"""Reference-comparison maintainer tool contracts."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from support.paths import load_tool_script
import subprocess
from support.paths import REPO_ROOT, load_artifact_tool
from gkx.core_ky_layout import rows_for_layout, source_ky_layout
from gkx.solvers_nonlinear_state_integration import DIVERGENCE_KNEE_STEPS
from support.paths import load_profiling_tool
import math
import jax.numpy as jnp
from gkx.terms.config import FieldState
from gkx.terms.config import TermConfig
import ast
import inspect
import re
import textwrap
import jax
from scripts.profiling._profiler_options import make_profile_options
from scripts.profiling.profile_startup_and_cache import (
    PhaseTiming,
    _write_phase_csv,
    _write_phase_json,
    build_low_rank_moment_cache,
    main_runtime_startup,
)
from gkx.solvers_nonlinear_diagnostic_integration import (
    integrate_nonlinear_explicit_diagnostics_state,
)
from support.paths import load_release_tool
from scripts.campaigns.portfolio_guard import (
    ReducedPortfolioArtifactGuardConfig,
    reduced_portfolio_artifact_guard_report,
)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1.0, 1.01, 1.02], 0),
        ([1.0, 1.01, 1.02, 0.75], None),  # early false plateau
        ([1.0, 1.04, 1.08, 1.12], None),  # accumulated sub-tolerance drift
        ([1.0, 0.75, 0.751, 0.752], 1),  # genuinely settled suffix
        ([1.0, 1.01], None),  # two refinements are required
        ([1.0, 1.01, 1.02, float("nan")], None),
        ([1.0, 1.01, 1.02, float("inf")], None),
        ([0.0, 0.0, 0.0], 0),
    ],
)
def test_refinement_requires_a_consistent_finite_suffix(values, expected):
    result = load_tool_script("campaigns", "convergence_protocol").refine(
        "velocity", range(len(values)), values.__getitem__, verbose=False
    )
    assert result.converged_value == expected
    assert result.to_dict()["converged"] == (expected is not None)


# ---- imported-linear growth-dump mode ----

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "comparison"))

from compare_gx_imported_linear import (
    _expand_gx_restart_state_to_full_positive_ky,
    _load_growth_dt,
    _load_gx_restart_state,
    _load_gx_restart_time,
    build_growth_dump_parser as growth_dump_build_parser,
)


def test_compare_gx_imported_growth_dump_parser_accepts_required_paths() -> None:
    args = growth_dump_build_parser().parse_args(
        [
            "--gx-dir-start",
            "/tmp/start",
            "--gx-dir-stop",
            "/tmp/stop",
            "--gx-out",
            "/tmp/run.out.nc",
            "--gx-input",
            "/tmp/run.in",
            "--geometry-file",
            "/tmp/geom.nc",
            "--time-index-start",
            "10",
            "--time-index-stop",
            "11",
        ]
    )
    assert args.gx_dir_start == Path("/tmp/start")
    assert args.gx_dir_stop == Path("/tmp/stop")
    assert args.gx_out == Path("/tmp/run.out.nc")
    assert args.gx_input == Path("/tmp/run.in")
    assert args.geometry_file == Path("/tmp/geom.nc")
    assert args.time_index_start == 10
    assert args.time_index_stop == 11


def test_compare_gx_imported_growth_dump_parser_accepts_restart_start() -> None:
    args = growth_dump_build_parser().parse_args(
        [
            "--gx-dir-start",
            "/tmp/start",
            "--gx-dir-stop",
            "/tmp/stop",
            "--gx-restart-start",
            "/tmp/restart.nc",
            "--gx-out",
            "/tmp/run.out.nc",
            "--gx-input",
            "/tmp/run.in",
            "--geometry-file",
            "/tmp/geom.nc",
            "--time-index-start",
            "10",
            "--time-index-stop",
            "11",
        ]
    )
    assert args.gx_restart_start == Path("/tmp/restart.nc")


def test_load_growth_dt_accepts_float64_scalar(tmp_path: Path) -> None:
    path = tmp_path / "diag_growth_dt_t45.bin"
    import numpy as np

    np.asarray([2.5e-4], dtype=np.float64).tofile(path)
    assert _load_growth_dt(path) == 2.5e-4


def test_load_gx_restart_state_transposes_to_gkx_layout(tmp_path: Path) -> None:
    import numpy as np
    from netCDF4 import Dataset

    path = tmp_path / "restart.nc"
    with Dataset(path, "w") as root:
        root.createDimension("Nspecies", 1)
        root.createDimension("Nm", 3)
        root.createDimension("Nl", 2)
        root.createDimension("Nz", 5)
        root.createDimension("Nkx", 1)
        root.createDimension("Nky", 4)
        root.createDimension("ri", 2)
        t = root.createVariable("time", "f8", ())
        t.assignValue(3.25)
        g = root.createVariable(
            "G", "f4", ("Nspecies", "Nm", "Nl", "Nz", "Nkx", "Nky", "ri")
        )
        raw = np.zeros((1, 3, 2, 5, 1, 4, 2), dtype=np.float32)
        raw[0, 2, 1, 4, 0, 3, 0] = 7.0
        raw[0, 2, 1, 4, 0, 3, 1] = -2.0
        g[:] = raw

    state = _load_gx_restart_state(path)
    assert state.shape == (1, 2, 3, 4, 1, 5)
    assert state[0, 1, 2, 3, 0, 4] == np.complex64(7.0 - 2.0j)
    assert _load_gx_restart_time(path) == 3.25


def test_expand_gx_restart_state_to_full_positive_ky_embeds_dealiased_kx() -> None:
    import numpy as np

    # ny_full=16 -> nyc_full=9, active naky=6; nx_full=4 -> active nakx=3
    active = np.zeros((1, 1, 1, 6, 3, 2), dtype=np.complex64)
    active[0, 0, 0, 5, 0, 1] = 1.0 + 2.0j
    active[0, 0, 0, 5, 1, 1] = 3.0 + 4.0j
    active[0, 0, 0, 5, 2, 1] = 5.0 + 6.0j
    full = _expand_gx_restart_state_to_full_positive_ky(active, ny_full=16, nx_full=4)
    assert full.shape == (1, 1, 1, 9, 4, 2)
    assert full[0, 0, 0, 5, 0, 1] == np.complex64(1.0 + 2.0j)
    assert full[0, 0, 0, 5, 1, 1] == np.complex64(3.0 + 4.0j)
    assert full[0, 0, 0, 5, 3, 1] == np.complex64(5.0 + 6.0j)
    assert full[0, 0, 0, 6, 0, 1] == 0.0j


# ---- test_compare_gx_imported_linear.py ----


import compare_gx_imported_linear as imported_linear

from compare_gx_imported_linear import (
    GXInputContract,
    _build_imported_initial_condition,
    _build_imported_linear_terms,
    _build_sample_steps,
    _gx_has_uniform_linear_dt,
    _resolve_imported_boundary,
    _infer_gx_linear_dt,
    _integrate_target_mode_series,
    _distribution_free_energy_by_ky,
    _gx_kyst_fac_mask_cached,
    _load_gx_input_contract,
    _match_local_kx_index,
    _resolve_imported_real_fft_ny,
    _run_single_ky,
    _resolve_internal_geometry_source,
    _select_gx_kx_index,
    _write_scan_rows,
    build_parser as imported_linear_build_parser,
)
from gkx.config import GeometryConfig, GridConfig
from gkx.geometry import SAlphaGeometry, sample_flux_tube_geometry
from gkx.core_grid import build_spectral_grid
from gkx.solvers_time_explicit import ExplicitTimeConfig
from gkx.operators.linear.params import LinearTerms
from gkx.config import RuntimeConfig
from gkx.operators.linear.params import Species


def test_compare_gx_imported_linear_parser_accepts_gx_input() -> None:
    args = imported_linear_build_parser().parse_args(
        [
            "--gx",
            "/tmp/run.out.nc",
            "--geometry-file",
            "/tmp/run.eik.nc",
            "--gx-input",
            "/tmp/run.in",
        ]
    )
    assert args.gx_input == Path("/tmp/run.in")


def test_compare_gx_imported_linear_parser_accepts_exact_init_file() -> None:
    args = imported_linear_build_parser().parse_args(
        [
            "--gx",
            "/tmp/run.out.nc",
            "--geometry-file",
            "/tmp/run.eik.nc",
            "--init-file",
            "/tmp/g_state.bin",
        ]
    )
    assert args.init_file == Path("/tmp/g_state.bin")


def test_compare_gx_imported_linear_parser_accepts_cache_and_sample_controls() -> None:
    args = imported_linear_build_parser().parse_args(
        [
            "--gx",
            "/tmp/run.out.nc",
            "--geometry-file",
            "/tmp/run.eik.nc",
            "--cache-dir",
            "/tmp/cache",
            "--reuse-cache",
            "--sample-step-stride",
            "3",
            "--max-samples",
            "12",
        ]
    )
    assert args.cache_dir == Path("/tmp/cache")
    assert args.reuse_cache is True
    assert args.sample_step_stride == 3
    assert args.max_samples == 12


def test_compare_gx_imported_linear_parser_accepts_project_mode_method() -> None:
    args = imported_linear_build_parser().parse_args(
        [
            "--gx",
            "/tmp/run.out.nc",
            "--geometry-file",
            "/tmp/run.eik.nc",
            "--mode-method",
            "project",
        ]
    )
    assert args.mode_method == "project"


def test_build_sample_steps_supports_stride_and_early_window() -> None:
    gx_time = np.linspace(0.0, 9.0, 10)
    assert np.array_equal(
        _build_sample_steps(gx_time, sample_step_stride=1, max_samples=None),
        np.arange(10),
    )
    assert np.array_equal(
        _build_sample_steps(gx_time, sample_step_stride=2, max_samples=None),
        np.arange(0, 10, 2),
    )
    assert np.array_equal(
        _build_sample_steps(gx_time, sample_step_stride=2, max_samples=3),
        np.asarray([0, 2, 4]),
    )
    assert np.array_equal(
        _build_sample_steps(
            gx_time, sample_step_stride=2, max_samples=3, sample_window="tail"
        ),
        np.asarray([4, 6, 8]),
    )


def test_load_gx_input_contract_reads_fix_aspect_and_species_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.in"
    path.write_text(
        """
[Dimensions]
 ntheta = 48
 nperiod = 1
 ny = 96
 nx = 96
 nspecies = 1

[Domain]
 y0 = 21.0
 boundary = "fix aspect"

[Physics]
 beta = 0.01

[Time]
 dt = 0.005
 scheme = "rk3"

[Initialization]
 init_field = "density"
 init_amp = 1.0e-3
 ikpar_init = 0

[Diagnostics]
 nwrite = 50

[species]
 z = [1.0, -1.0]
 mass = [1.0, 0.00027]
 dens = [1.0, 1.0]
 temp = [1.0, 1.0]
 tprim = [3.0, 0.0]
 fprim = [1.0, 0.0]
 vnewk = [0.01, 0.0]

[Boltzmann]
 add_Boltzmann_species = true
 Boltzmann_type = "electrons"
 tau_fac = 1.0

[Dissipation]
 hypercollisions = true
 hyper = true
 D_hyper = 0.05
""".strip()
    )

    contract = _load_gx_input_contract(path)
    assert contract.Nx == 96
    assert contract.Ny == 96
    assert contract.nperiod == 1
    assert contract.ntheta == 48
    assert contract.npol is None
    assert contract.alpha is None
    assert contract.torflux is None
    assert contract.nlaguerre == 8
    assert contract.nhermite == 16
    assert contract.boundary == "fix aspect"
    assert contract.geo_option == "s-alpha"
    assert contract.y0 == 21.0
    assert contract.fapar == 1.0
    assert contract.fbpar == 1.0
    assert contract.beta == 0.01
    assert contract.tau_e == 1.0
    assert contract.dt == 0.005
    assert contract.scheme == "rk3"
    assert contract.nwrite == 50
    assert contract.init_field == "density"
    assert contract.init_amp == 1.0e-3
    assert contract.init_single is False
    assert contract.gaussian_init is False
    assert contract.kpar_init == 0.0
    assert contract.random_seed == 22
    assert contract.hypercollisions is True
    assert contract.hyper is True
    assert contract.D_hyper == 0.05
    assert contract.damp_ends_amp == 0.1
    assert contract.damp_ends_widthfrac == 1.0 / 8.0
    assert contract.restart_with_perturb is False
    assert contract.restart_scale == 1.0
    assert len(contract.species) == 1
    assert contract.species[0].charge == 1.0
    assert contract.species[0].tprim == 3.0


def test_compare_gx_imported_linear_parser_defaults_hl_dims_to_gx_contract() -> None:
    args = imported_linear_build_parser().parse_args(
        [
            "--gx",
            "/tmp/run.out.nc",
            "--geometry-file",
            "/tmp/run.eik.nc",
        ]
    )
    assert args.Nl is None
    assert args.Nm is None


def test_write_scan_rows_preserves_extended_metric_columns(tmp_path: Path) -> None:
    out = tmp_path / "scan.csv"
    rows = [
        {
            "ky": 0.2,
            "peak_abs_omega_ref": 0.4,
            "mean_abs_omega": 0.01,
            "mean_rel_omega": 0.02,
            "peak_abs_gamma_ref": 0.03,
            "mean_abs_gamma": 0.004,
            "mean_rel_gamma": 0.5,
            "mean_abs_Wg": 1.0e-5,
            "mean_rel_Wg": 0.03,
            "mean_abs_Wphi": 2.0e-5,
            "mean_rel_Wphi": 0.04,
            "mean_abs_Wapar": 0.0,
            "mean_rel_Wapar": 0.0,
        },
        {
            "ky": 0.1,
            "peak_abs_omega_ref": 0.2,
            "mean_abs_omega": 0.005,
            "mean_rel_omega": 0.01,
            "peak_abs_gamma_ref": 0.01,
            "mean_abs_gamma": 0.002,
            "mean_rel_gamma": 0.25,
            "mean_abs_Wg": 5.0e-6,
            "mean_rel_Wg": 0.02,
            "mean_abs_Wphi": 1.0e-5,
            "mean_rel_Wphi": 0.03,
            "mean_abs_Wapar": 0.0,
            "mean_rel_Wapar": 0.0,
            "mean_abs_Phi2": 3.0e-5,
            "mean_rel_Phi2": 0.05,
        },
    ]

    df = _write_scan_rows(rows, out)

    assert list(df["ky"]) == [0.1, 0.2]
    assert "peak_abs_gamma_ref" in df.columns
    assert "mean_abs_Wg" in df.columns
    assert "mean_abs_Wphi" in df.columns
    assert "mean_abs_Phi2" in df.columns

    written = out.read_text()
    assert "mean_abs_Wg" in written
    assert "peak_abs_omega_ref" in written


def test_imported_linear_zero_shat_promotes_to_periodic_boundary() -> None:
    assert _resolve_imported_boundary("linked", zero_shat=True) == "periodic"
    assert _resolve_imported_boundary("periodic", zero_shat=True) == "periodic"
    assert _resolve_imported_boundary("linked", zero_shat=False) == "linked"


def test_load_gx_input_contract_promotes_near_zero_shear_to_zero_shat(
    tmp_path: Path,
) -> None:
    path = tmp_path / "kaw_like.in"
    path.write_text(
        """
restart_with_perturb = true
scale = 0.125

[Dimensions]
 ntheta = 16
 nperiod = 1
 nky = 2
 nkx = 1
 nspecies = 1

[Domain]
 y0 = 100.0
 boundary = "linked"

[Physics]
 beta = 0.01

[Geometry]
 geo_option = "slab"
 shat = 1.0e-8
""".strip()
    )

    contract = _load_gx_input_contract(path)

    assert contract.s_hat == pytest.approx(1.0e-8)
    assert contract.zero_shat is True
    assert (
        _resolve_imported_boundary(contract.boundary, zero_shat=contract.zero_shat)
        == "periodic"
    )


def test_load_gx_input_contract_reads_vmec_geometry_contract(tmp_path: Path) -> None:
    path = tmp_path / "w7x.in"
    path.write_text(
        """
[Dimensions]
 ntheta = 256
 nperiod = 1
 nky = 28
 nkx = 1

[Domain]
 y0 = 10.0
 boundary = "linked"

[Geometry]
 geo_option = "nc"
 alpha = 0.0
 torflux = 0.64
 npol = 6.0
""".strip()
    )

    contract = _load_gx_input_contract(path)
    assert contract.Nx == 1
    assert contract.Ny == 28
    assert contract.nperiod == 1
    assert contract.ntheta == 256
    assert contract.alpha == pytest.approx(0.0)
    assert contract.torflux == pytest.approx(0.64)
    assert contract.npol == pytest.approx(6.0)


def test_load_gx_input_contract_parses_restart_contract(tmp_path: Path) -> None:
    path = tmp_path / "restart_like.in"
    path.write_text(
        """
restart_with_perturb = true
scale = 0.125

[Dimensions]
 ntheta = 16
 nperiod = 1
 nky = 2
 nkx = 1
 nspecies = 1

[Domain]
 y0 = 100.0
 boundary = "linked"

[Geometry]
 geo_option = "slab"
 shat = 0.0
""".strip()
    )

    contract = _load_gx_input_contract(path)

    assert contract.restart_with_perturb is True
    assert contract.restart_scale == pytest.approx(0.125)


def test_imported_linear_uses_raw_damp_ends_rate() -> None:
    contract = _dummy_gx_contract(init_single=False)
    dt = 0.2
    params = imported_linear.build_linear_params(
        contract.species,
        tau_e=contract.tau_e,
        kpar_scale=1.0,
        beta=contract.beta,
    )
    params = replace(
        params,
        D_hyper=float(contract.D_hyper),
        damp_ends_amp=float(contract.damp_ends_amp),
        damp_ends_widthfrac=float(contract.damp_ends_widthfrac),
    )
    assert float(params.damp_ends_amp) == pytest.approx(0.1)
    assert float(params.damp_ends_amp) != pytest.approx(0.1 / dt)


def test_infer_gx_linear_dt_prefers_explicit_input_dt() -> None:
    contract = replace(_dummy_gx_contract(init_single=False), dt=0.025, nwrite=50)
    gx_time = np.asarray([1.25, 2.50, 3.75], dtype=float)
    assert _infer_gx_linear_dt(gx_time, contract) == pytest.approx(0.025)


def test_infer_gx_linear_dt_uses_diagnostic_spacing_without_input_dt() -> None:
    contract = replace(_dummy_gx_contract(init_single=False), dt=None, nwrite=100)
    gx_time = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=float)
    assert _infer_gx_linear_dt(gx_time, contract) == pytest.approx(0.005)


def test_gx_has_uniform_linear_dt_true_for_constant_spacing() -> None:
    contract = replace(_dummy_gx_contract(init_single=False), dt=None, nwrite=10)
    gx_time = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float)
    assert _gx_has_uniform_linear_dt(gx_time, contract) is True


def test_gx_has_uniform_linear_dt_false_for_variable_spacing() -> None:
    contract = replace(_dummy_gx_contract(init_single=False), dt=None, nwrite=10)
    gx_time = np.asarray([0.1, 0.21, 0.33, 0.46], dtype=float)
    assert _gx_has_uniform_linear_dt(gx_time, contract) is False


def test_gx_has_uniform_linear_dt_ignores_single_truncated_final_interval() -> None:
    contract = replace(_dummy_gx_contract(init_single=False), dt=None, nwrite=10)
    gx_time = np.asarray([0.1, 0.2, 0.3, 0.35], dtype=float)
    assert _gx_has_uniform_linear_dt(gx_time, contract) is True


@pytest.mark.skipif(
    not Path(".cache/gx_clean_main/linear/hsx/hsx_linear.in").exists(),
    reason="Requires local cache file",
)
def test_build_imported_initial_condition_uses_runtime_multikx_startup() -> None:
    class DummyGeom:
        s_hat = 1.0

    contract = _load_gx_input_contract(
        Path(".cache/gx_clean_main/linear/hsx/hsx_linear.in")
    )
    grid_full = build_spectral_grid(
        GridConfig(
            Nx=9,
            Ny=10,
            Nz=8,
            Lx=62.8,
            Ly=2.0 * np.pi * contract.y0,
            boundary="periodic",
            y0=contract.y0,
            nperiod=1,
            ntheta=8,
        )
    )
    g0 = _build_imported_initial_condition(
        grid=grid_full,
        geom=DummyGeom(),
        gx_contract=contract,
        species=contract.species,
        ky_index=1,
        kx_index=0,
        Nl=8,
        Nm=4,
    )
    g0_np = np.asarray(g0)
    nonzero_kx = np.flatnonzero(np.any(np.abs(g0_np[0, 0, 0, 1]) > 0.0, axis=-1))
    assert nonzero_kx.size > 1


def test_match_local_kx_index_uses_kx_value_not_raw_index() -> None:
    grid_kx = np.asarray([0.0, 0.05, 0.10, 0.15, -0.15, -0.10, -0.05], dtype=float)
    assert _match_local_kx_index(grid_kx, -0.10) == 5
    assert _match_local_kx_index(grid_kx, 0.15) == 3


def test_select_gx_kx_index_defaults_to_kx_zero_branch() -> None:
    gx_kx = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=float)
    assert _select_gx_kx_index(gx_kx, None) == 2
    assert _select_gx_kx_index(gx_kx, _dummy_gx_contract(init_single=False)) == 2


def test_select_gx_kx_index_honors_explicit_single_mode_startup() -> None:
    gx_kx = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=float)
    contract = replace(_dummy_gx_contract(init_single=True), ikx_single=4)
    assert _select_gx_kx_index(gx_kx, contract) == 4


def test_resolve_imported_real_fft_ny_uses_full_gx_ky_layout() -> None:
    gx_ky = np.asarray([0.0] + [0.05 * i for i in range(1, 16)], dtype=float)
    contract = replace(_dummy_gx_contract(init_single=False), Ny=16)
    assert _resolve_imported_real_fft_ny(gx_ky, contract) == 46


def test_resolve_imported_real_fft_ny_recovers_miller_gx_nky_contract() -> None:
    gx_ky = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    contract = replace(_dummy_gx_contract(init_single=False), Ny=6)
    assert _resolve_imported_real_fft_ny(gx_ky, contract) == 16


def test_resolve_imported_real_fft_ny_keeps_single_positive_ky_unmasked() -> None:
    gx_ky = np.asarray([0.0, 0.01], dtype=float)
    contract = replace(_dummy_gx_contract(init_single=False), Ny=2)
    assert _resolve_imported_real_fft_ny(gx_ky, contract) == 4


def test_resolve_imported_real_fft_ny_accepts_full_diag_state_ky_block() -> None:
    gx_ky = np.asarray([0.0, 0.01, 0.02], dtype=float)
    contract = replace(_dummy_gx_contract(init_single=False), Ny=2)
    assert _resolve_imported_real_fft_ny(gx_ky, contract) == 4


def _dummy_gx_contract(*, init_single: bool) -> GXInputContract:
    return GXInputContract(
        Nx=8,
        Ny=8,
        nperiod=1,
        ntheta=8,
        npol=1.0,
        alpha=0.0,
        torflux=0.5,
        nlaguerre=8,
        nhermite=16,
        boundary="periodic",
        geo_option="s-alpha",
        s_hat=0.0,
        zero_shat=False,
        y0=10.0,
        fapar=0.0,
        fbpar=0.0,
        species=(
            Species(
                charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=0.0, fprim=0.0
            ),
        ),
        tau_e=0.0,
        beta=0.0,
        dt=0.1,
        scheme="rk4",
        nwrite=1,
        init_field="density",
        init_amp=1.0e-5,
        init_single=init_single,
        ikx_single=0,
        iky_single=1,
        gaussian_init=False,
        gaussian_width=0.5,
        gaussian_envelope_constant=1.0,
        gaussian_envelope_sine=0.0,
        kpar_init=0.0,
        random_seed=22,
        init_electrons_only=False,
        random_init=False,
        hypercollisions=False,
        hyper=False,
        D_hyper=0.0,
        damp_ends_amp=0.1,
        damp_ends_widthfrac=1.0 / 8.0,
        restart_with_perturb=False,
        restart_scale=1.0,
    )


def test_build_imported_linear_terms_honors_em_switches() -> None:
    electrostatic = _build_imported_linear_terms(_dummy_gx_contract(init_single=False))
    assert electrostatic.apar == 0.0
    assert electrostatic.bpar == 0.0

    electromagnetic = _build_imported_linear_terms(
        replace(
            _dummy_gx_contract(init_single=False),
            fapar=1.0,
            fbpar=1.0,
            hypercollisions=True,
            hyper=True,
        )
    )
    assert electromagnetic.apar == 1.0
    assert electromagnetic.bpar == 1.0
    assert electromagnetic.hypercollisions == 1.0
    assert electromagnetic.hyperdiffusion == 1.0


def test_run_single_ky_uses_full_grid_for_imported_multimode(monkeypatch) -> None:
    grid_full = SimpleNamespace(
        ky=np.asarray([0.0, 0.1, 0.2], dtype=float),
        kx=np.asarray([0.0, 0.1], dtype=float),
        z=np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=float),
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        imported_linear,
        "_build_imported_initial_condition",
        lambda **_: np.zeros((1, 1, 1, 3, 2, 4), dtype=np.complex64),
    )
    monkeypatch.setattr(
        imported_linear, "build_linear_cache", lambda *_args, **_kwargs: "cache"
    )

    def _fake_integrate(**kwargs):
        captured["grid"] = kwargs["grid"]
        captured["g_shape"] = tuple(np.asarray(kwargs["G0"]).shape)
        captured["ky_index"] = kwargs["ky_index"]
        return tuple(np.zeros(2, dtype=float) for _ in range(6))

    monkeypatch.setattr(
        imported_linear, "_integrate_target_mode_series", _fake_integrate
    )

    _run_single_ky(
        ky_target=0.1,
        geom=SimpleNamespace(),
        grid_full=grid_full,
        params=SimpleNamespace(),
        time_cfg=ExplicitTimeConfig(dt=0.1, t_max=0.2, sample_stride=1, fixed_dt=True),
        gx_contract=_dummy_gx_contract(init_single=False),
        species=(
            Species(
                charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=0.0, fprim=0.0
            ),
        ),
        Nl=1,
        Nm=1,
        reference_times=np.asarray([0.1, 0.2], dtype=float),
        output_steps=np.asarray([0, 1], dtype=int),
        mode_method="z_index",
        kx_index=0,
        terms=LinearTerms(),
    )

    assert captured["grid"] is grid_full
    assert captured["g_shape"] == (1, 1, 1, 3, 2, 4)
    assert captured["ky_index"] == 1


def test_run_single_ky_preserves_single_ky_fallback_without_gx_contract(
    monkeypatch,
) -> None:
    grid_full = build_spectral_grid(
        GridConfig(
            Nx=4,
            Ny=6,
            Nz=4,
            Lx=10.0,
            Ly=20.0,
            boundary="periodic",
            y0=10.0,
        )
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        imported_linear,
        "_build_imported_initial_condition",
        lambda **_: np.zeros(
            (1, 1, 1, grid_full.ky.size, grid_full.kx.size, grid_full.z.size),
            dtype=np.complex64,
        ),
    )
    monkeypatch.setattr(
        imported_linear, "build_linear_cache", lambda *_args, **_kwargs: "cache"
    )

    def _fake_integrate(**kwargs):
        captured["grid_ky"] = int(kwargs["grid"].ky.size)
        captured["g_shape"] = tuple(np.asarray(kwargs["G0"]).shape)
        captured["ky_index"] = kwargs["ky_index"]
        return tuple(np.zeros(2, dtype=float) for _ in range(6))

    monkeypatch.setattr(
        imported_linear, "_integrate_target_mode_series", _fake_integrate
    )

    _run_single_ky(
        ky_target=float(grid_full.ky[1]),
        geom=SimpleNamespace(),
        grid_full=grid_full,
        params=SimpleNamespace(),
        time_cfg=ExplicitTimeConfig(dt=0.1, t_max=0.2, sample_stride=1, fixed_dt=True),
        gx_contract=None,
        species=(
            Species(
                charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=0.0, fprim=0.0
            ),
        ),
        Nl=1,
        Nm=1,
        reference_times=np.asarray([0.1, 0.2], dtype=float),
        output_steps=np.asarray([0, 1], dtype=int),
        mode_method="z_index",
        kx_index=0,
        terms=LinearTerms(),
    )

    assert captured["grid_ky"] == 1
    assert captured["g_shape"][3] == 1
    assert captured["ky_index"] == 0


def test_gx_kyst_fac_mask_cached_uses_positive_half_storage_on_full_ky_grid() -> None:
    cache = SimpleNamespace(
        ky=np.asarray([-0.2, 0.0, 0.2], dtype=np.float32),
        kx=np.asarray([0.0, 0.1], dtype=np.float32),
        dealias_mask=np.asarray([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )
    fac = np.asarray(_gx_kyst_fac_mask_cached(cache, use_dealias=True), dtype=float)
    np.testing.assert_allclose(
        fac,
        np.asarray(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
            ],
            dtype=float,
        ),
    )


def test_distribution_free_energy_by_ky_matches_gx_positive_ky_storage_contract() -> (
    None
):
    cache = SimpleNamespace(
        ky=np.asarray([-0.2, 0.0, 0.2], dtype=np.float32),
        kx=np.asarray([0.0], dtype=np.float32),
        dealias_mask=np.asarray([[1.0], [1.0], [1.0]], dtype=np.float32),
    )
    params = SimpleNamespace(density=1.0, temp=1.0)
    vol_fac = jnp.asarray([1.0], dtype=jnp.float32)
    G = jnp.ones((1, 1, 1, 3, 1, 1), dtype=jnp.complex64)
    Wg = np.asarray(
        _distribution_free_energy_by_ky(G, cache, params, vol_fac), dtype=float
    )
    assert np.allclose(Wg, np.asarray([0.0, 0.5, 1.0], dtype=float))


def test_select_geometry_source_prefers_gx_output_for_vmec_generated_runs() -> None:
    gx_out = Path("/tmp/run.out.nc").resolve()
    geom = Path("/tmp/run.eik.nc").resolve()
    vmec_contract = replace(_dummy_gx_contract(init_single=False), geo_option="vmec")
    desc_contract = replace(_dummy_gx_contract(init_single=False), geo_option="desc")
    nc_contract = replace(_dummy_gx_contract(init_single=False), geo_option="nc")
    assert (
        _resolve_internal_geometry_source(
            geometry_file=geom, runtime_config=None, gx_contract=vmec_contract
        )
        == geom
    )
    assert (
        _resolve_internal_geometry_source(
            geometry_file=gx_out, runtime_config=None, gx_contract=vmec_contract
        )
        == gx_out
    )
    assert (
        _resolve_internal_geometry_source(
            geometry_file=gx_out, runtime_config=None, gx_contract=desc_contract
        )
        == gx_out
    )
    assert (
        _resolve_internal_geometry_source(
            geometry_file=geom, runtime_config=None, gx_contract=nc_contract
        )
        == geom
    )


def test_resolve_internal_geometry_source_uses_gx_grid_contract_for_internal_miller(
    monkeypatch,
) -> None:
    runtime_path = Path("/tmp/runtime_miller.toml")
    captured: dict[str, object] = {}
    cfg = RuntimeConfig(
        grid=GridConfig(boundary="periodic", y0=28.2, ntheta=24, nperiod=1),
        geometry=GeometryConfig(
            model="miller", q=1.4, s_hat=0.8, R0=2.77778, R_geo=2.77778
        ),
    )
    gx_contract = replace(
        _dummy_gx_contract(init_single=False),
        boundary="linked",
        y0=20.0,
        ntheta=32,
        nperiod=2,
    )
    out = Path("/tmp/internal_miller.eiknc.nc").resolve()

    monkeypatch.setattr(
        imported_linear, "load_runtime_from_toml", lambda _path: (cfg, {})
    )

    def _fake_generate_runtime_miller_eik(runtime_cfg, *, force):
        captured["boundary"] = runtime_cfg.grid.boundary
        captured["y0"] = runtime_cfg.grid.y0
        captured["ntheta"] = runtime_cfg.grid.ntheta
        captured["nperiod"] = runtime_cfg.grid.nperiod
        captured["force"] = force
        return out

    monkeypatch.setattr(
        imported_linear,
        "generate_runtime_miller_eik",
        _fake_generate_runtime_miller_eik,
    )

    resolved = _resolve_internal_geometry_source(
        geometry_file=None,
        runtime_config=runtime_path,
        gx_contract=gx_contract,
    )

    assert resolved == out
    assert captured == {
        "boundary": "linked",
        "y0": 20.0,
        "ntheta": 32,
        "nperiod": 2,
        "force": True,
    }


def test_resolve_internal_geometry_source_uses_gx_vmec_geometry_contract(
    monkeypatch,
) -> None:
    runtime_path = Path("/tmp/runtime_vmec.toml")
    captured: dict[str, object] = {}
    cfg = RuntimeConfig(
        grid=GridConfig(boundary="fix aspect", y0=21.0, ntheta=48, nperiod=1),
        geometry=GeometryConfig(
            model="vmec",
            vmec_file="/tmp/wout.nc",
            alpha=0.25,
            torflux=0.5,
            npol=1.0,
        ),
    )
    gx_contract = replace(
        _dummy_gx_contract(init_single=False),
        boundary="linked",
        y0=10.0,
        ntheta=256,
        nperiod=1,
        alpha=0.0,
        torflux=0.64,
        npol=6.0,
    )
    out = Path("/tmp/internal_vmec.eiknc.nc").resolve()

    monkeypatch.setattr(
        imported_linear, "load_runtime_from_toml", lambda _path: (cfg, {})
    )

    def _fake_generate_runtime_vmec_eik(runtime_cfg, *, force):
        captured["boundary"] = runtime_cfg.grid.boundary
        captured["y0"] = runtime_cfg.grid.y0
        captured["ntheta"] = runtime_cfg.grid.ntheta
        captured["nperiod"] = runtime_cfg.grid.nperiod
        captured["alpha"] = runtime_cfg.geometry.alpha
        captured["torflux"] = runtime_cfg.geometry.torflux
        captured["npol"] = runtime_cfg.geometry.npol
        captured["force"] = force
        return out

    monkeypatch.setattr(
        imported_linear, "generate_runtime_vmec_eik", _fake_generate_runtime_vmec_eik
    )

    resolved = _resolve_internal_geometry_source(
        geometry_file=None,
        runtime_config=runtime_path,
        gx_contract=gx_contract,
    )

    assert resolved == out
    assert captured == {
        "boundary": "linked",
        "y0": 10.0,
        "ntheta": 256,
        "nperiod": 1,
        "alpha": 0.0,
        "torflux": 0.64,
        "npol": 6.0,
        "force": True,
    }


def test_integrate_target_mode_series_collects_requested_sample_count(
    monkeypatch,
) -> None:
    monkeypatch.setattr(imported_linear.jax, "jit", lambda fn, donate_argnums=None: fn)
    monkeypatch.setattr(
        imported_linear, "ensure_flux_tube_geometry_data", lambda geom, _theta: geom
    )
    monkeypatch.setattr(
        imported_linear,
        "assemble_rhs_cached",
        lambda *_args, **_kwargs: (
            None,
            SimpleNamespace(phi=jnp.zeros((2, 2, 3), dtype=jnp.complex64), apar=None),
        ),
    )
    monkeypatch.setattr(
        imported_linear,
        "_linear_explicit_step",
        lambda G_state, *_args, **_kwargs: (
            G_state,
            SimpleNamespace(phi=jnp.zeros((2, 2, 3), dtype=jnp.complex64), apar=None),
        ),
    )
    monkeypatch.setattr(
        imported_linear,
        "_instantaneous_growth_rate_step",
        lambda *_args, **_kwargs: (
            jnp.ones((2, 2), dtype=jnp.float32),
            jnp.full((2, 2), 2.0, dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        imported_linear,
        "_distribution_free_energy_by_ky",
        lambda *_args, **_kwargs: jnp.asarray([0.0, 3.0]),
    )
    monkeypatch.setattr(
        imported_linear,
        "_electrostatic_field_energy_by_ky",
        lambda *_args, **_kwargs: jnp.asarray([0.0, 4.0]),
    )
    monkeypatch.setattr(
        imported_linear,
        "_magnetic_vector_potential_energy_by_ky",
        lambda *_args, **_kwargs: jnp.asarray([0.0, 5.0]),
    )
    monkeypatch.setattr(
        imported_linear,
        "_linear_frequency_bound",
        lambda *_args, **_kwargs: np.asarray([0.0, 0.0, 0.0]),
    )

    gamma, omega, Wg, Wphi, Wapar, Phi2 = _integrate_target_mode_series(
        G0=jnp.zeros((1, 1, 1, 2, 2, 3), dtype=jnp.complex64),
        grid=SimpleNamespace(dealias_mask=np.ones((2, 2), dtype=bool), z=np.arange(3)),
        geom=SimpleNamespace(
            s_hat=0.0,
            gradpar=lambda: 1.0,
            metric_coeffs=lambda theta: (
                jnp.ones_like(theta),
                jnp.zeros_like(theta),
                jnp.ones_like(theta),
            ),
            drift_coeffs=lambda theta: (
                jnp.zeros_like(theta),
                jnp.zeros_like(theta),
                jnp.zeros_like(theta),
                jnp.zeros_like(theta),
            ),
        ),
        cache=SimpleNamespace(jacobian=jnp.ones(3, dtype=jnp.float32)),
        params=SimpleNamespace(),
        time_cfg=ExplicitTimeConfig(dt=0.1, t_max=0.21, sample_stride=1, fixed_dt=True),
        terms=LinearTerms(),
        mode_method="z_index",
        ky_index=1,
        kx_index=0,
        reference_times=np.asarray([0.1, 0.2, 0.3], dtype=float),
        output_steps=np.asarray([0, 1, 2], dtype=int),
    )

    np.testing.assert_allclose(gamma, np.ones(3, dtype=float))
    np.testing.assert_allclose(omega, np.full(3, 2.0, dtype=float))
    np.testing.assert_allclose(Wg, np.full(3, 3.0, dtype=float))
    np.testing.assert_allclose(Wphi, np.full(3, 4.0, dtype=float))
    np.testing.assert_allclose(Wapar, np.full(3, 5.0, dtype=float))
    np.testing.assert_allclose(Phi2, np.zeros(3, dtype=float))


def test_integrate_target_mode_series_normalizes_imported_geometry_before_omega_max(
    monkeypatch,
) -> None:
    monkeypatch.setattr(imported_linear.jax, "jit", lambda fn, donate_argnums=None: fn)
    monkeypatch.setattr(
        imported_linear,
        "assemble_rhs_cached",
        lambda *_args, **_kwargs: (
            None,
            SimpleNamespace(phi=jnp.zeros((1, 1, 4), dtype=jnp.complex64), apar=None),
        ),
    )
    monkeypatch.setattr(
        imported_linear,
        "_linear_explicit_step",
        lambda G_state, *_args, **_kwargs: (
            G_state,
            SimpleNamespace(phi=jnp.zeros((1, 1, 4), dtype=jnp.complex64), apar=None),
        ),
    )
    monkeypatch.setattr(
        imported_linear,
        "_instantaneous_growth_rate_step",
        lambda *_args, **_kwargs: (
            jnp.asarray([[0.0]], dtype=float),
            jnp.asarray([[0.0]], dtype=float),
        ),
    )
    monkeypatch.setattr(
        imported_linear,
        "_distribution_free_energy_by_ky",
        lambda *_args, **_kwargs: jnp.asarray([0.0]),
    )
    monkeypatch.setattr(
        imported_linear,
        "_electrostatic_field_energy_by_ky",
        lambda *_args, **_kwargs: jnp.asarray([0.0]),
    )
    monkeypatch.setattr(
        imported_linear,
        "_magnetic_vector_potential_energy_by_ky",
        lambda *_args, **_kwargs: jnp.asarray([0.0]),
    )

    analytic = SAlphaGeometry.from_config(
        imported_linear.GeometryConfig(
            model="s-alpha", q=1.4, s_hat=0.8, epsilon=0.18, R0=1.0
        )
    )
    theta_solver = jnp.linspace(-jnp.pi, jnp.pi, 4, endpoint=False)
    theta_closed = jnp.linspace(-jnp.pi, jnp.pi, 5)
    sampled_closed = sample_flux_tube_geometry(analytic, theta_closed)
    geom = replace(sampled_closed, theta_closed_interval=True)

    captured: dict[str, float] = {}

    def _fake_omega_max(grid_arg, geom_arg, *_args, **_kwargs):
        theta_arg = np.asarray(geom_arg.theta, dtype=float)
        captured["theta_len"] = float(theta_arg.shape[0])
        captured["theta_last"] = float(theta_arg[-1])
        captured["grid_z_len"] = float(np.asarray(grid_arg.z).shape[0])
        return np.asarray([0.0, 0.0, 0.0], dtype=float)

    monkeypatch.setattr(imported_linear, "_linear_frequency_bound", _fake_omega_max)

    _integrate_target_mode_series(
        G0=jnp.zeros((1, 1, 1, 1, 1, 4), dtype=jnp.complex64),
        grid=SimpleNamespace(
            dealias_mask=np.ones((1, 1), dtype=bool),
            z=np.asarray(theta_solver, dtype=float),
        ),
        geom=geom,
        cache=SimpleNamespace(jacobian=jnp.ones(4, dtype=jnp.float32)),
        params=SimpleNamespace(),
        time_cfg=ExplicitTimeConfig(dt=0.1, t_max=0.1, sample_stride=1, fixed_dt=True),
        terms=LinearTerms(),
        mode_method="z_index",
        ky_index=0,
        kx_index=0,
        reference_times=np.asarray([0.1], dtype=float),
        output_steps=np.asarray([0], dtype=int),
    )

    assert captured["theta_len"] == captured["grid_z_len"] == 4.0
    assert captured["theta_last"] != pytest.approx(float(theta_closed[-1]))


def test_integrate_target_mode_series_uses_elapsed_sample_interval(monkeypatch) -> None:
    monkeypatch.setattr(imported_linear.jax, "jit", lambda fn, donate_argnums=None: fn)
    monkeypatch.setattr(
        imported_linear, "ensure_flux_tube_geometry_data", lambda geom, _theta: geom
    )
    monkeypatch.setattr(
        imported_linear,
        "assemble_rhs_cached",
        lambda *_args, **_kwargs: (
            None,
            SimpleNamespace(phi=jnp.zeros((1, 1, 1), dtype=jnp.complex64), apar=None),
        ),
    )

    step_count = {"n": 0}

    def _fake_step(G_state, *_args, **_kwargs):
        step_count["n"] += 1
        phi_val = float(step_count["n"])
        phi = jnp.full((1, 1, 1), phi_val, dtype=jnp.complex64)
        return G_state, SimpleNamespace(phi=phi, apar=None)

    monkeypatch.setattr(imported_linear, "_linear_explicit_step", _fake_step)
    captured: dict[str, object] = {}

    def _fake_growth(phi, phi_prev, dt_step, **_kwargs):
        captured["phi"] = np.asarray(phi)
        captured["phi_prev"] = np.asarray(phi_prev)
        captured["dt"] = float(dt_step)
        return jnp.ones((1, 1), dtype=jnp.float32), jnp.ones((1, 1), dtype=jnp.float32)

    monkeypatch.setattr(
        imported_linear, "_instantaneous_growth_rate_step", _fake_growth
    )
    monkeypatch.setattr(
        imported_linear,
        "_distribution_free_energy_by_ky",
        lambda *_args, **_kwargs: jnp.asarray([1.0]),
    )
    monkeypatch.setattr(
        imported_linear,
        "_electrostatic_field_energy_by_ky",
        lambda *_args, **_kwargs: jnp.asarray([1.0]),
    )
    monkeypatch.setattr(
        imported_linear,
        "_magnetic_vector_potential_energy_by_ky",
        lambda *_args, **_kwargs: jnp.asarray([0.0]),
    )
    monkeypatch.setattr(
        imported_linear,
        "_linear_frequency_bound",
        lambda *_args, **_kwargs: np.asarray([0.0, 0.0, 0.0]),
    )

    _integrate_target_mode_series(
        G0=jnp.zeros((1, 1, 1, 1, 1, 1), dtype=jnp.complex64),
        grid=SimpleNamespace(dealias_mask=np.ones((1, 1), dtype=bool), z=np.arange(1)),
        geom=SimpleNamespace(
            s_hat=0.0,
            gradpar=lambda: 1.0,
            metric_coeffs=lambda theta: (
                jnp.ones_like(theta),
                jnp.zeros_like(theta),
                jnp.ones_like(theta),
            ),
            drift_coeffs=lambda theta: (
                jnp.zeros_like(theta),
                jnp.zeros_like(theta),
                jnp.zeros_like(theta),
                jnp.zeros_like(theta),
            ),
        ),
        cache=SimpleNamespace(jacobian=jnp.ones(1, dtype=jnp.float32)),
        params=SimpleNamespace(),
        time_cfg=ExplicitTimeConfig(dt=0.1, t_max=0.2, sample_stride=1, fixed_dt=True),
        terms=LinearTerms(),
        mode_method="z_index",
        ky_index=0,
        kx_index=0,
        reference_times=np.asarray([0.2], dtype=float),
        output_steps=np.asarray([0], dtype=int),
    )

    np.testing.assert_allclose(
        captured["phi_prev"], np.zeros((1, 1, 1), dtype=np.complex64)
    )
    np.testing.assert_allclose(
        captured["phi"], np.full((1, 1, 1), 2.0, dtype=np.complex64)
    )
    assert np.isclose(float(captured["dt"]), 0.2)


def test_integrate_target_mode_series_downsamples_output_without_sparsifying_growth_interval(
    monkeypatch,
) -> None:
    monkeypatch.setattr(imported_linear.jax, "jit", lambda fn, donate_argnums=None: fn)
    monkeypatch.setattr(
        imported_linear, "ensure_flux_tube_geometry_data", lambda geom, _theta: geom
    )
    monkeypatch.setattr(
        imported_linear,
        "assemble_rhs_cached",
        lambda *_args, **_kwargs: (
            None,
            SimpleNamespace(phi=jnp.zeros((1, 1, 1), dtype=jnp.complex64), apar=None),
        ),
    )

    step_count = {"n": 0}

    def _fake_step(G_state, *_args, **_kwargs):
        step_count["n"] += 1
        phi_val = float(step_count["n"])
        phi = jnp.full((1, 1, 1), phi_val + 1.0j * phi_val, dtype=jnp.complex64)
        return G_state, SimpleNamespace(phi=phi, apar=None)

    monkeypatch.setattr(imported_linear, "_linear_explicit_step", _fake_step)
    growth_calls: list[tuple[np.ndarray, np.ndarray, float]] = []

    def _fake_growth(phi, phi_prev, dt_step, **_kwargs):
        growth_calls.append((np.asarray(phi), np.asarray(phi_prev), float(dt_step)))
        n = len(growth_calls)
        return (
            jnp.full((1, 1), float(n), dtype=jnp.float32),
            jnp.full((1, 1), 10.0 * float(n), dtype=jnp.float32),
        )

    monkeypatch.setattr(
        imported_linear, "_instantaneous_growth_rate_step", _fake_growth
    )
    monkeypatch.setattr(
        imported_linear,
        "_distribution_free_energy_by_ky",
        lambda *_args, **_kwargs: jnp.asarray([1.0]),
    )
    monkeypatch.setattr(
        imported_linear,
        "_electrostatic_field_energy_by_ky",
        lambda *_args, **_kwargs: jnp.asarray([1.0]),
    )
    monkeypatch.setattr(
        imported_linear,
        "_magnetic_vector_potential_energy_by_ky",
        lambda *_args, **_kwargs: jnp.asarray([0.0]),
    )
    monkeypatch.setattr(
        imported_linear,
        "_linear_frequency_bound",
        lambda *_args, **_kwargs: np.asarray([0.0, 0.0, 0.0]),
    )

    gamma, omega, *_rest = _integrate_target_mode_series(
        G0=jnp.zeros((1, 1, 1, 1, 1, 1), dtype=jnp.complex64),
        grid=SimpleNamespace(dealias_mask=np.ones((1, 1), dtype=bool), z=np.arange(1)),
        geom=SimpleNamespace(
            s_hat=0.0,
            gradpar=lambda: 1.0,
            metric_coeffs=lambda theta: (
                jnp.ones_like(theta),
                jnp.zeros_like(theta),
                jnp.ones_like(theta),
            ),
            drift_coeffs=lambda theta: (
                jnp.zeros_like(theta),
                jnp.zeros_like(theta),
                jnp.zeros_like(theta),
                jnp.zeros_like(theta),
            ),
        ),
        cache=SimpleNamespace(jacobian=jnp.ones(1, dtype=jnp.float32)),
        params=SimpleNamespace(),
        time_cfg=ExplicitTimeConfig(dt=0.1, t_max=0.3, sample_stride=1, fixed_dt=True),
        terms=LinearTerms(),
        mode_method="z_index",
        ky_index=0,
        kx_index=0,
        reference_times=np.asarray([0.1, 0.2, 0.3], dtype=float),
        output_steps=np.asarray([2], dtype=int),
    )

    assert len(growth_calls) == 3
    np.testing.assert_allclose(
        growth_calls[-1][1], np.full((1, 1, 1), 2.0 + 2.0j, dtype=np.complex64)
    )
    np.testing.assert_allclose(
        growth_calls[-1][0], np.full((1, 1, 1), 3.0 + 3.0j, dtype=np.complex64)
    )
    assert np.isclose(growth_calls[-1][2], 0.1)
    np.testing.assert_allclose(gamma, np.asarray([3.0], dtype=float))
    np.testing.assert_allclose(omega, np.asarray([30.0], dtype=float))


def test_write_scan_rows_checkpoints_sorted_csv(tmp_path: Path) -> None:
    out = tmp_path / "scan.csv"
    df = _write_scan_rows(
        [
            {"ky": 0.3, "mean_abs_gamma": 3.0},
            {"ky": 0.1, "mean_abs_gamma": 1.0},
        ],
        out,
    )
    assert list(df["ky"]) == [0.1, 0.3]
    saved = np.genfromtxt(out, delimiter=",", names=True)
    np.testing.assert_allclose(
        np.asarray(saved["ky"], dtype=float), np.asarray([0.1, 0.3], dtype=float)
    )


# ---- test_compare_gx_nonlinear_diagnostics.py ----


def _write_minimal_gx_nc(path: Path, ntime: int = 5) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    with Dataset(path, "w") as root:
        root.createDimension("time", ntime)
        root.createDimension("species", 2)
        grids = root.createGroup("Grids")
        diags = root.createGroup("Diagnostics")

        tvar = grids.createVariable("time", "f8", ("time",))
        tvar[:] = np.linspace(0.0, 1.0, ntime)

        phi2 = diags.createVariable("Phi2_t", "f8", ("time",))
        phi2[:] = np.linspace(0.1, 0.2, ntime)

        for name in ["Wg_st", "Wphi_st", "HeatFlux_st", "ParticleFlux_st"]:
            var = diags.createVariable(name, "f8", ("time", "species"))
            series = np.linspace(0.1, 0.2, ntime)[:, None]
            var[:, :] = np.concatenate([series, 2.0 * series], axis=1)

        wapar = diags.createVariable("Wapar_st", "f8", ("time", "species"))
        wapar[:, :] = np.repeat(np.linspace(0.3, 0.4, ntime)[:, None], 2, axis=1)


def _write_minimal_gkx_nc(path: Path, ntime: int = 5) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    with Dataset(path, "w") as root:
        root.createDimension("time", ntime)
        root.createDimension("species", 2)
        grids = root.createGroup("Grids")
        diags = root.createGroup("Diagnostics")

        tvar = grids.createVariable("time", "f8", ("time",))
        tvar[:] = np.linspace(0.0, 1.0, ntime)

        phi2 = diags.createVariable("Phi2_t", "f8", ("time",))
        phi2[:] = np.linspace(0.2, 0.3, ntime)

        for name in ["Wg_st", "Wphi_st", "HeatFlux_st", "ParticleFlux_st"]:
            var = diags.createVariable(name, "f8", ("time", "species"))
            series = np.linspace(0.2, 0.3, ntime)[:, None]
            var[:, :] = np.concatenate([series, 2.0 * series], axis=1)

        wapar = diags.createVariable("Wapar_st", "f8", ("time", "species"))
        wapar[:, :] = np.repeat(np.linspace(0.4, 0.5, ntime)[:, None], 2, axis=1)


def _write_minimal_gkx_csv(path: Path, ntime: int = 5) -> None:
    t = np.linspace(0.0, 1.0, ntime)
    data = np.column_stack(
        [
            t,
            np.zeros_like(t),  # gamma
            np.zeros_like(t),  # omega
            np.linspace(0.1, 0.2, ntime),  # Wg
            np.linspace(0.2, 0.3, ntime),  # Wphi
            np.linspace(0.3, 0.4, ntime),  # Wapar
            np.linspace(0.6, 0.9, ntime),  # energy
            np.linspace(0.01, 0.02, ntime),  # heat flux
            np.linspace(0.03, 0.04, ntime),  # particle flux
        ]
    )
    header = "t,gamma,omega,Wg,Wphi,Wapar,energy,heat_flux,particle_flux"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def test_compare_gx_nonlinear_diagnostics_plot(tmp_path: Path) -> None:
    pytest.importorskip("netCDF4")
    os.environ.setdefault("MPLBACKEND", "Agg")

    gx_path = tmp_path / "gx.out.nc"
    sp_path = tmp_path / "gkx.csv"
    out_path = tmp_path / "diag_compare.png"
    summary_path = tmp_path / "diag_compare.summary.json"

    _write_minimal_gx_nc(gx_path)
    _write_minimal_gkx_csv(sp_path)

    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear as mod

        argv = [
            "compare_gx_nonlinear_diagnostics.py",
            "--gx",
            str(gx_path),
            "--gkx",
            str(sp_path),
            "--tmin",
            "0.25",
            "--tmax",
            "1.0",
            "--out",
            str(out_path),
            "--summary-json",
            str(summary_path),
            "--summary-case",
            "cyclone_nonlinear_window",
            "--summary-source",
            "minimal GX fixture",
            "--gate-mean-rel",
            "2.0",
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            assert mod.run_diagnostics() == 0
        finally:
            sys.argv = old_argv
    finally:
        sys.path.remove(str(tools_dir))

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["gate_mean_rel"] == 2.0
    assert summary["case"] == "cyclone_nonlinear_window"
    assert summary["source"] == "minimal GX fixture"
    assert summary["tmin"] == 0.25
    assert summary["tmax"] == 1.0
    assert summary["gate_report"]["case"] == "cyclone_nonlinear_window"
    assert summary["gate_report"]["source"] == "minimal GX fixture"
    assert {row["metric"] for row in summary["summary"]} >= {"Wg", "Wphi", "HeatFlux"}
    assert isinstance(summary["gate_passed"], bool)
    assert "Infinity" not in summary_path.read_text(encoding="utf-8")
    assert "NaN" not in summary_path.read_text(encoding="utf-8")


def test_compare_gx_nonlinear_diagnostics_uses_single_species_wapar(
    tmp_path: Path,
) -> None:
    pytest.importorskip("netCDF4")

    gx_path = tmp_path / "gx.out.nc"
    _write_minimal_gx_nc(gx_path)

    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear as mod

        loaded = mod._load_gx_diag(gx_path)
    finally:
        sys.path.remove(str(tools_dir))

    t = np.linspace(0.0, 1.0, 5)
    assert np.allclose(loaded["Wg"], 3.0 * np.linspace(0.1, 0.2, 5))
    assert np.allclose(loaded["Wphi"], 3.0 * np.linspace(0.1, 0.2, 5))
    assert np.allclose(loaded["heat_flux"], 3.0 * np.linspace(0.1, 0.2, 5))
    assert np.allclose(loaded["particle_flux"], 3.0 * np.linspace(0.1, 0.2, 5))
    assert np.allclose(loaded["Wapar"], np.linspace(0.3, 0.4, 5))
    assert np.allclose(loaded["t"], t)


def test_compare_gx_nonlinear_diagnostics_loads_gkx_out_nc(tmp_path: Path) -> None:
    pytest.importorskip("netCDF4")

    gkx_path = tmp_path / "gkx.out.nc"
    _write_minimal_gkx_nc(gkx_path)

    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear as mod

        loaded = mod._load_gkx(gkx_path)
    finally:
        sys.path.remove(str(tools_dir))

    t = np.linspace(0.0, 1.0, 5)
    assert np.allclose(loaded["t"], t)
    assert np.allclose(loaded["phi2"], np.linspace(0.2, 0.3, 5))
    assert np.allclose(loaded["Wg"], 3.0 * np.linspace(0.2, 0.3, 5))
    assert np.allclose(loaded["Wphi"], 3.0 * np.linspace(0.2, 0.3, 5))
    assert np.allclose(loaded["heat_flux"], 3.0 * np.linspace(0.2, 0.3, 5))
    assert np.allclose(loaded["particle_flux"], 3.0 * np.linspace(0.2, 0.3, 5))
    assert np.allclose(loaded["Wapar"], np.linspace(0.4, 0.5, 5))


def test_compare_gx_nonlinear_diagnostics_interp_summary() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear as mod

        mean_rel, max_rel, final_rel = mod._interp_summary(
            np.array([0.0, 1.0, 2.0]),
            np.array([2.0, 4.0, 6.0]),
            np.array([0.0, 2.0]),
            np.array([1.0, 3.0]),
        )
    finally:
        sys.path.remove(str(tools_dir))

    assert np.isclose(mean_rel, 1.0)
    assert np.isclose(max_rel, 1.0)
    assert np.isclose(final_rel, 1.0)


def test_compare_gx_nonlinear_diagnostics_apply_time_window() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear as mod

        series = {
            "t": np.array([0.0, 1.0, 2.0, 3.0]),
            "Wg": np.array([10.0, 11.0, 12.0, 13.0]),
        }
        windowed = mod._apply_time_window(series, tmin=1.0, tmax=2.0)
    finally:
        sys.path.remove(str(tools_dir))

    assert np.allclose(windowed["t"], [1.0, 2.0])
    assert np.allclose(windowed["Wg"], [11.0, 12.0])


# ---- test_compare_gx_nonlinear_terms.py ----


def test_compare_gx_nonlinear_terms_parser_accepts_runtime_config() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear as mod
    finally:
        sys.path.remove(str(tools_dir))

    parser = mod.build_terms_parser()
    args = parser.parse_args(
        [
            "--gx-dir",
            "gx_dump",
            "--gx-out",
            "gx.out.nc",
            "--config",
            "runtime.toml",
            "--ky",
            "0.4",
        ]
    )

    assert args.gx_dir == Path("gx_dump")
    assert args.gx_out == Path("gx.out.nc")
    assert args.config == Path("runtime.toml")
    assert args.ky == 0.4


def test_build_runtime_compare_context_overrides_grid_from_dump(monkeypatch) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear as mod
    finally:
        sys.path.remove(str(tools_dir))

    cfg = SimpleNamespace(grid=SimpleNamespace(Nx=8, Ny=8, Nz=8, y0=None))
    captured: dict[str, object] = {}

    monkeypatch.setattr(mod, "load_runtime_from_toml", lambda _path: (cfg, None))
    monkeypatch.setattr(
        mod,
        "replace",
        lambda obj, **updates: SimpleNamespace(**(obj.__dict__ | updates)),
    )

    def _fake_build_runtime_geometry(cfg_use):
        captured["cfg_use"] = cfg_use
        return "geom"

    monkeypatch.setattr(mod, "build_runtime_geometry", _fake_build_runtime_geometry)
    monkeypatch.setattr(
        mod, "apply_imported_geometry_grid_defaults", lambda _geom, grid: grid
    )
    grid_obj = SimpleNamespace(
        ky=np.array([0.0, 0.2, -0.2]), kx=np.array([0.0]), z=np.array([0.0, 1.0])
    )
    monkeypatch.setattr(mod, "build_spectral_grid", lambda _grid: grid_obj)
    monkeypatch.setattr(
        mod, "build_runtime_linear_params", lambda *_args, **_kwargs: "params"
    )
    monkeypatch.setattr(mod, "build_runtime_term_config", lambda _cfg: "terms")

    cfg_use, geom, grid, params, term_cfg = mod._build_runtime_compare_context(
        Path("runtime.toml"),
        nx=3,
        ny_full=6,
        nz=5,
        nl=2,
        nm=4,
        ky_vals_nyc=np.array([0.2, 0.4], dtype=float),
        y0_override=None,
    )

    assert cfg_use.grid.Nx == 3
    assert cfg_use.grid.Ny == 6
    assert cfg_use.grid.Nz == 5
    assert cfg_use.grid.y0 == 5.0
    assert captured["cfg_use"] is cfg_use
    assert geom == "geom"
    assert grid is grid_obj
    assert params == "params"
    assert term_cfg == "terms"


def test_pick_species_dump_prefers_species_suffix(tmp_path: Path) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear as mod
    finally:
        sys.path.remove(str(tools_dir))

    suffixed = tmp_path / "nl_total_s0.bin"
    plain = tmp_path / "nl_total.bin"
    suffixed.write_bytes(b"s")
    plain.write_bytes(b"p")

    picked = mod._pick_species_dump(tmp_path, "nl_total", 0)

    assert picked == suffixed


def test_pick_first_existing_uses_diag_state_kxky_fallback(tmp_path: Path) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear as mod
    finally:
        sys.path.remove(str(tools_dir))

    diag_kx = tmp_path / "diag_state_kx_t23.bin"
    diag_ky = tmp_path / "diag_state_ky_t23.bin"
    diag_kx.write_bytes(b"kx")
    diag_ky.write_bytes(b"ky")

    picked_kx = mod._pick_first_existing(
        tmp_path / "nl_kx.bin", *sorted(tmp_path.glob("diag_state_kx_t*.bin"))
    )
    picked_ky = mod._pick_first_existing(
        tmp_path / "nl_ky.bin", *sorted(tmp_path.glob("diag_state_ky_t*.bin"))
    )

    assert picked_kx == diag_kx
    assert picked_ky == diag_ky


def test_resolve_dealias_mask_rebuilds_to_compared_shape() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear as mod
    finally:
        sys.path.remove(str(tools_dir))

    mask = mod._resolve_dealias_mask(np.ones((4, 4), dtype=bool), ny=10, nx=4)

    assert mask.shape == (10, 4)


def test_synth_positive_and_full_ky_rebuild_dump_grid() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear as mod
    finally:
        sys.path.remove(str(tools_dir))

    ky_pos = mod._synth_positive_ky(nyc=6, y0=10.0)
    ky_full = mod._synth_full_ky(nyc=6, y0=10.0)

    assert np.allclose(ky_pos, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    assert np.allclose(ky_full, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, -0.4, -0.3, -0.2, -0.1])


# ---- test_compare_gx_rhs_terms.py ----

from gkx.benchmarking_shared import (
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
)
from scripts.comparison.reference_params import (
    _build_initial_condition,
    _two_species_params,
)
from gkx.config import KBMBaseCase
from gkx.core_grid import select_ky_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.terms.assembly import assemble_rhs_terms_cached, compute_fields_cached
from gkx.workflows.runtime.toml import load_runtime_from_toml


def test_manual_linear_contributions_match_assembly_for_multispecies_kbm() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_rhs_terms as mod
    finally:
        sys.path.remove(str(tools_dir))

    cfg = KBMBaseCase(
        grid=GridConfig(
            Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=8, nperiod=2
        )
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = _two_species_params(
        cfg.model,
        kpar_scale=float(geom.gradpar()),
        omega_d_scale=KBM_OMEGA_D_SCALE,
        omega_star_scale=KBM_OMEGA_STAR_SCALE,
        rho_star=KBM_RHO_STAR,
        nhermite=6,
    )
    grid_full = build_spectral_grid(cfg.grid)
    ky_idx = int(np.argmin(np.abs(np.asarray(grid_full.ky) - 0.3)))
    grid = select_ky_grid(grid_full, ky_idx)
    cache = build_linear_cache(grid, geom, params, 4, 6)

    G0_single = _build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=4,
        Nm=6,
        init_cfg=cfg.init,
    )
    G = np.zeros((2, 4, 6, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G[1] = np.asarray(G0_single, dtype=np.complex64)
    G_j = jnp.asarray(G)

    term_cfg = TermConfig(hypercollisions=0.0, end_damping=0.0, bpar=0.0)
    rhs_total, fields_ref, contrib_ref = assemble_rhs_terms_cached(
        G_j, cache, params, terms=term_cfg
    )
    fields = compute_fields_cached(
        G_j, cache, params, terms=term_cfg, use_custom_vjp=False
    )
    fields_manual, contrib_manual = mod._manual_linear_contributions_from_fields(
        G_j,
        cache,
        params,
        term_cfg,
        phi=np.asarray(fields.phi),
        apar=np.asarray(fields.apar),
        bpar=np.asarray(
            fields.bpar if fields.bpar is not None else np.zeros_like(fields.phi)
        ),
    )

    assert np.allclose(np.asarray(fields_manual.phi), np.asarray(fields_ref.phi))
    assert np.allclose(np.asarray(fields_manual.apar), np.asarray(fields_ref.apar))
    for key in (
        "streaming",
        "mirror",
        "curvature",
        "gradb",
        "diamagnetic",
        "collisions",
    ):
        assert np.allclose(
            np.asarray(contrib_manual[key]), np.asarray(contrib_ref[key])
        )
    contrib_sum = sum(np.asarray(contrib_manual[key]) for key in contrib_manual)
    assert np.allclose(contrib_sum, np.asarray(rhs_total))


def test_compare_gx_rhs_terms_parser_defaults_to_dump_metadata() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_rhs_terms as mod
    finally:
        sys.path.remove(str(tools_dir))

    parser = mod.build_parser()
    args = parser.parse_args(["--gx-dir", "/tmp/gx", "--gx-out", "/tmp/gx.out.nc"])

    assert args.Nl is None
    assert args.Nm is None
    assert args.y0 is None


def test_compare_gx_rhs_terms_parser_accepts_runtime_config() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_rhs_terms as mod
    finally:
        sys.path.remove(str(tools_dir))

    parser = mod.build_parser()
    args = parser.parse_args(
        [
            "--gx-dir",
            "/tmp/gx",
            "--gx-out",
            "/tmp/gx.out.nc",
            "--config",
            "/tmp/runtime.toml",
        ]
    )

    assert args.config == Path("/tmp/runtime.toml")


def test_compare_gx_rhs_terms_parser_accepts_imported_geometry_args() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_rhs_terms as mod
    finally:
        sys.path.remove(str(tools_dir))

    parser = mod.build_parser()
    args = parser.parse_args(
        [
            "--gx-dir",
            "/tmp/gx",
            "--gx-out",
            "/tmp/gx.out.nc",
            "--gx-input",
            "/tmp/gx.in",
            "--geometry-file",
            "/tmp/geom.nc",
        ]
    )

    assert args.gx_input == Path("/tmp/gx.in")
    assert args.geometry_file == Path("/tmp/geom.nc")


def test_compare_gx_rhs_terms_runtime_context_overrides_grid_from_dump(
    monkeypatch,
) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "scripts" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_rhs_terms as mod
    finally:
        sys.path.remove(str(tools_dir))

    cfg = type(
        "Cfg", (), {"grid": type("Grid", (), {"Nx": 8, "Ny": 8, "Nz": 8, "y0": None})()}
    )()
    captured: dict[str, object] = {}

    monkeypatch.setattr(mod, "load_runtime_from_toml", lambda _path: (cfg, None))
    monkeypatch.setattr(
        mod,
        "replace",
        lambda obj, **updates: type("Obj", (), obj.__dict__ | updates)(),
    )

    def _fake_build_runtime_geometry(cfg_use):
        captured["cfg_use"] = cfg_use
        return "geom"

    monkeypatch.setattr(mod, "build_runtime_geometry", _fake_build_runtime_geometry)
    monkeypatch.setattr(
        mod, "apply_imported_geometry_grid_defaults", lambda _geom, grid: grid
    )
    grid_obj = type(
        "GridObj", (), {"ky": np.array([0.0, 0.2, -0.2]), "kx": np.array([0.0])}
    )()
    monkeypatch.setattr(mod, "build_spectral_grid", lambda _grid: grid_obj)
    monkeypatch.setattr(
        mod, "build_runtime_linear_params", lambda *_args, **_kwargs: "params"
    )
    monkeypatch.setattr(
        mod,
        "build_runtime_term_config",
        lambda _cfg: TermConfig(hypercollisions=1.0, end_damping=1.0),
    )

    cfg_use, geom, grid_full, params, term_cfg = mod._build_runtime_compare_context(
        Path("runtime.toml"),
        nx=3,
        ny_full=6,
        nz=5,
        nm=4,
        ky_vals=np.array([0.2, 0.4], dtype=float),
        y0_override=None,
    )

    assert cfg_use.grid.Nx == 3
    assert cfg_use.grid.Ny == 6
    assert cfg_use.grid.Nz == 5
    assert cfg_use.grid.y0 == 5.0
    assert captured["cfg_use"] is cfg_use
    assert geom == "geom"
    assert grid_full is grid_obj
    assert params == "params"
    assert term_cfg.hypercollisions == 0.0
    assert term_cfg.end_damping == 0.0


def _runtime_config_from_kbm_case(cfg: KBMBaseCase) -> RuntimeConfig:
    """The shipped KBM runtime deck with one parsed case's grid, geometry and drives."""

    runtime_cfg, _raw = load_runtime_from_toml(
        Path(__file__).resolve().parents[3]
        / "examples/06_electromagnetic/case_full.toml"
    )
    ion, electron = runtime_cfg.species
    model = cfg.model
    return replace(
        runtime_cfg,
        grid=cfg.grid,
        geometry=cfg.geometry,
        init=cfg.init,
        physics=replace(runtime_cfg.physics, beta=float(model.beta)),
        species=(
            replace(ion, tprim=float(model.tprim_i), fprim=float(model.fprim)),
            replace(
                electron,
                mass=1.0 / float(model.mass_ratio),
                temperature=float(model.Te_over_Ti),
                tprim=float(model.tprim_e),
                fprim=float(model.fprim),
            ),
        ),
    )


def test_runtime_linear_accepts_vmec_and_desc_eik_geometry_aliases(
    tmp_path: Path,
) -> None:
    from gkx.runtime import (
        run_runtime_linear,
    )

    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    grid = GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=8, nperiod=2)
    cfg = KBMBaseCase(grid=grid)
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, grid.Nz + 1)
    analytic = SAlphaGeometry.from_config(cfg.geometry)
    sampled = sample_flux_tube_geometry(analytic, theta)
    path = tmp_path / "geom.eik.nc"
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = np.asarray(sampled.bmag_profile)
        root.createVariable("gds2", "f8", ("z",))[:] = np.asarray(sampled.gds2_profile)
        root.createVariable("gds21", "f8", ("z",))[:] = np.asarray(
            sampled.gds21_profile
        )
        root.createVariable("gds22", "f8", ("z",))[:] = np.asarray(
            sampled.gds22_profile
        )
        root.createVariable("cvdrift", "f8", ("z",))[:] = np.asarray(sampled.cv_profile)
        root.createVariable("gbdrift", "f8", ("z",))[:] = np.asarray(sampled.gb_profile)
        root.createVariable("cvdrift0", "f8", ("z",))[:] = np.asarray(
            sampled.cv0_profile
        )
        root.createVariable("gbdrift0", "f8", ("z",))[:] = np.asarray(
            sampled.gb0_profile
        )
        root.createVariable("jacob", "f8", ("z",))[:] = np.asarray(
            sampled.jacobian_profile
        )
        root.createVariable("grho", "f8", ("z",))[:] = np.asarray(sampled.grho_profile)
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(
            theta.size, sampled.gradpar_value
        )
        root.createVariable("q", "f8", ())[:] = sampled.q
        root.createVariable("shat", "f8", ())[:] = sampled.s_hat
        root.createVariable("Rmaj", "f8", ())[:] = sampled.R0
        root.createVariable("kxfac", "f8", ())[:] = sampled.kxfac
        root.createVariable("scale", "f8", ())[:] = sampled.theta_scale
        root.createVariable("nfp", "f8", ())[:] = sampled.nfp
        root.createVariable("alpha", "f8", ())[:] = sampled.alpha

    for model in ("vmec-eik", "desc-eik"):
        cfg_nc = replace(
            cfg,
            geometry=replace(
                cfg.geometry,
                model=model,
                geometry_file=str(path),
            ),
        )
        runtime_cfg = _runtime_config_from_kbm_case(cfg_nc)
        result = run_runtime_linear(
            runtime_cfg,
            ky_target=0.3,
            Nl=4,
            Nm=6,
            dt=0.01,
            steps=40,
            solver="explicit_time",
            sample_stride=2,
        )
        assert np.isfinite(result.gamma)
        assert np.isfinite(result.omega)


def test_rhs_term_diagnostics_etg_uses_canonical_runtime_contract() -> None:
    from scripts.comparison import compare_gx_rhs_terms as mod

    args = type(
        "Args",
        (),
        {
            "Nx": 1,
            "Ny": 8,
            "Nz": 16,
            "Lx": 6.28,
            "Ly": 6.28,
            "boundary": "linked",
            "y0": 0.2,
            "ntheta": 8,
            "nperiod": 1,
            "Nm": 4,
            "drift_scale": 1.0,
            "tprim_e": 6.0,
        },
    )()
    cfg, params, species_index, drift_scale, drive_scale, rho_scale = mod._case_config(
        "etg", args
    )

    assert cfg.species[0].tprim == pytest.approx(6.0)
    assert cfg.physics.adiabatic_ions is True
    assert species_index == 0
    assert float(params.tau_e) == pytest.approx(1.0)
    assert (drift_scale, drive_scale, rho_scale) == pytest.approx((1.0, 1.0, 1.0))


def test_write_rhs_term_diagnostics_seed_state_handles_multispecies_tem() -> None:
    from scripts.comparison import compare_gx_rhs_terms as mod

    args = type(
        "Args",
        (),
        {
            "Nx": 1,
            "Ny": 8,
            "Nz": 24,
            "Lx": 62.8,
            "Ly": 62.8,
            "boundary": "linked",
            "y0": 10.0,
            "ntheta": 8,
            "nperiod": 2,
            "Nm": 6,
            "drift_scale": 1.0,
        },
    )()
    cfg, params, init_species_index, *_ = mod._case_config("tem", args)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid_full = build_spectral_grid(cfg.grid)
    ky_idx = int(np.argmin(np.abs(np.asarray(grid_full.ky) - 0.3)))
    grid = select_ky_grid(grid_full, ky_idx)

    G0 = mod._build_seed_state(
        cfg=cfg,
        geom=geom,
        grid=grid,
        params=params,
        Nl=4,
        Nm=6,
        init_species_index=init_species_index,
    )

    assert np.asarray(G0).shape == (2, 4, 6, grid.ky.size, grid.kx.size, grid.z.size)
    assert np.any(np.abs(np.asarray(G0[1])) > 0.0)
    assert np.allclose(np.asarray(G0[0]), 0.0)


# ---- test_reference_comparison_contracts.py ----

from support.paths import load_comparison_tool  # noqa: E402


def test_imported_window_parser_accepts_required_args() -> None:
    mod = load_comparison_tool("compare_gx_imported_linear")
    args = mod.build_window_parser().parse_args(
        [
            "--gx-dir",
            "/tmp/gx",
            "--gx-out",
            "/tmp/run.out.nc",
            "--gx-input",
            "/tmp/run.in",
            "--geometry-file",
            "/tmp/run.eik.nc",
            "--time-index-start",
            "0",
            "--time-index-stop",
            "1",
        ]
    )
    assert args.gx_dir == Path("/tmp/gx")
    assert args.gx_out == Path("/tmp/run.out.nc")
    assert args.gx_input == Path("/tmp/run.in")
    assert args.geometry_file == Path("/tmp/run.eik.nc")
    assert args.time_index_start == 0
    assert args.time_index_stop == 1


# ---- from test_nonlinear_gradient_evidence_contracts.py ----
# The nonlinear-autodiff claims must stay attached to their generators.
#
# Commits 612e1311 and a7b41968 removed the tools and the JSON that produced the
# headline adjoint numbers, leaving the docs page and the figure resting on
# literals typed into the plotting script. These tests pin the other direction:
# the tracked measurements exist, the figure reads them, the generators that write


STATIC = REPO_ROOT / "docs" / "_static"
LADDER = STATIC / "nonlinear_heat_flux_gradient_window_rk3.json"
PARITY = STATIC / "nonlinear_window_device_parity.json"
MEMORY = (
    STATIC / "nonlinear_adjoint_checkpointing_cpu32.json",
    STATIC / "nonlinear_adjoint_checkpointing_gpu32.json",
)
GENERATORS = (
    "scripts/campaigns/nonlinear_gradient_window.py",
    "scripts/profiling/profile_nonlinear_adjoint_checkpointing.py",
    "scripts/profiling/profile_nonlinear_window_device_parity.py",
)


def _ladder_tool():
    return load_tool_script("campaigns", "nonlinear_gradient_window")


def test_gradient_ladder_requires_compatible_clean_state_source(
    monkeypatch,
) -> None:
    tool = _ladder_tool()
    provenance = {
        "repository_root": str(REPO_ROOT),
        "git_commit": "current",
        "git_dirty": False,
    }

    assert (
        tool._require_compatible_state_source(
            {
                "gkx_git_commit": np.asarray("current"),
                "gkx_git_dirty": np.asarray(0),
            },
            provenance,
        )
        == "current"
    )
    with pytest.raises(SystemExit, match="no GKX source provenance"):
        tool._require_compatible_state_source({}, provenance)
    monkeypatch.setattr(tool, "_gkx_source_tree_matches", lambda *_args: False)
    with pytest.raises(SystemExit, match="differs from current source"):
        tool._require_compatible_state_source(
            {
                "gkx_git_commit": np.asarray("old"),
                "gkx_git_dirty": np.asarray(0),
            },
            provenance,
        )


@pytest.mark.parametrize("relative", GENERATORS)
def test_generator_scripts_are_present(relative: str) -> None:
    assert (REPO_ROOT / relative).is_file()


def test_gradient_window_imports_from_the_repository_package() -> None:
    """The profiler imports the campaign through ``scripts.campaigns``."""

    command = (
        "import sys; "
        f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
        "from scripts.campaigns.nonlinear_gradient_window import build_window_case; "
        "assert callable(build_window_case)"
    )
    subprocess.run([sys.executable, "-I", "-c", command], check=True)


def test_gradient_window_nz_override_wins_over_shipped_ntheta() -> None:
    case = _ladder_tool().build_window_case(
        REPO_ROOT / "benchmarks" / "cases/cyclone_nonlinear_t400.toml",
        {"Nx": 6, "Ny": 4, "Nz": 10},
    )

    # The ky extent is the stored row count of the deck's own layout: Ny = 4
    # rows on the two-sided axis, Nyc = 3 on the half one. Either way it is
    # the Ny override, not the shipped deck's Ny, that sets it.
    grid = case["grid"]
    ky_rows = rows_for_layout(4, source_ky_layout(grid))
    assert int(grid.ky.size) == ky_rows
    assert case["shape"][-3:] == (ky_rows, 6, 10)
    assert grid.z.size == 10


def test_docs_page_names_every_generator() -> None:
    page = (REPO_ROOT / "docs" / "nonlinear_autodiff.rst").read_text()
    for relative in GENERATORS:
        assert relative in page, f"{relative} is not documented as regenerable"


def test_figure_builder_reads_measurements_rather_than_literals() -> None:
    module = load_artifact_tool("build_nonlinear_autodiff_figure")
    assert module.LADDER == LADDER
    assert {path for _label, path in module.MEMORY_PROFILES} == set(MEMORY)
    for path in (module.LADDER, *(p for _l, p in module.MEMORY_PROFILES)):
        assert path.is_file(), f"missing tracked measurement {path}"
    # The knee shading on the figure and the knee the ladder reports have to be
    # the same threshold, or the picture and the number disagree.
    assert module.TOLERANCE == _ladder_defaults()["tolerance"]


def _ladder_defaults() -> dict:
    parser = _ladder_tool().build_parser()
    return {action.dest: action.default for action in parser._actions}


def test_tracked_ladder_shows_the_knee_the_runtime_guard_uses() -> None:
    ladder = json.loads(LADDER.read_text())
    tolerance = _ladder_defaults()["tolerance"]
    rows = [
        dict(row, agrees=row["ad_fd_relative_error"] <= tolerance)
        for row in ladder["rows"]
    ]
    knee = _ladder_tool().locate_knee(rows)
    assert knee["divergence_knee_steps"] == DIVERGENCE_KNEE_STEPS
    assert knee["knee_bracket"] == [DIVERGENCE_KNEE_STEPS, 2 * DIVERGENCE_KNEE_STEPS]


def test_tracked_ladder_agrees_with_finite_differences_below_the_knee() -> None:
    ladder = json.loads(LADDER.read_text())
    below = [r for r in ladder["rows"] if r["window"] <= DIVERGENCE_KNEE_STEPS]
    above = [r for r in ladder["rows"] if r["window"] > DIVERGENCE_KNEE_STEPS]
    assert below and above
    assert max(r["ad_fd_relative_error"] for r in below) < 1.0e-8
    assert min(r["ad_fd_relative_error"] for r in above) > 1.0e-6
    # A ladder that never diverges would satisfy the two bounds above only by
    # accident; require the gradient itself to take off past the knee.
    assert above[0]["abs_gradient"] > 10.0 * below[-1]["abs_gradient"]


@pytest.mark.parametrize("path", MEMORY)
def test_checkpoint_memory_profiles_show_a_real_reduction(path: Path) -> None:
    profile = json.loads(path.read_text())
    policies = {row["checkpoint"]: row for row in profile["rows"]}
    assert set(policies) == {"step", "block"}
    assert policies["block"]["temp_bytes"] < policies["step"]["temp_bytes"]
    assert profile["temp_reduction"] > 10.0
    # Rematerialization is the trade; a profile claiming free memory would mean
    # the two policies did not compile to different programs.
    assert profile["runtime_ratio"] > 1.0


def test_device_parity_artifact_compares_one_identical_case() -> None:
    parity = json.loads(PARITY.read_text())
    assert len(parity["runs"]) >= 2
    backends = {run["default_backend"] for run in parity["runs"].values()}
    assert {"cpu", "gpu"} <= backends
    comparisons = {(row["left"], row["right"]): row for row in parity["comparisons"]}
    assert comparisons
    for row in comparisons.values():
        assert row["gradient_relative_difference"] < 1.0e-12
        assert row["value_relative_difference"] < 1.0e-12


# ---- from test_nonlinear_sharding_profile_contracts.py ----


def _load_sharding_tool_module():
    return load_profiling_tool("profile_nonlinear_sharding")


def test_profile_nonlinear_sharding_parser_defaults_to_tracked_artifact() -> None:
    mod = _load_sharding_tool_module()
    args = mod.build_parser().parse_args([])

    assert args.out_json == mod.DEFAULT_OUT
    assert args.sharding == "auto"
    assert args.sharding_options is None
    assert args.method == "rk2"
    assert args.warmups == 1
    assert args.repeats == 3
    assert args.allow_unsafe_cpu_state_sharding is False
    assert (
        mod._artifact_path_for_contract(args.out_json)
        == "docs/_static/nonlinear_sharding_profile.json"
    )


def test_profile_nonlinear_sharding_documented_script_entrypoint() -> None:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/profiling/profile_nonlinear_sharding.py"),
            "--help",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "numerical identity gate" in result.stdout


def test_profile_nonlinear_sharding_problem_excites_nonlinear_bracket() -> None:
    mod = _load_sharding_tool_module()
    args = SimpleNamespace(nx=8, ny=8, nz=12, nl=2, nm=3, amplitude=1.0e-4)
    state, cache, params = mod._build_problem(args)
    nonlinear_terms = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
        nonlinear=1.0,
    )
    rhs, _fields = mod.nonlinear_rhs_cached(
        state,
        cache,
        params,
        nonlinear_terms,
        compressed_real_fft=True,
        laguerre_mode="grid",
    )

    assert int(jnp.count_nonzero(jnp.abs(state) > 0.0)) > int(args.nz)
    assert float(jnp.max(jnp.abs(rhs))) > 0.0
    assert (
        mod._initial_nonlinear_activity(state, cache, params, laguerre_mode="grid")
        > 0.0
    )


def test_profile_nonlinear_sharding_source_contract_is_machine_readable(
    tmp_path: Path,
) -> None:
    mod = _load_sharding_tool_module()
    out_json = tmp_path / "profile.json"
    argv = [
        "--out-json",
        str(out_json),
        "--sharding",
        "kx",
        "--warmups",
        "0",
        "--repeats",
        "2",
    ]
    args = mod.build_parser().parse_args(argv)

    contract = mod._source_contract(args, argv, backend="gpu", device_count=2)

    assert contract["backend"] == "gpu"
    assert contract["source_contract_version"] == 1
    assert contract["device_count"] == 2
    assert contract["sharding_axis"] == "kx"
    assert contract["source_artifact"] == str(out_json.resolve())
    assert contract["timing_warmup_repeat"] == {"warmups": 0, "repeats": 2}
    assert contract["allow_unsafe_cpu_state_sharding"] is False
    assert contract["profile_command_argv"][-len(argv) :] == argv
    assert (
        "scripts/profiling/profile_nonlinear_sharding.py" in contract["profile_command"]
    )
    assert {"python", "gkx", "jax", "jaxlib", "numpy"} <= set(
        contract["software_versions"]
    )
    assert all(contract["software_versions"].values())
    assert contract["git_revision"]
    assert isinstance(contract["git_dirty"], bool)


def test_profile_nonlinear_sharding_helpers_report_stats_and_unique_specs() -> None:
    mod = _load_sharding_tool_module()

    stats = mod._time_stats([3.0, 1.0, 2.0])

    assert stats["min"] == 1.0
    assert stats["median"] == 2.0
    assert stats["mean"] == 2.0
    assert stats["max"] == 3.0
    assert mod._sharding_specs("auto", "ky,kx,ky,z") == ["auto", "ky", "kx", "z"]
    assert mod._sharding_specs("auto,kx", None) == ["auto", "kx"]


def test_profile_nonlinear_sharding_reports_best_identity_candidate() -> None:
    mod = _load_sharding_tool_module()

    best = mod._best_identity_preserving_candidate(
        {
            "auto": {
                "identity_gate_pass": True,
                "engineering_speedup_median": 0.8,
                "state_sharding_active": True,
            },
            "kx": {
                "identity_gate_pass": True,
                "engineering_speedup_median": 1.2,
                "state_sharding_active": True,
            },
            "z": {
                "identity_gate_pass": False,
                "engineering_speedup_median": 3.0,
                "state_sharding_active": True,
            },
        }
    )

    assert best == {
        "spec": "kx",
        "engineering_speedup_median": 1.2,
        "state_sharding_active": True,
        "identity_gate_pass": True,
    }


def test_profile_nonlinear_sharding_excludes_inactive_speedup_candidates() -> None:
    mod = _load_sharding_tool_module()

    best = mod._best_identity_preserving_candidate(
        {
            "auto": {
                "identity_gate_pass": True,
                "engineering_speedup_median": 4.0,
                "state_sharding_active": False,
            },
            "kx": {
                "identity_gate_pass": True,
                "engineering_speedup_median": 1.1,
                "state_sharding_active": True,
            },
        }
    )

    assert best["spec"] == "kx"
    assert best["engineering_speedup_median"] == 1.1


def test_profile_nonlinear_sharding_skips_unsafe_cpu_state_sharding() -> None:
    mod = _load_sharding_tool_module()

    assert (
        mod._skip_unsafe_cpu_state_sharding(
            backend="cpu",
            device_count=4,
            state_sharding_active=True,
            allow_unsafe_cpu_state_sharding=False,
        )
        is True
    )
    assert (
        mod._skip_unsafe_cpu_state_sharding(
            backend="cpu",
            device_count=4,
            state_sharding_active=True,
            allow_unsafe_cpu_state_sharding=True,
        )
        is False
    )
    assert (
        mod._skip_unsafe_cpu_state_sharding(
            backend="gpu",
            device_count=2,
            state_sharding_active=True,
            allow_unsafe_cpu_state_sharding=False,
        )
        is False
    )

    row = mod._candidate_failure(
        state_sharding_active=True,
        error=mod.CPU_WHOLE_STATE_SHARDING_SKIP_REASON,
        skip_reason="cpu_whole_state_pjit_sharding_unsafe_for_fft_layout",
    )

    assert row["identity_gate_pass"] is False
    assert row["state_sharding_active"] is True
    assert row["skip_reason"] == "cpu_whole_state_pjit_sharding_unsafe_for_fft_layout"
    assert "unsafe_for_fft_layout" in row["error"]


def test_profile_nonlinear_sharding_diagnostic_metrics_compare_rhs_and_phi(
    monkeypatch,
) -> None:
    mod = _load_sharding_tool_module()

    def fake_rhs(
        state, cache, params, terms, *, compressed_real_fft=True, laguerre_mode="grid"
    ):
        del cache, params, terms, compressed_real_fft, laguerre_mode
        arr = jnp.asarray(state)
        return 2.0 * arr, FieldState(
            phi=jnp.sum(arr, axis=(0, 1)), apar=None, bpar=None
        )

    monkeypatch.setattr(mod, "nonlinear_rhs_cached", fake_rhs)

    reference = jnp.ones((2, 2, 1, 1, 3), dtype=jnp.complex64)
    candidate = reference.at[0, 0, 0, 0, 0].add(1.0e-3)

    metrics = mod._nonlinear_diagnostic_identity_metrics(
        reference,
        candidate,
        cache=object(),
        params=object(),
        terms=object(),
        compressed_real_fft=True,
        laguerre_mode="grid",
    )

    assert metrics["max_abs_rhs_error"] == pytest.approx(2.0e-3, rel=1.0e-4)
    assert metrics["max_abs_phi_error"] == pytest.approx(1.0e-3, rel=1.0e-4)
    assert metrics["max_rel_rhs_error"] > 0.0
    assert metrics["max_rel_phi_error"] > 0.0


# Sweep-driver contracts for the same nonlinear sharding profiling lane.
def _load_sweep_tool_module():
    return load_profiling_tool("profile_nonlinear_sharding")


def test_nonlinear_sharding_sweep_subcommand_parser_defaults_to_bounded_artifact() -> (
    None
):
    mod = _load_sweep_tool_module()

    args = mod.build_sweep_parser().parse_args([])

    assert args.out_prefix == mod.DEFAULT_SWEEP_PREFIX
    assert args.backend == "cpu"
    assert args.devices == [1, 2]
    assert args.sharding_options == "auto,kx"
    assert args.timeout_s == 300.0
    assert args.office_gpu_xlarge is False


def test_nonlinear_sharding_sweep_subcommand_office_gpu_preset_is_canonical() -> None:
    mod = _load_sweep_tool_module()

    args = mod.apply_sweep_preset(
        mod.build_sweep_parser().parse_args(["--office-gpu-xlarge"])
    )

    assert args.backend == "gpu"
    assert args.devices == [1, 2]
    assert (args.nx, args.ny, args.nz, args.nl, args.nm, args.steps) == (
        48,
        96,
        128,
        4,
        8,
        12,
    )
    assert args.sharding_options == "auto,kx"
    assert args.out_prefix == mod.OFFICE_GPU_XLARGE_PREFIX
    assert args.trace is True


def test_nonlinear_sharding_sweep_subcommand_device_env_is_backend_specific() -> None:
    mod = _load_sweep_tool_module()

    cpu_env = mod._device_env({"XLA_FLAGS": "--foo=bar"}, backend="cpu", devices=4)
    replaced_cpu_env = mod._device_env(
        {"XLA_FLAGS": "--foo=bar --xla_force_host_platform_device_count=8"},
        backend="cpu",
        devices=2,
    )
    gpu_env = mod._device_env({}, backend="gpu", devices=2)

    assert cpu_env["JAX_PLATFORMS"] == "cpu"
    assert "--xla_force_host_platform_device_count=4" in cpu_env["XLA_FLAGS"]
    assert (
        "--xla_force_host_platform_device_count=8" not in replaced_cpu_env["XLA_FLAGS"]
    )
    assert "--xla_force_host_platform_device_count=2" in replaced_cpu_env["XLA_FLAGS"]
    assert "--foo=bar" in replaced_cpu_env["XLA_FLAGS"]
    assert gpu_env["JAX_PLATFORMS"] == "cuda"
    assert gpu_env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert gpu_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_nonlinear_sharding_sweep_subcommand_selects_fastest_identity_candidate() -> (
    None
):
    mod = _load_sweep_tool_module()
    payload = {
        "source_contract_version": 1,
        "backend": "gpu",
        "device_count": 2,
        "default_backend": "gpu",
        "sharding_axis": "kx",
        "profile_command": "python scripts/profiling/profile_nonlinear_sharding.py --sharding kx",
        "profile_command_argv": [
            "python",
            "scripts/profiling/profile_nonlinear_sharding.py",
            "--sharding",
            "kx",
        ],
        "source_artifact": "/tmp/profile.json",
        "software_versions": {
            "python": "3.11.0",
            "gkx": "test",
            "jax": "0.test",
            "jaxlib": "0.test",
            "numpy": "2.test",
        },
        "timing_warmup_repeat": {"warmups": 1, "repeats": 3},
        "state_shape": [4, 8, 17, 32, 64],
        "state_sharding_requested": "auto",
        "serial_stats_s": {"median": 10.0},
        "best_identity_preserving_candidate": {"spec": "kx"},
        "sharded_results": {
            "auto": {
                "state_sharding_active": True,
                "stats_s": {"median": 8.0},
                "identity_gate_pass": True,
                "max_abs_state_error": 0.0,
                "max_rel_state_error": 0.0,
                "error": None,
            },
            "kx": {
                "state_sharding_active": True,
                "stats_s": {"median": 5.0},
                "identity_gate_pass": True,
                "max_abs_state_error": 0.0,
                "max_rel_state_error": 0.0,
                "error": None,
            },
        },
    }

    row = mod._row_from_payload(payload, requested_devices=2)

    assert row["best_spec"] == "kx"
    assert row["parallel_median_s"] == 5.0
    assert row["same_process_speedup"] == 2.0
    assert row["identity_gate_pass"] is True
    assert row["source_contract_version"] == 1
    assert row["profile_command"].startswith(
        "python scripts/profiling/profile_nonlinear_sharding.py"
    )
    assert row["profile_command_argv"][-2:] == ["--sharding", "kx"]
    assert row["source_artifact"] == "/tmp/profile.json"
    assert row["software_versions"]["gkx"] == "test"
    assert row["timing_warmup_repeat"] == {"warmups": 1, "repeats": 3}
    assert row["profile_backend"] == "gpu"
    assert row["profile_device_count"] == 2
    assert row["profile_sharding_axis"] == "kx"


def test_nonlinear_sharding_sweep_subcommand_json_clean_replaces_nonfinite() -> None:
    mod = _load_sweep_tool_module()

    cleaned = mod._json_clean({"bad": math.inf, "ok": 1.0})

    assert cleaned == {"bad": None, "ok": 1.0}


def test_nonlinear_sharding_sweep_subcommand_records_timeout_rows(monkeypatch) -> None:
    mod = _load_sweep_tool_module()

    def _raise_timeout(*_args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd="profile",
            timeout=float(kwargs["timeout"]),
            output="stdout tail",
            stderr="stderr tail",
        )

    monkeypatch.setattr(mod.subprocess, "run", _raise_timeout)

    summary = mod.run_sweep(
        backend="cpu",
        devices=[2],
        nx=4,
        ny=4,
        nz=4,
        nl=1,
        nm=1,
        dt=0.02,
        steps=1,
        method="rk2",
        sharding="auto",
        sharding_options="auto,kx",
        laguerre_mode="grid",
        warmups=0,
        repeats=1,
        timeout_s=0.5,
        trace=False,
    )

    assert summary["identity_passed"] is False
    assert summary["speedup_passed"] is False
    assert summary["status"] == "diagnostic_identity_only"
    assert summary["rows"][0]["parallel_median_s"] is None
    assert "timed out" in summary["rows"][0]["error"]
    assert "stderr tail" in summary["rows"][0]["error"]
    assert summary["speedup_blockers"] == ["cpu_2devices_identity_failed"]


def test_nonlinear_sharding_sweep_subcommand_marks_identity_only_slowdown(
    monkeypatch,
) -> None:
    mod = _load_sweep_tool_module()

    def _fake_run(cmd, **_kwargs):
        out_json = Path(cmd[cmd.index("--out-json") + 1])
        device_count = 2 if "2devices" in out_json.name else 1
        spec = "kx" if device_count == 2 else "auto"
        median = 20.0 if device_count == 2 else 10.0
        payload = {
            "device_count": device_count,
            "default_backend": "gpu",
            "state_shape": [1],
            "state_sharding_requested": "auto",
            "serial_stats_s": {"median": 10.0},
            "best_identity_preserving_candidate": {"spec": spec},
            "sharded_results": {
                spec: {
                    "state_sharding_active": device_count > 1,
                    "stats_s": {"median": median},
                    "identity_gate_pass": True,
                    "max_abs_state_error": 0.0,
                    "max_rel_state_error": 0.0,
                    "error": None,
                }
            },
        }
        out_json.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    summary = mod.run_sweep(
        backend="gpu",
        devices=[1, 2],
        nx=4,
        ny=4,
        nz=4,
        nl=1,
        nm=1,
        dt=0.02,
        steps=1,
        method="rk2",
        sharding="auto",
        sharding_options="auto,kx",
        laguerre_mode="grid",
        warmups=0,
        repeats=1,
        timeout_s=1.0,
        trace=False,
    )

    assert summary["identity_passed"] is True
    assert summary["speedup_passed"] is False
    assert summary["status"] == "diagnostic_identity_only"
    assert summary["rows"][1]["strong_speedup_vs_1_device"] == 0.5
    assert summary["speedup_blockers"] == ["gpu_2devices_speedup_0.5_below_1"]


def test_nonlinear_sharding_sweep_subcommand_preserves_failed_profile_json(
    monkeypatch,
) -> None:
    mod = _load_sweep_tool_module()

    def _fake_run(cmd, **_kwargs):
        out_json = Path(cmd[cmd.index("--out-json") + 1])
        payload = {
            "device_count": 4,
            "default_backend": "cpu",
            "state_shape": [4, 8, 17, 32, 64],
            "state_sharding_requested": "auto",
            "serial_stats_s": {"median": 10.0},
            "best_identity_preserving_candidate": {
                "spec": None,
                "identity_gate_pass": False,
                "state_sharding_active": False,
                "engineering_speedup_median": None,
            },
            "sharded_results": {
                "auto": {
                    "state_sharding_active": True,
                    "stats_s": None,
                    "identity_gate_pass": False,
                    "max_abs_state_error": None,
                    "max_rel_state_error": None,
                    "error": "skipped: cpu_whole_state_pjit_sharding_unsafe_for_fft_layout",
                    "skip_reason": "cpu_whole_state_pjit_sharding_unsafe_for_fft_layout",
                }
            },
        }
        out_json.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 2, "profile json written", "")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    summary = mod.run_sweep(
        backend="cpu",
        devices=[4],
        nx=4,
        ny=4,
        nz=4,
        nl=1,
        nm=1,
        dt=0.02,
        steps=1,
        method="rk2",
        sharding="auto",
        sharding_options="auto",
        laguerre_mode="grid",
        warmups=0,
        repeats=1,
        timeout_s=1.0,
        trace=False,
    )

    assert summary["identity_passed"] is False
    assert summary["rows"][0]["profile_returncode"] == 2
    assert summary["rows"][0]["state_sharding_active"] is True
    assert "unsafe_for_fft_layout" in summary["rows"][0]["error"]
    assert summary["profiles"]["4"]["profile_returncode"] == 2


# ---- from test_runtime_and_scaling_profile_contracts.py ----


runtime_kernels = load_profiling_tool("profile_runtime_kernels")
linear_trace = runtime_kernels
nonlinear_trace = runtime_kernels


def test_cyclone_runtime_profiler_default_config_exists() -> None:
    args = runtime_kernels.build_cyclone_parser().parse_args([])

    assert (REPO_ROOT / args.config).is_file()
    assert args.repeats == 1
    assert args.resolved_diagnostics is True
    assert args.reuse_prepared_simulation is False
    assert args.out is None


def test_prepared_profile_summary_fingerprints_numerical_outputs() -> None:
    result = (
        jnp.asarray([0.0, 0.5]),
        SimpleNamespace(
            heat_flux_t=jnp.asarray([2.0, 4.0]),
            dt_t=jnp.asarray([0.1, 0.2]),
        ),
        jnp.arange(6, dtype=jnp.float32).reshape(1, 2, 3).astype(jnp.complex64),
        SimpleNamespace(phi=jnp.asarray([3.0j, 4.0])),
    )

    summary = runtime_kernels._prepared_result_summary(result)

    assert summary["time"]["shape"] == [2]
    assert summary["time"]["max_abs"] == 0.5
    assert summary["final_state"]["shape"] == [1, 2, 3]
    assert summary["final_state"]["finite_fraction"] == 1.0
    np.testing.assert_allclose(summary["phi"]["l2_norm"], 5.0)
    np.testing.assert_allclose(summary["heat_flux"]["sum_real"], 6.0)
    np.testing.assert_allclose(summary["dt"]["max_abs"], 0.2)


def test_runtime_profile_normalizes_peak_rss_units() -> None:
    assert runtime_kernels._peak_rss_bytes(123, system="Darwin") == 123
    assert runtime_kernels._peak_rss_bytes(123, system="Linux") == 123 * 1024


def test_runtime_startup_profiler_keywords_match_integration_contract() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(main_runtime_startup)))
    integration_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "integrate_nonlinear_explicit_diagnostics_state"
    ]

    assert len(integration_calls) == 1
    passed_keywords = {
        keyword.arg for keyword in integration_calls[0].keywords if keyword.arg
    }
    accepted_keywords = set(
        inspect.signature(integrate_nonlinear_explicit_diagnostics_state).parameters
    )
    assert passed_keywords <= accepted_keywords, passed_keywords - accepted_keywords


def test_sspx3_stage_profile_preserves_identity_without_speedup_claim() -> None:
    profile = json.loads(
        (REPO_ROOT / "docs/_static/linear_sspx3_stage_profile.json").read_text(
            encoding="utf-8"
        )
    )

    assert profile["identity_gate"]["passed"] is True
    assert profile["before"]["finite"] is True
    assert profile["after"]["finite"] is True
    assert profile["performance_gate"]["measurable_speedup"] is False
    assert profile["performance_gate"]["passed"] is False
    assert abs(profile["performance_gate"]["relative_median_time_change"]) < 0.03


def test_prepared_nonlinear_cpu_gpu_profiles_are_matched_and_clean() -> None:
    cpu = json.loads(
        (
            REPO_ROOT / "docs/_static/prepared_nonlinear_runtime_cpu_profile.json"
        ).read_text(encoding="utf-8")
    )
    gpu = json.loads(
        (
            REPO_ROOT / "docs/_static/prepared_nonlinear_runtime_gpu_profile.json"
        ).read_text(encoding="utf-8")
    )

    for profile in (cpu, gpu):
        assert profile["git_revision"] == cpu["git_revision"]
        assert profile["git_dirty"] is False
        assert profile["reuse_prepared_simulation"] is True
        assert profile["resolved_diagnostics"] is False
        assert profile["steps"] == 200
        assert profile["method"] == "rk3"
        assert profile["fixed_dt"] is False
        assert profile["sample_stride"] == 10
        assert profile["diagnostics_stride"] == 10
        assert profile["software"] == cpu["software"]
        assert "result_summary" in profile
        assert profile["memory_summary"]["host_peak_rss_bytes"] > 0
    assert cpu["backend"] == "cpu"
    assert gpu["backend"] == "gpu"
    assert cpu["run_median_s"] / gpu["run_median_s"] >= 5.0
    expected_shapes = {
        "time": [21],
        "final_state": [1, 4, 8, 64, 64, 24],
        "phi": [64, 64, 24],
        "heat_flux": [21],
        "dt": [21],
    }
    for name, expected_shape in expected_shapes.items():
        assert (
            cpu["result_summary"][name]["shape"] == gpu["result_summary"][name]["shape"]
        )
        assert cpu["result_summary"][name]["shape"] == expected_shape
        assert cpu["result_summary"][name]["finite_fraction"] == 1.0
        assert gpu["result_summary"][name]["finite_fraction"] == 1.0
        # The full nonlinear state is a more sensitive accumulated trajectory
        # than the scalar diagnostics after 200 adaptive steps.  Keep its
        # observed CPU/GPU norm drift explicit instead of letting the old
        # mislabeled time vector provide a falsely exact state gate.
        rtol = 1.0e-3 if name == "final_state" else 1.0e-5
        np.testing.assert_allclose(
            cpu["result_summary"][name]["l2_norm"],
            gpu["result_summary"][name]["l2_norm"],
            rtol=rtol,
            atol=1.0e-12,
        )


def test_resolved_diagnostic_profiles_are_identity_gated_and_bounded() -> None:
    for backend in ("cpu", "gpu"):
        compact = json.loads(
            (
                REPO_ROOT
                / f"docs/_static/prepared_nonlinear_runtime_{backend}_profile.json"
            ).read_text(encoding="utf-8")
        )
        resolved = json.loads(
            (
                REPO_ROOT
                / f"docs/_static/prepared_nonlinear_runtime_{backend}_resolved_profile.json"
            ).read_text(encoding="utf-8")
        )
        assert compact["git_revision"] == resolved["git_revision"]
        assert compact["software"] == resolved["software"]
        assert compact["resolved_diagnostics"] is False
        assert resolved["resolved_diagnostics"] is True
        assert compact["steps"] == resolved["steps"] == 200
        assert resolved["run_median_s"] / compact["run_median_s"] <= 1.25
        assert (
            resolved["memory_summary"]["host_peak_rss_bytes"]
            / compact["memory_summary"]["host_peak_rss_bytes"]
            <= 1.10
        )
        for name in ("time", "final_state", "phi", "heat_flux", "dt"):
            np.testing.assert_allclose(
                compact["result_summary"][name]["l2_norm"],
                resolved["result_summary"][name]["l2_norm"],
                rtol=1.0e-7,
                atol=1.0e-12,
            )
        if backend == "gpu":
            compact_peak = compact["memory_summary"]["device_stats"][
                "peak_bytes_in_use"
            ]
            resolved_peak = resolved["memory_summary"]["device_stats"][
                "peak_bytes_in_use"
            ]
            assert resolved_peak / compact_peak <= 1.10


def test_make_profile_options_defaults_disable_python_and_host_tracers() -> None:
    opts = make_profile_options()
    assert opts.python_tracer_level == 0
    assert opts.host_tracer_level == 0


def test_make_profile_options_accepts_explicit_levels() -> None:
    opts = make_profile_options(python_tracer_level=1, host_tracer_level=2)
    assert opts.python_tracer_level == 1
    assert opts.host_tracer_level == 2


def test_profile_linear_cache_uses_low_rank_moment_factors() -> None:
    params = SimpleNamespace(
        nu_hermite=0.5,
        nu_laguerre=0.25,
        p_hyper=4,
        p_hyper_l=3,
        p_hyper_m=5,
        p_hyper_lm=2,
    )

    cache = build_low_rank_moment_cache(
        nl=3, nm=4, params=params, real_dtype=jnp.float32
    )

    assert cache["lb_lam"].shape == (3, 4)
    assert cache["collision_lam"].shape == (0,)
    assert cache["hyper_ratio"].shape == (3, 4, 1, 1, 1)
    assert cache["sqrt_p"].shape == (1, 1, 4, 1, 1, 1)
    assert cache["mask_const"].dtype == jnp.bool_


def test_profile_runtime_startup_writes_csv_and_json(tmp_path: Path) -> None:
    phases = [
        PhaseTiming(phase="a", seconds=1.25, note="first"),
        PhaseTiming(phase="b", seconds=2.75, note="second"),
    ]
    csv_path = tmp_path / "startup.csv"
    json_path = tmp_path / "startup.json"

    _write_phase_csv(csv_path, phases)
    _write_phase_json(json_path, phases, {"config": "case.toml", "device_count": 1})

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "phase,seconds,note" in csv_text
    assert "a,1.25,first" in csv_text

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["config"] == "case.toml"
    assert payload["startup_total_s"] == 4.0
    assert payload["phases"][1]["phase"] == "b"


def test_full_linear_trace_hlo_token_counts_are_coarse_but_stable() -> None:
    hlo = """
    ROOT fusion.1 = f32[2] fusion(arg), kind=kLoop
    fft.2 = c64[2] fft(arg), fft_type=FFT
    scatter.3 = f32[2] scatter(arg)
    """
    counts = linear_trace._hlo_token_counts(hlo)

    assert counts["fusion"] >= 1
    assert counts["fft"] >= 2
    assert counts["scatter"] >= 1
    assert counts["gather"] == 0


def test_hlo_op_counts_read_op_names_not_metadata() -> None:
    hlo = """
  %copy.6 = c64[2,4]{1,0} copy(%t), metadata={op_name="jit(f)/concatenate"}
  %concatenate.0 = c64[2,8]{1,0} concatenate(%copy.6, %c), dimensions={1}
  %gather.2 = f32[3] gather(%concatenate.0, %i), metadata={op_name="jit(f)/fft"}
  ROOT %copy.4 = f64[5] copy(%x)
  %while.1 = (c64[2], s32[]) while(%tuple), condition=%cond, body=%body
"""
    counts = runtime_kernels._hlo_op_counts(hlo)

    assert (counts["copy"], counts["concatenate"], counts["gather"]) == (2, 1, 1)
    assert counts["fft"] == 0
    assert counts["bytes_written"] == 8 * 8 + 16 * 8 + 5 * 8


def test_hlo_op_counts_price_captured_arrays_as_module_literals() -> None:
    """The ledger must price what a captured array costs the executable.

    The reference-route contract (queue row Q18) puts both nonlinear routes on
    the captured graph, so the module carries its cache, parameter and policy
    arrays as literals. ``constant_bytes`` is that payload and is counted apart
    from ``bytes_written``, which prices materialized copies.
    """

    hlo = """
  %constant.1 = f32[64]{0} constant({...})
  %constant.2 = s32[8]{0} constant({...})
  %copy.3 = f32[64]{0} copy(%constant.1)
"""
    counts = runtime_kernels._hlo_op_counts(hlo)

    assert counts["constant_bytes"] == 64 * 4 + 8 * 4
    assert counts["bytes_written"] == 64 * 4
    assert counts["copy"] == 1


def test_hlo_op_counts_charge_only_buffers_that_are_written() -> None:
    """A copy inside a fusion body is an index, not a buffer (queue row Q27).

    XLA:CPU emits a fusion body as one loop nest, so a ``copy`` written inside
    one -- typically re-laying-out a fusion *parameter* so the consumer can
    read it in the order it wants -- allocates nothing; only the fusion's ROOT
    owns an output buffer.  ``bytes_written`` charges every such instruction
    the full size of its shape, which is why cloning one producer into more
    consumer fusions moved it by tens of per cent with no traffic behind it.
    ``materialized_bytes`` is the subset that owns a buffer, and the two
    partition ``bytes_written`` exactly.
    """

    hlo = """
%fused_computation.1 (param_0: f32[4]) -> f32[4] {
  %param_0.1 = f32[4]{0} parameter(0)
  %copy.1 = f32[4]{0} copy(%param_0.1)
  ROOT %copy.2 = f32[4]{0} copy(%copy.1)
}

%body.7 (arg: f32[8]) -> f32[8] {
  %arg.1 = f32[8]{0} parameter(0)
  ROOT %copy.3 = f32[8]{0} copy(%arg.1)
}

ENTRY %main (x: f32[4]) -> f32[4] {
  %x.1 = f32[4]{0} parameter(0)
  %fusion.1 = f32[4]{0} fusion(%x.1), kind=kLoop, calls=%fused_computation.1
  %while.1 = f32[8]{0} while(%w), condition=%cond.6, body=%body.7
  ROOT %copy.4 = f32[4]{0} copy(%fusion.1)
}
"""

    counts = runtime_kernels._hlo_op_counts(hlo)

    assert counts["copy"] == 4
    # every copy, fusion interiors included -- the historical total
    assert counts["bytes_written"] == 4 * 4 + 4 * 4 + 8 * 4 + 4 * 4
    # the fusion ROOT, the while body's copy and the entry copy own buffers
    assert counts["materialized_bytes"] == 4 * 4 + 8 * 4 + 4 * 4
    # only %copy.1, written inside the fusion body and not its root
    assert counts["fused_interior_bytes"] == 4 * 4
    assert (
        counts["materialized_bytes"] + counts["fused_interior_bytes"]
        == counts["bytes_written"]
    )


def test_hlo_op_counts_without_computation_headers_are_all_materialized() -> None:
    """A bare instruction list has no fusion bodies, so nothing is interior.

    The older ledger snippets in this file are written that way, and their
    ``bytes_written`` must keep its value.
    """

    hlo = """
  %copy.6 = c64[2,4]{1,0} copy(%t)
  %concatenate.0 = c64[2,8]{1,0} concatenate(%copy.6, %c), dimensions={1}
"""

    counts = runtime_kernels._hlo_op_counts(hlo)

    assert counts["fused_interior_bytes"] == 0
    assert counts["materialized_bytes"] == counts["bytes_written"] == 8 * 8 + 16 * 8


def test_compiled_memory_stats_report_the_compilers_buffer_assignment() -> None:
    """``memory_analysis`` is the independent check on the text counts.

    It comes from XLA's buffer assignment rather than from the printed module,
    so an instruction a fusion emits as index arithmetic cannot inflate it. The
    step ledger records it beside the op counts for that reason.
    """

    stats: dict[str, int] = {}
    # An explicit dtype: the CI shards run with JAX_ENABLE_X64, under which an
    # unannotated zeros() is float64 and the argument is twice this size.
    runtime_kernels._compiled_hlo_text(
        lambda x: jnp.sum(x * 2.0),
        jnp.zeros((8, 8), dtype=jnp.float32),
        stats=stats,
    )

    assert set(stats) >= {
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
    }
    assert stats["argument_size_in_bytes"] == 8 * 8 * 4
    assert all(value >= 0 for value in stats.values())


def test_nonlinear_step_hlo_routes_name_one_reference_graph() -> None:
    """``--route runtime`` and ``--route diagnostics`` are the same graph.

    Queue row Q18 gave ``integrate_nonlinear_explicit_diagnostics_state`` the
    prepared route's jit, so the ledger must not keep lowering a separate
    module for the runtime. ``--route eager-scan`` keeps the retired operand
    placement so the rejected option stays priceable.
    """

    parser = runtime_kernels.build_nonlinear_step_hlo_parser()
    choices = parser.parse_known_args(["--route", "eager-scan"])[0]
    assert choices.route == "eager-scan"

    source = textwrap.dedent(inspect.getsource(runtime_kernels.main_nonlinear_step_hlo))
    tree = ast.parse(source)
    selectors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.IfExp)
        and isinstance(node.body, ast.Name)
        and node.body.id == "_eager_scan_hlo"
    ]
    assert len(selectors) == 1
    assert isinstance(selectors[0].orelse, ast.Name)
    assert selectors[0].orelse.id == "_diagnostics_scan_hlo"


def test_eager_scan_route_still_lowers_scan_body_arrays_as_arguments() -> None:
    """The retired placement must stay measurable, and stay the rejected one.

    A jit of the same function embeds the closed-over array as a constant.
    That captured graph is what both shipped routes now compile; the bound
    module below is the eager scan the contract retired.
    """

    weights = jnp.linspace(0.5, 1.5, 64, dtype=jnp.float32)

    def run(state: jnp.ndarray) -> jnp.ndarray:
        def step(carry: jnp.ndarray, _unused: None) -> tuple[jnp.ndarray, None]:
            return carry * weights, None

        return jax.lax.scan(step, state, None, length=3)[0]

    state = jnp.ones(64, dtype=jnp.float32)
    constant = re.compile(r"= f32\[64\]\S* constant\(")
    entry = re.compile(r"entry_computation_layout=\{\((?P<arguments>[^)]*)\)->")
    captured = runtime_kernels._compiled_hlo_text(run, state)
    bound = runtime_kernels._scan_equation_hlo(run, state)

    def entry_arguments(text: str) -> int:
        match = entry.search(text)
        assert match is not None
        return match.group("arguments").count("f32[64]")

    assert constant.search(captured) and entry_arguments(captured) == 1
    assert not constant.search(bound) and entry_arguments(bound) == 2


def test_full_linear_trace_summary_contains_metadata() -> None:
    payload = linear_trace._build_summary(
        config="benchmarks/cases/cyclone_nonlinear_miller.toml",
        backend="cpu",
        nl=4,
        nm=8,
        repeats=3,
        state="z_wave",
        z_variation_norm=0.2,
        compile_execute_seconds=1.5,
        warm_seconds=0.1,
        rhs_norm=2.0,
        phi_norm=3.0,
        hlo_text="ROOT add.1 = f32[] add(a, b)\n",
        trace_dir=Path("tools_out/trace"),
        memory_profile=Path("tools_out/memory.prof"),
        hlo_out=Path("tools_out/hlo.txt"),
        force_electrostatic_fields=True,
        source="gkx.operators.linear.rhs.linear_rhs_cached",
    )

    assert payload["kind"] == "full_linear_rhs_trace_summary"
    assert payload["case"] == "cyclone_nonlinear_miller"
    assert payload["backend"] == "cpu"
    assert payload["warm_seconds"] == 0.1
    assert payload["hlo_token_counts"]["add"] >= 1
    assert payload["force_electrostatic_fields"] is True
    assert payload["source"] == "gkx.operators.linear.rhs.linear_rhs_cached"
    assert payload["trace_dir"] == "tools_out/trace"
    assert "kernel-level optimization targets" in payload["claim_scope"]


def test_full_linear_trace_summary_json_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    linear_trace._write_summary_json({"kind": "full", "value": 2.0}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "kind": "full",
        "value": 2.0,
    }


def test_full_linear_trace_inject_z_wave_adds_parallel_variation() -> None:
    state = jnp.zeros((1, 4, 3, 2, 1, 5), dtype=jnp.complex64)

    out = linear_trace._inject_z_wave(
        state, ky_index=1, kx_index=0, amplitude=0.2, z_mode=1
    )

    assert linear_trace._z_variation_norm(out) > 0.0
    assert jnp.linalg.norm(out[0, 0, 2, 1, 0]) > 0.0


def test_full_nonlinear_trace_summary_contains_metadata() -> None:
    payload = nonlinear_trace._build_nonlinear_summary(
        config="benchmarks/cases/cyclone_nonlinear_miller.toml",
        backend="gpu",
        nl=4,
        nm=8,
        repeats=5,
        state="initial",
        laguerre_mode="grid",
        compressed_real_fft=True,
        z_variation_norm=0.0,
        compile_execute_seconds=2.0,
        warm_seconds=0.01,
        rhs_norm=1.0,
        phi_norm=0.1,
        apar_norm=0.0,
        bpar_norm=0.0,
        hlo_text="ROOT multiply.1 = f32[] multiply(a, b)\nfft.2 = c64[] fft(c)\n",
        trace_dir=Path("tools_out/nonlinear_trace"),
        memory_profile=Path("tools_out/nonlinear.prof"),
        hlo_out=Path("tools_out/nonlinear.hlo.txt"),
        electrostatic_specialized=True,
    )

    assert payload["kind"] == "full_nonlinear_rhs_trace_summary"
    assert payload["case"] == "cyclone_nonlinear_miller"
    assert payload["backend"] == "gpu"
    assert payload["laguerre_mode"] == "grid"
    assert payload["compressed_real_fft"] is True
    assert payload["hlo_token_counts"]["multiply"] >= 1
    assert payload["hlo_token_counts"]["fft"] >= 1
    assert payload["electrostatic_specialized"] is True
    assert payload["trace_dir"] == "tools_out/nonlinear_trace"
    assert "transport runtime claim" in payload["claim_scope"]


def test_full_nonlinear_trace_field_norm_handles_missing_em_fields() -> None:
    assert nonlinear_trace._field_norm(None) == 0.0


# Tracked parallel and RHS-term profile artifacts. Their profilers were retired
# in SLIM-SCRIPTS tranche 3 (recovery SHA in plan/research/2026-09-22-slim-tools/MAP.md);
# the committed JSONs stay as fixtures and keep these contracts.


def test_tracked_mixed_species_hermite_profile_is_scoped_and_identity_gated() -> None:
    artifact = (
        REPO_ROOT / "docs" / "_static" / "linear_rhs_species_hermite_profile_cpu.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["decomposition_axis"] == "species_hermite"
    assert payload["requested_devices"] == 4
    assert payload["actual_devices"] == 4
    assert payload["identity_passed"] is True
    integration = payload["integration"]
    assert integration["identity_passed"] is True
    assert integration["speedup_passed"] is (integration["speedup"] > 1.0)
    assert integration["state_identity"]["max_abs_error"] <= payload["atol"]
    assert integration["field_history_identity"]["max_abs_error"] <= payload["atol"]
    assert (
        "mixed species-Hermite collision-free integration" in integration["claim_scope"]
    )
    assert payload["max_rel_error"] <= payload["rtol"]
    assert payload["max_abs_error"] <= payload["atol"]
    assert payload["max_phi_abs_error"] <= payload["atol"]
    assert payload["speedup"] > 1.0
    assert "not a GPU or general scaling claim" in payload["claim_scope"]
    assert len(payload["git_revision"]) == 40


def test_linear_rhs_terms_tracked_miller_profile_is_active_artifact() -> None:
    path = REPO_ROOT / "docs" / "_static" / "linear_rhs_terms_profile_miller_cpu.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["kind"] == "linear_rhs_terms_profile_summary"
    assert payload["case"] == "runtime_cyclone_nonlinear_miller"
    assert payload["state"] == "z_wave_linear_kick"
    assert payload["full_linear_rhs_seconds"] > 0.0
    assert payload["rows"]["streaming"]["norm"] > 0.0
    assert payload["dominant_nonzero_norm_term"] == "streaming"


# ---- from test_check_vmec_boozer_gates.py ----


holdout_mod = load_release_tool("check_vmec_boozer_gates")


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _aggregate_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        "passed": True,
        "claim_scope": "reduced aggregate objective plumbing",
        "samples": [
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 1,
                "weight": 0.5,
            },
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 2,
                "weight": 0.5,
            },
        ],
    }


def _line_search_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_line_search_report",
        "passed": True,
        "samples": [
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 1,
                "weight": 0.5,
            },
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 2,
                "weight": 0.5,
            },
        ],
    }


def _ensemble_payload(*, passed: bool = True) -> dict[str, object]:
    return {
        "kind": "nonlinear_window_ensemble_report",
        "claim_level": "replicated_nonlinear_window_uncertainty_gate_not_simulation_claim",
        "passed": passed,
        "gate_report": {"passed": passed},
    }


def test_aggregate_holdout_gate_blocks_without_surface_or_field_line_holdout(
    tmp_path: Path,
) -> None:
    aggregate = _write_json(tmp_path / "aggregate.json", _aggregate_payload())
    line_search = _write_json(tmp_path / "line_search.json", _line_search_payload())

    report = holdout_mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
    )

    assert report["passed"] is False
    assert report["promotion_gate"]["blockers"] == [
        "passed_holdout_surface_or_field_line_artifact",
        "passed_replicated_nonlinear_window_ensemble",
    ]
    assert report["training_sample_summary"]["alphas"] == ["0"]


def test_aggregate_holdout_gate_rejects_ky_only_holdout(tmp_path: Path) -> None:
    aggregate = _write_json(tmp_path / "aggregate.json", _aggregate_payload())
    line_search = _write_json(tmp_path / "line_search.json", _line_search_payload())
    ky_only = _write_json(
        tmp_path / "ky_only.json",
        {
            "passed": True,
            "claim_level": "passed_grid_convergence_candidate_for_transport_holdout",
            "samples": [
                {"surface_index": None, "alpha": 0.0, "selected_ky_index": 7},
            ],
        },
    )

    report = holdout_mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
        holdout_artifacts=(ky_only,),
    )

    assert report["passed"] is False
    assert report["holdout_artifacts"][0]["passed"] is True
    assert report["holdout_artifacts"][0]["heldout_surface_or_field_line"] is False
    assert "k_y-only" in report["promotion_gate"]["requirements"][4]


def test_aggregate_holdout_gate_accepts_passed_field_line_holdout(
    tmp_path: Path,
) -> None:
    aggregate = _write_json(tmp_path / "aggregate.json", _aggregate_payload())
    line_search = _write_json(tmp_path / "line_search.json", _line_search_payload())
    ensemble = _write_json(tmp_path / "ensemble.json", _ensemble_payload())
    holdout = _write_json(
        tmp_path / "alpha_holdout.json",
        {
            "promotion_gate": {"passed": True},
            "claim_level": "passed_grid_convergence_candidate_for_transport_holdout",
            "samples": [
                {"surface_index": None, "alpha": 0.75, "selected_ky_index": 1},
            ],
        },
    )

    report = holdout_mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
        holdout_artifacts=(holdout,),
        nonlinear_ensemble_artifacts=(ensemble,),
    )

    assert report["passed"] is True
    assert report["promotion_gate"]["blockers"] == []
    assert report["holdout_artifacts"][0]["qualifies_for_promotion"] is True
    assert (
        report["nonlinear_ensemble_artifacts"][0][
            "qualifies_for_production_nonlinear_promotion"
        ]
        is True
    )
    assert report["gates"][-2]["detail"].endswith("held-out field-line alpha=0.75")


def test_aggregate_holdout_gate_rejects_non_ensemble_nonlinear_artifact(
    tmp_path: Path,
) -> None:
    aggregate = _write_json(tmp_path / "aggregate.json", _aggregate_payload())
    line_search = _write_json(tmp_path / "line_search.json", _line_search_payload())
    holdout = _write_json(
        tmp_path / "alpha_holdout.json",
        {
            "promotion_gate": {"passed": True},
            "claim_level": "passed_grid_convergence_candidate_for_transport_holdout",
            "samples": [{"surface_index": None, "alpha": 0.75, "selected_ky_index": 1}],
        },
    )
    single_window = _write_json(
        tmp_path / "single_window.json",
        {
            "kind": "nonlinear_window_convergence_report",
            "passed": True,
            "gate_report": {"passed": True},
        },
    )

    report = holdout_mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
        holdout_artifacts=(holdout,),
        nonlinear_ensemble_artifacts=(single_window,),
    )

    assert report["passed"] is False
    assert report["promotion_gate"]["blockers"] == [
        "passed_replicated_nonlinear_window_ensemble"
    ]
    assert (
        report["nonlinear_ensemble_artifacts"][0]["is_nonlinear_window_ensemble"]
        is False
    )


def test_aggregate_holdout_gate_records_readiness_manifest_blockers(
    tmp_path: Path,
) -> None:
    aggregate = _write_json(tmp_path / "aggregate.json", _aggregate_payload())
    line_search = _write_json(tmp_path / "line_search.json", _line_search_payload())
    holdout = _write_json(
        tmp_path / "alpha_holdout.json",
        {
            "promotion_gate": {"passed": True},
            "claim_level": "passed_grid_convergence_candidate_for_transport_holdout",
            "samples": [{"surface_index": None, "alpha": 0.75, "selected_ky_index": 1}],
        },
    )
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "kind": "nonlinear_window_ensemble_readiness_manifest",
            "passed": False,
            "promotion_gate": {
                "passed": False,
                "blockers": ["seed_and_timestep_replicates_present"],
            },
            "missing_artifacts": [
                {
                    "case": "case_a",
                    "variant_axis": "seed",
                    "missing_count": 2,
                }
            ],
        },
    )

    report = holdout_mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
        holdout_artifacts=(holdout,),
        nonlinear_ensemble_artifacts=(manifest,),
    )

    row = report["nonlinear_ensemble_artifacts"][0]
    assert report["passed"] is False
    assert row["is_nonlinear_window_readiness_manifest"] is True
    assert row["readiness_blockers"] == ["seed_and_timestep_replicates_present"]
    assert row["missing_artifacts"][0]["variant_axis"] == "seed"
    assert row["qualifies_for_production_nonlinear_promotion"] is False


def test_aggregate_holdout_gate_rejects_non_promotable_holdout_scope(
    tmp_path: Path,
) -> None:
    aggregate = _write_json(tmp_path / "aggregate.json", _aggregate_payload())
    line_search = _write_json(tmp_path / "line_search.json", _line_search_payload())
    startup_holdout = _write_json(
        tmp_path / "startup_holdout.json",
        {
            "passed": True,
            "claim_level": "startup_transient_nonlinear_plumbing_fd_audit_not_transport_average",
            "transport_average_gate": False,
            "heldout_samples": [
                {"surface_index": 3, "alpha": 0.0, "selected_ky_index": 1},
            ],
        },
    )

    report = holdout_mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
        holdout_artifacts=(startup_holdout,),
    )

    assert report["passed"] is False
    assert report["holdout_artifacts"][0]["n_samples"] == 1
    assert report["holdout_artifacts"][0]["heldout_surface_or_field_line"] is True
    assert report["holdout_artifacts"][0]["qualifies_for_promotion"] is False
    assert (
        "transport_average_gate_false"
        in report["holdout_artifacts"][0]["claim_scope_blockers"]
    )
    assert (
        "passed_replicated_nonlinear_window_ensemble"
        in report["promotion_gate"]["blockers"]
    )


def test_aggregate_holdout_gate_main_writes_json(tmp_path: Path) -> None:
    aggregate = _write_json(tmp_path / "aggregate.json", _aggregate_payload())
    line_search = _write_json(tmp_path / "line_search.json", _line_search_payload())
    out = tmp_path / "report.json"

    result = holdout_mod.main_aggregate_holdout(
        [
            "--aggregate-artifact",
            str(aggregate),
            "--line-search-artifact",
            str(line_search),
            "--json-out",
            str(out),
        ]
    )

    assert result == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["kind"] == "vmec_boozer_aggregate_holdout_promotion_gate"
    assert saved["passed"] is False


# VMEC/Boozer reduced portfolio guard assertions
portfolio_mod = holdout_mod


def _row_artifact() -> dict[str, object]:
    samples = [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 2, "weight": 0.25},
    ]
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        "artifact_kind": "vmec_boozer_multi_point_objective_gate",
        "builder": "scripts/artifacts/build_vmec_boozer_aggregate_objective_gate.py multi-point",
        "passed": True,
        "source_scope": "mode21_vmec_boozer_state_multi_point",
        "claim_scope": "real VMEC/Boozer reduced QL rows; not a nonlinear turbulent transport claim",
        "next_action": "Nonlinear transport optimization still requires separate long-window gates.",
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "input_path": "/tmp/input.nfp4_QH_warm_start",
        "wout_path": "/tmp/wout_nfp4_QH_warm_start.nc",
        "options": {"mboz": 21, "nboz": 21},
        "n_samples": 4,
        "samples": samples,
        "objective_names": [
            "gamma",
            "omega",
            "kperp_eff2",
            "mixing_length_heat_flux_proxy",
        ],
        "minus_sample_values": [0.7, 0.9, 0.8, 1.0],
        "base_sample_values": [0.8, 1.0, 0.9, 1.1],
        "plus_sample_values": [0.9, 1.1, 1.0, 1.2],
        "minus_objective_table": [
            [0.10, -0.2, 0.5, 0.7],
            [0.12, -0.3, 0.6, 0.9],
            [0.11, -0.1, 0.4, 0.8],
            [0.13, -0.4, 0.7, 1.0],
        ],
        "base_objective_table": [
            [0.11, -0.2, 0.5, 0.8],
            [0.13, -0.3, 0.6, 1.0],
            [0.12, -0.1, 0.4, 0.9],
            [0.14, -0.4, 0.7, 1.1],
        ],
        "plus_objective_table": [
            [0.12, -0.2, 0.5, 0.9],
            [0.14, -0.3, 0.6, 1.1],
            [0.13, -0.1, 0.4, 1.0],
            [0.15, -0.4, 0.7, 1.2],
        ],
        "base_value": 0.95,
        "minus_value": 0.85,
        "plus_value": 1.05,
        "central_derivative": 1.0,
        "response_abs": 0.2,
        "curvature_ratio": 0.01,
        "finite_values": True,
        "finite_difference_consistent": True,
        "response_resolved": True,
    }


def _gradient_artifact() -> dict[str, object]:
    return {
        "kind": "mode21_vmec_boozer_quasilinear_gradient_gate",
        "passed": True,
        "objective_gates": [
            {
                "objective": "gamma",
                "parameter": "Rcos_mid_surface_m1",
                "passed": True,
                "implicit": 10.0,
                "finite_difference": 10.01,
                "abs_error": 0.01,
                "rel_error": 0.001,
            },
            {
                "objective": "mixing_length_heat_flux_proxy",
                "parameter": "Rcos_mid_surface_m1",
                "passed": True,
                "implicit": 20.0,
                "finite_difference": 20.02,
                "abs_error": 0.02,
                "rel_error": 0.001,
            },
        ],
    }


def test_reduced_portfolio_guard_passes_real_metadata_contract() -> None:
    report = reduced_portfolio_artifact_guard_report(
        _row_artifact(),
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is True
    assert report["provenance_gate"]["passed"] is True
    assert report["coverage_gate"]["n_alphas"] == 2
    assert report["coverage_gate"]["n_ky"] == 2
    assert report["portfolio_reducer_gate"]["contract"]["row_shape"] == [1, 2, 2, 1]
    assert report["ad_fd_gradient_gate"]["has_growth_ad_fd_gate"] is True
    assert report["ad_fd_gradient_gate"]["has_quasilinear_ad_fd_gate"] is True
    assert report["claim_scope_gate"]["passed"] is True


@pytest.mark.parametrize(
    ("reduction", "base_value", "expected_shape"),
    [
        ("weighted_mean", 0.95, [1, 2, 2, 1]),
        ("max", 1.0, [1, 2, 2, 1]),
    ],
)
def test_reduced_portfolio_guard_accepts_declared_reducer_semantics(
    reduction: str,
    base_value: float,
    expected_shape: list[int],
) -> None:
    artifact = _row_artifact()
    artifact["reduction"] = reduction
    if reduction == "max":
        artifact["base_sample_values"] = [0.5, 0.75, 0.875, 1.0]
    artifact["base_value"] = base_value

    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is True
    assert report["portfolio_reducer_gate"]["reduction"] == reduction
    assert report["portfolio_reducer_gate"]["contract"]["row_shape"] == expected_shape


def test_reduced_portfolio_guard_distinguishes_physical_torflux_surfaces() -> None:
    artifact = _row_artifact()
    artifact["samples"] = [
        {
            "surface_index": None,
            "torflux": 0.5,
            "surface": 0.5,
            "alpha": 0.0,
            "ky": 0.1,
            "selected_ky_index": 1,
            "weight": 0.25,
        },
        {
            "surface_index": None,
            "torflux": 0.5,
            "surface": 0.5,
            "alpha": 0.0,
            "ky": 0.2,
            "selected_ky_index": 2,
            "weight": 0.25,
        },
        {
            "surface_index": None,
            "torflux": 0.7,
            "surface": 0.7,
            "alpha": 0.0,
            "ky": 0.1,
            "selected_ky_index": 1,
            "weight": 0.25,
        },
        {
            "surface_index": None,
            "torflux": 0.7,
            "surface": 0.7,
            "alpha": 0.0,
            "ky": 0.2,
            "selected_ky_index": 2,
            "weight": 0.25,
        },
    ]
    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
        config=ReducedPortfolioArtifactGuardConfig(min_alphas=1),
    )

    assert report["passed"] is True
    assert report["coverage_gate"]["n_surfaces"] == 2
    assert report["coverage_gate"]["n_alphas"] == 1
    assert report["coverage_gate"]["n_ky"] == 2
    assert report["portfolio_reducer_gate"]["contract"]["row_shape"] == [2, 1, 2, 1]


def test_reduced_portfolio_guard_rejects_duplicate_or_incomplete_sample_grids() -> None:
    duplicate = _row_artifact()
    duplicate["samples"] = [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 2, "weight": 0.25},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        reduced_portfolio_artifact_guard_report(
            duplicate,
            gradient_artifacts=[_gradient_artifact()],
        )

    incomplete = _row_artifact()
    incomplete["samples"] = incomplete["samples"][:-1]  # type: ignore[index]
    incomplete["base_sample_values"] = incomplete["base_sample_values"][:-1]  # type: ignore[index]
    incomplete["minus_sample_values"] = incomplete["minus_sample_values"][:-1]  # type: ignore[index]
    incomplete["plus_sample_values"] = incomplete["plus_sample_values"][:-1]  # type: ignore[index]
    incomplete["base_objective_table"] = incomplete["base_objective_table"][:-1]  # type: ignore[index]
    incomplete["minus_objective_table"] = incomplete["minus_objective_table"][:-1]  # type: ignore[index]
    incomplete["plus_objective_table"] = incomplete["plus_objective_table"][:-1]  # type: ignore[index]
    with pytest.raises(ValueError, match="complete rectangular"):
        reduced_portfolio_artifact_guard_report(
            incomplete,
            gradient_artifacts=[_gradient_artifact()],
        )


def test_reduced_portfolio_guard_marks_bad_gradient_gate_without_crashing() -> None:
    bad_gradient = {
        "kind": "mode21_vmec_boozer_quasilinear_gradient_gate",
        "passed": False,
        "objective_gates": [
            {
                "objective": "gamma",
                "passed": True,
                "implicit": 1.0,
                "finite_difference": float("nan"),
                "abs_error": 0.0,
                "rel_error": 0.0,
            },
            "not-a-dict",
        ],
    }

    report = reduced_portfolio_artifact_guard_report(
        _row_artifact(),
        gradient_artifacts=[bad_gradient],  # type: ignore[list-item]
    )

    assert report["passed"] is False
    assert report["ad_fd_gradient_gate"]["passed"] is False
    assert report["ad_fd_gradient_gate"]["finite_ad_fd_values"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda artifact: artifact.update({"reduction": "median"}),
            "artifact reduction",
        ),
        (
            lambda artifact: artifact.update({"base_sample_values": [0.8, 1.0]}),
            "base_sample_values",
        ),
        (
            lambda artifact: artifact.update(
                {"base_objective_table": [0.8, 1.0, 0.9, 1.1]}
            ),
            "two-dimensional",
        ),
        (
            lambda artifact: artifact.update(
                {"samples": [*artifact["samples"][:-1], "bad-sample"]}
            ),
            "all samples",
        ),
    ],
)
def test_reduced_portfolio_guard_rejects_malformed_artifact_shapes(
    mutator,
    message: str,
) -> None:
    artifact = _row_artifact()
    mutator(artifact)

    with pytest.raises(ValueError, match=message):
        reduced_portfolio_artifact_guard_report(
            artifact,
            gradient_artifacts=[_gradient_artifact()],
        )


def test_reduced_portfolio_guard_reports_objective_and_provenance_blockers() -> None:
    artifact = _row_artifact()
    artifact["objective_names"] = ["omega", "kperp_eff2"]
    artifact["options"] = {"mboz": 8, "nboz": 8}
    artifact["input_path"] = ""
    artifact["wout_path"] = ""

    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is False
    assert report["objective_name_gate"]["passed"] is False
    assert report["objective_name_gate"]["has_growth_objective"] is False
    assert report["objective_name_gate"]["has_quasilinear_objective"] is False
    assert report["provenance_gate"]["passed"] is False
    assert report["provenance_gate"]["mboz"] == 8
    assert report["provenance_gate"]["has_input_and_wout_paths"] is False


def test_reduced_portfolio_guard_reports_unresolved_finite_difference_diagnostics() -> (
    None
):
    artifact = _row_artifact()
    artifact["response_resolved"] = False
    artifact["finite_difference_consistent"] = False
    artifact["plus_value"] = float("inf")

    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is False
    assert report["finite_difference_gate"]["passed"] is False
    assert report["finite_difference_gate"]["response_resolved"] is False
    assert report["finite_difference_gate"]["finite_difference_consistent"] is False
    assert (
        report["finite_difference_gate"]["finite_scalar_fields"]["plus_value"] is False
    )


def test_reduced_portfolio_guard_fails_single_alpha_or_missing_gradient_gate() -> None:
    artifact = _row_artifact()
    artifact["samples"] = [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.5},
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.5},
    ]
    artifact["base_sample_values"] = [0.8, 1.0]
    artifact["minus_sample_values"] = [0.7, 0.9]
    artifact["plus_sample_values"] = [0.9, 1.1]
    artifact["base_objective_table"] = [[0.11, -0.2, 0.5, 0.8], [0.13, -0.3, 0.6, 1.0]]
    artifact["minus_objective_table"] = [[0.10, -0.2, 0.5, 0.7], [0.12, -0.3, 0.6, 0.9]]
    artifact["plus_objective_table"] = [[0.12, -0.2, 0.5, 0.9], [0.14, -0.3, 0.6, 1.1]]
    artifact["base_value"] = 0.9
    artifact["minus_value"] = 0.8
    artifact["plus_value"] = 1.0

    report = reduced_portfolio_artifact_guard_report(artifact, gradient_artifacts=[])

    assert report["passed"] is False
    assert report["coverage_gate"]["passed"] is False
    assert report["ad_fd_gradient_gate"]["passed"] is False


def test_reduced_portfolio_guard_fails_production_nonlinear_claim() -> None:
    artifact = _row_artifact()
    artifact["objective"] = "nonlinear_heat_flux"
    artifact["claim_scope"] = (
        "production nonlinear turbulent transport optimization claim"
    )

    report = reduced_portfolio_artifact_guard_report(
        artifact,
        gradient_artifacts=[_gradient_artifact()],
    )

    assert report["passed"] is False
    assert report["claim_scope_gate"]["passed"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_alphas": 0}, "min_alphas"),
        ({"min_ky": 0}, "min_ky"),
        ({"min_objectives": 0}, "min_objectives"),
        ({"min_boozer_mode": 0}, "min_boozer_mode"),
        ({"value_rtol": -1.0e-8}, "tolerances"),
        ({"value_atol": -1.0e-8}, "tolerances"),
    ],
)
def test_reduced_portfolio_guard_validates_config(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ReducedPortfolioArtifactGuardConfig(**kwargs)


def test_tool_writes_guard_artifact(tmp_path: Path) -> None:
    row_path = tmp_path / "row.json"
    gradient_path = tmp_path / "gradient.json"
    out_path = tmp_path / "guard.json"
    row_path.write_text(json.dumps(_row_artifact()), encoding="utf-8")
    gradient_path.write_text(json.dumps(_gradient_artifact()), encoding="utf-8")

    payload = portfolio_mod.build_vmec_boozer_reduced_portfolio_guard_payload(
        row_artifact=row_path,
        gradient_artifacts=[gradient_path],
    )
    written = portfolio_mod.write_vmec_boozer_reduced_portfolio_guard_artifact(
        payload, out=out_path
    )

    assert Path(written) == out_path
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert data["row_artifact"] == str(row_path)


def test_tool_exposes_reducer_value_tolerances(tmp_path: Path) -> None:
    row = _row_artifact()
    row["base_value"] = 0.95000004
    row_path = tmp_path / "row.json"
    gradient_path = tmp_path / "gradient.json"
    row_path.write_text(json.dumps(row), encoding="utf-8")
    gradient_path.write_text(json.dumps(_gradient_artifact()), encoding="utf-8")

    strict_payload = portfolio_mod.build_vmec_boozer_reduced_portfolio_guard_payload(
        row_artifact=row_path,
        gradient_artifacts=[gradient_path],
    )
    loose_payload = portfolio_mod.build_vmec_boozer_reduced_portfolio_guard_payload(
        row_artifact=row_path,
        gradient_artifacts=[gradient_path],
        value_rtol=1.0e-6,
        value_atol=1.0e-6,
    )

    assert strict_payload["portfolio_reducer_gate"]["passed"] is False
    assert strict_payload["passed"] is False
    assert loose_payload["portfolio_reducer_gate"]["passed"] is True
    assert loose_payload["passed"] is True


def test_tool_main_returns_nonzero_for_failed_guard(tmp_path: Path) -> None:
    row = _row_artifact()
    row["options"] = {"mboz": 8, "nboz": 8}
    row_path = tmp_path / "row.json"
    gradient_path = tmp_path / "gradient.json"
    row_path.write_text(json.dumps(row), encoding="utf-8")
    gradient_path.write_text(json.dumps(_gradient_artifact()), encoding="utf-8")

    result = portfolio_mod.main_reduced_portfolio_guard(
        [
            "--row-artifact",
            str(row_path),
            "--gradient-artifact",
            str(gradient_path),
            "--out",
            str(tmp_path / "guard.json"),
        ]
    )

    assert result == 1
