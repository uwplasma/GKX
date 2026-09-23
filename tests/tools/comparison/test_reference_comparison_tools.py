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

import jax.numpy as jnp

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
    _build_initial_condition,
    _two_species_params,
)
from gkx.config import KBMBaseCase
from gkx.core_grid import select_ky_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.terms.assembly import assemble_rhs_terms_cached, compute_fields_cached
from gkx.terms.config import TermConfig
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
        / "examples/linear/axisymmetric/runtime_kbm.toml"
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
    from gkx.runtime import run_runtime_linear

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
