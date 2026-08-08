"""Reference-comparison maintainer tool contracts."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

# ---- imported-linear growth-dump mode ----

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "comparison"))

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
from gkx.core.grid import build_spectral_grid
from gkx.solvers.time.explicit import ExplicitTimeConfig
from gkx.operators.linear.params import LinearTerms
from gkx.workflows.runtime.config import RuntimeConfig
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


# ---- test_compare_gx_kbm.py ----

import pandas as pd


def test_compare_gx_kbm_parser_defaults_hl_dims_to_gx_contract() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    args = mod.build_parser().parse_args(["--gx", "/tmp/run.out.nc"])
    assert args.Nl is None
    assert args.Nm is None


def test_compare_gx_kbm_prepare_gx_reference_preserves_full_grid_metadata(
    monkeypatch,
) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    gx_time = np.array([0.0, 1.0], dtype=float)
    gx_ky = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    gx_omega_series = np.arange(2 * 4 * 2, dtype=float).reshape(2, 4, 2)

    monkeypatch.setattr(
        mod,
        "_load_gx_omega_gamma",
        lambda _path: (gx_time, gx_ky, gx_omega_series, 0.01, 1.4, 0.8, 0.18, 2.77778),
    )

    prepared = mod._prepare_gx_reference(
        Path("dummy.nc"), ky_arg="0.3", y0_fallback=10.0
    )

    gx_time_sel, gx_ky_sel, gx_omega_sel, beta, q, shat, eps, rmaj, nky_full, y0 = (
        prepared
    )
    assert np.array_equal(gx_time_sel, gx_time)
    assert np.array_equal(gx_ky_sel, np.array([0.3]))
    assert np.array_equal(gx_omega_sel, gx_omega_series[:, [2], :])
    assert beta == 0.01
    assert q == 1.4
    assert shat == 0.8
    assert eps == 0.18
    assert rmaj == 2.77778
    assert nky_full == 4
    assert np.isclose(y0, 10.0)


def test_compare_gx_kbm_checkpoints_partial_rows(tmp_path: Path) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    out = tmp_path / "kbm.csv"
    rows = [
        {"ky": 0.2, "solver": "gx_time", "gamma": 1.0},
        {"ky": 0.3, "solver": "gx_time", "gamma": 2.0},
    ]

    mod._write_rows(out, rows[:1])
    first = pd.read_csv(out)
    assert list(first["ky"]) == [0.2]

    mod._write_rows(out, rows)
    second = pd.read_csv(out)
    assert list(second["ky"]) == [0.2, 0.3]
    assert list(second["gamma"]) == [1.0, 2.0]


def test_compare_gx_kbm_continuation_score_prefers_overlap() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    smooth = mod._candidate_objective(
        rel_gamma=0.10,
        rel_omega=0.10,
        eig_overlap_gx=0.80,
        eig_overlap_prev=0.95,
        gamma_weight=1.0,
        omega_weight=1.0,
        gx_overlap_weight=1.0,
        prev_overlap_weight=2.0,
    )
    jump = mod._candidate_objective(
        rel_gamma=0.08,
        rel_omega=0.08,
        eig_overlap_gx=0.82,
        eig_overlap_prev=0.20,
        gamma_weight=1.0,
        omega_weight=1.0,
        gx_overlap_weight=1.0,
        prev_overlap_weight=2.0,
    )

    assert smooth < jump


def test_compare_gx_kbm_run_candidate_uses_gx_shift_for_krylov(monkeypatch) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    captured: dict[str, object] = {}

    def _fake_run_runtime_linear(_cfg, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(gamma=0.1, omega=0.2)

    monkeypatch.setattr(mod, "run_runtime_linear", _fake_run_runtime_linear)

    args = SimpleNamespace(
        time_fit_signal="auto",
        Nl=16,
        Nm=48,
        dt=0.01,
        steps=4000,
        method="rk4",
        mode_method="z_index",
        no_auto_window=False,
        tmin=None,
        tmax=None,
        sample_stride=1,
        krylov_gx_shift=True,
        krylov_gx_shift_source="target",
    )

    result = mod._run_candidate(
        args,
        cfg=KBMBaseCase(),
        ky_value=0.3,
        beta_value=0.015,
        solver_name="krylov",
        gx_gamma=0.219,
        gx_omega=1.141,
    )

    assert result.gamma == 0.1
    krylov_cfg = captured["krylov_cfg"]
    assert krylov_cfg is not None
    assert krylov_cfg.shift == complex(0.219, -1.141)
    assert krylov_cfg.shift_source == "target"
    assert krylov_cfg.omega_sign == 0
    assert krylov_cfg.omega_target_factor == 0.0
    assert krylov_cfg.shift_selection == "shift"


def test_compare_gx_kbm_run_candidate_skips_gx_shift_for_non_krylov(
    monkeypatch,
) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    captured: dict[str, object] = {}

    def _fake_run_runtime_linear(_cfg, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(gamma=0.1, omega=0.2)

    monkeypatch.setattr(mod, "run_runtime_linear", _fake_run_runtime_linear)

    args = SimpleNamespace(
        time_fit_signal="auto",
        Nl=16,
        Nm=48,
        dt=0.01,
        steps=4000,
        method="rk4",
        mode_method="z_index",
        no_auto_window=False,
        tmin=None,
        tmax=None,
        sample_stride=1,
        krylov_gx_shift=True,
        krylov_gx_shift_source="target",
    )

    mod._run_candidate(
        args,
        cfg=KBMBaseCase(),
        ky_value=0.3,
        beta_value=0.015,
        solver_name="gx_time",
        gx_gamma=0.219,
        gx_omega=1.141,
    )

    assert captured["krylov_cfg"] is None
    assert captured["solver"] == "explicit_time"


def test_compare_gx_kbm_run_candidate_honors_mode_method_override(monkeypatch) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    captured: dict[str, object] = {}

    def _fake_run_runtime_linear(_cfg, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(gamma=0.1, omega=0.2)

    monkeypatch.setattr(mod, "run_runtime_linear", _fake_run_runtime_linear)

    args = SimpleNamespace(
        time_fit_signal="auto",
        Nl=16,
        Nm=48,
        dt=0.01,
        steps=4000,
        method="rk4",
        mode_method="project",
        no_auto_window=False,
        tmin=None,
        tmax=None,
        sample_stride=1,
        krylov_gx_shift=False,
        krylov_gx_shift_source="target",
    )

    mod._run_candidate(
        args,
        cfg=KBMBaseCase(),
        ky_value=0.3,
        beta_value=0.015,
        solver_name="gx_time",
        mode_method_override="max",
        gx_gamma=0.219,
        gx_omega=1.141,
    )

    assert captured["mode_method"] == "max"
    assert captured["solver"] == "explicit_time"


def test_compare_gx_kbm_run_candidate_strips_late_fit_suffix(monkeypatch) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    captured: dict[str, object] = {}

    def _fake_run_runtime_linear(_cfg, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(gamma=0.1, omega=0.2)

    monkeypatch.setattr(mod, "run_runtime_linear", _fake_run_runtime_linear)

    args = SimpleNamespace(
        time_fit_signal="auto",
        Nl=16,
        Nm=48,
        dt=0.01,
        steps=4000,
        method="rk4",
        mode_method="project",
        no_auto_window=False,
        tmin=None,
        tmax=None,
        sample_stride=1,
        krylov_gx_shift=False,
        krylov_gx_shift_source="target",
    )

    mod._run_candidate(
        args,
        cfg=KBMBaseCase(),
        ky_value=0.3,
        beta_value=0.015,
        solver_name="gx_time",
        mode_method_override="project_late",
        gx_gamma=0.219,
        gx_omega=1.141,
    )

    assert captured["mode_method"] == "project"
    assert captured["solver"] == "explicit_time"


def test_compare_gx_kbm_run_candidate_cached_reuses_gx_time_trajectory(
    monkeypatch,
) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    calls: list[str] = []

    def _fake_run_candidate(*_args, **_kwargs):
        calls.append("run")
        return SimpleNamespace(
            gamma=0.1,
            omega=0.2,
            t=[0.0, 1.0],
            phi_t=[[[[1.0 + 0.0j]]], [[[2.0 + 0.0j]]]],
            selection=SimpleNamespace(ky_index=0, kx_index=0, z_index=0),
        )

    def _fake_recompute(args, result, *, mode_method):
        calls.append(f"recompute:{mode_method}")
        return SimpleNamespace(
            gamma=1.1 if mode_method == "project" else 1.2,
            omega=-2.1 if mode_method == "project" else -2.2,
            t=result.t,
            phi_t=result.phi_t,
            selection=result.selection,
        )

    monkeypatch.setattr(mod, "_run_candidate", _fake_run_candidate)
    monkeypatch.setattr(mod, "_recompute_time_history_growth", _fake_recompute)

    args = SimpleNamespace(
        mode_method="project",
        dt=0.01,
        steps=4000,
        method="rk4",
    )
    cache: dict[tuple[object, ...], object] = {}

    result_project = mod._run_candidate_cached(
        args,
        cfg=KBMBaseCase(),
        ky_value=0.3,
        beta_value=0.015,
        solver_name="gx_time",
        mode_method_override="project",
        result_cache=cache,
        gx_time_ref=None,
        gx_gamma=0.2,
        gx_omega=-1.0,
    )
    result_max = mod._run_candidate_cached(
        args,
        cfg=KBMBaseCase(),
        ky_value=0.3,
        beta_value=0.015,
        solver_name="gx_time",
        mode_method_override="max",
        result_cache=cache,
        gx_time_ref=None,
        gx_gamma=0.2,
        gx_omega=-1.0,
    )

    assert calls == ["run", "recompute:project", "recompute:max"]
    assert result_project.gamma == 1.1
    assert result_max.gamma == 1.2


def test_compare_gx_kbm_run_candidate_cached_loads_saved_trajectory(
    monkeypatch, tmp_path: Path
) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    base = SimpleNamespace(
        gamma=0.1,
        omega=0.2,
        ky=0.3,
        t=np.array([0.0, 1.0], dtype=float),
        phi_t=np.array([[[[1.0 + 0.0j]]], [[[2.0 + 0.0j]]]], dtype=np.complex128),
        gamma_t=np.array([[[0.1]], [[0.2]]], dtype=float),
        omega_t=np.array([[[1.0]], [[1.1]]], dtype=float),
        selection=SimpleNamespace(ky_index=0, kx_index=0, z_index=0),
    )
    mod._save_trajectory(tmp_path / "kbm_ky_0p3000_trajectory.npz", base)

    def _unexpected_run(*_args, **_kwargs):
        raise AssertionError("trajectory reuse should avoid rerunning gx_time dynamics")

    def _fake_recompute(_args, result, *, mode_method, t_ref=None):
        assert t_ref is None
        assert np.array_equal(np.asarray(result.t), np.asarray(base.t))
        assert mode_method == "project"
        return SimpleNamespace(
            gamma=0.77,
            omega=1.55,
            t=result.t,
            field_history=result.field_history,
            selection=result.selection,
        )

    monkeypatch.setattr(mod, "_run_candidate", _unexpected_run)
    monkeypatch.setattr(mod, "_recompute_time_history_growth_on_grid", _fake_recompute)

    args = SimpleNamespace(
        mode_method="project",
        dt=0.01,
        steps=4000,
        method="rk4",
        trajectory_dir=tmp_path,
        reuse_trajectory=True,
    )
    cache: dict[tuple[object, ...], object] = {}

    result = mod._run_candidate_cached(
        args,
        cfg=KBMBaseCase(),
        ky_value=0.3,
        beta_value=0.015,
        solver_name="gx_time",
        mode_method_override="project",
        result_cache=cache,
        gx_time_ref=None,
        gx_gamma=0.2,
        gx_omega=-1.0,
    )

    assert result.gamma == 0.77
    assert result.omega == 1.55


def test_compare_gx_kbm_recompute_on_gx_time_grid(monkeypatch) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))
    from gkx.benchmarking.shared import LinearRunResult

    captured: dict[str, object] = {}

    def _fake_recompute(_args, result, *, mode_method: str):
        captured["t"] = np.asarray(result.t)
        captured["phi_shape"] = np.asarray(result.phi_t).shape
        captured["mode_method"] = mode_method
        return result

    monkeypatch.setattr(mod, "_recompute_time_history_growth", _fake_recompute)

    result = LinearRunResult(
        t=np.array([0.0, 1.0, 2.0], dtype=float),
        phi_t=np.array(
            [[[[1.0 + 0.0j]]], [[[2.0 + 0.0j]]], [[[3.0 + 0.0j]]]], dtype=np.complex128
        ),
        gamma=0.0,
        omega=0.0,
        ky=0.3,
        selection=SimpleNamespace(ky_index=0, kx_index=0, z_index=0),
    )
    gx_time = np.array([0.0, 0.5, 1.5], dtype=float)

    sampled = mod._recompute_time_history_growth_on_grid(
        SimpleNamespace(),
        result,
        mode_method="project",
        t_ref=gx_time,
    )

    assert np.array_equal(np.asarray(captured["t"]), np.asarray(result.t))
    assert captured["phi_shape"] == np.asarray(result.phi_t).shape
    assert captured["mode_method"] == "project"
    assert np.array_equal(np.asarray(sampled.t), np.asarray(result.t))
    assert np.array_equal(np.asarray(sampled.phi_t), np.asarray(result.phi_t))


def test_compare_gx_kbm_recompute_on_gx_time_grid_prefers_instantaneous_omega_series() -> (
    None
):
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))
    from gkx.benchmarking.shared import LinearRunResult

    result = LinearRunResult(
        t=np.array([0.0, 1.0, 2.0], dtype=float),
        phi_t=np.array(
            [[[[1.0 + 0.0j]]], [[[2.0 + 0.0j]]], [[[3.0 + 0.0j]]]], dtype=np.complex128
        ),
        gamma=0.0,
        omega=0.0,
        ky=0.3,
        selection=SimpleNamespace(ky_index=0, kx_index=0, z_index=0),
        gamma_t=np.array([[[1.0]], [[3.0]], [[5.0]]], dtype=float),
        omega_t=np.array([[[2.0]], [[4.0]], [[6.0]]], dtype=float),
    )
    gx_time = np.array([0.5, 1.5], dtype=float)

    sampled = mod._recompute_time_history_growth_on_grid(
        SimpleNamespace(gx_avg_fraction=0.5),
        result,
        mode_method="z_index",
        t_ref=gx_time,
    )

    assert np.isclose(sampled.gamma, 4.0)
    assert np.isclose(sampled.omega, 5.0)
    assert np.array_equal(np.asarray(sampled.t), np.asarray(result.t))
    assert np.array_equal(np.asarray(sampled.phi_t), np.asarray(result.phi_t))


def test_compare_gx_kbm_recompute_project_uses_fit_window(monkeypatch) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))
    from gkx.benchmarking.shared import LinearRunResult

    calls: dict[str, object] = {}

    def _fake_extract(phi_t, sel, *, method: str):
        del phi_t, sel
        calls["method"] = method
        return np.array([1.0 + 0.0j, 1.1 - 0.1j, 1.2 - 0.2j], dtype=np.complex128)

    def _fake_fit_auto(t, signal, **kwargs):
        del t, kwargs
        calls["signal_len"] = int(np.asarray(signal).shape[0])
        return 0.33, 1.44, 0.0, 0.0

    monkeypatch.setattr(mod, "extract_mode_time_series", _fake_extract)
    monkeypatch.setattr(mod, "fit_growth_rate_auto", _fake_fit_auto)
    monkeypatch.setattr(
        mod,
        "instantaneous_growth_rate_from_phi",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected GX ratio fit")
        ),
    )

    result = LinearRunResult(
        t=np.array([0.0, 1.0, 2.0], dtype=float),
        phi_t=np.array(
            [[[[1.0 + 0.0j]]], [[[2.0 + 0.0j]]], [[[3.0 + 0.0j]]]], dtype=np.complex128
        ),
        gamma=0.0,
        omega=0.0,
        ky=0.3,
        selection=SimpleNamespace(ky_index=0, kx_index=0, z_index=0),
    )

    out = mod._recompute_time_history_growth(
        SimpleNamespace(tmin=None, tmax=None),
        result,
        mode_method="project",
    )

    assert calls["method"] == "project"
    assert calls["signal_len"] == 3
    assert np.isclose(out.gamma, 0.33)
    assert np.isclose(out.omega, 1.44)
    assert np.isclose(out.fit_window_tmin, 0.0)
    assert np.isclose(out.fit_window_tmax, 0.0)


def test_compare_gx_kbm_recompute_project_late_uses_late_fit_policy(
    monkeypatch,
) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))
    from gkx.benchmarking.shared import LinearRunResult

    calls: dict[str, object] = {}

    def _fake_extract(phi_t, sel, *, method: str):
        del phi_t, sel
        calls["method"] = method
        return np.array([1.0 + 0.0j, 1.1 - 0.1j, 1.2 - 0.2j], dtype=np.complex128)

    def _fake_fit_auto(t, signal, **kwargs):
        del t, signal
        calls["kwargs"] = kwargs
        return 0.22, 0.88, 6.0, 9.6

    monkeypatch.setattr(mod, "extract_mode_time_series", _fake_extract)
    monkeypatch.setattr(mod, "fit_growth_rate_auto", _fake_fit_auto)

    result = LinearRunResult(
        t=np.array([0.0, 1.0, 2.0], dtype=float),
        phi_t=np.array(
            [[[[1.0 + 0.0j]]], [[[2.0 + 0.0j]]], [[[3.0 + 0.0j]]]], dtype=np.complex128
        ),
        gamma=0.0,
        omega=0.0,
        ky=0.3,
        selection=SimpleNamespace(ky_index=0, kx_index=0, z_index=0),
    )

    out = mod._recompute_time_history_growth(
        SimpleNamespace(tmin=None, tmax=None),
        result,
        mode_method="project_late",
    )

    assert calls["method"] == "project"
    assert calls["kwargs"]["window_method"] == "fixed"
    assert calls["kwargs"]["window_fraction"] == pytest.approx(
        mod.LATE_PROJECT_WINDOW_FRACTION
    )
    assert calls["kwargs"]["min_points"] == mod.LATE_PROJECT_MIN_POINTS
    assert calls["kwargs"]["start_fraction"] == pytest.approx(
        mod.LATE_PROJECT_START_FRACTION
    )
    assert calls["kwargs"]["growth_weight"] == pytest.approx(
        mod.LATE_PROJECT_GROWTH_WEIGHT
    )
    assert np.isclose(out.gamma, 0.22)
    assert np.isclose(out.omega, 0.88)
    assert np.isclose(out.fit_window_tmin, 6.0)
    assert np.isclose(out.fit_window_tmax, 9.6)


def test_compare_gx_kbm_run_candidate_allows_shift_source_override(monkeypatch) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    captured: dict[str, object] = {}

    def _fake_run_runtime_linear(_cfg, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(gamma=0.1, omega=0.2)

    monkeypatch.setattr(mod, "run_runtime_linear", _fake_run_runtime_linear)

    args = SimpleNamespace(
        time_fit_signal="auto",
        Nl=16,
        Nm=48,
        dt=0.01,
        steps=4000,
        method="rk4",
        mode_method="z_index",
        no_auto_window=False,
        tmin=None,
        tmax=None,
        sample_stride=1,
        krylov_gx_shift=True,
        krylov_gx_shift_source="propagator",
    )

    mod._run_candidate(
        args,
        cfg=KBMBaseCase(),
        ky_value=0.3,
        beta_value=0.015,
        solver_name="krylov",
        gx_gamma=0.219,
        gx_omega=1.141,
    )

    krylov_cfg = captured["krylov_cfg"]
    assert krylov_cfg is not None
    assert krylov_cfg.shift_source == "propagator"


def test_compare_gx_kbm_parser_defaults_to_project_mode() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    parser = mod.build_parser()
    args = parser.parse_args(["--gx", "kbm.out.nc"])

    assert args.mode_method == "project"
    assert args.steps is None
    assert args.branch_policy == "continuation"
    assert args.gx_input is None
    assert (
        args.branch_solvers
        == "gx_time@project,gx_time@project_late,gx_time@svd,gx_time@svd_late,gx_time@max,gx_time@z_index,krylov,time"
    )


def test_compare_gx_kbm_loads_gx_input_contract(tmp_path: Path) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    gx_in = tmp_path / "kbm_miller.in"
    gx_in.write_text(
        """
[Dimensions]
ntheta = 32
nperiod = 2

[Domain]
y0 = 10.0

[Physics]
beta = 0.015

[Initialization]
init_field = "all"
gaussian_init = true
init_electrons_only = true

[Geometry]
rhoc = 0.5
Rmaj = 2.77778
qinp = 1.4
shat = 0.8

[species]
z = [1.0, -1.0]
mass = [1.0, 2.7e-4]
temp = [1.0, 1.0]
tprim = [2.49, 2.49]
fprim = [0.8, 0.8]
""".strip()
    )

    contract = mod._load_kbm_reference_input_contract(gx_in)
    assert contract.mass_ratio == pytest.approx(1.0 / 2.7e-4)
    assert contract.init_electrons_only is True
    assert contract.init_field == "all"
    assert contract.eps == pytest.approx(0.5 / 2.77778)


def test_compare_gx_kbm_runtime_conversion_preserves_physical_case() -> None:
    from tools.comparison import compare_gx_kbm as mod
    from gkx.config import KBMBaseCase

    case = KBMBaseCase()
    runtime = mod._runtime_config_from_kbm_case(case)

    assert runtime.grid == case.grid
    assert runtime.geometry == case.geometry
    assert runtime.init == case.init
    assert runtime.physics.beta == pytest.approx(case.model.beta)
    assert runtime.species[0].tprim == pytest.approx(case.model.tprim_i)
    assert runtime.species[1].tprim == pytest.approx(case.model.tprim_e)
    assert runtime.species[1].mass == pytest.approx(1.0 / case.model.mass_ratio)


def test_compare_gx_kbm_candidate_row_captures_branch_metrics() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    result = SimpleNamespace(gamma=0.8, omega=-1.5)
    row = mod._candidate_row(
        ky=0.3,
        solver="gx_time",
        result=result,
        gx_gamma=1.0,
        gx_omega=-2.0,
        eig_overlap_gx=0.9,
        eig_rel_l2=0.1,
        eig_overlap_prev=0.8,
        branch_score=0.42,
        selected=True,
    )

    assert row["ky"] == 0.3
    assert row["solver"] == "gx_time"
    assert row["gamma"] == 0.8
    assert row["omega"] == -1.5
    assert row["rel_gamma"] == pytest.approx(0.2)
    assert row["rel_omega"] == pytest.approx(0.25)
    assert row["eig_overlap_gx"] == 0.9
    assert row["eig_rel_l2"] == 0.1
    assert row["eig_overlap_prev"] == 0.8
    assert row["branch_score"] == 0.42
    assert row["selected"] is True
    assert np.isnan(row["fit_window_tmin"])
    assert np.isnan(row["fit_window_tmax"])


def test_compare_gx_kbm_candidate_row_captures_fit_window() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    result = SimpleNamespace(
        gamma=0.8, omega=-1.5, fit_window_tmin=6.0, fit_window_tmax=9.6
    )
    row = mod._candidate_row(
        ky=0.3,
        solver="gx_time@project_late",
        result=result,
        gx_gamma=1.0,
        gx_omega=-2.0,
        eig_overlap_gx=0.9,
        eig_rel_l2=0.1,
        eig_overlap_prev=0.8,
        branch_score=0.42,
        selected=True,
    )

    assert row["fit_window_tmin"] == pytest.approx(6.0)
    assert row["fit_window_tmax"] == pytest.approx(9.6)


def test_compare_gx_kbm_branch_gate_report_from_rows() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    rows = [
        {"ky": 0.1, "gamma": 0.10, "omega": 0.60, "eig_overlap_prev": np.nan},
        {"ky": 0.2, "gamma": 0.105, "omega": 0.62, "eig_overlap_prev": 0.98},
        {"ky": 0.3, "gamma": 0.110, "omega": 0.64, "eig_overlap_prev": 0.97},
    ]

    report = mod._branch_gate_report_from_rows(
        rows,
        max_rel_gamma_jump=0.1,
        max_rel_omega_jump=0.1,
        min_successive_overlap=0.95,
    )

    assert report is not None
    assert report["case"] == "kbm_linear_branch_continuity"
    assert report["passed"] is True
    assert {gate["metric"] for gate in report["gates"]} == {
        "max_rel_gamma_jump",
        "max_rel_omega_jump",
        "successive_overlap_deficit",
    }

    assert mod._branch_gate_report_from_rows(rows[:1]) is None


def test_compare_gx_kbm_parse_candidate_spec_supports_mode_override() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    solver, mode_method, label = mod._parse_candidate_spec("gx_time@max")

    assert solver == "gx_time"
    assert mode_method == "max"
    assert label == "gx_time@max"


def test_compare_gx_kbm_parse_candidate_spec_without_override() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    solver, mode_method, label = mod._parse_candidate_spec("krylov")

    assert solver == "krylov"
    assert mode_method is None
    assert label == "krylov"


def test_compare_gx_kbm_loads_npz_reference(tmp_path: Path) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    path = tmp_path / "kbm_ref.npz"
    np.savez(
        path,
        time=np.array([0.1, 0.2], dtype=float),
        ky=np.array([0.1, 0.2, 0.3], dtype=float),
        omega_series=np.zeros((2, 3, 2), dtype=float),
        beta=np.asarray(0.015),
        q=np.asarray(1.4),
        shat=np.asarray(0.8),
        eps=np.asarray(0.18),
        rmaj=np.asarray(2.77778),
        theta=np.array([-1.0, 0.0, 1.0], dtype=float),
        phi_modes=np.array(
            [
                [1.0 + 0.0j, 2.0 + 0.0j, 1.0 + 0.0j],
                [2.0 + 0.0j, 4.0 + 0.0j, 2.0 + 0.0j],
                [1.0j, 2.0j, 1.0j],
            ],
            dtype=np.complex128,
        ),
    )

    time, ky, omega, beta, q, shat, eps, rmaj = mod._load_gx_omega_gamma(path)
    theta, mode = mod._load_gx_eigenfunction(path, 0.2)

    assert np.allclose(time, np.array([0.1, 0.2]))
    assert np.allclose(ky, np.array([0.1, 0.2, 0.3]))
    assert omega.shape == (2, 3, 2)
    assert beta == pytest.approx(0.015)
    assert q == pytest.approx(1.4)
    assert shat == pytest.approx(0.8)
    assert eps == pytest.approx(0.18)
    assert rmaj == pytest.approx(2.77778)
    assert np.allclose(theta, np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(mode, np.array([0.5 + 0.0j, 1.0 + 0.0j, 0.5 + 0.0j]))


def test_compare_gx_kbm_npz_zero_geometry_scalars_fall_back_to_defaults(
    tmp_path: Path,
) -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    path = tmp_path / "kbm_ref_zero_geom.npz"
    np.savez(
        path,
        time=np.array([0.1, 0.2], dtype=float),
        ky=np.array([0.1, 0.2, 0.3], dtype=float),
        omega_series=np.zeros((2, 3, 2), dtype=float),
        beta=np.asarray(0.015),
        q=np.asarray(0.0),
        shat=np.asarray(0.0),
        eps=np.asarray(0.0),
        rmaj=np.asarray(0.0),
        theta=np.array([-1.0, 0.0, 1.0], dtype=float),
        phi_modes=np.ones((3, 3), dtype=np.complex128),
    )

    _time, _ky, _omega, beta, q, shat, eps, rmaj = mod._load_gx_omega_gamma(path)

    assert beta == pytest.approx(0.015)
    assert q is None
    assert shat is None
    assert eps is None
    assert rmaj is None


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

    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
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

    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
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

    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
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
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
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
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
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
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
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
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
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
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
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
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
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
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear as mod
    finally:
        sys.path.remove(str(tools_dir))

    mask = mod._resolve_dealias_mask(np.ones((4, 4), dtype=bool), ny=10, nx=4)

    assert mask.shape == (10, 4)


def test_synth_positive_and_full_ky_rebuild_dump_grid() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
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

from gkx.benchmarking.shared import (
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
    _build_initial_condition,
    _two_species_params,
)
from gkx.config import KBMBaseCase
from gkx.core.grid import select_ky_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.terms.assembly import assemble_rhs_terms_cached, compute_fields_cached
from gkx.terms.config import TermConfig


def test_manual_linear_contributions_match_assembly_for_multispecies_kbm() -> None:
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
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
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
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
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
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
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
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
    tools_dir = Path(__file__).resolve().parents[3] / "tools" / "comparison"
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


def test_runtime_linear_accepts_vmec_and_desc_eik_geometry_aliases(
    tmp_path: Path,
) -> None:
    from gkx.runtime import run_runtime_linear
    from tools.comparison.compare_gx_kbm import _runtime_config_from_kbm_case

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


def test_ky_diagnostics_build_problem_seeds_multispecies_tem() -> None:
    from tools.comparison import ky_diagnostics as mod

    _cfg, grid, _geom, params, _terms, G0 = mod._build_problem("tem", 0.3, None, 4, 6)

    assert np.asarray(G0).shape == (2, 4, 6, grid.ky.size, grid.kx.size, grid.z.size)
    assert int(np.atleast_1d(np.asarray(params.charge_sign)).shape[0]) == 2
    assert np.any(np.abs(np.asarray(G0[1])) > 0.0)
    assert np.allclose(np.asarray(G0[0]), 0.0)


def test_ky_diagnostics_etg_uses_canonical_boltzmann_ion_contract() -> None:
    from tools.comparison import ky_diagnostics as mod

    cfg, _grid, _geom, params, terms, state = mod._build_problem(
        "etg", 10.0, None, 2, 4
    )

    assert cfg.physics.adiabatic_ions is True
    assert cfg.physics.adiabatic_electrons is False
    assert state.shape[0] == 2  # Single-species states omit a species axis.
    assert float(params.tau_e) == pytest.approx(1.0)
    assert float(params.omega_d_scale) == pytest.approx(1.0)
    assert float(params.hypercollisions_const) == pytest.approx(0.0)
    assert float(params.hypercollisions_kz) == pytest.approx(1.0)
    assert terms.apar == pytest.approx(0.0)


def test_rhs_term_diagnostics_etg_uses_canonical_runtime_contract() -> None:
    from tools.comparison import compare_gx_rhs_terms as mod

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
    from tools.comparison import compare_gx_rhs_terms as mod

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


# ---- compare_runtime.py ----

from netCDF4 import Dataset


def test_compare_runtime_stress_matrix_parser_and_case_layout(tmp_path: Path) -> None:
    from tools.comparison import compare_runtime as mod

    parser = mod.build_stress_matrix_parser()
    args = parser.parse_args(
        ["--comparison-repo", str(tmp_path), "--cases", "kaw", "--Nl", "6"]
    )
    cases = mod._linear_stress_cases(tmp_path)

    assert args.comparison_repo == tmp_path
    assert args.cases == ["kaw"]
    assert args.Nl == 6
    assert set(cases) == {"kaw", "cyclone_ke", "kbm_miller"}
    assert cases["kaw"][0].name == "kaw_betahat10.0_kp0.01_correct.out.nc"


def test_compare_runtime_stress_case_writes_labeled_frame(
    tmp_path: Path, monkeypatch
) -> None:
    from tools.comparison import compare_runtime as mod

    output = tmp_path / "case.out.nc"
    input_file = tmp_path / "case.in"
    output.touch()
    input_file.touch()
    captured: list[list[str]] = []

    def _run(command, *, check):
        assert check is True
        captured.append(command)
        out = Path(command[command.index("--out") + 1])
        out.write_text("ky,gamma\n0.2,0.1\n", encoding="utf-8")

    monkeypatch.setattr(mod.subprocess, "run", _run)
    frame = mod._run_linear_stress_case(
        name="case",
        output=output,
        input_file=input_file,
        out_csv=tmp_path / "result.csv",
        nl=4,
        nm=8,
    )

    assert frame.to_dict("records") == [{"case": "case", "ky": 0.2, "gamma": 0.1}]
    assert captured[0][captured[0].index("--Nl") + 1] == "4"
    assert captured[0][captured[0].index("--Nm") + 1] == "8"


def test_compare_runtime_startup_select_ky_block_slices_third_to_last_axis() -> None:
    from tools.comparison import compare_runtime as mod

    arr = np.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)
    sliced = mod._select_ky_block(arr, 1)

    assert sliced.shape == (2, 3, 1, 5, 6)
    assert np.array_equal(sliced[:, :, 0, :, :], arr[:, :, 1, :, :])


def test_compare_runtime_startup_infers_full_ny_from_positive_ky() -> None:
    from tools.comparison import compare_runtime as mod

    assert mod._full_ny_from_positive_ky(np.array([0.1, 0.2, 0.3, 0.4])) == 10


def test_compare_runtime_startup_parser_requires_core_args() -> None:
    from tools.comparison import compare_runtime as mod

    parser = mod.build_startup_parser()
    args = parser.parse_args(
        [
            "--gx-dir",
            "gx_dump",
            "--gx-out",
            "gx.out.nc",
            "--config",
            "runtime.toml",
            "--ky",
            "0.3",
        ]
    )

    assert args.gx_dir == Path("gx_dump")
    assert args.gx_out == Path("gx.out.nc")
    assert args.config == Path("runtime.toml")
    assert args.ky == 0.3


def test_compare_runtime_startup_builds_full_grid_before_slicing(
    tmp_path: Path, monkeypatch
) -> None:
    from tools.comparison import compare_runtime as mod

    gx_out = tmp_path / "gx.out.nc"
    with Dataset(gx_out, "w") as ds:
        grids = ds.createGroup("Grids")
        grids.createDimension("ky", 2)
        ky = grids.createVariable("ky", "f8", ("ky",))
        ky[:] = [0.1, 0.2]

    monkeypatch.setattr(
        mod,
        "_load_shape",
        lambda _path: {"nspec": 1, "nl": 1, "nm": 1, "nyc": 2, "nx": 1, "nz": 2},
    )
    monkeypatch.setattr(
        mod, "_load_bin", lambda *_args, **_kwargs: np.zeros(4, dtype=np.complex64)
    )
    gx_g = np.arange(4, dtype=np.complex64).reshape(1, 1, 1, 2, 1, 2)
    gx_phi = np.arange(4, dtype=np.complex64).reshape(2, 1, 2)
    monkeypatch.setattr(mod, "_reshape_gx", lambda *_args, **_kwargs: gx_g)
    monkeypatch.setattr(mod, "_load_field", lambda *_args, **_kwargs: gx_phi)
    monkeypatch.setattr(
        mod,
        "replace",
        lambda obj, **updates: SimpleNamespace(**(obj.__dict__ | updates)),
    )
    monkeypatch.setattr(
        mod,
        "load_runtime_from_toml",
        lambda _path: (
            SimpleNamespace(
                grid=SimpleNamespace(Nx=1, Ny=4, Nz=2, y0=10.0), species=[object()]
            ),
            None,
        ),
    )
    monkeypatch.setattr(mod, "build_runtime_geometry", lambda _cfg: object())
    monkeypatch.setattr(
        mod, "apply_imported_geometry_grid_defaults", lambda _geom, grid: grid
    )
    grid_full = SimpleNamespace(ky=np.array([0.0, 0.1, 0.2, -0.1]), kx=np.array([0.0]))
    monkeypatch.setattr(mod, "build_spectral_grid", lambda _grid: grid_full)
    monkeypatch.setattr(
        mod, "build_runtime_linear_params", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(mod, "_species_to_linear", lambda _species: [object()])

    captured: dict[str, object] = {}

    def _fake_build_initial_condition(
        grid, _geom, _cfg, *, ky_index, kx_index, Nl, Nm, nspecies
    ):
        captured["grid"] = grid
        captured["ky_index"] = ky_index
        captured["kx_index"] = kx_index
        captured["Nl"] = Nl
        captured["Nm"] = Nm
        captured["nspecies"] = nspecies
        return np.arange(8, dtype=np.complex64).reshape(1, 1, 1, 4, 1, 2)

    monkeypatch.setattr(mod, "_build_initial_condition", _fake_build_initial_condition)
    monkeypatch.setattr(mod, "build_linear_cache", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mod, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        mod,
        "compute_fields_cached",
        lambda *_args, **_kwargs: SimpleNamespace(
            phi=np.arange(8, dtype=np.complex64).reshape(4, 1, 2), apar=None
        ),
    )

    summaries: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def _fake_summary(name, ref, test):
        summaries.append(
            (name, tuple(np.asarray(ref).shape), tuple(np.asarray(test).shape))
        )

    monkeypatch.setattr(mod, "_summary", _fake_summary)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_runtime.py startup",
            "--gx-dir",
            str(tmp_path),
            "--gx-out",
            str(gx_out),
            "--config",
            str(tmp_path / "runtime.toml"),
            "--ky",
            "0.2",
        ],
    )

    mod.main_startup()

    assert captured["grid"] is grid_full
    assert captured["ky_index"] == 2
    assert captured["kx_index"] == 0
    assert ("g_state", (1, 1, 1, 1, 1, 2), (1, 1, 1, 1, 1, 2)) in summaries
    assert ("phi", (1, 1, 2), (1, 1, 2)) in summaries


def test_compare_runtime_diagnostic_state_parser_requires_core_args() -> None:
    from tools.comparison import compare_runtime as mod

    parser = mod.build_diagnostic_state_parser()
    args = parser.parse_args(
        [
            "--gx-dir",
            "gx_dump",
            "--gx-out",
            "gx.out.nc",
            "--config",
            "runtime.toml",
            "--time-index",
            "10",
        ]
    )

    assert args.gx_dir == Path("gx_dump")
    assert args.gx_out == Path("gx.out.nc")
    assert args.config == Path("runtime.toml")
    assert args.time_index == 10


def test_compare_runtime_diagnostic_state_builds_positive_ky_grid_and_writes_csv(
    tmp_path: Path, monkeypatch
) -> None:
    from tools.comparison import compare_runtime as mod

    gx_out = tmp_path / "gx.out.nc"
    with Dataset(gx_out, "w") as ds:
        ds.createDimension("l", 1)
        ds.createDimension("m", 1)
        ds.createDimension("s", 1)
        ds.createDimension("kx", 1)
        ds.createDimension("ky", 2)
        ds.createDimension("theta", 2)
        ds.createDimension("time", 1)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        ky = grids.createVariable("ky", "f8", ("ky",))
        ky[:] = [0.1, 0.2]
        kx = grids.createVariable("kx", "f8", ("kx",))
        kx[:] = [0.0]
        time = grids.createVariable("time", "f8", ("time",))
        time[:] = [3.0]
        for name, value in {
            "Wg_kyst": 1.5,
            "Wphi_kyst": 2.5,
            "Wapar_kyst": 0.0,
            "HeatFlux_kyst": 3.5,
            "ParticleFlux_kyst": 0.0,
        }.items():
            var = diag.createVariable(name, "f8", ("time", "s", "ky"))
            var[:] = value
        heat_s = diag.createVariable("HeatFlux_st", "f8", ("time", "s"))
        heat_s[:] = [[3.5]]
        pflux_s = diag.createVariable("ParticleFlux_st", "f8", ("time", "s"))
        pflux_s[:] = [[0.0]]

    gx_dir = tmp_path / "gx_dump"
    gx_dir.mkdir()
    np.arange(4, dtype=np.complex64).tofile(gx_dir / "diag_state_G_s0_t0.bin")
    np.arange(4, dtype=np.complex64).tofile(gx_dir / "diag_state_phi_t0.bin")
    np.arange(4, dtype=np.float32).tofile(gx_dir / "diag_state_kperp2_t0.bin")
    np.arange(2, dtype=np.float32).tofile(gx_dir / "diag_state_fluxfac_t0.bin")
    np.array([0.0], dtype=np.float32).tofile(gx_dir / "diag_state_kx_t0.bin")
    np.array([0.1, 0.2], dtype=np.float32).tofile(gx_dir / "diag_state_ky_t0.bin")

    monkeypatch.setattr(
        mod,
        "replace",
        lambda obj, **updates: SimpleNamespace(**(obj.__dict__ | updates)),
    )
    monkeypatch.setattr(
        mod,
        "load_runtime_from_toml",
        lambda _path: (
            SimpleNamespace(
                grid=SimpleNamespace(Nx=1, Ny=4, Nz=2, y0=10.0),
                species=[object()],
                normalization=SimpleNamespace(flux_scale=1.0, wphi_scale=1.0),
            ),
            None,
        ),
    )
    monkeypatch.setattr(mod, "build_runtime_geometry", lambda _cfg: object())
    monkeypatch.setattr(
        mod, "ensure_flux_tube_geometry_data", lambda geom, _theta: geom
    )
    monkeypatch.setattr(
        mod, "apply_imported_geometry_grid_defaults", lambda _geom, grid: grid
    )
    grid_full = SimpleNamespace(
        ky=np.array([0.0, 0.1, 0.2, -0.1]), kx=np.array([0.0]), z=np.array([0.0, 1.0])
    )
    monkeypatch.setattr(mod, "build_spectral_grid", lambda _grid: grid_full)
    grid_pos = SimpleNamespace(
        ky=np.array([0.1, 0.2]), kx=np.array([0.0]), z=np.array([0.0, 1.0])
    )
    captured: dict[str, object] = {}

    def _fake_select_real_fft_ky_grid(grid, ky_vals):
        captured["grid"] = grid
        captured["ky_vals"] = np.asarray(ky_vals)
        return grid_pos

    monkeypatch.setattr(mod, "select_real_fft_ky_grid", _fake_select_real_fft_ky_grid)
    monkeypatch.setattr(
        mod, "build_runtime_linear_params", lambda *_args, **_kwargs: object()
    )
    cache = SimpleNamespace(
        kperp2=np.arange(4, dtype=np.float32).reshape(2, 1, 2),
        bmag=np.ones(2),
        kperp2_bmag=False,
    )
    monkeypatch.setattr(mod, "build_linear_cache", lambda *_args, **_kwargs: cache)
    monkeypatch.setattr(mod, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        mod,
        "compute_fields_cached",
        lambda *_args, **_kwargs: SimpleNamespace(
            phi=np.arange(4, dtype=np.complex64).reshape(2, 1, 2), apar=None, bpar=None
        ),
    )
    monkeypatch.setattr(
        mod,
        "fieldline_quadrature_weights",
        lambda *_args, **_kwargs: (np.array([0.4, 0.6]), np.array([0.0, 1.0])),
    )
    monkeypatch.setattr(mod, "distribution_free_energy", lambda *_args, **_kwargs: 1.5)
    monkeypatch.setattr(
        mod, "electrostatic_field_energy", lambda *_args, **_kwargs: 2.5
    )
    monkeypatch.setattr(
        mod, "magnetic_vector_potential_energy", lambda *_args, **_kwargs: 0.0
    )
    monkeypatch.setattr(mod, "heat_flux_total", lambda *_args, **_kwargs: 3.5)
    monkeypatch.setattr(mod, "particle_flux_total", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        mod, "heat_flux_species", lambda *_args, **_kwargs: np.array([3.5])
    )
    monkeypatch.setattr(
        mod, "particle_flux_species", lambda *_args, **_kwargs: np.array([0.0])
    )

    summaries: list[str] = []
    monkeypatch.setattr(
        mod, "_summary", lambda name, *_args, **_kwargs: summaries.append(name)
    )

    out_csv = tmp_path / "diag.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_runtime.py diagnostic-state",
            "--gx-dir",
            str(gx_dir),
            "--gx-out",
            str(gx_out),
            "--config",
            str(tmp_path / "runtime.toml"),
            "--time-index",
            "0",
            "--out",
            str(out_csv),
        ],
    )

    mod.main_diagnostic_state()

    assert captured["grid"] is grid_full
    assert np.array_equal(captured["ky_vals"], np.array([0.1, 0.2], dtype=np.float32))
    assert summaries == ["kperp2", "fluxfac", "kx", "ky", "phi"]
    text = out_csv.read_text()
    assert "metric,gx_out,gkx_dump" in text
    assert "Wg" in text


def test_compare_runtime_window_parser_requires_core_args() -> None:
    from tools.comparison import compare_runtime as mod

    parser = mod.build_window_parser()
    args = parser.parse_args(
        [
            "--gx-dir",
            "gx_dump",
            "--gx-out",
            "gx.out.nc",
            "--config",
            "runtime.toml",
            "--time-index-start",
            "10",
            "--time-index-stop",
            "11",
        ]
    )

    assert args.gx_dir == Path("gx_dump")
    assert args.gx_out == Path("gx.out.nc")
    assert args.config == Path("runtime.toml")
    assert args.time_index_start == 10
    assert args.time_index_stop == 11


def test_compare_runtime_window_parser_accepts_optional_ky_and_steps() -> None:
    from tools.comparison import compare_runtime as mod

    parser = mod.build_window_parser()
    args = parser.parse_args(
        [
            "--gx-dir",
            "gx_dump",
            "--gx-out",
            "gx.out.nc",
            "--config",
            "runtime.toml",
            "--time-index-start",
            "10",
            "--time-index-stop",
            "11",
            "--steps",
            "50",
            "--ky",
            "0.3",
        ]
    )
    assert args.steps == 50
    assert args.ky == 0.3


def test_compare_runtime_window_writes_csv(tmp_path: Path, monkeypatch) -> None:
    from tools.comparison import compare_runtime as mod

    gx_out = tmp_path / "gx.out.nc"
    with Dataset(gx_out, "w") as ds:
        ds.createDimension("l", 1)
        ds.createDimension("m", 1)
        ds.createDimension("s", 1)
        ds.createDimension("time", 2)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        time = grids.createVariable("time", "f8", ("time",))
        time[:] = [1.0, 2.0]
        for name, value in {
            "Wg_kyst": [1.0, 2.0],
            "Wphi_kyst": [3.0, 4.0],
            "Wapar_kyst": [0.0, 0.0],
            "HeatFlux_kyst": [5.0, 6.0],
            "ParticleFlux_kyst": [0.0, 0.0],
        }.items():
            var = diag.createVariable(name, "f8", ("time", "s"))
            var[:] = np.asarray(value, dtype=float)[:, None]
        heat_s = diag.createVariable("HeatFlux_st", "f8", ("time", "s"))
        heat_s[:] = [[5.0], [6.0]]
        pflux_s = diag.createVariable("ParticleFlux_st", "f8", ("time", "s"))
        pflux_s[:] = [[0.0], [0.0]]

    gx_dir = tmp_path / "gx_dump"
    gx_dir.mkdir()
    np.ones(2, dtype=np.complex64).tofile(gx_dir / "diag_state_G_s0_t0.bin")
    np.ones(2, dtype=np.complex64).tofile(gx_dir / "diag_state_G_s0_t1.bin")
    np.ones(2, dtype=np.complex64).tofile(gx_dir / "diag_state_phi_t0.bin")
    np.ones(2, dtype=np.complex64).tofile(gx_dir / "diag_state_phi_t1.bin")
    np.array([0.0], dtype=np.float32).tofile(gx_dir / "diag_state_kx_t0.bin")
    np.array([0.1], dtype=np.float32).tofile(gx_dir / "diag_state_ky_t0.bin")

    monkeypatch.setattr(
        mod,
        "replace",
        lambda obj, **updates: SimpleNamespace(**(obj.__dict__ | updates)),
    )
    monkeypatch.setattr(
        mod,
        "load_runtime_from_toml",
        lambda _path: (
            SimpleNamespace(
                grid=SimpleNamespace(Nx=1, Ny=1, Nz=2, y0=10.0),
                time=SimpleNamespace(
                    dt=0.1,
                    method="rk3",
                    nonlinear_dealias=False,
                    laguerre_nonlinear_mode="grid",
                    fixed_dt=False,
                    dt_min=1.0e-6,
                    dt_max=None,
                    cfl=1.0,
                    cfl_fac=1.73,
                    collision_split=False,
                    collision_scheme="implicit",
                    implicit_restart=20,
                    implicit_preconditioner=None,
                ),
                run=SimpleNamespace(ky=0.1),
                normalization=SimpleNamespace(flux_scale=1.0, wphi_scale=1.0),
                init=SimpleNamespace(
                    init_file=None, init_file_scale=1.0, init_file_mode="replace"
                ),
            ),
            None,
        ),
    )
    monkeypatch.setattr(mod, "build_runtime_geometry", lambda _cfg: object())
    monkeypatch.setattr(
        mod, "apply_imported_geometry_grid_defaults", lambda _geom, grid: grid
    )
    grid_full = SimpleNamespace(
        ky=np.array([0.1]), kx=np.array([0.0]), z=np.array([0.0, 1.0])
    )
    monkeypatch.setattr(mod, "build_spectral_grid", lambda _grid: grid_full)
    monkeypatch.setattr(mod, "select_real_fft_ky_grid", lambda grid, _ky: grid)
    monkeypatch.setattr(
        mod, "ensure_flux_tube_geometry_data", lambda geom, _theta: geom
    )
    monkeypatch.setattr(
        mod, "build_runtime_linear_params", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        mod, "build_linear_cache", lambda *_args, **_kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(mod, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        mod,
        "_load_real_vector_auto",
        lambda path: (
            np.array([0.0], dtype=np.float32)
            if "kx" in path.name
            else np.array([0.1], dtype=np.float32)
        ),
    )
    monkeypatch.setattr(
        mod,
        "_load_species_state",
        lambda *_args, **_kwargs: np.ones((1, 1, 1, 1, 1, 2), dtype=np.complex64),
    )
    monkeypatch.setattr(
        mod,
        "_load_field",
        lambda *_args, **_kwargs: np.ones((1, 1, 2), dtype=np.complex64),
    )
    monkeypatch.setattr(mod, "_maybe_load_field", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mod,
        "compute_fields_cached",
        lambda *_args, **_kwargs: SimpleNamespace(
            phi=np.ones((1, 1, 2), dtype=np.complex64),
            apar=None,
            bpar=None,
        ),
    )
    monkeypatch.setattr(mod, "_summary", lambda *_args, **_kwargs: None)

    diag = SimpleNamespace(
        t=np.array([1.0], dtype=float),
        dt_t=np.array([0.02], dtype=float),
        gamma_t=np.array([0.0], dtype=float),
        omega_t=np.array([0.0], dtype=float),
        Wg_t=np.array([2.0], dtype=float),
        Wphi_t=np.array([4.0], dtype=float),
        Wapar_t=np.array([0.0], dtype=float),
        energy_t=np.array([6.0], dtype=float),
        heat_flux_t=np.array([6.0], dtype=float),
        particle_flux_t=np.array([0.0], dtype=float),
        heat_flux_species_t=np.array([[6.0]], dtype=float),
        particle_flux_species_t=np.array([[0.0]], dtype=float),
    )
    monkeypatch.setattr(
        mod,
        "run_runtime_nonlinear",
        lambda *_args, **_kwargs: SimpleNamespace(
            diagnostics=diag,
        ),
    )

    out_csv = tmp_path / "window.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_runtime.py window",
            "--gx-dir",
            str(gx_dir),
            "--gx-out",
            str(gx_out),
            "--config",
            str(tmp_path / "runtime.toml"),
            "--time-index-start",
            "0",
            "--time-index-stop",
            "1",
            "--steps",
            "50",
            "--out",
            str(out_csv),
        ],
    )

    mod.main_window()

    text = out_csv.read_text()
    assert "time_index_start,time_index_stop" in text
    assert "Wg" in text


# ---- W7-X zonal-reference comparison contracts ----

from support.paths import load_artifact_tool, load_comparison_tool


def _load_tool_module():
    return load_artifact_tool("build_w7x_zonal_reference_artifacts")


def _write_reference(tmp_path: Path) -> tuple[Path, Path]:
    traces = []
    residuals = []
    for kx in (0.05, 0.07, 0.10, 0.30):
        tmax = 3500.0 if kx == 0.05 else 2000.0
        t = np.linspace(0.0, tmax, 21)
        for code, offset in (("stella", -0.002), ("GENE", 0.002)):
            residual = 0.1 + kx + offset
            for tv in t:
                traces.append(
                    {
                        "kx_rhoi": kx,
                        "code": code,
                        "t_vti_over_a": tv,
                        "response": residual + np.exp(-tv / 200.0),
                    }
                )
            residuals.append(
                {
                    "panel": "x",
                    "kx_rhoi": kx,
                    "code": code,
                    "residual_mean": residual,
                    "residual_median": residual,
                    "residual_min": residual,
                    "residual_max": residual,
                    "n_pixels": 5,
                }
            )
    trace_csv = tmp_path / "ref_traces.csv"
    residual_csv = tmp_path / "ref_residuals.csv"
    pd.DataFrame(traces).to_csv(trace_csv, index=False)
    pd.DataFrame(residuals).to_csv(residual_csv, index=False)
    return trace_csv, residual_csv


def _write_summary(
    tmp_path: Path,
    *,
    tmax_scale: float = 1.0,
    residual_shift: float = 0.0,
    initial_level: float = 1.0,
) -> Path:
    rows = []
    for kx in (0.05, 0.07, 0.10, 0.30):
        ref_tmax = 3500.0 if kx == 0.05 else 2000.0
        rows.append(
            {
                "kx_target": kx,
                "residual_level": 0.1 + kx + residual_shift,
                "residual_std": 0.01,
                "tmax": ref_tmax * tmax_scale,
                "initial_level": initial_level,
            }
        )
    path = tmp_path / "gkx_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_w7x_zonal_reference_comparison_passes_closed_synthetic_case(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    ref_traces, ref_residuals = _write_reference(tmp_path)
    summary = _write_summary(tmp_path)

    rows, report = mod.build_comparison(
        gkx_summary=summary,
        reference_traces=ref_traces,
        reference_residuals=ref_residuals,
    )

    assert report.passed is True
    assert rows["coverage_ratio"].min() == 1.0
    assert rows["residual_abs_error"].max() <= 1.0e-12


def test_w7x_zonal_reference_comparison_fails_short_window(tmp_path: Path) -> None:
    mod = _load_tool_module()
    ref_traces, ref_residuals = _write_reference(tmp_path)
    summary = _write_summary(tmp_path, tmax_scale=0.03)

    rows, report = mod.build_comparison(
        gkx_summary=summary,
        reference_traces=ref_traces,
        reference_residuals=ref_residuals,
    )

    assert report.passed is False
    assert rows["coverage_ratio"].max() < 0.98
    failed = [gate.metric for gate in report.gates if not gate.passed]
    assert "time_coverage_kx050" in failed


def test_w7x_zonal_reference_trace_metrics_use_summary_initial_level(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    ref_traces, ref_residuals = _write_reference(tmp_path)
    initial_level = 2.0
    summary = _write_summary(tmp_path, initial_level=initial_level)
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    refs = pd.read_csv(ref_traces)
    combined_rows = []
    for kx, group in refs.groupby("kx_rhoi"):
        mean_trace = group.pivot_table(
            index="t_vti_over_a", columns="code", values="response", aggfunc="mean"
        )
        mean_trace = mean_trace.sort_index().mean(axis=1)
        token = mod.kx_token(float(kx))
        time_col = "t_reference" if np.isclose(float(kx), 0.05) else "t"
        pd.DataFrame(
            {
                time_col: np.asarray(mean_trace.index, dtype=float),
                "phi_zonal_real": initial_level * np.asarray(mean_trace, dtype=float),
            }
        ).to_csv(trace_dir / f"w7x_test4_kx{token}.csv", index=False)
        for time_value, value in zip(mean_trace.index, mean_trace, strict=True):
            combined_rows.append(
                {
                    "kx_target": float(kx),
                    "t_reference": float(time_value),
                    "phi_zonal_real": initial_level * float(value),
                }
            )
    combined_trace = tmp_path / "combined_traces.csv"
    pd.DataFrame(combined_rows).to_csv(combined_trace, index=False)

    rows, report = mod.build_comparison(
        gkx_summary=summary,
        reference_traces=ref_traces,
        reference_residuals=ref_residuals,
        gkx_trace_dir=trace_dir,
        envelope_atol=1.0e-12,
    )

    assert report.passed is True
    assert rows["trace_available"].min() == 1
    assert rows["tail_mean_abs_error"].max() <= 1.0e-12

    combined_rows_out, combined_report = mod.build_comparison(
        gkx_summary=summary,
        reference_traces=ref_traces,
        reference_residuals=ref_residuals,
        gkx_traces=combined_trace,
        envelope_atol=1.0e-12,
    )
    assert combined_report.passed is True
    assert combined_rows_out["tail_mean_abs_error"].max() <= 1.0e-12


def test_w7x_zonal_reference_main_writes_open_json(tmp_path: Path) -> None:
    mod = _load_tool_module()
    ref_traces, ref_residuals = _write_reference(tmp_path)
    summary = _write_summary(tmp_path, residual_shift=1.0)
    out_csv = tmp_path / "compare.csv"
    out_json = tmp_path / "compare.json"
    out_png = tmp_path / "compare.png"

    rc = mod.main(
        [
            "compare",
            "--gkx-summary",
            str(summary),
            "--reference-traces",
            str(ref_traces),
            "--reference-residuals",
            str(ref_residuals),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
            "--out-png",
            str(out_png),
        ]
    )

    assert rc == 2
    assert out_csv.exists()
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["validation_status"] == "open"
    assert payload["gate_index_include"] is False


# ---- test_make_reference_panels.py ----

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools" / "comparison"

from make_reference_panels import (  # noqa: E402
    STATIC,
    _autocrop_image,
    _linear_table_rows,
    _load_imported_linear,
    _load_imported_linear_lastvalue,
    _load_secondary,
    _plot_imported_linear,
    _plot_secondary,
    _secondary_table_rows,
    build_parser as reference_panels_build_parser,
)


def _write_plot(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0.0, 1.0], [0.0, 1.0], linewidth=2.0)
    ax.set_title(title)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def _write_imported_linear_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "ky": [0.1, 0.2, 0.3],
            "mean_abs_omega": [1.0e-5, 2.0e-5, 3.0e-5],
            "mean_rel_omega": [1.0e-3, 2.0e-3, 3.0e-3],
            "mean_abs_gamma": [4.0e-5, 5.0e-5, 6.0e-5],
            "mean_rel_gamma": [4.0e-2, 5.0e-2, 6.0e-2],
            "mean_rel_Wg": [1.0e-2, 2.0e-2, 3.0e-2],
            "mean_rel_Wphi": [1.5e-2, 2.5e-2, 3.5e-2],
            "mean_rel_Wapar": [0.0, 0.0, 0.0],
        }
    ).to_csv(path, index=False)


def _write_imported_lastvalue_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "ky": [0.1, 0.2, 0.3],
            "rel_gamma": [5.0e-3, 4.0e-3, 3.0e-3],
            "rel_omega": [6.0e-4, 5.0e-4, 4.0e-4],
            "gamma": [0.1, 0.2, 0.3],
            "gamma_gx": [0.1, 0.2, 0.3],
            "omega": [-0.2, -0.3, -0.4],
            "omega_gx": [-0.2, -0.3, -0.4],
        }
    ).to_csv(path, index=False)


def _write_linear_mismatch_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "ky": [0.05, 0.10, 0.20],
            "gamma": [0.01, 0.03, 0.08],
            "omega": [0.03, 0.06, 0.13],
            "gamma_gx": [0.011, 0.029, 0.079],
            "omega_gx": [0.031, 0.061, 0.129],
        }
    ).to_csv(path, index=False)


def _write_secondary_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "ky": [0.1],
            "kx": [0.05],
            "gamma_gx": [4.9],
            "gamma_sp": [4.91],
            "rel_gamma": [2.0e-3],
            "omega_gx": [-1.6e-4],
            "omega_sp": [3.0e-7],
        }
    ).to_csv(path, index=False)


def test_load_secondary_adds_abs_omega_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "secondary.csv"
    _write_secondary_csv(path)
    df = _load_secondary(path)
    assert "abs_omega" in df.columns
    assert float(df.loc[0, "abs_omega"]) == pytest.approx(1.603e-4)


def test_secondary_table_rows_format_expected_values(tmp_path: Path) -> None:
    path = tmp_path / "secondary.csv"
    pd.DataFrame(
        {
            "ky": [0.0],
            "kx": [-0.05],
            "gamma_gx": [4.901835],
            "gamma_sp": [4.901937],
            "rel_gamma": [2.1e-5],
            "omega_gx": [-1.6e-4],
            "omega_sp": [2.6e-7],
            "abs_omega": [1.6026e-4],
        }
    ).to_csv(path, index=False)
    rows = _secondary_table_rows(_load_secondary(path))
    assert rows == [
        [
            "(0.00, -0.05)",
            "4.901835",
            "4.901937",
            "2.10e-05",
            "-1.60e-04",
            "2.60e-07",
            "1.60e-04",
        ]
    ]


def test_parser_defaults_to_repository_static_assets() -> None:
    args = reference_panels_build_parser().parse_args(["summary"])
    assert (
        args.secondary_csv
        == STATIC / "comparison" / "secondary_reference_out_compare.csv"
    )
    assert STATIC == ROOT / "docs" / "_static"


def test_load_imported_linear_requires_expected_columns(tmp_path: Path) -> None:
    path = tmp_path / "linear.csv"
    pd.DataFrame({"ky": [0.1], "mean_rel_omega": [1.0e-3]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        _load_imported_linear(path)


def test_load_imported_linear_lastvalue_requires_expected_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "linear_lastvalue.csv"
    pd.DataFrame({"ky": [0.1], "rel_gamma": [1.0e-3]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        _load_imported_linear_lastvalue(path)


def test_linear_table_rows_formats_scan_metrics(tmp_path: Path) -> None:
    path = tmp_path / "linear.csv"
    _write_imported_linear_csv(path)
    rows = _linear_table_rows(_load_imported_linear(path))
    assert rows[0] == ["0.100", "1.00e-05", "4.00e-05", "1.00e-02", "1.50e-02"]


def test_plot_imported_linear_adds_lines_and_log_axis() -> None:
    df = pd.DataFrame(
        {
            "ky": [0.1, 0.2],
            "mean_rel_omega": [1.0e-3, 2.0e-3],
            "mean_rel_gamma": [3.0e-2, 4.0e-2],
            "mean_rel_Wg": [5.0e-5, 6.0e-5],
            "mean_rel_Wphi": [7.0e-5, 8.0e-5],
            "mean_rel_Wapar": [0.0, 0.0],
        }
    )
    lastvalue = pd.DataFrame(
        {
            "ky": [0.1, 0.2],
            "rel_gamma": [5.0e-3, 6.0e-3],
            "rel_omega": [7.0e-4, 8.0e-4],
            "gamma": [0.1, 0.2],
            "gamma_gx": [0.1, 0.2],
            "omega": [-0.2, -0.3],
            "omega_gx": [-0.2, -0.3],
        }
    )
    fig, ax = plt.subplots()
    try:
        _plot_imported_linear(
            ax, df, "Imported", lastvalue=lastvalue, note="late-time closure"
        )
        assert ax.get_yscale() == "log"
        assert len(ax.lines) == 6
        assert any(text.get_text() == "late-time closure" for text in ax.texts)
    finally:
        plt.close(fig)


def test_plot_secondary_adds_gamma_lines_and_abs_omega_bars() -> None:
    df = pd.DataFrame(
        {
            "ky": [0.0, 0.1],
            "kx": [-0.05, 0.05],
            "gamma_gx": [4.9, 4.9],
            "gamma_sp": [4.9, 4.9],
            "abs_omega": [1.0e-4, 2.0e-4],
        }
    )
    fig, ax = plt.subplots()
    try:
        _plot_secondary(ax, df, "Secondary")
        assert len(ax.lines) == 2
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)


def test_autocrop_image_trims_uniform_border() -> None:
    image = np.ones((10, 12, 3), dtype=float)
    image[3:7, 4:9, :] = 0.0
    cropped = _autocrop_image(image, white_threshold=0.95, pad_pixels=0)
    assert cropped.shape == (4, 5, 3)


def test_reference_panel_subcommands_render_outputs(tmp_path: Path) -> None:
    cyclone_linear = tmp_path / "cyclone_linear.csv"
    kbm_linear = tmp_path / "kbm_linear.csv"
    cyclone_nl = tmp_path / "cyclone_nl.png"
    kbm_nl = tmp_path / "kbm_nl.png"
    cyclone_kbm = tmp_path / "cyclone_kbm.png"
    w7x = tmp_path / "w7x.png"
    hsx = tmp_path / "hsx.png"
    w7x_csv = tmp_path / "w7x.csv"
    hsx_csv = tmp_path / "hsx.csv"
    w7x_lastvalue_csv = tmp_path / "w7x_last.csv"
    hsx_lastvalue_csv = tmp_path / "hsx_last.csv"
    secondary = tmp_path / "secondary.csv"

    _write_linear_mismatch_csv(cyclone_linear)
    _write_linear_mismatch_csv(kbm_linear)
    _write_plot(cyclone_nl, "Cyclone")
    _write_plot(kbm_nl, "KBM")
    _write_plot(cyclone_kbm, "Cyclone KBM")
    _write_plot(w7x, "W7-X")
    _write_plot(hsx, "HSX")
    _write_imported_linear_csv(w7x_csv)
    _write_imported_linear_csv(hsx_csv)
    _write_imported_lastvalue_csv(w7x_lastvalue_csv)
    _write_imported_lastvalue_csv(hsx_lastvalue_csv)
    _write_secondary_csv(secondary)

    tokamak_out = tmp_path / "tokamak_panel.png"
    publication_png = tmp_path / "publication_panel.png"
    publication_pdf = tmp_path / "publication_panel.pdf"
    summary_out = tmp_path / "summary_panel.png"
    base_env = {**os.environ, "PYTHONPATH": str(TOOLS)}

    subprocess.run(
        [
            sys.executable,
            str(TOOLS / "make_reference_panels.py"),
            "tokamak",
            "--cyclone-linear",
            str(cyclone_linear),
            "--kbm-linear",
            str(kbm_linear),
            "--cyclone-nonlinear-panel",
            str(cyclone_nl),
            "--kbm-nonlinear-panel",
            str(kbm_nl),
            "--out",
            str(tokamak_out),
        ],
        check=True,
        cwd=ROOT,
        env=base_env,
    )
    subprocess.run(
        [
            sys.executable,
            str(TOOLS / "make_reference_panels.py"),
            "publication",
            "--cyclone-kbm-panel",
            str(cyclone_kbm),
            "--w7x-panel",
            str(w7x),
            "--hsx-panel",
            str(hsx),
            "--w7x-linear-csv",
            str(w7x_csv),
            "--hsx-linear-csv",
            str(hsx_csv),
            "--w7x-linear-lastvalue-csv",
            str(w7x_lastvalue_csv),
            "--hsx-linear-lastvalue-csv",
            str(hsx_lastvalue_csv),
            "--out",
            str(publication_png),
            "--pdf-out",
            str(publication_pdf),
        ],
        check=True,
        cwd=ROOT,
        env=base_env,
    )
    subprocess.run(
        [
            sys.executable,
            str(TOOLS / "make_reference_panels.py"),
            "summary",
            "--cyclone-kbm-panel",
            str(cyclone_kbm),
            "--w7x-panel",
            str(w7x),
            "--hsx-panel",
            str(hsx),
            "--w7x-linear-csv",
            str(w7x_csv),
            "--hsx-linear-csv",
            str(hsx_csv),
            "--w7x-linear-lastvalue-csv",
            str(w7x_lastvalue_csv),
            "--hsx-linear-lastvalue-csv",
            str(hsx_lastvalue_csv),
            "--secondary-csv",
            str(secondary),
            "--out",
            str(summary_out),
        ],
        check=True,
        cwd=ROOT,
        env=base_env,
    )

    for path in [tokamak_out, publication_png, publication_pdf, summary_out]:
        assert path.exists()
        assert path.stat().st_size > 0


# ---- test_reference_comparison_contracts.py ----


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
