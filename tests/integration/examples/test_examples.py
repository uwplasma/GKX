"""Fast integration checks for shipped example workflows."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from support.paths import REPO_ROOT, load_repo_script
from examples.theory_and_demos.autodiff_inverse_growth import run_demo as run_inverse_growth_demo
from examples.theory_and_demos.autodiff_inverse_twomode import run_demo as run_twomode_demo
from examples.theory_and_demos.quasilinear_implicit_sensitivity import (
    run_demo as run_implicit_sensitivity_demo,
)
from gkx.config import CycloneBaseCase, GridConfig, TimeConfig
from gkx.core.grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry, sample_flux_tube_geometry
from gkx.operators.linear.params import LinearParams
from gkx.solvers.time.runners import (
    integrate_linear_from_config,
    integrate_nonlinear_from_config,
)
from gkx.terms.config import TermConfig


_PARALLEL_EXAMPLE = (
    REPO_ROOT / "examples" / "parallelization" / "independent_ky_runtime_batch_scan.py"
)
_PARALLEL_CONFIG = (
    REPO_ROOT / "examples" / "parallelization" / "runtime_batch_ky_scan.toml"
)


def _load_parallel_example_module():
    return load_repo_script(
        _PARALLEL_EXAMPLE.relative_to(REPO_ROOT),
        module_name="independent_ky_runtime_batch_scan",
        write_bytecode=False,
    )


def test_autodiff_inverse_growth_demo_summary(tmp_path: Path) -> None:
    summary = run_inverse_growth_demo(
        outdir=tmp_path,
        steps=24,
        dt=0.05,
        ky_index=1,
        kx_index=0,
        z_index=0,
        tprim_true=2.2,
        fprim_true=0.8,
        tprim_init=1.8,
        fprim_init=1.1,
        gd_steps=4,
        gd_lr=0.5,
        plot=False,
        write_files=False,
    )
    assert max(summary["jac_rel_error"]) < 0.05
    assert summary["loss_final"] >= 0.0
    assert max(summary["observable_abs_error"]) < 5.0e-2
    cov = summary["covariance"]
    assert cov[0][0] > 0.0
    assert cov[1][1] > 0.0
    assert summary["sensitivity_map_rank"] >= 1
    assert summary["jacobian_condition_number"] > 0.0
    assert len(summary["covariance_std"]) == 2
    assert summary["uq_ellipse_area_1sigma"] >= 0.0


def test_autodiff_twomode_demo_summary(tmp_path: Path) -> None:
    summary = run_twomode_demo(
        outdir=tmp_path,
        steps=24,
        dt=0.05,
        ky_indices=(1, 3),
        kx_index=0,
        z_index=0,
        tprim_true=2.2,
        fprim_true=0.8,
        tprim_init=1.8,
        fprim_init=1.1,
        gd_steps=6,
        gd_lr=0.2,
        plot=False,
        write_files=False,
    )
    assert max(summary["jac_rel_error"]) < 0.05
    assert max(summary["parameter_abs_error"]) < 1.0e-2
    assert max(summary["observable_abs_error"]) < 1.0e-4
    cov = summary["covariance"]
    assert cov[0][0] > 0.0
    assert cov[1][1] > 0.0
    assert summary["sensitivity_map_rank"] == 2
    assert summary["jacobian_condition_number"] < 1.0e4
    assert len(summary["covariance_std"]) == 2
    assert summary["uq_ellipse_area_1sigma"] >= 0.0


def test_quasilinear_implicit_sensitivity_demo_summary(tmp_path: Path) -> None:
    summary = run_implicit_sensitivity_demo(
        outdir=tmp_path, plot=False, write_files=False
    )

    assert summary["passed"] is True
    assert summary["branch_isolated"] is True
    assert summary["sensitivity_method"] == "implicit_left_right_eigenpair"
    assert len(summary["observable_labels"]) == 5
    assert len(summary["parameter_labels"]) == 2
    jac_impl = np.asarray(summary["jacobian_implicit"], dtype=float)
    jac_fd = np.asarray(summary["jacobian_fd"], dtype=float)
    np.testing.assert_allclose(jac_impl, jac_fd, rtol=5.0e-2, atol=2.0e-3)


def test_runtime_batch_ky_scan_example_uses_independent_workers(
    monkeypatch,
) -> None:
    import gkx.runtime as runtime

    example = _load_parallel_example_module()
    calls: list[float] = []

    def _unexpected_combined_batch(*_args, **_kwargs):
        raise AssertionError(
            "strategy='batch' example must not use the combined-ky solver path"
        )

    def _fake_run_runtime_linear(_cfg, **kwargs):
        ky = float(kwargs["ky_target"])
        calls.append(ky)
        return SimpleNamespace(gamma=1.0 + ky, omega=-(2.0 + ky), quasilinear=None)

    monkeypatch.setattr(runtime, "_run_runtime_scan_batch", _unexpected_combined_batch)
    monkeypatch.setattr(runtime, "run_runtime_linear", _fake_run_runtime_linear)

    scan = example.run_example(_PARALLEL_CONFIG)

    np.testing.assert_allclose(sorted(calls), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(scan.ky, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(scan.gamma, [1.1, 1.2, 1.3])
    np.testing.assert_allclose(scan.omega, [-2.1, -2.2, -2.3])
    assert scan.parallel is not None
    assert scan.parallel["source"] == "runtime_config"
    assert scan.parallel["requested_workers"] == 2
    assert scan.parallel["effective_workers"] == 2
    assert scan.parallel["executor"] == "thread"
    assert "independent ky workers" in scan.parallel["identity_contract"]


def test_example_smoke_diffrax() -> None:
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(
        t_max=0.2,
        dt=0.1,
        use_diffrax=True,
        diffrax_solver="Tsit5",
        diffrax_adaptive=True,
        diffrax_rtol=1.0e-3,
        diffrax_atol=1.0e-6,
        diffrax_max_steps=20000,
        progress_bar=False,
    )
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    _, phi_t = integrate_linear_from_config(G, grid, geom, params, cfg.time)
    assert phi_t.shape[0] == 2


def test_example_smoke_nonlinear_scan() -> None:
    grid_cfg = GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2", use_diffrax=False)
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    terms = TermConfig(nonlinear=1.0)

    for seed in (0, 1):
        G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
        G = G.at[0, 0, seed, 0, :].set(1.0e-3 + 0.0j)
        _, fields_t = integrate_nonlinear_from_config(
            G,
            grid,
            geom,
            params,
            cfg.time,
            terms=terms,
        )
        assert fields_t.phi.shape[0] == 2


def test_example_smoke_nonlinear_scan_with_sampled_geometry() -> None:
    grid_cfg = GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2", use_diffrax=False)
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = sample_flux_tube_geometry(SAlphaGeometry.from_config(cfg.geometry), grid.z)
    params = LinearParams()
    terms = TermConfig(nonlinear=1.0)

    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    G = G.at[0, 0, 0, 0, :].set(1.0e-3 + 0.0j)
    _, fields_t = integrate_nonlinear_from_config(
        G,
        grid,
        geom,
        params,
        cfg.time,
        terms=terms,
    )

    assert fields_t.phi.shape[0] == 2
