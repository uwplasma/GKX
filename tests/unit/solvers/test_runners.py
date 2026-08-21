"""Config-driven runner tests."""

import dataclasses

import jax.numpy as jnp
import pytest

import gkx.solvers.time.runners as runners
from gkx.config import CycloneBaseCase, GridConfig, TimeConfig
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid
from gkx.operators.linear.params import LinearParams
from gkx.solvers.time.runners import (
    integrate_linear_from_config,
    integrate_nonlinear_from_config,
)
from gkx.terms.config import FieldState


def test_integrate_linear_from_config():
    """TimeConfig should map into the linear integrator."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2", use_diffrax=False)
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    _, phi_t = integrate_linear_from_config(G, grid, geom, params, cfg.time)
    assert phi_t.shape[0] == 2


def test_integrate_linear_from_config_forwards_parallel(monkeypatch):
    """Runtime parallel policy should reach the fixed-step linear integrator."""

    captured = {}
    parallel = object()

    def fake_integrate_linear(*args, **kwargs):
        captured["parallel"] = kwargs["parallel"]
        return "G", "phi"

    monkeypatch.setattr(runners, "integrate_linear", fake_integrate_linear)

    grid_cfg = GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2", use_diffrax=False)
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)

    assert integrate_linear_from_config(
        G, grid, geom, params, cfg.time, parallel=parallel
    ) == ("G", "phi")
    assert captured["parallel"] is parallel


def test_integrate_linear_from_config_rejects_parallel_diffrax() -> None:
    grid_cfg = GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2", use_diffrax=True)
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)

    with pytest.raises(NotImplementedError, match="fixed-step cached"):
        integrate_linear_from_config(
            G,
            grid,
            geom,
            params,
            cfg.time,
            parallel=type("P", (), {"strategy": "velocity"})(),
        )


def test_integrate_nonlinear_from_config_routes_fixed_step_state_sharding(monkeypatch):
    """Non-diffrax nonlinear runs should honor TimeConfig.state_sharding."""

    captured = {}

    def fake_cache(grid, geom, params, nl, nm):
        captured["cache_shape"] = (nl, nm)
        return "cache"

    def fake_sharded(G0, cache, params, **kwargs):
        captured["kwargs"] = kwargs
        captured["cache"] = cache
        return G0 + 1.0, FieldState(phi=jnp.ones((2, 1, 1, 1), dtype=G0.dtype))

    monkeypatch.setattr(
        runners, "resolve_state_sharding", lambda G0, spec: "mesh" if spec else None
    )
    monkeypatch.setattr(runners, "build_linear_cache", fake_cache)
    monkeypatch.setattr(runners, "integrate_nonlinear_sharded", fake_sharded)

    grid_cfg = GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(
        t_max=0.2, dt=0.1, method="rk2", use_diffrax=False, state_sharding="ky"
    )
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)

    G_out, fields = integrate_nonlinear_from_config(G, grid, geom, params, cfg.time)

    assert captured["cache_shape"] == (2, 2)
    assert captured["cache"] == "cache"
    assert captured["kwargs"]["state_sharding"] == "mesh"
    assert captured["kwargs"]["return_fields"] is True
    assert G_out.shape == G.shape
    assert fields.phi.shape[0] == 2


def test_integrate_nonlinear_from_config_rejects_ungated_z_state_sharding():
    """The config path should not expose exploratory z-FFT sharding as release-grade."""

    grid_cfg = GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(
        t_max=0.2, dt=0.1, method="rk2", use_diffrax=False, state_sharding="z"
    )
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)

    with pytest.raises(ValueError, match="z FFT axis"):
        integrate_nonlinear_from_config(G, grid, geom, params, cfg.time)


def test_integrate_linear_from_config_applies_selected_collision_operator():
    """A selected moment operator must change the linear evolution."""

    grid_cfg = GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(
        grid=grid_cfg,
        time=TimeConfig(t_max=0.4, dt=0.05, method="rk2", use_diffrax=False),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(nu=0.05)

    def evolve(name):
        time_cfg = dataclasses.replace(cfg.time, collision_operator=name)
        # Nl * Nm must match the tabulated eight-moment drift-kinetic matrix.
        G = jnp.zeros(
            (4, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex128
        )
        G = G.at[0, 0, 1, 0, :].set(1.0e-3)
        return integrate_linear_from_config(G, grid, geom, params, time_cfg)[0]

    baseline = evolve("lenard_bernstein")
    sugama = evolve("sugama")
    improved = evolve("improved_sugama")

    # "none" and "lenard_bernstein" both keep the built-in diagonal term.
    assert jnp.allclose(baseline, evolve("none"))
    # Each moment operator replaces that term, so all three must differ.
    assert not jnp.allclose(baseline, sugama)
    assert not jnp.allclose(baseline, improved)
    assert not jnp.allclose(sugama, improved)
    assert jnp.all(jnp.isfinite(sugama)) and jnp.all(jnp.isfinite(improved))


def test_integrate_linear_from_config_reports_moment_basis_mismatch():
    """A basis the tabulated matrix cannot act on must fail with guidance."""

    grid_cfg = GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(
        grid=grid_cfg,
        time=TimeConfig(
            t_max=0.2,
            dt=0.1,
            method="rk2",
            use_diffrax=False,
            collision_operator="sugama",
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(nu=0.05)
    G = jnp.zeros((3, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex128)

    with pytest.raises(ValueError, match="8-moment"):
        integrate_linear_from_config(G, grid, geom, params, cfg.time)


def test_config_collision_operator_rejects_unsupported_solver_paths():
    """Unsupported paths must raise instead of silently ignoring the setting."""

    grid_cfg = GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(
        grid=grid_cfg,
        time=TimeConfig(
            t_max=0.2, dt=0.1, method="rk2", use_diffrax=True,
            collision_operator="sugama",
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(nu=0.05)
    G = jnp.zeros((4, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex128)

    with pytest.raises(NotImplementedError, match="diffrax linear"):
        integrate_linear_from_config(G, grid, geom, params, cfg.time)
    with pytest.raises(NotImplementedError, match="diffrax nonlinear"):
        integrate_nonlinear_from_config(G, grid, geom, params, cfg.time)
