"""Orchestration above the integrators: config routing, sharding, gradient gates.

Nothing here integrates anything itself; each test checks that the layer above
hands the right arguments to the layer below. runners maps a TimeConfig onto the
linear and nonlinear integrators and forwards the runtime parallel and sharding
policy; gkx.parallel.integrators wraps those same integrators in a sharded driver
(validated against a mocked pjit, so it is device-count independent); and the
VMEC/Boozer gradient-gate modules drive a solver objective from the objective
side, the outermost caller of all.

Absorbed from test_sharded_integrators.py and test_solver_gradient_gate_modules.py;
the origin markers below delimit each block.
"""

from __future__ import annotations

from gkx.config import CycloneBaseCase, GridConfig, TimeConfig
from gkx.core_grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry
from gkx.operators.linear.params import LinearParams
from gkx.parallel import integrators as integrator_module
from gkx.parallel.integrators import (
    integrate_linear_sharded,
    integrate_nonlinear_sharded,
)
from gkx.solvers_time_runners import (
    integrate_linear_from_config,
    integrate_nonlinear_from_config,
)
from gkx.terms.config import FieldState
from types import SimpleNamespace
from typing import Any
import dataclasses
import gkx.solvers_time_runners as runners
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
import pytest


def test_integrate_linear_from_config():
    """TimeConfig should map into the linear integrator."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2")
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
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2")
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)

    assert integrate_linear_from_config(
        G, grid, geom, params, cfg.time, parallel=parallel
    ) == ("G", "phi")
    assert captured["parallel"] is parallel


def test_integrate_nonlinear_from_config_routes_fixed_step_state_sharding(monkeypatch):
    """Nonlinear runs should honor TimeConfig.state_sharding."""

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
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2", state_sharding="ky")
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
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2", state_sharding="z")
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
        time=TimeConfig(t_max=0.4, dt=0.05, method="rk2"),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(nu=0.05)

    def evolve(name):
        time_cfg = dataclasses.replace(cfg.time, collision_operator=name)
        # (Nl, Nm) must be the (J+1, P+1) basis of the drift-kinetic matrix.
        G = jnp.zeros(
            (2, 4, int(grid.ky.size), cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex128
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


def test_integrate_linear_from_config_requires_the_table_moment_basis():
    """Only the table's (Nl, Nm) = (J+1, P+1) = (2, 4) may run.

    The transposed (4, 2) has the right moment count, so without the check it
    ran silently on the wrong moments (the state is packed m*Nl + l).
    """

    cfg = CycloneBaseCase(
        grid=GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0),
        time=TimeConfig(t_max=0.1, dt=0.05, method="rk2", collision_operator="sugama"),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(nu=0.05)

    def run(nl, nm):
        G = jnp.zeros((nl, nm, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), jnp.complex128)
        G = G.at[0, 0, 1, 0, :].set(1.0e-3)
        return integrate_linear_from_config(G, grid, geom, params, cfg.time)[0]

    for nl, nm in ((3, 3), (4, 2)):
        with pytest.raises(ValueError, match=r"8-moment basis \(Nl, Nm\) = \(2, 4\)"):
            run(nl, nm)
    with pytest.raises(ValueError, match="Set Nl=2, Nm=4"):
        run(4, 2)
    assert jnp.all(jnp.isfinite(run(2, 4)))


def test_check_moment_basis_names_the_finite_wavelength_table_layout():
    """Each shipped finite-Larmor table accepts only its own (J+1, P+1)."""

    from gkx.operators.linear.collision_factory import collision_operator_from_config

    species = {name: jnp.ones(1) for name in ("density", "mass", "temperature")}
    check = runners._check_moment_basis_matches_operator
    for (nl, nm), bad in (((2, 4), (4, 2)), ((3, 6), (6, 3))):
        op = collision_operator_from_config(
            "coulomb_finite_kperp", moments=nl * nm, **species
        )
        check(op, "coulomb_finite_kperp", jnp.zeros((nl, nm, 1, 1, 1)))
        with pytest.raises(ValueError, match=f"Set Nl={nl}, Nm={nm}"):
            check(op, "coulomb_finite_kperp", jnp.zeros(bad + (1, 1, 1)))


def test_krylov_and_explicit_runtime_refuse_a_moment_collision_operator():
    """Paths that cannot carry the operator must refuse, not run as LB."""

    from gkx.config import RuntimeConfig, RuntimeSpeciesConfig
    from gkx.runtime import run_runtime_linear

    base = RuntimeConfig()
    cfg = dataclasses.replace(
        base,
        grid=dataclasses.replace(base.grid, Nx=1, Ny=4, Nz=8, boundary="periodic"),
        time=dataclasses.replace(base.time, collision_operator="sugama"),
        species=(RuntimeSpeciesConfig(name="ion"),),
    )
    for solver, path in (
        ("krylov", "Krylov eigenvalue"),
        ("explicit_time", "explicit"),
    ):
        with pytest.raises(NotImplementedError, match=path):
            run_runtime_linear(cfg, ky_target=0.2, Nl=2, Nm=4, solver=solver)


def test_config_collision_operator_rejects_unsupported_solver_paths(monkeypatch):
    """Unsupported paths must raise instead of silently ignoring the setting."""

    monkeypatch.setattr(
        runners, "resolve_state_sharding", lambda G0, spec: "mesh" if spec else None
    )

    grid_cfg = GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(
        grid=grid_cfg,
        time=TimeConfig(
            t_max=0.2,
            dt=0.1,
            method="rk2",
            state_sharding="ky",
            collision_operator="sugama",
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(nu=0.05)
    G = jnp.zeros((4, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex128)

    with pytest.raises(NotImplementedError, match="sharded nonlinear"):
        integrate_nonlinear_from_config(G, grid, geom, params, cfg.time)


# ---- from test_sharded_integrators.py ----


def test_integrate_linear_sharded_rejects_nonpositive_steps() -> None:
    with pytest.raises(ValueError):
        integrate_linear_sharded(
            jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64), None, None, dt=0.1, steps=0
        )


def test_integrate_linear_sharded_runs_with_mocked_pjit(monkeypatch) -> None:
    calls = {"rhs": 0, "shard": 0, "put": 0}

    def fake_rhs(G, cache, params, terms=None, dt=None):
        calls["rhs"] += 1
        return jnp.ones_like(G), None

    monkeypatch.setattr("gkx.parallel.integrators.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr("gkx.parallel.integrators.pjit", lambda fn, **kwargs: fn)
    monkeypatch.setattr(
        "gkx.parallel.integrators.jax.lax.with_sharding_constraint",
        lambda state, sharding: calls.__setitem__("shard", calls["shard"] + 1) or state,
    )
    monkeypatch.setattr(
        "gkx.parallel.integrators.jax.device_put",
        lambda state, sharding: calls.__setitem__("put", calls["put"] + 1) or state,
    )

    G0 = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)
    out = integrate_linear_sharded(
        G0, SimpleNamespace(), SimpleNamespace(), dt=0.5, steps=2, state_sharding="mesh"
    )
    assert out.shape == G0.shape
    assert calls["rhs"] >= 2
    assert calls["put"] == 1
    assert calls["shard"] >= 3


def test_integrate_linear_sharded_no_sharding_path(monkeypatch) -> None:
    calls = {"rhs": 0}

    def fake_rhs(G, cache, params, terms=None, dt=None):
        calls["rhs"] += 1
        return jnp.ones_like(G), None

    monkeypatch.setattr("gkx.parallel.integrators.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr("gkx.parallel.integrators.pjit", lambda fn, **kwargs: fn)

    G0 = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)
    out = integrate_linear_sharded(
        G0, SimpleNamespace(), SimpleNamespace(), dt=0.25, steps=2
    )

    assert out.shape == G0.shape
    assert calls["rhs"] == 2
    assert jnp.allclose(out, 0.5)


def _cache_stub() -> SimpleNamespace:
    return SimpleNamespace(ky=jnp.asarray([0.0]), kx=jnp.asarray([0.0]))


def test_integrate_nonlinear_sharded_rejects_bad_options() -> None:
    G0 = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)
    with pytest.raises(ValueError, match="steps"):
        integrate_nonlinear_sharded(
            G0, _cache_stub(), SimpleNamespace(), dt=0.1, steps=0
        )
    with pytest.raises(ValueError, match="method"):
        integrate_nonlinear_sharded(
            G0, _cache_stub(), SimpleNamespace(), dt=0.1, steps=1, method="bad"
        )


def _one_device_ky_sharding() -> NamedSharding:
    """A real ky sharding on a one-device mesh, available on every host."""

    mesh = Mesh(np.array(jax.devices()[:1]), ("d",))
    return NamedSharding(mesh, PartitionSpec(None, None, "d", None, None))


def test_integrate_nonlinear_sharded_runs_with_a_real_sharding(monkeypatch) -> None:
    calls = {"rhs": 0}

    def fake_rhs(
        G,
        cache,
        params,
        terms=None,
        *,
        compressed_real_fft=True,
        laguerre_mode="grid",
        external_phi=None,
    ):
        calls["rhs"] += 1
        return jnp.ones_like(G), FieldState(phi=jnp.ones((1, 1, 1), dtype=G.dtype))

    monkeypatch.setattr("gkx.parallel.integrators.nonlinear_rhs_cached", fake_rhs)
    integrator_module._compiled_nonlinear_sharded_runner.cache_clear()
    sharding = _one_device_ky_sharding()

    G0 = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)
    # The fake RHS ignores the cache and parameters; empty pytrees let them
    # through the real jit and shard_map.
    G_final, fields_t = integrate_nonlinear_sharded(
        G0,
        (),
        (),
        dt=0.5,
        steps=2,
        method="rk2",
        state_sharding=sharding,
        compressed_real_fft=False,
    )

    assert G_final.shape == G0.shape
    assert G_final.sharding.spec == sharding.spec
    assert fields_t.phi.shape[0] == 2
    assert np.allclose(np.asarray(G_final), 1.0)
    assert calls["rhs"] >= 2


def _stub_sharding(spec: tuple, **mesh_shape: int) -> SimpleNamespace:
    return SimpleNamespace(spec=spec, mesh=SimpleNamespace(shape=mesh_shape))


def test_state_split_pads_an_extent_the_devices_do_not_divide() -> None:
    """``Nyc = 5`` on the ``ky >= 0`` layout is padded to 6 for two devices."""

    split = integrator_module._resolve_state_split(
        (1, 3, 4, 5, 4, 16), _stub_sharding((None, None, None, "d"), d=2)
    )
    assert (split.axis, split.mesh_axis, split.count) == (3, "d", 2)
    assert (split.extent, split.padded) == (5, 6)
    divisible = integrator_module._resolve_state_split(
        (1, 3, 4, 8, 4, 16), _stub_sharding((None, None, None, "d"), d=4)
    )
    assert divisible.padded == divisible.extent == 8


@pytest.mark.parametrize(
    "spec", [(None, None, None, "d", "e"), (None, None, None, ("d", "e")), ()]
)
def test_state_split_rejects_anything_but_one_axis_on_one_mesh_axis(spec) -> None:
    with pytest.raises(ValueError, match="exactly one state axis") as excinfo:
        integrator_module._resolve_state_split(
            (1, 3, 4, 8, 4, 16), _stub_sharding(spec, d=2, e=2)
        )
    assert "(1, 3, 4, 8, 4, 16)" in str(excinfo.value)


@pytest.mark.parametrize(
    "method", ["euler", "rk2", "rk3", "rk3_heun", "rk3_classic", "rk4", "sspx3"]
)
def test_integrate_nonlinear_sharded_explicit_methods_constant_rhs(
    monkeypatch, method: str
) -> None:
    """All explicit nonlinear sharded methods should preserve a constant RHS update."""

    calls = {"rhs": 0}

    def fake_rhs(
        G,
        cache,
        params,
        terms=None,
        *,
        compressed_real_fft=True,
        laguerre_mode="grid",
        external_phi=None,
    ):
        calls["rhs"] += 1
        return jnp.ones_like(G), FieldState(phi=jnp.ones((1, 1, 1), dtype=G.dtype))

    monkeypatch.setattr("gkx.parallel.integrators.nonlinear_rhs_cached", fake_rhs)
    monkeypatch.setattr("gkx.parallel.integrators.pjit", lambda fn, **kwargs: fn)

    G0 = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)
    G_final, fields_t = integrate_nonlinear_sharded(
        G0,
        _cache_stub(),
        SimpleNamespace(),
        dt=0.1,
        steps=1,
        method=method,
        compressed_real_fft=False,
    )

    assert G_final.shape == G0.shape
    assert fields_t.phi.shape[0] == 1
    assert jnp.allclose(G_final, 0.1)
    assert calls["rhs"] >= 2


def test_integrate_nonlinear_sharded_final_only_path(monkeypatch) -> None:
    calls = {"rhs": 0}

    def fake_rhs(
        G,
        cache,
        params,
        terms=None,
        *,
        compressed_real_fft=True,
        laguerre_mode="grid",
        external_phi=None,
    ):
        calls["rhs"] += 1
        return 2.0 * jnp.ones_like(G), FieldState(
            phi=jnp.ones((1, 1, 1), dtype=G.dtype)
        )

    monkeypatch.setattr("gkx.parallel.integrators.nonlinear_rhs_cached", fake_rhs)
    monkeypatch.setattr("gkx.parallel.integrators.pjit", lambda fn, **kwargs: fn)

    G0 = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)
    out = integrate_nonlinear_sharded(
        G0,
        _cache_stub(),
        SimpleNamespace(),
        dt=0.25,
        steps=2,
        method="euler",
        return_fields=False,
    )

    assert out.shape == G0.shape
    assert jnp.allclose(out, 1.0)
    # lax.scan traces one RHS site for the final-only graph. The field-history
    # path has a second RHS site to recover fields at the accepted state.
    assert calls["rhs"] == 1


def test_integrate_nonlinear_sharded_reuses_compiled_runner(monkeypatch) -> None:
    calls = {"compile": 0}

    def fake_pjit(fn, **_kwargs):
        calls["compile"] += 1
        return fn

    def fake_rhs(
        G,
        cache,
        params,
        terms=None,
        *,
        compressed_real_fft=True,
        laguerre_mode="grid",
        external_phi=None,
    ):
        del cache, params, terms, compressed_real_fft, laguerre_mode, external_phi
        return jnp.ones_like(G), FieldState(phi=jnp.ones((1, 1, 1), dtype=G.dtype))

    monkeypatch.setattr(integrator_module, "pjit", fake_pjit)
    monkeypatch.setattr(integrator_module, "nonlinear_rhs_cached", fake_rhs)
    integrator_module._compiled_nonlinear_sharded_runner.cache_clear()
    G0 = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)

    for _ in range(2):
        out = integrate_nonlinear_sharded(
            G0,
            _cache_stub(),
            SimpleNamespace(),
            dt=0.1,
            steps=2,
            method="euler",
            return_fields=False,
        )
        assert jnp.allclose(out, 0.2)

    assert calls["compile"] == 1
    integrator_module._compiled_nonlinear_sharded_runner.cache_clear()


# ---- from test_solver_gradient_gate_modules.py ----


def _fake_state_bundle(_case_name: str) -> dict[str, object]:
    return {
        "case_name": "fake",
        "input_path": "input.fake",
        "wout_path": "wout.fake.nc",
        "state": object(),
        "runtime": object(),
        "inp": object(),
        "wout": object(),
    }


def _fake_state_array(_state: object, _parameter_family: str) -> np.ndarray:
    return np.zeros((5, 3), dtype=float)


def _fake_replace_state_coefficient(
    _state: object,
    parameter_family: str,
    _base_coeff: np.ndarray,
    radial_index: int,
    mode_index: int,
    delta: float,
) -> dict[str, object]:
    return {
        "family": parameter_family,
        "radial_index": radial_index,
        "mode_index": mode_index,
        "delta": float(delta),
    }


def _fake_parameter_name(
    parameter_family: str,
    radial_index: int,
    mode_index: int,
    *,
    default_mid_surface: int,
) -> str:
    suffix = "mid" if radial_index == default_mid_surface else f"r{radial_index}"
    return f"{parameter_family}_{suffix}_m{mode_index}"


def _fake_objective_vector(
    traced_state: dict[str, object], *_args: Any, **_kwargs: Any
) -> np.ndarray:
    x = float(traced_state["delta"])
    return np.asarray([1.0 + 3.0 * x + x * x, 0.1 + x, 2.0, 3.0, 4.0, 5.0 + x])


def _fake_objective_table(
    traced_state: dict[str, object],
    *_args: Any,
    **_kwargs: Any,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    x = float(traced_state["delta"])
    table = np.asarray(
        [
            [1.0 + 2.0 * x, 0.0, 1.0, 1.0, 1.0, 2.0],
            [2.0 + 4.0 * x, 0.0, 1.0, 1.0, 1.0, 4.0],
        ],
        dtype=float,
    )
    metadata = [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1},
        {"surface_index": 2, "alpha": 0.0, "selected_ky_index": 1},
    ]
    return table, metadata


def _parabola_fd_report(**kwargs: Any) -> dict[str, object]:
    delta = float(kwargs.get("base_delta", 0.0))
    value = 1.0 + (delta - 0.03) ** 2
    derivative = 2.0 * (delta - 0.03)
    return {
        "passed": True,
        "base_value": value,
        "central_derivative": derivative,
        "curvature_ratio": 0.0,
        "n_samples": 1,
        "samples": [{"surface_index": None, "alpha": 0.0, "selected_ky_index": 1}],
    }


def _one_mode_context(**kwargs: Any) -> dict[str, object]:
    cfg = SimpleNamespace(grid=SimpleNamespace(Nx=1, Ny=4, Nz=int(kwargs["ntheta"])))

    def matrix_fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([[1.0 + 2.0 * x[0] + 1j * (0.5 - x[0])]])

    return {
        "case_name": kwargs["case_name"],
        "cfg": cfg,
        "parameter_names": ("Rcos_mid_m1",),
        "parameter_indices": {"Rcos": [2, 1]},
        "surface_index": kwargs["surface_index"],
        "mboz": int(kwargs["mboz"]),
        "nboz": int(kwargs["nboz"]),
        "surface_stencil_width": kwargs["surface_stencil_width"],
        "n_laguerre": int(kwargs["n_laguerre"]),
        "n_hermite": int(kwargs["n_hermite"]),
        "state_shape": (1,),
        "matrix_fn": matrix_fn,
    }


def _fake_ql_features(
    eigenvalue: jnp.ndarray,
    _eigenvector: jnp.ndarray,
    x: jnp.ndarray,
    _context: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    gamma = jnp.real(eigenvalue)
    omega = jnp.imag(eigenvalue)
    kperp = 2.0 + x[0]
    heat = 3.0 + 2.0 * x[0]
    proxy = gamma * heat / kperp
    return gamma, omega, kperp, heat, proxy


def _fake_window_metrics(
    gamma: jnp.ndarray,
    kperp: jnp.ndarray,
    heat: jnp.ndarray,
    *,
    dt: float,
    steps: int,
    tail_fraction: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    scale = jnp.asarray(float(dt) * int(steps) * float(tail_fraction))
    return gamma * heat * scale, heat / kperp, gamma - kperp
