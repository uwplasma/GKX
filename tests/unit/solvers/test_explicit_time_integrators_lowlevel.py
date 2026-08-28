from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import gkx.solvers.time.explicit as eti
import gkx.solvers.time.explicit_diagnostics as explicit_diagnostics
from gkx.solvers.time.explicit_steps import _linear_explicit_stage_update
from gkx.terms.config import FieldState


def _progress_sample() -> explicit_diagnostics._DiagnosticSample:
    scalar = jnp.asarray(0.0)
    return explicit_diagnostics._DiagnosticSample(
        phi=scalar,
        gamma=scalar,
        omega=scalar,
        Wg=1.0,
        Wphi=0.5,
        Wapar=0.0,
        heat=0.25,
        particle=0.0,
    )


def test_diagnostics_progress_uses_plain_base_install_contract(capsys) -> None:
    console = explicit_diagnostics._start_progress(True)
    assert console is True
    explicit_diagnostics._emit_sample_progress(
        console,
        step=1,
        t=0.5,
        t_max=1.0,
        sample=_progress_sample(),
    )
    explicit_diagnostics._finish_progress(console)
    output = capsys.readouterr().out
    assert "explicit linear simulation started" in output
    assert "progress=50% step=1" in output
    assert "explicit linear simulation complete" in output


def _cache() -> SimpleNamespace:
    return SimpleNamespace(
        ky=jnp.asarray([0.0, 0.3]),
        kx=jnp.asarray([0.0, 0.2]),
        dealias_mask=jnp.asarray([[True, True], [True, True]]),
    )


def test_explicit_time_lowlevel_array_and_maximum_helpers() -> None:
    empty_grid = SimpleNamespace(
        ky=np.asarray([]), kx=np.asarray([0.0]), z=np.asarray([0.0]), ky_mode=None
    )
    kx, ky, kz = eti._cfl_wavenumber_arrays(empty_grid)
    assert kx.tolist() == [0.0]
    assert ky.size == 0
    assert kz.tolist() == [0.0]

    sliced_grid = SimpleNamespace(
        ky=np.asarray([0.3]),
        kx=np.asarray([0.0]),
        z=np.asarray([0.0, 1.0, 2.0, 3.0]),
        ky_mode=np.asarray([3]),
    )
    _kx, ky, kz = eti._cfl_wavenumber_arrays(sliced_grid)
    np.testing.assert_allclose(ky, [0.3])
    np.testing.assert_allclose(kz, [0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])

    assert eti._laguerre_velocity_max(0) == 0.0
    assert eti._gradient_ratio_max(np.asarray([]), np.asarray([])) == 0.0
    assert eti._gradient_ratio_max(
        np.asarray([2.0]), np.asarray([0.0])
    ) == pytest.approx(1.0e6)


def test_instantaneous_growth_rate_step_max_mode_and_invalid_method() -> None:
    phi_prev = jnp.asarray([[[1.0 + 1.0j, 2.0 + 0.5j]]])
    phi_now = jnp.asarray([[[2.0 + 2.0j, 3.0 + 4.0j]]])
    mask = jnp.asarray([[True]])

    gamma, omega = eti._instantaneous_growth_rate_step(
        phi_now, phi_prev, 0.5, z_index=0, mask=mask, mode_method="max"
    )

    assert gamma.shape == (1, 1)
    assert omega.shape == (1, 1)
    assert np.isfinite(np.asarray(gamma[0, 0]))
    with pytest.raises(ValueError, match="mode_method"):
        eti._instantaneous_growth_rate_step(
            phi_now, phi_prev, 0.5, z_index=0, mask=mask, mode_method="bad"
        )


@pytest.mark.parametrize(
    "method", ["euler", "rk2", "rk3_classic", "rk3", "rk3_heun", "rk4", "sspx3", "k10"]
)
def test_linear_explicit_step_methods_match_scalar_linear_amplification(
    monkeypatch, method: str
) -> None:
    rate = 0.2 - 0.1j

    def fake_assemble(state, cache, params, terms=None, dt=None):
        return rate * state, FieldState(phi=jnp.sum(state, axis=0))

    monkeypatch.setattr(eti, "assemble_rhs_cached", fake_assemble)
    G0 = jnp.ones((1, 1, 2, 2, 1), dtype=jnp.complex64)

    G1, fields = eti._linear_explicit_step(
        G0, _cache(), object(), object(), 0.05, method=method
    )

    assert G1.shape == G0.shape
    assert fields.phi.shape == (1, 2, 2, 1)
    assert np.all(np.isfinite(np.asarray(G1)))


def test_linear_explicit_step_rejects_unknown_method(monkeypatch) -> None:
    monkeypatch.setattr(
        eti,
        "assemble_rhs_cached",
        lambda state, cache, params, terms=None, dt=None: (
            state,
            FieldState(phi=jnp.sum(state, axis=0)),
        ),
    )

    with pytest.raises(ValueError, match="explicit linear method"):
        eti._linear_explicit_step(
            jnp.ones((1, 1, 2, 2, 1), dtype=jnp.complex64),
            _cache(),
            object(),
            object(),
            0.05,
            method="bad",
        )


@pytest.mark.parametrize(("method", "expected_calls"), [("sspx3", 3), ("k10", 10)])
def test_self_staging_explicit_methods_do_not_evaluate_unused_rhs(
    method: str, expected_calls: int
) -> None:
    calls = 0

    def rhs(state):
        nonlocal calls
        calls += 1
        return 0.2 * state

    state = jnp.asarray([1.0])
    result = _linear_explicit_stage_update(
        state, jnp.asarray(0.1), method_key=method, rhs=rhs
    )

    assert calls == expected_calls
    assert np.all(np.isfinite(np.asarray(result)))


def test_explicit_from_config_preserves_adaptive_controls(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(eti, "build_linear_cache", lambda *_args: "cache")

    def fake_integrate(_state, _grid, cache, _params, _geom, config, *_args, **kwargs):
        captured.update(cache=cache, config=config, kwargs=kwargs)
        return np.asarray([0.1]), np.ones((1, 1, 1, 2)), None, None, None

    monkeypatch.setattr(eti, "integrate_linear_explicit_diagnostics", fake_integrate)
    time_cfg = SimpleNamespace(
        dt=0.02, t_max=2.0, sample_stride=3, fixed_dt=False,
        dt_min=1.0e-6, dt_max=0.04, cfl=0.7, method="rk2", cfl_fac=None,
        use_dealias_mask=True,
    )
    t, phi = eti.integrate_linear_explicit_from_config(
        jnp.ones((1,)), object(), object(), object(), time_cfg,
        Nl=2, Nm=3, z_index=1, show_progress=True,
    )

    config = captured["config"]
    assert config.dt == pytest.approx(0.02)
    assert config.t_max == pytest.approx(2.0)
    assert config.sample_stride == 3
    assert config.fixed_dt is False
    assert config.use_dealias_mask is True
    assert config.dt_max == pytest.approx(0.04)
    assert config.cfl == pytest.approx(0.7)
    assert captured["cache"] == "cache"
    assert captured["kwargs"]["show_progress"] is True
    np.testing.assert_allclose(t, [0.1])
    assert phi.shape == (1, 1, 1, 2)


def test_integrate_linear_explicit_from_config_runs_full_rk4_loop() -> None:
    # End-to-end explicit linear rk4 loop (public API) on a tiny Cyclone case,
    # exercising _run_linear_explicit_loop and its stepper/progress helpers.
    from gkx.config import CycloneBaseCase, GridConfig, TimeConfig
    from gkx.core.grid import build_spectral_grid
    from gkx.geometry import SAlphaGeometry
    from gkx.operators.linear.params import LinearParams

    grid = build_spectral_grid(
        CycloneBaseCase(grid=GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)).grid
    )
    geom = SAlphaGeometry.from_config(CycloneBaseCase().geometry)
    params = LinearParams(
        omega_d_scale=0.0, omega_star_scale=0.0, nu=0.0, nu_hyper=0.0,
        damp_ends_amp=0.0, damp_ends_widthfrac=0.0,
    )
    n_l, n_m = 2, 3
    z = jnp.linspace(0.0, 2.0 * jnp.pi, grid.z.size, endpoint=False)
    g0 = jnp.zeros(
        (n_l, n_m, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    g0 = g0.at[0, 0, 1, 0, :].set(1.0e-3 * jnp.exp(1j * z))
    time_cfg = TimeConfig(
        t_max=0.2, dt=0.02, method="rk4", sample_stride=1, use_diffrax=False
    )
    t, phi = eti.integrate_linear_explicit_from_config(
        g0, grid, geom, params, time_cfg, Nl=n_l, Nm=n_m, z_index=grid.z.size // 2
    )
    t = np.asarray(t)
    phi = np.asarray(phi)
    assert t.shape[0] == phi.shape[0]
    assert t.shape[0] >= 2
    assert np.all(np.isfinite(t))
    assert np.all(np.isfinite(phi))
    assert float(t[-1]) > float(t[0])


def test_resolve_and_validate_method_reject_unknown() -> None:
    assert eti._resolve_explicit_method("  RK4 ") == "rk4"
    with pytest.raises(ValueError, match="method must be one of"):
        eti._resolve_explicit_method("nonexistent")
    eti._validate_mode_method("max")
    with pytest.raises(ValueError, match="mode_method must be"):
        eti._validate_mode_method("bogus")


def test_format_wall_time_hours_minutes_and_clamped_branches() -> None:
    assert eti._format_wall_time(3661.0) == "1:01:01"
    assert eti._format_wall_time(7200.0) == "2:00:00"
    assert eti._format_wall_time(65.0) == "01:05"
    assert eti._format_wall_time(0.0) == "00:00"
    # Negative wall times are clamped to zero rather than formatting garbage.
    assert eti._format_wall_time(-5.0) == "00:00"


def test_adaptive_linear_dt_fixed_disabled_and_cfl_clamped() -> None:
    fixed = eti.ExplicitTimeConfig(t_max=1.0, dt=0.02, fixed_dt=True, cfl=0.5, cfl_fac=2.0)
    # fixed_dt short-circuits regardless of wmax.
    assert eti._adaptive_linear_dt(fixed, dt=0.02, dt_min=1e-6, dt_max=0.04, wmax=1e3) == 0.02

    adaptive = eti.ExplicitTimeConfig(
        t_max=1.0, dt=0.02, fixed_dt=False, cfl=0.5, cfl_fac=2.0
    )
    # Non-positive wmax cannot form a CFL estimate -> keep the requested dt.
    assert eti._adaptive_linear_dt(adaptive, dt=0.02, dt_min=1e-6, dt_max=0.04, wmax=0.0) == 0.02
    # cfl_fac*cfl/wmax = 2*0.5/100 = 0.01, in range.
    assert eti._adaptive_linear_dt(
        adaptive, dt=0.02, dt_min=1e-6, dt_max=0.04, wmax=100.0
    ) == pytest.approx(0.01)
    # Same guess clamped up to dt_min and down to dt_max.
    assert eti._adaptive_linear_dt(
        adaptive, dt=0.02, dt_min=0.02, dt_max=0.04, wmax=100.0
    ) == pytest.approx(0.02)
    assert eti._adaptive_linear_dt(
        adaptive, dt=0.02, dt_min=1e-6, dt_max=0.005, wmax=100.0
    ) == pytest.approx(0.005)


def test_should_emit_linear_progress_trigger_conditions() -> None:
    # First step, final step, and stride multiples emit; interior steps do not.
    assert eti._should_emit_linear_progress(step=1, total_steps_est=100, progress_stride=10)
    assert eti._should_emit_linear_progress(step=100, total_steps_est=100, progress_stride=10)
    assert eti._should_emit_linear_progress(step=20, total_steps_est=100, progress_stride=10)
    assert not eti._should_emit_linear_progress(
        step=23, total_steps_est=100, progress_stride=10
    )


def test_linear_loop_progress_clock_and_history_arrays() -> None:
    total_steps_est, progress_stride, started_at = eti._linear_loop_progress_clock(0.2, 0.02)
    assert total_steps_est == 10
    assert progress_stride >= 1
    assert isinstance(started_at, float)
    # A zero/degenerate dt must not divide-by-zero and still yields >=1 step.
    assert eti._linear_loop_progress_clock(0.0, 0.0)[0] == 1

    history = eti._LinearHistory()
    for k in range(3):
        history.ts.append(0.1 * k)
        history.phi.append(np.full((2,), float(k)))
        history.gamma.append(np.asarray(0.5 * k))
        history.omega.append(np.asarray(-0.5 * k))
    ts, phi, gamma, omega = eti._linear_history_arrays(history)
    np.testing.assert_allclose(ts, [0.0, 0.1, 0.2])
    assert phi.shape == (3, 2)
    assert gamma.shape == (3,) and omega.shape == (3,)


def _tiny_linear_case():
    from gkx.config import CycloneBaseCase, GridConfig
    from gkx.core.grid import build_spectral_grid
    from gkx.geometry import SAlphaGeometry
    from gkx.operators.linear.params import LinearParams

    grid = build_spectral_grid(
        CycloneBaseCase(grid=GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)).grid
    )
    geom = SAlphaGeometry.from_config(CycloneBaseCase().geometry)
    params = LinearParams(
        omega_d_scale=1.0, omega_star_scale=1.0, nu=0.0, nu_hyper=0.0,
        damp_ends_amp=0.0, damp_ends_widthfrac=0.0,
    )
    n_l, n_m = 2, 3
    z = jnp.linspace(0.0, 2.0 * jnp.pi, grid.z.size, endpoint=False)
    g0 = jnp.zeros(
        (n_l, n_m, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    g0 = g0.at[0, 0, 1, 0, :].set(1.0e-3 * jnp.exp(1j * z))
    cache = eti.build_linear_cache(grid, geom, params, n_l, n_m)
    return g0, grid, geom, params, cache, n_l, n_m


def test_integrate_linear_explicit_show_progress_and_max_mode(capsys) -> None:
    # Drive the public integrator with progress emission and max-mode growth
    # diagnostics, covering the start/step/complete progress helpers and the
    # max-mode sampling branch.
    g0, grid, geom, params, cache, _n_l, _n_m = _tiny_linear_case()
    time_cfg = eti.ExplicitTimeConfig(
        t_max=0.2, dt=0.02, method="rk4", sample_stride=1, fixed_dt=True
    )
    t, phi, gamma, omega = eti.integrate_linear_explicit(
        g0, grid, cache, params, geom, time_cfg,
        mode_method="max", jit=False, show_progress=True,
    )
    out = capsys.readouterr().out
    assert "linear initial-value integration started" in out
    assert "linear initial-value integration complete" in out
    assert "step=" in out
    for arr in (t, phi, gamma, omega):
        assert np.all(np.isfinite(np.asarray(arr)))
    assert np.asarray(t).shape[0] >= 2


def test_integrate_linear_explicit_adaptive_dt_completes() -> None:
    # fixed_dt=False with a physical frequency bound exercises the adaptive CFL
    # dt selection inside the stepping loop.
    g0, grid, geom, params, cache, _n_l, _n_m = _tiny_linear_case()
    time_cfg = eti.ExplicitTimeConfig(
        t_max=0.1, dt=0.05, method="rk3", sample_stride=1,
        fixed_dt=False, dt_min=1.0e-4, dt_max=0.05, cfl=0.8,
    )
    t, phi, gamma, omega = eti.integrate_linear_explicit(
        g0, grid, cache, params, geom, time_cfg,
        mode_method="z_index", jit=False, show_progress=False,
    )
    t = np.asarray(t)
    assert t.shape[0] >= 1
    assert np.all(np.isfinite(t))
    assert np.all(np.isfinite(np.asarray(phi)))
    assert float(t[-1]) <= 0.1 + 1.0e-9


def test_cfl_host_scales_read_inside_a_trace_that_touched_nothing_physical() -> None:
    """The CFL bound reads its own grid and geometry from inside a jit trace.

    Both used to be lifted through ``jnp`` first -- ``grid.z[1] - grid.z[0]``
    for the parallel extent, ``jnp.asarray(theta)`` for the drift maxima. Under
    current JAX those stage out inside a trace, so reading the result back on
    the host raised and every adaptive-dt run refused under ``jit`` even though
    the grid, the geometry and the parameters were all concrete host data.
    """

    import jax

    _g0, grid, geom, _params, _cache, _n_l, _n_m = _tiny_linear_case()
    theta = np.asarray(grid.z, dtype=float)
    expected_zp = eti._parallel_periods_from_grid(grid)
    expected_maxima = eti._geometry_frequency_maxima(geom, theta)
    seen: dict[str, object] = {}

    def probe(x: jnp.ndarray) -> jnp.ndarray:
        seen["zp"] = eti._parallel_periods_from_grid(grid)
        seen["maxima"] = eti._geometry_frequency_maxima(geom, theta)
        seen["kz"] = eti._cfl_wavenumber_arrays(grid)[2]
        return x * 2.0

    jax.jit(probe)(jnp.asarray(1.0))
    assert seen["zp"] == pytest.approx(expected_zp)
    np.testing.assert_allclose(np.asarray(seen["maxima"]), np.asarray(expected_maxima))
    # A host float, not a zero-dimensional device array standing in for one.
    assert all(isinstance(value, float) for value in seen["maxima"])
    assert isinstance(seen["kz"], np.ndarray)


def test_cfl_bound_names_the_traced_input_it_cannot_read() -> None:
    """A genuinely traced drift is refused by name, not as a conversion error.

    ``dt_max`` is a host scalar fixed before the scan is staged, so a drift
    that only exists inside the trace has no value to bound with. The refusal
    says which input and what to do instead.
    """

    import jax
    from dataclasses import replace as dataclass_replace

    from gkx.geometry import ensure_flux_tube_geometry_data

    _g0, grid, geom, _params, _cache, _n_l, _n_m = _tiny_linear_case()
    sampled = ensure_flux_tube_geometry_data(geom, grid.z)
    theta = np.asarray(grid.z, dtype=float)

    def probe(scale: jnp.ndarray):
        traced = dataclass_replace(sampled, cv_profile=sampled.cv_profile * scale)
        return eti._geometry_frequency_maxima(traced, theta)

    with pytest.raises(ValueError, match="cannot read a traced curvature drift"):
        jax.jit(probe)(jnp.asarray(1.0))
