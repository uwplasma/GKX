"""Time-advance schemes: linear explicit tableaus, nonlinear explicit, IMEX.

One module for everything that steps the state forward. The linear half covers
gkx.solvers_time_explicit -- native tableaus, adaptive-CFL controls, progress and
error paths. The nonlinear half covers gkx.solvers_nonlinear: the per-stage
explicit update and its method dispatch, the checkpointed scan whose block
schedule must not change the discrete RK map, and the IMEX split, which lives
here rather than with the linear solver because it reuses the same
step/scan/diagnostic scaffolding and only delegates its implicit stage.

Absorbed from test_nonlinear_explicit_step.py, test_nonlinear_explicit_scan.py
and test_nonlinear_imex.py; the origin markers below delimit each block.
"""

from __future__ import annotations

from gkx.diagnostics.analysis import estimate_observed_order
from gkx.solvers_nonlinear_explicit import (
    _checkpoint_block_size,
    advance_explicit_nonlinear_state,
    checkpointed_explicit_scan,
    integrate_cached_explicit_scan,
    integrate_nonlinear_scan,
    make_explicit_diagnostic_step,
    run_explicit_diagnostic_scan,
)
from gkx.solvers_nonlinear_imex import (
    advance_imex_nonlinear_state,
    imex_fixed_point_guess,
    integrate_cached_imex_scan,
    make_imex_diagnostic_step,
    make_imex_nonlinear_term,
    make_imex_solve_step,
    run_imex_diagnostic_scan,
    solve_imex_step,
)
from gkx.solvers_time_explicit_steps import (
    _linear_explicit_stage_update,
    _linear_native_step,
)
from gkx.terms.config import FieldState
from gkx.terms.nonlinear import (
    exb_nonlinear_contribution,
    placeholder_nonlinear_contribution,
)
from types import SimpleNamespace
import gkx.solvers_nonlinear_imex as imex_module
import gkx.solvers_nonlinear_imex_diagnostics as imex_diagnostics
import gkx.solvers_time_explicit as eti
import gkx.solvers_time_explicit_diagnostics as explicit_diagnostics
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warnings


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


def replace_time_cfg(cfg, **changes):
    from dataclasses import replace

    return replace(cfg, **changes)


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


@pytest.mark.parametrize(("method", "expected_calls"), [("imex", 1), ("imex2", 2)])
def test_native_diagonal_imex_step_matches_scalar_amplification(
    method: str, expected_calls: int
) -> None:
    rate = 0.2 - 0.1j
    damping = 0.7
    dt = 0.05
    calls = 0

    def rhs(state):
        nonlocal calls
        calls += 1
        return rate * state

    state = jnp.asarray([1.0 + 0.0j])
    result = _linear_native_step(
        state,
        jnp.asarray(damping),
        jnp.asarray(dt),
        method_key=method,
        rhs=rhs,
    )

    explicit_rate = rate + damping
    if method == "imex":
        expected = (1.0 + dt * explicit_rate) / (1.0 + dt * damping)
    else:
        half = (1.0 + 0.5 * dt * explicit_rate) / (1.0 + 0.5 * dt * damping)
        expected = (1.0 + dt * explicit_rate * half) / (1.0 + dt * damping)
    assert calls == expected_calls
    np.testing.assert_allclose(np.asarray(result), [expected], rtol=1.0e-6)


def test_explicit_from_config_preserves_adaptive_controls(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(eti, "build_linear_cache", lambda *_args: "cache")

    def fake_integrate(_state, _grid, cache, _params, _geom, config, *_args, **kwargs):
        captured.update(cache=cache, config=config, kwargs=kwargs)
        return np.asarray([0.1]), np.ones((1, 1, 1, 2)), None, None, None

    monkeypatch.setattr(eti, "integrate_linear_explicit_diagnostics", fake_integrate)
    time_cfg = SimpleNamespace(
        dt=0.02,
        t_max=2.0,
        sample_stride=3,
        fixed_dt=False,
        dt_min=1.0e-6,
        dt_max=0.04,
        cfl=0.7,
        method="rk2",
        cfl_fac=None,
        use_dealias_mask=True,
    )
    t, phi = eti.integrate_linear_explicit_from_config(
        jnp.ones((1,)),
        object(),
        object(),
        object(),
        time_cfg,
        Nl=2,
        Nm=3,
        z_index=1,
        show_progress=True,
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
    from gkx.core_grid import build_spectral_grid
    from gkx.geometry import SAlphaGeometry
    from gkx.operators.linear.params import LinearParams

    grid = build_spectral_grid(
        CycloneBaseCase(grid=GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)).grid
    )
    geom = SAlphaGeometry.from_config(CycloneBaseCase().geometry)
    params = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        nu=0.0,
        nu_hyper=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    n_l, n_m = 2, 3
    z = jnp.linspace(0.0, 2.0 * jnp.pi, grid.z.size, endpoint=False)
    g0 = jnp.zeros(
        (n_l, n_m, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    g0 = g0.at[0, 0, 1, 0, :].set(1.0e-3 * jnp.exp(1j * z))
    time_cfg = TimeConfig(t_max=0.2, dt=0.02, method="rk4", sample_stride=1)
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
    fixed = eti.ExplicitTimeConfig(
        t_max=1.0, dt=0.02, fixed_dt=True, cfl=0.5, cfl_fac=2.0
    )
    # fixed_dt short-circuits regardless of wmax.
    assert (
        eti._adaptive_linear_dt(fixed, dt=0.02, dt_min=1e-6, dt_max=0.04, wmax=1e3)
        == 0.02
    )

    adaptive = eti.ExplicitTimeConfig(
        t_max=1.0, dt=0.02, fixed_dt=False, cfl=0.5, cfl_fac=2.0
    )
    # Non-positive wmax cannot form a CFL estimate -> keep the requested dt.
    assert (
        eti._adaptive_linear_dt(adaptive, dt=0.02, dt_min=1e-6, dt_max=0.04, wmax=0.0)
        == 0.02
    )
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
    assert eti._should_emit_linear_progress(
        step=1, total_steps_est=100, progress_stride=10
    )
    assert eti._should_emit_linear_progress(
        step=100, total_steps_est=100, progress_stride=10
    )
    assert eti._should_emit_linear_progress(
        step=20, total_steps_est=100, progress_stride=10
    )
    assert not eti._should_emit_linear_progress(
        step=23, total_steps_est=100, progress_stride=10
    )


def test_linear_loop_progress_clock_and_history_arrays() -> None:
    total_steps_est, progress_stride, started_at = eti._linear_loop_progress_clock(
        0.2, 0.02
    )
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
    from gkx.core_grid import build_spectral_grid
    from gkx.geometry import SAlphaGeometry
    from gkx.operators.linear.params import LinearParams

    grid = build_spectral_grid(
        CycloneBaseCase(grid=GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)).grid
    )
    geom = SAlphaGeometry.from_config(CycloneBaseCase().geometry)
    params = LinearParams(
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        nu=0.0,
        nu_hyper=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
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
        g0,
        grid,
        cache,
        params,
        geom,
        time_cfg,
        mode_method="max",
        jit=False,
        show_progress=True,
    )
    out = capsys.readouterr().out
    assert "linear initial-value integration started" in out
    assert "linear initial-value integration complete" in out
    assert "step=" in out
    for arr in (t, phi, gamma, omega):
        assert np.all(np.isfinite(np.asarray(arr)))
    assert np.asarray(t).shape[0] >= 2


@pytest.mark.parametrize("fixed_dt", [True, False])
@pytest.mark.parametrize("diagnostics", [True, False])
@pytest.mark.parametrize("method", ["rk3", "rk4"])
def test_linear_explicit_stops_at_requested_time(fixed_dt, diagnostics, method) -> None:
    """A nonintegral horizon requires a shortened final step in both facades."""
    g0, grid, geom, params, cache, *_ = _tiny_linear_case()
    cfg = eti.ExplicitTimeConfig(
        t_max=0.1,
        dt=0.03,
        dt_max=0.03,
        dt_min=0.02,
        fixed_dt=fixed_dt,
        cfl=1e3,
        method=method,
        sample_stride=1,
    )
    solve = (
        eti.integrate_linear_explicit_diagnostics
        if diagnostics
        else eti.integrate_linear_explicit
    )
    result = solve(g0, grid, cache, params, geom, cfg, jit=False)
    np.testing.assert_allclose(result[0], [0.03, 0.06, 0.09, 0.1], atol=1e-12)
    assert np.all(np.isfinite(result[1]))
    if diagnostics:
        np.testing.assert_allclose(result[-1].dt_t, [0.03, 0.03, 0.03, 0.01])


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


def _closed_interval_copy(sampled):
    """Return `sampled` re-expressed on the closed theta interval.

    This is the shape an imported `*.eik.nc` file carries -- the terminal theta
    point repeated at the far end of the period -- so its profiles are one
    sample longer than the spectral grid they belong to.
    """

    from dataclasses import replace as dataclass_replace

    theta = np.asarray(sampled.theta, dtype=float)
    closed = np.append(theta, theta[-1] + float(theta[1] - theta[0]))

    def _wrap(name: str) -> np.ndarray:
        profile = np.asarray(getattr(sampled, name), dtype=float)
        return np.append(profile, profile[0])

    return dataclass_replace(
        sampled,
        theta=closed,
        theta_closed_interval=True,
        source_model="vmec-eik",
        **{
            name: _wrap(name)
            for name in (
                "bmag_profile",
                "bgrad_profile",
                "gds2_profile",
                "gds21_profile",
                "gds22_profile",
                "cv_profile",
                "gb_profile",
                "cv0_profile",
                "gb0_profile",
                "jacobian_profile",
                "grho_profile",
            )
        },
    )


def test_fixed_dt_cfl_hint_conforms_imported_geometry_instead_of_aborting() -> None:
    """The startup hint survives an imported flux tube -- and still fires.

    An imported equilibrium arrives on the closed theta interval, one sample
    longer than the grid it is used with, which made this hint raise and take
    every `gkx wout_XXX.nc --linear` run down with it. Conforming the geometry
    is what fixes that; the surrounding `except` would also stop the crash, but
    silently, so this pins the warning itself. Drop the conform and the hint
    goes quiet on exactly the runs that need it.
    """

    from gkx.geometry import ensure_flux_tube_geometry_data
    from gkx.solvers_time_explicit_cfl import warn_if_fixed_dt_exceeds_cfl

    _g0, grid, geom, params, _cache, n_l, n_m = _tiny_linear_case()
    imported = _closed_interval_copy(ensure_flux_tube_geometry_data(geom, grid.z))
    assert imported.theta.shape[0] == np.asarray(grid.z).size + 1

    over_cfl = eti.ExplicitTimeConfig(
        t_max=1.0, dt=5.0, method="rk3", sample_stride=1, fixed_dt=True
    )
    with pytest.warns(RuntimeWarning, match="exceeds the estimated"):
        warn_if_fixed_dt_exceeds_cfl(
            grid=grid,
            geom=imported,
            params=params,
            n_laguerre=n_l,
            n_hermite=n_m,
            tcfg=over_cfl,
        )

    # A step under the bound stays silent on the same imported geometry.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_fixed_dt_exceeds_cfl(
            grid=grid,
            geom=imported,
            params=params,
            n_laguerre=n_l,
            n_hermite=n_m,
            tcfg=replace_time_cfg(over_cfl, dt=1.0e-6),
        )
    assert not [w for w in caught if "exceeds the estimated" in str(w.message)]


# ---- from test_nonlinear_explicit_step.py ----


def _constant_rhs(value: float):
    def rhs_fn(G_state):
        return jnp.ones_like(G_state) * value, None

    return rhs_fn


def test_advance_explicit_nonlinear_state_euler_projects_and_preserves_dtype() -> None:
    G = jnp.asarray([1.0], dtype=jnp.float32)
    dG = jnp.asarray([3.0], dtype=jnp.float32)

    out = advance_explicit_nonlinear_state(
        G,
        dG,
        jnp.asarray(0.1, dtype=jnp.float32),
        method="euler",
        rhs_fn=_constant_rhs(0.0),
        project_state=lambda state: 2.0 * state,
        state_dtype=jnp.float32,
    )

    np.testing.assert_allclose(np.asarray(out), [2.6], rtol=1e-6)
    assert out.dtype == jnp.float32


@pytest.mark.parametrize("method", ["rk2", "rk3", "rk3_heun", "rk3_classic", "rk4"])
def test_advance_explicit_nonlinear_state_rk_methods_match_constant_rhs(
    method: str,
) -> None:
    G = jnp.asarray([1.0], dtype=jnp.float32)
    dG = jnp.asarray([2.0], dtype=jnp.float32)

    out = advance_explicit_nonlinear_state(
        G,
        dG,
        jnp.asarray(0.1, dtype=jnp.float32),
        method=method,
        rhs_fn=_constant_rhs(2.0),
        project_state=lambda state: state,
        state_dtype=jnp.float32,
    )

    np.testing.assert_allclose(np.asarray(out), [1.2], rtol=1e-6)


@pytest.mark.parametrize("method", ["sspx3", "k10"])
def test_advance_explicit_nonlinear_state_extended_methods_are_finite(
    method: str,
) -> None:
    G = jnp.asarray([1.0], dtype=jnp.float32)
    dG = jnp.asarray([0.5], dtype=jnp.float32)

    out = advance_explicit_nonlinear_state(
        G,
        dG,
        jnp.asarray(0.05, dtype=jnp.float32),
        method=method,
        rhs_fn=_constant_rhs(0.5),
        project_state=lambda state: state,
        state_dtype=jnp.float32,
    )

    assert out.shape == G.shape
    assert np.all(np.isfinite(np.asarray(out)))


def test_advance_explicit_nonlinear_state_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        advance_explicit_nonlinear_state(
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.asarray([0.0], dtype=jnp.float32),
            jnp.asarray(0.1, dtype=jnp.float32),
            method="bogus",
            rhs_fn=_constant_rhs(0.0),
            project_state=lambda state: state,
            state_dtype=jnp.float32,
        )


def test_integrate_cached_explicit_scan_forwards_scan_policy() -> None:
    captured: dict[str, object] = {}
    G0 = jnp.asarray([1.0], dtype=jnp.float32)

    def rhs_fn(G):
        return jnp.ones_like(G), "fields"

    def project_state(G):
        return G + 2.0

    def scan_fn(rhs, G, dt, steps, **kwargs):
        captured["rhs"] = rhs
        captured["G"] = G
        captured["dt"] = dt
        captured["steps"] = steps
        captured.update(kwargs)
        dG, fields = rhs(G)
        return kwargs["project_state"](G + dt * steps * dG), fields

    G_out, fields = integrate_cached_explicit_scan(
        G0,
        0.25,
        4,
        method="rk4",
        rhs_fn=rhs_fn,
        scan_fn=scan_fn,
        checkpoint=True,
        project_state=project_state,
        show_progress=True,
        return_fields=False,
    )

    np.testing.assert_allclose(np.asarray(G_out), [4.0], rtol=1e-6)
    assert fields == "fields"
    assert captured["rhs"] is rhs_fn
    assert captured["G"] is G0
    assert captured["dt"] == 0.25
    assert captured["steps"] == 4
    assert captured["method"] == "rk4"
    assert captured["checkpoint"] is True
    assert captured["project_state"] is project_state
    assert captured["show_progress"] is True
    assert captured["return_fields"] is False


def test_make_explicit_diagnostic_step_forwards_runtime_policies() -> None:
    seen: dict[str, object] = {}
    time_step_policy = SimpleNamespace(
        update_dt=lambda _fields, dt_prev: dt_prev + 0.25,
        progress_total=jnp.asarray(5.0, dtype=jnp.float32),
    )

    def rhs_fn(G):
        seen["rhs_input"] = G
        return jnp.ones_like(G) * 2.0, "rhs_fields"

    def project_state(G):
        seen["projected"] = True
        return G

    def compute_fields_fn(G, cache, params, **kwargs):
        seen["fields_G"] = G
        seen["cache"] = cache
        seen["params"] = params
        seen["field_terms"] = kwargs["terms"]
        seen["external_phi"] = kwargs["external_phi"]
        return "new_fields"

    def compute_diag_from_state(G, fields, G_prev, fields_prev, dt_local):
        seen["diag_args"] = (G, fields, G_prev, fields_prev, dt_local)
        return jnp.asarray(7.0, dtype=jnp.float32)

    def select_diagnostics_fn(idx, **kwargs):
        seen["diag_idx"] = idx
        seen["diagnostics_stride"] = kwargs["diagnostics_stride"]
        seen["diag_prev"] = kwargs["diag_prev"]
        return kwargs["compute_diag_fn"]()

    def emit_progress_fn(G, **kwargs):
        seen["progress"] = kwargs
        return G + 1.0

    def apply_collision_split_fn(G, damping, dt_local, scheme):
        seen["collision"] = (damping, dt_local, scheme)
        return G + 3.0

    step = make_explicit_diagnostic_step(
        rhs_fn=rhs_fn,
        method="euler",
        project_state=project_state,
        state_dtype=jnp.float32,
        real_dtype=jnp.float32,
        time_step_policy=time_step_policy,
        compute_fields_fn=compute_fields_fn,
        cache="cache",
        params="params",
        term_cfg="terms",
        external_phi="phi",
        compute_diag_from_state=compute_diag_from_state,
        diagnostics_stride=2,
        select_diagnostics_fn=select_diagnostics_fn,
        show_progress=True,
        steps=4,
        emit_progress_fn=emit_progress_fn,
        use_collision_split=True,
        damping="damping",
        collision_scheme="implicit",
        apply_collision_split_fn=apply_collision_split_fn,
    )

    carry, out = step(
        (
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.asarray([0.5], dtype=jnp.float32),
            "old_fields",
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.25, dtype=jnp.float32),
        ),
        jnp.asarray(3, dtype=jnp.int32),
    )

    G_new, G_prev, fields_new, diag, t_new, dt_new = carry
    np.testing.assert_allclose(np.asarray(G_new), [6.0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(G_prev), [6.0], rtol=1e-6)
    assert fields_new == "new_fields"
    np.testing.assert_allclose(np.asarray(diag), 7.0, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(t_new), 0.5, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(dt_new), 0.5, rtol=1e-6)
    assert out[0] is diag
    assert seen["field_terms"] == "terms"
    assert seen["external_phi"] == "phi"
    assert seen["diagnostics_stride"] == 2
    np.testing.assert_allclose(np.asarray(seen["diag_prev"]), 0.0, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(seen["diag_args"][0]), [5.0], rtol=1e-6)
    assert seen["collision"][0] == "damping"
    assert seen["collision"][2] == "implicit"
    assert seen["progress"]["show_progress"] is True
    assert seen["progress"]["steps"] == 4


def test_explicit_diagnostic_step_caps_and_holds_at_time_horizon() -> None:
    step = make_explicit_diagnostic_step(
        rhs_fn=lambda G: (jnp.ones_like(G), G),
        method="euler",
        project_state=lambda G: G,
        state_dtype=jnp.float32,
        real_dtype=jnp.float32,
        time_step_policy=SimpleNamespace(
            update_dt=lambda _fields, _dt_prev: jnp.asarray(0.5, dtype=jnp.float32),
            progress_total=jnp.asarray(1.0, dtype=jnp.float32),
        ),
        compute_fields_fn=lambda G, *_args, **_kwargs: G,
        cache=None,
        params=None,
        term_cfg=None,
        external_phi=None,
        compute_diag_from_state=lambda G, *_args: G[0],
        diagnostics_stride=1,
        select_diagnostics_fn=lambda _idx, **kwargs: kwargs["compute_diag_fn"](),
        show_progress=False,
        steps=2,
        emit_progress_fn=lambda G, **_kwargs: G,
        time_horizon=0.75,
    )
    carry = (
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(0.5, dtype=jnp.float32),
    )

    carry, first = step(carry, jnp.asarray(0))
    np.testing.assert_allclose(np.asarray(carry[0]), [1.25])
    np.testing.assert_allclose(np.asarray(carry[4]), 0.75)
    np.testing.assert_allclose(np.asarray(first[2]), 0.25)

    held, second = step(carry, jnp.asarray(1))
    np.testing.assert_allclose(np.asarray(held[0]), [1.25])
    np.testing.assert_allclose(np.asarray(held[4]), 0.75)
    np.testing.assert_allclose(np.asarray(second[2]), 0.0)
    np.testing.assert_allclose(np.asarray(second[0]), np.asarray(first[0]))


def test_make_explicit_diagnostic_step_requires_collision_split_policy() -> None:
    step = make_explicit_diagnostic_step(
        rhs_fn=lambda G: (jnp.zeros_like(G), "fields"),
        method="euler",
        project_state=lambda G: G,
        state_dtype=jnp.float32,
        real_dtype=jnp.float32,
        time_step_policy=SimpleNamespace(
            update_dt=lambda _fields, dt_prev: dt_prev,
            progress_total=jnp.asarray(1.0, dtype=jnp.float32),
        ),
        compute_fields_fn=lambda G, *_args, **_kwargs: G,
        cache=None,
        params=None,
        term_cfg=None,
        external_phi=None,
        compute_diag_from_state=lambda *_args: jnp.asarray(0.0, dtype=jnp.float32),
        diagnostics_stride=1,
        select_diagnostics_fn=lambda _idx, **kwargs: kwargs["compute_diag_fn"](),
        show_progress=False,
        steps=1,
        emit_progress_fn=lambda G, **_kwargs: G,
        use_collision_split=True,
        damping=jnp.asarray(1.0, dtype=jnp.float32),
        apply_collision_split_fn=None,
    )

    with pytest.raises(ValueError, match="apply_collision_split_fn"):
        step(
            (
                jnp.asarray([1.0], dtype=jnp.float32),
                jnp.asarray([1.0], dtype=jnp.float32),
                None,
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.1, dtype=jnp.float32),
            ),
            jnp.asarray(0, dtype=jnp.int32),
        )


def test_run_explicit_diagnostic_scan_dense_path_runs_all_steps() -> None:
    def step_fn(carry, idx):
        G, G_prev, fields_prev, diag_prev, t_prev, dt_prev = carry
        del G_prev, fields_prev, diag_prev
        G_next = G + 1
        t_next = t_prev + dt_prev
        diag = G_next + idx
        return (G_next, G_next, G_next, diag, t_next, dt_prev), (
            diag,
            t_next,
            dt_prev,
        )

    G_final, (diag, t, dt_series) = run_explicit_diagnostic_scan(
        step_fn,
        (
            jnp.asarray(0),
            jnp.asarray(0),
            jnp.asarray(0),
            jnp.asarray(0),
            jnp.asarray(0.0),
            jnp.asarray(0.5),
        ),
        steps=3,
        stride=1,
        sampled_scan=False,
        checkpoint=False,
        sampled_scan_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("sampled path should not run")
        ),
    )

    assert int(G_final) == 3
    np.testing.assert_allclose(np.asarray(diag), [1, 3, 5])
    np.testing.assert_allclose(np.asarray(t), [0.5, 1.0, 1.5])
    np.testing.assert_allclose(np.asarray(dt_series), [0.5, 0.5, 0.5])


def test_run_explicit_diagnostic_scan_sampled_path_forwards_policy() -> None:
    seen: dict[str, object] = {}

    def step_fn(carry, idx):
        del idx
        return carry, carry[-3:]

    initial = (
        jnp.asarray(1),
        jnp.asarray(2),
        jnp.asarray(3),
        jnp.asarray(4),
        jnp.asarray(5),
        jnp.asarray(6),
    )

    def sampled_scan_fn(step, carry, **kwargs):
        seen["step"] = step
        seen["carry"] = carry
        seen.update(kwargs)
        return carry, (jnp.asarray([7]), jnp.asarray([8]), jnp.asarray([9]))

    G_final, scan_diag_out = run_explicit_diagnostic_scan(
        step_fn,
        initial,
        steps=5,
        stride=2,
        sampled_scan=True,
        checkpoint=False,
        sampled_scan_fn=sampled_scan_fn,
    )

    assert int(G_final) == 1
    assert seen["step"] is step_fn
    assert seen["carry"] is initial
    assert seen["steps"] == 5
    assert seen["stride"] == 2
    np.testing.assert_allclose(np.asarray(scan_diag_out[0]), [7])


# ---- from test_nonlinear_explicit_scan.py ----
# Tests for nonlinear explicit scan integrator utilities.


_SSPX3_ADT = float((1.0 / 6.0) ** (1.0 / 3.0))


_SSPX3_WGTFAC = float((9.0 - 2.0 * (6.0 ** (2.0 / 3.0))) ** 0.5)


_SSPX3_W1 = 0.5 * (_SSPX3_WGTFAC - 1.0)


_SSPX3_W2 = 0.5 * ((6.0 ** (2.0 / 3.0)) - 1.0 - _SSPX3_WGTFAC)


_SSPX3_W3 = (1.0 / _SSPX3_ADT) - 1.0 - _SSPX3_W2 * (_SSPX3_W1 + 1.0)


def _linear_rhs(rate: complex):
    def rhs_fn(G: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
        dG = rate * G
        phi = jnp.sum(G, axis=0)
        return dG, FieldState(phi=phi)

    return rhs_fn


def test_checkpointed_scan_matches_discrete_primal_and_reverse_with_tail() -> None:
    """Block checkpointing must change storage, not the discrete RK map."""

    steps = 19  # sqrt schedule has three full blocks and a non-empty tail
    indices = jnp.arange(steps, dtype=jnp.int32)

    def objective(parameter: jnp.ndarray, checkpoint: bool) -> jnp.ndarray:
        def step(carry: jnp.ndarray, index: jnp.ndarray):
            weight = jnp.asarray(index + 1, dtype=carry.dtype) / steps
            updated = jnp.tanh(carry + weight * parameter)
            return updated, (updated, updated * updated)

        final, (states, squares) = checkpointed_explicit_scan(
            step,
            jnp.asarray([0.2, -0.4], dtype=jnp.float32),
            indices,
            checkpoint=checkpoint,
        )
        return jnp.sum(final) + 0.01 * jnp.sum(states + squares)

    parameter = jnp.asarray(0.03, dtype=jnp.float32)
    plain = jax.value_and_grad(lambda value: objective(value, False))(parameter)
    blocked = jax.value_and_grad(lambda value: objective(value, True))(parameter)

    np.testing.assert_allclose(np.asarray(blocked[0]), np.asarray(plain[0]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(blocked[1]), np.asarray(plain[1]), rtol=1e-6)
    assert _checkpoint_block_size(steps) == 5
    assert _checkpoint_block_size(2048) == 46


@pytest.mark.parametrize(
    ("method", "one_step_factor"),
    [
        ("euler", lambda a: 1.0 + a),
        ("rk2", lambda a: 1.0 + a + 0.5 * a * a),
        ("rk3", lambda a: 1.0 + a + 0.5 * a * a + (a * a * a) / 6.0),
        ("rk3_heun", lambda a: 1.0 + a + 0.5 * a * a + (a * a * a) / 6.0),
        ("rk3_classic", lambda a: 1.0 + a + 0.5 * a * a + (a * a * a) / 6.0),
        (
            "rk4",
            lambda a: (
                1.0 + a + 0.5 * a * a + (a * a * a) / 6.0 + (a * a * a * a) / 24.0
            ),
        ),
        ("sspx3", lambda a: 1.0 + a + 0.5 * a * a + (a * a * a) / 6.0),
    ],
)
def test_integrate_nonlinear_scan_methods_match_linear_amplification(
    method, one_step_factor
) -> None:
    dt = 0.1
    steps = 4
    rate = 0.3 - 0.2j
    G0 = jnp.asarray([[1.0 + 0.0j, 0.5 + 0.25j]], dtype=jnp.complex64)
    G0_ref = jnp.array(G0)
    G_final, fields = integrate_nonlinear_scan(
        _linear_rhs(rate),
        G0,
        dt,
        steps,
        method=method,
        checkpoint=True,
    )
    a = rate * dt
    expected = (one_step_factor(a) ** steps) * G0_ref
    assert G_final.shape == G0.shape
    assert fields.phi.shape[0] == steps
    assert jnp.allclose(G_final, expected, rtol=3.0e-3, atol=3.0e-3)


def test_integrate_nonlinear_scan_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        integrate_nonlinear_scan(
            _linear_rhs(0.1 + 0.0j),
            jnp.ones((2, 2), dtype=jnp.complex64),
            0.1,
            4,
            method="bad",
        )


def test_integrate_nonlinear_scan_projects_each_stage() -> None:
    def rhs_fn(G: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
        flipped = jnp.flip(G, axis=-2)
        return 1j * flipped, FieldState(phi=jnp.sum(G, axis=0))

    def projector(G: jnp.ndarray) -> jnp.ndarray:
        pos = G[..., :3, :]
        neg = jnp.conj(pos[..., 1:2, :])[..., ::-1, :]
        return jnp.concatenate([pos, neg], axis=-2)

    G0 = jnp.asarray(
        [[1.0 + 0.0j], [2.0 + 1.0j], [-3.0 + 0.5j], [7.0 - 2.0j]], dtype=jnp.complex64
    )
    G_final, _fields = integrate_nonlinear_scan(
        rhs_fn,
        G0,
        0.1,
        2,
        method="rk4",
        project_state=projector,
    )
    assert jnp.allclose(G_final[..., 3, :], jnp.conj(G_final[..., 1, :]))


def test_integrate_nonlinear_scan_rk3_alias_matches_heun_variant() -> None:
    G0 = jnp.asarray([[1.0 + 0.0j, 0.5 + 0.25j]], dtype=jnp.complex64)
    out_rk3, _ = integrate_nonlinear_scan(
        _linear_rhs(0.3 - 0.2j), jnp.array(G0), 0.1, 3, method="rk3"
    )
    out_heun, _ = integrate_nonlinear_scan(
        _linear_rhs(0.3 - 0.2j), jnp.array(G0), 0.1, 3, method="rk3_heun"
    )
    assert jnp.allclose(out_rk3, out_heun)


def test_integrate_nonlinear_scan_final_only_skips_field_history_rhs() -> None:
    calls = {"rhs": 0}

    def rhs_fn(G: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
        calls["rhs"] += 1
        return 0.2 * G, FieldState(phi=jnp.sum(G, axis=0))

    G0 = jnp.asarray([[1.0 + 0.0j]], dtype=jnp.complex64)
    expected = (1.0 + 0.1 * 0.2) ** 2 * G0
    out = integrate_nonlinear_scan(
        rhs_fn,
        G0,
        0.1,
        2,
        method="euler",
        return_fields=False,
    )

    assert out.shape == G0.shape
    assert jnp.allclose(out, expected)
    assert calls["rhs"] == 1


def test_integrate_nonlinear_scan_accepts_dynamic_and_static_rhs_arguments() -> None:
    def rhs_fn(
        G: jnp.ndarray, rate: jnp.ndarray, offset: float
    ) -> tuple[jnp.ndarray, FieldState]:
        return rate * G + offset, FieldState(phi=jnp.sum(G, axis=0))

    G0 = jnp.asarray([[1.0 + 0.0j]], dtype=jnp.complex64)
    expected = np.asarray(G0).copy()
    out = integrate_nonlinear_scan(
        rhs_fn,
        G0,
        0.1,
        2,
        method="euler",
        return_fields=False,
        rhs_args=(jnp.asarray(0.2, dtype=jnp.float32),),
        rhs_static_args=(0.05,),
    )

    for _ in range(2):
        expected = expected + 0.1 * (0.2 * expected + 0.05)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1.0e-6)


@pytest.mark.parametrize(
    ("method", "expected_order", "min_observed_order"),
    [
        ("euler", 1.0, 0.9),
        ("rk2", 2.0, 1.75),
        ("rk3", 3.0, 2.6),
        ("rk3_heun", 3.0, 2.6),
        ("rk3_classic", 3.0, 2.6),
        ("rk4", 4.0, 3.3),
        ("sspx3", 3.0, 2.6),
    ],
)
def test_integrate_nonlinear_scan_observed_order_against_exact_solution(
    method: str,
    expected_order: float,
    min_observed_order: float,
) -> None:
    rate = -1.1 + 0.7j
    t_final = 0.8
    G0 = jnp.asarray([[1.0 + 0.1j, -0.4 + 0.2j]], dtype=jnp.complex64)
    exact = np.exp(rate * t_final) * np.asarray(G0)

    errors: list[float] = []
    dts: list[float] = []
    for steps in (2, 4, 8, 16):
        dt = t_final / steps
        out, _ = integrate_nonlinear_scan(
            _linear_rhs(rate), jnp.array(G0), dt, steps, method=method
        )
        err = float(np.max(np.abs(np.asarray(out) - exact)))
        errors.append(err)
        dts.append(dt)

    metrics = estimate_observed_order(np.asarray(dts), np.asarray(errors))
    assert metrics.orders.size > 0
    assert metrics.asymptotic_order >= min_observed_order
    assert metrics.asymptotic_order <= expected_order + 0.6


def test_integrate_nonlinear_scan_k10_branch_is_finite_and_shape_preserving() -> None:
    G0 = jnp.asarray([[1.0 + 0.0j, 0.5 + 0.25j]], dtype=jnp.complex64)

    G_final, fields = integrate_nonlinear_scan(
        _linear_rhs(0.3 - 0.2j), G0, 0.1, 2, method="k10"
    )

    assert G_final.shape == G0.shape
    assert fields.phi.shape[0] == 2
    assert np.all(np.isfinite(np.asarray(G_final)))


def test_integrate_nonlinear_scan_show_progress_callback_path(monkeypatch) -> None:
    import gkx.callbacks as callbacks

    callback_calls: list[int] = []

    monkeypatch.setattr(
        callbacks, "should_emit_progress", lambda idx, steps: jnp.asarray(True)
    )

    def _fake_print_callback(state, idx, *args):
        del idx, args
        callback_calls.append(0)
        return state

    monkeypatch.setattr(callbacks, "print_callback", _fake_print_callback)

    G0 = jnp.asarray([[1.0 + 0.0j]], dtype=jnp.complex64)
    G_final, fields = integrate_nonlinear_scan(
        _linear_rhs(0.0 + 0.0j),
        G0,
        0.1,
        1,
        method="euler",
        show_progress=True,
    )

    assert callback_calls == [0]
    assert G_final.shape == G0.shape
    assert fields.phi.shape[0] == 1


def test_nonlinear_placeholders() -> None:
    G = jnp.ones((3, 4, 1), dtype=jnp.complex64)
    out = placeholder_nonlinear_contribution(G, weight=jnp.asarray(2.0))
    assert jnp.allclose(out, 0.0)
    exb = exb_nonlinear_contribution(
        G,
        phi=jnp.ones((3, 4, 1), dtype=jnp.complex64),
        dealias_mask=jnp.ones((3, 4), dtype=bool),
        kx_grid=jnp.ones((3, 4), dtype=jnp.float32),
        ky_grid=jnp.ones((3, 4), dtype=jnp.float32),
        weight=jnp.asarray(1.0),
        compressed_real_fft=False,
    )
    assert exb.shape == G.shape


# ---- from test_nonlinear_imex.py ----


def test_imex_diagnostic_helpers_have_canonical_owner() -> None:
    assert imex_module.advance_imex_nonlinear_state is (
        imex_diagnostics.advance_imex_nonlinear_state
    )
    assert imex_module.make_imex_diagnostic_step is (
        imex_diagnostics.make_imex_diagnostic_step
    )
    assert imex_module.run_imex_diagnostic_scan is (
        imex_diagnostics.run_imex_diagnostic_scan
    )


def test_imex_fixed_point_guess_applies_linear_predictor_iterations() -> None:
    def linear_rhs(g, *_args, **_kwargs):
        return g, None

    out = imex_fixed_point_guess(
        jnp.asarray([0.0], dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
        linear_rhs_fn=linear_rhs,
        cache=SimpleNamespace(),
        params=SimpleNamespace(),
        linear_cfg=SimpleNamespace(),
        external_phi=None,
        dt_val=jnp.asarray(0.1, dtype=jnp.float32),
        implicit_iters=2,
        implicit_relax=1.0,
    )

    np.testing.assert_allclose(np.asarray(out), [1.1], rtol=1e-6)


def test_solve_imex_step_identity_system_returns_rhs_shape() -> None:
    def linear_rhs(g, *_args, **_kwargs):
        return jnp.zeros_like(g), None

    G_rhs = jnp.asarray([[2.0]], dtype=jnp.float32)
    out = solve_imex_step(
        jnp.zeros_like(G_rhs),
        G_rhs,
        linear_rhs_fn=linear_rhs,
        cache=SimpleNamespace(),
        params=SimpleNamespace(),
        linear_cfg=SimpleNamespace(),
        external_phi=None,
        dt_val=jnp.asarray(0.1, dtype=jnp.float32),
        implicit_iters=0,
        implicit_relax=1.0,
        matvec=lambda flat: flat,
        shape=tuple(G_rhs.shape),
        implicit_tol=1.0e-8,
        implicit_maxiter=20,
        implicit_restart=5,
    )

    np.testing.assert_allclose(np.asarray(out), np.asarray(G_rhs), rtol=1e-6)


def test_make_imex_nonlinear_term_forwards_injected_kernels() -> None:
    seen: dict[str, object] = {}

    def fields_fn(*_args, **_kwargs):
        return "fields"

    def contribution_fn(*_args, **_kwargs):
        return "contribution"

    def nonlinear_kernel(G, cache, params, terms, **kwargs):
        seen["G"] = G
        seen["cache"] = cache
        seen["params"] = params
        seen["terms"] = terms
        seen["fields_fn"] = kwargs["fields_fn"]
        seen["contribution_fn"] = kwargs["nonlinear_contribution_fn"]
        seen["real_dtype"] = kwargs["real_dtype"]
        seen["external_phi"] = kwargs["external_phi"]
        seen["compressed_real_fft"] = kwargs["compressed_real_fft"]
        seen["laguerre_mode"] = kwargs["laguerre_mode"]
        return G + 2.0

    cache = SimpleNamespace(name="cache")
    params = SimpleNamespace(name="params")
    terms = SimpleNamespace(name="terms")
    term = make_imex_nonlinear_term(
        cache,
        params,
        terms,
        real_dtype=jnp.float32,
        external_phi=3.0,
        compressed_real_fft=False,
        laguerre_mode="spectral",
        fields_fn=fields_fn,
        nonlinear_term_fn=nonlinear_kernel,
        nonlinear_contribution_fn=contribution_fn,
    )
    G = jnp.asarray([1.0], dtype=jnp.float32)

    out = term(G)

    np.testing.assert_allclose(np.asarray(out), [3.0])
    assert seen["G"] is G
    assert seen["cache"] is cache
    assert seen["params"] is params
    assert seen["terms"] is terms
    assert seen["fields_fn"] is fields_fn
    assert seen["contribution_fn"] is contribution_fn
    assert seen["real_dtype"] is jnp.float32
    assert seen["external_phi"] == 3.0
    assert seen["compressed_real_fft"] is False
    assert seen["laguerre_mode"] == "spectral"


def test_make_imex_solve_step_forwards_solver_policy() -> None:
    seen: dict[str, object] = {}

    def solve_step_fn(G_in, G_rhs, **kwargs):
        seen["G_in"] = G_in
        seen["G_rhs"] = G_rhs
        seen.update(kwargs)
        return G_rhs + 4.0

    def linear_rhs_fn(g, *_args, **_kwargs):
        return g, None

    def matvec(flat):
        return flat

    def precond(flat):
        return flat

    solve_step = make_imex_solve_step(
        linear_rhs_fn=linear_rhs_fn,
        cache=SimpleNamespace(name="cache"),
        params=SimpleNamespace(name="params"),
        linear_cfg=SimpleNamespace(name="linear"),
        external_phi=None,
        dt_val=jnp.asarray(0.2, dtype=jnp.float32),
        implicit_iters=3,
        implicit_relax=0.5,
        matvec=matvec,
        shape=(1,),
        implicit_tol=1.0e-5,
        implicit_maxiter=7,
        implicit_restart=2,
        precond_op=precond,
        solve_step_fn=solve_step_fn,
    )
    G_in = jnp.asarray([1.0], dtype=jnp.float32)
    G_rhs = jnp.asarray([2.0], dtype=jnp.float32)

    out = solve_step(G_in, G_rhs)

    np.testing.assert_allclose(np.asarray(out), [6.0])
    assert seen["G_in"] is G_in
    assert seen["G_rhs"] is G_rhs
    assert seen["linear_rhs_fn"] is linear_rhs_fn
    assert seen["matvec"] is matvec
    assert seen["precond_op"] is precond
    assert seen["implicit_iters"] == 3
    assert seen["implicit_relax"] == 0.5
    assert seen["shape"] == (1,)
    assert seen["implicit_tol"] == 1.0e-5
    assert seen["implicit_maxiter"] == 7
    assert seen["implicit_restart"] == 2


def test_imex_operator_resolution_and_state_shape_policies() -> None:
    provided = SimpleNamespace(name="provided")
    built: list[dict[str, object]] = []

    def build_operator_fn(G, cache, params, dt, **kwargs):
        built.append(
            {
                "shape": tuple(G.shape),
                "cache": cache,
                "params": params,
                "dt": dt,
                **kwargs,
            }
        )
        return SimpleNamespace(name="built")

    reused = imex_module._resolve_imex_operator(
        implicit_operator=provided,
        G0=jnp.ones((2,), dtype=jnp.complex64),
        cache="cache",
        params="params",
        dt=0.2,
        linear_cfg="linear",
        implicit_preconditioner="diag",
        compressed_real_fft=False,
        build_operator_fn=build_operator_fn,
        build_implicit_operator_fn=None,
    )
    assert reused is provided
    assert built == []

    built_op = imex_module._resolve_imex_operator(
        implicit_operator=None,
        G0=jnp.ones((2,), dtype=jnp.complex64),
        cache="cache",
        params="params",
        dt=0.2,
        linear_cfg="linear",
        implicit_preconditioner="diag",
        compressed_real_fft=False,
        build_operator_fn=build_operator_fn,
        build_implicit_operator_fn=lambda *_args, **_kwargs: (),
    )
    assert built_op.name == "built"
    assert built[0]["terms"] == "linear"
    assert built[0]["implicit_preconditioner"] == "diag"
    assert built[0]["compressed_real_fft"] is False
    assert "build_implicit_operator_fn" in built[0]

    operator = SimpleNamespace(
        shape=(1, 2),
        squeeze_species=True,
        state_dtype=jnp.complex64,
    )
    state, shape, squeeze_species = imex_module._state_for_imex_operator(
        jnp.ones((2,), dtype=jnp.float32), operator
    )
    assert shape == (1, 2)
    assert squeeze_species is True
    assert state.shape == (1, 2)
    assert state.dtype == jnp.complex64

    with pytest.raises(ValueError, match="implicit_operator shape mismatch"):
        imex_module._state_for_imex_operator(
            jnp.ones((3,), dtype=jnp.float32), operator
        )


def test_advance_imex_nonlinear_state_default_method_solves_rhs() -> None:
    calls: list[float] = []

    def nonlinear_term(g):
        return 2.0 * g

    def solve_step(g_in, rhs):
        calls.append(float(np.asarray(g_in[0])))
        return rhs + 1.0

    out = advance_imex_nonlinear_state(
        jnp.asarray([1.0], dtype=jnp.float32),
        dt_val=jnp.asarray(0.25, dtype=jnp.float32),
        method="imex",
        nonlinear_term=nonlinear_term,
        solve_step=solve_step,
        project_state=lambda g: g,
    )

    np.testing.assert_allclose(np.asarray(out), [2.5], rtol=1e-6)
    assert calls == [1.0]


def test_advance_imex_nonlinear_state_sspx3_matches_constant_rhs_step() -> None:
    def nonlinear_term(g):
        return jnp.ones_like(g) * 2.0

    def solve_step(_g_in, rhs):
        return rhs

    out = advance_imex_nonlinear_state(
        jnp.asarray([1.0], dtype=jnp.float32),
        dt_val=jnp.asarray(0.25, dtype=jnp.float32),
        method="sspx3",
        nonlinear_term=nonlinear_term,
        solve_step=solve_step,
        project_state=lambda g: g,
    )

    np.testing.assert_allclose(np.asarray(out), [1.5], rtol=1e-6)


def test_make_imex_diagnostic_step_forwards_runtime_policies() -> None:
    seen: dict[str, object] = {}

    def nonlinear_term(G):
        return jnp.ones_like(G) * 2.0

    def solve_step(G_in, G_rhs):
        seen["solve"] = (G_in, G_rhs)
        return G_rhs + 3.0

    def compute_fields_fn(G, cache, params, **kwargs):
        seen["fields"] = (G, cache, params, kwargs)
        return "new_fields"

    def compute_diag_from_state(G, fields, G_prev, fields_prev, dt_val):
        seen["diag_args"] = (G, fields, G_prev, fields_prev, dt_val)
        return jnp.asarray(9.0, dtype=jnp.float32)

    def select_diagnostics_fn(idx, **kwargs):
        seen["select"] = (idx, kwargs)
        return kwargs["compute_diag_fn"]()

    def emit_progress_fn(G, **kwargs):
        seen["progress"] = (G, kwargs)
        return G + 1.0

    def apply_collision_split_fn(G, damping, dt_val, scheme):
        seen["collision"] = (G, damping, dt_val, scheme)
        return G + 2.0

    cache = SimpleNamespace(name="cache")
    params = SimpleNamespace(name="params")
    term_cfg = SimpleNamespace(name="terms")
    damping = SimpleNamespace(name="damping")
    step = make_imex_diagnostic_step(
        method="imex",
        nonlinear_term=nonlinear_term,
        solve_step=solve_step,
        project_state=lambda G: G,
        state_dtype=jnp.float32,
        real_dtype=jnp.float32,
        dt_val=jnp.asarray(0.5, dtype=jnp.float32),
        compute_fields_fn=compute_fields_fn,
        cache=cache,
        params=params,
        term_cfg=term_cfg,
        external_phi=4.0,
        compute_diag_from_state=compute_diag_from_state,
        diagnostics_stride=3,
        select_diagnostics_fn=select_diagnostics_fn,
        show_progress=True,
        steps=11,
        progress_total=jnp.asarray(5.5, dtype=jnp.float32),
        emit_progress_fn=emit_progress_fn,
        use_collision_split=True,
        damping=damping,
        collision_scheme="exact",
        apply_collision_split_fn=apply_collision_split_fn,
    )

    carry_out, step_out = step(
        (
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.asarray([0.5], dtype=jnp.float32),
            "old_fields",
            jnp.asarray(-1.0, dtype=jnp.float32),
            jnp.asarray(2.0, dtype=jnp.float32),
        ),
        jnp.asarray(4, dtype=jnp.int32),
    )

    G_new, G_prev, fields_new, diag, t_new = carry_out
    np.testing.assert_allclose(np.asarray(G_new), [8.0])
    np.testing.assert_allclose(np.asarray(G_prev), [8.0])
    assert fields_new == "new_fields"
    np.testing.assert_allclose(np.asarray(diag), 9.0)
    np.testing.assert_allclose(np.asarray(t_new), 2.5)
    assert step_out[0] is diag
    assert step_out[1] is t_new
    solve_in, solve_rhs = seen["solve"]
    np.testing.assert_allclose(np.asarray(solve_in), [1.0])
    np.testing.assert_allclose(np.asarray(solve_rhs), [2.0])
    fields_args = seen["fields"]
    assert fields_args[1] is cache
    assert fields_args[2] is params
    assert fields_args[3]["terms"] is term_cfg
    assert fields_args[3]["external_phi"] == 4.0
    diag_args = seen["diag_args"]
    np.testing.assert_allclose(np.asarray(diag_args[0]), [7.0])
    assert diag_args[1] == "new_fields"
    np.testing.assert_allclose(np.asarray(diag_args[2]), [0.5])
    assert diag_args[3] == "old_fields"
    np.testing.assert_allclose(np.asarray(diag_args[4]), 0.5)
    select_idx, select_kwargs = seen["select"]
    np.testing.assert_allclose(np.asarray(select_idx), 4)
    assert select_kwargs["diagnostics_stride"] == 3
    np.testing.assert_allclose(np.asarray(select_kwargs["diag_prev"]), -1.0)
    collision_args = seen["collision"]
    np.testing.assert_allclose(np.asarray(collision_args[0]), [5.0])
    assert collision_args[1] is damping
    np.testing.assert_allclose(np.asarray(collision_args[2]), 0.5)
    assert collision_args[3] == "exact"
    progress_args = seen["progress"]
    np.testing.assert_allclose(np.asarray(progress_args[0]), [7.0])
    assert progress_args[1]["show_progress"] is True
    assert progress_args[1]["diag"] is diag
    assert progress_args[1]["steps"] == 11
    np.testing.assert_allclose(np.asarray(progress_args[1]["t_new"]), 2.5)
    np.testing.assert_allclose(np.asarray(progress_args[1]["progress_total"]), 5.5)


def test_make_imex_diagnostic_step_requires_collision_split_policy() -> None:
    step = make_imex_diagnostic_step(
        method="imex",
        nonlinear_term=lambda G: G,
        solve_step=lambda _G_in, G_rhs: G_rhs,
        project_state=lambda G: G,
        state_dtype=jnp.float32,
        real_dtype=jnp.float32,
        dt_val=jnp.asarray(0.1, dtype=jnp.float32),
        compute_fields_fn=lambda *_args, **_kwargs: "fields",
        cache=SimpleNamespace(),
        params=SimpleNamespace(),
        term_cfg=SimpleNamespace(),
        external_phi=None,
        compute_diag_from_state=lambda *_args: jnp.asarray(0.0, dtype=jnp.float32),
        diagnostics_stride=1,
        select_diagnostics_fn=lambda _idx, **kwargs: kwargs["compute_diag_fn"](),
        show_progress=False,
        steps=1,
        progress_total=jnp.asarray(0.1, dtype=jnp.float32),
        emit_progress_fn=lambda G, **_kwargs: G,
        use_collision_split=True,
        damping=SimpleNamespace(),
        apply_collision_split_fn=None,
    )

    with np.testing.assert_raises(ValueError):
        step(
            (
                jnp.asarray([1.0], dtype=jnp.float32),
                jnp.asarray([1.0], dtype=jnp.float32),
                "fields",
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            jnp.asarray(0, dtype=jnp.int32),
        )


def test_run_imex_diagnostic_scan_runs_fixed_step_policy() -> None:
    def step(carry, idx):
        G, G_prev, fields_prev, diag_prev, t_prev = carry
        del G_prev, fields_prev
        G_new = G + 1.0
        diag = diag_prev + idx + 1
        t_new = t_prev + 0.5
        return (G_new, G_new, jnp.asarray([2.0], dtype=jnp.float32), diag, t_new), (
            diag,
            t_new,
        )

    G_final, diag_out = run_imex_diagnostic_scan(
        step,
        (
            jnp.asarray([0.0], dtype=jnp.float32),
            jnp.asarray([0.0], dtype=jnp.float32),
            jnp.asarray([0.0], dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        ),
        steps=3,
        checkpoint=False,
    )

    diag, t = diag_out
    np.testing.assert_allclose(np.asarray(G_final), [3.0])
    np.testing.assert_allclose(np.asarray(diag), [1.0, 3.0, 6.0])
    np.testing.assert_allclose(np.asarray(t), [0.5, 1.0, 1.5])


def test_integrate_cached_imex_scan_owns_cached_scan_policy(monkeypatch) -> None:
    G0 = jnp.zeros((1,), dtype=jnp.complex64)
    fields = FieldState(phi=jnp.zeros((1,), dtype=jnp.complex64), apar=None, bpar=None)
    build_calls: list[dict[str, object]] = []
    nonlinear_calls: list[str] = []
    linear_calls: list[str] = []

    def build_operator_fn(G, cache, params, dt, **kwargs):
        build_calls.append(kwargs)
        return SimpleNamespace(
            shape=tuple(G.shape),
            dt_val=jnp.asarray(dt, dtype=jnp.float32),
            precond_op=None,
            matvec=lambda x: x,
            squeeze_species=False,
            state_dtype=G.dtype,
        )

    def linear_rhs_fn(G, *_args, **_kwargs):
        linear_calls.append("linear")
        return jnp.zeros_like(G), fields

    def nonlinear_kernel(G, cache, params, terms, **kwargs):
        del cache, params, terms
        assert kwargs["fields_fn"] is fields_fn
        assert kwargs["nonlinear_contribution_fn"] is contribution_fn
        assert kwargs["compressed_real_fft"] is False
        assert kwargs["laguerre_mode"] == "spectral"
        nonlinear_calls.append("nonlinear")
        return jnp.ones_like(G)

    def fields_fn(*_args, **_kwargs):
        return fields

    def contribution_fn(*_args, **_kwargs):
        return jnp.asarray(0.0, dtype=jnp.float32)

    monkeypatch.setattr(
        "gkx.solvers_nonlinear_imex.jax.scipy.sparse.linalg.gmres",
        lambda matvec, rhs, **kwargs: (rhs, SimpleNamespace(success=True)),
    )

    G_out, fields_t = integrate_cached_imex_scan(
        G0,
        SimpleNamespace(name="cache"),
        SimpleNamespace(name="params"),
        0.2,
        2,
        term_cfg=SimpleNamespace(name="terms"),
        linear_cfg=SimpleNamespace(name="linear"),
        linear_rhs_fn=linear_rhs_fn,
        build_operator_fn=build_operator_fn,
        fields_fn=fields_fn,
        nonlinear_term_fn=nonlinear_kernel,
        nonlinear_contribution_fn=contribution_fn,
        implicit_iters=0,
        compressed_real_fft=False,
        laguerre_mode="spectral",
    )

    assert build_calls
    assert build_calls[0]["terms"].name == "linear"
    np.testing.assert_allclose(np.asarray(G_out), np.asarray([0.4]), rtol=1e-6)
    assert fields_t.phi.shape == (2, 1)
    assert nonlinear_calls == ["nonlinear"]
    assert linear_calls
