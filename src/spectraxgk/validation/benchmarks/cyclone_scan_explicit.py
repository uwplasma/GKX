"""Reference-aligned explicit-time policies for Cyclone ky scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


def explicit_time_config_for_scan_point(
    *,
    dt_i: float,
    steps_i: int,
    reference_aligned: bool,
    time_cfg: Any | None,
    time_base: Any,
    hooks: Any,
) -> Any:
    """Build the explicit-time configuration for one ky scan point."""

    t_max_val = dt_i * float(steps_i)
    if reference_aligned and time_cfg is None:
        fixed_dt_i = True
        dt_min_i = dt_i
        dt_max_i: float | None = dt_i
        cfl_i = 1.0
        cfl_fac_i = 1.0
    else:
        fixed_dt_i = bool(time_base.fixed_dt)
        dt_min_i = float(time_base.dt_min)
        dt_max_i = None if time_base.dt_max is None else float(time_base.dt_max)
        cfl_i = float(time_base.cfl)
        cfl_fac_i = hooks.resolve_cfl_fac(str(time_base.method), time_base.cfl_fac)
    return hooks.explicit_time_config(
        dt=dt_i,
        t_max=t_max_val,
        sample_stride=1,
        fixed_dt=fixed_dt_i,
        dt_min=dt_min_i,
        dt_max=dt_max_i,
        cfl=cfl_i,
        cfl_fac=cfl_fac_i,
    )


def explicit_reselection_target(
    *,
    explicit_growth_ok: bool,
    prev_omega: float | None,
    prev_prev_omega: float | None,
) -> float | None:
    """Predict the next branch frequency used when explicit fits jump branch."""

    target_omega = (
        prev_omega if (explicit_growth_ok and prev_omega is not None) else None
    )
    if (
        target_omega is not None
        and prev_prev_omega is not None
        and prev_omega is not None
        and prev_omega > prev_prev_omega
    ):
        target_omega = prev_omega + (prev_omega - prev_prev_omega)
    return target_omega


def krylov_reselected_frequency(
    G0: Any,
    cache: Any,
    params: Any,
    terms: Any,
    kcfg: Any,
    hooks: Any,
    diagnostic_norm: str,
) -> tuple[float, float]:
    """Re-evaluate a scan point with Krylov mode selection after a bad fit."""

    eig, _vec = hooks.dominant_eigenpair(
        jnp.array(G0),
        cache,
        params,
        terms=terms,
        krylov_dim=kcfg.krylov_dim,
        restarts=kcfg.restarts,
        omega_min_factor=kcfg.omega_min_factor,
        omega_target_factor=kcfg.omega_target_factor,
        omega_cap_factor=kcfg.omega_cap_factor,
        omega_sign=kcfg.omega_sign,
        method=kcfg.method,
        power_iters=kcfg.power_iters,
        power_dt=kcfg.power_dt,
        shift=kcfg.shift,
        shift_source=kcfg.shift_source,
        shift_tol=kcfg.shift_tol,
        shift_maxiter=kcfg.shift_maxiter,
        shift_restart=kcfg.shift_restart,
        shift_solve_method=kcfg.shift_solve_method,
        shift_preconditioner=kcfg.shift_preconditioner,
        shift_selection=kcfg.shift_selection,
        mode_family=kcfg.mode_family,
        fallback_method=kcfg.fallback_method,
        fallback_real_floor=kcfg.fallback_real_floor,
    )
    gamma_k = float(np.real(eig))
    omega_k = float(abs(-np.imag(eig)))
    return hooks.normalize_growth_rate(gamma_k, omega_k, params, diagnostic_norm)


def choose_reselected_frequency(
    *,
    gamma: float,
    omega: float,
    gamma_k: float,
    omega_k: float,
    target_omega: float,
) -> tuple[float, float]:
    """Pick the frequency closest to the continuation target without growth jumps."""

    candidates: list[tuple[float, float]] = [(float(gamma), float(abs(omega)))]
    gamma_base = abs(float(gamma))
    gamma_delta_limit = max(3.0 * gamma_base, gamma_base + 0.05, 1.0e-3)
    if (
        np.isfinite(gamma_k)
        and np.isfinite(omega_k)
        and gamma_k > 0.0
        and abs(gamma_k - float(gamma)) <= gamma_delta_limit
    ):
        candidates.append((gamma_k, omega_k))

    def _score(candidate: tuple[float, float]) -> float:
        g_val, o_val = candidate
        penalty = 0.0 if g_val > 0.0 else 1.0e3
        return penalty + abs(o_val - target_omega)

    return min(candidates, key=_score)


@dataclass(frozen=True)
class _CycloneExplicitPoint:
    """Prepared state for one ky point in the explicit Cyclone scan."""

    index: int
    ky_value: float
    grid: Any
    state: Any
    cache: Any
    dt: float
    steps: int


@dataclass(frozen=True)
class _CycloneExplicitFit:
    """Raw explicit-time fit before optional branch reselection."""

    gamma: float
    omega: float
    growth_ok: bool


@dataclass
class _CycloneExplicitContinuation:
    """Previous-frequency state used by explicit branch reselection."""

    prev_omega: float | None = None
    prev_prev_omega: float | None = None

    def update(self, omega: float) -> None:
        self.prev_prev_omega = self.prev_omega
        self.prev_omega = float(omega)


def _prepare_explicit_scan_point(
    *,
    index: int,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    hooks: Any,
) -> _CycloneExplicitPoint:
    """Build grid, initial condition, cache, and time step for one ky point."""

    ky_val = float(ky_values[index])
    ky_index = hooks.select_ky_index(np.asarray(grid_full.ky), ky_val)
    grid = hooks.select_ky_grid(grid_full, ky_index)
    state = hooks.build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=n_laguerre,
        Nm=n_hermite,
        init_cfg=init_cfg,
    )
    cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)
    dt_i = float(dt[index]) if isinstance(dt, np.ndarray) else float(dt)
    steps_i = int(steps[index]) if isinstance(steps, np.ndarray) else int(steps)
    return _CycloneExplicitPoint(
        index=index,
        ky_value=ky_val,
        grid=grid,
        state=state,
        cache=cache,
        dt=dt_i,
        steps=steps_i,
    )


def _fit_explicit_scan_point(
    point: _CycloneExplicitPoint,
    *,
    geom: Any,
    params: Any,
    terms: Any,
    explicit_time_cfg: Any,
    diagnostic_norm: str,
    hooks: Any,
    show_progress: bool,
) -> _CycloneExplicitFit:
    """Integrate one explicit-time trace and fit its instantaneous growth."""

    t, phi_t, _g_t, _o_t = hooks.integrate_linear_explicit(
        jnp.array(point.state),
        point.grid,
        point.cache,
        params,
        geom,
        explicit_time_cfg,
        terms=terms,
        mode_method="z_index",
        show_progress=show_progress,
    )
    sel_local = hooks.mode_selection(
        ky_index=0,
        kx_index=0,
        z_index=hooks.midplane_index(point.grid),
    )
    try:
        gamma, omega, _g, _o, _t_mid = hooks.instantaneous_growth_rate_from_phi(
            phi_t,
            t,
            sel_local,
            navg_fraction=0.5,
            mode_method="z_index",
        )
        gamma, omega = hooks.normalize_growth_rate(
            gamma,
            omega,
            params,
            diagnostic_norm,
        )
        return _CycloneExplicitFit(float(gamma), float(omega), True)
    except ValueError:
        return _CycloneExplicitFit(float("nan"), float("nan"), False)


def _explicit_scan_needs_reselection(
    *,
    reference_aligned: bool,
    fit: _CycloneExplicitFit,
    continuation: _CycloneExplicitContinuation,
    index: int,
) -> bool:
    """Return whether explicit branch history suggests a Krylov reselection."""

    return (
        (reference_aligned and fit.growth_ok)
        and continuation.prev_omega is not None
        and continuation.prev_omega > 0.0
        and (
            fit.omega <= 0.0
            or ((index >= 2) and (fit.omega < 0.85 * continuation.prev_omega))
        )
    )


def _reselected_explicit_scan_fit(
    point: _CycloneExplicitPoint,
    fit: _CycloneExplicitFit,
    *,
    params: Any,
    terms: Any,
    kcfg: Any,
    diagnostic_norm: str,
    continuation: _CycloneExplicitContinuation,
    hooks: Any,
) -> _CycloneExplicitFit:
    """Apply Krylov reselection when explicit-time branch tracking fails."""

    target_omega = explicit_reselection_target(
        explicit_growth_ok=fit.growth_ok,
        prev_omega=continuation.prev_omega,
        prev_prev_omega=continuation.prev_prev_omega,
    )
    gamma_k, omega_k = krylov_reselected_frequency(
        point.state,
        point.cache,
        params,
        terms,
        kcfg,
        hooks,
        diagnostic_norm,
    )
    if not fit.growth_ok:
        return _CycloneExplicitFit(gamma_k, omega_k, True)
    assert target_omega is not None
    gamma, omega = choose_reselected_frequency(
        gamma=fit.gamma,
        omega=fit.omega,
        gamma_k=gamma_k,
        omega_k=omega_k,
        target_omega=target_omega,
    )
    return _CycloneExplicitFit(gamma, omega, True)


def _final_explicit_scan_fit(
    point: _CycloneExplicitPoint,
    fit: _CycloneExplicitFit,
    *,
    params: Any,
    terms: Any,
    kcfg: Any,
    diagnostic_norm: str,
    reference_aligned: bool,
    continuation: _CycloneExplicitContinuation,
    hooks: Any,
) -> _CycloneExplicitFit:
    """Resolve sign convention and optional branch reselection for one point."""

    if reference_aligned and continuation.prev_omega is None and fit.omega < 0.0:
        fit = _CycloneExplicitFit(fit.gamma, abs(fit.omega), fit.growth_ok)
    if _explicit_scan_needs_reselection(
        reference_aligned=reference_aligned,
        fit=fit,
        continuation=continuation,
        index=point.index,
    ) or not fit.growth_ok:
        return _reselected_explicit_scan_fit(
            point,
            fit,
            params=params,
            terms=terms,
            kcfg=kcfg,
            diagnostic_norm=diagnostic_norm,
            continuation=continuation,
            hooks=hooks,
        )
    return fit


def _append_explicit_scan_point(
    *,
    point: _CycloneExplicitPoint,
    geom: Any,
    params: Any,
    terms: Any,
    time_cfg: Any | None,
    time_base: Any,
    kcfg: Any,
    reference_aligned: bool,
    diagnostic_norm: str,
    continuation: _CycloneExplicitContinuation,
    gamma_out: np.ndarray,
    omega_out: np.ndarray,
    hooks: Any,
    show_progress: bool,
) -> None:
    """Evaluate one explicit scan point and update continuation/output arrays."""

    explicit_time_cfg = explicit_time_config_for_scan_point(
        dt_i=point.dt,
        steps_i=point.steps,
        reference_aligned=reference_aligned,
        time_cfg=time_cfg,
        time_base=time_base,
        hooks=hooks,
    )
    fit = _fit_explicit_scan_point(
        point,
        geom=geom,
        params=params,
        terms=terms,
        explicit_time_cfg=explicit_time_cfg,
        diagnostic_norm=diagnostic_norm,
        hooks=hooks,
        show_progress=show_progress,
    )
    fit = _final_explicit_scan_fit(
        point,
        fit,
        params=params,
        terms=terms,
        kcfg=kcfg,
        diagnostic_norm=diagnostic_norm,
        reference_aligned=reference_aligned,
        continuation=continuation,
        hooks=hooks,
    )
    gamma_out[point.index] = fit.gamma
    omega_out[point.index] = fit.omega
    continuation.update(fit.omega)


def run_explicit_time_cyclone_scan(
    *,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    terms: Any,
    cfg: Any,
    time_cfg: Any | None,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    krylov_cfg: Any | None,
    krylov_default: Any,
    reference_aligned: bool,
    diagnostic_norm: str,
    show_progress: bool,
    hooks: Any,
) -> Any:
    """Run the reference-aligned explicit-time branch for a Cyclone ky scan."""

    if ky_values.size == 0:
        return hooks.cyclone_scan_result(
            ky=ky_values,
            gamma=np.array([]),
            omega=np.array([]),
        )

    gamma_out = np.zeros_like(ky_values, dtype=float)
    omega_out = np.zeros_like(ky_values, dtype=float)
    continuation = _CycloneExplicitContinuation()
    kcfg = krylov_cfg or krylov_default
    time_base = time_cfg or cfg.time
    for idx, _ky_val in enumerate(ky_values):
        point = _prepare_explicit_scan_point(
            index=idx,
            ky_values=ky_values,
            grid_full=grid_full,
            geom=geom,
            params=params,
            init_cfg=init_cfg,
            n_laguerre=n_laguerre,
            n_hermite=n_hermite,
            dt=dt,
            steps=steps,
            hooks=hooks,
        )
        _append_explicit_scan_point(
            point=point,
            geom=geom,
            params=params,
            terms=terms,
            time_cfg=time_cfg,
            time_base=time_base,
            kcfg=kcfg,
            reference_aligned=reference_aligned,
            diagnostic_norm=diagnostic_norm,
            continuation=continuation,
            gamma_out=gamma_out,
            omega_out=omega_out,
            hooks=hooks,
            show_progress=show_progress,
        )
    return hooks.cyclone_scan_result(ky=ky_values, gamma=gamma_out, omega=omega_out)


__all__ = [
    "choose_reselected_frequency",
    "explicit_reselection_target",
    "explicit_time_config_for_scan_point",
    "krylov_reselected_frequency",
    "run_explicit_time_cyclone_scan",
]
