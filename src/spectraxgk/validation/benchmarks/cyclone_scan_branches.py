"""Solver branches for the Cyclone ky-scan benchmark."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import ModeSelection, ModeSelectionBatch
from spectraxgk.validation.benchmarks.defaults import _iter_ky_batches
from spectraxgk.validation.benchmarks.defaults import CycloneScanResult
from spectraxgk.validation.benchmarks.defaults import indexed_float_value

@dataclass(frozen=True)
class CycloneScanHooks:
    """Patchable numerical hooks supplied by the Cyclone scan owner module."""

    cyclone_scan_result: type[CycloneScanResult]
    explicit_time_config: type
    mode_selection: type[ModeSelection]
    mode_selection_batch: type[ModeSelectionBatch]
    select_ky_index: Callable[..., int]
    select_ky_grid: Callable[..., Any]
    build_initial_condition: Callable[..., Any]
    build_linear_cache: Callable[..., Any]
    integrate_linear_explicit: Callable[..., Any]
    integrate_linear: Callable[..., Any]
    integrate_linear_diagnostics: Callable[..., Any]
    integrate_linear_from_config: Callable[..., Any]
    integrate_linear_diffrax_streaming: Callable[..., Any]
    instantaneous_growth_rate_from_phi: Callable[
        ..., tuple[float, float, Any, Any, Any]
    ]
    dominant_eigenpair: Callable[..., tuple[Any, Any]]
    extract_mode_time_series: Callable[..., np.ndarray]
    select_fit_signal_auto: Callable[..., tuple[np.ndarray, str, float, float]]
    run_cyclone_linear: Callable[..., Any]
    normalize_growth_rate: Callable[..., tuple[float, float]]
    resolve_streaming_window: Callable[..., tuple[float, float]]
    midplane_index: Callable[..., int]
    resolve_cfl_fac: Callable[..., float]


# Krylov and explicit-time branch policies are local here so Cyclone scan
# behavior has one patchable owner during validation.
def seed_from_explicit_trace(
    G0: Any,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    cfg_use: Any,
    hooks: Any,
    *,
    show_progress: bool,
) -> tuple[bool, bool, float, float]:
    """Estimate the starting branch from a short explicit trace."""

    gamma_seed = 0.0
    omega_seed = 0.0
    try:
        t_seed = min(150.0, float(cfg_use.power_dt) * 15000.0)
        explicit_time_cfg = hooks.explicit_time_config(
            dt=float(cfg_use.power_dt),
            t_max=t_seed,
            sample_stride=1,
            fixed_dt=True,
        )
        t_short, phi_seed, _g_t, _o_t = hooks.integrate_linear_explicit(
            jnp.array(G0),
            grid,
            cache,
            params,
            geom,
            explicit_time_cfg,
            terms=terms,
            mode_method="z_index",
            show_progress=show_progress,
        )
        sel = hooks.mode_selection(
            ky_index=0,
            kx_index=0,
            z_index=hooks.midplane_index(grid),
        )
        gamma_seed, omega_seed, _g, _o, _t_mid = (
            hooks.instantaneous_growth_rate_from_phi(
                phi_seed,
                t_short,
                sel,
                navg_fraction=0.5,
                mode_method="z_index",
            )
        )
        omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
        seed_ok = omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
    except Exception:
        seed_ok = False
        omega_ok = False
    return seed_ok, omega_ok, float(gamma_seed), float(omega_seed)


def reduced_seed_from_explicit_trace(
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    cfg_use: Any,
    hooks: Any,
    *,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    show_progress: bool,
) -> tuple[bool, bool, float, float]:
    """Build a smaller velocity-space seed trace when the full seed fails."""

    n_laguerre_seed = min(n_laguerre, 16)
    n_hermite_seed = min(n_hermite, 12)
    try:
        cache_seed = hooks.build_linear_cache(
            grid,
            geom,
            params,
            n_laguerre_seed,
            n_hermite_seed,
        )
        G0_seed = hooks.build_initial_condition(
            grid,
            geom,
            ky_index=0,
            kx_index=0,
            Nl=n_laguerre_seed,
            Nm=n_hermite_seed,
            init_cfg=init_cfg,
        )
    except Exception:
        return False, False, 0.0, 0.0
    return seed_from_explicit_trace(
        G0_seed,
        grid,
        cache_seed,
        params,
        geom,
        terms,
        cfg_use,
        hooks,
        show_progress=show_progress,
    )


def seed_shift(
    prev_eig: complex | None,
    *,
    omega_ok: bool,
    seed_ok: bool,
    gamma_seed: float,
    omega_seed: float,
) -> complex | None:
    """Choose the Krylov shift implied by continuation or a trace seed."""

    if prev_eig is not None and np.isfinite(prev_eig):
        return prev_eig
    if omega_ok:
        return complex(float(gamma_seed) if seed_ok else 0.0, float(-omega_seed))
    return None


def use_explicit_seed(
    gamma: float,
    omega: float,
    *,
    seed_ok: bool,
    gamma_seed: float,
    omega_seed: float,
) -> bool:
    """Return whether the explicit trace should override the eigen-solver result."""

    if not seed_ok:
        return False
    seed_strong = (gamma_seed > 0.0) and (abs(omega_seed) > 1.0e-6)
    if not seed_strong:
        return False
    omega_tol = 0.15 * max(abs(omega_seed), 1.0e-6)
    gamma_tol = 0.15 * max(abs(gamma_seed), 1.0e-6)
    return (
        not np.isfinite(gamma)
        or not np.isfinite(omega)
        or (gamma_seed > 0.0 and gamma < 0.0)
        or abs(omega - omega_seed) > omega_tol
        or abs(gamma - gamma_seed) > gamma_tol
    )

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

def _empty_scan_result(ky_values: np.ndarray, hooks: Any) -> CycloneScanResult:
    return hooks.cyclone_scan_result(
        ky=ky_values,
        gamma=np.array([]),
        omega=np.array([]),
    )


@dataclass(frozen=True)
class _CycloneKrylovPoint:
    """Prepared state for one ky point in the Cyclone Krylov scan."""

    index: int
    ky_value: float
    grid: Any
    state: Any
    cache: Any


@dataclass(frozen=True)
class _CycloneKrylovSeed:
    """Explicit-trace seed information used to stabilize branch following."""

    seed_ok: bool
    omega_ok: bool
    gamma: float
    omega: float


@dataclass
class _CycloneKrylovContinuation:
    """Mutable branch-following state carried between ky points."""

    v_ref: Any = None
    prev_eig: complex | None = None


def _prepare_krylov_scan_point(
    *,
    index: int,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    hooks: Any,
) -> _CycloneKrylovPoint:
    """Build grid, initial condition, and cache for one Cyclone scan point."""

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
    return _CycloneKrylovPoint(
        index=index,
        ky_value=ky_val,
        grid=grid,
        state=state,
        cache=cache,
    )


def _explicit_seed_for_krylov_point(
    point: _CycloneKrylovPoint,
    *,
    geom: Any,
    params: Any,
    terms: Any,
    cfg_use: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    prev_eig: complex | None,
    hooks: Any,
    show_progress: bool,
) -> _CycloneKrylovSeed:
    """Resolve the full or reduced explicit seed used by the Krylov branch."""

    seed_ok = False
    omega_ok = False
    gamma_seed = 0.0
    omega_seed = 0.0
    if prev_eig is None:
        seed_ok, omega_ok, gamma_seed, omega_seed = seed_from_explicit_trace(
            point.state,
            point.grid,
            point.cache,
            params,
            geom,
            terms,
            cfg_use,
            hooks,
            show_progress=show_progress,
        )
    if not seed_ok:
        seed_ok, omega_ok, gamma_seed, omega_seed = (
            reduced_seed_from_explicit_trace(
                point.grid,
                geom,
                params,
                terms,
                cfg_use,
                hooks,
                init_cfg=init_cfg,
                n_laguerre=n_laguerre,
                n_hermite=n_hermite,
                show_progress=show_progress,
            )
        )
    return _CycloneKrylovSeed(
        seed_ok=bool(seed_ok),
        omega_ok=bool(omega_ok),
        gamma=float(gamma_seed),
        omega=float(omega_seed),
    )


def _dominant_krylov_scan_pair(
    point: _CycloneKrylovPoint,
    *,
    params: Any,
    terms: Any,
    cfg_use: Any,
    continuation: _CycloneKrylovContinuation,
    seed: _CycloneKrylovSeed,
    hooks: Any,
) -> tuple[Any, Any]:
    """Run the dominant-eigenpair solve for one prepared Cyclone scan point."""

    shift = seed_shift(
        continuation.prev_eig,
        omega_ok=seed.omega_ok,
        seed_ok=seed.seed_ok,
        gamma_seed=seed.gamma,
        omega_seed=seed.omega,
    )
    return hooks.dominant_eigenpair(
        point.state,
        point.cache,
        params,
        terms=terms,
        v_ref=continuation.v_ref,
        select_overlap=continuation.v_ref is not None,
        krylov_dim=cfg_use.krylov_dim,
        restarts=cfg_use.restarts,
        omega_min_factor=cfg_use.omega_min_factor,
        omega_target_factor=cfg_use.omega_target_factor,
        omega_cap_factor=cfg_use.omega_cap_factor,
        omega_sign=cfg_use.omega_sign,
        method=cfg_use.method,
        power_iters=cfg_use.power_iters,
        power_dt=cfg_use.power_dt,
        shift=shift if shift is not None else cfg_use.shift,
        shift_source=cfg_use.shift_source,
        shift_tol=cfg_use.shift_tol,
        shift_maxiter=cfg_use.shift_maxiter,
        shift_restart=cfg_use.shift_restart,
        shift_solve_method=cfg_use.shift_solve_method,
        shift_preconditioner=cfg_use.shift_preconditioner,
        shift_selection=cfg_use.shift_selection,
        mode_family=cfg_use.mode_family,
        fallback_method=cfg_use.fallback_method,
        fallback_real_floor=cfg_use.fallback_real_floor,
    )


def _raw_krylov_scan_rates(
    *,
    eig: Any,
    vec: Any,
    seed: _CycloneKrylovSeed,
    continuation: _CycloneKrylovContinuation,
) -> tuple[float, float]:
    """Return raw rates and update continuation before normalization."""

    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    if use_explicit_seed(
        gamma,
        omega,
        seed_ok=seed.seed_ok,
        gamma_seed=seed.gamma,
        omega_seed=seed.omega,
    ):
        gamma = float(seed.gamma)
        omega = float(seed.omega)
    else:
        continuation.v_ref = vec
    continuation.prev_eig = complex(float(gamma), float(-omega))
    return gamma, omega


def _append_krylov_scan_point(
    *,
    point: _CycloneKrylovPoint,
    geom: Any,
    params: Any,
    terms: Any,
    cfg_use: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    diagnostic_norm: str,
    continuation: _CycloneKrylovContinuation,
    gamma_out: np.ndarray,
    omega_out: np.ndarray,
    hooks: Any,
    show_progress: bool,
) -> None:
    """Evaluate one ky point and write normalized scan outputs."""

    seed = _explicit_seed_for_krylov_point(
        point,
        geom=geom,
        params=params,
        terms=terms,
        cfg_use=cfg_use,
        init_cfg=init_cfg,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        prev_eig=continuation.prev_eig,
        hooks=hooks,
        show_progress=show_progress,
    )
    eig, vec = _dominant_krylov_scan_pair(
        point,
        params=params,
        terms=terms,
        cfg_use=cfg_use,
        continuation=continuation,
        seed=seed,
        hooks=hooks,
    )
    gamma, omega = _raw_krylov_scan_rates(
        eig=eig,
        vec=vec,
        seed=seed,
        continuation=continuation,
    )
    gamma, omega = hooks.normalize_growth_rate(
        gamma,
        omega,
        params,
        diagnostic_norm,
    )
    gamma_out[point.index] = gamma
    omega_out[point.index] = omega


def run_krylov_cyclone_scan(
    *,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    terms: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    mode_follow: bool,
    krylov_cfg: Any | None,
    krylov_default: Any,
    diagnostic_norm: str,
    show_progress: bool,
    hooks: Any,
) -> CycloneScanResult:
    """Run the Krylov branch-following path for a Cyclone ky scan."""

    if ky_values.size == 0:
        return _empty_scan_result(ky_values, hooks)

    order = np.argsort(ky_values) if mode_follow else np.arange(ky_values.size)
    gamma_out = np.zeros_like(ky_values, dtype=float)
    omega_out = np.zeros_like(ky_values, dtype=float)
    continuation = _CycloneKrylovContinuation()
    cfg_use = krylov_cfg or krylov_default
    for idx in order:
        point = _prepare_krylov_scan_point(
            index=int(idx),
            ky_values=ky_values,
            grid_full=grid_full,
            geom=geom,
            params=params,
            init_cfg=init_cfg,
            n_laguerre=n_laguerre,
            n_hermite=n_hermite,
            hooks=hooks,
        )
        _append_krylov_scan_point(
            point=point,
            geom=geom,
            params=params,
            terms=terms,
            cfg_use=cfg_use,
            init_cfg=init_cfg,
            n_laguerre=n_laguerre,
            n_hermite=n_hermite,
            diagnostic_norm=diagnostic_norm,
            continuation=continuation,
            gamma_out=gamma_out,
            omega_out=omega_out,
            hooks=hooks,
            show_progress=show_progress,
        )
    return hooks.cyclone_scan_result(ky=ky_values, gamma=gamma_out, omega=omega_out)


def _valid_time_branch_growth(
    gamma_val: float,
    omega_val: float,
    *,
    require_positive: bool,
) -> bool:
    if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
        return False
    if require_positive and gamma_val <= 0.0:
        return False
    return True


def _resolve_time_branch_growth(
    gamma: float,
    omega: float,
    *,
    ky_value: float,
    n_laguerre: int,
    n_hermite: int,
    dt: float,
    steps: int,
    method: str,
    params: Any,
    cfg: Any,
    time_cfg: Any | None,
    krylov_cfg: Any | None,
    diagnostic_norm: str,
    auto_solver: bool,
    require_positive: bool,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> tuple[float, float]:
    """Return fitted growth/frequency, using the Krylov fallback if required."""

    if not auto_solver or _valid_time_branch_growth(
        gamma, omega, require_positive=require_positive
    ):
        return gamma, omega

    result = hooks.run_cyclone_linear(
        ky_target=float(ky_value),
        Nl=n_laguerre,
        Nm=n_hermite,
        dt=dt,
        steps=steps,
        method=method,
        params=params,
        cfg=cfg,
        time_cfg=time_cfg,
        solver="krylov",
        krylov_cfg=krylov_cfg,
        diagnostic_norm=diagnostic_norm,
        fit_signal="phi",
        show_progress=show_progress,
    )
    return float(result.gamma), float(result.omega)


@dataclass(frozen=True)
class _CycloneTimeScanBatch:
    batch_start: int
    ky_slice: np.ndarray
    valid_count: int
    ky_local: np.ndarray
    grid: Any
    selection: ModeSelection | ModeSelectionBatch
    dt: float
    steps: int
    state: Any
    cache: Any


@dataclass(frozen=True)
class _CycloneHistoryFitOptions:
    n_laguerre: int
    n_hermite: int
    method: str
    params: Any
    cfg: Any
    time_cfg: Any | None
    krylov_cfg: Any | None
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    max_amp_fraction: float
    max_fraction: float
    end_fraction: float
    phase_weight: float
    length_weight: float
    min_r2: float
    late_penalty: float
    min_slope: float | None
    min_slope_frac: float
    slope_var_weight: float
    window_method: str
    mode_method: str
    mode_only: bool
    fit_key: str
    diagnostic_norm: str
    auto_solver: bool
    fit_policy: Any
    hooks: CycloneScanHooks
    show_progress: bool


@dataclass(frozen=True)
class _CycloneTimeRunOptions:
    ky_values: np.ndarray
    grid_full: Any
    geom: Any
    params: Any
    terms: Any
    cfg: Any
    time_cfg: Any | None
    init_cfg: Any
    n_laguerre: int
    n_hermite: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    sample_stride: int | None
    mode_method: str
    mode_only: bool
    fit_key: str
    need_density: bool
    diagnostic_norm: str
    use_jit: bool
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    use_batch: bool
    hooks: CycloneScanHooks
    show_progress: bool


@dataclass
class _CycloneScanOutput:
    gammas: list[float]
    omegas: list[float]
    ky: list[float]

    @classmethod
    def empty(cls) -> "_CycloneScanOutput":
        return cls(gammas=[], omegas=[], ky=[])


@dataclass(frozen=True)
class _CycloneTimeScanControls:
    run_options: _CycloneTimeRunOptions
    fit_options: _CycloneHistoryFitOptions


@dataclass(frozen=True)
class _CycloneTimeScanInputs:
    ky_values: np.ndarray
    grid_full: Any
    geom: Any
    params: Any
    terms: Any
    cfg: Any
    time_cfg: Any | None
    init_cfg: Any
    n_laguerre: int
    n_hermite: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    krylov_cfg: Any | None
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    max_amp_fraction: float
    max_fraction: float
    end_fraction: float
    phase_weight: float
    length_weight: float
    min_r2: float
    late_penalty: float
    min_slope: float | None
    min_slope_frac: float
    slope_var_weight: float
    window_method: str
    mode_method: str
    mode_only: bool
    sample_stride: int | None
    fit_key: str
    need_density: bool
    diagnostic_norm: str
    use_jit: bool
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    auto_solver: bool
    use_batch: bool
    fit_policy: Any
    hooks: CycloneScanHooks
    show_progress: bool


def _cyclone_time_scan_inputs_from_locals(values: dict[str, Any]) -> _CycloneTimeScanInputs:
    """Pack public ``run_time_cyclone_scan`` arguments once for internal routing."""

    return _CycloneTimeScanInputs(
        **{field.name: values[field.name] for field in fields(_CycloneTimeScanInputs)}
    )


def _iter_cyclone_time_scan_batches(options: _CycloneTimeRunOptions):
    if options.use_batch:
        return _iter_ky_batches(
            options.ky_values,
            ky_batch=options.ky_batch,
            fixed_batch_shape=options.fixed_batch_shape,
        )
    return _iter_ky_batches(options.ky_values, ky_batch=1, fixed_batch_shape=False)


def _prepare_cyclone_time_batch(
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    use_batch: bool,
    grid_full: Any,
    geom: Any,
    params: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    hooks: CycloneScanHooks,
) -> _CycloneTimeScanBatch:
    if use_batch:
        ky_indices = [
            hooks.select_ky_index(np.asarray(grid_full.ky), float(ky))
            for ky in ky_slice
        ]
        grid = hooks.select_ky_grid(grid_full, ky_indices)
        ky_local = np.arange(len(ky_indices))
        selection: ModeSelection | ModeSelectionBatch = hooks.mode_selection_batch(
            ky_local.astype(int), 0, hooks.midplane_index(grid)
        )
        dt_i = float(dt)
        steps_i = int(steps)
    else:
        ky_indices = [
            hooks.select_ky_index(np.asarray(grid_full.ky), float(ky_slice[0]))
        ]
        grid = hooks.select_ky_grid(grid_full, ky_indices[0])
        ky_local = np.arange(len(ky_indices))
        selection = hooks.mode_selection(
            ky_index=0, kx_index=0, z_index=hooks.midplane_index(grid)
        )
        dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)

    state = hooks.build_initial_condition(
        grid,
        geom,
        ky_index=ky_local,
        kx_index=0,
        Nl=n_laguerre,
        Nm=n_hermite,
        init_cfg=init_cfg,
    )
    cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)
    return _CycloneTimeScanBatch(
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        ky_local=ky_local,
        grid=grid,
        selection=selection,
        dt=dt_i,
        steps=steps_i,
        state=state,
        cache=cache,
    )


def _cyclone_time_config_for_batch(
    time_cfg: Any | None,
    *,
    dt: float,
    steps: int,
    sample_stride: int | None,
) -> Any | None:
    if time_cfg is None:
        return None
    time_cfg_i = replace(time_cfg, dt=dt, t_max=dt * steps)
    if sample_stride is not None:
        time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)
    return time_cfg_i


def _append_cyclone_streaming_results(
    batch: _CycloneTimeScanBatch,
    *,
    geom: Any,
    params: Any,
    terms: Any,
    time_cfg: Any,
    tmin: float | None,
    tmax: float | None,
    start_fraction: float,
    window_fraction: float,
    mode_method: str,
    streaming_amp_floor: float,
    diagnostic_norm: str,
    hooks: CycloneScanHooks,
    show_progress: bool,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> None:
    t_total = float(time_cfg.t_max)
    tmin_i, tmax_i = hooks.resolve_streaming_window(
        t_total,
        indexed_float_value(tmin, batch.batch_start),
        indexed_float_value(tmax, batch.batch_start),
        start_fraction,
        window_fraction,
        1.0,
    )
    _, gamma_vals, omega_vals = hooks.integrate_linear_diffrax_streaming(
        batch.state,
        batch.grid,
        geom,
        params,
        dt=batch.dt,
        steps=batch.steps,
        method=time_cfg.diffrax_solver,
        cache=batch.cache,
        terms=terms,
        adaptive=False,
        rtol=time_cfg.diffrax_rtol,
        atol=time_cfg.diffrax_atol,
        max_steps=time_cfg.diffrax_max_steps,
        progress_bar=time_cfg.progress_bar,
        checkpoint=time_cfg.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal="phi",
        show_progress=show_progress,
        mode_ky_indices=batch.ky_local[: batch.valid_count],
        mode_kx_index=0,
        mode_z_index=hooks.midplane_index(batch.grid),
        mode_method=mode_method,
        amp_floor=streaming_amp_floor,
        return_state=False,
    )
    gamma_arr = np.asarray(gamma_vals)
    omega_arr = np.asarray(omega_vals)
    for local_idx in range(batch.valid_count):
        gamma_i, omega_i = hooks.normalize_growth_rate(
            float(gamma_arr[local_idx]),
            float(omega_arr[local_idx]),
            params,
            diagnostic_norm,
        )
        gammas.append(gamma_i)
        omegas.append(omega_i)
        ky_out.append(float(batch.ky_slice[local_idx]))


def _integrate_cyclone_time_history(
    batch: _CycloneTimeScanBatch,
    *,
    geom: Any,
    params: Any,
    terms: Any,
    time_cfg: Any | None,
    method: str,
    mode_method: str,
    mode_only: bool,
    sample_stride: int | None,
    fit_key: str,
    need_density: bool,
    use_jit: bool,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    if time_cfg is not None:
        save_field = (
            "phi+density"
            if fit_key == "auto"
            else ("density" if fit_key == "density" else "phi")
        )
        save_mode = None if fit_key == "auto" else (batch.selection if mode_only else None)
        _, saved = hooks.integrate_linear_from_config(
            batch.state,
            batch.grid,
            geom,
            params,
            time_cfg,
            cache=batch.cache,
            terms=terms,
            save_mode=save_mode,
            mode_method=mode_method,
            save_field=save_field,
            density_species_index=0 if need_density else None,
        )
        if fit_key == "auto":
            phi_t, density_t = saved
            return np.asarray(phi_t), np.asarray(density_t), time_cfg.sample_stride
        return np.asarray(saved), None, time_cfg.sample_stride

    stride = 1 if sample_stride is None else int(sample_stride)
    if use_jit and not need_density:
        _, phi_out_time = hooks.integrate_linear(
            batch.state,
            batch.grid,
            geom,
            params,
            dt=batch.dt,
            steps=batch.steps,
            method=method,
            cache=batch.cache,
            terms=terms,
            sample_stride=stride,
            show_progress=show_progress,
        )
        return np.asarray(phi_out_time), None, stride

    diag = hooks.integrate_linear_diagnostics(
        batch.state,
        batch.grid,
        geom,
        params,
        dt=batch.dt,
        steps=batch.steps,
        method=method,
        cache=batch.cache,
        terms=terms,
        sample_stride=stride,
        species_index=None,
        record_hl_energy=False,
    )
    density_t = np.asarray(diag[2]) if len(diag) > 2 else None
    return np.asarray(diag[1]), density_t, stride


def _resolve_and_append_cyclone_fit(
    *,
    gamma: float,
    omega: float,
    ky_val: float,
    batch: _CycloneTimeScanBatch,
    options: _CycloneHistoryFitOptions,
    output: _CycloneScanOutput,
) -> None:
    gamma, omega = _resolve_time_branch_growth(
        gamma,
        omega,
        ky_value=float(ky_val),
        n_laguerre=options.n_laguerre,
        n_hermite=options.n_hermite,
        dt=batch.dt,
        steps=batch.steps,
        method=options.method,
        params=options.params,
        cfg=options.cfg,
        time_cfg=options.time_cfg,
        krylov_cfg=options.krylov_cfg,
        diagnostic_norm=options.diagnostic_norm,
        auto_solver=options.auto_solver,
        require_positive=options.require_positive,
        hooks=options.hooks,
        show_progress=options.show_progress,
    )
    output.gammas.append(gamma)
    output.omegas.append(omega)
    output.ky.append(float(ky_val))


def _auto_history_fit_for_local_mode(
    *,
    t: np.ndarray,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    batch: _CycloneTimeScanBatch,
    local_idx: int,
    options: _CycloneHistoryFitOptions,
) -> tuple[float, float]:
    sel_local = options.hooks.mode_selection(
        ky_index=local_idx,
        kx_index=0,
        z_index=options.hooks.midplane_index(batch.grid),
    )
    _signal, _name, gamma, omega = options.hooks.select_fit_signal_auto(
        t,
        phi_t,
        density_t,
        sel_local,
        mode_method=options.mode_method,
        tmin=indexed_float_value(options.tmin, batch.batch_start + local_idx),
        tmax=indexed_float_value(options.tmax, batch.batch_start + local_idx),
        window_fraction=options.window_fraction,
        min_points=options.min_points,
        start_fraction=options.start_fraction,
        growth_weight=options.growth_weight,
        require_positive=options.require_positive,
        min_amp_fraction=options.min_amp_fraction,
        max_amp_fraction=options.max_amp_fraction,
        window_method=options.window_method,
        max_fraction=options.max_fraction,
        end_fraction=options.end_fraction,
        num_windows=8,
        phase_weight=options.phase_weight,
        length_weight=options.length_weight,
        min_r2=options.min_r2,
        late_penalty=options.late_penalty,
        min_slope=options.min_slope,
        min_slope_frac=options.min_slope_frac,
        slope_var_weight=options.slope_var_weight,
    )
    return options.hooks.normalize_growth_rate(
        gamma, omega, options.params, options.diagnostic_norm
    )


def _history_signal_for_local_mode(
    *,
    phi_t: np.ndarray,
    signal_t: np.ndarray | None,
    batch: _CycloneTimeScanBatch,
    local_idx: int,
    options: _CycloneHistoryFitOptions,
) -> np.ndarray:
    if signal_t is not None:
        return signal_t[:, local_idx] if signal_t.ndim > 1 else signal_t
    sel_local = options.hooks.mode_selection(
        ky_index=local_idx,
        kx_index=0,
        z_index=options.hooks.midplane_index(batch.grid),
    )
    return options.hooks.extract_mode_time_series(
        phi_t, sel_local, method=options.mode_method
    )


def _history_fit_for_local_mode(
    *,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    signal_t: np.ndarray | None,
    t: np.ndarray,
    batch: _CycloneTimeScanBatch,
    local_idx: int,
    stride: int,
    options: _CycloneHistoryFitOptions,
) -> tuple[float, float]:
    if signal_t is None and options.fit_key == "auto":
        return _auto_history_fit_for_local_mode(
            t=t,
            phi_t=phi_t,
            density_t=density_t,
            batch=batch,
            local_idx=local_idx,
            options=options,
        )
    signal = _history_signal_for_local_mode(
        phi_t=phi_t,
        signal_t=signal_t,
        batch=batch,
        local_idx=local_idx,
        options=options,
    )
    return options.fit_policy.fit_signal(
        signal,
        idx=batch.batch_start + local_idx,
        dt=batch.dt,
        stride=stride,
        params=options.params,
        diagnostic_norm=options.diagnostic_norm,
    )


def _append_cyclone_history_fit_results(
    batch: _CycloneTimeScanBatch,
    *,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    stride: int,
    options: _CycloneHistoryFitOptions,
    output: _CycloneScanOutput,
) -> None:
    signal_t = phi_t if options.mode_only and phi_t.ndim == 2 else None
    t = np.arange(phi_t.shape[0]) * batch.dt * stride
    for local_idx in range(batch.valid_count):
        ky_val = batch.ky_slice[local_idx]
        gamma, omega = _history_fit_for_local_mode(
            phi_t=phi_t,
            density_t=density_t,
            signal_t=signal_t,
            t=t,
            batch=batch,
            local_idx=local_idx,
            stride=stride,
            options=options,
        )
        _resolve_and_append_cyclone_fit(
            gamma=gamma,
            omega=omega,
            ky_val=float(ky_val),
            batch=batch,
            options=options,
            output=output,
        )


def _append_cyclone_streaming_batch_if_requested(
    batch: _CycloneTimeScanBatch,
    *,
    time_cfg: Any | None,
    run_options: _CycloneTimeRunOptions,
    fit_options: _CycloneHistoryFitOptions,
    output: _CycloneScanOutput,
) -> bool:
    """Append Diffrax streaming fits when the batch selected that path."""

    if time_cfg is None or not time_cfg.use_diffrax or not run_options.streaming_fit:
        return False
    _append_cyclone_streaming_results(
        batch,
        geom=run_options.geom,
        params=run_options.params,
        terms=run_options.terms,
        time_cfg=time_cfg,
        tmin=fit_options.tmin,
        tmax=fit_options.tmax,
        start_fraction=fit_options.start_fraction,
        window_fraction=fit_options.window_fraction,
        mode_method=run_options.mode_method,
        streaming_amp_floor=run_options.streaming_amp_floor,
        diagnostic_norm=run_options.diagnostic_norm,
        hooks=run_options.hooks,
        show_progress=run_options.show_progress,
        gammas=output.gammas,
        omegas=output.omegas,
        ky_out=output.ky,
    )
    return True


def _append_cyclone_integrated_history_results(
    batch: _CycloneTimeScanBatch,
    *,
    time_cfg: Any | None,
    run_options: _CycloneTimeRunOptions,
    fit_options: _CycloneHistoryFitOptions,
    output: _CycloneScanOutput,
) -> None:
    """Integrate one Cyclone time batch and append fitted local modes."""

    phi_t_np, density_np, stride = _integrate_cyclone_time_history(
        batch,
        geom=run_options.geom,
        params=run_options.params,
        terms=run_options.terms,
        time_cfg=time_cfg,
        method=run_options.method,
        mode_method=run_options.mode_method,
        mode_only=run_options.mode_only,
        sample_stride=run_options.sample_stride,
        fit_key=run_options.fit_key,
        need_density=run_options.need_density,
        use_jit=run_options.use_jit,
        hooks=run_options.hooks,
        show_progress=run_options.show_progress,
    )
    _append_cyclone_history_fit_results(
        batch,
        phi_t=phi_t_np,
        density_t=density_np,
        stride=stride,
        options=fit_options,
        output=output,
    )


def _append_cyclone_time_batch_results(
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    run_options: _CycloneTimeRunOptions,
    fit_options: _CycloneHistoryFitOptions,
    output: _CycloneScanOutput,
) -> None:
    batch = _prepare_cyclone_time_batch(
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        use_batch=run_options.use_batch,
        grid_full=run_options.grid_full,
        geom=run_options.geom,
        params=run_options.params,
        init_cfg=run_options.init_cfg,
        n_laguerre=run_options.n_laguerre,
        n_hermite=run_options.n_hermite,
        dt=run_options.dt,
        steps=run_options.steps,
        hooks=run_options.hooks,
    )
    time_cfg_i = _cyclone_time_config_for_batch(
        run_options.time_cfg,
        dt=batch.dt,
        steps=batch.steps,
        sample_stride=run_options.sample_stride,
    )
    if _append_cyclone_streaming_batch_if_requested(
        batch,
        time_cfg=time_cfg_i,
        run_options=run_options,
        fit_options=fit_options,
        output=output,
    ):
        return

    _append_cyclone_integrated_history_results(
        batch,
        time_cfg=time_cfg_i,
        run_options=run_options,
        fit_options=fit_options,
        output=output,
    )


def _build_cyclone_time_run_options(
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
    method: str,
    sample_stride: int | None,
    mode_method: str,
    mode_only: bool,
    fit_key: str,
    need_density: bool,
    diagnostic_norm: str,
    use_jit: bool,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    use_batch: bool,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> _CycloneTimeRunOptions:
    """Pack Cyclone time-scan controls that are shared by every ky batch."""

    return _CycloneTimeRunOptions(
        ky_values=ky_values,
        grid_full=grid_full,
        geom=geom,
        params=params,
        terms=terms,
        cfg=cfg,
        time_cfg=time_cfg,
        init_cfg=init_cfg,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        mode_method=mode_method,
        mode_only=mode_only,
        fit_key=fit_key,
        need_density=need_density,
        diagnostic_norm=diagnostic_norm,
        use_jit=use_jit,
        ky_batch=ky_batch,
        fixed_batch_shape=fixed_batch_shape,
        streaming_fit=streaming_fit,
        streaming_amp_floor=streaming_amp_floor,
        use_batch=use_batch,
        hooks=hooks,
        show_progress=show_progress,
    )


def _build_cyclone_history_fit_options(
    *,
    n_laguerre: int,
    n_hermite: int,
    method: str,
    params: Any,
    cfg: Any,
    time_cfg: Any | None,
    krylov_cfg: Any | None,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    max_amp_fraction: float,
    max_fraction: float,
    end_fraction: float,
    phase_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    min_slope: float | None,
    min_slope_frac: float,
    slope_var_weight: float,
    window_method: str,
    mode_method: str,
    mode_only: bool,
    fit_key: str,
    diagnostic_norm: str,
    auto_solver: bool,
    fit_policy: Any,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> _CycloneHistoryFitOptions:
    """Pack growth/frequency fitting controls for Cyclone time histories."""

    return _CycloneHistoryFitOptions(
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        method=method,
        params=params,
        cfg=cfg,
        time_cfg=time_cfg,
        krylov_cfg=krylov_cfg,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        max_amp_fraction=max_amp_fraction,
        max_fraction=max_fraction,
        end_fraction=end_fraction,
        phase_weight=phase_weight,
        length_weight=length_weight,
        min_r2=min_r2,
        late_penalty=late_penalty,
        min_slope=min_slope,
        min_slope_frac=min_slope_frac,
        slope_var_weight=slope_var_weight,
        window_method=window_method,
        mode_method=mode_method,
        mode_only=mode_only,
        fit_key=fit_key,
        diagnostic_norm=diagnostic_norm,
        auto_solver=auto_solver,
        fit_policy=fit_policy,
        hooks=hooks,
        show_progress=show_progress,
    )


def _prepare_cyclone_time_scan_controls(
    inputs: _CycloneTimeScanInputs,
) -> _CycloneTimeScanControls:
    """Build all shared Cyclone time-scan controls before batch execution."""

    run_options = _build_cyclone_time_run_options(
        ky_values=inputs.ky_values,
        grid_full=inputs.grid_full,
        geom=inputs.geom,
        params=inputs.params,
        terms=inputs.terms,
        cfg=inputs.cfg,
        time_cfg=inputs.time_cfg,
        init_cfg=inputs.init_cfg,
        n_laguerre=inputs.n_laguerre,
        n_hermite=inputs.n_hermite,
        dt=inputs.dt,
        steps=inputs.steps,
        method=inputs.method,
        sample_stride=inputs.sample_stride,
        mode_method=inputs.mode_method,
        mode_only=inputs.mode_only,
        fit_key=inputs.fit_key,
        need_density=inputs.need_density,
        diagnostic_norm=inputs.diagnostic_norm,
        use_jit=inputs.use_jit,
        ky_batch=inputs.ky_batch,
        fixed_batch_shape=inputs.fixed_batch_shape,
        streaming_fit=inputs.streaming_fit,
        streaming_amp_floor=inputs.streaming_amp_floor,
        use_batch=inputs.use_batch,
        hooks=inputs.hooks,
        show_progress=inputs.show_progress,
    )
    fit_options = _build_cyclone_history_fit_options(
        n_laguerre=inputs.n_laguerre,
        n_hermite=inputs.n_hermite,
        method=inputs.method,
        params=inputs.params,
        cfg=inputs.cfg,
        time_cfg=inputs.time_cfg,
        krylov_cfg=inputs.krylov_cfg,
        tmin=inputs.tmin,
        tmax=inputs.tmax,
        window_fraction=inputs.window_fraction,
        min_points=inputs.min_points,
        start_fraction=inputs.start_fraction,
        growth_weight=inputs.growth_weight,
        require_positive=inputs.require_positive,
        min_amp_fraction=inputs.min_amp_fraction,
        max_amp_fraction=inputs.max_amp_fraction,
        max_fraction=inputs.max_fraction,
        end_fraction=inputs.end_fraction,
        phase_weight=inputs.phase_weight,
        length_weight=inputs.length_weight,
        min_r2=inputs.min_r2,
        late_penalty=inputs.late_penalty,
        min_slope=inputs.min_slope,
        min_slope_frac=inputs.min_slope_frac,
        slope_var_weight=inputs.slope_var_weight,
        window_method=inputs.window_method,
        mode_method=inputs.mode_method,
        mode_only=inputs.mode_only,
        fit_key=inputs.fit_key,
        diagnostic_norm=inputs.diagnostic_norm,
        auto_solver=inputs.auto_solver,
        fit_policy=inputs.fit_policy,
        hooks=inputs.hooks,
        show_progress=inputs.show_progress,
    )
    return _CycloneTimeScanControls(
        run_options=run_options,
        fit_options=fit_options,
    )


def _run_cyclone_time_scan_batches(
    *,
    run_options: _CycloneTimeRunOptions,
    fit_options: _CycloneHistoryFitOptions,
) -> _CycloneScanOutput:
    """Execute all Cyclone time-scan batches and collect fitted rows."""

    output = _CycloneScanOutput.empty()
    for batch_start, ky_slice, valid_count in _iter_cyclone_time_scan_batches(
        run_options
    ):
        _append_cyclone_time_batch_results(
            batch_start=batch_start,
            ky_slice=ky_slice,
            valid_count=valid_count,
            run_options=run_options,
            fit_options=fit_options,
            output=output,
        )
    return output


def _cyclone_time_scan_result(
    output: _CycloneScanOutput,
    *,
    hooks: CycloneScanHooks,
) -> CycloneScanResult:
    """Pack fitted Cyclone scan rows into the public result type."""

    return hooks.cyclone_scan_result(
        ky=np.array(output.ky),
        gamma=np.array(output.gammas),
        omega=np.array(output.omegas),
    )


def run_time_cyclone_scan(
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
    method: str,
    krylov_cfg: Any | None,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    max_amp_fraction: float,
    max_fraction: float,
    end_fraction: float,
    phase_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    min_slope: float | None,
    min_slope_frac: float,
    slope_var_weight: float,
    window_method: str,
    mode_method: str,
    mode_only: bool,
    sample_stride: int | None,
    fit_key: str,
    need_density: bool,
    diagnostic_norm: str,
    use_jit: bool,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    auto_solver: bool,
    use_batch: bool,
    fit_policy: Any,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> CycloneScanResult:
    """Run the standard Cyclone scan time-integration branches."""

    inputs = _cyclone_time_scan_inputs_from_locals(locals())
    controls = _prepare_cyclone_time_scan_controls(inputs)
    output = _run_cyclone_time_scan_batches(
        run_options=controls.run_options,
        fit_options=controls.fit_options,
    )
    return _cyclone_time_scan_result(output, hooks=hooks)


__all__ = [
    "CycloneScanHooks",
    "choose_reselected_frequency",
    "explicit_reselection_target",
    "explicit_time_config_for_scan_point",
    "krylov_reselected_frequency",
    "reduced_seed_from_explicit_trace",
    "seed_from_explicit_trace",
    "seed_shift",
    "use_explicit_seed",
    "_resolve_time_branch_growth",
    "_valid_time_branch_growth",
    "run_explicit_time_cyclone_scan",
    "run_krylov_cyclone_scan",
    "run_time_cyclone_scan",
]
