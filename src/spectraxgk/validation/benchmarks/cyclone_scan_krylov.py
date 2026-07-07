"""Krylov branch-following policy for Cyclone ky scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spectraxgk.validation.benchmarks.defaults import CycloneScanResult
from spectraxgk.validation.benchmarks.cyclone_scan_seed import (
    reduced_seed_from_explicit_trace,
    seed_from_explicit_trace,
    seed_shift,
    use_explicit_seed,
)


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


__all__ = ["run_krylov_cyclone_scan"]
