"""Krylov branch-following policy for Cyclone ky scans."""

from __future__ import annotations

from typing import Any

import numpy as np

from spectraxgk.validation.benchmarks.reference import CycloneScanResult
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
    v_ref = None
    prev_eig: complex | None = None
    cfg_use = krylov_cfg or krylov_default
    for idx in order:
        ky_val = float(ky_values[idx])
        ky_index = hooks.select_ky_index(np.asarray(grid_full.ky), ky_val)
        grid = hooks.select_ky_grid(grid_full, ky_index)
        G0 = hooks.build_initial_condition(
            grid,
            geom,
            ky_index=0,
            kx_index=0,
            Nl=n_laguerre,
            Nm=n_hermite,
            init_cfg=init_cfg,
        )
        cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)
        seed_ok = False
        omega_ok = False
        gamma_seed = 0.0
        omega_seed = 0.0
        if prev_eig is None:
            seed_ok, omega_ok, gamma_seed, omega_seed = seed_from_explicit_trace(
                G0,
                grid,
                cache,
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
                    grid,
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

        shift = seed_shift(
            prev_eig,
            omega_ok=omega_ok,
            seed_ok=seed_ok,
            gamma_seed=gamma_seed,
            omega_seed=omega_seed,
        )
        eig, vec = hooks.dominant_eigenpair(
            G0,
            cache,
            params,
            terms=terms,
            v_ref=v_ref,
            select_overlap=v_ref is not None,
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
        gamma = float(np.real(eig))
        omega = float(-np.imag(eig))
        if use_explicit_seed(
            gamma,
            omega,
            seed_ok=seed_ok,
            gamma_seed=gamma_seed,
            omega_seed=omega_seed,
        ):
            gamma = float(gamma_seed)
            omega = float(omega_seed)
        else:
            v_ref = vec
        prev_eig = complex(float(gamma), float(-omega))
        gamma, omega = hooks.normalize_growth_rate(
            gamma, omega, params, diagnostic_norm
        )
        gamma_out[idx] = gamma
        omega_out[idx] = omega
    return hooks.cyclone_scan_result(ky=ky_values, gamma=gamma_out, omega=omega_out)


__all__ = ["run_krylov_cyclone_scan"]
