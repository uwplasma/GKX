"""Reference-aligned explicit-time policies for Cyclone ky scans."""

from __future__ import annotations

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
    prev_omega: float | None = None
    prev_prev_omega: float | None = None
    kcfg = krylov_cfg or krylov_default
    time_base = time_cfg or cfg.time
    for idx, ky_val in enumerate(ky_values):
        ky_index = hooks.select_ky_index(np.asarray(grid_full.ky), float(ky_val))
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
        dt_i = float(dt[idx]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[idx]) if isinstance(steps, np.ndarray) else int(steps)
        explicit_time_cfg = explicit_time_config_for_scan_point(
            dt_i=dt_i,
            steps_i=steps_i,
            reference_aligned=reference_aligned,
            time_cfg=time_cfg,
            time_base=time_base,
            hooks=hooks,
        )
        t, phi_t, _g_t, _o_t = hooks.integrate_linear_explicit(
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
        sel_local = hooks.mode_selection(
            ky_index=0,
            kx_index=0,
            z_index=hooks.midplane_index(grid),
        )
        explicit_growth_ok = True
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
        except ValueError:
            explicit_growth_ok = False
            gamma = float("nan")
            omega = float("nan")
        if reference_aligned and prev_omega is None and omega < 0.0:
            omega = abs(omega)
        need_reselect = (
            (reference_aligned and explicit_growth_ok)
            and prev_omega is not None
            and prev_omega > 0.0
            and (omega <= 0.0 or ((idx >= 2) and (omega < 0.85 * prev_omega)))
        )
        if need_reselect or not explicit_growth_ok:
            target_omega = explicit_reselection_target(
                explicit_growth_ok=explicit_growth_ok,
                prev_omega=prev_omega,
                prev_prev_omega=prev_prev_omega,
            )
            gamma_k, omega_k = krylov_reselected_frequency(
                G0,
                cache,
                params,
                terms,
                kcfg,
                hooks,
                diagnostic_norm,
            )
            if not explicit_growth_ok:
                gamma, omega = gamma_k, omega_k
            else:
                assert target_omega is not None
                gamma, omega = choose_reselected_frequency(
                    gamma=gamma,
                    omega=omega,
                    gamma_k=gamma_k,
                    omega_k=omega_k,
                    target_omega=target_omega,
                )
        gamma_out[idx] = gamma
        omega_out[idx] = omega
        prev_prev_omega = prev_omega
        prev_omega = float(omega)
    return hooks.cyclone_scan_result(ky=ky_values, gamma=gamma_out, omega=omega_out)


# Historical private names remain available through the old import surface.
_explicit_time_config_for_scan_point = explicit_time_config_for_scan_point
_explicit_reselection_target = explicit_reselection_target
_krylov_reselected_frequency = krylov_reselected_frequency
_choose_reselected_frequency = choose_reselected_frequency

__all__ = [
    "_choose_reselected_frequency",
    "_explicit_reselection_target",
    "_explicit_time_config_for_scan_point",
    "_krylov_reselected_frequency",
    "choose_reselected_frequency",
    "explicit_reselection_target",
    "explicit_time_config_for_scan_point",
    "krylov_reselected_frequency",
    "run_explicit_time_cyclone_scan",
]
