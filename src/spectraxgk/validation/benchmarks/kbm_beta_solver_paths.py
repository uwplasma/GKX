"""Solver-path helpers for fixed-ky KBM beta scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from spectraxgk.solvers.time.explicit import ExplicitTimeConfig


@dataclass(frozen=True)
class KBMBetaExplicitHooks:
    """Patchable numerical hooks used by the explicit-time KBM beta path."""

    integrate_linear_explicit_diagnostics: Callable[..., Any]
    instantaneous_growth_rate_from_phi: Callable[..., Any]
    windowed_growth_rate_from_omega_series: Callable[..., Any]
    extract_mode_time_series: Callable[..., Any]
    fit_growth_rate_auto: Callable[..., Any]
    normalize_growth_rate: Callable[..., tuple[float, float]]
    resolve_cfl_fac: Callable[..., float]


@dataclass(frozen=True)
class KBMBetaKrylovHooks:
    """Patchable numerical hooks used by the Krylov KBM beta path."""

    dominant_eigenpair: Callable[..., Any]
    use_multi_target_krylov: Callable[..., bool]
    normalize_growth_rate: Callable[..., tuple[float, float]]


@dataclass(frozen=True)
class KBMBetaKrylovResult:
    """Krylov growth result plus continuation state for the next beta point."""

    gamma: float
    omega: float
    prev_vec: Any
    prev_eig: Any
    fallback_to_time: bool = False


def fit_kbm_beta_explicit_time_sample(
    *,
    G0_jax: Any,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    dt_i: float,
    steps_i: int,
    time_cfg: Any,
    sample_stride: int | None,
    mode_method: str,
    sel: Any,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    diagnostic_norm: str,
    hooks: KBMBetaExplicitHooks,
) -> tuple[float, float]:
    """Run and fit one explicit-time KBM beta sample."""

    explicit_mode_method = mode_method if mode_method in {"z_index", "max"} else "z_index"
    explicit_time_cfg = ExplicitTimeConfig(
        dt=dt_i,
        t_max=dt_i * steps_i,
        sample_stride=max(int(sample_stride or 1), 1),
        fixed_dt=bool(time_cfg.fixed_dt) if time_cfg is not None else False,
        use_dealias_mask=bool(getattr(time_cfg, "use_dealias_mask", False))
        if time_cfg is not None
        else False,
        dt_min=float(time_cfg.dt_min) if time_cfg is not None else 1.0e-7,
        dt_max=float(time_cfg.dt_max)
        if (time_cfg is not None and time_cfg.dt_max is not None)
        else None,
        cfl=float(time_cfg.cfl) if time_cfg is not None else 0.9,
        cfl_fac=(
            hooks.resolve_cfl_fac(str(time_cfg.method), time_cfg.cfl_fac)
            if time_cfg is not None
            else float(ExplicitTimeConfig.cfl_fac)
        ),
    )
    t_arr, phi_t, gamma_t, omega_t, _diagnostics = (
        hooks.integrate_linear_explicit_diagnostics(
            G0_jax,
            grid,
            cache,
            params,
            geom,
            explicit_time_cfg,
            terms=terms,
            mode_method=explicit_mode_method,
            z_index=sel.z_index,
            jit=True,
        )
    )
    if t_arr.size > 1:
        phi_np = np.asarray(phi_t)
        t_np = np.asarray(t_arr, dtype=float)
        if mode_method in {"z_index", "max"}:
            try:
                gamma, omega, _gamma_t, _omega_t, _t_mid = (
                    hooks.instantaneous_growth_rate_from_phi(
                        phi_np,
                        t_np,
                        sel,
                        navg_fraction=0.5,
                        mode_method=mode_method,
                    )
                )
            except ValueError:
                try:
                    gamma, omega, _gamma_t, _omega_t = (
                        hooks.windowed_growth_rate_from_omega_series(
                            np.asarray(gamma_t),
                            np.asarray(omega_t),
                            sel,
                            navg_fraction=0.5,
                        )
                    )
                except ValueError:
                    signal = hooks.extract_mode_time_series(
                        phi_np, sel, method=mode_method
                    )
                    gamma, omega, _tmin, _tmax = hooks.fit_growth_rate_auto(
                        t_np,
                        signal,
                        window_method="fixed",
                        window_fraction=window_fraction,
                        min_points=min_points,
                        start_fraction=start_fraction,
                        growth_weight=growth_weight,
                        require_positive=require_positive,
                        min_amp_fraction=min_amp_fraction,
                    )
        else:
            signal = hooks.extract_mode_time_series(phi_np, sel, method=mode_method)
            gamma, omega, _tmin, _tmax = hooks.fit_growth_rate_auto(
                t_np,
                signal,
                window_method="fixed",
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
    else:
        gamma = float("nan")
        omega = float("nan")
    return hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)


def solve_kbm_beta_krylov_sample(
    *,
    beta: float,
    cfg: Any,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    solver_key: str,
    krylov_cfg_use: Any,
    use_continuation: bool,
    prev_vec: Any,
    prev_eig: Any,
    kbm_target_factors: Sequence[float] | None,
    kbm_beta_transition: float | None,
    diagnostic_norm: str,
    is_valid_growth: Callable[[float, float], bool],
    hooks: KBMBetaKrylovHooks,
) -> KBMBetaKrylovResult:
    """Run one Krylov KBM beta sample and decide whether time fallback is needed."""

    shift_val = krylov_cfg_use.shift
    shift_selection = krylov_cfg_use.shift_selection
    if use_continuation and prev_eig is not None:
        shift_val = complex(np.asarray(prev_eig))

    targets: Sequence[float] | None = kbm_target_factors if kbm_target_factors else None
    use_multi_target = hooks.use_multi_target_krylov(
        krylov_cfg_use,
        targets,
        shift=shift_val,
    )
    if use_multi_target:
        assert targets is not None
        beta_transition = (
            float(cfg.model.beta)
            if kbm_beta_transition is None
            else float(kbm_beta_transition)
        )
        eig_candidates = []
        vec_candidates = []
        for target in targets:
            eig_i, vec_i = hooks.dominant_eigenpair(
                G0_jax,
                cache,
                params,
                terms=terms,
                v_ref=None,
                select_overlap=False,
                krylov_dim=krylov_cfg_use.krylov_dim,
                restarts=krylov_cfg_use.restarts,
                omega_min_factor=krylov_cfg_use.omega_min_factor,
                omega_target_factor=float(target),
                omega_cap_factor=krylov_cfg_use.omega_cap_factor,
                omega_sign=krylov_cfg_use.omega_sign,
                method=krylov_cfg_use.method,
                power_iters=krylov_cfg_use.power_iters,
                power_dt=krylov_cfg_use.power_dt,
                shift=None,
                shift_source="target",
                shift_tol=krylov_cfg_use.shift_tol,
                shift_maxiter=krylov_cfg_use.shift_maxiter,
                shift_restart=krylov_cfg_use.shift_restart,
                shift_solve_method=krylov_cfg_use.shift_solve_method,
                shift_preconditioner=krylov_cfg_use.shift_preconditioner,
                shift_selection="targeted",
                mode_family=krylov_cfg_use.mode_family,
                fallback_method=krylov_cfg_use.fallback_method,
                fallback_real_floor=krylov_cfg_use.fallback_real_floor,
            )
            eig_candidates.append(eig_i)
            vec_candidates.append(vec_i)
        if len(eig_candidates) >= 2 and np.isfinite(beta_transition):
            idx = 1 if float(beta) >= beta_transition else 0
            eig = eig_candidates[idx]
            vec = vec_candidates[idx]
        else:
            eig_arr = np.asarray([complex(np.asarray(e)) for e in eig_candidates])
            growth = np.real(eig_arr)
            if np.all(~np.isfinite(growth)):
                eig = eig_candidates[0]
                vec = vec_candidates[0]
            else:
                idx = int(np.nanargmax(np.where(np.isfinite(growth), growth, -np.inf)))
                eig = eig_candidates[idx]
                vec = vec_candidates[idx]
    else:
        eig, vec = hooks.dominant_eigenpair(
            G0_jax,
            cache,
            params,
            terms=terms,
            v_ref=prev_vec,
            select_overlap=use_continuation,
            krylov_dim=krylov_cfg_use.krylov_dim,
            restarts=krylov_cfg_use.restarts,
            omega_min_factor=krylov_cfg_use.omega_min_factor,
            omega_target_factor=krylov_cfg_use.omega_target_factor,
            omega_cap_factor=krylov_cfg_use.omega_cap_factor,
            omega_sign=krylov_cfg_use.omega_sign,
            method=krylov_cfg_use.method,
            power_iters=krylov_cfg_use.power_iters,
            power_dt=krylov_cfg_use.power_dt,
            shift=shift_val,
            shift_source=krylov_cfg_use.shift_source,
            shift_tol=krylov_cfg_use.shift_tol,
            shift_maxiter=krylov_cfg_use.shift_maxiter,
            shift_restart=krylov_cfg_use.shift_restart,
            shift_solve_method=krylov_cfg_use.shift_solve_method,
            shift_preconditioner=krylov_cfg_use.shift_preconditioner,
            shift_selection=shift_selection,
            mode_family=krylov_cfg_use.mode_family,
            fallback_method=krylov_cfg_use.fallback_method,
            fallback_real_floor=krylov_cfg_use.fallback_real_floor,
        )
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    if krylov_cfg_use.omega_sign != 0:
        omega = float(np.sign(krylov_cfg_use.omega_sign)) * abs(omega)
    gamma, omega = hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    if solver_key == "auto" and not is_valid_growth(gamma, omega):
        return KBMBetaKrylovResult(
            gamma=gamma,
            omega=omega,
            prev_vec=prev_vec,
            prev_eig=prev_eig,
            fallback_to_time=True,
        )
    if use_continuation:
        prev_vec = vec
        prev_eig = eig
    return KBMBetaKrylovResult(
        gamma=gamma,
        omega=omega,
        prev_vec=prev_vec,
        prev_eig=prev_eig,
        fallback_to_time=False,
    )
