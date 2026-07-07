"""Solver-path policies for the KBM single-ky linear benchmark."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from spectraxgk.diagnostics.analysis import (
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    instantaneous_growth_rate_from_phi,
    windowed_growth_rate_from_omega_series,
)
from spectraxgk.validation.benchmarks.defaults import KBM_KRYLOV_DEFAULT
from spectraxgk.diagnostics.growth_rates import _normalize_growth_rate
from spectraxgk.validation.benchmarks.reference import LinearRunResult
from spectraxgk.validation.benchmarks.scan import scan_window_valid
from spectraxgk.validation.benchmarks.defaults import _kbm_use_multi_target_krylov
from spectraxgk.config import resolve_cfl_fac
from spectraxgk.solvers.time.explicit import (
    ExplicitTimeConfig,
    integrate_linear_explicit_diagnostics,
)
from spectraxgk.operators.linear.params import linear_terms_to_term_config
from spectraxgk.solvers.linear.krylov import dominant_eigenpair
from spectraxgk.terms.assembly import compute_fields_cached

_PATCHABLE_NAMES = (
    "extract_mode_time_series",
    "fit_growth_rate",
    "fit_growth_rate_auto",
    "instantaneous_growth_rate_from_phi",
    "windowed_growth_rate_from_omega_series",
    "KBM_KRYLOV_DEFAULT",
    "_normalize_growth_rate",
    "LinearRunResult",
    "scan_window_valid",
    "_kbm_use_multi_target_krylov",
    "resolve_cfl_fac",
    "ExplicitTimeConfig",
    "integrate_linear_explicit_diagnostics",
    "linear_terms_to_term_config",
    "dominant_eigenpair",
    "compute_fields_cached",
)


def sync_path_hooks(source: dict[str, Any]) -> None:
    """Mirror the KBM linear owner module's patchable hooks into this module."""

    for name in _PATCHABLE_NAMES:
        if name in source:
            globals()[name] = source[name]


def fit_kbm_window(
    signal: np.ndarray,
    t_arr: np.ndarray,
    *,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> tuple[float, float]:
    """Fit a KBM time trace with the runner's automatic-window fallback policy."""

    use_auto = auto_window and tmin is None and tmax is None
    if not use_auto and not scan_window_valid(t_arr, tmin, tmax):
        use_auto = True
    auto_fit_kwargs: dict[str, Any] = {
        "window_fraction": window_fraction,
        "min_points": min_points,
        "start_fraction": start_fraction,
        "growth_weight": growth_weight,
        "require_positive": require_positive,
        "min_amp_fraction": min_amp_fraction,
    }
    if use_auto:
        gamma_val, omega_val, _tmin, _tmax = fit_growth_rate_auto(
            t_arr, signal, **auto_fit_kwargs
        )
    else:
        try:
            gamma_val, omega_val = fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)
        except ValueError:
            gamma_val, omega_val, _tmin, _tmax = fit_growth_rate_auto(
                t_arr, signal, **auto_fit_kwargs
            )
    return gamma_val, omega_val


def _build_kbm_explicit_time_config(
    *,
    dt: float,
    steps: int,
    time_cfg: Any,
    sample_stride: int | None,
) -> ExplicitTimeConfig:
    """Build the explicit-time policy used by reference-aligned KBM runs."""

    return ExplicitTimeConfig(
        dt=dt,
        t_max=dt * steps,
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
            resolve_cfl_fac(str(time_cfg.method), time_cfg.cfl_fac)
            if time_cfg is not None
            else float(ExplicitTimeConfig.cfl_fac)
        ),
    )


def _fit_kbm_projected_rates(
    *,
    phi_t_np: np.ndarray,
    gamma_t: Any,
    omega_t: Any,
    t_out: np.ndarray,
    sel: Any,
    mode_method: str,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> tuple[float, float]:
    """Fit KBM rates from instantaneous diagnostics with robust fallbacks."""

    try:
        gamma, omega, _g_t, _o_t, _t_mid = instantaneous_growth_rate_from_phi(
            phi_t_np,
            t_out,
            sel,
            navg_fraction=0.5,
            mode_method=mode_method,
        )
        return gamma, omega
    except ValueError:
        try:
            gamma, omega, _g_t, _o_t = windowed_growth_rate_from_omega_series(
                np.asarray(gamma_t),
                np.asarray(omega_t),
                sel,
                navg_fraction=0.5,
            )
            return gamma, omega
        except ValueError:
            signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
            return fit_kbm_window(
                signal,
                t_out,
                auto_window=auto_window,
                tmin=tmin,
                tmax=tmax,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )


def _fit_kbm_trace_rates(
    *,
    phi_t_np: np.ndarray,
    t_out: np.ndarray,
    sel: Any,
    mode_method: str,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> tuple[float, float]:
    """Fit KBM rates directly from the selected complex field trace."""

    signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
    if auto_window and tmin is None and tmax is None:
        gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
            t_out,
            signal,
            window_method="fixed",
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
        )
        return gamma, omega
    return fit_kbm_window(
        signal,
        t_out,
        auto_window=auto_window,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
    )


def _fit_kbm_explicit_rates(
    *,
    phi_t_np: np.ndarray,
    gamma_t: Any,
    omega_t: Any,
    t_out: np.ndarray,
    sel: Any,
    mode_method: str,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> tuple[float, float]:
    """Extract the explicit-time KBM growth rate and frequency."""

    if t_out.size <= 1:
        return float("nan"), float("nan")
    if mode_method in {"z_index", "max"}:
        return _fit_kbm_projected_rates(
            phi_t_np=phi_t_np,
            gamma_t=gamma_t,
            omega_t=omega_t,
            t_out=t_out,
            sel=sel,
            mode_method=mode_method,
            auto_window=auto_window,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
        )
    return _fit_kbm_trace_rates(
        phi_t_np=phi_t_np,
        t_out=t_out,
        sel=sel,
        mode_method=mode_method,
        auto_window=auto_window,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
    )


def _dominant_kbm_krylov_eigenpair(
    *,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg: Any,
    omega_target_factor: float,
    shift: Any,
    shift_source: str,
    shift_selection: str,
) -> tuple[Any, Any]:
    """Call the KBM Krylov eigensolver with an explicit target policy."""

    return dominant_eigenpair(
        G0_jax,
        cache,
        params,
        terms=terms,
        v_ref=None,
        select_overlap=False,
        krylov_dim=krylov_cfg.krylov_dim,
        restarts=krylov_cfg.restarts,
        omega_min_factor=krylov_cfg.omega_min_factor,
        omega_target_factor=omega_target_factor,
        omega_cap_factor=krylov_cfg.omega_cap_factor,
        omega_sign=krylov_cfg.omega_sign,
        method=krylov_cfg.method,
        power_iters=krylov_cfg.power_iters,
        power_dt=krylov_cfg.power_dt,
        shift=shift,
        shift_source=shift_source,
        shift_tol=krylov_cfg.shift_tol,
        shift_maxiter=krylov_cfg.shift_maxiter,
        shift_restart=krylov_cfg.shift_restart,
        shift_solve_method=krylov_cfg.shift_solve_method,
        shift_preconditioner=krylov_cfg.shift_preconditioner,
        shift_selection=shift_selection,
        mode_family=krylov_cfg.mode_family,
        fallback_method=krylov_cfg.fallback_method,
        fallback_real_floor=krylov_cfg.fallback_real_floor,
    )


def _collect_kbm_target_candidates(
    *,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg: Any,
    targets: Sequence[float],
) -> tuple[list[Any], list[Any]]:
    """Evaluate the KBM Krylov solve at each target frequency factor."""

    eig_candidates = []
    vec_candidates = []
    for target in targets:
        eig_i, vec_i = _dominant_kbm_krylov_eigenpair(
            G0_jax=G0_jax,
            cache=cache,
            params=params,
            terms=terms,
            krylov_cfg=krylov_cfg,
            omega_target_factor=float(target),
            shift=None,
            shift_source="target",
            shift_selection="targeted",
        )
        eig_candidates.append(eig_i)
        vec_candidates.append(vec_i)
    return eig_candidates, vec_candidates


def _select_kbm_target_candidate(
    *,
    eig_candidates: Sequence[Any],
    beta_use: float,
    beta_transition: float,
) -> int:
    """Choose the KBM branch candidate from beta-continuity or max growth."""

    if len(eig_candidates) >= 2 and np.isfinite(beta_transition):
        return 1 if beta_use >= beta_transition else 0
    eig_arr = np.asarray([complex(np.asarray(e)) for e in eig_candidates])
    growth = np.real(eig_arr)
    finite_growth = np.where(np.isfinite(growth), growth, -np.inf)
    return 0 if np.all(~np.isfinite(growth)) else int(np.nanargmax(finite_growth))


def _run_kbm_multi_target_krylov(
    *,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    beta_use: float,
    cfg_use: Any,
    krylov_cfg: Any,
    targets: Sequence[float],
    kbm_beta_transition: float | None,
) -> tuple[Any, Any]:
    """Run the KBM multi-target branch-continuity Krylov policy."""

    beta_transition = (
        float(cfg_use.model.beta)
        if kbm_beta_transition is None
        else float(kbm_beta_transition)
    )
    eig_candidates, vec_candidates = _collect_kbm_target_candidates(
        G0_jax=G0_jax,
        cache=cache,
        params=params,
        terms=terms,
        krylov_cfg=krylov_cfg,
        targets=targets,
    )
    idx = _select_kbm_target_candidate(
        eig_candidates=eig_candidates,
        beta_use=beta_use,
        beta_transition=beta_transition,
    )
    return eig_candidates[idx], vec_candidates[idx]


def run_kbm_explicit_time_path(
    *,
    G0_jax: Any,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    sel: Any,
    ky_target: float,
    dt: float,
    steps: int,
    time_cfg: Any,
    sample_stride: int | None,
    mode_method: str,
    diagnostic_norm: str,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> LinearRunResult:
    """Run KBM's reference-aligned explicit diagnostics branch."""

    explicit_mode_method = mode_method if mode_method in {"z_index", "max"} else "z_index"
    explicit_time_cfg = _build_kbm_explicit_time_config(
        dt=dt,
        steps=steps,
        time_cfg=time_cfg,
        sample_stride=sample_stride,
    )
    t_arr, phi_t, gamma_t, omega_t, _diag = integrate_linear_explicit_diagnostics(
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
    t_out = np.asarray(t_arr, dtype=float)
    phi_t_np = np.asarray(phi_t)
    gamma, omega = _fit_kbm_explicit_rates(
        phi_t_np=phi_t_np,
        gamma_t=gamma_t,
        omega_t=omega_t,
        t_out=t_out,
        sel=sel,
        mode_method=mode_method,
        auto_window=auto_window,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
    )
    gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    return LinearRunResult(
        t=t_out,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(ky_target),
        selection=sel,
        gamma_t=np.asarray(gamma_t),
        omega_t=np.asarray(omega_t),
    )


def _run_kbm_single_target_krylov(
    *,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg: Any,
    shift_val: Any,
) -> tuple[Any, Any]:
    """Run the KBM Krylov policy with the configured single target."""

    return _dominant_kbm_krylov_eigenpair(
        G0_jax=G0_jax,
        cache=cache,
        params=params,
        terms=terms,
        krylov_cfg=krylov_cfg,
        omega_target_factor=krylov_cfg.omega_target_factor,
        shift=shift_val,
        shift_source=krylov_cfg.shift_source,
        shift_selection=krylov_cfg.shift_selection,
    )


def _kbm_krylov_result(
    *,
    eig: Any,
    vec: Any,
    cache: Any,
    params: Any,
    terms: Any,
    sel: Any,
    ky_target: float,
    diagnostic_norm: str,
    krylov_cfg: Any,
) -> LinearRunResult:
    """Package a KBM Krylov eigenpair as a linear benchmark result."""

    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    if krylov_cfg.omega_sign != 0:
        omega = float(np.sign(krylov_cfg.omega_sign)) * abs(omega)
    gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    term_cfg = linear_terms_to_term_config(terms)
    phi = compute_fields_cached(vec, cache, params, terms=term_cfg).phi
    return LinearRunResult(
        t=np.array([0.0], dtype=float),
        phi_t=np.asarray(phi)[None, ...],
        gamma=gamma,
        omega=omega,
        ky=float(ky_target),
        selection=sel,
    )


def run_kbm_krylov_path(
    *,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    sel: Any,
    ky_target: float,
    beta_use: float,
    cfg_use: Any,
    krylov_cfg: Any,
    kbm_target_factors: Sequence[float] | None,
    kbm_beta_transition: float | None,
    diagnostic_norm: str,
) -> LinearRunResult:
    """Run KBM's single-target or beta-transition multi-target Krylov branch."""

    krylov_cfg_use = krylov_cfg or KBM_KRYLOV_DEFAULT
    shift_val = krylov_cfg_use.shift
    targets: Sequence[float] | None = kbm_target_factors if kbm_target_factors else None
    use_multi_target = _kbm_use_multi_target_krylov(
        krylov_cfg_use,
        targets,
        shift=shift_val,
    )
    if use_multi_target:
        assert targets is not None
        eig, vec = _run_kbm_multi_target_krylov(
            G0_jax=G0_jax,
            cache=cache,
            params=params,
            terms=terms,
            beta_use=beta_use,
            cfg_use=cfg_use,
            krylov_cfg=krylov_cfg_use,
            targets=targets,
            kbm_beta_transition=kbm_beta_transition,
        )
    else:
        eig, vec = _run_kbm_single_target_krylov(
            G0_jax=G0_jax,
            cache=cache,
            params=params,
            terms=terms,
            krylov_cfg=krylov_cfg_use,
            shift_val=shift_val,
        )
    return _kbm_krylov_result(
        eig=eig,
        vec=vec,
        cache=cache,
        params=params,
        terms=terms,
        sel=sel,
        ky_target=ky_target,
        diagnostic_norm=diagnostic_norm,
        krylov_cfg=krylov_cfg_use,
    )
