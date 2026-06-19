"""Solver-path helpers for fixed-ky KBM beta scans."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
class KBMBetaTimeHooks:
    """Patchable numerical hooks used by the saved-time KBM beta path."""

    integrate_linear_diffrax_streaming: Callable[..., Any]
    integrate_linear_from_config: Callable[..., Any]
    integrate_linear_diagnostics: Callable[..., Any]
    integrate_linear: Callable[..., Any]
    resolve_streaming_window: Callable[..., tuple[float, float]]
    midplane_index: Callable[..., int]
    select_fit_signal_auto: Callable[..., tuple[Any, str, float, float]]
    extract_mode_only_signal: Callable[..., Any]
    select_fit_signal: Callable[..., Any]
    normalize_growth_rate: Callable[..., tuple[float, float]]


@dataclass(frozen=True)
class KBMBetaKrylovResult:
    """Krylov growth result plus continuation state for the next beta point."""

    gamma: float
    omega: float
    prev_vec: Any
    prev_eig: Any
    fallback_to_time: bool = False


_KBM_KRYLOV_FORWARD_KEYS = (
    "krylov_dim restarts omega_min_factor omega_target_factor omega_cap_factor omega_sign method "
    "power_iters power_dt shift shift_source shift_tol shift_maxiter shift_restart shift_solve_method "
    "shift_preconditioner shift_selection mode_family fallback_method fallback_real_floor"
).split()


def _dominant_kbm_eigenpair(
    hooks: KBMBetaKrylovHooks,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg_use: Any,
    **overrides: Any,
) -> tuple[Any, Any]:
    """Call the KBM Krylov solver with one shared target/shift policy."""

    kwargs = {
        "terms": terms,
        "v_ref": None,
        "select_overlap": False,
        **{name: getattr(krylov_cfg_use, name) for name in _KBM_KRYLOV_FORWARD_KEYS},
        **overrides,
    }
    return hooks.dominant_eigenpair(G0_jax, cache, params, **kwargs)


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
    diagnostic_norm: str,
    fit_policy: Any,
    hooks: KBMBetaExplicitHooks,
) -> tuple[float, float]:
    """Run and fit one explicit-time KBM beta sample."""

    explicit_mode_method = (
        mode_method if mode_method in {"z_index", "max"} else "z_index"
    )
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
        gamma, omega = _fit_explicit_growth_history(
            phi_t=phi_t,
            t_arr=t_arr,
            gamma_t=gamma_t,
            omega_t=omega_t,
            mode_method=mode_method,
            sel=sel,
            fit_policy=fit_policy,
            hooks=hooks,
        )
    else:
        gamma = float("nan")
        omega = float("nan")
    return hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)


def _fit_explicit_growth_history(
    *,
    phi_t: Any,
    t_arr: Any,
    gamma_t: Any,
    omega_t: Any,
    mode_method: str,
    sel: Any,
    fit_policy: Any,
    hooks: KBMBetaExplicitHooks,
) -> tuple[float, float]:
    """Fit the explicit-time trace using the KBM fallback ladder."""

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
            return gamma, omega
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
                return gamma, omega
            except ValueError:
                pass

    signal = hooks.extract_mode_time_series(phi_np, sel, method=mode_method)
    gamma, omega, _tmin, _tmax = hooks.fit_growth_rate_auto(
        t_np,
        signal,
        window_method="fixed",
        window_fraction=fit_policy.window_fraction,
        min_points=fit_policy.min_points,
        start_fraction=fit_policy.start_fraction,
        growth_weight=fit_policy.growth_weight,
        require_positive=fit_policy.require_positive,
        min_amp_fraction=fit_policy.min_amp_fraction,
    )
    return gamma, omega


def fit_kbm_beta_time_sample(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    cache: Any,
    params: Any,
    terms: Any,
    dt_i: float,
    steps_i: int,
    method: str,
    time_cfg: Any,
    sample_stride: int | None,
    fit_key: str,
    streaming_fit: bool,
    streaming_amp_floor: float,
    mode_only: bool,
    mode_method: str,
    sel: Any,
    tmin: Any,
    tmax: Any,
    sample_index: int,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    diagnostic_norm: str,
    density_species_index: int,
    fit_policy: Any,
    hooks: KBMBetaTimeHooks,
) -> tuple[float, float]:
    """Run and fit one saved-time or streaming KBM beta sample."""

    time_cfg_i = _sample_time_config(time_cfg, dt_i, steps_i, sample_stride)
    if time_cfg_i is not None and time_cfg_i.use_diffrax and streaming_fit:
        return _fit_streaming_time_sample(
            G0_jax=G0_jax,
            grid=grid,
            geom=geom,
            cache=cache,
            params=params,
            terms=terms,
            dt_i=dt_i,
            steps_i=steps_i,
            time_cfg_i=time_cfg_i,
            fit_key=fit_key,
            streaming_amp_floor=streaming_amp_floor,
            mode_method=mode_method,
            tmin=tmin,
            tmax=tmax,
            sample_index=sample_index,
            window_fraction=window_fraction,
            start_fraction=start_fraction,
            diagnostic_norm=diagnostic_norm,
            density_species_index=density_species_index,
            hooks=hooks,
        )

    phi_t, density_t, stride = _integrate_time_sample_series(
        G0_jax=G0_jax,
        grid=grid,
        geom=geom,
        cache=cache,
        params=params,
        terms=terms,
        dt_i=dt_i,
        steps_i=steps_i,
        method=method,
        time_cfg_i=time_cfg_i,
        sample_stride=sample_stride,
        fit_key=fit_key,
        mode_only=mode_only,
        mode_method=mode_method,
        sel=sel,
        density_species_index=density_species_index,
        hooks=hooks,
    )
    return _fit_saved_time_sample(
        phi_t=phi_t,
        density_t=density_t,
        dt_i=dt_i,
        stride=stride,
        fit_key=fit_key,
        mode_only=mode_only,
        mode_method=mode_method,
        sel=sel,
        tmin=tmin,
        tmax=tmax,
        sample_index=sample_index,
        diagnostic_norm=diagnostic_norm,
        density_species_index=density_species_index,
        params=params,
        fit_policy=fit_policy,
        hooks=hooks,
    )


def _sample_time_config(
    time_cfg: Any,
    dt_i: float,
    steps_i: int,
    sample_stride: int | None,
) -> Any | None:
    if time_cfg is None:
        return None
    cfg_i = replace(time_cfg, dt=dt_i, t_max=dt_i * steps_i)
    if sample_stride is not None:
        cfg_i = replace(cfg_i, sample_stride=sample_stride)
    return cfg_i


def _fit_streaming_time_sample(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    cache: Any,
    params: Any,
    terms: Any,
    dt_i: float,
    steps_i: int,
    time_cfg_i: Any,
    fit_key: str,
    streaming_amp_floor: float,
    mode_method: str,
    tmin: Any,
    tmax: Any,
    sample_index: int,
    window_fraction: float,
    start_fraction: float,
    diagnostic_norm: str,
    density_species_index: int,
    hooks: KBMBetaTimeHooks,
) -> tuple[float, float]:
    tmin_i, tmax_i = hooks.resolve_streaming_window(
        float(time_cfg_i.t_max),
        _indexed_float_value(tmin, sample_index),
        _indexed_float_value(tmax, sample_index),
        start_fraction,
        window_fraction,
        1.0,
    )
    _, gamma_vals, omega_vals = hooks.integrate_linear_diffrax_streaming(
        G0_jax,
        grid,
        geom,
        params,
        dt=dt_i,
        steps=steps_i,
        method=time_cfg_i.diffrax_solver,
        cache=cache,
        terms=terms,
        adaptive=time_cfg_i.diffrax_adaptive,
        rtol=time_cfg_i.diffrax_rtol,
        atol=time_cfg_i.diffrax_atol,
        max_steps=time_cfg_i.diffrax_max_steps,
        progress_bar=time_cfg_i.progress_bar,
        checkpoint=time_cfg_i.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal=fit_key,
        mode_ky_indices=[0],
        mode_kx_index=0,
        mode_z_index=hooks.midplane_index(grid),
        mode_method=mode_method,
        amp_floor=streaming_amp_floor,
        density_species_index=density_species_index if fit_key == "density" else None,
        return_state=False,
    )
    gamma = float(np.asarray(gamma_vals)[0])
    omega = float(np.asarray(omega_vals)[0])
    return hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)


def _integrate_time_sample_series(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    cache: Any,
    params: Any,
    terms: Any,
    dt_i: float,
    steps_i: int,
    method: str,
    time_cfg_i: Any | None,
    sample_stride: int | None,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    sel: Any,
    density_species_index: int,
    hooks: KBMBetaTimeHooks,
) -> tuple[Any, Any | None, int]:
    if time_cfg_i is not None and time_cfg_i.use_diffrax:
        stride = int(time_cfg_i.sample_stride)
        phi_t, density_t = _integrate_config_time_series(
            G0_jax=G0_jax,
            grid=grid,
            geom=geom,
            cache=cache,
            params=params,
            terms=terms,
            time_cfg_i=time_cfg_i,
            fit_key=fit_key,
            mode_only=mode_only,
            mode_method=mode_method,
            sel=sel,
            density_species_index=density_species_index,
            hooks=hooks,
        )
        return phi_t, density_t, stride

    stride = (
        int(time_cfg_i.sample_stride)
        if time_cfg_i is not None
        else 1 if sample_stride is None else int(sample_stride)
    )
    phi_t, density_t = _integrate_saved_time_series(
        G0_jax=G0_jax,
        grid=grid,
        geom=geom,
        params=params,
        cache=cache,
        terms=terms,
        dt_i=dt_i,
        steps_i=steps_i,
        method=method,
        stride=stride,
        fit_key=fit_key,
        density_species_index=density_species_index,
        hooks=hooks,
    )
    return phi_t, density_t, stride


def _integrate_config_time_series(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    cache: Any,
    params: Any,
    terms: Any,
    time_cfg_i: Any,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    sel: Any,
    density_species_index: int,
    hooks: KBMBetaTimeHooks,
) -> tuple[Any, Any | None]:
    save_mode_method = mode_method if mode_method in {"z_index", "max"} else "z_index"
    _, phi_t = hooks.integrate_linear_from_config(
        G0_jax,
        grid,
        geom,
        params,
        time_cfg_i,
        cache=cache,
        terms=terms,
        save_mode=sel if mode_only else None,
        mode_method=save_mode_method,
        save_field="phi+density"
        if fit_key == "auto"
        else ("density" if fit_key == "density" else "phi"),
        density_species_index=density_species_index
        if fit_key in {"density", "auto"}
        else None,
    )
    if fit_key == "auto":
        phi_values, density_values = phi_t
        return phi_values, density_values
    return phi_t, None


def _fit_saved_time_sample(
    *,
    phi_t: Any,
    density_t: Any | None,
    dt_i: float,
    stride: int,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    sel: Any,
    tmin: Any,
    tmax: Any,
    sample_index: int,
    diagnostic_norm: str,
    density_species_index: int,
    params: Any,
    fit_policy: Any,
    hooks: KBMBetaTimeHooks,
) -> tuple[float, float]:
    phi_t_np = np.asarray(phi_t)
    density_np = None if density_t is None else np.asarray(density_t)
    if fit_key == "density" and density_np is None:
        density_np = phi_t_np
    if fit_key == "auto":
        _signal, _name, gamma, omega = hooks.select_fit_signal_auto(
            np.arange(phi_t_np.shape[0]) * dt_i * stride,
            phi_t_np,
            density_np,
            sel,
            mode_method=mode_method,
            tmin=_indexed_float_value(tmin, sample_index),
            tmax=_indexed_float_value(tmax, sample_index),
            num_windows=8,
            **fit_policy.auto_kwargs(),
        )
        return hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)

    signal = _select_time_fit_signal(
        phi_t_np=phi_t_np,
        density_np=density_np,
        fit_key=fit_key,
        mode_only=mode_only,
        mode_method=mode_method,
        sel=sel,
        density_species_index=density_species_index,
        hooks=hooks,
    )
    return fit_policy.fit_signal(
        signal,
        idx=sample_index,
        dt=dt_i,
        stride=stride,
        params=params,
        diagnostic_norm=diagnostic_norm,
    )


def _select_time_fit_signal(
    *,
    phi_t_np: Any,
    density_np: Any | None,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    sel: Any,
    density_species_index: int,
    hooks: KBMBetaTimeHooks,
) -> Any:
    if (
        mode_only
        and fit_key == "density"
        and density_np is not None
        and density_np.ndim <= 3
    ):
        return hooks.extract_mode_only_signal(
            density_np,
            local_idx=0,
            species_index=density_species_index,
        )
    if mode_only and phi_t_np.ndim <= 2:
        return hooks.extract_mode_only_signal(phi_t_np, local_idx=0)
    return hooks.select_fit_signal(
        phi_t_np,
        density_np,
        sel,
        fit_signal=fit_key,
        mode_method=mode_method,
    )


def _integrate_saved_time_series(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    params: Any,
    cache: Any,
    terms: Any,
    dt_i: float,
    steps_i: int,
    method: str,
    stride: int,
    fit_key: str,
    density_species_index: int,
    hooks: KBMBetaTimeHooks,
) -> tuple[Any, Any | None]:
    if fit_key in {"density", "auto"}:
        diag_out = hooks.integrate_linear_diagnostics(
            G0_jax,
            grid,
            geom,
            params,
            dt=dt_i,
            steps=steps_i,
            method=method,
            cache=cache,
            terms=terms,
            sample_stride=stride,
            species_index=density_species_index,
        )
        return diag_out[1], diag_out[2] if len(diag_out) > 2 else None

    _, phi_t = hooks.integrate_linear(
        G0_jax,
        grid,
        geom,
        params,
        dt=dt_i,
        steps=steps_i,
        method=method,
        cache=cache,
        terms=terms,
        sample_stride=stride,
    )
    return phi_t, None


def _select_kbm_beta_eigen_candidate(
    *,
    beta: float,
    beta_transition: float,
    eig_candidates: Sequence[Any],
    vec_candidates: Sequence[Any],
) -> tuple[Any, Any]:
    """Select the KBM branch from targeted Krylov candidates."""

    if len(eig_candidates) >= 2 and np.isfinite(beta_transition):
        idx = 1 if float(beta) >= float(beta_transition) else 0
    else:
        growth = np.real(np.asarray([complex(np.asarray(e)) for e in eig_candidates]))
        idx = (
            0
            if np.all(~np.isfinite(growth))
            else int(np.nanargmax(np.where(np.isfinite(growth), growth, -np.inf)))
        )
    return eig_candidates[idx], vec_candidates[idx]


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
        target_results = [
            _dominant_kbm_eigenpair(
                hooks,
                G0_jax,
                cache,
                params,
                terms,
                krylov_cfg_use,
                omega_target_factor=float(target),
                shift=None,
                shift_source="target",
                shift_selection="targeted",
            )
            for target in targets
        ]
        eig_candidates, vec_candidates = zip(*target_results, strict=True)
        eig, vec = _select_kbm_beta_eigen_candidate(
            beta=beta,
            beta_transition=beta_transition,
            eig_candidates=eig_candidates,
            vec_candidates=vec_candidates,
        )
    else:
        eig, vec = _dominant_kbm_eigenpair(
            hooks,
            G0_jax,
            cache,
            params,
            terms,
            krylov_cfg_use,
            v_ref=prev_vec,
            select_overlap=use_continuation,
            shift=shift_val,
            shift_selection=shift_selection,
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


def _indexed_float_value(value: Any, idx: int) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return float(arr[int(idx)])
