"""Solver branches for the Cyclone ky-scan benchmark."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import ModeSelection, ModeSelectionBatch
from spectraxgk.validation.benchmarks.batching import _iter_ky_batches
from spectraxgk.validation.benchmarks.reference import CycloneScanResult
from spectraxgk.validation.benchmarks.scan import indexed_float_value


@dataclass(frozen=True)
class CycloneScanHooks:
    """Patchable numerical hooks supplied by the public Cyclone facade."""

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
    instantaneous_growth_rate_from_phi: Callable[..., tuple[float, float, Any, Any, Any]]
    dominant_eigenpair: Callable[..., tuple[Any, Any]]
    extract_mode_time_series: Callable[..., np.ndarray]
    select_fit_signal_auto: Callable[..., tuple[np.ndarray, str, float, float]]
    run_cyclone_linear: Callable[..., Any]
    normalize_growth_rate: Callable[..., tuple[float, float]]
    resolve_streaming_window: Callable[..., tuple[float, float]]
    midplane_index: Callable[..., int]
    resolve_cfl_fac: Callable[..., float]


def _empty_scan_result(ky_values: np.ndarray, hooks: CycloneScanHooks) -> CycloneScanResult:
    return hooks.cyclone_scan_result(
        ky=ky_values,
        gamma=np.array([]),
        omega=np.array([]),
    )


def _seed_from_explicit_trace(
    G0: Any,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    cfg_use: Any,
    hooks: CycloneScanHooks,
    *,
    show_progress: bool,
) -> tuple[bool, bool, float, float]:
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
        gamma_seed, omega_seed, _g, _o, _t_mid = hooks.instantaneous_growth_rate_from_phi(
            phi_seed,
            t_short,
            sel,
            navg_fraction=0.5,
            mode_method="z_index",
        )
        omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
        seed_ok = omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
    except Exception:
        seed_ok = False
        omega_ok = False
    return seed_ok, omega_ok, float(gamma_seed), float(omega_seed)


def _reduced_seed_from_explicit_trace(
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    cfg_use: Any,
    hooks: CycloneScanHooks,
    *,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    show_progress: bool,
) -> tuple[bool, bool, float, float]:
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
    return _seed_from_explicit_trace(
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


def _seed_shift(
    prev_eig: complex | None,
    *,
    omega_ok: bool,
    seed_ok: bool,
    gamma_seed: float,
    omega_seed: float,
) -> complex | None:
    if prev_eig is not None and np.isfinite(prev_eig):
        return prev_eig
    if omega_ok:
        return complex(float(gamma_seed) if seed_ok else 0.0, float(-omega_seed))
    return None


def _use_explicit_seed(
    gamma: float,
    omega: float,
    *,
    seed_ok: bool,
    gamma_seed: float,
    omega_seed: float,
) -> bool:
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
    hooks: CycloneScanHooks,
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
            seed_ok, omega_ok, gamma_seed, omega_seed = _seed_from_explicit_trace(
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
            seed_ok, omega_ok, gamma_seed, omega_seed = _reduced_seed_from_explicit_trace(
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

        shift = _seed_shift(
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
        if _use_explicit_seed(
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
        gamma, omega = hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)
        gamma_out[idx] = gamma
        omega_out[idx] = omega
    return hooks.cyclone_scan_result(ky=ky_values, gamma=gamma_out, omega=omega_out)


def _explicit_time_config_for_scan_point(
    *,
    dt_i: float,
    steps_i: int,
    reference_aligned: bool,
    time_cfg: Any | None,
    time_base: Any,
    hooks: CycloneScanHooks,
) -> Any:
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


def _explicit_reselection_target(
    *,
    explicit_growth_ok: bool,
    prev_omega: float | None,
    prev_prev_omega: float | None,
) -> float | None:
    target_omega = prev_omega if (explicit_growth_ok and prev_omega is not None) else None
    if (
        target_omega is not None
        and prev_prev_omega is not None
        and prev_omega is not None
        and prev_omega > prev_prev_omega
    ):
        target_omega = prev_omega + (prev_omega - prev_prev_omega)
    return target_omega


def _krylov_reselected_frequency(
    G0: Any,
    cache: Any,
    params: Any,
    terms: Any,
    kcfg: Any,
    hooks: CycloneScanHooks,
    diagnostic_norm: str,
) -> tuple[float, float]:
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


def _choose_reselected_frequency(
    *,
    gamma: float,
    omega: float,
    gamma_k: float,
    omega_k: float,
    target_omega: float,
) -> tuple[float, float]:
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
    hooks: CycloneScanHooks,
) -> CycloneScanResult:
    """Run the reference-aligned explicit-time branch for a Cyclone ky scan."""

    if ky_values.size == 0:
        return _empty_scan_result(ky_values, hooks)

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
        explicit_time_cfg = _explicit_time_config_for_scan_point(
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
            target_omega = _explicit_reselection_target(
                explicit_growth_ok=explicit_growth_ok,
                prev_omega=prev_omega,
                prev_prev_omega=prev_prev_omega,
            )
            gamma_k, omega_k = _krylov_reselected_frequency(
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
                gamma, omega = _choose_reselected_frequency(
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

    if use_batch:
        ky_iter = _iter_ky_batches(
            ky_values,
            ky_batch=ky_batch,
            fixed_batch_shape=fixed_batch_shape,
        )
    else:
        ky_iter = _iter_ky_batches(ky_values, ky_batch=1, fixed_batch_shape=False)

    gammas: list[float] = []
    omegas: list[float] = []
    ky_out: list[float] = []
    for batch_start, ky_slice, valid_count in ky_iter:
        sel_scan: ModeSelection | ModeSelectionBatch
        if use_batch:
            ky_indices = [
                hooks.select_ky_index(np.asarray(grid_full.ky), float(ky))
                for ky in ky_slice
            ]
            grid = hooks.select_ky_grid(grid_full, ky_indices)
            ky_local = np.arange(len(ky_indices))
            sel_scan = hooks.mode_selection_batch(
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
            sel_scan = hooks.mode_selection(
                ky_index=0, kx_index=0, z_index=hooks.midplane_index(grid)
            )
            dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
            steps_i = (
                int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)
            )

        G0_jax = hooks.build_initial_condition(
            grid,
            geom,
            ky_index=ky_local,
            kx_index=0,
            Nl=n_laguerre,
            Nm=n_hermite,
            init_cfg=init_cfg,
        )
        cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)

        time_cfg_i = None
        if time_cfg is not None:
            time_cfg_i = replace(time_cfg, dt=dt_i, t_max=dt_i * steps_i)
            if sample_stride is not None:
                time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)

        if time_cfg_i is not None and time_cfg_i.use_diffrax and streaming_fit:
            t_total = float(time_cfg_i.t_max)
            tmin_i, tmax_i = hooks.resolve_streaming_window(
                t_total,
                indexed_float_value(tmin, batch_start),
                indexed_float_value(tmax, batch_start),
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
                adaptive=False,
                rtol=time_cfg_i.diffrax_rtol,
                atol=time_cfg_i.diffrax_atol,
                max_steps=time_cfg_i.diffrax_max_steps,
                progress_bar=time_cfg_i.progress_bar,
                checkpoint=time_cfg_i.checkpoint,
                tmin=tmin_i,
                tmax=tmax_i,
                fit_signal="phi",
                show_progress=show_progress,
                mode_ky_indices=ky_local[:valid_count],
                mode_kx_index=0,
                mode_z_index=hooks.midplane_index(grid),
                mode_method=mode_method,
                amp_floor=streaming_amp_floor,
                return_state=False,
            )
            gamma_arr = np.asarray(gamma_vals)
            omega_arr = np.asarray(omega_vals)
            for local_idx in range(valid_count):
                gamma_i, omega_i = hooks.normalize_growth_rate(
                    float(gamma_arr[local_idx]),
                    float(omega_arr[local_idx]),
                    params,
                    diagnostic_norm,
                )
                gammas.append(gamma_i)
                omegas.append(omega_i)
                ky_out.append(float(ky_slice[local_idx]))
            continue

        if time_cfg_i is not None:
            save_field = (
                "phi+density"
                if fit_key == "auto"
                else ("density" if fit_key == "density" else "phi")
            )
            save_mode = None if fit_key == "auto" else (sel_scan if mode_only else None)
            _, saved = hooks.integrate_linear_from_config(
                G0_jax,
                grid,
                geom,
                params,
                time_cfg_i,
                cache=cache,
                terms=terms,
                save_mode=save_mode,
                mode_method=mode_method,
                save_field=save_field,
                density_species_index=0 if need_density else None,
            )
            if fit_key == "auto":
                phi_t, density_t = saved
                phi_t = np.asarray(phi_t)
                density_t = np.asarray(density_t)
            else:
                phi_t = np.asarray(saved)
                density_t = None
            stride = time_cfg_i.sample_stride
        else:
            stride = 1 if sample_stride is None else int(sample_stride)
            if use_jit and not need_density:
                _, phi_out_time = hooks.integrate_linear(
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
                    show_progress=show_progress,
                )
                phi_t = phi_out_time
                density_t = None
            else:
                diag = hooks.integrate_linear_diagnostics(
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
                    species_index=None,
                    record_hl_energy=False,
                )
                phi_t = np.asarray(diag[1])
                density_t = np.asarray(diag[2]) if len(diag) > 2 else None

        phi_t_np = np.asarray(phi_t)
        signal_t = phi_t_np if mode_only and phi_t_np.ndim == 2 else None
        density_np = None if density_t is None else np.asarray(density_t)
        t = np.arange(phi_t_np.shape[0]) * dt_i * stride
        for local_idx in range(valid_count):
            ky_val = ky_slice[local_idx]
            if signal_t is None:
                sel_local = hooks.mode_selection(
                    ky_index=local_idx, kx_index=0, z_index=hooks.midplane_index(grid)
                )
                if fit_key == "auto":
                    _signal, _name, gamma, omega = hooks.select_fit_signal_auto(
                        t,
                        phi_t_np,
                        density_np,
                        sel_local,
                        mode_method=mode_method,
                        tmin=indexed_float_value(tmin, batch_start + local_idx),
                        tmax=indexed_float_value(tmax, batch_start + local_idx),
                        window_fraction=window_fraction,
                        min_points=min_points,
                        start_fraction=start_fraction,
                        growth_weight=growth_weight,
                        require_positive=require_positive,
                        min_amp_fraction=min_amp_fraction,
                        max_amp_fraction=max_amp_fraction,
                        window_method=window_method,
                        max_fraction=max_fraction,
                        end_fraction=end_fraction,
                        num_windows=8,
                        phase_weight=phase_weight,
                        length_weight=length_weight,
                        min_r2=min_r2,
                        late_penalty=late_penalty,
                        min_slope=min_slope,
                        min_slope_frac=min_slope_frac,
                        slope_var_weight=slope_var_weight,
                    )
                    gamma, omega = hooks.normalize_growth_rate(
                        gamma, omega, params, diagnostic_norm
                    )
                    if auto_solver and not _valid_time_branch_growth(
                        gamma,
                        omega,
                        require_positive=require_positive,
                    ):
                        res = hooks.run_cyclone_linear(
                            ky_target=float(ky_val),
                            Nl=n_laguerre,
                            Nm=n_hermite,
                            dt=dt_i,
                            steps=steps_i,
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
                        gamma = float(res.gamma)
                        omega = float(res.omega)
                    gammas.append(gamma)
                    omegas.append(omega)
                    ky_out.append(float(ky_val))
                    continue
                signal = hooks.extract_mode_time_series(
                    phi_t_np, sel_local, method=mode_method
                )
            else:
                signal = signal_t[:, local_idx] if signal_t.ndim > 1 else signal_t
            gamma, omega = fit_policy.fit_signal(
                signal,
                idx=batch_start + local_idx,
                dt=dt_i,
                stride=stride,
                params=params,
                diagnostic_norm=diagnostic_norm,
            )
            if auto_solver and not _valid_time_branch_growth(
                gamma,
                omega,
                require_positive=require_positive,
            ):
                res = hooks.run_cyclone_linear(
                    ky_target=float(ky_val),
                    Nl=n_laguerre,
                    Nm=n_hermite,
                    dt=dt_i,
                    steps=steps_i,
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
                gamma = float(res.gamma)
                omega = float(res.omega)
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_val))
    return hooks.cyclone_scan_result(
        ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas)
    )
