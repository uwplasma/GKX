"""Solver-path policies for ETG linear ky scans."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    instantaneous_growth_rate_from_phi,
)
from spectraxgk.validation.benchmarks.batching import _resolve_streaming_window
from spectraxgk.validation.benchmarks.defaults import ETG_KRYLOV_DEFAULT
from spectraxgk.validation.benchmarks.fit_signals import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.scan import indexed_float_value
from spectraxgk.validation.benchmarks.solver_policy import _midplane_index
from spectraxgk.linear import integrate_linear, integrate_linear_diagnostics
from spectraxgk.solvers.linear.krylov import dominant_eigenpair
from spectraxgk.solvers.time.diffrax import integrate_linear_diffrax_streaming
from spectraxgk.solvers.time.runners import integrate_linear_from_config
from spectraxgk.validation.benchmarks.etg_linear import run_etg_linear

_PATCHABLE_NAMES = (
    "ModeSelection",
    "instantaneous_growth_rate_from_phi",
    "_resolve_streaming_window",
    "ETG_KRYLOV_DEFAULT",
    "_extract_mode_only_signal",
    "_normalize_growth_rate",
    "_select_fit_signal",
    "_select_fit_signal_auto",
    "indexed_float_value",
    "_midplane_index",
    "integrate_linear",
    "integrate_linear_diagnostics",
    "dominant_eigenpair",
    "integrate_linear_diffrax_streaming",
    "integrate_linear_from_config",
    "run_etg_linear",
)


def sync_path_hooks(source: dict[str, Any]) -> None:
    """Mirror the ETG scan owner module's patchable hooks into this module."""

    for name in _PATCHABLE_NAMES:
        if name in source:
            globals()[name] = source[name]


@dataclass(frozen=True)
class ETGTimeBatchResult:
    """Time-path data for one ETG scan batch after optional streaming handling."""

    handled: bool
    phi_t: np.ndarray | None = None
    density_t: np.ndarray | None = None
    t: np.ndarray | None = None
    stride: int = 1


def run_etg_krylov_batch(
    *,
    G0_jax: jnp.ndarray,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg: Any,
    prev_vec: jnp.ndarray | None,
    prev_eig: complex | None,
    diagnostic_norm: str,
) -> tuple[float, float, jnp.ndarray | None, complex | None]:
    """Run one ETG Krylov scan point with continuation-aware branch selection."""

    cfg_use = krylov_cfg or ETG_KRYLOV_DEFAULT
    use_cont = bool(cfg_use.continuation)
    v0_use = G0_jax
    v_ref = None
    shift_override = cfg_use.shift
    shift_selection_use = cfg_use.shift_selection
    if use_cont and prev_vec is not None and prev_vec.shape == G0_jax.shape:
        v0_use = prev_vec
        v_ref = prev_vec
        if cfg_use.method.strip().lower() == "shift_invert" and prev_eig is not None:
            if shift_override is None:
                shift_override = prev_eig
                shift_selection_use = "shift"
    select_overlap = (
        use_cont
        and v_ref is not None
        and (cfg_use.continuation_selection.strip().lower() == "overlap")
    )
    eig, vec = dominant_eigenpair(
        v0_use,
        cache,
        params,
        terms=terms,
        v_ref=v_ref,
        select_overlap=select_overlap,
        krylov_dim=cfg_use.krylov_dim,
        restarts=cfg_use.restarts,
        omega_min_factor=cfg_use.omega_min_factor,
        omega_target_factor=cfg_use.omega_target_factor,
        omega_cap_factor=cfg_use.omega_cap_factor,
        omega_sign=cfg_use.omega_sign,
        method=cfg_use.method,
        power_iters=cfg_use.power_iters,
        power_dt=cfg_use.power_dt,
        shift=shift_override,
        shift_source=cfg_use.shift_source,
        shift_tol=cfg_use.shift_tol,
        shift_maxiter=cfg_use.shift_maxiter,
        shift_restart=cfg_use.shift_restart,
        shift_solve_method=cfg_use.shift_solve_method,
        shift_preconditioner=cfg_use.shift_preconditioner,
        shift_selection=shift_selection_use,
        mode_family=cfg_use.mode_family,
        fallback_method=cfg_use.fallback_method,
        fallback_real_floor=cfg_use.fallback_real_floor,
    )
    if use_cont:
        eig_host = complex(np.asarray(eig))
        if np.isfinite(eig_host.real) and np.isfinite(eig_host.imag):
            prev_vec = vec
            prev_eig = eig_host
        else:
            prev_vec = None
            prev_eig = None
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    if cfg_use.omega_sign != 0:
        omega = float(np.sign(cfg_use.omega_sign)) * abs(omega)
    gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    return gamma, omega, prev_vec, prev_eig


def run_etg_time_batch(
    *,
    G0_jax: jnp.ndarray,
    grid: Any,
    geom: Any,
    params: Any,
    cache: Any,
    terms: Any,
    time_cfg: Any,
    dt_i: float,
    steps_i: int,
    method: str,
    sample_stride: int | None,
    fit_key: str,
    need_density: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    mode_method: str,
    mode_only: bool,
    sel: Any,
    batch_start: int,
    valid_count: int,
    ky_slice: np.ndarray,
    tmin: float | None,
    tmax: float | None,
    start_fraction: float,
    window_fraction: float,
    electron_index: int,
    diagnostic_norm: str,
    show_progress: bool,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> ETGTimeBatchResult:
    """Integrate one ETG time-path batch and append streaming-fit results if used."""

    time_cfg_i = None
    if time_cfg is not None:
        time_cfg_i = replace(time_cfg, dt=dt_i, t_max=dt_i * steps_i)
        if sample_stride is not None:
            time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)

    if time_cfg_i is not None and time_cfg_i.use_diffrax and streaming_fit:
        t_total = float(time_cfg_i.t_max)
        tmin_i, tmax_i = _resolve_streaming_window(
            t_total,
            indexed_float_value(tmin, batch_start),
            indexed_float_value(tmax, batch_start),
            start_fraction,
            window_fraction,
            1.0,
        )
        _, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
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
            mode_ky_indices=np.arange(valid_count, dtype=int),
            mode_kx_index=0,
            mode_z_index=_midplane_index(grid),
            mode_method=mode_method,
            amp_floor=streaming_amp_floor,
            density_species_index=electron_index if fit_key == "density" else None,
            return_state=False,
        )
        gamma_arr = np.asarray(gamma_vals)
        omega_arr = np.asarray(omega_vals)
        for local_idx in range(valid_count):
            gamma_i, omega_i = _normalize_growth_rate(
                float(gamma_arr[local_idx]),
                float(omega_arr[local_idx]),
                params,
                diagnostic_norm,
            )
            gammas.append(gamma_i)
            omegas.append(omega_i)
            ky_out.append(float(ky_slice[local_idx]))
        return ETGTimeBatchResult(handled=True)

    if time_cfg_i is not None:
        save_field = "phi+density" if fit_key == "auto" else ("density" if fit_key == "density" else "phi")
        save_mode = None if fit_key == "auto" else (sel if (mode_only and fit_key == "phi") else None)
        _, saved = integrate_linear_from_config(
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
            density_species_index=electron_index if need_density else None,
            show_progress=show_progress,
        )
        if fit_key == "auto":
            phi_t, density_t = saved
        else:
            phi_t = saved
            density_t = None
        stride = int(time_cfg_i.sample_stride)
    else:
        stride = 1 if sample_stride is None else int(sample_stride)
        if need_density:
            diag = integrate_linear_diagnostics(
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
                species_index=1,
                show_progress=show_progress,
            )
            phi_t = diag[1]
            density_t = diag[2] if len(diag) > 2 else None
        else:
            _, phi_out_time = integrate_linear(
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

    phi_t_np = np.asarray(phi_t)
    density_np = None if density_t is None else np.asarray(density_t)
    if fit_key == "density" and density_np is None:
        density_np = phi_t_np
    t = np.arange(phi_t_np.shape[0]) * dt_i * stride
    return ETGTimeBatchResult(
        handled=False,
        phi_t=phi_t_np,
        density_t=density_np,
        t=t,
        stride=stride,
    )


def append_etg_time_fit_results(
    *,
    result: ETGTimeBatchResult,
    ky_slice: np.ndarray,
    valid_count: int,
    batch_start: int,
    fit_key: str,
    fit_policy: Any,
    params: Any,
    diagnostic_norm: str,
    mode_method: str,
    mode_only: bool,
    mode_z_index: int,
    gx_growth: bool,
    gx_navg_fraction: float,
    auto_solver: bool,
    require_positive: bool,
    cfg: Any,
    Nl: int,
    Nm: int,
    dt_i: float,
    steps_i: int,
    method: str,
    krylov_cfg: Any,
    show_progress: bool,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> None:
    """Fit and append ETG growth/frequency values from a saved time batch."""

    if result.phi_t is None or result.t is None:
        raise ValueError("ETG time-batch result has no saved signal to fit")
    phi_t_np = result.phi_t
    density_np = result.density_t
    t = result.t

    def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
        if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
            return False
        if require_positive and gamma_val <= 0.0:
            return False
        return True

    for local_idx in range(valid_count):
        ky_val = ky_slice[local_idx]
        if fit_key == "auto":
            sel_local = ModeSelection(
                ky_index=local_idx,
                kx_index=0,
                z_index=mode_z_index,
            )
            _signal, _name, gamma, omega = _select_fit_signal_auto(
                t,
                phi_t_np,
                density_np,
                sel_local,
                mode_method=mode_method,
                tmin=indexed_float_value(fit_policy.tmin, batch_start + local_idx),
                tmax=indexed_float_value(fit_policy.tmax, batch_start + local_idx),
                window_fraction=fit_policy.window_fraction,
                min_points=fit_policy.min_points,
                start_fraction=fit_policy.start_fraction,
                growth_weight=fit_policy.growth_weight,
                require_positive=fit_policy.require_positive,
                min_amp_fraction=fit_policy.min_amp_fraction,
                max_amp_fraction=fit_policy.max_amp_fraction,
                window_method=fit_policy.window_method,
                max_fraction=fit_policy.max_fraction,
                end_fraction=fit_policy.end_fraction,
                num_windows=8,
                phase_weight=fit_policy.phase_weight,
                length_weight=fit_policy.length_weight,
                min_r2=fit_policy.min_r2,
                late_penalty=fit_policy.late_penalty,
                min_slope=fit_policy.min_slope,
                min_slope_frac=fit_policy.min_slope_frac,
                slope_var_weight=fit_policy.slope_var_weight,
            )
            gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
        else:
            if mode_only and fit_key == "phi" and phi_t_np.ndim <= 2:
                signal = _extract_mode_only_signal(phi_t_np, local_idx=local_idx)
            else:
                sel_local = ModeSelection(ky_index=local_idx, kx_index=0, z_index=mode_z_index)
                signal = _select_fit_signal(
                    phi_t_np,
                    density_np,
                    sel_local,
                    fit_signal=fit_key,
                    mode_method=mode_method,
                )
            if gx_growth and fit_key == "phi":
                sel_local = ModeSelection(ky_index=local_idx, kx_index=0, z_index=mode_z_index)
                gamma, omega, _gamma_t, _omega_t, _t_mid = instantaneous_growth_rate_from_phi(
                    phi_t_np,
                    t,
                    sel_local,
                    navg_fraction=gx_navg_fraction,
                    mode_method=mode_method,
                )
                gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
            else:
                gamma, omega = fit_policy.fit_signal(
                    signal,
                    idx=batch_start + local_idx,
                    dt=dt_i,
                    stride=result.stride,
                    params=params,
                    diagnostic_norm=diagnostic_norm,
                )
        if auto_solver and not _is_valid_growth(gamma, omega):
            res = run_etg_linear(
                ky_target=float(ky_val),
                cfg=cfg,
                Nl=Nl,
                Nm=Nm,
                dt=dt_i,
                steps=steps_i,
                method=method,
                params=params,
                solver="krylov",
                krylov_cfg=krylov_cfg,
                diagnostic_norm=diagnostic_norm,
                fit_signal="phi",
                show_progress=show_progress,
            )
            gamma = float(res.gamma)
            omega = float(res.omega)
        gammas.append(float(gamma))
        omegas.append(float(omega))
        ky_out.append(float(ky_val))
