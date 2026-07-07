"""Solver-path policies for ETG linear ky scans."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    instantaneous_growth_rate_from_phi,
)
from spectraxgk.validation.benchmarks.batching import _resolve_streaming_window
from spectraxgk.validation.benchmarks.defaults import ETG_KRYLOV_DEFAULT
from spectraxgk.diagnostics.growth_rates import (
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
from spectraxgk.validation.benchmarks.etg_linear import (
    _ETG_KRYLOV_FORWARD_KEYS,
    run_etg_linear,
)

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


@dataclass(frozen=True)
class _ETGTimeFitContext:
    ky_slice: np.ndarray
    valid_count: int
    batch_start: int
    fit_key: str
    fit_policy: Any
    params: Any
    diagnostic_norm: str
    mode_method: str
    mode_only: bool
    mode_z_index: int
    reference_growth_window: bool
    reference_navg_fraction: float
    auto_solver: bool
    require_positive: bool
    cfg: Any
    Nl: int
    Nm: int
    dt_i: float
    steps_i: int
    method: str
    krylov_cfg: Any
    show_progress: bool
    gammas: list[float]
    omegas: list[float]
    ky_out: list[float]


@dataclass(frozen=True)
class _ETGTimeBatchContext:
    G0_jax: jnp.ndarray
    grid: Any
    geom: Any
    params: Any
    cache: Any
    terms: Any
    time_cfg: Any
    dt_i: float
    steps_i: int
    method: str
    sample_stride: int | None
    fit_key: str
    need_density: bool
    streaming_fit: bool
    streaming_amp_floor: float
    mode_method: str
    mode_only: bool
    sel: Any
    batch_start: int
    valid_count: int
    ky_slice: np.ndarray
    tmin: float | None
    tmax: float | None
    start_fraction: float
    window_fraction: float
    electron_index: int
    diagnostic_norm: str
    show_progress: bool
    gammas: list[float]
    omegas: list[float]
    ky_out: list[float]


def _etg_time_batch_context_from_locals(values: dict[str, Any]) -> _ETGTimeBatchContext:
    """Pack ``run_etg_time_batch`` arguments for internal routing."""

    return _ETGTimeBatchContext(
        **{field.name: values[field.name] for field in fields(_ETGTimeBatchContext)}
    )


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
    krylov_kwargs = {
        "terms": terms,
        "v_ref": v_ref,
        "select_overlap": select_overlap,
        **{name: getattr(cfg_use, name) for name in _ETG_KRYLOV_FORWARD_KEYS},
        "shift": shift_override,
        "shift_selection": shift_selection_use,
    }
    eig, vec = dominant_eigenpair(v0_use, cache, params, **krylov_kwargs)
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


def _etg_time_config_for_batch(context: _ETGTimeBatchContext) -> Any | None:
    if context.time_cfg is None:
        return None
    time_cfg_i = replace(
        context.time_cfg,
        dt=context.dt_i,
        t_max=context.dt_i * context.steps_i,
    )
    if context.sample_stride is not None:
        time_cfg_i = replace(time_cfg_i, sample_stride=context.sample_stride)
    return time_cfg_i


def _append_etg_streaming_time_results(
    context: _ETGTimeBatchContext,
    *,
    time_cfg_i: Any,
) -> None:
    t_total = float(time_cfg_i.t_max)
    tmin_i, tmax_i = _resolve_streaming_window(
        t_total,
        indexed_float_value(context.tmin, context.batch_start),
        indexed_float_value(context.tmax, context.batch_start),
        context.start_fraction,
        context.window_fraction,
        1.0,
    )
    _, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
        context.G0_jax,
        context.grid,
        context.geom,
        context.params,
        dt=context.dt_i,
        steps=context.steps_i,
        method=time_cfg_i.diffrax_solver,
        cache=context.cache,
        terms=context.terms,
        adaptive=time_cfg_i.diffrax_adaptive,
        rtol=time_cfg_i.diffrax_rtol,
        atol=time_cfg_i.diffrax_atol,
        max_steps=time_cfg_i.diffrax_max_steps,
        progress_bar=time_cfg_i.progress_bar,
        checkpoint=time_cfg_i.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal=context.fit_key,
        mode_ky_indices=np.arange(context.valid_count, dtype=int),
        mode_kx_index=0,
        mode_z_index=_midplane_index(context.grid),
        mode_method=context.mode_method,
        amp_floor=context.streaming_amp_floor,
        density_species_index=context.electron_index
        if context.fit_key == "density"
        else None,
        return_state=False,
    )
    gamma_arr = np.asarray(gamma_vals)
    omega_arr = np.asarray(omega_vals)
    for local_idx in range(context.valid_count):
        gamma_i, omega_i = _normalize_growth_rate(
            float(gamma_arr[local_idx]),
            float(omega_arr[local_idx]),
            context.params,
            context.diagnostic_norm,
        )
        context.gammas.append(gamma_i)
        context.omegas.append(omega_i)
        context.ky_out.append(float(context.ky_slice[local_idx]))


def _configured_etg_time_history(
    context: _ETGTimeBatchContext,
    *,
    time_cfg_i: Any,
) -> tuple[Any, Any | None, int]:
    save_field = (
        "phi+density"
        if context.fit_key == "auto"
        else ("density" if context.fit_key == "density" else "phi")
    )
    save_mode = None
    if context.fit_key != "auto" and context.mode_only and context.fit_key == "phi":
        save_mode = context.sel
    _, saved = integrate_linear_from_config(
        context.G0_jax,
        context.grid,
        context.geom,
        context.params,
        time_cfg_i,
        cache=context.cache,
        terms=context.terms,
        save_mode=save_mode,
        mode_method=context.mode_method,
        save_field=save_field,
        density_species_index=context.electron_index
        if context.need_density
        else None,
        show_progress=context.show_progress,
    )
    if context.fit_key == "auto":
        phi_t, density_t = saved
    else:
        phi_t = saved
        density_t = None
    return phi_t, density_t, int(time_cfg_i.sample_stride)


def _unconfigured_etg_time_history(
    context: _ETGTimeBatchContext,
) -> tuple[Any, Any | None, int]:
    stride = 1 if context.sample_stride is None else int(context.sample_stride)
    if context.need_density:
        diag = integrate_linear_diagnostics(
            context.G0_jax,
            context.grid,
            context.geom,
            context.params,
            dt=context.dt_i,
            steps=context.steps_i,
            method=context.method,
            cache=context.cache,
            terms=context.terms,
            sample_stride=stride,
            species_index=1,
            show_progress=context.show_progress,
        )
        return diag[1], diag[2] if len(diag) > 2 else None, stride
    _, phi_out_time = integrate_linear(
        context.G0_jax,
        context.grid,
        context.geom,
        context.params,
        dt=context.dt_i,
        steps=context.steps_i,
        method=context.method,
        cache=context.cache,
        terms=context.terms,
        sample_stride=stride,
        show_progress=context.show_progress,
    )
    return phi_out_time, None, stride


def _pack_etg_time_history_result(
    *,
    phi_t: Any,
    density_t: Any | None,
    dt_i: float,
    stride: int,
    fit_key: str,
) -> ETGTimeBatchResult:
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

    context = _etg_time_batch_context_from_locals(locals())
    time_cfg_i = _etg_time_config_for_batch(context)
    if time_cfg_i is not None and time_cfg_i.use_diffrax and streaming_fit:
        _append_etg_streaming_time_results(context, time_cfg_i=time_cfg_i)
        return ETGTimeBatchResult(handled=True)

    if time_cfg_i is not None:
        phi_t, density_t, stride = _configured_etg_time_history(
            context, time_cfg_i=time_cfg_i
        )
    else:
        phi_t, density_t, stride = _unconfigured_etg_time_history(context)
    return _pack_etg_time_history_result(
        phi_t=phi_t,
        density_t=density_t,
        dt_i=dt_i,
        stride=stride,
        fit_key=fit_key,
    )


def _valid_etg_growth(
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


def _etg_local_selection(local_idx: int, context: _ETGTimeFitContext) -> ModeSelection:
    return ModeSelection(
        ky_index=local_idx,
        kx_index=0,
        z_index=context.mode_z_index,
    )


def _auto_etg_time_fit(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> tuple[float, float]:
    if result.phi_t is None or result.t is None:
        raise ValueError("ETG time-batch result has no saved signal to fit")
    _signal, _name, gamma, omega = _select_fit_signal_auto(
        result.t,
        result.phi_t,
        result.density_t,
        _etg_local_selection(local_idx, context),
        mode_method=context.mode_method,
        tmin=indexed_float_value(context.fit_policy.tmin, context.batch_start + local_idx),
        tmax=indexed_float_value(context.fit_policy.tmax, context.batch_start + local_idx),
        window_fraction=context.fit_policy.window_fraction,
        min_points=context.fit_policy.min_points,
        start_fraction=context.fit_policy.start_fraction,
        growth_weight=context.fit_policy.growth_weight,
        require_positive=context.fit_policy.require_positive,
        min_amp_fraction=context.fit_policy.min_amp_fraction,
        max_amp_fraction=context.fit_policy.max_amp_fraction,
        window_method=context.fit_policy.window_method,
        max_fraction=context.fit_policy.max_fraction,
        end_fraction=context.fit_policy.end_fraction,
        num_windows=8,
        phase_weight=context.fit_policy.phase_weight,
        length_weight=context.fit_policy.length_weight,
        min_r2=context.fit_policy.min_r2,
        late_penalty=context.fit_policy.late_penalty,
        min_slope=context.fit_policy.min_slope,
        min_slope_frac=context.fit_policy.min_slope_frac,
        slope_var_weight=context.fit_policy.slope_var_weight,
    )
    return _normalize_growth_rate(
        gamma, omega, context.params, context.diagnostic_norm
    )


def _direct_etg_time_signal(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> np.ndarray:
    if result.phi_t is None:
        raise ValueError("ETG time-batch result has no saved signal to fit")
    if context.mode_only and context.fit_key == "phi" and result.phi_t.ndim <= 2:
        return _extract_mode_only_signal(result.phi_t, local_idx=local_idx)
    return _select_fit_signal(
        result.phi_t,
        result.density_t,
        _etg_local_selection(local_idx, context),
        fit_signal=context.fit_key,
        mode_method=context.mode_method,
    )


def _reference_window_etg_time_fit(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> tuple[float, float]:
    if result.phi_t is None or result.t is None:
        raise ValueError("ETG time-batch result has no saved signal to fit")
    gamma, omega, _gamma_t, _omega_t, _t_mid = instantaneous_growth_rate_from_phi(
        result.phi_t,
        result.t,
        _etg_local_selection(local_idx, context),
        navg_fraction=context.reference_navg_fraction,
        mode_method=context.mode_method,
    )
    return _normalize_growth_rate(
        gamma, omega, context.params, context.diagnostic_norm
    )


def _direct_etg_time_fit(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> tuple[float, float]:
    if context.reference_growth_window and context.fit_key == "phi":
        return _reference_window_etg_time_fit(
            result, local_idx=local_idx, context=context
        )
    signal = _direct_etg_time_signal(result, local_idx=local_idx, context=context)
    return context.fit_policy.fit_signal(
        signal,
        idx=context.batch_start + local_idx,
        dt=context.dt_i,
        stride=result.stride,
        params=context.params,
        diagnostic_norm=context.diagnostic_norm,
    )


def _resolve_etg_time_fit(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> tuple[float, float]:
    if context.fit_key == "auto":
        return _auto_etg_time_fit(result, local_idx=local_idx, context=context)
    return _direct_etg_time_fit(result, local_idx=local_idx, context=context)


def _fallback_etg_krylov_fit(
    *,
    ky_val: float,
    context: _ETGTimeFitContext,
) -> tuple[float, float]:
    res = run_etg_linear(
        ky_target=float(ky_val),
        cfg=context.cfg,
        Nl=context.Nl,
        Nm=context.Nm,
        dt=context.dt_i,
        steps=context.steps_i,
        method=context.method,
        params=context.params,
        solver="krylov",
        krylov_cfg=context.krylov_cfg,
        diagnostic_norm=context.diagnostic_norm,
        fit_signal="phi",
        show_progress=context.show_progress,
    )
    return float(res.gamma), float(res.omega)


def _append_resolved_etg_time_fit(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> None:
    ky_val = float(context.ky_slice[local_idx])
    gamma, omega = _resolve_etg_time_fit(
        result, local_idx=local_idx, context=context
    )
    if context.auto_solver and not _valid_etg_growth(
        gamma, omega, require_positive=context.require_positive
    ):
        gamma, omega = _fallback_etg_krylov_fit(ky_val=ky_val, context=context)
    context.gammas.append(float(gamma))
    context.omegas.append(float(omega))
    context.ky_out.append(ky_val)


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
    reference_growth_window: bool,
    reference_navg_fraction: float,
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
    context = _ETGTimeFitContext(
        ky_slice=ky_slice,
        valid_count=valid_count,
        batch_start=batch_start,
        fit_key=fit_key,
        fit_policy=fit_policy,
        params=params,
        diagnostic_norm=diagnostic_norm,
        mode_method=mode_method,
        mode_only=mode_only,
        mode_z_index=mode_z_index,
        reference_growth_window=reference_growth_window,
        reference_navg_fraction=reference_navg_fraction,
        auto_solver=auto_solver,
        require_positive=require_positive,
        cfg=cfg,
        Nl=Nl,
        Nm=Nm,
        dt_i=dt_i,
        steps_i=steps_i,
        method=method,
        krylov_cfg=krylov_cfg,
        show_progress=show_progress,
        gammas=gammas,
        omegas=omegas,
        ky_out=ky_out,
    )
    for local_idx in range(valid_count):
        _append_resolved_etg_time_fit(result, local_idx=local_idx, context=context)
