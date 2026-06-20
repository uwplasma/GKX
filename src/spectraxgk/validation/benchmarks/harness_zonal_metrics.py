"""Zonal-response metric extraction for benchmark validation traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectraxgk.validation.benchmarks.harness_timeseries import (
    _analytic_signal,
    _explicit_time_window,
    _leading_window,
    _tail_window,
)
from spectraxgk.validation.gates import ZonalFlowResponseMetrics


@dataclass(frozen=True)
class _ZonalWindowState:
    t_arr: np.ndarray
    policy: str
    damping_mode: str
    frequency_mode: str
    initial_level: float
    tail_tmin: float
    tail_tmax: float
    response_norm: np.ndarray
    residual_norm: float
    residual_std_norm: float
    response_rms: float
    detrended_norm: np.ndarray
    fit_mask: np.ndarray
    fit_tmin: float
    fit_tmax: float


@dataclass(frozen=True)
class _ZonalPeakFitState:
    max_peak_idx: np.ndarray
    min_peak_idx: np.ndarray
    peak_idx: np.ndarray
    gam_damping: float
    peak_fit_count: int
    gam_frequency: float


def _coerce_zonal_trace(t: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_arr = np.asarray(t, dtype=float)
    resp = np.asarray(response, dtype=float)
    if t_arr.ndim != 1 or resp.ndim != 1 or t_arr.size != resp.size:
        raise ValueError("t and response must be one-dimensional arrays of equal length")
    if t_arr.size < 4:
        raise ValueError("zonal-flow response requires at least four samples")

    finite = np.isfinite(t_arr) & np.isfinite(resp)
    t_arr = t_arr[finite]
    resp = resp[finite]
    if t_arr.size < 4:
        raise ValueError("zonal-flow response requires at least four finite samples")
    return t_arr, resp


def _normalized_zonal_options(
    *,
    initial_policy: str,
    peak_fit_max_peaks: int | None,
    damping_fit_mode: str,
    frequency_fit_mode: str,
    hilbert_trim_fraction: float,
) -> tuple[str, str, str]:
    policy = str(initial_policy).strip().lower().replace("-", "_")
    if policy not in {"window_abs_mean", "first_abs"}:
        raise ValueError("initial_policy must be one of {'window_abs_mean', 'first_abs'}")
    if peak_fit_max_peaks is not None and int(peak_fit_max_peaks) <= 0:
        raise ValueError("peak_fit_max_peaks must be > 0 when provided")
    damping_mode = str(damping_fit_mode).strip().lower().replace("-", "_")
    if damping_mode not in {"combined_envelope", "branchwise_extrema"}:
        raise ValueError("damping_fit_mode must be one of {'combined_envelope', 'branchwise_extrema'}")
    frequency_mode = str(frequency_fit_mode).strip().lower().replace("-", "_")
    if frequency_mode not in {"peak_spacing", "hilbert_phase"}:
        raise ValueError("frequency_fit_mode must be one of {'peak_spacing', 'hilbert_phase'}")
    if not 0.0 <= float(hilbert_trim_fraction) < 0.5:
        raise ValueError("hilbert_trim_fraction must be in [0, 0.5)")
    return policy, damping_mode, frequency_mode


def _initial_response_level(
    *,
    t_arr: np.ndarray,
    resp: np.ndarray,
    initial_fraction: float,
    policy: str,
    initial_level_override: float | None,
) -> float:
    if initial_level_override is not None:
        initial_level = float(initial_level_override)
    elif policy == "first_abs":
        initial_level = float(abs(resp[0]))
    else:
        lead_mask, _lead_tmin, _lead_tmax = _leading_window(t_arr, float(initial_fraction))
        initial_vals = resp[lead_mask]
        if initial_vals.size == 0:
            raise ValueError("response windows must be non-empty")
        initial_level = float(np.mean(np.abs(initial_vals)))
    if initial_level <= 0.0 or not np.isfinite(initial_level):
        raise ValueError("initial response level must be finite and positive")
    return initial_level


def _residual_window_metrics(
    *,
    t_arr: np.ndarray,
    resp: np.ndarray,
    tail_fraction: float,
    initial_level: float,
) -> tuple[np.ndarray, float, float, np.ndarray, float, float, float]:
    tail_mask, tail_tmin, tail_tmax = _tail_window(t_arr, float(tail_fraction))
    tail_vals = resp[tail_mask]
    if tail_vals.size == 0:
        raise ValueError("response windows must be non-empty")

    residual = float(np.mean(tail_vals))
    residual_std = float(np.std(tail_vals))
    response_norm = resp / initial_level
    residual_norm = residual / initial_level
    residual_std_norm = residual_std / initial_level
    response_rms = float(np.sqrt(np.mean(np.square(response_norm[tail_mask]))))
    return (
        tail_mask,
        float(tail_tmin if tail_tmin is not None else t_arr[0]),
        float(tail_tmax if tail_tmax is not None else t_arr[-1]),
        response_norm,
        residual_norm,
        residual_std_norm,
        response_rms,
    )


def _zonal_peak_indices(detrended_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if detrended_norm.size < 3:
        empty = np.asarray([], dtype=int)
        return empty, empty, empty
    max_peak_idx = (
        np.flatnonzero(
            (detrended_norm[1:-1] > detrended_norm[:-2])
            & (detrended_norm[1:-1] >= detrended_norm[2:])
            & (detrended_norm[1:-1] > 1.0e-12)
        )
        + 1
    )
    min_peak_idx = (
        np.flatnonzero(
            (detrended_norm[1:-1] < detrended_norm[:-2])
            & (detrended_norm[1:-1] <= detrended_norm[2:])
            & (detrended_norm[1:-1] < -1.0e-12)
        )
        + 1
    )
    return max_peak_idx, min_peak_idx, np.sort(np.concatenate([max_peak_idx, min_peak_idx]))


def _limited_peak_fit(
    times: np.ndarray,
    values: np.ndarray,
    peak_fit_max_peaks: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if peak_fit_max_peaks is None or not times.size:
        return times, values
    nfit = min(int(peak_fit_max_peaks), int(times.size))
    return times[:nfit], values[:nfit]


def _combined_envelope_damping(
    *,
    peak_times: np.ndarray,
    peak_values: np.ndarray,
    fit_mask: np.ndarray,
    peak_idx: np.ndarray,
    peak_fit_max_peaks: int | None,
) -> tuple[float, int]:
    peak_fit_times = peak_times[fit_mask[peak_idx]]
    peak_fit_values = peak_values[fit_mask[peak_idx]]
    peak_fit_times, peak_fit_values = _limited_peak_fit(
        peak_fit_times,
        peak_fit_values,
        peak_fit_max_peaks,
    )
    valid = np.isfinite(peak_fit_values) & (peak_fit_values > 0.0)
    if np.count_nonzero(valid) < 2:
        return float("nan"), int(peak_fit_times.size)
    slope, _offset = np.polyfit(peak_fit_times[valid], np.log(peak_fit_values[valid]), 1)
    return float(-slope), int(peak_fit_times.size)


def _branchwise_extrema_damping(
    *,
    t_arr: np.ndarray,
    detrended_norm: np.ndarray,
    fit_mask: np.ndarray,
    max_peak_idx: np.ndarray,
    min_peak_idx: np.ndarray,
    peak_fit_max_peaks: int | None,
) -> tuple[float, int]:
    branch_gammas: list[float] = []
    branch_counts: list[int] = []
    for branch_idx in (max_peak_idx, min_peak_idx):
        idx = branch_idx[fit_mask[branch_idx]]
        if peak_fit_max_peaks is not None and idx.size:
            idx = idx[: min(int(peak_fit_max_peaks), int(idx.size))]
        amp = np.abs(detrended_norm[idx])
        valid = np.isfinite(amp) & (amp > 0.0)
        if np.count_nonzero(valid) >= 2:
            slope, _offset = np.polyfit(t_arr[idx][valid], np.log(amp[valid]), 1)
            branch_gammas.append(float(-slope))
            branch_counts.append(int(np.count_nonzero(valid)))
    if not branch_gammas:
        return float("nan"), 0
    return float(np.mean(branch_gammas)), int(np.sum(branch_counts))


def _fit_peak_times_for_frequency(
    *,
    peak_times: np.ndarray,
    fit_mask: np.ndarray,
    peak_idx: np.ndarray,
    peak_fit_max_peaks: int | None,
    damping_mode: str,
) -> np.ndarray:
    fit_peak_times = peak_times[fit_mask[peak_idx]]
    if peak_fit_max_peaks is not None and damping_mode == "combined_envelope" and fit_peak_times.size:
        fit_peak_times = fit_peak_times[: min(int(peak_fit_max_peaks), int(fit_peak_times.size))]
    return fit_peak_times


def _peak_spacing_frequency(
    *,
    fit_peak_times: np.ndarray,
    peak_times: np.ndarray,
    fit_mask: np.ndarray,
    peak_idx: np.ndarray,
) -> float:
    freq_peak_times = fit_peak_times if fit_peak_times.size >= 2 else peak_times[fit_mask[peak_idx]]
    if freq_peak_times.size < 2:
        return float("nan")
    dt_peaks = np.diff(freq_peak_times)
    dt_peaks = dt_peaks[np.isfinite(dt_peaks) & (dt_peaks > 0.0)]
    return float(np.pi / np.mean(dt_peaks)) if dt_peaks.size else float("nan")


def _hilbert_phase_frequency(
    *,
    t_arr: np.ndarray,
    detrended_norm: np.ndarray,
    fit_mask: np.ndarray,
    hilbert_trim_fraction: float,
) -> float:
    fit_t = t_arr[fit_mask]
    fit_signal = detrended_norm[fit_mask]
    if fit_t.size < 8:
        return float("nan")
    analytic = _analytic_signal(fit_signal)
    phase = np.unwrap(np.angle(analytic))
    omega = np.gradient(phase, fit_t)
    trim = int(np.floor(float(hilbert_trim_fraction) * fit_t.size))
    trim_mask = np.ones_like(fit_t, dtype=bool)
    if trim > 0:
        trim_mask[:trim] = False
        trim_mask[-trim:] = False
    amp = np.abs(analytic)
    valid = np.isfinite(omega) & np.isfinite(amp) & (amp > 1.0e-6) & trim_mask
    return float(np.mean(omega[valid])) if np.count_nonzero(valid) >= 2 else float("nan")


def _zonal_damping_fit(
    *,
    damping_mode: str,
    peak_times: np.ndarray,
    peak_values: np.ndarray,
    fit_mask: np.ndarray,
    peak_idx: np.ndarray,
    t_arr: np.ndarray,
    detrended_norm: np.ndarray,
    max_peak_idx: np.ndarray,
    min_peak_idx: np.ndarray,
    peak_fit_max_peaks: int | None,
) -> tuple[float, int]:
    if damping_mode == "combined_envelope":
        return _combined_envelope_damping(
            peak_times=peak_times,
            peak_values=peak_values,
            fit_mask=fit_mask,
            peak_idx=peak_idx,
            peak_fit_max_peaks=peak_fit_max_peaks,
        )
    return _branchwise_extrema_damping(
        t_arr=t_arr,
        detrended_norm=detrended_norm,
        fit_mask=fit_mask,
        max_peak_idx=max_peak_idx,
        min_peak_idx=min_peak_idx,
        peak_fit_max_peaks=peak_fit_max_peaks,
    )


def _zonal_frequency_fit(
    *,
    frequency_mode: str,
    peak_times: np.ndarray,
    fit_mask: np.ndarray,
    peak_idx: np.ndarray,
    peak_fit_max_peaks: int | None,
    damping_mode: str,
    t_arr: np.ndarray,
    detrended_norm: np.ndarray,
    hilbert_trim_fraction: float,
) -> float:
    fit_peak_times = _fit_peak_times_for_frequency(
        peak_times=peak_times,
        fit_mask=fit_mask,
        peak_idx=peak_idx,
        peak_fit_max_peaks=peak_fit_max_peaks,
        damping_mode=damping_mode,
    )
    if frequency_mode == "peak_spacing":
        return _peak_spacing_frequency(
            fit_peak_times=fit_peak_times,
            peak_times=peak_times,
            fit_mask=fit_mask,
            peak_idx=peak_idx,
        )
    return _hilbert_phase_frequency(
        t_arr=t_arr,
        detrended_norm=detrended_norm,
        fit_mask=fit_mask,
        hilbert_trim_fraction=hilbert_trim_fraction,
    )


def _zonal_metric_result(
    *,
    initial_level: float,
    policy: str,
    residual_norm: float,
    residual_std_norm: float,
    response_rms: float,
    gam_frequency: float,
    gam_damping: float,
    damping_mode: str,
    frequency_mode: str,
    peak_fit_count: int,
    tmin: float,
    tmax: float,
    fit_tmin: float,
    fit_tmax: float,
    t_arr: np.ndarray,
    response_norm: np.ndarray,
    detrended_norm: np.ndarray,
    max_peak_idx: np.ndarray,
    min_peak_idx: np.ndarray,
    peak_idx: np.ndarray,
) -> ZonalFlowResponseMetrics:
    peak_values = np.abs(detrended_norm[peak_idx])
    return ZonalFlowResponseMetrics(
        initial_level=initial_level,
        initial_policy=policy,
        residual_level=residual_norm,
        residual_std=residual_std_norm,
        response_rms=response_rms,
        gam_frequency=gam_frequency,
        gam_damping_rate=gam_damping,
        damping_method=damping_mode,
        frequency_method=frequency_mode,
        peak_count=int(peak_idx.size),
        peak_fit_count=int(peak_fit_count),
        tmin=tmin,
        tmax=tmax,
        fit_tmin=float(fit_tmin),
        fit_tmax=float(fit_tmax),
        peak_times=np.asarray(t_arr[peak_idx], dtype=float),
        peak_envelope=np.asarray(peak_values, dtype=float),
        max_peak_times=np.asarray(t_arr[max_peak_idx], dtype=float),
        max_peak_values=np.asarray(response_norm[max_peak_idx], dtype=float),
        min_peak_times=np.asarray(t_arr[min_peak_idx], dtype=float),
        min_peak_values=np.asarray(response_norm[min_peak_idx], dtype=float),
    )


def _zonal_window_state(
    t: np.ndarray,
    response: np.ndarray,
    *,
    tail_fraction: float,
    initial_fraction: float,
    initial_policy: str,
    initial_level_override: float | None,
    peak_fit_max_peaks: int | None,
    damping_fit_mode: str,
    frequency_fit_mode: str,
    fit_window_tmin: float | None,
    fit_window_tmax: float | None,
    hilbert_trim_fraction: float,
) -> _ZonalWindowState:
    t_arr, resp = _coerce_zonal_trace(t, response)
    policy, damping_mode, frequency_mode = _normalized_zonal_options(
        initial_policy=initial_policy,
        peak_fit_max_peaks=peak_fit_max_peaks,
        damping_fit_mode=damping_fit_mode,
        frequency_fit_mode=frequency_fit_mode,
        hilbert_trim_fraction=hilbert_trim_fraction,
    )
    initial_level = _initial_response_level(
        t_arr=t_arr,
        resp=resp,
        initial_fraction=initial_fraction,
        policy=policy,
        initial_level_override=initial_level_override,
    )
    (
        _tail_mask,
        tail_tmin,
        tail_tmax,
        response_norm,
        residual_norm,
        residual_std_norm,
        response_rms,
    ) = _residual_window_metrics(
        t_arr=t_arr,
        resp=resp,
        tail_fraction=tail_fraction,
        initial_level=initial_level,
    )
    fit_mask, fit_tmin, fit_tmax = _explicit_time_window(
        t_arr, tmin=fit_window_tmin, tmax=fit_window_tmax
    )
    return _ZonalWindowState(
        t_arr=t_arr,
        policy=policy,
        damping_mode=damping_mode,
        frequency_mode=frequency_mode,
        initial_level=initial_level,
        tail_tmin=tail_tmin,
        tail_tmax=tail_tmax,
        response_norm=response_norm,
        residual_norm=residual_norm,
        residual_std_norm=residual_std_norm,
        response_rms=response_rms,
        detrended_norm=response_norm - residual_norm,
        fit_mask=fit_mask,
        fit_tmin=fit_tmin,
        fit_tmax=fit_tmax,
    )


def _zonal_peak_fit_state(
    state: _ZonalWindowState,
    *,
    peak_fit_max_peaks: int | None,
    hilbert_trim_fraction: float,
) -> _ZonalPeakFitState:
    max_peak_idx, min_peak_idx, peak_idx = _zonal_peak_indices(state.detrended_norm)
    peak_times = state.t_arr[peak_idx]
    peak_values = np.abs(state.detrended_norm[peak_idx])
    gam_damping, peak_fit_count = _zonal_damping_fit(
        damping_mode=state.damping_mode,
        peak_times=peak_times,
        peak_values=peak_values,
        fit_mask=state.fit_mask,
        peak_idx=peak_idx,
        t_arr=state.t_arr,
        detrended_norm=state.detrended_norm,
        max_peak_idx=max_peak_idx,
        min_peak_idx=min_peak_idx,
        peak_fit_max_peaks=peak_fit_max_peaks,
    )
    gam_frequency = _zonal_frequency_fit(
        frequency_mode=state.frequency_mode,
        peak_times=peak_times,
        fit_mask=state.fit_mask,
        peak_idx=peak_idx,
        peak_fit_max_peaks=peak_fit_max_peaks,
        damping_mode=state.damping_mode,
        t_arr=state.t_arr,
        detrended_norm=state.detrended_norm,
        hilbert_trim_fraction=hilbert_trim_fraction,
    )
    return _ZonalPeakFitState(
        max_peak_idx=max_peak_idx,
        min_peak_idx=min_peak_idx,
        peak_idx=peak_idx,
        gam_damping=gam_damping,
        peak_fit_count=peak_fit_count,
        gam_frequency=gam_frequency,
    )


def zonal_flow_response_metrics(
    t: np.ndarray,
    response: np.ndarray,
    *,
    tail_fraction: float = 0.3,
    initial_fraction: float = 0.1,
    initial_policy: str = "window_abs_mean",
    initial_level_override: float | None = None,
    peak_fit_max_peaks: int | None = None,
    damping_fit_mode: str = "combined_envelope",
    frequency_fit_mode: str = "peak_spacing",
    fit_window_tmin: float | None = None,
    fit_window_tmax: float | None = None,
    hilbert_trim_fraction: float = 0.2,
) -> ZonalFlowResponseMetrics:
    """Estimate residual level and GAM envelope metrics from a zonal response.

    The input ``response`` should be a scalar zonal observable such as zonal
    potential or a normalized zonal-energy proxy on a uniform time trace.
    ``initial_policy="first_abs"`` follows Rosenbluth-Hinton/GAM convention by
    normalizing to the initial potential magnitude; ``"window_abs_mean"`` keeps
    the older robust behavior for generic noisy traces. ``initial_level_override``
    supports benchmarks whose published normalization is an external initial
    amplitude, for example a Gaussian potential maximum rather than the first
    line-averaged sample.
    """

    state = _zonal_window_state(
        t,
        response,
        tail_fraction=tail_fraction,
        initial_fraction=initial_fraction,
        initial_policy=initial_policy,
        initial_level_override=initial_level_override,
        peak_fit_max_peaks=peak_fit_max_peaks,
        damping_fit_mode=damping_fit_mode,
        frequency_fit_mode=frequency_fit_mode,
        fit_window_tmin=fit_window_tmin,
        fit_window_tmax=fit_window_tmax,
        hilbert_trim_fraction=hilbert_trim_fraction,
    )
    peaks = _zonal_peak_fit_state(
        state,
        peak_fit_max_peaks=peak_fit_max_peaks,
        hilbert_trim_fraction=hilbert_trim_fraction,
    )
    return _zonal_metric_result(
        initial_level=state.initial_level,
        policy=state.policy,
        residual_norm=state.residual_norm,
        residual_std_norm=state.residual_std_norm,
        response_rms=state.response_rms,
        gam_frequency=peaks.gam_frequency,
        gam_damping=peaks.gam_damping,
        damping_mode=state.damping_mode,
        frequency_mode=state.frequency_mode,
        peak_fit_count=peaks.peak_fit_count,
        tmin=state.tail_tmin,
        tmax=state.tail_tmax,
        fit_tmin=state.fit_tmin,
        fit_tmax=state.fit_tmax,
        t_arr=state.t_arr,
        response_norm=state.response_norm,
        detrended_norm=state.detrended_norm,
        max_peak_idx=peaks.max_peak_idx,
        min_peak_idx=peaks.min_peak_idx,
        peak_idx=peaks.peak_idx,
    )


__all__ = ["zonal_flow_response_metrics"]
