"""Zonal-response metric extraction for benchmark validation traces."""

from __future__ import annotations

import numpy as np

from spectraxgk.validation.benchmarks.harness_timeseries import (
    _analytic_signal,
    _explicit_time_window,
    _leading_window,
    _tail_window,
)
from spectraxgk.validation.gates import ZonalFlowResponseMetrics


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

    t_arr = np.asarray(t, dtype=float)
    resp = np.asarray(response, dtype=float)
    if t_arr.ndim != 1 or resp.ndim != 1 or t_arr.size != resp.size:
        raise ValueError(
            "t and response must be one-dimensional arrays of equal length"
        )
    if t_arr.size < 4:
        raise ValueError("zonal-flow response requires at least four samples")

    finite = np.isfinite(t_arr) & np.isfinite(resp)
    t_arr = t_arr[finite]
    resp = resp[finite]
    if t_arr.size < 4:
        raise ValueError("zonal-flow response requires at least four finite samples")

    policy = str(initial_policy).strip().lower().replace("-", "_")
    if policy not in {"window_abs_mean", "first_abs"}:
        raise ValueError(
            "initial_policy must be one of {'window_abs_mean', 'first_abs'}"
        )
    if peak_fit_max_peaks is not None and int(peak_fit_max_peaks) <= 0:
        raise ValueError("peak_fit_max_peaks must be > 0 when provided")
    damping_mode = str(damping_fit_mode).strip().lower().replace("-", "_")
    if damping_mode not in {"combined_envelope", "branchwise_extrema"}:
        raise ValueError(
            "damping_fit_mode must be one of {'combined_envelope', 'branchwise_extrema'}"
        )
    frequency_mode = str(frequency_fit_mode).strip().lower().replace("-", "_")
    if frequency_mode not in {"peak_spacing", "hilbert_phase"}:
        raise ValueError(
            "frequency_fit_mode must be one of {'peak_spacing', 'hilbert_phase'}"
        )
    if not 0.0 <= float(hilbert_trim_fraction) < 0.5:
        raise ValueError("hilbert_trim_fraction must be in [0, 0.5)")

    tail_mask, tail_tmin, tail_tmax = _tail_window(t_arr, float(tail_fraction))
    tail_vals = resp[tail_mask]
    if tail_vals.size == 0:
        raise ValueError("response windows must be non-empty")

    if initial_level_override is not None:
        initial_level = float(initial_level_override)
    elif policy == "first_abs":
        initial_level = float(abs(resp[0]))
    else:
        lead_mask, _lead_tmin, _lead_tmax = _leading_window(
            t_arr, float(initial_fraction)
        )
        initial_vals = resp[lead_mask]
        if initial_vals.size == 0:
            raise ValueError("response windows must be non-empty")
        initial_level = float(np.mean(np.abs(initial_vals)))
    if initial_level <= 0.0 or not np.isfinite(initial_level):
        raise ValueError("initial response level must be finite and positive")

    residual = float(np.mean(tail_vals))
    residual_std = float(np.std(tail_vals))
    response_norm = resp / initial_level
    residual_norm = residual / initial_level
    residual_std_norm = residual_std / initial_level
    response_rms = float(np.sqrt(np.mean(np.square(response_norm[tail_mask]))))

    detrended_norm = response_norm - residual_norm
    fit_mask, fit_tmin, fit_tmax = _explicit_time_window(
        t_arr,
        tmin=fit_window_tmin,
        tmax=fit_window_tmax,
    )

    max_peak_idx = np.asarray([], dtype=int)
    min_peak_idx = np.asarray([], dtype=int)
    peak_idx = np.asarray([], dtype=int)
    if detrended_norm.size >= 3:
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
        peak_idx = np.sort(np.concatenate([max_peak_idx, min_peak_idx]))
    peak_times = t_arr[peak_idx]
    peak_values = np.abs(detrended_norm[peak_idx])
    max_peak_times = t_arr[max_peak_idx]
    max_peak_values = response_norm[max_peak_idx]
    min_peak_times = t_arr[min_peak_idx]
    min_peak_values = response_norm[min_peak_idx]

    gam_frequency = float("nan")
    gam_damping = float("nan")
    peak_fit_count = 0
    if damping_mode == "combined_envelope":
        peak_fit_times = peak_times[fit_mask[peak_idx]]
        peak_fit_values = peak_values[fit_mask[peak_idx]]
        if peak_fit_max_peaks is not None and peak_fit_times.size:
            nfit = min(int(peak_fit_max_peaks), int(peak_fit_times.size))
            peak_fit_times = peak_fit_times[:nfit]
            peak_fit_values = peak_fit_values[:nfit]
        peak_fit_count = int(peak_fit_times.size)
        valid = np.isfinite(peak_fit_values) & (peak_fit_values > 0.0)
        if np.count_nonzero(valid) >= 2:
            slope, _offset = np.polyfit(
                peak_fit_times[valid], np.log(peak_fit_values[valid]), 1
            )
            gam_damping = float(-slope)
    else:
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
        if branch_gammas:
            gam_damping = float(np.mean(branch_gammas))
            peak_fit_count = int(np.sum(branch_counts))

    fit_peak_times = peak_times[fit_mask[peak_idx]]
    if (
        peak_fit_max_peaks is not None
        and damping_mode == "combined_envelope"
        and fit_peak_times.size
    ):
        fit_peak_times = fit_peak_times[
            : min(int(peak_fit_max_peaks), int(fit_peak_times.size))
        ]

    if frequency_mode == "peak_spacing":
        freq_peak_times = (
            fit_peak_times
            if fit_peak_times.size >= 2
            else peak_times[fit_mask[peak_idx]]
        )
        if freq_peak_times.size >= 2:
            dt_peaks = np.diff(freq_peak_times)
            dt_peaks = dt_peaks[np.isfinite(dt_peaks) & (dt_peaks > 0.0)]
            if dt_peaks.size:
                gam_frequency = float(np.pi / np.mean(dt_peaks))
    else:
        fit_t = t_arr[fit_mask]
        fit_signal = detrended_norm[fit_mask]
        if fit_t.size >= 8:
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
            if np.count_nonzero(valid) >= 2:
                gam_frequency = float(np.mean(omega[valid]))

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
        peak_count=int(peak_times.size),
        peak_fit_count=int(peak_fit_count),
        tmin=float(tail_tmin if tail_tmin is not None else t_arr[0]),
        tmax=float(tail_tmax if tail_tmax is not None else t_arr[-1]),
        fit_tmin=float(fit_tmin),
        fit_tmax=float(fit_tmax),
        peak_times=np.asarray(peak_times, dtype=float),
        peak_envelope=np.asarray(peak_values, dtype=float),
        max_peak_times=np.asarray(max_peak_times, dtype=float),
        max_peak_values=np.asarray(max_peak_values, dtype=float),
        min_peak_times=np.asarray(min_peak_times, dtype=float),
        min_peak_values=np.asarray(min_peak_values, dtype=float),
    )



__all__ = ["zonal_flow_response_metrics"]
