"""Helpers for zonal-response validation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pandas import DataFrame
else:
    DataFrame = Any

from gkx.artifacts.figure_style import set_plot_style
from scripts.checks._gates.validation_gates import ZonalFlowResponseMetrics


def _tail_window(
    t: np.ndarray, tail_fraction: float
) -> tuple[np.ndarray, float | None, float | None]:
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional")
    if t.size == 0:
        raise ValueError("t must be non-empty")
    start = max(0, int(np.floor((1.0 - float(tail_fraction)) * t.size)))
    mask = np.zeros_like(t, dtype=bool)
    mask[start:] = True
    if not np.any(mask):
        mask[-1] = True
    tt = np.asarray(t[mask], dtype=float)
    return mask, float(tt[0]) if tt.size else None, float(tt[-1]) if tt.size else None


def _leading_window(
    t: np.ndarray,
    lead_fraction: float,
) -> tuple[np.ndarray, float | None, float | None]:
    if not 0.0 < float(lead_fraction) <= 1.0:
        raise ValueError("lead_fraction must be in (0, 1]")
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional")
    if t.size == 0:
        raise ValueError("t must be non-empty")
    stop = max(1, int(np.ceil(float(lead_fraction) * t.size)))
    mask = np.zeros_like(t, dtype=bool)
    mask[:stop] = True
    tt = np.asarray(t[mask], dtype=float)
    return mask, float(tt[0]) if tt.size else None, float(tt[-1]) if tt.size else None


def _explicit_time_window(
    t: np.ndarray,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
) -> tuple[np.ndarray, float, float]:
    mask = np.ones_like(t, dtype=bool)
    if tmin is not None:
        mask &= t >= float(tmin)
    if tmax is not None:
        mask &= t <= float(tmax)
    if not np.any(mask):
        raise ValueError("explicit fit window is empty")
    tt = np.asarray(t[mask], dtype=float)
    return mask, float(tt[0]), float(tt[-1])


def _analytic_signal(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    spec = np.fft.fft(x)
    filt = np.zeros(x.size, dtype=float)
    if x.size % 2 == 0:
        filt[0] = 1.0
        filt[x.size // 2] = 1.0
        filt[1 : x.size // 2] = 2.0
    else:
        filt[0] = 1.0
        filt[1 : (x.size + 1) // 2] = 2.0
    return np.fft.ifft(spec * filt)


# Zonal-response metrics used by benchmark and manuscript validation gates.
_DAMPING_FIT_MODES = frozenset(
    {"combined_envelope", "branchwise_extrema", "period_rms_envelope"}
)


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
    damping_fit_tmin: float = float("nan")
    damping_fit_tmax: float = float("nan")


def _coerce_zonal_trace(
    t: np.ndarray, response: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
        raise ValueError(
            "initial_policy must be one of {'window_abs_mean', 'first_abs'}"
        )
    if peak_fit_max_peaks is not None and int(peak_fit_max_peaks) <= 0:
        raise ValueError("peak_fit_max_peaks must be > 0 when provided")
    damping_mode = str(damping_fit_mode).strip().lower().replace("-", "_")
    if damping_mode not in _DAMPING_FIT_MODES:
        raise ValueError(
            f"damping_fit_mode must be one of {sorted(_DAMPING_FIT_MODES)}"
        )
    frequency_mode = str(frequency_fit_mode).strip().lower().replace("-", "_")
    if frequency_mode not in {"peak_spacing", "hilbert_phase"}:
        raise ValueError(
            "frequency_fit_mode must be one of {'peak_spacing', 'hilbert_phase'}"
        )
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
        lead_mask, _lead_tmin, _lead_tmax = _leading_window(
            t_arr, float(initial_fraction)
        )
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


def _zonal_peak_indices(
    detrended_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return (
        max_peak_idx,
        min_peak_idx,
        np.sort(np.concatenate([max_peak_idx, min_peak_idx])),
    )


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
    slope, _offset = np.polyfit(
        peak_fit_times[valid], np.log(peak_fit_values[valid]), 1
    )
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


def _sliding_period_envelope(
    t_arr: np.ndarray,
    detrended_norm: np.ndarray,
    *,
    period: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, envelope)`` from a sliding one-period RMS of the trace.

    For ``y(t) = C(t) + A exp(-gamma t) cos(omega t + phi)`` with ``C`` varying
    slowly on the oscillation period, the RMS of ``y`` about its *own* running
    mean over exactly one period is ``A exp(-gamma t)`` times a factor that does
    not depend on ``t``. So ``log(envelope)`` has slope ``-gamma`` whatever the
    offset does, and every sample inside the window enters the average with
    weight ``1/width`` -- no single extremum, and therefore no single output
    sample, can move the fit. Refining the diagnostic output cadence refines the
    quadrature of the same continuum integral instead of changing which points
    are fitted, which is what makes the estimate cadence-independent.
    """

    spacing = float(np.median(np.diff(t_arr))) if t_arr.size > 1 else 0.0
    if (
        not np.isfinite(spacing)
        or spacing <= 0.0
        or not np.isfinite(period)
        or period <= 0.0
    ):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    width = int(round(float(period) / spacing))
    if width % 2 == 0:
        width += 1
    if width < 5 or 2 * width > int(t_arr.size):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    kernel = np.full(width, 1.0 / float(width), dtype=float)
    # Both passes are "valid" convolutions so no output ever sees the zero
    # padding "same" would splice in at the ends: an edge-contaminated local
    # mean would otherwise leak one full window into the squared deviations.
    half = width // 2
    size = int(t_arr.size)
    local_mean = np.convolve(detrended_norm, kernel, mode="valid")
    deviation = detrended_norm[half : size - half] - local_mean
    local_var = np.convolve(np.square(deviation), kernel, mode="valid")
    times = np.asarray(t_arr[width - 1 : size - width + 1], dtype=float)
    return times, np.sqrt(2.0 * np.maximum(local_var, 0.0))


def _amplitude_weighted_log_slope(
    times: np.ndarray,
    envelope: np.ndarray,
) -> tuple[float, int]:
    """Weighted log-linear slope of an envelope, with weights ``envelope**2``.

    Additive noise of size ``delta`` on an envelope sample of size ``A`` costs
    ``delta / A`` in the log, so the inverse-variance weight of a log-envelope
    point is proportional to ``A**2``. That is also what keeps the late,
    near-zero end of the window -- where recurrence and round-off live -- from
    dominating a fit whose early points carry the actual signal.
    """

    valid = np.isfinite(times) & np.isfinite(envelope) & (envelope > 0.0)
    count = int(np.count_nonzero(valid))
    if count < 4:
        return float("nan"), count
    x = times[valid]
    y = np.log(envelope[valid])
    weights = np.square(envelope[valid])
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return float("nan"), count
    weights = weights / total
    x_mean = float(np.sum(weights * x))
    y_mean = float(np.sum(weights * y))
    denominator = float(np.sum(weights * np.square(x - x_mean)))
    if denominator <= 0.0:
        return float("nan"), count
    slope = float(np.sum(weights * (x - x_mean) * (y - y_mean)) / denominator)
    return -slope, count


def _period_rms_envelope_damping(
    *,
    t_arr: np.ndarray,
    detrended_norm: np.ndarray,
    gam_frequency: float,
    damping_fit_start_periods: float,
    damping_fit_periods: float,
) -> tuple[float, int, float, float]:
    """Fit the GAM damping over a whole number of GAM periods of the envelope.

    The window is stated in periods of the measured oscillation rather than in
    samples or in an absolute time, so it does not move when the output cadence
    or the timestep changes. ``damping_fit_start_periods`` skips the initial
    relaxation that precedes the first full oscillation.
    """

    if not np.isfinite(gam_frequency) or gam_frequency <= 0.0:
        return float("nan"), 0, float("nan"), float("nan")
    period = 2.0 * np.pi / float(gam_frequency)
    window_tmin = float(t_arr[0]) + float(damping_fit_start_periods) * period
    window_tmax = window_tmin + float(damping_fit_periods) * period
    times, envelope = _sliding_period_envelope(t_arr, detrended_norm, period=period)
    if times.size == 0:
        return float("nan"), 0, window_tmin, window_tmax
    inside = (times >= window_tmin) & (times <= window_tmax)
    damping, count = _amplitude_weighted_log_slope(times[inside], envelope[inside])
    return damping, count, window_tmin, window_tmax


def _fit_peak_times_for_frequency(
    *,
    peak_times: np.ndarray,
    fit_mask: np.ndarray,
    peak_idx: np.ndarray,
    peak_fit_max_peaks: int | None,
    damping_mode: str,
) -> np.ndarray:
    fit_peak_times = peak_times[fit_mask[peak_idx]]
    if (
        peak_fit_max_peaks is not None
        and damping_mode == "combined_envelope"
        and fit_peak_times.size
    ):
        fit_peak_times = fit_peak_times[
            : min(int(peak_fit_max_peaks), int(fit_peak_times.size))
        ]
    return fit_peak_times


def _peak_spacing_frequency(
    *,
    fit_peak_times: np.ndarray,
    peak_times: np.ndarray,
    fit_mask: np.ndarray,
    peak_idx: np.ndarray,
) -> float:
    freq_peak_times = (
        fit_peak_times if fit_peak_times.size >= 2 else peak_times[fit_mask[peak_idx]]
    )
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
    return (
        float(np.mean(omega[valid])) if np.count_nonzero(valid) >= 2 else float("nan")
    )


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
    gam_frequency: float,
    damping_fit_start_periods: float,
    damping_fit_periods: float,
) -> tuple[float, int, float, float]:
    if damping_mode == "period_rms_envelope":
        return _period_rms_envelope_damping(
            t_arr=t_arr,
            detrended_norm=detrended_norm,
            gam_frequency=gam_frequency,
            damping_fit_start_periods=damping_fit_start_periods,
            damping_fit_periods=damping_fit_periods,
        )
    if damping_mode == "combined_envelope":
        damping, count = _combined_envelope_damping(
            peak_times=peak_times,
            peak_values=peak_values,
            fit_mask=fit_mask,
            peak_idx=peak_idx,
            peak_fit_max_peaks=peak_fit_max_peaks,
        )
    else:
        damping, count = _branchwise_extrema_damping(
            t_arr=t_arr,
            detrended_norm=detrended_norm,
            fit_mask=fit_mask,
            max_peak_idx=max_peak_idx,
            min_peak_idx=min_peak_idx,
            peak_fit_max_peaks=peak_fit_max_peaks,
        )
    windowed = t_arr[fit_mask]
    if windowed.size:
        return damping, count, float(windowed[0]), float(windowed[-1])
    return damping, count, float("nan"), float("nan")


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
    damping_fit_tmin: float,
    damping_fit_tmax: float,
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
        damping_fit_tmin=float(damping_fit_tmin),
        damping_fit_tmax=float(damping_fit_tmax),
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
    damping_fit_start_periods: float,
    damping_fit_periods: float,
) -> _ZonalPeakFitState:
    max_peak_idx, min_peak_idx, peak_idx = _zonal_peak_indices(state.detrended_norm)
    peak_times = state.t_arr[peak_idx]
    peak_values = np.abs(state.detrended_norm[peak_idx])
    # The frequency is fitted first because ``period_rms_envelope`` states its
    # own window in GAM periods, so it needs the measured oscillation frequency.
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
    gam_damping, peak_fit_count, damping_tmin, damping_tmax = _zonal_damping_fit(
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
        gam_frequency=gam_frequency,
        damping_fit_start_periods=damping_fit_start_periods,
        damping_fit_periods=damping_fit_periods,
    )
    return _ZonalPeakFitState(
        max_peak_idx=max_peak_idx,
        min_peak_idx=min_peak_idx,
        peak_idx=peak_idx,
        gam_damping=gam_damping,
        peak_fit_count=peak_fit_count,
        gam_frequency=gam_frequency,
        damping_fit_tmin=damping_tmin,
        damping_fit_tmax=damping_tmax,
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
    damping_fit_start_periods: float = 1.0,
    damping_fit_periods: float = 3.0,
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

    ``damping_fit_mode="period_rms_envelope"`` is the estimator to prefer for a
    *gated* damping rate. The extrema-based modes fit a handful of hand-picked
    peaks, so a near-zero-crossing wiggle that one output cadence resolves and
    another does not enters the fit and moves the answer by tens of percent; the
    period-RMS envelope averages every sample over whole GAM periods and does
    not. ``damping_fit_start_periods`` and ``damping_fit_periods`` state that
    window in periods of the measured oscillation, so it is fixed in physical
    time rather than in samples or in a hard-coded absolute time.
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
        damping_fit_start_periods=damping_fit_start_periods,
        damping_fit_periods=damping_fit_periods,
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
        damping_fit_tmin=peaks.damping_fit_tmin,
        damping_fit_tmax=peaks.damping_fit_tmax,
        t_arr=state.t_arr,
        response_norm=state.response_norm,
        detrended_norm=state.detrended_norm,
        max_peak_idx=peaks.max_peak_idx,
        min_peak_idx=peaks.min_peak_idx,
        peak_idx=peaks.peak_idx,
    )


# Zonal-response plotting helpers.
@dataclass(frozen=True)
class _ZonalPlotData:
    t: np.ndarray
    response_norm: np.ndarray
    residual: float
    residual_std: float
    env_t: np.ndarray
    env_y: np.ndarray
    fit_count: int
    fit_tmin: float
    fit_tmax: float
    damping_method: str
    frequency_method: str
    max_peak_t: np.ndarray
    max_peak_y: np.ndarray
    min_peak_t: np.ndarray
    min_peak_y: np.ndarray
    metrics: Any


def _validated_zonal_trace(
    t: np.ndarray, response: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    t_arr = np.asarray(t, dtype=float)
    resp = np.asarray(response, dtype=float)
    if t_arr.ndim != 1 or resp.ndim != 1 or t_arr.size != resp.size:
        raise ValueError(
            "t and response must be one-dimensional arrays of equal length"
        )
    return t_arr, resp


def _array_metric(metrics: Any, name: str) -> np.ndarray:
    value = getattr(metrics, name, np.asarray([], dtype=float))
    return np.asarray(value, dtype=float)


def _zonal_plot_data(
    t: np.ndarray, response: np.ndarray, metrics: Any
) -> _ZonalPlotData:
    response_norm = response / float(metrics.initial_level)
    return _ZonalPlotData(
        t=t,
        response_norm=response_norm,
        residual=float(metrics.residual_level),
        residual_std=float(metrics.residual_std),
        env_t=np.asarray(metrics.peak_times, dtype=float),
        env_y=np.asarray(metrics.peak_envelope, dtype=float),
        fit_count=int(getattr(metrics, "peak_fit_count", len(metrics.peak_times))),
        fit_tmin=float(getattr(metrics, "fit_tmin", t[0])),
        fit_tmax=float(getattr(metrics, "fit_tmax", t[-1])),
        damping_method=str(getattr(metrics, "damping_method", "combined_envelope")),
        frequency_method=str(getattr(metrics, "frequency_method", "peak_spacing")),
        max_peak_t=_array_metric(metrics, "max_peak_times"),
        max_peak_y=_array_metric(metrics, "max_peak_values"),
        min_peak_t=_array_metric(metrics, "min_peak_times"),
        min_peak_y=_array_metric(metrics, "min_peak_values"),
        metrics=metrics,
    )


def _fit_window_mask(values: np.ndarray, data: _ZonalPlotData) -> np.ndarray:
    return (values >= data.fit_tmin) & (values <= data.fit_tmax)


def _plot_branchwise_points(
    ax: plt.Axes,
    *,
    data: _ZonalPlotData,
) -> None:
    if data.damping_method != "branchwise_extrema":
        return
    if data.max_peak_t.size:
        keep = _fit_window_mask(data.max_peak_t, data)
        ax.plot(
            data.max_peak_t[keep],
            data.max_peak_y[keep],
            linestyle="none",
            marker="o",
            color="#2a9d8f",
            markersize=5.2,
            label="maxima fit points",
        )
    if data.min_peak_t.size:
        keep = _fit_window_mask(data.min_peak_t, data)
        ax.plot(
            data.min_peak_t[keep],
            data.min_peak_y[keep],
            linestyle="none",
            marker="o",
            color="#7b2cbf",
            markersize=5.2,
            label="minima fit points",
        )


def _plot_normalized_response_panel(
    ax: plt.Axes,
    *,
    data: _ZonalPlotData,
    y_label: str,
) -> None:
    ax.plot(
        data.t, data.response_norm, color="#0f4c81", linewidth=2.2, label="response"
    )
    ax.axhline(
        data.residual,
        color="#c44e52",
        linestyle="--",
        linewidth=2.0,
        label="residual",
    )
    ax.axvspan(data.fit_tmin, data.fit_tmax, color="#d9ead3", alpha=0.22, linewidth=0.0)
    _plot_branchwise_points(ax, data=data)
    ax.fill_between(
        data.t,
        data.residual - data.residual_std,
        data.residual + data.residual_std,
        color="#c44e52",
        alpha=0.15,
        linewidth=0.0,
    )
    ax.set_xlabel("t")
    ax.set_ylabel(y_label)
    ax.set_title("Normalized response")
    ax.legend(loc="best", frameon=False)


def _combined_envelope_fit(
    data: _ZonalPlotData,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    fit_mask = _fit_window_mask(data.env_t, data)
    fit_env_t = data.env_t[fit_mask]
    fit_env_y = data.env_y[fit_mask]
    finite_damping = np.isfinite(float(data.metrics.gam_damping_rate))
    if data.damping_method != "combined_envelope" or data.fit_count < 2:
        return None
    if not finite_damping or not fit_env_t.size:
        return None

    fit_n = min(data.fit_count, fit_env_t.size)
    fit_t = fit_env_t[:fit_n]
    fit = fit_env_y[0] * np.exp(
        -float(data.metrics.gam_damping_rate) * (fit_t - fit_t[0])
    )
    label = (
        "envelope fit"
        if fit_n == fit_env_t.size
        else f"envelope fit (first {fit_n} peaks)"
    )
    return fit_t, fit, label


def _plot_envelope_panel(ax: plt.Axes, *, data: _ZonalPlotData) -> None:
    ax.plot(
        data.t,
        np.maximum(np.abs(data.response_norm - data.residual), 1.0e-14),
        color="#4c956c",
        linewidth=2.0,
        alpha=0.5,
    )
    if data.env_t.size:
        ax.plot(
            data.env_t,
            data.env_y,
            color="#c44e52",
            marker="o",
            linewidth=1.8,
            label="envelope peaks",
        )
    fit = _combined_envelope_fit(data)
    if fit is not None:
        fit_t, fit_y, label = fit
        ax.plot(
            fit_t, fit_y, color="#2a9d8f", linestyle="--", linewidth=2.0, label=label
        )
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("envelope")
    ax.set_title("GAM envelope")
    if data.env_t.size:
        ax.legend(loc="best", frameon=False)


def _metrics_annotation(data: _ZonalPlotData) -> str:
    metrics = data.metrics
    norm_policy = getattr(metrics, "initial_policy", "window_abs_mean")
    return (
        f"residual = {metrics.residual_level:.4f}\n"
        f"std = {metrics.residual_std:.4f}\n"
        f"ω_GAM = {metrics.gam_frequency:.4f}\n"
        f"γ_damp = {metrics.gam_damping_rate:.4f}\n"
        f"fit_peaks = {data.fit_count}\n"
        f"norm = {norm_policy}\n"
        f"damp = {data.damping_method}\n"
        f"freq = {data.frequency_method}\n"
        f"fit_t = [{data.fit_tmin:.1f}, {data.fit_tmax:.1f}]"
    )


def _annotate_envelope_panel(ax: plt.Axes, *, data: _ZonalPlotData) -> None:
    ax.text(
        0.03,
        0.97,
        _metrics_annotation(data),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "#cccccc",
        },
    )


def _style_zonal_axes(axes: np.ndarray) -> None:
    for axis in axes:
        axis.grid(True, alpha=0.25)


def zonal_flow_response_figure(
    t: np.ndarray,
    response: np.ndarray,
    *,
    metrics: Any = None,
    title: str = "Zonal-flow response",
    y_label: str = "normalized response",
) -> tuple[plt.Figure, np.ndarray]:
    """Render a zonal-flow response trace and its envelope summary."""

    set_plot_style()
    t_arr, resp = _validated_zonal_trace(t, response)
    if metrics is None:
        metrics = zonal_flow_response_metrics(t_arr, resp)
    data = _zonal_plot_data(t_arr, resp, metrics)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    _plot_normalized_response_panel(axes[0], data=data, y_label=y_label)
    _plot_envelope_panel(axes[1], data=data)
    _annotate_envelope_panel(axes[1], data=data)
    _style_zonal_axes(axes)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


__all__ = ["zonal_flow_response_figure"]
