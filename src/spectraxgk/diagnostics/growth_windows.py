"""Fit-window selection utilities for linear growth-rate diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np

from spectraxgk.diagnostics.growth_fit import _log_amp_phase


def _validated_fit_inputs(t: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if signal.ndim != 1:
        raise ValueError("signal must be 1D")
    if t.shape[0] != signal.shape[0]:
        raise ValueError("t and signal must have same length")

    finite = np.isfinite(signal)
    if not np.all(finite):
        t = t[finite]
        signal = signal[finite]
        if t.size < 2:
            raise ValueError("not enough finite points to fit")
    if t.shape[0] < 2:
        raise ValueError("not enough points to fit")
    return t, signal


def _r2_score(y: np.ndarray, yfit: np.ndarray) -> float:
    ss_res = float(np.sum((y - yfit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0.0:
        return -np.inf
    return 1.0 - ss_res / ss_tot


def _least_squares_line(tt: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    A = np.vstack([tt, np.ones_like(tt)]).T
    slope, offset = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(offset), slope * tt + offset


@dataclass(frozen=True)
class _LoglinearWindowOptions:
    min_points: int
    start_fraction: float
    max_fraction: float
    end_fraction: float
    num_windows: int
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    max_amp_fraction: float
    phase_weight: float
    length_weight: float
    min_r2: float
    late_penalty: float
    min_slope: float | None
    min_slope_frac: float
    slope_var_weight: float


@dataclass(frozen=True)
class _LoglinearSearchState:
    t: np.ndarray
    log_amp: np.ndarray
    phase: np.ndarray
    amp_lin: np.ndarray
    slope_series: np.ndarray
    lengths: np.ndarray
    start_index: int
    end_index_max: int
    slope_thresh: float
    max_amp: float


def _amplitude_start_index(
    amp_lin: np.ndarray, *, start_index: int, min_amp_fraction: float
) -> int:
    if min_amp_fraction <= 0.0:
        return start_index
    amp_thresh = min_amp_fraction * float(np.max(amp_lin))
    above = np.where(amp_lin >= amp_thresh)[0]
    return max(start_index, int(above[0])) if above.size else start_index


def select_fit_window(
    t: np.ndarray,
    signal: np.ndarray,
    window_fraction: float = 0.3,
    min_points: int = 20,
    start_fraction: float = 0.0,
    growth_weight: float = 0.0,
    require_positive: bool = False,
    min_amp_fraction: float = 0.0,
) -> Tuple[float, float]:
    """Pick a time window with the most exponential-like behavior."""

    t, signal = _validated_fit_inputs(t, signal)
    n = t.shape[0]
    window = max(min_points, int(window_fraction * n))
    window = min(window, n)
    if window < 2:
        raise ValueError("window too short")
    if not (0.0 <= start_fraction < 1.0):
        raise ValueError("start_fraction must be in [0, 1)")
    if growth_weight < 0.0:
        raise ValueError("growth_weight must be >= 0")
    if not (0.0 <= min_amp_fraction < 1.0):
        raise ValueError("min_amp_fraction must be in [0, 1)")

    log_amp, phase = _log_amp_phase(signal)
    amp_lin = np.abs(signal)

    best_score = -np.inf
    best_slice = (0, window)
    found_positive = False
    start_index = _amplitude_start_index(
        amp_lin, start_index=int(start_fraction * n), min_amp_fraction=min_amp_fraction
    )
    for start in range(start_index, n - window + 1):
        end = start + window
        tt = t[start:end]
        gamma, _offset, amp_fit = _least_squares_line(tt, log_amp[start:end])
        _phase_slope, _phase_off, phase_fit = _least_squares_line(tt, phase[start:end])
        if require_positive and gamma <= 0.0:
            continue
        if require_positive:
            found_positive = True
        score = _r2_score(log_amp[start:end], amp_fit) + _r2_score(
            phase[start:end], phase_fit
        )
        if growth_weight > 0.0:
            score += growth_weight * float(gamma)
        score += 0.01 * (start / max(1, n - window))
        if score > best_score:
            best_score = score
            best_slice = (start, end)
    if require_positive and not found_positive:
        return select_fit_window(
            t,
            signal,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=False,
            min_amp_fraction=min_amp_fraction,
        )

    tmin = float(t[best_slice[0]])
    tmax = float(t[best_slice[1] - 1])
    return tmin, tmax


def _validate_loglinear_options(options: _LoglinearWindowOptions) -> None:
    if options.min_points < 2:
        raise ValueError("min_points must be >= 2")
    if not (0.0 <= options.start_fraction < 1.0):
        raise ValueError("start_fraction must be in [0, 1)")
    if not (0.0 < options.max_fraction <= 1.0):
        raise ValueError("max_fraction must be in (0, 1]")
    if not (0.0 < options.end_fraction <= 1.0):
        raise ValueError("end_fraction must be in (0, 1]")
    if options.num_windows < 1:
        raise ValueError("num_windows must be >= 1")
    if options.growth_weight < 0.0:
        raise ValueError("growth_weight must be >= 0")
    if not (0.0 <= options.min_amp_fraction < 1.0):
        raise ValueError("min_amp_fraction must be in [0, 1)")
    if not (0.0 < options.max_amp_fraction <= 1.0):
        raise ValueError("max_amp_fraction must be in (0, 1]")
    if options.late_penalty < 0.0:
        raise ValueError("late_penalty must be >= 0")
    if options.min_slope_frac < 0.0:
        raise ValueError("min_slope_frac must be >= 0")
    if options.slope_var_weight < 0.0:
        raise ValueError("slope_var_weight must be >= 0")


def _prepare_loglinear_search_state(
    t: np.ndarray,
    signal: np.ndarray,
    options: _LoglinearWindowOptions,
) -> _LoglinearSearchState:
    n = t.shape[0]
    log_amp, phase = _log_amp_phase(signal)
    amp_lin = np.abs(signal)
    slope_thresh, slope_series = _loglinear_slope_threshold(
        log_amp,
        t,
        min_slope=options.min_slope,
        min_slope_frac=options.min_slope_frac,
    )
    return _LoglinearSearchState(
        t=t,
        log_amp=log_amp,
        phase=phase,
        amp_lin=amp_lin,
        slope_series=slope_series,
        lengths=_loglinear_lengths(
            n,
            min_points=options.min_points,
            max_fraction=options.max_fraction,
            num_windows=options.num_windows,
        ),
        start_index=_amplitude_start_index(
            amp_lin,
            start_index=int(options.start_fraction * n),
            min_amp_fraction=options.min_amp_fraction,
        ),
        end_index_max=max(int(options.end_fraction * n), 2),
        slope_thresh=slope_thresh,
        max_amp=_loglinear_amp_cap(amp_lin),
    )


def _loglinear_lengths(
    n: int, *, min_points: int, max_fraction: float, num_windows: int
) -> np.ndarray:
    max_points = max(min_points, int(max_fraction * n))
    max_points = min(max_points, n)
    lengths = np.unique(np.linspace(min_points, max_points, num=num_windows).astype(int))
    lengths = lengths[lengths >= 2]
    if lengths.size == 0:
        raise ValueError("no valid window lengths")
    return lengths


def _loglinear_slope_threshold(
    log_amp: np.ndarray,
    t: np.ndarray,
    *,
    min_slope: float | None,
    min_slope_frac: float,
) -> tuple[float, np.ndarray]:
    slope_series = np.gradient(log_amp, t)
    slope_pos = slope_series[np.isfinite(slope_series) & (slope_series > 0.0)]
    slope_ref = float(np.percentile(slope_pos, 90)) if slope_pos.size else 0.0
    slope_thresh = min_slope if min_slope is not None else 0.0
    if min_slope_frac > 0.0 and slope_ref > 0.0:
        slope_thresh = max(slope_thresh, min_slope_frac * slope_ref)
    return slope_thresh, slope_series


def _loglinear_amp_cap(amp_lin: np.ndarray) -> float:
    amp_finite = amp_lin[np.isfinite(amp_lin)]
    if amp_finite.size:
        amp_ref = float(np.percentile(amp_finite, 95.0))
        if not np.isfinite(amp_ref) or amp_ref <= 0.0:
            amp_ref = float(np.max(amp_finite))
    else:
        amp_ref = float(np.nanmax(amp_lin))
    return amp_ref if amp_ref > 0.0 else float(np.max(amp_lin))


def _score_loglinear_candidate(
    *,
    t: np.ndarray,
    log_amp: np.ndarray,
    phase: np.ndarray,
    amp_lin: np.ndarray,
    slope_series: np.ndarray,
    start: int,
    end: int,
    n: int,
    require_positive: bool,
    max_amp_fraction: float,
    max_amp: float,
    phase_weight: float,
    growth_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    slope_thresh: float,
    slope_var_weight: float,
) -> tuple[float | None, bool]:
    tt = t[start:end]
    gamma, _offset, log_fit = _least_squares_line(tt, log_amp[start:end])
    _phase_slope, _phase_off, phase_fit = _least_squares_line(tt, phase[start:end])
    r2_log = _r2_score(log_amp[start:end], log_fit)
    r2_phase = _r2_score(phase[start:end], phase_fit)
    if r2_log < min_r2:
        return None, False
    if require_positive and gamma <= 0.0:
        return None, False
    if slope_thresh > 0.0 and gamma < slope_thresh:
        return None, False

    found_positive = bool(require_positive)
    if max_amp_fraction < 1.0:
        if float(np.max(amp_lin[start:end])) > max_amp_fraction * max_amp:
            return None, found_positive

    score = r2_log + phase_weight * r2_phase
    if growth_weight > 0.0:
        score += growth_weight * float(gamma)
    if length_weight > 0.0:
        score += length_weight * float(end - start) / float(n)
    if slope_var_weight > 0.0:
        slope_std = float(np.std(slope_series[start:end]))
        score -= slope_var_weight * (slope_std / (abs(gamma) + 1.0e-12))
    if late_penalty > 0.0:
        score -= late_penalty * (start / max(1, n - (end - start)))
    return score, found_positive


def _search_loglinear_windows(
    *,
    t: np.ndarray,
    log_amp: np.ndarray,
    phase: np.ndarray,
    amp_lin: np.ndarray,
    slope_series: np.ndarray,
    lengths: np.ndarray,
    start_index: int,
    end_index_max: int,
    require_positive: bool,
    max_amp_fraction: float,
    max_amp: float,
    phase_weight: float,
    growth_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    slope_thresh: float,
    slope_var_weight: float,
) -> tuple[float, tuple[int, int], bool]:
    n = t.shape[0]
    best_score = -np.inf
    best_slice = (0, int(lengths[0]))
    found_positive = False
    for window in lengths:
        if window > n:
            continue
        for start in range(start_index, n - int(window) + 1):
            end = start + int(window)
            if end > end_index_max:
                continue
            score, positive = _score_loglinear_candidate(
                t=t,
                log_amp=log_amp,
                phase=phase,
                amp_lin=amp_lin,
                slope_series=slope_series,
                start=start,
                end=end,
                n=n,
                require_positive=require_positive,
                max_amp_fraction=max_amp_fraction,
                max_amp=max_amp,
                phase_weight=phase_weight,
                growth_weight=growth_weight,
                length_weight=length_weight,
                min_r2=min_r2,
                late_penalty=late_penalty,
                slope_thresh=slope_thresh,
                slope_var_weight=slope_var_weight,
            )
            found_positive = found_positive or positive
            if score is not None and score > best_score:
                best_score = score
                best_slice = (start, end)
    return best_score, best_slice, found_positive


def _fallback_loglinear_slice(
    *,
    t: np.ndarray,
    log_amp: np.ndarray,
    phase: np.ndarray,
    lengths: np.ndarray,
    fallback_start: int,
    fallback_end_index_max: int,
    phase_weight: float,
) -> tuple[float, tuple[int, int]]:
    n = t.shape[0]
    best_score = -np.inf
    best_slice = (0, int(lengths[0]))
    for window in lengths:
        if window > n:
            continue
        for start in range(fallback_start, n - int(window) + 1):
            end = start + int(window)
            if end > fallback_end_index_max:
                continue
            tt = t[start:end]
            _gamma, _offset, log_fit = _least_squares_line(tt, log_amp[start:end])
            _phase_slope, _phase_off, phase_fit = _least_squares_line(
                tt, phase[start:end]
            )
            score = _r2_score(log_amp[start:end], log_fit) + phase_weight * _r2_score(
                phase[start:end], phase_fit
            )
            if score > best_score:
                best_score = score
                best_slice = (start, end)
    return best_score, best_slice


def _search_loglinear_best_slice(
    state: _LoglinearSearchState,
    options: _LoglinearWindowOptions,
) -> tuple[tuple[int, int], bool]:
    n = state.t.shape[0]
    best_score, best_slice, found_positive = _search_loglinear_windows(
        t=state.t,
        log_amp=state.log_amp,
        phase=state.phase,
        amp_lin=state.amp_lin,
        slope_series=state.slope_series,
        lengths=state.lengths,
        start_index=state.start_index,
        end_index_max=state.end_index_max,
        require_positive=options.require_positive,
        max_amp_fraction=options.max_amp_fraction,
        max_amp=state.max_amp,
        phase_weight=options.phase_weight,
        growth_weight=options.growth_weight,
        length_weight=options.length_weight,
        min_r2=options.min_r2,
        late_penalty=options.late_penalty,
        slope_thresh=state.slope_thresh,
        slope_var_weight=options.slope_var_weight,
    )
    if best_score != -np.inf:
        return best_slice, found_positive

    fallback_start = min(state.start_index, max(0, n - 2))
    fallback_end_index_max = (
        state.end_index_max if state.end_index_max >= options.min_points else n
    )
    _best_score, best_slice = _fallback_loglinear_slice(
        t=state.t,
        log_amp=state.log_amp,
        phase=state.phase,
        lengths=state.lengths,
        fallback_start=fallback_start,
        fallback_end_index_max=fallback_end_index_max,
        phase_weight=options.phase_weight,
    )
    return best_slice, found_positive


def _select_fit_window_loglinear_impl(
    t: np.ndarray,
    signal: np.ndarray,
    options: _LoglinearWindowOptions,
) -> Tuple[float, float]:
    state = _prepare_loglinear_search_state(t, signal, options)
    best_slice, found_positive = _search_loglinear_best_slice(state, options)
    if options.require_positive and not found_positive:
        return _select_fit_window_loglinear_impl(
            t,
            signal,
            replace(options, require_positive=False),
        )
    return float(t[best_slice[0]]), float(t[best_slice[1] - 1])


def select_fit_window_loglinear(
    t: np.ndarray,
    signal: np.ndarray,
    min_points: int = 20,
    start_fraction: float = 0.0,
    max_fraction: float = 0.8,
    end_fraction: float = 0.9,
    num_windows: int = 8,
    growth_weight: float = 0.0,
    require_positive: bool = False,
    min_amp_fraction: float = 0.0,
    max_amp_fraction: float = 0.9,
    phase_weight: float = 0.2,
    length_weight: float = 0.05,
    min_r2: float = 0.0,
    late_penalty: float = 0.1,
    min_slope: float | None = None,
    min_slope_frac: float = 0.0,
    slope_var_weight: float = 0.0,
) -> Tuple[float, float]:
    """Select a window where log-amplitude is closest to linear."""

    t, signal = _validated_fit_inputs(t, signal)
    options = _LoglinearWindowOptions(
        min_points=min_points,
        start_fraction=start_fraction,
        max_fraction=max_fraction,
        end_fraction=end_fraction,
        num_windows=num_windows,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        max_amp_fraction=max_amp_fraction,
        phase_weight=phase_weight,
        length_weight=length_weight,
        min_r2=min_r2,
        late_penalty=late_penalty,
        min_slope=min_slope,
        min_slope_frac=min_slope_frac,
        slope_var_weight=slope_var_weight,
    )
    _validate_loglinear_options(options)
    return _select_fit_window_loglinear_impl(t, signal, options)


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


def late_time_window(
    t: np.ndarray, *, tail_fraction: float = 0.4
) -> tuple[float, float]:
    """Return the start/end of a late-time tail window.

    This is the windowing convention used for manuscript-facing eigenfunction
    extraction when the growth-rate fit window is not the same object as the
    late-time mode-shape window.
    """

    _mask, tmin, tmax = _tail_window(np.asarray(t, dtype=float), float(tail_fraction))
    if tmin is None or tmax is None:
        raise ValueError("late-time window requires a non-empty time axis")
    return float(tmin), float(tmax)


def _tail_stats(arr: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(arr, dtype=float)[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


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
