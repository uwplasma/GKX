"""Fit-window selection utilities for linear growth-rate diagnostics."""

from __future__ import annotations

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


def _validate_loglinear_options(
    *,
    min_points: int,
    start_fraction: float,
    max_fraction: float,
    end_fraction: float,
    num_windows: int,
    growth_weight: float,
    min_amp_fraction: float,
    max_amp_fraction: float,
    late_penalty: float,
    min_slope_frac: float,
    slope_var_weight: float,
) -> None:
    if min_points < 2:
        raise ValueError("min_points must be >= 2")
    if not (0.0 <= start_fraction < 1.0):
        raise ValueError("start_fraction must be in [0, 1)")
    if not (0.0 < max_fraction <= 1.0):
        raise ValueError("max_fraction must be in (0, 1]")
    if not (0.0 < end_fraction <= 1.0):
        raise ValueError("end_fraction must be in (0, 1]")
    if num_windows < 1:
        raise ValueError("num_windows must be >= 1")
    if growth_weight < 0.0:
        raise ValueError("growth_weight must be >= 0")
    if not (0.0 <= min_amp_fraction < 1.0):
        raise ValueError("min_amp_fraction must be in [0, 1)")
    if not (0.0 < max_amp_fraction <= 1.0):
        raise ValueError("max_amp_fraction must be in (0, 1]")
    if late_penalty < 0.0:
        raise ValueError("late_penalty must be >= 0")
    if min_slope_frac < 0.0:
        raise ValueError("min_slope_frac must be >= 0")
    if slope_var_weight < 0.0:
        raise ValueError("slope_var_weight must be >= 0")


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
    n = t.shape[0]
    _validate_loglinear_options(
        min_points=min_points,
        start_fraction=start_fraction,
        max_fraction=max_fraction,
        end_fraction=end_fraction,
        num_windows=num_windows,
        growth_weight=growth_weight,
        min_amp_fraction=min_amp_fraction,
        max_amp_fraction=max_amp_fraction,
        late_penalty=late_penalty,
        min_slope_frac=min_slope_frac,
        slope_var_weight=slope_var_weight,
    )
    end_index_max = max(int(end_fraction * n), 2)
    lengths = _loglinear_lengths(
        n, min_points=min_points, max_fraction=max_fraction, num_windows=num_windows
    )

    log_amp, phase = _log_amp_phase(signal)
    amp_lin = np.abs(signal)
    slope_thresh, slope_series = _loglinear_slope_threshold(
        log_amp, t, min_slope=min_slope, min_slope_frac=min_slope_frac
    )
    start_index = _amplitude_start_index(
        amp_lin, start_index=int(start_fraction * n), min_amp_fraction=min_amp_fraction
    )
    best_score, best_slice, found_positive = _search_loglinear_windows(
        t=t,
        log_amp=log_amp,
        phase=phase,
        amp_lin=amp_lin,
        slope_series=slope_series,
        lengths=lengths,
        start_index=start_index,
        end_index_max=end_index_max,
        require_positive=require_positive,
        max_amp_fraction=max_amp_fraction,
        max_amp=_loglinear_amp_cap(amp_lin),
        phase_weight=phase_weight,
        growth_weight=growth_weight,
        length_weight=length_weight,
        min_r2=min_r2,
        late_penalty=late_penalty,
        slope_thresh=slope_thresh,
        slope_var_weight=slope_var_weight,
    )

    if best_score == -np.inf:
        fallback_start = min(start_index, max(0, n - 2))
        fallback_end_index_max = end_index_max if end_index_max >= min_points else n
        best_score, best_slice = _fallback_loglinear_slice(
            t=t,
            log_amp=log_amp,
            phase=phase,
            lengths=lengths,
            fallback_start=fallback_start,
            fallback_end_index_max=fallback_end_index_max,
            phase_weight=phase_weight,
        )

    if require_positive and not found_positive:
        return select_fit_window_loglinear(
            t,
            signal,
            min_points=min_points,
            start_fraction=start_fraction,
            max_fraction=max_fraction,
            end_fraction=end_fraction,
            num_windows=num_windows,
            growth_weight=growth_weight,
            require_positive=False,
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

    tmin = float(t[best_slice[0]])
    tmax = float(t[best_slice[1] - 1])
    return tmin, tmax
