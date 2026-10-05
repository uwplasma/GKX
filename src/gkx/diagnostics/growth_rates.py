"""Growth-rate, frequency, least-squares, and fit-window diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple
import warnings

import numpy as np

from gkx.diagnostics.modes import ModeSelection, extract_mode_time_series


# Fit-window selection.


def _log_amp_phase(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(log|signal|, unwrapped phase)`` with robust scaling."""

    if signal.size == 0:
        raise ValueError("signal must be non-empty")
    signal = np.asarray(signal)
    finite = np.isfinite(signal)
    scale = float(np.max(np.abs(signal[finite]))) if np.any(finite) else 1.0
    signal = np.where(finite, signal, 0.0) if not np.all(finite) else signal
    scale = scale if np.isfinite(scale) and scale > 0.0 else 1.0
    scaled = signal / scale
    log_amp = np.log(np.maximum(np.abs(scaled), 1.0e-30)) + np.log(scale)
    return log_amp, np.unwrap(np.angle(scaled))


def _validated_fit_inputs(
    t: np.ndarray, signal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


def _phase_r2_score(phase: np.ndarray, fit: np.ndarray) -> float:
    """R^2 of a phase fit; a constant phase (zero frequency) is a perfect fit."""

    if np.ptp(phase) == 0.0:
        return 1.0
    return _r2_score(phase, fit)


def _least_squares_coefficients(tt: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.vstack([tt, np.ones_like(tt)]).T
    return np.linalg.lstsq(A, y, rcond=None)[0]


def _least_squares_line(
    tt: np.ndarray, y: np.ndarray
) -> tuple[float, float, np.ndarray]:
    slope, offset = _least_squares_coefficients(tt, y)
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
        score = _r2_score(log_amp[start:end], amp_fit) + _phase_r2_score(
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
    lengths = np.unique(
        np.linspace(min_points, max_points, num=num_windows).astype(int)
    )
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
    r2_phase = _phase_r2_score(phase[start:end], phase_fit)
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
            score = _r2_score(
                log_amp[start:end], log_fit
            ) + phase_weight * _phase_r2_score(phase[start:end], phase_fit)
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


def _windowed_fit_inputs(
    t: np.ndarray,
    signal: np.ndarray,
    *,
    tmin: float | None,
    tmax: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, signal = _validated_fit_inputs(t, signal)
    mask = np.ones_like(t, dtype=bool)
    if tmin is not None:
        mask &= t >= tmin
    if tmax is not None:
        mask &= t <= tmax
    tt = t[mask]
    if tt.size < 2:
        raise ValueError("not enough points to fit")
    log_amp, phase = _log_amp_phase(signal[mask])
    return tt, log_amp, phase


def fit_growth_rate(
    t: np.ndarray,
    signal: np.ndarray,
    tmin: float | None = None,
    tmax: float | None = None,
) -> Tuple[float, float]:
    """Fit gamma and omega from a complex signal ~ exp((gamma - i*omega) t)."""

    tt, log_amp, phase = _windowed_fit_inputs(t, signal, tmin=tmin, tmax=tmax)
    gamma, _offset = _least_squares_coefficients(tt, log_amp)
    phase_slope, _phase_offset = _least_squares_coefficients(tt, phase)
    return float(gamma), float(-phase_slope)


def fit_growth_rate_with_stats(
    t: np.ndarray,
    signal: np.ndarray,
    tmin: float | None = None,
    tmax: float | None = None,
) -> Tuple[float, float, float, float]:
    """Fit gamma/omega and return (gamma, omega, r2_log_amp, r2_phase)."""

    tt, log_amp, phase = _windowed_fit_inputs(t, signal, tmin=tmin, tmax=tmax)
    gamma, _offset, log_fit = _least_squares_line(tt, log_amp)
    phase_slope, _phase_offset, phase_fit = _least_squares_line(tt, phase)
    return (
        gamma,
        -phase_slope,
        _r2_score(log_amp, log_fit),
        _r2_score(phase, phase_fit),
    )


__all__ = [
    "GrowthRateFitStats",
    "_log_amp_phase",
    "fit_growth_rate",
    "fit_growth_rate_auto",
    "fit_growth_rate_auto_with_stats",
    "fit_growth_rate_uncertainty",
    "fit_growth_rate_with_stats",
    "instantaneous_growth_rate_from_phi",
    "select_fit_window",
    "select_fit_window_loglinear",
    "select_fit_window_stationary",
]


@dataclass(frozen=True)
class GrowthRateFitStats:
    """Log-linear gamma/omega fit with standard errors over one window."""

    gamma: float
    omega: float
    gamma_stderr: float
    omega_stderr: float
    r2_log: float
    r2_phase: float
    n_points: int
    tmin: float
    tmax: float


def _slope_stderr(tt: np.ndarray, resid: np.ndarray) -> float:
    """Slope standard error from the linear-fit covariance.

    Residuals of a log-amplitude fit oscillate at twice the mode frequency, so
    successive samples are strongly correlated and the independent-sample OLS
    covariance understates the slope uncertainty. The standard AR(1)
    correction inflates it by ``sqrt((1 + rho) / (1 - rho))`` with ``rho`` the
    lag-one residual autocorrelation.
    """

    n = tt.size
    t_var = float(np.sum((tt - np.mean(tt)) ** 2))
    if n <= 2 or t_var <= 0.0:
        return float("inf")
    ss_res = float(np.dot(resid, resid))
    stderr = np.sqrt(ss_res / (n - 2) / t_var)
    if ss_res > 0.0:
        rho = float(np.dot(resid[1:], resid[:-1]) / ss_res)
        rho = min(max(rho, 0.0), 0.98)
        stderr *= np.sqrt((1.0 + rho) / (1.0 - rho))
    return float(stderr)


def fit_growth_rate_uncertainty(
    t: np.ndarray,
    signal: np.ndarray,
    tmin: float | None = None,
    tmax: float | None = None,
) -> GrowthRateFitStats:
    """Fit gamma/omega over a window and report stderr and R^2 diagnostics."""

    tt, log_amp, phase = _windowed_fit_inputs(t, signal, tmin=tmin, tmax=tmax)
    gamma, _offset, log_fit = _least_squares_line(tt, log_amp)
    phase_slope, _phase_off, phase_fit = _least_squares_line(tt, phase)
    return GrowthRateFitStats(
        gamma=float(gamma),
        omega=float(-phase_slope),
        gamma_stderr=_slope_stderr(tt, log_amp - log_fit),
        omega_stderr=_slope_stderr(tt, phase - phase_fit),
        r2_log=_r2_score(log_amp, log_fit),
        r2_phase=_r2_score(phase, phase_fit),
        n_points=int(tt.size),
        tmin=float(tt[0]),
        tmax=float(tt[-1]),
    )


def instantaneous_growth_rate_from_phi(
    phi_t: np.ndarray,
    t: np.ndarray | None,
    sel: ModeSelection,
    *,
    navg_fraction: float = 0.5,
    use_last: bool = False,
    mode_method: str = "z_index",
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute instantaneous growth and frequency from complex mode ratios.

    Returns ``(gamma_avg, omega_avg, gamma_t, omega_t, t_mid)``.
    """

    if phi_t.ndim != 4:
        raise ValueError("phi_t must have shape (t, ky, kx, z)")
    if t is None:
        t = np.arange(phi_t.shape[0], dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if t.shape[0] != phi_t.shape[0]:
        raise ValueError("t and phi_t must have consistent time dimension")
    if phi_t.shape[0] < 2:
        raise ValueError("phi_t must have at least two time samples")
    if mode_method not in {"z_index", "max", "project", "svd"}:
        raise ValueError(
            "mode_method must be one of {'z_index', 'max', 'project', 'svd'}"
        )

    signal = extract_mode_time_series(phi_t, sel, method=mode_method)
    phi_now = signal[1:]
    phi_prev = signal[:-1]
    dt = np.diff(t)
    dt = np.where(dt == 0.0, 1.0, dt)

    ratio = np.full_like(phi_now, np.nan + 1.0j * np.nan)
    mask = (phi_prev != 0.0) & np.isfinite(phi_prev) & np.isfinite(phi_now)
    ratio[mask] = phi_now[mask] / phi_prev[mask]

    gamma = np.log(np.abs(ratio)) / dt
    omega = -np.angle(ratio) / dt
    t_mid = 0.5 * (t[1:] + t[:-1])
    finite = np.isfinite(gamma) & np.isfinite(omega)
    gamma = gamma[finite]
    omega = omega[finite]
    t_mid = t_mid[finite]
    if gamma.size == 0:
        raise ValueError("No finite instantaneous growth-rate samples available")

    if use_last:
        gamma_avg = float(gamma[-1])
        omega_avg = float(omega[-1])
    else:
        istart = int(len(gamma) * navg_fraction)
        gamma_avg = float(np.mean(gamma[istart:]))
        omega_avg = float(np.mean(omega[istart:]))
    return gamma_avg, omega_avg, gamma, omega, t_mid


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with edge renormalization (no padding bias)."""

    if window <= 1 or values.size < 2:
        return values
    kernel = np.ones(int(window), dtype=float)
    return np.convolve(values, kernel, mode="same") / np.convolve(
        np.ones_like(values), kernel, mode="same"
    )


def _stationary_window_from_end(
    gamma_s: np.ndarray,
    *,
    end: int,
    seed: int,
    k_std: float,
    rel_tolerance: float = 0.02,
) -> tuple[int, float] | None:
    """Extend a seed window backward from ``end`` while gamma(t) stays stationary.

    Returns ``(start, gamma_mean)`` or ``None`` when the seed itself is not a
    positive-growth region (e.g. it sits in a recurrence or overflow tail).
    """

    i0 = end - seed
    if i0 < 0:
        return None
    seg = gamma_s[i0:end]
    mu = float(np.mean(seg))
    if mu <= 0.0:
        return None
    sigma = float(np.std(seg))
    # Acceptance band: k*std, but floored at a relative fraction of the mean
    # growth rate and capped so a noisy seed cannot declare arbitrarily
    # wandering gamma(t) "stationary". The floor matters more than it looks:
    # gamma(t) is smoothed before this test, so on a clean run its residual
    # scatter is orders of magnitude below any physically meaningful drift,
    # and a pure k*std band would refuse windows that are stationary to well
    # under a percent.
    tol = min(max(k_std * sigma, float(rel_tolerance) * mu) + 1.0e-12, 0.75 * mu)
    total = float(np.sum(seg))
    count = seed
    i = i0 - 1
    while i >= 0:
        if abs(gamma_s[i] - total / count) > tol:
            break
        total += float(gamma_s[i])
        count += 1
        i -= 1
    return i + 1, total / count


def select_fit_window_stationary(
    t: np.ndarray,
    signal: np.ndarray,
    *,
    min_points: int = 20,
    k_std: float = 3.0,
    rel_tolerance: float = 0.02,
    min_growth_times: float = 2.0,
    smooth_points: int | None = None,
    end_scan_fraction: float = 0.5,
    max_end_candidates: int = 32,
) -> Tuple[float, float] | None:
    """Select the longest late-time window over which gamma(t) is stationary.

    The instantaneous growth rate gamma(t) is computed from the log-amplitude
    derivative of the signal, smoothed with a short rolling mean, and windows
    are grown backward from a set of candidate late-time end points: a point
    joins the window while it stays within ``k_std`` standard deviations (or
    ``rel_tolerance`` times the mean, whichever band is wider) of the running
    window mean. Candidate end points earlier than the final
    sample let the search step over a trailing recurrence/overflow tail. A
    window is only accepted when it spans at least ``min_growth_times``
    growth times (``min_growth_times / gamma``) and ``min_points`` samples;
    ``None`` is returned when no such window exists, so callers can fall back
    to another selection method.
    """

    t, signal = _validated_fit_inputs(t, signal)
    try:
        _g, _w, gamma_t, _omega_t, t_mid = instantaneous_growth_rate_from_phi(
            np.asarray(signal, dtype=np.complex128).reshape(-1, 1, 1, 1),
            np.asarray(t, dtype=float),
            ModeSelection(ky_index=0, kx_index=0, z_index=0),
            mode_method="z_index",
        )
    except ValueError:
        return None
    n = gamma_t.size
    seed = max(4, min(int(min_points), n // 4))
    if n < max(int(min_points), 2 * seed):
        return None

    smooth = smooth_points if smooth_points is not None else max(3, n // 25)
    gamma_s = _rolling_mean(gamma_t, int(smooth))

    e_lo = max(int(np.ceil(n * (1.0 - float(end_scan_fraction)))), seed)
    ends = np.unique(
        np.linspace(n, e_lo, num=min(int(max_end_candidates), n - e_lo + 1)).astype(int)
    )[::-1]
    best: tuple[float, float, float] | None = None  # (span, tmin, tmax)
    for end in ends:
        window = _stationary_window_from_end(
            gamma_s,
            end=int(end),
            seed=seed,
            k_std=float(k_std),
            rel_tolerance=float(rel_tolerance),
        )
        if window is None:
            continue
        start, gamma_est = window
        if gamma_est <= 0.0 or end - start < int(min_points):
            continue
        span = float(t_mid[end - 1] - t_mid[start])
        if span * gamma_est < float(min_growth_times):
            continue
        if best is None or span > best[0]:
            best = (span, float(t_mid[start]), float(t_mid[end - 1]))
    if best is None:
        return None
    return best[1], best[2]


def fit_growth_rate_auto(
    t: np.ndarray,
    signal: np.ndarray,
    tmin: float | None = None,
    tmax: float | None = None,
    window_fraction: float = 0.3,
    min_points: int = 20,
    start_fraction: float = 0.0,
    growth_weight: float = 0.0,
    require_positive: bool = False,
    min_amp_fraction: float = 0.0,
    max_amp_fraction: float = 0.9,
    window_method: str = "stationary",
    max_fraction: float = 0.8,
    end_fraction: float = 0.9,
    num_windows: int = 8,
    phase_weight: float = 0.2,
    length_weight: float = 0.05,
    min_r2: float = 0.0,
    late_penalty: float = 0.1,
    min_slope: float | None = None,
    min_slope_frac: float = 0.0,
    slope_var_weight: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Fit gamma/omega with optional auto-selected window.

    ``window_method="stationary"`` (the default) selects the longest late-time
    window over which the instantaneous growth rate is stationary and falls
    back to the ``"loglinear"`` score search with a ``RuntimeWarning`` when no
    stationary window exists.
    """

    if t.ndim != 1 or signal.ndim != 1:
        raise ValueError("t and signal must be 1D")
    if t.shape[0] != signal.shape[0]:
        raise ValueError("t and signal must have same length")
    if t.size < 2:
        return 0.0, 0.0, 0.0, 0.0

    finite = np.isfinite(signal)
    if not np.all(finite):
        t = t[finite]
        signal = signal[finite]
        if t.size < 2:
            return 0.0, 0.0, 0.0, 0.0

    if window_method not in {"stationary", "loglinear", "fixed"}:
        raise ValueError("window_method must be 'stationary', 'loglinear', or 'fixed'")
    if tmin is None and tmax is None:
        if window_method == "stationary":
            window = select_fit_window_stationary(t, signal, min_points=min_points)
            if window is not None:
                tmin, tmax = window
            else:
                warnings.warn(
                    "no stationary growth window found; falling back to "
                    "log-linear auto window selection",
                    RuntimeWarning,
                )
                window_method = "loglinear"
        if window_method == "loglinear":
            tmin, tmax = select_fit_window_loglinear(
                t,
                signal,
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
        elif window_method == "fixed":
            tmin, tmax = select_fit_window(
                t,
                signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
    gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)
    tmin_out = float(tmin) if tmin is not None else float(t[0])
    tmax_out = float(tmax) if tmax is not None else float(t[-1])
    return gamma, omega, tmin_out, tmax_out


def fit_growth_rate_auto_with_stats(
    t: np.ndarray,
    signal: np.ndarray,
    tmin: float | None = None,
    tmax: float | None = None,
    window_fraction: float = 0.3,
    min_points: int = 20,
    start_fraction: float = 0.0,
    growth_weight: float = 0.0,
    require_positive: bool = False,
    min_amp_fraction: float = 0.0,
    max_amp_fraction: float = 0.9,
    window_method: str = "stationary",
    max_fraction: float = 0.8,
    end_fraction: float = 0.9,
    num_windows: int = 8,
    phase_weight: float = 0.2,
    length_weight: float = 0.05,
    min_r2: float = 0.0,
    late_penalty: float = 0.1,
    min_slope: float | None = None,
    min_slope_frac: float = 0.0,
    slope_var_weight: float = 0.0,
) -> Tuple[float, float, float, float, float, float]:
    """Fit gamma/omega and report selected window plus R^2 scores."""

    gamma, omega, tmin_out, tmax_out = fit_growth_rate_auto(
        t,
        signal,
        tmin=tmin,
        tmax=tmax,
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
        num_windows=num_windows,
        phase_weight=phase_weight,
        length_weight=length_weight,
        min_r2=min_r2,
        late_penalty=late_penalty,
        min_slope=min_slope,
        min_slope_frac=min_slope_frac,
        slope_var_weight=slope_var_weight,
    )
    try:
        _gamma, _omega, r2_log, r2_phase = fit_growth_rate_with_stats(
            t, signal, tmin=tmin_out, tmax=tmax_out
        )
    except ValueError:
        r2_log = -np.inf
        r2_phase = -np.inf
    return gamma, omega, tmin_out, tmax_out, float(r2_log), float(r2_phase)
