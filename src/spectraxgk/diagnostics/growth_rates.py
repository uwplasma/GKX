"""Public growth-rate, frequency, and fit-window diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import warnings

import numpy as np

from spectraxgk.diagnostics.growth_fit import (
    _log_amp_phase,
    fit_growth_rate,
    fit_growth_rate_with_stats,
)
from spectraxgk.diagnostics.modes import ModeSelection, extract_mode_time_series
from spectraxgk.diagnostics.normalization import apply_diagnostic_normalization
from spectraxgk.diagnostics.growth_windows import (
    select_fit_window,
    select_fit_window_loglinear,
)
from spectraxgk.operators.linear.params import LinearParams

__all__ = [
    "_log_amp_phase",
    "fit_growth_rate",
    "fit_growth_rate_auto",
    "fit_growth_rate_auto_with_stats",
    "fit_growth_rate_with_stats",
    "instantaneous_growth_rate_from_phi",
    "select_fit_window",
    "select_fit_window_loglinear",
    "windowed_growth_rate_from_omega_series",
]


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
        raise ValueError("mode_method must be one of {'z_index', 'max', 'project', 'svd'}")

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


def windowed_growth_rate_from_omega_series(
    gamma_t: np.ndarray,
    omega_t: np.ndarray,
    sel: ModeSelection,
    *,
    navg_fraction: float = 0.5,
    use_last: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Average a resolved ``(time, ky, kx)`` growth/frequency series."""

    if gamma_t.ndim != 3 or omega_t.ndim != 3:
        raise ValueError("gamma_t and omega_t must have shape (t, ky, kx)")
    if gamma_t.shape != omega_t.shape:
        raise ValueError("gamma_t and omega_t must have matching shape")
    if sel.ky_index >= gamma_t.shape[1] or sel.kx_index >= gamma_t.shape[2]:
        raise ValueError("ModeSelection indices out of range for omega series")

    gamma = np.asarray(gamma_t[:, sel.ky_index, sel.kx_index], dtype=float)
    omega = np.asarray(omega_t[:, sel.ky_index, sel.kx_index], dtype=float)
    finite = np.isfinite(gamma) & np.isfinite(omega)
    gamma = gamma[finite]
    omega = omega[finite]
    if gamma.size == 0:
        raise ValueError("No finite growth/frequency series samples available")
    if use_last:
        return float(gamma[-1]), float(omega[-1]), gamma, omega

    istart = int(len(gamma) * navg_fraction)
    istart = max(0, min(istart, len(gamma) - 1))
    return float(np.mean(gamma[istart:])), float(np.mean(omega[istart:])), gamma, omega


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
    window_method: str = "loglinear",
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
    """Fit gamma/omega with optional auto-selected window."""

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

    if tmin is None and tmax is None:
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
        else:
            raise ValueError("window_method must be 'loglinear' or 'fixed'")
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
    window_method: str = "loglinear",
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

# Fit-signal selection helpers used by benchmark and runtime linear diagnostics.
@dataclass(frozen=True)
class _AutoFitSignalOptions:
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    max_amp_fraction: float
    window_method: str
    max_fraction: float
    end_fraction: float
    num_windows: int
    phase_weight: float
    length_weight: float
    min_r2: float
    late_penalty: float
    min_slope: float | None
    min_slope_frac: float
    slope_var_weight: float


@dataclass(frozen=True)
class _FitSignalCandidate:
    signal: np.ndarray
    name: str
    gamma: float
    omega: float
    score: float


def _select_fit_signal(
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    sel: ModeSelection,
    *,
    fit_signal: str,
    mode_method: str,
    fallback: bool = True,
) -> np.ndarray:
    def _extract(arr: np.ndarray) -> np.ndarray:
        return extract_mode_time_series(arr, sel, method=mode_method)

    def _is_valid(arr: np.ndarray) -> bool:
        finite = np.isfinite(arr)
        return int(np.count_nonzero(finite)) >= 2

    if fit_signal == "phi":
        signal = _extract(phi_t)
        if fallback and not _is_valid(signal) and density_t is not None:
            alt = _extract(density_t)
            if _is_valid(alt):
                return alt
        if not _is_valid(signal):
            warnings.warn(
                "Fit signal has insufficient finite samples; falling back to zeros.",
                RuntimeWarning,
            )
            return np.zeros(phi_t.shape[0], dtype=np.complex128)
        return signal
    if fit_signal == "density":
        if density_t is None:
            raise ValueError("density_t must be provided when fit_signal='density'")
        signal = _extract(density_t)
        if fallback and not _is_valid(signal):
            alt = _extract(phi_t)
            if _is_valid(alt):
                return alt
        if not _is_valid(signal):
            warnings.warn(
                "Fit signal has insufficient finite samples; falling back to zeros.",
                RuntimeWarning,
            )
            return np.zeros(phi_t.shape[0], dtype=np.complex128)
        return signal
    raise ValueError("fit_signal must be 'phi' or 'density'")


def _score_fit_signal_auto(
    t: np.ndarray,
    signal: np.ndarray,
    *,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    max_amp_fraction: float,
    window_method: str,
    max_fraction: float,
    end_fraction: float,
    num_windows: int,
    phase_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    min_slope: float | None,
    min_slope_frac: float,
    slope_var_weight: float,
) -> tuple[float, float, float]:
    """Score a candidate fit signal using auto-window stats."""

    try:
        gamma, omega, _tmin, _tmax, r2_log, r2_phase = fit_growth_rate_auto_with_stats(
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
    except ValueError:
        return 0.0, 0.0, -np.inf

    if not np.isfinite(gamma) or not np.isfinite(omega):
        return gamma, omega, -np.inf
    if require_positive and gamma <= 0.0:
        return gamma, omega, -np.inf
    if r2_log < min_r2:
        return gamma, omega, -np.inf
    score = float(r2_log + phase_weight * r2_phase + growth_weight * gamma)
    return gamma, omega, score


def _score_fit_signal_candidate(
    t: np.ndarray,
    source: np.ndarray,
    sel: ModeSelection,
    *,
    name: str,
    mode_method: str,
    options: _AutoFitSignalOptions,
) -> _FitSignalCandidate:
    signal = extract_mode_time_series(source, sel, method=mode_method)
    gamma, omega, score = _score_fit_signal_auto(
        t,
        signal,
        tmin=options.tmin,
        tmax=options.tmax,
        window_fraction=options.window_fraction,
        min_points=options.min_points,
        start_fraction=options.start_fraction,
        growth_weight=options.growth_weight,
        require_positive=options.require_positive,
        min_amp_fraction=options.min_amp_fraction,
        max_amp_fraction=options.max_amp_fraction,
        window_method=options.window_method,
        max_fraction=options.max_fraction,
        end_fraction=options.end_fraction,
        num_windows=options.num_windows,
        phase_weight=options.phase_weight,
        length_weight=options.length_weight,
        min_r2=options.min_r2,
        late_penalty=options.late_penalty,
        min_slope=options.min_slope,
        min_slope_frac=options.min_slope_frac,
        slope_var_weight=options.slope_var_weight,
    )
    return _FitSignalCandidate(
        signal=signal,
        name=name,
        gamma=float(gamma),
        omega=float(omega),
        score=float(score),
    )


def _best_fit_signal_candidate(
    current: _FitSignalCandidate,
    candidate: _FitSignalCandidate,
) -> _FitSignalCandidate:
    if candidate.score > current.score:
        return candidate
    return current


def _select_fit_signal_auto(
    t: np.ndarray,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    sel: ModeSelection,
    *,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    max_amp_fraction: float,
    window_method: str,
    max_fraction: float,
    end_fraction: float,
    num_windows: int,
    phase_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    min_slope: float | None,
    min_slope_frac: float,
    slope_var_weight: float,
) -> tuple[np.ndarray, str, float, float]:
    """Choose between phi/density signals based on fit quality."""

    options = _AutoFitSignalOptions(
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
    best = _score_fit_signal_candidate(
        t,
        phi_t,
        sel,
        name="phi",
        mode_method=mode_method,
        options=options,
    )
    if density_t is not None:
        best = _best_fit_signal_candidate(
            best,
            _score_fit_signal_candidate(
                t,
                density_t,
                sel,
                name="density",
                mode_method=mode_method,
                options=options,
            ),
        )

    return best.signal, best.name, best.gamma, best.omega


def _extract_mode_only_signal(
    source: np.ndarray,
    *,
    local_idx: int,
    species_index: int | None = None,
) -> np.ndarray:
    """Extract a 1D time trace from reduced mode-only outputs."""

    arr = np.asarray(source)
    if arr.ndim == 0:
        return np.asarray([arr], dtype=np.complex128)
    if arr.ndim == 1:
        return arr

    # Some save modes return (t, species, ky). Select requested species first.
    if species_index is not None and arr.ndim >= 3 and arr.shape[1] > 0:
        idx = min(max(int(species_index), 0), arr.shape[1] - 1)
        arr = arr[:, idx, ...]

    if arr.ndim == 2:
        idx = min(max(int(local_idx), 0), arr.shape[1] - 1)
        return arr[:, idx]

    # Final fallback: flatten non-time axes and select one column.
    arr2 = arr.reshape(arr.shape[0], -1)
    idx = min(max(int(local_idx), 0), arr2.shape[1] - 1)
    return arr2[:, idx]


def _normalize_growth_rate(
    gamma: float,
    omega: float,
    params: LinearParams,
    diagnostic_norm: str,
) -> tuple[float, float]:
    return apply_diagnostic_normalization(
        gamma,
        omega,
        rho_star=float(np.asarray(params.rho_star)),
        diagnostic_norm=diagnostic_norm,
    )
