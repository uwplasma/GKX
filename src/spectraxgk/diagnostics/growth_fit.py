"""Least-squares growth-rate and frequency fits for complex mode traces."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def fit_growth_rate(
    t: np.ndarray,
    signal: np.ndarray,
    tmin: float | None = None,
    tmax: float | None = None,
) -> Tuple[float, float]:
    """Fit gamma and omega from a complex signal ~ exp((gamma - i*omega) t)."""

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

    mask = np.ones_like(t, dtype=bool)
    if tmin is not None:
        mask &= t >= tmin
    if tmax is not None:
        mask &= t <= tmax

    tt = t[mask]
    yy = signal[mask]
    if tt.size < 2:
        raise ValueError("not enough points to fit")

    log_amp, phase = _log_amp_phase(yy)

    A = np.vstack([tt, np.ones_like(tt)]).T
    gamma, _ = np.linalg.lstsq(A, log_amp, rcond=None)[0]
    omega, _ = np.linalg.lstsq(A, phase, rcond=None)[0]
    return float(gamma), float(-omega)


def fit_growth_rate_with_stats(
    t: np.ndarray,
    signal: np.ndarray,
    tmin: float | None = None,
    tmax: float | None = None,
) -> Tuple[float, float, float, float]:
    """Fit gamma/omega and return (gamma, omega, r2_log_amp, r2_phase)."""

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

    mask = np.ones_like(t, dtype=bool)
    if tmin is not None:
        mask &= t >= tmin
    if tmax is not None:
        mask &= t <= tmax

    tt = t[mask]
    yy = signal[mask]
    if tt.size < 2:
        raise ValueError("not enough points to fit")

    log_amp, phase = _log_amp_phase(yy)
    A = np.vstack([tt, np.ones_like(tt)]).T
    gamma, offset = np.linalg.lstsq(A, log_amp, rcond=None)[0]
    phase_slope, phase_off = np.linalg.lstsq(A, phase, rcond=None)[0]
    log_fit = gamma * tt + offset
    phase_fit = phase_slope * tt + phase_off

    def r2_score(y: np.ndarray, yfit: np.ndarray) -> float:
        ss_res = float(np.sum((y - yfit) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot <= 0.0:
            return -np.inf
        return 1.0 - ss_res / ss_tot

    r2_log = r2_score(log_amp, log_fit)
    r2_phase = r2_score(phase, phase_fit)
    return float(gamma), float(-phase_slope), r2_log, r2_phase



def _log_amp_phase(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (log|signal|, unwrapped phase) with robust scaling."""

    if signal.size == 0:
        raise ValueError("signal must be non-empty")
    signal = np.asarray(signal)
    finite = np.isfinite(signal)
    if np.any(finite):
        scale = float(np.max(np.abs(signal[finite])))
    else:
        scale = 1.0
    if not np.all(finite):
        signal = np.where(finite, signal, 0.0)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scaled = signal / scale
    amp = np.abs(scaled)
    log_amp = np.log(np.maximum(amp, 1.0e-30)) + np.log(scale)
    phase = np.unwrap(np.angle(scaled))
    return log_amp, phase
