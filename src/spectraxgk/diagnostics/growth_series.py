"""Growth-rate and frequency diagnostics from resolved mode time series."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from spectraxgk.diagnostics.modes import ModeSelection, extract_mode_time_series


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

    Returns (gamma_avg, omega_avg, gamma_t, omega_t, t_mid).
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

    log_amp = np.log(np.abs(ratio))
    phase = np.angle(ratio)
    gamma = log_amp / dt
    omega = -phase / dt
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
    """Compute window-averaged growth and frequency from precomputed time series.

    Parameters
    ----------
    gamma_t, omega_t:
        Arrays with shape ``(t, ky, kx)``.
    sel:
        Mode selection used to choose the ``(ky, kx)`` series.
    navg_fraction:
        Fractional start index for late-time averaging.
    use_last:
        If true, use the last finite sample instead of late-time average.
    """

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

