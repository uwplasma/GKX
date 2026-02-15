"""Analysis helpers for extracting growth rates and mode signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ModeSelection:
    ky_index: int
    kx_index: int
    z_index: int = 0


def select_ky_index(ky: np.ndarray, ky_target: float) -> int:
    """Return the index of ky closest to ky_target."""

    return int(np.argmin(np.abs(ky - ky_target)))


def extract_mode(phi_t: np.ndarray, sel: ModeSelection) -> np.ndarray:
    """Extract a complex mode time series from phi_t(t, ky, kx, z)."""

    return phi_t[:, sel.ky_index, sel.kx_index, sel.z_index]


def fit_growth_rate(
    t: np.ndarray,
    signal: np.ndarray,
    tmin: float | None = None,
    tmax: float | None = None,
) -> Tuple[float, float]:
    """Fit gamma and omega from a complex signal ~ exp((gamma + i*omega) t)."""

    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if signal.ndim != 1:
        raise ValueError("signal must be 1D")
    if t.shape[0] != signal.shape[0]:
        raise ValueError("t and signal must have same length")

    mask = np.ones_like(t, dtype=bool)
    if tmin is not None:
        mask &= t >= tmin
    if tmax is not None:
        mask &= t <= tmax

    tt = t[mask]
    yy = signal[mask]
    if tt.size < 2:
        raise ValueError("not enough points to fit")

    amp = np.abs(yy)
    phase = np.unwrap(np.angle(yy))

    A = np.vstack([tt, np.ones_like(tt)]).T
    gamma, _ = np.linalg.lstsq(A, np.log(amp), rcond=None)[0]
    omega, _ = np.linalg.lstsq(A, phase, rcond=None)[0]
    return float(gamma), float(omega)
