"""Benchmark utilities for linear Cyclone base case comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from importlib import resources


@dataclass(frozen=True)
class CycloneReference:
    ky: np.ndarray
    omega: np.ndarray
    gamma: np.ndarray


def load_cyclone_reference() -> CycloneReference:
    """Load GX Cyclone base case reference data (adiabatic electrons)."""

    data_path = resources.files("spectraxgk").joinpath("data", "cyclone_gx_adiabatic_ref.csv")
    arr = np.loadtxt(data_path, delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


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
