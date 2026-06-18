"""Small numerical compatibility helpers for VMEC imported geometry."""

from __future__ import annotations

import numpy as np

from spectraxgk.geometry_backends.kernels import finite_diff_nonuniform, nperiod_contract

def nperiod_set(
    values: np.ndarray, theta: np.ndarray, npol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Contract *values* / *theta* to theta in [-npol*pi, npol*pi]."""

    v = np.asarray(values)
    t = np.asarray(theta)
    if v.shape != t.shape:
        raise ValueError("values and theta must have the same shape")
    v_out, t_out = nperiod_contract(v, t, float(npol))
    return np.asarray(v_out), np.asarray(t_out)


def dermv(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Second-order non-uniform finite-difference derivative (1-D)."""

    v = np.asarray(values)
    x = np.asarray(grid)
    if v.ndim != 1 or x.ndim != 1:
        raise ValueError("dermv expects 1D arrays")
    if v.shape[0] != x.shape[0]:
        raise ValueError("values and grid must have identical lengths")
    return np.asarray(finite_diff_nonuniform(v, x))
