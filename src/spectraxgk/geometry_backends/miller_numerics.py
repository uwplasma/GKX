"""Numerical helper functions for the internal Miller backend."""

from __future__ import annotations

import numpy as np

from spectraxgk.geometry_backends.kernels import (
    centered_reflected_difference,
    extend_nperiod_data,
    reflect_and_append,
    weighted_centered_difference,
)


def _safe_denom(values: np.ndarray | float, eps: float = 1.0e-30) -> np.ndarray | float:
    """Avoid division by zero while preserving the sign of nonzero inputs."""

    arr = np.asarray(values, dtype=float)
    safe = np.where(np.abs(arr) < eps, np.where(arr < 0.0, -eps, eps), arr)
    if np.isscalar(values):
        return float(safe)
    return safe


def derm(arr: np.ndarray, ch: str, par: str = "e") -> np.ndarray:
    """NumPy wrapper around JAX-backed Miller ``derm``."""

    axis = "l" if ch == "l" else "r"
    return np.asarray(
        centered_reflected_difference(np.asarray(arr), axis=axis, parity=par)
    )


def dermv(arr: np.ndarray, brr: np.ndarray, ch: str, par: str = "e") -> np.ndarray:
    """NumPy wrapper around JAX-backed Miller ``dermv``."""

    axis = "l" if ch == "l" else "r"
    return np.asarray(
        weighted_centered_difference(
            np.asarray(arr), np.asarray(brr), axis=axis, parity=par
        )
    )


def nperiod_data_extend(
    arr: np.ndarray, nperiod: int, istheta: int = 0, par: str = "e"
) -> np.ndarray:
    """NumPy wrapper around JAX-backed Miller period-extension helper."""

    return np.asarray(
        extend_nperiod_data(
            np.asarray(arr), int(nperiod), istheta=bool(istheta), parity=par
        )
    )


def reflect_n_append(arr: np.ndarray, ch: str) -> np.ndarray:
    """NumPy wrapper around JAX-backed Miller reflection helper."""

    return np.asarray(reflect_and_append(np.asarray(arr), parity=ch))


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """NumPy equivalent of SciPy cumulative trapezoid with initial=0.

    Supports 1D or 2D arrays. For 2D y with 1D x, broadcasts x and integrates per-row.
    """

    yy = np.asarray(y, dtype=float)
    xx = np.asarray(x, dtype=float)

    # Handle 1D case
    if yy.ndim == 1:
        if xx.ndim != 1:
            raise ValueError("x and y must have same number of dimensions")
        if yy.shape[0] != xx.shape[0]:
            raise ValueError("x and y must have the same length")
        if yy.shape[0] < 2:
            return np.zeros_like(yy)
        area = 0.5 * (yy[1:] + yy[:-1]) * (xx[1:] - xx[:-1])
        return np.concatenate(([0.0], np.cumsum(area)))

    # Handle 2D case (integrate along axis)
    if yy.ndim == 2:
        # Normalize axis
        if axis < 0:
            axis = yy.ndim + axis

        if axis == 1:
            # Integrate along columns (per-row integration) with 1D x broadcasting
            if xx.ndim not in (1, 2):
                raise ValueError("For 2D y with axis=1, x must be 1D or 2D")

            if xx.ndim == 1:
                # 1D x: broadcast across rows
                if yy.shape[1] != xx.shape[0]:
                    raise ValueError(
                        f"x length ({xx.shape[0]}) must match y's column count ({yy.shape[1]})"
                    )
                result = np.zeros_like(yy)
                for i in range(yy.shape[0]):
                    area = 0.5 * (yy[i, 1:] + yy[i, :-1]) * (xx[1:] - xx[:-1])
                    result[i, 1:] = np.cumsum(area)
                return result
            else:
                # 2D x: element-wise matching
                if yy.shape != xx.shape:
                    raise ValueError(
                        f"y shape {yy.shape} must match x shape {xx.shape}"
                    )
                result = np.zeros_like(yy)
                for i in range(yy.shape[0]):
                    area = 0.5 * (yy[i, 1:] + yy[i, :-1]) * (xx[i, 1:] - xx[i, :-1])
                    result[i, 1:] = np.cumsum(area)
                return result
        else:
            raise NotImplementedError(
                f"cumulative_trapezoid with axis={axis} not yet supported"
            )

    raise ValueError(f"cumulative_trapezoid supports ndim=1 or 2, got {yy.ndim}")


def to_ballooning(
    theta_ex: np.ndarray, profile_ex: np.ndarray, *, parity: str
) -> tuple[np.ndarray, np.ndarray]:
    """Convert extended imported-geometry profiles to ballooning-space representation."""

    theta_ball = reflect_n_append(theta_ex, "o")
    prof_ball = reflect_n_append(profile_ex, parity)
    return theta_ball, prof_ball


__all__ = [
    "_safe_denom",
    "cumulative_trapezoid",
    "derm",
    "dermv",
    "nperiod_data_extend",
    "reflect_n_append",
    "to_ballooning",
]
