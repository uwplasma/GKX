"""Quasilinear spectrum integration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

def integrated_quasilinear_flux_from_spectrum(
    spectrum_csv: str | Path,
    *,
    column: str = "saturated_heat_flux_total",
    ky_column: str = "ky",
    method: str = "sum",
    delta_ky: float | None = None,
) -> dict[str, Any]:
    """Integrate one quasilinear spectrum column into a scalar flux estimate.

    ``method="sum"`` preserves the discrete spectral-sum convention used by
    most runtime diagnostics. ``method="trapezoid"`` is available for smooth
    scan studies where the CSV is treated as a sampled function of ``ky``.
    """

    path = Path(spectrum_csv)
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    names = set(data.dtype.names or ())
    if column not in names:
        raise ValueError(f"{path} is missing quasilinear column '{column}'")
    values = np.asarray(data[column], dtype=float)
    finite = np.isfinite(values)
    if ky_column in names:
        ky = np.asarray(data[ky_column], dtype=float)
        finite &= np.isfinite(ky)
    else:
        ky = np.arange(values.size, dtype=float)
    if not np.any(finite):
        raise ValueError(f"{path} contains no finite samples in column '{column}'")
    values = values[finite]
    ky = ky[finite]

    method_key = method.strip().lower()
    if method_key == "sum":
        estimate = float(np.sum(values))
        if delta_ky is not None:
            width = float(delta_ky)
            if not np.isfinite(width) or width <= 0.0:
                raise ValueError("delta_ky must be finite and positive")
            estimate *= width
    elif method_key == "mean":
        estimate = float(np.mean(values))
    elif method_key == "trapezoid":
        if values.size < 2:
            raise ValueError(
                "trapezoid integration requires at least two finite spectrum samples"
            )
        order = np.argsort(ky)
        estimate = float(np.trapezoid(values[order], ky[order]))
    else:
        raise ValueError("method must be one of {'sum', 'mean', 'trapezoid'}")

    return {
        "estimate": estimate,
        "method": method_key,
        "column": str(column),
        "ky_column": str(ky_column),
        "delta_ky": None if delta_ky is None else float(delta_ky),
        "n_samples": int(values.size),
        "ky_min": float(np.min(ky)),
        "ky_max": float(np.max(ky)),
        "artifact": str(path),
    }



__all__ = ["integrated_quasilinear_flux_from_spectrum"]
