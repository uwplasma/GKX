"""Readers for legacy GX grouped NetCDF outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GXLegacyCetgOutput:
    """Minimal legacy GX cETG diagnostic contract."""

    time: np.ndarray
    ky: np.ndarray
    kx: np.ndarray
    kz: np.ndarray
    x: np.ndarray
    y: np.ndarray
    W: np.ndarray
    Phi2: np.ndarray
    qflux: np.ndarray
    pflux: np.ndarray


_NETCDF_FILL_FLOAT = np.float64(9.969209968386869e36)


def _read_var(group, name: str) -> np.ndarray:
    return np.asarray(group.variables[name][:], dtype=float)


def _looks_like_fill(arr: np.ndarray) -> bool:
    arr_f = np.asarray(arr, dtype=float)
    if arr_f.size == 0:
        return True
    finite = np.isfinite(arr_f)
    if not np.any(finite):
        return True
    vals = arr_f[finite]
    return bool(np.all(np.abs(vals) >= 0.99 * _NETCDF_FILL_FLOAT))


def load_gx_legacy_cetg_output(path: str | Path) -> GXLegacyCetgOutput:
    """Load the grouped legacy GX cETG NetCDF format."""

    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("netCDF4 is required to load legacy GX cETG outputs") from exc

    root = Dataset(Path(path), "r")
    try:
        spectra = root.groups["Spectra"]
        fluxes = root.groups["Fluxes"]
        W = _read_var(spectra, "W")
        if _looks_like_fill(W):
            Wkx = _read_var(spectra, "Wkxst")
            W = np.sum(Wkx, axis=tuple(range(1, Wkx.ndim)))
        Phi2 = _read_var(spectra, "Phi2t")
        if _looks_like_fill(Phi2):
            Phi2kx = _read_var(spectra, "Phi2kxt")
            Phi2 = np.sum(Phi2kx, axis=tuple(range(1, Phi2kx.ndim)))
        qflux = _read_var(fluxes, "qflux")
        pflux = _read_var(fluxes, "pflux")
        if _looks_like_fill(pflux):
            pflux = np.zeros_like(qflux)
        return GXLegacyCetgOutput(
            time=np.asarray(root.variables["time"][:], dtype=float),
            ky=np.asarray(root.variables["ky"][:], dtype=float),
            kx=np.asarray(root.variables["kx"][:], dtype=float),
            kz=np.asarray(root.variables["kz"][:], dtype=float),
            x=np.asarray(root.variables["x"][:], dtype=float),
            y=np.asarray(root.variables["y"][:], dtype=float),
            W=W,
            Phi2=Phi2,
            qflux=qflux,
            pflux=pflux,
        )
    finally:
        root.close()
