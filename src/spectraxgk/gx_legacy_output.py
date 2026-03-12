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
        return GXLegacyCetgOutput(
            time=np.asarray(root.variables["time"][:], dtype=float),
            ky=np.asarray(root.variables["ky"][:], dtype=float),
            kx=np.asarray(root.variables["kx"][:], dtype=float),
            kz=np.asarray(root.variables["kz"][:], dtype=float),
            x=np.asarray(root.variables["x"][:], dtype=float),
            y=np.asarray(root.variables["y"][:], dtype=float),
            W=np.asarray(spectra.variables["W"][:], dtype=float),
            Phi2=np.asarray(spectra.variables["Phi2t"][:], dtype=float),
            qflux=np.asarray(fluxes.variables["qflux"][:], dtype=float),
            pflux=np.asarray(fluxes.variables["pflux"][:], dtype=float),
        )
    finally:
        root.close()
