"""Restart-file writer for nonlinear NetCDF bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from spectraxgk.artifacts.io import _ensure_parent
from spectraxgk.artifacts.spectral_layout import _restart_to_netcdf_layout


def _write_restart_netcdf(
    Dataset: Any,
    restart_path: str | Path,
    state: Any,
    time_vals: np.ndarray,
) -> str | None:
    """Write the compact restart file and return its path when state exists."""

    if state is None:
        return None
    restart_state_layout = _restart_to_netcdf_layout(np.asarray(state))
    path = Path(restart_path)
    _ensure_parent(path)
    with Dataset(path, "w") as root:
        root.createDimension("Nspecies", restart_state_layout.shape[0])
        root.createDimension("Nm", restart_state_layout.shape[1])
        root.createDimension("Nl", restart_state_layout.shape[2])
        root.createDimension("Nz", restart_state_layout.shape[3])
        root.createDimension("Nkx", restart_state_layout.shape[4])
        root.createDimension("Nky", restart_state_layout.shape[5])
        root.createDimension("ri", 2)
        root.createVariable(
            "G", "f4", ("Nspecies", "Nm", "Nl", "Nz", "Nkx", "Nky", "ri")
        )[:, :, :, :, :, :, :] = restart_state_layout
        time_last = float(time_vals[-1]) if time_vals.size else 0.0
        root.createVariable("time", "f8", ())[:] = time_last
    return str(path)


__all__ = ["_write_restart_netcdf"]
