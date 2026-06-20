"""NetCDF writer for internal Miller imported-geometry profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def write_miller_eik_netcdf(
    path: Path, profiles: dict[str, np.ndarray | float]
) -> None:
    """Write root-level imported Miller ``*.eiknc.nc`` output."""

    try:
        import netCDF4 as nc
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError(
            "netCDF4 is required for internal Miller eik writeout"
        ) from exc

    theta = np.asarray(profiles["theta"], dtype=float)
    ntheta2 = int(theta.shape[0] - 1)
    if ntheta2 < 2:
        raise ValueError("Insufficient theta samples for Miller eik writeout")

    with nc.Dataset(path, "w") as ds:
        ds.createDimension("z", ntheta2)

        def _vec(name: str) -> Any:
            return ds.createVariable(name, "f8", ("z",))

        _vec("theta")[:] = np.asarray(profiles["theta"], dtype=float)[:-1]
        _vec("bmag")[:] = np.asarray(profiles["bmag"], dtype=float)[:-1]
        _vec("gradpar")[:] = np.asarray(profiles["gradpar"], dtype=float)[:-1]
        _vec("grho")[:] = np.asarray(profiles["grho"], dtype=float)[:-1]
        _vec("gds2")[:] = np.asarray(profiles["gds2"], dtype=float)[:-1]
        _vec("gds21")[:] = np.asarray(profiles["gds21"], dtype=float)[:-1]
        _vec("gds22")[:] = np.asarray(profiles["gds22"], dtype=float)[:-1]
        _vec("gbdrift")[:] = np.asarray(profiles["gbdrift"], dtype=float)[:-1]
        _vec("gbdrift0")[:] = np.asarray(profiles["gbdrift0"], dtype=float)[:-1]
        _vec("cvdrift")[:] = np.asarray(profiles["cvdrift"], dtype=float)[:-1]
        _vec("cvdrift0")[:] = np.asarray(profiles["cvdrift0"], dtype=float)[:-1]
        _vec("jacob")[:] = np.asarray(profiles["jacob"], dtype=float)[:-1]
        _vec("Rplot")[:] = np.asarray(profiles["Rplot"], dtype=float)[:-1]
        _vec("Zplot")[:] = np.asarray(profiles["Zplot"], dtype=float)[:-1]
        _vec("aprime")[:] = np.asarray(profiles["aprime"], dtype=float)[:-1]

        ds.createVariable("drhodpsi", "f8").assignValue(float(profiles["drhodpsi"]))
        ds.createVariable("kxfac", "f8").assignValue(float(profiles["kxfac"]))
        ds.createVariable("Rmaj", "f8").assignValue(float(profiles["Rmaj"]))
        ds.createVariable("q", "f8").assignValue(float(profiles["q"]))
        ds.createVariable("shat", "f8").assignValue(float(profiles["shat"]))


__all__ = ["write_miller_eik_netcdf"]
