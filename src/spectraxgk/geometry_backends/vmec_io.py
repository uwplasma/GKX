"""NetCDF writeout for VMEC imported geometry."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np

def write_vmec_eik_netcdf(
    path: Path,
    profiles: dict[str, Any],
    *,
    request: Any,
) -> None:
    """Write an imported VMEC ``*.eik.nc`` file.

    The output format matches the imported-geometry NetCDF contract.
    Drift coefficients are stored at the pre-loader-normalization (2x) level;
    the loader (``load_imported_geometry_netcdf``) divides by 2 on read.
    """

    try:
        import netCDF4 as _nc
    except ImportError as exc:
        raise ImportError("netCDF4 is required for VMEC eik writeout") from exc

    theta = np.asarray(profiles["theta"], dtype=float)
    ntheta = int(theta.shape[0])

    dpsidrho = float(profiles["dpsidrho"])
    drhodpsi = 1.0 / abs(dpsidrho) if abs(dpsidrho) > 1.0e-30 else 1.0e30
    gradpar_val = float(profiles["gradpar"][0])
    bmag = np.asarray(profiles["bmag"], dtype=float)
    jacob = 1.0 / np.abs(drhodpsi * gradpar_val * bmag)

    with _nc.Dataset(path, "w") as ds:
        ds.createDimension("z", ntheta)
        ds.createDimension("3", 3)

        def _v(name: str, dtype: str = "f8", dims: tuple = ("z",)) -> Any:
            return ds.createVariable(name, dtype, dims)

        _v("theta")[:] = theta
        _v("theta_PEST")[:] = np.asarray(profiles["theta_PEST"], dtype=float)
        _v("bmag")[:] = bmag
        _v("gradpar")[:] = np.asarray(profiles["gradpar"], dtype=float)
        _v("grho")[:] = np.asarray(profiles["grho"], dtype=float)
        _v("gds2")[:] = np.asarray(profiles["gds2"], dtype=float)
        _v("gds21")[:] = np.asarray(profiles["gds21"], dtype=float)
        _v("gds22")[:] = np.asarray(profiles["gds22"], dtype=float)
        _v("gbdrift")[:] = np.asarray(profiles["gbdrift"], dtype=float)
        _v("gbdrift0")[:] = np.asarray(profiles["gbdrift0"], dtype=float)
        _v("cvdrift")[:] = np.asarray(profiles["cvdrift"], dtype=float)
        _v("cvdrift0")[:] = np.asarray(profiles["cvdrift0"], dtype=float)
        _v("jacob")[:] = jacob
        _v("Rplot")[:] = np.asarray(profiles["Rplot"], dtype=float)
        _v("Zplot")[:] = np.asarray(profiles["Zplot"], dtype=float)
        _v("grad_x", dims=("3", "z"))[:, :] = np.asarray(
            profiles["grad_x"], dtype=float
        )
        _v("grad_y", dims=("3", "z"))[:, :] = np.asarray(
            profiles["grad_y"], dtype=float
        )
        _v("b_vec", dims=("3", "z"))[:, :] = np.asarray(profiles["b_vec"], dtype=float)

        ds.createVariable("drhodpsi", "f8").assignValue(drhodpsi)
        ds.createVariable("kxfac", "f8").assignValue(float(profiles.get("kxfac", 1.0)))
        ds.createVariable("Rmaj", "f8").assignValue(float(profiles["Rmaj"]))
        ds.createVariable("q", "f8").assignValue(float(profiles["q"]))
        ds.createVariable("shat", "f8").assignValue(float(profiles["shat"]))
        ds.createVariable("scale", "f8").assignValue(float(profiles["scale"]))
        ds.createVariable("alpha", "f8").assignValue(float(profiles["alpha"]))
        ds.createVariable("zeta_center", "f8").assignValue(
            float(profiles["zeta_center"])
        )
        ds.createVariable("nfp", "i4").assignValue(int(profiles["nfp"]))


def _write_vmec_eik_netcdf_atomically(
    path: Path,
    profiles: dict[str, Any],
    *,
    request: Any,
) -> None:
    """Write a VMEC eik file through a unique temp file, then atomically replace.

    W7-X validation sweeps commonly launch multiple identical VMEC geometry
    requests at once. Writing the shared cache file directly can expose a
    partially written netCDF to another process. A per-process temp path keeps
    the final cache path all-or-nothing.
    """

    final_path = Path(path).expanduser().resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_name(
        f".{final_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        write_vmec_eik_netcdf(temp_path, profiles, request=request)
        os.replace(temp_path, final_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
