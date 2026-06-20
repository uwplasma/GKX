"""Final-field ``*.big.nc`` writer for nonlinear NetCDF bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from spectraxgk.artifacts.io import _ensure_parent
from spectraxgk.artifacts.nonlinear_netcdf_geometry import (
    _particle_moments,
    _write_geometry_group,
)
from spectraxgk.artifacts.spectral_layout import (
    _dealiased_spectral_field,
    _spectral_species_to_ri,
    _spectral_to_ri,
    _spectral_to_xy,
    _state_basis_moments,
    _write_runtime_root_metadata,
)


def _write_big_netcdf(
    Dataset: Any,
    big_path: str | Path,
    result: Any,
    cfg: Any,
    *,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    theta: np.ndarray,
    kx_vals: np.ndarray,
    ky_vals: np.ndarray,
    nspecies: int,
    nl: int,
    nm: int,
    time_vals: np.ndarray,
) -> str | None:
    """Write final spectral/real-space fields and moments when fields exist."""

    if result.fields is None:
        return None
    path = Path(big_path)
    _ensure_parent(path)
    phi_full = np.asarray(result.fields.phi)
    apar_full = (
        np.zeros_like(phi_full)
        if result.fields.apar is None
        else np.asarray(result.fields.apar)
    )
    bpar_full = (
        np.zeros_like(phi_full)
        if result.fields.bpar is None
        else np.asarray(result.fields.bpar)
    )
    phi_active = _dealiased_spectral_field(phi_full)
    apar_active = _dealiased_spectral_field(apar_full)
    bpar_active = _dealiased_spectral_field(bpar_full)
    basis_moments = (
        _state_basis_moments(np.asarray(result.state))
        if result.state is not None
        else {}
    )
    particle_moments = (
        _particle_moments(np.asarray(result.state), cfg)
        if result.state is not None
        else {}
    )
    with Dataset(path, "w") as root:
        root.createDimension("ri", 2)
        root.createDimension("x", x_vals.size)
        root.createDimension("y", y_vals.size)
        root.createDimension("theta", theta.size)
        root.createDimension("kx", kx_vals.size)
        root.createDimension("ky", ky_vals.size)
        root.createDimension("kz", theta.size)
        root.createDimension("m", nm)
        root.createDimension("l", nl)
        root.createDimension("s", nspecies)
        root.createDimension("time", 1)
        _write_runtime_root_metadata(root, cfg, nspecies=nspecies, nl=nl, nm=nm)
        grids = root.createGroup("Grids")
        grids.createVariable("time", "f8", ("time",))[:] = np.asarray(
            [float(time_vals[-1]) if time_vals.size else 0.0], dtype=np.float64
        )
        grids.createVariable("kx", "f4", ("kx",))[:] = kx_vals
        grids.createVariable("ky", "f4", ("ky",))[:] = ky_vals
        grids.createVariable("kz", "f4", ("kz",))[:] = theta
        grids.createVariable("x", "f4", ("x",))[:] = x_vals
        grids.createVariable("y", "f4", ("y",))[:] = y_vals
        grids.createVariable("theta", "f4", ("theta",))[:] = theta
        geom_group = root.createGroup("Geometry")
        _write_geometry_group(geom_group, cfg)
        diag_group = root.createGroup("Diagnostics")
        diag_group.createVariable("Phi", "f4", ("time", "ky", "kx", "theta", "ri"))[
            0, ...
        ] = _spectral_to_ri(phi_active)
        diag_group.createVariable(
            "Apar", "f4", ("time", "ky", "kx", "theta", "ri")
        )[0, ...] = _spectral_to_ri(apar_active)
        diag_group.createVariable(
            "Bpar", "f4", ("time", "ky", "kx", "theta", "ri")
        )[0, ...] = _spectral_to_ri(bpar_active)
        diag_group.createVariable("PhiXY", "f4", ("time", "y", "x", "theta"))[
            0, ...
        ] = _spectral_to_xy(phi_full)
        diag_group.createVariable("AparXY", "f4", ("time", "y", "x", "theta"))[
            0, ...
        ] = _spectral_to_xy(apar_full)
        diag_group.createVariable("BparXY", "f4", ("time", "y", "x", "theta"))[
            0, ...
        ] = _spectral_to_xy(bpar_full)
        for name, values in basis_moments.items():
            active = _dealiased_spectral_field(values, ky_axis=1, kx_axis=2)
            diag_group.createVariable(
                name, "f4", ("time", "s", "ky", "kx", "theta", "ri")
            )[0, ...] = _spectral_species_to_ri(active)
            diag_group.createVariable(
                f"{name}XY", "f4", ("time", "s", "y", "x", "theta")
            )[0, ...] = np.real(np.fft.ifft2(values, axes=(1, 2))).astype(
                np.float32, copy=False
            )
        for name, values in particle_moments.items():
            active = _dealiased_spectral_field(values, ky_axis=1, kx_axis=2)
            diag_group.createVariable(
                name, "f4", ("time", "s", "ky", "kx", "theta", "ri")
            )[0, ...] = _spectral_species_to_ri(active)
            diag_group.createVariable(
                f"{name}XY", "f4", ("time", "s", "y", "x", "theta")
            )[0, ...] = np.real(np.fft.ifft2(values, axes=(1, 2))).astype(
                np.float32, copy=False
            )
    return str(path)


__all__ = ["_write_big_netcdf"]
