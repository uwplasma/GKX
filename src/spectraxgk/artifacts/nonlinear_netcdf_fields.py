"""Final-field ``*.big.nc`` writer for nonlinear NetCDF bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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


def _final_field_arrays(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return final field arrays, filling absent electromagnetic fields by zero."""

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
    return phi_full, apar_full, bpar_full


def _final_moment_arrays(result: Any, cfg: Any) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    """Return basis and particle moments for the final state, if available."""

    if result.state is None:
        return {}, {}
    state = np.asarray(result.state)
    return _state_basis_moments(state), _particle_moments(state, cfg)


def _create_big_dimensions(
    root: Any,
    *,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    theta: np.ndarray,
    kx_vals: np.ndarray,
    ky_vals: np.ndarray,
    nspecies: int,
    nl: int,
    nm: int,
) -> None:
    """Create the fixed dimensions used by the final-field NetCDF bundle."""

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


def _write_big_grids(
    root: Any,
    *,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    theta: np.ndarray,
    kx_vals: np.ndarray,
    ky_vals: np.ndarray,
    time_vals: np.ndarray,
) -> None:
    """Write grid coordinates for the final-field NetCDF bundle."""

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


def _write_final_field_diagnostics(
    diag_group: Any,
    *,
    phi_full: np.ndarray,
    apar_full: np.ndarray,
    bpar_full: np.ndarray,
) -> None:
    """Write spectral and real-space final fields to Diagnostics."""

    phi_active = _dealiased_spectral_field(phi_full)
    apar_active = _dealiased_spectral_field(apar_full)
    bpar_active = _dealiased_spectral_field(bpar_full)
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


def _write_moment_diagnostics(
    diag_group: Any, moments: Mapping[str, np.ndarray]
) -> None:
    """Write spectral and real-space species moments to Diagnostics."""

    for name, values in moments.items():
        active = _dealiased_spectral_field(values, ky_axis=1, kx_axis=2)
        diag_group.createVariable(
            name, "f4", ("time", "s", "ky", "kx", "theta", "ri")
        )[0, ...] = _spectral_species_to_ri(active)
        diag_group.createVariable(
            f"{name}XY", "f4", ("time", "s", "y", "x", "theta")
        )[0, ...] = np.real(np.fft.ifft2(values, axes=(1, 2))).astype(
            np.float32, copy=False
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
    phi_full, apar_full, bpar_full = _final_field_arrays(result)
    basis_moments, particle_moments = _final_moment_arrays(result, cfg)
    with Dataset(path, "w") as root:
        _create_big_dimensions(
            root,
            x_vals=x_vals,
            y_vals=y_vals,
            theta=theta,
            kx_vals=kx_vals,
            ky_vals=ky_vals,
            nspecies=nspecies,
            nl=nl,
            nm=nm,
        )
        _write_runtime_root_metadata(root, cfg, nspecies=nspecies, nl=nl, nm=nm)
        _write_big_grids(
            root,
            x_vals=x_vals,
            y_vals=y_vals,
            theta=theta,
            kx_vals=kx_vals,
            ky_vals=ky_vals,
            time_vals=time_vals,
        )
        geom_group = root.createGroup("Geometry")
        _write_geometry_group(geom_group, cfg)
        diag_group = root.createGroup("Diagnostics")
        _write_final_field_diagnostics(
            diag_group,
            phi_full=phi_full,
            apar_full=apar_full,
            bpar_full=bpar_full,
        )
        _write_moment_diagnostics(diag_group, basis_moments)
        _write_moment_diagnostics(diag_group, particle_moments)
    return str(path)


__all__ = ["_write_big_netcdf"]
