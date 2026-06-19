"""Nonlinear NetCDF output schema writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from spectraxgk.diagnostics import SimulationDiagnostics
from spectraxgk.geometry import (
    apply_geometry_grid_defaults,
    ensure_flux_tube_geometry_data,
)
from spectraxgk.core.grid import (
    build_spectral_grid,
    real_fft_ordered_kx,
    real_fft_unique_ky,
)
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.runtime import (
    build_runtime_geometry,
    build_runtime_linear_params,
)
from spectraxgk.artifacts.spectral_layout import (
    _dealiased_kx_values,
    _dealiased_ky_values,
    _real_space_axis,
    _require_netcdf4,
    _restart_to_netcdf_layout,
    _write_runtime_root_metadata,
)
from spectraxgk.artifacts.io import (
    _ensure_parent,
    _netcdf_bundle_base,
)
from spectraxgk.artifacts.nonlinear_diagnostics import _resolve_restart_path
from spectraxgk.artifacts.nonlinear_netcdf_fields import _write_big_netcdf
from spectraxgk.artifacts.nonlinear_netcdf_diagnostics import _write_diagnostics_group
from spectraxgk.artifacts import nonlinear_netcdf_geometry as _geometry_helpers


def _sync_geometry_helper_dependencies() -> None:
    """Sync patchable geometry helper dependencies on this facade module."""

    _geometry_helpers.apply_geometry_grid_defaults = apply_geometry_grid_defaults
    _geometry_helpers.ensure_flux_tube_geometry_data = ensure_flux_tube_geometry_data
    _geometry_helpers.build_spectral_grid = build_spectral_grid
    _geometry_helpers.real_fft_ordered_kx = real_fft_ordered_kx
    _geometry_helpers.real_fft_unique_ky = real_fft_unique_ky
    _geometry_helpers.build_linear_cache = build_linear_cache
    _geometry_helpers.build_runtime_geometry = build_runtime_geometry
    _geometry_helpers.build_runtime_linear_params = build_runtime_linear_params


def _build_output_grid_and_geometry(cfg: Any) -> tuple[Any, Any]:
    """Resolve artifact output onto the same geometry-implied grid as the solver."""

    _sync_geometry_helper_dependencies()
    return _geometry_helpers._build_output_grid_and_geometry(cfg)


def _particle_moments(state: np.ndarray, cfg: Any) -> dict[str, np.ndarray]:
    _sync_geometry_helper_dependencies()
    return _geometry_helpers._particle_moments(state, cfg)


def _write_geometry_group(
    group: Any,
    cfg: Any,
    *,
    grid: Any | None = None,
    geom: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    _sync_geometry_helper_dependencies()
    return _geometry_helpers._write_geometry_group(group, cfg, grid=grid, geom=geom)


def _write_input_parameters_group(group: Any, cfg: Any, geom: Any) -> None:
    return _geometry_helpers._write_input_parameters_group(group, cfg, geom)


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


def _write_nonlinear_netcdf_outputs(
    out: str | Path, result: Any, cfg: Any
) -> dict[str, str]:
    Dataset = _require_netcdf4()
    out_path = Path(out)
    base = _netcdf_bundle_base(out_path)
    out_nc_path = Path(f"{base}.out.nc")
    restart_path = _resolve_restart_path(out_path, cfg, for_write=True)
    big_path = Path(f"{base}.big.nc")

    diag: SimulationDiagnostics | None = result.diagnostics
    if diag is None:
        raise ValueError(
            "nonlinear NetCDF output artifacts require nonlinear diagnostics output"
        )

    grid, geom_data = _build_output_grid_and_geometry(cfg)
    theta = np.asarray(grid.z, dtype=np.float32)
    kx_vals = _dealiased_kx_values(np.asarray(grid.kx))
    ky_vals = _dealiased_ky_values(np.asarray(grid.ky))
    full_nx = int(np.asarray(grid.kx).size)
    full_ny = int(np.asarray(grid.ky).size)
    active_nx = int(kx_vals.size)
    active_ny = int(ky_vals.size)
    nspecies = (
        int(np.asarray(result.state).shape[0])
        if result.state is not None and np.asarray(result.state).ndim == 6
        else len(cfg.species)
    )
    time_vals = np.asarray(diag.t, dtype=np.float64)
    x_vals = _real_space_axis(int(grid.kx.size), float(2.0 * np.pi * grid.x0))
    y_extent = float(2.0 * np.pi * grid.y0)
    y_vals = _real_space_axis(int(grid.ky.size), y_extent)
    nl = (
        int(np.asarray(result.state).shape[1])
        if result.state is not None and np.asarray(result.state).ndim == 6
        else 1
    )
    nm = (
        int(np.asarray(result.state).shape[2])
        if result.state is not None and np.asarray(result.state).ndim == 6
        else 1
    )

    _ensure_parent(out_nc_path)
    with Dataset(out_nc_path, "w") as root:
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
        root.createDimension("time", time_vals.size)
        _write_runtime_root_metadata(root, cfg, nspecies=nspecies, nl=nl, nm=nm)

        grids = root.createGroup("Grids")
        grids.createVariable("time", "f8", ("time",))[:] = time_vals
        grids.createVariable("kx", "f4", ("kx",))[:] = kx_vals
        grids.createVariable("ky", "f4", ("ky",))[:] = ky_vals
        grids.createVariable("kz", "f4", ("kz",))[:] = theta
        grids.createVariable("x", "f4", ("x",))[:] = x_vals
        grids.createVariable("y", "f4", ("y",))[:] = y_vals
        grids.createVariable("theta", "f4", ("theta",))[:] = theta

        geom_group = root.createGroup("Geometry")
        _write_geometry_group(geom_group, cfg, grid=grid, geom=geom_data)

        _write_diagnostics_group(
            root,
            diag,
            cfg,
            nspecies=nspecies,
            full_nx=full_nx,
            full_ny=full_ny,
            active_nx=active_nx,
            active_ny=active_ny,
        )
        inputs = root.createGroup("Inputs")
        _write_input_parameters_group(inputs, cfg, geom_data)

    paths = {"out": str(out_nc_path)}

    restart_written = _write_restart_netcdf(
        Dataset,
        restart_path,
        result.state,
        time_vals,
    )
    if restart_written is not None:
        paths["restart"] = restart_written

    big_written = _write_big_netcdf(
        Dataset,
        big_path,
        result,
        cfg,
        x_vals=x_vals,
        y_vals=y_vals,
        theta=theta,
        kx_vals=kx_vals,
        ky_vals=ky_vals,
        nspecies=nspecies,
        nl=nl,
        nm=nm,
        time_vals=time_vals,
    )
    if big_written is not None:
        paths["big"] = big_written

    return paths


__all__ = [
    "_build_output_grid_and_geometry",
    "_particle_moments",
    "_write_diagnostics_group",
    "_write_geometry_group",
    "_write_input_parameters_group",
    "_write_nonlinear_netcdf_outputs",
]
