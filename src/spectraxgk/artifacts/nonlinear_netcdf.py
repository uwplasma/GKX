"""Nonlinear NetCDF output schema writer."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class _NonlinearNetCDFPaths:
    out_nc_path: Path
    restart_path: Path
    big_path: Path


@dataclass(frozen=True)
class _NonlinearNetCDFLayout:
    grid: Any
    geom_data: Any
    theta: np.ndarray
    kx_vals: np.ndarray
    ky_vals: np.ndarray
    full_nx: int
    full_ny: int
    active_nx: int
    active_ny: int
    nspecies: int
    time_vals: np.ndarray
    x_vals: np.ndarray
    y_vals: np.ndarray
    nl: int
    nm: int


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


def _nonlinear_netcdf_paths(out: str | Path, cfg: Any) -> _NonlinearNetCDFPaths:
    out_path = Path(out)
    base = _netcdf_bundle_base(out_path)
    return _NonlinearNetCDFPaths(
        out_nc_path=Path(f"{base}.out.nc"),
        restart_path=Path(_resolve_restart_path(out_path, cfg, for_write=True)),
        big_path=Path(f"{base}.big.nc"),
    )


def _required_nonlinear_diagnostics(result: Any) -> SimulationDiagnostics:
    diag: SimulationDiagnostics | None = result.diagnostics
    if diag is None:
        raise ValueError(
            "nonlinear NetCDF output artifacts require nonlinear diagnostics output"
        )
    return diag


def _state_axis_size(state: Any, axis: int, default: int) -> int:
    if state is None:
        return int(default)
    state_array = np.asarray(state)
    return int(state_array.shape[axis]) if state_array.ndim == 6 else int(default)


def _nonlinear_netcdf_layout(
    result: Any,
    cfg: Any,
    diag: SimulationDiagnostics,
) -> _NonlinearNetCDFLayout:
    grid, geom_data = _build_output_grid_and_geometry(cfg)
    theta = np.asarray(grid.z, dtype=np.float32)
    kx_vals = _dealiased_kx_values(np.asarray(grid.kx))
    ky_vals = _dealiased_ky_values(np.asarray(grid.ky))
    full_nx = int(np.asarray(grid.kx).size)
    full_ny = int(np.asarray(grid.ky).size)
    active_nx = int(kx_vals.size)
    active_ny = int(ky_vals.size)
    nspecies = _state_axis_size(result.state, 0, len(cfg.species))
    time_vals = np.asarray(diag.t, dtype=np.float64)
    x_vals = _real_space_axis(int(grid.kx.size), float(2.0 * np.pi * grid.x0))
    y_extent = float(2.0 * np.pi * grid.y0)
    y_vals = _real_space_axis(int(grid.ky.size), y_extent)
    return _NonlinearNetCDFLayout(
        grid=grid,
        geom_data=geom_data,
        theta=theta,
        kx_vals=kx_vals,
        ky_vals=ky_vals,
        full_nx=full_nx,
        full_ny=full_ny,
        active_nx=active_nx,
        active_ny=active_ny,
        nspecies=nspecies,
        time_vals=time_vals,
        x_vals=x_vals,
        y_vals=y_vals,
        nl=_state_axis_size(result.state, 1, 1),
        nm=_state_axis_size(result.state, 2, 1),
    )


def _create_nonlinear_dimensions(root: Any, layout: _NonlinearNetCDFLayout) -> None:
    root.createDimension("ri", 2)
    root.createDimension("x", layout.x_vals.size)
    root.createDimension("y", layout.y_vals.size)
    root.createDimension("theta", layout.theta.size)
    root.createDimension("kx", layout.kx_vals.size)
    root.createDimension("ky", layout.ky_vals.size)
    root.createDimension("kz", layout.theta.size)
    root.createDimension("m", layout.nm)
    root.createDimension("l", layout.nl)
    root.createDimension("s", layout.nspecies)
    root.createDimension("time", layout.time_vals.size)


def _write_grids_group(root: Any, layout: _NonlinearNetCDFLayout) -> None:
    grids = root.createGroup("Grids")
    grids.createVariable("time", "f8", ("time",))[:] = layout.time_vals
    grids.createVariable("kx", "f4", ("kx",))[:] = layout.kx_vals
    grids.createVariable("ky", "f4", ("ky",))[:] = layout.ky_vals
    grids.createVariable("kz", "f4", ("kz",))[:] = layout.theta
    grids.createVariable("x", "f4", ("x",))[:] = layout.x_vals
    grids.createVariable("y", "f4", ("y",))[:] = layout.y_vals
    grids.createVariable("theta", "f4", ("theta",))[:] = layout.theta


def _write_primary_nonlinear_netcdf(
    Dataset: Any,
    out_nc_path: Path,
    *,
    diag: SimulationDiagnostics,
    cfg: Any,
    layout: _NonlinearNetCDFLayout,
) -> None:
    _ensure_parent(out_nc_path)
    with Dataset(out_nc_path, "w") as root:
        _create_nonlinear_dimensions(root, layout)
        _write_runtime_root_metadata(
            root,
            cfg,
            nspecies=layout.nspecies,
            nl=layout.nl,
            nm=layout.nm,
        )
        _write_grids_group(root, layout)

        geom_group = root.createGroup("Geometry")
        _write_geometry_group(
            geom_group,
            cfg,
            grid=layout.grid,
            geom=layout.geom_data,
        )

        _write_diagnostics_group(
            root,
            diag,
            cfg,
            nspecies=layout.nspecies,
            full_nx=layout.full_nx,
            full_ny=layout.full_ny,
            active_nx=layout.active_nx,
            active_ny=layout.active_ny,
        )
        inputs = root.createGroup("Inputs")
        _write_input_parameters_group(inputs, cfg, layout.geom_data)


def _write_optional_nonlinear_netcdf_artifacts(
    Dataset: Any,
    paths: _NonlinearNetCDFPaths,
    *,
    result: Any,
    cfg: Any,
    layout: _NonlinearNetCDFLayout,
) -> dict[str, str]:
    written: dict[str, str] = {}
    restart_written = _write_restart_netcdf(
        Dataset,
        paths.restart_path,
        result.state,
        layout.time_vals,
    )
    if restart_written is not None:
        written["restart"] = restart_written

    big_written = _write_big_netcdf(
        Dataset,
        paths.big_path,
        result,
        cfg,
        x_vals=layout.x_vals,
        y_vals=layout.y_vals,
        theta=layout.theta,
        kx_vals=layout.kx_vals,
        ky_vals=layout.ky_vals,
        nspecies=layout.nspecies,
        nl=layout.nl,
        nm=layout.nm,
        time_vals=layout.time_vals,
    )
    if big_written is not None:
        written["big"] = big_written
    return written


def _write_nonlinear_netcdf_outputs(
    out: str | Path, result: Any, cfg: Any
) -> dict[str, str]:
    Dataset = _require_netcdf4()
    paths = _nonlinear_netcdf_paths(out, cfg)
    diag = _required_nonlinear_diagnostics(result)
    layout = _nonlinear_netcdf_layout(result, cfg, diag)

    _write_primary_nonlinear_netcdf(
        Dataset,
        paths.out_nc_path,
        diag=diag,
        cfg=cfg,
        layout=layout,
    )

    written = {"out": str(paths.out_nc_path)}
    written.update(
        _write_optional_nonlinear_netcdf_artifacts(
            Dataset,
            paths,
            result=result,
            cfg=cfg,
            layout=layout,
        )
    )
    return written


__all__ = [
    "_build_output_grid_and_geometry",
    "_particle_moments",
    "_write_diagnostics_group",
    "_write_geometry_group",
    "_write_input_parameters_group",
    "_write_nonlinear_netcdf_outputs",
]
