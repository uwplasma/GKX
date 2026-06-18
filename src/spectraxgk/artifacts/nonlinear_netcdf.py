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
    _complex_to_ri,
    _condense_kx_for_output,
    _condense_ky_for_output,
    _condense_kykx_for_output,
    _dealiased_kx_values,
    _dealiased_ky_values,
    _real_space_axis,
    _require_netcdf4,
    _species_matrix,
    _write_runtime_root_metadata,
)
from spectraxgk.artifacts.io import (
    _ensure_parent,
    _netcdf_bundle_base,
)
from spectraxgk.artifacts.nonlinear_diagnostics import (
    _resolved_species_time,
    _resolve_restart_path,
)
from spectraxgk.artifacts.nonlinear_netcdf_fields import _write_big_netcdf
from spectraxgk.artifacts import nonlinear_netcdf_geometry as _geometry_helpers
from spectraxgk.artifacts.nonlinear_netcdf_restart import _write_restart_netcdf


def _sync_geometry_helper_dependencies() -> None:
    """Preserve the historical monkeypatch seam on this facade module."""

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

        diag_group = root.createGroup("Diagnostics")
        resolved = diag.resolved
        phi2_kx_out = None
        phi2_ky_out = None
        phi2_kykx_out = None
        if resolved is not None and resolved.Phi2_kxkyt is not None:
            # The NetCDF schema stores Phi2 on the rFFT-positive ky view.  Deriving the one-
            # dimensional spectra from the condensed two-dimensional spectrum
            # keeps Phi2_t, Phi2_kxt, and Phi2_kyt mutually consistent when
            # SPECTRAX-GK evolved a full Hermitian ky layout internally.
            phi2_kykx_out = _condense_kykx_for_output(
                np.asarray(resolved.Phi2_kxkyt, dtype=np.float32),
                full_ny=full_ny,
                full_nx=full_nx,
                active_ny=active_ny,
                active_nx=active_nx,
            )
            phi2_kx_out = np.sum(phi2_kykx_out, axis=1)
            phi2_ky_out = np.sum(phi2_kykx_out, axis=2)
            phi2_t = np.sum(phi2_kykx_out, axis=(1, 2))
        elif resolved is not None and resolved.Phi2_kyt is not None:
            phi2_ky_out = _condense_ky_for_output(
                np.asarray(resolved.Phi2_kyt, dtype=np.float32),
                full_ny=full_ny,
                active_ny=active_ny,
            )
            phi2_t = np.sum(phi2_ky_out, axis=1)
        elif resolved is not None and resolved.Phi2_kxt is not None:
            phi2_kx_out = _condense_kx_for_output(
                np.asarray(resolved.Phi2_kxt, dtype=np.float32),
                full_nx=full_nx,
                active_nx=active_nx,
            )
            phi2_t = np.sum(phi2_kx_out, axis=1)
        else:
            phi2_t = np.asarray(diag.Wphi_t, dtype=np.float32)
        diag_group.createVariable("Phi2_t", "f4", ("time",))[:] = phi2_t
        wg_s = _species_matrix(np.asarray(diag.Wg_t, dtype=np.float32), nspecies, None)
        wphi_s = _species_matrix(
            np.asarray(diag.Wphi_t, dtype=np.float32), nspecies, None
        )
        wapar_s = _species_matrix(
            np.asarray(diag.Wapar_t, dtype=np.float32), nspecies, None
        )
        heat_s = _species_matrix(
            np.asarray(diag.heat_flux_t, dtype=np.float32),
            nspecies,
            None
            if diag.heat_flux_species_t is None
            else np.asarray(diag.heat_flux_species_t, dtype=np.float32),
        )
        pflux_s = _species_matrix(
            np.asarray(diag.particle_flux_t, dtype=np.float32),
            nspecies,
            None
            if diag.particle_flux_species_t is None
            else np.asarray(diag.particle_flux_species_t, dtype=np.float32),
        )
        turb_heat_s = _species_matrix(
            np.asarray(
                np.zeros_like(diag.heat_flux_t)
                if diag.turbulent_heating_t is None
                else diag.turbulent_heating_t,
                dtype=np.float32,
            ),
            nspecies,
            None
            if diag.turbulent_heating_species_t is None
            else np.asarray(diag.turbulent_heating_species_t, dtype=np.float32),
        )
        diag_group.createVariable("Wg_st", "f4", ("time", "s"))[:, :] = wg_s
        diag_group.createVariable("Wphi_st", "f4", ("time", "s"))[:, :] = wphi_s
        diag_group.createVariable("Wapar_st", "f4", ("time", "s"))[:, :] = wapar_s
        diag_group.createVariable("HeatFlux_st", "f4", ("time", "s"))[:, :] = heat_s
        diag_group.createVariable("ParticleFlux_st", "f4", ("time", "s"))[:, :] = (
            pflux_s
        )
        heat_es_st = _resolved_species_time(
            None if resolved is None else resolved.HeatFluxES_kxst,
            fallback=heat_s if cfg.physics.electrostatic else np.zeros_like(heat_s),
        )
        heat_apar_st = _resolved_species_time(
            None if resolved is None else resolved.HeatFluxApar_kxst,
            fallback=np.zeros_like(heat_s),
        )
        heat_bpar_st = _resolved_species_time(
            None if resolved is None else resolved.HeatFluxBpar_kxst,
            fallback=np.zeros_like(heat_s),
        )
        pflux_es_st = _resolved_species_time(
            None if resolved is None else resolved.ParticleFluxES_kxst,
            fallback=pflux_s if cfg.physics.electrostatic else np.zeros_like(pflux_s),
        )
        pflux_apar_st = _resolved_species_time(
            None if resolved is None else resolved.ParticleFluxApar_kxst,
            fallback=np.zeros_like(pflux_s),
        )
        pflux_bpar_st = _resolved_species_time(
            None if resolved is None else resolved.ParticleFluxBpar_kxst,
            fallback=np.zeros_like(pflux_s),
        )
        diag_group.createVariable("HeatFluxES_st", "f4", ("time", "s"))[:, :] = (
            heat_es_st
        )
        diag_group.createVariable("HeatFluxApar_st", "f4", ("time", "s"))[:, :] = (
            heat_apar_st
        )
        diag_group.createVariable("HeatFluxBpar_st", "f4", ("time", "s"))[:, :] = (
            heat_bpar_st
        )
        diag_group.createVariable("ParticleFluxES_st", "f4", ("time", "s"))[:, :] = (
            pflux_es_st
        )
        diag_group.createVariable("ParticleFluxApar_st", "f4", ("time", "s"))[:, :] = (
            pflux_apar_st
        )
        diag_group.createVariable("ParticleFluxBpar_st", "f4", ("time", "s"))[:, :] = (
            pflux_bpar_st
        )
        turb_heat_st = _resolved_species_time(
            None if resolved is None else resolved.TurbulentHeating_kxst,
            fallback=turb_heat_s,
        )
        diag_group.createVariable("TurbulentHeating_st", "f4", ("time", "s"))[:, :] = (
            turb_heat_st
        )
        if resolved is not None:
            if phi2_kx_out is not None:
                diag_group.createVariable("Phi2_kxt", "f4", ("time", "kx"))[:, :] = (
                    phi2_kx_out
                )
            if phi2_ky_out is not None:
                diag_group.createVariable("Phi2_kyt", "f4", ("time", "ky"))[:, :] = (
                    phi2_ky_out
                )
            if phi2_kykx_out is not None:
                diag_group.createVariable("Phi2_kxkyt", "f4", ("time", "ky", "kx"))[
                    :, :, :
                ] = phi2_kykx_out
            if resolved.Phi2_zt is not None:
                diag_group.createVariable("Phi2_zt", "f4", ("time", "theta"))[:, :] = (
                    np.asarray(resolved.Phi2_zt, dtype=np.float32)
                )
            if resolved.Phi2_zonal_t is not None:
                diag_group.createVariable("Phi2_zonal_t", "f4", ("time",))[:] = (
                    np.asarray(resolved.Phi2_zonal_t, dtype=np.float32)
                )
            if resolved.Phi2_zonal_kxt is not None:
                diag_group.createVariable("Phi2_zonal_kxt", "f4", ("time", "kx"))[
                    :, :
                ] = _condense_kx_for_output(
                    np.asarray(resolved.Phi2_zonal_kxt, dtype=np.float32),
                    full_nx=full_nx,
                    active_nx=active_nx,
                )
            if resolved.Phi2_zonal_zt is not None:
                diag_group.createVariable("Phi2_zonal_zt", "f4", ("time", "theta"))[
                    :, :
                ] = np.asarray(resolved.Phi2_zonal_zt, dtype=np.float32)
            if resolved.Phi_zonal_mode_kxt is not None:
                diag_group.createVariable(
                    "Phi_zonal_mode_kxt", "f4", ("time", "kx", "ri")
                )[:, :, :] = _complex_to_ri(
                    _condense_kx_for_output(
                        np.asarray(resolved.Phi_zonal_mode_kxt),
                        full_nx=full_nx,
                        active_nx=active_nx,
                    )
                )
            if resolved.Phi_zonal_line_kxt is not None:
                diag_group.createVariable(
                    "Phi_zonal_line_kxt", "f4", ("time", "kx", "ri")
                )[:, :, :] = _complex_to_ri(
                    _condense_kx_for_output(
                        np.asarray(resolved.Phi_zonal_line_kxt),
                        full_nx=full_nx,
                        active_nx=active_nx,
                    )
                )
            metric_specs = (
                (
                    "Wg",
                    resolved.Wg_kxst,
                    resolved.Wg_kyst,
                    resolved.Wg_kxkyst,
                    resolved.Wg_zst,
                ),
                (
                    "Wphi",
                    resolved.Wphi_kxst,
                    resolved.Wphi_kyst,
                    resolved.Wphi_kxkyst,
                    resolved.Wphi_zst,
                ),
                (
                    "Wapar",
                    resolved.Wapar_kxst,
                    resolved.Wapar_kyst,
                    resolved.Wapar_kxkyst,
                    resolved.Wapar_zst,
                ),
                (
                    "HeatFlux",
                    resolved.HeatFlux_kxst,
                    resolved.HeatFlux_kyst,
                    resolved.HeatFlux_kxkyst,
                    resolved.HeatFlux_zst,
                ),
                (
                    "ParticleFlux",
                    resolved.ParticleFlux_kxst,
                    resolved.ParticleFlux_kyst,
                    resolved.ParticleFlux_kxkyst,
                    resolved.ParticleFlux_zst,
                ),
            )
            for prefix, kx_arr, ky_arr, kykx_arr, z_arr in metric_specs:
                if kx_arr is not None:
                    diag_group.createVariable(
                        f"{prefix}_kxst", "f4", ("time", "s", "kx")
                    )[:, :, :] = _condense_kx_for_output(
                        np.asarray(kx_arr, dtype=np.float32),
                        full_nx=full_nx,
                        active_nx=active_nx,
                    )
                if ky_arr is not None:
                    diag_group.createVariable(
                        f"{prefix}_kyst", "f4", ("time", "s", "ky")
                    )[:, :, :] = _condense_ky_for_output(
                        np.asarray(ky_arr, dtype=np.float32),
                        full_ny=full_ny,
                        active_ny=active_ny,
                    )
                if kykx_arr is not None:
                    diag_group.createVariable(
                        f"{prefix}_kxkyst", "f4", ("time", "s", "ky", "kx")
                    )[:, :, :, :] = _condense_kykx_for_output(
                        np.asarray(kykx_arr, dtype=np.float32),
                        full_ny=full_ny,
                        full_nx=full_nx,
                        active_ny=active_ny,
                        active_nx=active_nx,
                    )
                if z_arr is not None:
                    diag_group.createVariable(
                        f"{prefix}_zst", "f4", ("time", "s", "theta")
                    )[:, :, :] = np.asarray(z_arr, dtype=np.float32)
            if resolved.Wg_lmst is not None:
                diag_group.createVariable("Wg_lmst", "f4", ("time", "s", "m", "l"))[
                    :, :, :, :
                ] = np.asarray(resolved.Wg_lmst, dtype=np.float32)
            split_metric_specs = (
                (
                    "HeatFluxES",
                    resolved.HeatFluxES_kxst,
                    resolved.HeatFluxES_kyst,
                    resolved.HeatFluxES_kxkyst,
                    resolved.HeatFluxES_zst,
                    resolved.HeatFlux_kxst,
                    resolved.HeatFlux_kyst,
                    resolved.HeatFlux_kxkyst,
                    resolved.HeatFlux_zst,
                    cfg.physics.electrostatic,
                ),
                (
                    "HeatFluxApar",
                    resolved.HeatFluxApar_kxst,
                    resolved.HeatFluxApar_kyst,
                    resolved.HeatFluxApar_kxkyst,
                    resolved.HeatFluxApar_zst,
                    None,
                    None,
                    None,
                    None,
                    False,
                ),
                (
                    "HeatFluxBpar",
                    resolved.HeatFluxBpar_kxst,
                    resolved.HeatFluxBpar_kyst,
                    resolved.HeatFluxBpar_kxkyst,
                    resolved.HeatFluxBpar_zst,
                    None,
                    None,
                    None,
                    None,
                    False,
                ),
                (
                    "ParticleFluxES",
                    resolved.ParticleFluxES_kxst,
                    resolved.ParticleFluxES_kyst,
                    resolved.ParticleFluxES_kxkyst,
                    resolved.ParticleFluxES_zst,
                    resolved.ParticleFlux_kxst,
                    resolved.ParticleFlux_kyst,
                    resolved.ParticleFlux_kxkyst,
                    resolved.ParticleFlux_zst,
                    cfg.physics.electrostatic,
                ),
                (
                    "ParticleFluxApar",
                    resolved.ParticleFluxApar_kxst,
                    resolved.ParticleFluxApar_kyst,
                    resolved.ParticleFluxApar_kxkyst,
                    resolved.ParticleFluxApar_zst,
                    None,
                    None,
                    None,
                    None,
                    False,
                ),
                (
                    "ParticleFluxBpar",
                    resolved.ParticleFluxBpar_kxst,
                    resolved.ParticleFluxBpar_kyst,
                    resolved.ParticleFluxBpar_kxkyst,
                    resolved.ParticleFluxBpar_zst,
                    None,
                    None,
                    None,
                    None,
                    False,
                ),
            )
            for (
                prefix,
                kx_arr,
                ky_arr,
                kykx_arr,
                z_arr,
                total_kx,
                total_ky,
                total_kykx,
                total_z,
                fallback_total,
            ) in split_metric_specs:
                use_kx = total_kx if kx_arr is None and fallback_total else kx_arr
                use_ky = total_ky if ky_arr is None and fallback_total else ky_arr
                use_kykx = (
                    total_kykx if kykx_arr is None and fallback_total else kykx_arr
                )
                use_z = total_z if z_arr is None and fallback_total else z_arr
                if use_kx is not None:
                    diag_group.createVariable(
                        f"{prefix}_kxst", "f4", ("time", "s", "kx")
                    )[:, :, :] = _condense_kx_for_output(
                        np.asarray(use_kx, dtype=np.float32),
                        full_nx=full_nx,
                        active_nx=active_nx,
                    )
                if use_ky is not None:
                    diag_group.createVariable(
                        f"{prefix}_kyst", "f4", ("time", "s", "ky")
                    )[:, :, :] = _condense_ky_for_output(
                        np.asarray(use_ky, dtype=np.float32),
                        full_ny=full_ny,
                        active_ny=active_ny,
                    )
                if use_kykx is not None:
                    diag_group.createVariable(
                        f"{prefix}_kxkyst", "f4", ("time", "s", "ky", "kx")
                    )[:, :, :, :] = _condense_kykx_for_output(
                        np.asarray(use_kykx, dtype=np.float32),
                        full_ny=full_ny,
                        full_nx=full_nx,
                        active_ny=active_ny,
                        active_nx=active_nx,
                    )
                if use_z is not None:
                    diag_group.createVariable(
                        f"{prefix}_zst", "f4", ("time", "s", "theta")
                    )[:, :, :] = np.asarray(use_z, dtype=np.float32)
            if resolved.TurbulentHeating_kxst is not None:
                diag_group.createVariable(
                    "TurbulentHeating_kxst", "f4", ("time", "s", "kx")
                )[:, :, :] = _condense_kx_for_output(
                    np.asarray(resolved.TurbulentHeating_kxst, dtype=np.float32),
                    full_nx=full_nx,
                    active_nx=active_nx,
                )
            if resolved.TurbulentHeating_kyst is not None:
                diag_group.createVariable(
                    "TurbulentHeating_kyst", "f4", ("time", "s", "ky")
                )[:, :, :] = _condense_ky_for_output(
                    np.asarray(resolved.TurbulentHeating_kyst, dtype=np.float32),
                    full_ny=full_ny,
                    active_ny=active_ny,
                )
            if resolved.TurbulentHeating_kxkyst is not None:
                diag_group.createVariable(
                    "TurbulentHeating_kxkyst", "f4", ("time", "s", "ky", "kx")
                )[:, :, :, :] = _condense_kykx_for_output(
                    np.asarray(resolved.TurbulentHeating_kxkyst, dtype=np.float32),
                    full_ny=full_ny,
                    full_nx=full_nx,
                    active_ny=active_ny,
                    active_nx=active_nx,
                )
            if resolved.TurbulentHeating_zst is not None:
                diag_group.createVariable(
                    "TurbulentHeating_zst", "f4", ("time", "s", "theta")
                )[:, :, :] = np.asarray(resolved.TurbulentHeating_zst, dtype=np.float32)

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
    "_write_geometry_group",
    "_write_input_parameters_group",
    "_write_nonlinear_netcdf_outputs",
]
