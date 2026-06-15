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
from spectraxgk.grids import (
    build_spectral_grid,
    real_fft_ordered_kx,
    real_fft_unique_ky,
)
from spectraxgk.linear import build_linear_cache
from spectraxgk.runtime import (
    build_runtime_geometry,
    build_runtime_linear_params,
)
from spectraxgk.netcdf_spectral_layout import (
    _complex_to_ri,
    _condense_kx_for_output,
    _condense_ky_for_output,
    _condense_kykx_for_output,
    _dealiased_spectral_field,
    _dealiased_kx_values,
    _dealiased_ky_values,
    _real_space_axis,
    _require_netcdf4,
    _restart_to_netcdf_layout,
    _species_matrix,
    _spectral_species_to_ri,
    _spectral_to_ri,
    _spectral_to_xy,
    _state_basis_moments,
    _write_runtime_root_metadata,
)
from spectraxgk.runtime_artifact_io import (
    _ensure_parent,
    _netcdf_bundle_base,
)
from spectraxgk.runtime_artifact_nonlinear_diagnostics import (
    _resolved_species_time,
    _resolve_restart_path,
)


def _build_output_grid_and_geometry(cfg: Any) -> tuple[Any, Any]:
    """Resolve artifact output onto the same geometry-implied grid as the solver."""

    geom_raw = build_runtime_geometry(cfg)
    grid_cfg = apply_geometry_grid_defaults(geom_raw, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    geom = ensure_flux_tube_geometry_data(geom_raw, grid.z)
    return grid, geom


def _particle_moments(state: np.ndarray, cfg: Any) -> dict[str, np.ndarray]:
    state_arr = np.asarray(state)
    ns, nl, nm, _ny, _nx, _nz = state_arr.shape
    grid, geom = _build_output_grid_and_geometry(cfg)
    params = build_runtime_linear_params(cfg, Nm=nm, geom=geom)
    cache = build_linear_cache(grid, geom, params, nl, nm)
    Jl = np.asarray(cache.Jl)
    JlB = np.asarray(cache.JlB)
    if Jl.ndim == 4:
        Jl = Jl[None, ...]
    if JlB.ndim == 4:
        JlB = JlB[None, ...]
    sqrt_b = np.sqrt(np.maximum(np.asarray(cache.kperp2, dtype=np.float32), 0.0))
    g0 = (
        state_arr[:, :, 0, ...]
        if nm >= 1
        else np.zeros((ns, nl) + state_arr.shape[3:], dtype=state_arr.dtype)
    )
    g1 = state_arr[:, :, 1, ...] if nm >= 2 else np.zeros_like(g0)
    g2 = state_arr[:, :, 2, ...] if nm >= 3 else np.zeros_like(g0)
    particle_density = np.sum(Jl * g0, axis=1)
    particle_upar = np.sum(Jl * g1, axis=1)
    particle_uperp = sqrt_b[None, ...] * np.sum(JlB * g0, axis=1)
    particle_temp = np.sqrt(2.0, dtype=np.float32) * np.sum(Jl * g2, axis=1)
    return {
        "ParticleDensity": particle_density,
        "ParticleUpar": particle_upar,
        "ParticleUperp": particle_uperp,
        "ParticleTemp": particle_temp,
    }


def _write_geometry_group(
    group: Any,
    cfg: Any,
    *,
    grid: Any | None = None,
    geom: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    if grid is None or geom is None:
        grid, geom = _build_output_grid_and_geometry(cfg)
    theta = np.asarray(grid.z, dtype=np.float32)
    group.createVariable("bmag", "f4", ("theta",))[:] = np.asarray(
        geom.bmag_profile, dtype=np.float32
    )
    group.createVariable("bgrad", "f4", ("theta",))[:] = np.asarray(
        geom.bgrad_profile, dtype=np.float32
    )
    group.createVariable("gbdrift", "f4", ("theta",))[:] = np.asarray(
        geom.gb_profile, dtype=np.float32
    )
    group.createVariable("gbdrift0", "f4", ("theta",))[:] = np.asarray(
        geom.gb0_profile, dtype=np.float32
    )
    group.createVariable("cvdrift", "f4", ("theta",))[:] = np.asarray(
        geom.cv_profile, dtype=np.float32
    )
    group.createVariable("cvdrift0", "f4", ("theta",))[:] = np.asarray(
        geom.cv0_profile, dtype=np.float32
    )
    group.createVariable("gds2", "f4", ("theta",))[:] = np.asarray(
        geom.gds2_profile, dtype=np.float32
    )
    group.createVariable("gds21", "f4", ("theta",))[:] = np.asarray(
        geom.gds21_profile, dtype=np.float32
    )
    group.createVariable("gds22", "f4", ("theta",))[:] = np.asarray(
        geom.gds22_profile, dtype=np.float32
    )
    group.createVariable("grho", "f4", ("theta",))[:] = np.asarray(
        geom.grho_profile, dtype=np.float32
    )
    group.createVariable("jacobian", "f4", ("theta",))[:] = np.asarray(
        geom.jacobian_profile, dtype=np.float32
    )
    group.createVariable("gradpar", "f4", ())[:] = np.float32(geom.gradpar_value)
    group.createVariable("nperiod", "i4", ())[:] = np.int32(
        cfg.grid.nperiod if cfg.grid.nperiod is not None else 1
    )
    group.createVariable("q", "f4", ())[:] = np.float32(geom.q)
    group.createVariable("shat", "f4", ())[:] = np.float32(geom.s_hat)
    group.createVariable("shift", "f4", ())[:] = np.float32(
        getattr(cfg.geometry, "shift", 0.0)
    )
    group.createVariable("rmaj", "f4", ())[:] = np.float32(geom.R0)
    group.createVariable("aminor", "f4", ())[:] = np.float32(geom.epsilon * geom.R0)
    group.createVariable("kxfac", "f4", ())[:] = np.float32(geom.kxfac)
    group.createVariable("drhodpsi", "f4", ())[:] = np.float32(1.0)
    group.createVariable("theta_scale", "f4", ())[:] = np.float32(geom.theta_scale)
    group.createVariable("nfp", "i4", ())[:] = np.int32(geom.nfp)
    group.createVariable("alpha", "f4", ())[:] = np.float32(geom.alpha)
    group.createVariable("zeta_center", "f4", ())[:] = np.float32(0.0)
    return (
        theta,
        np.asarray(real_fft_ordered_kx(grid.kx), dtype=np.float32),
        np.asarray(real_fft_unique_ky(grid.ky), dtype=np.float32),
        geom,
    )


def _write_input_parameters_group(group: Any, cfg: Any, geom: Any) -> None:
    group.createVariable("igeo", "i4", ())[:] = np.int32(
        0 if str(cfg.geometry.model).lower() == "miller" else 1
    )
    group.createVariable("slab", "i4", ())[:] = np.int32(
        1 if str(cfg.geometry.model).lower() == "slab" else 0
    )
    group.createVariable("const_curv", "i4", ())[:] = np.int32(0)
    group.createVariable("geofile_dum", "i4", ())[:] = np.int32(
        1 if getattr(cfg.geometry, "geometry_file", None) else 0
    )
    group.createVariable("drhodpsi", "f4", ())[:] = np.float32(1.0)
    group.createVariable("kxfac", "f4", ())[:] = np.float32(geom.kxfac)
    group.createVariable("Rmaj", "f4", ())[:] = np.float32(geom.R0)
    group.createVariable("shift", "f4", ())[:] = np.float32(
        getattr(cfg.geometry, "shift", 0.0)
    )
    group.createVariable("eps", "f4", ())[:] = np.float32(geom.epsilon)
    group.createVariable("q", "f4", ())[:] = np.float32(geom.q)
    group.createVariable("shat", "f4", ())[:] = np.float32(geom.s_hat)
    group.createVariable("kappa", "f4", ())[:] = np.float32(
        getattr(cfg.geometry, "kappa", 1.0)
    )
    group.createVariable("kappa_prime", "f4", ())[:] = np.float32(
        getattr(cfg.geometry, "akappri", 0.0)
    )
    group.createVariable("tri", "f4", ())[:] = np.float32(
        getattr(cfg.geometry, "tri", 0.0)
    )
    group.createVariable("tri_prime", "f4", ())[:] = np.float32(
        getattr(cfg.geometry, "tripri", 0.0)
    )
    group.createVariable("beta", "f4", ())[:] = np.float32(cfg.physics.beta)
    group.createVariable("zero_shat", "i4", ())[:] = np.int32(
        abs(float(geom.s_hat)) < 1.0e-30
    )
    group.createVariable("B_ref", "f4", ())[:] = np.float32(geom.B0)
    group.createVariable("a_ref", "f4", ())[:] = np.float32(
        max(float(geom.epsilon * geom.R0), 1.0)
    )
    group.createVariable("grhoavg", "f4", ())[:] = np.float32(
        np.mean(np.asarray(geom.grho_profile, dtype=np.float32))
    )
    group.createVariable("surfarea", "f4", ())[:] = np.float32(np.nan)


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

    if result.state is not None:
        restart_state_layout = _restart_to_netcdf_layout(np.asarray(result.state))
        _ensure_parent(restart_path)
        with Dataset(restart_path, "w") as root:
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
        paths["restart"] = str(restart_path)

    if result.fields is not None:
        _ensure_parent(big_path)
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
        with Dataset(big_path, "w") as root:
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
        paths["big"] = str(big_path)

    return paths


__all__ = [
    "_build_output_grid_and_geometry",
    "_particle_moments",
    "_write_geometry_group",
    "_write_input_parameters_group",
    "_write_nonlinear_netcdf_outputs",
]
