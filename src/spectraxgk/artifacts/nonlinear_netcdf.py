"""Write nonlinear NetCDF output, restart, diagnostics, and final-field bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from spectraxgk.artifacts.io import _ensure_parent, _netcdf_bundle_base
from spectraxgk.artifacts.nonlinear_diagnostics import (
    _resolve_restart_path,
    _resolved_species_time,
)
from spectraxgk.artifacts.spectral_layout import (
    _complex_to_ri,
    _condense_kx_for_output,
    _condense_ky_for_output,
    _condense_kykx_for_output,
    _dealiased_kx_values,
    _dealiased_ky_values,
    _dealiased_spectral_field,
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
from spectraxgk.core.grid import (
    build_spectral_grid,
    real_fft_ordered_kx,
    real_fft_unique_ky,
)
from spectraxgk.diagnostics import SimulationDiagnostics
from spectraxgk.geometry import (
    apply_geometry_grid_defaults,
    ensure_flux_tube_geometry_data,
)
from spectraxgk.operators.linear.cache_builder import build_linear_cache
from spectraxgk.runtime import build_runtime_geometry, build_runtime_linear_params


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


def _final_moment_arrays(
    result: Any, cfg: Any
) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
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
    diag_group.createVariable("Apar", "f4", ("time", "ky", "kx", "theta", "ri"))[
        0, ...
    ] = _spectral_to_ri(apar_active)
    diag_group.createVariable("Bpar", "f4", ("time", "ky", "kx", "theta", "ri"))[
        0, ...
    ] = _spectral_to_ri(bpar_active)
    diag_group.createVariable("PhiXY", "f4", ("time", "y", "x", "theta"))[0, ...] = (
        _spectral_to_xy(phi_full)
    )
    diag_group.createVariable("AparXY", "f4", ("time", "y", "x", "theta"))[0, ...] = (
        _spectral_to_xy(apar_full)
    )
    diag_group.createVariable("BparXY", "f4", ("time", "y", "x", "theta"))[0, ...] = (
        _spectral_to_xy(bpar_full)
    )


def _write_moment_diagnostics(
    diag_group: Any, moments: Mapping[str, np.ndarray]
) -> None:
    """Write spectral and real-space species moments to Diagnostics."""

    for name, values in moments.items():
        active = _dealiased_spectral_field(values, ky_axis=1, kx_axis=2)
        diag_group.createVariable(name, "f4", ("time", "s", "ky", "kx", "theta", "ri"))[
            0, ...
        ] = _spectral_species_to_ri(active)
        diag_group.createVariable(f"{name}XY", "f4", ("time", "s", "y", "x", "theta"))[
            0, ...
        ] = np.real(np.fft.ifft2(values, axes=(1, 2))).astype(np.float32, copy=False)


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


def _write_resolved_species_spectra(
    diag_group: Any,
    prefix: str,
    *,
    kx_arr: Any,
    ky_arr: Any,
    kykx_arr: Any,
    z_arr: Any,
    full_nx: int,
    full_ny: int,
    active_nx: int,
    active_ny: int,
) -> None:
    """Write optional ``(time, species, spectral)`` resolved diagnostics."""

    if kx_arr is not None:
        diag_group.createVariable(f"{prefix}_kxst", "f4", ("time", "s", "kx"))[
            :, :, :
        ] = _condense_kx_for_output(
            np.asarray(kx_arr, dtype=np.float32),
            full_nx=full_nx,
            active_nx=active_nx,
        )
    if ky_arr is not None:
        diag_group.createVariable(f"{prefix}_kyst", "f4", ("time", "s", "ky"))[
            :, :, :
        ] = _condense_ky_for_output(
            np.asarray(ky_arr, dtype=np.float32),
            full_ny=full_ny,
            active_ny=active_ny,
        )
    if kykx_arr is not None:
        diag_group.createVariable(f"{prefix}_kxkyst", "f4", ("time", "s", "ky", "kx"))[
            :, :, :, :
        ] = _condense_kykx_for_output(
            np.asarray(kykx_arr, dtype=np.float32),
            full_ny=full_ny,
            full_nx=full_nx,
            active_ny=active_ny,
            active_nx=active_nx,
        )
    if z_arr is not None:
        diag_group.createVariable(f"{prefix}_zst", "f4", ("time", "s", "theta"))[
            :, :, :
        ] = np.asarray(z_arr, dtype=np.float32)


def _phi2_outputs_for_netcdf(
    resolved: Any,
    diag: Any,
    *,
    full_nx: int,
    full_ny: int,
    active_nx: int,
    active_ny: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return mutually consistent Phi2 spectra for the NetCDF schema."""

    phi2_kx_out = None
    phi2_ky_out = None
    phi2_kykx_out = None
    if resolved is not None and resolved.Phi2_kxkyt is not None:
        # The NetCDF schema stores Phi2 on the rFFT-positive ky view. Deriving
        # one-dimensional spectra from the condensed two-dimensional spectrum
        # keeps Phi2_t, Phi2_kxt, and Phi2_kyt mutually consistent when the
        # solver evolved a full Hermitian ky layout internally.
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
    return np.asarray(phi2_t, dtype=np.float32), phi2_kx_out, phi2_ky_out, phi2_kykx_out


def _write_phi2_diagnostics(
    diag_group: Any,
    resolved: Any,
    *,
    phi2_kx_out: np.ndarray | None,
    phi2_ky_out: np.ndarray | None,
    phi2_kykx_out: np.ndarray | None,
) -> None:
    """Write resolved nonlinear potential-energy diagnostics."""

    if resolved is None:
        return
    if phi2_kx_out is not None:
        diag_group.createVariable("Phi2_kxt", "f4", ("time", "kx"))[:, :] = phi2_kx_out
    if phi2_ky_out is not None:
        diag_group.createVariable("Phi2_kyt", "f4", ("time", "ky"))[:, :] = phi2_ky_out
    if phi2_kykx_out is not None:
        diag_group.createVariable("Phi2_kxkyt", "f4", ("time", "ky", "kx"))[:, :, :] = (
            phi2_kykx_out
        )
    if resolved.Phi2_zt is not None:
        diag_group.createVariable("Phi2_zt", "f4", ("time", "theta"))[:, :] = (
            np.asarray(resolved.Phi2_zt, dtype=np.float32)
        )


def _write_zonal_phi_diagnostics(
    diag_group: Any,
    resolved: Any,
    *,
    full_nx: int,
    active_nx: int,
) -> None:
    """Write zonal potential diagnostics and preserve complex-valued fields."""

    if resolved is None:
        return
    if resolved.Phi2_zonal_t is not None:
        diag_group.createVariable("Phi2_zonal_t", "f4", ("time",))[:] = np.asarray(
            resolved.Phi2_zonal_t, dtype=np.float32
        )
    if resolved.Phi2_zonal_kxt is not None:
        diag_group.createVariable("Phi2_zonal_kxt", "f4", ("time", "kx"))[:, :] = (
            _condense_kx_for_output(
                np.asarray(resolved.Phi2_zonal_kxt, dtype=np.float32),
                full_nx=full_nx,
                active_nx=active_nx,
            )
        )
    if resolved.Phi2_zonal_zt is not None:
        diag_group.createVariable("Phi2_zonal_zt", "f4", ("time", "theta"))[:, :] = (
            np.asarray(resolved.Phi2_zonal_zt, dtype=np.float32)
        )
    if resolved.Phi_zonal_mode_kxt is not None:
        diag_group.createVariable("Phi_zonal_mode_kxt", "f4", ("time", "kx", "ri"))[
            :, :, :
        ] = _complex_to_ri(
            _condense_kx_for_output(
                np.asarray(resolved.Phi_zonal_mode_kxt),
                full_nx=full_nx,
                active_nx=active_nx,
            )
        )
    if resolved.Phi_zonal_line_kxt is not None:
        diag_group.createVariable("Phi_zonal_line_kxt", "f4", ("time", "kx", "ri"))[
            :, :, :
        ] = _complex_to_ri(
            _condense_kx_for_output(
                np.asarray(resolved.Phi_zonal_line_kxt),
                full_nx=full_nx,
                active_nx=active_nx,
            )
        )


def _species_history_matrices(diag: Any, nspecies: int) -> dict[str, np.ndarray]:
    """Return base ``(time, species)`` histories for the diagnostics group."""

    return {
        "Wg_st": _species_matrix(
            np.asarray(diag.Wg_t, dtype=np.float32), nspecies, None
        ),
        "Wphi_st": _species_matrix(
            np.asarray(diag.Wphi_t, dtype=np.float32), nspecies, None
        ),
        "Wapar_st": _species_matrix(
            np.asarray(diag.Wapar_t, dtype=np.float32), nspecies, None
        ),
        "HeatFlux_st": _species_matrix(
            np.asarray(diag.heat_flux_t, dtype=np.float32),
            nspecies,
            None
            if diag.heat_flux_species_t is None
            else np.asarray(diag.heat_flux_species_t, dtype=np.float32),
        ),
        "ParticleFlux_st": _species_matrix(
            np.asarray(diag.particle_flux_t, dtype=np.float32),
            nspecies,
            None
            if diag.particle_flux_species_t is None
            else np.asarray(diag.particle_flux_species_t, dtype=np.float32),
        ),
        "TurbulentHeating_st": _species_matrix(
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
        ),
    }


def _write_species_history_variables(
    diag_group: Any, histories: dict[str, np.ndarray]
) -> None:
    for name, values in histories.items():
        if name == "TurbulentHeating_st":
            continue
        diag_group.createVariable(name, "f4", ("time", "s"))[:, :] = values


def _resolved_species_history(
    resolved: Any, attr: str, fallback: np.ndarray
) -> np.ndarray:
    return _resolved_species_time(
        None if resolved is None else getattr(resolved, attr),
        fallback=fallback,
    )


def _write_split_species_histories(
    diag_group: Any,
    resolved: Any,
    cfg: Any,
    histories: dict[str, np.ndarray],
) -> None:
    heat_s = histories["HeatFlux_st"]
    pflux_s = histories["ParticleFlux_st"]
    turb_heat_s = histories["TurbulentHeating_st"]
    zero_heat = np.zeros_like(heat_s)
    zero_pflux = np.zeros_like(pflux_s)
    split_specs = (
        (
            "HeatFluxES_st",
            "HeatFluxES_kxst",
            heat_s if cfg.physics.electrostatic else zero_heat,
        ),
        ("HeatFluxApar_st", "HeatFluxApar_kxst", zero_heat),
        ("HeatFluxBpar_st", "HeatFluxBpar_kxst", zero_heat),
        (
            "ParticleFluxES_st",
            "ParticleFluxES_kxst",
            pflux_s if cfg.physics.electrostatic else zero_pflux,
        ),
        ("ParticleFluxApar_st", "ParticleFluxApar_kxst", zero_pflux),
        ("ParticleFluxBpar_st", "ParticleFluxBpar_kxst", zero_pflux),
        ("TurbulentHeating_st", "TurbulentHeating_kxst", turb_heat_s),
    )
    for output_name, resolved_attr, fallback in split_specs:
        diag_group.createVariable(output_name, "f4", ("time", "s"))[:, :] = (
            _resolved_species_history(resolved, resolved_attr, fallback)
        )


def _resolved_spectrum_arrays(resolved: Any, prefix: str) -> tuple[Any, Any, Any, Any]:
    return (
        getattr(resolved, f"{prefix}_kxst"),
        getattr(resolved, f"{prefix}_kyst"),
        getattr(resolved, f"{prefix}_kxkyst"),
        getattr(resolved, f"{prefix}_zst"),
    )


def _write_named_resolved_spectra(
    diag_group: Any,
    prefix: str,
    arrays: tuple[Any, Any, Any, Any],
    *,
    full_nx: int,
    full_ny: int,
    active_nx: int,
    active_ny: int,
) -> None:
    kx_arr, ky_arr, kykx_arr, z_arr = arrays
    _write_resolved_species_spectra(
        diag_group,
        prefix,
        kx_arr=kx_arr,
        ky_arr=ky_arr,
        kykx_arr=kykx_arr,
        z_arr=z_arr,
        full_nx=full_nx,
        full_ny=full_ny,
        active_nx=active_nx,
        active_ny=active_ny,
    )


def _write_primary_resolved_spectra(
    diag_group: Any,
    resolved: Any,
    *,
    full_nx: int,
    full_ny: int,
    active_nx: int,
    active_ny: int,
) -> None:
    for prefix in ("Wg", "Wphi", "Wapar", "HeatFlux", "ParticleFlux"):
        _write_named_resolved_spectra(
            diag_group,
            prefix,
            _resolved_spectrum_arrays(resolved, prefix),
            full_nx=full_nx,
            full_ny=full_ny,
            active_nx=active_nx,
            active_ny=active_ny,
        )
    if resolved.Wg_lmst is not None:
        diag_group.createVariable("Wg_lmst", "f4", ("time", "s", "m", "l"))[
            :, :, :, :
        ] = np.asarray(resolved.Wg_lmst, dtype=np.float32)


def _write_split_resolved_spectra(
    diag_group: Any,
    resolved: Any,
    cfg: Any,
    *,
    full_nx: int,
    full_ny: int,
    active_nx: int,
    active_ny: int,
) -> None:
    split_specs = (
        ("HeatFluxES", "HeatFlux", cfg.physics.electrostatic),
        ("HeatFluxApar", None, False),
        ("HeatFluxBpar", None, False),
        ("ParticleFluxES", "ParticleFlux", cfg.physics.electrostatic),
        ("ParticleFluxApar", None, False),
        ("ParticleFluxBpar", None, False),
        ("TurbulentHeating", None, False),
    )
    for prefix, total_prefix, fallback_total in split_specs:
        arrays = _resolved_spectrum_arrays(resolved, prefix)
        if fallback_total and total_prefix is not None:
            totals = _resolved_spectrum_arrays(resolved, total_prefix)
            arrays = tuple(
                total if arr is None else arr for arr, total in zip(arrays, totals)
            )
        _write_named_resolved_spectra(
            diag_group,
            prefix,
            arrays,
            full_nx=full_nx,
            full_ny=full_ny,
            active_nx=active_nx,
            active_ny=active_ny,
        )


def _write_diagnostics_group(
    root: Any,
    diag: Any,
    cfg: Any,
    *,
    nspecies: int,
    full_nx: int,
    full_ny: int,
    active_nx: int,
    active_ny: int,
) -> None:
    """Write scalar, species, resolved, and split nonlinear diagnostics."""
    diag_group = root.createGroup("Diagnostics")
    resolved = diag.resolved
    phi2_t, phi2_kx_out, phi2_ky_out, phi2_kykx_out = _phi2_outputs_for_netcdf(
        resolved,
        diag,
        full_nx=full_nx,
        full_ny=full_ny,
        active_nx=active_nx,
        active_ny=active_ny,
    )
    diag_group.createVariable("Phi2_t", "f4", ("time",))[:] = phi2_t
    histories = _species_history_matrices(diag, nspecies)
    _write_species_history_variables(diag_group, histories)
    _write_split_species_histories(diag_group, resolved, cfg, histories)
    if resolved is None:
        return
    _write_phi2_diagnostics(
        diag_group,
        resolved,
        phi2_kx_out=phi2_kx_out,
        phi2_ky_out=phi2_ky_out,
        phi2_kykx_out=phi2_kykx_out,
    )
    _write_zonal_phi_diagnostics(
        diag_group,
        resolved,
        full_nx=full_nx,
        active_nx=active_nx,
    )
    _write_primary_resolved_spectra(
        diag_group,
        resolved,
        full_nx=full_nx,
        full_ny=full_ny,
        active_nx=active_nx,
        active_ny=active_ny,
    )
    _write_split_resolved_spectra(
        diag_group,
        resolved,
        cfg,
        full_nx=full_nx,
        full_ny=full_ny,
        active_nx=active_nx,
        active_ny=active_ny,
    )


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
