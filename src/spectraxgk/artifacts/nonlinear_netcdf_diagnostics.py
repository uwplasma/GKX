"""Diagnostic-history writer for nonlinear NetCDF output bundles."""

from __future__ import annotations

from typing import Any

import numpy as np

from spectraxgk.artifacts.nonlinear_diagnostics import _resolved_species_time
from spectraxgk.artifacts.spectral_layout import (
    _complex_to_ri,
    _condense_kx_for_output,
    _condense_ky_for_output,
    _condense_kykx_for_output,
    _species_matrix,
)


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
        diag_group.createVariable("Phi2_zonal_t", "f4", ("time",))[:] = (
            np.asarray(resolved.Phi2_zonal_t, dtype=np.float32)
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
            _write_resolved_species_spectra(
                diag_group,
                prefix,
                kx_arr=use_kx,
                ky_arr=use_ky,
                kykx_arr=use_kykx,
                z_arr=use_z,
                full_nx=full_nx,
                full_ny=full_ny,
                active_nx=active_nx,
                active_ny=active_ny,
            )
        _write_resolved_species_spectra(
            diag_group,
            "TurbulentHeating",
            kx_arr=resolved.TurbulentHeating_kxst,
            ky_arr=resolved.TurbulentHeating_kyst,
            kykx_arr=resolved.TurbulentHeating_kxkyst,
            z_arr=resolved.TurbulentHeating_zst,
            full_nx=full_nx,
            full_ny=full_ny,
            active_nx=active_nx,
            active_ny=active_ny,
        )



__all__ = ["_write_diagnostics_group"]
