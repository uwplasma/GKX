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



def _species_history_matrices(diag: Any, nspecies: int) -> dict[str, np.ndarray]:
    """Return base ``(time, species)`` histories for the diagnostics group."""

    return {
        "Wg_st": _species_matrix(np.asarray(diag.Wg_t, dtype=np.float32), nspecies, None),
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


def _resolved_species_history(resolved: Any, attr: str, fallback: np.ndarray) -> np.ndarray:
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
        ("HeatFluxES_st", "HeatFluxES_kxst", heat_s if cfg.physics.electrostatic else zero_heat),
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
            arrays = tuple(total if arr is None else arr for arr, total in zip(arrays, totals))
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



__all__ = ["_write_diagnostics_group"]
