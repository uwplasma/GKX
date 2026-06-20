"""Diagnostic tuple assembly for nonlinear time integration.

The nonlinear integrators keep their public facade in ``spectraxgk.nonlinear``.
This module owns the pure state-to-diagnostics assembly logic, with diagnostic
kernels injected by the facade so tests and interactive debugging can still
monkeypatch the public module-level functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from spectraxgk.core.grid import SpectralGrid
from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.terms.config import FieldState


@dataclass(frozen=True)
class NonlinearDiagnosticKernels:
    """Facade-injected diagnostic callables used by nonlinear integrators."""

    instantaneous_growth_rate_step: Callable[..., Any]
    phi2_resolved: Callable[..., Any]
    zonal_phi_mode_kxt: Callable[..., Any]
    zonal_phi_line_kxt: Callable[..., Any]
    distribution_free_energy: Callable[..., Any]
    distribution_free_energy_resolved: Callable[..., Any]
    electrostatic_field_energy: Callable[..., Any]
    electrostatic_field_energy_resolved: Callable[..., Any]
    magnetic_vector_potential_energy: Callable[..., Any]
    magnetic_vector_potential_energy_resolved: Callable[..., Any]
    heat_flux_species: Callable[..., Any]
    heat_flux_resolved_species: Callable[..., Any]
    heat_flux_channel_resolved_species: Callable[..., Any]
    particle_flux_species: Callable[..., Any]
    particle_flux_resolved_species: Callable[..., Any]
    particle_flux_channel_resolved_species: Callable[..., Any]
    turbulent_heating_species: Callable[..., Any]
    turbulent_heating_resolved_species: Callable[..., Any]


@dataclass(frozen=True)
class _DiagnosticFieldPair:
    phi: jnp.ndarray
    apar: jnp.ndarray
    bpar: jnp.ndarray
    phi_prev_step: jnp.ndarray
    apar_prev_step: jnp.ndarray
    bpar_prev_step: jnp.ndarray


@dataclass(frozen=True)
class _ResolvedFieldGroups:
    phi2: tuple[Any, ...]
    phi_zonal_mode_kxt: Any
    phi_zonal_line_kxt: Any
    free_energy: tuple[Any, ...]
    electrostatic_energy: tuple[Any, ...]
    magnetic_energy: tuple[Any, ...]


@dataclass(frozen=True)
class _ResolvedTransportGroups:
    heat_flux: tuple[Any, ...]
    heat_channels: tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]
    particle_flux: tuple[Any, ...]
    particle_channels: tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]
    turbulent_heating: tuple[Any, ...]


def _diagnostic_field_pair(
    fields_state: FieldState, fields_prev_step: FieldState
) -> _DiagnosticFieldPair:
    """Return present/previous fields with disabled EM components zero-filled."""

    phi = fields_state.phi
    apar = fields_state.apar if fields_state.apar is not None else jnp.zeros_like(phi)
    bpar = fields_state.bpar if fields_state.bpar is not None else jnp.zeros_like(phi)
    phi_prev_step = fields_prev_step.phi
    apar_prev_step = (
        fields_prev_step.apar
        if fields_prev_step.apar is not None
        else jnp.zeros_like(phi_prev_step)
    )
    bpar_prev_step = (
        fields_prev_step.bpar
        if fields_prev_step.bpar is not None
        else jnp.zeros_like(phi_prev_step)
    )
    return _DiagnosticFieldPair(
        phi=phi,
        apar=apar,
        bpar=bpar,
        phi_prev_step=phi_prev_step,
        apar_prev_step=apar_prev_step,
        bpar_prev_step=bpar_prev_step,
    )


def _mode_growth_frequency(
    fields: _DiagnosticFieldPair,
    dt_step: jnp.ndarray,
    *,
    mask: jnp.ndarray,
    z_idx: int,
    omega_ky_index: int | None,
    omega_kx_index: int | None,
    real_dtype: Any,
    kernels: NonlinearDiagnosticKernels,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return monitored nonlinear growth, frequency, and optional mode value."""

    gamma_modes, omega_modes = kernels.instantaneous_growth_rate_step(
        fields.phi, fields.phi_prev_step, dt_step, z_index=z_idx, mask=mask
    )
    if omega_ky_index is not None:
        ky_i = int(np.clip(omega_ky_index, 0, int(gamma_modes.shape[0]) - 1))
        kx_i = int(np.clip(omega_kx_index or 0, 0, int(gamma_modes.shape[1]) - 1))
        gamma = jnp.nan_to_num(
            gamma_modes[ky_i, kx_i], nan=jnp.asarray(0.0, dtype=real_dtype)
        )
        omega = jnp.nan_to_num(
            omega_modes[ky_i, kx_i], nan=jnp.asarray(0.0, dtype=real_dtype)
        )
        phi_mode = fields.phi[ky_i, kx_i, z_idx]
    else:
        gamma = jnp.nan_to_num(
            jnp.nanmean(jnp.where(mask, gamma_modes, jnp.nan)),
            nan=jnp.asarray(0.0, dtype=real_dtype),
        )
        omega = jnp.nan_to_num(
            jnp.nanmean(jnp.where(mask, omega_modes, jnp.nan)),
            nan=jnp.asarray(0.0, dtype=real_dtype),
        )
        phi_mode = jnp.asarray(0.0 + 0.0j, dtype=fields.phi.dtype)
    return gamma, omega, phi_mode


def _compute_unresolved_diagnostic_tuple(
    G_state: jnp.ndarray,
    fields: _DiagnosticFieldPair,
    G_prev_step: jnp.ndarray,
    dt_step: jnp.ndarray,
    *,
    gamma: jnp.ndarray,
    omega: jnp.ndarray,
    phi_mode: jnp.ndarray,
    grid: SpectralGrid,
    cache: LinearCache,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    flux_fac: jnp.ndarray,
    use_dealias: bool,
    flux_scale: float,
    wphi_scale: float,
    kernels: NonlinearDiagnosticKernels,
) -> tuple[Any, ...]:
    """Build scalar nonlinear diagnostics when resolved spectra are disabled."""

    Wg_val = kernels.distribution_free_energy(
        G_state, grid, params, vol_fac, use_dealias=use_dealias
    )
    Wphi_val = kernels.electrostatic_field_energy(
        fields.phi,
        cache,
        params,
        vol_fac,
        use_dealias=use_dealias,
        wphi_scale=wphi_scale,
    )
    Wapar_val = kernels.magnetic_vector_potential_energy(
        fields.apar, cache, vol_fac, use_dealias=use_dealias
    )
    heat_species = kernels.heat_flux_species(
        G_state,
        fields.phi,
        fields.apar,
        fields.bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    pflux_species = kernels.particle_flux_species(
        G_state,
        fields.phi,
        fields.apar,
        fields.bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    turbulent_heat_species = kernels.turbulent_heating_species(
        G_state,
        G_prev_step,
        fields.phi,
        fields.apar,
        fields.bpar,
        fields.phi_prev_step,
        fields.apar_prev_step,
        fields.bpar_prev_step,
        cache,
        grid,
        params,
        vol_fac,
        dt_step,
        use_dealias=use_dealias,
    )
    return (
        gamma,
        omega,
        Wg_val,
        Wphi_val,
        Wapar_val,
        jnp.sum(heat_species),
        jnp.sum(pflux_species),
        jnp.sum(turbulent_heat_species),
        heat_species,
        pflux_species,
        turbulent_heat_species,
        phi_mode,
        (),
    )


def _compute_resolved_field_groups(
    G_state: jnp.ndarray,
    fields: _DiagnosticFieldPair,
    *,
    grid: SpectralGrid,
    cache: LinearCache,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    nspecies: int,
    use_dealias: bool,
    wphi_scale: float,
    kernels: NonlinearDiagnosticKernels,
) -> _ResolvedFieldGroups:
    """Evaluate resolved field-energy and potential spectra."""

    return _ResolvedFieldGroups(
        phi2=tuple(
            kernels.phi2_resolved(fields.phi, grid, vol_fac, use_dealias=use_dealias)
        ),
        phi_zonal_mode_kxt=kernels.zonal_phi_mode_kxt(fields.phi, grid, vol_fac),
        phi_zonal_line_kxt=kernels.zonal_phi_line_kxt(fields.phi, grid),
        free_energy=tuple(
            kernels.distribution_free_energy_resolved(
                G_state,
                grid,
                params,
                vol_fac,
                use_dealias=use_dealias,
            )
        ),
        electrostatic_energy=tuple(
            kernels.electrostatic_field_energy_resolved(
                fields.phi,
                cache,
                params,
                vol_fac,
                use_dealias=use_dealias,
                wphi_scale=wphi_scale,
            )
        ),
        magnetic_energy=tuple(
            kernels.magnetic_vector_potential_energy_resolved(
                fields.apar,
                cache,
                vol_fac,
                nspecies=nspecies,
                use_dealias=use_dealias,
            )
        ),
    )


def _as_three_channel_tuple(
    channels: Any,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    """Normalize ES/Apar/Bpar kernel output to a typed tuple."""

    es, apar, bpar = channels
    return tuple(es), tuple(apar), tuple(bpar)


def _compute_resolved_transport_groups(
    G_state: jnp.ndarray,
    fields: _DiagnosticFieldPair,
    G_prev_step: jnp.ndarray,
    dt_step: jnp.ndarray,
    *,
    grid: SpectralGrid,
    cache: LinearCache,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    vol_fac: jnp.ndarray,
    use_dealias: bool,
    flux_scale: float,
    kernels: NonlinearDiagnosticKernels,
) -> _ResolvedTransportGroups:
    """Evaluate resolved heat, particle, and turbulent-heating spectra."""

    heat_channels = kernels.heat_flux_channel_resolved_species(
        G_state,
        fields.phi,
        fields.apar,
        fields.bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    particle_channels = kernels.particle_flux_channel_resolved_species(
        G_state,
        fields.phi,
        fields.apar,
        fields.bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    turbulent_heating = kernels.turbulent_heating_resolved_species(
        G_state,
        G_prev_step,
        fields.phi,
        fields.apar,
        fields.bpar,
        fields.phi_prev_step,
        fields.apar_prev_step,
        fields.bpar_prev_step,
        cache,
        grid,
        params,
        vol_fac,
        dt_step,
        use_dealias=use_dealias,
    )
    return _ResolvedTransportGroups(
        heat_flux=tuple(
            kernels.heat_flux_resolved_species(
                G_state,
                fields.phi,
                fields.apar,
                fields.bpar,
                cache,
                grid,
                params,
                flux_fac,
                use_dealias=use_dealias,
                flux_scale=flux_scale,
            )
        ),
        heat_channels=_as_three_channel_tuple(heat_channels),
        particle_flux=tuple(
            kernels.particle_flux_resolved_species(
                G_state,
                fields.phi,
                fields.apar,
                fields.bpar,
                cache,
                grid,
                params,
                flux_fac,
                use_dealias=use_dealias,
                flux_scale=flux_scale,
            )
        ),
        particle_channels=_as_three_channel_tuple(particle_channels),
        turbulent_heating=tuple(turbulent_heating),
    )


def _channel_resolved_tail(
    channels: tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]],
) -> tuple[Any, ...]:
    """Return ES/Apar/Bpar channel spectra after the species scalar slot."""

    es, apar, bpar = channels
    return (*es[1:5], *apar[1:5], *bpar[1:5])


def _pack_resolved_schema(
    field_groups: _ResolvedFieldGroups,
    transport_groups: _ResolvedTransportGroups,
) -> tuple[Any, ...]:
    """Pack resolved spectra in the NetCDF/diagnostic metadata schema order."""

    return (
        *field_groups.phi2[1:8],
        field_groups.phi_zonal_mode_kxt,
        field_groups.phi_zonal_line_kxt,
        *field_groups.free_energy[1:6],
        *field_groups.electrostatic_energy[1:5],
        *field_groups.magnetic_energy[1:5],
        *transport_groups.heat_flux[1:5],
        *_channel_resolved_tail(transport_groups.heat_channels),
        *transport_groups.particle_flux[1:5],
        *_channel_resolved_tail(transport_groups.particle_channels),
        *transport_groups.turbulent_heating[1:5],
    )


def _compute_resolved_diagnostic_tuple(
    G_state: jnp.ndarray,
    fields: _DiagnosticFieldPair,
    G_prev_step: jnp.ndarray,
    dt_step: jnp.ndarray,
    *,
    gamma: jnp.ndarray,
    omega: jnp.ndarray,
    phi_mode: jnp.ndarray,
    grid: SpectralGrid,
    cache: LinearCache,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    flux_fac: jnp.ndarray,
    nspecies: int,
    use_dealias: bool,
    flux_scale: float,
    wphi_scale: float,
    kernels: NonlinearDiagnosticKernels,
) -> tuple[Any, ...]:
    """Build scalar plus resolved nonlinear diagnostics in schema order."""

    field_groups = _compute_resolved_field_groups(
        G_state,
        fields,
        grid=grid,
        cache=cache,
        params=params,
        vol_fac=vol_fac,
        nspecies=nspecies,
        use_dealias=use_dealias,
        wphi_scale=wphi_scale,
        kernels=kernels,
    )
    transport_groups = _compute_resolved_transport_groups(
        G_state,
        fields,
        G_prev_step,
        dt_step,
        grid=grid,
        cache=cache,
        params=params,
        flux_fac=flux_fac,
        vol_fac=vol_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
        kernels=kernels,
    )
    return (
        gamma,
        omega,
        jnp.sum(field_groups.free_energy[0]),
        jnp.sum(field_groups.electrostatic_energy[0]),
        jnp.sum(field_groups.magnetic_energy[0]),
        jnp.sum(transport_groups.heat_flux[0]),
        jnp.sum(transport_groups.particle_flux[0]),
        jnp.sum(transport_groups.turbulent_heating[0]),
        transport_groups.heat_flux[0],
        transport_groups.particle_flux[0],
        transport_groups.turbulent_heating[0],
        phi_mode,
        _pack_resolved_schema(field_groups, transport_groups),
    )


def compute_nonlinear_diagnostic_tuple(
    G_state: jnp.ndarray,
    fields_state: FieldState,
    G_prev_step: jnp.ndarray,
    fields_prev_step: FieldState,
    dt_step: jnp.ndarray,
    *,
    grid: SpectralGrid,
    cache: LinearCache,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    flux_fac: jnp.ndarray,
    mask: jnp.ndarray,
    z_idx: int,
    use_dealias: bool,
    real_dtype: Any,
    omega_ky_index: int | None,
    omega_kx_index: int | None,
    flux_scale: float,
    wphi_scale: float,
    resolved_diagnostics: bool,
    kernels: NonlinearDiagnosticKernels,
) -> tuple[Any, ...]:
    """Build the nonlinear scan diagnostic tuple for one state."""

    fields = _diagnostic_field_pair(fields_state, fields_prev_step)
    gamma, omega, phi_mode = _mode_growth_frequency(
        fields,
        dt_step,
        mask=mask,
        z_idx=z_idx,
        omega_ky_index=omega_ky_index,
        omega_kx_index=omega_kx_index,
        real_dtype=real_dtype,
        kernels=kernels,
    )
    if not resolved_diagnostics:
        return _compute_unresolved_diagnostic_tuple(
            G_state,
            fields,
            G_prev_step,
            dt_step,
            gamma=gamma,
            omega=omega,
            phi_mode=phi_mode,
            grid=grid,
            cache=cache,
            params=params,
            vol_fac=vol_fac,
            flux_fac=flux_fac,
            use_dealias=use_dealias,
            flux_scale=flux_scale,
            wphi_scale=wphi_scale,
            kernels=kernels,
        )
    nspecies = int(G_state.shape[0]) if G_state.ndim == 6 else 1
    return _compute_resolved_diagnostic_tuple(
        G_state,
        fields,
        G_prev_step,
        dt_step,
        gamma=gamma,
        omega=omega,
        phi_mode=phi_mode,
        grid=grid,
        cache=cache,
        params=params,
        vol_fac=vol_fac,
        flux_fac=flux_fac,
        nspecies=nspecies,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
        wphi_scale=wphi_scale,
        kernels=kernels,
    )


def make_nonlinear_diagnostic_tuple_fn(
    *,
    grid: SpectralGrid,
    cache: LinearCache,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    flux_fac: jnp.ndarray,
    mask: jnp.ndarray,
    z_idx: int,
    use_dealias: bool,
    real_dtype: Any,
    omega_ky_index: int | None,
    omega_kx_index: int | None,
    flux_scale: float,
    wphi_scale: float,
    resolved_diagnostics: bool,
    kernels: NonlinearDiagnosticKernels,
) -> Callable[[jnp.ndarray, FieldState, jnp.ndarray, FieldState, jnp.ndarray], tuple[Any, ...]]:
    """Return a reusable state-to-diagnostic tuple closure for scan policies."""

    def compute_diag_from_state(
        G_state: jnp.ndarray,
        fields_state: FieldState,
        G_prev_step: jnp.ndarray,
        fields_prev_step: FieldState,
        dt_step: jnp.ndarray,
    ) -> tuple[Any, ...]:
        return compute_nonlinear_diagnostic_tuple(
            G_state,
            fields_state,
            G_prev_step,
            fields_prev_step,
            dt_step,
            grid=grid,
            cache=cache,
            params=params,
            vol_fac=vol_fac,
            flux_fac=flux_fac,
            mask=mask,
            z_idx=z_idx,
            use_dealias=use_dealias,
            real_dtype=real_dtype,
            omega_ky_index=omega_ky_index,
            omega_kx_index=omega_kx_index,
            flux_scale=flux_scale,
            wphi_scale=wphi_scale,
            resolved_diagnostics=resolved_diagnostics,
            kernels=kernels,
        )

    return compute_diag_from_state


__all__ = [
    "NonlinearDiagnosticKernels",
    "compute_nonlinear_diagnostic_tuple",
    "make_nonlinear_diagnostic_tuple_fn",
]
