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

    gamma_modes, omega_modes = kernels.instantaneous_growth_rate_step(
        phi, phi_prev_step, dt_step, z_index=z_idx, mask=mask
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
        phi_mode = phi[ky_i, kx_i, z_idx]
    else:
        gamma = jnp.nan_to_num(
            jnp.nanmean(jnp.where(mask, gamma_modes, jnp.nan)),
            nan=jnp.asarray(0.0, dtype=real_dtype),
        )
        omega = jnp.nan_to_num(
            jnp.nanmean(jnp.where(mask, omega_modes, jnp.nan)),
            nan=jnp.asarray(0.0, dtype=real_dtype),
        )
        phi_mode = jnp.asarray(0.0 + 0.0j, dtype=phi.dtype)
    nspecies = int(G_state.shape[0]) if G_state.ndim == 6 else 1
    if not resolved_diagnostics:
        Wg_val = kernels.distribution_free_energy(
            G_state, grid, params, vol_fac, use_dealias=use_dealias
        )
        Wphi_val = kernels.electrostatic_field_energy(
            phi,
            cache,
            params,
            vol_fac,
            use_dealias=use_dealias,
            wphi_scale=wphi_scale,
        )
        Wapar_val = kernels.magnetic_vector_potential_energy(
            apar, cache, vol_fac, use_dealias=use_dealias
        )
        heat_species = kernels.heat_flux_species(
            G_state,
            phi,
            apar,
            bpar,
            cache,
            grid,
            params,
            flux_fac,
            use_dealias=use_dealias,
            flux_scale=flux_scale,
        )
        pflux_species = kernels.particle_flux_species(
            G_state,
            phi,
            apar,
            bpar,
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
            phi,
            apar,
            bpar,
            phi_prev_step,
            apar_prev_step,
            bpar_prev_step,
            cache,
            grid,
            params,
            vol_fac,
            dt_step,
            use_dealias=use_dealias,
        )
        heat_val = jnp.sum(heat_species)
        pflux_val = jnp.sum(pflux_species)
        turbulent_heat_val = jnp.sum(turbulent_heat_species)
        return (
            gamma,
            omega,
            Wg_val,
            Wphi_val,
            Wapar_val,
            heat_val,
            pflux_val,
            turbulent_heat_val,
            heat_species,
            pflux_species,
            turbulent_heat_species,
            phi_mode,
            (),
        )
    (
        phi2_val,
        phi2_kxt,
        phi2_kyt,
        phi2_kxkyt,
        phi2_zt,
        phi2_zonal_t,
        phi2_zonal_kxt,
        phi2_zonal_zt,
    ) = kernels.phi2_resolved(phi, grid, vol_fac, use_dealias=use_dealias)
    phi_zonal_mode_kxt = kernels.zonal_phi_mode_kxt(phi, grid, vol_fac)
    phi_zonal_line_kxt = kernels.zonal_phi_line_kxt(phi, grid)
    Wg_st, Wg_kxst, Wg_kyst, Wg_kxkyst, Wg_zst, Wg_lmst = (
        kernels.distribution_free_energy_resolved(
            G_state,
            grid,
            params,
            vol_fac,
            use_dealias=use_dealias,
        )
    )
    Wphi_st, Wphi_kxst, Wphi_kyst, Wphi_kxkyst, Wphi_zst = (
        kernels.electrostatic_field_energy_resolved(
            phi,
            cache,
            params,
            vol_fac,
            use_dealias=use_dealias,
            wphi_scale=wphi_scale,
        )
    )
    Wapar_st, Wapar_kxst, Wapar_kyst, Wapar_kxkyst, Wapar_zst = (
        kernels.magnetic_vector_potential_energy_resolved(
            apar,
            cache,
            vol_fac,
            nspecies=nspecies,
            use_dealias=use_dealias,
        )
    )
    heat_species, HeatFlux_kxst, HeatFlux_kyst, HeatFlux_kxkyst, HeatFlux_zst = (
        kernels.heat_flux_resolved_species(
            G_state,
            phi,
            apar,
            bpar,
            cache,
            grid,
            params,
            flux_fac,
            use_dealias=use_dealias,
            flux_scale=flux_scale,
        )
    )
    heat_es, heat_apar, heat_bpar = kernels.heat_flux_channel_resolved_species(
        G_state,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    (
        pflux_species,
        ParticleFlux_kxst,
        ParticleFlux_kyst,
        ParticleFlux_kxkyst,
        ParticleFlux_zst,
    ) = kernels.particle_flux_resolved_species(
        G_state,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    pflux_es, pflux_apar, pflux_bpar = kernels.particle_flux_channel_resolved_species(
        G_state,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    (
        turbulent_heat_species,
        TurbulentHeating_kxst,
        TurbulentHeating_kyst,
        TurbulentHeating_kxkyst,
        TurbulentHeating_zst,
    ) = kernels.turbulent_heating_resolved_species(
        G_state,
        G_prev_step,
        phi,
        apar,
        bpar,
        phi_prev_step,
        apar_prev_step,
        bpar_prev_step,
        cache,
        grid,
        params,
        vol_fac,
        dt_step,
        use_dealias=use_dealias,
    )
    Wg_val = jnp.sum(Wg_st)
    Wphi_val = jnp.sum(Wphi_st)
    Wapar_val = jnp.sum(Wapar_st)
    heat_val = jnp.sum(heat_species)
    pflux_val = jnp.sum(pflux_species)
    turbulent_heat_val = jnp.sum(turbulent_heat_species)
    return (
        gamma,
        omega,
        Wg_val,
        Wphi_val,
        Wapar_val,
        heat_val,
        pflux_val,
        turbulent_heat_val,
        heat_species,
        pflux_species,
        turbulent_heat_species,
        phi_mode,
        (
            phi2_kxt,
            phi2_kyt,
            phi2_kxkyt,
            phi2_zt,
            phi2_zonal_t,
            phi2_zonal_kxt,
            phi2_zonal_zt,
            phi_zonal_mode_kxt,
            phi_zonal_line_kxt,
            Wg_kxst,
            Wg_kyst,
            Wg_kxkyst,
            Wg_zst,
            Wg_lmst,
            Wphi_kxst,
            Wphi_kyst,
            Wphi_kxkyst,
            Wphi_zst,
            Wapar_kxst,
            Wapar_kyst,
            Wapar_kxkyst,
            Wapar_zst,
            HeatFlux_kxst,
            HeatFlux_kyst,
            HeatFlux_kxkyst,
            HeatFlux_zst,
            heat_es[1],
            heat_es[2],
            heat_es[3],
            heat_es[4],
            heat_apar[1],
            heat_apar[2],
            heat_apar[3],
            heat_apar[4],
            heat_bpar[1],
            heat_bpar[2],
            heat_bpar[3],
            heat_bpar[4],
            ParticleFlux_kxst,
            ParticleFlux_kyst,
            ParticleFlux_kxkyst,
            ParticleFlux_zst,
            pflux_es[1],
            pflux_es[2],
            pflux_es[3],
            pflux_es[4],
            pflux_apar[1],
            pflux_apar[2],
            pflux_apar[3],
            pflux_apar[4],
            pflux_bpar[1],
            pflux_bpar[2],
            pflux_bpar[3],
            pflux_bpar[4],
            TurbulentHeating_kxst,
            TurbulentHeating_kyst,
            TurbulentHeating_kxkyst,
            TurbulentHeating_zst,
        ),
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
