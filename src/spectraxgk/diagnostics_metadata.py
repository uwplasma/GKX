"""Diagnostic container dataclasses for GX-style runs."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


ArrayLike = jnp.ndarray | np.ndarray


@dataclass(frozen=True)
class ResolvedDiagnostics:
    """Optional resolved nonlinear diagnostics stored per sample."""

    Phi2_kxt: ArrayLike | None = None
    Phi2_kyt: ArrayLike | None = None
    Phi2_kxkyt: ArrayLike | None = None
    Phi2_zt: ArrayLike | None = None
    Phi2_zonal_t: ArrayLike | None = None
    Phi2_zonal_kxt: ArrayLike | None = None
    Phi2_zonal_zt: ArrayLike | None = None
    Phi_zonal_mode_kxt: ArrayLike | None = None
    Phi_zonal_line_kxt: ArrayLike | None = None
    Wg_kxst: ArrayLike | None = None
    Wg_kyst: ArrayLike | None = None
    Wg_kxkyst: ArrayLike | None = None
    Wg_zst: ArrayLike | None = None
    Wg_lmst: ArrayLike | None = None
    Wphi_kxst: ArrayLike | None = None
    Wphi_kyst: ArrayLike | None = None
    Wphi_kxkyst: ArrayLike | None = None
    Wphi_zst: ArrayLike | None = None
    Wapar_kxst: ArrayLike | None = None
    Wapar_kyst: ArrayLike | None = None
    Wapar_kxkyst: ArrayLike | None = None
    Wapar_zst: ArrayLike | None = None
    HeatFlux_kxst: ArrayLike | None = None
    HeatFlux_kyst: ArrayLike | None = None
    HeatFlux_kxkyst: ArrayLike | None = None
    HeatFlux_zst: ArrayLike | None = None
    HeatFluxES_kxst: ArrayLike | None = None
    HeatFluxES_kyst: ArrayLike | None = None
    HeatFluxES_kxkyst: ArrayLike | None = None
    HeatFluxES_zst: ArrayLike | None = None
    HeatFluxApar_kxst: ArrayLike | None = None
    HeatFluxApar_kyst: ArrayLike | None = None
    HeatFluxApar_kxkyst: ArrayLike | None = None
    HeatFluxApar_zst: ArrayLike | None = None
    HeatFluxBpar_kxst: ArrayLike | None = None
    HeatFluxBpar_kyst: ArrayLike | None = None
    HeatFluxBpar_kxkyst: ArrayLike | None = None
    HeatFluxBpar_zst: ArrayLike | None = None
    ParticleFlux_kxst: ArrayLike | None = None
    ParticleFlux_kyst: ArrayLike | None = None
    ParticleFlux_kxkyst: ArrayLike | None = None
    ParticleFlux_zst: ArrayLike | None = None
    ParticleFluxES_kxst: ArrayLike | None = None
    ParticleFluxES_kyst: ArrayLike | None = None
    ParticleFluxES_kxkyst: ArrayLike | None = None
    ParticleFluxES_zst: ArrayLike | None = None
    ParticleFluxApar_kxst: ArrayLike | None = None
    ParticleFluxApar_kyst: ArrayLike | None = None
    ParticleFluxApar_kxkyst: ArrayLike | None = None
    ParticleFluxApar_zst: ArrayLike | None = None
    ParticleFluxBpar_kxst: ArrayLike | None = None
    ParticleFluxBpar_kyst: ArrayLike | None = None
    ParticleFluxBpar_kxkyst: ArrayLike | None = None
    ParticleFluxBpar_zst: ArrayLike | None = None
    TurbulentHeating_kxst: ArrayLike | None = None
    TurbulentHeating_kyst: ArrayLike | None = None
    TurbulentHeating_kxkyst: ArrayLike | None = None
    TurbulentHeating_zst: ArrayLike | None = None


@dataclass(frozen=True)
class SimulationDiagnostics:
    """Streaming diagnostics at each sample time."""

    t: ArrayLike
    dt_t: ArrayLike
    dt_mean: ArrayLike
    gamma_t: ArrayLike
    omega_t: ArrayLike
    Wg_t: ArrayLike
    Wphi_t: ArrayLike
    Wapar_t: ArrayLike
    heat_flux_t: ArrayLike
    particle_flux_t: ArrayLike
    energy_t: ArrayLike
    heat_flux_species_t: ArrayLike | None = None
    particle_flux_species_t: ArrayLike | None = None
    turbulent_heating_t: ArrayLike | None = None
    turbulent_heating_species_t: ArrayLike | None = None
    phi_mode_t: ArrayLike | None = None
    resolved: ResolvedDiagnostics | None = None


# Compatibility aliases preserved while the broader rename propagates through
# the comparison and audit tooling.
GXResolvedDiagnostics = ResolvedDiagnostics
GXDiagnostics = SimulationDiagnostics
