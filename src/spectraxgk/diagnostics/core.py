"""Simulation diagnostics public facade."""

from __future__ import annotations

from spectraxgk.diagnostics.channels import (
    _heat_flux_channel_contrib_species,
    _particle_flux_channel_contrib_species,
    _turbulent_heating_contrib_species,
)
from spectraxgk.diagnostics.energy import (
    _masked_abs2,
    distribution_free_energy,
    electrostatic_field_energy,
    magnetic_vector_potential_energy,
    runtime_energy_total,
    total_energy,
)
from spectraxgk.diagnostics.metadata import (
    ArrayLike,
    ResolvedDiagnostics,
    SimulationDiagnostics,
)
from spectraxgk.diagnostics.resolved import (
    _reduce_scalar_kykxz,
    _reduce_species_kykxz,
    distribution_free_energy_resolved,
    electrostatic_field_energy_resolved,
    heat_flux_channel_resolved_species,
    heat_flux_resolved_species,
    magnetic_vector_potential_energy_resolved,
    particle_flux_channel_resolved_species,
    particle_flux_resolved_species,
    phi2_resolved,
    turbulent_heating_resolved_species,
    zonal_phi_line_kxt,
    zonal_phi_mode_kxt,
)
from spectraxgk.diagnostics.transport import (
    heat_flux_channel_species,
    heat_flux_species,
    heat_flux_total,
    particle_flux_channel_species,
    particle_flux_species,
    particle_flux_total,
    turbulent_heating_species,
    turbulent_heating_total,
)
from spectraxgk.diagnostics.weights import (
    _cached_hermitian_mode_weight,
    _hermitian_mode_weight,
    _jl_family,
    _species_array,
    _transport_mode_weight,
    fieldline_quadrature_weights,
)

__all__ = [
    "ArrayLike",
    "ResolvedDiagnostics",
    "SimulationDiagnostics",
    "_cached_hermitian_mode_weight",
    "_heat_flux_channel_contrib_species",
    "_hermitian_mode_weight",
    "_jl_family",
    "_masked_abs2",
    "_particle_flux_channel_contrib_species",
    "_reduce_scalar_kykxz",
    "_reduce_species_kykxz",
    "_species_array",
    "_transport_mode_weight",
    "_turbulent_heating_contrib_species",
    "distribution_free_energy",
    "distribution_free_energy_resolved",
    "electrostatic_field_energy",
    "electrostatic_field_energy_resolved",
    "fieldline_quadrature_weights",
    "heat_flux_channel_resolved_species",
    "heat_flux_channel_species",
    "heat_flux_resolved_species",
    "heat_flux_species",
    "heat_flux_total",
    "magnetic_vector_potential_energy",
    "magnetic_vector_potential_energy_resolved",
    "particle_flux_channel_resolved_species",
    "particle_flux_channel_species",
    "particle_flux_resolved_species",
    "particle_flux_species",
    "particle_flux_total",
    "phi2_resolved",
    "runtime_energy_total",
    "total_energy",
    "turbulent_heating_resolved_species",
    "turbulent_heating_species",
    "turbulent_heating_total",
    "zonal_phi_line_kxt",
    "zonal_phi_mode_kxt",
]
