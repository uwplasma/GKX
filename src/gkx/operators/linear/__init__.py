"""Linear operator facade with cycle-free lazy imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "cache_model": ["LinearCache"],
    "cache_builder": ["build_linear_cache", "update_linear_cache_for_sheared_kx"],
    "cache_arrays": [
        "collision_damping",
        "hypercollision_damping",
    ],
    "collision_tables": [
        "EqualSpeciesFiniteWavelengthSugamaOperator",
        "TabulatedMultispeciesCollisionOperator",
        "interpolate_collision_moment_matrix",
    ],
    "collisions": [
        "DriftKineticMomentCollisionOperator",
        "EqualSpeciesFiniteWavelengthCoulombOperator",
        "FiniteWavelengthCoulombOperator",
        "apply_collision_moment_matrix",
        "apply_finite_wavelength_coulomb_moment_operator",
        "apply_multispecies_collision_moment_matrix",
        "assemble_drift_kinetic_improved_sugama_matrix",
        "assemble_drift_kinetic_sugama_matrix",
        "drift_kinetic_improved_sugama_pair_matrices",
        "drift_kinetic_sugama_pair_matrices",
        "interpolate_collision_diagonal_table",
        "interpolate_collision_pair_table",
        "load_collision_moment_matrix",
        "parallel_electric_field_source",
        "solve_driven_collision_response",
    ],
    "moments": [
        "build_H",
        "compute_b",
        "diamagnetic_drive_coeffs",
        "energy_operator",
        "grad_z_periodic",
        "hermite_streaming",
        "lenard_bernstein_eigenvalues",
        "quasineutrality_phi",
        "streaming_term",
    ],
    "params": [
        "COLLISION_OPERATOR_NAMES",
        "LinearParams",
        "LinearTerms",
        "Preconditioner",
        "PreconditionerSpec",
        "collision_operator_from_config",
        "linear_terms_to_term_config",
        "term_config_to_linear_terms",
    ],
    "rhs": ["linear_rhs", "linear_rhs_cached"],
    "streaming": [
        "abs_z_linked_fft",
        "apply_hermite_v",
        "apply_hermite_v2",
        "apply_laguerre_x",
        "grad_z_linked_fft",
        "shift_axis",
        "streaming_ladder_term",
    ],
}
_MODULE_BY_NAME = {
    name: f"gkx.operators.linear.{module}"
    for module, names in _EXPORTS.items()
    for name in names
}
__all__ = list(_MODULE_BY_NAME)


def __getattr__(name: str) -> Any:
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
