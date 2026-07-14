"""Linear operator facade with cycle-free lazy imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "cache_model": ["LinearCache"],
    "cache_builder": ["build_linear_cache", "update_linear_cache_for_sheared_kx"],
    "cache_arrays": [
        "_build_end_damping_profile_array",
        "_build_gyroaverage_cache_arrays",
        "_build_low_rank_moment_cache_arrays",
        "_numpy_dtype_for_jax",
        "apply_collision_moment_matrix",
        "apply_multispecies_collision_moment_matrix",
        "collision_damping",
        "drift_kinetic_sugama_pair_matrices",
        "hypercollision_damping",
        "interpolate_collision_moment_matrix",
        "load_collision_moment_matrix",
    ],
    "linked": [
        "_build_linked_end_damping_profile",
        "_build_linked_fft_maps",
        "_signed_to_index",
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
        "LinearParams",
        "LinearTerms",
        "Preconditioner",
        "PreconditionerSpec",
        "_as_species_array",
        "_check_nonnegative",
        "_check_positive",
        "_is_tracer",
        "_resolve_implicit_preconditioner",
        "_x64_enabled",
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
    name: f"spectraxgk.operators.linear.{module}"
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
