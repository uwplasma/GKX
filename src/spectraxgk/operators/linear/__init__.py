"""Linear operator kernels and cache-building helpers."""

from __future__ import annotations

from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.cache_builder import build_linear_cache
from spectraxgk.operators.linear.cache_arrays import (
    _build_end_damping_profile_array,
    _build_gyroaverage_cache_arrays,
    _build_low_rank_moment_cache_arrays,
    _numpy_dtype_for_jax,
    collision_damping,
    hypercollision_damping,
)
from spectraxgk.operators.linear.linked import (
    _build_linked_end_damping_profile,
    _build_linked_fft_maps,
    _signed_to_index,
)
from spectraxgk.operators.linear.moments import (
    apply_hermite_v,
    apply_hermite_v2,
    apply_laguerre_x,
    build_H,
    compute_b,
    diamagnetic_drive_coeffs,
    energy_operator,
    grad_z_periodic,
    hermite_streaming,
    lenard_bernstein_eigenvalues,
    quasineutrality_phi,
    shift_axis,
    streaming_term,
)
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    Preconditioner,
    PreconditionerSpec,
    _as_species_array,
    _check_nonnegative,
    _check_positive,
    _is_tracer,
    _resolve_implicit_preconditioner,
    _x64_enabled,
    linear_terms_to_term_config,
    term_config_to_linear_terms,
)
from spectraxgk.operators.linear.rhs import linear_rhs, linear_rhs_cached

__all__ = [
    "LinearCache",
    "LinearParams",
    "LinearTerms",
    "Preconditioner",
    "PreconditionerSpec",
    "_as_species_array",
    "_build_end_damping_profile_array",
    "_build_gyroaverage_cache_arrays",
    "_build_linked_end_damping_profile",
    "_build_linked_fft_maps",
    "_build_low_rank_moment_cache_arrays",
    "_check_nonnegative",
    "_check_positive",
    "_is_tracer",
    "_numpy_dtype_for_jax",
    "_resolve_implicit_preconditioner",
    "_signed_to_index",
    "_x64_enabled",
    "apply_hermite_v",
    "apply_hermite_v2",
    "apply_laguerre_x",
    "build_H",
    "build_linear_cache",
    "collision_damping",
    "compute_b",
    "diamagnetic_drive_coeffs",
    "energy_operator",
    "grad_z_periodic",
    "hermite_streaming",
    "hypercollision_damping",
    "lenard_bernstein_eigenvalues",
    "linear_terms_to_term_config",
    "linear_rhs",
    "linear_rhs_cached",
    "quasineutrality_phi",
    "shift_axis",
    "streaming_term",
    "term_config_to_linear_terms",
]
