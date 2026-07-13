"""Linear electrostatic gyrokinetic building blocks (Hermite-Laguerre)."""

from __future__ import annotations

from spectraxgk.operators.linear.linked import (
    _build_linked_end_damping_profile,  # noqa: F401 - linear API helper re-export
    _build_linked_fft_maps,  # noqa: F401 - linear API helper re-export
    _signed_to_index,  # noqa: F401 - linear API helper re-export
)
from spectraxgk.operators.linear.moments import (
    apply_hermite_v,  # noqa: F401 - linear API helper re-export
    apply_hermite_v2,  # noqa: F401 - linear API helper re-export
    apply_laguerre_x,  # noqa: F401 - linear API helper re-export
    build_H,  # noqa: F401 - linear API helper re-export
    compute_b,  # noqa: F401 - linear API helper re-export
    diamagnetic_drive_coeffs,  # noqa: F401 - linear API helper re-export
    energy_operator,  # noqa: F401 - linear API helper re-export
    grad_z_periodic,  # noqa: F401 - linear API helper re-export
    hermite_streaming,  # noqa: F401 - linear API helper re-export
    lenard_bernstein_eigenvalues,  # noqa: F401 - linear API helper re-export
    quasineutrality_phi,  # noqa: F401 - linear API helper re-export
    shift_axis,  # noqa: F401 - linear API helper re-export
    streaming_term,  # noqa: F401 - linear API helper re-export
)
from spectraxgk.operators.linear.cache_model import (
    LinearCache,  # noqa: F401 - linear API type re-export
)
from spectraxgk.operators.linear.cache_builder import (
    build_linear_cache,  # noqa: F401 - linear API helper re-export
)
from spectraxgk.operators.linear.cache_arrays import (
    _build_end_damping_profile_array,  # noqa: F401 - linear API helper re-export
    _build_gyroaverage_cache_arrays,  # noqa: F401 - linear API helper re-export
    _build_low_rank_moment_cache_arrays,  # noqa: F401 - linear API helper re-export
    _numpy_dtype_for_jax,  # noqa: F401 - linear API helper re-export
    collision_damping,  # noqa: F401 - linear API helper re-export
    hypercollision_damping,  # noqa: F401 - linear API helper re-export
)
from spectraxgk.operators.linear.params import (
    LinearParams,  # noqa: F401 - linear API type re-export
    LinearTerms,  # noqa: F401 - linear API type re-export
    Preconditioner,  # noqa: F401 - linear API type alias re-export
    PreconditionerSpec,  # noqa: F401 - linear API type alias re-export
    _as_species_array,  # noqa: F401 - linear API helper re-export
    _check_nonnegative,  # noqa: F401 - linear API helper re-export
    _check_positive,  # noqa: F401 - linear API helper re-export
    _is_tracer,  # noqa: F401 - linear API helper re-export
    _resolve_implicit_preconditioner,  # noqa: F401 - linear API helper re-export
    _x64_enabled,  # noqa: F401 - linear API helper re-export
    linear_terms_to_term_config,  # noqa: F401 - linear API helper re-export
    term_config_to_linear_terms,  # noqa: F401 - linear API helper re-export
)
from spectraxgk.operators.linear.rhs import (
    linear_rhs,  # noqa: F401 - linear API helper re-export
    linear_rhs_cached,  # noqa: F401 - linear API helper re-export
)
from spectraxgk.solvers.linear.implicit import (
    _build_implicit_operator,  # noqa: F401 - linear API helper re-export
    _integrate_linear_implicit_cached,  # noqa: F401 - linear API helper re-export
)
from spectraxgk.solvers.linear.parallel import (
    _FUSED_ELECTROSTATIC_SLICE_KERNEL_CACHE,  # noqa: F401 - linear API helper re-export
    _electrostatic_streaming_field_rhs,  # noqa: F401 - linear API helper re-export
    _is_electrostatic_field_terms,  # noqa: F401 - linear API helper re-export
    _is_electrostatic_slice_terms,  # noqa: F401 - linear API helper re-export
    _is_streaming_only_terms,  # noqa: F401 - linear API helper re-export
    _linear_rhs_electrostatic_slices_velocity_sharded_fused,  # noqa: F401 - linear API helper re-export
    _resolve_parallel_devices,  # noqa: F401 - linear API helper re-export
    _streaming_electrostatic_from_phi_velocity_sharded,  # noqa: F401 - linear API helper re-export
    linear_rhs_electrostatic_species_sharded,  # noqa: F401 - linear API helper re-export
    linear_rhs_electrostatic_slices_velocity_sharded,  # noqa: F401 - linear API helper re-export
    linear_rhs_parallel_cached,  # noqa: F401 - linear API helper re-export
    linear_rhs_streaming_electrostatic_velocity_sharded,  # noqa: F401 - linear API helper re-export
    linear_rhs_electrostatic_species_hermite_sharded,  # noqa: F401 - linear API helper re-export
    linear_rhs_streaming_velocity_sharded,  # noqa: F401 - linear API helper re-export
    prepare_electrostatic_species_inputs,  # noqa: F401 - linear API helper re-export
)


from spectraxgk.solvers.linear.integrators import (
    _integrate_linear_cached,  # noqa: F401 - linear API helper re-export
    _integrate_linear_cached_donate,  # noqa: F401 - linear API helper re-export
    _integrate_linear_cached_impl,  # noqa: F401 - linear API helper re-export
    integrate_linear,  # noqa: F401 - linear API helper re-export
    integrate_linear_diagnostics,  # noqa: F401 - linear API helper re-export
)
