"""Public linear gyrokinetic operators, caches, and integrators."""

from spectraxgk.operators.linear.cache_arrays import (
    collision_damping,
    hypercollision_damping,
)
from spectraxgk.operators.linear.cache_builder import (
    build_linear_cache,
    update_linear_cache_for_sheared_kx,
)
from spectraxgk.operators.linear.cache_model import LinearCache
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
    linear_terms_to_term_config,
    term_config_to_linear_terms,
)
from spectraxgk.operators.linear.rhs import linear_rhs, linear_rhs_cached
from spectraxgk.solvers.linear.integrators import (
    integrate_linear,
    integrate_linear_diagnostics,
)
from spectraxgk.solvers.linear.parallel import (
    linear_rhs_electrostatic_slices_velocity_sharded,
    linear_rhs_electrostatic_species_hermite_sharded,
    linear_rhs_electrostatic_species_sharded,
    linear_rhs_parallel_cached,
    linear_rhs_streaming_electrostatic_velocity_sharded,
    linear_rhs_streaming_velocity_sharded,
    prepare_electrostatic_species_hermite_state,
    prepare_electrostatic_species_inputs,
)

__all__ = [
    "LinearCache",
    "LinearParams",
    "LinearTerms",
    "Preconditioner",
    "PreconditionerSpec",
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
    "integrate_linear",
    "integrate_linear_diagnostics",
    "lenard_bernstein_eigenvalues",
    "linear_rhs",
    "linear_rhs_cached",
    "linear_rhs_electrostatic_slices_velocity_sharded",
    "linear_rhs_electrostatic_species_hermite_sharded",
    "linear_rhs_electrostatic_species_sharded",
    "linear_rhs_parallel_cached",
    "linear_rhs_streaming_electrostatic_velocity_sharded",
    "linear_rhs_streaming_velocity_sharded",
    "linear_terms_to_term_config",
    "prepare_electrostatic_species_hermite_state",
    "prepare_electrostatic_species_inputs",
    "quasineutrality_phi",
    "shift_axis",
    "streaming_term",
    "term_config_to_linear_terms",
    "update_linear_cache_for_sheared_kx",
]
