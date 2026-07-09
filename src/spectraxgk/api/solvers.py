"""Public solvers API exports."""

from spectraxgk.diagnostics.quasilinear_transport import (
    QuasilinearTransportResult,
    compute_quasilinear_from_linear_state,
    effective_kperp2,
    mixing_length_amplitude2_jax,
    phi_norm2,
    quasilinear_feature_objective,
    saturation_amplitude2,
    saturated_flux_from_linear_weight,
    shape_aware_power_law_objective,
)
from spectraxgk.operators import hermite_streaming
from spectraxgk.linear import (
    LinearCache,
    LinearParams,
    LinearTerms,
    build_linear_cache,
    integrate_linear,
    linear_terms_to_term_config,
    linear_rhs,
    linear_rhs_cached,
    linear_rhs_electrostatic_slices_velocity_sharded,
    linear_rhs_streaming_electrostatic_velocity_sharded,
    linear_rhs_parallel_cached,
    linear_rhs_streaming_velocity_sharded,
    term_config_to_linear_terms,
)
from spectraxgk.solvers.linear.krylov import (
    KrylovConfig,
    dominant_eigenpair,
    dominant_eigenvalue,
)
from spectraxgk.nonlinear import (
    IMEXLinearOperator,
    build_nonlinear_imex_operator,
    integrate_nonlinear,
    integrate_nonlinear_cached,
    integrate_nonlinear_explicit_diagnostics,
    nonlinear_rhs_cached,
)
from spectraxgk.core.species import Species, build_linear_params
from spectraxgk.solvers.time.diffrax_linear import integrate_linear_diffrax
from spectraxgk.solvers.time.diffrax_nonlinear import integrate_nonlinear_diffrax
from spectraxgk.solvers.time.diffrax_streaming import integrate_linear_diffrax_streaming
from spectraxgk.parallel.integrators import (
    integrate_linear_sharded,
    integrate_nonlinear_sharded,
)
from spectraxgk.solvers.time.explicit import (
    ExplicitTimeConfig,
    integrate_linear_explicit,
    integrate_linear_explicit_diagnostics,
)

__all__ = [
    "QuasilinearTransportResult",
    "compute_quasilinear_from_linear_state",
    "effective_kperp2",
    "mixing_length_amplitude2_jax",
    "phi_norm2",
    "quasilinear_feature_objective",
    "saturation_amplitude2",
    "saturated_flux_from_linear_weight",
    "shape_aware_power_law_objective",
    "hermite_streaming",
    "LinearParams",
    "LinearTerms",
    "LinearCache",
    "build_linear_cache",
    "linear_terms_to_term_config",
    "term_config_to_linear_terms",
    "linear_rhs",
    "linear_rhs_cached",
    "linear_rhs_electrostatic_slices_velocity_sharded",
    "linear_rhs_streaming_electrostatic_velocity_sharded",
    "linear_rhs_parallel_cached",
    "linear_rhs_streaming_velocity_sharded",
    "integrate_linear",
    "KrylovConfig",
    "dominant_eigenpair",
    "dominant_eigenvalue",
    "integrate_linear_diffrax",
    "integrate_linear_diffrax_streaming",
    "integrate_linear_sharded",
    "integrate_nonlinear_sharded",
    "integrate_nonlinear",
    "integrate_nonlinear_cached",
    "integrate_nonlinear_explicit_diagnostics",
    "integrate_nonlinear_diffrax",
    "build_nonlinear_imex_operator",
    "IMEXLinearOperator",
    "ExplicitTimeConfig",
    "integrate_linear_explicit",
    "integrate_linear_explicit_diagnostics",
    "nonlinear_rhs_cached",
    "Species",
    "build_linear_params",
]
