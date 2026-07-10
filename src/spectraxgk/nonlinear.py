"""Public nonlinear gyrokinetic facade.

Implementation lives in focused nonlinear core, diagnostic, operator, and solver
modules.  This facade keeps the stable import surface small while making the
owner modules explicit for tests and development.
"""

from __future__ import annotations

from spectraxgk.solvers.nonlinear.state_integration import (
    _linear_rhs_jit_for_terms,
    integrate_nonlinear,
    integrate_nonlinear_cached,
    integrate_nonlinear_imex_cached,
    nonlinear_rhs_cached,
)
from spectraxgk.solvers.nonlinear.diagnostic_integration import (
    _EXPLICIT_DIAGNOSTIC_OPTION_KEYS,
    _IMEX_DIAGNOSTIC_OPTION_KEYS,
    _explicit_nonlinear_diagnostics_deps,
    _imex_nonlinear_diagnostics_deps,
    _integrate_nonlinear_explicit_diagnostics_impl,
    _nonlinear_diagnostic_kernels,
    _options_from_scope,
    PreparedExplicitNonlinearDiagnostics,
    integrate_nonlinear_explicit_diagnostics,
    integrate_nonlinear_explicit_diagnostics_state,
    integrate_nonlinear_imex_diagnostics,
    prepare_nonlinear_explicit_diagnostics,
)
from spectraxgk.operators.nonlinear.diagnostics import (
    _pack_resolved_diagnostics,
    _sample_axis0,
    _sample_indices_with_final,
    build_nonlinear_simulation_diagnostics,
    finalize_nonlinear_scan_diagnostics,
    maybe_emit_nonlinear_progress,
    run_sampled_explicit_diagnostic_scan,
    sampled_scan_intervals,
    select_nonlinear_step_diagnostics,
)
from spectraxgk.operators.nonlinear.policies import (
    IMEXLinearOperator,
    NonlinearCollisionSplitPolicy,
    NonlinearDiagnosticSetup,
    NonlinearTimeStepPolicy,
    _apply_collision_split,
    _collision_damping,
    _diagnostic_omega_mode_mask,
    _make_fixed_mode_projector,
    _make_hermitian_projector,
    _make_nonlinear_state_projector,
    _nonlinear_cfl_frequency_components,
    build_nonlinear_collision_split_policy,
    build_nonlinear_diagnostic_setup,
    build_nonlinear_imex_operator,
    build_nonlinear_time_step_policy,
)

__all__ = [
    "IMEXLinearOperator",
    "NonlinearCollisionSplitPolicy",
    "NonlinearDiagnosticSetup",
    "NonlinearTimeStepPolicy",
    "PreparedExplicitNonlinearDiagnostics",
    "_EXPLICIT_DIAGNOSTIC_OPTION_KEYS",
    "_IMEX_DIAGNOSTIC_OPTION_KEYS",
    "_apply_collision_split",
    "_collision_damping",
    "_diagnostic_omega_mode_mask",
    "_explicit_nonlinear_diagnostics_deps",
    "_imex_nonlinear_diagnostics_deps",
    "_integrate_nonlinear_explicit_diagnostics_impl",
    "_linear_rhs_jit_for_terms",
    "_make_fixed_mode_projector",
    "_make_hermitian_projector",
    "_make_nonlinear_state_projector",
    "_nonlinear_cfl_frequency_components",
    "_nonlinear_diagnostic_kernels",
    "_options_from_scope",
    "_pack_resolved_diagnostics",
    "_sample_axis0",
    "_sample_indices_with_final",
    "build_nonlinear_collision_split_policy",
    "build_nonlinear_diagnostic_setup",
    "build_nonlinear_imex_operator",
    "build_nonlinear_simulation_diagnostics",
    "build_nonlinear_time_step_policy",
    "finalize_nonlinear_scan_diagnostics",
    "integrate_nonlinear",
    "integrate_nonlinear_cached",
    "integrate_nonlinear_explicit_diagnostics",
    "integrate_nonlinear_explicit_diagnostics_state",
    "integrate_nonlinear_imex_cached",
    "integrate_nonlinear_imex_diagnostics",
    "maybe_emit_nonlinear_progress",
    "nonlinear_rhs_cached",
    "prepare_nonlinear_explicit_diagnostics",
    "run_sampled_explicit_diagnostic_scan",
    "sampled_scan_intervals",
    "select_nonlinear_step_diagnostics",
]
