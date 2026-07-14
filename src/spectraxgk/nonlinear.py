"""Public nonlinear gyrokinetic facade.

Implementation lives in focused nonlinear core, diagnostic, operator, and solver
modules.  This facade keeps the stable import surface small while making the
owner modules explicit for tests and development.
"""

from __future__ import annotations

from spectraxgk.solvers.nonlinear.state_integration import (
    integrate_nonlinear,
    integrate_nonlinear_cached,
    integrate_nonlinear_imex_cached,
    integrate_nonlinear_sheared,
    integrate_nonlinear_sheared_transport,
    nonlinear_rhs_cached,
    ShearedTransportTrace,
)
from spectraxgk.solvers.nonlinear.diagnostic_integration import (
    PreparedExplicitNonlinearDiagnostics,
    integrate_nonlinear_explicit_diagnostics,
    integrate_nonlinear_explicit_diagnostics_state,
    integrate_nonlinear_imex_diagnostics,
    prepare_nonlinear_explicit_diagnostics,
)
from spectraxgk.operators.nonlinear.diagnostics import (
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
    ShearingCoordinateUpdate,
    advance_shearing_coordinates,
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
    "ShearingCoordinateUpdate",
    "build_nonlinear_collision_split_policy",
    "build_nonlinear_diagnostic_setup",
    "build_nonlinear_imex_operator",
    "build_nonlinear_simulation_diagnostics",
    "build_nonlinear_time_step_policy",
    "advance_shearing_coordinates",
    "finalize_nonlinear_scan_diagnostics",
    "integrate_nonlinear",
    "integrate_nonlinear_cached",
    "integrate_nonlinear_explicit_diagnostics",
    "integrate_nonlinear_explicit_diagnostics_state",
    "integrate_nonlinear_imex_cached",
    "integrate_nonlinear_sheared",
    "integrate_nonlinear_sheared_transport",
    "integrate_nonlinear_imex_diagnostics",
    "maybe_emit_nonlinear_progress",
    "nonlinear_rhs_cached",
    "prepare_nonlinear_explicit_diagnostics",
    "run_sampled_explicit_diagnostic_scan",
    "sampled_scan_intervals",
    "select_nonlinear_step_diagnostics",
    "ShearedTransportTrace",
]
