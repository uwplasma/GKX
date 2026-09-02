"""Nonlinear solver policies."""

from __future__ import annotations

from gkx.solvers_nonlinear_explicit import (
    advance_explicit_nonlinear_state,
    checkpoint_explicit_step,
    integrate_cached_explicit_scan,
    make_explicit_diagnostic_step,
    run_explicit_diagnostic_scan,
)
from gkx.solvers_nonlinear_diagnostics import (
    ExplicitNonlinearDiagnosticsDeps,
    IMEXNonlinearDiagnosticsDeps,
    integrate_explicit_nonlinear_diagnostics_impl,
    integrate_imex_nonlinear_diagnostics_impl,
)
from gkx.solvers_nonlinear_diagnostic_integration import (
    integrate_nonlinear_explicit_diagnostics,
    integrate_nonlinear_explicit_diagnostics_state,
    integrate_nonlinear_imex_diagnostics,
)
from gkx.solvers_nonlinear_imex import (
    advance_imex_nonlinear_state,
    imex_fixed_point_guess,
    integrate_cached_imex_scan,
    make_imex_diagnostic_step,
    make_imex_nonlinear_term,
    make_imex_solve_step,
    run_imex_diagnostic_scan,
    solve_imex_step,
)
from gkx.solvers_nonlinear_state_integration import (
    DIVERGENCE_KNEE_STEPS,
    integrate_nonlinear,
    integrate_nonlinear_cached,
    integrate_nonlinear_imex_cached,
    integrate_nonlinear_sheared,
    integrate_nonlinear_sheared_transport,
    nonlinear_heat_flux_window,
    nonlinear_rhs_cached,
    ShearedTransportTrace,
)

__all__ = [
    "advance_explicit_nonlinear_state",
    "DIVERGENCE_KNEE_STEPS",
    "advance_imex_nonlinear_state",
    "checkpoint_explicit_step",
    "ExplicitNonlinearDiagnosticsDeps",
    "integrate_explicit_nonlinear_diagnostics_impl",
    "IMEXNonlinearDiagnosticsDeps",
    "integrate_imex_nonlinear_diagnostics_impl",
    "integrate_nonlinear",
    "integrate_nonlinear_cached",
    "integrate_nonlinear_explicit_diagnostics",
    "integrate_nonlinear_explicit_diagnostics_state",
    "integrate_nonlinear_imex_cached",
    "integrate_nonlinear_sheared",
    "integrate_nonlinear_sheared_transport",
    "integrate_nonlinear_imex_diagnostics",
    "integrate_cached_explicit_scan",
    "imex_fixed_point_guess",
    "make_explicit_diagnostic_step",
    "nonlinear_heat_flux_window",
    "nonlinear_rhs_cached",
    "run_explicit_diagnostic_scan",
    "integrate_cached_imex_scan",
    "make_imex_diagnostic_step",
    "make_imex_nonlinear_term",
    "make_imex_solve_step",
    "run_imex_diagnostic_scan",
    "solve_imex_step",
    "ShearedTransportTrace",
]
