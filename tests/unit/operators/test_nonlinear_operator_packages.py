from __future__ import annotations

import gkx.operators as operators
import gkx.operators.linear as linear_operators
import gkx.operators.nonlinear as nonlinear_operators
import gkx.operators.nonlinear.diagnostic_state as operator_diagnostics
import gkx.operators.nonlinear.rhs as operator_rhs
import gkx.solvers.nonlinear as nonlinear_solvers
import gkx.solvers.nonlinear.diagnostics as solver_diagnostics
import gkx.solvers.nonlinear.explicit as solver_explicit
import gkx.solvers.nonlinear.imex as solver_imex


def test_operator_package_preserves_public_linear_export_identity() -> None:
    assert operators.hermite_streaming is linear_operators.hermite_streaming


def test_nonlinear_operator_package_reexports_rhs_implementation() -> None:
    assert nonlinear_operators.RhsCallable is operator_rhs.RhsCallable
    assert (
        nonlinear_operators.linear_rhs_jit_for_terms_impl
        is operator_rhs.linear_rhs_jit_for_terms_impl
    )
    assert (
        nonlinear_operators.nonlinear_rhs_cached_impl
        is operator_rhs.nonlinear_rhs_cached_impl
    )
    assert (
        nonlinear_operators.nonlinear_em_term_cached_impl
        is operator_rhs.nonlinear_em_term_cached_impl
    )


def test_nonlinear_operator_package_reexports_diagnostic_implementation() -> None:
    assert nonlinear_operators.NonlinearDiagnosticKernels is (
        operator_diagnostics.NonlinearDiagnosticKernels
    )
    assert (
        nonlinear_operators.compute_nonlinear_diagnostic_tuple
        is operator_diagnostics.compute_nonlinear_diagnostic_tuple
    )
    assert (
        nonlinear_operators.make_nonlinear_diagnostic_tuple_fn
        is operator_diagnostics.make_nonlinear_diagnostic_tuple_fn
    )


def test_nonlinear_solver_package_reexports_implementations() -> None:
    assert (
        nonlinear_solvers.advance_explicit_nonlinear_state
        is solver_explicit.advance_explicit_nonlinear_state
    )
    assert (
        nonlinear_solvers.checkpoint_explicit_step
        is solver_explicit.checkpoint_explicit_step
    )
    assert (
        nonlinear_solvers.integrate_cached_explicit_scan
        is solver_explicit.integrate_cached_explicit_scan
    )
    assert (
        nonlinear_solvers.make_explicit_diagnostic_step
        is solver_explicit.make_explicit_diagnostic_step
    )
    assert (
        nonlinear_solvers.run_explicit_diagnostic_scan
        is solver_explicit.run_explicit_diagnostic_scan
    )
    assert (
        nonlinear_solvers.ExplicitNonlinearDiagnosticsDeps
        is solver_diagnostics.ExplicitNonlinearDiagnosticsDeps
    )
    assert (
        nonlinear_solvers.integrate_explicit_nonlinear_diagnostics_impl
        is solver_diagnostics.integrate_explicit_nonlinear_diagnostics_impl
    )
    assert (
        nonlinear_solvers.IMEXNonlinearDiagnosticsDeps
        is solver_diagnostics.IMEXNonlinearDiagnosticsDeps
    )
    assert (
        nonlinear_solvers.integrate_imex_nonlinear_diagnostics_impl
        is solver_diagnostics.integrate_imex_nonlinear_diagnostics_impl
    )
    assert (
        nonlinear_solvers.advance_imex_nonlinear_state
        is solver_imex.advance_imex_nonlinear_state
    )
    assert (
        nonlinear_solvers.imex_fixed_point_guess is solver_imex.imex_fixed_point_guess
    )
    assert (
        nonlinear_solvers.integrate_cached_imex_scan
        is solver_imex.integrate_cached_imex_scan
    )
    assert (
        nonlinear_solvers.make_imex_diagnostic_step
        is solver_imex.make_imex_diagnostic_step
    )
    assert (
        nonlinear_solvers.make_imex_nonlinear_term
        is solver_imex.make_imex_nonlinear_term
    )
    assert nonlinear_solvers.make_imex_solve_step is solver_imex.make_imex_solve_step
    assert (
        nonlinear_solvers.run_imex_diagnostic_scan
        is solver_imex.run_imex_diagnostic_scan
    )
    assert nonlinear_solvers.solve_imex_step is solver_imex.solve_imex_step
