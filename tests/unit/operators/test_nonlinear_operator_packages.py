from __future__ import annotations

import gkx.operators as operators
import gkx.operators.linear as linear_operators
import gkx.operators.nonlinear as nonlinear_operators
import gkx.operators.nonlinear.diagnostic_state as operator_diagnostics
import gkx.operators.nonlinear.rhs as operator_rhs
from gkx.terms.assembly import linear_rhs_jit_for_terms


def test_operator_package_preserves_public_linear_export_identity() -> None:
    assert operators.hermite_streaming is linear_operators.hermite_streaming


def test_nonlinear_operator_package_reexports_rhs_implementation() -> None:
    assert nonlinear_operators.RhsCallable is operator_rhs.RhsCallable
    assert (
        nonlinear_operators.linear_rhs_jit_for_terms_impl
        is operator_rhs.linear_rhs_jit_for_terms_impl
        is linear_rhs_jit_for_terms
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
