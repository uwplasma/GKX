"""Public nonlinear operators, loaded lazily to avoid assembly import cycles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from gkx.operators.nonlinear.diagnostic_state import (
        NonlinearDiagnosticKernels,
        compute_nonlinear_diagnostic_tuple,
        make_nonlinear_diagnostic_tuple_fn,
    )
    from gkx.operators.nonlinear.rhs import (
        RhsCallable,
        linear_rhs_jit_for_terms_impl,
        nonlinear_em_term_cached_impl,
        nonlinear_rhs_cached_impl,
    )

__all__ = [
    "NonlinearDiagnosticKernels",
    "RhsCallable",
    "compute_nonlinear_diagnostic_tuple",
    "linear_rhs_jit_for_terms_impl",
    "make_nonlinear_diagnostic_tuple_fn",
    "nonlinear_em_term_cached_impl",
    "nonlinear_rhs_cached_impl",
]

_DIAGNOSTIC_EXPORTS = {
    "NonlinearDiagnosticKernels",
    "compute_nonlinear_diagnostic_tuple",
    "make_nonlinear_diagnostic_tuple_fn",
}


def __getattr__(name: str) -> Any:
    if name in _DIAGNOSTIC_EXPORTS:
        from gkx.operators.nonlinear import diagnostic_state

        return getattr(diagnostic_state, name)
    if name in __all__:
        from gkx.operators.nonlinear import rhs

        return getattr(rhs, name)
    raise AttributeError(name)
