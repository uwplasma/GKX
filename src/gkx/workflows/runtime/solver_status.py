"""Solver-status checks at the runtime host boundary.

Implicit and IMEX scans carry ``ImplicitSolveStats`` because a traced step
cannot raise. The runtime asks for those stats only from methods that take
implicit solves and refuses an unconverged run here, as the eigen gates refuse
an uncertified pair.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from gkx.solvers_linear_implicit import (
    ImplicitSolveSummary,
    require_converged_implicit_solves,
)

_IMPLICIT_SOLVE_METHODS = {
    "linear": frozenset({"implicit"}),
    "nonlinear": frozenset({"imex", "semi-implicit"}),
}


def solve_stats_request(method: Any, *, kind: str) -> dict[str, Any]:
    """Return ``return_solve_stats=True`` when ``method`` takes implicit solves."""

    key = str(method).strip().lower()
    return {"return_solve_stats": True} if key in _IMPLICIT_SOLVE_METHODS[kind] else {}


def checked_solve_summary(
    outputs: Sequence[Any], request: dict[str, Any], *, label: str
) -> ImplicitSolveSummary | None:
    """Fail closed on the stats a requested integrator appended last.

    Returns ``None`` when no stats were requested, so explicit runs keep a null
    status rather than a fabricated converged one.
    """

    if not request:
        return None
    return require_converged_implicit_solves(outputs[-1], label=label)


__all__ = ["checked_solve_summary", "solve_stats_request"]
