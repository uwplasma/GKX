"""Shared structured-algebra policies and time-integration helpers."""

from __future__ import annotations

from collections.abc import Callable

import jax
from solvax import KrylovSolution, gmres


def solve_gmres(
    matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    x0: jax.Array | None,
    preconditioner: Callable[[jax.Array], jax.Array] | None,
    tolerance: float,
    max_restarts: int,
    restart: int,
) -> KrylovSolution:
    """Solve a complex matrix-free system with the admitted FGMRES backend."""

    return gmres(
        matvec,
        rhs,
        x0=x0,
        precond=preconditioner,
        restart=max(int(restart), 1),
        rtol=tolerance,
        atol=0.0,
        max_restarts=max(int(max_restarts), 1),
    )


__all__ = ["solve_gmres"]
