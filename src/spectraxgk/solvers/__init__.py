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
    method: str,
) -> KrylovSolution:
    """Solve a complex matrix-free system with the admitted FGMRES backend.

    ``batched`` and ``incremental`` are accepted while the TOML schema moves
    from the former JAX least-squares implementation. Both now select SOLVAX's
    incremental unitary-Givens FGMRES; the names no longer change physics or
    orthogonalization. Unknown values fail before tracing instead of silently
    changing a solve policy.
    """

    normalized = str(method).strip().lower()
    if normalized not in {"gmres", "batched", "incremental", "solvax"}:
        raise ValueError(
            "GMRES method must be 'gmres', 'solvax', 'batched', or "
            "'incremental'; "
            f"got {method!r}"
        )
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
