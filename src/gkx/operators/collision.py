"""Shared interface for gyrokinetic collision operators."""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol, runtime_checkable


class CollisionContext(NamedTuple):
    """Distribution, Hamiltonian, fields, cache, and parameters seen by collisions."""

    distribution: Any
    hamiltonian: Any
    fields: Any
    cache: Any
    parameters: Any


@runtime_checkable
class CollisionOperator(Protocol):
    """JAX-compatible collision model returning a state-shaped RHS term."""

    def apply(self, context: CollisionContext) -> Any:
        """Return the unit-weight collision contribution."""


@runtime_checkable
class SplitCollisionOperator(CollisionOperator, Protocol):
    """Collision model with a mathematically valid finite-time update."""

    def split_step(self, context: CollisionContext, dt: Any) -> Any:
        """Advance the unit-weight collision model by ``dt``."""


__all__ = ["CollisionContext", "CollisionOperator", "SplitCollisionOperator"]
