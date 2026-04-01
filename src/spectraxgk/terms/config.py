"""Configuration and state containers for term-wise RHS assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FieldState:
    """Field variables for the gyrokinetic system."""

    phi: jnp.ndarray
    apar: jnp.ndarray | None = None
    bpar: jnp.ndarray | None = None

    def tree_flatten(self):
        children = (self.phi, self.apar, self.bpar)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


# Signature for nonlinear RHS functions: G -> (dG, fields)
RHSFn = Callable[[jnp.ndarray], Tuple[jnp.ndarray, FieldState]]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TermConfig:
    """Switches for term-wise RHS assembly (1.0 = on, 0.0 = off)."""

    streaming: float = 1.0
    mirror: float = 1.0
    curvature: float = 1.0
    gradb: float = 1.0
    diamagnetic: float = 1.0
    collisions: float = 1.0
    hypercollisions: float = 1.0
    hyperdiffusion: float = 0.0
    end_damping: float = 1.0
    apar: float = 1.0
    bpar: float = 1.0
    nonlinear: float = 0.0

    def tree_flatten(self):
        children = (
            self.streaming,
            self.mirror,
            self.curvature,
            self.gradb,
            self.diamagnetic,
            self.collisions,
            self.hypercollisions,
            self.hyperdiffusion,
            self.end_damping,
            self.apar,
            self.bpar,
            self.nonlinear,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
