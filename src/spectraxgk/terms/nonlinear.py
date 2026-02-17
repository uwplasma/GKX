"""Nonlinear E×B term placeholders (to be implemented)."""

from __future__ import annotations

import jax.numpy as jnp


def exb_nonlinear_contribution(
    G: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    """Return the nonlinear E×B contribution (placeholder)."""

    _ = (G, phi, dealias_mask, weight)
    raise NotImplementedError("Nonlinear E×B term is not implemented yet.")


def placeholder_nonlinear_contribution(
    G: jnp.ndarray,
    *,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    """Return a zero nonlinear contribution to validate IO shapes."""

    return jnp.zeros_like(G) * weight
