"""Gyroaveraging coefficients for Laguerre velocity space."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import i0e

from spectraxgk.basis import laguerre


def gamma0(b: jnp.ndarray) -> jnp.ndarray:
    """Compute Gamma_0(b) = exp(-b) I_0(b) using i0e for stability."""

    b = jnp.asarray(b)
    return i0e(b)


def J_l_all(b: jnp.ndarray, l_max: int) -> jnp.ndarray:
    """Gyroaveraging coefficients J_l(b) = exp(-b/2) L_l(b) for l=0..l_max."""

    if l_max < 0:
        raise ValueError("l_max must be >= 0")
    b = jnp.asarray(b)
    L = laguerre(b, l_max)
    return jnp.exp(-0.5 * b)[None, ...] * L


def sum_Jl2(b: jnp.ndarray, l_max: int) -> jnp.ndarray:
    """Truncated sum of J_l(b)^2, useful for Gamma_0 convergence checks."""

    Jl = J_l_all(b, l_max)
    return jnp.sum(Jl * Jl, axis=0)
