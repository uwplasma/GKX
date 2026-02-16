"""Gyroaveraging coefficients for Laguerre velocity space."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import gammaln, i0e


def gamma0(b: jnp.ndarray) -> jnp.ndarray:
    """Compute Gamma_0(b) = exp(-b) I_0(b) using i0e for stability."""

    b = jnp.asarray(b)
    return i0e(b)


def J_l_all(b: jnp.ndarray, l_max: int) -> jnp.ndarray:
    """Gyroaveraging coefficients matching the GX Laguerre-Hermite convention."""

    if l_max < 0:
        raise ValueError("l_max must be >= 0")
    b = jnp.asarray(b)
    l = jnp.arange(l_max + 1, dtype=b.dtype)
    l_shape = (l_max + 1,) + (1,) * b.ndim
    l = l.reshape(l_shape)
    b_safe = jnp.maximum(0.5 * b, 1.0e-30)
    log_term = l * jnp.log(b_safe) - gammaln(l + 1.0)
    coef = jnp.exp(log_term)
    sign = jnp.where((l % 2) == 0, 1.0, -1.0)
    Jl = jnp.exp(-0.5 * b)[None, ...] * sign * coef
    if b.ndim > 0:
        mask = (b == 0.0)[None, ...] & (l > 0)
        Jl = jnp.where(mask, 0.0, Jl)
    return Jl


def sum_Jl2(b: jnp.ndarray, l_max: int) -> jnp.ndarray:
    """Truncated sum of J_l(b)^2, useful for Gamma_0 convergence checks."""

    Jl = J_l_all(b, l_max)
    return jnp.sum(Jl * Jl, axis=0)
