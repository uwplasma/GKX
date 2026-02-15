"""Hermite and Laguerre basis utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


def hermite_physicists(x: jnp.ndarray, n_max: int) -> jnp.ndarray:
    """Physicists' Hermite polynomials H_n(x) for n=0..n_max.

    Weight: exp(-x**2). Recurrence:
        H_0 = 1
        H_1 = 2x
        H_{n+1} = 2x H_n - 2n H_{n-1}
    """

    x = jnp.asarray(x)
    if n_max < 0:
        raise ValueError("n_max must be >= 0")
    if n_max == 0:
        return jnp.expand_dims(jnp.ones_like(x), axis=0)
    h0 = jnp.ones_like(x)
    h1 = 2.0 * x

    def step(carry, n):
        h_prev, h_curr = carry
        h_next = 2.0 * x * h_curr - 2.0 * n * h_prev
        return (h_curr, h_next), h_next

    _, tail = jax.lax.scan(step, (h0, h1), jnp.arange(1, n_max))
    return jnp.concatenate([h0[None, ...], h1[None, ...], tail], axis=0)


def hermite_normed(x: jnp.ndarray, n_max: int) -> jnp.ndarray:
    """Normalized Hermite functions with weight exp(-x**2).

    psi_n = H_n(x) / sqrt(2**n * n! * sqrt(pi))
    """

    h = hermite_physicists(x, n_max)
    n = jnp.arange(0, n_max + 1)
    log_norm = 0.5 * (n * jnp.log(2.0) + gammaln(n + 1) + 0.5 * jnp.log(jnp.pi))
    norm = jnp.exp(log_norm)
    return h / norm[:, None]


def laguerre(x: jnp.ndarray, l_max: int) -> jnp.ndarray:
    """Laguerre polynomials L_l(x) for l=0..l_max.

    Weight: exp(-x). Recurrence:
        L_0 = 1
        L_1 = 1 - x
        (l+1) L_{l+1} = (2l+1-x) L_l - l L_{l-1}
    """

    x = jnp.asarray(x)
    if l_max < 0:
        raise ValueError("l_max must be >= 0")
    if l_max == 0:
        return jnp.expand_dims(jnp.ones_like(x), axis=0)
    l0 = jnp.ones_like(x)
    l1 = 1.0 - x

    def step(carry, l):
        l_prev, l_curr = carry
        l_next = ((2.0 * l + 1.0 - x) * l_curr - l * l_prev) / (l + 1.0)
        return (l_curr, l_next), l_next

    _, tail = jax.lax.scan(step, (l0, l1), jnp.arange(1, l_max))
    return jnp.concatenate([l0[None, ...], l1[None, ...], tail], axis=0)


def hermite_ladder_coeffs(n_max: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return sqrt(n+1) and sqrt(n) arrays for Hermite ladder operators."""

    if n_max < 0:
        raise ValueError("n_max must be >= 0")
    n = jnp.arange(0, n_max + 1)
    return jnp.sqrt(n + 1.0), jnp.sqrt(n)
