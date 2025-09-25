# hermite_ops.py
"""
Hermite ladder operators and LB collisions (dimensionless).
Streaming in Fourier uses off-diagonals k*sqrt(n+1); in x-space we use the same
ladder sqrt(n+1) and let the spatial derivative provide the “k”.

Field closure:
  Fourier:  E_k = i c0/k, and only dc1/dt has E  -> H_field[1,0] = 1/k
  x-space:  E = P @ c0 (nonlocal), coupled only into n=1 equation.
"""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["N"])
def ladder_sqrt(N: int) -> jnp.ndarray:
    """Return vector s with s[n] = sqrt(n+1), length N-1."""
    n = jnp.arange(N - 1, dtype=jnp.float64)
    return jnp.sqrt(n + 1.0)


@partial(jax.jit, static_argnames=["N"])
def streaming_block_fourier(k: float, N: int) -> jnp.ndarray:
    """Fourier-space streaming Hermite block: off-diagonals k*sqrt(n+1)."""
    s = ladder_sqrt(N)
    H = jnp.zeros((N, N), dtype=jnp.float64)
    off = jnp.asarray(k, jnp.float64) * s
    H = H.at[jnp.arange(N - 1), jnp.arange(1, N)].set(off)  # super
    H = H.at[jnp.arange(1, N), jnp.arange(N - 1)].set(off)  # sub
    return H


@partial(jax.jit, static_argnames=["N"])
def field_one_sided_fourier(k: float, N: int) -> jnp.ndarray:
    """Only dc1/dt gets E term, with E_k = i c0 / k  => H_field[1,0] = 1/k."""
    H = jnp.zeros((N, N), dtype=jnp.float64)
    k = jnp.asarray(k, jnp.float64)
    return jax.lax.cond(
        jnp.abs(k) > 0.0,
        lambda HH: HH.at[1, 0].set(1.0 / k),
        lambda HH: HH,
        H,
    )


@partial(jax.jit, static_argnames=["N"])
def build_collision_matrix(N: int, nu0: float, hyper_p: int = 0, cutoff: int = 3) -> jnp.ndarray:
    """
    Lenard–Bernstein with optional hyper factor; zero for n < cutoff:
      nu(n) = nu0 * (n/(N-1))^p,   C = diag(nu(n) * n)
    """
    n = jnp.arange(N, dtype=jnp.float64)
    mask = (n >= cutoff).astype(jnp.float64)
    denom = jnp.maximum(N - 1.0, 1.0)
    p = jnp.asarray(hyper_p, dtype=jnp.float64)
    scale = jnp.power(n / denom, p)
    nu_n = jnp.asarray(nu0, jnp.float64) * scale * mask
    return jnp.diag(nu_n * n)
