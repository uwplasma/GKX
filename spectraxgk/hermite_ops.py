"""
Hermite ladder operators and collisions in our normalization.

We work in dimensionless units (v_th = 1, lambda_D = 1, omega_p = 1).
Streaming (in Fourier) was: off_n = k * sqrt(n+1). In real-space, the k is
replaced by a spatial derivative operator. We keep the same ladder factors
sqrt(n+1) connecting n <-> n+1.

Field closure: in Fourier, E_k = i c_{0k}/k, and only dc1/dt gets the E term
(one-sided coupling). In x-space this becomes E = -∂_x φ, with φ solving
-∂_x^2 φ = c0. This induces a non-local x-operator mapping c0 -> E.
"""

from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=['N'])
def ladder_sqrt(N: int) -> jnp.ndarray:
    """Return sqrt(n+1) for n=0..N-2 as a length-(N-1) vector."""
    n = jnp.arange(N - 1, dtype=jnp.float64)
    return jnp.sqrt(n + 1.0)

@partial(jax.jit, static_argnames=['N'])
def streaming_block_fourier(k: float, N: int) -> jnp.ndarray:
    """
    Fourier-space streaming Hermite block H_stream with off-diagonals k*sqrt(n+1).
    This is the same normalization you converged to in previous scripts.
    """
    off = k * ladder_sqrt(N)
    H = jnp.zeros((N, N), dtype=jnp.float64)
    H = H.at[jnp.arange(N - 1), jnp.arange(1, N)].set(off)
    H = H.at[jnp.arange(1, N), jnp.arange(N - 1)].set(off)
    return H

@partial(jax.jit, static_argnames=['N'])
def field_one_sided_fourier(k: float, N: int) -> jnp.ndarray:
    """
    Only dc1/dt gets E term, with E_k = i c0 / k  => H_field[1,0] = 1/k
    """
    H = jnp.zeros((N, N), dtype=jnp.float64)
    return jax.lax.cond(jnp.abs(k) > 0.0,
                        lambda HH: HH.at[1, 0].set(1.0 / k),
                        lambda HH: HH, H)

@partial(jax.jit, static_argnames=['N'])
def build_collision_matrix(N: int, nu0: float, hyper_p: int = 0, cutoff: int = 3) -> jnp.ndarray:
    """
    Lenard–Bernstein with hyper factor; zero for n<cutoff:
    nu(n) = nu0 * (n/(N-1))^p for n>=cutoff; else 0.
    C = diag( nu(n) * n )
    """
    n = jnp.arange(N, dtype=jnp.float64)
    mask = (n >= cutoff).astype(jnp.float64)
    denom = jnp.maximum(N - 1.0, 1.0)
    p = jnp.asarray(hyper_p, dtype=jnp.float64)
    scale = jnp.power(n / denom, p)
    nu_n = nu0 * scale * mask
    return jnp.diag(nu_n * n)

def block_diag_repeat(mat: jnp.ndarray, reps: int) -> jnp.ndarray:
    """kronecker(I_reps, mat) without leaving JAX. Shape: (reps*N, reps*N)."""
    N = mat.shape[0]
    eye = jnp.eye(reps, dtype=mat.dtype)
    return jnp.kron(eye, mat)
