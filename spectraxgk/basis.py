from __future__ import annotations
import jax.numpy as jnp


def hermite_coupling_factors(Nn: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return sqrt(n) and sqrt(n+1) arrays of length Nn with sqrt(0)=0.
    These appear in the v_|| streaming operator couplings.
    """
    n = jnp.arange(Nn)
    sqrt_n = jnp.sqrt(n)
    sqrt_np1 = jnp.sqrt(n + 1.0)
    return sqrt_n, sqrt_np1


def lb_eigenvalues(Nn: int, Nm: int, alpha: float = 1.0, beta: float = 2.0) -> jnp.ndarray:
    """Diagonal Lenard-Bernstein rates in Hermite-Laguerre space.
    A common model is \lambda_{n,m} = alpha * n + beta * m. In 3D (two \perp dims),
    a frequently used choice is alpha=1, beta=2 (reflecting two perpendicular degrees).
    Overall collision frequency `nu` scales these rates.
    """
    n = jnp.arange(Nn)
    m = jnp.arange(Nm)
    lam = alpha * n[:, None] + beta * m[None, :]
    return lam