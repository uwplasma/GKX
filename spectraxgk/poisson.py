# poisson.py
"""
Poisson solvers mapping c0(x) -> E(x) = -∂x φ with -∂x^2 φ = c0 on a DG0 grid.

- Periodic: spectral inversion (k!=0), set ϕ̂(0)=0; return dense P with vmap.
- Neumann / Dirichlet: FD Laplacian + inverse, then E = -φ_x; return dense P.
"""

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("Nx",))
def _periodic_P(Nx: int, L: float) -> jnp.ndarray:
    """Dense operator P so that E = P @ c0 (periodic box)."""
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(Nx, d=(L / Nx))
    k2 = k * k

    def apply(c0: jnp.ndarray) -> jnp.ndarray:
        c0h = jnp.fft.fft(c0)
        ph = jnp.where(k2 == 0.0, 0.0 + 0.0j, -c0h / k2)
        Eh = -1j * k * ph
        return jnp.real(jnp.fft.ifft(Eh))

    identity_matrix = jnp.eye(Nx, dtype=jnp.float64)
    cols = jax.vmap(apply, in_axes=0)(identity_matrix)  # (Nx, Nx)
    return cols.T.astype(jnp.float64)


@partial(jax.jit, static_argnames=("Nx",))
def _fd_P(Nx: int, L: float, bc: Literal["neumann", "dirichlet"]) -> jnp.ndarray:
    """
    Dense P via FD Laplacian solve and gradient.
    Laplacian A φ = -c0; E = -∂x φ -> build G then P = G @ A^{-1}.
    """
    dx = jnp.asarray(L, jnp.float64) / jnp.asarray(Nx, jnp.float64)

    # Laplacian
    main = -2.0 * jnp.ones((Nx,), jnp.float64)
    off = 1.0 * jnp.ones((Nx - 1,), jnp.float64)
    A = jnp.diag(main) + jnp.diag(off, 1) + jnp.diag(off, -1)
    if bc == "neumann":
        A = A.at[0, 0].set(-1.0).at[-1, -1].set(-1.0)
    elif bc == "dirichlet":
        pass
    A = A / (dx * dx)

    # Gradient E = -φ_x (we’ll build G so that E = G φ)
    G = jnp.zeros((Nx, Nx), dtype=jnp.float64)
    # interior centered
    idx = jnp.arange(1, Nx - 1)
    G = G.at[idx, idx - 1].add(-0.5 / dx)
    G = G.at[idx, idx + 1].add(+0.5 / dx)
    # edges one-sided
    if bc == "neumann":
        G = G.at[0, 1].set(+1.0 / dx)
        G = G.at[-1, -2].set(-1.0 / dx)
    elif bc == "dirichlet":
        G = G.at[0, 0].set(-1.0 / dx).at[0, 1].set(+1.0 / dx)
        G = G.at[-1, -2].set(-1.0 / dx).at[-1, -1].set(+1.0 / dx)

    Ainv = jnp.linalg.inv(A)
    P = G @ Ainv
    return P.astype(jnp.float64)


def build_P(Nx: int, L: float, bc_kind: str) -> jnp.ndarray:
    """Return dense P so that E = P @ c0 for the given boundary condition."""
    if bc_kind == "periodic":
        return _periodic_P(Nx, L)
    elif bc_kind == "neumann":
        return _fd_P(Nx, L, "neumann")
    elif bc_kind == "dirichlet":
        return _fd_P(Nx, L, "dirichlet")
    raise ValueError("bc.kind must be periodic|neumann|dirichlet")
