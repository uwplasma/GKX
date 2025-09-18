"""
Poisson solvers mapping c0(x) -> E(x) = -∂x φ with -∂x^2 φ = c0.

- Periodic: spectral FFT-based invert Laplacian (k!=0), k=0 set φ̂(0)=0.
- Neumann: solve tridiagonal for φ'' = -c0 with φ'(0)=φ'(L)=0 (unique up to constant; set mean φ=0).
- Dirichlet: φ(0)=φ(L)=0.
All return a linear operator P such that E = P @ c0_vec for DG0 grid.
"""

from typing import Literal
import numpy as np
import jax
import jax.numpy as jnp

def _periodic_P(Nx: int, L: float) -> jnp.ndarray:
    x = jnp.linspace(0.0, L, Nx, endpoint=False)
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(Nx, d=L / Nx)  # physical wavenumbers
    k2 = k * k
    # Build spectral operator: φ̂ = -c0̂ / k^2 (k!=0), Ê = -ik φ̂
    def apply(c0):
        c0h = jnp.fft.fft(c0)
        ph = jnp.where(k2 == 0, 0.0 + 0.0j, -c0h / k2)
        Eh = -1j * k * ph
        E = jnp.real(jnp.fft.ifft(Eh))
        return E
    # Materialize a dense matrix P by applying to basis vectors (small Nx is fine)
    # If Nx large, prefer a lazy LinearOperator; but dense is simple and JAX-friendly for now.
    I = jnp.eye(Nx)
    cols = jax.vmap(apply, in_axes=0)(I)
    return cols.T  # P @ c0

def _fd_tridiag(Nx: int, L: float, bc: Literal["neumann","dirichlet"]) -> jnp.ndarray:
    """
    Build finite-difference inverse Laplacian composed with negative gradient E=-φ_x.
    Return dense P so that E = P @ c0.
    """
    dx = L / Nx
    # Laplacian matrix A (Nx x Nx)
    main = -2.0 * jnp.ones((Nx,))
    off  = 1.0 * jnp.ones((Nx - 1,))
    A = jnp.diag(main) + jnp.diag(off, 1) + jnp.diag(off, -1)
    if bc == "neumann":
        A = A.at[0,0].set(-1.0).at[-1,-1].set(-1.0)
    elif bc == "dirichlet":
        pass
    A = A / (dx * dx)
    # Right-hand side: A φ = -c0
    # Gradient matrix G: E = -G φ; use centered differences interior.
    G = jnp.zeros((Nx, Nx))
    # interior centered
    G = G.at[1:-1, :-2].add(-0.5 / dx)
    G = G.at[1:-1, 2: ].add( 0.5 / dx)
    # boundaries one-sided
    if bc == "neumann":
        G = G.at[0, 1].set( 1.0 / dx)
        G = G.at[-1, -2].set(-1.0 / dx)
    elif bc == "dirichlet":
        G = G.at[0, 0].set(-1.0 / dx).at[0,1].set(1.0 / dx)
        G = G.at[-1,-2].set(-1.0 / dx).at[-1,-1].set(1.0 / dx)

    # P = (-G) @ A^{-1} @ (-I)  => G @ A^{-1}
    Ainv = jnp.linalg.inv(A)
    P = G @ Ainv
    return P

def build_P(Nx: int, L: float, bc_kind: str) -> jnp.ndarray:
    if bc_kind == "periodic":
        return _periodic_P(Nx, L)
    elif bc_kind == "neumann":
        return _fd_tridiag(Nx, L, "neumann")
    elif bc_kind == "dirichlet":
        return _fd_tridiag(Nx, L, "dirichlet")
    else:
        raise ValueError("bc.kind must be periodic|neumann|dirichlet")
