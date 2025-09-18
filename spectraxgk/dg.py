"""
DG0 (finite-volume) real-space discretization in x with Hermite in v.

State: c[n, i] for Hermite index n=0..N-1 and cell i=0..Nx-1.
Semi-discrete system (dimensionless):
  ∂_t c_n + sqrt(n+1)*∂_x c_{n+1} + sqrt(n)*∂_x c_{n-1} + δ_{n1} * E = -ν(n) n c_n,
with E(x) = -∂_x φ, and -∂_x^2 φ = c_0(x).

DG0 upwind for ∂_x: we build a (Nx x Nx) matrix D depending on BC.
Field term is a nonlocal linear map: E = P @ c0  with P from poisson.build_P.

We assemble a big real linear operator A_real on the vectorized state C=[c_0; c_1; ...; c_{N-1}] (size N*Nx).
"""

from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from hermite_ops import ladder_sqrt, build_collision_matrix, block_diag_repeat
from poisson import build_P

def upwind_D(Nx: int, L: float, bc_kind: str) -> jnp.ndarray:
    """DG0 upwind derivative matrix D: approximates ∂_x (flux form) on cell averages."""
    dx = L / Nx
    D = jnp.zeros((Nx, Nx))
    # interior upwind (positive characteristic speed for both ladders => use right/left symmetrically)
    # We'll use central flux for simplicity; you can switch to true upwind by BC sign.
    D = D.at[jnp.arange(1, Nx), jnp.arange(0, Nx-1)].set(-1.0 / (2*dx))
    D = D.at[jnp.arange(0, Nx-1), jnp.arange(1, Nx)].set( 1.0 / (2*dx))
    # boundaries
    if bc_kind == "periodic":
        D = D.at[0, -1].set(-1.0 / (2*dx))
        D = D.at[-1, 0].set( 1.0 / (2*dx))
    elif bc_kind == "dirichlet":
        # one-sided at edges
        D = D.at[0, 0].set(-1.0 / dx).at[0, 1].set(1.0 / dx)
        D = D.at[-1, -2].set(-1.0 / dx).at[-1, -1].set(1.0 / dx)
    elif bc_kind == "neumann":
        # ghost gradient ~0 => copy interior stencil with zero flux => zero at ends
        D = D.at[0, 0].set(-1.0 / (2*dx)).at[0, 1].set(1.0 / (2*dx))
        D = D.at[-1, -2].set(-1.0 / (2*dx)).at[-1, -1].set(1.0 / (2*dx))
    else:
        raise ValueError("bc_kind must be periodic|dirichlet|neumann")
    return D

def assemble_A_real(Nx: int, L: float, N: int,
                    nu0: float, hyper_p: int, cutoff: int,
                    bc_kind: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build the real linear operator for the stacked state C (N*Nx,):
    dC/dt = A_real @ C
    Returns: (A_real, P) where P maps c0 -> E (P is Nx x Nx)
    """
    D = upwind_D(Nx, L, bc_kind)           # Nx x Nx
    P = build_P(Nx, L, bc_kind)            # Nx x Nx  (E = P @ c0)
    sqrt = ladder_sqrt(N)                  # length N-1

    # Streaming couplings coupled with D:
    # n -> n+1 via +sqrt[n]*∂_x c_{n+1}, and n -> n-1 via +sqrt[n-1]*∂_x c_{n-1}
    # We build a block (N blocks of size Nx) matrix.
    blocks = []
    for n in range(N):
        row = [jnp.zeros((Nx, Nx)) for _ in range(N)]
        if n + 1 < N:
            row[n+1] =  row[n+1] + sqrt[n] * D
        if n - 1 >= 0:
            row[n-1] =  row[n-1] + jnp.sqrt(n) * D
        # Field coupling: only n==1 receives +E term => +I_c0_to_E @ c0
        if n == 1:
            row[0] = row[0] + P  # dc1/dt += E = P @ c0
        blocks.append(row)
    A = jnp.block(blocks)  # (N*Nx, N*Nx)

    # Collisions (diagonal in n): C = diag(nu(n)*n) acting pointwise in x; block-diag repeat.
    Cn = build_collision_matrix(N, nu0, hyper_p, cutoff)
    C_big = block_diag_repeat(Cn, Nx)  # (N*Nx, N*Nx)
    A_real = A - C_big
    return A_real, P

def initial_condition(Nx: int, L: float, N: int, kind: str, amplitude: float, k: float|None,
                      shift: float|None, seed_c1: bool) -> jnp.ndarray:
    """
    Build c(x,0): shape (N, Nx). For DG0: real coefficients.
    - landau: c0(x,0) = amplitude * cos(k x)
    - two_stream: f0 ~ 0.5[ M(u-Δ) + M(u+Δ) ]; linearization seed in c1 via shift.
      Here we seed c0 with small spatial perturbation and c1 optional.
    """
    x = jnp.linspace(0.0, L, Nx, endpoint=False)
    C = jnp.zeros((N, Nx), dtype=jnp.float64)
    if kind == "landau":
        if k is None:
            raise ValueError("landau needs init.k")
        C = C.at[0, :].set(amplitude * jnp.cos(k * x))
        if seed_c1:
            C = C.at[1, :].set(0.1 * amplitude * jnp.cos(k * x))
    elif kind == "two_stream":
        k0 = k if k is not None else (2.0 * jnp.pi / L)
        C = C.at[0, :].set(amplitude * jnp.cos(k0 * x))
        if seed_c1:
            C = C.at[1, :].set(0.1 * amplitude * jnp.cos(k0 * x))
        # shift enters the equilibrium, but here we just seed perturbations
    else:
        raise ValueError("init.type must be landau|two_stream")
    return C

def solve_dg(A_real: jnp.ndarray, C0: jnp.ndarray, tmax: float, nt: int,
             backend: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evolve dC/dt = A_real C with either eig (fast for static A) or Diffrax Tsit5.
    C0 comes as (N, Nx), we flatten and then reshape back to (N, Nx, nt).
    """
    N, Nx = C0.shape
    y0 = jnp.reshape(C0, (N*Nx,))

    ts = jnp.linspace(0.0, tmax, nt, dtype=jnp.float64)
    if backend == "eig":
        w, V = jnp.linalg.eig(A_real)
        Vinv = jnp.linalg.inv(V)
        alpha = Vinv @ y0
        Y = V @ (jnp.exp(w[:, None] * ts[None, :]) * alpha[:, None])    # (N*Nx, nt)
    else:
        from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt
        term = ODETerm(lambda t, y, A: A @ y)
        sol = diffeqsolve(term, Tsit5(), 0.0, float(tmax), y0=y0, args=A_real,
                          dt0=1e-3, saveat=SaveAt(ts=ts), max_steps=2_000_000)
        Y = sol.ys.T  # (N*Nx, nt)

    C_t = jnp.reshape(Y, (N, Nx, nt))
    return np.asarray(ts), np.asarray(C_t)
