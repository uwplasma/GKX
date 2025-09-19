# dg.py
"""
DG0 (finite-volume) real-space discretization in x with Hermite in v.

State: c[n, i] for Hermite index n=0..N-1 and cell i=0..Nx-1.

Semi-discrete (dimensionless):
  ∂_t c_n + sqrt(n+1)*∂_x c_{n+1} + sqrt(n)*∂_x c_{n-1} + δ_{n1} * E(x) = -ν(n) n c_n,
with E(x) = -∂_x φ, and -∂_x^2 φ = c_0(x).

DG0 (here: central flux by default) builds a dense Nx×Nx derivative operator D
depending on boundary condition. Field term is a nonlocal linear map: E = P @ c0.

We assemble a big real operator A_real on vectorized state C=[c_0; c_1; ...; c_{N-1}] (size N*Nx):
  dC/dt = A_real @ C
"""

from typing import Tuple
import jax
import jax.numpy as jnp

from hermite_ops import (
    ladder_sqrt,
    build_collision_matrix,
)
from poisson import build_P


# -------------------- DG derivative --------------------
@jax.jit
def _D_periodic(Nx: int, L: float) -> jnp.ndarray:
    """Central-difference derivative on a periodic grid of cell averages."""
    dx = jnp.asarray(L, jnp.float64) / jnp.asarray(Nx, jnp.float64)
    # Shift matrices via roll
    eye = jnp.eye(Nx, dtype=jnp.float64)
    S_plus  = jnp.roll(eye, -1, axis=1)   # right neighbor (i -> i+1)
    S_minus = jnp.roll(eye,  1, axis=1)   # left  neighbor (i -> i-1)
    return (S_plus - S_minus) * (0.5 / dx)


@jax.jit
def _D_dirichlet(Nx: int, L: float) -> jnp.ndarray:
    """One-sided at boundaries; centered in interior."""
    dx = jnp.asarray(L, jnp.float64) / jnp.asarray(Nx, jnp.float64)
    D = jnp.zeros((Nx, Nx), dtype=jnp.float64)
    # interior centered
    idx = jnp.arange(1, Nx - 1)
    D = D.at[idx, idx - 1].set(-0.5 / dx)
    D = D.at[idx, idx + 1].set(+0.5 / dx)
    # edges one-sided
    D = D.at[0, 0].set(-1.0 / dx).at[0, 1].set(+1.0 / dx)
    D = D.at[-1, -2].set(-1.0 / dx).at[-1, -1].set(+1.0 / dx)
    return D


@jax.jit
def _D_neumann(Nx: int, L: float) -> jnp.ndarray:
    """Zero-gradient at ends (use interior stencil copied to edges)."""
    dx = jnp.asarray(L, jnp.float64) / jnp.asarray(Nx, jnp.float64)
    D = jnp.zeros((Nx, Nx), dtype=jnp.float64)
    # interior centered
    idx = jnp.arange(1, Nx - 1)
    D = D.at[idx, idx - 1].set(-0.5 / dx)
    D = D.at[idx, idx + 1].set(+0.5 / dx)
    # shallow edges (mirror interior centered)
    D = D.at[0, 0].set(-0.5 / dx).at[0, 1].set(+0.5 / dx)
    D = D.at[-1, -2].set(-0.5 / dx).at[-1, -1].set(+0.5 / dx)
    return D


def upwind_D(Nx: int, L: float, bc_kind: str) -> jnp.ndarray:
    """Choose a DG0 derivative consistent with the boundary condition."""
    if bc_kind == "periodic":
        return _D_periodic(Nx, L)
    elif bc_kind == "dirichlet":
        return _D_dirichlet(Nx, L)
    elif bc_kind == "neumann":
        return _D_neumann(Nx, L)
    raise ValueError("bc_kind must be periodic|dirichlet|neumann")


# -------------------- Assembly via Kronecker products --------------------
def _hermite_streaming_blocks(N: int) -> jnp.ndarray:
    """
    Hermite “ladder” coupling matrix S such that:
      (S)_{n,n+1} = sqrt(n+1), (S)_{n,n-1} = sqrt(n)
    i.e. S = diag(s, +1) + diag(s, -1) where s_n = sqrt(n+1).
    """
    s = ladder_sqrt(N)                             # (N-1,)
    S = jnp.zeros((N, N), dtype=jnp.float64)
    S = S.at[jnp.arange(N - 1), jnp.arange(1, N)].set(s)   # super
    S = S.at[jnp.arange(1, N), jnp.arange(N - 1)].set(s)   # sub
    return S


def _field_selector(N: int) -> jnp.ndarray:
    """Hermite selector E10 with (1,0)=1 (only n=1 receives E from c0)."""
    E10 = jnp.zeros((N, N), dtype=jnp.float64)
    E10 = E10.at[1, 0].set(1.0)
    return E10


def _kron(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Lightweight Kronecker (A⊗B) using einsum; keeps dtype & JIT-friendly."""
    a0, a1 = A.shape
    b0, b1 = B.shape
    out = jnp.einsum("ij,kl->ikjl", A, B)
    return out.reshape((a0 * b0, a1 * b1))


def assemble_A_real(
    Nx: int,
    L: float,
    N: int,
    nu0: float,
    hyper_p: int,
    cutoff: int,
    bc_kind: str,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build A_real for stacked state C (N*Nx,):
      dC/dt = A_real @ C

    Returns:
      A_real : (N*Nx, N*Nx)
      P      : (Nx, Nx)   (E = P @ c0)
    """
    # x-operators
    D = upwind_D(Nx, L, bc_kind)             # (Nx, Nx)
    P = build_P(Nx, L, bc_kind)              # (Nx, Nx), E = P @ c0
    I_x = jnp.eye(Nx, dtype=jnp.float64)

    # Hermite operators
    S = _hermite_streaming_blocks(N)         # (N, N)
    E10 = _field_selector(N)                 # (N, N)

    # Streaming: sum over ± ladders -> S @ ∂_x
    A_stream = _kron(S, D)                   # (N*Nx, N*Nx)

    # Field: only n=1 receives +E(x), with E = P @ c0
    # Block at (n=1, m=0) equals P -> kron(E10, P)
    A_field = _kron(E10, P)

    # Collisions: diagonal in n, pointwise in x: kron(Cn, I_x)
    Cn = build_collision_matrix(N, nu0, hyper_p, cutoff)  # (N, N)
    A_coll = _kron(Cn, I_x)

    # Total real generator
    A_real = (A_stream + A_field) - A_coll
    return A_real.astype(jnp.float64), P.astype(jnp.float64)


# -------------------- Initial condition & solve --------------------
def initial_condition(
    Nx: int,
    L: float,
    N: int,
    kind: str,
    amplitude: float,
    k: float | None,
    shift: float | None,
    seed_c1: bool,
) -> jnp.ndarray:
    """
    Build c(x,0): shape (N, Nx). Real DG0 coefficients.

    landau:     c0(x,0) = A cos(k x), optional c1 seed
    two_stream: same scaffold for perturbation; (shift) affects equilibrium (not modeled here).
    """
    x = jnp.linspace(0.0, L, Nx, endpoint=False, dtype=jnp.float64)
    C0 = jnp.zeros((N, Nx), dtype=jnp.float64)

    if kind == "landau":
        if k is None:
            raise ValueError("landau needs init.k")
        C0 = C0.at[0, :].set(amplitude * jnp.cos(jnp.asarray(k, jnp.float64) * x))
        if seed_c1:
            C0 = C0.at[1, :].set(0.1 * amplitude * jnp.cos(jnp.asarray(k, jnp.float64) * x))

    elif kind == "two_stream":
        k0 = jnp.asarray(k if k is not None else (2.0 * jnp.pi / L), jnp.float64)
        C0 = C0.at[0, :].set(amplitude * jnp.cos(k0 * x))
        if seed_c1:
            C0 = C0.at[1, :].set(0.1 * amplitude * jnp.cos(k0 * x))
        # (shift) would modify f0; here we only seed perturbations.

    else:
        raise ValueError("init.type must be landau|two_stream")

    return C0


def solve_dg(
    A_real: jnp.ndarray,
    C0: jnp.ndarray,
    tmax: float,
    nt: int,
    backend: str,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evolve dC/dt = A_real C with either eig (fast for static A) or Diffrax Tsit5.
    C0: (N, Nx), returns C_t: (N, Nx, nt) (real).
    """
    N, Nx = C0.shape
    y0 = jnp.reshape(C0, (N * Nx,))
    ts = jnp.linspace(0.0, tmax, nt, dtype=jnp.float64)

    if backend == "eig":
        # JIT-friendly batched exp via eig
        w, V = jnp.linalg.eig(A_real.astype(jnp.complex128))
        Vinv = jnp.linalg.inv(V)
        alpha = Vinv @ y0
        phases = jnp.exp(w[:, None] * ts[None, :])             # (N*Nx, nt)
        Yc = V @ (phases * alpha[:, None])                     # complex
        Y = jnp.real(Yc)                                       # A_real is real
    else:
        from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController
        term = ODETerm(lambda t, y, A: A @ y)
        controller = PIDController(rtol=1e-7, atol=1e-10, jump_ts=ts)
        sol = diffeqsolve(
            term, Tsit5(),
            t0=0.0, t1=float(tmax), dt0=1e-3,
            y0=y0, args=A_real,
            stepsize_controller=controller,
            saveat=SaveAt(ts=ts),
            max_steps=2_000_000
        )
        Y = sol.ys.T  # (N*Nx, nt)

    C_t = jnp.reshape(Y, (N, Nx, nt))
    return jnp.asarray(ts), jnp.asarray(C_t)
