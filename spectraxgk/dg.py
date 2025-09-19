"""
DG0 (finite-volume) real-space discretization in x with Hermite in v. (multi-species only)

State ordering (multi species, S): C = concat_s [c^{(s)}_0; ...; c^{(s)}_{N-1}]  (size S*N*Nx)

Semi-discrete (linearized around drifting Maxwellian, common velocity normalization):
  ∂_t δc^{(s)}_n
    + vth_s * (√(n+1) ∂_x δc^{(s)}_{n+1} + √n ∂_x δc^{(s)}_{n-1})
    + u0_s   * ∂_x δc^{(s)}_n
    + δ_{n1} * (q_s/m_s) E(x)
  = -ν_s(n) n δc^{(s)}_n

Poisson:  -∂_x^2 φ = Σ_s q_s δc^{(s)}_0,   E = -∂_x φ  => E = P @ (Σ_s q_s δc^{(s)}_0)

We build a big real operator A_real (S*N*Nx by S*N*Nx) so that dC/dt = A_real @ C.
"""

from typing import Tuple, Sequence, Optional
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from hermite_ops import ladder_sqrt, build_collision_matrix
from poisson import build_P


# -------------------- DG derivative --------------------
@jax.jit
def _D_periodic(Nx: int, L: float) -> jnp.ndarray:
    dx = jnp.asarray(L, jnp.float64) / jnp.asarray(Nx, jnp.float64)
    eye = jnp.eye(Nx, dtype=jnp.float64)
    S_plus  = jnp.roll(eye, -1, axis=1)
    S_minus = jnp.roll(eye,  1, axis=1)
    return (S_plus - S_minus) * (0.5 / dx)

@jax.jit
def _D_dirichlet(Nx: int, L: float) -> jnp.ndarray:
    dx = jnp.asarray(L, jnp.float64) / jnp.asarray(Nx, jnp.float64)
    D = jnp.zeros((Nx, Nx), dtype=jnp.float64)
    idx = jnp.arange(1, Nx - 1)
    D = D.at[idx, idx - 1].set(-0.5 / dx)
    D = D.at[idx, idx + 1].set(+0.5 / dx)
    D = D.at[0, 0].set(-1.0 / dx).at[0, 1].set(+1.0 / dx)
    D = D.at[-1, -2].set(-1.0 / dx).at[-1, -1].set(+1.0 / dx)
    return D

@jax.jit
def _D_neumann(Nx: int, L: float) -> jnp.ndarray:
    dx = jnp.asarray(L, jnp.float64) / jnp.asarray(Nx, jnp.float64)
    D = jnp.zeros((Nx, Nx), dtype=jnp.float64)
    idx = jnp.arange(1, Nx - 1)
    D = D.at[idx, idx - 1].set(-0.5 / dx)
    D = D.at[idx, idx + 1].set(+0.5 / dx)
    D = D.at[0, 0].set(-0.5 / dx).at[0, 1].set(+0.5 / dx)
    D = D.at[-1, -2].set(-0.5 / dx).at[-1, -1].set(+0.5 / dx)
    return D

def upwind_D(Nx: int, L: float, bc_kind: str) -> jnp.ndarray:
    if bc_kind == "periodic":
        return _D_periodic(Nx, L)
    elif bc_kind == "dirichlet":
        return _D_dirichlet(Nx, L)
    elif bc_kind == "neumann":
        return _D_neumann(Nx, L)
    raise ValueError("bc_kind must be periodic|dirichlet|neumann")


# -------------------- Kronecker helpers --------------------
def _kron(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    a0, a1 = A.shape
    b0, b1 = B.shape
    out = jnp.einsum("ij,kl->ikjl", A, B)
    return out.reshape((a0 * b0, a1 * b1))

def _block_diag(blocks: Sequence[jnp.ndarray]) -> jnp.ndarray:
    S = len(blocks)
    z = jnp.zeros_like(blocks[0])
    rows = []
    for i in range(S):
        rows.append([blocks[i] if i == j else z for j in range(S)])
    return jnp.block(rows)


# -------------------- Hermite structure --------------------
def _hermite_streaming_blocks(N: int) -> jnp.ndarray:
    s = ladder_sqrt(N)  # (N-1,)
    S = jnp.zeros((N, N), dtype=jnp.float64)
    S = S.at[jnp.arange(N - 1), jnp.arange(1, N)].set(s)   # super
    S = S.at[jnp.arange(1, N), jnp.arange(N - 1)].set(s)   # sub
    return S

def _field_selector(N: int) -> jnp.ndarray:
    E10 = jnp.zeros((N, N), dtype=jnp.float64)
    E10 = E10.at[1, 0].set(1.0)  # n=1 receives E from n=0
    return E10


# -------------------- Multi-species assembly --------------------
def assemble_A_real_multispecies(
    Nx: int,
    L: float,
    N: int,
    species: Sequence,   # expects q, m, n0, vth, u0, nu0, hyper_p, collide_cutoff
    bc_kind: str,
) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
    """
    Build A_real for S species, size (S*N*Nx, S*N*Nx). P maps a scalar c0(x) to E(x) for the grid/bc:
        E = P @ c0
    Field coupling across species is embedded in A_real via (q_s/m_s) P q_r between (n=1 <- n=0) blocks.
    Returns (A_real, P, meta).
    """
    S_count = len(species)
    D = upwind_D(Nx, L, bc_kind)                  # (Nx,Nx)
    P = build_P(Nx, L, bc_kind)                   # (Nx,Nx)
    I_x = jnp.eye(Nx, dtype=jnp.float64)
    S_h = _hermite_streaming_blocks(N)            # (N,N)
    I_h = jnp.eye(N, dtype=jnp.float64)

    # Per-species diagonal parts
    diag_blocks = []
    for sp in species:
        vth = float(getattr(sp, "vth", 1.0))
        u0  = float(getattr(sp, "u0",  0.0))
        # streaming + drift
        A_stream = _kron(vth * S_h, D)            # vth_s * S ⊗ D
        A_drift  = _kron(I_h, u0 * D)             # u0_s  * I ⊗ D
        # collisions
        Cn = build_collision_matrix(N,
                                    float(getattr(sp, "nu0", 0.0)),
                                    int(getattr(sp, "hyper_p", 0)),
                                    int(getattr(sp, "collide_cutoff", 3)))
        A_coll = _kron(Cn, I_x)
        diag_blocks.append((A_stream + A_drift) - A_coll)   # (N*Nx, N*Nx)

    A_diag = _block_diag(diag_blocks)             # (S*N*Nx, S*N*Nx)

    # Cross-species field coupling (n=1 <- n=0) via P, weighted by q_s/m_s and q_r
    SNNx = S_count * N * Nx
    A_field = jnp.zeros((SNNx, SNNx), dtype=jnp.float64)

    def blk_slice(spec_idx: int, n: int) -> slice:
        base = spec_idx * (N * Nx)
        return slice(base + n * Nx, base + (n + 1) * Nx)

    for si, sp_s in enumerate(species):
        q_s = float(sp_s.q); m_s = float(sp_s.m)
        r_slice = blk_slice(si, 1)   # receiver: n=1 of species s
        for rj, sp_r in enumerate(species):
            q_r = float(sp_r.q)
            c_slice = blk_slice(rj, 0)   # source: n=0 of species r
            A_field = A_field.at[r_slice, c_slice].set((q_s / m_s) * (P * q_r))

    A_real = (A_diag + A_field).astype(jnp.float64)
    meta = {}  # reserved for future diagnostics if needed
    return A_real, P.astype(jnp.float64), meta


def initial_condition_multispecies(
    Nx: int,
    L: float,
    N: int,
    species: Sequence,
) -> jnp.ndarray:
    """
    Return C0_Snx: (S, N, Nx) with species-specific cosine perturbations:
      c0^{(s)}(x,0) = amplitude_s * cos(2π * k_s * x / L)
      optional small seed in n=1 if species.seed_c1 is True.
    """
    x = jnp.linspace(0.0, L, Nx, endpoint=False, dtype=jnp.float64)
    S = len(species)
    C0 = jnp.zeros((S, N, Nx), dtype=jnp.float64)

    for si, sp in enumerate(species):
        amp = float(getattr(sp, "amplitude", 0.0))
        ks  = getattr(sp, "k", None)   # cycles in box
        if ks is None:
            continue
        phase = 2.0 * jnp.pi * float(ks) * x / float(L)
        cosx  = jnp.cos(phase)  # (Nx,)
        C0 = C0.at[si, 0, :].set(amp * cosx)
        if bool(getattr(sp, "seed_c1", False)):
            C0 = C0.at[si, 1, :].set(0.1 * amp * cosx)

    return C0


# -------------------- Multispecies solver (coupled) --------------------
def solve_dg_multispecies(
    A_real: jnp.ndarray,   # (S*N*Nx, S*N*Nx)
    C0_Snx: jnp.ndarray,   # (S, N, Nx)
    tmax: float,
    nt: int,
    backend: str,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evolve the full coupled multispecies system:
      dC/dt = A_real C
    where C is the stacked vector of shape (S*N*Nx,).

    Returns:
      ts   : (nt,)
      C_St : (S, N, Nx, nt)
    """
    S, N, Nx = C0_Snx.shape
    y0 = jnp.reshape(C0_Snx, (S * N * Nx,))
    ts = jnp.linspace(0.0, tmax, nt, dtype=jnp.float64)

    if backend == "eig":
        w, V = jnp.linalg.eig(A_real.astype(jnp.complex128))
        Vinv = jnp.linalg.inv(V)
        alpha = Vinv @ y0
        phases = jnp.exp(w[:, None] * ts[None, :])
        Yc = V @ (phases * alpha[:, None])    # complex
        Y = jnp.real(Yc)                      # A_real is real
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
        Y = sol.ys.T  # (S*N*Nx, nt)

    C_St = jnp.reshape(Y, (S, N, Nx, nt))
    return jnp.asarray(ts), jnp.asarray(C_St)
