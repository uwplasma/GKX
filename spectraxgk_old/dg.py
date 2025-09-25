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

from collections.abc import Sequence
from functools import partial

import jax
from diffrax import ODETerm, PIDController, SaveAt, TqdmProgressMeter, Tsit5, diffeqsolve

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from spectraxgk.hermite_ops import build_collision_matrix, ladder_sqrt
from spectraxgk.poisson import build_P


# -------------------- DG derivative --------------------
@partial(jax.jit, static_argnames=("Nx",))
def _D_periodic(Nx: int, L: float) -> jnp.ndarray:
    dx = jnp.asarray(L, jnp.float64) / jnp.asarray(Nx, jnp.float64)
    eye = jnp.eye(Nx, dtype=jnp.float64)
    S_plus = jnp.roll(eye, -1, axis=1)
    S_minus = jnp.roll(eye, 1, axis=1)
    return (S_plus - S_minus) * (0.5 / dx)


@partial(jax.jit, static_argnames=("Nx",))
def _D_dirichlet(Nx: int, L: float) -> jnp.ndarray:
    dx = jnp.asarray(L, jnp.float64) / jnp.asarray(Nx, jnp.float64)
    D = jnp.zeros((Nx, Nx), dtype=jnp.float64)
    idx = jnp.arange(1, Nx - 1)
    D = D.at[idx, idx - 1].set(-0.5 / dx)
    D = D.at[idx, idx + 1].set(+0.5 / dx)
    D = D.at[0, 0].set(-1.0 / dx).at[0, 1].set(+1.0 / dx)
    D = D.at[-1, -2].set(-1.0 / dx).at[-1, -1].set(+1.0 / dx)
    return D


@partial(jax.jit, static_argnames=("Nx",))
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
    S = S.at[jnp.arange(N - 1), jnp.arange(1, N)].set(s)  # super
    S = S.at[jnp.arange(1, N), jnp.arange(N - 1)].set(s)  # sub
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
    species: Sequence,  # expects objects with attributes: q, m, n0, vth, u0, nu0, hyper_p, collide_cutoff
    bc_kind: str,
):
    """
    Build A_real for S species, size (S*N*Nx, S*N*Nx). Also return P (Poisson map),
    D (DG derivative), and x (grid).  Layout is block-diagonal per species for
    streaming/drift/collisions, plus cross-species field coupling on (n=1 <- n=0).
    """
    # --- grids & operators in x ---
    Nx_i = int(Nx)
    L_f = float(L)
    x = jnp.linspace(0.0, L_f, Nx_i, endpoint=False, dtype=jnp.float64)
    D = upwind_D(Nx_i, L_f, bc_kind)  # (Nx,Nx)
    P = build_P(Nx_i, L_f, bc_kind)  # (Nx,Nx)
    I_x = jnp.eye(Nx_i, dtype=jnp.float64)

    # --- Hermite operators ---
    S_h = _hermite_streaming_blocks(N)  # (N,N)
    I_h = jnp.eye(N, dtype=jnp.float64)

    # --- per-species diagonal blocks (streaming + drift - collisions) ---
    diag_blocks = []
    for sp in species:
        vth = float(getattr(sp, "vth", 1.0))
        u0 = float(getattr(sp, "u0", 0.0))
        # streaming + drift
        A_stream = _kron(vth * S_h, D)  # vth_s * S ⊗ D
        A_drift = _kron(I_h, u0 * D)  # u0_s  * I ⊗ D
        # collisions
        Cn = build_collision_matrix(
            N,
            float(getattr(sp, "nu0", 0.0)),
            int(getattr(sp, "hyper_p", 0)),
            int(getattr(sp, "collide_cutoff", 3)),
        )
        A_coll = _kron(Cn, I_x)
        diag_blocks.append((A_stream + A_drift) - A_coll)  # (N*Nx, N*Nx)

    A_diag = _block_diag(diag_blocks)  # (S*N*Nx, S*N*Nx)

    # --- cross-species field coupling: (n=1 of s) gets (q_s/m_s) * E; E = P * sum_r q_r c0^{(r)} ---
    S_count = len(species)
    SNNx = S_count * N * Nx_i
    A_field = jnp.zeros((SNNx, SNNx), dtype=jnp.float64)

    def blk_slice(spec_idx: int, n: int) -> slice:
        base = spec_idx * (N * Nx_i)
        return slice(base + n * Nx_i, base + (n + 1) * Nx_i)

    for si, sp_s in enumerate(species):
        q_s = float(sp_s.q)
        m_s = float(sp_s.m)
        r_slice = blk_slice(si, 1)  # receiver: species s, n=1
        for rj, sp_r in enumerate(species):
            q_r = float(sp_r.q)
            c_slice = blk_slice(rj, 0)  # source: species r, n=0
            # place (q_s/m_s) * (P * q_r) into that cross-block
            A_field = A_field.at[r_slice, c_slice].set((q_s / m_s) * (P * q_r))

    A_real = A_diag + A_field
    return (
        A_real.astype(jnp.float64),
        P.astype(jnp.float64),
        D.astype(jnp.float64),
        x.astype(jnp.float64),
    )


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
        ks = getattr(sp, "k", None)  # cycles in box
        if ks is None:
            continue
        phase = 2.0 * jnp.pi * float(ks) * x / float(L)
        cosx = jnp.cos(phase)  # (Nx,)
        C0 = C0.at[si, 0, :].set(amp * cosx)
        if bool(getattr(sp, "seed_c1", False)):
            C0 = C0.at[si, 1, :].set(0.1 * amp * cosx)

    return C0


# -------------------- Multispecies solver (coupled) --------------------
def solve_dg_multispecies(
    A_real: jnp.ndarray,  # (S*N*Nx, S*N*Nx)
    C0_Snx: jnp.ndarray,  # (S, N, Nx)
    tmax: float,
    nt: int,
    backend: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        Yc = V @ (phases * alpha[:, None])  # complex
        Y = jnp.real(Yc)  # A_real is real
    else:
        term = ODETerm(lambda t, y, A: A @ y)
        controller = PIDController(rtol=1e-7, atol=1e-10, jump_ts=ts)
        sol = diffeqsolve(
            term,
            Tsit5(),
            t0=0.0,
            t1=float(tmax),
            dt0=1e-3,
            y0=y0,
            args=A_real,
            stepsize_controller=controller,
            saveat=SaveAt(ts=ts),
            progress_meter=TqdmProgressMeter(),
            max_steps=2_000_000,
        )
        Y = sol.ys.T  # (S*N*Nx, nt)

    C_St = jnp.reshape(Y, (S, N, Nx, nt))
    return jnp.asarray(ts), jnp.asarray(C_St)


def hermite_du_matrix(N: int) -> jnp.ndarray:
    D = jnp.zeros((N, N), dtype=jnp.float64)
    n = jnp.arange(N)
    if N > 1:
        D = D.at[jnp.arange(N - 1), jnp.arange(1, N)].set(-jnp.sqrt((n[: N - 1] + 1) / 2.0))
        D = D.at[jnp.arange(1, N), jnp.arange(N - 1)].set(jnp.sqrt(n[1:] / 2.0))
    return D


def solve_dg_multispecies_linear(
    A_real: jnp.ndarray,
    C0_Snx: jnp.ndarray,  # (S,N,Nx)
    tmax: float,
    nt: int,
    backend: str,
):
    """
    Linear multispecies DG: evolve each species with the same *linear* operator A_real
    (which already includes inter-species linear field coupling via P and charges).
    """
    S, N, Nx = C0_Snx.shape
    ts = jnp.linspace(0.0, float(tmax), int(nt), dtype=jnp.float64)

    def lin_solve(C0):
        y0 = jnp.reshape(C0, (N * Nx,))
        term = ODETerm(lambda t, y, A: A @ y)
        controller = PIDController(rtol=1e-7, atol=1e-10, jump_ts=ts)
        sol = diffeqsolve(
            term,
            Tsit5(),
            t0=0.0,
            t1=float(tmax),
            dt0=1e-3,
            y0=y0,
            args=A_real,
            stepsize_controller=controller,
            saveat=SaveAt(ts=ts),
            progress_meter=TqdmProgressMeter(),
            max_steps=4_000_000,
        )
        Y = sol.ys.T  # (N*Nx,nt)
        return jnp.reshape(Y, (N, Nx, nt))

    C_list = [lin_solve(C0_Snx[s]) for s in range(S)]
    C_St = jnp.stack(C_list, axis=0)  # (S,N,Nx,nt)
    return ts, C_St


def solve_dg_multispecies_nonlinear(
    A_real: jnp.ndarray,  # linear part (streaming + drift + collisions + linear field)
    P: jnp.ndarray,  # (Nx,Nx) E = P @ rho
    species,
    C0_Snx: jnp.ndarray,  # (S,N,Nx)
    tmax: float,
    nt: int,
    backend: str,
):
    """
    Nonlinear DG: y' = A_real y + NL(y) with NL_s = (q_s/m_s) * (E(x)/vth_s) * (∂_u δf_s)_proj
    where δf_s is represented by Hermite coefficients C_s[:, x], and ∂_u is a fixed (N×N) matrix.
    """
    if backend != "diffrax":
        # Nonlinear system needs a time-dependent RHS; keep Tsit5
        backend = "diffrax"

    S, N, Nx = C0_Snx.shape
    ts = jnp.linspace(0.0, float(tmax), int(nt), dtype=jnp.float64)
    Du = hermite_du_matrix(N)  # (N,N)

    # Pack S species into one long vector
    def pack(C_St):
        return jnp.reshape(C_St, (S * N * Nx,))

    def unpack(y):
        return jnp.reshape(y, (S, N, Nx))

    # Apply A_real on each species (block-diagonal in species)
    def apply_linear(C_St):
        # (S,N,Nx) -> (S,N,Nx)   via matvec per species
        def one(C):
            Y = A_real @ jnp.reshape(C, (N * Nx,))
            return jnp.reshape(Y, (N, Nx))

        return jax.vmap(one, in_axes=0)(C_St)

    # Nonlinear term: build E(x) from rho = Σ_s q_s c0_s, then Hermite-du per species
    q = jnp.asarray([sp.q for sp in species], jnp.float64)[:, None]  # (S,1)
    vth = jnp.asarray([max(float(getattr(sp, "vth", 1.0)), 1e-30) for sp in species], jnp.float64)[
        :, None
    ]

    def apply_nonlinear(C_St):
        # rho(x) = Σ_s q_s * c0_s(x)
        c0_Sx = C_St[:, 0, :]  # (S,Nx)
        rho_x = jnp.sum(q * c0_Sx, axis=0)  # (Nx,)
        E_x = P @ rho_x  # (Nx,)
        # per-species: (q_s/m_s)/vth_s * (Du @ C_s) * E(x)
        ms = jnp.asarray([sp.m for sp in species], jnp.float64)[:, None]
        fac = (jnp.asarray([sp.q for sp in species], jnp.float64)[:, None] / ms) / vth  # (S,1)

        def one_species(C_s, f):
            G = Du @ C_s  # (N,Nx)
            return f * (G * E_x[None, :])  # (N,Nx)

        return jax.vmap(one_species, in_axes=(0, 0))(C_St, fac)  # (S,N,Nx)

    def rhs(t, y, _):
        C = unpack(y)
        L = apply_linear(C)
        Nl = apply_nonlinear(C)
        return pack(L + Nl)

    y0 = pack(C0_Snx)
    term = ODETerm(rhs)
    controller = PIDController(rtol=1e-7, atol=1e-10, jump_ts=ts)
    sol = diffeqsolve(
        term,
        Tsit5(),
        t0=0.0,
        t1=float(tmax),
        dt0=1e-3,
        y0=y0,
        args=None,
        stepsize_controller=controller,
        progress_meter=TqdmProgressMeter(),
        saveat=SaveAt(ts=ts),
        max_steps=6_000_000,
    )
    Y = sol.ys.T  # (S*N*Nx, nt)
    C_St = jnp.reshape(Y, (S, N, Nx, int(nt)))
    return ts, C_St
