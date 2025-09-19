"""
Fourier–Hermite mode bank (multi-species only).
We build, per k, a coupled block system across species:
  A(k) = (-i) * (H_tot + H_field) - C_tot
and evolve either by eig or Diffrax (real 2M system under the hood).

ICs are species-driven: each species s seeds its Hermite n=0 (and optionally n=1)
on the spatial Fourier mode matching its requested cycles k_s,
i.e. cos(2π k_s x / L) → equal power at ±k0_phys, so each matching k gets amp/2.
"""

from typing import Tuple, Sequence
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from hermite_ops import (
    streaming_block_fourier,   # H_stream(k,N) real
    field_one_sided_fourier,   # H_field(k,N)  real
    build_collision_matrix     # C(N; nu0, hyper_p, cutoff) real
)

# Optional Diffrax
try:
    from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController
    HAS_DIFFRAX = True
except Exception:
    HAS_DIFFRAX = False


# ---------------- Utilities ----------------
def _match_mode(k_val: float, k0: float, tol: float) -> bool:
    # Match by |k| since cos injects power at ±k0
    return float(jnp.abs(jnp.abs(k_val) - jnp.abs(k0))) <= tol

@jax.jit
def _drift_shift(H: jnp.ndarray, k: float, u0: float) -> jnp.ndarray:
    """Add species drift: H <- H + (k*u0) * I_N (Doppler shift)."""
    N = H.shape[0]
    return H + (k * u0) * jnp.eye(N, dtype=H.dtype)

@jax.jit
def _block_diag(blocks: jnp.ndarray) -> jnp.ndarray:
    """
    Block-diagonal using a batched trick:
      blocks: (S, N, N) -> (S*N, S*N)
    """
    S, N, _ = blocks.shape
    eyeS = jnp.eye(S, dtype=blocks.dtype)
    # Kronecker eyeS ⊗ block[i] and sum with mask
    def one_row(i):
        row = []
        for j in range(S):
            row.append(jnp.where(i == j, blocks[i], jnp.zeros_like(blocks[i])))
        return jnp.block(row)
    return jnp.block([[jnp.where(i == j, blocks[i], jnp.zeros_like(blocks[i]))
                       for j in range(S)] for i in range(S)])


# ---------------- Per-k coupled operator across species ----------------
def _assemble_A_multi_for_k(
    k: float,
    species: Sequence,   # expects .q, .m, .u0, .nu0, .hyper_p, .collide_cutoff
    N: int,
) -> jnp.ndarray:
    """
    Build big complex A(k) for S species, size (S*N, S*N):

      A = (-i) * ( H_tot + H_field ) - C_tot

    - Per-species blocks: H_s = streaming + field-one-sided + drift (k*u0_s I)
                          C_s = collisions
    - Field coupling:
        E_k = i/k * sum_r q_r * c0^{(r)}
        dc1^{(s)}/dt += (q_s/m_s) * E_k
        =>
        H_field[(s,1),(r,0)] = (q_s/m_s) * (q_r) * (1/k)    (real), k=0 -> 0
    """
    S = len(species)

    # per-species H_s and C_s (real float64)
    H_blocks = []
    C_blocks = []
    for sp in species:
        Hs = streaming_block_fourier(k, N) + field_one_sided_fourier(k, N)
        Hs = _drift_shift(Hs, k, float(getattr(sp, "u0", 0.0)))
        Cs = build_collision_matrix(
            N,
            float(getattr(sp, "nu0", 0.0)),
            int(getattr(sp, "hyper_p", 0)),
            int(getattr(sp, "collide_cutoff", 3)),
        )
        H_blocks.append(Hs)
        C_blocks.append(Cs)

    H_tot = _block_diag(jnp.stack(H_blocks, axis=0))   # (S*N, S*N)
    C_tot = _block_diag(jnp.stack(C_blocks, axis=0))   # (S*N, S*N)

    # field coupling H_field: only (n=1 <- n=0) across species
    SNN = S * N
    H_field = jnp.zeros((SNN, SNN), dtype=jnp.float64)
    inv_k = 0.0 if (k == 0.0) else (1.0 / k)
    for si, sp_s in enumerate(species):
        for rj, sp_r in enumerate(species):
            row = si * N + 1  # n=1
            col = rj * N + 0  # n=0
            H_field = H_field.at[row, col].set(
                (float(sp_s.q) / float(sp_s.m)) * (float(sp_r.q)) * inv_k
            )

    # complex generator
    A = (-1j) * (H_tot + H_field).astype(jnp.complex128) - C_tot.astype(jnp.complex128)
    return A


# ---------------- Diffrax block evolve (2M real system) ----------------
def _diffrax_evolve_block(Ar: jnp.ndarray, Ai: jnp.ndarray, base: jnp.ndarray,
                          ts: jnp.ndarray, tmax: float) -> jnp.ndarray:
    if not HAS_DIFFRAX:
        raise RuntimeError("Diffrax not installed.")

    def rhs(t, y, args):
        Ar, Ai = args
        M = Ar.shape[0]
        x = y[:M]; z = y[M:]
        dx = Ar @ x - Ai @ z
        dz = Ai @ x + Ar @ z
        return jnp.concatenate([dx, dz])

    y0 = jnp.concatenate([jnp.real(base), jnp.imag(base)])
    ctrl = PIDController(rtol=1e-7, atol=1e-10, jump_ts=ts)
    sol = diffeqsolve(
        ODETerm(rhs), Tsit5(),
        t0=0.0, t1=float(tmax), dt0=1e-3,
        y0=y0, args=(Ar, Ai),
        stepsize_controller=ctrl, saveat=SaveAt(ts=ts),
        max_steps=2_000_000
    )
    M = Ar.shape[0]
    Xr, Xi = sol.ys[:, :M], sol.ys[:, M:]
    return (Xr + 1j * Xi).T


# ---------------- Public API (multi-species only) ----------------
def run_bank_multispecies(
    kvals: jnp.ndarray,
    N: int,
    species: Sequence,   # expects: q,m,n0,vth,u0,nu0,hyper_p,collide_cutoff, amplitude, k, seed_c1?
    backend: str,
    tmax: float,
    nt: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Multi-species Fourier–Hermite evolution.

    Returns:
      ts:      (nt,)
      C_kSnt:  (Nk, S, N, nt) complex
      Ek_kt:   (Nk, nt)       complex    with E_k = i/k * sum_s q_s c0^{(s)}
    """
    kvals = kvals.astype(jnp.float64)
    ts = jnp.linspace(0.0, tmax, nt, dtype=jnp.float64)
    S = len(species)

    # Infer L from Δk (kvals = 2π n / L). Handle degenerate Nk=1.
    if kvals.size < 2:
        dk = 2.0 * jnp.pi
    else:
        dk = float(kvals[1] - kvals[0])
    L_inferred = 2.0 * jnp.pi / max(dk, 1e-30)

    # Precompute target physical wavenumbers & amplitudes per species
    k0_phys_by_s = []
    amp_by_s = []
    seedc1_by_s = []
    for sp in species:
        ks = getattr(sp, "k", None)  # cycles in box
        k0_phys = 0.0 if ks is None else (2.0 * jnp.pi * float(ks) / L_inferred)
        k0_phys_by_s.append(float(k0_phys))
        amp_by_s.append(float(getattr(sp, "amplitude", 0.0)))
        seedc1_by_s.append(bool(getattr(sp, "seed_c1", False)))

    C_list = []
    E_list = []

    tol = 0.5 * abs(float(dk))  # tolerant mode match

    for k in kvals:
        # IC for this specific k: amp/2 on matching species (n=0), optional small n=1
        base_blocks = []
        for sidx, sp in enumerate(species):
            amp_s = amp_by_s[sidx]
            c0 = jnp.zeros((N,), jnp.complex128)
            if _match_mode(float(k), k0_phys_by_s[sidx], tol=tol):
                c0 = c0.at[0].set(0.5 * amp_s)           # cos → split into ±k
                if seedc1_by_s[sidx]:
                    c0 = c0.at[1].set(0.05 * amp_s)
            base_blocks.append(c0)
        base_k = jnp.concatenate(base_blocks, axis=0)  # (S*N,)

        # Coupled operator and evolution
        A = _assemble_A_multi_for_k(float(k), species, N)  # (S*N, S*N)
        if backend == "eig":
            w, V = jnp.linalg.eig(A)
            Vinv = jnp.linalg.inv(V)
            alpha = Vinv @ base_k
            phases = jnp.exp(w[:, None] * ts[None, :])
            Y = V @ (phases * alpha[:, None])               # (S*N, nt)
        else:
            Ar = jnp.real(A); Ai = jnp.imag(A)
            Y  = _diffrax_evolve_block(Ar, Ai, base_k, ts, tmax)

        C_Snt = Y.reshape((S, N, nt))               # (S, N, nt)
        C_list.append(C_Snt)

        # Field: E_k = i/k * sum_s q_s c0^{(s)} (safe k=0 → 0)
        c0_by_s = C_Snt[:, 0, :]                            # (S, nt)
        q = jnp.asarray([sp.q for sp in species], jnp.float64)[:, None]
        rho_k = jnp.sum(q * c0_by_s, axis=0)                # (nt,)
        inv_k = 0.0 if (k == 0.0) else (1.0 / k)
        E_list.append(1j * inv_k * rho_k)                   # (nt,)

    C_kSnt = jnp.stack(C_list, axis=0)   # (Nk, S, N, nt)
    Ek_kt  = jnp.stack(E_list,  axis=0)  # (Nk, nt)
    return jnp.asarray(ts), jnp.asarray(C_kSnt), jnp.asarray(Ek_kt)
