# fourier.py
"""
Fourier–Hermite mode bank
Supports:
  - Linear evolution (block-coupled across species in k-space)
  - Nonlinear evolution via pseudo-spectral x-product:
      (q_s/m_s) * E(x) * (1/vth_s) * ∂_u δf_s

State per k: for S species, N Hermite modes:
  C_kSnt[k, s, n, t] ∈ ℂ

IC per species s:
  seed n=0 on spatial mode matching k_s (cycles in box):
    cos(2π k_s x / L) injects equal power at ±k0_phys = 2π k_s / L
  ⇒ each matching k gets amplitude_s / 2 on c_{n=0}.
"""

from collections.abc import Sequence

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from diffrax import (
    ODETerm,
    PIDController,
    SaveAt,
    TqdmProgressMeter,
    Tsit5,
    diffeqsolve,
)

from spectraxgk.hermite_ops import (
    build_collision_matrix,  # C(N; nu0, hyper_p, cutoff) real
    field_one_sided_fourier,  # H_field(k,N)  real
    streaming_block_fourier,  # H_stream(k,N) real
)


# ---------------- Utilities ----------------
def _match_mode_array(k_val, k0, tol):
    """Return a JAX bool: | |k_val| - |k0| | <= tol."""
    k_val = jnp.asarray(k_val, jnp.float64)
    k0 = jnp.asarray(k0, jnp.float64)
    tol = jnp.asarray(tol, jnp.float64)
    return jnp.abs(jnp.abs(k_val) - jnp.abs(k0)) <= tol


@jax.jit
def _drift_shift(H: jnp.ndarray, k, u0: float) -> jnp.ndarray:
    """Add species drift: H <- H + (k*u0) * I_N (Doppler shift)."""
    k = jnp.asarray(k, jnp.float64)
    N = H.shape[0]
    return H + (k * u0) * jnp.eye(N, dtype=H.dtype)


def _block_diag_list(blocks: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """
    Lightweight block diagonal assembly with jnp.block (keeps JAX-compat).
    blocks: list of (N,N) -> (S*N, S*N)
    """
    S = len(blocks)
    zero = jnp.zeros_like(blocks[0])
    rows = []
    for i in range(S):
        rows.append([blocks[i] if i == j else zero for j in range(S)])
    return jnp.block(rows)


def _assemble_A_multi_for_k(
    k,  # JAX scalar
    species: Sequence,  # expects .q, .m, .u0, .nu0, .hyper_p, .collide_cutoff
    N: int,
) -> jnp.ndarray:
    """
    Build big complex A(k) for S species, size (S*N, S*N):

      A = (-i) * ( H_tot + H_field ) - C_tot

    per-species:
      H_s = streaming_block_fourier(k,N) + field_one_sided_fourier(k,N) + (k u0_s) I
      C_s = build_collision_matrix(...)
    field coupling (only n=1 <- n=0):
      H_field[(s,1),(r,0)] = (q_s/m_s) * (q_r) * (1/k)   (k=0 → 0 via jnp.where)
    """
    S = len(species)
    k = jnp.asarray(k, jnp.float64)

    # per-species real blocks
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

    H_tot = _block_diag_list(H_blocks)  # (S*N, S*N)
    C_tot = _block_diag_list(C_blocks)  # (S*N, S*N)

    # field coupling
    SNN = S * N
    H_field = jnp.zeros((SNN, SNN), dtype=jnp.float64)
    inv_k = jnp.where(k != 0.0, 1.0 / k, 0.0)
    for si, sp_s in enumerate(species):
        qs, ms = float(sp_s.q), float(sp_s.m)
        for rj, sp_r in enumerate(species):
            qr = float(sp_r.q)
            row = si * N + 1  # n=1
            col = rj * N + 0  # n=0
            H_field = H_field.at[row, col].set((qs / ms) * qr * inv_k)

    # complex generator
    A = (-1j) * (H_tot + H_field).astype(jnp.complex128) - C_tot.astype(jnp.complex128)
    return A


def _build_linear_Ak_blocks(kvals: jnp.ndarray, N: int, species: Sequence) -> jnp.ndarray:
    """
    Build big A(k) (S*N x S*N) for each k and stack: (Nk, S*N, S*N) complex.
    """
    kvals = jnp.asarray(kvals, jnp.float64)

    def one_k(kk):
        return _assemble_A_multi_for_k(kk, species, N)

    return jax.vmap(one_k, in_axes=0)(kvals)


def _dealias_mask(kvals: jnp.ndarray, frac: float) -> jnp.ndarray:
    """
    2/3 rule (default): keep |k| <= frac * k_max. Returns (Nk,) float mask {0,1}.
    """
    kvals = jnp.asarray(kvals, jnp.float64)
    if (frac <= 0.0) or (frac >= 1.0):
        return jnp.ones_like(kvals, dtype=jnp.float64)
    kmax = jnp.max(jnp.abs(kvals))
    return (jnp.abs(kvals) <= frac * kmax).astype(jnp.float64)


# ---------------- Diffrax block evolve (2M real system) ----------------
def _diffrax_evolve_block(
    Ar: jnp.ndarray, Ai: jnp.ndarray, base: jnp.ndarray, ts: jnp.ndarray, tmax: float
) -> jnp.ndarray:
    def rhs(t, y, args):
        Ar, Ai = args
        M = Ar.shape[0]
        x = y[:M]
        z = y[M:]
        dx = Ar @ x - Ai @ z
        dz = Ai @ x + Ar @ z
        return jnp.concatenate([dx, dz])

    y0 = jnp.concatenate([jnp.real(base), jnp.imag(base)])
    ctrl = PIDController(rtol=1e-7, atol=1e-10, jump_ts=ts)
    sol = diffeqsolve(
        ODETerm(rhs),
        Tsit5(),
        t0=0.0,
        t1=float(tmax),
        dt0=1e-3,
        y0=y0,
        args=(Ar, Ai),
        stepsize_controller=ctrl,
        saveat=SaveAt(ts=ts),
        progress_meter=TqdmProgressMeter(),
        max_steps=2_000_000,
    )
    M = Ar.shape[0]
    Xr, Xi = sol.ys[:, :M], sol.ys[:, M:]
    return (Xr + 1j * Xi).T


# ---------------- Linear multi-species public API ----------------
def run_bank_multispecies_linear(
    kvals: jnp.ndarray,
    N: int,
    species: Sequence,  # q,m,n0,vth,u0,nu0,hyper_p,collide_cutoff, amplitude, k, seed_c1?
    backend: str,
    tmax: float,
    nt: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Linear multi-species Fourier–Hermite evolution.

    Returns:
      ts:      (nt,)
      C_kSnt:  (Nk, S, N, nt) complex
      Ek_kt:   (Nk, nt)       complex  (E_k = i/k * Σ_s q_s c0^{(s)}, safe k=0→0)
    """
    kvals = jnp.asarray(kvals, jnp.float64)
    ts = jnp.linspace(0.0, float(tmax), int(nt), dtype=jnp.float64)
    S = len(species)
    Nk = int(kvals.shape[0])

    # infer L from Δk (k = 2π n / L)

    dk = 2.0 * jnp.pi if Nk < 2 else kvals[1] - kvals[0]  # noqa: PLR2004
    dk = jnp.asarray(dk, jnp.float64)
    L_inferred = 2.0 * jnp.pi / jnp.maximum(jnp.abs(dk), 1e-30)

    # per-species target |k0|
    k0_phys = []
    amp = []
    seedc1 = []
    for sp in species:
        cyc = getattr(sp, "k", None)
        amp.append(float(getattr(sp, "amplitude", 0.0)))
        seedc1.append(bool(getattr(sp, "seed_c1", False)))
        if cyc is None:
            k0_phys.append(0.0)
        else:
            k0_phys.append(2.0 * jnp.pi * float(cyc) / L_inferred)
    k0_phys = jnp.asarray(k0_phys, jnp.float64)
    amp = jnp.asarray(amp, jnp.float64)

    # build linear operators for each k
    A_kk = _build_linear_Ak_blocks(kvals, N, species)  # (Nk, S*N, S*N) complex

    C_list = []
    E_list = []

    # tolerance for matching |k| ≈ |k0|
    tol = 0.5 * jnp.abs(dk)

    for ki in range(Nk):
        kk = kvals[ki]
        # IC on this k: half amplitude for matching species (cos → ±k)
        base_blocks = []
        for si, _sp in enumerate(species):
            c0 = jnp.zeros((N,), jnp.complex128)
            is_match = _match_mode_array(kk, k0_phys[si], tol)
            c0 = c0.at[0].set(jnp.where(is_match, 0.5 * amp[si], 0.0))
            if seedc1[si]:
                c0 = c0.at[1].set(jnp.where(is_match, 0.05 * amp[si], 0.0))
            base_blocks.append(c0)
        base_k = jnp.concatenate(base_blocks, axis=0)  # (S*N,)

        # evolve linearly with A(k)
        Ak = A_kk[ki, :, :]
        if backend == "eig":
            w, V = jnp.linalg.eig(Ak)
            Vinv = jnp.linalg.inv(V)
            alpha = Vinv @ base_k
            phases = jnp.exp(w[:, None] * ts[None, :])  # (S*N, nt)
            Y = V @ (phases * alpha[:, None])  # (S*N, nt)
        else:
            Ar = jnp.real(Ak)
            Ai = jnp.imag(Ak)
            Y = _diffrax_evolve_block(Ar, Ai, base_k, ts, tmax)  # (S*N, nt)

        C_Snt = Y.reshape((S, N, ts.shape[0]))  # (S, N, nt)
        C_list.append(C_Snt)

        # field diagnostic: E_k = i/k * Σ_s q_s c0^{(s)}  (safe k=0→0)
        c0_by_s = C_Snt[:, 0, :]  # (S, nt)
        q = jnp.asarray([float(sp.q) for sp in species], jnp.float64)[:, None]
        rho_k_t = jnp.sum(q * c0_by_s, axis=0)  # (nt,)
        inv_k = jnp.where(kk != 0.0, 1.0 / kk, 0.0)
        E_list.append(1j * inv_k * rho_k_t)  # (nt,)

    C_kSnt = jnp.stack(C_list, axis=0)  # (Nk, S, N, nt)
    Ek_kt = jnp.stack(E_list, axis=0)  # (Nk, nt)
    return ts, C_kSnt, Ek_kt


# ---------- Hermite ∂/∂u operator (orthonormal physicists’ Hermite) ----------
def hermite_du_matrix(N: int) -> jnp.ndarray:
    """
    ∂_u φ_n = √(n/2) φ_{n-1} - √((n+1)/2) φ_{n+1}
    Matrix D with D[n,m] such that (∂_u f)_n = Σ_m D[n,m] f_m.
    """
    D = jnp.zeros((N, N), dtype=jnp.float64)
    if N > 1:
        n = jnp.arange(N, dtype=jnp.float64)
        # lower diag: n <- n+1   ( -√((n+1)/2) )
        D = D.at[jnp.arange(N - 1), jnp.arange(1, N)].set(-jnp.sqrt((n[: N - 1] + 1.0) / 2.0))
        # upper diag: n <- n-1   (  √(n/2) )
        D = D.at[jnp.arange(1, N), jnp.arange(N - 1)].set(jnp.sqrt(n[1:] / 2.0))
    return D


# ---------------- Nonlinear multi-species public API ----------------
def run_bank_multispecies_nonlinear(
    kvals: jnp.ndarray,
    N: int,
    species: Sequence,
    backend: str,
    tmax: float,
    nt: int,
    dealias_frac: float = 2.0 / 3.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Nonlinear electrostatic Vlasov–Poisson, multispecies (Fourier–Hermite).
    Pseudo-spectral NL in x, Hermite matrix in v.

    Returns:
      ts:      (nt,)
      C_kSnt:  (Nk, S, N, nt) complex
      Ek_kt:   (Nk, nt) complex
    """
    kvals = jnp.asarray(kvals, jnp.float64)
    Nk = int(kvals.shape[0])
    S = len(species)
    ts = jnp.linspace(0.0, float(tmax), int(nt), dtype=jnp.float64)

    # Linear part per-k (streaming + collisions + linear E⋅∂_v f0)
    A_kk = _build_linear_Ak_blocks(kvals, N, species)  # (Nk, S*N, S*N)

    # Hermite ∂_u and k-helpers for Poisson
    Du = hermite_du_matrix(N)  # (N,N) real
    Du = jnp.asarray(Du, jnp.float64)
    inv_k = jnp.where(kvals != 0.0, 1.0 / kvals, 0.0)  # (Nk,)
    i_over_k = 1j * inv_k  # (Nk,)
    dealias = _dealias_mask(kvals, dealias_frac)  # (Nk,)
    dealias_c = dealias.astype(jnp.complex128)

    # Infer L from Δk for mapping cycles→physical k0
    dk = 2.0 * jnp.pi if Nk < 2 else kvals[1] - kvals[0]  # noqa: PLR2004
    dk = jnp.asarray(dk, jnp.float64)
    L_inferred = 2.0 * jnp.pi / jnp.maximum(jnp.abs(dk), 1e-30)

    # IC: C_kSnt0 (Nk, S, N) at t=0
    k0_phys = []
    amp = []
    for sp in species:
        cyc = getattr(sp, "k", None)
        amp.append(float(getattr(sp, "amplitude", 0.0)))
        if cyc is None:
            k0_phys.append(0.0)
        else:
            k0_phys.append(2.0 * jnp.pi * float(cyc) / L_inferred)
    k0_phys = jnp.asarray(k0_phys, jnp.float64)
    amp = jnp.asarray(amp, jnp.float64)

    # build mask per species for matching |k| ≈ |k0|
    tol = 0.5 * jnp.abs(dk)
    # C_kSnt0 init
    C_kSnt0 = jnp.zeros((Nk, S, N), dtype=jnp.complex128)
    for si in range(S):
        match_si = _match_mode_array(kvals, k0_phys[si], tol)  # (Nk,)
        seed = (0.5 * amp[si]) * match_si.astype(jnp.float64)  # (Nk,)
        C_kSnt0 = C_kSnt0.at[:, si, 0].set(C_kSnt0[:, si, 0] + seed)  # put in n=0

    # pack/unpack for real 2M integration
    M = Nk * S * N

    def pack(C_kSnt):
        z = jnp.reshape(C_kSnt, (M,))
        return jnp.concatenate([jnp.real(z), jnp.imag(z)], axis=0)

    def unpack(y):
        yr, yi = jnp.split(y, 2, axis=0)
        z = yr + 1j * yi
        return jnp.reshape(z, (Nk, S, N))

    # linear term: vmap over k to apply A(k) @ y_k
    def lin_term(C_kSnt):
        Y = C_kSnt.reshape((Nk, S * N))  # (Nk,S*N)

        def one_k(Ak, yk):
            return Ak @ yk  # (S*N,)

        Z = jax.vmap(one_k, in_axes=(0, 0))(A_kk, Y)  # (Nk,S*N)
        return Z.reshape((Nk, S, N))

    # nonlinear term via physical x product: (q_s/m_s)/vth_s * E(x) * (∂_u δf_s)
    q = jnp.asarray([float(getattr(sp, "q", -1.0)) for sp in species], jnp.float64)  # (S,)
    m = jnp.asarray([float(getattr(sp, "m", 1.0)) for sp in species], jnp.float64)  # (S,)
    vT = jnp.asarray([float(getattr(sp, "vth", 1.0)) for sp in species], jnp.float64)  # (S,)
    scale_s = (q / m) / jnp.maximum(vT, 1e-30)  # (S,)

    def nl_term(C_kSnt):
        # E from Poisson: E_k = (i/k) * Σ_s q_s c0_{k,s}   (k=0→0)
        c0_kS = C_kSnt[:, :, 0]  # (Nk,S)
        rho_k = (c0_kS * q[None, :]).sum(axis=1)  # (Nk,)
        E_k = i_over_k * rho_k  # (Nk,)
        E_k = dealias_c * E_k
        E_x = jnp.real(jnp.fft.ifft(E_k, axis=0))  # (Nx,) with Nx=Nk

        # to x-space: (S,N,Nx)
        C_SNx = jnp.transpose(jnp.fft.ifft(jnp.transpose(C_kSnt, (1, 2, 0)), axis=2), (0, 1, 2))
        C_SNx = jnp.real(C_SNx)

        # ∂_u in Hermite: (N,N) @ (N,Nx) per species
        out_SNx = []
        for si in range(S):
            Gn = Du @ C_SNx[si, :, :]  # (N,Nx)
            out_SNx.append(scale_s[si] * (Gn * E_x[None, :]))
        NL_SNx = jnp.stack(out_SNx, axis=0)  # (S,N,Nx)

        # back to k and dealiased
        NL_kSN = jnp.transpose(
            jnp.fft.fft(jnp.transpose(NL_SNx, (0, 1, 2)), axis=2), (2, 0, 1)
        )  # (Nk,S,N)
        NL_kSN = dealias_c[:, None, None] * NL_kSN
        return NL_kSN

    # full RHS in complex space
    def rhs_complex(t, C_kSnt):
        return lin_term(C_kSnt) + nl_term(C_kSnt)

    # integrate
    if backend == "eig":
        # eig is not applicable for nonlinear; switch to diffrax
        backend = "diffrax"

    y0 = pack(C_kSnt0)

    if backend == "diffrax":

        def rhs_real(t, y, _):
            C = unpack(y)
            dC = rhs_complex(t, C)
            return pack(dC)

        term = ODETerm(rhs_real)
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
            saveat=SaveAt(ts=ts),
            progress_meter=TqdmProgressMeter(),
            max_steps=4_000_000,
        )
        ys = sol.ys  # (nt, 2M)
        C_seq = [unpack(ys[i]) for i in range(int(nt))]
        C_kSnt = jnp.stack(C_seq, axis=-1)  # (Nk,S,N,nt)
    else:
        raise RuntimeError("Nonlinear Fourier mode requires backend='diffrax' currently.")

    # diagnostics: E_k(t) from c0
    c0_k_t = C_kSnt[:, :, 0, :]  # (Nk,S,nt)
    q_vec = jnp.asarray([float(getattr(sp, "q", -1.0)) for sp in species], jnp.float64)
    rho_kt = jnp.sum(c0_k_t * q_vec[None, :, None], axis=1)  # (Nk,nt)
    inv_k_col = jnp.where(kvals[:, None] != 0.0, 1.0 / kvals[:, None], 0.0)
    Ek_kt = 1j * inv_k_col * rho_kt

    return ts, C_kSnt, Ek_kt
