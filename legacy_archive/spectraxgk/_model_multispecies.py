# spectraxgk/_model_multispecies.py
from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial

__all__ = [
    "enforce_conjugate_symmetry_fftshifted",
    "build_Hk_from_Gk_phi",
    "solve_phi_quasineutrality_multispecies",
    "collision_lenard_bernstein_conserving_multispecies",
    "rhs_gk_multispecies",
    "cheap_diagnostics_multispecies",
    "shift_m",
]

DEN_EPS = 1e-30


def enforce_conjugate_symmetry_fftshifted(Ak: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Enforce Ak(k) = conj(Ak(-k)) in fftshifted ordering.
    """
    iy, ix, iz = params["conj_y"], params["conj_x"], params["conj_z"]
    Aneg = jnp.take(Ak, iy, axis=-3)
    Aneg = jnp.take(Aneg, ix, axis=-2)
    Aneg = jnp.take(Aneg, iz, axis=-1)
    return 0.5 * (Ak + jnp.conj(Aneg))


def build_Hk_from_Gk_phi(Gk: jnp.ndarray, phi_k: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    H_{s,ℓ,m} = G_{s,ℓ,m} + (q_s/T_s) * J_ℓ(b_s) * φ * δ_{m0}

    This is the standard h = g + (q/T) J0 φ mapping in LH moments.
    """
    rdt = Gk.real.dtype
    q_s = jnp.asarray(params["q_s"], dtype=rdt)
    T_s = jnp.asarray(params["T_s"], dtype=rdt)
    a_s = (q_s / T_s)[:, None, None, None, None]             # (Ns,1,1,1,1)
    Jl_s = jnp.asarray(params["Jl_s"], dtype=rdt)             # (Ns,Nl,Ny,Nx,Nz)

    add_m0 = a_s * (Jl_s * phi_k[None, None, ...])            # (Ns,Nl,Ny,Nx,Nz)
    m0 = Gk[:, :, 0, ...] + add_m0.astype(Gk.dtype)
    rest = Gk[:, :, 1:, ...]
    return jnp.concatenate([m0[:, :, None, ...], rest], axis=2)


def shift_m(H: jnp.ndarray, dm: int) -> jnp.ndarray:
    """
    Shift Hermite index m by dm with zero-fill at boundaries.
    """
    if dm == 0:
        return H
    z = jnp.zeros_like(H[:, :, :1, ...])
    if dm == +1:
        return jnp.concatenate([H[:, :, 1:, ...], z], axis=2)
    if dm == -1:
        return jnp.concatenate([z, H[:, :, :-1, ...]], axis=2)
    raise ValueError("dm must be -1,0,+1")


def _shift_l(H: jnp.ndarray, dl: int) -> jnp.ndarray:
    """
    Shift Laguerre index ℓ by dl with zero-fill at boundaries.
    """
    if dl == 0:
        return H
    z = jnp.zeros_like(H[:, :1, ...])
    if dl == +1:
        return jnp.concatenate([H[:, 1:, ...], z], axis=1)
    if dl == -1:
        return jnp.concatenate([z, H[:, :-1, ...]], axis=1)
    raise ValueError("dl must be -1,0,+1")


def solve_phi_quasineutrality_multispecies(Gk: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Solve electrostatic quasineutrality in Fourier space:

      den(k) * φ_k = Σ_s q_s n0_s Σ_ℓ J_ℓ(b_s) * G_{s,ℓ,m=0}

    with:
      den(k) = Σ_s (q_s^2 n0_s / T_s) * (1 - Γ0_s(k)) + λ_D^2 k^2

    and a gauge fix:
      φ(k=0) = 0

    Dealiasing:
      φ is zeroed where mask23 is False.
    """
    q_s, n0_s = params["q_s"], params["n0_s"]
    Jl_s = params["Jl_s"]                  # (Ns,Nl,Ny,Nx,Nz)
    den = params["den_qn"]                 # (Ny,Nx,Nz)
    mask23 = params["mask23"]

    g_m0 = Gk[:, :, 0, ...]                # (Ns,Nl,Ny,Nx,Nz)
    num_s = jnp.sum(Jl_s * g_m0, axis=1)   # (Ns,Ny,Nx,Nz)
    num = jnp.sum((q_s * n0_s)[:, None, None, None] * num_s, axis=0)

    phi = jnp.where(jnp.abs(den) > DEN_EPS, num / den, 0.0 + 0.0j)

    Ny, Nx, Nz = phi.shape
    phi = phi.at[Ny//2, Nx//2, Nz//2].set(0.0 + 0.0j)

    return jnp.where(mask23, phi, 0.0 + 0.0j)


def solve_phi_from_H_multispecies(Hk: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Quasineutrality solved using H as the state.

    Using h = g + (q/T) J_l phi δ_{m0} and the standard GK quasineutrality,
    one obtains a closed form:

        den_h(k) * phi_k = Σ_s q_s n0_s Σ_l J_l(b_s) * H_{s,l,m=0}(k)

    where:
        den_h(k) = Σ_s (q_s^2 n0_s / T_s) + lambda_D^2 k^2
               = params["den_h"].
    """
    q_s, n0_s = params["q_s"], params["n0_s"]
    Jl_s = params["Jl_s"]          # (Ns,Nl,Ny,Nx,Nz)
    den_h = params["den_h"]        # (Ny,Nx,Nz)
    mask23 = params["mask23"]

    h_m0 = Hk[:, :, 0, ...]                         # (Ns,Nl,Ny,Nx,Nz)
    num_s = jnp.sum(Jl_s * h_m0, axis=1)            # (Ns,Ny,Nx,Nz)
    num = jnp.sum((q_s * n0_s)[:, None, None, None] * num_s, axis=0)

    phi = jnp.where(jnp.abs(den_h) > DEN_EPS, num / den_h, 0.0 + 0.0j)
    Ny, Nx, Nz = phi.shape
    phi = phi.at[Ny//2, Nx//2, Nz//2].set(0.0 + 0.0j)
    return jnp.where(mask23, phi, 0.0 + 0.0j)

def build_Gk_from_Hk_phi(Hk: jnp.ndarray, phi_k: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Inverse map: G = H - (q/T) J_l phi δ_{m0}
    """
    rdt = Hk.real.dtype
    q_s = jnp.asarray(params["q_s"], dtype=rdt)
    T_s = jnp.asarray(params["T_s"], dtype=rdt)
    a_s = (q_s / T_s)[:, None, None, None, None]     # (Ns,1,1,1,1)
    Jl_s = jnp.asarray(params["Jl_s"], dtype=rdt)     # (Ns,Nl,Ny,Nx,Nz)
    sub_m0 = a_s * (Jl_s * phi_k[None, None, ...])    # (Ns,Nl,Ny,Nx,Nz)
    g_m0 = Hk[:, :, 0, ...] - sub_m0.astype(Hk.dtype)
    rest = Hk[:, :, 1:, ...]
    return jnp.concatenate([g_m0[:, :, None, ...], rest], axis=2)

def collision_lenard_bernstein_conserving_multispecies(Hk: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Conserving Lenard–Bernstein model operator in LH moments (sparse form).
    """
    nu_s = params["nu_s"]
    b_s = params["b_s"]
    Jl_s = params["Jl_s"]
    Jm1_s = params["Jl_m1_s"]
    coeff_T_s = params["coeff_T_s"]
    ell = params["ell"]
    m_index = params["m_index"]

    def one_species(Hs, nus, bs, Jls, Jm1s, coeffTs):
        rdt = Hs.real.dtype
        nus = jnp.asarray(nus, dtype=rdt)
        bs = jnp.asarray(bs, dtype=rdt)
        Jls = jnp.asarray(Jls, dtype=rdt)
        Jm1s = jnp.asarray(Jm1s, dtype=rdt)
        coeffTs = jnp.asarray(coeffTs, dtype=rdt)
        ell_loc = jnp.asarray(ell, dtype=rdt)[:, None, ...]               # (Nl,1,Ny,Nx,Nz)
        m_loc = jnp.asarray(m_index, dtype=rdt)[None, :, None, None, None]

        def _do(_):
            # Velocity moments needed for conserving form
            u_par  = jnp.sum(Jls * Hs[:, 1, ...], axis=0)
            T_par  = jnp.sqrt(jnp.asarray(2.0, dtype=rdt)) * jnp.sum(Jls * Hs[:, 2, ...], axis=0)
            u_perp = jnp.sqrt(bs) * jnp.sum((Jls + Jm1s) * Hs[:, 0, ...], axis=0)

            Top = (jnp.asarray(1.5, dtype=rdt) * coeffTs)
            T_perp = jnp.sum(Top * Hs[:, 0, ...], axis=0)
            Ttot = (T_par + 2.0 * T_perp)

            C = -nus * (bs[None, None, ...] + 2.0 * ell_loc + m_loc) * Hs
            C = C.at[:, 1, ...].add(nus * (Jls * u_par))
            C = C.at[:, 2, ...].add(nus * jnp.sqrt(jnp.asarray(2.0/3.0, dtype=rdt)) * (Jls * Ttot))
            C0_add = nus * (jnp.sqrt(bs) * (Jls + Jm1s) * u_perp + coeffTs * Ttot)
            C = C.at[:, 0, ...].add(C0_add)
            return C.astype(Hs.dtype)

        return jax.lax.cond(nus == 0.0, lambda _: jnp.zeros_like(Hs), _do, operand=None)

    return jax.vmap(one_species)(Hk, nu_s, b_s, Jl_s, Jm1_s, coeff_T_s)


def cheap_diagnostics_multispecies(Gk: jnp.ndarray, params: dict) -> dict:
    """
    Diagnostics consistent with evolving G (Eq. 3.20).

    Conserved quadratic (collisionless, nonlinear-off) includes field/polarization:
      W = 1/2 Σ_s n0_s T_s ||G_s||^2  +  1/2 Σ_k den_qn(k) |phi_k|^2
    where den_qn is the quasineutrality/polarization denominator used to solve phi(G).
    (Multi-species generalization of the single-species expressions around Eq. 3.20–3.24.)
    """
    rdt = Gk.real.dtype
    # Ensure we're on the same manifold as RHS (dealias is enforced inside RHS too)
    mask23 = params["mask23"]
    Gk = jnp.where(mask23[None, None, None, ...], Gk, 0.0 + 0.0j)
    if bool(params.get("enforce_reality", True)):
        Gk = enforce_conjugate_symmetry_fftshifted(Gk, params)

    phi_k = solve_phi_quasineutrality_multispecies(Gk, params)
    den_qn = jnp.asarray(params["den_qn"], dtype=rdt)

    n0 = jnp.asarray(params["n0_s"], dtype=rdt)
    T  = jnp.asarray(params["T_s"],  dtype=rdt)

    absG2 = jnp.abs(Gk) ** 2
    W_g_s = 0.5 * (n0 * T) * jnp.sum(absG2, axis=(1,2,3,4,5))
    W_g   = jnp.sum(W_g_s)

    W_field = 0.5 * jnp.sum(den_qn * (jnp.abs(phi_k) ** 2))
    W_free = W_g + W_field

    # Helpful probes
    Hk = build_Hk_from_Gk_phi(Gk, phi_k, params)
    max_abs_phi = jnp.max(jnp.abs(phi_k))
    max_abs_G   = jnp.max(jnp.abs(Gk))

    # Hermite spectrum from G (shape (Ns,Nh))
    E_m = 0.5 * jnp.sum(absG2, axis=(1,3,4,5))  # sum over ℓ,k

    return dict(
        W_total=W_free,
        W_free=W_free,
        W_s=W_g_s,
        W_g=W_g, W_g_s=W_g_s,
        W_field=W_field,
        phi_rms=jnp.sqrt(jnp.mean(jnp.abs(phi_k) ** 2)),
        max_abs_phi=max_abs_phi,
        max_abs_G=max_abs_G,
        E_m=E_m,
    )



def _ifftz_shifted(Ak: jnp.ndarray) -> jnp.ndarray:
    # Only along z, where Ak is fftshifted.
    return jnp.fft.ifft(jnp.fft.ifftshift(Ak, axes=(-1,)), axis=-1, norm="forward")


def _fftz_shifted(A: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.fftshift(jnp.fft.fft(A, axis=-1, norm="forward"), axes=(-1,))


def _mul_gradlnB_z_in_fourier(Ak: jnp.ndarray, gradlnB_z: jnp.ndarray) -> jnp.ndarray:
    """
    Multiply by grad_parallel ln B(z), assuming gradlnB depends only on z.

    We do this as: FFT_z^{-1} Ak -> A(z), multiply by gradlnB(z), FFT_z back.
    This is efficient and avoids full 3D convolutions.
    """
    Az = _ifftz_shifted(Ak)
    Az = Az * gradlnB_z.reshape((1,) * (Az.ndim - 1) + (Az.shape[-1],))
    return _fftz_shifted(Az)


@partial(jax.jit, static_argnames=("Nh", "Nl"))
def rhs_gk_multispecies(Gk: jnp.ndarray, params: dict, Nh: int, Nl: int):
    """
    Multispecies electrostatic GK RHS in Fourier–Laguerre–Hermite moments.

    Parallel streaming operator matches the Laguerre–Hermite form, including the
    optional ∇∥ ln B coupling between ℓ and m when enable_gradB_parallel=True.

    Nonlinear term kept as pseudo-spectral E×B (can be disabled).

    Returns:
      dGk, phi_k
    """
    kx, ky, kz = params["kx_grid"], params["ky_grid"], params["kz_grid"]
    mask23 = params["mask23"]
    alpha_kln = params["alpha_kln"]
    sqrt_mp, sqrt_mm = params["sqrt_m_plus"], params["sqrt_m_minus"]

    vth_s, U_s = params["vth_s"], params["Upar_s"]

    enable_gradB_parallel = jnp.asarray(params.get("enable_gradB_parallel", False), dtype=jnp.bool_)
    enable_streaming  = jnp.asarray(params.get("enable_streaming", True), dtype=jnp.bool_)
    enable_nonlinear  = jnp.asarray(params.get("enable_nonlinear", True), dtype=jnp.bool_)
    enable_collisions = jnp.asarray(params.get("enable_collisions", True), dtype=jnp.bool_)
    enforce_reality   = jnp.asarray(params.get("enforce_reality", True), dtype=jnp.bool_)

    # Dealias always
    Gk = jnp.where(mask23[None, None, None, ...], Gk, 0.0 + 0.0j)

    # Optional symmetry (keeps real-space fields real)
    Gk = jax.lax.cond(
        enforce_reality,
        lambda x: enforce_conjugate_symmetry_fftshifted(x, params),
        lambda x: x,
        Gk,
    )

    # Fields from G (closed form)
    phi_k = solve_phi_quasineutrality_multispecies(Gk, params)

    Hk = build_Hk_from_Gk_phi(Gk, phi_k, params)

    # Fourier derivative convention used elsewhere in this file: ∂z -> (1j*kz).
    # Eq. (3.20) has "+ v_ts ∇_||(...)" on the LHS, so the RHS contribution is "- v_ts ∇_||(...)".
    # Hence we use (-1j*kz) for the RHS streaming pieces.
    kz_term = (-1j * kz)[None, None, None, ...]  # (1,1,1,Ny,Nx,Nz)
    vth = vth_s[:, None, None, None, None, None]
    U = U_s[:, None, None, None, None, None]

    # ---- Parallel streaming + optional gradlnB couplings ----
    def _stream(_):
        Hp1 = shift_m(Hk, +1)  # H_{m+1}
        Hm1 = shift_m(Hk, -1)  # H_{m-1}

        # ladder = sqrt(m+1) H_{m+1} + sqrt(m) H_{m-1}
        ladder = (
            (sqrt_mp[None, None, :, None, None, None] * Hp1)
            + (sqrt_mm[None, None, :, None, None, None] * Hm1)
        )

        # base: - vth * ∂z ladder  ->  (-i kz) vth ladder
        stream_base = kz_term * vth * ladder

        # Mean parallel flow advection is a parallel convective-derivative piece.
        # For energy consistency it must act on the same object as streaming: H(G,phi(G)).
        # RHS piece: - U ∂z H -> (-i kz) U H
        stream_U = kz_term * U * Hk

        def _no_gradB(_):
            return stream_base + stream_U

        def _with_gradB(_):
            # gradlnB(z) terms (z-only profile)
            gradlnB_z = jnp.asarray(params["gradlnB_z"], dtype=Hk.real.dtype)  # (Nz,)

            # Implement Eq. (3.20) ∇_||lnB bracket exactly:
            #   [ -(ℓ+1)√(m+1) H_{ℓ,m+1}  - ℓ√(m+1) H_{ℓ-1,m+1}
            #     +  ℓ√m     H_{ℓ,m-1}   + (ℓ+1)√m   H_{ℓ+1,m-1} ] ∇_|| ln B
            ell = jnp.asarray(params["ell_vec"], dtype=Hk.real.dtype)              # (Nl,)
            ell_b   = ell[None, :, None, None, None, None]                         # (1,Nl,1,1,1,1)
            ellp1_b = (ell + 1.0)[None, :, None, None, None, None]                 # (1,Nl,1,1,1,1)
            s_mp = sqrt_mp[None, None, :, None, None, None]                        # (1,1,Nh,1,1,1)
            s_mm = sqrt_mm[None, None, :, None, None, None]                        # (1,1,Nh,1,1,1)

            Hp1_lm1 = _shift_l(Hp1, -1)  # H_{ℓ-1,m+1}
            Hm1_lp1 = _shift_l(Hm1, +1)  # H_{ℓ+1,m-1}

            Bstuff = (
                (-ellp1_b * s_mp) * Hp1
                + (-ell_b   * s_mp) * Hp1_lm1
                + ( ell_b   * s_mm) * Hm1
                + ( ellp1_b * s_mm) * Hm1_lp1
            )

            # multiply by gradlnB(z) in real space (z-only FFTs)
            Bstuff_grad = _mul_gradlnB_z_in_fourier(Bstuff, gradlnB_z)

            # contribution: - vth * (Bstuff * gradlnB)
            stream_gradB = -vth * Bstuff_grad

            return stream_base + stream_U + stream_gradB

        return jax.lax.cond(enable_gradB_parallel, _with_gradB, _no_gradB, operand=None)

    stream = jax.lax.cond(enable_streaming, _stream, lambda _: jnp.zeros_like(Hk), operand=None)

    # NOTE:
    # The GK moment equation evolves G, but the parallel streaming operator depends on
    # H(G,phi(G)). The expression above already computes dG (using H inside the flux).
    dG_stream = stream

    # ---- Nonlinear E×B (pseudo-spectral) ----
    def _nonlinear(_):
        Jl_s = params["Jl_s"]  # (Ns,Nl,Ny,Nx,Nz)

        # species+laguerre dependent gyroaveraged potential
        phi_sk = Jl_s * phi_k[None, None, ...]  # (Ns,Nl,Ny,Nx,Nz)

        # vE in Fourier: vEx=-∂yφ, vEy=+∂xφ
        vEx_k = -1j * ky[None, None, ...] * phi_sk
        vEy_k =  1j * kx[None, None, ...] * phi_sk

        def ifftn_shifted(Ak):
            return jnp.fft.ifftn(jnp.fft.ifftshift(Ak, axes=(-3, -2, -1)),
                                 axes=(-3, -2, -1), norm="forward")

        def fftn_shifted(A):
            return jnp.fft.fftshift(jnp.fft.fftn(A, axes=(-3, -2, -1), norm="forward"),
                                    axes=(-3, -2, -1))

        vEx = ifftn_shifted(vEx_k)
        vEy = ifftn_shifted(vEy_k)

        dGdx = ifftn_shifted((1j * kx)[None, None, None, ...] * Gk)
        dGdy = ifftn_shifted((1j * ky)[None, None, None, ...] * Gk)

        # alpha_kln: (k, l, n)
        # vE:        (s, k, x,y,z)   i.e. Laguerre index k
        # dH:        (s, n, m, x,y,z) i.e. Laguerre index n
        NL = (
            jnp.einsum("kln,skxyz,snmxyz->slmxyz", alpha_kln, vEx, dGdx, optimize=True)
            + jnp.einsum("kln,skxyz,snmxyz->slmxyz", alpha_kln, vEy, dGdy, optimize=True)
        )
        NL_k = fftn_shifted(NL)

        return jnp.where(mask23[None, None, None, ...], NL_k, 0.0 + 0.0j)

    NL_k = jax.lax.cond(enable_nonlinear, _nonlinear, lambda _: jnp.zeros_like(Gk), operand=None)

    # ---- Collisions ----
    def _coll(_):
        C = collision_lenard_bernstein_conserving_multispecies(Hk, params)
        return jnp.where(mask23[None, None, None, ...], C, 0.0 + 0.0j)

    Ck = jax.lax.cond(enable_collisions, _coll, lambda _: jnp.zeros_like(Hk), operand=None)

    # Convention: dH = -NL + streaming + collisions
    dGk = -NL_k + dG_stream + Ck

    dGk = jax.lax.cond(
         enforce_reality,
         lambda x: enforce_conjugate_symmetry_fftshifted(x, params),
         lambda x: x,
        dGk,
     )

    return dGk, phi_k
