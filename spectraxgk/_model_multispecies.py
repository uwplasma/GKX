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
]

def build_Hk_from_Gk_phi(Gk: jnp.ndarray, phi_k: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    H_s = G_s + (q_s/T_s) * J_l(b_s) * phi * δ_{m0}, done without padding.
    Gk shape: (Ns,Nl,Nh,Ny,Nx,Nz), phi_k shape: (Ny,Nx,Nz)
    """
    rdt = Gk.real.dtype
    q_s = jnp.asarray(params["q_s"], dtype=rdt)
    T_s = jnp.asarray(params["T_s"], dtype=rdt)
    a_s = (q_s / T_s)[:, None, None, None, None]                 # (Ns,1,1,1,1)
    Jl_s = jnp.asarray(params["Jl_s"], dtype=rdt)                 # (Ns,Nl,Ny,Nx,Nz)
    add_m0 = a_s * (Jl_s * phi_k[None, None, ...])                # (Ns,Nl,Ny,Nx,Nz)
    return Gk.at[:, :, 0, ...].add(add_m0.astype(Gk.dtype))

def enforce_conjugate_symmetry_fftshifted(Ak: jnp.ndarray, params: dict) -> jnp.ndarray:
    iy, ix, iz = params["conj_y"], params["conj_x"], params["conj_z"]
    Aneg = jnp.take(Ak, iy, axis=-3)
    Aneg = jnp.take(Aneg, ix, axis=-2)
    Aneg = jnp.take(Aneg, iz, axis=-1)
    return 0.5 * (Ak + jnp.conj(Aneg))

def shift_m(H: jnp.ndarray, dm: int) -> jnp.ndarray:
    # H shape: (..., Nh, Ny, Nx, Nz) with Hermite axis = -4 if leading dims exist
    # Here we always store (Ns,Nl,Nh,Ny,Nx,Nz) so Hermite axis = 2.
    if dm == 0:
        return H
    z = jnp.zeros_like(H[:, :, :1, ...])
    if dm == +1:
        return jnp.concatenate([H[:, :, 1:, ...], z], axis=2)
    if dm == -1:
        return jnp.concatenate([z, H[:, :, :-1, ...]], axis=2)
    raise ValueError("dm must be -1,0,+1")

def solve_phi_quasineutrality_multispecies(Gk: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Multi-species electrostatic quasineutrality (kinetic ions + kinetic electrons):
      den(k) * phi_k = sum_s q_s n0_s * sum_l J_l(b_s) * g_{s,l,m=0}

    where den(k) = sum_s (q_s^2 n0_s / T_s) * (1 - Gamma0_s).
    """
    q_s, n0_s = params["q_s"], params["n0_s"]
    Jl_s = params["Jl_s"]        # (Ns,Nl,Ny,Nx,Nz)
    den = params["den_qn"]       # (Ny,Nx,Nz)
    mask23 = params["mask23"]

    # num = Σ_s q_s n0_s Σ_l J_l g_{m=0}
    g_m0 = Gk[:, :, 0, ...]  # (Ns,Nl,Ny,Nx,Nz)
    num_s = jnp.sum(Jl_s * g_m0, axis=1)                          # (Ns,Ny,Nx,Nz)
    num = jnp.sum((q_s * n0_s)[:, None, None, None] * num_s, axis=0)

    phi = jnp.where(jnp.abs(den) > 1e-30, num / den, 0.0 + 0.0j)

    Ny, Nx, Nz = phi.shape
    phi = phi.at[Ny//2, Nx//2, Nz//2].set(0.0 + 0.0j)  # gauge

    return phi * mask23

def collision_lenard_bernstein_conserving_multispecies(Hk: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Apply conserving Lenard–Bernstein operator species-by-species.
    Hk shape: (Ns,Nl,Nh,Ny,Nx,Nz)
    """
    nu_s = params["nu_s"]                # (Ns,)
    b_s = params["b_s"]                  # (Ns,Ny,Nx,Nz)
    Jl_s = params["Jl_s"]                # (Ns,Nl,Ny,Nx,Nz)
    Jm1_s = params["Jl_m1_s"]
    coeff_T_s = params["coeff_T_s"]      # (Ns,Nl,Ny,Nx,Nz)
    ell = params["ell"]                  # (Nl,1,1,1)
    m_index = params["m_index"]   # (Nh,) real dtype chosen at init

    def one_species(Hs, nus, bs, Jls, Jm1s, coeffTs):
        # Hs: (Nl,Nh,Ny,Nx,Nz)
        # Keep everything in the same precision as Hs to avoid cond branch dtype mismatch.
        rdt = Hs.real.dtype
        nus = jnp.asarray(nus, dtype=rdt)
        bs = jnp.asarray(bs, dtype=rdt)
        Jls = jnp.asarray(Jls, dtype=rdt)
        Jm1s = jnp.asarray(Jm1s, dtype=rdt)
        coeffTs = jnp.asarray(coeffTs, dtype=rdt)
        ell_loc = jnp.asarray(ell, dtype=rdt)                  # (Nl,1,1,1)
        m_loc = jnp.asarray(m_index, dtype=rdt)[None, :, None, None, None]     # (1,Nh,1,1,1)

        def _do(_):
            # moments (k-space)
            u_par  = jnp.sum(Jls * Hs[:, 1, ...], axis=0)
            T_par  = jnp.sqrt(2.0) * jnp.sum(Jls * Hs[:, 2, ...], axis=0)
            u_perp = jnp.sqrt(bs) * jnp.sum((Jls + Jm1s) * Hs[:, 0, ...], axis=0)
            Top = (jnp.asarray(1.5, dtype=rdt) * coeffTs)  # reconstruct (ell*Jm1+2ell*Jl+(ell+1)Jp1)
            T_perp = jnp.sum(Top * Hs[:, 0, ...], axis=0)
            Ttot = (T_par + 2.0 * T_perp)

            # base damping
            C = -nus * (bs[None, None, ...] + 2.0 * ell_loc[:, None, ...] + m_loc) * Hs

            # conserving corrections
            C = C.at[:, 1, ...].add(nus * (Jls * u_par))
            C = C.at[:, 2, ...].add(nus * jnp.sqrt(2.0/3.0) * (Jls * Ttot))
            C0_add = nus * (jnp.sqrt(bs) * (Jls + Jm1s) * u_perp + coeffTs * Ttot)
            C = C.at[:, 0, ...].add(C0_add)
            return C.astype(Hs.dtype)

        return jax.lax.cond(nus == 0.0,
                            lambda _: jnp.zeros_like(Hs),
                            _do,
                            operand=None)

    return jax.vmap(one_species)(Hk, nu_s, b_s, Jl_s, Jm1_s, coeff_T_s)

def cheap_diagnostics_multispecies(Gk: jnp.ndarray, params: dict) -> dict:
    phi_k = solve_phi_quasineutrality_multispecies(Gk, params)
    Hk = build_Hk_from_Gk_phi(Gk, phi_k, params)

    Wtot = 0.5 * jnp.sum(jnp.abs(Hk)**2)
    phi_rms = jnp.sqrt(jnp.mean(jnp.abs(phi_k)**2))

    return dict(W_total=Wtot, phi_rms=phi_rms)

@partial(jax.jit, static_argnames=("Nh","Nl"))
def rhs_gk_multispecies(Gk: jnp.ndarray, params: dict, Nh: int, Nl: int):
    """
    Multispecies electrostatic GK in slab, Laguerre–Hermite velocity basis.
    State: Gk (Ns,Nl,Nh,Ny,Nx,Nz) represents g_s.
    """
    kx, ky, kz = params["kx_grid"], params["ky_grid"], params["kz_grid"]
    mask23 = params["mask23"]
    alpha_kln = params["alpha_kln"]
    sqrt_mp, sqrt_mm = params["sqrt_m_plus"], params["sqrt_m_minus"]

    q_s, T_s, vth_s, U_s = params["q_s"], params["T_s"], params["vth_s"], params["Upar_s"]
    Jl_s = params["Jl_s"]

    enable_streaming = params.get("enable_streaming", True)
    enable_nonlinear = params.get("enable_nonlinear", True)
    enable_collisions = params.get("enable_collisions", True)
    enforce_reality = params.get("enforce_reality", True)

    # dealias
    Gk = Gk * mask23[None, None, None, ...]

    # optional symmetry
    Gk = jax.lax.cond(enforce_reality,
                      lambda x: enforce_conjugate_symmetry_fftshifted(x, params),
                      lambda x: x,
                      Gk)

    # fields (kinetic QN)
    phi_k = solve_phi_quasineutrality_multispecies(Gk, params)

    # h_s = g_s + (q_s/T_s) J0 phi δ_{m0}
    Hk = build_Hk_from_Gk_phi(Gk, phi_k, params)

    # streaming: includes drift U_s as diagonal phase advection in z
    def _stream(_):
        Hp1 = shift_m(Hk, +1)
        Hm1 = shift_m(Hk, -1)
        kz_term = (-1j * kz)[None, None, None, ...]  # broadcast over Ns,Nl,Nh

        # v_parallel part via Hermite ladder
        vth = vth_s[:, None, None, None, None, None]  # (Ns,1,1,1,1,1)
        ladder = (
            (sqrt_mp[None, None, :, None, None, None] * Hp1)
            + (sqrt_mm[None, None, :, None, None, None] * Hm1)
        )
        stream_v = kz_term * vth * ladder

        # mean drift U_s: -(ikz) U_s Hk
        U = U_s[:, None, None, None, None, None]
        stream_U = kz_term * U * Hk
        return stream_v + stream_U

    stream = jax.lax.cond(enable_streaming, _stream, lambda _: jnp.zeros_like(Hk), operand=None)

    # Nonlinear ExB (optional; expensive and irrelevant in 1D)
    def _nonlinear(_):
        # Species-dependent gyroaveraged potential for ExB: <phi>_s = J0_s phi
        phi_sk = Jl_s * phi_k[None, None, ...]          # (Ns,Nl,Ny,Nx,Nz)
        vEx_k = -1j * ky[None, None, ...] * phi_sk
        vEy_k =  1j * kx[None, None, ...] * phi_sk

        # If you want this term fast, don’t run it in 1D; keep for 3D turbulence only.
        def ifftn_shifted(Ak):
            return jnp.fft.ifftn(jnp.fft.ifftshift(Ak, axes=(-3,-2,-1)),
                                 axes=(-3,-2,-1), norm="forward")

        def fftn_shifted(A):
            return jnp.fft.fftshift(jnp.fft.fftn(A, axes=(-3,-2,-1), norm="forward"),
                                    axes=(-3,-2,-1))

        vEx = ifftn_shifted(vEx_k)
        vEy = ifftn_shifted(vEy_k)

        dHdx = ifftn_shifted((1j*kx)[None, None, None, ...] * Hk)
        dHdy = ifftn_shifted((1j*ky)[None, None, None, ...] * Hk)

        # Laguerre convolution: alpha_kln
        NL = (
            jnp.einsum("kln,snxyz,skmxyz->slmxyz", alpha_kln, vEx, dHdx, optimize=True)
            + jnp.einsum("kln,snxyz,skmxyz->slmxyz", alpha_kln, vEy, dHdy, optimize=True)
        )
        return fftn_shifted(NL) * mask23[None, None, None, ...]

    NL_k = jax.lax.cond(enable_nonlinear, _nonlinear, lambda _: jnp.zeros_like(Hk), operand=None)

    # collisions
    def _coll(_):
        C = collision_lenard_bernstein_conserving_multispecies(Hk, params)
        return C * mask23[None, None, None, ...]
    Ck = jax.lax.cond(enable_collisions, _coll, lambda _: jnp.zeros_like(Hk), operand=None)

    dGk = -NL_k + stream + Ck

    dGk = jax.lax.cond(enforce_reality,
                       lambda x: enforce_conjugate_symmetry_fftshifted(x, params),
                       lambda x: x,
                       dGk)

    return dGk, phi_k
