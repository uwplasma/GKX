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

def enforce_conjugate_symmetry_fftshifted(Ak: jnp.ndarray, params: dict) -> jnp.ndarray:
    iy, ix, iz = params["conj_y"], params["conj_x"], params["conj_z"]
    Aneg = jnp.take(Ak, iy, axis=-3)
    Aneg = jnp.take(Aneg, ix, axis=-2)
    Aneg = jnp.take(Aneg, iz, axis=-1)
    return 0.5 * (Ak + jnp.conj(Aneg))

def build_Hk_from_Gk_phi(Gk: jnp.ndarray, phi_k: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    H_s = G_s + (q_s/T_s) * J_l(b_s) * phi * δ_{m0}
    Implemented without scatter/pad: concatenate m=0 slice + remaining slices.
    """
    rdt = Gk.real.dtype
    q_s = jnp.asarray(params["q_s"], dtype=rdt)
    T_s = jnp.asarray(params["T_s"], dtype=rdt)
    a_s = (q_s / T_s)[:, None, None, None, None]           # (Ns,1,1,1,1)
    Jl_s = jnp.asarray(params["Jl_s"], dtype=rdt)           # (Ns,Nl,Ny,Nx,Nz)

    add_m0 = a_s * (Jl_s * phi_k[None, None, ...])          # (Ns,Nl,Ny,Nx,Nz)
    m0 = Gk[:, :, 0, ...] + add_m0.astype(Gk.dtype)
    rest = Gk[:, :, 1:, ...]
    return jnp.concatenate([m0[:, :, None, ...], rest], axis=2)

def shift_m(H: jnp.ndarray, dm: int) -> jnp.ndarray:
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
    den(k) * phi_k = sum_s q_s n0_s * sum_l J_l(b_s) * g_{s,l,m=0}
    den includes optional Debye: den += k^2 * lambda_D^2 (set in init; default 0).
    """
    q_s, n0_s = params["q_s"], params["n0_s"]
    Jl_s = params["Jl_s"]          # (Ns,Nl,Ny,Nx,Nz)
    den = params["den_qn"]         # (Ny,Nx,Nz)
    mask23_c = params.get("mask23_c", params["mask23"].astype(Gk.dtype))

    g_m0 = Gk[:, :, 0, ...]                           # (Ns,Nl,Ny,Nx,Nz)
    num_s = jnp.sum(Jl_s * g_m0, axis=1)              # (Ns,Ny,Nx,Nz)
    num = jnp.sum((q_s * n0_s)[:, None, None, None] * num_s, axis=0)

    phi = jnp.where(jnp.abs(den) > 1e-30, num / den, 0.0 + 0.0j)

    Ny, Nx, Nz = phi.shape
    phi = phi.at[Ny//2, Nx//2, Nz//2].set(0.0 + 0.0j)  # gauge

    return phi * mask23_c

def collision_lenard_bernstein_conserving_multispecies(Hk: jnp.ndarray, params: dict) -> jnp.ndarray:
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
        ell_loc = jnp.asarray(ell, dtype=rdt)[:, None, ...]         # (Nl,1,Ny,Nx,Nz)
        m_loc = jnp.asarray(m_index, dtype=rdt)[None, :, None, None, None]

        def _do(_):
            u_par  = jnp.sum(Jls * Hs[:, 1, ...], axis=0)
            T_par  = jnp.sqrt(jnp.asarray(2.0, dtype=rdt)) * jnp.sum(Jls * Hs[:, 2, ...], axis=0)
            u_perp = jnp.sqrt(bs) * jnp.sum((Jls + Jm1s) * Hs[:, 0, ...], axis=0)

            # (ell*Jm1 + 2ell*Jl + (ell+1)Jp1) = (3/2) * coeffTs
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
    Diagnostics designed for papers/debugging:
      - W_total = 0.5 ||H||^2
      - W_s per species
      - W_phi proxy = 0.5 * sum Re(den_qn) * |phi|^2
      - D_coll total + per species (>=0) when collisions enabled
      - phi_rms, max|phi|, max|G|
      - Hermite spectrum E_m(s,m) = 0.5 sum_{l,k} |H|^2
    """
    phi_k = solve_phi_quasineutrality_multispecies(Gk, params)
    Hk = build_Hk_from_Gk_phi(Gk, phi_k, params)

    # "g-energy"
    absG2 = jnp.abs(Gk) ** 2
    W_g_s = 0.5 * jnp.sum(absG2, axis=(1,2,3,4,5))
    W_g = jnp.sum(W_g_s)

    # "h-energy" (useful diagnostic, but not the linear invariant)
    absH2 = jnp.abs(Hk) ** 2
    W_h_s = 0.5 * jnp.sum(absH2, axis=(1,2,3,4,5))
    W_h = jnp.sum(W_h_s)

    # field/polarization-like term (den already includes optional Debye k^2 lambda_D^2)
    den = jnp.asarray(params["den_qn"], dtype=Gk.real.dtype)
    W_phi = 0.5 * jnp.sum(jnp.real(den) * (jnp.abs(phi_k) ** 2))

    # Free energy to use for "streaming conserves energy" checks
    W_free = W_g + W_phi

    phi_rms = jnp.sqrt(jnp.mean(jnp.abs(phi_k) ** 2))
    max_abs_phi = jnp.max(jnp.abs(phi_k))
    max_abs_G = jnp.max(jnp.abs(Gk))

    # Hermite spectrum from H (often what you want to visualize recurrence)
    E_m = 0.5 * jnp.sum(absH2, axis=(1,3,4,5))  # (Ns,Nh)

    return dict(
       W_total=W_free,     # keep key for existing code/plots
        W_free=W_free,
        W_g=W_g, W_g_s=W_g_s,
        W_h=W_h, W_h_s=W_h_s,
        W_phi=W_phi,
        phi_rms=phi_rms,
        max_abs_phi=max_abs_phi,
        max_abs_G=max_abs_G,
        E_m=E_m,
    )

@partial(jax.jit, static_argnames=("Nh","Nl"))
def rhs_gk_multispecies(Gk: jnp.ndarray, params: dict, Nh: int, Nl: int):
    kx, ky, kz = params["kx_grid"], params["ky_grid"], params["kz_grid"]
    mask23_c = params.get("mask23_c", params["mask23"].astype(Gk.dtype))
    alpha_kln = params["alpha_kln"]
    sqrt_mp, sqrt_mm = params["sqrt_m_plus"], params["sqrt_m_minus"]

    vth_s, U_s = params["vth_s"], params["Upar_s"]

    enable_streaming = params.get("enable_streaming", True)
    enable_nonlinear = params.get("enable_nonlinear", True)
    enable_collisions = params.get("enable_collisions", True)
    enforce_reality = params.get("enforce_reality", True)

    # dealias
    Gk = Gk * mask23_c[None, None, None, ...]

    # optional symmetry
    Gk = jax.lax.cond(enforce_reality,
                      lambda x: enforce_conjugate_symmetry_fftshifted(x, params),
                      lambda x: x,
                      Gk)

    # fields
    phi_k = solve_phi_quasineutrality_multispecies(Gk, params)
    Hk = build_Hk_from_Gk_phi(Gk, phi_k, params)

    # streaming (Hermite ladder + mean drift)
    def _stream(_):
        Hp1 = shift_m(Hk, +1)
        Hm1 = shift_m(Hk, -1)
        kz_term = (-1j * kz)[None, None, None, ...]  # (1,1,1,Ny,Nx,Nz)

        vth = vth_s[:, None, None, None, None, None]
        ladder = (
            (sqrt_mp[None, None, :, None, None, None] * Hp1)
            + (sqrt_mm[None, None, :, None, None, None] * Hm1)
        )
        stream_v = kz_term * vth * ladder

        U = U_s[:, None, None, None, None, None]
        stream_U = kz_term * U * Hk
        return stream_v + stream_U

    stream = jax.lax.cond(enable_streaming, _stream, lambda _: jnp.zeros_like(Hk), operand=None)

    # nonlinear ExB (kept for 3D; will be zero when kx=ky=0 anyway)
    def _nonlinear(_):
        Jl_s = params["Jl_s"]
        phi_sk = Jl_s * phi_k[None, None, ...]  # (Ns,Nl,Ny,Nx,Nz)
        vEx_k = -1j * ky[None, None, ...] * phi_sk
        vEy_k =  1j * kx[None, None, ...] * phi_sk

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

        NL = (
            jnp.einsum("kln,snxyz,skmxyz->slmxyz", alpha_kln, vEx, dHdx, optimize=True)
            + jnp.einsum("kln,snxyz,skmxyz->slmxyz", alpha_kln, vEy, dHdy, optimize=True)
        )
        return fftn_shifted(NL) * mask23_c[None, None, None, ...]

    NL_k = jax.lax.cond(enable_nonlinear, _nonlinear, lambda _: jnp.zeros_like(Hk), operand=None)

    def _coll(_):
        C = collision_lenard_bernstein_conserving_multispecies(Hk, params)
        return C * mask23_c[None, None, None, ...]
    Ck = jax.lax.cond(enable_collisions, _coll, lambda _: jnp.zeros_like(Hk), operand=None)

    dGk = -NL_k + stream + Ck

    dGk = jax.lax.cond(enforce_reality,
                       lambda x: enforce_conjugate_symmetry_fftshifted(x, params),
                       lambda x: x,
                       dGk)

    return dGk, phi_k
