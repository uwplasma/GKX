# spectraxgk/_model.py
"""
Core slab electrostatic GK model in Laguerre–Hermite moments.

This file provides:
  - fft/ifft helpers for arrays stored in fftshift ordering
  - quasineutrality with adiabatic (Boltzmann) electrons
  - a conserving Lenard–Bernstein collision operator in LH space
  - the RHS: streaming + nonlinear E×B + collisions

Key stability/conservation fix vs your previous version:
  - DO NOT take `.real` after inverse FFTs in the nonlinear term.
    Dropping imaginary parts breaks the antisymmetry that gives energy conservation
    in pseudo-spectral Poisson brackets (even if the true physical fields are real).
    If you want to enforce reality, do it by enforcing conjugate symmetry in k-space,
    not by truncating `.real` in the middle of the calculation.

Performance notes:
  - Nonlinear term is O(Nl^2 * Nh * Ngrid). Keep Nl modest.
  - 2/3 de-aliasing is applied in k-space via params["mask23"].
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial

__all__ = [
    "ifft_shifted",
    "fft_shifted",
    "enforce_conjugate_symmetry_fftshifted",
    "solve_phi_from_quasineutrality_boltzmann_e",
    "collision_lenard_bernstein_conserving",
    "rhs_laguerre_hermite_gk",
    "cheap_diagnostics_from_state",
]


def ifft_shifted(Ak: jnp.ndarray) -> jnp.ndarray:
    """
    Inverse FFT of an array stored in fftshift ordering on the last 3 axes.
    Uses norm="forward" so fft_shifted/ifft_shifted are exact inverses.
    """
    return jnp.fft.ifftn(
        jnp.fft.ifftshift(Ak, axes=(-3, -2, -1)),
        axes=(-3, -2, -1),
        norm="forward",
    )


def fft_shifted(A: jnp.ndarray) -> jnp.ndarray:
    """
    FFT then fftshift on the last 3 axes.
    Uses norm="forward" so fft_shifted/ifft_shifted are exact inverses.
    """
    return jnp.fft.fftshift(
        jnp.fft.fftn(A, axes=(-3, -2, -1), norm="forward"),
        axes=(-3, -2, -1),
    )


def enforce_conjugate_symmetry_fftshifted(Gk: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Enforce conjugate symmetry in fftshifted ordering:
        G(k) = conj(G(-k))
    This guarantees real fields after iFFT (up to roundoff), without dropping `.real`.

    Requires precomputed index arrays in params:
      params["conj_y"], params["conj_x"], params["conj_z"].
    """
    iy = params["conj_y"]
    ix = params["conj_x"]
    iz = params["conj_z"]
    G_conj = jnp.conj(Gk[..., iy][:, :, :, ix][:, :, :, :, iz])
    return 0.5 * (Gk + G_conj)


def _pad_m_axis(H: jnp.ndarray) -> jnp.ndarray:
    return jnp.pad(H, ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0)))


def shift_m(H: jnp.ndarray, dm: int) -> jnp.ndarray:
    """
    Zero-padded shift along Hermite index m (axis=1).

      dm = +1  -> returns H_{m+1} (last entry is 0)
      dm = -1  -> returns H_{m-1} (first entry is 0)
      dm =  0  -> identity
    """
    if dm == 0:
        return H
    P = _pad_m_axis(H)
    _, Nh, _, _, _ = H.shape
    return P[:, 1 + dm : 1 + dm + Nh, :, :, :]


def solve_phi_from_quasineutrality_boltzmann_e(Gk: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Adiabatic-electron quasineutrality in Fourier space (slab).

    Using truncated Laguerre sums:
      num = Σ_l J_l(b) * G_{l,m=0}
      den = 1/tau_e + 1 - Σ_l J_l(b)^2
      phi = num / den, with phi(k=0)=0 gauge.

    Returns phi_k in fftshift ordering and dealiased by mask23.
    """
    Jl = params["Jl_grid"]          # (Nl,Ny,Nx,Nz)
    den = params["den_qn"]          # (Ny,Nx,Nz)
    mask23 = params["mask23"]

    Gm0 = Gk[:, 0, ...]             # (Nl,Ny,Nx,Nz)
    num = jnp.sum(Jl * Gm0, axis=0) # (Ny,Nx,Nz)

    phi = jnp.where(jnp.abs(den) > 1e-30, num / den, 0.0 + 0.0j)

    Ny, Nx, Nz = phi.shape
    phi = phi.at[Ny // 2, Nx // 2, Nz // 2].set(0.0 + 0.0j)

    return phi * mask23


def collision_lenard_bernstein_conserving(Hk: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Conserving Lenard–Bernstein collisions in Laguerre–Hermite space (electrostatic slab).

    Returns Ck with the same shape as Hk.
    """
    nu = jnp.asarray(params["nu"], dtype=jnp.float64)

    def _do(_):
        b = params["b_grid"]     # (Ny,Nx,Nz)
        Jl = params["Jl_grid"]   # (Nl,Ny,Nx,Nz)
        Jm1 = params["Jl_m1"]
        Jp1 = params["Jl_p1"]

        Nl, Nh, Ny, Nx, Nz = Hk.shape
        ell = jnp.arange(Nl, dtype=jnp.float64)[:, None, None, None]      # (Nl,1,1,1)
        m = jnp.arange(Nh, dtype=jnp.float64)[None, :, None, None, None]  # (1,Nh,1,1,1)

        # moments in k-space
        u_par = jnp.sum(Jl * Hk[:, 1, ...], axis=0)                        # (Ny,Nx,Nz)
        T_par = jnp.sqrt(2.0) * jnp.sum(Jl * Hk[:, 2, ...], axis=0)         # (Ny,Nx,Nz)
        u_perp = jnp.sqrt(b) * jnp.sum((Jl + Jm1) * Hk[:, 0, ...], axis=0)  # (Ny,Nx,Nz)
        T_perp = jnp.sum(
            (ell * Jm1 + 2.0 * ell * Jl + (ell + 1.0) * Jp1) * Hk[:, 0, ...],
            axis=0,
        )  # (Ny,Nx,Nz)

        Ttot = (T_par + 2.0 * T_perp)

        # base damping (diagonal in LH)
        base = -nu * (b[None, None, ...] + 2.0 * ell[:, None, ...] + m) * Hk
        C = base

        # conserving corrections
        C = C.at[:, 1, ...].add(nu * (Jl * u_par))
        C = C.at[:, 2, ...].add(nu * jnp.sqrt(2.0 / 3.0) * (Jl * Ttot))

        coeff_T = (2.0 / 3.0) * (ell * Jm1 + 2.0 * ell * Jl + (ell + 1.0) * Jp1)
        C0_add = nu * (jnp.sqrt(b) * (Jl + Jm1) * u_perp + coeff_T * Ttot)
        C = C.at[:, 0, ...].add(C0_add)
        return C

    return jax.lax.cond(nu == 0.0, lambda _: jnp.zeros_like(Hk), _do, operand=None)


def cheap_diagnostics_from_state(Gk: jnp.ndarray, params: dict) -> dict:
    """
    Cheap diagnostics computed from the *current* state only (no history needed).
    Designed to be called inside SaveAt(fn=...).

    Returns a small pytree of scalars/arrays.
    """
    phi_k = solve_phi_from_quasineutrality_boltzmann_e(Gk, params)
    Jl = params["Jl_grid"]

    Hk = Gk.at[:, 0, ...].add(Jl * phi_k)

    Wg = 0.5 * jnp.sum(jnp.abs(Gk) ** 2)
    Wphi = 0.5 * (1.0 + 1.0 / params["tau_e"]) * jnp.sum(jnp.abs(phi_k) ** 2)
    Wtot = Wg + Wphi

    # Collision dissipation proxy: D = -Re <H, C(H)>  (non-negative for a dissipative operator)
    Ck = collision_lenard_bernstein_conserving(Hk, params)
    Dcoll = -jnp.real(jnp.vdot(Hk, Ck))  # scalar

    # A couple of “sanity” norms
    max_abs_G = jnp.max(jnp.abs(Gk))
    max_abs_phi = jnp.max(jnp.abs(phi_k))

    return dict(
        W_g=Wg,
        W_phi=Wphi,
        W_total=Wtot,
        D_coll=Dcoll,
        max_abs_G=max_abs_G,
        max_abs_phi=max_abs_phi,
    )


@partial(jax.jit, static_argnames=("Nh", "Nl"))
def rhs_laguerre_hermite_gk(Gk: jnp.ndarray, params: dict, Nh: int, Nl: int):
    """
    RHS for electrostatic slab GK in Laguerre–Hermite moments.

    Feature toggles in params (all default True in initialization below):
      params["enable_streaming"]
      params["enable_nonlinear"]
      params["enable_collisions"]
      params["enforce_reality"]   (enforce conjugate symmetry each RHS evaluation)

    Returns:
      dGk, phi_k
    """
    kx = params["kx_grid"]
    ky = params["ky_grid"]
    kz = params["kz_grid"]
    mask23 = params["mask23"]
    alpha_kln = params["alpha_kln"]
    sqrt_m_plus = params["sqrt_m_plus"]
    sqrt_m_minus = params["sqrt_m_minus"]
    vti = params["vti"]

    enable_streaming = params.get("enable_streaming", True)
    enable_nonlinear = params.get("enable_nonlinear", True)
    enable_collisions = params.get("enable_collisions", True)
    enforce_reality = params.get("enforce_reality", True)

    # de-alias state
    Gk = Gk * mask23[None, None, ...]

    # enforce conjugate symmetry (helps long runs + prevents creeping complex physical fields)
    Gk = jax.lax.cond(
        enforce_reality,
        lambda x: enforce_conjugate_symmetry_fftshifted(x, params),
        lambda x: x,
        Gk,
    )

    # fields
    phi_k = solve_phi_from_quasineutrality_boltzmann_e(Gk, params)
    Jl = params["Jl_grid"]

    # H = G + J_l(b)*phi δ_{m0}
    Hk = Gk.at[:, 0, ...].add(Jl * phi_k)

    # --- streaming (linear) ---
    def _stream(_):
        H_p1 = shift_m(Hk, +1)
        H_m1 = shift_m(Hk, -1)
        return -1j * kz[None, None, ...] * vti * (
            (sqrt_m_plus[None, :, None, None, None] * H_p1)
            + (sqrt_m_minus[None, :, None, None, None] * H_m1)
        )

    stream = jax.lax.cond(enable_streaming, _stream, lambda _: jnp.zeros_like(Hk), operand=None)

    # --- nonlinear E×B term (pseudo-spectral Poisson bracket) ---
    def _nonlinear(_):
        phi_nk = Jl * phi_k                                  # (Nl,Ny,Nx,Nz)
        vEx_k = -1j * ky[None, ...] * phi_nk                 # (Nl,Ny,Nx,Nz)
        vEy_k =  1j * kx[None, ...] * phi_nk                 # (Nl,Ny,Nx,Nz)

        # IMPORTANT: keep complex (do NOT drop `.real`)
        vEx = ifft_shifted(vEx_k)                            # (Nl,Ny,Nx,Nz)
        vEy = ifft_shifted(vEy_k)

        dHdx = ifft_shifted((1j * kx)[None, None, ...] * Hk)  # (Nl,Nh,Ny,Nx,Nz)
        dHdy = ifft_shifted((1j * ky)[None, None, ...] * Hk)

        NL_realspace = (
            jnp.einsum("kln,nxyz,kmxyz->lmxyz", alpha_kln, vEx, dHdx, optimize=True)
            + jnp.einsum("kln,nxyz,kmxyz->lmxyz", alpha_kln, vEy, dHdy, optimize=True)
        )
        return fft_shifted(NL_realspace) * mask23[None, None, ...]

    NL_k = jax.lax.cond(enable_nonlinear, _nonlinear, lambda _: jnp.zeros_like(Hk), operand=None)

    # --- collisions ---
    def _coll(_):
        return collision_lenard_bernstein_conserving(Hk, params) * mask23[None, None, ...]

    Ck = jax.lax.cond(enable_collisions, _coll, lambda _: jnp.zeros_like(Hk), operand=None)

    dGk = -NL_k + stream + Ck

    # re-apply symmetry to derivative too (prevents symmetry drift)
    dGk = jax.lax.cond(
        enforce_reality,
        lambda x: enforce_conjugate_symmetry_fftshifted(x, params),
        lambda x: x,
        dGk,
    )

    return dGk, phi_k
