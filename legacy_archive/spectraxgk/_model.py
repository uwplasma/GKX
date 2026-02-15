# spectraxgk/_model.py
"""
Core slab electrostatic GK model in Laguerre–Hermite moments.

This file provides:
  - fft/ifft helpers for arrays stored in fftshift ordering
  - conjugate-symmetry enforcement in fftshift ordering
  - quasineutrality with adiabatic (Boltzmann) electrons
  - a conserving Lenard–Bernstein collision operator in LH space
  - the RHS: streaming + nonlinear E×B + collisions
  - cheap diagnostics for SaveAt(fn=...)

IMPORTANT:
  - Do NOT take `.real` mid-calculation in the nonlinear term.
    Reality should be enforced by conjugate symmetry in k-space.
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
    Enforce conjugate symmetry in fftshifted ordering on the LAST THREE axes:
        G(k_y,k_x,k_z) = conj(G(-k_y,-k_x,-k_z))

    Requires precomputed index arrays in params:
      params["conj_y"], params["conj_x"], params["conj_z"].

    This implementation is axis-correct and works for any leading dims,
    e.g. (Nl,Nh,Ny,Nx,Nz) or (Ny,Nx,Nz).
    """
    iy = params["conj_y"]
    ix = params["conj_x"]
    iz = params["conj_z"]

    Gneg = jnp.take(Gk, iy, axis=-3)
    Gneg = jnp.take(Gneg, ix, axis=-2)
    Gneg = jnp.take(Gneg, iz, axis=-1)

    Gconj = jnp.conj(Gneg)
    return 0.5 * (Gk + Gconj)


def _pad_m_axis(H: jnp.ndarray) -> jnp.ndarray:
    # Pad Hermite axis (axis=1) with zeros.
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
    z = jnp.zeros_like(H[:, :1, ...])
    if dm == +1:
        # H_{m+1} -> drop m=0, append 0 at end
        return jnp.concatenate([H[:, 1:, ...], z], axis=1)
    if dm == -1:
        # H_{m-1} -> prepend 0, drop last
        return jnp.concatenate([z, H[:, :-1, ...]], axis=1)
    raise ValueError("dm must be -1,0,+1")



def solve_phi_from_quasineutrality_boltzmann_e(Gk: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Adiabatic-electron quasineutrality in Fourier space (slab).

    Truncated Laguerre sums:
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
    Conserving Lenard–Bernstein collisions in Laguerre–Hermite space.

    Returns Ck with the same shape as Hk.

    Note: Dissipation/energy tests are most consistent when you measure free energy
    in terms of H (since collisions act on H).
    """

    use_x64 = bool(jax.config.read("jax_enable_x64"))
    rdt = jnp.float64 if use_x64 else jnp.float32
    cdt = jnp.complex128 if use_x64 else jnp.complex64

    nu = jnp.asarray(params["nu"], dtype=rdt)
    def _do(_):
        b = params["b_grid"]     # (Ny,Nx,Nz)
        Jl = params["Jl_grid"]   # (Nl,Ny,Nx,Nz)
        Jm1 = params["Jl_m1"]
        Jp1 = params["Jl_p1"]

        Nl, Nh, Ny, Nx, Nz = Hk.shape
        ell = jnp.arange(Nl, dtype=rdt)[:, None, None, None]      # (Nl,1,1,1)
        m = jnp.arange(Nh, dtype=rdt)[None, :, None, None, None]  # (1,Nh,1,1,1)

        # moments in k-space
        u_par = jnp.sum(Jl * Hk[:, 1, ...], axis=0)                        # (Ny,Nx,Nz)
        T_par = jnp.sqrt(2.0) * jnp.sum(Jl * Hk[:, 2, ...], axis=0)         # (Ny,Nx,Nz)
        u_perp = jnp.sqrt(b) * jnp.sum((Jl + Jm1) * Hk[:, 0, ...], axis=0)  # (Ny,Nx,Nz)
        T_perp = jnp.sum(
            (ell * Jm1 + 2.0 * ell * Jl + (ell + 1.0) * Jp1) * Hk[:, 0, ...],
            axis=0,
        )  # (Ny,Nx,Nz)

        Ttot = (T_par + 2.0 * T_perp)  # (Ny,Nx,Nz)

        # base damping (diagonal in LH)
        C = -nu * (b[None, None, ...] + 2.0 * ell[:, None, ...] + m) * Hk

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
    Cheap diagnostics computed from current state only.

    IMPORTANT: We define W_total as a "free-energy-like" quantity based on H,
    since both the nonlinear bracket and the collisions are implemented on H.

      H = G + J_l(b) * phi * δ_{m0}

    We still report W_phi separately, but we do NOT force W_total = W_g + W_phi.
    """
    phi_k = solve_phi_from_quasineutrality_boltzmann_e(Gk, params)
    Jl = params["Jl_grid"]
    Hk = Gk.at[:, 0, ...].add(Jl * phi_k)

    use_x64 = bool(jax.config.read("jax_enable_x64"))
    rdt = jnp.float64 if use_x64 else jnp.float32
    cdt = jnp.complex128 if use_x64 else jnp.complex64

    # "moment/free energy" based on H (most consistent with RHS + collisions)
    W_h = 0.5 * jnp.sum(jnp.abs(Hk) ** 2)

    # keep a field-energy-like proxy separately for plotting
    Wphi = 0.5 * (1.0 + 1.0 / params["tau_e"]) * jnp.sum(jnp.abs(phi_k) ** 2)

    # define W_total as the conserved/dissipated quantity for tests/papers
    Wtot = W_h

    enable_collisions = params.get("enable_collisions", True)
    def _dcoll(_):
        Ck = collision_lenard_bernstein_conserving(Hk, params)
        return -jnp.real(jnp.vdot(Hk, Ck))  # should be >=0 for dissipative operator
    Dcoll = jax.lax.cond(enable_collisions, _dcoll, lambda _: jnp.array(0.0, dtype=rdt), operand=None)

    max_abs_G = jnp.max(jnp.abs(Gk))
    max_abs_phi = jnp.max(jnp.abs(phi_k))

    return dict(
        W_g=W_h,          # keep key for backwards compat with plots/tests
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

    Feature toggles in params:
      enable_streaming, enable_nonlinear, enable_collisions, enforce_reality
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

    # enforce conjugate symmetry (optional)
    Gk = jax.lax.cond(
        enforce_reality,
        lambda x: enforce_conjugate_symmetry_fftshifted(x, params),
        lambda x: x,
        Gk,
    )

    # fields
    phi_k = solve_phi_from_quasineutrality_boltzmann_e(Gk, params)
    Jl = params["Jl_grid"]
    Hk = Gk.at[:, 0, ...].add(Jl * phi_k)

    # streaming
    def _stream(_):
        Hp1 = shift_m(Hk, +1)
        Hm1 = shift_m(Hk, -1)
        return -1j * kz[None, None, ...] * vti * (
            (sqrt_m_plus[None, :, None, None, None] * Hp1)
            + (sqrt_m_minus[None, :, None, None, None] * Hm1)
        )

    stream = jax.lax.cond(enable_streaming, _stream, lambda _: jnp.zeros_like(Hk), operand=None)

    # nonlinear E×B
    def _nonlinear(_):
        phi_nk = Jl * phi_k
        vEx_k = -1j * ky[None, ...] * phi_nk
        vEy_k =  1j * kx[None, ...] * phi_nk

        vEx = ifft_shifted(vEx_k)
        vEy = ifft_shifted(vEy_k)

        dHdx = ifft_shifted((1j * kx)[None, None, ...] * Hk)
        dHdy = ifft_shifted((1j * ky)[None, None, ...] * Hk)

        NL_rs = (
            jnp.einsum("kln,nxyz,kmxyz->lmxyz", alpha_kln, vEx, dHdx, optimize=True)
            + jnp.einsum("kln,nxyz,kmxyz->lmxyz", alpha_kln, vEy, dHdy, optimize=True)
        )
        return fft_shifted(NL_rs) * mask23[None, None, ...]

    NL_k = jax.lax.cond(enable_nonlinear, _nonlinear, lambda _: jnp.zeros_like(Hk), operand=None)

    # collisions
    def _coll(_):
        return collision_lenard_bernstein_conserving(Hk, params) * mask23[None, None, ...]
    Ck = jax.lax.cond(enable_collisions, _coll, lambda _: jnp.zeros_like(Hk), operand=None)

    dGk = -NL_k + stream + Ck

    # keep symmetry of derivative too (prevents drift)
    dGk = jax.lax.cond(
        enforce_reality,
        lambda x: enforce_conjugate_symmetry_fftshifted(x, params),
        lambda x: x,
        dGk,
    )

    return dGk, phi_k
