# spectraxgk/_initialization_multispecies.py
from __future__ import annotations

import os
import jax
jax.config.update("jax_enable_x64", os.environ.get("SPECTRAX_X64", "0") == "1")
import jax.numpy as jnp

from ._hl_basis import (
    kgrid_fftshifted,
    twothirds_mask,
    alpha_tensor,
    J_l_all,
    conjugate_index_fftshifted,
)

__all__ = ["initialize_simulation_parameters_multispecies"]

def initialize_simulation_parameters_multispecies(
    user_parameters=None,
    Nx=33, Ny=33, Nz=17,
    Nh=24, Nl=8,
    timesteps=200, dt=1e-2,
):
    if user_parameters is None:
        user_parameters = {}

    # Defaults
    p = dict(
        Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi,
        Nx=Nx, Ny=Ny, Nz=Nz,
        Nh=Nh, Nl=Nl,
        t_max=1.0,
        timesteps=timesteps,
        dt=dt,

        enable_streaming=True,
        enable_nonlinear=True,
        enable_collisions=True,
        enforce_reality=True,

        # initial perturbation mode (fftshift indices are Ny//2 + ny0 etc)
        nx0=1, ny0=0, nz0=0,
        pert_amp=1e-6,

        # species list must be provided for real kinetic runs
        # each: {name,q,T,n0,rho,vth,nu,Upar}
        # NOTE: do not keep Python 'species' metadata (dicts/strings) in params
        # that will be passed into jitted functions.
        species=[],
    )
    p.update(user_parameters)

    # Extract and then REMOVE from params so JAX never sees strings.
    species = list(p.pop("species", []))
    if len(species) == 0:
        raise ValueError("input_parameters must include a non-empty 'species' list for multispecies GK.")

    use_x64 = bool(jax.config.read("jax_enable_x64"))
    rdt = jnp.float64 if use_x64 else jnp.float32
    cdt = jnp.complex128 if use_x64 else jnp.complex64

    Ns = len(species)

    # Wavenumber grids (fftshifted, rad/length)
    kx, ky, kz = kgrid_fftshifted(p["Lx"], p["Ly"], p["Lz"], Nx, Ny, Nz)
    kperp2 = kx**2 + ky**2

    mask23 = twothirds_mask(Ny, Nx, Nz)
    alpha_kln = alpha_tensor(Nl)

    # Species arrays
    q_s   = jnp.array([sp.get("q", 1.0)   for sp in species], dtype=rdt)
    T_s   = jnp.array([sp.get("T", 1.0)   for sp in species], dtype=rdt)
    n0_s  = jnp.array([sp.get("n0", 1.0)  for sp in species], dtype=rdt)
    rho_s = jnp.array([sp.get("rho", 1.0) for sp in species], dtype=rdt)
    vth_s = jnp.array([sp.get("vth", 1.0) for sp in species], dtype=rdt)
    nu_s  = jnp.array([sp.get("nu", 0.0)  for sp in species], dtype=rdt)
    U_s   = jnp.array([sp.get("Upar", 0.0) for sp in species], dtype=rdt)

    # Per-species b_s(k) and Laguerre gyroaverage coefficients J_l(b_s)
    # Shapes: (Ns, Nl, Ny, Nx, Nz)
    b_s = (kperp2[None, ...] * (rho_s[:, None, None, None] ** 2))
    Jl_s = jax.vmap(lambda b: J_l_all(b, Nl))(b_s)  # vmap over species

    # For collisions: shifted J arrays in l
    Jm1_s = jnp.concatenate([jnp.zeros_like(Jl_s[:, :1]), Jl_s[:, :-1]], axis=1)
    Jp1_s = jnp.concatenate([Jl_s[:, 1:], jnp.zeros_like(Jl_s[:, :1])], axis=1)

    # Precompute Gamma0_s = sum_l J_l^2
    Gamma0_s = jnp.sum(Jl_s * Jl_s, axis=1)  # (Ns,Ny,Nx,Nz)

    # Quasineutrality denominator:
    # den = Σ_s (q_s^2 n0_s / T_s) * (1 - Gamma0_s)
    den_qn = jnp.sum((q_s*q_s * n0_s / T_s)[:, None, None, None] * (1.0 - Gamma0_s), axis=0)

    # Hermite ladder coefficients
    m = jnp.arange(Nh, dtype=rdt)
    sqrt_m_plus = jnp.sqrt(m + 1.0)
    sqrt_m_minus = jnp.sqrt(m)

    # Conjugate index maps for fftshift ordering
    conj_y = jnp.array([conjugate_index_fftshifted(i, Ny) for i in range(Ny)], dtype=jnp.int32)
    conj_x = jnp.array([conjugate_index_fftshifted(i, Nx) for i in range(Nx)], dtype=jnp.int32)
    conj_z = jnp.array([conjugate_index_fftshifted(i, Nz) for i in range(Nz)], dtype=jnp.int32)

    # Initial condition: perturb g_{l=0,m=0} at one k-mode for all species (small)
    G0 = jnp.zeros((Ns, Nl, Nh, Ny, Nx, Nz), dtype=cdt)

    ky0 = Ny//2 + int(p["ny0"])
    kx0 = Nx//2 + int(p["nx0"])
    kz0 = Nz//2 + int(p["nz0"])

    amp = jnp.array(p["pert_amp"], dtype=rdt)
    G0 = G0.at[:, 0, 0, ky0, kx0, kz0].set((amp + 0.0j).astype(G0.dtype))

    ky1 = conjugate_index_fftshifted(ky0, Ny)
    kx1 = conjugate_index_fftshifted(kx0, Nx)
    kz1 = conjugate_index_fftshifted(kz0, Nz)
    G0 = G0.at[:, 0, 0, ky1, kx1, kz1].set((amp + 0.0j).astype(G0.dtype))

    # Precompute collision scaffolding to avoid per-call overhead
    ell = jnp.arange(Nl, dtype=rdt)[:, None, None, None]

    m_index = jnp.arange(Nh, dtype=rdt)
    coeff_T = (2.0/3.0) * (
        ell[None, ...] * Jm1_s
        + 2.0 * ell[None, ...] * Jl_s
        + (ell[None, ...] + 1.0) * Jp1_s
    )  # (Ns,Nl,Ny,Nx,Nz)

    p.update(
        Ns=Ns,
        kx_grid=kx, ky_grid=ky, kz_grid=kz,
        kperp2_grid=kperp2,
        mask23=mask23,
        alpha_kln=alpha_kln,
        sqrt_m_plus=sqrt_m_plus,
        sqrt_m_minus=sqrt_m_minus,
        conj_y=conj_y, conj_x=conj_x, conj_z=conj_z,

        # NOTE: do NOT store Python species dicts (contain strings) in params passed to JAX-jitted code.
        q_s=q_s, T_s=T_s, n0_s=n0_s, rho_s=rho_s, vth_s=vth_s, nu_s=nu_s, Upar_s=U_s,

        # gyroaverage
        b_s=b_s,
        Jl_s=Jl_s, Jl_m1_s=Jm1_s, Jl_p1_s=Jp1_s,
        Gamma0_s=Gamma0_s,
        den_qn=den_qn,

        # collisions cached pieces
        ell=ell,
        coeff_T_s=coeff_T,
        m_index=m_index,

        # IC
        Gk_0=G0,
    )
    return p
