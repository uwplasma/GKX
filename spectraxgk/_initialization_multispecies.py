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

def _build_perturb_mask(species: list[dict], spec) -> list[float]:
    """
    spec can be:
      - None (=> all ones)
      - list[int] indices
      - list[str] names (matches species[i]["name"])
      - dict name->weight
      - list[bool] length Ns
      - list[float] length Ns
    Returns weights (float) length Ns.
    """
    Ns = len(species)
    if spec is None:
        return [1.0] * Ns

    if isinstance(spec, dict):
        out = [0.0] * Ns
        for i, sp in enumerate(species):
            nm = sp.get("name", str(i))
            if nm in spec:
                out[i] = float(spec[nm])
        return out

    if isinstance(spec, (list, tuple)):
        if len(spec) == Ns and all(isinstance(x, (bool, int, float)) for x in spec):
            return [float(x) for x in spec]
        if all(isinstance(x, int) for x in spec):
            out = [0.0] * Ns
            for i in spec:
                out[int(i)] = 1.0
            return out
        if all(isinstance(x, str) for x in spec):
            out = [0.0] * Ns
            names = [sp.get("name", str(i)) for i, sp in enumerate(species)]
            for nm in spec:
                if nm in names:
                    out[names.index(nm)] = 1.0
            return out

    # fallback: all
    return [1.0] * Ns


def initialize_simulation_parameters_multispecies(
    user_parameters=None,
    Nx=33, Ny=33, Nz=17,
    Nh=24, Nl=8,
    timesteps=200, dt=1e-2,
):
    if user_parameters is None:
        user_parameters = {}

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

        nx0=1, ny0=0, nz0=0,
        pert_amp=1e-6,

        # NEW: optional finite-Debye term for Poisson-like behavior:
        # den(k) += (k^2 * lambda_D^2). Default 0.0 keeps original GK limit.
        lambda_D=0.0,

        # NEW: which species get the IC perturbation (see _build_perturb_mask)
        perturb_species=None,

        species=[],
    )
    p.update(user_parameters)


    # Extract and REMOVE from params so JAX never sees strings/lists-of-strings.
    perturb_species_spec = p.pop("perturb_species", None)

    species = list(p.pop("species", []))
    if len(species) == 0:
        raise ValueError("input_parameters must include a non-empty 'species' list for multispecies GK.")

    use_x64 = bool(jax.config.read("jax_enable_x64"))
    rdt = jnp.float64 if use_x64 else jnp.float32
    cdt = jnp.complex128 if use_x64 else jnp.complex64

    Ns = len(species)

    # Wavenumbers (fftshifted, rad/length)
    kx, ky, kz = kgrid_fftshifted(p["Lx"], p["Ly"], p["Lz"], Nx, Ny, Nz)
    k2 = kx**2 + ky**2 + kz**2
    kperp2 = kx**2 + ky**2

    mask23_bool = twothirds_mask(Ny, Nx, Nz)
    mask23_c = mask23_bool.astype(cdt)
    mask23_r = mask23_bool.astype(rdt)

    alpha_kln = alpha_tensor(Nl).astype(rdt)

    # Species arrays
    q_s   = jnp.array([sp.get("q", 1.0)   for sp in species], dtype=rdt)
    T_s   = jnp.array([sp.get("T", 1.0)   for sp in species], dtype=rdt)
    n0_s  = jnp.array([sp.get("n0", 1.0)  for sp in species], dtype=rdt)
    rho_s = jnp.array([sp.get("rho", 1.0) for sp in species], dtype=rdt)
    vth_s = jnp.array([sp.get("vth", 1.0) for sp in species], dtype=rdt)
    nu_s  = jnp.array([sp.get("nu", 0.0)  for sp in species], dtype=rdt)
    U_s   = jnp.array([sp.get("Upar", 0.0) for sp in species], dtype=rdt)

    # b_s(k) and J_l(b_s)
    b_s = (kperp2[None, ...] * (rho_s[:, None, None, None] ** 2))  # (Ns,Ny,Nx,Nz)
    Jl_s = jax.vmap(lambda b: J_l_all(b, Nl).astype(rdt))(b_s)      # (Ns,Nl,Ny,Nx,Nz)

    Jm1_s = jnp.concatenate([jnp.zeros_like(Jl_s[:, :1]), Jl_s[:, :-1]], axis=1)
    Jp1_s = jnp.concatenate([Jl_s[:, 1:], jnp.zeros_like(Jl_s[:, :1])], axis=1)

    Gamma0_s = jnp.sum(Jl_s * Jl_s, axis=1)  # (Ns,Ny,Nx,Nz)

    den_qn = jnp.sum((q_s*q_s * n0_s / T_s)[:, None, None, None] * (1.0 - Gamma0_s), axis=0)

    # Optional Debye term (for k_perp=0 1D runs)
    lambda_D = jnp.asarray(p.get("lambda_D", 0.0), dtype=rdt)
    den_qn = den_qn + (lambda_D * lambda_D) * k2

    m = jnp.arange(Nh, dtype=rdt)
    sqrt_m_plus = jnp.sqrt(m + 1.0)
    sqrt_m_minus = jnp.sqrt(m)

    conj_y = jnp.array([conjugate_index_fftshifted(i, Ny) for i in range(Ny)], dtype=jnp.int32)
    conj_x = jnp.array([conjugate_index_fftshifted(i, Nx) for i in range(Nx)], dtype=jnp.int32)
    conj_z = jnp.array([conjugate_index_fftshifted(i, Nz) for i in range(Nz)], dtype=jnp.int32)

    # IC
    G0 = jnp.zeros((Ns, Nl, Nh, Ny, Nx, Nz), dtype=cdt)

    ky0 = Ny//2 + int(p["ny0"])
    kx0 = Nx//2 + int(p["nx0"])
    kz0 = Nz//2 + int(p["nz0"])

    amp = jnp.array(p["pert_amp"], dtype=rdt)

    weights = _build_perturb_mask(species, perturb_species_spec)
    w_s = jnp.array(weights, dtype=rdt)  # (Ns,)

    # Apply per-species weights so you can avoid exact charge cancellation
    G0 = G0.at[:, 0, 0, ky0, kx0, kz0].set((w_s * amp).astype(rdt).astype(cdt))

    ky1 = conjugate_index_fftshifted(ky0, Ny)
    kx1 = conjugate_index_fftshifted(kx0, Nx)
    kz1 = conjugate_index_fftshifted(kz0, Nz)
    G0 = G0.at[:, 0, 0, ky1, kx1, kz1].set((w_s * amp).astype(rdt).astype(cdt))

    # Collisions cached pieces
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
        k2_grid=k2,
        kperp2_grid=kperp2,
        mask23=mask23_bool,
        mask23_r=mask23_r,
        mask23_c=mask23_c,
        alpha_kln=alpha_kln,
        sqrt_m_plus=sqrt_m_plus,
        sqrt_m_minus=sqrt_m_minus,
        conj_y=conj_y, conj_x=conj_x, conj_z=conj_z,

        q_s=q_s, T_s=T_s, n0_s=n0_s, rho_s=rho_s, vth_s=vth_s, nu_s=nu_s, Upar_s=U_s,
        lambda_D=lambda_D,

        b_s=b_s,
        Jl_s=Jl_s, Jl_m1_s=Jm1_s, Jl_p1_s=Jp1_s,
        Gamma0_s=Gamma0_s,
        den_qn=den_qn,

        ell=ell,
        coeff_T_s=coeff_T,
        m_index=m_index,

        perturb_weights_s=w_s,

        Gk_0=G0,
    )
    return p
