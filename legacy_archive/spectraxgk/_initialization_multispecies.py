# spectraxgk/_initialization_multispecies.py
from __future__ import annotations

import os
import jax
jax.config.update("jax_enable_x64", os.environ.get("SPECTRAX_X64", "0") == "1")
import jax.numpy as jnp
from jax.scipy.special import i0e

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
    Convert user-facing 'perturb_species' spec into per-species weights.

    spec can be:
      - None (=> all ones)
      - list[int] indices
      - list[str] names (matches species[i]["name"])
      - dict name->weight
      - list[bool] length Ns
      - list[float] length Ns

    Returns: list[float] length Ns.
    NOTE: Strings/lists-of-strings are NEVER stored into the returned params dict
          (keeps JIT safe).
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

    return [1.0] * Ns


def _make_gradlnB_profile(Lz: float, Nz: int, *, B_eps: float, B_mode: int) -> jnp.ndarray:
    """
    Real-space profile for grad_parallel ln B in a simple 1D slab model:
      B(z) = 1 + B_eps * cos(B_mode * 2π z/Lz)
      grad ln B = d/dz ln B = B'(z)/B(z)

    Returned array has shape (Nz,) on the uniform real-space grid z_j = j*Lz/Nz.
    """
    z = jnp.linspace(0.0, Lz, Nz, endpoint=False)
    k = (2.0 * jnp.pi / Lz) * float(B_mode)
    B = 1.0 + float(B_eps) * jnp.cos(k * z)
    dBdz = -float(B_eps) * k * jnp.sin(k * z)
    return dBdz / B


def initialize_simulation_parameters_multispecies(
    user_parameters=None,
    Nx=33, Ny=33, Nz=17,
    Nh=24, Nl=8,
    timesteps=200, dt=1e-2,
):
    """
    Build a JAX-friendly params dict for multispecies electrostatic GK in a
    Fourier–Laguerre–Hermite representation.

    Key design points:
      - No strings or lists-of-strings are stored in the returned dict.
      - Arrays are JAX arrays; small metadata (bool/ints) are Python scalars.
      - Optional grad_parallel ln B profile is stored in real space (Nz,) and
        applied via z-only FFTs in the model.
    """
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
        enable_gradB_parallel=False,   # NEW: include ∇∥ ln B couplings
        enable_nonlinear=True,
        enable_collisions=True,
        enforce_reality=True,

        nx0=1, ny0=0, nz0=0,
        pert_amp=1e-6,

        # Optional finite-Debye term: den += k^2 * lambda_D^2 (helps k_perp=0)
        lambda_D=0.0,

        # Which species get IC perturbation (user-facing; removed before returning)
        perturb_species=None,

        # Simple B(z) model controls (only used if enable_gradB_parallel)
        B_eps=0.0,     # amplitude in B(z)=1+B_eps*cos(...)
        B_mode=1,      # mode number

        species=[],
    )
    p.update(user_parameters)

    # Strip user-facing non-JAX things immediately.
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
    mask23_r = mask23_bool.astype(rdt)

    # Laguerre triple product tensor (for nonlinear term)
    alpha_kln = alpha_tensor(Nl).astype(rdt)

    # Species arrays
    q_s   = jnp.array([sp.get("q", 1.0)    for sp in species], dtype=rdt)
    T_s   = jnp.array([sp.get("T", 1.0)    for sp in species], dtype=rdt)
    n0_s  = jnp.array([sp.get("n0", 1.0)   for sp in species], dtype=rdt)
    rho_s = jnp.array([sp.get("rho", 1.0)  for sp in species], dtype=rdt)
    vth_s = jnp.array([sp.get("vth", 1.0)  for sp in species], dtype=rdt)
    nu_s  = jnp.array([sp.get("nu", 0.0)   for sp in species], dtype=rdt)
    U_s   = jnp.array([sp.get("Upar", 0.0) for sp in species], dtype=rdt)

    # b_s(k) and J_l(b_s)
    b_s = (kperp2[None, ...] * (rho_s[:, None, None, None] ** 2))            # (Ns,Ny,Nx,Nz)
    Jl_s = jax.vmap(lambda b: J_l_all(b, Nl).astype(rdt))(b_s)               # (Ns,Nl,Ny,Nx,Nz)
    Jm1_s = jnp.concatenate([jnp.zeros_like(Jl_s[:, :1]), Jl_s[:, :-1]], axis=1)
    Jp1_s = jnp.concatenate([Jl_s[:, 1:], jnp.zeros_like(Jl_s[:, :1])], axis=1)
    # Exact polarization function Γ0(b) = e^{-b} I0(b).  Using i0e(b)=e^{-b}I0(b) is stable.
    # This is required for the free-energy cancellation in the collisionless linear system.
    Gamma0_s = i0e(b_s).astype(rdt)                                          # (Ns,Ny,Nx,Nz)

    # Quasineutrality denominator (multi-species generalization of the usual form):
    den_qn = jnp.sum((q_s*q_s * n0_s / T_s)[:, None, None, None] * (1.0 - Gamma0_s), axis=0)

    lambda_D = jnp.asarray(p.get("lambda_D", 0.0), dtype=rdt)
    den_qn = den_qn + (lambda_D * lambda_D) * k2

    # Useful scalar (appears in the closed-form inversion when solving phi from H)
    sum_q2n0_over_T = jnp.sum((q_s * q_s) * n0_s / T_s).astype(rdt)
    den_h = sum_q2n0_over_T + (lambda_D * lambda_D) * k2   # (Ny,Nx,Nz)

    # Hermite ladder coefficients
    m = jnp.arange(Nh, dtype=rdt)
    sqrt_m_plus = jnp.sqrt(m + 1.0)
    sqrt_m_minus = jnp.sqrt(m)

    # Conjugate indices for fftshifted symmetry
    conj_y = jnp.array([conjugate_index_fftshifted(i, Ny) for i in range(Ny)], dtype=jnp.int32)
    conj_x = jnp.array([conjugate_index_fftshifted(i, Nx) for i in range(Nx)], dtype=jnp.int32)
    conj_z = jnp.array([conjugate_index_fftshifted(i, Nz) for i in range(Nz)], dtype=jnp.int32)

    # Initial condition for G (user-facing), then convert to H for evolution.
    G0 = jnp.zeros((Ns, Nl, Nh, Ny, Nx, Nz), dtype=cdt)

    ky0 = Ny//2 + int(p["ny0"])
    kx0 = Nx//2 + int(p["nx0"])
    kz0 = Nz//2 + int(p["nz0"])

    amp = jnp.array(p["pert_amp"], dtype=rdt)
    weights = _build_perturb_mask(species, perturb_species_spec)
    w_s = jnp.array(weights, dtype=rdt)  # (Ns,)

    G0 = G0.at[:, 0, 0, ky0, kx0, kz0].set((w_s * amp).astype(cdt))

    # Ensure conjugate mode is also set (keeps real-space fields real when requested)
    ky1 = conjugate_index_fftshifted(ky0, Ny)
    kx1 = conjugate_index_fftshifted(kx0, Nx)
    kz1 = conjugate_index_fftshifted(kz0, Nz)
    G0 = G0.at[:, 0, 0, ky1, kx1, kz1].set((w_s * amp).astype(cdt))

    # --- Build phi0 from G0 (quasineutrality) and convert to H0 = G0 + (q/T) J_l phi δ_{m0} ---
    g_m0 = G0[:, :, 0, ...]                               # (Ns,Nl,Ny,Nx,Nz)
    num_s = jnp.sum(Jl_s * g_m0, axis=1)                  # (Ns,Ny,Nx,Nz)
    num = jnp.sum((q_s * n0_s)[:, None, None, None] * num_s, axis=0)
    phi0 = jnp.where(jnp.abs(den_qn) > 1e-30, num / den_qn, 0.0 + 0.0j).astype(cdt)
    phi0 = phi0.at[Ny//2, Nx//2, Nz//2].set(0.0 + 0.0j)
    phi0 = jnp.where(mask23_bool, phi0, 0.0 + 0.0j)

    a_s = (q_s / T_s)[:, None, None, None, None]          # (Ns,1,1,1,1)
    add_m0 = a_s * (Jl_s * phi0[None, None, ...])         # (Ns,Nl,Ny,Nx,Nz)
    H0_m0 = (g_m0 + add_m0).astype(cdt)
    H0 = G0.at[:, :, 0, ...].set(H0_m0)

    # Collisions cached pieces
    ell = jnp.arange(Nl, dtype=rdt)[:, None, None, None]   # (Nl,1,1,1)
    ell_vec = jnp.arange(Nl, dtype=rdt)                    # (Nl,)
    m_index = jnp.arange(Nh, dtype=rdt)

    coeff_T = (2.0/3.0) * (
        ell[None, ...] * Jm1_s
        + 2.0 * ell[None, ...] * Jl_s
        + (ell[None, ...] + 1.0) * Jp1_s
    )  # (Ns,Nl,Ny,Nx,Nz)

    # Optional grad_parallel ln B profile in real space (Nz,)
    enable_gradB_parallel = bool(p.get("enable_gradB_parallel", False))
    B_eps = float(p.get("B_eps", 0.0))
    B_mode = int(p.get("B_mode", 1))
    gradlnB_z = _make_gradlnB_profile(float(p["Lz"]), Nz, B_eps=B_eps, B_mode=B_mode).astype(rdt)

    p.update(
        Ns=Ns,
        kx_grid=kx.astype(rdt),
        ky_grid=ky.astype(rdt),
        kz_grid=kz.astype(rdt),
        k2_grid=k2.astype(rdt),
        kperp2_grid=kperp2.astype(rdt),

        mask23=mask23_bool,
        mask23_r=mask23_r,

        alpha_kln=alpha_kln,
        sqrt_m_plus=sqrt_m_plus,
        sqrt_m_minus=sqrt_m_minus,
        conj_y=conj_y, conj_x=conj_x, conj_z=conj_z,

        q_s=q_s, T_s=T_s, n0_s=n0_s, rho_s=rho_s, vth_s=vth_s, nu_s=nu_s, Upar_s=U_s,
        lambda_D=lambda_D,

        b_s=b_s,
        Jl_s=Jl_s, Jl_m1_s=Jm1_s, Jl_p1_s=Jp1_s,
        Gamma0_s=Gamma0_s,
        den_qn=den_qn.astype(rdt),
        den_h=den_h.astype(rdt),
        sum_q2n0_over_T=sum_q2n0_over_T,

        ell=ell,
        ell_vec=ell_vec,
        coeff_T_s=coeff_T,
        m_index=m_index,

        perturb_weights_s=w_s,

        enable_gradB_parallel=enable_gradB_parallel,
        B_eps=jnp.asarray(B_eps, dtype=rdt),
        B_mode=jnp.asarray(B_mode, dtype=jnp.int32),
        gradlnB_z=gradlnB_z,

        Gk_0=G0,          # optional: keep for inspection
        Hk_0=H0,          # <-- evolve this
    )
    return p
