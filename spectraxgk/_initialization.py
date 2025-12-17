# spectraxgk/_initialization.py
import jax
jax.config.update("jax_enable_x64", True)  # keep for now; consider disabling for speed once stable
import jax.numpy as jnp

from ._hl_basis import (
    kgrid_fftshifted,
    twothirds_mask,
    alpha_tensor,
    J_l_all,
    conjugate_index_fftshifted,
)

__all__ = ["initialize_simulation_parameters", "load_parameters"]

try:
    import tomllib
except ModuleNotFoundError:
    import pip._vendor.tomli as tomllib


def initialize_simulation_parameters(
    user_parameters=None,
    Nx=33, Ny=33, Nz=17,
    Nh=24, Nl=8,
    timesteps=200, dt=1e-2,
):
    """
    Electrostatic slab GK / drift-kinetic core in Laguerre–Hermite moments.
    """
    if user_parameters is None:
        user_parameters = {}

    p = dict(
        # domain
        Lx=2 * jnp.pi, Ly=2 * jnp.pi, Lz=2 * jnp.pi,
        Nx=Nx, Ny=Ny, Nz=Nz,

        # LH resolution
        Nh=Nh, Nl=Nl,

        # time
        t_max=1.0,
        timesteps=timesteps,
        dt=dt,

        # normalized physics
        vti=1.0,
        rho_i=1.0,
        tau_e=1.0,
        nu=0.0,

        # model feature toggles (for physics tests)
        enable_streaming=True,
        enable_nonlinear=True,
        enable_collisions=True,
        enforce_reality=True,   # enforce conjugate symmetry each RHS eval

        # initial perturbation
        nx0=1, ny0=0, nz0=0,
        pert_amp=1e-3,
    )
    p.update(user_parameters)

    # Wavenumber grids in fftshifted ordering (rad/length)
    kx, ky, kz = kgrid_fftshifted(p["Lx"], p["Ly"], p["Lz"], Nx, Ny, Nz)
    k2 = kx**2 + ky**2 + kz**2
    kperp2 = kx**2 + ky**2
    b = kperp2 * (p["rho_i"] ** 2)

    mask23 = twothirds_mask(Ny, Nx, Nz)

    # Precompute J_l(b)
    Jl = J_l_all(b, Nl)  # (Nl, Ny, Nx, Nz)

    # Shifted J arrays for collision moments
    Jm1 = jnp.concatenate([jnp.zeros_like(Jl[:1]), Jl[:-1]], axis=0)
    Jp1 = jnp.concatenate([Jl[1:], jnp.zeros_like(Jl[:1])], axis=0)

    # Quasineutrality denominator (truncated approximation)
    sumJ2 = jnp.sum(Jl * Jl, axis=0)
    den_qn = (1.0 / p["tau_e"]) + 1.0 - sumJ2

    # Laguerre convolution tensor (cached)
    alpha_kln = alpha_tensor(Nl)  # (Nl,Nl,Nl)

    # Hermite ladder coefficients for streaming
    m = jnp.arange(Nh, dtype=jnp.float64)
    sqrt_m_plus = jnp.sqrt(m + 1.0)
    sqrt_m_minus = jnp.sqrt(m)

    # Conjugate-index arrays for enforcing reality
    conj_y = jnp.array([conjugate_index_fftshifted(i, Ny) for i in range(Ny)], dtype=jnp.int32)
    conj_x = jnp.array([conjugate_index_fftshifted(i, Nx) for i in range(Nx)], dtype=jnp.int32)
    conj_z = jnp.array([conjugate_index_fftshifted(i, Nz) for i in range(Nz)], dtype=jnp.int32)

    # Initial condition: density perturbation in (l=0,m=0) at a single k mode + conjugate
    G0 = jnp.zeros((Nl, Nh, Ny, Nx, Nz), dtype=jnp.complex128)
    ky0 = Ny // 2 + int(p["ny0"])
    kx0 = Nx // 2 + int(p["nx0"])
    kz0 = Nz // 2 + int(p["nz0"])

    amp = jnp.array(p["pert_amp"], dtype=jnp.float64)
    G0 = G0.at[0, 0, ky0, kx0, kz0].set(amp + 0.0j)

    ky1 = conjugate_index_fftshifted(ky0, Ny)
    kx1 = conjugate_index_fftshifted(kx0, Nx)
    kz1 = conjugate_index_fftshifted(kz0, Nz)
    G0 = G0.at[0, 0, ky1, kx1, kz1].set(amp + 0.0j)

    p.update(
        kx_grid=kx, ky_grid=ky, kz_grid=kz,
        k2_grid=k2,
        kperp2_grid=kperp2,
        b_grid=b,
        mask23=mask23,
        Jl_grid=Jl,
        Jl_m1=Jm1,
        Jl_p1=Jp1,
        den_qn=den_qn,
        alpha_kln=alpha_kln,
        sqrt_m_plus=sqrt_m_plus,
        sqrt_m_minus=sqrt_m_minus,
        conj_y=conj_y,
        conj_x=conj_x,
        conj_z=conj_z,
        Gk_0=G0,
    )
    return p


def load_parameters(toml_file):
    cfg = tomllib.load(open(toml_file, "rb"))
    input_parameters = cfg.get("input_parameters", {})
    solver_parameters = cfg.get("solver_parameters", {})
    solver_parameters.setdefault("adaptive_time_step", True)
    return input_parameters, solver_parameters
