"""Linear electrostatic gyrokinetic building blocks (Hermite-Laguerre)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from spectraxgk.basis import hermite_ladder_coeffs
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.gyroaverage import J_l_all
from spectraxgk.grids import SpectralGrid


@dataclass(frozen=True)
class LinearParams:
    """Parameters for the linear electrostatic operator."""

    tau_e: float = 1.0
    vth: float = 1.0
    rho: float = 1.0
    kpar_scale: float = 1.0


def grad_z_periodic(f: jnp.ndarray, dz: float) -> jnp.ndarray:
    """Centered periodic derivative along the last axis."""

    if dz <= 0.0:
        raise ValueError("dz must be > 0")
    return (jnp.roll(f, -1, axis=-1) - jnp.roll(f, 1, axis=-1)) / (2.0 * dz)


def compute_b(grid: SpectralGrid, geom: SAlphaGeometry, rho: float) -> jnp.ndarray:
    """Compute b = rho^2 * k_perp^2(kx, ky, theta) for s-alpha geometry."""

    if rho <= 0.0:
        raise ValueError("rho must be > 0")
    kx0 = grid.kx[None, :, None]
    ky = grid.ky[:, None, None]
    theta = grid.z[None, None, :]
    kx_eff = geom.kx_effective(kx0, ky, theta)
    kperp2 = kx_eff * kx_eff + ky * ky
    return (rho * rho) * kperp2


def quasineutrality_phi(G: jnp.ndarray, Jl: jnp.ndarray, tau_e: float) -> jnp.ndarray:
    """Solve electrostatic quasineutrality for phi.

    Uses an adiabatic electron closure:
        (tau_e + 1 - sum_l J_l^2) * phi = sum_l J_l * G_{l,m=0}
    """

    if tau_e <= 0.0:
        raise ValueError("tau_e must be > 0")
    Gm0 = G[:, 0, ...]
    num = jnp.sum(Jl * Gm0, axis=0)
    den = tau_e + 1.0 - jnp.sum(Jl * Jl, axis=0)
    den_safe = jnp.where(den == 0.0, jnp.inf, den)
    return num / den_safe


def build_H(G: jnp.ndarray, Jl: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Map G -> H = G + J_l(b) * phi * delta_{m0}."""

    return G.at[:, 0, ...].add(Jl * phi)


def streaming_term(H: jnp.ndarray, dz: float, vth: float) -> jnp.ndarray:
    """Streaming term using Hermite ladder and real-space z derivative."""

    if vth <= 0.0:
        raise ValueError("vth must be > 0")
    dH_dz = grad_z_periodic(H, dz)
    Nm = H.shape[1]
    sqrt_p, sqrt_m = hermite_ladder_coeffs(Nm - 1)
    sqrt_p = sqrt_p[:Nm]
    sqrt_m = sqrt_m[:Nm]

    pad = ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0))
    dH_pad = jnp.pad(dH_dz, pad)
    dH_plus = dH_pad[:, 2:, ...]
    dH_minus = dH_pad[:, :-2, ...]

    ladder = sqrt_p[None, :, None, None, None] * dH_plus + sqrt_m[None, :, None, None, None] * dH_minus
    return vth * ladder


def linear_rhs(
    G: jnp.ndarray,
    grid: SpectralGrid,
    geom: SAlphaGeometry,
    params: LinearParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the linear RHS and electrostatic potential.

    Parameters
    ----------
    G : jnp.ndarray
        Laguerre-Hermite moments with shape (Nl, Nm, Ny, Nx, Nz).
    grid : SpectralGrid
        Flux-tube spectral grid.
    geom : SAlphaGeometry
        Analytic s-alpha geometry.
    params : LinearParams
        Physical and normalization parameters.
    """

    if G.ndim != 5:
        raise ValueError("G must have shape (Nl, Nm, Ny, Nx, Nz)")
    dz = float(grid.z[1] - grid.z[0])
    b = compute_b(grid, geom, params.rho)
    Jl = J_l_all(b, l_max=G.shape[0] - 1)
    phi = quasineutrality_phi(G, Jl, params.tau_e)
    ky0 = grid.ky == 0.0
    kx0 = grid.kx == 0.0
    mask0 = ky0[:, None] & kx0[None, :]
    phi = phi.at[mask0, :].set(0.0)
    H = build_H(G, Jl, phi)
    stream = streaming_term(H, dz, params.vth)
    dG = -params.kpar_scale * stream
    return dG, phi
