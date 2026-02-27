"""GX-style diagnostics for gyrokinetic simulations."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.gyroaverage import gamma0
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import LinearCache, LinearParams
from spectraxgk.terms.operators import shift_axis


@dataclass(frozen=True)
class GXDiagnostics:
    """Streaming GX-style diagnostics at each sample time."""

    t: jnp.ndarray
    gamma_t: jnp.ndarray
    omega_t: jnp.ndarray
    Wg_t: jnp.ndarray
    Wphi_t: jnp.ndarray
    Wapar_t: jnp.ndarray
    heat_flux_t: jnp.ndarray
    particle_flux_t: jnp.ndarray
    energy_t: jnp.ndarray


def gx_volume_factors(geom: SAlphaGeometry, grid: SpectralGrid) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (vol_fac, flux_fac) matching GX's volume weights."""

    theta = grid.z
    bmag = geom.bmag(theta)
    gradpar = jnp.asarray(geom.gradpar())
    jacobian = 1.0 / (jnp.abs(gradpar) * bmag)
    vol_fac = jacobian / jnp.sum(jacobian)
    flux_fac = jacobian / jnp.sum(jacobian)
    return vol_fac, flux_fac


def _gx_fac_mask(grid: SpectralGrid) -> jnp.ndarray:
    """Return GX-style fac*mask for (ky, kx) weighting."""

    ky = grid.ky
    fac = jnp.where(ky == 0.0, 1.0, 2.0)
    fac = fac[:, None] * jnp.ones((1, grid.kx.size), dtype=fac.dtype)
    mask = grid.dealias_mask.astype(fac.dtype)
    return fac * mask


def _gx_fac_mask_nonzero(grid: SpectralGrid) -> jnp.ndarray:
    """Return fac*mask that excludes ky=0 contributions (for fluxes)."""

    ky = grid.ky
    fac = jnp.where(ky == 0.0, 0.0, 2.0)
    fac = fac[:, None] * jnp.ones((1, grid.kx.size), dtype=fac.dtype)
    mask = grid.dealias_mask.astype(fac.dtype)
    return fac * mask


def _species_array(val: float | jnp.ndarray, ns: int) -> jnp.ndarray:
    arr = jnp.asarray(val)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (ns,))
    return arr


def gx_Wg(G: jnp.ndarray, grid: SpectralGrid, params: LinearParams, vol_fac: jnp.ndarray) -> jnp.ndarray:
    """GX Wg diagnostic (free energy in g)."""

    fac = _gx_fac_mask(grid)
    fac = fac[None, None, None, :, :, None]
    vol = vol_fac[None, None, None, None, None, :]

    if G.ndim == 5:
        Gs = G[None, ...]
    else:
        Gs = G

    ns = Gs.shape[0]
    nt = _species_array(params.density, ns) * _species_array(params.temp, ns)
    nt = nt[:, None, None, None, None, None]

    g2 = jnp.abs(Gs) ** 2
    return 0.5 * jnp.sum(g2 * fac * vol * nt)


def gx_Wphi_krehm(
    phi: jnp.ndarray,
    grid: SpectralGrid,
    params: LinearParams,
    vol_fac: jnp.ndarray,
) -> jnp.ndarray:
    """GX Krehm electrostatic energy (Wphi) diagnostic."""

    kx = grid.kx[None, :]
    ky = grid.ky[:, None]
    kperp2 = kx * kx + ky * ky
    rho = jnp.asarray(params.rho)
    if rho.ndim == 0:
        rho = rho[None]
    rho2 = rho * rho
    fac = _gx_fac_mask(grid)
    vol = vol_fac[None, None, :]

    wphi = 0.0
    for rho2_s in rho2:
        b = 0.5 * kperp2 * rho2_s
        gam0 = gamma0(b)
        weight = (1.0 - gam0)[:, :, None] * fac[:, :, None]
        contrib = 0.5 * (2.0 / rho2_s) * jnp.abs(phi) ** 2 * weight * vol
        wphi = wphi + jnp.sum(contrib)
    return wphi


def gx_Wapar_krehm(
    apar: jnp.ndarray,
    grid: SpectralGrid,
) -> jnp.ndarray:
    """GX Krehm magnetic energy (Wapar) diagnostic."""

    kx = grid.kx[None, :]
    ky = grid.ky[:, None]
    kperp2 = kx * kx + ky * ky
    fac = _gx_fac_mask(grid)
    weight = fac[:, :, None]
    contrib = 0.5 * kperp2[:, :, None] * jnp.abs(apar) ** 2 * weight
    return jnp.sum(contrib)


def _jl_family(cache: LinearCache) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (Jl, JlB, Jfac) arrays in GX conventions."""

    Jl = cache.Jl
    if Jl.ndim == 6:
        Jl_s = Jl
    else:
        Jl_s = Jl[None, ...]
    JlB = cache.JlB
    if JlB.ndim == 6:
        JlB_s = JlB
    else:
        JlB_s = JlB[None, ...]

    Nl = Jl_s.shape[1]
    l = jnp.arange(Nl, dtype=Jl_s.dtype)[:, None, None, None, None]
    l = l[None, ...]
    Jl_m1 = shift_axis(Jl_s, -1, axis=1)
    Jl_p1 = shift_axis(Jl_s, 1, axis=1)
    JflrA = l * Jl_m1 + 2.0 * l * Jl_s + (l + 1.0) * Jl_p1
    Jfac = 1.5 * Jl_s + JflrA
    return Jl_s, JlB_s, Jfac


def gx_heat_flux(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
) -> jnp.ndarray:
    """GX heat flux diagnostic (gyroBohm units)."""

    if G.ndim == 5:
        Gs = G[None, ...]
    else:
        Gs = G
    ns = Gs.shape[0]
    fac = _gx_fac_mask_nonzero(grid)
    fac = fac[:, :, None]
    flx = flux_fac[None, None, :]

    ky = grid.ky[:, None, None]
    vphi = 1.0j * ky * phi
    vapar = 1.0j * ky * apar
    vbpar = 1.0j * ky * bpar

    Jl, JlB, Jfac = _jl_family(cache)
    sqrt2 = jnp.sqrt(2.0)
    sqrt32 = jnp.sqrt(1.5)

    flux = 0.0
    rho = _species_array(params.rho, ns)
    nt = _species_array(params.density, ns) * _species_array(params.temp, ns)
    vth = _species_array(params.vth, ns)
    tz = _species_array(params.tz, ns)

    kperp2 = cache.kperp2
    for s in range(ns):
        b_s = kperp2 * rho[s] * rho[s]
        Jl_s = Jl[s]
        JlB_s = JlB[s]
        Jfac_s = Jfac[s]
        G_s = Gs[s]
        Nm = G_s.shape[1]

        def _get_m(m_idx: int) -> jnp.ndarray:
            if Nm <= m_idx:
                return jnp.zeros_like(G_s[:, 0, ...])
            return G_s[:, m_idx, ...]

        G0 = _get_m(0)
        G1 = _get_m(1)
        G2 = _get_m(2)
        G3 = _get_m(3)

        p_bar = jnp.sum(Jfac_s * G0 + (1.0 / sqrt2) * Jl_s * G2, axis=0)
        q_bar = jnp.sum(
            Jfac_s * G1 + Jl_s * (sqrt32 * G3 + G1),
            axis=0,
        )
        Jfac_m1 = shift_axis(Jfac_s, -1, axis=0)
        qB_bar = jnp.sum(Jfac_s * G0 + Jfac_m1 * G0 + (1.0 / sqrt2) * JlB_s * G2, axis=0)

        fg = (
            jnp.conj(vphi) * p_bar
            - vth[s] * jnp.conj(vapar) * q_bar
            + tz[s] * jnp.conj(vbpar) * qB_bar
        )
        flux_s = jnp.sum((fg * 2.0 * flx * fac).real) * nt[s]
        flux = flux + flux_s
    return flux


def gx_particle_flux(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
) -> jnp.ndarray:
    """GX particle flux diagnostic."""

    if G.ndim == 5:
        Gs = G[None, ...]
    else:
        Gs = G
    ns = Gs.shape[0]
    fac = _gx_fac_mask_nonzero(grid)
    fac = fac[:, :, None]
    flx = flux_fac[None, None, :]

    ky = grid.ky[:, None, None]
    vphi = 1.0j * ky * phi
    vapar = 1.0j * ky * apar
    vbpar = 1.0j * ky * bpar

    Jl, JlB, _ = _jl_family(cache)
    flux = 0.0
    rho = _species_array(params.rho, ns)
    dens = _species_array(params.density, ns)
    vth = _species_array(params.vth, ns)
    tz = _species_array(params.tz, ns)

    kperp2 = cache.kperp2
    for s in range(ns):
        b_s = kperp2 * rho[s] * rho[s]
        Jl_s = Jl[s]
        JlB_s = JlB[s]
        G_s = Gs[s]
        Nm = G_s.shape[1]

        def _get_m(m_idx: int) -> jnp.ndarray:
            if Nm <= m_idx:
                return jnp.zeros_like(G_s[:, 0, ...])
            return G_s[:, m_idx, ...]

        G0 = _get_m(0)
        G1 = _get_m(1)

        n_bar = jnp.sum(Jl_s * G0, axis=0)
        u_bar = jnp.sum(Jl_s * G1, axis=0)
        uB_bar = jnp.sum(JlB_s * G0, axis=0)

        fg = jnp.conj(vphi) * n_bar - vth[s] * jnp.conj(vapar) * u_bar + tz[s] * jnp.conj(vbpar) * uB_bar
        flux_s = jnp.sum((fg * 2.0 * flx * fac).real) * dens[s]
        flux = flux + flux_s
    return flux


def gx_energy_total(Wg: jnp.ndarray, Wphi: jnp.ndarray, Wapar: jnp.ndarray) -> jnp.ndarray:
    return Wg + Wphi + Wapar
