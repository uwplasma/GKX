"""Per-mode channel contribution kernels for GX diagnostics."""

from __future__ import annotations

import jax.numpy as jnp

from spectraxgk.diagnostics_weights import (
    _gx_fac_mask,
    _gx_fac_mask_nonzero,
    _jl_family,
    _species_array,
)
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import LinearCache, LinearParams
from spectraxgk.terms.operators import shift_axis


def _gx_heat_flux_channel_contrib_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool,
    flux_scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if G.ndim == 5:
        Gs = G[None, ...]
    else:
        Gs = G
    ns = Gs.shape[0]
    fac = _gx_fac_mask_nonzero(grid, use_dealias=use_dealias)[:, :, None]
    flx = flux_fac[None, None, :]
    ky = grid.ky[:, None, None]
    vphi = 1.0j * ky * phi
    vapar = 1.0j * ky * apar
    vbpar = 1.0j * ky * bpar
    Jl, JlB, Jfac = _jl_family(cache)
    sqrt2 = jnp.sqrt(2.0)
    sqrt32 = jnp.sqrt(1.5)
    nt = _species_array(params.density, ns) * _species_array(params.temp, ns)
    vth = _species_array(params.vth, ns)
    tz = _species_array(params.tz, ns)
    es_species = []
    apar_species = []
    bpar_species = []
    for s in range(ns):
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
        q_bar = jnp.sum(Jfac_s * G1 + Jl_s * (sqrt32 * G3 + G1), axis=0)
        Jfac_m1 = shift_axis(Jfac_s, -1, axis=0)
        qB_bar = jnp.sum(
            Jfac_s * G0 + Jfac_m1 * G0 + (1.0 / sqrt2) * JlB_s * G2,
            axis=0,
        )
        weight = 2.0 * flx * fac * nt[s] * flux_scale
        es_species.append((jnp.conj(vphi) * p_bar * weight).real)
        apar_species.append((-vth[s] * jnp.conj(vapar) * q_bar * weight).real)
        bpar_species.append((tz[s] * jnp.conj(vbpar) * qB_bar * weight).real)
    return (
        jnp.stack(es_species, axis=0),
        jnp.stack(apar_species, axis=0),
        jnp.stack(bpar_species, axis=0),
    )


def _gx_particle_flux_channel_contrib_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool,
    flux_scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if G.ndim == 5:
        Gs = G[None, ...]
    else:
        Gs = G
    ns = Gs.shape[0]
    zero_shape = (ns, grid.ky.size, grid.kx.size, grid.z.size)
    if ns == 1:
        zero = jnp.zeros(zero_shape, dtype=jnp.real(phi).dtype)
        return zero, zero, zero
    fac = _gx_fac_mask_nonzero(grid, use_dealias=use_dealias)[:, :, None]
    flx = flux_fac[None, None, :]
    ky = grid.ky[:, None, None]
    vphi = 1.0j * ky * phi
    vapar = 1.0j * ky * apar
    vbpar = 1.0j * ky * bpar
    Jl, JlB, _ = _jl_family(cache)
    dens = _species_array(params.density, ns)
    vth = _species_array(params.vth, ns)
    tz = _species_array(params.tz, ns)
    es_species = []
    apar_species = []
    bpar_species = []
    for s in range(ns):
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
        weight = 2.0 * flx * fac * dens[s] * flux_scale
        es_species.append((jnp.conj(vphi) * n_bar * weight).real)
        apar_species.append((-vth[s] * jnp.conj(vapar) * u_bar * weight).real)
        bpar_species.append((tz[s] * jnp.conj(vbpar) * uB_bar * weight).real)
    return (
        jnp.stack(es_species, axis=0),
        jnp.stack(apar_species, axis=0),
        jnp.stack(bpar_species, axis=0),
    )


def _gx_turbulent_heating_contrib_species(
    G: jnp.ndarray,
    G_old: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    phi_old: jnp.ndarray,
    apar_old: jnp.ndarray,
    bpar_old: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    dt: jnp.ndarray | float,
    *,
    use_dealias: bool,
) -> jnp.ndarray:
    if G.ndim == 5:
        Gs = G[None, ...]
    else:
        Gs = G
    if G_old.ndim == 5:
        G_old_s = G_old[None, ...]
    else:
        G_old_s = G_old

    ns = Gs.shape[0]
    fac = _gx_fac_mask(grid, use_dealias=use_dealias)[:, :, None]
    vol = vol_fac[None, None, :]
    Jl, JlB, _ = _jl_family(cache)
    dens = _species_array(params.density, ns)
    vth = _species_array(params.vth, ns)
    tz = _species_array(params.tz, ns)
    zt = jnp.where(tz == 0.0, 0.0, 1.0 / tz)
    real_dtype = jnp.real(phi).dtype
    dt_arr = jnp.asarray(dt, dtype=real_dtype)
    dt_safe = jnp.where(dt_arr == 0.0, jnp.asarray(jnp.inf, dtype=real_dtype), dt_arr)

    dphidt = (phi - phi_old) / dt_safe
    dadt = (apar - apar_old) / dt_safe
    dbdt = (bpar - bpar_old) / dt_safe

    species_contrib = []
    for s in range(ns):
        Jl_s = Jl[s]
        JlB_s = JlB[s]
        G_s = Gs[s]
        G_prev_s = G_old_s[s]

        def _get_m(source: jnp.ndarray, m_idx: int) -> jnp.ndarray:
            if source.shape[1] <= m_idx:
                return jnp.zeros_like(source[:, 0, ...])
            return source[:, m_idx, ...]

        G0 = _get_m(G_s, 0)
        G1 = _get_m(G_s, 1)
        G0_old = _get_m(G_prev_s, 0)
        G1_old = _get_m(G_prev_s, 1)

        H0_old = (
            G0_old + zt[s] * Jl_s * phi_old[None, ...] + JlB_s * bpar_old[None, ...]
        )
        H1_old = G1_old - zt[s] * vth[s] * Jl_s * apar_old[None, ...]
        H0 = G0 + zt[s] * Jl_s * phi[None, ...] + JlB_s * bpar[None, ...]
        H1 = G1 - zt[s] * vth[s] * Jl_s * apar[None, ...]

        n_bar_old = jnp.sum(Jl_s * H0_old, axis=0)
        u_bar_old = jnp.sum(Jl_s * H1_old, axis=0)
        uB_bar_old = jnp.sum(JlB_s * H0_old, axis=0)
        n_bar = jnp.sum(Jl_s * H0, axis=0)
        u_bar = jnp.sum(Jl_s * H1, axis=0)
        uB_bar = jnp.sum(JlB_s * H0, axis=0)

        dn_bardt = (n_bar - n_bar_old) / dt_safe
        du_bardt = (u_bar - u_bar_old) / dt_safe
        duB_bardt = (uB_bar - uB_bar_old) / dt_safe

        h_dchidt = (
            jnp.conj(dphidt) * n_bar_old
            - vth[s] * jnp.conj(dadt) * u_bar_old
            + tz[s] * jnp.conj(dbdt) * uB_bar_old
        )
        chi_dhdt = (
            jnp.conj(phi_old) * dn_bardt
            - vth[s] * jnp.conj(apar_old) * du_bardt
            + tz[s] * jnp.conj(bpar_old) * duB_bardt
        )
        species_contrib.append(
            0.5 * (h_dchidt.real - chi_dhdt.real) * dens[s] * fac * vol
        )

    return jnp.stack(species_contrib, axis=0)
