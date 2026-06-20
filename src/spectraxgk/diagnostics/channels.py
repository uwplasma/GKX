"""Per-mode channel contribution kernels for runtime diagnostics."""

from __future__ import annotations

import jax.numpy as jnp

from spectraxgk.diagnostics.weights import (
    _hermitian_mode_weight,
    _transport_mode_weight,
    _jl_family,
    _species_array,
)
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.terms.operators import shift_axis


def _mask_modes(value: jnp.ndarray, active: jnp.ndarray) -> jnp.ndarray:
    """Zero inactive spectral modes before products or moment reductions."""

    return jnp.where(active, value, jnp.asarray(0, dtype=value.dtype))


def _heat_flux_channel_contrib_species(
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
    fac = _transport_mode_weight(grid, use_dealias=use_dealias)[:, :, None]
    active = fac != 0.0
    flx = flux_fac[None, None, :]
    ky = grid.ky[:, None, None]
    vphi = _mask_modes(1.0j * ky * phi, active)
    vapar = _mask_modes(1.0j * ky * apar, active)
    vbpar = _mask_modes(1.0j * ky * bpar, active)
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
        G_s = _mask_modes(Gs[s], active[None, None, ...])
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


def _particle_flux_channel_contrib_species(
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
    fac = _transport_mode_weight(grid, use_dealias=use_dealias)[:, :, None]
    active = fac != 0.0
    flx = flux_fac[None, None, :]
    ky = grid.ky[:, None, None]
    vphi = _mask_modes(1.0j * ky * phi, active)
    vapar = _mask_modes(1.0j * ky * apar, active)
    vbpar = _mask_modes(1.0j * ky * bpar, active)
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
        G_s = _mask_modes(Gs[s], active[None, None, ...])
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


def _with_species_axis(G: jnp.ndarray) -> jnp.ndarray:
    return G[None, ...] if G.ndim == 5 else G


def _safe_heating_dt(dt: jnp.ndarray | float, real_dtype: jnp.dtype) -> jnp.ndarray:
    dt_arr = jnp.asarray(dt, dtype=real_dtype)
    return jnp.where(dt_arr == 0.0, jnp.asarray(jnp.inf, dtype=real_dtype), dt_arr)


def _masked_time_derivative(
    value: jnp.ndarray,
    old_value: jnp.ndarray,
    active: jnp.ndarray,
    dt_safe: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    value_masked = _mask_modes(value, active)
    old_masked = _mask_modes(old_value, active)
    return value_masked, old_masked, (value_masked - old_masked) / dt_safe


def _moment_or_zero(source: jnp.ndarray, m_idx: int) -> jnp.ndarray:
    if source.shape[1] <= m_idx:
        return jnp.zeros_like(source[:, 0, ...])
    return source[:, m_idx, ...]


def _heating_moments(
    G_s: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    Jl_s: jnp.ndarray,
    JlB_s: jnp.ndarray,
    zt_s: jnp.ndarray,
    vth_s: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    G0 = _moment_or_zero(G_s, 0)
    G1 = _moment_or_zero(G_s, 1)
    H0 = G0 + zt_s * Jl_s * phi[None, ...] + JlB_s * bpar[None, ...]
    H1 = G1 - zt_s * vth_s * Jl_s * apar[None, ...]
    return (
        jnp.sum(Jl_s * H0, axis=0),
        jnp.sum(Jl_s * H1, axis=0),
        jnp.sum(JlB_s * H0, axis=0),
    )


def _turbulent_heating_species_term(
    G_s: jnp.ndarray,
    G_prev_s: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    phi_old: jnp.ndarray,
    apar_old: jnp.ndarray,
    bpar_old: jnp.ndarray,
    dphidt: jnp.ndarray,
    dadt: jnp.ndarray,
    dbdt: jnp.ndarray,
    active: jnp.ndarray,
    dt_safe: jnp.ndarray,
    Jl_s: jnp.ndarray,
    JlB_s: jnp.ndarray,
    dens_s: jnp.ndarray,
    vth_s: jnp.ndarray,
    tz_s: jnp.ndarray,
    zt_s: jnp.ndarray,
    fac: jnp.ndarray,
    vol: jnp.ndarray,
) -> jnp.ndarray:
    G_s = _mask_modes(G_s, active[None, None, ...])
    G_prev_s = _mask_modes(G_prev_s, active[None, None, ...])
    n_old, u_old, uB_old = _heating_moments(
        G_prev_s,
        phi=phi_old,
        apar=apar_old,
        bpar=bpar_old,
        Jl_s=Jl_s,
        JlB_s=JlB_s,
        zt_s=zt_s,
        vth_s=vth_s,
    )
    n_now, u_now, uB_now = _heating_moments(
        G_s,
        phi=phi,
        apar=apar,
        bpar=bpar,
        Jl_s=Jl_s,
        JlB_s=JlB_s,
        zt_s=zt_s,
        vth_s=vth_s,
    )
    dn_bardt = (n_now - n_old) / dt_safe
    du_bardt = (u_now - u_old) / dt_safe
    duB_bardt = (uB_now - uB_old) / dt_safe
    h_dchidt = (
        jnp.conj(dphidt) * n_old
        - vth_s * jnp.conj(dadt) * u_old
        + tz_s * jnp.conj(dbdt) * uB_old
    )
    chi_dhdt = (
        jnp.conj(phi_old) * dn_bardt
        - vth_s * jnp.conj(apar_old) * du_bardt
        + tz_s * jnp.conj(bpar_old) * duB_bardt
    )
    return 0.5 * (h_dchidt.real - chi_dhdt.real) * dens_s * fac * vol


def _turbulent_heating_contrib_species(
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
    Gs = _with_species_axis(G)
    G_old_s = _with_species_axis(G_old)
    ns = Gs.shape[0]
    fac = _hermitian_mode_weight(grid, use_dealias=use_dealias)[:, :, None]
    active = fac != 0.0
    vol = vol_fac[None, None, :]
    Jl, JlB, _ = _jl_family(cache)
    dens = _species_array(params.density, ns)
    vth = _species_array(params.vth, ns)
    tz = _species_array(params.tz, ns)
    zt = jnp.where(tz == 0.0, 0.0, 1.0 / tz)
    real_dtype = jnp.real(phi).dtype
    dt_safe = _safe_heating_dt(dt, real_dtype)
    phi_masked, phi_old_masked, dphidt = _masked_time_derivative(
        phi, phi_old, active, dt_safe
    )
    apar_masked, apar_old_masked, dadt = _masked_time_derivative(
        apar, apar_old, active, dt_safe
    )
    bpar_masked, bpar_old_masked, dbdt = _masked_time_derivative(
        bpar, bpar_old, active, dt_safe
    )

    species_contrib = []
    for s in range(ns):
        species_contrib.append(
            _turbulent_heating_species_term(
                Gs[s],
                G_old_s[s],
                phi=phi_masked,
                apar=apar_masked,
                bpar=bpar_masked,
                phi_old=phi_old_masked,
                apar_old=apar_old_masked,
                bpar_old=bpar_old_masked,
                dphidt=dphidt,
                dadt=dadt,
                dbdt=dbdt,
                active=active,
                dt_safe=dt_safe,
                Jl_s=Jl[s],
                JlB_s=JlB[s],
                dens_s=dens[s],
                vth_s=vth[s],
                tz_s=tz[s],
                zt_s=zt[s],
                fac=fac,
                vol=vol,
            )
        )

    return jnp.stack(species_contrib, axis=0)
