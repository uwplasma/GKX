"""Simulation diagnostics for gyrokinetic runs."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from spectraxgk.gyroaverage import gamma0
from spectraxgk.diagnostics_channels import (
    _gx_heat_flux_channel_contrib_species,
    _gx_particle_flux_channel_contrib_species,
    _gx_turbulent_heating_contrib_species,
)
from spectraxgk.diagnostics_metadata import (
    ArrayLike,
    GXDiagnostics,
    GXResolvedDiagnostics,
    ResolvedDiagnostics,
    SimulationDiagnostics,
)
from spectraxgk.diagnostics_weights import (
    _gx_fac_mask,
    _gx_fac_mask_cached,
    _gx_fac_mask_nonzero,
    _jl_family,
    _species_array,
    gx_volume_factors,
)
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import LinearCache, LinearParams


__all__ = [
    "ArrayLike",
    "GXDiagnostics",
    "GXResolvedDiagnostics",
    "ResolvedDiagnostics",
    "SimulationDiagnostics",
    "_gx_fac_mask",
    "_gx_fac_mask_cached",
    "_gx_fac_mask_nonzero",
    "_gx_heat_flux_channel_contrib_species",
    "_gx_particle_flux_channel_contrib_species",
    "_gx_turbulent_heating_contrib_species",
    "_jl_family",
    "_reduce_scalar_kykxz",
    "_reduce_species_kykxz",
    "_species_array",
    "gx_Wapar",
    "gx_Wapar_krehm",
    "gx_Wapar_resolved",
    "gx_Wg",
    "gx_Wg_resolved",
    "gx_Wphi",
    "gx_Wphi_krehm",
    "gx_Wphi_resolved",
    "gx_energy_total",
    "gx_heat_flux",
    "gx_heat_flux_resolved_species",
    "gx_heat_flux_species",
    "gx_heat_flux_split_resolved_species",
    "gx_heat_flux_split_species",
    "gx_particle_flux",
    "gx_particle_flux_resolved_species",
    "gx_particle_flux_species",
    "gx_particle_flux_split_resolved_species",
    "gx_particle_flux_split_species",
    "gx_phi2_resolved",
    "gx_phi_zonal_line_kxt",
    "gx_phi_zonal_mode_kxt",
    "gx_turbulent_heating",
    "gx_turbulent_heating_resolved_species",
    "gx_turbulent_heating_species",
    "gx_volume_factors",
    "total_energy",
]


def gx_Wg(
    G: jnp.ndarray,
    grid: SpectralGrid,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
) -> jnp.ndarray:
    """GX Wg diagnostic (free energy in g)."""

    fac = _gx_fac_mask(grid, use_dealias=use_dealias)
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
    *,
    kx: jnp.ndarray | None = None,
    ky: jnp.ndarray | None = None,
    use_dealias: bool = True,
    gx_real_fft: bool = False,
    wphi_scale: float = 1.0,
) -> jnp.ndarray:
    """GX Krehm electrostatic energy (Wphi) diagnostic."""

    kx_arr = grid.kx if kx is None else kx
    ky_arr = grid.ky if ky is None else ky
    kx = jnp.asarray(kx_arr)[None, :]
    ky = jnp.asarray(ky_arr)[:, None]
    mask = grid.dealias_mask
    kperp2 = kx * kx + ky * ky
    rho = jnp.asarray(params.rho)
    if rho.ndim == 0:
        rho = rho[None]
    rho2 = rho * rho
    if use_dealias:
        mask = mask.astype(ky.dtype)
    else:
        mask = jnp.ones_like(mask, dtype=ky.dtype)
    if gx_real_fft:
        fac = jnp.where(ky[:, 0] > 0.0, 2.0, jnp.where(ky[:, 0] == 0.0, 1.0, 0.0))[:, None]
    else:
        has_negative = bool(np.any(np.asarray(ky_arr) < 0.0))
        if has_negative:
            fac = jnp.ones((ky.shape[0], kx.shape[1]), dtype=ky.dtype)
        else:
            fac = jnp.where(ky[:, 0] == 0.0, 1.0, 2.0)[:, None]
    fac = fac * mask
    vol = vol_fac[None, None, :]

    wphi = jnp.asarray(0.0, dtype=jnp.real(phi).dtype)
    for rho2_s in rho2:
        b = 0.5 * kperp2 * rho2_s
        gam0 = gamma0(b)
        weight = (1.0 - gam0)[:, :, None] * fac[:, :, None]
        contrib = 0.5 * (2.0 / rho2_s) * jnp.abs(phi) ** 2 * weight * vol
        wphi = wphi + jnp.sum(contrib)
    return wphi * jnp.asarray(wphi_scale, dtype=jnp.real(phi).dtype)


def gx_Wphi(
    phi: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    wphi_scale: float = 1.0,
) -> jnp.ndarray:
    """Standard GX electrostatic free energy diagnostic."""

    fac = _gx_fac_mask_cached(cache, use_dealias=use_dealias)
    weight = fac[:, :, None] * vol_fac[None, None, :]
    rho = jnp.asarray(params.rho)
    if rho.ndim == 0:
        rho = rho[None]
    rho2 = rho * rho

    wphi = jnp.asarray(0.0, dtype=jnp.real(phi).dtype)
    phi2 = jnp.abs(phi) ** 2
    for rho2_s in rho2:
        b = cache.kperp2 * rho2_s
        contrib = 0.5 * phi2 * (1.0 - gamma0(b)) * weight
        wphi = wphi + jnp.sum(contrib)
    return wphi * jnp.asarray(wphi_scale, dtype=jnp.real(phi).dtype)


def gx_Wapar_krehm(
    apar: jnp.ndarray,
    grid: SpectralGrid,
    *,
    kx: jnp.ndarray | None = None,
    ky: jnp.ndarray | None = None,
    use_dealias: bool = True,
) -> jnp.ndarray:
    """GX Krehm magnetic energy (Wapar) diagnostic."""

    kx_arr = grid.kx if kx is None else kx
    ky_arr = grid.ky if ky is None else ky
    kx = jnp.asarray(kx_arr)[None, :]
    ky = jnp.asarray(ky_arr)[:, None]
    kperp2 = kx * kx + ky * ky
    fac = _gx_fac_mask(grid, use_dealias=use_dealias)
    weight = fac[:, :, None]
    contrib = 0.5 * kperp2[:, :, None] * jnp.abs(apar) ** 2 * weight
    return jnp.sum(contrib)


def gx_Wapar(
    apar: jnp.ndarray,
    cache: LinearCache,
    vol_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
) -> jnp.ndarray:
    """Standard GX magnetic free energy diagnostic."""

    fac = _gx_fac_mask_cached(cache, use_dealias=use_dealias)
    weight = fac[:, :, None] * vol_fac[None, None, :]
    bmag2 = cache.bmag[None, None, :] ** 2 if cache.kperp2_bmag else 1.0
    contrib = 0.5 * jnp.abs(apar) ** 2 * cache.kperp2 * bmag2 * weight
    return jnp.sum(contrib)


def gx_heat_flux_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> jnp.ndarray:
    """GX heat flux diagnostic per species (gyroBohm units)."""

    es_contrib, apar_contrib, bpar_contrib = _gx_heat_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return jnp.sum(es_contrib + apar_contrib + bpar_contrib, axis=(1, 2, 3))


def gx_heat_flux_split_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ES, Apar, and Bpar heat-flux channels per species."""

    es_contrib, apar_contrib, bpar_contrib = _gx_heat_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return (
        jnp.sum(es_contrib, axis=(1, 2, 3)),
        jnp.sum(apar_contrib, axis=(1, 2, 3)),
        jnp.sum(bpar_contrib, axis=(1, 2, 3)),
    )


def gx_heat_flux(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> jnp.ndarray:
    """GX total heat flux diagnostic."""

    return jnp.sum(
        gx_heat_flux_species(
            G,
            phi,
            apar,
            bpar,
            cache,
            grid,
            params,
            flux_fac,
            use_dealias=use_dealias,
            flux_scale=flux_scale,
        )
    )


def gx_particle_flux_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> jnp.ndarray:
    """GX particle flux diagnostic per species."""

    es_contrib, apar_contrib, bpar_contrib = _gx_particle_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return jnp.sum(es_contrib + apar_contrib + bpar_contrib, axis=(1, 2, 3))


def gx_particle_flux_split_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ES, Apar, and Bpar particle-flux channels per species."""

    es_contrib, apar_contrib, bpar_contrib = _gx_particle_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return (
        jnp.sum(es_contrib, axis=(1, 2, 3)),
        jnp.sum(apar_contrib, axis=(1, 2, 3)),
        jnp.sum(bpar_contrib, axis=(1, 2, 3)),
    )


def gx_particle_flux(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> jnp.ndarray:
    """GX total particle flux diagnostic."""

    return jnp.sum(
        gx_particle_flux_species(
            G,
            phi,
            apar,
            bpar,
            cache,
            grid,
            params,
            flux_fac,
            use_dealias=use_dealias,
            flux_scale=flux_scale,
        )
    )


def gx_turbulent_heating_species(
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
    use_dealias: bool = True,
) -> jnp.ndarray:
    """GX turbulent heating diagnostic per species."""

    contrib = _gx_turbulent_heating_contrib_species(
        G,
        G_old,
        phi,
        apar,
        bpar,
        phi_old,
        apar_old,
        bpar_old,
        cache,
        grid,
        params,
        vol_fac,
        dt,
        use_dealias=use_dealias,
    )
    return jnp.sum(contrib, axis=(1, 2, 3))


def gx_turbulent_heating(
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
    use_dealias: bool = True,
) -> jnp.ndarray:
    """GX total turbulent heating diagnostic."""

    return jnp.sum(
        gx_turbulent_heating_species(
            G,
            G_old,
            phi,
            apar,
            bpar,
            phi_old,
            apar_old,
            bpar_old,
            cache,
            grid,
            params,
            vol_fac,
            dt,
            use_dealias=use_dealias,
        )
    )


def total_energy(Wg: ArrayLike, Wphi: ArrayLike, Wapar: ArrayLike) -> ArrayLike:
    return Wg + Wphi + Wapar


gx_energy_total = total_energy


def _reduce_scalar_kykxz(
    contrib: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reduce a ``(ky, kx, z)`` contribution to GX spectra views."""

    return (
        jnp.sum(contrib, axis=(0, 2)),
        jnp.sum(contrib, axis=(1, 2)),
        jnp.sum(contrib, axis=2),
        jnp.sum(contrib, axis=(0, 1)),
        jnp.sum(contrib),
    )


def _reduce_species_kykxz(
    contrib: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reduce a ``(species, ky, kx, z)`` contribution to GX spectra views."""

    return (
        jnp.sum(contrib, axis=(1, 2, 3)),
        jnp.sum(contrib, axis=(1, 3)),
        jnp.sum(contrib, axis=(2, 3)),
        jnp.sum(contrib, axis=3),
        jnp.sum(contrib, axis=(1, 2)),
    )


def gx_phi2_resolved(
    phi: jnp.ndarray,
    grid: SpectralGrid,
    vol_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return GX-style resolved ``Phi2`` reductions from a single field state."""

    fac = _gx_fac_mask(grid, use_dealias=use_dealias)
    contrib = jnp.abs(phi) ** 2 * fac[:, :, None] * vol_fac[None, None, :]
    phi2_kxt, phi2_kyt, phi2_kxkyt, phi2_zt, phi2_t = _reduce_scalar_kykxz(contrib)
    zonal_mask = (jnp.asarray(grid.ky) == 0.0).astype(contrib.dtype)[:, None, None]
    zonal = contrib * zonal_mask
    phi2_zonal_kxt, _phi2_zonal_kyt, _phi2_zonal_kxkyt, phi2_zonal_zt, phi2_zonal_t = _reduce_scalar_kykxz(zonal)
    return (
        phi2_t,
        phi2_kxt,
        phi2_kyt,
        phi2_kxkyt,
        phi2_zt,
        phi2_zonal_t,
        phi2_zonal_kxt,
        phi2_zonal_zt,
    )


def gx_phi_zonal_mode_kxt(
    phi: jnp.ndarray,
    grid: SpectralGrid,
    vol_fac: jnp.ndarray,
) -> jnp.ndarray:
    """Return the signed, volume-averaged zonal potential history per ``k_x``.

    This is the minimal complex zonal observable needed to construct
    Rosenbluth-Hinton / GAM response traces from nonlinear diagnostics without
    collapsing immediately to a positive-definite energy proxy.
    """

    zonal_mask = (jnp.asarray(grid.ky) == 0.0).astype(phi.dtype)[:, None, None]
    zonal = phi * zonal_mask
    return jnp.sum(zonal * vol_fac[None, None, :], axis=(0, 2))


def gx_phi_zonal_line_kxt(
    phi: jnp.ndarray,
    grid: SpectralGrid,
) -> jnp.ndarray:
    """Return the signed, unweighted line-averaged zonal potential per ``k_x``."""

    zonal_mask = (jnp.asarray(grid.ky) == 0.0).astype(phi.dtype)[:, None, None]
    zonal = phi * zonal_mask
    return jnp.sum(zonal, axis=(0, 2)) / jnp.asarray(phi.shape[-1], dtype=phi.real.dtype)


def gx_Wg_resolved(
    G: jnp.ndarray,
    grid: SpectralGrid,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return GX-style resolved free-energy reductions from a single state."""

    fac = _gx_fac_mask(grid, use_dealias=use_dealias)[None, None, None, :, :, None]
    vol = vol_fac[None, None, None, None, None, :]
    if G.ndim == 5:
        Gs = G[None, ...]
    else:
        Gs = G
    ns = Gs.shape[0]
    nt = _species_array(params.density, ns) * _species_array(params.temp, ns)
    contrib = 0.5 * jnp.abs(Gs) ** 2 * fac * vol * nt[:, None, None, None, None, None]
    Wg_kxst = jnp.sum(contrib, axis=(1, 2, 3, 5))
    Wg_kyst = jnp.sum(contrib, axis=(1, 2, 4, 5))
    Wg_kxkyst = jnp.sum(contrib, axis=(1, 2, 5))
    Wg_zst = jnp.sum(contrib, axis=(1, 2, 3, 4))
    Wg_st = jnp.sum(contrib, axis=(1, 2, 3, 4, 5))
    Wg_lmst = jnp.transpose(jnp.sum(contrib, axis=(3, 4, 5)), (0, 2, 1))
    return Wg_st, Wg_kxst, Wg_kyst, Wg_kxkyst, Wg_zst, Wg_lmst


def gx_Wphi_resolved(
    phi: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    wphi_scale: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return GX-style resolved electrostatic-energy reductions per species."""

    fac = _gx_fac_mask_cached(cache, use_dealias=use_dealias)
    weight = fac[:, :, None] * vol_fac[None, None, :]
    rho = jnp.asarray(params.rho)
    if rho.ndim == 0:
        rho = rho[None]
    phi2 = jnp.abs(phi) ** 2
    contrib_species = []
    scale = jnp.asarray(wphi_scale, dtype=jnp.real(phi).dtype)
    for rho2_s in rho * rho:
        b = cache.kperp2 * rho2_s
        contrib_species.append(0.5 * phi2 * (1.0 - gamma0(b)) * weight * scale)
    contrib = jnp.stack(contrib_species, axis=0)
    return _reduce_species_kykxz(contrib)


def gx_Wapar_resolved(
    apar: jnp.ndarray,
    cache: LinearCache,
    vol_fac: jnp.ndarray,
    *,
    nspecies: int,
    use_dealias: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return GX-style resolved magnetic-energy reductions per species."""

    fac = _gx_fac_mask_cached(cache, use_dealias=use_dealias)
    weight = fac[:, :, None] * vol_fac[None, None, :]
    bmag2 = cache.bmag[None, None, :] ** 2 if cache.kperp2_bmag else 1.0
    total = 0.5 * jnp.abs(apar) ** 2 * cache.kperp2 * bmag2 * weight
    ns = max(int(nspecies), 1)
    contrib = jnp.broadcast_to(total[None, ...] / float(ns), (ns,) + total.shape)
    return _reduce_species_kykxz(contrib)


def gx_heat_flux_resolved_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return GX-style resolved heat-flux reductions per species."""

    es_contrib, apar_contrib, bpar_contrib = _gx_heat_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return _reduce_species_kykxz(es_contrib + apar_contrib + bpar_contrib)


def gx_heat_flux_split_resolved_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Return resolved ES, Apar, and Bpar heat-flux channels per species."""

    es_contrib, apar_contrib, bpar_contrib = _gx_heat_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return (
        _reduce_species_kykxz(es_contrib),
        _reduce_species_kykxz(apar_contrib),
        _reduce_species_kykxz(bpar_contrib),
    )


def gx_particle_flux_resolved_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return GX-style resolved particle-flux reductions per species."""

    es_contrib, apar_contrib, bpar_contrib = _gx_particle_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return _reduce_species_kykxz(es_contrib + apar_contrib + bpar_contrib)


def gx_particle_flux_split_resolved_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
]:
    """Return resolved ES, Apar, and Bpar particle-flux channels per species."""

    es_contrib, apar_contrib, bpar_contrib = _gx_particle_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return (
        _reduce_species_kykxz(es_contrib),
        _reduce_species_kykxz(apar_contrib),
        _reduce_species_kykxz(bpar_contrib),
    )


def gx_turbulent_heating_resolved_species(
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
    use_dealias: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return GX-style resolved turbulent-heating reductions per species."""

    contrib = _gx_turbulent_heating_contrib_species(
        G,
        G_old,
        phi,
        apar,
        bpar,
        phi_old,
        apar_old,
        bpar_old,
        cache,
        grid,
        params,
        vol_fac,
        dt,
        use_dealias=use_dealias,
    )
    return _reduce_species_kykxz(contrib)
