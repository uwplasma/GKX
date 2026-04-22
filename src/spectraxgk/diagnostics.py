"""Simulation diagnostics for gyrokinetic runs."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry import FluxTubeGeometryData, FluxTubeGeometryLike
from spectraxgk.gyroaverage import gamma0
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import LinearCache, LinearParams
from spectraxgk.terms.operators import shift_axis


ArrayLike = jnp.ndarray | np.ndarray


@dataclass(frozen=True)
class ResolvedDiagnostics:
    """Optional resolved nonlinear diagnostics stored per sample."""

    Phi2_kxt: ArrayLike | None = None
    Phi2_kyt: ArrayLike | None = None
    Phi2_kxkyt: ArrayLike | None = None
    Phi2_zt: ArrayLike | None = None
    Phi2_zonal_t: ArrayLike | None = None
    Phi2_zonal_kxt: ArrayLike | None = None
    Phi2_zonal_zt: ArrayLike | None = None
    Phi_zonal_mode_kxt: ArrayLike | None = None
    Wg_kxst: ArrayLike | None = None
    Wg_kyst: ArrayLike | None = None
    Wg_kxkyst: ArrayLike | None = None
    Wg_zst: ArrayLike | None = None
    Wg_lmst: ArrayLike | None = None
    Wphi_kxst: ArrayLike | None = None
    Wphi_kyst: ArrayLike | None = None
    Wphi_kxkyst: ArrayLike | None = None
    Wphi_zst: ArrayLike | None = None
    Wapar_kxst: ArrayLike | None = None
    Wapar_kyst: ArrayLike | None = None
    Wapar_kxkyst: ArrayLike | None = None
    Wapar_zst: ArrayLike | None = None
    HeatFlux_kxst: ArrayLike | None = None
    HeatFlux_kyst: ArrayLike | None = None
    HeatFlux_kxkyst: ArrayLike | None = None
    HeatFlux_zst: ArrayLike | None = None
    HeatFluxES_kxst: ArrayLike | None = None
    HeatFluxES_kyst: ArrayLike | None = None
    HeatFluxES_kxkyst: ArrayLike | None = None
    HeatFluxES_zst: ArrayLike | None = None
    HeatFluxApar_kxst: ArrayLike | None = None
    HeatFluxApar_kyst: ArrayLike | None = None
    HeatFluxApar_kxkyst: ArrayLike | None = None
    HeatFluxApar_zst: ArrayLike | None = None
    HeatFluxBpar_kxst: ArrayLike | None = None
    HeatFluxBpar_kyst: ArrayLike | None = None
    HeatFluxBpar_kxkyst: ArrayLike | None = None
    HeatFluxBpar_zst: ArrayLike | None = None
    ParticleFlux_kxst: ArrayLike | None = None
    ParticleFlux_kyst: ArrayLike | None = None
    ParticleFlux_kxkyst: ArrayLike | None = None
    ParticleFlux_zst: ArrayLike | None = None
    ParticleFluxES_kxst: ArrayLike | None = None
    ParticleFluxES_kyst: ArrayLike | None = None
    ParticleFluxES_kxkyst: ArrayLike | None = None
    ParticleFluxES_zst: ArrayLike | None = None
    ParticleFluxApar_kxst: ArrayLike | None = None
    ParticleFluxApar_kyst: ArrayLike | None = None
    ParticleFluxApar_kxkyst: ArrayLike | None = None
    ParticleFluxApar_zst: ArrayLike | None = None
    ParticleFluxBpar_kxst: ArrayLike | None = None
    ParticleFluxBpar_kyst: ArrayLike | None = None
    ParticleFluxBpar_kxkyst: ArrayLike | None = None
    ParticleFluxBpar_zst: ArrayLike | None = None
    TurbulentHeating_kxst: ArrayLike | None = None
    TurbulentHeating_kyst: ArrayLike | None = None
    TurbulentHeating_kxkyst: ArrayLike | None = None
    TurbulentHeating_zst: ArrayLike | None = None


@dataclass(frozen=True)
class SimulationDiagnostics:
    """Streaming diagnostics at each sample time."""

    t: ArrayLike
    dt_t: ArrayLike
    dt_mean: ArrayLike
    gamma_t: ArrayLike
    omega_t: ArrayLike
    Wg_t: ArrayLike
    Wphi_t: ArrayLike
    Wapar_t: ArrayLike
    heat_flux_t: ArrayLike
    particle_flux_t: ArrayLike
    energy_t: ArrayLike
    heat_flux_species_t: ArrayLike | None = None
    particle_flux_species_t: ArrayLike | None = None
    turbulent_heating_t: ArrayLike | None = None
    turbulent_heating_species_t: ArrayLike | None = None
    phi_mode_t: ArrayLike | None = None
    resolved: ResolvedDiagnostics | None = None


# Compatibility aliases preserved while the broader rename propagates through
# the comparison and audit tooling.
GXResolvedDiagnostics = ResolvedDiagnostics
GXDiagnostics = SimulationDiagnostics


def gx_volume_factors(geom: FluxTubeGeometryLike, grid: SpectralGrid) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (vol_fac, flux_fac) matching GX's volume weights."""

    theta = grid.z
    if isinstance(geom, FluxTubeGeometryData):
        jacobian = geom.jacobian(theta)
        grho = geom.grho(theta)
    else:
        bmag = geom.bmag(theta)
        gradpar = jnp.asarray(geom.gradpar())
        jacobian = 1.0 / (jnp.abs(gradpar) * bmag)
        grho = jnp.ones_like(jacobian)
    vol_fac = jacobian / jnp.sum(jacobian)
    flux_fac = jacobian / jnp.sum(jacobian * grho)
    return vol_fac, flux_fac


def _gx_fac_mask(grid: SpectralGrid, *, use_dealias: bool) -> jnp.ndarray:
    """Return GX-style fac*mask for (ky, kx) weighting."""

    ky = grid.ky
    has_negative = jnp.any(ky < 0.0)
    fac = jnp.where(has_negative, 1.0, jnp.where(ky == 0.0, 1.0, 2.0))
    fac = fac[:, None] * jnp.ones((1, grid.kx.size), dtype=fac.dtype)
    if use_dealias:
        mask = grid.dealias_mask.astype(fac.dtype)
    else:
        mask = jnp.ones_like(fac)
    return fac * mask


def _gx_fac_mask_nonzero(grid: SpectralGrid, *, use_dealias: bool) -> jnp.ndarray:
    """Return fac*mask that excludes ky=0 and uses GX positive-ky weighting."""

    ky = grid.ky
    # GX flux kernels operate on the rFFT-positive ky set and already include
    # the Hermitian pair factor of 2 in the kernel expression itself. For the
    # full-ky SPECTRAX layout we therefore keep unit weight on ky>0 here.
    fac = jnp.where(ky > 0.0, 1.0, 0.0)
    fac = fac[:, None] * jnp.ones((1, grid.kx.size), dtype=fac.dtype)
    if use_dealias:
        mask = grid.dealias_mask.astype(fac.dtype)
    else:
        mask = jnp.ones_like(fac)
    return fac * mask


def _gx_fac_mask_cached(cache: LinearCache, *, use_dealias: bool) -> jnp.ndarray:
    """Return GX-style fac*mask from a linear cache."""

    ky = cache.ky
    has_negative = jnp.any(ky < 0.0)
    fac = jnp.where(has_negative, 1.0, jnp.where(ky == 0.0, 1.0, 2.0))
    fac = fac[:, None] * jnp.ones((1, cache.kx.size), dtype=fac.dtype)
    if use_dealias:
        mask = cache.dealias_mask.astype(fac.dtype)
    else:
        mask = jnp.ones_like(fac)
    return fac * mask


def _species_array(val: float | jnp.ndarray, ns: int) -> jnp.ndarray:
    arr = jnp.asarray(val)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (ns,))
    return arr


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


def _jl_family(cache: LinearCache) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (Jl, JlB, Jfac) arrays in GX conventions."""

    Jl = cache.Jl
    if Jl.ndim == 5:
        Jl_s = Jl
    elif Jl.ndim == 4:
        Jl_s = Jl[None, ...]
    else:
        raise ValueError(f"unexpected Jl rank {Jl.ndim}; expected 4 or 5")
    JlB = cache.JlB
    if JlB.ndim == 5:
        JlB_s = JlB
    elif JlB.ndim == 4:
        JlB_s = JlB[None, ...]
    else:
        raise ValueError(f"unexpected JlB rank {JlB.ndim}; expected 4 or 5")

    Nl = Jl_s.shape[1]
    l = jnp.arange(Nl, dtype=Jl_s.dtype)[None, :, None, None, None]
    Jl_m1 = shift_axis(Jl_s, -1, axis=1)
    Jl_p1 = shift_axis(Jl_s, 1, axis=1)
    JflrA = l * Jl_m1 + 2.0 * l * Jl_s + (l + 1.0) * Jl_p1
    Jfac = 1.5 * Jl_s + JflrA
    return Jl_s, JlB_s, Jfac


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
        qB_bar = jnp.sum(Jfac_s * G0 + Jfac_m1 * G0 + (1.0 / sqrt2) * JlB_s * G2, axis=0)
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
        Nm = G_s.shape[1]

        def _get_m(source: jnp.ndarray, m_idx: int) -> jnp.ndarray:
            if source.shape[1] <= m_idx:
                return jnp.zeros_like(source[:, 0, ...])
            return source[:, m_idx, ...]

        G0 = _get_m(G_s, 0)
        G1 = _get_m(G_s, 1)
        G0_old = _get_m(G_prev_s, 0)
        G1_old = _get_m(G_prev_s, 1)

        H0_old = G0_old + zt[s] * Jl_s * phi_old[None, ...] + JlB_s * bpar_old[None, ...]
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
        species_contrib.append(0.5 * (h_dchidt.real - chi_dhdt.real) * dens[s] * fac * vol)

    return jnp.stack(species_contrib, axis=0)


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
