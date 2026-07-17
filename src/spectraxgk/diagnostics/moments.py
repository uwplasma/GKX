"""Gyrokinetic quadrature, energy, transport-channel, and spectral diagnostics."""

from __future__ import annotations

import jax.numpy as jnp

from spectraxgk.core.grid import SpectralGrid
from spectraxgk.core.velocity import gamma0
from spectraxgk.diagnostics.metadata import ArrayLike
from spectraxgk.geometry import (
    FluxTubeGeometryData,
    FluxTubeGeometryLike,
    ensure_flux_tube_geometry_data,
)
from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.operators.linear.streaming import shift_axis


def fieldline_quadrature_weights(
    geom: FluxTubeGeometryLike, grid: SpectralGrid
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (vol_fac, flux_fac) for field-line volume and flux-surface weighting."""

    theta = grid.z
    if isinstance(geom, FluxTubeGeometryData):
        geom_data = ensure_flux_tube_geometry_data(geom, theta)
        jacobian = geom_data.jacobian(theta)
        grho = geom_data.grho(theta)
    else:
        bmag = geom.bmag(theta)
        gradpar = jnp.asarray(geom.gradpar())
        jacobian = 1.0 / (jnp.abs(gradpar) * bmag)
        grho = jnp.ones_like(jacobian)
    vol_fac = jacobian / jnp.sum(jacobian)
    flux_fac = jacobian / jnp.sum(jacobian * grho)
    return vol_fac, flux_fac


def _hermitian_mode_weight(grid: SpectralGrid, *, use_dealias: bool) -> jnp.ndarray:
    """Return Hermitian spectral weight for (ky, kx) weighting."""

    ky = grid.ky
    has_negative = jnp.any(ky < 0.0)
    fac = jnp.where(has_negative, 1.0, jnp.where(ky == 0.0, 1.0, 2.0))
    fac = fac[:, None] * jnp.ones((1, grid.kx.size), dtype=fac.dtype)
    if use_dealias:
        mask = grid.dealias_mask.astype(fac.dtype)
    else:
        mask = jnp.ones_like(fac)
    return fac * mask


def _transport_mode_weight(grid: SpectralGrid, *, use_dealias: bool) -> jnp.ndarray:
    """Return fac*mask that excludes ky=0 and uses positive-ky transport weighting."""

    ky = grid.ky
    # Positive-rFFT transport kernels include the Hermitian pair factor of 2 in
    # the kernel expression itself. For the full-ky SPECTRAX layout we therefore
    # keep unit weight on ky>0 here.
    fac = jnp.where(ky > 0.0, 1.0, 0.0)
    fac = fac[:, None] * jnp.ones((1, grid.kx.size), dtype=fac.dtype)
    if use_dealias:
        mask = grid.dealias_mask.astype(fac.dtype)
    else:
        mask = jnp.ones_like(fac)
    return fac * mask


def _cached_hermitian_mode_weight(
    cache: LinearCache, *, use_dealias: bool
) -> jnp.ndarray:
    """Return Hermitian spectral weight from a linear cache."""

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


def _jl_family(cache: LinearCache) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (Jl, JlB, Jfac) arrays in Laguerre-Hermite conventions."""

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
    ell = jnp.arange(Nl, dtype=Jl_s.dtype)[None, :, None, None, None]
    Jl_m1 = shift_axis(Jl_s, -1, axis=1)
    Jl_p1 = shift_axis(Jl_s, 1, axis=1)
    JflrA = ell * Jl_m1 + 2.0 * ell * Jl_s + (ell + 1.0) * Jl_p1
    Jfac = 1.5 * Jl_s + JflrA
    return Jl_s, JlB_s, Jfac


def _masked_abs2(value: jnp.ndarray, active: jnp.ndarray) -> jnp.ndarray:
    """Return ``abs(value)**2`` after zeroing inactive spectral modes."""

    zero = jnp.asarray(0, dtype=value.dtype)
    return jnp.abs(jnp.where(active, value, zero)) ** 2


def distribution_free_energy(
    G: jnp.ndarray,
    grid: SpectralGrid,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
) -> jnp.ndarray:
    """Distribution free-energy diagnostic (free energy in g)."""

    fac = _hermitian_mode_weight(grid, use_dealias=use_dealias)
    fac = fac[None, None, None, :, :, None]
    vol = vol_fac[None, None, None, None, None, :]

    if G.ndim == 5:
        Gs = G[None, ...]
    else:
        Gs = G

    ns = Gs.shape[0]
    nt = _species_array(params.density, ns) * _species_array(params.temp, ns)
    nt = nt[:, None, None, None, None, None]

    g2 = _masked_abs2(Gs, fac != 0.0)
    return 0.5 * jnp.sum(g2 * fac * vol * nt)


def electrostatic_field_energy(
    phi: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    wphi_scale: float = 1.0,
) -> jnp.ndarray:
    """Electrostatic field-energy diagnostic."""

    fac = _cached_hermitian_mode_weight(cache, use_dealias=use_dealias)
    weight = fac[:, :, None] * vol_fac[None, None, :]
    rho = jnp.asarray(params.rho)
    if rho.ndim == 0:
        rho = rho[None]
    rho2 = rho * rho

    wphi = jnp.asarray(0.0, dtype=jnp.real(phi).dtype)
    active = fac[:, :, None] != 0.0
    phi2 = _masked_abs2(phi, active)
    for rho2_s in rho2:
        b = cache.kperp2 * rho2_s
        contrib = 0.5 * phi2 * (1.0 - gamma0(b)) * weight
        wphi = wphi + jnp.sum(contrib)
    return wphi * jnp.asarray(wphi_scale, dtype=jnp.real(phi).dtype)


def magnetic_vector_potential_energy(
    apar: jnp.ndarray,
    cache: LinearCache,
    vol_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
) -> jnp.ndarray:
    """Magnetic vector-potential field-energy diagnostic."""

    fac = _cached_hermitian_mode_weight(cache, use_dealias=use_dealias)
    weight = fac[:, :, None] * vol_fac[None, None, :]
    bmag2 = cache.bmag[None, None, :] ** 2 if cache.kperp2_bmag else 1.0
    contrib = 0.5 * _masked_abs2(apar, weight != 0.0) * cache.kperp2 * bmag2 * weight
    return jnp.sum(contrib)


def total_energy(Wg: ArrayLike, Wphi: ArrayLike, Wapar: ArrayLike) -> ArrayLike:
    return Wg + Wphi + Wapar


runtime_energy_total = total_energy


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


def _reduce_scalar_kykxz(
    contrib: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reduce a ``(ky, kx, z)`` contribution to spectral diagnostics views."""

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
    """Reduce a ``(species, ky, kx, z)`` contribution to spectral diagnostics views."""

    return (
        jnp.sum(contrib, axis=(1, 2, 3)),
        jnp.sum(contrib, axis=(1, 3)),
        jnp.sum(contrib, axis=(2, 3)),
        jnp.sum(contrib, axis=3),
        jnp.sum(contrib, axis=(1, 2)),
    )


def phi2_resolved(
    phi: jnp.ndarray,
    grid: SpectralGrid,
    vol_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Return Resolved ``Phi2`` reductions from a single field state."""

    fac = _hermitian_mode_weight(grid, use_dealias=use_dealias)
    active = fac[:, :, None] != 0.0
    contrib = _masked_abs2(phi, active) * fac[:, :, None] * vol_fac[None, None, :]
    phi2_kxt, phi2_kyt, phi2_kxkyt, phi2_zt, phi2_t = _reduce_scalar_kykxz(contrib)
    zonal_mask = (jnp.asarray(grid.ky) == 0.0).astype(contrib.dtype)[:, None, None]
    zonal = contrib * zonal_mask
    phi2_zonal_kxt, _phi2_zonal_kyt, _phi2_zonal_kxkyt, phi2_zonal_zt, phi2_zonal_t = (
        _reduce_scalar_kykxz(zonal)
    )
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


def zonal_phi_mode_kxt(
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


def zonal_phi_line_kxt(
    phi: jnp.ndarray,
    grid: SpectralGrid,
) -> jnp.ndarray:
    """Return the signed, unweighted line-averaged zonal potential per ``k_x``."""

    zonal_mask = (jnp.asarray(grid.ky) == 0.0).astype(phi.dtype)[:, None, None]
    zonal = phi * zonal_mask
    return jnp.sum(zonal, axis=(0, 2)) / jnp.asarray(
        phi.shape[-1], dtype=phi.real.dtype
    )


def distribution_free_energy_resolved(
    G: jnp.ndarray,
    grid: SpectralGrid,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Return Resolved free-energy reductions from a single state."""

    fac = _hermitian_mode_weight(grid, use_dealias=use_dealias)[
        None, None, None, :, :, None
    ]
    vol = vol_fac[None, None, None, None, None, :]
    if G.ndim == 5:
        Gs = G[None, ...]
    else:
        Gs = G
    ns = Gs.shape[0]
    nt = _species_array(params.density, ns) * _species_array(params.temp, ns)
    g2 = _masked_abs2(Gs, fac != 0.0)
    contrib = 0.5 * g2 * fac * vol * nt[:, None, None, None, None, None]
    Wg_kxst = jnp.sum(contrib, axis=(1, 2, 3, 5))
    Wg_kyst = jnp.sum(contrib, axis=(1, 2, 4, 5))
    Wg_kxkyst = jnp.sum(contrib, axis=(1, 2, 5))
    Wg_zst = jnp.sum(contrib, axis=(1, 2, 3, 4))
    Wg_st = jnp.sum(contrib, axis=(1, 2, 3, 4, 5))
    Wg_lmst = jnp.transpose(jnp.sum(contrib, axis=(3, 4, 5)), (0, 2, 1))
    return Wg_st, Wg_kxst, Wg_kyst, Wg_kxkyst, Wg_zst, Wg_lmst


def electrostatic_field_energy_resolved(
    phi: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    wphi_scale: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return Resolved electrostatic-energy reductions per species."""

    fac = _cached_hermitian_mode_weight(cache, use_dealias=use_dealias)
    weight = fac[:, :, None] * vol_fac[None, None, :]
    rho = jnp.asarray(params.rho)
    if rho.ndim == 0:
        rho = rho[None]
    active = fac[:, :, None] != 0.0
    phi2 = _masked_abs2(phi, active)
    contrib_species = []
    scale = jnp.asarray(wphi_scale, dtype=jnp.real(phi).dtype)
    for rho2_s in rho * rho:
        b = cache.kperp2 * rho2_s
        contrib_species.append(0.5 * phi2 * (1.0 - gamma0(b)) * weight * scale)
    contrib = jnp.stack(contrib_species, axis=0)
    return _reduce_species_kykxz(contrib)


def magnetic_vector_potential_energy_resolved(
    apar: jnp.ndarray,
    cache: LinearCache,
    vol_fac: jnp.ndarray,
    *,
    nspecies: int,
    use_dealias: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return Resolved magnetic-energy reductions per species."""

    fac = _cached_hermitian_mode_weight(cache, use_dealias=use_dealias)
    weight = fac[:, :, None] * vol_fac[None, None, :]
    bmag2 = cache.bmag[None, None, :] ** 2 if cache.kperp2_bmag else 1.0
    total = 0.5 * _masked_abs2(apar, weight != 0.0) * cache.kperp2 * bmag2 * weight
    ns = max(int(nspecies), 1)
    contrib = jnp.broadcast_to(total[None, ...] / float(ns), (ns,) + total.shape)
    return _reduce_species_kykxz(contrib)


def heat_flux_resolved_species(
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
    """Return Resolved heat-flux reductions per species."""

    es_contrib, apar_contrib, bpar_contrib = _heat_flux_channel_contrib_species(
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


def heat_flux_channel_resolved_species(
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

    es_contrib, apar_contrib, bpar_contrib = _heat_flux_channel_contrib_species(
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


def particle_flux_resolved_species(
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
    """Return Resolved particle-flux reductions per species."""

    es_contrib, apar_contrib, bpar_contrib = _particle_flux_channel_contrib_species(
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


def particle_flux_channel_resolved_species(
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

    es_contrib, apar_contrib, bpar_contrib = _particle_flux_channel_contrib_species(
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


def turbulent_heating_resolved_species(
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
    """Return Resolved turbulent-heating reductions per species."""

    contrib = _turbulent_heating_contrib_species(
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


__all__ = [
    "ArrayLike",
    "_cached_hermitian_mode_weight",
    "_heat_flux_channel_contrib_species",
    "_hermitian_mode_weight",
    "_jl_family",
    "_masked_abs2",
    "_particle_flux_channel_contrib_species",
    "_reduce_scalar_kykxz",
    "_reduce_species_kykxz",
    "_species_array",
    "_transport_mode_weight",
    "_turbulent_heating_contrib_species",
    "distribution_free_energy",
    "distribution_free_energy_resolved",
    "electrostatic_field_energy",
    "electrostatic_field_energy_resolved",
    "fieldline_quadrature_weights",
    "heat_flux_channel_resolved_species",
    "heat_flux_resolved_species",
    "magnetic_vector_potential_energy",
    "magnetic_vector_potential_energy_resolved",
    "particle_flux_channel_resolved_species",
    "particle_flux_resolved_species",
    "phi2_resolved",
    "runtime_energy_total",
    "total_energy",
    "turbulent_heating_resolved_species",
    "zonal_phi_line_kxt",
    "zonal_phi_mode_kxt",
]
