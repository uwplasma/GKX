"""Scalar free-energy and field-energy diagnostics."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from spectraxgk.core.grid import SpectralGrid
from spectraxgk.core.velocity import gamma0
from spectraxgk.diagnostics.metadata import ArrayLike
from spectraxgk.diagnostics.weights import (
    _cached_hermitian_mode_weight,
    _hermitian_mode_weight,
    _species_array,
)
from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.params import LinearParams

__all__ = [
    "_masked_abs2",
    "distribution_free_energy",
    "electrostatic_field_energy",
    "electrostatic_field_energy_krehm",
    "magnetic_vector_potential_energy",
    "magnetic_vector_potential_energy_krehm",
    "runtime_energy_total",
    "total_energy",
]


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


def electrostatic_field_energy_krehm(
    phi: jnp.ndarray,
    grid: SpectralGrid,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    *,
    kx: jnp.ndarray | None = None,
    ky: jnp.ndarray | None = None,
    use_dealias: bool = True,
    compressed_real_fft: bool = False,
    wphi_scale: float = 1.0,
) -> jnp.ndarray:
    """Krehm electrostatic field-energy (Wphi) diagnostic."""

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
    if compressed_real_fft:
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
        active = fac[:, :, None] != 0.0
        weight = (1.0 - gam0)[:, :, None] * fac[:, :, None]
        contrib = 0.5 * (2.0 / rho2_s) * _masked_abs2(phi, active) * weight * vol
        wphi = wphi + jnp.sum(contrib)
    return wphi * jnp.asarray(wphi_scale, dtype=jnp.real(phi).dtype)


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


def magnetic_vector_potential_energy_krehm(
    apar: jnp.ndarray,
    grid: SpectralGrid,
    *,
    kx: jnp.ndarray | None = None,
    ky: jnp.ndarray | None = None,
    use_dealias: bool = True,
) -> jnp.ndarray:
    """Krehm magnetic vector-potential energy (Wapar) diagnostic."""

    kx_arr = grid.kx if kx is None else kx
    ky_arr = grid.ky if ky is None else ky
    kx = jnp.asarray(kx_arr)[None, :]
    ky = jnp.asarray(ky_arr)[:, None]
    kperp2 = kx * kx + ky * ky
    fac = _hermitian_mode_weight(grid, use_dealias=use_dealias)
    weight = fac[:, :, None]
    contrib = 0.5 * kperp2[:, :, None] * _masked_abs2(apar, weight != 0.0) * weight
    return jnp.sum(contrib)


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


