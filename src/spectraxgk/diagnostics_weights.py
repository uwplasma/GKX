"""Shared GX diagnostic weighting, mask, and gyroaverage helpers."""

from __future__ import annotations

import jax.numpy as jnp

from spectraxgk.geometry import (
    FluxTubeGeometryData,
    FluxTubeGeometryLike,
    ensure_flux_tube_geometry_data,
)
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import LinearCache
from spectraxgk.terms.operators import shift_axis


def gx_volume_factors(
    geom: FluxTubeGeometryLike, grid: SpectralGrid
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (vol_fac, flux_fac) matching GX's volume weights."""

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
    ell = jnp.arange(Nl, dtype=Jl_s.dtype)[None, :, None, None, None]
    Jl_m1 = shift_axis(Jl_s, -1, axis=1)
    Jl_p1 = shift_axis(Jl_s, 1, axis=1)
    JflrA = ell * Jl_m1 + 2.0 * ell * Jl_s + (ell + 1.0) * Jl_p1
    Jfac = 1.5 * Jl_s + JflrA
    return Jl_s, JlB_s, Jfac
