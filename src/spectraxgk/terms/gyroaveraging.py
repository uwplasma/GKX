"""Laguerre-space gyroaveraging helpers for nonlinear gyrokinetic terms."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectraxgk.core.velocity import bessel_j0, bessel_j1


def _laguerre_to_grid(G: jnp.ndarray, laguerre_to_grid: jnp.ndarray) -> jnp.ndarray:
    """Transform Laguerre moments to the muB quadrature grid."""
    G = jnp.asarray(G)
    laguerre_to_grid = jnp.asarray(laguerre_to_grid)
    return jnp.einsum(
        "slmyxz,lj->sjmyxz",
        G,
        laguerre_to_grid,
        precision=jax.lax.Precision.HIGHEST,
    )


def _laguerre_to_spectral(
    g_mu: jnp.ndarray, laguerre_to_spectral: jnp.ndarray
) -> jnp.ndarray:
    """Transform muB quadrature-grid values back to Laguerre moments."""
    g_mu = jnp.asarray(g_mu)
    laguerre_to_spectral = jnp.asarray(laguerre_to_spectral)
    return jnp.einsum(
        "sjmyxz,jl->slmyxz",
        g_mu,
        laguerre_to_spectral,
        precision=jax.lax.Precision.HIGHEST,
    )


def _laguerre_j0_field(
    field: jnp.ndarray,
    b: jnp.ndarray,
    roots: jnp.ndarray,
    factor: float,
) -> jnp.ndarray:
    """Apply J0(field) on the Laguerre quadrature grid."""
    b = jnp.asarray(b)
    roots = jnp.asarray(roots)
    field = jnp.asarray(field)
    if b.ndim == 3:
        b = b[None, ...]
    if roots.ndim == 0:
        roots = roots[None]
    alpha = jnp.sqrt(
        jnp.maximum(0.0, 2.0 * roots[None, :, None, None, None] * b[:, None, ...])
    )
    j0 = bessel_j0(alpha)
    field_b = field[None, None, ...]
    return j0 * field_b * jnp.asarray(factor, dtype=field.dtype)


def _laguerre_j0_field_precomputed(
    field: jnp.ndarray,
    j0: jnp.ndarray,
    factor: float,
) -> jnp.ndarray:
    field = jnp.asarray(field)
    field_b = field[None, None, ...]
    return j0 * field_b * jnp.asarray(factor, dtype=field.dtype)


def _laguerre_bpar_correction(
    bpar: jnp.ndarray,
    b: jnp.ndarray,
    roots: jnp.ndarray,
    tz: jnp.ndarray,
    factor: float,
) -> jnp.ndarray:
    """Return the bpar correction term on the Laguerre quadrature grid."""
    b = jnp.asarray(b)
    roots = jnp.asarray(roots)
    bpar = jnp.asarray(bpar)
    if b.ndim == 3:
        b = b[None, ...]
    if roots.ndim == 0:
        roots = roots[None]
    tz_arr = jnp.asarray(tz)
    if tz_arr.ndim == 0:
        tz_arr = tz_arr[None]
    alpha = jnp.sqrt(
        jnp.maximum(0.0, 2.0 * roots[None, :, None, None, None] * b[:, None, ...])
    )
    j1 = bessel_j1(alpha)
    j1_over_alpha = jnp.where(alpha < 1.0e-8, 0.5, j1 / alpha)
    coeff = (
        tz_arr[:, None, None, None, None]
        * 2.0
        * roots[None, :, None, None, None]
        * j1_over_alpha
    )
    bpar_b = bpar[None, None, ...]
    return coeff * bpar_b * jnp.asarray(factor, dtype=bpar.dtype)


def _laguerre_bpar_correction_precomputed(
    bpar: jnp.ndarray,
    j1_over_alpha: jnp.ndarray,
    roots: jnp.ndarray,
    tz: jnp.ndarray,
    factor: float,
) -> jnp.ndarray:
    bpar = jnp.asarray(bpar)
    tz_arr = jnp.asarray(tz)
    if tz_arr.ndim == 0:
        tz_arr = tz_arr[None]
    coeff = (
        tz_arr[:, None, None, None, None]
        * 2.0
        * roots[None, :, None, None, None]
        * j1_over_alpha
    )
    bpar_b = bpar[None, None, ...]
    return coeff * bpar_b * jnp.asarray(factor, dtype=bpar.dtype)

