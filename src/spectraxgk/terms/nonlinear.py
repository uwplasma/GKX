"""Nonlinear E×B term placeholders (to be implemented)."""

from __future__ import annotations

import jax.numpy as jnp


def _fft2_xy(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.fft2(x, axes=(-3, -2))


def _ifft2_xy(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.ifft2(x, axes=(-3, -2))


def _broadcast_mask(mask: jnp.ndarray, ndim: int) -> jnp.ndarray:
    shape = (1,) * (ndim - 3) + mask.shape + (1,)
    return jnp.reshape(mask, shape)


def _broadcast_grid(grid: jnp.ndarray, ndim: int) -> jnp.ndarray:
    shape = (1,) * (ndim - 3) + grid.shape + (1,)
    return jnp.reshape(grid, shape)


def exb_nonlinear_contribution(
    G: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    """Return the nonlinear E×B contribution using a pseudospectral bracket."""

    complex_dtype = jnp.result_type(G, phi, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=complex_dtype)).dtype
    imag = jnp.asarray(1j, dtype=complex_dtype)

    G_hat = jnp.asarray(G, dtype=complex_dtype)
    phi_hat = jnp.asarray(phi, dtype=complex_dtype)

    mask = jnp.asarray(dealias_mask, dtype=real_dtype)
    G_hat = G_hat * _broadcast_mask(mask, G_hat.ndim)
    phi_hat = phi_hat * _broadcast_mask(mask, phi_hat.ndim)

    kx = jnp.asarray(kx_grid, dtype=real_dtype)
    ky = jnp.asarray(ky_grid, dtype=real_dtype)
    kx_b = _broadcast_grid(kx, G_hat.ndim)
    ky_b = _broadcast_grid(ky, G_hat.ndim)
    kx_phi = _broadcast_grid(kx, phi_hat.ndim)
    ky_phi = _broadcast_grid(ky, phi_hat.ndim)

    dphi_dx = _ifft2_xy(imag * kx_phi * phi_hat)
    dphi_dy = _ifft2_xy(imag * ky_phi * phi_hat)
    dG_dx = _ifft2_xy(imag * kx_b * G_hat)
    dG_dy = _ifft2_xy(imag * ky_b * G_hat)

    phi_shape = (1,) * (G_hat.ndim - 3) + dphi_dx.shape
    dphi_dx_b = jnp.reshape(dphi_dx, phi_shape)
    dphi_dy_b = jnp.reshape(dphi_dy, phi_shape)
    bracket = dphi_dx_b * dG_dy - dphi_dy_b * dG_dx

    bracket_hat = _fft2_xy(bracket)
    bracket_hat = bracket_hat * _broadcast_mask(mask, bracket_hat.ndim)

    return -jnp.asarray(weight, dtype=real_dtype) * bracket_hat


def placeholder_nonlinear_contribution(
    G: jnp.ndarray,
    *,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    """Return a zero nonlinear contribution to validate IO shapes."""

    return jnp.zeros_like(G) * weight
