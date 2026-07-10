"""Pseudo-spectral bracket kernels for nonlinear gyrokinetic terms."""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from spectraxgk.core.grid import real_fft_mesh

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


def _apply_mask_xy(field: jnp.ndarray, mask: jnp.ndarray | None) -> jnp.ndarray:
    if mask is None:
        return field
    real_dtype = jnp.real(jnp.empty((), dtype=field.dtype)).dtype
    mask_b = _broadcast_mask(jnp.asarray(mask, dtype=real_dtype), field.ndim)
    return field * mask_b


def _broadcast_to_G(x: jnp.ndarray, G: jnp.ndarray) -> jnp.ndarray:
    if x.ndim == G.ndim:
        return x
    if x.ndim == G.ndim - 1:
        return jnp.expand_dims(x, axis=-4)
    if x.ndim == 3:
        shape = (1,) * (G.ndim - 3) + x.shape
        return jnp.reshape(x, shape)
    if x.ndim < G.ndim:
        shape = (1,) * (G.ndim - x.ndim) + x.shape
        return jnp.reshape(x, shape)
    return x


def _stack_fields(G_hat: jnp.ndarray, fields: Sequence[jnp.ndarray]) -> jnp.ndarray:
    stacked = []
    for field in fields:
        stacked.append(_broadcast_to_G(jnp.asarray(field), G_hat))
    return jnp.stack(stacked, axis=0)


def _spectral_bracket_multi_real_fft(
    G_hat: jnp.ndarray,
    chi_hat_stack: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
    fft_norm: float | None = None,
) -> jnp.ndarray:
    complex_dtype = jnp.result_type(G_hat, chi_hat_stack, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=complex_dtype)).dtype
    imag = jnp.asarray(1j, dtype=complex_dtype)

    G_hat = jnp.asarray(G_hat, dtype=complex_dtype)
    chi_hat_stack = jnp.asarray(chi_hat_stack, dtype=complex_dtype)

    mask = jnp.asarray(dealias_mask, dtype=real_dtype)
    kx = jnp.asarray(kx_grid, dtype=real_dtype)
    ky = jnp.asarray(ky_grid, dtype=real_dtype)
    if fft_norm is None:
        fft_norm_val = float(ky_grid.shape[0] * ky_grid.shape[1])
    else:
        fft_norm_val = float(fft_norm)
    ifft_scale = jnp.asarray(fft_norm_val, dtype=real_dtype)
    fft_scale = jnp.asarray(1.0 / fft_norm_val, dtype=real_dtype)

    ny_full = int(ky.shape[0])
    _, ky_vals, kx_nyc, ky_nyc = real_fft_mesh(kx, ky)
    nyc = int(ky_vals.shape[0])

    G_nyc = G_hat[..., :nyc, :, :]
    chi_nyc = chi_hat_stack[..., :nyc, :, :]
    axes = (-2, -3)
    kx_b = _broadcast_grid(kx_nyc, G_nyc.ndim)
    ky_b = _broadcast_grid(ky_nyc, G_nyc.ndim)
    kx_chi = _broadcast_grid(kx_nyc, chi_nyc.ndim)
    ky_chi = _broadcast_grid(ky_nyc, chi_nyc.ndim)
    gradients = jnp.stack(
        [
            imag * kx_b * G_nyc,
            imag * ky_b * G_nyc,
            imag * kx_chi * chi_nyc,
            imag * ky_chi * chi_nyc,
        ],
        axis=0,
    )
    gradients = (
        jnp.fft.irfft2(gradients, s=(kx.shape[1], ny_full), axes=axes)
        * ifft_scale
    )
    dG_dx, dG_dy, dchi_dx, dchi_dy = gradients

    bracket = dG_dx[None, ...] * dchi_dy - dG_dy[None, ...] * dchi_dx

    bracket_hat_nyc = jnp.fft.rfft2(bracket, axes=axes) * fft_scale
    mask_nyc = mask[:nyc, :]
    bracket_hat_nyc = bracket_hat_nyc * _broadcast_mask(mask_nyc, bracket_hat_nyc.ndim)
    if ny_full > 1:
        neg_hi = nyc - 1 if (ny_full % 2 == 0) else nyc
        neg = jnp.conj(bracket_hat_nyc[..., 1:neg_hi, :, :])
        neg = neg[..., ::-1, :, :]
        if kx.shape[1] > 1:
            kx_neg = jnp.concatenate(
                [jnp.asarray([0], dtype=jnp.int32), jnp.arange(kx.shape[1] - 1, 0, -1, dtype=jnp.int32)]
            )
            neg = neg[..., kx_neg, :]
        bracket_hat = jnp.concatenate([bracket_hat_nyc, neg], axis=-3)
    else:
        bracket_hat = bracket_hat_nyc
    return jnp.asarray(kxfac, dtype=real_dtype) * bracket_hat


def _spectral_bracket_multi_full(
    G_hat: jnp.ndarray,
    chi_hat_stack: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
    fft_norm: float | None = None,
) -> jnp.ndarray:
    complex_dtype = jnp.result_type(G_hat, chi_hat_stack, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=complex_dtype)).dtype
    imag = jnp.asarray(1j, dtype=complex_dtype)

    G_hat = jnp.asarray(G_hat, dtype=complex_dtype)
    chi_hat_stack = jnp.asarray(chi_hat_stack, dtype=complex_dtype)

    mask = jnp.asarray(dealias_mask, dtype=real_dtype)
    kx = jnp.asarray(kx_grid, dtype=real_dtype)
    ky = jnp.asarray(ky_grid, dtype=real_dtype)
    if fft_norm is None:
        fft_norm_val = float(ky_grid.shape[0] * ky_grid.shape[1])
    else:
        fft_norm_val = float(fft_norm)
    ifft_scale = jnp.asarray(fft_norm_val, dtype=real_dtype)
    fft_scale = jnp.asarray(1.0 / fft_norm_val, dtype=real_dtype)

    kx_b = _broadcast_grid(kx, G_hat.ndim)
    ky_b = _broadcast_grid(ky, G_hat.ndim)
    grad_G = jnp.stack([imag * kx_b * G_hat, imag * ky_b * G_hat], axis=0)
    grad_G = _ifft2_xy(grad_G) * ifft_scale
    dG_dx = grad_G[0]
    dG_dy = grad_G[1]

    kx_chi = _broadcast_grid(kx, chi_hat_stack.ndim)
    ky_chi = _broadcast_grid(ky, chi_hat_stack.ndim)
    grad_chi = jnp.stack([imag * kx_chi * chi_hat_stack, imag * ky_chi * chi_hat_stack], axis=0)
    grad_chi = _ifft2_xy(grad_chi) * ifft_scale
    dchi_dx = grad_chi[0]
    dchi_dy = grad_chi[1]

    bracket = dG_dx[None, ...] * dchi_dy - dG_dy[None, ...] * dchi_dx

    bracket_hat = _fft2_xy(bracket) * fft_scale
    bracket_hat = bracket_hat * _broadcast_mask(mask, bracket_hat.ndim)
    return jnp.asarray(kxfac, dtype=real_dtype) * bracket_hat


def _spectral_bracket(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
    fft_norm: float | None = None,
    compressed_real_fft: bool = True,
) -> jnp.ndarray:
    if compressed_real_fft:
        return _spectral_bracket_real_fft(
            G_hat,
            chi_hat,
            kx_grid=kx_grid,
            ky_grid=ky_grid,
            dealias_mask=dealias_mask,
            kxfac=kxfac,
            fft_norm=fft_norm,
        )
    return _spectral_bracket_full(
        G_hat,
        chi_hat,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=kxfac,
        fft_norm=fft_norm,
    )


def _spectral_bracket_real_fft(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
    fft_norm: float | None = None,
) -> jnp.ndarray:
    complex_dtype = jnp.result_type(G_hat, chi_hat, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=complex_dtype)).dtype
    imag = jnp.asarray(1j, dtype=complex_dtype)

    G_hat = jnp.asarray(G_hat, dtype=complex_dtype)
    chi_hat = _broadcast_to_G(jnp.asarray(chi_hat, dtype=complex_dtype), G_hat)

    mask = jnp.asarray(dealias_mask, dtype=real_dtype)
    kx = jnp.asarray(kx_grid, dtype=real_dtype)
    ky = jnp.asarray(ky_grid, dtype=real_dtype)
    if fft_norm is None:
        fft_norm_val = float(ky_grid.shape[0] * ky_grid.shape[1])
    else:
        fft_norm_val = float(fft_norm)
    ifft_scale = jnp.asarray(fft_norm_val, dtype=real_dtype)
    fft_scale = jnp.asarray(1.0 / fft_norm_val, dtype=real_dtype)

    ny_full = int(ky.shape[0])
    _, ky_vals, kx_nyc, ky_nyc = real_fft_mesh(kx, ky)
    nyc = int(ky_vals.shape[0])

    G_nyc = G_hat[..., :nyc, :, :]
    chi_nyc = chi_hat[..., :nyc, :, :]
    axes = (-2, -3)
    kx_b = _broadcast_grid(kx_nyc, G_nyc.ndim)
    ky_b = _broadcast_grid(ky_nyc, G_nyc.ndim)
    grad_G = jnp.stack([imag * kx_b * G_nyc, imag * ky_b * G_nyc], axis=0)
    grad_G = jnp.fft.irfft2(grad_G, s=(kx.shape[1], ny_full), axes=axes) * ifft_scale
    dG_dx = grad_G[0]
    dG_dy = grad_G[1]

    kx_chi = _broadcast_grid(kx_nyc, chi_nyc.ndim)
    ky_chi = _broadcast_grid(ky_nyc, chi_nyc.ndim)
    grad_chi = jnp.stack([imag * kx_chi * chi_nyc, imag * ky_chi * chi_nyc], axis=0)
    grad_chi = jnp.fft.irfft2(grad_chi, s=(kx.shape[1], ny_full), axes=axes) * ifft_scale
    dchi_dx = grad_chi[0]
    dchi_dy = grad_chi[1]

    bracket = dG_dx * dchi_dy - dG_dy * dchi_dx

    bracket_hat_nyc = jnp.fft.rfft2(bracket, axes=axes) * fft_scale
    mask_nyc = mask[:nyc, :]
    bracket_hat_nyc = bracket_hat_nyc * _broadcast_mask(mask_nyc, bracket_hat_nyc.ndim)
    if ny_full > 1:
        neg_hi = nyc - 1 if (ny_full % 2 == 0) else nyc
        neg = jnp.conj(bracket_hat_nyc[..., 1:neg_hi, :, :])
        neg = neg[..., ::-1, :, :]
        if kx.shape[1] > 1:
            kx_neg = jnp.concatenate(
                [jnp.asarray([0], dtype=jnp.int32), jnp.arange(kx.shape[1] - 1, 0, -1, dtype=jnp.int32)]
            )
            neg = neg[..., kx_neg, :]
        bracket_hat = jnp.concatenate([bracket_hat_nyc, neg], axis=-3)
    else:
        bracket_hat = bracket_hat_nyc
    return jnp.asarray(kxfac, dtype=real_dtype) * bracket_hat


def _spectral_bracket_full(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
    fft_norm: float | None = None,
) -> jnp.ndarray:
    complex_dtype = jnp.result_type(G_hat, chi_hat, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=complex_dtype)).dtype
    imag = jnp.asarray(1j, dtype=complex_dtype)

    G_hat = jnp.asarray(G_hat, dtype=complex_dtype)
    chi_hat = _broadcast_to_G(jnp.asarray(chi_hat, dtype=complex_dtype), G_hat)

    mask = jnp.asarray(dealias_mask, dtype=real_dtype)
    kx = jnp.asarray(kx_grid, dtype=real_dtype)
    ky = jnp.asarray(ky_grid, dtype=real_dtype)
    if fft_norm is None:
        fft_norm_val = float(ky_grid.shape[0] * ky_grid.shape[1])
    else:
        fft_norm_val = float(fft_norm)
    ifft_scale = jnp.asarray(fft_norm_val, dtype=real_dtype)
    fft_scale = jnp.asarray(1.0 / fft_norm_val, dtype=real_dtype)

    kx_b = _broadcast_grid(kx, G_hat.ndim)
    ky_b = _broadcast_grid(ky, G_hat.ndim)
    kx_chi = _broadcast_grid(kx, chi_hat.ndim)
    ky_chi = _broadcast_grid(ky, chi_hat.ndim)
    gradients = jnp.stack(
        [
            imag * kx_b * G_hat,
            imag * ky_b * G_hat,
            imag * kx_chi * chi_hat,
            imag * ky_chi * chi_hat,
        ],
        axis=0,
    )
    dG_dx, dG_dy, dchi_dx, dchi_dy = _ifft2_xy(gradients) * ifft_scale

    bracket = dG_dx * dchi_dy - dG_dy * dchi_dx

    bracket_hat = _fft2_xy(bracket) * fft_scale
    bracket_hat = bracket_hat * _broadcast_mask(mask, bracket_hat.ndim)
    return jnp.asarray(kxfac, dtype=real_dtype) * bracket_hat


def _spectral_bracket_multi(
    G_hat: jnp.ndarray,
    chi_hat_stack: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
    fft_norm: float | None = None,
    compressed_real_fft: bool = True,
) -> jnp.ndarray:
    if compressed_real_fft:
        return _spectral_bracket_multi_real_fft(
            G_hat,
            chi_hat_stack,
            kx_grid=kx_grid,
            ky_grid=ky_grid,
            dealias_mask=dealias_mask,
            kxfac=kxfac,
            fft_norm=fft_norm,
        )
    return _spectral_bracket_multi_full(
        G_hat,
        chi_hat_stack,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=kxfac,
        fft_norm=fft_norm,
    )
