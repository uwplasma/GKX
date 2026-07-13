"""Pseudo-spectral bracket kernels for nonlinear gyrokinetic terms."""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from spectraxgk.core.grid import real_fft_mesh


def _fft2_xy(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.fft2(x, axes=(-3, -2))


def _ifft2_xy(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.ifft2(x, axes=(-3, -2))


def _ifft2_xy_with_radial_phase(
    x: jnp.ndarray, radial_phase: jnp.ndarray
) -> jnp.ndarray:
    """Transform a shearing wave after applying its fractional radial phase."""

    radial = jnp.fft.ifft(x, axis=-2)
    radial *= _broadcast_grid(radial_phase, radial.ndim)
    return jnp.fft.ifft(radial, axis=-3)


def _fft2_xy_remove_radial_phase(
    x: jnp.ndarray, radial_phase: jnp.ndarray
) -> jnp.ndarray:
    """Return a physical field to its nearest-cell shearing-wave coefficients."""

    binormal = jnp.fft.fft(x, axis=-3)
    binormal *= _broadcast_grid(jnp.conj(radial_phase), binormal.ndim)
    return jnp.fft.fft(binormal, axis=-2)


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
    return jnp.stack(
        [_broadcast_to_G(jnp.asarray(field), G_hat) for field in fields], axis=0
    )


def _fft_scales(
    ky_grid: jnp.ndarray,
    *,
    real_dtype: jnp.dtype,
    fft_norm: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    norm = (
        float(ky_grid.shape[0] * ky_grid.shape[1])
        if fft_norm is None
        else float(fft_norm)
    )
    return (
        jnp.asarray(norm, dtype=real_dtype),
        jnp.asarray(1.0 / norm, dtype=real_dtype),
    )


def _complete_hermitian_ky(
    positive_ky: jnp.ndarray,
    *,
    ny_full: int,
    nx: int,
) -> jnp.ndarray:
    if ny_full <= 1:
        return positive_ky
    nyc = int(positive_ky.shape[-3])
    neg_hi = nyc - 1 if ny_full % 2 == 0 else nyc
    negative_ky = jnp.conj(positive_ky[..., 1:neg_hi, :, :])
    negative_ky = negative_ky[..., ::-1, :, :]
    if nx > 1:
        conjugate_kx = jnp.concatenate(
            [
                jnp.asarray([0], dtype=jnp.int32),
                jnp.arange(nx - 1, 0, -1, dtype=jnp.int32),
            ]
        )
        negative_ky = negative_ky[..., conjugate_kx, :]
    return jnp.concatenate([positive_ky, negative_ky], axis=-3)


def _spectral_bracket_real_fft_core(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
    fft_norm: float | None = None,
    radial_phase: jnp.ndarray | None = None,
    multiple_fields: bool,
) -> jnp.ndarray:
    if radial_phase is not None:
        raise NotImplementedError(
            "radial shearing phases currently require compressed_real_fft=False"
        )
    complex_dtype = jnp.result_type(G_hat, chi_hat, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=complex_dtype)).dtype
    imag = jnp.asarray(1j, dtype=complex_dtype)
    G_hat = jnp.asarray(G_hat, dtype=complex_dtype)
    chi_hat = jnp.asarray(chi_hat, dtype=complex_dtype)
    mask = jnp.asarray(dealias_mask, dtype=real_dtype)
    kx = jnp.asarray(kx_grid, dtype=real_dtype)
    ky = jnp.asarray(ky_grid, dtype=real_dtype)
    ifft_scale, fft_scale = _fft_scales(
        ky_grid, real_dtype=real_dtype, fft_norm=fft_norm
    )

    ny_full = int(ky.shape[0])
    _, ky_values, kx_nyc, ky_nyc = real_fft_mesh(kx, ky)
    nyc = int(ky_values.shape[0])
    G_nyc = G_hat[..., :nyc, :, :]
    chi_nyc = chi_hat[..., :nyc, :, :]
    axes = (-2, -3)

    kx_b = _broadcast_grid(kx_nyc, G_nyc.ndim)
    ky_b = _broadcast_grid(ky_nyc, G_nyc.ndim)
    grad_G = jnp.stack([imag * kx_b * G_nyc, imag * ky_b * G_nyc], axis=0)
    grad_G = jnp.fft.irfft2(grad_G, s=(kx.shape[1], ny_full), axes=axes) * ifft_scale
    dG_dx, dG_dy = grad_G

    kx_chi = _broadcast_grid(kx_nyc, chi_nyc.ndim)
    ky_chi = _broadcast_grid(ky_nyc, chi_nyc.ndim)
    grad_chi = jnp.stack([imag * kx_chi * chi_nyc, imag * ky_chi * chi_nyc], axis=0)
    grad_chi = (
        jnp.fft.irfft2(grad_chi, s=(kx.shape[1], ny_full), axes=axes) * ifft_scale
    )
    dchi_dx, dchi_dy = grad_chi

    if multiple_fields:
        bracket = dG_dx[None, ...] * dchi_dy - dG_dy[None, ...] * dchi_dx
    else:
        bracket = dG_dx * dchi_dy - dG_dy * dchi_dx
    positive_ky = jnp.fft.rfft2(bracket, axes=axes) * fft_scale
    positive_ky *= _broadcast_mask(mask[:nyc, :], positive_ky.ndim)
    bracket_hat = _complete_hermitian_ky(
        positive_ky, ny_full=ny_full, nx=int(kx.shape[1])
    )
    return jnp.asarray(kxfac, dtype=real_dtype) * bracket_hat


def _spectral_bracket_full_core(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
    fft_norm: float | None = None,
    radial_phase: jnp.ndarray | None = None,
    multiple_fields: bool,
) -> jnp.ndarray:
    complex_dtype = jnp.result_type(G_hat, chi_hat, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=complex_dtype)).dtype
    imag = jnp.asarray(1j, dtype=complex_dtype)
    G_hat = jnp.asarray(G_hat, dtype=complex_dtype)
    chi_hat = jnp.asarray(chi_hat, dtype=complex_dtype)
    mask = jnp.asarray(dealias_mask, dtype=real_dtype)
    kx = jnp.asarray(kx_grid, dtype=real_dtype)
    ky = jnp.asarray(ky_grid, dtype=real_dtype)
    ifft_scale, fft_scale = _fft_scales(
        ky_grid, real_dtype=real_dtype, fft_norm=fft_norm
    )
    phase = None
    if radial_phase is not None:
        phase = jnp.asarray(radial_phase, dtype=complex_dtype)
        if tuple(phase.shape) != tuple(kx.shape):
            raise ValueError("radial_phase must have shape (ky, x)")

    kx_b = _broadcast_grid(kx, G_hat.ndim)
    ky_b = _broadcast_grid(ky, G_hat.ndim)
    grad_G = jnp.stack([imag * kx_b * G_hat, imag * ky_b * G_hat], axis=0)
    if phase is None:
        dG_dx, dG_dy = _ifft2_xy(grad_G) * ifft_scale
    else:
        dG_dx, dG_dy = _ifft2_xy_with_radial_phase(grad_G, phase) * ifft_scale

    kx_chi = _broadcast_grid(kx, chi_hat.ndim)
    ky_chi = _broadcast_grid(ky, chi_hat.ndim)
    grad_chi = jnp.stack([imag * kx_chi * chi_hat, imag * ky_chi * chi_hat], axis=0)
    if phase is None:
        dchi_dx, dchi_dy = _ifft2_xy(grad_chi) * ifft_scale
    else:
        dchi_dx, dchi_dy = _ifft2_xy_with_radial_phase(grad_chi, phase) * ifft_scale

    if multiple_fields:
        bracket = dG_dx[None, ...] * dchi_dy - dG_dy[None, ...] * dchi_dx
    else:
        bracket = dG_dx * dchi_dy - dG_dy * dchi_dx
    bracket_hat = (
        _fft2_xy(bracket)
        if phase is None
        else _fft2_xy_remove_radial_phase(bracket, phase)
    ) * fft_scale
    bracket_hat *= _broadcast_mask(mask, bracket_hat.ndim)
    return jnp.asarray(kxfac, dtype=real_dtype) * bracket_hat


def _spectral_bracket_multi_real_fft(
    G_hat: jnp.ndarray,
    chi_hat_stack: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    return _spectral_bracket_real_fft_core(
        G_hat, chi_hat_stack, multiple_fields=True, **kwargs
    )


def _spectral_bracket_multi_full(
    G_hat: jnp.ndarray,
    chi_hat_stack: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    return _spectral_bracket_full_core(
        G_hat, chi_hat_stack, multiple_fields=True, **kwargs
    )


def _spectral_bracket_real_fft(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    return _spectral_bracket_real_fft_core(
        G_hat,
        _broadcast_to_G(jnp.asarray(chi_hat), G_hat),
        multiple_fields=False,
        **kwargs,
    )


def _spectral_bracket_full(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    **kwargs,
) -> jnp.ndarray:
    return _spectral_bracket_full_core(
        G_hat,
        _broadcast_to_G(jnp.asarray(chi_hat), G_hat),
        multiple_fields=False,
        **kwargs,
    )


def _spectral_bracket(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    *,
    compressed_real_fft: bool = True,
    **kwargs,
) -> jnp.ndarray:
    kernel = (
        _spectral_bracket_real_fft if compressed_real_fft else _spectral_bracket_full
    )
    return kernel(G_hat, chi_hat, **kwargs)


def _spectral_bracket_multi(
    G_hat: jnp.ndarray,
    chi_hat_stack: jnp.ndarray,
    *,
    compressed_real_fft: bool = True,
    **kwargs,
) -> jnp.ndarray:
    kernel = (
        _spectral_bracket_multi_real_fft
        if compressed_real_fft
        else _spectral_bracket_multi_full
    )
    return kernel(G_hat, chi_hat_stack, **kwargs)
