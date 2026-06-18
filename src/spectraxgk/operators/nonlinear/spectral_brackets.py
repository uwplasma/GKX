"""Pseudo-spectral field, bracket, and RHS micro-routes."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from spectraxgk.operators.nonlinear.spectral_state import _validate_spectral_state_shape


def _spectral_wave_numbers(ny: int, nx: int, dtype: Any) -> tuple[jax.Array, jax.Array]:
    ky = jnp.fft.fftfreq(ny, d=1.0 / float(ny)).astype(dtype)
    kx = jnp.fft.fftfreq(nx, d=1.0 / float(nx)).astype(dtype)
    return ky, kx


def _field_from_spectral_density(density_hat: jax.Array) -> jax.Array:
    ny, nx, _nz = (int(item) for item in density_hat.shape)
    real_dtype = jnp.real(density_hat).dtype
    ky, kx = _spectral_wave_numbers(ny, nx, real_dtype)
    kperp2 = ky[:, None, None] ** 2 + kx[None, :, None] ** 2
    phi_hat = density_hat / (1.0 + kperp2)
    return phi_hat.at[0, 0, :].set(0.0)


def _field_from_state(state_hat: jax.Array) -> jax.Array:
    density_hat = jnp.sum(state_hat[:, 0, :, :, :], axis=0)
    return _field_from_spectral_density(density_hat)


def _spectral_bracket(state_hat: jax.Array, phi_hat: jax.Array) -> jax.Array:
    _nl, _nm, ny, nx, _nz = _validate_spectral_state_shape(tuple(state_hat.shape))
    real_dtype = jnp.real(state_hat).dtype
    ky, kx = _spectral_wave_numbers(ny, nx, real_dtype)
    ky_state = ky[None, None, :, None, None]
    kx_state = kx[None, None, None, :, None]
    ky_field = ky[:, None, None]
    kx_field = kx[None, :, None]

    state_dx = jnp.fft.ifft2(1j * kx_state * state_hat, axes=(-3, -2))
    state_dy = jnp.fft.ifft2(1j * ky_state * state_hat, axes=(-3, -2))
    phi_dx = jnp.fft.ifft2(1j * kx_field * phi_hat, axes=(0, 1))
    phi_dy = jnp.fft.ifft2(1j * ky_field * phi_hat, axes=(0, 1))
    bracket_xy = (
        phi_dx[None, None, :, :, :] * state_dy - phi_dy[None, None, :, :, :] * state_dx
    )
    return jnp.fft.fft2(bracket_xy, axes=(-3, -2))


def _pencil_ifft2(arr: jax.Array, *, y_axis: int, x_axis: int) -> jax.Array:
    """Return a 2D inverse FFT through explicit x-then-y pencil stages."""

    y_axis = y_axis % arr.ndim
    x_axis = x_axis % arr.ndim
    x_transformed = jnp.fft.ifft(arr, axis=x_axis)
    transposed = jnp.swapaxes(x_transformed, y_axis, x_axis)
    y_transformed = jnp.fft.ifft(transposed, axis=x_axis)
    return jnp.swapaxes(y_transformed, y_axis, x_axis)


def _pencil_fft2(arr: jax.Array, *, y_axis: int, x_axis: int) -> jax.Array:
    """Return a 2D forward FFT through explicit x-then-y pencil stages."""

    y_axis = y_axis % arr.ndim
    x_axis = x_axis % arr.ndim
    x_transformed = jnp.fft.fft(arr, axis=x_axis)
    transposed = jnp.swapaxes(x_transformed, y_axis, x_axis)
    y_transformed = jnp.fft.fft(transposed, axis=x_axis)
    return jnp.swapaxes(y_transformed, y_axis, x_axis)


def _pencil_spectral_bracket(state_hat: jax.Array, phi_hat: jax.Array) -> jax.Array:
    """Return the pseudo-spectral bracket using pencil FFT staging.

    This function is the local algorithmic route that a distributed pencil FFT
    implementation should follow: stack derivative operands, transform through
    explicit axis-transpose stages, multiply in physical space, and transform
    the bracket back without first reconstructing logical output tiles.
    """

    _nl, _nm, ny, nx, _nz = _validate_spectral_state_shape(tuple(state_hat.shape))
    real_dtype = jnp.real(state_hat).dtype
    ky, kx = _spectral_wave_numbers(ny, nx, real_dtype)
    ky_state = ky[None, None, :, None, None]
    kx_state = kx[None, None, None, :, None]
    ky_field = ky[:, None, None]
    kx_field = kx[None, :, None]

    state_grad_hat = jnp.stack(
        [1j * kx_state * state_hat, 1j * ky_state * state_hat],
        axis=0,
    )
    state_grad_xy = _pencil_ifft2(state_grad_hat, y_axis=-3, x_axis=-2)
    state_dx = state_grad_xy[0]
    state_dy = state_grad_xy[1]

    field_grad_hat = jnp.stack(
        [1j * kx_field * phi_hat, 1j * ky_field * phi_hat],
        axis=0,
    )
    field_grad_xy = _pencil_ifft2(field_grad_hat, y_axis=1, x_axis=2)
    phi_dx = field_grad_xy[0]
    phi_dy = field_grad_xy[1]

    bracket_xy = (
        phi_dx[None, None, :, :, :] * state_dy - phi_dy[None, None, :, :, :] * state_dx
    )
    return _pencil_fft2(bracket_xy, y_axis=-3, x_axis=-2)


def _pencil_spectral_bracket_z_chunked(
    state_hat: jax.Array,
    phi_hat: jax.Array,
    *,
    z_chunk_size: int,
) -> jax.Array:
    """Return the pencil bracket by processing independent z slabs.

    The nonlinear bracket has no coupling along ``z`` inside this local
    pseudo-spectral micro-route. Chunking the local z extent therefore preserves
    the operator while reducing cuFFT batched-plan pressure on GPUs.
    """

    _nl, _nm, _ny, _nx, nz = _validate_spectral_state_shape(tuple(state_hat.shape))
    chunk_size = int(z_chunk_size)
    if chunk_size < 1:
        raise ValueError("z_chunk_size must be at least one")
    if chunk_size >= nz:
        return _pencil_spectral_bracket(state_hat, phi_hat)

    bracket_chunks: list[jax.Array] = []
    for start in range(0, nz, chunk_size):
        size = min(chunk_size, nz - start)
        state_chunk = jax.lax.dynamic_slice_in_dim(
            state_hat,
            start,
            size,
            axis=-1,
        )
        phi_chunk = jax.lax.dynamic_slice_in_dim(
            phi_hat,
            start,
            size,
            axis=-1,
        )
        bracket_chunks.append(_pencil_spectral_bracket(state_chunk, phi_chunk))
    return jnp.concatenate(bracket_chunks, axis=-1)


def _spectral_rhs_from_bracket(bracket_hat: jax.Array) -> jax.Array:
    """Return the ExB advection contribution used by the identity micro-route."""

    return -bracket_hat


def _serial_nonlinear_spectral_rhs(
    state_hat: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    field = _field_from_state(state_hat)
    bracket = _spectral_bracket(state_hat, field)
    rhs = _spectral_rhs_from_bracket(bracket)
    return field, bracket, rhs


def _pencil_nonlinear_spectral_rhs(
    state_hat: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    field = _field_from_state(state_hat)
    bracket = _pencil_spectral_bracket(state_hat, field)
    rhs = _spectral_rhs_from_bracket(bracket)
    return field, bracket, rhs


def _pencil_nonlinear_spectral_rhs_z_chunked(
    state_hat: jax.Array,
    *,
    z_chunk_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    field = _field_from_state(state_hat)
    bracket = _pencil_spectral_bracket_z_chunked(
        state_hat,
        field,
        z_chunk_size=z_chunk_size,
    )
    rhs = _spectral_rhs_from_bracket(bracket)
    return field, bracket, rhs


__all__ = [
    "_field_from_spectral_density",
    "_field_from_state",
    "_pencil_fft2",
    "_pencil_ifft2",
    "_pencil_nonlinear_spectral_rhs",
    "_pencil_nonlinear_spectral_rhs_z_chunked",
    "_pencil_spectral_bracket",
    "_pencil_spectral_bracket_z_chunked",
    "_serial_nonlinear_spectral_rhs",
    "_spectral_bracket",
    "_spectral_rhs_from_bracket",
    "_spectral_wave_numbers",
]
