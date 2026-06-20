"""State layout, spectral masks, and projection utilities for cETG."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from spectraxgk.core.grid import SpectralGrid


def _to_internal_state(G: jnp.ndarray) -> jnp.ndarray:
    G_arr = jnp.asarray(G)
    if G_arr.ndim == 4 and G_arr.shape[0] == 2:
        return G_arr
    if (
        G_arr.ndim != 6
        or G_arr.shape[0] != 1
        or G_arr.shape[1] != 2
        or G_arr.shape[2] != 1
    ):
        raise ValueError(
            "cETG state must have shape (1, 2, 1, Ny, Nx, Nz) or (2, Ny, Nx, Nz)"
        )
    return jnp.stack([G_arr[0, 0, 0], G_arr[0, 1, 0]], axis=0)


def _from_internal_state(G: jnp.ndarray) -> jnp.ndarray:
    G_arr = jnp.asarray(G)
    if G_arr.ndim != 4 or G_arr.shape[0] != 2:
        raise ValueError("internal cETG state must have shape (2, Ny, Nx, Nz)")
    return G_arr[None, :, None, :, :, :]


def _xy_mask(grid: SpectralGrid, dtype: jnp.dtype) -> jnp.ndarray:
    return jnp.asarray(grid.dealias_mask, dtype=dtype)[None, :, :, None]


def _kz_grid(grid: SpectralGrid) -> jnp.ndarray:
    z = np.asarray(grid.z, dtype=float)
    if z.size < 2:
        return jnp.zeros((z.size,), dtype=float)
    dz = float(z[1] - z[0])
    return 2.0 * jnp.pi * jnp.fft.fftfreq(z.size, d=dz)


def _kz_mask(grid: SpectralGrid, dtype: jnp.dtype, *, dealias_kz: bool) -> jnp.ndarray:
    if not dealias_kz:
        return jnp.ones((grid.z.size,), dtype=dtype)
    kz_frac = jnp.fft.fftfreq(grid.z.size)
    return jnp.asarray(jnp.abs(kz_frac) < (1.0 / 3.0), dtype=dtype)


def _apply_kz_filter(
    arr: jnp.ndarray, grid: SpectralGrid, *, dealias_kz: bool
) -> jnp.ndarray:
    if not dealias_kz or int(grid.z.size) <= 1:
        return arr
    arr_k = jnp.fft.fft(arr, axis=-1)
    mask = _kz_mask(grid, arr_k.real.dtype, dealias_kz=True)
    # The periodic-z dealias contract uses forward/inverse FFT pairs without
    # the 1/N rescale on the inverse, so the filtered field carries an Nz
    # factor.
    return jnp.fft.ifft(arr_k * mask, axis=-1) * jnp.asarray(
        float(grid.z.size), dtype=arr_k.real.dtype
    )


def _dz2(arr: jnp.ndarray, grid: SpectralGrid) -> jnp.ndarray:
    if int(grid.z.size) <= 1:
        return jnp.zeros_like(arr)
    kz = _kz_grid(grid).astype(jnp.real(arr).dtype)
    arr_k = jnp.fft.fft(arr, axis=-1)
    return jnp.fft.ifft(-(kz**2) * arr_k, axis=-1)


def _use_hermitian_reconstruction(
    grid: SpectralGrid, *, compressed_real_fft: bool
) -> bool:
    return bool(compressed_real_fft) and bool(
        np.any(np.asarray(grid.ky, dtype=float) < 0.0)
    )


def _project_state(
    G: jnp.ndarray,
    grid: SpectralGrid,
    *,
    compressed_real_fft: bool,
) -> jnp.ndarray:
    G_proj = jnp.asarray(G)
    G_proj = G_proj * _xy_mask(grid, jnp.real(G_proj).dtype)

    if not _use_hermitian_reconstruction(grid, compressed_real_fft=compressed_real_fft):
        return G_proj
    ny_full = int(grid.ky.size)
    nyc = ny_full // 2 + 1
    if nyc <= 2:
        return G_proj
    pos = G_proj[:, :nyc, :, :]
    neg_hi = nyc - 1 if (ny_full % 2 == 0) else nyc
    neg = jnp.conj(pos[:, 1:neg_hi, :, :])[:, ::-1, :, :]
    nx = int(grid.kx.size)
    if nx > 1:
        kx_neg = jnp.concatenate(
            [
                jnp.asarray([0], dtype=jnp.int32),
                jnp.arange(nx - 1, 0, -1, dtype=jnp.int32),
            ]
        )
        neg = neg[:, :, kx_neg, :]
    return jnp.concatenate([pos, neg], axis=1)


__all__ = [
    "_apply_kz_filter",
    "_dz2",
    "_from_internal_state",
    "_kz_grid",
    "_project_state",
    "_to_internal_state",
    "_xy_mask",
]
