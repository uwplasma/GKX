"""Low-level Hermite/Laguerre operators for gyrokinetic terms."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectraxgk.basis import hermite_ladder_coeffs
from spectraxgk.terms.validation import _check_positive


def grad_z_periodic(
    f: jnp.ndarray, dz: float | jnp.ndarray | None = None, kz: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Spectral periodic derivative along the last axis."""

    if kz is None:
        if dz is None:
            raise ValueError("Either dz or kz must be provided")
        _check_positive(dz, "dz")
    n = f.shape[-1]
    if kz is None:
        dz_val = jnp.asarray(dz, dtype=jnp.real(f).dtype)
        kz = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dz_val)
    f_hat = jnp.fft.fft(f, axis=-1)
    df_hat = (1j * kz) * f_hat
    return jnp.fft.ifft(df_hat, axis=-1)


def _shift_kx_linked(
    f: jnp.ndarray,
    kx_link: jnp.ndarray,
    kx_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Shift along kx for each ky using precomputed link indices."""

    f_ky = jnp.moveaxis(f, -2, 0)
    kx_mask = kx_mask.astype(f.dtype)

    def _gather_ky(f_slice: jnp.ndarray, idx: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        gathered = jnp.take(f_slice, idx, axis=-1)
        return gathered * mask

    shifted = jax.vmap(_gather_ky, in_axes=(0, 0, 0))(f_ky, kx_link, kx_mask)
    return jnp.moveaxis(shifted, 0, -2)


def _grad_z_linked_fd(
    f: jnp.ndarray,
    dz: float | jnp.ndarray,
    kx_link_plus: jnp.ndarray,
    kx_link_minus: jnp.ndarray,
    kx_mask_plus: jnp.ndarray,
    kx_mask_minus: jnp.ndarray,
) -> jnp.ndarray:
    """Finite-difference z-derivative with twist-shift kx linking at the ends."""

    _check_positive(dz, "dz")
    dz_val = jnp.asarray(dz, dtype=jnp.real(f).dtype)
    f_roll_p1 = jnp.roll(f, -1, axis=-1)
    f_roll_m1 = jnp.roll(f, 1, axis=-1)
    f_z0 = f[..., 0]
    f_zm1 = f[..., -1]
    f_roll_p1 = f_roll_p1.at[..., -1].set(_shift_kx_linked(f_z0, kx_link_plus, kx_mask_plus))
    f_roll_m1 = f_roll_m1.at[..., 0].set(_shift_kx_linked(f_zm1, kx_link_minus, kx_mask_minus))
    return (f_roll_p1 - f_roll_m1) / (2.0 * dz_val)


def grad_z_linked_fft(
    f: jnp.ndarray,
    dz: float | jnp.ndarray,
    linked_indices: tuple[jnp.ndarray, ...],
    linked_kz: tuple[jnp.ndarray, ...],
    linked_inverse_permutation: jnp.ndarray | None = None,
    linked_full_cover: bool = False,
    linked_gather_map: jnp.ndarray | None = None,
    linked_gather_mask: jnp.ndarray | None = None,
    linked_use_gather: bool = False,
) -> jnp.ndarray:
    """Spectral z-derivative using GX-style linked FFT chains."""

    _check_positive(dz, "dz")
    if len(linked_indices) != len(linked_kz):
        raise ValueError("linked_indices and linked_kz must have the same length")
    if not linked_indices:
        raise ValueError("linked_indices cannot be empty for linked FFT derivative")

    Ny = f.shape[-3]
    Nx = f.shape[-2]
    Nz = f.shape[-1]
    lead_shape = f.shape[:-3]
    f_perm = jnp.swapaxes(f, -3, -2)
    f_flat = f_perm.reshape(*lead_shape, Nx * Ny, Nz)
    chain_updates: list[jnp.ndarray] = []
    chain_indices: list[jnp.ndarray] = []

    for idx_map, kz_link in zip(linked_indices, linked_kz):
        if idx_map.ndim != 2:
            raise ValueError("linked index maps must have shape (nChains, nLinks)")
        nChains, nLinks = idx_map.shape
        idx_flat = idx_map.reshape(-1)
        f_link = jnp.take(f_flat, idx_flat, axis=-2)
        f_link = f_link.reshape(*lead_shape, nChains, nLinks * Nz)
        f_hat = jnp.fft.fft(f_link, axis=-1)
        df_hat = (1j * kz_link) * f_hat
        df_link = jnp.fft.ifft(df_hat, axis=-1)
        df_link = df_link.reshape(*lead_shape, nChains * nLinks, Nz)
        chain_updates.append(df_link)
        chain_indices.append(idx_flat)

    if linked_use_gather or (linked_gather_map is not None and linked_gather_mask is not None):
        updates_cat = jnp.concatenate(chain_updates, axis=-2)
        gather_map = jnp.asarray(linked_gather_map, dtype=jnp.int32)
        gather_mask = jnp.asarray(linked_gather_mask, dtype=updates_cat.dtype)
        updates_full = jnp.take(updates_cat, gather_map, axis=-2)
        mask_shape = (1,) * (updates_full.ndim - 2) + (gather_mask.shape[0], 1)
        updates_full = updates_full * gather_mask.reshape(mask_shape)
        updates_full = updates_full.reshape(*lead_shape, Nx, Ny, Nz)
        return jnp.swapaxes(updates_full, -3, -2)

    if linked_full_cover:
        if linked_inverse_permutation is None:
            raise ValueError("linked_inverse_permutation required when linked_full_cover is True")
        updates_cat = jnp.concatenate(chain_updates, axis=-2)
        inv = jnp.asarray(linked_inverse_permutation, dtype=jnp.int32)
        df_flat = jnp.take(updates_cat, inv, axis=-2)
        df_full = df_flat.reshape(*lead_shape, Nx, Ny, Nz)
        return jnp.swapaxes(df_full, -3, -2)

    raise ValueError("linked FFT maps require gather_map/mask or full cover to avoid scatter")


def abs_z_linked_fft(
    f: jnp.ndarray,
    linked_indices: tuple[jnp.ndarray, ...],
    linked_kz: tuple[jnp.ndarray, ...],
    linked_inverse_permutation: jnp.ndarray | None = None,
    linked_full_cover: bool = False,
    linked_gather_map: jnp.ndarray | None = None,
    linked_gather_mask: jnp.ndarray | None = None,
    linked_use_gather: bool = False,
) -> jnp.ndarray:
    """Apply |kz| in linked-FFT space (GX abs_dz equivalent)."""

    if len(linked_indices) != len(linked_kz):
        raise ValueError("linked_indices and linked_kz must have the same length")
    if not linked_indices:
        raise ValueError("linked_indices cannot be empty for linked FFT operator")

    Ny = f.shape[-3]
    Nx = f.shape[-2]
    Nz = f.shape[-1]
    lead_shape = f.shape[:-3]
    f_perm = jnp.swapaxes(f, -3, -2)
    f_flat = f_perm.reshape(*lead_shape, Nx * Ny, Nz)
    chain_updates: list[jnp.ndarray] = []
    chain_indices: list[jnp.ndarray] = []

    for idx_map, kz_link in zip(linked_indices, linked_kz):
        if idx_map.ndim != 2:
            raise ValueError("linked index maps must have shape (nChains, nLinks)")
        nChains, nLinks = idx_map.shape
        idx_flat = idx_map.reshape(-1)
        f_link = jnp.take(f_flat, idx_flat, axis=-2)
        f_link = f_link.reshape(*lead_shape, nChains, nLinks * Nz)
        f_hat = jnp.fft.fft(f_link, axis=-1)
        df_hat = jnp.abs(kz_link) * f_hat
        df_link = jnp.fft.ifft(df_hat, axis=-1)
        df_link = df_link.reshape(*lead_shape, nChains * nLinks, Nz)
        chain_updates.append(df_link)
        chain_indices.append(idx_flat)

    if linked_use_gather or (linked_gather_map is not None and linked_gather_mask is not None):
        updates_cat = jnp.concatenate(chain_updates, axis=-2)
        gather_map = jnp.asarray(linked_gather_map, dtype=jnp.int32)
        gather_mask = jnp.asarray(linked_gather_mask, dtype=updates_cat.dtype)
        updates_full = jnp.take(updates_cat, gather_map, axis=-2)
        mask_shape = (1,) * (updates_full.ndim - 2) + (gather_mask.shape[0], 1)
        updates_full = updates_full * gather_mask.reshape(mask_shape)
        updates_full = updates_full.reshape(*lead_shape, Nx, Ny, Nz)
        return jnp.swapaxes(updates_full, -3, -2)

    if linked_full_cover:
        if linked_inverse_permutation is None:
            raise ValueError("linked_inverse_permutation required when linked_full_cover is True")
        updates_cat = jnp.concatenate(chain_updates, axis=-2)
        inv = jnp.asarray(linked_inverse_permutation, dtype=jnp.int32)
        df_flat = jnp.take(updates_cat, inv, axis=-2)
        df_full = df_flat.reshape(*lead_shape, Nx, Ny, Nz)
        return jnp.swapaxes(df_full, -3, -2)

    raise ValueError("linked FFT maps require gather_map/mask or full cover to avoid scatter")


def shift_axis(arr: jnp.ndarray, offset: int, axis: int) -> jnp.ndarray:
    """Shift an array along an axis with zero padding (non-periodic)."""

    axis = axis % arr.ndim
    if offset == 0:
        return arr
    axis_len = arr.shape[axis]
    if abs(offset) >= axis_len:
        return jnp.zeros_like(arr)
    if offset > 0:
        pad_shape = [arr.shape[i] if i != axis else offset for i in range(arr.ndim)]
        zeros = jnp.zeros(pad_shape, dtype=arr.dtype)
        arr_pad = jnp.concatenate([arr, zeros], axis=axis)
        return jax.lax.slice_in_dim(arr_pad, offset, offset + arr.shape[axis], axis=axis)
    pad_shape = [arr.shape[i] if i != axis else -offset for i in range(arr.ndim)]
    zeros = jnp.zeros(pad_shape, dtype=arr.dtype)
    arr_pad = jnp.concatenate([zeros, arr], axis=axis)
    return jax.lax.slice_in_dim(arr_pad, 0, arr.shape[axis], axis=axis)


def apply_hermite_v(G: jnp.ndarray) -> jnp.ndarray:
    """Multiply Hermite coefficients by v_parallel (ladder form)."""

    axis_m = -4
    Nm = G.shape[axis_m]
    sqrt_p, sqrt_m = hermite_ladder_coeffs(Nm - 1)
    sqrt_p = sqrt_p[:Nm]
    sqrt_m = sqrt_m[:Nm]

    pad = [(0, 0)] * G.ndim
    pad[axis_m] = (1, 1)
    G_pad = jnp.pad(G, pad)
    slc_plus = [slice(None)] * G.ndim
    slc_minus = [slice(None)] * G.ndim
    slc_plus[axis_m] = slice(2, None)
    slc_minus[axis_m] = slice(0, -2)
    G_plus = G_pad[tuple(slc_plus)]
    G_minus = G_pad[tuple(slc_minus)]
    shape = [1] * G.ndim
    shape[axis_m] = Nm
    sqrt_p = sqrt_p.reshape(shape)
    sqrt_m = sqrt_m.reshape(shape)
    return sqrt_p * G_plus + sqrt_m * G_minus


def apply_hermite_v2(G: jnp.ndarray) -> jnp.ndarray:
    """Multiply Hermite coefficients by v_parallel^2."""

    return apply_hermite_v(apply_hermite_v(G))


def apply_laguerre_x(G: jnp.ndarray) -> jnp.ndarray:
    """Multiply Laguerre coefficients by the perpendicular energy variable."""

    axis_l = -5
    Nl = G.shape[axis_l]
    l = jnp.arange(Nl)
    pad = [(0, 0)] * G.ndim
    pad[axis_l] = (1, 1)
    G_pad = jnp.pad(G, pad)
    slc_plus = [slice(None)] * G.ndim
    slc_minus = [slice(None)] * G.ndim
    slc_plus[axis_l] = slice(2, None)
    slc_minus[axis_l] = slice(0, -2)
    G_plus = G_pad[tuple(slc_plus)]
    G_minus = G_pad[tuple(slc_minus)]
    l_shape = [1] * G.ndim
    l_shape[axis_l] = Nl
    l_col = l.reshape(l_shape)
    return (
        (2.0 * l_col + 1.0) * G
        - (l_col + 1.0) * G_plus
        - l_col * G_minus
    )


def streaming_term(
    H: jnp.ndarray,
    kz: jnp.ndarray,
    vth: float | jnp.ndarray,
    sqrt_p: jnp.ndarray,
    sqrt_m: jnp.ndarray,
    *,
    dz: float | jnp.ndarray | None = None,
    kx_link_plus: jnp.ndarray | None = None,
    kx_link_minus: jnp.ndarray | None = None,
    kx_mask_plus: jnp.ndarray | None = None,
    kx_mask_minus: jnp.ndarray | None = None,
    linked_indices: tuple[jnp.ndarray, ...] | None = None,
    linked_kz: tuple[jnp.ndarray, ...] | None = None,
    linked_inverse_permutation: jnp.ndarray | None = None,
    linked_full_cover: bool = False,
    linked_gather_map: jnp.ndarray | None = None,
    linked_gather_mask: jnp.ndarray | None = None,
    linked_use_gather: bool = False,
    use_twist_shift: bool = False,
) -> jnp.ndarray:
    """Streaming term using Hermite ladder and real-space z derivative."""

    _check_positive(vth, "vth")
    if use_twist_shift:
        if dz is None:
            raise ValueError("dz must be provided for twist-shift boundaries")
        if linked_indices is not None and linked_kz is not None:
            dH_dz = grad_z_linked_fft(
                H,
                dz=dz,
                linked_indices=linked_indices,
                linked_kz=linked_kz,
                linked_inverse_permutation=linked_inverse_permutation,
                linked_full_cover=linked_full_cover,
                linked_gather_map=linked_gather_map,
                linked_gather_mask=linked_gather_mask,
                linked_use_gather=linked_use_gather,
            )
        else:
            if kx_link_plus is None or kx_link_minus is None:
                raise ValueError("kx_link arrays must be provided for twist-shift boundaries")
            if kx_mask_plus is None or kx_mask_minus is None:
                raise ValueError("kx_link masks must be provided for twist-shift boundaries")
            dH_dz = _grad_z_linked_fd(
                H,
                dz=dz,
                kx_link_plus=kx_link_plus,
                kx_link_minus=kx_link_minus,
                kx_mask_plus=kx_mask_plus,
                kx_mask_minus=kx_mask_minus,
            )
    else:
        dH_dz = grad_z_periodic(H, kz=kz)
    axis_m = -4
    pad = [(0, 0)] * H.ndim
    pad[axis_m] = (1, 1)
    H_pad = jnp.pad(dH_dz, pad)
    slc_plus = [slice(None)] * H.ndim
    slc_minus = [slice(None)] * H.ndim
    slc_plus[axis_m] = slice(2, None)
    slc_minus[axis_m] = slice(0, -2)
    H_plus = H_pad[tuple(slc_plus)]
    H_minus = H_pad[tuple(slc_minus)]
    return vth * (sqrt_p * H_plus + sqrt_m * H_minus)
