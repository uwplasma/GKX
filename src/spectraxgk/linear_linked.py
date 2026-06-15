"""Linked-boundary FFT maps and damping profiles for linear operators."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = [
    "_build_linked_end_damping_profile",
    "_build_linked_fft_maps",
    "_signed_to_index",
]


def _signed_to_index(idx: int, n: int) -> int:
    half = (n + 1) // 2
    if 0 <= idx < half:
        return idx
    if half <= idx + n < n:
        return idx + n
    return -1


def _build_linked_fft_maps(
    kx: np.ndarray,
    ky: np.ndarray,
    y0: float,
    jtwist: int,
    dz: float,
    nz: int,
    real_dtype: jnp.dtype,
    ky_mode: np.ndarray | None = None,
) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]:
    """Construct linked-chain FFT index maps for the parallel derivative."""

    ny = ky.size
    nx = kx.size
    if ky_mode is not None:
        naky = int(np.asarray(ky_mode, dtype=int).reshape(-1).size)
    else:
        naky = 1 + (ny - 1) // 3
    if nx < 4:
        nakx = nx
    else:
        nakx = 1 + 2 * ((nx - 1) // 3)
    if nakx <= 0 or naky <= 0:
        return (), ()

    nshift = nx - nakx
    idx_left = -np.ones((naky, nakx), dtype=int)
    idx_right = -np.ones((naky, nakx), dtype=int)

    ky_mode_arr: np.ndarray | None = None
    if ky_mode is not None:
        ky_mode_arr = np.asarray(ky_mode, dtype=int).reshape(-1)
    for idx in range(nakx):
        idx0 = idx if idx < (nakx + 1) // 2 else idx - nakx
        for idy in range(naky):
            idy_mode = int(ky_mode_arr[idy]) if ky_mode_arr is not None else idy
            if idy_mode == 0:
                idx_l = idx0
                idx_r = idx0
            else:
                idx_l = idx0 + idy_mode * jtwist
                idx_r = idx0 - idy_mode * jtwist
            idx_left[idy, idx] = _signed_to_index(idx_l, nakx)
            idx_right[idy, idx] = _signed_to_index(idx_r, nakx)

    links_l = np.zeros((naky, nakx), dtype=int)
    links_r = np.zeros((naky, nakx), dtype=int)
    for idx in range(nakx):
        for idy in range(naky):
            idx_star = idx
            while idx_star != idx_left[idy, idx_star] and idx_left[idy, idx_star] >= 0:
                links_l[idy, idx] += 1
                idx_star = idx_left[idy, idx_star]
            idx_star = idx
            while idx_star != idx_right[idy, idx_star] and idx_right[idy, idx_star] >= 0:
                links_r[idy, idx] += 1
                idx_star = idx_right[idy, idx_star]

    n_k = np.zeros(naky * nakx, dtype=int)
    k = 0
    for idx in range(nakx):
        for idy in range(naky):
            n_k[k] = 1 + links_l[idy, idx] + links_r[idy, idx]
            k += 1

    n_k_sorted = np.sort(n_k)
    unique_vals = np.unique(n_k_sorted)
    n_links = unique_vals.astype(int)
    n_chains = np.zeros_like(n_links)
    for i, val in enumerate(n_links):
        count = int(np.sum(n_k_sorted == val))
        n_chains[i] = count // val if val > 0 else 0

    linked_indices: list[jnp.ndarray] = []
    linked_kz: list[jnp.ndarray] = []
    for nlinks_val, nchains_val in zip(n_links, n_chains):
        if nlinks_val <= 0 or nchains_val <= 0:
            continue
        link_kx = np.zeros((nchains_val, nlinks_val), dtype=np.int32)
        link_ky = np.zeros((nchains_val, nlinks_val), dtype=np.int32)
        n = 0
        for idy in range(naky):
            for idx in range(nakx):
                np_k = 1 + links_l[idy, idx] + links_r[idy, idx]
                if np_k != nlinks_val:
                    continue
                p = links_l[idy, idx]
                if p != 0:
                    continue
                idx0 = idx if idx < (nakx + 1) // 2 else idx + nshift
                link_ky[n, 0] = idy
                link_kx[n, 0] = idx0
                idx_r = idx
                for p in range(1, nlinks_val):
                    idx_r = idx_right[idy, idx_r]
                    link_ky[n, p] = idy
                    if idx_r < (nakx + 1) // 2:
                        link_kx[n, p] = idx_r
                    else:
                        link_kx[n, p] = idx_r + nshift
                n += 1
        idx_flat = link_ky + ny * link_kx
        linked_indices.append(jnp.asarray(idx_flat, dtype=jnp.int32))
        nzL = int(nlinks_val) * int(nz)
        kz_linked = 2.0 * np.pi * np.fft.fftfreq(nzL, d=float(dz))
        linked_kz.append(jnp.asarray(kz_linked, dtype=real_dtype))

    return tuple(linked_indices), tuple(linked_kz)


def _build_linked_end_damping_profile(
    *,
    linked_indices: tuple[jnp.ndarray, ...],
    ny: int,
    nx: int,
    nz: int,
    widthfrac: float,
    ky_mode: np.ndarray | None = None,
) -> np.ndarray:
    """Construct the GX linked-boundary damping profile on the full FFT grid."""

    profile = np.zeros((ny, nx, nz), dtype=float)
    if not linked_indices or widthfrac <= 0.0 or ny <= 0 or nx <= 0 or nz <= 0:
        return profile
    ky_mode_arr: np.ndarray | None = None
    if ky_mode is not None:
        ky_mode_arr = np.asarray(ky_mode, dtype=np.int32).reshape(-1)
        if ky_mode_arr.size < ny:
            raise ValueError("ky_mode must have at least ny entries for linked end damping")
    if nx > 1:
        kx_neg = np.concatenate(([0], np.arange(nx - 1, 0, -1, dtype=np.int32)))
    else:
        kx_neg = np.asarray([0], dtype=np.int32)

    for idx_map_j in linked_indices:
        idx_map = np.asarray(idx_map_j, dtype=np.int32)
        if idx_map.ndim != 2 or idx_map.size == 0:
            continue
        nlinks = int(idx_map.shape[1])
        width = int(nz * nlinks * float(widthfrac))
        if width <= 0:
            continue
        chain_extent = nz * nlinks
        for chain in idx_map:
            for p, idx_flat in enumerate(chain):
                ky_idx = int(idx_flat % ny)
                kx_idx = int(idx_flat // ny)
                ky_phys = int(ky_mode_arr[ky_idx]) if ky_mode_arr is not None else ky_idx
                if ky_phys == 0:
                    continue
                if ky_mode_arr is not None:
                    mirror_matches = np.flatnonzero(ky_mode_arr == -ky_phys)
                    mirror_ky = int(mirror_matches[0]) if mirror_matches.size else ky_idx
                else:
                    mirror_ky = (-ky_idx) % ny
                mirror_kx = int(kx_neg[kx_idx])
                for idz in range(nz):
                    idzp = idz + nz * p
                    nu = 0.0
                    if idzp <= width:
                        x = float(idzp) / float(width)
                        nu = 1.0 - 2.0 * x * x / (1.0 + x**4)
                    elif idzp >= chain_extent - width:
                        x = float(chain_extent - idzp) / float(width)
                        nu = 1.0 - 2.0 * x * x / (1.0 + x**4)
                    profile[ky_idx, kx_idx, idz] = nu
                    if mirror_ky != ky_idx:
                        profile[mirror_ky, mirror_kx, idz] = nu
    return profile
