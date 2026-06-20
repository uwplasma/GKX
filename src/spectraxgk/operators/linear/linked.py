"""Linked-boundary FFT maps and damping profiles for linear operators."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

__all__ = [
    "_build_linked_end_damping_profile",
    "_build_linked_fft_maps",
    "_signed_to_index",
]


@dataclass(frozen=True)
class _LinkedActiveModes:
    naky: int
    nakx: int
    nshift: int
    ky_mode: np.ndarray | None


@dataclass(frozen=True)
class _LinkedNeighborMaps:
    left: np.ndarray
    right: np.ndarray


@dataclass(frozen=True)
class _LinkedChainCounts:
    left: np.ndarray
    right: np.ndarray
    n_links: np.ndarray
    n_chains: np.ndarray


def _signed_to_index(idx: int, n: int) -> int:
    half = (n + 1) // 2
    if 0 <= idx < half:
        return idx
    if half <= idx + n < n:
        return idx + n
    return -1


def _linked_active_modes(
    *,
    nx: int,
    ny: int,
    ky_mode: np.ndarray | None,
) -> _LinkedActiveModes:
    if ky_mode is not None:
        ky_mode_arr = np.asarray(ky_mode, dtype=int).reshape(-1)
        naky = int(ky_mode_arr.size)
    else:
        ky_mode_arr = None
        naky = 1 + (ny - 1) // 3
    if nx < 4:
        nakx = nx
    else:
        nakx = 1 + 2 * ((nx - 1) // 3)
    return _LinkedActiveModes(
        naky=naky,
        nakx=nakx,
        nshift=nx - nakx,
        ky_mode=ky_mode_arr,
    )


def _linked_neighbor_maps(
    *,
    active: _LinkedActiveModes,
    jtwist: int,
) -> _LinkedNeighborMaps:
    idx_left = -np.ones((active.naky, active.nakx), dtype=int)
    idx_right = -np.ones((active.naky, active.nakx), dtype=int)
    for idx in range(active.nakx):
        idx0 = idx if idx < (active.nakx + 1) // 2 else idx - active.nakx
        for idy in range(active.naky):
            idy_mode = int(active.ky_mode[idy]) if active.ky_mode is not None else idy
            if idy_mode == 0:
                idx_l = idx0
                idx_r = idx0
            else:
                idx_l = idx0 + idy_mode * jtwist
                idx_r = idx0 - idy_mode * jtwist
            idx_left[idy, idx] = _signed_to_index(idx_l, active.nakx)
            idx_right[idy, idx] = _signed_to_index(idx_r, active.nakx)
    return _LinkedNeighborMaps(left=idx_left, right=idx_right)


def _linked_counts_in_direction(
    neighbor_map: np.ndarray,
) -> np.ndarray:
    counts = np.zeros_like(neighbor_map, dtype=int)
    naky, nakx = neighbor_map.shape
    for idx in range(nakx):
        for idy in range(naky):
            idx_star = idx
            while idx_star != neighbor_map[idy, idx_star] and neighbor_map[idy, idx_star] >= 0:
                counts[idy, idx] += 1
                idx_star = neighbor_map[idy, idx_star]
    return counts


def _linked_chain_counts(neighbors: _LinkedNeighborMaps) -> _LinkedChainCounts:
    links_l = _linked_counts_in_direction(neighbors.left)
    links_r = _linked_counts_in_direction(neighbors.right)
    n_k = (1 + links_l + links_r).reshape(-1)
    n_k_sorted = np.sort(n_k)
    n_links = np.unique(n_k_sorted).astype(int)
    n_chains = np.zeros_like(n_links)
    for i, val in enumerate(n_links):
        count = int(np.sum(n_k_sorted == val))
        n_chains[i] = count // val if val > 0 else 0
    return _LinkedChainCounts(
        left=links_l,
        right=links_r,
        n_links=n_links,
        n_chains=n_chains,
    )


def _full_kx_index(idx: int, active: _LinkedActiveModes) -> int:
    if idx < (active.nakx + 1) // 2:
        return idx
    return idx + active.nshift


def _linked_chain_indices_for_length(
    *,
    active: _LinkedActiveModes,
    neighbors: _LinkedNeighborMaps,
    counts: _LinkedChainCounts,
    ny: int,
    nlinks_val: int,
    nchains_val: int,
) -> np.ndarray:
    link_kx = np.zeros((nchains_val, nlinks_val), dtype=np.int32)
    link_ky = np.zeros((nchains_val, nlinks_val), dtype=np.int32)
    n = 0
    for idy in range(active.naky):
        for idx in range(active.nakx):
            np_k = 1 + counts.left[idy, idx] + counts.right[idy, idx]
            if np_k != nlinks_val or counts.left[idy, idx] != 0:
                continue
            link_ky[n, 0] = idy
            link_kx[n, 0] = _full_kx_index(idx, active)
            idx_r = idx
            for p in range(1, nlinks_val):
                idx_r = neighbors.right[idy, idx_r]
                link_ky[n, p] = idy
                link_kx[n, p] = _full_kx_index(idx_r, active)
            n += 1
    return link_ky + ny * link_kx


def _linked_kz_for_length(
    *,
    nlinks_val: int,
    nz: int,
    dz: float,
    real_dtype: jnp.dtype,
) -> jnp.ndarray:
    nz_linked = int(nlinks_val) * int(nz)
    kz_linked = 2.0 * np.pi * np.fft.fftfreq(nz_linked, d=float(dz))
    return jnp.asarray(kz_linked, dtype=real_dtype)


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
    active = _linked_active_modes(nx=nx, ny=ny, ky_mode=ky_mode)
    if active.nakx <= 0 or active.naky <= 0:
        return (), ()

    neighbors = _linked_neighbor_maps(active=active, jtwist=jtwist)
    counts = _linked_chain_counts(neighbors)

    linked_indices: list[jnp.ndarray] = []
    linked_kz: list[jnp.ndarray] = []
    for nlinks_val, nchains_val in zip(counts.n_links, counts.n_chains):
        if nlinks_val <= 0 or nchains_val <= 0:
            continue
        idx_flat = _linked_chain_indices_for_length(
            active=active,
            neighbors=neighbors,
            counts=counts,
            ny=ny,
            nlinks_val=int(nlinks_val),
            nchains_val=int(nchains_val),
        )
        linked_indices.append(jnp.asarray(idx_flat, dtype=jnp.int32))
        linked_kz.append(
            _linked_kz_for_length(
                nlinks_val=int(nlinks_val),
                nz=nz,
                dz=dz,
                real_dtype=real_dtype,
            )
        )

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
    """Construct the linked-boundary damping profile on the full FFT grid."""

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
