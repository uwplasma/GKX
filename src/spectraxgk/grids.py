"""Spectral grid utilities for flux-tube geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.config import GridConfig


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SpectralGrid:
    kx: jnp.ndarray
    ky: jnp.ndarray
    z: jnp.ndarray
    kx_grid: jnp.ndarray
    ky_grid: jnp.ndarray
    dealias_mask: jnp.ndarray
    y0: float
    x0: float
    boundary: str
    jtwist: int | None
    non_twist: bool
    kxfac: float
    ky_mode: jnp.ndarray | None = None

    def tree_flatten(self):
        children = (
            self.kx,
            self.ky,
            self.z,
            self.kx_grid,
            self.ky_grid,
            self.dealias_mask,
        )
        aux_data = (
            self.y0,
            self.x0,
            self.boundary,
            self.jtwist,
            self.non_twist,
            self.kxfac,
            self.ky_mode,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        y0, x0, boundary, jtwist, non_twist, kxfac, ky_mode = aux_data
        return cls(
            *children,
            y0=y0,
            x0=x0,
            boundary=boundary,
            jtwist=jtwist,
            non_twist=non_twist,
            kxfac=kxfac,
            ky_mode=ky_mode,
        )


def _fftfreq_phys(n: int, L: float) -> jnp.ndarray:
    """Physical wave numbers for an FFT grid of length L."""

    return 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=L / n)


def twothirds_mask(Ny: int, Nx: int) -> jnp.ndarray:
    """2/3 dealiasing mask for 2D Fourier grids."""

    ky = jnp.fft.fftfreq(Ny)
    kx = jnp.fft.fftfreq(Nx)
    # GX excludes the boundary shell |k| = 1/3 and keeps only the strict 2/3 interior.
    ky_ok = jnp.abs(ky) < (1.0 / 3.0)
    kx_ok = jnp.abs(kx) < (1.0 / 3.0)
    return ky_ok[:, None] & kx_ok[None, :]


def gx_real_fft_ky(ky: jnp.ndarray) -> jnp.ndarray:
    """Return GX's compressed real-FFT ``ky`` convention.

    GX stores the unique ``ky`` block with non-negative values, including the
    positive Nyquist mode when ``Ny`` is even.
    """

    ky_arr = jnp.asarray(ky)
    if ky_arr.ndim == 0:
        raise ValueError("ky must be at least 1D")
    ky_1d = ky_arr if ky_arr.ndim == 1 else ky_arr[:, 0]
    nyc = 1 + int(ky_1d.shape[0]) // 2
    return jnp.abs(ky_1d[:nyc])


def gx_real_fft_kx(kx: jnp.ndarray) -> jnp.ndarray:
    """Return GX's native ``kx`` ordering for real-FFT nonlinear kernels.

    GX keeps the x-axis in full complex-FFT order but uses a positive Nyquist
    multiplier when ``Nx`` is even.
    """

    kx_arr = jnp.asarray(kx)
    if kx_arr.ndim == 0:
        raise ValueError("kx must be at least 1D")
    kx_1d = kx_arr if kx_arr.ndim == 1 else kx_arr[0, :]
    nx = int(kx_1d.shape[0])
    if nx == 0 or (nx % 2) != 0:
        return kx_1d
    return kx_1d.at[nx // 2].set(jnp.abs(kx_1d[nx // 2]))


def gx_real_fft_mesh(
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return GX-style compressed ``(kx, ky)`` multipliers and meshgrids."""

    kx = gx_real_fft_kx(kx_grid)
    ky = gx_real_fft_ky(ky_grid)
    ky_mesh, kx_mesh = jnp.meshgrid(ky, kx, indexing="ij")
    return kx, ky, kx_mesh, ky_mesh


def build_spectral_grid(cfg: GridConfig) -> SpectralGrid:
    Lx = cfg.Lx
    Ly = 2.0 * jnp.pi * cfg.y0 if cfg.y0 is not None else cfg.Ly
    y0 = float(cfg.y0) if cfg.y0 is not None else float(Ly) / (2.0 * jnp.pi)
    x0 = float(Lx) / (2.0 * jnp.pi)

    zp = cfg.zp
    if zp is None:
        if cfg.nperiod is not None:
            zp = 2 * cfg.nperiod - 1
        elif cfg.ntheta is not None:
            zp = 1

    Nz = cfg.Nz
    if cfg.ntheta is not None:
        Nz = int(cfg.ntheta) * int(zp if zp is not None else 1)
        z_min = -jnp.pi * float(zp if zp is not None else 1)
        z_max = jnp.pi * float(zp if zp is not None else 1)
    else:
        z_min = cfg.z_min
        z_max = cfg.z_max

    kx = _fftfreq_phys(cfg.Nx, Lx)
    ky = _fftfreq_phys(cfg.Ny, Ly)
    z = jnp.linspace(z_min, z_max, Nz, endpoint=False)
    ky_grid, kx_grid = jnp.meshgrid(ky, kx, indexing="ij")
    mask = twothirds_mask(cfg.Ny, cfg.Nx)
    return SpectralGrid(
        kx=kx,
        ky=ky,
        z=z,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=mask,
        y0=y0,
        x0=x0,
        boundary=str(cfg.boundary),
        jtwist=cfg.jtwist,
        non_twist=bool(cfg.non_twist),
        kxfac=float(cfg.kxfac),
        ky_mode=None,
    )


def select_ky_grid(
    grid: SpectralGrid,
    ky_index: int | jnp.ndarray | np.ndarray | Sequence[int],
) -> SpectralGrid:
    """Return a grid sliced down to one or more ky indices."""

    ky_idx = jnp.asarray(ky_index, dtype=jnp.int32)
    if ky_idx.ndim == 0:
        ky_idx = ky_idx[None]
    ky = jnp.take(grid.ky, ky_idx, axis=0)
    ky_grid = jnp.take(grid.ky_grid, ky_idx, axis=0)
    kx_grid = jnp.take(grid.kx_grid, ky_idx, axis=0)
    mask = jnp.take(grid.dealias_mask, ky_idx, axis=0)
    ky_mode = jnp.rint(ky * grid.y0).astype(jnp.int32)
    return SpectralGrid(
        kx=grid.kx,
        ky=ky,
        z=grid.z,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=mask,
        y0=grid.y0,
        x0=grid.x0,
        boundary=grid.boundary,
        jtwist=grid.jtwist,
        non_twist=grid.non_twist,
        kxfac=grid.kxfac,
        ky_mode=ky_mode,
    )


def select_gx_real_fft_ky_grid(
    grid: SpectralGrid,
    ky_values: jnp.ndarray | np.ndarray | Sequence[float],
) -> SpectralGrid:
    """Return a GX real-FFT positive-``ky`` view of ``grid``.

    GX nonlinear dumps store the compressed non-negative ``ky`` block, including
    the unique Nyquist row when ``Ny`` is even. This helper keeps the matching
    leading dealias rows from the full FFT grid while replacing the ``ky``
    coordinates with the explicit positive-frequency values from the dump.
    """

    ky_vals = jnp.asarray(ky_values, dtype=grid.ky.dtype)
    if ky_vals.ndim != 1 or ky_vals.size == 0:
        raise ValueError("ky_values must be a non-empty 1D array")
    nky = int(ky_vals.shape[0])
    if nky > int(grid.ky.shape[0]):
        raise ValueError("ky_values length cannot exceed the full grid ky length")
    kx_vals = gx_real_fft_kx(grid.kx)
    mask = jnp.take(grid.dealias_mask, jnp.arange(nky, dtype=jnp.int32), axis=0)
    kx_grid = jnp.broadcast_to(kx_vals[None, :], (nky, kx_vals.shape[0]))
    ky_grid = jnp.broadcast_to(ky_vals[:, None], (nky, kx_vals.shape[0]))
    ky_mode = jnp.rint(ky_vals * grid.y0).astype(jnp.int32)
    return SpectralGrid(
        kx=kx_vals,
        ky=ky_vals,
        z=grid.z,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=mask,
        y0=grid.y0,
        x0=grid.x0,
        boundary=grid.boundary,
        jtwist=grid.jtwist,
        non_twist=grid.non_twist,
        kxfac=grid.kxfac,
        ky_mode=ky_mode,
    )
