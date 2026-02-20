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

    def tree_flatten(self):
        children = (
            self.kx,
            self.ky,
            self.z,
            self.kx_grid,
            self.ky_grid,
            self.dealias_mask,
        )
        aux_data = (self.y0, self.x0, self.boundary, self.jtwist, self.non_twist, self.kxfac)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        y0, x0, boundary, jtwist, non_twist, kxfac = aux_data
        return cls(
            *children,
            y0=y0,
            x0=x0,
            boundary=boundary,
            jtwist=jtwist,
            non_twist=non_twist,
            kxfac=kxfac,
        )


def _fftfreq_phys(n: int, L: float) -> jnp.ndarray:
    """Physical wave numbers for an FFT grid of length L."""

    return 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=L / n)


def twothirds_mask(Ny: int, Nx: int) -> jnp.ndarray:
    """2/3 dealiasing mask for 2D Fourier grids."""

    ky = jnp.fft.fftfreq(Ny)
    kx = jnp.fft.fftfreq(Nx)
    ky_ok = jnp.abs(ky) <= (1.0 / 3.0)
    kx_ok = jnp.abs(kx) <= (1.0 / 3.0)
    return ky_ok[:, None] & kx_ok[None, :]


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
    )
