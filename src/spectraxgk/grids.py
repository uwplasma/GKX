"""Spectral grid utilities for flux-tube geometry."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

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

    def tree_flatten(self):
        children = (
            self.kx,
            self.ky,
            self.z,
            self.kx_grid,
            self.ky_grid,
            self.dealias_mask,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


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
    return SpectralGrid(kx=kx, ky=ky, z=z, kx_grid=kx_grid, ky_grid=ky_grid, dealias_mask=mask)


def select_ky_grid(grid: SpectralGrid, ky_index: int) -> SpectralGrid:
    """Return a grid sliced down to a single ky index."""

    ky = grid.ky[ky_index : ky_index + 1]
    ky_grid = grid.ky_grid[ky_index : ky_index + 1, :]
    kx_grid = grid.kx_grid[ky_index : ky_index + 1, :]
    mask = grid.dealias_mask[ky_index : ky_index + 1, :]
    return SpectralGrid(
        kx=grid.kx,
        ky=ky,
        z=grid.z,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=mask,
    )
