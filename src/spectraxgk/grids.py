"""Spectral grid utilities for flux-tube geometry."""

from __future__ import annotations

from dataclasses import dataclass
import jax.numpy as jnp

from spectraxgk.config import GridConfig


@dataclass(frozen=True)
class SpectralGrid:
    kx: jnp.ndarray
    ky: jnp.ndarray
    z: jnp.ndarray
    kx_grid: jnp.ndarray
    ky_grid: jnp.ndarray
    dealias_mask: jnp.ndarray


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
    kx = _fftfreq_phys(cfg.Nx, cfg.Lx)
    ky = _fftfreq_phys(cfg.Ny, cfg.Ly)
    z = jnp.linspace(cfg.z_min, cfg.z_max, cfg.Nz, endpoint=False)
    ky_grid, kx_grid = jnp.meshgrid(ky, kx, indexing="ij")
    mask = twothirds_mask(cfg.Ny, cfg.Nx)
    return SpectralGrid(kx=kx, ky=ky, z=z, kx_grid=kx_grid, ky_grid=ky_grid, dealias_mask=mask)
