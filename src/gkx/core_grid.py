"""Spectral grid utilities for flux-tube geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from gkx.config import GridConfig
from gkx.core_ky_layout import (
    FULL,
    KyLayout,
    half_ky_values,
    half_twothirds_mask,
    ky_layout_of,
    nyc_from_ny,
)


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
    #: Length of the two-sided ``ky`` axis this grid's rows were taken from,
    #: or ``None`` when the rows are not a complete axis.  A half-spectrum
    #: block cannot say how long its own full axis is
    #: (:mod:`gkx.core_ky_layout`), and the reduction weights of an even grid's
    #: Nyquist row depend on the answer, so the grid carries it.
    ny_full: int | None = None

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
            self.ny_full,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        y0, x0, boundary, jtwist, non_twist, kxfac, ky_mode, ny_full = aux_data
        return cls(
            *children,
            y0=y0,
            x0=x0,
            boundary=boundary,
            jtwist=jtwist,
            non_twist=non_twist,
            kxfac=kxfac,
            ky_mode=ky_mode,
            ny_full=ny_full,
        )

    @property
    def ky_layout(self) -> KyLayout:
        """Return whether this grid's ``ky`` rows are the full axis or its half.

        Reads :attr:`ny_full`, which is the only thing that can tell them
        apart; a grid that does not carry one is a selection of modes and is
        reported as :data:`~gkx.core_ky_layout.FULL`.
        """

        return ky_layout_of(int(self.ky.shape[0]), self.ny_full)


def _fftfreq_phys(n: int, L: float) -> jnp.ndarray:
    """Physical wave numbers for an FFT grid of length L."""

    return 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=L / n)


def twothirds_mask(Ny: int, Nx: int, *, ky_layout: KyLayout = FULL) -> jnp.ndarray:
    """2/3 dealiasing mask for 2D Fourier grids.

    ``Ny`` is always the length of the two-sided axis.  ``ky_layout`` selects
    how many rows are returned: the full axis, or the ``ky >= 0`` block that a
    half-spectrum state carries.
    """

    if ky_layout != FULL:
        return half_twothirds_mask(Ny, Nx)
    ky = jnp.fft.fftfreq(Ny)
    kx = jnp.fft.fftfreq(Nx)
    # The two-thirds dealiased convolution rule keeps only the strict interior.
    ky_ok = jnp.abs(ky) < (1.0 / 3.0)
    kx_ok = jnp.abs(kx) < (1.0 / 3.0)
    return ky_ok[:, None] & kx_ok[None, :]


def real_fft_unique_ky(ky: jnp.ndarray) -> jnp.ndarray:
    """Return the compressed non-negative `ky` block for a real FFT."""

    ky_arr = jnp.asarray(ky)
    if ky_arr.ndim == 0:
        raise ValueError("ky must be at least 1D")
    ky_1d = ky_arr if ky_arr.ndim == 1 else ky_arr[:, 0]
    return half_ky_values(ky_1d)


def real_fft_ordered_kx(kx: jnp.ndarray) -> jnp.ndarray:
    """Return the `kx` ordering used with real-FFT nonlinear kernels."""

    kx_arr = jnp.asarray(kx)
    if kx_arr.ndim == 0:
        raise ValueError("kx must be at least 1D")
    kx_1d = kx_arr if kx_arr.ndim == 1 else kx_arr[0, :]
    nx = int(kx_1d.shape[0])
    if nx == 0 or (nx % 2) != 0:
        return kx_1d
    return kx_1d.at[nx // 2].set(jnp.abs(kx_1d[nx // 2]))


def real_fft_mesh(
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return compressed real-FFT `(kx, ky)` multipliers and meshgrids."""

    kx = real_fft_ordered_kx(kx_grid)
    ky = real_fft_unique_ky(ky_grid)
    ky_mesh, kx_mesh = jnp.meshgrid(ky, kx, indexing="ij")
    return kx, ky, kx_mesh, ky_mesh


def _gyrokinetic_moment_shape(
    state: jnp.ndarray, *, name: str = "G0"
) -> tuple[int, int]:
    """Return ``(Nl, Nm)`` with or without a leading species axis."""

    if state.ndim == 5:
        return int(state.shape[0]), int(state.shape[1])
    if state.ndim == 6:
        return int(state.shape[1]), int(state.shape[2])
    raise ValueError(
        f"{name} must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
    )


def build_spectral_grid(cfg: GridConfig, *, ky_layout: KyLayout = FULL) -> SpectralGrid:
    """Return the spectral grid of ``cfg`` with its ``ky`` axis in ``ky_layout``.

    ``FULL`` is the two-sided ``fftfreq`` axis of length ``Ny``.  ``HALF``
    keeps the ``Nyc = 1 + Ny // 2`` non-negative rows, which is the layout a
    half-spectrum evolved state uses (plan 5.3 N3); the grid still records the
    full length in :attr:`SpectralGrid.ny_full`, because ``Nyc`` alone cannot
    supply it.
    """

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
    ky_full = _fftfreq_phys(cfg.Ny, Ly)
    ky = ky_full if ky_layout == FULL else half_ky_values(ky_full)
    z = jnp.linspace(z_min, z_max, Nz, endpoint=False)
    ky_grid, kx_grid = jnp.meshgrid(ky, kx, indexing="ij")
    mask = twothirds_mask(cfg.Ny, cfg.Nx, ky_layout=ky_layout)
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
        ny_full=int(ky_full.shape[0]),
    )


def _parent_ny_full(grid: SpectralGrid) -> int:
    """Return the two-sided ``ky`` length behind `grid`'s rows."""

    if grid.ny_full is not None:
        return int(grid.ny_full)
    return int(grid.ky.shape[0])


def _selected_ny_full(grid: SpectralGrid, ky_vals: jnp.ndarray) -> int | None:
    """Return ``ny_full`` for a selection, or ``None`` when it is a subset.

    Only a complete axis may carry the parent length: the reduction weights
    read the Nyquist row by index, and an index into an arbitrary selection of
    modes names the wrong row.  The two complete cases are the full axis itself
    and the leading ``ky >= 0`` block, and the block is recognized by value so
    that a list of wave numbers read from another code's dump -- which
    :func:`select_real_fft_ky_grid` also accepts -- is not mistaken for it.
    """

    parent = _parent_ny_full(grid)
    rows = int(ky_vals.shape[0])
    if rows == parent:
        return parent
    if rows != nyc_from_ny(parent):
        return None
    try:
        selected = np.asarray(ky_vals, dtype=float)
        expected = np.abs(np.asarray(grid.ky, dtype=float)[:rows])
    except (TypeError, ValueError):
        return None
    return parent if np.array_equal(selected, expected) else None


def select_ky_grid(
    grid: SpectralGrid,
    ky_index: int | jnp.ndarray | np.ndarray | Sequence[int],
) -> SpectralGrid:
    """Return a linear-solver grid sliced down to one or more ky indices.

    The parent grid's two-thirds mask belongs to nonlinear FFT products.  A
    linear ky scan must not zero a selected high-ky mode just because that row
    would be dealiased in a nonlinear convolution, so sliced linear grids carry
    an all-true mask.
    """

    ky_idx = jnp.asarray(ky_index, dtype=jnp.int32)
    if ky_idx.ndim == 0:
        ky_idx = ky_idx[None]
    ky = jnp.take(grid.ky, ky_idx, axis=0)
    ky_grid = jnp.take(grid.ky_grid, ky_idx, axis=0)
    kx_grid = jnp.take(grid.kx_grid, ky_idx, axis=0)
    mask = jnp.ones_like(jnp.take(grid.dealias_mask, ky_idx, axis=0), dtype=bool)
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
        ny_full=_selected_ny_full(grid, ky),
    )


def select_real_fft_ky_grid(
    grid: SpectralGrid,
    ky_values: jnp.ndarray | np.ndarray | Sequence[float],
) -> SpectralGrid:
    """Return a positive-`ky` real-FFT view of `grid`."""

    ky_vals = jnp.asarray(ky_values, dtype=grid.ky.dtype)
    if ky_vals.ndim != 1 or ky_vals.size == 0:
        raise ValueError("ky_values must be a non-empty 1D array")
    nky = int(ky_vals.shape[0])
    if nky > int(grid.ky.shape[0]):
        raise ValueError("ky_values length cannot exceed the full grid ky length")
    kx_vals = real_fft_ordered_kx(grid.kx)
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
        ny_full=_selected_ny_full(grid, ky_vals),
    )
