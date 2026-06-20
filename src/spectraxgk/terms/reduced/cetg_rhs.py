"""Field solve and RHS assembly for the cETG reduced model."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from spectraxgk.core.grid import SpectralGrid
from spectraxgk.terms.brackets import _spectral_bracket_multi
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.reduced.cetg_model import CETGModelParams
from spectraxgk.terms.reduced.cetg_state import (
    _apply_kz_filter,
    _dz2,
    _project_state,
    _to_internal_state,
    _xy_mask,
)


def cetg_fields(
    G: jnp.ndarray,
    grid: SpectralGrid,
    params: CETGModelParams,
    *,
    apply_kz_dealias: bool = True,
) -> FieldState:
    """Solve the cETG electrostatic field equation."""

    G_int = _to_internal_state(G)
    phi = -jnp.asarray(params.tau_fac, dtype=jnp.real(G_int).dtype) * G_int[0]
    phi = phi * _xy_mask(grid, jnp.real(phi).dtype)[0]
    if apply_kz_dealias:
        phi = _apply_kz_filter(phi, grid, dealias_kz=params.dealias_kz)
    return FieldState(phi=phi, apar=None, bpar=None)


def _cetg_linear_rhs(
    G: jnp.ndarray,
    fields: FieldState,
    terms: TermConfig,
    grid: SpectralGrid,
    params: CETGModelParams,
) -> jnp.ndarray:
    G_int = _to_internal_state(G)
    density = G_int[0]
    temperature = G_int[1]
    phi = fields.phi
    gpar2 = jnp.asarray(params.gradpar * params.gradpar, dtype=jnp.real(G_int).dtype)
    c1 = jnp.asarray(params.c1, dtype=jnp.real(G_int).dtype)
    C12 = jnp.asarray(params.C12, dtype=jnp.real(G_int).dtype)
    C23 = jnp.asarray(params.C23, dtype=jnp.real(G_int).dtype)

    rhs0 = 0.5 * gpar2 * c1 * (density + C12 * temperature - phi)
    rhs1 = (gpar2 / 3.0) * c1 * (C12 * density + C23 * temperature - C12 * phi)
    rhs = jnp.stack([_dz2(rhs0, grid), _dz2(rhs1, grid)], axis=0)

    ky = jnp.asarray(grid.ky, dtype=jnp.real(G_int).dtype)[:, None, None]
    rhs = rhs.at[1].add(-0.5j * ky * phi)
    if float(terms.hyperdiffusion) != 0.0 and float(params.D_hyper) != 0.0:
        kx = jnp.asarray(grid.kx, dtype=jnp.real(G_int).dtype)[None, :, None]
        k2 = kx * kx + ky * ky
        Dfac = jnp.asarray(params.D_hyper, dtype=jnp.real(G_int).dtype) * (
            k2 ** jnp.asarray(params.nu_hyper, dtype=jnp.real(G_int).dtype)
        )
        rhs = (
            rhs
            - jnp.asarray(float(terms.hyperdiffusion), dtype=jnp.real(G_int).dtype)
            * Dfac[None, ...]
            * G_int
        )
    rhs = rhs * _xy_mask(grid, jnp.real(rhs).dtype)
    return _apply_kz_filter(rhs, grid, dealias_kz=params.dealias_kz)


def _cetg_nonlinear_rhs(
    G: jnp.ndarray,
    fields: FieldState,
    grid: SpectralGrid,
    *,
    compressed_real_fft: bool,
) -> jnp.ndarray:
    G_int = _to_internal_state(G)
    phi = fields.phi
    bracket = _spectral_bracket_multi(
        G_int,
        phi[None, ...],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(grid.kxfac, dtype=jnp.real(G_int).dtype),
        compressed_real_fft=compressed_real_fft,
    )[0]
    bracket = 0.5 * bracket
    bracket = bracket * _xy_mask(grid, jnp.real(bracket).dtype)
    return bracket


def cetg_rhs(
    G: jnp.ndarray,
    grid: SpectralGrid,
    params: CETGModelParams,
    terms: TermConfig,
    *,
    compressed_real_fft: bool,
    fields_override: FieldState | None = None,
) -> tuple[jnp.ndarray, FieldState]:
    """Return the full cETG RHS and the electrostatic fields."""

    G_int = _to_internal_state(G)
    fields = (
        cetg_fields(G_int, grid, params) if fields_override is None else fields_override
    )
    rhs = _cetg_linear_rhs(G_int, fields, terms, grid, params)
    if float(terms.nonlinear) != 0.0:
        rhs = rhs + jnp.asarray(
            float(terms.nonlinear), dtype=jnp.real(rhs).dtype
        ) * _cetg_nonlinear_rhs(
            G_int,
            fields,
            grid,
            compressed_real_fft=compressed_real_fft,
        )
    rhs = _project_state(rhs, grid, compressed_real_fft=compressed_real_fft)
    return rhs, fields


def _cetg_linear_omega_max(grid: SpectralGrid, params: CETGModelParams) -> float:
    ny = int(grid.ky.size)
    nz = int(grid.z.size)
    ky_max = (
        float(abs(np.asarray(grid.ky, dtype=float)[(ny - 1) // 3])) if ny > 1 else 0.0
    )
    z0 = abs(float(params.z0))
    kz_max = (float(nz) / 3.0) * float(params.gradpar) / z0
    cfac = 0.5 * float(params.c1) * float(np.sqrt(1.0 + (params.C12 - 1.0)))
    return float(cfac * np.sqrt(max(ky_max, 0.0)) * kz_max)


def _cetg_nonlinear_omega_components(
    phi: jnp.ndarray,
    grid: SpectralGrid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    complex_dtype = jnp.result_type(phi, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=complex_dtype)).dtype
    imag = jnp.asarray(1j, dtype=complex_dtype)
    fft_norm = float(grid.ky.size * grid.kx.size)
    ifft_scale = jnp.asarray(fft_norm, dtype=real_dtype)
    kx = jnp.asarray(grid.kx_grid, dtype=real_dtype)
    ky = jnp.asarray(grid.ky_grid, dtype=real_dtype)
    dphi_dx = jnp.fft.ifft2(imag * kx[:, :, None] * phi, axes=(-3, -2)) * ifft_scale
    dphi_dy = jnp.fft.ifft2(imag * ky[:, :, None] * phi, axes=(-3, -2)) * ifft_scale
    vmax_x = jnp.max(jnp.abs(dphi_dy))
    vmax_y = jnp.max(jnp.abs(dphi_dx))
    nx = int(grid.kx.size)
    ny = int(grid.ky.size)
    kx_max = (
        float(abs(np.asarray(grid.kx, dtype=float)[(nx - 1) // 3])) if nx > 1 else 0.0
    )
    ky_max = (
        float(abs(np.asarray(grid.ky, dtype=float)[(ny - 1) // 3])) if ny > 1 else 0.0
    )
    return jnp.asarray(kx_max, dtype=real_dtype) * vmax_x, jnp.asarray(
        ky_max, dtype=real_dtype
    ) * vmax_y


__all__ = [
    "_cetg_linear_omega_max",
    "_cetg_linear_rhs",
    "_cetg_nonlinear_omega_components",
    "_cetg_nonlinear_rhs",
    "cetg_fields",
    "cetg_rhs",
]
