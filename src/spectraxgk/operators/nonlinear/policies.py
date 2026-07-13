"""Helper policies and operators for nonlinear gyrokinetic drivers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import jax
import numpy as np

from spectraxgk.core.grid import SpectralGrid, real_fft_mesh
from spectraxgk.solvers.linear.implicit import _build_implicit_operator
from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.params import (
    LinearParams,
    term_config_to_linear_terms,
)
from spectraxgk.operators.nonlinear.collisions import (
    NonlinearCollisionSplitPolicy,
    _apply_collision_split,
    _collision_damping,
    build_nonlinear_collision_split_policy,
)
from spectraxgk.operators.nonlinear.projection import (
    ShearingCoordinateUpdate,
    _make_fixed_mode_projector,
    _make_hermitian_projector,
    _make_nonlinear_state_projector,
    advance_shearing_coordinates,
)
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.operators.nonlinear.brackets import _broadcast_grid, _ifft2_xy

__all__ = [
    "IMEXLinearOperator",
    "NonlinearCollisionSplitPolicy",
    "NonlinearDiagnosticSetup",
    "NonlinearTimeStepPolicy",
    "ShearingCoordinateUpdate",
    "_apply_collision_split",
    "_collision_damping",
    "_nonlinear_cfl_frequency_components",
    "_diagnostic_omega_mode_mask",
    "_make_fixed_mode_projector",
    "_make_hermitian_projector",
    "_make_nonlinear_state_projector",
    "advance_shearing_coordinates",
    "build_nonlinear_collision_split_policy",
    "build_nonlinear_diagnostic_setup",
    "build_nonlinear_imex_operator",
    "build_nonlinear_time_step_policy",
]


@dataclass(frozen=True)
class IMEXLinearOperator:
    """Reusable matrix-free linear operator for nonlinear IMEX solves."""

    state_dtype: jnp.dtype
    shape: tuple[int, ...]
    dt_val: jnp.ndarray
    precond_op: Callable[[jnp.ndarray], jnp.ndarray] | None
    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    squeeze_species: bool


@dataclass(frozen=True)
class NonlinearDiagnosticSetup:
    """Shared cache, weights, masks, and projection policy for diagnostics."""

    geom: Any
    cache: LinearCache
    vol_fac: jnp.ndarray
    flux_fac: jnp.ndarray
    mask: jnp.ndarray
    z_idx: int
    use_dealias: bool
    project_state: Callable[[jnp.ndarray], jnp.ndarray]


@dataclass(frozen=True)
class NonlinearTimeStepPolicy:
    """Initial step, progress horizon, and adaptive update callable."""

    dt_init: jnp.ndarray
    progress_total: jnp.ndarray
    update_dt: Callable[[FieldState, jnp.ndarray], jnp.ndarray]


@dataclass(frozen=True)
class _TimeStepLimits:
    dt_init: jnp.ndarray
    progress_total: jnp.ndarray
    dt_min: jnp.ndarray
    dt_max: jnp.ndarray
    cfl: jnp.ndarray
    cfl_fac: jnp.ndarray


@dataclass(frozen=True)
class _NonlinearCFLBounds:
    kx_max: jnp.ndarray
    ky_max: jnp.ndarray
    vpar_max: jnp.ndarray
    muB_max: jnp.ndarray
    kxfac: jnp.ndarray
    linear_omega: jnp.ndarray


def _nonlinear_moment_counts(G0: jnp.ndarray) -> tuple[int, int]:
    if G0.ndim == 5:
        return int(G0.shape[0]), int(G0.shape[1])
    if G0.ndim == 6:
        return int(G0.shape[1]), int(G0.shape[2])
    raise ValueError(
        "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
    )


def build_nonlinear_diagnostic_setup(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: Any,
    params: LinearParams,
    *,
    cache: LinearCache | None,
    use_dealias_mask: bool,
    z_index: int | None,
    compressed_real_fft: bool,
    fixed_mode_ky_index: int | None,
    fixed_mode_kx_index: int | None,
    ensure_geometry_fn: Callable[..., Any],
    build_cache_fn: Callable[..., LinearCache],
    quadrature_weights_fn: Callable[..., tuple[jnp.ndarray, jnp.ndarray]],
    omega_mask_fn: Callable[..., jnp.ndarray],
    midplane_index_fn: Callable[[int], int],
) -> NonlinearDiagnosticSetup:
    """Build the shared diagnostic setup used by explicit and IMEX scans."""

    geom_eff = ensure_geometry_fn(geom, grid.z)
    if cache is None:
        nl, nm = _nonlinear_moment_counts(G0)
        cache = build_cache_fn(grid, geom_eff, params, nl, nm)

    vol_fac, flux_fac = quadrature_weights_fn(geom_eff, grid)
    mask = omega_mask_fn(grid, cache, compressed_real_fft=compressed_real_fft)
    z_idx = midplane_index_fn(grid.z.size) if z_index is None else int(z_index)
    project_state = _make_nonlinear_state_projector(
        G0,
        ky_vals=np.asarray(grid.ky),
        nx=int(grid.kx.size),
        compressed_real_fft=compressed_real_fft,
        fixed_mode_ky_index=fixed_mode_ky_index,
        fixed_mode_kx_index=fixed_mode_kx_index,
    )
    return NonlinearDiagnosticSetup(
        geom=geom_eff,
        cache=cache,
        vol_fac=vol_fac,
        flux_fac=flux_fac,
        mask=mask,
        z_idx=z_idx,
        use_dealias=bool(use_dealias_mask),
        project_state=project_state,
    )


def _build_time_step_limits(
    *,
    method: str,
    dt: float,
    steps: int,
    fixed_dt: bool,
    dt_min: float,
    dt_max: float | None,
    cfl: float,
    cfl_fac: float | None,
    real_dtype: Any,
    resolve_cfl_fac_fn: Callable[[str, float | None], float],
) -> _TimeStepLimits:
    dt_init = jnp.asarray(dt, dtype=real_dtype)
    progress_total = (
        jnp.asarray(float(steps) * float(dt), dtype=real_dtype)
        if fixed_dt
        else jnp.asarray(jnp.nan, dtype=real_dtype)
    )
    return _TimeStepLimits(
        dt_init=dt_init,
        progress_total=progress_total,
        dt_min=jnp.asarray(dt_min, dtype=real_dtype),
        dt_max=jnp.asarray(dt if dt_max is None else dt_max, dtype=real_dtype),
        cfl=jnp.asarray(cfl, dtype=real_dtype),
        cfl_fac=jnp.asarray(resolve_cfl_fac_fn(method, cfl_fac), dtype=real_dtype),
    )


def _build_nonlinear_cfl_bounds(
    grid: SpectralGrid,
    geom: Any,
    params: LinearParams,
    cache: LinearCache,
    *,
    real_dtype: Any,
    linear_frequency_bound_fn: Callable[..., Any],
    laguerre_velocity_max_fn: Callable[[int], float],
) -> _NonlinearCFLBounds:
    nx = int(grid.kx.size)
    ny = int(grid.ky.size)
    kx = jnp.asarray(cache.kx, dtype=real_dtype)
    ky = jnp.asarray(cache.ky, dtype=real_dtype)
    nl = int(cache.l.shape[0])
    nm = int(cache.m.shape[1])
    vtmax = jnp.max(jnp.abs(jnp.asarray(params.vth, dtype=real_dtype)))
    tzmax = jnp.max(jnp.abs(jnp.asarray(params.tz, dtype=real_dtype)))
    return _NonlinearCFLBounds(
        kx_max=jnp.abs(kx[(nx - 1) // 3]) if nx > 1 else jnp.zeros((), real_dtype),
        ky_max=jnp.abs(ky[(ny - 1) // 3]) if ny > 1 else jnp.zeros((), real_dtype),
        vpar_max=2.0 * jnp.sqrt(jnp.asarray(max(nm, 1), real_dtype)) * vtmax,
        muB_max=jnp.asarray(laguerre_velocity_max_fn(nl), real_dtype) * tzmax,
        kxfac=jnp.asarray(cache.kxfac, dtype=real_dtype),
        linear_omega=jnp.asarray(
            linear_frequency_bound_fn(
                grid,
                geom,
                params,
                nl,
                nm,
                include_diamagnetic_drive=False,
            ),
            dtype=real_dtype,
        ),
    )


def _make_nonlinear_dt_update(
    grid: SpectralGrid,
    cache: LinearCache,
    *,
    fixed_dt: bool,
    compressed_real_fft: bool,
    real_dtype: Any,
    limits: _TimeStepLimits,
    bounds: _NonlinearCFLBounds,
    cfl_frequency_components_fn: Callable[..., tuple[jnp.ndarray, jnp.ndarray]],
) -> Callable[[FieldState, jnp.ndarray], jnp.ndarray]:
    def update_dt(fields_state: FieldState, dt_prev: jnp.ndarray) -> jnp.ndarray:
        if fixed_dt:
            return jnp.asarray(dt_prev, dtype=real_dtype)
        omega_nl_x, omega_nl_y = cfl_frequency_components_fn(
            fields_state,
            grid,
            cache,
            compressed_real_fft=compressed_real_fft,
            kx_max=bounds.kx_max,
            ky_max=bounds.ky_max,
            kxfac=bounds.kxfac,
            vpar_max=bounds.vpar_max,
            muB_max=bounds.muB_max,
        )
        wmax = (
            jnp.maximum(bounds.linear_omega[0], omega_nl_x)
            + jnp.maximum(bounds.linear_omega[1], omega_nl_y)
            + bounds.linear_omega[2]
        )
        dt_guess = jnp.where(wmax > 0.0, limits.cfl_fac * limits.cfl / wmax, dt_prev)
        return jnp.asarray(
            jnp.clip(dt_guess, limits.dt_min, limits.dt_max), dtype=real_dtype
        )

    return update_dt


def build_nonlinear_time_step_policy(
    grid: SpectralGrid,
    geom: Any,
    params: LinearParams,
    cache: LinearCache,
    *,
    method: str,
    dt: float,
    steps: int,
    fixed_dt: bool,
    dt_min: float,
    dt_max: float | None,
    cfl: float,
    cfl_fac: float | None,
    compressed_real_fft: bool,
    real_dtype: Any,
    resolve_cfl_fac_fn: Callable[[str, float | None], float],
    linear_frequency_bound_fn: Callable[..., Any],
    laguerre_velocity_max_fn: Callable[[int], float],
    cfl_frequency_components_fn: Callable[..., tuple[jnp.ndarray, jnp.ndarray]],
) -> NonlinearTimeStepPolicy:
    """Build the fixed/adaptive nonlinear time-step update policy."""

    limits = _build_time_step_limits(
        method=method,
        dt=dt,
        steps=steps,
        fixed_dt=fixed_dt,
        dt_min=dt_min,
        dt_max=dt_max,
        cfl=cfl,
        cfl_fac=cfl_fac,
        real_dtype=real_dtype,
        resolve_cfl_fac_fn=resolve_cfl_fac_fn,
    )
    if fixed_dt:

        def fixed_update(_fields: FieldState, dt_prev: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(dt_prev, dtype=real_dtype)

        return NonlinearTimeStepPolicy(
            dt_init=limits.dt_init,
            progress_total=limits.progress_total,
            update_dt=fixed_update,
        )
    bounds = _build_nonlinear_cfl_bounds(
        grid,
        geom,
        params,
        cache,
        real_dtype=real_dtype,
        linear_frequency_bound_fn=linear_frequency_bound_fn,
        laguerre_velocity_max_fn=laguerre_velocity_max_fn,
    )
    update_dt = _make_nonlinear_dt_update(
        grid,
        cache,
        fixed_dt=fixed_dt,
        compressed_real_fft=compressed_real_fft,
        real_dtype=real_dtype,
        limits=limits,
        bounds=bounds,
        cfl_frequency_components_fn=cfl_frequency_components_fn,
    )

    return NonlinearTimeStepPolicy(
        dt_init=limits.dt_init,
        progress_total=limits.progress_total,
        update_dt=update_dt,
    )


def _compressed_real_cfl_gradient(
    field: jnp.ndarray,
    *,
    nyc: int,
    kx_b: jnp.ndarray,
    ky_b: jnp.ndarray,
    imag: jnp.ndarray,
    ifft_scale: jnp.ndarray,
    grid: SpectralGrid,
    use_batched_fft: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return physical-space gradients for an rFFT-compressed spectral field."""

    field_nyc = field[:nyc, :, :]
    if use_batched_fft:
        grad = jnp.stack([imag * kx_b * field_nyc, imag * ky_b * field_nyc], axis=0)
        grad = (
            jnp.fft.irfft2(grad, s=(grid.kx.size, grid.ky.size), axes=(-2, -3))
            * ifft_scale
        )
        return grad[0], grad[1]
    dfdx = jnp.fft.irfft2(
        imag * kx_b * field_nyc,
        s=(grid.kx.size, grid.ky.size),
        axes=(-2, -3),
    )
    dfdy = jnp.fft.irfft2(
        imag * ky_b * field_nyc,
        s=(grid.kx.size, grid.ky.size),
        axes=(-2, -3),
    )
    return dfdx * ifft_scale, dfdy * ifft_scale


def _full_complex_cfl_gradient(
    field: jnp.ndarray,
    *,
    kx_b: jnp.ndarray,
    ky_b: jnp.ndarray,
    imag: jnp.ndarray,
    ifft_scale: jnp.ndarray,
    use_batched_fft: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return physical-space gradients for a full complex spectral field."""

    if use_batched_fft:
        grad = (
            _ifft2_xy(jnp.stack([imag * kx_b * field, imag * ky_b * field], axis=0))
            * ifft_scale
        )
        return grad[0], grad[1]
    dfdx = _ifft2_xy(imag * kx_b * field) * ifft_scale
    dfdy = _ifft2_xy(imag * ky_b * field) * ifft_scale
    return dfdx, dfdy


def _field_gradient_for_cfl(
    field: jnp.ndarray,
    *,
    compressed_real_fft: bool,
    nyc: int,
    kx_b: jnp.ndarray,
    ky_b: jnp.ndarray,
    imag: jnp.ndarray,
    ifft_scale: jnp.ndarray,
    grid: SpectralGrid,
    use_batched_fft: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Dispatch the nonlinear CFL gradient transform for the current layout."""

    if compressed_real_fft:
        return _compressed_real_cfl_gradient(
            field,
            nyc=nyc,
            kx_b=kx_b,
            ky_b=ky_b,
            imag=imag,
            ifft_scale=ifft_scale,
            grid=grid,
            use_batched_fft=use_batched_fft,
        )
    return _full_complex_cfl_gradient(
        field,
        kx_b=kx_b,
        ky_b=ky_b,
        imag=imag,
        ifft_scale=ifft_scale,
        use_batched_fft=use_batched_fft,
    )


def _accumulate_electromagnetic_cfl_gradients(
    dphi_dx: jnp.ndarray,
    dphi_dy: jnp.ndarray,
    fields: FieldState,
    *,
    grad_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    vpar_max: float,
    muB_max: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Add optional Apar/Bpar gradient speeds to the electrostatic CFL speeds."""

    speed_x = jnp.abs(dphi_dx)
    speed_y = jnp.abs(dphi_dy)
    if fields.apar is not None:
        dap_dx, dap_dy = grad_fn(fields.apar)
        speed_x = speed_x + vpar_max * jnp.abs(dap_dx)
        speed_y = speed_y + vpar_max * jnp.abs(dap_dy)
    if fields.bpar is not None:
        dbp_dx, dbp_dy = grad_fn(fields.bpar)
        speed_x = speed_x + muB_max * jnp.abs(dbp_dx)
        speed_y = speed_y + muB_max * jnp.abs(dbp_dy)
    return speed_x, speed_y


def _cfl_frequencies_from_physical_speeds(
    speed_x: jnp.ndarray,
    speed_y: jnp.ndarray,
    *,
    real_dtype: jnp.dtype,
    kxfac: jnp.ndarray,
    kx_max: float,
    ky_max: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reduce physical-space nonlinear speeds to x/y CFL frequencies."""

    vmax_x = jnp.max(speed_y)
    vmax_y = jnp.max(speed_x)
    scale = jnp.asarray(0.5, dtype=real_dtype)
    omega_x = jnp.abs(kxfac) * jnp.asarray(kx_max, dtype=real_dtype) * vmax_x * scale
    omega_y = jnp.abs(kxfac) * jnp.asarray(ky_max, dtype=real_dtype) * vmax_y * scale
    return jnp.asarray(omega_x, dtype=real_dtype), jnp.asarray(
        omega_y, dtype=real_dtype
    )


def _nonlinear_cfl_frequency_components(
    fields: FieldState,
    grid: SpectralGrid,
    cache: LinearCache,
    *,
    compressed_real_fft: bool,
    kx_max: float,
    ky_max: float,
    kxfac: float,
    vpar_max: float,
    muB_max: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Nonlinear x/y CFL frequency components from grad(phi, apar, bpar)."""

    phi = fields.phi

    ny = int(grid.ky.size)
    nyc = 1 + ny // 2

    real_dtype = jnp.real(jnp.empty((), dtype=phi.dtype)).dtype
    kxfac_val = jnp.asarray(kxfac, dtype=real_dtype)
    imag = jnp.asarray(1j, dtype=phi.dtype)

    fft_norm = float(grid.ky.size * grid.kx.size)
    ifft_scale = jnp.asarray(fft_norm, dtype=real_dtype)
    use_batched_fft = jax.default_backend() != "cpu"

    if compressed_real_fft:
        _, ky_vals, kx_nyc, ky_nyc = real_fft_mesh(cache.kx_grid, cache.ky_grid)
        nyc = int(ky_vals.shape[0])
        kx_b = _broadcast_grid(kx_nyc, phi[:nyc, :, :].ndim)
        ky_b = _broadcast_grid(ky_nyc, phi[:nyc, :, :].ndim)
    else:
        kx_b = _broadcast_grid(cache.kx_grid, phi.ndim)
        ky_b = _broadcast_grid(cache.ky_grid, phi.ndim)

    def grad_fn(field: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _field_gradient_for_cfl(
            field,
            compressed_real_fft=compressed_real_fft,
            nyc=nyc,
            kx_b=kx_b,
            ky_b=ky_b,
            imag=imag,
            ifft_scale=ifft_scale,
            grid=grid,
            use_batched_fft=use_batched_fft,
        )

    dphi_dx, dphi_dy = grad_fn(phi)
    speed_x, speed_y = _accumulate_electromagnetic_cfl_gradients(
        dphi_dx,
        dphi_dy,
        fields,
        grad_fn=grad_fn,
        vpar_max=vpar_max,
        muB_max=muB_max,
    )
    return _cfl_frequencies_from_physical_speeds(
        speed_x,
        speed_y,
        real_dtype=real_dtype,
        kxfac=kxfac_val,
        kx_max=kx_max,
        ky_max=ky_max,
    )


def _diagnostic_omega_mode_mask(
    grid: SpectralGrid,
    cache: LinearCache,
    *,
    compressed_real_fft: bool,
) -> jnp.ndarray:
    """Mask used to reduce mode-wise nonlinear omega/gamma diagnostics."""

    ny = int(grid.ky.size)
    nx = int(grid.kx.size)
    if compressed_real_fft and bool(np.any(np.asarray(grid.ky) < 0.0)):
        # Full-ky SPECTRAX layout stores the rFFT-unique modes in the first
        # Ny//2+1 entries, including the Nyquist row when Ny is even.
        ky_unique = jnp.arange(ny, dtype=jnp.int32)[:, None] < (ny // 2 + 1)
    else:
        ky_unique = jnp.asarray(cache.ky)[:, None] >= 0.0
    return jnp.asarray(grid.dealias_mask, dtype=bool) & jnp.broadcast_to(
        ky_unique, (ny, nx)
    )


def build_nonlinear_imex_operator(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    *,
    terms: TermConfig | None = None,
    implicit_preconditioner: str | None = None,
    compressed_real_fft: bool = True,
    build_implicit_operator_fn: Callable[..., tuple[Any, ...]] | None = None,
) -> IMEXLinearOperator:
    """Build and cache the matrix-free linear operator used by nonlinear IMEX."""

    del compressed_real_fft
    term_cfg = terms or TermConfig()
    linear_terms = term_config_to_linear_terms(term_cfg)
    if build_implicit_operator_fn is None:
        build_implicit_operator_fn = _build_implicit_operator
    G, shape, _size, dt_val, precond_op, matvec, squeeze_species = (
        build_implicit_operator_fn(
            G0,
            cache,
            params,
            dt,
            linear_terms,
            implicit_preconditioner,
        )
    )
    return IMEXLinearOperator(
        state_dtype=G.dtype,
        shape=shape,
        dt_val=dt_val,
        precond_op=precond_op,
        matvec=matvec,
        squeeze_species=squeeze_species,
    )
