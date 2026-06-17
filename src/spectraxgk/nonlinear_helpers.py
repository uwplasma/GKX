"""Helper policies and operators for nonlinear gyrokinetic drivers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import jax
import numpy as np

from spectraxgk.grids import SpectralGrid, real_fft_mesh
from spectraxgk.solvers.linear.implicit import _build_implicit_operator
from spectraxgk.operators.linear.cache import (
    LinearCache,
    collision_damping as _base_collision_damping,
    hypercollision_damping,
)
from spectraxgk.operators.linear.params import (
    LinearParams,
    term_config_to_linear_terms,
)
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.nonlinear import _broadcast_grid, _ifft2_xy

__all__ = [
    "IMEXLinearOperator",
    "NonlinearDiagnosticSetup",
    "NonlinearTimeStepPolicy",
    "_apply_collision_split",
    "_collision_damping",
    "_nonlinear_cfl_frequency_components",
    "_diagnostic_omega_mode_mask",
    "_make_fixed_mode_projector",
    "_make_hermitian_projector",
    "_make_nonlinear_state_projector",
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

    dt_init = jnp.asarray(dt, dtype=real_dtype)
    progress_total = (
        jnp.asarray(float(steps) * float(dt), dtype=real_dtype)
        if fixed_dt
        else jnp.asarray(jnp.nan, dtype=real_dtype)
    )
    dt_min_val = jnp.asarray(dt_min, dtype=real_dtype)
    # Explicit-time default behavior: when dt_max is unset, dt_max == dt.
    dt_max_val = jnp.asarray(dt if dt_max is None else dt_max, dtype=real_dtype)
    cfl_val = jnp.asarray(cfl, dtype=real_dtype)
    cfl_fac_val = jnp.asarray(resolve_cfl_fac_fn(method, cfl_fac), dtype=real_dtype)

    nx = int(grid.kx.size)
    ny = int(grid.ky.size)
    kx_np = np.asarray(cache.kx, dtype=float)
    ky_np = np.asarray(cache.ky, dtype=float)
    kx_max = float(abs(kx_np[(nx - 1) // 3])) if nx > 1 else 0.0
    ky_max = float(abs(ky_np[(ny - 1) // 3])) if ny > 1 else 0.0
    vtmax = float(np.max(np.abs(np.asarray(params.vth, dtype=float))))
    tzmax = float(np.max(np.abs(np.asarray(params.tz, dtype=float))))
    nl = int(cache.l.shape[0])
    nm = int(cache.m.shape[1])
    vpar_max = 2.0 * float(np.sqrt(max(nm, 1))) * vtmax
    muB_max = laguerre_velocity_max_fn(nl) * tzmax
    kxfac_val = float(np.asarray(cache.kxfac))
    linear_omega = jnp.asarray(
        linear_frequency_bound_fn(
            grid,
            geom,
            params,
            nl,
            nm,
            include_diamagnetic_drive=False,
        ),
        dtype=real_dtype,
    )

    def update_dt(fields_state: FieldState, dt_prev: jnp.ndarray) -> jnp.ndarray:
        if fixed_dt:
            return jnp.asarray(dt_prev, dtype=real_dtype)
        omega_nl_x, omega_nl_y = cfl_frequency_components_fn(
            fields_state,
            grid,
            cache,
            compressed_real_fft=compressed_real_fft,
            kx_max=kx_max,
            ky_max=ky_max,
            kxfac=kxfac_val,
            vpar_max=vpar_max,
            muB_max=muB_max,
        )
        wmax = (
            jnp.maximum(linear_omega[0], omega_nl_x)
            + jnp.maximum(linear_omega[1], omega_nl_y)
            + linear_omega[2]
        )
        dt_guess = jnp.where(wmax > 0.0, cfl_fac_val * cfl_val / wmax, dt_prev)
        return jnp.asarray(jnp.clip(dt_guess, dt_min_val, dt_max_val), dtype=real_dtype)

    return NonlinearTimeStepPolicy(
        dt_init=dt_init,
        progress_total=progress_total,
        update_dt=update_dt,
    )


def _make_hermitian_projector(
    ky_vals: np.ndarray, nx: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Project full-ky states onto the compressed real-FFT Hermitian manifold."""

    ny_full = int(ky_vals.size)
    nyc = ny_full // 2 + 1
    use_hermitian = nyc > 2 and bool(np.any(np.asarray(ky_vals) < 0.0))
    if not use_hermitian:
        return lambda G_state: G_state

    neg_hi = nyc - 1 if (ny_full % 2 == 0) else nyc
    if nx > 1:
        kx_neg = jnp.asarray(
            np.concatenate(([0], np.arange(nx - 1, 0, -1))), dtype=jnp.int32
        )
    else:
        kx_neg = None

    def project(G_state: jnp.ndarray) -> jnp.ndarray:
        pos = G_state[..., :nyc, :, :]
        neg = jnp.conj(pos[..., 1:neg_hi, :, :])[..., ::-1, :, :]
        if kx_neg is not None:
            neg = neg[..., kx_neg, :]
        return jnp.concatenate([pos, neg], axis=-3)

    return project


def _make_nonlinear_state_projector(
    fixed_state: jnp.ndarray | None,
    *,
    ky_vals: np.ndarray,
    nx: int,
    compressed_real_fft: bool,
    fixed_mode_ky_index: int | None,
    fixed_mode_kx_index: int | None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Compose fixed-mode and Hermitian projections for nonlinear state scans."""

    fixed_projector = _make_fixed_mode_projector(
        fixed_state,
        ky_index=fixed_mode_ky_index,
        kx_index=fixed_mode_kx_index,
    )
    hermitian_projector = (
        _make_hermitian_projector(np.asarray(ky_vals), nx=int(nx))
        if compressed_real_fft
        else (lambda G_state: G_state)
    )

    def project(G_state: jnp.ndarray) -> jnp.ndarray:
        if fixed_projector is not None:
            G_state = fixed_projector(G_state)
        return hermitian_projector(G_state)

    return project


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
    apar = fields.apar
    bpar = fields.bpar

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
        phi_nyc = phi[:nyc, :, :]
        kx_b = _broadcast_grid(kx_nyc, phi_nyc.ndim)
        ky_b = _broadcast_grid(ky_nyc, phi_nyc.ndim)
        if use_batched_fft:
            grad_phi = jnp.stack(
                [imag * kx_b * phi_nyc, imag * ky_b * phi_nyc], axis=0
            )
            grad_phi = (
                jnp.fft.irfft2(
                    grad_phi, s=(grid.kx.size, grid.ky.size), axes=(-2, -3)
                )
                * ifft_scale
            )
            dphi_dx = grad_phi[0]
            dphi_dy = grad_phi[1]
        else:
            dphi_dx = jnp.fft.irfft2(
                imag * kx_b * phi_nyc,
                s=(grid.kx.size, grid.ky.size),
                axes=(-2, -3),
            )
            dphi_dy = jnp.fft.irfft2(
                imag * ky_b * phi_nyc,
                s=(grid.kx.size, grid.ky.size),
                axes=(-2, -3),
            )
            dphi_dx = dphi_dx * ifft_scale
            dphi_dy = dphi_dy * ifft_scale

        def _grad_real(field: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            field_nyc = field[:nyc, :, :]
            if use_batched_fft:
                grad = jnp.stack(
                    [imag * kx_b * field_nyc, imag * ky_b * field_nyc], axis=0
                )
                grad = (
                    jnp.fft.irfft2(
                        grad, s=(grid.kx.size, grid.ky.size), axes=(-2, -3)
                    )
                    * ifft_scale
                )
                return grad[0], grad[1]
            dfx = jnp.fft.irfft2(
                imag * kx_b * field_nyc,
                s=(grid.kx.size, grid.ky.size),
                axes=(-2, -3),
            )
            dfy = jnp.fft.irfft2(
                imag * ky_b * field_nyc,
                s=(grid.kx.size, grid.ky.size),
                axes=(-2, -3),
            )
            return dfx * ifft_scale, dfy * ifft_scale

    else:
        kx_b = _broadcast_grid(cache.kx_grid, phi.ndim)
        ky_b = _broadcast_grid(cache.ky_grid, phi.ndim)
        if use_batched_fft:
            grad_phi = (
                _ifft2_xy(jnp.stack([imag * kx_b * phi, imag * ky_b * phi], axis=0))
                * ifft_scale
            )
            dphi_dx = grad_phi[0]
            dphi_dy = grad_phi[1]
        else:
            dphi_dx = _ifft2_xy(imag * kx_b * phi) * ifft_scale
            dphi_dy = _ifft2_xy(imag * ky_b * phi) * ifft_scale

        def _grad_real(field: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            if use_batched_fft:
                grad = (
                    _ifft2_xy(
                        jnp.stack([imag * kx_b * field, imag * ky_b * field], axis=0)
                    )
                    * ifft_scale
                )
                return grad[0], grad[1]
            dfx = _ifft2_xy(imag * kx_b * field) * ifft_scale
            dfy = _ifft2_xy(imag * ky_b * field) * ifft_scale
            return dfx, dfy

    dphi_dx = jnp.abs(dphi_dx)
    dphi_dy = jnp.abs(dphi_dy)

    if apar is not None:
        dap_dx, dap_dy = _grad_real(apar)
        dphi_dx = dphi_dx + vpar_max * jnp.abs(dap_dx)
        dphi_dy = dphi_dy + vpar_max * jnp.abs(dap_dy)
    if bpar is not None:
        dbp_dx, dbp_dy = _grad_real(bpar)
        dphi_dx = dphi_dx + muB_max * jnp.abs(dbp_dx)
        dphi_dy = dphi_dy + muB_max * jnp.abs(dbp_dy)

    vmax_x = jnp.max(dphi_dy)
    vmax_y = jnp.max(dphi_dx)
    scale = jnp.asarray(0.5, dtype=real_dtype)
    omega_x = (
        jnp.abs(kxfac_val) * jnp.asarray(kx_max, dtype=real_dtype) * vmax_x * scale
    )
    omega_y = (
        jnp.abs(kxfac_val) * jnp.asarray(ky_max, dtype=real_dtype) * vmax_y * scale
    )
    return jnp.asarray(omega_x, dtype=real_dtype), jnp.asarray(
        omega_y, dtype=real_dtype
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


def _collision_damping(
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    real_dtype: jnp.dtype,
    *,
    squeeze_species: bool,
) -> jnp.ndarray:
    """Assemble collision + hypercollision damping for operator splitting."""

    damping = _base_collision_damping(
        cache, params, real_dtype, squeeze_species=squeeze_species
    )
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    coll_w = jnp.asarray(term_cfg.collisions, dtype=real_dtype)
    hyper_w = jnp.asarray(term_cfg.hypercollisions, dtype=real_dtype)
    if squeeze_species and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]

    damping = coll_w * damping + hyper_w * hyper_damp
    return damping.astype(real_dtype)


def _apply_collision_split(
    G: jnp.ndarray,
    damping: jnp.ndarray,
    dt_local: jnp.ndarray,
    scheme: str,
) -> jnp.ndarray:
    """Apply a diagonal collision/hypercollision split update."""

    scheme_key = scheme.strip().lower()
    if scheme_key in {"implicit", "imex"}:
        return G / (1.0 + dt_local * damping)
    if scheme_key in {"exp", "sts", "rkc", "rkc2"}:
        # For diagonal collision operators the exponential update is exact and
        # behaves like a stabilized explicit (STS/RKC) limit.
        return G * jnp.exp(-dt_local * damping)
    raise ValueError("collision_scheme must be one of {'implicit', 'exp', 'sts', 'rkc'}")


def _make_fixed_mode_projector(
    fixed_state: jnp.ndarray | None,
    *,
    ky_index: int | None,
    kx_index: int | None,
) -> Callable[[jnp.ndarray], jnp.ndarray] | None:
    """Return a projector that keeps one Fourier mode equal to ``fixed_state``."""

    if fixed_state is None or ky_index is None or kx_index is None:
        return None
    ky_i = int(ky_index)
    kx_i = int(kx_index)
    fixed_block = jnp.asarray(fixed_state)[..., ky_i : ky_i + 1, kx_i : kx_i + 1, :]

    def project(G_state: jnp.ndarray) -> jnp.ndarray:
        return G_state.at[..., ky_i : ky_i + 1, kx_i : kx_i + 1, :].set(
            fixed_block
        )

    return project


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
