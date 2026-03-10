"""Nonlinear gyrokinetic drivers built on term-wise RHS assembly."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.config import resolve_cfl_fac
from spectraxgk.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from spectraxgk.grids import SpectralGrid, gx_real_fft_mesh
from spectraxgk.linear import (
    LinearCache,
    LinearParams,
    _build_implicit_operator,
    build_linear_cache,
    hypercollision_damping,
    term_config_to_linear_terms,
)
from spectraxgk.terms.assembly import assemble_rhs_cached_jit, compute_fields_cached
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.integrators import integrate_nonlinear as integrate_nonlinear_scan
from spectraxgk.terms.nonlinear import _broadcast_grid, _ifft2_xy, nonlinear_em_contribution
from spectraxgk.gx_integrators import _gx_growth_rate_step, _gx_midplane_index
from spectraxgk.diagnostics import (
    GXDiagnostics,
    gx_energy_total,
    gx_heat_flux,
    gx_heat_flux_species,
    gx_particle_flux,
    gx_particle_flux_species,
    gx_volume_factors,
    gx_Wapar,
    gx_Wg,
    gx_Wphi,
)
from spectraxgk.gx_integrators import _gx_laguerre_vmax

_SSPX3_ADT = float((1.0 / 6.0) ** (1.0 / 3.0))
_SSPX3_WGTFAC = float((9.0 - 2.0 * (6.0 ** (2.0 / 3.0))) ** 0.5)
_SSPX3_W1 = 0.5 * (_SSPX3_WGTFAC - 1.0)
_SSPX3_W2 = 0.5 * ((6.0 ** (2.0 / 3.0)) - 1.0 - _SSPX3_WGTFAC)
_SSPX3_W3 = (1.0 / _SSPX3_ADT) - 1.0 - _SSPX3_W2 * (_SSPX3_W1 + 1.0)


@dataclass(frozen=True)
class IMEXLinearOperator:
    """Reusable matrix-free linear operator for nonlinear IMEX solves."""

    state_dtype: jnp.dtype
    shape: tuple[int, ...]
    dt_val: jnp.ndarray
    precond_op: Callable[[jnp.ndarray], jnp.ndarray] | None
    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    squeeze_species: bool


def nonlinear_rhs_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: TermConfig | None = None,
    *,
    gx_real_fft: bool = True,
    laguerre_mode: str = "grid",
) -> Tuple[jnp.ndarray, FieldState]:
    """Compute a nonlinear RHS using linear terms plus a placeholder nonlinear term."""

    term_cfg = terms or TermConfig()
    dG, fields = assemble_rhs_cached_jit(G, cache, params, term_cfg)
    if term_cfg.nonlinear != 0.0:
        real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
        weight = jnp.asarray(term_cfg.nonlinear, dtype=real_dtype)
        dG = dG + nonlinear_em_contribution(
            G,
            phi=fields.phi,
            apar=fields.apar,
            bpar=fields.bpar,
            Jl=cache.Jl,
            JlB=cache.JlB,
            tz=jnp.asarray(params.tz),
            vth=jnp.asarray(params.vth),
            sqrt_m=cache.sqrt_m,
            sqrt_m_p1=cache.sqrt_m_p1,
            kx_grid=cache.kx_grid,
            ky_grid=cache.ky_grid,
            dealias_mask=cache.dealias_mask,
            kxfac=cache.kxfac,
            weight=weight,
            apar_weight=float(term_cfg.apar),
            bpar_weight=float(term_cfg.bpar),
            laguerre_to_grid=cache.laguerre_to_grid,
            laguerre_to_spectral=cache.laguerre_to_spectral,
            laguerre_roots=cache.laguerre_roots,
            laguerre_j0=cache.laguerre_j0,
            laguerre_j1_over_alpha=cache.laguerre_j1_over_alpha,
            b=cache.b,
            gx_real_fft=gx_real_fft,
            laguerre_mode=laguerre_mode,
        )
    return dG, fields


def _make_hermitian_projector(ky_vals: np.ndarray, nx: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Project full-ky states onto the GX real-FFT Hermitian manifold."""

    ny_full = int(ky_vals.size)
    nyc = ny_full // 2 + 1
    use_hermitian = nyc > 2 and bool(np.any(np.asarray(ky_vals) < 0.0))
    if not use_hermitian:
        return lambda G_state: G_state

    neg_hi = nyc - 1 if (ny_full % 2 == 0) else nyc
    if nx > 1:
        kx_neg = jnp.asarray(np.concatenate(([0], np.arange(nx - 1, 0, -1))), dtype=jnp.int32)
    else:
        kx_neg = None

    def project(G_state: jnp.ndarray) -> jnp.ndarray:
        pos = G_state[..., :nyc, :, :]
        neg = jnp.conj(pos[..., 1:neg_hi, :, :])[..., ::-1, :, :]
        if kx_neg is not None:
            neg = neg[..., kx_neg, :]
        return jnp.concatenate([pos, neg], axis=-3)

    return project


def _gx_nonlinear_omega_max(
    fields: FieldState,
    grid: SpectralGrid,
    cache: LinearCache,
    *,
    gx_real_fft: bool,
    kx_max: float,
    ky_max: float,
    kxfac: float,
    vpar_max: float,
    muB_max: float,
) -> jnp.ndarray:
    """GX-style nonlinear max frequency estimate from grad(phi,apar,bpar)."""

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

    if gx_real_fft:
        _, ky_vals, kx_nyc, ky_nyc = gx_real_fft_mesh(cache.kx_grid, cache.ky_grid)
        nyc = int(ky_vals.shape[0])
        phi_nyc = phi[:nyc, :, :]
        kx_b = _broadcast_grid(kx_nyc, phi_nyc.ndim)
        ky_b = _broadcast_grid(ky_nyc, phi_nyc.ndim)
        dphi_dx = jnp.fft.irfft2(imag * kx_b * phi_nyc, s=(grid.kx.size, grid.ky.size), axes=(-2, -3))
        dphi_dy = jnp.fft.irfft2(imag * ky_b * phi_nyc, s=(grid.kx.size, grid.ky.size), axes=(-2, -3))
        dphi_dx = dphi_dx * ifft_scale
        dphi_dy = dphi_dy * ifft_scale

        def _grad_real(field: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            field_nyc = field[:nyc, :, :]
            dfx = jnp.fft.irfft2(imag * kx_b * field_nyc, s=(grid.kx.size, grid.ky.size), axes=(-2, -3))
            dfy = jnp.fft.irfft2(imag * ky_b * field_nyc, s=(grid.kx.size, grid.ky.size), axes=(-2, -3))
            return dfx * ifft_scale, dfy * ifft_scale
    else:
        kx_b = _broadcast_grid(cache.kx_grid, phi.ndim)
        ky_b = _broadcast_grid(cache.ky_grid, phi.ndim)
        dphi_dx = _ifft2_xy(imag * kx_b * phi) * ifft_scale
        dphi_dy = _ifft2_xy(imag * ky_b * phi) * ifft_scale

        def _grad_real(field: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    omega_x = jnp.abs(kxfac_val) * jnp.asarray(kx_max, dtype=real_dtype) * vmax_x * scale
    omega_y = jnp.abs(kxfac_val) * jnp.asarray(ky_max, dtype=real_dtype) * vmax_y * scale
    return jnp.asarray(omega_x + omega_y, dtype=real_dtype)


def _gx_omega_mode_mask(
    grid: SpectralGrid,
    cache: LinearCache,
    *,
    gx_real_fft: bool,
) -> jnp.ndarray:
    """Mask used to reduce mode-wise GX omega/gamma diagnostics."""

    ny = int(grid.ky.size)
    nx = int(grid.kx.size)
    if gx_real_fft and bool(np.any(np.asarray(grid.ky) < 0.0)):
        # Full-ky SPECTRAX layout stores the rFFT-unique modes in the first
        # Ny//2+1 entries, including the Nyquist row when Ny is even.
        ky_unique = jnp.arange(ny, dtype=jnp.int32)[:, None] < (ny // 2 + 1)
    else:
        ky_unique = jnp.asarray(cache.ky)[:, None] >= 0.0
    return jnp.asarray(grid.dealias_mask, dtype=bool) & jnp.broadcast_to(ky_unique, (ny, nx))


def _collision_damping(
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    real_dtype: jnp.dtype,
    *,
    squeeze_species: bool,
) -> jnp.ndarray:
    """Assemble collision + hypercollision damping for operator splitting."""

    lb_lam = cache.lb_lam.astype(real_dtype)
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    coll_w = jnp.asarray(term_cfg.collisions, dtype=real_dtype)
    hyper_w = jnp.asarray(term_cfg.hypercollisions, dtype=real_dtype)

    if lb_lam.ndim == 6:
        ns = lb_lam.shape[0]
        nu = jnp.asarray(params.nu, dtype=real_dtype)
        if nu.ndim == 0:
            nu = jnp.broadcast_to(nu, (ns,))
        damping = nu[:, None, None, None, None, None] * lb_lam
        if squeeze_species:
            damping = damping[0]
            hyper_damp = hyper_damp[0]
    else:
        damping = jnp.asarray(params.nu, dtype=real_dtype) * lb_lam

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
        return G_state.at[..., ky_i : ky_i + 1, kx_i : kx_i + 1, :].set(fixed_block)

    return project


def integrate_nonlinear_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    gx_real_fft: bool = True,
    laguerre_mode: str = "grid",
) -> tuple[jnp.ndarray, FieldState]:
    """Integrate the nonlinear system using a cached geometry object."""

    term_cfg = terms or TermConfig()
    if method in {"imex", "semi-implicit"}:
        return integrate_nonlinear_imex_cached(
            G0,
            cache,
            params,
            dt,
            steps,
            terms=term_cfg,
            checkpoint=checkpoint,
            gx_real_fft=gx_real_fft,
            laguerre_mode=laguerre_mode,
        )

    def rhs_fn(G):
        return nonlinear_rhs_cached(
            G,
            cache,
            params,
            term_cfg,
            gx_real_fft=gx_real_fft,
            laguerre_mode=laguerre_mode,
        )

    project_state = None
    if gx_real_fft:
        project_state = _make_hermitian_projector(np.asarray(cache.ky), int(np.asarray(cache.kx).size))

    return integrate_nonlinear_scan(
        rhs_fn,
        G0,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
        project_state=project_state,
    )


def integrate_nonlinear(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    gx_real_fft: bool = True,
    laguerre_mode: str = "grid",
) -> tuple[jnp.ndarray, FieldState]:
    """Integrate the nonlinear system using built-in cache construction."""

    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError("G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
        cache = build_linear_cache(grid, geom_eff, params, Nl, Nm)
    return integrate_nonlinear_cached(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        terms=terms,
        checkpoint=checkpoint,
        gx_real_fft=gx_real_fft,
        laguerre_mode=laguerre_mode,
    )


def _integrate_nonlinear_gx_diagnostics_impl(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "rk3",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    diagnostics_stride: int = 1,
    use_dealias_mask: bool = False,
    z_index: int | None = None,
    gx_real_fft: bool = True,
    laguerre_mode: str = "grid",
    omega_ky_index: int | None = None,
    omega_kx_index: int | None = None,
    flux_scale: float = 1.0,
    wphi_scale: float = 1.0,
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    cfl: float = 0.9,
    cfl_fac: float | None = None,
    collision_split: bool = False,
    collision_scheme: str = "implicit",
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: str | None = None,
    fixed_mode_ky_index: int | None = None,
    fixed_mode_kx_index: int | None = None,
) -> tuple[jnp.ndarray, GXDiagnostics, jnp.ndarray, FieldState]:
    """Integrate nonlinear system and return GX-style diagnostics plus final state."""

    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError("G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
        cache = build_linear_cache(grid, geom_eff, params, Nl, Nm)

    term_cfg = terms or TermConfig()
    if method in {"imex", "semi-implicit"}:
        raise ValueError("Final-state GX diagnostics helper only supports explicit methods")
    vol_fac, flux_fac = gx_volume_factors(geom_eff, grid)
    mask = _gx_omega_mode_mask(grid, cache, gx_real_fft=gx_real_fft)
    z_idx = _gx_midplane_index(grid.z.size) if z_index is None else int(z_index)
    use_dealias = bool(use_dealias_mask)
    use_hermitian = bool(gx_real_fft) and bool(np.any(np.asarray(grid.ky) < 0.0))
    ny_full = int(grid.ky.size)
    nyc = ny_full // 2 + 1
    nx = int(grid.kx.size)
    if nx > 1:
        kx_neg = jnp.concatenate(
            [jnp.asarray([0], dtype=jnp.int32), jnp.arange(nx - 1, 0, -1, dtype=jnp.int32)]
        )
    else:
        kx_neg = jnp.asarray([0], dtype=jnp.int32)

    fixed_projector = _make_fixed_mode_projector(
        G0,
        ky_index=fixed_mode_ky_index,
        kx_index=fixed_mode_kx_index,
    )

    def _project_state(G_state: jnp.ndarray) -> jnp.ndarray:
        if fixed_projector is not None:
            G_state = fixed_projector(G_state)
        if not use_hermitian or nyc <= 2:
            return G_state
        pos = G_state[..., :nyc, :, :]
        neg_hi = nyc - 1 if (ny_full % 2 == 0) else nyc
        neg = jnp.conj(pos[..., 1:neg_hi, :, :])[..., ::-1, :, :]
        if nx > 1:
            neg = neg[..., kx_neg, :]
        return jnp.concatenate([pos, neg], axis=-3)

    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    G0 = _project_state(G0)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_init = jnp.asarray(dt, dtype=real_dtype)
    dt_min_val = jnp.asarray(dt_min, dtype=real_dtype)
    # GX default behavior: when dt_max is unset, dt_max == dt.
    dt_max_val = jnp.asarray(dt if dt_max is None else dt_max, dtype=real_dtype)
    cfl_val = jnp.asarray(cfl, dtype=real_dtype)
    cfl_fac_val = jnp.asarray(resolve_cfl_fac(method, cfl_fac), dtype=real_dtype)

    nx = int(grid.kx.size)
    ny = int(grid.ky.size)
    kx_np = np.asarray(cache.kx, dtype=float)
    ky_np = np.asarray(cache.ky, dtype=float)
    kx_max = float(abs(kx_np[(nx - 1) // 3])) if nx > 1 else 0.0
    ky_max = float(abs(ky_np[(ny - 1) // 3])) if ny > 1 else 0.0
    vtmax = float(np.max(np.abs(np.asarray(params.vth, dtype=float))))
    tzmax = float(np.max(np.abs(np.asarray(params.tz, dtype=float))))
    nm = int(cache.sqrt_m.shape[0])
    nl = int(cache.l.shape[0])
    vpar_max = 2.0 * float(np.sqrt(max(nm, 1))) * vtmax
    muB_max = _gx_laguerre_vmax(nl) * tzmax
    kxfac_val = float(np.asarray(cache.kxfac))
    squeeze_species = G0.ndim == 5 and cache.lb_lam.ndim == 6
    use_collision_split = bool(collision_split) and (
        float(term_cfg.collisions) != 0.0 or float(term_cfg.hypercollisions) != 0.0
    )
    damping = None
    if use_collision_split:
        damping = _collision_damping(cache, params, term_cfg, real_dtype, squeeze_species=squeeze_species)

    def _update_dt(fields_state: FieldState, dt_prev: jnp.ndarray) -> jnp.ndarray:
        if fixed_dt:
            return jnp.asarray(dt_prev, dtype=real_dtype)
        wmax = _gx_nonlinear_omega_max(
            fields_state,
            grid,
            cache,
            gx_real_fft=gx_real_fft,
            kx_max=kx_max,
            ky_max=ky_max,
            kxfac=kxfac_val,
            vpar_max=vpar_max,
            muB_max=muB_max,
        )
        dt_guess = jnp.where(wmax > 0.0, cfl_fac_val * cfl_val / wmax, dt_prev)
        return jnp.asarray(jnp.clip(dt_guess, dt_min_val, dt_max_val), dtype=real_dtype)

    def rhs_fn(G):
        return nonlinear_rhs_cached(
            G,
            cache,
            params,
            term_cfg,
            gx_real_fft=gx_real_fft,
            laguerre_mode=laguerre_mode,
        )

    fields0 = compute_fields_cached(G0, cache, params, terms=term_cfg)
    phi_prev = fields0.phi

    def _compute_diag_from_state(
        G_state: jnp.ndarray,
        fields_state: FieldState,
        phi_prev_step: jnp.ndarray,
        dt_step: jnp.ndarray,
    ):
        phi = fields_state.phi
        apar = fields_state.apar if fields_state.apar is not None else jnp.zeros_like(phi)
        bpar = fields_state.bpar if fields_state.bpar is not None else jnp.zeros_like(phi)

        gamma_modes, omega_modes = _gx_growth_rate_step(
            phi, phi_prev_step, dt_step, z_index=z_idx, mask=mask
        )
        if omega_ky_index is not None:
            ky_i = int(np.clip(omega_ky_index, 0, int(gamma_modes.shape[0]) - 1))
            kx_i = int(np.clip(omega_kx_index or 0, 0, int(gamma_modes.shape[1]) - 1))
            gamma = jnp.nan_to_num(gamma_modes[ky_i, kx_i], nan=jnp.asarray(0.0, dtype=real_dtype))
            omega = jnp.nan_to_num(omega_modes[ky_i, kx_i], nan=jnp.asarray(0.0, dtype=real_dtype))
        else:
            gamma = jnp.nan_to_num(
                jnp.nanmean(jnp.where(mask, gamma_modes, jnp.nan)),
                nan=jnp.asarray(0.0, dtype=real_dtype),
            )
            omega = jnp.nan_to_num(
                jnp.nanmean(jnp.where(mask, omega_modes, jnp.nan)),
                nan=jnp.asarray(0.0, dtype=real_dtype),
            )
        Wg_val = gx_Wg(G_state, grid, params, vol_fac, use_dealias=use_dealias)
        Wphi_val = gx_Wphi(
            phi,
            cache,
            params,
            vol_fac,
            use_dealias=use_dealias,
            wphi_scale=wphi_scale,
        )
        Wapar_val = gx_Wapar(apar, cache, vol_fac, use_dealias=use_dealias)
        heat_species = gx_heat_flux_species(
            G_state,
            phi,
            apar,
            bpar,
            cache,
            grid,
            params,
            flux_fac,
            use_dealias=use_dealias,
            flux_scale=flux_scale,
        )
        pflux_species = gx_particle_flux_species(
            G_state,
            phi,
            apar,
            bpar,
            cache,
            grid,
            params,
            flux_fac,
            use_dealias=use_dealias,
            flux_scale=flux_scale,
        )
        heat_val = jnp.sum(heat_species)
        pflux_val = jnp.sum(pflux_species)
        return (
            gamma,
            omega,
            Wg_val,
            Wphi_val,
            Wapar_val,
            heat_val,
            pflux_val,
            heat_species,
            pflux_species,
        )

    def step(carry, idx):
        G, phi_prev_step, diag_prev, t_prev, dt_prev = carry
        dG, fields = rhs_fn(G)
        dt_local = jnp.asarray(_update_dt(fields, dt_prev), dtype=real_dtype)
        if method == "euler":
            G_new = G + dt_local * dG
        elif method == "rk2":
            k1 = dG
            G_half = _project_state(G + 0.5 * dt_local * k1)
            k2, _ = rhs_fn(G_half)
            G_new = G + dt_local * k2
        elif method == "rk3":
            k1 = dG
            G1 = _project_state(G + dt_local * k1)
            k2, _ = rhs_fn(G1)
            G2 = _project_state(0.75 * G + 0.25 * (G1 + dt_local * k2))
            k3, _ = rhs_fn(G2)
            G_new = (1.0 / 3.0) * G + (2.0 / 3.0) * (G2 + dt_local * k3)
        elif method == "rk3_gx":
            k1 = dG
            G1 = _project_state(G + (dt_local / 3.0) * k1)
            k2, _ = rhs_fn(G1)
            G2 = _project_state(G + (2.0 * dt_local / 3.0) * k2)
            k3, _ = rhs_fn(G2)
            G3 = _project_state(G + 0.75 * dt_local * k3)
            G_new = G3 + 0.25 * dt_local * k1
        elif method == "rk4":
            k1 = dG
            G2 = _project_state(G + 0.5 * dt_local * k1)
            k2, _ = rhs_fn(G2)
            G3 = _project_state(G + 0.5 * dt_local * k2)
            k3, _ = rhs_fn(G3)
            G4 = _project_state(G + dt_local * k3)
            k4, _ = rhs_fn(G4)
            G_new = G + (dt_local / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        elif method == "sspx3":
            def _sspx3_euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
                dG_state, _fields_state = rhs_fn(G_state)
                return _project_state(G_state + (_SSPX3_ADT * dt_local) * dG_state)

            G1 = _sspx3_euler_step(G)
            G2_euler = _sspx3_euler_step(G1)
            G2 = _project_state((1.0 - _SSPX3_W1) * G + (_SSPX3_W1 - 1.0) * G1 + G2_euler)
            G3 = _sspx3_euler_step(G2)
            G_new = (
                (1.0 - _SSPX3_W2 - _SSPX3_W3) * G
                + _SSPX3_W3 * G1
                + (_SSPX3_W2 - 1.0) * G2
                + G3
            )
        elif method == "k10":
            def _k10_euler_step(G_state):
                dG_state, _ = rhs_fn(G_state)
                return _project_state(G_state + (dt_local / 6.0) * dG_state)

            G_q1 = G
            G_q2 = G
            for _ in range(5):
                G_q1 = _k10_euler_step(G_q1)

            G_q2 = 0.04 * G_q2 + 0.36 * G_q1
            G_q1 = 15.0 * G_q2 - 5.0 * G_q1

            for _ in range(4):
                G_q1 = _k10_euler_step(G_q1)

            dG_final, _ = rhs_fn(G_q1)
            G_new = G_q2 + 0.6 * G_q1 + 0.1 * dt_local * dG_final
        else:
            raise ValueError(
                "method must be one of {'euler', 'rk2', 'rk3', 'rk3_gx', 'rk4', 'k10', 'sspx3'}"
            )
        if use_collision_split and damping is not None:
            G_new = _apply_collision_split(G_new, damping, dt_local, collision_scheme)
        G_new = _project_state(G_new)
        # Keep scan carry dtype stable under mixed-precision scalar constants.
        G_new = jnp.asarray(G_new, dtype=state_dtype)
        t_new = jnp.asarray(t_prev + dt_local, dtype=real_dtype)
        fields_new = compute_fields_cached(G_new, cache, params, terms=term_cfg)
        phi_new = fields_new.phi

        def _compute_diag(_):
            return _compute_diag_from_state(G_new, fields_new, phi_prev_step, dt_local)

        def _reuse_diag(_):
            return diag_prev

        diag_stride = int(max(diagnostics_stride, 1))
        do_diag = (idx % diag_stride) == 0
        diag = jax.lax.cond(do_diag, _compute_diag, _reuse_diag, operand=None)
        return (G_new, phi_new, diag, t_new, dt_local), (diag, t_new, dt_local)

    step_fn = jax.checkpoint(step) if checkpoint else step
    dt0 = jnp.asarray(_update_dt(fields0, dt_init), dtype=real_dtype)
    diag_zero = _compute_diag_from_state(G0, fields0, phi_prev, dt0)
    idx = jnp.arange(steps, dtype=jnp.int32)
    (G_final, _phi_last, _diag_last, _t_last, _dt_last), diag_out = jax.lax.scan(
        step_fn,
        (
            G0,
            phi_prev,
            diag_zero,
            jnp.asarray(0.0, dtype=real_dtype),
            dt0,
        ),
        idx,
        length=steps,
    )

    diag, t, dt_series = diag_out
    gamma_t, omega_t, Wg_t, Wphi_t, Wapar_t, heat_t, pflux_t, heat_s_t, pflux_s_t = diag

    stride = int(max(sample_stride, diagnostics_stride, 1))
    if stride > 1:
        gamma_t = gamma_t[::stride]
        omega_t = omega_t[::stride]
        Wg_t = Wg_t[::stride]
        Wphi_t = Wphi_t[::stride]
        Wapar_t = Wapar_t[::stride]
        heat_t = heat_t[::stride]
        pflux_t = pflux_t[::stride]
        heat_s_t = heat_s_t[::stride, ...]
        pflux_s_t = pflux_s_t[::stride, ...]
        t = t[::stride]
        dt_series = dt_series[::stride]

    dt_mean = jnp.mean(dt_series)
    energy_t = gx_energy_total(Wg_t, Wphi_t, Wapar_t)
    diag_out = GXDiagnostics(
        t=t,
        dt_t=dt_series,
        dt_mean=dt_mean,
        gamma_t=gamma_t,
        omega_t=omega_t,
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=heat_t,
        particle_flux_t=pflux_t,
        energy_t=energy_t,
        heat_flux_species_t=heat_s_t,
        particle_flux_species_t=pflux_s_t,
    )
    fields_final = compute_fields_cached(G_final, cache, params, terms=term_cfg)
    return t, diag_out, G_final, fields_final


def integrate_nonlinear_gx_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "rk3",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    diagnostics_stride: int = 1,
    use_dealias_mask: bool = False,
    z_index: int | None = None,
    gx_real_fft: bool = True,
    laguerre_mode: str = "grid",
    omega_ky_index: int | None = None,
    omega_kx_index: int | None = None,
    flux_scale: float = 1.0,
    wphi_scale: float = 1.0,
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    cfl: float = 0.9,
    cfl_fac: float | None = None,
    collision_split: bool = False,
    collision_scheme: str = "implicit",
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: str | None = None,
    fixed_mode_ky_index: int | None = None,
    fixed_mode_kx_index: int | None = None,
) -> tuple[jnp.ndarray, GXDiagnostics]:
    """Integrate nonlinear system and return GX-style diagnostics."""

    if method in {"imex", "semi-implicit"}:
        return integrate_nonlinear_imex_gx_diagnostics(
            G0,
            grid,
            geom,
            params,
            dt=dt,
            steps=steps,
            method=method,
            cache=cache,
            terms=terms,
            checkpoint=checkpoint,
            sample_stride=sample_stride,
            diagnostics_stride=diagnostics_stride,
            use_dealias_mask=use_dealias_mask,
            z_index=z_index,
            gx_real_fft=gx_real_fft,
            laguerre_mode=laguerre_mode,
            omega_ky_index=omega_ky_index,
            omega_kx_index=omega_kx_index,
            flux_scale=flux_scale,
            wphi_scale=wphi_scale,
            collision_split=collision_split,
            collision_scheme=collision_scheme,
            implicit_tol=implicit_tol,
            implicit_maxiter=implicit_maxiter,
            implicit_iters=implicit_iters,
            implicit_relax=implicit_relax,
            implicit_restart=implicit_restart,
            implicit_solve_method=implicit_solve_method,
            implicit_preconditioner=implicit_preconditioner,
            fixed_mode_ky_index=fixed_mode_ky_index,
            fixed_mode_kx_index=fixed_mode_kx_index,
        )

    t, diag_out, _G_final, _fields_final = _integrate_nonlinear_gx_diagnostics_impl(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        method=method,
        cache=cache,
        terms=terms,
        checkpoint=checkpoint,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        use_dealias_mask=use_dealias_mask,
        z_index=z_index,
        gx_real_fft=gx_real_fft,
        laguerre_mode=laguerre_mode,
        omega_ky_index=omega_ky_index,
        omega_kx_index=omega_kx_index,
        flux_scale=flux_scale,
        wphi_scale=wphi_scale,
        fixed_dt=fixed_dt,
        dt_min=dt_min,
        dt_max=dt_max,
        cfl=cfl,
        cfl_fac=cfl_fac,
        collision_split=collision_split,
        collision_scheme=collision_scheme,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_restart=implicit_restart,
        implicit_solve_method=implicit_solve_method,
        implicit_preconditioner=implicit_preconditioner,
        fixed_mode_ky_index=fixed_mode_ky_index,
        fixed_mode_kx_index=fixed_mode_kx_index,
    )
    return t, diag_out


def integrate_nonlinear_gx_diagnostics_state(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "rk3",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    diagnostics_stride: int = 1,
    use_dealias_mask: bool = False,
    z_index: int | None = None,
    gx_real_fft: bool = True,
    laguerre_mode: str = "grid",
    omega_ky_index: int | None = None,
    omega_kx_index: int | None = None,
    flux_scale: float = 1.0,
    wphi_scale: float = 1.0,
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    cfl: float = 0.9,
    cfl_fac: float | None = None,
    collision_split: bool = False,
    collision_scheme: str = "implicit",
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: str | None = None,
    fixed_mode_ky_index: int | None = None,
    fixed_mode_kx_index: int | None = None,
) -> tuple[jnp.ndarray, GXDiagnostics, jnp.ndarray, FieldState]:
    """Integrate nonlinear system and return GX diagnostics plus the final state."""

    if method in {"imex", "semi-implicit"}:
        raise ValueError("integrate_nonlinear_gx_diagnostics_state only supports explicit methods")

    return _integrate_nonlinear_gx_diagnostics_impl(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        method=method,
        cache=cache,
        terms=terms,
        checkpoint=checkpoint,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        use_dealias_mask=use_dealias_mask,
        z_index=z_index,
        gx_real_fft=gx_real_fft,
        laguerre_mode=laguerre_mode,
        omega_ky_index=omega_ky_index,
        omega_kx_index=omega_kx_index,
        flux_scale=flux_scale,
        wphi_scale=wphi_scale,
        fixed_dt=fixed_dt,
        dt_min=dt_min,
        dt_max=dt_max,
        cfl=cfl,
        cfl_fac=cfl_fac,
        collision_split=collision_split,
        collision_scheme=collision_scheme,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_restart=implicit_restart,
        implicit_solve_method=implicit_solve_method,
        implicit_preconditioner=implicit_preconditioner,
        fixed_mode_ky_index=fixed_mode_ky_index,
        fixed_mode_kx_index=fixed_mode_kx_index,
    )


def integrate_nonlinear_imex_gx_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "imex",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    diagnostics_stride: int = 1,
    use_dealias_mask: bool = False,
    z_index: int | None = None,
    gx_real_fft: bool = True,
    laguerre_mode: str = "grid",
    omega_ky_index: int | None = None,
    omega_kx_index: int | None = None,
    flux_scale: float = 1.0,
    wphi_scale: float = 1.0,
    collision_split: bool = False,
    collision_scheme: str = "implicit",
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: str | None = None,
    fixed_mode_ky_index: int | None = None,
    fixed_mode_kx_index: int | None = None,
) -> tuple[jnp.ndarray, GXDiagnostics]:
    """IMEX nonlinear integrator with GX diagnostics."""

    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError("G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
        cache = build_linear_cache(grid, geom_eff, params, Nl, Nm)

    term_cfg = terms or TermConfig()
    linear_cfg = replace(term_cfg, nonlinear=0.0)
    if collision_split:
        linear_cfg = replace(linear_cfg, collisions=0.0, hypercollisions=0.0)

    vol_fac, flux_fac = gx_volume_factors(geom_eff, grid)
    mask = _gx_omega_mode_mask(grid, cache, gx_real_fft=gx_real_fft)
    z_idx = _gx_midplane_index(grid.z.size) if z_index is None else int(z_index)
    use_dealias = bool(use_dealias_mask)
    use_hermitian = bool(gx_real_fft) and bool(np.any(np.asarray(grid.ky) < 0.0))
    ny_full = int(grid.ky.size)
    nyc = ny_full // 2 + 1
    nx = int(grid.kx.size)
    if nx > 1:
        kx_neg = jnp.concatenate(
            [jnp.asarray([0], dtype=jnp.int32), jnp.arange(nx - 1, 0, -1, dtype=jnp.int32)]
        )
    else:
        kx_neg = jnp.asarray([0], dtype=jnp.int32)

    fixed_projector = _make_fixed_mode_projector(
        G0,
        ky_index=fixed_mode_ky_index,
        kx_index=fixed_mode_kx_index,
    )

    def _project_state(G_state: jnp.ndarray) -> jnp.ndarray:
        if fixed_projector is not None:
            G_state = fixed_projector(G_state)
        if not use_hermitian or nyc <= 2:
            return G_state
        pos = G_state[..., :nyc, :, :]
        neg_hi = nyc - 1 if (ny_full % 2 == 0) else nyc
        neg = jnp.conj(pos[..., 1:neg_hi, :, :])[..., ::-1, :, :]
        if nx > 1:
            neg = neg[..., kx_neg, :]
        return jnp.concatenate([pos, neg], axis=-3)

    initial_state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=initial_state_dtype)
    G0 = _project_state(G0)

    implicit_operator = build_nonlinear_imex_operator(
        G0,
        cache,
        params,
        dt,
        terms=linear_cfg,
        implicit_preconditioner=implicit_preconditioner,
        gx_real_fft=gx_real_fft,
    )

    # Keep the scan carry in the same dtype as the implicit operator, especially
    # under x64 where the operator promotes complex64 inputs to complex128.
    state_dtype = implicit_operator.state_dtype
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)

    squeeze_species = implicit_operator.squeeze_species
    if squeeze_species and G0.ndim == len(implicit_operator.shape) - 1:
        G0 = G0[None, ...]
    use_collision_split = bool(collision_split) and (
        float(term_cfg.collisions) != 0.0 or float(term_cfg.hypercollisions) != 0.0
    )
    damping = None
    if use_collision_split:
        damping = _collision_damping(cache, params, term_cfg, real_dtype, squeeze_species=squeeze_species)

    def nonlinear_term(G_in: jnp.ndarray) -> jnp.ndarray:
        if term_cfg.nonlinear == 0.0:
            return jnp.zeros_like(G_in)
        weight = jnp.asarray(term_cfg.nonlinear, dtype=real_dtype)
        fields = compute_fields_cached(G_in, cache, params, terms=term_cfg)
        return nonlinear_em_contribution(
            G_in,
            phi=fields.phi,
            apar=fields.apar,
            bpar=fields.bpar,
            Jl=cache.Jl,
            JlB=cache.JlB,
            tz=jnp.asarray(params.tz),
            vth=jnp.asarray(params.vth),
            sqrt_m=cache.sqrt_m,
            sqrt_m_p1=cache.sqrt_m_p1,
            kx_grid=cache.kx_grid,
            ky_grid=cache.ky_grid,
            dealias_mask=cache.dealias_mask,
            kxfac=cache.kxfac,
            weight=weight,
            apar_weight=float(term_cfg.apar),
            bpar_weight=float(term_cfg.bpar),
            laguerre_to_grid=cache.laguerre_to_grid,
            laguerre_to_spectral=cache.laguerre_to_spectral,
            laguerre_roots=cache.laguerre_roots,
            laguerre_j0=cache.laguerre_j0,
            laguerre_j1_over_alpha=cache.laguerre_j1_over_alpha,
            b=cache.b,
            gx_real_fft=gx_real_fft,
            laguerre_mode=laguerre_mode,
        )

    def fixed_point(G_in: jnp.ndarray, G_rhs: jnp.ndarray) -> jnp.ndarray:
        def body(_i, g):
            dG, _fields = assemble_rhs_cached_jit(g, cache, params, linear_cfg)
            g_next = G_rhs + dt_val * dG
            return (1.0 - implicit_relax) * g + implicit_relax * g_next

        return jax.lax.fori_loop(0, max(int(implicit_iters), 0), body, G_in)

    def solve_step(G_in: jnp.ndarray, G_rhs: jnp.ndarray) -> jnp.ndarray:
        G_guess = fixed_point(G_in, G_rhs)
        sol, _info = jax.scipy.sparse.linalg.gmres(
            implicit_operator.matvec,
            G_rhs.reshape(-1),
            x0=G_guess.reshape(-1),
            tol=implicit_tol,
            maxiter=implicit_maxiter,
            restart=implicit_restart,
            M=implicit_operator.precond_op,
            solve_method=implicit_solve_method,
        )
        return sol.reshape(implicit_operator.shape)

    def _compute_diag_from_state(
        G_state: jnp.ndarray,
        fields_state: FieldState,
        phi_prev_step: jnp.ndarray,
        dt_step: jnp.ndarray,
    ):
        phi = fields_state.phi
        apar = fields_state.apar if fields_state.apar is not None else jnp.zeros_like(phi)
        bpar = fields_state.bpar if fields_state.bpar is not None else jnp.zeros_like(phi)

        gamma_modes, omega_modes = _gx_growth_rate_step(
            phi, phi_prev_step, dt_step, z_index=z_idx, mask=mask
        )
        if omega_ky_index is not None:
            ky_i = int(np.clip(omega_ky_index, 0, int(gamma_modes.shape[0]) - 1))
            kx_i = int(np.clip(omega_kx_index or 0, 0, int(gamma_modes.shape[1]) - 1))
            gamma = jnp.nan_to_num(gamma_modes[ky_i, kx_i], nan=jnp.asarray(0.0, dtype=real_dtype))
            omega = jnp.nan_to_num(omega_modes[ky_i, kx_i], nan=jnp.asarray(0.0, dtype=real_dtype))
        else:
            gamma = jnp.nan_to_num(
                jnp.nanmean(jnp.where(mask, gamma_modes, jnp.nan)),
                nan=jnp.asarray(0.0, dtype=real_dtype),
            )
            omega = jnp.nan_to_num(
                jnp.nanmean(jnp.where(mask, omega_modes, jnp.nan)),
                nan=jnp.asarray(0.0, dtype=real_dtype),
            )
        Wg_val = gx_Wg(G_state, grid, params, vol_fac, use_dealias=use_dealias)
        Wphi_val = gx_Wphi(
            phi,
            cache,
            params,
            vol_fac,
            use_dealias=use_dealias,
            wphi_scale=wphi_scale,
        )
        Wapar_val = gx_Wapar(apar, cache, vol_fac, use_dealias=use_dealias)
        heat_species = gx_heat_flux_species(
            G_state,
            phi,
            apar,
            bpar,
            cache,
            grid,
            params,
            flux_fac,
            use_dealias=use_dealias,
            flux_scale=flux_scale,
        )
        pflux_species = gx_particle_flux_species(
            G_state,
            phi,
            apar,
            bpar,
            cache,
            grid,
            params,
            flux_fac,
            use_dealias=use_dealias,
            flux_scale=flux_scale,
        )
        heat_val = jnp.sum(heat_species)
        pflux_val = jnp.sum(pflux_species)
        return (
            gamma,
            omega,
            Wg_val,
            Wphi_val,
            Wapar_val,
            heat_val,
            pflux_val,
            heat_species,
            pflux_species,
        )

    fields0 = compute_fields_cached(G0, cache, params, terms=term_cfg)
    phi_prev = fields0.phi

    def step(carry, idx):
        G, phi_prev_step, diag_prev, t_prev = carry
        rhs = G + dt_val * nonlinear_term(G)
        if method == "sspx3":
            def _euler_step(G_state: jnp.ndarray, dt_stage: jnp.ndarray) -> jnp.ndarray:
                rhs_stage = G_state + dt_stage * nonlinear_term(G_state)
                return solve_step(G_state, rhs_stage)

            G1 = _euler_step(G, _SSPX3_ADT * dt_val)
            G2_euler = _euler_step(G1, _SSPX3_ADT * dt_val)
            G2 = _project_state((1.0 - _SSPX3_W1) * G + (_SSPX3_W1 - 1.0) * G1 + G2_euler)
            G3 = _euler_step(G2, _SSPX3_ADT * dt_val)
            G_new = (
                (1.0 - _SSPX3_W2 - _SSPX3_W3) * G
                + _SSPX3_W3 * G1
                + (_SSPX3_W2 - 1.0) * G2
                + G3
            )
        else:
            G_new = solve_step(G, rhs)
        if use_collision_split and damping is not None:
            G_new = _apply_collision_split(G_new, damping, dt_val, collision_scheme)
        G_new = _project_state(G_new)
        # Keep scan carry dtype stable under mixed-precision scalar constants.
        G_new = jnp.asarray(G_new, dtype=state_dtype)
        t_new = t_prev + dt_val
        fields_new = compute_fields_cached(G_new, cache, params, terms=term_cfg)
        phi_new = fields_new.phi

        def _compute_diag(_):
            return _compute_diag_from_state(G_new, fields_new, phi_prev_step, dt_val)

        def _reuse_diag(_):
            return diag_prev

        diag_stride = int(max(diagnostics_stride, 1))
        do_diag = (idx % diag_stride) == 0
        diag = jax.lax.cond(do_diag, _compute_diag, _reuse_diag, operand=None)
        return (G_new, phi_new, diag, t_new), (diag, t_new)

    step_fn = jax.checkpoint(step) if checkpoint else step
    diag_zero = _compute_diag_from_state(G0, fields0, phi_prev, dt_val)
    idx = jnp.arange(steps, dtype=jnp.int32)
    (G_final, _phi_last, _diag_last, _t_last), diag_out = jax.lax.scan(
        step_fn,
        (
            G0,
            phi_prev,
            diag_zero,
            jnp.asarray(0.0, dtype=real_dtype),
        ),
        idx,
        length=steps,
    )

    diag, t = diag_out
    gamma_t, omega_t, Wg_t, Wphi_t, Wapar_t, heat_t, pflux_t, heat_s_t, pflux_s_t = diag
    dt_series = jnp.ones_like(t) * dt_val

    stride = int(max(sample_stride, diagnostics_stride, 1))
    if stride > 1:
        gamma_t = gamma_t[::stride]
        omega_t = omega_t[::stride]
        Wg_t = Wg_t[::stride]
        Wphi_t = Wphi_t[::stride]
        Wapar_t = Wapar_t[::stride]
        heat_t = heat_t[::stride]
        pflux_t = pflux_t[::stride]
        heat_s_t = heat_s_t[::stride, ...]
        pflux_s_t = pflux_s_t[::stride, ...]
        t = t[::stride]
        dt_series = dt_series[::stride]

    dt_mean = jnp.mean(dt_series)
    energy_t = gx_energy_total(Wg_t, Wphi_t, Wapar_t)
    diag_out = GXDiagnostics(
        t=t,
        dt_t=dt_series,
        dt_mean=dt_mean,
        gamma_t=gamma_t,
        omega_t=omega_t,
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=heat_t,
        particle_flux_t=pflux_t,
        energy_t=energy_t,
        heat_flux_species_t=heat_s_t,
        particle_flux_species_t=pflux_s_t,
    )
    return t, diag_out


def build_nonlinear_imex_operator(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    *,
    terms: TermConfig | None = None,
    implicit_preconditioner: str | None = None,
    gx_real_fft: bool = True,
) -> IMEXLinearOperator:
    """Build and cache the matrix-free linear operator used by nonlinear IMEX."""

    term_cfg = terms or TermConfig()
    linear_terms = term_config_to_linear_terms(term_cfg)
    G, shape, _size, dt_val, precond_op, matvec, squeeze_species = _build_implicit_operator(
        G0,
        cache,
        params,
        dt,
        linear_terms,
        implicit_preconditioner,
    )
    return IMEXLinearOperator(
        state_dtype=G.dtype,
        shape=shape,
        dt_val=dt_val,
        precond_op=precond_op,
        matvec=matvec,
        squeeze_species=squeeze_species,
    )


def integrate_nonlinear_imex_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: str | None = None,
    implicit_operator: IMEXLinearOperator | None = None,
    gx_real_fft: bool = True,
    laguerre_mode: str = "grid",
) -> tuple[jnp.ndarray, FieldState]:
    """IMEX integrator: implicit linear operator, explicit nonlinear term."""

    term_cfg = terms or TermConfig()
    linear_cfg = replace(term_cfg, nonlinear=0.0)

    linear_terms = term_config_to_linear_terms(linear_cfg)

    precond_op: Callable[[jnp.ndarray], jnp.ndarray] | None
    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    if implicit_operator is None:
        G, shape, _size, dt_val, precond_op, matvec, squeeze_species = _build_implicit_operator(
            G0,
            cache,
            params,
            dt,
            linear_terms,
            implicit_preconditioner,
        )
    else:
        shape = implicit_operator.shape
        dt_val = implicit_operator.dt_val
        precond_op = implicit_operator.precond_op
        matvec = implicit_operator.matvec
        squeeze_species = implicit_operator.squeeze_species
        G = jnp.asarray(G0, dtype=implicit_operator.state_dtype)
        if squeeze_species and G.ndim == len(shape) - 1:
            G = G[None, ...]
        if G.shape != shape:
            raise ValueError(
                "implicit_operator shape mismatch: "
                f"expected {shape}, got {tuple(G.shape)}"
            )

    def nonlinear_term(G_in: jnp.ndarray) -> jnp.ndarray:
        if term_cfg.nonlinear == 0.0:
            return jnp.zeros_like(G_in)
        weight = jnp.asarray(term_cfg.nonlinear, dtype=jnp.real(jnp.empty((), G_in.dtype)).dtype)
        fields = compute_fields_cached(G_in, cache, params, terms=term_cfg)
        return nonlinear_em_contribution(
            G_in,
            phi=fields.phi,
            apar=fields.apar,
            bpar=fields.bpar,
            Jl=cache.Jl,
            JlB=cache.JlB,
            tz=jnp.asarray(params.tz),
            vth=jnp.asarray(params.vth),
            sqrt_m=cache.sqrt_m,
            sqrt_m_p1=cache.sqrt_m_p1,
            kx_grid=cache.kx_grid,
            ky_grid=cache.ky_grid,
            dealias_mask=cache.dealias_mask,
            kxfac=cache.kxfac,
            weight=weight,
            apar_weight=float(term_cfg.apar),
            bpar_weight=float(term_cfg.bpar),
            laguerre_to_grid=cache.laguerre_to_grid,
            laguerre_to_spectral=cache.laguerre_to_spectral,
            laguerre_roots=cache.laguerre_roots,
            laguerre_j0=cache.laguerre_j0,
            laguerre_j1_over_alpha=cache.laguerre_j1_over_alpha,
            b=cache.b,
            gx_real_fft=gx_real_fft,
            laguerre_mode=laguerre_mode,
        )

    def fixed_point(G_in: jnp.ndarray, G_rhs: jnp.ndarray) -> jnp.ndarray:
        def body(_i, g):
            dG, _fields = assemble_rhs_cached_jit(g, cache, params, linear_cfg)
            g_next = G_rhs + dt_val * dG
            return (1.0 - implicit_relax) * g + implicit_relax * g_next

        return jax.lax.fori_loop(0, max(int(implicit_iters), 0), body, G_in)

    def solve_step(G_in: jnp.ndarray, G_rhs: jnp.ndarray) -> jnp.ndarray:
        G_guess = fixed_point(G_in, G_rhs)
        sol, _info = jax.scipy.sparse.linalg.gmres(
            matvec,
            G_rhs.reshape(-1),
            x0=G_guess.reshape(-1),
            tol=implicit_tol,
            maxiter=implicit_maxiter,
            restart=implicit_restart,
            M=precond_op,
            solve_method=implicit_solve_method,
        )
        return sol.reshape(shape)

    def step(G_in, _):
        rhs = G_in + dt_val * nonlinear_term(G_in)
        G_new = solve_step(G_in, rhs)
        _dG_new, fields_new = assemble_rhs_cached_jit(G_new, cache, params, linear_cfg)
        return G_new, fields_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    G_out, fields_t = jax.lax.scan(step_fn, G, None, length=steps)
    G_out = G_out[0] if squeeze_species else G_out
    return G_out, fields_t
