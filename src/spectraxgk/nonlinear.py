"""Nonlinear gyrokinetic drivers built on term-wise RHS assembly."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import (
    LinearCache,
    LinearParams,
    _build_implicit_operator,
    build_linear_cache,
    term_config_to_linear_terms,
)
from spectraxgk.terms.assembly import assemble_rhs_cached_jit, compute_fields_cached
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.integrators import integrate_nonlinear as integrate_nonlinear_scan
from spectraxgk.terms.nonlinear import nonlinear_em_contribution
from spectraxgk.gx_integrators import _gx_growth_rate_step, _gx_midplane_index
from spectraxgk.diagnostics import (
    GXDiagnostics,
    gx_energy_total,
    gx_heat_flux,
    gx_particle_flux,
    gx_volume_factors,
    gx_Wapar_krehm,
    gx_Wg,
    gx_Wphi_krehm,
)


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

    return integrate_nonlinear_scan(
        rhs_fn,
        G0,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
    )


def integrate_nonlinear(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: SAlphaGeometry,
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

    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError("G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
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


def integrate_nonlinear_gx_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: SAlphaGeometry,
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
    flux_scale: float = 2.0,
    wphi_scale: float = 1.0,
) -> tuple[jnp.ndarray, GXDiagnostics]:
    """Integrate nonlinear system and return GX-style diagnostics."""

    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError("G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
        cache = build_linear_cache(grid, geom, params, Nl, Nm)

    term_cfg = terms or TermConfig()
    vol_fac, flux_fac = gx_volume_factors(geom, grid)
    mask = jnp.asarray(grid.dealias_mask, dtype=bool)
    z_idx = _gx_midplane_index(grid.z.size) if z_index is None else int(z_index)
    use_dealias = bool(use_dealias_mask)
    rho_star = float(getattr(params, "rho_star", 1.0))
    kx_phys = cache.kx / rho_star
    ky_phys = cache.ky / rho_star
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

    def _enforce_hermitian(G_state: jnp.ndarray) -> jnp.ndarray:
        if not use_hermitian or nyc <= 2:
            return G_state
        pos = G_state[..., :nyc, :, :]
        neg = jnp.conj(pos[..., 1 : nyc - 1, :, :])[..., ::-1, :, :]
        if nx > 1:
            neg = neg[..., kx_neg, :]
        return jnp.concatenate([pos, neg], axis=-3)

    state_dtype = jnp.result_type(G0, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    G0 = _enforce_hermitian(G0)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)

    def rhs_fn(G):
        return nonlinear_rhs_cached(
            G,
            cache,
            params,
            term_cfg,
            gx_real_fft=gx_real_fft,
            laguerre_mode=laguerre_mode,
        )

    _dG0, fields0 = rhs_fn(G0)
    phi_prev = fields0.phi

    def _compute_diag_from_state(G_state, phi_last):
        _dG_state, fields_state = rhs_fn(G_state)
        phi = fields_state.phi
        apar = fields_state.apar if fields_state.apar is not None else jnp.zeros_like(phi)
        bpar = fields_state.bpar if fields_state.bpar is not None else jnp.zeros_like(phi)

        gamma, omega = _gx_growth_rate_step(phi, phi_last, dt_val, z_index=z_idx, mask=mask)
        Wg_val = gx_Wg(G_state, grid, params, vol_fac, use_dealias=use_dealias)
        Wphi_val = gx_Wphi_krehm(
            phi,
            grid,
            params,
            vol_fac,
            kx=kx_phys,
            ky=ky_phys,
            use_dealias=use_dealias,
            gx_real_fft=gx_real_fft,
            wphi_scale=wphi_scale,
        )
        Wapar_val = gx_Wapar_krehm(apar, grid, kx=kx_phys, ky=ky_phys, use_dealias=use_dealias)
        heat_val = gx_heat_flux(
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
        pflux_val = gx_particle_flux(
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
        return (gamma, omega, Wg_val, Wphi_val, Wapar_val, heat_val, pflux_val), phi

    def step(carry, idx):
        G, phi_last, diag_prev = carry
        dG, _ = rhs_fn(G)
        if method == "euler":
            G_new = G + dt_val * dG
        elif method == "rk2":
            k1 = dG
            k2, _ = rhs_fn(G + 0.5 * dt_val * k1)
            G_new = G + dt_val * k2
        elif method == "rk3":
            k1 = dG
            G1 = G + dt_val * k1
            k2, _ = rhs_fn(G1)
            G2 = 0.75 * G + 0.25 * (G1 + dt_val * k2)
            k3, _ = rhs_fn(G2)
            G_new = (1.0 / 3.0) * G + (2.0 / 3.0) * (G2 + dt_val * k3)
        elif method == "rk3_gx":
            k1 = dG
            G1 = G + (dt_val / 3.0) * k1
            k2, _ = rhs_fn(G1)
            G2 = G + (2.0 * dt_val / 3.0) * k2
            k3, _ = rhs_fn(G2)
            G3 = G + 0.75 * dt_val * k3
            G_new = G3 + 0.25 * dt_val * k1
        elif method == "rk4":
            k1 = dG
            k2, _ = rhs_fn(G + 0.5 * dt_val * k1)
            k3, _ = rhs_fn(G + 0.5 * dt_val * k2)
            k4, _ = rhs_fn(G + dt_val * k3)
            G_new = G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        elif method == "k10":
            def _euler_step(G_state):
                dG_state, _ = rhs_fn(G_state)
                return G_state + (dt_val / 6.0) * dG_state

            G_q1 = G
            G_q2 = G
            for _ in range(5):
                G_q1 = _euler_step(G_q1)

            G_q2 = 0.04 * G_q2 + 0.36 * G_q1
            G_q1 = 15.0 * G_q2 - 5.0 * G_q1

            for _ in range(4):
                G_q1 = _euler_step(G_q1)

            dG_final, _ = rhs_fn(G_q1)
            G_new = G_q2 + 0.6 * G_q1 + 0.1 * dt_val * dG_final
        else:
            raise ValueError(
                "method must be one of {'euler', 'rk2', 'rk3', 'rk3_gx', 'rk4', 'k10'}"
            )
        G_new = _enforce_hermitian(G_new)

        def _compute_diag(_):
            return _compute_diag_from_state(G_new, phi_last)

        def _reuse_diag(_):
            return diag_prev, phi_last

        diag_stride = int(max(diagnostics_stride, 1))
        do_diag = (idx % diag_stride) == 0
        diag, phi = jax.lax.cond(do_diag, _compute_diag, _reuse_diag, operand=None)
        return (G_new, phi, diag), diag

    step_fn = jax.checkpoint(step) if checkpoint else step
    diag_zero, _phi0 = _compute_diag_from_state(G0, phi_prev)
    idx = jnp.arange(steps, dtype=jnp.int32)
    (G_final, _phi_last, _diag_last), diag = jax.lax.scan(
        step_fn, (G0, phi_prev, diag_zero), idx, length=steps
    )

    gamma_t, omega_t, Wg_t, Wphi_t, Wapar_t, heat_t, pflux_t = diag
    t = dt_val * (jnp.arange(steps, dtype=real_dtype) + 1.0)

    stride = int(max(sample_stride, diagnostics_stride, 1))
    if stride > 1:
        gamma_t = gamma_t[::stride]
        omega_t = omega_t[::stride]
        Wg_t = Wg_t[::stride]
        Wphi_t = Wphi_t[::stride]
        Wapar_t = Wapar_t[::stride]
        heat_t = heat_t[::stride]
        pflux_t = pflux_t[::stride]
        t = t[::stride]

    energy_t = gx_energy_total(Wg_t, Wphi_t, Wapar_t)
    diag_out = GXDiagnostics(
        t=t,
        gamma_t=gamma_t,
        omega_t=omega_t,
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=heat_t,
        particle_flux_t=pflux_t,
        energy_t=energy_t,
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
