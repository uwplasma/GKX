"""Matrix-free Krylov solvers for linear gyrokinetic operators."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres

from spectraxgk.linear import (
    LinearCache,
    LinearParams,
    LinearTerms,
    _as_species_array,
    hypercollision_damping,
    linear_terms_to_term_config,
)
from spectraxgk.terms.assembly import assemble_rhs_cached
from spectraxgk.terms.config import TermConfig


@dataclass(frozen=True)
class KrylovConfig:
    """Controls for the Krylov-based eigen solver."""

    krylov_dim: int = 24
    restarts: int = 2
    omega_min_factor: float = 0.0
    omega_target_factor: float = 0.0
    omega_cap_factor: float = 2.0
    omega_sign: int = 0
    method: str = "propagator"
    power_iters: int = 200
    power_dt: float = 0.01
    shift: complex | None = None
    shift_source: str = "propagator"
    shift_tol: float = 1.0e-4
    shift_maxiter: int = 50
    shift_restart: int = 20
    shift_solve_method: str = "batched"
    shift_preconditioner: str | None = "damping"
    shift_selection: str = "targeted"
    mode_family: str = "auto"
    fallback_method: str = "propagator"
    fallback_real_floor: float = -1.0e-6
    continuation: bool = False
    continuation_selection: str = "overlap"


def _normalize(v: jnp.ndarray) -> jnp.ndarray:
    norm = jnp.linalg.norm(v)
    norm_safe = jnp.where(norm == 0.0, 1.0, norm)
    return v / norm_safe


@jax.jit
def _assemble_rhs_cached_novjp(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
) -> tuple[jnp.ndarray, object]:
    return assemble_rhs_cached(G, cache, params, terms=term_cfg, use_custom_vjp=False)


def _apply_operator(
    v: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
) -> jnp.ndarray:
    dG, _fields = _assemble_rhs_cached_novjp(v, cache, params, term_cfg)
    return dG


def _compute_damping(
    v: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
) -> jnp.ndarray:
    real_dtype = jnp.real(v).dtype
    lb_lam = cache.lb_lam.astype(real_dtype)
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    if lb_lam.ndim == 6:
        ns = lb_lam.shape[0]
        nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)
        damping = nu[:, None, None, None, None, None] * lb_lam + hyper_damp
        if v.ndim == 5:
            damping = damping[0]
    else:
        damping = jnp.asarray(params.nu, dtype=real_dtype) * lb_lam + hyper_damp
    return damping.astype(real_dtype)


def _omega_scale(cache: LinearCache, params: LinearParams) -> jnp.ndarray:
    ky_scale = jnp.max(jnp.abs(cache.ky))
    rlt_i = jnp.abs(params.R_over_LTi)
    rlt_e = jnp.abs(params.R_over_LTe)
    rln = jnp.abs(params.R_over_Ln)

    def _max_scalar(arr: jnp.ndarray) -> jnp.ndarray:
        return arr if arr.ndim == 0 else jnp.max(arr)

    drive_i = _max_scalar(rlt_i)
    drive_e = _max_scalar(rlt_e)
    drive_n = _max_scalar(rln)
    drive = jnp.maximum(drive_i, jnp.maximum(drive_e, drive_n))
    return ky_scale * jnp.maximum(drive, 1.0e-8)


def _mode_family_sign(mode_family: str) -> int:
    key = mode_family.strip().lower()
    if key in {"ion", "itg", "cyclone", "positive"}:
        return 1
    if key in {"electron", "etg", "tem", "kbm", "negative"}:
        return -1
    return 0


def _select_by_overlap(
    eigvecs: jnp.ndarray,
    V: jnp.ndarray,
    v_ref: jnp.ndarray,
    mask: jnp.ndarray,
    fallback_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Select eigenpair with maximal overlap to v_ref within mask."""
    beta = jnp.tensordot(jnp.conj(V), v_ref, axes=v_ref.ndim)
    overlap = jnp.abs(jnp.dot(jnp.conj(beta), eigvecs))
    overlap_masked = jnp.where(mask, overlap, -jnp.inf)
    has_mask = jnp.any(mask)
    idx = jnp.argmax(overlap_masked)
    return jnp.where(has_mask, idx, fallback_idx)


def _select_by_target(
    real_part: jnp.ndarray,
    imag_part: jnp.ndarray,
    mask: jnp.ndarray,
    omega_scale: jnp.ndarray,
    omega_target_factor: float,
    omega_sign: int,
    fallback_idx: jnp.ndarray,
) -> jnp.ndarray:
    omega_target_factor_val = jnp.asarray(omega_target_factor, dtype=imag_part.dtype)
    use_target = omega_target_factor_val > 0.0
    omega_target = omega_target_factor_val * omega_scale
    omega_sign_val = jnp.asarray(omega_sign, dtype=imag_part.dtype)
    use_sign = omega_sign_val != 0.0
    omega_target = jnp.where(use_sign, jnp.sign(omega_sign_val) * jnp.abs(omega_target), omega_target)
    use_mask = jnp.any(mask)
    mask_use = jnp.where(use_mask, mask, jnp.ones_like(mask, dtype=bool))
    mask_pos = real_part >= 0.0
    mask_target = mask_use
    has_pos = jnp.any(mask_target & mask_pos)
    mask_target = jnp.where(has_pos, mask_target & mask_pos, mask_target)
    dist = jnp.abs(imag_part - omega_target)
    dist_masked = jnp.where(mask_target, dist, jnp.inf)
    idx_target = jnp.argmin(dist_masked)
    has_target = jnp.any(mask_target)
    use_choice = use_target & has_target
    return jnp.where(use_choice, idx_target, fallback_idx)


def _advance_imex2(
    v: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    damping = _compute_damping(v, cache, params)
    dG = _apply_operator(v, cache, params, term_cfg)
    dG_explicit = dG + damping * v
    v_half = (v + 0.5 * dt * dG_explicit) / (1.0 + 0.5 * dt * damping)
    dG_half = _apply_operator(v_half, cache, params, term_cfg)
    dG_half_exp = dG_half + damping * v_half
    return (v + dt * dG_half_exp) / (1.0 + dt * damping)


def _build_shift_invert_precond(
    v: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    sigma: jnp.ndarray,
    mode: str | None,
) -> tuple[jnp.ndarray | None, Callable[[jnp.ndarray], jnp.ndarray] | None]:
    if mode is None or mode.lower() == "none":
        return None, None
    mode_key = mode.lower()
    if mode_key == "damping":
        damping = _compute_damping(v, cache, params)
        diag = -damping.astype(v.dtype) - sigma
        safe = jnp.where(jnp.abs(diag) > 0.0, diag, 1.0 + 0.0j)
        precond = 1.0 / safe
        shape = v.shape
        size = v.size

        def apply_precond_damping(x_flat: jnp.ndarray) -> jnp.ndarray:
            x = x_flat.reshape(shape)
            return (x * precond).reshape(size)

        return precond, apply_precond_damping

    if mode_key not in {
        "hermite-line",
        "hermite_line",
        "hermite",
        "streaming-line",
        "streaming_line",
        "hermite-line-coarse",
        "hermite_line_coarse",
        "hermite_coarse",
        "streaming-line-coarse",
    }:
        return None, None

    # Hermite-line preconditioners rely on real-valued tridiagonal solves, but
    # shift-invert uses complex coefficients (streaming i*kz). Until a complex
    # tridiagonal solve is implemented, fall back to the diagonal damping
    # preconditioner when hermite-line is requested.
    damping = _compute_damping(v, cache, params)
    diag = -damping.astype(v.dtype) - sigma
    safe = jnp.where(jnp.abs(diag) > 0.0, diag, 1.0 + 0.0j)
    precond = 1.0 / safe
    shape = v.shape
    size = v.size

    def apply_precond_fallback(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        return (x * precond).reshape(size)

    return precond, apply_precond_fallback

    shape = v.shape
    size = v.size
    state_dtype = v.dtype
    real_dtype = jnp.real(v).dtype
    imag = jnp.asarray(1j, dtype=state_dtype)
    sigma_val = jnp.asarray(sigma, dtype=state_dtype)
    sigma_safe = jnp.where(jnp.abs(sigma_val) > 1.0e-12, sigma_val, 1.0e-12 + 0.0j)

    w_stream = jnp.asarray(term_cfg.streaming, dtype=real_dtype)
    kpar_scale = jnp.asarray(params.kpar_scale, dtype=real_dtype)
    sqrt_m_line = cache.sqrt_m_ladder.reshape(-1).astype(real_dtype)
    sqrt_p_line = cache.sqrt_p.reshape(-1).astype(real_dtype)

    def _solve_hermite_lines_fft(
        x: jnp.ndarray,
        *,
        kz: jnp.ndarray,
        vth: jnp.ndarray,
    ) -> jnp.ndarray:
        """Invert (-sigma I + L_stream) via FFT(z) + tridiagonal(m).

        This is a matrix-free streaming preconditioner for shift-invert GMRES
        solves. It ignores curvature/mirror/etc. but captures the stiff Hermite
        coupling from streaming exactly (for periodic z).
        """

        x_hat = jnp.fft.fft(x, axis=-1)
        x_hat_mlast = jnp.moveaxis(x_hat, 2, -1)  # (Ns, Nl, Ny, Nx, Nz, Nm)
        coeff = (
            (-w_stream * kpar_scale)
            * vth[:, None, None, None, None]
            * (imag * kz)[None, None, None, None, :]
        )
        coeff = coeff[..., None]  # (Ns, 1, 1, 1, Nz, 1)
        dl = coeff * sqrt_m_line
        du = coeff * sqrt_p_line
        du = du.at[..., -1].set(jnp.asarray(0.0, dtype=du.dtype))
        d = jnp.ones_like(du) * (-sigma_safe)
        batch_shape = x_hat_mlast.shape
        dl = jnp.broadcast_to(dl, batch_shape)
        d = jnp.broadcast_to(d, batch_shape)
        du = jnp.broadcast_to(du, batch_shape)
        # `tridiagonal_solve` requires all operands to share the same dtype.
        # Under x64 this pathway can mix complex64 state with complex128 coeffs.
        solve_dtype = jnp.result_type(x_hat_mlast, dl, d, du)
        y_hat_mlast = jax.lax.linalg.tridiagonal_solve(
            dl.astype(solve_dtype),
            d.astype(solve_dtype),
            du.astype(solve_dtype),
            x_hat_mlast.astype(solve_dtype)[..., None],
        )[..., 0]
        y_hat_mlast = y_hat_mlast.astype(x_hat_mlast.dtype)
        y_hat = jnp.moveaxis(y_hat_mlast, -1, 2)
        return jnp.fft.ifft(y_hat, axis=-1)

    def _solve_hermite_lines_linked(
        x: jnp.ndarray,
        *,
        vth: jnp.ndarray,
    ) -> jnp.ndarray:
        """Linked-FFT variant of the Hermite-line streaming preconditioner."""

        if not cache.linked_indices:
            return _solve_hermite_lines_fft(x, kz=cache.kz, vth=vth)

        Ny = x.shape[-3]
        Nx = x.shape[-2]
        Nz = x.shape[-1]
        lead_shape = x.shape[:-3]
        x_flat = x.reshape(*lead_shape, Ny * Nx, Nz)
        y_flat = jnp.zeros_like(x_flat)

        def _scatter_unique(
            target: jnp.ndarray, idx_flat: jnp.ndarray, updates: jnp.ndarray
        ) -> jnp.ndarray:
            idx = jnp.asarray(idx_flat, dtype=jnp.int32)
            target_t = jnp.moveaxis(target, -2, 0)
            updates_t = jnp.moveaxis(updates, -2, 0)
            idx = idx[:, None]
            dnums = jax.lax.ScatterDimensionNumbers(
                update_window_dims=tuple(range(1, updates_t.ndim)),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
            )
            out_t = jax.lax.scatter(
                target_t,
                idx,
                updates_t,
                dnums,
                unique_indices=True,
            )
            return jnp.moveaxis(out_t, 0, -2)

        for idx_map, kz_link in zip(cache.linked_indices, cache.linked_kz):
            nChains, nLinks = idx_map.shape
            idx_flat = idx_map.reshape(-1)
            x_link = jnp.take(x_flat, idx_flat, axis=-2)
            x_link = x_link.reshape(*lead_shape, nChains, nLinks * Nz)
            x_hat = jnp.fft.fft(x_link, axis=-1)
            x_hat_mlast = jnp.moveaxis(x_hat, 2, -1)  # (Ns, Nl, nChains, nfreq, Nm)
            coeff = (
                (-w_stream * kpar_scale)
                * vth[:, None, None, None]
                * (imag * kz_link)[None, None, None, :]
            )
            coeff = coeff[..., None]  # (Ns, 1, 1, nfreq, 1)
            dl = coeff * sqrt_m_line
            du = coeff * sqrt_p_line
            du = du.at[..., -1].set(jnp.asarray(0.0, dtype=du.dtype))
            d = jnp.ones_like(du) * (-sigma_safe)
            batch_shape = x_hat_mlast.shape
            dl = jnp.broadcast_to(dl, batch_shape)
            d = jnp.broadcast_to(d, batch_shape)
            du = jnp.broadcast_to(du, batch_shape)
            solve_dtype = jnp.result_type(x_hat_mlast, dl, d, du)
            y_hat_mlast = jax.lax.linalg.tridiagonal_solve(
                dl.astype(solve_dtype),
                d.astype(solve_dtype),
                du.astype(solve_dtype),
                x_hat_mlast.astype(solve_dtype)[..., None],
            )[..., 0]
            y_hat_mlast = y_hat_mlast.astype(x_hat_mlast.dtype)
            y_hat = jnp.moveaxis(y_hat_mlast, -1, 2)
            y_link = jnp.fft.ifft(y_hat, axis=-1)
            y_link = y_link.reshape(*lead_shape, nChains * nLinks, Nz)
            y_flat = _scatter_unique(y_flat, idx_flat, y_link)

        return y_flat.reshape(*lead_shape, Ny, Nx, Nz)

    def _project_kx_coarse(x: jnp.ndarray) -> jnp.ndarray:
        """Coarse-space projection/prolongation for twist/shift coupling."""

        if not cache.use_twist_shift or not cache.linked_indices:
            x_mean = jnp.mean(x, axis=-2, keepdims=True)
            return jnp.broadcast_to(x_mean, x.shape)

        Ny = x.shape[-3]
        Nx = x.shape[-2]
        Nz = x.shape[-1]
        lead_shape = x.shape[:-3]
        x_flat = x.reshape(*lead_shape, Ny * Nx, Nz)
        y_flat = jnp.zeros_like(x_flat)

        def _scatter_unique(
            target: jnp.ndarray, idx_flat: jnp.ndarray, updates: jnp.ndarray
        ) -> jnp.ndarray:
            idx = jnp.asarray(idx_flat, dtype=jnp.int32)
            target_t = jnp.moveaxis(target, -2, 0)
            updates_t = jnp.moveaxis(updates, -2, 0)
            idx = idx[:, None]
            dnums = jax.lax.ScatterDimensionNumbers(
                update_window_dims=tuple(range(1, updates_t.ndim)),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
            )
            out_t = jax.lax.scatter(
                target_t,
                idx,
                updates_t,
                dnums,
                unique_indices=True,
            )
            return jnp.moveaxis(out_t, 0, -2)

        for idx_map in cache.linked_indices:
            nChains, nLinks = idx_map.shape
            idx_flat = idx_map.reshape(-1)
            x_link = jnp.take(x_flat, idx_flat, axis=-2)
            x_link = x_link.reshape(*lead_shape, nChains, nLinks, Nz)
            x_mean = jnp.mean(x_link, axis=-2, keepdims=True)
            x_mean = jnp.broadcast_to(x_mean, x_link.shape)
            x_updates = x_mean.reshape(*lead_shape, nChains * nLinks, Nz)
            y_flat = _scatter_unique(y_flat, idx_flat, x_updates)

        return y_flat.reshape(*lead_shape, Ny, Nx, Nz)

    def apply_precond_hermite_line(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        squeeze_species = False
        if x.ndim == 5:
            x = x[None, ...]
            squeeze_species = True
        ns = x.shape[0]
        vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)
        y = (
            _solve_hermite_lines_linked(x, vth=vth)
            if cache.use_twist_shift
            else _solve_hermite_lines_fft(x, kz=cache.kz, vth=vth)
        )
        if squeeze_species:
            y = y[0]
        return y.reshape(size)

    def apply_precond_hermite_line_coarse(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        squeeze_species = False
        if x.ndim == 5:
            x = x[None, ...]
            squeeze_species = True
        ns = x.shape[0]
        vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)

        def solve(x_in: jnp.ndarray) -> jnp.ndarray:
            return (
                _solve_hermite_lines_linked(x_in, vth=vth)
                if cache.use_twist_shift
                else _solve_hermite_lines_fft(x_in, kz=cache.kz, vth=vth)
            )

        x_line = solve(x)
        x_coarse_in = _project_kx_coarse(x)
        x_coarse_full = solve(x_coarse_in)
        x_line_coarse_full = _project_kx_coarse(x_line)
        y = x_line + (x_coarse_full - x_line_coarse_full)
        if squeeze_species:
            y = y[0]
        return y.reshape(size)

    if mode_key in {"hermite-line-coarse", "hermite_line_coarse", "hermite_coarse", "streaming-line-coarse"}:
        return None, apply_precond_hermite_line_coarse
    return None, apply_precond_hermite_line


@partial(jax.jit, static_argnames=("iterations",))
def dominant_eigenpair_power(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    iterations: int,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Power iteration on an explicit-Euler propagator to target the rightmost mode."""

    dt_val = jnp.asarray(dt, dtype=jnp.real(v0).dtype)

    def step(state, _):
        v, _mu = state
        v_next_raw = _advance_imex2(v, cache, params, term_cfg, dt_val)
        denom = jnp.vdot(v, v)
        denom = jnp.where(denom == 0.0, 1.0, denom)
        mu = jnp.vdot(v, v_next_raw) / denom
        v_next = _normalize(v_next_raw)
        return (v_next, mu), None

    v0 = _normalize(v0)
    (v, mu), _ = jax.lax.scan(step, (v0, jnp.asarray(0.0, dtype=v0.dtype)), None, length=iterations)
    eig = jnp.log(mu) / dt_val
    return eig, v


def _arnoldi(
    v0: jnp.ndarray,
    apply_op,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    krylov_dim: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    v0 = _normalize(v0)
    V = jnp.zeros((krylov_dim + 1,) + v0.shape, dtype=v0.dtype)
    H = jnp.zeros((krylov_dim + 1, krylov_dim), dtype=v0.dtype)
    V = V.at[0].set(v0)

    def outer(i, carry):
        V, H = carry
        w = apply_op(V[i], cache, params, term_cfg)

        def inner(j, inner_carry):
            w, H = inner_carry
            h = jnp.vdot(V[j], w)
            w = w - h * V[j]
            H = H.at[j, i].set(h)
            return w, H

        def reorth(j, inner_carry):
            w, H = inner_carry
            h = jnp.vdot(V[j], w)
            w = w - h * V[j]
            H = H.at[j, i].add(h)
            return w, H

        w, H = jax.lax.fori_loop(0, i + 1, inner, (w, H))
        w, H = jax.lax.fori_loop(0, i + 1, reorth, (w, H))
        h_next = jnp.linalg.norm(w)
        H = H.at[i + 1, i].set(h_next)
        v_next = jnp.where(h_next > 0.0, w / h_next, w)
        V = V.at[i + 1].set(v_next)
        return V, H

    V, H = jax.lax.fori_loop(0, krylov_dim, outer, (V, H))
    return V, H


@partial(
    jax.jit,
    static_argnames=(
        "krylov_dim",
        "restarts",
        "shift_preconditioner",
        "gmres_restart",
        "gmres_maxiter",
        "gmres_solve_method",
        "select_targeted",
        "select_growth",
        "select_overlap",
    ),
)
def dominant_eigenpair_shift_invert_cached(
    v0: jnp.ndarray,
    v_ref: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    krylov_dim: int,
    restarts: int,
    sigma: jnp.ndarray,
    omega_min_factor: float,
    omega_target_factor: float,
    omega_cap_factor: float,
    omega_sign: int,
    gmres_tol: float,
    gmres_maxiter: int,
    gmres_restart: int,
    gmres_solve_method: str,
    shift_preconditioner: str | None,
    select_targeted: bool,
    select_growth: bool,
    select_overlap: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Restarted shift-invert Arnoldi with GMRES solves."""

    sigma_val = jnp.asarray(sigma, dtype=v0.dtype)
    shape = v0.shape
    size = v0.size
    _precond, precond_op = _build_shift_invert_precond(
        v0, cache, params, term_cfg, sigma_val, shift_preconditioner
    )

    def matvec(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        return (_apply_operator(x, cache, params, term_cfg) - sigma_val * x).reshape(size)

    def apply_shift_invert(x: jnp.ndarray, _cache, _params, _term_cfg) -> jnp.ndarray:
        b = x.reshape(size)
        x0 = precond_op(b) if precond_op is not None else b
        sol, _info = gmres(
            matvec,
            b,
            x0=x0,
            tol=gmres_tol,
            maxiter=gmres_maxiter,
            restart=gmres_restart,
            M=precond_op,
            solve_method=gmres_solve_method,
        )
        return sol.reshape(shape)

    def restart_body(i, state):
        v, _eig_prev = state
        V, H = _arnoldi(v, apply_shift_invert, cache, params, term_cfg, krylov_dim)
        Hk = H[:krylov_dim, :krylov_dim]
        eigvals, eigvecs = jnp.linalg.eig(Hk)
        safe = jnp.where(jnp.abs(eigvals) > 1.0e-14, eigvals, 1.0e-14 + 0.0j)
        lam = sigma_val + 1.0 / safe
        real_part = jnp.real(lam)
        imag_part = jnp.imag(lam)
        finite = jnp.isfinite(real_part) & jnp.isfinite(imag_part)
        omega_scale = _omega_scale(cache, params)
        omega_cap = omega_cap_factor * omega_scale
        omega_min = omega_min_factor * omega_scale
        use_cap = omega_cap_factor > 0.0
        use_min = omega_min_factor > 0.0
        mask0 = jnp.abs(imag_part) <= omega_cap
        mask0 = jnp.where(use_min, mask0 & (jnp.abs(imag_part) >= omega_min), mask0)
        mask0 = mask0 & finite
        mask0_any = jnp.any(mask0)
        omega_sign_val = jnp.asarray(omega_sign, dtype=imag_part.dtype)
        use_sign = omega_sign_val != 0.0
        mask_sign = (omega_sign_val * imag_part) >= 0.0
        mask = jnp.where(use_sign, mask0 & mask_sign, mask0)
        use_sign_mask = jnp.any(mask)
        mask = jnp.where(use_sign_mask, mask, mask0)
        dist = jnp.abs(lam - sigma_val)
        dist = jnp.where(finite, dist, jnp.inf)
        dist_masked = jnp.where(mask, dist, jnp.inf)
        idx_masked = jnp.argmin(dist_masked)
        idx_all = jnp.argmin(dist)
        has_mask = jnp.any(mask)
        idx = jnp.where(has_mask, idx_masked, idx_all)
        real_masked = jnp.where(mask, real_part, -jnp.inf)
        if select_growth:
            idx_growth = jnp.argmax(real_masked)
            has_growth = jnp.any(mask & (real_part >= 0.0))
            idx = jnp.where(has_growth, idx_growth, idx)
        if select_targeted:
            idx = _select_by_target(
                real_part,
                imag_part,
                mask,
                omega_scale,
                omega_target_factor,
                omega_sign,
                idx,
            )
        if select_overlap:
            idx = _select_by_overlap(eigvecs, V[:krylov_dim], v_ref, mask, idx)
        y = eigvecs[:, idx]
        v_next = jnp.tensordot(jnp.conj(y), V[:krylov_dim], axes=1)
        v_next = _normalize(v_next)
        eig_out = jnp.where(mask0_any, lam[idx], jnp.nan + 1.0j * jnp.nan)
        return v_next, eig_out

    v, eig = jax.lax.fori_loop(
        0, restarts, restart_body, (v0, jnp.asarray(0.0, dtype=v0.dtype))
    )
    return eig, v


@partial(jax.jit, static_argnames=("krylov_dim", "restarts", "select_overlap"))
def dominant_eigenpair_cached(
    v0: jnp.ndarray,
    v_ref: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    krylov_dim: int,
    restarts: int,
    omega_min_factor: float,
    omega_target_factor: float,
    omega_cap_factor: float,
    omega_sign: int,
    select_overlap: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Approximate the dominant eigenvalue (max real part) with restarted Arnoldi."""

    v = v0
    eig0 = jnp.asarray(0.0, dtype=v0.dtype)

    def restart_body(i, state):
        v, _eig_prev = state
        V, H = _arnoldi(v, _apply_operator, cache, params, term_cfg, krylov_dim)
        Hk = H[:krylov_dim, :krylov_dim]
        eigvals, eigvecs = jnp.linalg.eig(Hk)
        real_part = jnp.real(eigvals)
        imag_part = jnp.imag(eigvals)
        omega_scale = _omega_scale(cache, params)
        omega_cap = omega_cap_factor * omega_scale
        omega_min = omega_min_factor * omega_scale
        use_cap = omega_cap_factor > 0.0
        use_min = omega_min_factor > 0.0
        mask0 = jnp.abs(imag_part) <= omega_cap
        mask0 = jnp.where(use_min, mask0 & (jnp.abs(imag_part) >= omega_min), mask0)
        mask0_any = jnp.any(mask0)
        omega_sign_val = jnp.asarray(omega_sign, dtype=imag_part.dtype)
        use_sign = omega_sign_val != 0.0
        mask_sign = (omega_sign_val * imag_part) >= 0.0
        mask = jnp.where(use_sign, mask0 & mask_sign, mask0)
        use_sign_mask = jnp.any(mask)
        mask = jnp.where(use_sign_mask, mask, mask0)
        real_masked = jnp.where(mask, real_part, -jnp.inf)
        idx_masked = jnp.argmax(real_masked)
        idx_small = jnp.argmin(jnp.abs(imag_part))
        use_mask = use_cap & jnp.any(mask)
        idx = jnp.where(use_mask, idx_masked, idx_small)
        idx = jnp.where(mask0_any, idx, jnp.argmax(real_part))
        idx = _select_by_target(
            real_part,
            imag_part,
            mask,
            omega_scale,
            omega_target_factor,
            omega_sign,
            idx,
        )
        if select_overlap:
            idx = _select_by_overlap(eigvecs, V[:krylov_dim], v_ref, mask, idx)
        eig = eigvals[idx]
        y = eigvecs[:, idx]
        v_next = jnp.tensordot(jnp.conj(y), V[:krylov_dim], axes=1)
        v_next = _normalize(v_next)
        return v_next, eig

    v, eig = jax.lax.fori_loop(0, restarts, restart_body, (v, eig0))
    return eig, v


@partial(jax.jit, static_argnames=("krylov_dim", "restarts", "select_overlap"))
def dominant_eigenpair_propagator_cached(
    v0: jnp.ndarray,
    v_ref: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    krylov_dim: int,
    restarts: int,
    dt: float,
    omega_min_factor: float,
    omega_target_factor: float,
    omega_cap_factor: float,
    omega_sign: int,
    select_overlap: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Arnoldi on a stable IMEX2 propagator; eigenvalue from Rayleigh quotient."""

    v = v0
    dt_val = jnp.asarray(dt, dtype=jnp.real(v0).dtype)

    def apply_prop(x, cache, params, term_cfg):
        return _advance_imex2(x, cache, params, term_cfg, dt_val)

    def restart_body(i, state):
        v, _eig_prev = state
        V, H = _arnoldi(v, apply_prop, cache, params, term_cfg, krylov_dim)
        Hk = H[:krylov_dim, :krylov_dim]
        eigvals, eigvecs = jnp.linalg.eig(Hk)
        lam = jnp.log(eigvals) / dt_val
        real_part = jnp.real(lam)
        imag_part = jnp.imag(lam)
        omega_scale = _omega_scale(cache, params)
        omega_cap = omega_cap_factor * omega_scale
        omega_min = omega_min_factor * omega_scale
        use_cap = omega_cap_factor > 0.0
        use_min = omega_min_factor > 0.0
        mask0 = jnp.abs(imag_part) <= omega_cap
        mask0 = jnp.where(use_min, mask0 & (jnp.abs(imag_part) >= omega_min), mask0)
        omega_sign_val = jnp.asarray(omega_sign, dtype=imag_part.dtype)
        use_sign = omega_sign_val != 0.0
        mask_sign = (omega_sign_val * imag_part) >= 0.0
        mask = jnp.where(use_sign, mask0 & mask_sign, mask0)
        use_sign_mask = jnp.any(mask)
        mask = jnp.where(use_sign_mask, mask, mask0)
        real_masked = jnp.where(mask, real_part, -jnp.inf)
        idx_masked = jnp.argmax(real_masked)
        idx_small = jnp.argmin(jnp.abs(imag_part))
        use_mask = use_cap & jnp.any(mask)
        idx = jnp.where(use_mask, idx_masked, idx_small)
        idx = _select_by_target(
            real_part,
            imag_part,
            mask,
            omega_scale,
            omega_target_factor,
            omega_sign,
            idx,
        )
        if select_overlap:
            idx = _select_by_overlap(eigvecs, V[:krylov_dim], v_ref, mask, idx)
        y = eigvecs[:, idx]
        v_next = jnp.tensordot(jnp.conj(y), V[:krylov_dim], axes=1)
        v_next = _normalize(v_next)
        return v_next, lam[idx]

    v, eig_sel = jax.lax.fori_loop(
        0, restarts, restart_body, (v, jnp.asarray(0.0, dtype=v0.dtype))
    )
    Lv = _apply_operator(v, cache, params, term_cfg)
    num = jnp.vdot(v, Lv)
    den = jnp.vdot(v, v)
    eig_rayleigh = jnp.where(den == 0.0, 0.0, num / den)
    sel_finite = jnp.isfinite(jnp.real(eig_sel)) & jnp.isfinite(jnp.imag(eig_sel))
    prefer_rayleigh = (~sel_finite) | (
        (jnp.real(eig_sel) <= 0.0) & (jnp.real(eig_rayleigh) > jnp.real(eig_sel))
    )
    eig = jnp.where(prefer_rayleigh, eig_rayleigh, eig_sel)
    return eig, v


def dominant_eigenpair(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    v_ref: jnp.ndarray | None = None,
    select_overlap: bool = False,
    krylov_dim: int = 24,
    restarts: int = 2,
    omega_min_factor: float = 0.0,
    omega_target_factor: float = 0.0,
    omega_cap_factor: float = 2.0,
    omega_sign: int = 0,
    method: str = "power",
    power_iters: int = 40,
    power_dt: float = 0.01,
    shift: complex | None = None,
    shift_source: str = "propagator",
    shift_tol: float = 1.0e-4,
    shift_maxiter: int = 50,
    shift_restart: int = 20,
    shift_solve_method: str = "batched",
    shift_preconditioner: str | None = "damping",
    shift_selection: str = "targeted",
    mode_family: str = "auto",
    fallback_method: str = "propagator",
    fallback_real_floor: float = -1.0e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Python wrapper for the cached Krylov solver."""

    term_cfg = linear_terms_to_term_config(terms)
    v_ref_use = v0 if v_ref is None else v_ref
    method_key = method.strip().lower()
    mode_family_sign = _mode_family_sign(mode_family)
    omega_sign_eff = int(omega_sign) if int(omega_sign) != 0 else mode_family_sign
    if method_key == "power":
        return dominant_eigenpair_power(
            v0,
            cache,
            params,
            term_cfg,
            iterations=max(int(power_iters), 1),
            dt=float(power_dt),
        )
    if method_key == "propagator":
        return dominant_eigenpair_propagator_cached(
            v0,
            v_ref_use,
            cache,
            params,
            term_cfg,
            krylov_dim=krylov_dim,
            restarts=max(int(restarts), 1),
            dt=float(power_dt),
            omega_min_factor=float(omega_min_factor),
            omega_target_factor=float(omega_target_factor),
            omega_cap_factor=float(omega_cap_factor),
            omega_sign=omega_sign_eff,
            select_overlap=bool(select_overlap),
        )
    if method_key == "shift_invert":
        restarts = max(int(restarts), 1)
        if shift is None:
            shift_source_key = shift_source.strip().lower()
            if shift_source_key == "propagator":
                shift_est, v_seed = dominant_eigenpair_propagator_cached(
                    v0,
                    v_ref_use,
                    cache,
                    params,
                    term_cfg,
                    krylov_dim=max(int(krylov_dim), 1),
                    restarts=1,
                    dt=float(power_dt),
                    omega_min_factor=float(omega_min_factor),
                    omega_target_factor=float(omega_target_factor),
                    omega_cap_factor=float(omega_cap_factor),
                    omega_sign=omega_sign_eff,
                    select_overlap=False,
                )
                sigma = shift_est
                v_init = v_seed
            elif shift_source_key == "target":
                omega_scale = _omega_scale(cache, params)
                omega_target = float(omega_target_factor) * omega_scale
                if omega_sign_eff != 0:
                    omega_target = float(jnp.sign(omega_sign_eff)) * jnp.abs(omega_target)
                sigma = 1j * omega_target
                v_init = v0
            else:
                shift_est, v_seed = dominant_eigenpair_power(
                    v0,
                    cache,
                    params,
                    term_cfg,
                    iterations=max(int(power_iters), 1),
                    dt=float(power_dt),
                )
                sigma = shift_est
                v_init = v_seed
        else:
            sigma = jnp.asarray(shift, dtype=v0.dtype)
            v_init = v0
        selection_key = shift_selection.strip().lower()
        select_targeted = selection_key in {"targeted", "target", "auto", "default"}
        select_growth = selection_key in {"targeted", "growth", "auto", "default"}
        eig_si, vec_si = dominant_eigenpair_shift_invert_cached(
            v_init,
            v_ref_use,
            cache,
            params,
            term_cfg,
            krylov_dim=krylov_dim,
            restarts=restarts,
            sigma=sigma,
            omega_min_factor=float(omega_min_factor),
            omega_target_factor=float(omega_target_factor),
            omega_cap_factor=float(omega_cap_factor),
            omega_sign=omega_sign_eff,
            gmres_tol=shift_tol,
            gmres_maxiter=max(int(shift_maxiter), 1),
            gmres_restart=max(int(shift_restart), 1),
            gmres_solve_method=shift_solve_method,
            shift_preconditioner=shift_preconditioner,
            select_targeted=select_targeted,
            select_growth=select_growth,
            select_overlap=bool(select_overlap),
        )
        fallback_key = fallback_method.strip().lower()
        eig_host = complex(np.asarray(eig_si))
        need_fallback = (
            not np.isfinite(eig_host.real)
            or not np.isfinite(eig_host.imag)
            or eig_host.real < float(fallback_real_floor)
        )
        if need_fallback and fallback_key != "none":
            if fallback_key == "propagator":
                return dominant_eigenpair_propagator_cached(
                    v0,
                    v_ref_use,
                    cache,
                    params,
                    term_cfg,
                    krylov_dim=krylov_dim,
                    restarts=max(int(restarts), 1),
                    dt=float(power_dt),
                    omega_min_factor=float(omega_min_factor),
                    omega_target_factor=float(omega_target_factor),
                    omega_cap_factor=float(omega_cap_factor),
                    omega_sign=omega_sign_eff,
                    select_overlap=False,
                )
            if fallback_key == "arnoldi":
                return dominant_eigenpair_cached(
                    v0,
                    v_ref_use,
                    cache,
                    params,
                    term_cfg,
                    krylov_dim=krylov_dim,
                    restarts=max(int(restarts), 1),
                    omega_min_factor=float(omega_min_factor),
                    omega_target_factor=float(omega_target_factor),
                    omega_cap_factor=float(omega_cap_factor),
                    omega_sign=omega_sign_eff,
                    select_overlap=bool(select_overlap),
                )
            if fallback_key == "power":
                return dominant_eigenpair_power(
                    v0,
                    cache,
                    params,
                    term_cfg,
                    iterations=max(int(power_iters), 1),
                    dt=float(power_dt),
                )
        return eig_si, vec_si
    if method_key != "arnoldi":
        raise ValueError(
            "Krylov method must be 'power', 'propagator', 'shift_invert', or 'arnoldi'"
        )

    restarts = max(int(restarts), 1)
    return dominant_eigenpair_cached(
        v0,
        v_ref_use,
        cache,
        params,
        term_cfg,
        krylov_dim=krylov_dim,
        restarts=restarts,
        omega_min_factor=float(omega_min_factor),
        omega_target_factor=float(omega_target_factor),
        omega_cap_factor=float(omega_cap_factor),
        omega_sign=omega_sign_eff,
        select_overlap=bool(select_overlap),
    )


def dominant_eigenvalue(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    krylov_dim: int = 24,
    restarts: int = 2,
) -> jnp.ndarray:
    eig, _vec = dominant_eigenpair(
        v0,
        cache,
        params,
        terms,
        krylov_dim=krylov_dim,
        restarts=restarts,
    )
    return eig
