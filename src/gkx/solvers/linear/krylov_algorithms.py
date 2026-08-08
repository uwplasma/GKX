"""Matrix-free operators and compiled Krylov eigenmode algorithms."""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from solvax import gmres

from gkx.operators.linear.cache_arrays import (
    collision_damping,
    hypercollision_damping,
)
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.params import LinearParams
from gkx.terms.assembly import assemble_rhs_cached
from gkx.terms.config import TermConfig


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
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    if v.ndim == 5 and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]
    damping = (
        collision_damping(cache, params, real_dtype, squeeze_species=(v.ndim == 5))
        + hyper_damp
    )
    return damping.astype(real_dtype)


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


def _omega_scale(cache: LinearCache, params: LinearParams) -> jnp.ndarray:
    """Return the frequency scale used for branch and target selection."""

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
    """Map named branch families to the sign convention for physical frequency."""

    key = mode_family.strip().lower()
    if key in {"ion", "itg", "cyclone", "kbm", "positive"}:
        return 1
    if key in {"electron", "etg", "tem", "negative"}:
        return -1
    return 0


def _physical_omega(imag_part: jnp.ndarray) -> jnp.ndarray:
    """Map eigenvalue imaginary part to reported physical frequency."""

    return -imag_part


def _select_by_overlap(
    eigvecs: jnp.ndarray,
    V: jnp.ndarray,
    v_ref: jnp.ndarray,
    mask: jnp.ndarray,
    fallback_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Select the eigenpair with maximal overlap to ``v_ref`` within ``mask``."""

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
    """Select the branch nearest the requested physical-frequency target."""

    omega_target_factor_val = jnp.asarray(omega_target_factor, dtype=imag_part.dtype)
    use_target = omega_target_factor_val > 0.0
    omega_target = omega_target_factor_val * omega_scale
    omega_phys = _physical_omega(imag_part)
    omega_sign_val = jnp.asarray(omega_sign, dtype=imag_part.dtype)
    use_sign = omega_sign_val != 0.0
    omega_target = jnp.where(
        use_sign,
        jnp.sign(omega_sign_val) * jnp.abs(omega_target),
        omega_target,
    )
    use_mask = jnp.any(mask)
    mask_use = jnp.where(use_mask, mask, jnp.ones_like(mask, dtype=bool))
    mask_pos = real_part >= 0.0
    mask_target = mask_use
    has_pos = jnp.any(mask_target & mask_pos)
    mask_target = jnp.where(has_pos, mask_target & mask_pos, mask_target)
    dist = jnp.abs(omega_phys - omega_target)
    dist_masked = jnp.where(mask_target, dist, jnp.inf)
    idx_target = jnp.argmin(dist_masked)
    has_target = jnp.any(mask_target)
    use_choice = use_target & has_target
    return jnp.where(use_choice, idx_target, fallback_idx)


# Accepted preconditioner names, grouped by the builder they select. Single
# source for validation and dispatch, so nothing is advertised without resolving.
SHIFT_PRECOND_LINE_NAMES: frozenset[str] = frozenset(
    {"hermite-line", "hermite_line", "hermite", "streaming-line", "streaming_line",
     "hermite-line-coarse", "hermite_line_coarse", "hermite_coarse",
     "streaming-line-coarse"}
)  # fmt: skip
SHIFT_PRECOND_FIELD_NAMES: frozenset[str] = frozenset(
    {"field-corrected", "field_corrected", "field-schur", "field_schur",
     "field-corrected-coarse", "field_corrected_coarse"}
)  # fmt: skip
SHIFT_PRECOND_NAMES: frozenset[str] = (
    SHIFT_PRECOND_LINE_NAMES | SHIFT_PRECOND_FIELD_NAMES | {"damping", "none"}
)


def _build_shift_invert_precond(
    v: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    sigma: jnp.ndarray,
    mode: str | None,
) -> tuple[jnp.ndarray | None, Callable[[jnp.ndarray], jnp.ndarray] | None]:
    """Build the preconditioner used inside shift-invert Krylov GMRES solves.

    ``None``/``"none"`` are the explicit opt-outs; any other unrecognized name
    raises. This used to return ``(None, None)`` for those too, so a typo
    disabled preconditioning while the solve reported itself as preconditioned.
    ``"auto"`` must not arrive here -- it is a real policy resolved one level up
    by :func:`gkx.solvers.linear.krylov._automatic_shift_preconditioner`, and
    mapping it to ``"damping"`` would hand a damping diagonal to a caller that
    asked for a physics-aware line solve.
    """

    mode_key = "none" if mode is None else mode.strip().lower()
    if mode_key == "none":
        return None, None
    if mode_key not in SHIFT_PRECOND_NAMES:
        raise ValueError(
            f"unknown shift-invert preconditioner {mode!r}; accepted names are "
            f"{', '.join(repr(n) for n in sorted(SHIFT_PRECOND_NAMES))}, or None to "
            "disable preconditioning. 'auto' is resolved above this layer by "
            "_automatic_shift_preconditioner; reaching here means it was bypassed."
        )
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

    # Import lazily to keep the core operator module independent of the
    # implicit integration policy at import time.
    from gkx.solvers.linear.implicit import (
        _build_field_corrected_shifted_preconditioner,
        _build_shifted_hermite_preconditioner,
    )

    coarse = mode_key.endswith(("-coarse", "_coarse"))
    if mode_key in SHIFT_PRECOND_FIELD_NAMES:
        apply_preconditioner = _build_field_corrected_shifted_preconditioner(
            v,
            cache,
            params,
            term_cfg,
            sigma,
            coarse=coarse,
        )
    else:
        apply_preconditioner = _build_shifted_hermite_preconditioner(
            v,
            cache,
            params,
            term_cfg,
            sigma,
            coarse=coarse,
        )
    damping = _compute_damping(v, cache, params)
    diagonal = -damping.astype(v.dtype) - sigma
    safe_diagonal = jnp.where(jnp.abs(diagonal) > 0.0, diagonal, 1.0 + 0.0j)
    damping_inverse = 1.0 / safe_diagonal
    shift_floor = jnp.sqrt(jnp.finfo(jnp.real(v).dtype).eps)
    use_line = jnp.abs(sigma) > shift_floor

    def apply_with_zero_shift_fallback(x_flat: jnp.ndarray) -> jnp.ndarray:
        line_result = apply_preconditioner(x_flat)
        damping_result = (x_flat.reshape(v.shape) * damping_inverse).reshape(v.size)
        return jnp.where(use_line, line_result, damping_result)

    return None, apply_with_zero_shift_fallback


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
    (v, mu), _ = jax.lax.scan(
        step, (v0, jnp.asarray(0.0, dtype=v0.dtype)), None, length=iterations
    )
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
        operator_scale = jnp.linalg.norm(w)

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
        real_dtype = jnp.real(jnp.empty((), dtype=v0.dtype)).dtype
        breakdown_tol = 10.0 * jnp.finfo(real_dtype).eps * operator_scale
        resolved = h_next > breakdown_tol
        H = H.at[i + 1, i].set(jnp.where(resolved, h_next, 0.0))
        safe_norm = jnp.where(resolved, h_next, 1.0)
        v_next = jnp.where(resolved, w / safe_norm, jnp.zeros_like(w))
        V = V.at[i + 1].set(v_next)
        return V, H

    V, H = jax.lax.fori_loop(0, krylov_dim, outer, (V, H))
    return V, H


def _shift_invert_apply_factory(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    sigma_val: jnp.ndarray,
    gmres_tol: float,
    gmres_maxiter: int,
    gmres_restart: int,
    gmres_solve_method: str,
    shift_preconditioner: str | None,
):
    if gmres_solve_method not in {"batched", "incremental", "flexible"}:
        raise ValueError(
            "shift_solve_method must be 'batched', 'incremental', or 'flexible'"
        )
    shape = v0.shape
    size = v0.size
    _precond, precond_op = _build_shift_invert_precond(
        v0, cache, params, term_cfg, sigma_val, shift_preconditioner
    )

    def matvec(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        return (_apply_operator(x, cache, params, term_cfg) - sigma_val * x).reshape(
            size
        )

    def apply_shift_invert(x: jnp.ndarray, _cache, _params, _term_cfg) -> jnp.ndarray:
        b = x.reshape(size)
        # SOLVAX FGMRES applies the structured inverse on the right, so its
        # least-squares norm is the physical residual ||b - (A-sigma I)x||.
        # A left-preconditioned solve can report convergence in a transformed
        # norm while this residual remains O(1), which previously forced an
        # expensive unpreconditioned retry for nearly every right-hand side.
        restart = min(max(gmres_restart, 1), gmres_maxiter, size)
        max_restarts = max(1, math.ceil(gmres_maxiter / restart))
        initial_guess = precond_op(b) if precond_op is not None else b
        solution = gmres(
            matvec,
            b,
            x0=initial_guess,
            precond=precond_op,
            rtol=gmres_tol,
            restart=restart,
            max_restarts=max_restarts,
        )
        return solution.x.reshape(shape)

    return apply_shift_invert


def _shift_invert_spectrum(
    eigvals: jnp.ndarray,
    sigma_val: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    real_dtype = jnp.real(eigvals).dtype
    scale = jnp.maximum(jnp.max(jnp.abs(eigvals)), 1.0)
    valid = jnp.abs(eigvals) > 100.0 * jnp.finfo(real_dtype).eps * scale
    safe = jnp.where(valid, eigvals, 1.0 + 0.0j)
    lam = sigma_val + 1.0 / safe
    lam = jnp.where(valid, lam, jnp.asarray(jnp.nan + 1j * jnp.nan, lam.dtype))
    real_part = jnp.real(lam)
    imag_part = jnp.imag(lam)
    finite = valid & jnp.isfinite(real_part) & jnp.isfinite(imag_part)
    return lam, real_part, imag_part, finite


def _shift_invert_frequency_masks(
    *,
    imag_part: jnp.ndarray,
    finite: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    omega_min_factor: float,
    omega_cap_factor: float,
    omega_sign: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return _frequency_masks_from_imaginary_part(
        imag_part=imag_part,
        finite=finite,
        cache=cache,
        params=params,
        omega_min_factor=omega_min_factor,
        omega_cap_factor=omega_cap_factor,
        omega_sign=omega_sign,
    )


def _shift_invert_nearest_shift_index(
    *,
    lam: jnp.ndarray,
    sigma_val: jnp.ndarray,
    finite: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    dist = jnp.abs(lam - sigma_val)
    dist = jnp.where(finite, dist, jnp.inf)
    dist_masked = jnp.where(mask, dist, jnp.inf)
    idx_masked = jnp.argmin(dist_masked)
    idx_all = jnp.argmin(dist)
    return jnp.where(jnp.any(mask), idx_masked, idx_all)


def _shift_invert_mode_index(
    *,
    eigvecs: jnp.ndarray,
    V: jnp.ndarray,
    v_ref: jnp.ndarray,
    lam: jnp.ndarray,
    sigma_val: jnp.ndarray,
    real_part: jnp.ndarray,
    imag_part: jnp.ndarray,
    finite: jnp.ndarray,
    mask: jnp.ndarray,
    omega_scale: jnp.ndarray,
    omega_target_factor: float,
    omega_sign: int,
    select_growth: bool,
    select_targeted: bool,
    select_overlap: bool,
    krylov_dim: int,
) -> jnp.ndarray:
    idx = _shift_invert_nearest_shift_index(
        lam=lam,
        sigma_val=sigma_val,
        finite=finite,
        mask=mask,
    )
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
    return idx


def _frequency_masks_from_imaginary_part(
    *,
    imag_part: jnp.ndarray,
    finite: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    omega_min_factor: float,
    omega_cap_factor: float,
    omega_sign: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    omega_scale = _omega_scale(cache, params)
    omega_cap = omega_cap_factor * omega_scale
    omega_min = omega_min_factor * omega_scale
    use_min = omega_min_factor > 0.0
    mask0 = jnp.abs(imag_part) <= omega_cap
    mask0 = jnp.where(use_min, mask0 & (jnp.abs(imag_part) >= omega_min), mask0)
    mask0 = mask0 & finite
    omega_phys = _physical_omega(imag_part)
    omega_sign_val = jnp.asarray(omega_sign, dtype=imag_part.dtype)
    use_sign = omega_sign_val != 0.0
    mask_sign = (omega_sign_val * omega_phys) >= 0.0
    signed_mask = jnp.where(use_sign, mask0 & mask_sign, mask0)
    mask = jnp.where(jnp.any(signed_mask), signed_mask, mask0)
    return mask0, mask, omega_scale


def _arnoldi_mode_index(
    *,
    eigvecs: jnp.ndarray,
    V: jnp.ndarray,
    v_ref: jnp.ndarray,
    real_part: jnp.ndarray,
    imag_part: jnp.ndarray,
    mask0: jnp.ndarray,
    mask: jnp.ndarray,
    omega_scale: jnp.ndarray,
    omega_target_factor: float,
    omega_cap_factor: float,
    omega_sign: int,
    select_overlap: bool,
    krylov_dim: int,
    fallback_to_growth_when_no_mask: bool,
) -> jnp.ndarray:
    real_masked = jnp.where(mask, real_part, -jnp.inf)
    idx_masked = jnp.argmax(real_masked)
    idx_small = jnp.argmin(jnp.abs(imag_part))
    use_mask = (omega_cap_factor > 0.0) & jnp.any(mask)
    idx = jnp.where(use_mask, idx_masked, idx_small)
    if fallback_to_growth_when_no_mask:
        idx = jnp.where(jnp.any(mask0), idx, jnp.argmax(real_part))
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
    return idx


def _ritz_vector_from_index(
    V: jnp.ndarray,
    eigvecs: jnp.ndarray,
    idx: jnp.ndarray,
    *,
    krylov_dim: int,
) -> jnp.ndarray:
    y = eigvecs[:, idx]
    return _normalize(jnp.tensordot(y, V[:krylov_dim], axes=1))


def _rayleigh_quotient(
    vector: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
) -> jnp.ndarray:
    """Return the least-residual eigenvalue for a fixed physical Ritz vector."""

    operator_vector = _apply_operator(vector, cache, params, term_cfg)
    denominator = jnp.vdot(vector, vector)
    safe_denominator = jnp.where(denominator != 0.0, denominator, 1.0 + 0.0j)
    return jnp.vdot(vector, operator_vector) / safe_denominator


def _advance_rk4(
    state: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Advance with a full-operator polynomial that preserves eigenvectors."""

    k1 = _apply_operator(state, cache, params, term_cfg)
    k2 = _apply_operator(state + 0.5 * dt * k1, cache, params, term_cfg)
    k3 = _apply_operator(state + 0.5 * dt * k2, cache, params, term_cfg)
    k4 = _apply_operator(state + dt * k3, cache, params, term_cfg)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _operator_arnoldi_restart_step(
    v: jnp.ndarray,
    v_ref: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    krylov_dim: int,
    omega_min_factor: float,
    omega_target_factor: float,
    omega_cap_factor: float,
    omega_sign: int,
    select_overlap: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    V, H = _arnoldi(v, _apply_operator, cache, params, term_cfg, krylov_dim)
    Hk = H[:krylov_dim, :krylov_dim]
    eigvals, eigvecs = jnp.linalg.eig(Hk)
    real_part = jnp.real(eigvals)
    imag_part = jnp.imag(eigvals)
    mask0, mask, omega_scale = _frequency_masks_from_imaginary_part(
        imag_part=imag_part,
        finite=jnp.ones_like(real_part, dtype=bool),
        cache=cache,
        params=params,
        omega_min_factor=omega_min_factor,
        omega_cap_factor=omega_cap_factor,
        omega_sign=omega_sign,
    )
    idx = _arnoldi_mode_index(
        eigvecs=eigvecs,
        V=V,
        v_ref=v_ref,
        real_part=real_part,
        imag_part=imag_part,
        mask0=mask0,
        mask=mask,
        omega_scale=omega_scale,
        omega_target_factor=omega_target_factor,
        omega_cap_factor=omega_cap_factor,
        omega_sign=omega_sign,
        select_overlap=select_overlap,
        krylov_dim=krylov_dim,
        fallback_to_growth_when_no_mask=True,
    )
    return _ritz_vector_from_index(V, eigvecs, idx, krylov_dim=krylov_dim), eigvals[idx]


def _propagator_arnoldi_restart_step(
    v: jnp.ndarray,
    v_ref: jnp.ndarray,
    apply_prop,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    krylov_dim: int,
    horizon: jnp.ndarray,
    growth_only: bool,
    omega_min_factor: float,
    omega_target_factor: float,
    omega_cap_factor: float,
    omega_sign: int,
    select_overlap: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    V, H = _arnoldi(v, apply_prop, cache, params, term_cfg, krylov_dim)
    Hk = H[:krylov_dim, :krylov_dim]
    eigvals, eigvecs = jnp.linalg.eig(Hk)
    lam = jnp.log(eigvals) / horizon
    if growth_only:
        if select_overlap:
            lifted = jnp.tensordot(
                eigvecs.T,
                V[:krylov_dim],
                axes=1,
            ).reshape(krylov_dim, -1)
            norms = jnp.linalg.norm(lifted, axis=1)
            safe_norms = jnp.where(norms > 0.0, norms, 1.0)
            lifted = lifted / safe_norms[:, None]
            reference = _normalize(v_ref).reshape(-1)
            idx = jnp.argmax(jnp.abs(lifted.conj() @ reference))
        else:
            # For P ~= exp(T A), |mu| = exp(T Re(lambda)). The phase may wrap
            # many times without changing the ordering by physical growth.
            idx = jnp.argmax(jnp.abs(eigvals))
        return (
            _ritz_vector_from_index(V, eigvecs, idx, krylov_dim=krylov_dim),
            lam[idx],
        )

    real_part = jnp.real(lam)
    imag_part = jnp.imag(lam)
    mask0, mask, omega_scale = _frequency_masks_from_imaginary_part(
        imag_part=imag_part,
        finite=jnp.ones_like(real_part, dtype=bool),
        cache=cache,
        params=params,
        omega_min_factor=omega_min_factor,
        omega_cap_factor=omega_cap_factor,
        omega_sign=omega_sign,
    )
    idx = _arnoldi_mode_index(
        eigvecs=eigvecs,
        V=V,
        v_ref=v_ref,
        real_part=real_part,
        imag_part=imag_part,
        mask0=mask0,
        mask=mask,
        omega_scale=omega_scale,
        omega_target_factor=omega_target_factor,
        omega_cap_factor=omega_cap_factor,
        omega_sign=omega_sign,
        select_overlap=select_overlap,
        krylov_dim=krylov_dim,
        fallback_to_growth_when_no_mask=False,
    )
    return _ritz_vector_from_index(V, eigvecs, idx, krylov_dim=krylov_dim), lam[idx]


def _shift_invert_restart_step(
    v: jnp.ndarray,
    v_ref: jnp.ndarray,
    apply_shift_invert,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    krylov_dim: int,
    sigma_val: jnp.ndarray,
    omega_min_factor: float,
    omega_target_factor: float,
    omega_cap_factor: float,
    omega_sign: int,
    select_targeted: bool,
    select_growth: bool,
    select_overlap: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    V, H = _arnoldi(v, apply_shift_invert, cache, params, term_cfg, krylov_dim)
    Hk = H[:krylov_dim, :krylov_dim]
    eigvals, eigvecs = jnp.linalg.eig(Hk)
    lam, real_part, imag_part, finite = _shift_invert_spectrum(eigvals, sigma_val)
    mask0, mask, omega_scale = _shift_invert_frequency_masks(
        imag_part=imag_part,
        finite=finite,
        cache=cache,
        params=params,
        omega_min_factor=omega_min_factor,
        omega_cap_factor=omega_cap_factor,
        omega_sign=omega_sign,
    )
    idx = _shift_invert_mode_index(
        eigvecs=eigvecs,
        V=V,
        v_ref=v_ref,
        lam=lam,
        sigma_val=sigma_val,
        real_part=real_part,
        imag_part=imag_part,
        finite=finite,
        mask=mask,
        omega_scale=omega_scale,
        omega_target_factor=omega_target_factor,
        omega_sign=omega_sign,
        select_growth=select_growth,
        select_targeted=select_targeted,
        select_overlap=select_overlap,
        krylov_dim=krylov_dim,
    )
    v_next = _ritz_vector_from_index(V, eigvecs, idx, krylov_dim=krylov_dim)
    # Inexact inner solves make sigma + 1/mu less accurate than the physical
    # Rayleigh quotient for the same Ritz vector. The latter minimizes the
    # two-norm residual over scalar eigenvalues and cannot weaken the outer gate.
    eig_refined = _rayleigh_quotient(v_next, cache, params, term_cfg)
    eig_out = jnp.where(jnp.any(mask0), eig_refined, jnp.nan + 1.0j * jnp.nan)
    return v_next, eig_out


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
    apply_shift_invert = _shift_invert_apply_factory(
        v0,
        cache,
        params,
        term_cfg,
        sigma_val=sigma_val,
        gmres_tol=gmres_tol,
        gmres_maxiter=gmres_maxiter,
        gmres_restart=gmres_restart,
        gmres_solve_method=gmres_solve_method,
        shift_preconditioner=shift_preconditioner,
    )

    def restart_body(i, state):
        del i
        v, _eig_prev = state
        v_next, eig_out = _shift_invert_restart_step(
            v,
            v_ref,
            apply_shift_invert,
            cache,
            params,
            term_cfg,
            krylov_dim=krylov_dim,
            sigma_val=sigma_val,
            omega_min_factor=omega_min_factor,
            omega_target_factor=omega_target_factor,
            omega_cap_factor=omega_cap_factor,
            omega_sign=omega_sign,
            select_targeted=select_targeted,
            select_growth=select_growth,
            select_overlap=select_overlap,
        )
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
        del i
        v, _eig_prev = state
        v_next, eig = _operator_arnoldi_restart_step(
            v,
            v_ref,
            cache,
            params,
            term_cfg,
            krylov_dim=krylov_dim,
            omega_min_factor=omega_min_factor,
            omega_target_factor=omega_target_factor,
            omega_cap_factor=omega_cap_factor,
            omega_sign=omega_sign,
            select_overlap=select_overlap,
        )
        return v_next, eig

    v, eig = jax.lax.fori_loop(0, restarts, restart_body, (v, eig0))
    return eig, v


@partial(
    jax.jit,
    static_argnames=(
        "krylov_dim",
        "restarts",
        "propagator_steps",
        "select_overlap",
    ),
)
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
    propagator_steps: int,
    omega_min_factor: float,
    omega_target_factor: float,
    omega_cap_factor: float,
    omega_sign: int,
    select_overlap: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Filter by finite-horizon growth and return a physical Rayleigh pair."""

    v = v0
    dt_val = jnp.asarray(dt, dtype=jnp.real(v0).dtype)
    horizon = dt_val * propagator_steps

    def apply_prop(x, cache, params, term_cfg):
        if propagator_steps == 1:
            return _advance_imex2(x, cache, params, term_cfg, dt_val)
        return jax.lax.fori_loop(
            0,
            propagator_steps,
            lambda _index, state: _advance_rk4(
                state,
                cache,
                params,
                term_cfg,
                dt_val,
            ),
            x,
        )

    def restart_body(i, state):
        del i
        v, _eig_prev = state
        return _propagator_arnoldi_restart_step(
            v,
            v_ref,
            apply_prop,
            cache,
            params,
            term_cfg,
            krylov_dim=krylov_dim,
            horizon=horizon,
            growth_only=propagator_steps > 1,
            omega_min_factor=omega_min_factor,
            omega_target_factor=omega_target_factor,
            omega_cap_factor=omega_cap_factor,
            omega_sign=omega_sign,
            select_overlap=select_overlap,
        )

    v, _eig_sel = jax.lax.fori_loop(
        0, restarts, restart_body, (v, jnp.asarray(0.0, dtype=v0.dtype))
    )
    return _rayleigh_quotient(v, cache, params, term_cfg), v


__all__ = [
    "_advance_imex2",
    "_advance_rk4",
    "_apply_operator",
    "_arnoldi",
    "_assemble_rhs_cached_novjp",
    "_build_shift_invert_precond",
    "_compute_damping",
    "_mode_family_sign",
    "_normalize",
    "_omega_scale",
    "_physical_omega",
    "_rayleigh_quotient",
    "_select_by_overlap",
    "_select_by_target",
    "dominant_eigenpair_cached",
    "dominant_eigenpair_power",
    "dominant_eigenpair_propagator_cached",
    "dominant_eigenpair_shift_invert_cached",
]
