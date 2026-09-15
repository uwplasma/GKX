"""Implicit linear solve policies for cache-backed gyrokinetic operators."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import lu_factor, lu_solve
from solvax import KrylovSolution, gmres, tridiagonal_solve

from gkx.operators.linear.cache_arrays import (
    collision_damping,
    hypercollision_damping,
    hypercollision_kz_coefficient,
)
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    PreconditionerSpec,
    _as_species_array,
    _resolve_implicit_preconditioner,
    _x64_enabled,
    linear_terms_to_term_config,
    term_config_to_linear_terms,
)
from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.operators.linear.streaming import _scatter_unique_linked_modes
from gkx.terms.assembly import assemble_rhs_cached_with_fields, compute_fields_cached
from gkx.terms.config import FieldState, TermConfig

__all__ = [
    "ImplicitSolveStats",
    "ImplicitSolveSummary",
    "_build_field_corrected_shifted_preconditioner",
    "_build_implicit_operator",
    "_build_shifted_hermite_preconditioner",
    "_integrate_linear_implicit_cached",
    "require_converged_implicit_solves",
]


class ImplicitSolveStats(NamedTuple):
    """Convergence summary of every implicit GMRES solve in one run.

    Each field is a scalar array, so the summary rides through ``lax.scan`` and
    ``fori_loop`` carries without a host callback. ``max_relative_residual`` is
    the largest SOLVAX true residual ``||b - A x|| / ||b||``; with right
    preconditioning this is the physical norm. ``unconverged_solves`` counts
    solves whose SOLVAX ``converged`` flag was false at ``rtol=implicit_tol``.
    """

    max_relative_residual: jax.Array
    max_iterations: jax.Array
    solves: jax.Array
    unconverged_solves: jax.Array


def _empty_implicit_solve_stats(state_dtype: Any) -> ImplicitSolveStats:
    zero = jnp.asarray(0, dtype=jnp.int32)
    residual = jnp.asarray(0.0, dtype=jnp.finfo(state_dtype).dtype)
    return ImplicitSolveStats(residual, zero, zero, zero)


class _GmresStatus(NamedTuple):
    """The status fields of one SOLVAX solve, as pytree auxiliary data."""

    residual_norm: jax.Array
    iterations: jax.Array
    converged: jax.Array


def _fold_implicit_solve_stats(
    total: ImplicitSolveStats,
    solution: KrylovSolution | _GmresStatus,
    rhs_flat: jnp.ndarray,
) -> ImplicitSolveStats:
    """Fold one solve into a running summary; no gradient flows through it."""

    squared = jnp.maximum(jnp.real(jnp.vdot(rhs_flat, rhs_flat)), 0.0)
    rhs_norm = jnp.sqrt(jnp.where(squared > 0, squared, 1.0))
    relative = jax.lax.stop_gradient(solution.residual_norm / rhs_norm)
    unconverged = jnp.where(jax.lax.stop_gradient(solution.converged), 0, 1)
    return ImplicitSolveStats(
        jnp.maximum(
            total.max_relative_residual,
            relative.astype(total.max_relative_residual.dtype),
        ),
        jnp.maximum(total.max_iterations, solution.iterations.astype(jnp.int32)),
        total.solves + 1,
        total.unconverged_solves + unconverged.astype(jnp.int32),
    )


@dataclass(frozen=True)
class ImplicitSolveSummary:
    """Host-side :class:`ImplicitSolveStats` of one run, as Python scalars."""

    max_relative_residual: float
    max_iterations: int
    solves: int
    unconverged_solves: int

    @property
    def converged(self) -> bool:
        return self.unconverged_solves == 0 and math.isfinite(
            self.max_relative_residual
        )

    @classmethod
    def from_stats(cls, stats: ImplicitSolveStats) -> ImplicitSolveSummary:
        return cls(
            max_relative_residual=float(np.asarray(stats.max_relative_residual)),
            max_iterations=int(np.asarray(stats.max_iterations)),
            solves=int(np.asarray(stats.solves)),
            unconverged_solves=int(np.asarray(stats.unconverged_solves)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "max_relative_residual": self.max_relative_residual,
            "max_iterations": self.max_iterations,
            "solves": self.solves,
            "unconverged_solves": self.unconverged_solves,
        }


def require_converged_implicit_solves(
    stats: ImplicitSolveStats, *, label: str
) -> ImplicitSolveSummary:
    """Fail closed at the host boundary when any implicit solve did not converge.

    The traced scans cannot raise, so they carry :class:`ImplicitSolveStats` and
    the runtime checks it here, as the eigen gates do for their residuals.
    """

    summary = ImplicitSolveSummary.from_stats(stats)
    if summary.converged:
        return summary
    raise RuntimeError(
        f"{label}: {summary.unconverged_solves} of {summary.solves} implicit GMRES "
        "solves did not converge "
        f"(max_relative_residual={summary.max_relative_residual:.6g}, "
        f"max_iterations={summary.max_iterations}); refusing to return an "
        "unconverged trajectory. Raise implicit_maxiter or implicit_restart, use "
        "a stronger implicit_preconditioner or a smaller dt; call the integrator "
        "with return_solve_stats=True to inspect such a trajectory"
    )


@dataclass(frozen=True)
class _ImplicitState:
    G: jnp.ndarray
    shape: tuple[int, ...]
    size: int
    dt_val: jnp.ndarray
    real_dtype: jnp.dtype
    state_dtype: jnp.dtype
    squeeze_species: bool
    terms: LinearTerms


@dataclass(frozen=True)
class _ImplicitPreconditionerData:
    precond_full: jnp.ndarray
    precond_damp: jnp.ndarray
    precond_pas: jnp.ndarray
    vth: jnp.ndarray
    w_stream: jnp.ndarray
    sqrt_m_line: jnp.ndarray
    sqrt_p_line: jnp.ndarray
    imag: jnp.ndarray
    hyper_kz: jnp.ndarray


@dataclass(frozen=True)
class _ImplicitSolveOptions:
    tol: float
    maxiter: int
    iters: int
    relax: float
    restart: int


_IMPLICIT_PRECONDITIONER_ALIASES = {
    "full": frozenset({"auto", "diag", "diagonal", "physics", "block"}),
    "damping": frozenset({"damping", "collisional", "hyper"}),
    "pas": frozenset({"pas", "pas-line", "pas_line"}),
    "pas_coarse": frozenset(
        {"pas-coarse", "pas_schur", "block-schur", "schur", "pas-hybrid"}
    ),
    "hermite_line": frozenset(
        {"hermite-line", "hermite_line", "hermite", "streaming-line", "streaming_line"}
    ),
    "hermite_line_coarse": frozenset(
        {
            "hermite-line-coarse",
            "hermite_line_coarse",
            "hermite_coarse",
            "streaming-line-coarse",
        }
    ),
    "identity": frozenset({"identity", "none", "off"}),
}


def _prepare_implicit_state(
    G0: jnp.ndarray,
    dt: complex | jax.Array,
    terms: LinearTerms | TermConfig | None,
) -> _ImplicitState:
    terms = (
        term_config_to_linear_terms(terms)
        if isinstance(terms, TermConfig)
        else LinearTerms()
        if terms is None
        else terms
    )
    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_dtype = jnp.result_type(real_dtype, jnp.asarray(dt).dtype)
    dt_val = jnp.asarray(dt, dtype=dt_dtype)

    squeeze_species = False
    if G.ndim == 5:
        G = G[None, ...]
        squeeze_species = True
    shape = G.shape
    return _ImplicitState(
        G=G,
        shape=shape,
        size=int(np.prod(np.asarray(shape))),
        dt_val=dt_val,
        real_dtype=real_dtype,
        state_dtype=state_dtype,
        squeeze_species=squeeze_species,
        terms=terms,
    )


def _build_implicit_preconditioner_data(
    cache: LinearCache,
    params: LinearParams,
    state: _ImplicitState,
) -> _ImplicitPreconditionerData:
    real_dtype = state.real_dtype
    hyper_damp = state.terms.hypercollisions * hypercollision_damping(
        cache, params, real_dtype
    )
    hyper_kz = state.terms.hypercollisions * hypercollision_kz_coefficient(
        cache, params, real_dtype
    )
    damping = (
        state.terms.collisions
        * collision_damping(cache, params, real_dtype, squeeze_species=False)
        + hyper_damp
    ).astype(real_dtype)

    ell = cache.l.astype(real_dtype)
    m = cache.m.astype(real_dtype)
    cv_d = cache.cv_d.astype(real_dtype)
    gb_d = cache.gb_d.astype(real_dtype)
    w_curv = jnp.asarray(state.terms.curvature, dtype=real_dtype)
    w_gradb = jnp.asarray(state.terms.gradb, dtype=real_dtype)
    diag = jnp.zeros_like(damping, dtype=state.state_dtype)
    imag = jnp.asarray(1j, dtype=state.state_dtype)
    ns = state.shape[0]
    tz = _as_species_array(params.tz, ns, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)
    tz_b = tz[:, None, None, None, None, None]
    vth_b = vth[:, None, None, None, None, None]
    omega_d_scale = jnp.asarray(params.omega_d_scale, dtype=real_dtype)
    diag = diag - imag * tz_b * omega_d_scale * (
        w_curv * cv_d[None, None, None, ...] * (2.0 * m + 1.0)
        + w_gradb * gb_d[None, None, None, ...] * (2.0 * ell + 1.0)
    )
    precond_full = 1.0 / (1.0 + state.dt_val * damping - state.dt_val * diag)
    precond_full = precond_full.astype(state.G.dtype)
    precond_damp = (1.0 / (1.0 + state.dt_val * damping)).astype(state.G.dtype)
    kpar = params.kpar_scale * cache.kz.astype(real_dtype)
    w_stream = jnp.asarray(state.terms.streaming, dtype=real_dtype)
    kpar_b = kpar[None, None, None, None, None, :]
    precond_pas = 1.0 / (
        1.0
        + state.dt_val * damping
        - state.dt_val * diag
        + imag * state.dt_val * w_stream * vth_b * kpar_b
    )
    return _ImplicitPreconditionerData(
        precond_full=precond_full.astype(state.G.dtype),
        precond_damp=precond_damp,
        precond_pas=precond_pas.astype(state.G.dtype),
        vth=vth,
        w_stream=w_stream,
        sqrt_m_line=cache.sqrt_m_ladder.reshape(-1).astype(real_dtype),
        sqrt_p_line=cache.sqrt_p.reshape(-1).astype(real_dtype),
        imag=imag,
        hyper_kz=jnp.broadcast_to(hyper_kz, (ns, 1, state.shape[2], 1, 1, 1)),
    )


def _solve_tridiagonal_last_axis(
    lower: jnp.ndarray,
    diagonal: jnp.ndarray,
    upper: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """Solve independent tridiagonal systems stored on the last axis.

    GKX stores the Hermite line last, while SOLVAX uses a leading
    system axis so every trailing dimension is an independent column. The two
    axis moves are views at the JAX level and keep the physics layout out of the
    reusable structured solver.
    """

    return jnp.moveaxis(
        tridiagonal_solve(
            jnp.moveaxis(lower, -1, 0),
            jnp.moveaxis(diagonal, -1, 0),
            jnp.moveaxis(upper, -1, 0),
            jnp.moveaxis(rhs, -1, 0),
            method="auto",
        ),
        0,
        -1,
    )


def _solve_hermite_lines_fft(
    x: jnp.ndarray,
    *,
    kz: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    state: _ImplicitState,
    data: _ImplicitPreconditionerData,
) -> jnp.ndarray:
    """Invert the averaged diagonal plus Hermite streaming line."""

    x_hat = jnp.fft.fft(x, axis=-1)
    x_hat_mlast = jnp.moveaxis(x_hat, 2, -1)
    coeff = (
        (
            state.dt_val
            * data.w_stream
            * jnp.asarray(params.kpar_scale, dtype=state.real_dtype)
        )
        * data.vth[:, None, None, None, None]
        * (data.imag * kz)[None, None, None, None, :]
    )
    coeff = coeff[..., None]
    dl = coeff * data.sqrt_m_line
    du = coeff * data.sqrt_p_line
    du = du.at[..., -1].set(jnp.asarray(0.0, dtype=du.dtype))
    # Drifts vary along z and therefore couple Fourier modes.  Their mean
    # keeps the separable principal symbol D(l,m,kx,ky) + S(kz,m).
    hyper_symbol = state.dt_val * data.hyper_kz * jnp.abs(kz)
    diagonal = jnp.reciprocal(data.precond_full) - hyper_symbol
    d = jnp.moveaxis(jnp.mean(diagonal, axis=-1, keepdims=True) + hyper_symbol, 2, -1)
    batch_shape = x_hat_mlast.shape
    dl = jnp.broadcast_to(dl, batch_shape)
    d = jnp.broadcast_to(d, batch_shape)
    du = jnp.broadcast_to(du, batch_shape)
    y_hat_mlast = _solve_tridiagonal_last_axis(dl, d, du, x_hat_mlast)
    y_hat = jnp.moveaxis(y_hat_mlast, -1, 2)
    return jnp.fft.ifft(y_hat, axis=-1).astype(x.dtype)


def _solve_hermite_lines_linked(
    x: jnp.ndarray,
    *,
    cache: LinearCache,
    params: LinearParams,
    state: _ImplicitState,
    data: _ImplicitPreconditionerData,
) -> jnp.ndarray:
    """Linked-FFT variant of the Hermite-line streaming preconditioner."""

    if not cache.linked_indices:
        return _solve_hermite_lines_fft(
            x,
            kz=cache.kz,
            cache=cache,
            params=params,
            state=state,
            data=data,
        )

    Ny = x.shape[-3]
    Nx = x.shape[-2]
    Nz = x.shape[-1]
    lead_shape = x.shape[:-3]
    x_flat = jnp.swapaxes(x, -3, -2).reshape(*lead_shape, Nx * Ny, Nz)
    y_flat = jnp.zeros_like(x_flat)
    # Average only real-space coefficients; |kz| belongs to each linked FFT.
    diagonal = jnp.reciprocal(
        data.precond_full
    ) - state.dt_val * data.hyper_kz * jnp.abs(cache.kz)
    diagonal_flat = jnp.swapaxes(diagonal, -3, -2).reshape(*lead_shape, Nx * Ny, Nz)

    for idx_map, kz_link in zip(cache.linked_indices, cache.linked_kz):
        nChains, nLinks = idx_map.shape
        idx_flat = idx_map.reshape(-1)
        x_link = jnp.take(x_flat, idx_flat, axis=-2)
        x_link = x_link.reshape(*lead_shape, nChains, nLinks * Nz)
        x_hat = jnp.fft.fft(x_link, axis=-1)
        x_hat_mlast = jnp.moveaxis(x_hat, 2, -1)
        coeff = (
            (
                state.dt_val
                * data.w_stream
                * jnp.asarray(params.kpar_scale, dtype=state.real_dtype)
            )
            * data.vth[:, None, None, None]
            * (data.imag * kz_link)[None, None, None, :]
        )
        coeff = coeff[..., None]
        dl = coeff * data.sqrt_m_line
        du = coeff * data.sqrt_p_line
        du = du.at[..., -1].set(jnp.asarray(0.0, dtype=du.dtype))
        diagonal_link = jnp.take(diagonal_flat, idx_flat, axis=-2)
        diagonal_link = diagonal_link.reshape(
            *lead_shape,
            nChains,
            nLinks * Nz,
        )
        d = jnp.moveaxis(jnp.mean(diagonal_link, axis=-1), 2, -1)[..., None, :]
        d = (
            d
            + state.dt_val
            * data.hyper_kz.reshape(state.shape[0], 1, 1, 1, state.shape[2])
            * jnp.abs(kz_link)[:, None]
        )
        batch_shape = x_hat_mlast.shape
        dl = jnp.broadcast_to(dl, batch_shape)
        d = jnp.broadcast_to(d, batch_shape)
        du = jnp.broadcast_to(du, batch_shape)
        y_hat_mlast = _solve_tridiagonal_last_axis(dl, d, du, x_hat_mlast)
        y_hat = jnp.moveaxis(y_hat_mlast, -1, 2)
        y_link = jnp.fft.ifft(y_hat, axis=-1).astype(x.dtype)
        y_link = y_link.reshape(*lead_shape, nChains * nLinks, Nz)
        y_flat = _scatter_unique_linked_modes(y_flat, idx_flat, y_link)

    return jnp.swapaxes(y_flat.reshape(*lead_shape, Nx, Ny, Nz), -3, -2)


def _project_kx_coarse(x: jnp.ndarray, cache: LinearCache) -> jnp.ndarray:
    """Project/prolong a coarse kx correction without breaking linked chains."""

    if not cache.use_twist_shift or not cache.linked_indices:
        x_mean = jnp.mean(x, axis=4, keepdims=True)
        return jnp.broadcast_to(x_mean, x.shape)

    Ny = x.shape[-3]
    Nx = x.shape[-2]
    Nz = x.shape[-1]
    lead_shape = x.shape[:-3]
    x_flat = jnp.swapaxes(x, -3, -2).reshape(*lead_shape, Nx * Ny, Nz)
    y_flat = jnp.zeros_like(x_flat)

    for idx_map in cache.linked_indices:
        nChains, nLinks = idx_map.shape
        idx_flat = idx_map.reshape(-1)
        x_link = jnp.take(x_flat, idx_flat, axis=-2)
        x_link = x_link.reshape(*lead_shape, nChains, nLinks, Nz)
        x_mean = jnp.mean(x_link, axis=-2, keepdims=True)
        x_mean = jnp.broadcast_to(x_mean, x_link.shape)
        x_updates = x_mean.reshape(*lead_shape, nChains * nLinks, Nz)
        y_flat = _scatter_unique_linked_modes(y_flat, idx_flat, x_updates)

    return jnp.swapaxes(y_flat.reshape(*lead_shape, Nx, Ny, Nz), -3, -2)


def _canonical_implicit_preconditioner(
    implicit_preconditioner: PreconditionerSpec,
) -> Callable[[jnp.ndarray], jnp.ndarray] | str:
    resolved = _resolve_implicit_preconditioner(implicit_preconditioner)
    if callable(resolved):
        return resolved
    key = resolved or "auto"
    for canonical, aliases in _IMPLICIT_PRECONDITIONER_ALIASES.items():
        if key in aliases:
            return canonical
    raise ValueError(f"Unknown implicit_preconditioner '{resolved}'")


def _apply_factor_preconditioner(
    x_flat: jnp.ndarray,
    *,
    state: _ImplicitState,
    factor: jnp.ndarray,
) -> jnp.ndarray:
    x = x_flat.reshape(state.shape)
    return (x * factor).reshape(state.size)


def _apply_pas_coarse_preconditioner(
    x_flat: jnp.ndarray,
    *,
    cache: LinearCache,
    state: _ImplicitState,
    data: _ImplicitPreconditionerData,
) -> jnp.ndarray:
    x = x_flat.reshape(state.shape)
    x_line = x * data.precond_pas
    x_coarse = _project_kx_coarse(x, cache) * data.precond_pas
    x_line_coarse = _project_kx_coarse(x_line, cache)
    return (x_line + (x_coarse - x_line_coarse)).reshape(state.size)


def _apply_hermite_line_preconditioner(
    x_flat: jnp.ndarray,
    *,
    cache: LinearCache,
    params: LinearParams,
    state: _ImplicitState,
    data: _ImplicitPreconditionerData,
) -> jnp.ndarray:
    x = x_flat.reshape(state.shape)
    hermite = (
        _solve_hermite_lines_linked(
            x, cache=cache, params=params, state=state, data=data
        )
        if cache.use_twist_shift
        else _solve_hermite_lines_fft(
            x,
            kz=cache.kz,
            cache=cache,
            params=params,
            state=state,
            data=data,
        )
    )
    return hermite.reshape(state.size)


def _apply_hermite_line_coarse_preconditioner(
    x_flat: jnp.ndarray,
    *,
    cache: LinearCache,
    params: LinearParams,
    state: _ImplicitState,
    data: _ImplicitPreconditionerData,
) -> jnp.ndarray:
    x = x_flat.reshape(state.shape)
    x_line = _apply_hermite_line_preconditioner(
        x.reshape(state.size), cache=cache, params=params, state=state, data=data
    ).reshape(state.shape)
    x_coarse_in = _project_kx_coarse(x, cache)
    x_coarse_full = _apply_hermite_line_preconditioner(
        x_coarse_in.reshape(state.size),
        cache=cache,
        params=params,
        state=state,
        data=data,
    ).reshape(state.shape)
    x_line_coarse_full = _project_kx_coarse(x_line, cache)
    return (x_line + (x_coarse_full - x_line_coarse_full)).reshape(state.size)


def _build_implicit_preconditioner_callable(
    canonical: str,
    *,
    cache: LinearCache,
    params: LinearParams,
    state: _ImplicitState,
    data: _ImplicitPreconditionerData,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if canonical == "full":
        return lambda x_flat: _apply_factor_preconditioner(
            x_flat, state=state, factor=data.precond_full
        )
    if canonical == "damping":
        return lambda x_flat: _apply_factor_preconditioner(
            x_flat, state=state, factor=data.precond_damp
        )
    if canonical == "pas":
        return lambda x_flat: _apply_factor_preconditioner(
            x_flat, state=state, factor=data.precond_pas
        )
    if canonical == "pas_coarse":
        return lambda x_flat: _apply_pas_coarse_preconditioner(
            x_flat, cache=cache, state=state, data=data
        )
    if canonical == "hermite_line":
        return lambda x_flat: _apply_hermite_line_preconditioner(
            x_flat, cache=cache, params=params, state=state, data=data
        )
    if canonical == "hermite_line_coarse":
        return lambda x_flat: _apply_hermite_line_coarse_preconditioner(
            x_flat, cache=cache, params=params, state=state, data=data
        )
    if canonical == "identity":
        return lambda x_flat: x_flat
    raise ValueError(f"Unknown canonical implicit_preconditioner '{canonical}'")


def _build_shifted_hermite_preconditioner(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | TermConfig,
    sigma: jnp.ndarray,
    *,
    coarse: bool = False,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Approximate ``(A - sigma I)^-1`` with the FFT/Hermite line solver.

    The implicit preconditioner approximates ``(I - dt A)^-1``.  Setting
    ``dt = 1 / sigma`` and multiplying by ``-1 / sigma`` gives the corresponding
    inverse for ``A - sigma I`` while reusing exactly the same physics-aware
    streaming, drift, and damping factors.
    """

    sigma_value = jnp.asarray(sigma, dtype=G0.dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=G0.dtype)).dtype
    shift_floor = jnp.sqrt(jnp.finfo(real_dtype).eps)
    safe_sigma = jnp.where(
        jnp.abs(sigma_value) > shift_floor,
        sigma_value,
        jnp.asarray(1.0 + 0.0j, dtype=G0.dtype),
    )
    state = _prepare_implicit_state(G0, 1.0 / safe_sigma, terms)
    data = _build_implicit_preconditioner_data(cache, params, state)
    canonical = "hermite_line_coarse" if coarse else "hermite_line"
    line_inverse = _build_implicit_preconditioner_callable(
        canonical,
        cache=cache,
        params=params,
        state=state,
        data=data,
    )

    def apply_shifted_preconditioner(x_flat: jnp.ndarray) -> jnp.ndarray:
        result = (-1.0 / safe_sigma) * line_inverse(x_flat)
        return result.astype(x_flat.dtype)

    return apply_shifted_preconditioner


def _pack_field_state(fields: FieldState) -> jnp.ndarray:
    """Pack only active linear field arrays into one low-moment vector."""

    arrays = [fields.phi]
    if fields.apar is not None:
        arrays.append(fields.apar)
    if fields.bpar is not None:
        arrays.append(fields.bpar)
    return jnp.concatenate(tuple(jnp.ravel(array) for array in arrays))


def _unpack_field_state(vector: jnp.ndarray, template: FieldState) -> FieldState:
    """Inverse of :func:`_pack_field_state` with static template shapes."""

    phi_size = template.phi.size
    offset = phi_size
    phi = vector[:phi_size].reshape(template.phi.shape)
    if template.apar is None:
        apar = None
    else:
        apar_size = template.apar.size
        apar = vector[offset : offset + apar_size].reshape(template.apar.shape)
        offset += apar_size
    if template.bpar is None:
        bpar = None
    else:
        bpar = vector[offset:].reshape(template.bpar.shape)
    return FieldState(phi=phi, apar=apar, bpar=bpar)


def _build_field_corrected_shifted_preconditioner(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | TermConfig,
    sigma: jnp.ndarray,
    *,
    coarse: bool = False,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    r"""Add the exact low-moment field coupling to the Hermite-line inverse.

    For fixed geometry the field solve is a linear moment map ``F: G -> f``.
    Linear gyrokinetic forcing by those fields is another map ``R: f -> dG``,
    so the shifted operator separates as

    ``A - sigma I = (A_0 - sigma I) + R F``.

    ``F`` has at most three active fields per spatial spectral point,
    independent of velocity resolution. Sequentially probing ``R`` and
    applying ``F`` avoids materializing either the kinetic operator or its
    transpose. The Hermite-line inverse approximates ``A_0 - sigma I`` while
    a velocity-independent capacitance solve represents the field coupling.
    """

    term_cfg = (
        terms if isinstance(terms, TermConfig) else linear_terms_to_term_config(terms)
    )
    base_flat = _build_shifted_hermite_preconditioner(
        G0,
        cache,
        params,
        term_cfg,
        sigma,
        coarse=coarse,
    )
    zero_state = jnp.zeros_like(G0)
    field_template = compute_fields_cached(
        zero_state,
        cache,
        params,
        terms=term_cfg,
        use_custom_vjp=False,
    )
    zero_fields = _unpack_field_state(
        jnp.zeros_like(_pack_field_state(field_template)),
        field_template,
    )
    zero_rhs = assemble_rhs_cached_with_fields(
        zero_state,
        cache,
        params,
        zero_fields,
        terms=term_cfg,
    )

    def field_map(state: jnp.ndarray) -> jnp.ndarray:
        fields = compute_fields_cached(
            state,
            cache,
            params,
            terms=term_cfg,
            use_custom_vjp=False,
        )
        return _pack_field_state(fields).astype(G0.dtype)

    def field_response(field_vector: jnp.ndarray) -> jnp.ndarray:
        fields = _unpack_field_state(field_vector, field_template)
        response = assemble_rhs_cached_with_fields(
            zero_state,
            cache,
            params,
            fields,
            terms=term_cfg,
        )
        return (response - zero_rhs).astype(G0.dtype)

    field_size = _pack_field_state(field_template).size

    def base_state_preconditioner(state: jnp.ndarray) -> jnp.ndarray:
        return base_flat(jnp.ravel(state)).reshape(G0.shape)

    solved_columns = jnp.moveaxis(
        jax.lax.map(
            lambda index: base_state_preconditioner(
                field_response(jax.nn.one_hot(index, field_size, dtype=G0.dtype))
            ),
            jnp.arange(field_size),
        ),
        0,
        -1,
    )
    capacitance = jnp.eye(field_size, dtype=G0.dtype) + jnp.moveaxis(
        jax.lax.map(field_map, jnp.moveaxis(solved_columns, -1, 0)),
        0,
        -1,
    )
    capacitance_lu = lu_factor(capacitance)

    def corrected(state: jnp.ndarray) -> jnp.ndarray:
        base = base_state_preconditioner(state)
        weights = lu_solve(capacitance_lu, field_map(base))
        return base - jnp.tensordot(solved_columns, weights, axes=(-1, 0))

    def apply_corrected(vector: jnp.ndarray) -> jnp.ndarray:
        return jnp.ravel(corrected(vector.reshape(G0.shape))).astype(vector.dtype)

    return apply_corrected


def _select_implicit_preconditioner(
    *,
    cache: LinearCache,
    params: LinearParams,
    state: _ImplicitState,
    data: _ImplicitPreconditionerData,
    implicit_preconditioner: PreconditionerSpec,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    canonical = _canonical_implicit_preconditioner(implicit_preconditioner)
    if callable(canonical):
        return canonical
    return _build_implicit_preconditioner_callable(
        canonical,
        cache=cache,
        params=params,
        state=state,
        data=data,
    )


def _build_implicit_matvec(
    *,
    cache: LinearCache,
    params: LinearParams,
    state: _ImplicitState,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def matvec(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(state.shape)
        dG, _phi = linear_rhs_cached(
            x,
            cache,
            params,
            terms=state.terms,
            use_jit=False,
            use_custom_vjp=False,
            dt=state.dt_val,
        )
        return (x - state.dt_val * dG).reshape(state.size)

    return matvec


def _build_implicit_operator(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    terms: LinearTerms | None,
    implicit_preconditioner: PreconditionerSpec,
) -> tuple[
    jnp.ndarray,
    tuple[int, ...],
    int,
    jnp.ndarray,
    Callable[[jnp.ndarray], jnp.ndarray],
    Callable[[jnp.ndarray], jnp.ndarray],
    bool,
]:
    state = _prepare_implicit_state(G0, dt, terms)
    data = _build_implicit_preconditioner_data(cache, params, state)
    precond_op = _select_implicit_preconditioner(
        cache=cache,
        params=params,
        state=state,
        data=data,
        implicit_preconditioner=implicit_preconditioner,
    )
    matvec = _build_implicit_matvec(cache=cache, params=params, state=state)
    return (
        state.G,
        state.shape,
        state.size,
        state.dt_val,
        precond_op,
        matvec,
        state.squeeze_species,
    )


def _implicit_fixed_point_guess(
    G_in: jnp.ndarray,
    *,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    dt_val: jnp.ndarray,
    implicit_iters: int,
    implicit_relax: float,
) -> jnp.ndarray:
    """Build a bounded fixed-point warm start for the implicit GMRES solve."""

    def body(_i, g):
        dG, _phi = linear_rhs_cached(
            g,
            cache,
            params,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
            dt=dt_val,
        )
        g_next = G_in + dt_val * dG
        return (1.0 - implicit_relax) * g + implicit_relax * g_next

    return jax.lax.fori_loop(0, max(int(implicit_iters), 0), body, G_in)


def _gmres_iteration_budget(maxiter: int, restart: int) -> tuple[int, int]:
    """Map an iteration budget onto SOLVAX's static cycle size and cycle count.

    ``maxiter`` counts Arnoldi iterations, as in the shift-invert inner solve:
    the cycle is capped at ``maxiter`` and ``ceil(maxiter / restart)`` cycles run.
    """

    maxiter = max(int(maxiter), 1)
    restart = min(max(int(restart), 1), maxiter)
    return restart, math.ceil(maxiter / restart)


def _implicit_gmres_step(
    G_in: jnp.ndarray,
    stats: ImplicitSolveStats,
    *,
    shape: tuple[int, ...],
    size: int,
    **kwargs: Any,
) -> tuple[jnp.ndarray, ImplicitSolveStats]:
    """Advance one implicit step with a fixed-point warm start and GMRES."""

    solution = _implicit_gmres_solution(G_in, size=size, **kwargs)
    folded = _fold_implicit_solve_stats(stats, solution, G_in.reshape(size))
    return solution.x.reshape(shape), folded


def _implicit_gmres_solution(
    G_in: jnp.ndarray,
    *,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    dt_val: jnp.ndarray,
    size: int,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    precond_op: Callable[[jnp.ndarray], jnp.ndarray],
    implicit_tol: float,
    implicit_maxiter: int,
    implicit_iters: int,
    implicit_relax: float,
    implicit_restart: int,
) -> KrylovSolution:
    """Return SOLVAX's flat solution with its true residual and converged flag.

    The scan routes keep only ``x``: their ``(G, phi_t)`` result has no solver
    status channel, and a host callback inside the traced scan is not added.
    """

    G_guess = _implicit_fixed_point_guess(
        G_in,
        cache=cache,
        params=params,
        terms=terms,
        dt_val=dt_val,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
    )
    restart, max_restarts = _gmres_iteration_budget(implicit_maxiter, implicit_restart)
    return gmres(
        matvec,
        G_in.reshape(size),
        x0=G_guess.reshape(size),
        precond=precond_op,
        restart=restart,
        rtol=implicit_tol,
        atol=0.0,
        max_restarts=max_restarts,
    )


def _implicit_phi_diagnostic(
    G: jnp.ndarray,
    *,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    dt_val: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the linear field diagnostic after an implicit step."""

    _dG, phi = linear_rhs_cached(
        G,
        cache,
        params,
        terms=terms,
        use_jit=False,
        use_custom_vjp=False,
        dt=dt_val,
    )
    return phi


def _validate_implicit_sample_policy(*, steps: int, sample_stride: int) -> None:
    """Validate saved-sample cadence before building JAX scan closures."""

    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")


ImplicitSolveStepFn = Callable[
    [jnp.ndarray, ImplicitSolveStats], tuple[jnp.ndarray, ImplicitSolveStats]
]


def _build_implicit_solve_step(
    *,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    dt_val: jnp.ndarray,
    size: int,
    shape: tuple[int, ...],
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    precond_op: Callable[[jnp.ndarray], jnp.ndarray],
    options: _ImplicitSolveOptions,
) -> ImplicitSolveStepFn:
    """Return the per-step GMRES solve closure used by scan paths.

    The closure maps ``(state, stats)`` to ``(next state, stats)``: every solve
    folds its SOLVAX status into the carried :class:`ImplicitSolveStats`.
    """

    def solve_step(
        G_in: jnp.ndarray, stats: ImplicitSolveStats
    ) -> tuple[jnp.ndarray, ImplicitSolveStats]:
        return _implicit_gmres_step(
            G_in,
            stats,
            cache=cache,
            params=params,
            terms=terms,
            dt_val=dt_val,
            size=size,
            shape=shape,
            matvec=matvec,
            precond_op=precond_op,
            implicit_tol=options.tol,
            implicit_maxiter=options.maxiter,
            implicit_iters=options.iters,
            implicit_relax=options.relax,
            implicit_restart=options.restart,
        )

    return solve_step


def _scan_implicit_outputs(
    G: jnp.ndarray,
    *,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    dt_val: jnp.ndarray,
    solve_step: ImplicitSolveStepFn,
    steps: int,
    sample_stride: int,
    checkpoint: bool,
) -> tuple[tuple[jnp.ndarray, ImplicitSolveStats], jnp.ndarray]:
    """Integrate implicit steps, carrying solve status, and save field diagnostics."""

    carry0 = (G, _empty_implicit_solve_stats(G.dtype))

    def step(carry, _):
        G_new, stats = solve_step(*carry)
        phi_new = _implicit_phi_diagnostic(
            G_new,
            cache=cache,
            params=params,
            terms=terms,
            dt_val=dt_val,
        )
        return (G_new, stats), phi_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    if sample_stride <= 1:
        return jax.lax.scan(step_fn, carry0, None, length=steps)

    def sample_step(carry, _):
        def inner_step(_i, inner_carry):
            return solve_step(*inner_carry)

        G_out_local, stats = jax.lax.fori_loop(0, sample_stride, inner_step, carry)
        phi_out = _implicit_phi_diagnostic(
            G_out_local,
            cache=cache,
            params=params,
            terms=terms,
            dt_val=dt_val,
        )
        return (G_out_local, stats), phi_out

    return jax.lax.scan(sample_step, carry0, None, length=steps // sample_stride)


def _integrate_linear_implicit_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    terms: LinearTerms | None = None,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: PreconditionerSpec = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    return_solve_stats: bool = False,
) -> tuple[Any, ...]:
    """Implicit linear integrator using GMRES with a diagonal preconditioner.

    Returns ``(G_out, phi_t)``, or ``(G_out, phi_t, stats)`` with
    ``return_solve_stats=True``. The traced scan cannot raise on an unconverged
    solve, so ``stats`` (:class:`ImplicitSolveStats`) is the convergence channel;
    the runtime checks it with :func:`require_converged_implicit_solves`.
    """
    terms = LinearTerms() if terms is None else terms
    _validate_implicit_sample_policy(steps=steps, sample_stride=sample_stride)

    G, shape, size, dt_val, precond_op, matvec, squeeze_species = (
        _build_implicit_operator(G0, cache, params, dt, terms, implicit_preconditioner)
    )
    solve_step = _build_implicit_solve_step(
        cache=cache,
        params=params,
        terms=terms,
        dt_val=dt_val,
        size=size,
        shape=shape,
        matvec=matvec,
        precond_op=precond_op,
        options=_ImplicitSolveOptions(
            tol=implicit_tol,
            maxiter=implicit_maxiter,
            iters=implicit_iters,
            relax=implicit_relax,
            restart=implicit_restart,
        ),
    )
    (G_out, solve_stats), phi_t = _scan_implicit_outputs(
        G,
        cache=cache,
        params=params,
        terms=terms,
        dt_val=dt_val,
        solve_step=solve_step,
        steps=steps,
        sample_stride=sample_stride,
        checkpoint=checkpoint,
    )

    G_out = G_out[0] if squeeze_species else G_out
    if return_solve_stats:
        return G_out, phi_t, solve_stats
    return G_out, phi_t
