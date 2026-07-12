"""Implicit linear solve policies for cache-backed gyrokinetic operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres
from solvax import tridiagonal_solve

from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.cache_arrays import (
    collision_damping,
    hypercollision_damping,
)
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    PreconditionerSpec,
    _as_species_array,
    _resolve_implicit_preconditioner,
    _x64_enabled,
)
from spectraxgk.operators.linear.rhs import linear_rhs_cached

__all__ = ["_build_implicit_operator", "_integrate_linear_implicit_cached"]


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


@dataclass(frozen=True)
class _ImplicitSolveOptions:
    tol: float
    maxiter: int
    iters: int
    relax: float
    restart: int
    solve_method: str


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
        {"hermite-line-coarse", "hermite_line_coarse", "hermite_coarse", "streaming-line-coarse"}
    ),
    "identity": frozenset({"identity", "none", "off"}),
}


def _prepare_implicit_state(
    G0: jnp.ndarray,
    dt: float,
    terms: LinearTerms | None,
) -> _ImplicitState:
    terms = LinearTerms() if terms is None else terms
    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)

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
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    damping = (
        collision_damping(cache, params, real_dtype, squeeze_species=False) + hyper_damp
    ).astype(real_dtype)

    ell = cache.l.astype(real_dtype)
    m = cache.m.astype(real_dtype)
    cv_d = cache.cv_d.astype(real_dtype)
    gb_d = cache.gb_d.astype(real_dtype)
    bgrad = cache.bgrad.astype(real_dtype)
    w_mirror = jnp.asarray(state.terms.mirror, dtype=real_dtype)
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
    bgrad = bgrad[None, None, None, None, None, :]
    mirror_diag = vth_b * (2.0 * ell + 1.0) * (2.0 * m + 1.0)
    mirror_weight = 0.2
    diag = diag - w_mirror * mirror_weight * bgrad * mirror_diag

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
    )


def _scatter_unique_spectral_modes(
    target: jnp.ndarray,
    idx_flat: jnp.ndarray,
    updates: jnp.ndarray,
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


def _solve_tridiagonal_last_axis(
    lower: jnp.ndarray,
    diagonal: jnp.ndarray,
    upper: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """Solve independent tridiagonal systems stored on the last axis.

    SPECTRAX-GK stores the Hermite line last, while SOLVAX uses a leading
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
    """Invert ``I - dt L_stream`` approximately via FFT(z) + tridiagonal(m)."""

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
    d = jnp.ones_like(du)
    batch_shape = x_hat_mlast.shape
    dl = jnp.broadcast_to(dl, batch_shape)
    d = jnp.broadcast_to(d, batch_shape)
    du = jnp.broadcast_to(du, batch_shape)
    y_hat_mlast = _solve_tridiagonal_last_axis(dl, d, du, x_hat_mlast)
    y_hat = jnp.moveaxis(y_hat_mlast, -1, 2)
    return jnp.fft.ifft(y_hat, axis=-1)


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
    x_flat = x.reshape(*lead_shape, Ny * Nx, Nz)
    y_flat = jnp.zeros_like(x_flat)

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
        d = jnp.ones_like(du)
        batch_shape = x_hat_mlast.shape
        dl = jnp.broadcast_to(dl, batch_shape)
        d = jnp.broadcast_to(d, batch_shape)
        du = jnp.broadcast_to(du, batch_shape)
        y_hat_mlast = _solve_tridiagonal_last_axis(dl, d, du, x_hat_mlast)
        y_hat = jnp.moveaxis(y_hat_mlast, -1, 2)
        y_link = jnp.fft.ifft(y_hat, axis=-1)
        y_link = y_link.reshape(*lead_shape, nChains * nLinks, Nz)
        y_flat = _scatter_unique_spectral_modes(y_flat, idx_flat, y_link)

    return y_flat.reshape(*lead_shape, Ny, Nx, Nz)


def _project_kx_coarse(x: jnp.ndarray, cache: LinearCache) -> jnp.ndarray:
    """Project/prolong a coarse kx correction without breaking linked chains."""

    if not cache.use_twist_shift or not cache.linked_indices:
        x_mean = jnp.mean(x, axis=4, keepdims=True)
        return jnp.broadcast_to(x_mean, x.shape)

    Ny = x.shape[-3]
    Nx = x.shape[-2]
    Nz = x.shape[-1]
    lead_shape = x.shape[:-3]
    x_flat = x.reshape(*lead_shape, Ny * Nx, Nz)
    y_flat = jnp.zeros_like(x_flat)

    for idx_map in cache.linked_indices:
        nChains, nLinks = idx_map.shape
        idx_flat = idx_map.reshape(-1)
        x_link = jnp.take(x_flat, idx_flat, axis=-2)
        x_link = x_link.reshape(*lead_shape, nChains, nLinks, Nz)
        x_mean = jnp.mean(x_link, axis=-2, keepdims=True)
        x_mean = jnp.broadcast_to(x_mean, x_link.shape)
        x_updates = x_mean.reshape(*lead_shape, nChains * nLinks, Nz)
        y_flat = _scatter_unique_spectral_modes(y_flat, idx_flat, x_updates)

    return y_flat.reshape(*lead_shape, Ny, Nx, Nz)


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
    x = x_flat.reshape(state.shape) * data.precond_full
    x = (
        _solve_hermite_lines_linked(x, cache=cache, params=params, state=state, data=data)
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
    return x.reshape(state.size)


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


def _implicit_gmres_step(
    G_in: jnp.ndarray,
    *,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    dt_val: jnp.ndarray,
    size: int,
    shape: tuple[int, ...],
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    precond_op: Callable[[jnp.ndarray], jnp.ndarray],
    implicit_tol: float,
    implicit_maxiter: int,
    implicit_iters: int,
    implicit_relax: float,
    implicit_restart: int,
    implicit_solve_method: str,
) -> jnp.ndarray:
    """Advance one implicit step with a fixed-point warm start and GMRES."""

    G_guess = _implicit_fixed_point_guess(
        G_in,
        cache=cache,
        params=params,
        terms=terms,
        dt_val=dt_val,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
    )
    sol, _info = gmres(
        matvec,
        G_in.reshape(size),
        x0=G_guess.reshape(size),
        tol=implicit_tol,
        maxiter=implicit_maxiter,
        restart=implicit_restart,
        M=precond_op,
        solve_method=implicit_solve_method,
    )
    return sol.reshape(shape)


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
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return the per-step GMRES solve closure used by scan paths."""

    def solve_step(G_in: jnp.ndarray) -> jnp.ndarray:
        return _implicit_gmres_step(
            G_in,
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
            implicit_solve_method=options.solve_method,
        )

    return solve_step


def _scan_implicit_outputs(
    G: jnp.ndarray,
    *,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    dt_val: jnp.ndarray,
    solve_step: Callable[[jnp.ndarray], jnp.ndarray],
    steps: int,
    sample_stride: int,
    checkpoint: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate implicit steps and collect saved field diagnostics."""

    def step(G_in, _):
        G_new = solve_step(G_in)
        phi_new = _implicit_phi_diagnostic(
            G_new,
            cache=cache,
            params=params,
            terms=terms,
            dt_val=dt_val,
        )
        return G_new, phi_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    if sample_stride <= 1:
        return jax.lax.scan(step_fn, G, None, length=steps)

    def sample_step(G_in, _):
        def inner_step(_i, g):
            return solve_step(g)

        G_out_local = jax.lax.fori_loop(0, sample_stride, inner_step, G_in)
        phi_out = _implicit_phi_diagnostic(
            G_out_local,
            cache=cache,
            params=params,
            terms=terms,
            dt_val=dt_val,
        )
        return G_out_local, phi_out

    return jax.lax.scan(sample_step, G, None, length=steps // sample_stride)


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
    implicit_solve_method: str = "batched",
    implicit_preconditioner: PreconditionerSpec = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Implicit linear integrator using GMRES with a diagonal preconditioner."""
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
            solve_method=implicit_solve_method,
        ),
    )
    G_out, phi_t = _scan_implicit_outputs(
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
    return G_out, phi_t
