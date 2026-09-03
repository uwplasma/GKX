"""Diagnostic sampling integration for linear fixed-step solves."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from gkx.core_grid import SpectralGrid, _gyrokinetic_moment_shape
from gkx.geometry import FluxTubeGeometryLike
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.cache_arrays import (
    collision_damping,
    hypercollision_damping,
)
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    PreconditionerSpec,
    _x64_enabled,
)
from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.solvers_linear_implicit import (
    _ImplicitSolveOptions,
    _build_implicit_operator,
    _build_implicit_solve_step,
)
from gkx.solvers_time_explicit_steps import _linear_native_step


def _validate_linear_sampling(*, steps: int, sample_stride: int) -> None:
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")


def _linear_cache_or_build(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    cache: LinearCache | None,
    *,
    cache_builder: Callable[..., LinearCache],
) -> LinearCache:
    if cache is not None:
        return cache
    Nl, Nm = _gyrokinetic_moment_shape(G0)
    return cache_builder(grid, geom, params, Nl, Nm)


def _initial_state(G0: jnp.ndarray) -> tuple[jnp.ndarray, Any]:
    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    return G, real_dtype


def _linear_damping(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    real_dtype: Any,
    *,
    include_collisions: bool = True,
) -> jnp.ndarray:
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    if G.ndim == 5 and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]
    damping = hyper_damp
    if include_collisions:
        # A moment collision operator replaces the built-in diagonal term;
        # keeping both would apply collisions twice.
        damping = damping + collision_damping(
            cache, params, real_dtype, squeeze_species=(G.ndim == 5)
        )
    return damping.astype(real_dtype)


def _rhs(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    dt_val: jnp.ndarray,
    collision_operator: Any | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return linear_rhs_cached(
        G,
        cache,
        params,
        terms=terms,
        use_jit=False,
        dt=dt_val,
        collision_operator=collision_operator,
    )


def _density_from_state(
    G: jnp.ndarray,
    cache: LinearCache,
    species_index: int | None,
) -> jnp.ndarray:
    Jl = cache.Jl
    if G.ndim == 5:
        Jl_s = Jl[0] if Jl.ndim == 5 else Jl
        return jnp.sum(Jl_s * G[:, 0, ...], axis=0)
    if Jl.ndim == 5:
        if species_index is None:
            return jnp.sum(jnp.sum(Jl * G[:, :, 0, ...], axis=1), axis=0)
        Jl_s = Jl[int(species_index)]
        return jnp.sum(Jl_s * G[int(species_index), :, 0, ...], axis=0)
    if species_index is None:
        return jnp.sum(jnp.sum(Jl[None, ...] * G[:, :, 0, ...], axis=1), axis=0)
    return jnp.sum(Jl * G[int(species_index), :, 0, ...], axis=0)


def _hl_energy_from_state(G: jnp.ndarray) -> jnp.ndarray:
    if G.ndim == 5:
        return jnp.sum(jnp.abs(G) ** 2, axis=(2, 3, 4))
    return jnp.sum(jnp.abs(G) ** 2, axis=(0, 3, 4, 5))


def _maybe_emit_progress(
    G: jnp.ndarray,
    idx: jnp.ndarray,
    steps: int,
    dt_val: jnp.ndarray,
    phi: jnp.ndarray,
    density: jnp.ndarray,
    *,
    show_progress: bool,
    step_multiplier: int = 1,
) -> jnp.ndarray:
    if not show_progress:
        return G
    from gkx.callbacks import print_callback, should_emit_progress

    completed_step = jnp.minimum((idx + 1) * step_multiplier, steps) - 1
    sim_time = jnp.minimum((idx + 1) * step_multiplier, steps) * dt_val
    sim_total = jnp.asarray(steps, dtype=dt_val.dtype) * dt_val
    phi_max = jnp.max(jnp.abs(phi))
    density_max = jnp.max(jnp.abs(density))
    return jax.lax.cond(
        should_emit_progress(completed_step, steps),
        lambda state: print_callback(
            state,
            completed_step,
            steps,
            0.0,
            0.0,
            phi_max,
            density_max,
            sim_time,
            sim_total,
            metric_labels=("|phi|_max", "|n|_max"),
        ),
        lambda state: state,
        G,
    )


def _diagnostic_sample(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    dt_val: jnp.ndarray,
    species_index: int | None,
    *,
    record_hl_energy: bool,
    collision_operator: Any | None = None,
) -> tuple[jnp.ndarray, ...]:
    _dG, phi = _rhs(G, cache, params, terms, dt_val, collision_operator)
    density = _density_from_state(G, cache, species_index)
    if record_hl_energy:
        return phi, density, _hl_energy_from_state(G)
    return phi, density


def _every_step_scan(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    *,
    dt_val: jnp.ndarray,
    steps: int,
    advance: Callable[[jnp.ndarray], jnp.ndarray],
    species_index: int | None,
    record_hl_energy: bool,
    show_progress: bool,
    collision_operator: Any | None = None,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    def step(G_in: jnp.ndarray, idx: jnp.ndarray):
        G_out = advance(G_in)
        outputs = _diagnostic_sample(
            G_out,
            cache,
            params,
            terms,
            dt_val,
            species_index,
            record_hl_energy=record_hl_energy,
            collision_operator=collision_operator,
        )
        G_out = _maybe_emit_progress(
            G_out,
            idx,
            steps,
            dt_val,
            outputs[0],
            outputs[1],
            show_progress=show_progress,
        )
        return G_out, outputs

    return jax.lax.scan(step, G0, jnp.arange(steps))


def _strided_sample_scan(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    *,
    dt_val: jnp.ndarray,
    steps: int,
    sample_stride: int,
    advance: Callable[[jnp.ndarray], jnp.ndarray],
    species_index: int | None,
    record_hl_energy: bool,
    show_progress: bool,
    collision_operator: Any | None = None,
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, ...]]:
    def sample_step(G_in: jnp.ndarray, idx: jnp.ndarray):
        def inner_step(_i: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
            return advance(g)

        G_out = jax.lax.fori_loop(0, sample_stride, inner_step, G_in)
        outputs = _diagnostic_sample(
            G_out,
            cache,
            params,
            terms,
            dt_val,
            species_index,
            record_hl_energy=record_hl_energy,
            collision_operator=collision_operator,
        )
        G_out = _maybe_emit_progress(
            G_out,
            idx,
            steps,
            dt_val,
            outputs[0],
            outputs[1],
            show_progress=show_progress,
            step_multiplier=sample_stride,
        )
        return G_out, outputs

    num_samples = steps // sample_stride
    return jax.lax.scan(sample_step, G0, jnp.arange(num_samples))


def integrate_linear_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "rk4",
    cache: LinearCache | None = None,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
    species_index: int | None = 0,
    record_hl_energy: bool = False,
    show_progress: bool = False,
    collision_operator: Any | None = None,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: PreconditionerSpec = None,
) -> (
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
):
    """Integrate and return (G_out, phi_t, density_t) for diagnostics."""

    terms_use = terms or LinearTerms()
    _validate_linear_sampling(steps=steps, sample_stride=sample_stride)
    cache_use = _linear_cache_or_build(
        G0, grid, geom, params, cache, cache_builder=build_linear_cache
    )
    G, real_dtype = _initial_state(G0)
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    squeeze_species = False
    if method == "implicit":
        if collision_operator is not None:
            raise NotImplementedError(
                "implicit integration does not support custom collision operators"
            )
        G, shape, size, dt_val, precond, matvec, squeeze_species = (
            _build_implicit_operator(
                G,
                cache_use,
                params,
                dt,
                terms_use,
                implicit_preconditioner,
            )
        )
        advance = _build_implicit_solve_step(
            cache=cache_use,
            params=params,
            terms=terms_use,
            dt_val=dt_val,
            size=size,
            shape=shape,
            matvec=matvec,
            precond_op=precond,
            options=_ImplicitSolveOptions(
                tol=implicit_tol,
                maxiter=implicit_maxiter,
                iters=implicit_iters,
                relax=implicit_relax,
                restart=implicit_restart,
            ),
        )
    else:
        damping = _linear_damping(
            G,
            cache_use,
            params,
            real_dtype,
            include_collisions=collision_operator is None,
        )

        def advance(state: jnp.ndarray) -> jnp.ndarray:
            return _linear_native_step(
                state,
                damping,
                dt_val,
                method_key=method,
                rhs=lambda value: _rhs(
                    value, cache_use, params, terms_use, dt_val, collision_operator
                )[0],
            )

    if sample_stride <= 1:
        G_out, outputs = _every_step_scan(
            G,
            cache_use,
            params,
            terms_use,
            dt_val=dt_val,
            steps=steps,
            advance=advance,
            species_index=species_index,
            record_hl_energy=record_hl_energy,
            show_progress=show_progress,
            collision_operator=collision_operator,
        )
    else:
        G_out, outputs = _strided_sample_scan(
            G,
            cache_use,
            params,
            terms_use,
            dt_val=dt_val,
            steps=steps,
            sample_stride=sample_stride,
            advance=advance,
            species_index=species_index,
            record_hl_energy=record_hl_energy,
            show_progress=show_progress,
            collision_operator=collision_operator,
        )
    if squeeze_species:
        G_out = G_out[0]
    if record_hl_energy:
        phi_t, density_t, hl_t = outputs
        return G_out, phi_t, density_t, hl_t
    phi_t, density_t = outputs
    return G_out, phi_t, density_t


__all__ = ["integrate_linear_diagnostics"]
