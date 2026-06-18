"""Linear time-integration and diagnostic sampling policies."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.operators.linear.cache import (
    LinearCache,
    build_linear_cache,
    collision_damping,
    hypercollision_damping,
)
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    PreconditionerSpec,
    _x64_enabled,
)
from spectraxgk.operators.linear.rhs import linear_rhs_cached
from spectraxgk.solvers.linear.implicit import _integrate_linear_implicit_cached
from spectraxgk.solvers.linear import integrator_diagnostics as _linear_diagnostics
from spectraxgk.solvers.linear.parallel import (
    _is_electrostatic_field_terms,
    linear_rhs_parallel_cached,
)

__all__ = [
    "_integrate_linear_cached",
    "_integrate_linear_cached_donate",
    "_integrate_linear_cached_impl",
    "integrate_linear",
    "integrate_linear_diagnostics",
]


_SSPX3_ADT = float((1.0 / 6.0) ** (1.0 / 3.0))
_SSPX3_WGTFAC = float((9.0 - 2.0 * (6.0 ** (2.0 / 3.0))) ** 0.5)
_SSPX3_W1 = 0.5 * (_SSPX3_WGTFAC - 1.0)
_SSPX3_W2 = 0.5 * ((6.0 ** (2.0 / 3.0)) - 1.0 - _SSPX3_WGTFAC)
_SSPX3_W3 = (1.0 / _SSPX3_ADT) - 1.0 - _SSPX3_W2 * (_SSPX3_W1 + 1.0)


def _integrate_linear_cached_impl(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    checkpoint: bool = False,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
    show_progress: bool = False,
    parallel: Any | None = None,
    force_electrostatic_fields: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using cached geometry arrays."""
    if method not in {"euler", "rk2", "rk4", "imex", "imex2", "sspx3"}:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk4', 'imex', 'imex2', 'sspx3'}"
        )
    if terms is None:
        terms = LinearTerms()

    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    if G0.ndim == 5 and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]
    damping = (
        collision_damping(cache, params, real_dtype, squeeze_species=(G0.ndim == 5))
        + hyper_damp
    )
    damping = damping.astype(real_dtype)

    parallel_strategy = (
        "serial"
        if parallel is None
        else str(getattr(parallel, "strategy", "serial")).lower().replace("-", "_")
    )

    def rhs(G_in: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if parallel_strategy == "serial":
            return linear_rhs_cached(
                G_in,
                cache,
                params,
                terms=terms,
                dt=dt_val,
                force_electrostatic_fields=force_electrostatic_fields,
            )
        return linear_rhs_parallel_cached(
            G_in, cache, params, terms=terms, parallel=parallel, dt=dt_val
        )

    def advance(G):
        dG, _phi = rhs(G)
        if method == "imex":
            dG_explicit = dG + damping * G
            return (G + dt_val * dG_explicit) / (1.0 + dt_val * damping)
        if method == "imex2":
            dG_explicit = dG + damping * G
            G_half = (G + 0.5 * dt_val * dG_explicit) / (1.0 + 0.5 * dt_val * damping)
            dG_half, _phi = rhs(G_half)
            dG_half_exp = dG_half + damping * G_half
            return (G + dt_val * dG_half_exp) / (1.0 + dt_val * damping)
        if method == "euler":
            return G + dt_val * dG
        if method == "rk2":
            k1 = dG
            k2, _ = rhs(G + 0.5 * dt_val * k1)
            return G + dt_val * k2
        if method == "sspx3":

            def _euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
                dG_state, _ = rhs(G_state)
                return G_state + (_SSPX3_ADT * dt_val) * dG_state

            G1 = _euler_step(G)
            G2_euler = _euler_step(G1)
            G2 = (1.0 - _SSPX3_W1) * G + (_SSPX3_W1 - 1.0) * G1 + G2_euler
            G3 = _euler_step(G2)
            return (
                (1.0 - _SSPX3_W2 - _SSPX3_W3) * G
                + _SSPX3_W3 * G1
                + (_SSPX3_W2 - 1.0) * G2
                + G3
            )
        k1 = dG
        k2, _ = rhs(G + 0.5 * dt_val * k1)
        k3, _ = rhs(G + 0.5 * dt_val * k2)
        k4, _ = rhs(G + dt_val * k3)
        return G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(G, idx):
        G_new = advance(G)
        _dG_new, phi_new = rhs(G_new)
        if show_progress:
            from spectraxgk.utils.callbacks import print_callback, should_emit_progress

            sim_time = (idx + 1) * dt_val
            sim_total = jnp.asarray(steps, dtype=dt_val.dtype) * dt_val
            phi_max = jnp.max(jnp.abs(phi_new))
            G_new = jax.lax.cond(
                should_emit_progress(idx, steps),
                lambda state: print_callback(
                    state,
                    idx,
                    steps,
                    0.0,
                    0.0,
                    phi_max,
                    0.0,
                    sim_time,
                    sim_total,
                    metric_labels=("|phi|_max", "|n|_max"),
                ),
                lambda state: state,
                G_new,
            )
        return G_new, phi_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    indices = jnp.arange(steps)
    if sample_stride <= 1:
        return jax.lax.scan(step_fn, G0, indices)

    def sample_step(G, idx):
        def inner_step(i, state):
            return advance(state)

        G_out = jax.lax.fori_loop(0, sample_stride, inner_step, G)
        _dG_out, phi_out = rhs(G_out)
        if show_progress:
            from spectraxgk.utils.callbacks import print_callback, should_emit_progress

            completed_idx = jnp.minimum((idx + 1) * sample_stride, steps) - 1
            sim_time = jnp.minimum((idx + 1) * sample_stride, steps) * dt_val
            sim_total = jnp.asarray(steps, dtype=dt_val.dtype) * dt_val
            phi_max = jnp.max(jnp.abs(phi_out))
            G_out = jax.lax.cond(
                should_emit_progress(completed_idx, steps),
                lambda state: print_callback(
                    state,
                    completed_idx,
                    steps,
                    0.0,
                    0.0,
                    phi_max,
                    0.0,
                    sim_time,
                    sim_total,
                    metric_labels=("|phi|_max", "|n|_max"),
                ),
                lambda state: state,
                G_out,
            )
        return G_out, phi_out

    num_samples = steps // sample_stride
    sample_indices = jnp.arange(num_samples)
    return jax.lax.scan(sample_step, G0, sample_indices)


@partial(
    jax.jit,
    static_argnames=(
        "steps",
        "method",
        "checkpoint",
        "sample_stride",
        "show_progress",
        "force_electrostatic_fields",
    ),
)
def _integrate_linear_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    checkpoint: bool = False,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
    show_progress: bool = False,
    force_electrostatic_fields: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _integrate_linear_cached_impl(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
        terms=terms,
        sample_stride=sample_stride,
        show_progress=show_progress,
        force_electrostatic_fields=force_electrostatic_fields,
    )


@partial(
    jax.jit,
    static_argnames=(
        "steps",
        "method",
        "checkpoint",
        "sample_stride",
        "show_progress",
        "force_electrostatic_fields",
    ),
    donate_argnums=(0,),
)
def _integrate_linear_cached_donate(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    checkpoint: bool = False,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
    show_progress: bool = False,
    force_electrostatic_fields: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _integrate_linear_cached_impl(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
        terms=terms,
        sample_stride=sample_stride,
        show_progress=show_progress,
        force_electrostatic_fields=force_electrostatic_fields,
    )


def integrate_linear(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    cache: LinearCache | None = None,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: PreconditionerSpec = None,
    terms: LinearTerms | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    donate: bool = False,
    show_progress: bool = False,
    parallel: Any | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using a fixed-step scheme."""
    if terms is None:
        terms = LinearTerms()
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError(
                "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
            )
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
    if method == "semi-implicit":
        method = "imex"
    parallel_strategy = (
        "serial"
        if parallel is None
        else str(getattr(parallel, "strategy", "serial")).lower().replace("-", "_")
    )
    force_electrostatic_fields = _is_electrostatic_field_terms(terms)
    if method == "implicit":
        if parallel_strategy != "serial":
            raise NotImplementedError(
                "parallel linear integration currently supports only explicit fixed-step methods"
            )
        return _integrate_linear_implicit_cached(
            G0,
            cache,
            params,
            dt=dt,
            steps=steps,
            terms=terms,
            implicit_tol=implicit_tol,
            implicit_maxiter=implicit_maxiter,
            implicit_iters=implicit_iters,
            implicit_relax=implicit_relax,
            implicit_restart=implicit_restart,
            implicit_solve_method=implicit_solve_method,
            implicit_preconditioner=implicit_preconditioner,
            checkpoint=checkpoint,
            sample_stride=sample_stride,
        )
    if parallel_strategy != "serial":
        if donate:
            raise NotImplementedError(
                "parallel linear integration does not currently support donated input buffers"
            )
        return _integrate_linear_cached_impl(
            G0,
            cache,
            params,
            dt,
            steps,
            method=method,
            checkpoint=checkpoint,
            terms=terms,
            sample_stride=sample_stride,
            show_progress=show_progress,
            parallel=parallel,
            force_electrostatic_fields=force_electrostatic_fields,
        )
    integrator = _integrate_linear_cached_donate if donate else _integrate_linear_cached
    return integrator(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
        terms=terms,
        sample_stride=sample_stride,
        show_progress=show_progress,
        force_electrostatic_fields=force_electrostatic_fields,
    )


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
) -> (
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
):
    """Integrate and return (G_out, phi_t, density_t) for diagnostics."""

    _linear_diagnostics.build_linear_cache = build_linear_cache
    _linear_diagnostics.collision_damping = collision_damping
    _linear_diagnostics.hypercollision_damping = hypercollision_damping
    _linear_diagnostics.linear_rhs_cached = linear_rhs_cached
    return _linear_diagnostics.integrate_linear_diagnostics(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        method=method,
        cache=cache,
        terms=terms,
        sample_stride=sample_stride,
        species_index=species_index,
        record_hl_energy=record_hl_energy,
        show_progress=show_progress,
    )
