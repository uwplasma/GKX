"""Linear time-integration and diagnostic sampling policies."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp

from spectraxgk.core.extension_points import CollisionOperator
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.cache_builder import build_linear_cache
from spectraxgk.operators.linear.cache_arrays import (
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
_LINEAR_METHODS = {"euler", "rk2", "rk4", "imex", "imex2", "sspx3"}


def _validate_linear_method(method: str) -> None:
    if method not in _LINEAR_METHODS:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk4', 'imex', 'imex2', 'sspx3'}"
        )


def _prepared_linear_state_and_damping(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    include_collisions: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    if G0.ndim == 5 and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]
    damping = hyper_damp
    if include_collisions:
        damping = damping + collision_damping(
            cache, params, real_dtype, squeeze_species=(G0.ndim == 5)
        )
    return G0, damping.astype(real_dtype)


def _linear_parallel_strategy(parallel: Any | None) -> str:
    if parallel is None:
        return "serial"
    return str(getattr(parallel, "strategy", "serial")).lower().replace("-", "_")


def _linear_rhs_callable(
    *,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    dt_val: jnp.ndarray,
    parallel: Any | None,
    parallel_strategy: str,
    force_electrostatic_fields: bool,
    collision_operator: CollisionOperator | None = None,
) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    def rhs(G_in: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if parallel_strategy == "serial":
            return linear_rhs_cached(
                G_in,
                cache,
                params,
                terms=terms,
                dt=dt_val,
                force_electrostatic_fields=force_electrostatic_fields,
                collision_operator=collision_operator,
            )
        return linear_rhs_parallel_cached(
            G_in, cache, params, terms=terms, parallel=parallel, dt=dt_val
        )

    return rhs


def _linear_phi_callable(
    *,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    rhs: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    parallel_strategy: str,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return the cheapest field-only diagnostic path for an updated state."""

    if parallel_strategy != "serial" or not isinstance(cache, LinearCache):
        return lambda value: rhs(value)[1]

    from spectraxgk.operators.linear.params import linear_terms_to_term_config
    from spectraxgk.terms.assembly import compute_fields_cached

    term_config = linear_terms_to_term_config(terms)

    def solve_phi(value: jnp.ndarray) -> jnp.ndarray:
        return compute_fields_cached(
            value, cache, params, terms=term_config, use_custom_vjp=True
        ).phi

    return solve_phi


def _sspx3_step(
    G: jnp.ndarray,
    *,
    rhs: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    dt_val: jnp.ndarray,
) -> jnp.ndarray:
    def euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
        dG_state, _ = rhs(G_state)
        return G_state + (_SSPX3_ADT * dt_val) * dG_state

    G1 = euler_step(G)
    G2_euler = euler_step(G1)
    G2 = (1.0 - _SSPX3_W1) * G + (_SSPX3_W1 - 1.0) * G1 + G2_euler
    G3 = euler_step(G2)
    return (
        (1.0 - _SSPX3_W2 - _SSPX3_W3) * G + _SSPX3_W3 * G1 + (_SSPX3_W2 - 1.0) * G2 + G3
    )


def _advance_linear_state(
    G: jnp.ndarray,
    *,
    rhs: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]],
    damping: jnp.ndarray,
    dt_val: jnp.ndarray,
    method: str,
) -> jnp.ndarray:
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
        k2, _ = rhs(G + 0.5 * dt_val * dG)
        return G + dt_val * k2
    if method == "sspx3":
        return _sspx3_step(G, rhs=rhs, dt_val=dt_val)
    k1 = dG
    k2, _ = rhs(G + 0.5 * dt_val * k1)
    k3, _ = rhs(G + 0.5 * dt_val * k2)
    k4, _ = rhs(G + dt_val * k3)
    return G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _maybe_emit_linear_progress(
    G: jnp.ndarray,
    *,
    idx: jnp.ndarray,
    steps: int,
    dt_val: jnp.ndarray,
    phi: jnp.ndarray,
    show_progress: bool,
) -> jnp.ndarray:
    if not show_progress:
        return G
    from spectraxgk.utils.callbacks import print_callback, should_emit_progress

    sim_time = (idx + 1) * dt_val
    sim_total = jnp.asarray(steps, dtype=dt_val.dtype) * dt_val
    phi_max = jnp.max(jnp.abs(phi))
    return jax.lax.cond(
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
        G,
    )


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
    collision_operator: CollisionOperator | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using cached geometry arrays."""
    _validate_linear_method(method)
    if terms is None:
        terms = LinearTerms()
    G0, damping = _prepared_linear_state_and_damping(
        G0,
        cache,
        params,
        include_collisions=collision_operator is None,
    )
    real_dtype = jnp.real(jnp.empty((), dtype=G0.dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    parallel_strategy = _linear_parallel_strategy(parallel)
    rhs = _linear_rhs_callable(
        cache=cache,
        params=params,
        terms=terms,
        dt_val=dt_val,
        parallel=parallel,
        parallel_strategy=parallel_strategy,
        force_electrostatic_fields=force_electrostatic_fields,
        collision_operator=collision_operator,
    )
    solve_phi = _linear_phi_callable(
        cache=cache,
        params=params,
        terms=terms,
        rhs=rhs,
        parallel_strategy=parallel_strategy,
    )

    def advance(G: jnp.ndarray) -> jnp.ndarray:
        return _advance_linear_state(
            G,
            rhs=rhs,
            damping=damping,
            dt_val=dt_val,
            method=method,
        )

    def step(G: jnp.ndarray, idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        G_new = advance(G)
        phi_new = solve_phi(G_new)
        return _maybe_emit_linear_progress(
            G_new,
            idx=idx,
            steps=steps,
            dt_val=dt_val,
            phi=phi_new,
            show_progress=show_progress,
        ), phi_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    indices = jnp.arange(steps)
    if sample_stride <= 1:
        return jax.lax.scan(step_fn, G0, indices)

    def sample_step(
        G: jnp.ndarray, idx: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        def inner_step(_i, state):
            return advance(state)

        G_out = jax.lax.fori_loop(0, sample_stride, inner_step, G)
        phi_out = solve_phi(G_out)
        completed_idx = jnp.minimum((idx + 1) * sample_stride, steps) - 1
        return _maybe_emit_linear_progress(
            G_out,
            idx=completed_idx,
            steps=steps,
            dt_val=dt_val,
            phi=phi_out,
            show_progress=show_progress,
        ), phi_out

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


def _validate_linear_sampling(*, steps: int, sample_stride: int) -> None:
    """Validate fixed-step sampling before JIT dispatch."""

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
) -> LinearCache:
    """Return a supplied linear cache or build one from the state dimensions."""

    if cache is not None:
        return cache
    if G0.ndim == 5:
        Nl, Nm = G0.shape[0], G0.shape[1]
    elif G0.ndim == 6:
        Nl, Nm = G0.shape[1], G0.shape[2]
    else:
        raise ValueError(
            "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
        )
    return build_linear_cache(grid, geom, params, Nl, Nm)


def _normalize_linear_method(method: str) -> str:
    """Map public aliases onto the fixed-step method names used internally."""

    return "imex" if method == "semi-implicit" else method


def _dispatch_implicit_linear(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    dt: float,
    steps: int,
    terms: LinearTerms,
    implicit_tol: float,
    implicit_maxiter: int,
    implicit_iters: int,
    implicit_relax: float,
    implicit_restart: int,
    implicit_preconditioner: PreconditionerSpec,
    checkpoint: bool,
    sample_stride: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Route the implicit linear integration policy."""

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
        implicit_preconditioner=implicit_preconditioner,
        checkpoint=checkpoint,
        sample_stride=sample_stride,
    )


def _dispatch_parallel_linear(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    dt: float,
    steps: int,
    method: str,
    terms: LinearTerms,
    checkpoint: bool,
    sample_stride: int,
    donate: bool,
    show_progress: bool,
    parallel: Any,
    force_electrostatic_fields: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Route explicit non-serial linear integration."""

    if donate:
        raise NotImplementedError(
            "parallel linear integration does not currently support donated input buffers"
        )
    route_axis = str(getattr(parallel, "axis", "hermite")).lower().replace("-", "_")
    if route_axis in {"s", "species"}:
        from spectraxgk.solvers.linear.parallel_electrostatic import (
            prepare_electrostatic_species_inputs,
        )

        G0, cache, params = prepare_electrostatic_species_inputs(
            G0,
            cache,
            params,
            num_devices=getattr(parallel, "num_devices", None),
            replicate_cache=False,
        )
        return _integrate_species_sharded_explicit(
            G0,
            cache,
            params,
            dt=dt,
            steps=steps,
            method=method,
            terms=terms,
            checkpoint=checkpoint,
            sample_stride=sample_stride,
            show_progress=show_progress,
            parallel=parallel,
            force_electrostatic_fields=force_electrostatic_fields,
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


def _integrate_species_sharded_explicit(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    dt: float,
    steps: int,
    method: str,
    terms: LinearTerms,
    checkpoint: bool,
    sample_stride: int,
    show_progress: bool,
    parallel: Any,
    force_electrostatic_fields: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Advance one species per device inside a named-collective ``pmap``."""

    from spectraxgk.solvers.linear.parallel_common import (
        _is_electrostatic_field_terms,
        _is_electrostatic_slice_terms,
        _resolve_parallel_devices,
    )

    skip_dissipation = _is_electrostatic_slice_terms(terms)
    if method in {"imex", "imex2"}:
        raise NotImplementedError(
            "species-parallel IMEX requires a separately gated local damping solve"
        )
    from spectraxgk.operators.linear.params import (
        _as_species_array,
        linear_terms_to_term_config,
    )
    from spectraxgk.terms.assembly import assemble_rhs_cached_with_fields
    from spectraxgk.terms.config import FieldState
    from spectraxgk.terms.fields import (
        solve_electrostatic_phi_species_shard,
        solve_fields_species_shard,
    )

    state, _damping = _prepared_linear_state_and_damping(G0, cache, params)
    real_dtype = jnp.real(jnp.empty((), dtype=state.dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    devices = _resolve_parallel_devices(
        num_devices=getattr(parallel, "num_devices", None)
    )
    ns = int(state.shape[0])
    if len(devices) != ns:
        raise ValueError("species integration requires one device per species")
    species_names = (
        "charge_sign",
        "density",
        "mass",
        "temp",
        "vth",
        "rho",
        "R_over_Ln",
        "R_over_LTi",
        "R_over_LTe",
        "nu",
        "tz",
    )
    species_values = tuple(
        _as_species_array(getattr(params, name), ns, name).astype(real_dtype)
        for name in species_names
    )
    term_config = linear_terms_to_term_config(terms)
    electrostatic_fields = force_electrostatic_fields or _is_electrostatic_field_terms(
        terms
    )

    def program(local_state, local_jl, local_jlb, local_b, *species_scalars):
        local_species = tuple(jnp.reshape(value, (1,)) for value in species_scalars)
        local_cache = replace(
            cache,
            Jl=local_jl[None, ...],
            JlB=local_jlb[None, ...],
            b=local_b[None, ...],
        )
        local_params = replace(
            params, **dict(zip(species_names, local_species, strict=True))
        )

        def local_fields(value):
            if electrostatic_fields:
                phi = solve_electrostatic_phi_species_shard(
                    value[None, ...],
                    local_cache,
                    local_params,
                    local_species[0],
                    local_species[1],
                    local_species[-1],
                )
                zero = jnp.zeros_like(phi)
                return FieldState(phi=phi, apar=zero, bpar=zero)
            return solve_fields_species_shard(
                value[None, ...],
                local_cache,
                local_params,
                local_species[0],
                local_species[1],
                local_species[3],
                local_species[2],
                local_species[-1],
                local_species[4],
                jnp.asarray(params.fapar * terms.apar, dtype=real_dtype),
                jnp.asarray(terms.bpar, dtype=real_dtype),
            )

        def local_rhs(value):
            state6 = value[None, ...]
            fields = local_fields(value)
            rhs = assemble_rhs_cached_with_fields(
                state6,
                local_cache,
                local_params,
                fields,
                terms=term_config,
                force_electrostatic_fields=force_electrostatic_fields,
                skip_dissipation=skip_dissipation,
            )
            return rhs[0], fields.phi

        def advance(value):
            k1, _ = local_rhs(value)
            if method == "euler":
                return value + dt_val * k1
            if method == "rk2":
                k2, _ = local_rhs(value + 0.5 * dt_val * k1)
                return value + dt_val * k2
            if method == "sspx3":
                return _sspx3_step(value, rhs=local_rhs, dt_val=dt_val)
            k2, _ = local_rhs(value + 0.5 * dt_val * k1)
            k3, _ = local_rhs(value + 0.5 * dt_val * k2)
            k4, _ = local_rhs(value + dt_val * k3)
            return value + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        def step(value, index):
            advanced = advance(value)
            phi = local_fields(advanced).phi
            if show_progress:
                advanced = jax.lax.cond(
                    jax.lax.axis_index("species") == 0,
                    lambda x: _maybe_emit_linear_progress(
                        x,
                        idx=index,
                        steps=steps,
                        dt_val=dt_val,
                        phi=phi,
                        show_progress=True,
                    ),
                    lambda x: x,
                    advanced,
                )
            return advanced, phi

        body = jax.checkpoint(step) if checkpoint else step
        return jax.lax.scan(body, local_state, jnp.arange(steps))

    mapped = jax.pmap(program, axis_name="species", devices=devices)
    final_state, fields_by_device = mapped(
        state,
        cache.Jl,
        cache.JlB,
        cache.b,
        *species_values,
    )
    return (
        final_state,
        fields_by_device[0, sample_stride - 1 :: sample_stride],
    )


def _dispatch_serial_linear(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    dt: float,
    steps: int,
    method: str,
    terms: LinearTerms,
    checkpoint: bool,
    sample_stride: int,
    donate: bool,
    show_progress: bool,
    force_electrostatic_fields: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Route explicit serial linear integration, optionally donating ``G0``."""

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


def _dispatch_explicit_linear(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    dt: float,
    steps: int,
    method: str,
    terms: LinearTerms,
    checkpoint: bool,
    sample_stride: int,
    donate: bool,
    show_progress: bool,
    parallel: Any | None,
    parallel_strategy: str,
    force_electrostatic_fields: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Route explicit linear integration through serial or parallel kernels."""

    if parallel_strategy != "serial":
        return _dispatch_parallel_linear(
            G0,
            cache,
            params,
            dt=dt,
            steps=steps,
            method=method,
            terms=terms,
            checkpoint=checkpoint,
            sample_stride=sample_stride,
            donate=donate,
            show_progress=show_progress,
            parallel=parallel,
            force_electrostatic_fields=force_electrostatic_fields,
        )
    return _dispatch_serial_linear(
        G0,
        cache,
        params,
        dt=dt,
        steps=steps,
        method=method,
        terms=terms,
        checkpoint=checkpoint,
        sample_stride=sample_stride,
        donate=donate,
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
    implicit_preconditioner: PreconditionerSpec = None,
    terms: LinearTerms | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    donate: bool = False,
    show_progress: bool = False,
    parallel: Any | None = None,
    collision_operator: CollisionOperator | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using a fixed-step scheme."""
    terms = LinearTerms() if terms is None else terms
    _validate_linear_sampling(steps=steps, sample_stride=sample_stride)
    cache = _linear_cache_or_build(G0, grid, geom, params, cache)
    method = _normalize_linear_method(method)
    parallel_strategy = _linear_parallel_strategy(parallel)
    force_electrostatic_fields = _is_electrostatic_field_terms(terms)
    if collision_operator is not None:
        if method == "implicit":
            raise NotImplementedError(
                "custom collision operators currently require an explicit or IMEX method"
            )
        if parallel_strategy != "serial":
            raise NotImplementedError(
                "custom collision operators currently require serial state integration"
            )
        if donate:
            raise NotImplementedError(
                "custom collision operators currently do not support donated state buffers"
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
            force_electrostatic_fields=force_electrostatic_fields,
            collision_operator=collision_operator,
        )
    if method == "implicit":
        if parallel_strategy != "serial":
            raise NotImplementedError(
                "parallel linear integration currently supports only explicit fixed-step methods"
            )
        return _dispatch_implicit_linear(
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
            implicit_preconditioner=implicit_preconditioner,
            checkpoint=checkpoint,
            sample_stride=sample_stride,
        )
    return _dispatch_explicit_linear(
        G0,
        cache,
        params,
        dt=dt,
        steps=steps,
        method=method,
        terms=terms,
        checkpoint=checkpoint,
        sample_stride=sample_stride,
        donate=donate,
        show_progress=show_progress,
        parallel=parallel,
        parallel_strategy=parallel_strategy,
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
