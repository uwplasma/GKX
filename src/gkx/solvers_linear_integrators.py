"""Linear time-integration and diagnostic sampling policies."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable
import jax
import jax.numpy as jnp

from gkx.operators.collision import CollisionOperator
from gkx.geometry import FluxTubeGeometryLike
from gkx.core_grid import (
    SpectralGrid,
    _gyrokinetic_moment_shape,
)
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
    _integrate_linear_implicit_cached,
    _validate_implicit_sample_policy,
    _ImplicitSolveOptions,
    _build_implicit_operator,
    ImplicitSolveStats,
    _build_implicit_solve_step,
    _empty_implicit_solve_stats,
)
from gkx.solvers_linear_parallel import (
    _is_electrostatic_field_terms,
    linear_rhs_parallel_cached,
)
from gkx.solvers_time_explicit_steps import (
    _linear_explicit_stage_update,
    _linear_native_step,
)


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
    terms: LinearTerms | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    terms = LinearTerms() if terms is None else terms
    hyper_damp = terms.hypercollisions * hypercollision_damping(
        cache, params, real_dtype
    )
    if G0.ndim == 5 and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]
    damping = hyper_damp
    if include_collisions:
        damping = damping + terms.collisions * collision_damping(
            cache, params, real_dtype, squeeze_species=G0.ndim == 5
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
    parallel: Any | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return the cheapest field-only diagnostic path for an updated state."""

    parallel_axis = str(getattr(parallel, "axis", "")).lower().replace("-", "_")
    mixed_velocity_route = parallel_axis in {"species_hermite", "s_m", "mixed"}
    if (
        parallel_strategy != "serial"
        and not mixed_velocity_route
        or not isinstance(cache, LinearCache)
    ):
        return lambda value: rhs(value)[1]

    from gkx.operators.linear.params import linear_terms_to_term_config
    from gkx.terms.assembly import compute_fields_cached

    term_config = linear_terms_to_term_config(terms)

    def solve_phi(value: jnp.ndarray) -> jnp.ndarray:
        return compute_fields_cached(
            value, cache, params, terms=term_config, use_custom_vjp=True
        ).phi

    return solve_phi


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
    from gkx.callbacks import print_callback, should_emit_progress

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
        terms=terms,
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
        parallel=parallel,
    )

    def advance(G: jnp.ndarray) -> jnp.ndarray:
        return _linear_native_step(
            G,
            damping,
            dt_val,
            method_key=method,
            rhs=lambda value: rhs(value)[0],
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


_LINEAR_INTEGRATOR_STATIC_ARGS = (
    "steps",
    "method",
    "checkpoint",
    "sample_stride",
    "show_progress",
    "force_electrostatic_fields",
)


def _bind_cached_linear_integrator(
    *, donate: bool
) -> Callable[..., tuple[jnp.ndarray, jnp.ndarray]]:
    def integrate(
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

    return jax.jit(
        integrate,
        static_argnames=_LINEAR_INTEGRATOR_STATIC_ARGS,
        donate_argnums=(0,) if donate else (),
    )


_integrate_linear_cached = _bind_cached_linear_integrator(donate=False)
_integrate_linear_cached_donate = _bind_cached_linear_integrator(donate=True)


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
    if route_axis in {"species_hermite", "s_m", "mixed"} and method not in {
        "euler",
        "rk2",
    }:
        raise NotImplementedError(
            "mixed species-Hermite integration is currently gated for Euler and RK2"
        )
    if route_axis in {"s", "species"}:
        from gkx.solvers_linear_parallel_electrostatic import (
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
    if route_axis in {"species_hermite", "s_m", "mixed"}:
        from gkx.solvers_linear_parallel_streaming import (
            prepare_electrostatic_species_hermite_state,
        )

        G0 = prepare_electrostatic_species_hermite_state(
            G0,
            species_chunks=int(G0.shape[0]),
            hermite_chunks=2,
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

    from gkx.solvers_linear_parallel_common import (
        _is_electrostatic_field_terms,
        _is_electrostatic_slice_terms,
        _resolve_parallel_devices,
    )

    skip_dissipation = _is_electrostatic_slice_terms(terms)
    if method in {"imex", "imex2"}:
        raise NotImplementedError(
            "species-parallel IMEX requires a separately gated local damping solve"
        )
    from gkx.operators.linear.params import (
        _as_species_array,
        linear_terms_to_term_config,
    )
    from gkx.terms.assembly import assemble_rhs_cached_with_fields
    from gkx.terms.config import FieldState
    from gkx.terms.fields import (
        solve_electrostatic_phi_species_shard,
        solve_fields_species_shard,
    )

    state, _damping = _prepared_linear_state_and_damping(G0, cache, params, terms=terms)
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
        "fprim",
        "tprim",
        "tprim_e",
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
            return _linear_explicit_stage_update(
                value,
                dt_val,
                method_key=method,
                rhs=lambda state: local_rhs(state)[0],
            )

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

        if sample_stride <= 1:
            body = jax.checkpoint(step) if checkpoint else step
            return jax.lax.scan(body, local_state, jnp.arange(steps))

        def sample_step(value, sample_index):
            advanced = jax.lax.fori_loop(
                0, sample_stride, lambda _index, carry: advance(carry), value
            )
            phi = local_fields(advanced).phi
            completed_index = jnp.minimum((sample_index + 1) * sample_stride, steps) - 1
            if show_progress:
                advanced = jax.lax.cond(
                    jax.lax.axis_index("species") == 0,
                    lambda x: _maybe_emit_linear_progress(
                        x,
                        idx=completed_index,
                        steps=steps,
                        dt_val=dt_val,
                        phi=phi,
                        show_progress=True,
                    ),
                    lambda x: x,
                    advanced,
                )
            return advanced, phi

        sample_body = jax.checkpoint(sample_step) if checkpoint else sample_step
        return jax.lax.scan(
            sample_body, local_state, jnp.arange(steps // sample_stride)
        )

    mapped = jax.pmap(
        program,
        axis_name="species",
        devices=devices,
        out_axes=(0, None),
    )
    final_state, field_history = mapped(
        state,
        cache.Jl,
        cache.JlB,
        cache.b,
        *species_values,
    )
    return (
        final_state,
        field_history,
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
    return_solve_stats: bool = False,
) -> tuple[Any, ...]:
    """Time integrate the linear system using a fixed-step scheme.

    ``return_solve_stats=True`` appends the convergence summary of the implicit
    GMRES solves (:class:`~gkx.solvers_linear_implicit.ImplicitSolveStats`) for
    ``method="implicit"`` and ``None`` for methods without an implicit solve.
    """
    if return_solve_stats and method != "implicit":
        return (*integrate_linear(**{**locals(), "return_solve_stats": False}), None)
    terms = LinearTerms() if terms is None else terms
    _validate_implicit_sample_policy(steps=steps, sample_stride=sample_stride)
    cache = _linear_cache_or_build(
        G0, grid, geom, params, cache, cache_builder=build_linear_cache
    )
    method = "imex" if method == "semi-implicit" else method
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
            return_solve_stats=return_solve_stats,
        )
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
    terms: LinearTerms | None = None,
) -> jnp.ndarray:
    terms = LinearTerms() if terms is None else terms
    hyper_damp = terms.hypercollisions * hypercollision_damping(
        cache, params, real_dtype
    )
    if G.ndim == 5 and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]
    damping = hyper_damp
    if include_collisions:
        damping = damping + terms.collisions * collision_damping(
            cache, params, real_dtype, squeeze_species=G.ndim == 5
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


# One step maps (state, carried solve status) to the same pair. Explicit
# methods carry None; the implicit route folds ImplicitSolveStats.
AdvanceFn = Callable[[jnp.ndarray, Any], tuple[jnp.ndarray, Any]]


def _every_step_scan(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    *,
    dt_val: jnp.ndarray,
    steps: int,
    advance: AdvanceFn,
    species_index: int | None,
    record_hl_energy: bool,
    show_progress: bool,
    collision_operator: Any | None = None,
    initial_stats: ImplicitSolveStats | None = None,
) -> tuple[tuple[jnp.ndarray, Any], tuple[jnp.ndarray, ...]]:
    def step(carry: tuple[jnp.ndarray, Any], idx: jnp.ndarray):
        G_out, stats = advance(*carry)
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
        return (G_out, stats), outputs

    return jax.lax.scan(step, (G0, initial_stats), jnp.arange(steps))


def _strided_sample_scan(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms,
    *,
    dt_val: jnp.ndarray,
    steps: int,
    sample_stride: int,
    advance: AdvanceFn,
    species_index: int | None,
    record_hl_energy: bool,
    show_progress: bool,
    collision_operator: Any | None = None,
    initial_stats: ImplicitSolveStats | None = None,
) -> tuple[tuple[jnp.ndarray, Any], tuple[jnp.ndarray, ...]]:
    def sample_step(carry: tuple[jnp.ndarray, Any], idx: jnp.ndarray):
        def inner_step(_i: jnp.ndarray, inner: tuple[jnp.ndarray, Any]):
            return advance(*inner)

        G_out, stats = jax.lax.fori_loop(0, sample_stride, inner_step, carry)
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
        return (G_out, stats), outputs

    num_samples = steps // sample_stride
    return jax.lax.scan(sample_step, (G0, initial_stats), jnp.arange(num_samples))


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
    return_solve_stats: bool = False,
) -> tuple[Any, ...]:
    """Integrate and return (G_out, phi_t, density_t) for diagnostics.

    ``record_hl_energy`` appends ``hl_t``. ``return_solve_stats=True`` appends
    the implicit GMRES convergence summary as the last element
    (:class:`~gkx.solvers_linear_implicit.ImplicitSolveStats`, or ``None`` for
    methods without an implicit solve).
    """

    terms_use = terms or LinearTerms()
    _validate_implicit_sample_policy(steps=steps, sample_stride=sample_stride)
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
            terms=terms_use,
        )

        def advance(state: jnp.ndarray, stats: Any) -> tuple[jnp.ndarray, Any]:
            next_state = _linear_native_step(
                state,
                damping,
                dt_val,
                method_key=method,
                rhs=lambda value: _rhs(
                    value, cache_use, params, terms_use, dt_val, collision_operator
                )[0],
            )
            return next_state, stats

    # Explicit methods carry None, which adds no leaves to the scan carry.
    initial_stats = (
        _empty_implicit_solve_stats(G.dtype) if method == "implicit" else None
    )
    if sample_stride <= 1:
        (G_out, solve_stats), outputs = _every_step_scan(
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
            initial_stats=initial_stats,
        )
    else:
        (G_out, solve_stats), outputs = _strided_sample_scan(
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
            initial_stats=initial_stats,
        )
    if squeeze_species:
        G_out = G_out[0]
    result = (G_out, *outputs)
    return (*result, solve_stats) if return_solve_stats else result


__all__ = [
    "_integrate_linear_cached",
    "_integrate_linear_cached_donate",
    "_integrate_linear_cached_impl",
    "integrate_linear",
    "integrate_linear_diagnostics",
]
