"""Saturated-window heat flux and sheared-transport drivers for nonlinear runs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace
from functools import partial
from typing import Any, Callable
from gkx.config import method_error, resolve_nonlinear_cfl_fac
from gkx.core_grid import SpectralGrid, _gyrokinetic_moment_shape
from gkx.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from gkx.operators.collision import CollisionOperator
from gkx.operators.fluxes import (
    heat_flux_species,
    heat_flux_total,
)
from gkx.operators.linear.cache_builder import (
    build_linear_cache,
    update_linear_cache_for_sheared_kx,
)
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.linked import (
    mask_supplied_state,
)
from gkx.operators.linear.params import LinearParams, _x64_enabled
from gkx.operators.moments import (
    fieldline_quadrature_weights,
)
from gkx.operators.nonlinear.policies import (
    _nonlinear_cfl_frequency_components,
    build_nonlinear_imex_operator,
    build_nonlinear_time_step_policy,
)
from gkx.operators.nonlinear.projection import (
    _make_hermitian_projector,
    advance_shearing_coordinates,
    hermitian_projector_for_signature,
    hermitian_projector_signature,
)
from gkx.operators.nonlinear.rhs import (
    nonlinear_em_term_cached_impl,
)
from gkx.solvers.linear.implicit import (
    ImplicitSolveStats,
    _build_implicit_operator,
    _empty_implicit_solve_stats,
)
from gkx.solvers.nonlinear.explicit import (
    advance_explicit_nonlinear_state,
    checkpointed_explicit_scan,
)
from gkx.solvers.nonlinear.imex import (
    solve_imex_step_with_stats,
)
from gkx.solvers.time_explicit import (
    _laguerre_velocity_max,
    _linear_frequency_bound,
)
from gkx.terms.assembly import (
    compute_fields_cached,
)
from gkx.terms.config import FieldState, TermConfig
from gkx.terms.nonlinear import nonlinear_em_contribution


from gkx.solvers.nonlinear.diagnostics import (
    ADJOINT_MEMORY_BUDGET_BYTES,
    DIVERGENCE_KNEE_STEPS,
    ShearedTransportTrace,
    _identity_state,
    _linear_rhs_jit_for_terms,
    _warn_if_window_exceeds_divergence_knee,
    nonlinear_rhs_cached,
)


@partial(
    jax.jit,
    static_argnames=(
        "dt",
        "count",
        "tail",
        "method",
        "term_cfg",
        "compressed_real_fft",
        "laguerre_mode",
        "collision_operator",
        "checkpoint",
        "projector_signature",
        "memory_budget_bytes",
    ),
)
def _nonlinear_heat_flux_window_total(
    initial_state: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_factor: jnp.ndarray,
    initial_heat: jnp.ndarray,
    indices: jnp.ndarray,
    *,
    dt: float,
    count: int,
    tail: int,
    method: str,
    term_cfg: TermConfig,
    compressed_real_fft: bool,
    laguerre_mode: str,
    collision_operator: CollisionOperator | None,
    checkpoint: bool,
    projector_signature: tuple[int, bool, int] | None,
    memory_budget_bytes: int | None = None,
) -> jnp.ndarray:
    """Compile a reusable heat-flux sum for fixed shapes and static options.

    Arrays are operands, not captured geometry constants. A module-level JIT
    avoids rebuilding eager scan/cond executables on each objective call and
    permits reuse through ``value_and_grad``. Projector layout is static and
    has no derivative; configuration or topology changes may recompile.

    Keep ``total_heat / tail`` outside this graph: fusing the mean into the
    accumulation changed float32 rounding in the reference comparison.
    """

    project_state: Callable[[jnp.ndarray], jnp.ndarray] = (
        _identity_state
        if projector_signature is None
        else hermitian_projector_for_signature(projector_signature)
    )

    def rhs(state: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
        return nonlinear_rhs_cached(
            state,
            cache,
            params,
            term_cfg,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            collision_operator=collision_operator,
            differentiable=True,
        )

    def advance(carry: tuple[jnp.ndarray, jnp.ndarray], index: jnp.ndarray):
        state, total_heat = carry
        state = project_state(state)
        derivative, _ = rhs(state)
        next_state = advance_explicit_nonlinear_state(
            state,
            derivative,
            jnp.asarray(dt, dtype=jnp.real(state).dtype),
            method=method,
            rhs_fn=rhs,
            project_state=project_state,
            state_dtype=state.dtype,
        )
        _, fields = rhs(next_state)
        zero = jnp.zeros_like(fields.phi)
        heat = heat_flux_total(
            next_state,
            fields.phi,
            zero if fields.apar is None else fields.apar,
            zero if fields.bpar is None else fields.bpar,
            cache,
            grid,
            params,
            flux_factor,
        )
        include = jnp.asarray(index >= count - tail, dtype=heat.dtype)
        return (next_state, total_heat + include * heat), None

    (_, total_heat), _ = checkpointed_explicit_scan(
        advance,
        (jax.lax.stop_gradient(initial_state), initial_heat),
        indices,
        checkpoint=checkpoint,
        memory_budget_bytes=memory_budget_bytes,
    )
    return total_heat


def nonlinear_heat_flux_window(
    saturated_state: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    terms: TermConfig | None = None,
    method: str = "rk2",
    tail_steps: int | None = None,
    checkpoint: bool = True,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    collision_operator: CollisionOperator | None = None,
    divergence_knee_steps: int | None = DIVERGENCE_KNEE_STEPS,
    adjoint_memory_budget_bytes: int | None = ADJOINT_MEMORY_BUDGET_BYTES,
) -> jnp.ndarray:
    r"""Mean physical heat flux over a differentiable nonlinear window.

    ``saturated_state`` is detached: first run GKX to saturation,
    then differentiate this finite window through ``geom`` and ``params``.
    Reverse-mode AD follows the exact discrete Runge--Kutta map. Block
    checkpointing retains :math:`O(\sqrt{N})` distribution states instead of
    :math:`O(N)` for a window of ``N`` steps.

    On a linked (twist-shift) deck ``saturated_state`` is projected onto the
    linked chain cover before the window starts: it is a state
    this function did not build, and the ExB bracket takes the whole state to
    real space while dealiasing only its output, so off-chain rows would alias
    back onto the chain rows and change the flux this objective returns. The
    projection does not change the gradient contract, which is that
    ``saturated_state`` is detached and only ``geom`` and ``params`` carry
    derivatives. Periodic and full-cover decks are untouched.

    ``collision_operator`` is the same custom model
    :func:`integrate_nonlinear` accepts, and must be passed here too: a run
    saturated with a custom operator and then differentiated without one is a
    derivative of different physics from the trajectory it starts on.

    ``steps`` above ``divergence_knee_steps`` warns. Past the measured knee the
    windowed adjoint grows with the leading Lyapunov exponent and stops being a
    useful design direction; pass ``None`` to silence the check when the knee
    has been checked against centered finite differences for the case at hand.

    ``adjoint_memory_budget_bytes`` bounds the reverse pass's checkpoint
    storage. When the block-only schedule fits, each step's forward runs twice
    in a gradient instead of three times; otherwise, or with ``None``, the
    nested :math:`O(\sqrt{N})` schedule is used. It changes cost, not the
    discrete adjoint.

    Host-side geometry/layout preparation stays here. The differentiated scan
    in :func:`_nonlinear_heat_flux_window_total` reuses its executable for
    matching shapes, topology and static options, with geometry arrays passed
    as operands. See ``docs/nonlinear_autodiff.rst`` for scope and evidence.
    """

    count = int(steps)
    tail = count if tail_steps is None else int(tail_steps)
    if count < 1 or not 1 <= tail <= count:
        raise ValueError("steps must be positive and tail_steps must lie within it")
    _warn_if_window_exceeds_divergence_knee(count, divergence_knee_steps)
    if saturated_state.ndim not in (5, 6):
        raise ValueError(
            "saturated_state must have shape (Nl, Nm, Ny, Nx, Nz) or "
            "(Ns, Nl, Nm, Ny, Nx, Nz)"
        )
    offset = saturated_state.ndim - 5
    Nl, Nm = saturated_state.shape[offset : offset + 2]

    term_cfg = terms or TermConfig(nonlinear=1.0)
    geometry = ensure_flux_tube_geometry_data(geom, grid.z)
    cache = build_linear_cache(grid, geometry, params, Nl=Nl, Nm=Nm)
    _volume_factor, flux_factor = fieldline_quadrature_weights(geometry, grid)
    projector_signature = (
        hermitian_projector_signature(
            np.asarray(grid.ky), int(np.asarray(grid.kx).size)
        )
        if compressed_real_fft
        else None
    )
    initial_state = jax.lax.stop_gradient(
        mask_supplied_state(jnp.asarray(saturated_state), cache)
    )
    heat_dtype = jnp.result_type(
        jnp.real(initial_state), flux_factor, *jax.tree_util.tree_leaves(params)
    )
    total_heat = _nonlinear_heat_flux_window_total(
        initial_state,
        cache,
        grid,
        params,
        flux_factor,
        jnp.zeros((), dtype=heat_dtype),
        jnp.arange(count),
        dt=dt,
        count=count,
        tail=tail,
        method=method,
        term_cfg=term_cfg,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        collision_operator=collision_operator,
        checkpoint=checkpoint,
        projector_signature=projector_signature,
        memory_budget_bytes=(
            None
            if adjoint_memory_budget_bytes is None
            else int(adjoint_memory_budget_bytes)
        ),
    )
    return total_heat / tail


def _integrate_nonlinear_sheared_scan(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    shear_rate: jnp.ndarray | float,
    method: str = "rk2",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    laguerre_mode: str = "grid",
    collision_operator: CollisionOperator | None = None,
    compressed_real_fft: bool = False,
    record_transport: bool = False,
    return_fields: bool = True,
    flux_scale: float = 1.0,
    differentiable: bool = False,
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    cfl: float = 0.9,
    cfl_fac: float | None = None,
    initial_time: jnp.ndarray | float = 0.0,
    initial_dt: jnp.ndarray | float | None = None,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
) -> tuple[jnp.ndarray, Any, ImplicitSolveStats | None]:
    """Run the shared shearing-coordinate scan with optional transport output.

    Returns the final state, the scan output, and the implicit solve status of
    a ``method="imex"`` run (``None`` for the explicit methods, which take no
    implicit solve and so add no carry leaf).
    """

    if str(grid.boundary).lower() not in {"periodic", "linked"} or bool(grid.non_twist):
        raise NotImplementedError(
            "sheared integration requires a periodic or linked standard flux tube"
        )
    if steps < 1:
        raise ValueError("steps must be at least one")
    method_key = str(method).lower()
    if method_key not in {"rk2", "rk3", "imex"}:
        raise method_error(method, {"rk2", "rk3", "imex"})
    if not fixed_dt and not record_transport:
        raise ValueError("adaptive sheared integration requires a transport trace")
    if method_key == "imex" and not fixed_dt:
        raise ValueError("sheared IMEX integration currently requires fixed_dt=True")
    if method_key == "imex" and collision_operator is not None:
        raise NotImplementedError(
            "sheared IMEX does not yet support custom collision operators"
        )
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    if cache is None:
        nl, nm = _gyrokinetic_moment_shape(G0)
        cache = build_linear_cache(grid, geom_eff, params, nl, nm)

    term_cfg = terms or TermConfig()
    linear_cfg = replace(term_cfg, nonlinear=0.0)
    linear_rhs_fn = _linear_rhs_jit_for_terms(linear_cfg)
    base_complex_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_complex_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_value = jnp.asarray(dt, dtype=real_dtype)
    initial_time_value = jnp.asarray(initial_time, dtype=real_dtype)
    initial_dt_value = jnp.asarray(
        dt if initial_dt is None else initial_dt, dtype=real_dtype
    )
    rho_star = jnp.asarray(params.rho_star, dtype=real_dtype)
    coordinate_kx = jnp.asarray(cache.kx, dtype=real_dtype)
    coordinate_ky = jnp.asarray(cache.ky, dtype=real_dtype)
    if int(coordinate_kx.size) > 1:
        coordinate_x0 = 1.0 / jnp.abs(coordinate_kx[1] - coordinate_kx[0])
    else:
        coordinate_x0 = jnp.asarray(grid.x0, dtype=real_dtype) / rho_star
    _, flux_fac = fieldline_quadrature_weights(geom_eff, grid)
    # Full-complex shearing coordinates must retain the real-field subspace.
    project_state = _make_hermitian_projector(
        np.asarray(grid.ky), int(np.asarray(grid.kx).size)
    )
    time_step_policy = None
    if not fixed_dt:
        time_step_policy = build_nonlinear_time_step_policy(
            grid,
            geom_eff,
            params,
            cache,
            method=method_key,
            dt=dt,
            steps=steps,
            fixed_dt=False,
            dt_min=dt_min,
            dt_max=dt_max,
            cfl=cfl,
            cfl_fac=cfl_fac,
            compressed_real_fft=compressed_real_fft,
            real_dtype=real_dtype,
            resolve_cfl_fac_fn=resolve_nonlinear_cfl_fac,
            linear_frequency_bound_fn=_linear_frequency_bound,
            laguerre_velocity_max_fn=_laguerre_velocity_max,
            cfl_frequency_components_fn=_nonlinear_cfl_frequency_components,
        )

    def coordinates(state: jnp.ndarray, time: jnp.ndarray, previous: jnp.ndarray):
        update = advance_shearing_coordinates(
            state,
            # The linked-boundary cache may adjust the radial spacing to close
            # the twist-shift chain. Working in the cache's normalized units
            # preserves both that adjustment and the periodic convention.
            kx=coordinate_kx,
            ky=coordinate_ky,
            x0=coordinate_x0,
            shear_rate=shear_rate,
            previous_time=previous,
            time=time,
            dealias_mask=grid.dealias_mask,
        )
        return update._replace(state=project_state(update.state))

    def cache_at(update):
        return update_linear_cache_for_sheared_kx(
            cache,
            grid,
            geom_eff,
            params,
            update.effective_kx,
        )

    def rhs_at(update):
        updated_cache = cache_at(update)
        derivative, fields = nonlinear_rhs_cached(
            update.state,
            updated_cache,
            params,
            term_cfg,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            collision_operator=collision_operator,
            radial_phase=update.phase,
            differentiable=differentiable,
        )
        return derivative, fields, updated_cache

    def sheared_fields(
        state,
        updated_cache,
        field_params,
        *,
        terms=None,
        external_phi=None,
    ):
        return compute_fields_cached(
            state,
            updated_cache,
            field_params,
            terms=terms,
            use_custom_vjp=not differentiable,
            external_phi=external_phi,
        )

    def fields_at(update):
        updated_cache = cache_at(update)
        fields = sheared_fields(
            update.state,
            updated_cache,
            params,
            terms=term_cfg,
        )
        return fields, updated_cache

    def advance(current, derivative, time, dt_local, solve_stats=None):
        """Advance one sheared step, folding any implicit solve into the status.

        ``solve_stats`` rides through the scan carry because a traced step
        cannot raise, as in the cached IMEX and implicit linear scans.
        Explicit methods pass ``None`` straight back, so they
        add no carry leaf and compile unchanged.
        """

        new_time = time + dt_local
        if method_key == "imex":
            current_cache = cache_at(current)
            nonlinear_term = nonlinear_em_term_cached_impl(
                current.state,
                current_cache,
                params,
                term_cfg,
                external_phi=None,
                compressed_real_fft=compressed_real_fft,
                laguerre_mode=laguerre_mode,
                radial_phase=current.phase,
                fields_fn=sheared_fields,
                nonlinear_contribution_fn=nonlinear_em_contribution,
            )
            endpoint_guess = coordinates(current.state, new_time, time)
            endpoint_rhs = coordinates(
                current.state + dt_local * nonlinear_term,
                new_time,
                time,
            )
            endpoint_cache = cache_at(endpoint_guess)
            operator = build_nonlinear_imex_operator(
                endpoint_guess.state,
                endpoint_cache,
                params,
                dt_local,
                terms=linear_cfg,
                implicit_preconditioner=implicit_preconditioner,
                compressed_real_fft=compressed_real_fft,
                build_implicit_operator_fn=_build_implicit_operator,
            )
            guess = jnp.asarray(endpoint_guess.state, dtype=operator.state_dtype)
            rhs = jnp.asarray(endpoint_rhs.state, dtype=operator.state_dtype)
            if operator.squeeze_species:
                guess = guess[None, ...]
                rhs = rhs[None, ...]
            solution, solve_stats = solve_imex_step_with_stats(
                guess,
                rhs,
                solve_stats,
                linear_rhs_fn=linear_rhs_fn,
                cache=endpoint_cache,
                params=params,
                linear_cfg=linear_cfg,
                external_phi=None,
                dt_val=operator.dt_val,
                implicit_iters=implicit_iters,
                implicit_relax=implicit_relax,
                matvec=operator.matvec,
                shape=operator.shape,
                implicit_tol=implicit_tol,
                implicit_maxiter=implicit_maxiter,
                implicit_restart=implicit_restart,
                precond_op=operator.precond_op,
            )
            if operator.squeeze_species:
                solution = solution[0]
            return (
                endpoint_rhs._replace(state=project_state(solution)),
                new_time,
                solve_stats,
            )
        if method_key == "rk2":
            midpoint_time = time + 0.5 * dt_local
            midpoint = coordinates(
                current.state + 0.5 * dt_local * derivative,
                midpoint_time,
                time,
            )
            midpoint_derivative, _, _ = rhs_at(midpoint)
            derivative_in_step_basis = coordinates(
                midpoint_derivative,
                time,
                midpoint_time,
            ).state
            trial = current.state + dt_local * derivative_in_step_basis
        else:
            stage1_time = time + dt_local / 3.0
            stage1 = coordinates(
                current.state + (dt_local / 3.0) * derivative,
                stage1_time,
                time,
            )
            stage1_derivative, _, _ = rhs_at(stage1)
            stage1_derivative_base = coordinates(
                stage1_derivative,
                time,
                stage1_time,
            ).state
            stage2_time = time + 2.0 * dt_local / 3.0
            stage2 = coordinates(
                current.state + (2.0 * dt_local / 3.0) * stage1_derivative_base,
                stage2_time,
                time,
            )
            stage2_derivative, _, _ = rhs_at(stage2)
            stage2_derivative_base = coordinates(
                stage2_derivative,
                time,
                stage2_time,
            ).state
            trial = (
                current.state
                + 0.25 * dt_local * derivative
                + 0.75 * dt_local * (stage2_derivative_base)
            )
        return coordinates(trial, new_time, time), new_time, solve_stats

    def local_dt(current_fields, dt_previous):
        if time_step_policy is None:
            return dt_value
        return time_step_policy.update_dt(current_fields, dt_previous)

    # The status leaf exists only on the route that takes an implicit solve, so
    # the explicit sheared scans keep exactly the carry they had.
    carries_solve_stats = method_key == "imex"

    def _stats_of(carry, position: int):
        return carry[position] if carries_solve_stats else None

    def _stats_tail(solve_stats):
        return (solve_stats,) if carries_solve_stats else ()

    def state_only_step(carry: tuple[Any, ...], index: jnp.ndarray):
        state, adaptive_time, dt_previous = carry[:3]
        solve_stats = _stats_of(carry, 3)
        fixed_time = (
            initial_time_value + jnp.asarray(index, dtype=real_dtype) * dt_value
        )
        time = fixed_time if fixed_dt else adaptive_time
        current = coordinates(state, time, time)
        if method_key == "imex":
            derivative = jnp.zeros_like(current.state)
            current_fields = None
        else:
            derivative, current_fields, _ = rhs_at(current)
        dt_local = local_dt(current_fields, dt_previous)
        advanced, new_time, solve_stats = advance(
            current, derivative, time, dt_local, solve_stats
        )
        next_carry = (
            jnp.asarray(advanced.state, dtype=state_dtype),
            jnp.asarray(new_time, dtype=real_dtype),
            jnp.asarray(dt_local, dtype=real_dtype),
        ) + _stats_tail(solve_stats)
        return next_carry, None

    def endpoint_step(carry: tuple[Any, ...], index: jnp.ndarray):
        state, adaptive_time, dt_previous, derivative, current_fields = carry[:5]
        solve_stats = _stats_of(carry, 5)
        del index
        time = adaptive_time
        current = coordinates(state, time, time)
        dt_local = local_dt(current_fields, dt_previous)
        advanced, new_time, solve_stats = advance(
            current, derivative, time, dt_local, solve_stats
        )
        if method_key == "imex":
            next_derivative = jnp.zeros_like(advanced.state)
            fields, advanced_cache = fields_at(advanced)
        else:
            next_derivative, fields, advanced_cache = rhs_at(advanced)
        next_carry = (
            jnp.asarray(advanced.state, dtype=state_dtype),
            jnp.asarray(new_time, dtype=real_dtype),
            jnp.asarray(dt_local, dtype=real_dtype),
            jnp.asarray(next_derivative, dtype=state_dtype),
            fields,
        ) + _stats_tail(solve_stats)
        if not record_transport:
            return next_carry, fields
        apar = jnp.zeros_like(fields.phi) if fields.apar is None else fields.apar
        bpar = jnp.zeros_like(fields.phi) if fields.bpar is None else fields.bpar
        heat_flux = heat_flux_species(
            advanced.state,
            fields.phi,
            apar,
            bpar,
            advanced_cache,
            grid,
            params,
            flux_fac,
            use_dealias=True,
            flux_scale=flux_scale,
        )
        return next_carry, (new_time, heat_flux)

    initial_state_carry = (
        jnp.asarray(project_state(G0), dtype=state_dtype),
        initial_time_value,
        initial_dt_value,
    )
    initial_stats = (
        _empty_implicit_solve_stats(state_dtype) if carries_solve_stats else None
    )
    if not record_transport and not return_fields:
        state_final_carry, output = jax.lax.scan(
            state_only_step,
            initial_state_carry + _stats_tail(initial_stats),
            jnp.arange(steps),
        )
        return state_final_carry[0], output, _stats_of(state_final_carry, 3)

    initial_update = coordinates(
        initial_state_carry[0], initial_state_carry[1], initial_state_carry[1]
    )
    if method_key == "imex":
        initial_derivative = jnp.zeros_like(initial_update.state)
        initial_fields, _ = fields_at(initial_update)
    else:
        initial_derivative, initial_fields, _ = rhs_at(initial_update)
    # Some field-solve policies use a lower internal precision. Match the scan
    # carry to the requested state precision just as every subsequent step does.
    initial_endpoint_carry = (
        initial_state_carry
        + (
            jnp.asarray(initial_derivative, dtype=state_dtype),
            initial_fields,
        )
        + _stats_tail(initial_stats)
    )
    endpoint_final_carry, output = jax.lax.scan(
        endpoint_step, initial_endpoint_carry, jnp.arange(steps)
    )
    return endpoint_final_carry[0], output, _stats_of(endpoint_final_carry, 5)


def integrate_nonlinear_sheared(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    shear_rate: jnp.ndarray | float,
    method: str = "rk2",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    laguerre_mode: str = "grid",
    collision_operator: CollisionOperator | None = None,
    compressed_real_fft: bool = False,
    differentiable: bool = False,
    return_fields: bool = True,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
    return_solve_stats: bool = False,
) -> tuple[Any, ...] | jnp.ndarray:
    """Integrate the standard-flux-tube shearing-coordinate foundation.

    This research path supports midpoint RK2, three-stage
    Heun RK3, and first-order IMEX. Stage states and derivatives are remapped to
    the stage coordinate basis before the RHS and back to the step basis before
    Runge--Kutta combinations. IMEX evaluates the explicit nonlinear term in the
    current basis and rebuilds the implicit linear operator in the endpoint
    basis. ``compressed_real_fft`` evaluates the nonlinear bracket in the
    equivalent canonical shearing-coordinate representation.

    ``return_solve_stats=True`` appends the implicit solve summary of a
    ``method="imex"`` run (:class:`~gkx.solvers.linear.implicit.ImplicitSolveStats`,
    ``None`` for the explicit methods). The traced scan cannot raise on a
    starved inner budget, so this is the convergence channel; a host-side caller
    turns it into a refusal with
    :func:`~gkx.solvers.linear.implicit.require_converged_implicit_solves`.
    """

    final_state, fields, solve_stats = _integrate_nonlinear_sheared_scan(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        shear_rate=shear_rate,
        method=method,
        cache=cache,
        terms=terms,
        laguerre_mode=laguerre_mode,
        collision_operator=collision_operator,
        compressed_real_fft=compressed_real_fft,
        differentiable=differentiable,
        return_fields=return_fields,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_restart=implicit_restart,
        implicit_preconditioner=implicit_preconditioner,
    )
    head: tuple[Any, ...] = (final_state, fields) if return_fields else (final_state,)
    if return_solve_stats:
        return (*head, solve_stats)
    return head if return_fields else final_state


def integrate_nonlinear_sheared_transport(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    shear_rate: jnp.ndarray | float,
    method: str = "rk2",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    laguerre_mode: str = "grid",
    collision_operator: CollisionOperator | None = None,
    compressed_real_fft: bool = False,
    flux_scale: float = 1.0,
    differentiable: bool = True,
    fixed_dt: bool = True,
    dt_min: float = 1.0e-7,
    dt_max: float | None = None,
    cfl: float = 0.9,
    cfl_fac: float | None = None,
    initial_time: jnp.ndarray | float = 0.0,
    initial_dt: jnp.ndarray | float | None = None,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
    return_solve_stats: bool = False,
) -> ShearedTransportTrace:
    """Integrate a sheared run and record canonical heat flux at every step.

    With ``fixed_dt=False``, ``steps`` is the accepted-step budget and ``time``
    records the resulting nonuniform physical-time grid. ``initial_time`` and
    ``initial_dt`` continue a prior trace without resetting the shearing basis.

    ``return_solve_stats=True`` fills the trace's ``solve_stats`` field with the
    implicit solve summary of a ``method="imex"`` run; it stays ``None``
    otherwise, so a caller that does not ask reads and compiles exactly what it
    did before.
    """

    final_state, samples, solve_stats = _integrate_nonlinear_sheared_scan(
        G0,
        grid,
        geom,
        params,
        dt,
        steps,
        shear_rate=shear_rate,
        method=method,
        cache=cache,
        terms=terms,
        laguerre_mode=laguerre_mode,
        collision_operator=collision_operator,
        compressed_real_fft=compressed_real_fft,
        record_transport=True,
        flux_scale=flux_scale,
        differentiable=differentiable,
        fixed_dt=fixed_dt,
        dt_min=dt_min,
        dt_max=dt_max,
        cfl=cfl,
        cfl_fac=cfl_fac,
        initial_time=initial_time,
        initial_dt=initial_dt,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_restart=implicit_restart,
        implicit_preconditioner=implicit_preconditioner,
    )
    time, heat_flux = samples
    return ShearedTransportTrace(
        final_state, time, heat_flux, solve_stats if return_solve_stats else None
    )


__all__ = [
    "integrate_nonlinear_sheared",
    "integrate_nonlinear_sheared_transport",
    "nonlinear_heat_flux_window",
]
