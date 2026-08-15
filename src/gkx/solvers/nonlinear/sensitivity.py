"""Matrix-free sensitivities and stability diagnostics for discrete trajectories."""

from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple, cast

import jax
import jax.numpy as jnp

from gkx.solvers.nonlinear.explicit import checkpointed_explicit_scan


DiscreteStep = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
InstantaneousObjective = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


class DiscreteNILSASResult(NamedTuple):
    """Finite-trajectory discrete NILSAS result and conditioning diagnostics."""

    value: jnp.ndarray
    gradient: jnp.ndarray
    coefficients: jnp.ndarray
    constraint_residual: jnp.ndarray
    kkt_condition_number: jnp.ndarray
    boundary_inhomogeneous_norms: jnp.ndarray


class DiscreteMSSResult(NamedTuple):
    """Matrix-free multiple-shooting shadowing result."""

    value: jnp.ndarray
    gradient: jnp.ndarray
    normal_residual: jnp.ndarray
    cg_iterations: jnp.ndarray


class DiscreteLyapunovResult(NamedTuple):
    """Leading finite-time Lyapunov exponents of a discrete map."""

    exponents: jnp.ndarray
    accumulated_log_growth: jnp.ndarray
    orthogonality_residual: jnp.ndarray


def discrete_lyapunov_exponents(
    step: DiscreteStep,
    initial_state: jnp.ndarray,
    parameters: jnp.ndarray,
    *,
    vector_count: int,
    interval_count: int,
    steps_per_interval: int,
    warmup_intervals: int = 0,
    step_size: float = 1.0,
    random_seed: int = 0,
) -> DiscreteLyapunovResult:
    """Estimate leading exponents by matrix-free Benettin QR iteration.

    Only Jacobian-vector products of ``step`` are used.  ``step_size`` converts
    the result from growth per map application to growth per physical time.
    """

    state = jnp.asarray(initial_state)
    params = jnp.asarray(parameters)
    vectors = int(vector_count)
    intervals = int(interval_count)
    interval_steps = int(steps_per_interval)
    warmups = int(warmup_intervals)
    if state.ndim != 1 or params.ndim != 1:
        raise ValueError("initial_state and parameters must be one-dimensional")
    if jnp.issubdtype(state.dtype, jnp.complexfloating):
        raise ValueError("initial_state must use a real representation")
    if not jnp.issubdtype(state.dtype, jnp.inexact):
        raise ValueError("initial_state must have an inexact dtype")
    if not 1 <= vectors <= int(state.size):
        raise ValueError("vector_count must be between one and the state size")
    if intervals < 1 or interval_steps < 1 or warmups < 0 or float(step_size) <= 0.0:
        raise ValueError(
            "interval counts, interval length, and step size must be positive"
        )

    basis = jax.random.normal(
        jax.random.key(int(random_seed)), (int(state.size), vectors), dtype=state.dtype
    )
    basis, _ = jnp.linalg.qr(basis, mode="reduced")
    log_growth = jnp.zeros((vectors,), dtype=state.dtype)

    @jax.jit
    def advance_interval(
        current: jnp.ndarray, tangent_basis: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def advance(carry, _unused: None):
            local_state, local_basis = carry
            next_state, linearized_step = jax.linearize(
                lambda values: step(values, params), local_state
            )
            next_basis = jax.vmap(linearized_step, in_axes=1, out_axes=1)(local_basis)
            return (next_state, next_basis), None

        (next_state, propagated), _ = jax.lax.scan(
            advance, (current, tangent_basis), None, length=interval_steps
        )
        orthonormal, triangular = jnp.linalg.qr(propagated, mode="reduced")
        tiny = jnp.finfo(state.dtype).tiny
        growth = jnp.log(jnp.maximum(jnp.abs(jnp.diag(triangular)), tiny))
        return next_state, orthonormal, growth

    for _ in range(warmups):
        state, basis, _growth = advance_interval(state, basis)
    for _ in range(intervals):
        state, basis, growth = advance_interval(state, basis)
        log_growth = log_growth + growth
    elapsed = float(intervals * interval_steps) * float(step_size)
    identity = jnp.eye(vectors, dtype=state.dtype)
    orthogonality = jnp.linalg.norm(basis.T @ basis - identity)
    return DiscreteLyapunovResult(
        exponents=log_growth / elapsed,
        accumulated_log_growth=log_growth,
        orthogonality_residual=orthogonality,
    )


@partial(
    jax.jit,
    static_argnames=("step", "objective", "steps", "checkpoint"),
)
def integrate_discrete_observable(
    step: DiscreteStep,
    objective: InstantaneousObjective,
    initial_state: jnp.ndarray,
    parameters: jnp.ndarray,
    *,
    steps: int,
    checkpoint: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate a discrete map and retain one scalar objective per endpoint."""

    def scan_step(state: jnp.ndarray, _index: jnp.ndarray):
        next_state = step(state, parameters)
        return next_state, objective(next_state, parameters)

    return checkpointed_explicit_scan(
        scan_step,
        initial_state,
        jnp.arange(int(steps)),
        checkpoint=checkpoint,
    )


def discrete_window_value_and_grad(
    step: DiscreteStep,
    objective: InstantaneousObjective,
    initial_state: jnp.ndarray,
    parameters: jnp.ndarray,
    *,
    steps: int,
    tail_steps: int | None = None,
    checkpoint: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Differentiate a post-saturation finite-window objective average."""

    count = int(steps)
    tail = count if tail_steps is None else int(tail_steps)
    if count < 1 or not 1 <= tail <= count:
        raise ValueError("steps must be positive and tail_steps must lie within it")
    detached = jax.lax.stop_gradient(jnp.asarray(initial_state))

    def mean_objective(values: jnp.ndarray) -> jnp.ndarray:
        _, samples = integrate_discrete_observable(
            step,
            objective,
            detached,
            values,
            steps=count,
            checkpoint=checkpoint,
        )
        return jnp.mean(samples[-tail:])

    return jax.value_and_grad(mean_objective)(jnp.asarray(parameters))


def _conjugate_gradient(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    right_hand_side: jnp.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve an SPD system while retaining a device-side iteration count."""

    initial = jnp.zeros_like(right_hand_side)
    residual = right_hand_side - matvec(initial)
    residual_norm_squared = jnp.vdot(residual, residual).real
    target = (float(tolerance) ** 2) * jnp.maximum(
        residual_norm_squared, jnp.asarray(1.0, dtype=residual.dtype)
    )

    def condition(carry):
        _solution, _residual, _direction, norm_squared, iteration = carry
        return jnp.logical_and(norm_squared > target, iteration < max_iterations)

    def iteration(carry):
        solution, residual_now, direction, norm_squared, count = carry
        product = matvec(direction)
        alpha = norm_squared / jnp.maximum(
            jnp.vdot(direction, product).real,
            jnp.finfo(right_hand_side.dtype).tiny,
        )
        solution = solution + alpha * direction
        residual_next = residual_now - alpha * product
        next_norm_squared = jnp.vdot(residual_next, residual_next).real
        beta = next_norm_squared / jnp.maximum(
            norm_squared, jnp.finfo(right_hand_side.dtype).tiny
        )
        direction = residual_next + beta * direction
        return solution, residual_next, direction, next_norm_squared, count + 1

    solution, final_residual, _direction, _norm_squared, iterations = (
        jax.lax.while_loop(
            condition,
            iteration,
            (initial, residual, residual, residual_norm_squared, jnp.asarray(0)),
        )
    )
    relative_residual = jnp.linalg.norm(final_residual) / jnp.maximum(
        jnp.linalg.norm(right_hand_side),
        jnp.asarray(1.0, dtype=right_hand_side.dtype),
    )
    return solution, relative_residual, iterations


def discrete_multiple_shooting_shadowing(
    step: DiscreteStep,
    objective: InstantaneousObjective,
    initial_state: jnp.ndarray,
    parameters: jnp.ndarray,
    *,
    segment_count: int,
    steps_per_segment: int,
    regularization: float = 1.0e-8,
    cg_tolerance: float = 1.0e-8,
    cg_max_iterations: int = 200,
) -> DiscreteMSSResult:
    """Differentiate a discrete average with matrix-free multiple shooting.

    Segment-boundary tangents minimize their Euclidean norm subject to the
    linearized continuity equations.  The Schur system ``B B.T`` is solved by
    conjugate gradients using segment-map JVPs and VJPs; neither state nor
    segment Jacobians are assembled.  An adjoint Schur solve makes the cost
    independent of the number of parameters.

    The formulation is for a discrete map.  For a time-continuous system with
    an exactly neutral flow direction, ``regularization`` must remain positive;
    a production continuous-time shadowing claim additionally needs the usual
    neutral-direction projection/time-dilation term.
    """

    state = jnp.asarray(initial_state)
    params = jnp.asarray(parameters)
    segments = int(segment_count)
    segment_steps = int(steps_per_segment)
    if state.ndim != 1 or params.ndim != 1:
        raise ValueError("initial_state and parameters must be one-dimensional")
    if jnp.issubdtype(state.dtype, jnp.complexfloating):
        raise ValueError("initial_state must use a real representation")
    if not jnp.issubdtype(state.dtype, jnp.inexact):
        raise ValueError("initial_state must have an inexact dtype")
    if segments < 1 or segment_steps < 1:
        raise ValueError("segment_count and steps_per_segment must be positive")
    if float(regularization) <= 0.0:
        raise ValueError("regularization must be positive")
    if float(cg_tolerance) <= 0.0 or int(cg_max_iterations) < 1:
        raise ValueError("CG tolerance and maximum iterations must be positive")
    total_steps = segments * segment_steps

    def run_segment(
        start: jnp.ndarray, values: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        def advance(current: jnp.ndarray, _unused: None):
            sample = objective(current, values) / total_steps
            return step(current, values), sample

        endpoint, samples = jax.lax.scan(advance, start, None, length=segment_steps)
        return endpoint, jnp.sum(samples)

    run_segment_jit = jax.jit(run_segment)
    boundaries = [state]
    for _ in range(segments):
        endpoint, _value = run_segment_jit(boundaries[-1], params)
        boundaries.append(endpoint)
    starts = jnp.stack(boundaries[:-1])

    def batch_segments(
        start_values: jnp.ndarray, parameter_values: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jax.vmap(run_segment, in_axes=(0, None))(start_values, parameter_values)

    endpoints, segment_values = batch_segments(starts, params)
    del endpoints
    value = jnp.sum(segment_values)

    def batch_end(start_values: jnp.ndarray) -> jnp.ndarray:
        return batch_segments(start_values, params)[0]

    def apply_transition(tangents: jnp.ndarray) -> jnp.ndarray:
        return jax.jvp(batch_end, (starts,), (tangents,))[1]

    transpose_transition = jax.linear_transpose(
        apply_transition, jnp.zeros_like(starts)
    )

    def apply_transition_transpose(cotangents: jnp.ndarray) -> jnp.ndarray:
        return transpose_transition(cotangents)[0]

    def apply_continuity(boundary_vectors: jnp.ndarray) -> jnp.ndarray:
        return boundary_vectors[1:] - apply_transition(boundary_vectors[:-1])

    def apply_continuity_transpose(cotangents: jnp.ndarray) -> jnp.ndarray:
        transition_bars = apply_transition_transpose(cotangents)
        return jnp.concatenate(
            (
                -transition_bars[:1],
                cotangents[:-1] - transition_bars[1:],
                cotangents[-1:],
            ),
            axis=0,
        )

    state_cost_gradients = jax.grad(
        lambda start_values: jnp.sum(batch_segments(start_values, params)[1])
    )(starts)
    boundary_cost_gradient = jnp.concatenate(
        (state_cost_gradients, jnp.zeros_like(starts[:1])), axis=0
    )
    schur_rhs = apply_continuity(boundary_cost_gradient)

    def normal_operator(cotangents: jnp.ndarray) -> jnp.ndarray:
        return (
            apply_continuity(apply_continuity_transpose(cotangents))
            + float(regularization) * cotangents
        )

    multipliers, normal_residual, iterations = _conjugate_gradient(
        normal_operator,
        schur_rhs,
        tolerance=float(cg_tolerance),
        max_iterations=int(cg_max_iterations),
    )
    direct_gradient = jax.grad(
        lambda parameter_values: jnp.sum(batch_segments(starts, parameter_values)[1])
    )(params)
    _, endpoint_pullback = jax.vjp(
        lambda parameter_values: batch_segments(starts, parameter_values)[0],
        params,
    )
    gradient = direct_gradient + endpoint_pullback(multipliers)[0]
    return DiscreteMSSResult(
        value=value,
        gradient=gradient,
        normal_residual=normal_residual,
        cg_iterations=iterations,
    )


def _validate_nilsas_inputs(
    initial_state: jnp.ndarray,
    parameters: jnp.ndarray,
    *,
    segment_count: int,
    steps_per_segment: int,
    homogeneous_adjoint_count: int,
    regularization: float,
) -> tuple[jnp.ndarray, jnp.ndarray, int, int, int]:
    state = jnp.asarray(initial_state)
    params = jnp.asarray(parameters)
    segments = int(segment_count)
    segment_steps = int(steps_per_segment)
    adjoints = int(homogeneous_adjoint_count)
    if state.ndim != 1 or params.ndim != 1:
        raise ValueError("initial_state and parameters must be one-dimensional arrays")
    if jnp.issubdtype(state.dtype, jnp.complexfloating):
        raise ValueError("initial_state must use a real representation")
    if not jnp.issubdtype(state.dtype, jnp.inexact):
        raise ValueError("initial_state must have an inexact dtype")
    if segments < 1 or segment_steps < 1:
        raise ValueError("segment_count and steps_per_segment must be positive")
    if not 1 <= adjoints <= int(state.size):
        raise ValueError(
            "homogeneous_adjoint_count must be between one and the state size"
        )
    if float(regularization) < 0.0:
        raise ValueError("regularization must be non-negative")
    return state, params, segments, segment_steps, adjoints


def _nilsas_coefficients(
    covariances: tuple[jnp.ndarray, ...],
    linear_terms: tuple[jnp.ndarray, ...],
    interface_maps: tuple[jnp.ndarray, ...],
    interface_offsets: tuple[jnp.ndarray, ...],
    *,
    regularization: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve the reduced constrained least-squares problem from NILSAS."""

    segment_count = len(covariances)
    adjoint_count = int(covariances[0].shape[0])
    dtype = covariances[0].dtype
    size = segment_count * adjoint_count
    covariance = jnp.zeros((size, size), dtype=dtype)
    linear = jnp.concatenate(linear_terms)
    eye = jnp.eye(adjoint_count, dtype=dtype)
    for index, block in enumerate(covariances):
        start = index * adjoint_count
        covariance = covariance.at[
            start : start + adjoint_count, start : start + adjoint_count
        ].set(block + float(regularization) * eye)

    if segment_count == 1:
        coefficients = -jnp.linalg.solve(covariance, linear)
        return (
            coefficients.reshape((1, adjoint_count)),
            jnp.asarray(0.0, dtype=dtype),
            jnp.linalg.cond(covariance),
        )

    constraint_count = (segment_count - 1) * adjoint_count
    constraint = jnp.zeros((constraint_count, size), dtype=dtype)
    target = jnp.zeros((constraint_count,), dtype=dtype)
    for index in range(1, segment_count):
        row = (index - 1) * adjoint_count
        left = (index - 1) * adjoint_count
        right = index * adjoint_count
        constraint = constraint.at[
            row : row + adjoint_count, left : left + adjoint_count
        ].set(eye)
        constraint = constraint.at[
            row : row + adjoint_count, right : right + adjoint_count
        ].set(-interface_maps[index])
        target = target.at[row : row + adjoint_count].set(interface_offsets[index])

    zeros = jnp.zeros((constraint_count, constraint_count), dtype=dtype)
    kkt = jnp.block([[covariance, constraint.T], [constraint, zeros]])
    solution = jnp.linalg.solve(kkt, jnp.concatenate((-linear, target)))
    coefficients = solution[:size]
    residual = jnp.linalg.norm(constraint @ coefficients - target) / jnp.maximum(
        jnp.linalg.norm(target), jnp.asarray(1.0, dtype=dtype)
    )
    return (
        coefficients.reshape((segment_count, adjoint_count)),
        residual,
        jnp.linalg.cond(kkt),
    )


def discrete_nilsas(
    step: DiscreteStep,
    objective: InstantaneousObjective,
    initial_state: jnp.ndarray,
    parameters: jnp.ndarray,
    *,
    segment_count: int,
    steps_per_segment: int,
    homogeneous_adjoint_count: int,
    random_seed: int = 0,
    regularization: float = 1.0e-10,
) -> DiscreteNILSASResult:
    """Differentiate a long-time average with discrete NILSAS.

    This implements Appendix B of Ni & Talnikar (2019).  Segment-local primal
    trajectories are recomputed, so storage is one segment plus the segment
    boundaries.  State Jacobians are never assembled: homogeneous and
    inhomogeneous adjoints use only vector-Jacobian products of ``step``.

    ``initial_state`` must be a flat real array. Complex spectral states should
    be packed as concatenated real and imaginary parts so QR and the reduced
    least-squares coefficients remain real.
    """

    state, params, segments, segment_steps, adjoint_count = _validate_nilsas_inputs(
        initial_state,
        parameters,
        segment_count=segment_count,
        steps_per_segment=steps_per_segment,
        homogeneous_adjoint_count=homogeneous_adjoint_count,
        regularization=regularization,
    )
    dtype = state.dtype
    state_size = int(state.size)

    def segment_trajectory(start: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def advance(current: jnp.ndarray, _unused: None):
            return step(current, params), current

        endpoint, starts = jax.lax.scan(advance, start, None, length=segment_steps)
        return endpoint, jnp.concatenate((starts, endpoint[None, :]), axis=0)

    def segment_endpoint(
        carry: tuple[jnp.ndarray, jnp.ndarray], _unused: None
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
        current, total = carry
        value = jnp.asarray(objective(current, params), dtype=dtype)
        return (step(current, params), total + value), None

    run_segment = jax.jit(segment_trajectory)
    run_endpoint = jax.jit(
        lambda start: jax.lax.scan(
            segment_endpoint,
            (start, jnp.asarray(0.0, dtype=dtype)),
            None,
            length=segment_steps,
        )[0]
    )

    boundaries = [state]
    value_sum = jnp.asarray(0.0, dtype=dtype)
    for _ in range(segments):
        endpoint, segment_value = run_endpoint(boundaries[-1])
        boundaries.append(endpoint)
        value_sum = value_sum + segment_value

    def backward_segment(
        states: jnp.ndarray,
        terminal_basis: jnp.ndarray,
        terminal_particular: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        zeros_cov = jnp.zeros((adjoint_count, adjoint_count), dtype=dtype)
        zeros_linear = jnp.zeros((adjoint_count,), dtype=dtype)

        def reverse_step(carry, current):
            basis_next, particular_next, covariance, linear = carry
            covariance = covariance + basis_next.T @ basis_next
            linear = linear + basis_next.T @ particular_next
            _, pullback = jax.vjp(lambda value: step(value, params), current)
            adjoints = jnp.concatenate((basis_next, particular_next[:, None]), axis=1)
            previous = jax.vmap(
                lambda cotangent: pullback(cotangent)[0],
                in_axes=1,
                out_axes=1,
            )(adjoints)
            objective_state_gradient = jax.grad(lambda value: objective(value, params))(
                current
            )
            return (
                previous[:, :adjoint_count],
                previous[:, adjoint_count] + objective_state_gradient,
                covariance,
                linear,
            ), None

        result, _ = jax.lax.scan(
            reverse_step,
            (terminal_basis, terminal_particular, zeros_cov, zeros_linear),
            states[:-1][::-1],
        )
        return result

    backward = jax.jit(backward_segment)
    key = jax.random.key(int(random_seed))
    random_basis = jax.random.normal(key, (state_size, adjoint_count), dtype=dtype)
    terminal_basis, _ = jnp.linalg.qr(random_basis, mode="reduced")
    terminal_particular = jnp.zeros_like(state)
    boundary_bases: list[jnp.ndarray | None] = [None] * (segments + 1)
    boundary_particulars: list[jnp.ndarray | None] = [None] * (segments + 1)
    boundary_bases[segments] = terminal_basis
    boundary_particulars[segments] = terminal_particular
    covariances: list[jnp.ndarray | None] = [None] * segments
    linear_terms: list[jnp.ndarray | None] = [None] * segments
    interface_maps: list[jnp.ndarray | None] = [None] * segments
    interface_offsets: list[jnp.ndarray | None] = [None] * segments

    for index in range(segments - 1, -1, -1):
        _, states = run_segment(boundaries[index])
        basis_zero, particular_zero, covariance, linear = backward(
            states,
            boundary_bases[index + 1],
            boundary_particulars[index + 1],
        )
        basis, interface_map = jnp.linalg.qr(basis_zero, mode="reduced")
        interface_offset = basis.T @ particular_zero
        particular = particular_zero - basis @ interface_offset
        boundary_bases[index] = basis
        boundary_particulars[index] = particular
        covariances[index] = covariance
        linear_terms[index] = linear
        interface_maps[index] = interface_map
        interface_offsets[index] = interface_offset

    coefficients, constraint_residual, condition = _nilsas_coefficients(
        tuple(cast(jnp.ndarray, value) for value in covariances),
        tuple(cast(jnp.ndarray, value) for value in linear_terms),
        tuple(cast(jnp.ndarray, value) for value in interface_maps),
        tuple(cast(jnp.ndarray, value) for value in interface_offsets),
        regularization=regularization,
    )

    def segment_sensitivity(
        states: jnp.ndarray,
        terminal_basis: jnp.ndarray,
        terminal_particular: jnp.ndarray,
        coefficient: jnp.ndarray,
    ) -> jnp.ndarray:
        def reverse_step(carry, current):
            basis_next, particular_next, gradient = carry
            _, pullback = jax.vjp(step, current, params)
            adjoints = jnp.concatenate((basis_next, particular_next[:, None]), axis=1)
            state_bars, parameter_bars = jax.vmap(
                pullback,
                in_axes=1,
                out_axes=(1, -1),
            )(adjoints)
            weights = jnp.concatenate((coefficient, jnp.asarray([1.0], dtype=dtype)))
            gradient = gradient + jnp.tensordot(
                parameter_bars, weights, axes=((-1,), (0,))
            )
            gradient = gradient + jax.grad(lambda value: objective(current, value))(
                params
            )
            objective_state_gradient = jax.grad(lambda value: objective(value, params))(
                current
            )
            return (
                state_bars[:, :adjoint_count],
                state_bars[:, adjoint_count] + objective_state_gradient,
                gradient,
            ), None

        result, _ = jax.lax.scan(
            reverse_step,
            (terminal_basis, terminal_particular, jnp.zeros_like(params)),
            states[:-1][::-1],
        )
        return result[2]

    sensitivity = jax.jit(segment_sensitivity)
    gradient_sum = jnp.zeros_like(params)
    for index in range(segments - 1, -1, -1):
        _, states = run_segment(boundaries[index])
        gradient_sum = gradient_sum + sensitivity(
            states,
            boundary_bases[index + 1],
            boundary_particulars[index + 1],
            coefficients[index],
        )

    total_steps = segments * segment_steps
    boundary_norms = jnp.stack(
        [jnp.linalg.norm(value) for value in boundary_particulars]
    )
    return DiscreteNILSASResult(
        value=value_sum / total_steps,
        gradient=gradient_sum / total_steps,
        coefficients=coefficients,
        constraint_residual=constraint_residual,
        kkt_condition_number=condition,
        boundary_inhomogeneous_norms=boundary_norms,
    )


__all__ = [
    "DiscreteLyapunovResult",
    "DiscreteMSSResult",
    "DiscreteNILSASResult",
    "discrete_lyapunov_exponents",
    "discrete_multiple_shooting_shadowing",
    "discrete_nilsas",
    "discrete_window_value_and_grad",
    "integrate_discrete_observable",
]
