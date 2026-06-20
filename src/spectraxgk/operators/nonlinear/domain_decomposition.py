"""Local nonlinear domain prototype gates for parallelization validation.

These routines exercise bounded halo decomposition on a deterministic local
stencil.  They are diagnostic identity gates only; they do not implement a
production nonlinear domain-decomposed solver route or a speedup claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from spectraxgk.operators.nonlinear.parallel_contracts_domain import (
    _NONLINEAR_DOMAIN_CLAIM_SCOPE,
    _NONLINEAR_DOMAIN_GATE_NAME,
    _NONLINEAR_DOMAIN_TRANSPORT_CLAIM_SCOPE,
    _NONLINEAR_DOMAIN_TRANSPORT_GATE_NAME,
    NonlinearDomainDecompositionPlan,
    NonlinearDomainIdentityReport,
    NonlinearDomainTransportWindowReport,
    _nonlinear_domain_identity_blockers,
    _nonlinear_domain_plan_blockers,
)


@dataclass(frozen=True)
class _DomainTransportTraces:
    mass: tuple[float, ...]
    free_energy: tuple[float, ...]
    flux_proxy: tuple[float, ...]


@dataclass(frozen=True)
class _DomainTransportWindowSamples:
    serial_state: jax.Array
    decomposed_state: jax.Array
    serial_traces: _DomainTransportTraces
    decomposed_traces: _DomainTransportTraces


@dataclass(frozen=True)
class _DomainTransportTraceErrors:
    mass_abs: float
    mass_rel: float
    free_energy_abs: float
    free_energy_rel: float
    flux_proxy_abs: float
    flux_proxy_rel: float


def build_nonlinear_domain_decomposition_plan(
    state_shape: tuple[int, ...],
    *,
    axis: int = 0,
    num_domains: int = 2,
    halo: int = 1,
) -> NonlinearDomainDecompositionPlan:
    """Build a static chunk plan for the local state-domain identity prototype."""

    if not state_shape:
        raise ValueError("state_shape must contain at least one axis")
    ndim = len(state_shape)
    canonical_axis = axis % ndim
    domain_size = int(state_shape[canonical_axis])
    if domain_size <= 0:
        raise ValueError("decomposed axis must be non-empty")
    if int(num_domains) < 1:
        raise ValueError("num_domains must be at least one")
    if int(num_domains) > domain_size:
        raise ValueError("num_domains cannot exceed decomposed axis size")
    if int(halo) != 1:
        raise ValueError("this prototype only supports a one-cell halo")

    base, remainder = divmod(domain_size, int(num_domains))
    chunk_sizes = tuple(
        base + (1 if idx < remainder else 0) for idx in range(int(num_domains))
    )
    return NonlinearDomainDecompositionPlan(
        state_shape=tuple(int(size) for size in state_shape),
        axis=canonical_axis,
        chunk_sizes=chunk_sizes,
        halo=int(halo),
    )


def deterministic_nonlinear_domain_state(
    shape: tuple[int, ...] = (6, 4),
) -> jax.Array:
    """Return a small deterministic complex state for identity gates."""

    if not shape:
        raise ValueError("shape must contain at least one axis")
    size = 1
    for axis_size in shape:
        if int(axis_size) <= 0:
            raise ValueError("shape entries must be positive")
        size *= int(axis_size)
    values = jnp.arange(size, dtype=jnp.float32).reshape(
        tuple(int(item) for item in shape)
    )
    scaled = values / jnp.asarray(max(size - 1, 1), dtype=values.dtype)
    return scaled + 0.125j * jnp.cos(2.0 * jnp.pi * scaled)


def _prototype_nonlinear_step_axis0(state: jax.Array, dt: float) -> jax.Array:
    left = jnp.roll(state, shift=1, axis=0)
    right = jnp.roll(state, shift=-1, axis=0)
    centered_gradient = 0.5 * (right - left)
    laplacian = right - 2.0 * state + left
    nonlinear_damping = state * jnp.real(jnp.conj(state) * state)
    rhs = (
        0.03125 * laplacian - 0.015625 * nonlinear_damping + 0.0625j * centered_gradient
    )
    return state + jnp.asarray(dt, dtype=jnp.real(state).dtype) * rhs


def prototype_nonlinear_domain_serial_step(
    state: jax.Array,
    *,
    axis: int = 0,
    dt: float = 0.05,
) -> jax.Array:
    """Apply the serial local nonlinear prototype step along one state axis."""

    moved = jnp.moveaxis(state, axis % state.ndim, 0)
    stepped = _prototype_nonlinear_step_axis0(moved, dt)
    return jnp.moveaxis(stepped, 0, axis % state.ndim)


def prototype_nonlinear_domain_decomposed_step(
    state: jax.Array,
    plan: NonlinearDomainDecompositionPlan,
    *,
    dt: float = 0.05,
) -> jax.Array:
    """Apply the same local nonlinear step through static halo chunks."""

    plan_blockers = _nonlinear_domain_plan_blockers(plan)
    if plan_blockers:
        raise ValueError(
            "invalid nonlinear domain decomposition plan: " + ", ".join(plan_blockers)
        )
    if tuple(state.shape) != plan.state_shape:
        raise ValueError("state shape does not match decomposition plan")

    moved = jnp.moveaxis(state, plan.axis, 0)
    domain_size = plan.domain_size
    chunks = []
    for offset, chunk_size in zip(plan.offsets, plan.chunk_sizes, strict=True):
        indices = (
            jnp.arange(offset - plan.halo, offset + chunk_size + plan.halo)
            % domain_size
        )
        local_state = jnp.take(moved, indices, axis=0)
        local_step = _prototype_nonlinear_step_axis0(local_state, dt)
        chunks.append(
            jax.lax.dynamic_slice_in_dim(local_step, plan.halo, chunk_size, axis=0)
        )
    stepped = jnp.concatenate(chunks, axis=0)
    return jnp.moveaxis(stepped, 0, plan.axis)


def nonlinear_domain_identity_report(
    serial_state: jax.Array,
    decomposed_state: jax.Array,
    plan: NonlinearDomainDecompositionPlan,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> NonlinearDomainIdentityReport:
    """Compare decomposed and serial states and fail closed on any mismatch."""

    blocked_reasons = _nonlinear_domain_identity_blockers(
        serial_state,
        decomposed_state,
        plan,
    )
    plan_valid = not _nonlinear_domain_plan_blockers(plan)
    boundary_indices = plan.boundary_indices if plan_valid else ()
    if tuple(serial_state.shape) == tuple(decomposed_state.shape):
        abs_error = jnp.abs(decomposed_state - serial_state)
        scale = jnp.maximum(
            jnp.abs(serial_state),
            jnp.asarray(atol, dtype=jnp.real(abs_error).dtype),
        )
        rel_error = abs_error / scale
        max_abs_error = float(jnp.max(abs_error))
        max_rel_error = float(jnp.max(rel_error))
        if boundary_indices:
            boundary_selector = jnp.asarray(boundary_indices, dtype=jnp.int32)
            boundary_abs_error = jnp.take(abs_error, boundary_selector, axis=plan.axis)
            boundary_rel_error = jnp.take(rel_error, boundary_selector, axis=plan.axis)
            boundary_max_abs_error = float(jnp.max(boundary_abs_error))
            boundary_max_rel_error = float(jnp.max(boundary_rel_error))
        else:
            boundary_max_abs_error = 0.0
            boundary_max_rel_error = 0.0
    else:
        max_abs_error = float("inf")
        max_rel_error = float("inf")
        boundary_max_abs_error = float("inf")
        boundary_max_rel_error = float("inf")

    identity_passed = bool(
        not blocked_reasons
        and max_abs_error <= float(atol)
        and max_rel_error <= float(rtol)
        and boundary_max_abs_error <= float(atol)
        and boundary_max_rel_error <= float(rtol)
    )
    return NonlinearDomainIdentityReport(
        gate_name=_NONLINEAR_DOMAIN_GATE_NAME,
        plan=plan,
        atol=float(atol),
        rtol=float(rtol),
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        plan_valid=plan_valid,
        blocked_reasons=blocked_reasons,
        identity_passed=identity_passed,
        decomposed_path_enabled=identity_passed,
        claim_scope=_NONLINEAR_DOMAIN_CLAIM_SCOPE,
        boundary_max_abs_error=boundary_max_abs_error,
        boundary_max_rel_error=boundary_max_rel_error,
        boundary_indices=boundary_indices,
    )


def nonlinear_domain_parallel_identity_gate(
    state: jax.Array,
    plan: NonlinearDomainDecompositionPlan,
    *,
    dt: float = 0.05,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> tuple[jax.Array, NonlinearDomainIdentityReport]:
    """Return a fail-closed decomposed prototype step and its identity report."""

    serial = prototype_nonlinear_domain_serial_step(state, axis=plan.axis, dt=dt)
    decomposed = prototype_nonlinear_domain_decomposed_step(state, plan, dt=dt)
    report = nonlinear_domain_identity_report(
        serial, decomposed, plan, atol=atol, rtol=rtol
    )
    gated_state = decomposed if report.decomposed_path_enabled else serial
    return gated_state, report


def _nonlinear_domain_transport_observables(
    state: jax.Array,
    plan: NonlinearDomainDecompositionPlan,
) -> tuple[float, float, float]:
    """Return scalar transport-window proxies for a domain-decomposed state."""

    real_state = jnp.real(state)
    mass = float(jnp.sum(real_state))
    free_energy = float(jnp.sum(jnp.abs(state) ** 2))
    axis_gradient = jnp.roll(real_state, shift=-1, axis=plan.axis) - real_state
    boundary_indices = plan.boundary_indices
    if boundary_indices:
        selector = jnp.asarray(boundary_indices, dtype=jnp.int32)
        boundary_gradient = jnp.take(axis_gradient, selector, axis=plan.axis)
    else:
        boundary_gradient = axis_gradient
    flux_proxy = float(jnp.mean(jnp.abs(boundary_gradient)))
    return mass, free_energy, flux_proxy


def _append_transport_observables(
    traces: dict[str, list[float]],
    state: jax.Array,
    plan: NonlinearDomainDecompositionPlan,
) -> None:
    mass, free_energy, flux_proxy = _nonlinear_domain_transport_observables(
        state,
        plan,
    )
    traces["mass"].append(mass)
    traces["free_energy"].append(free_energy)
    traces["flux_proxy"].append(flux_proxy)


def _relative_trace_error(
    reference: tuple[float, ...],
    candidate: tuple[float, ...],
    *,
    floor: float,
) -> tuple[float, float]:
    if len(reference) != len(candidate):
        return float("inf"), float("inf")
    reference_arr = jnp.asarray(reference, dtype=jnp.float32)
    candidate_arr = jnp.asarray(candidate, dtype=jnp.float32)
    abs_error = jnp.abs(candidate_arr - reference_arr)
    max_abs = float(jnp.max(abs_error))
    scale = jnp.maximum(
        jnp.abs(reference_arr), jnp.asarray(floor, dtype=reference_arr.dtype)
    )
    max_rel = float(jnp.max(abs_error / scale))
    return max_abs, max_rel


def _trace_drift(trace: tuple[float, ...]) -> float:
    if len(trace) < 2:
        return 0.0
    return float(trace[-1] - trace[0])


def _new_domain_transport_traces() -> dict[str, list[float]]:
    return {
        "mass": [],
        "free_energy": [],
        "flux_proxy": [],
    }


def _freeze_domain_transport_traces(
    traces: dict[str, list[float]],
) -> _DomainTransportTraces:
    return _DomainTransportTraces(
        mass=tuple(traces["mass"]),
        free_energy=tuple(traces["free_energy"]),
        flux_proxy=tuple(traces["flux_proxy"]),
    )


def _blocked_domain_transport_window_report(
    plan: NonlinearDomainDecompositionPlan,
    *,
    steps: int,
    dt: float,
    atol: float,
    rtol: float,
    plan_valid: bool,
    blocked_reasons: tuple[str, ...],
) -> NonlinearDomainTransportWindowReport:
    return NonlinearDomainTransportWindowReport(
        gate_name=_NONLINEAR_DOMAIN_TRANSPORT_GATE_NAME,
        plan=plan,
        steps=int(steps),
        dt=float(dt),
        atol=float(atol),
        rtol=float(rtol),
        max_abs_state_error=float("inf"),
        max_rel_state_error=float("inf"),
        max_abs_boundary_error=float("inf"),
        max_rel_boundary_error=float("inf"),
        mass_trace_max_abs_error=float("inf"),
        mass_trace_max_rel_error=float("inf"),
        free_energy_trace_max_abs_error=float("inf"),
        free_energy_trace_max_rel_error=float("inf"),
        flux_proxy_trace_max_abs_error=float("inf"),
        flux_proxy_trace_max_rel_error=float("inf"),
        serial_mass_drift=float("inf"),
        decomposed_mass_drift=float("inf"),
        serial_free_energy_drift=float("inf"),
        decomposed_free_energy_drift=float("inf"),
        plan_valid=plan_valid,
        blocked_reasons=blocked_reasons,
        identity_passed=False,
        decomposed_path_enabled=False,
        claim_scope=_NONLINEAR_DOMAIN_TRANSPORT_CLAIM_SCOPE,
        boundary_indices=plan.boundary_indices if plan_valid else (),
    )


def _run_domain_transport_window(
    state: jax.Array,
    plan: NonlinearDomainDecompositionPlan,
    *,
    steps: int,
    dt: float,
) -> _DomainTransportWindowSamples:
    serial_state = state
    decomposed_state = state
    serial_trace_lists = _new_domain_transport_traces()
    decomposed_trace_lists = _new_domain_transport_traces()
    _append_transport_observables(serial_trace_lists, serial_state, plan)
    _append_transport_observables(decomposed_trace_lists, decomposed_state, plan)
    for _ in range(int(steps)):
        serial_state = prototype_nonlinear_domain_serial_step(
            serial_state,
            axis=plan.axis,
            dt=dt,
        )
        decomposed_state = prototype_nonlinear_domain_decomposed_step(
            decomposed_state,
            plan,
            dt=dt,
        )
        _append_transport_observables(serial_trace_lists, serial_state, plan)
        _append_transport_observables(decomposed_trace_lists, decomposed_state, plan)
    return _DomainTransportWindowSamples(
        serial_state=serial_state,
        decomposed_state=decomposed_state,
        serial_traces=_freeze_domain_transport_traces(serial_trace_lists),
        decomposed_traces=_freeze_domain_transport_traces(decomposed_trace_lists),
    )


def _domain_transport_trace_errors(
    serial: _DomainTransportTraces,
    decomposed: _DomainTransportTraces,
    *,
    floor: float,
) -> _DomainTransportTraceErrors:
    mass_abs, mass_rel = _relative_trace_error(serial.mass, decomposed.mass, floor=floor)
    free_energy_abs, free_energy_rel = _relative_trace_error(
        serial.free_energy,
        decomposed.free_energy,
        floor=floor,
    )
    flux_proxy_abs, flux_proxy_rel = _relative_trace_error(
        serial.flux_proxy,
        decomposed.flux_proxy,
        floor=floor,
    )
    return _DomainTransportTraceErrors(
        mass_abs=mass_abs,
        mass_rel=mass_rel,
        free_energy_abs=free_energy_abs,
        free_energy_rel=free_energy_rel,
        flux_proxy_abs=flux_proxy_abs,
        flux_proxy_rel=flux_proxy_rel,
    )


def _domain_transport_identity_passed(
    state_report: NonlinearDomainIdentityReport,
    errors: _DomainTransportTraceErrors,
    *,
    atol: float,
    rtol: float,
) -> bool:
    return bool(
        state_report.identity_passed
        and errors.mass_abs <= float(atol)
        and errors.mass_rel <= float(rtol)
        and errors.free_energy_abs <= float(atol)
        and errors.free_energy_rel <= float(rtol)
        and errors.flux_proxy_abs <= float(atol)
        and errors.flux_proxy_rel <= float(rtol)
    )


def _domain_transport_window_report(
    plan: NonlinearDomainDecompositionPlan,
    samples: _DomainTransportWindowSamples,
    state_report: NonlinearDomainIdentityReport,
    errors: _DomainTransportTraceErrors,
    *,
    steps: int,
    dt: float,
    atol: float,
    rtol: float,
    plan_valid: bool,
    blocked_reasons: tuple[str, ...],
) -> NonlinearDomainTransportWindowReport:
    identity_passed = _domain_transport_identity_passed(
        state_report,
        errors,
        atol=atol,
        rtol=rtol,
    )
    serial = samples.serial_traces
    decomposed = samples.decomposed_traces
    return NonlinearDomainTransportWindowReport(
        gate_name=_NONLINEAR_DOMAIN_TRANSPORT_GATE_NAME,
        plan=plan,
        steps=int(steps),
        dt=float(dt),
        atol=float(atol),
        rtol=float(rtol),
        max_abs_state_error=state_report.max_abs_error,
        max_rel_state_error=state_report.max_rel_error,
        max_abs_boundary_error=state_report.boundary_max_abs_error,
        max_rel_boundary_error=state_report.boundary_max_rel_error,
        mass_trace_max_abs_error=errors.mass_abs,
        mass_trace_max_rel_error=errors.mass_rel,
        free_energy_trace_max_abs_error=errors.free_energy_abs,
        free_energy_trace_max_rel_error=errors.free_energy_rel,
        flux_proxy_trace_max_abs_error=errors.flux_proxy_abs,
        flux_proxy_trace_max_rel_error=errors.flux_proxy_rel,
        serial_mass_drift=_trace_drift(serial.mass),
        decomposed_mass_drift=_trace_drift(decomposed.mass),
        serial_free_energy_drift=_trace_drift(serial.free_energy),
        decomposed_free_energy_drift=_trace_drift(decomposed.free_energy),
        plan_valid=plan_valid,
        blocked_reasons=blocked_reasons,
        identity_passed=identity_passed,
        decomposed_path_enabled=identity_passed,
        claim_scope=_NONLINEAR_DOMAIN_TRANSPORT_CLAIM_SCOPE,
        boundary_indices=state_report.boundary_indices,
        serial_mass_trace=serial.mass,
        decomposed_mass_trace=decomposed.mass,
        serial_free_energy_trace=serial.free_energy,
        decomposed_free_energy_trace=decomposed.free_energy,
        serial_flux_proxy_trace=serial.flux_proxy,
        decomposed_flux_proxy_trace=decomposed.flux_proxy,
    )


def nonlinear_domain_transport_window_identity_gate(
    state: jax.Array,
    plan: NonlinearDomainDecompositionPlan,
    *,
    dt: float = 0.025,
    steps: int = 4,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> NonlinearDomainTransportWindowReport:
    """Validate a multi-step serial-vs-decomposed transport window.

    The gate is deliberately stricter than a final-state check: it compares
    state identity, decomposed-boundary identity, and per-step scalar traces for
    mass, free-energy proxy, and boundary-flux proxy. The scalar drifts are
    compared between serial and decomposed paths; they are not claimed to be
    conserved by this damped diagnostic stencil.
    """

    if int(steps) < 1:
        raise ValueError("steps must be at least one")

    plan_blockers = _nonlinear_domain_plan_blockers(plan)
    plan_valid = not plan_blockers
    state_shape = tuple(int(size) for size in state.shape)
    blocked_reasons = list(plan_blockers)
    if state_shape != plan.state_shape:
        blocked_reasons.append("state_shape_does_not_match_plan")

    if blocked_reasons:
        return _blocked_domain_transport_window_report(
            plan,
            steps=steps,
            dt=dt,
            atol=atol,
            rtol=rtol,
            plan_valid=plan_valid,
            blocked_reasons=tuple(blocked_reasons),
        )

    samples = _run_domain_transport_window(state, plan, steps=steps, dt=dt)
    state_report = nonlinear_domain_identity_report(
        samples.serial_state,
        samples.decomposed_state,
        plan,
        atol=atol,
        rtol=rtol,
    )
    errors = _domain_transport_trace_errors(
        samples.serial_traces,
        samples.decomposed_traces,
        floor=atol,
    )
    return _domain_transport_window_report(
        plan=plan,
        samples=samples,
        state_report=state_report,
        errors=errors,
        steps=steps,
        dt=dt,
        atol=atol,
        rtol=rtol,
        plan_valid=plan_valid,
        blocked_reasons=tuple(blocked_reasons),
    )


__all__ = [
    "build_nonlinear_domain_decomposition_plan",
    "deterministic_nonlinear_domain_state",
    "nonlinear_domain_identity_report",
    "nonlinear_domain_parallel_identity_gate",
    "nonlinear_domain_transport_window_identity_gate",
    "prototype_nonlinear_domain_decomposed_step",
    "prototype_nonlinear_domain_serial_step",
]
