"""Planning contract and identity gates for nonlinear parallelization strategies.

The production-facing paths in this module remain policy metadata. The small
state-domain utilities below are conservative diagnostic prototypes: they only
enable a decomposed nonlinear state update after direct numerical identity
against the serial reference on the same deterministic operation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp


NonlinearParallelStrategyName = Literal[
    "independent_ky_scan",
    "uq_ensemble",
    "whole_state_kx_ky",
    "velocity_species_hermite",
    "fft_axis_domain",
]
ParallelReadiness = Literal["release_ready", "diagnostic", "blocked"]


@dataclass(frozen=True)
class NonlinearDomainDecompositionPlan:
    """Static decomposition plan for a local nonlinear state-domain prototype."""

    state_shape: tuple[int, ...]
    axis: int
    chunk_sizes: tuple[int, ...]
    halo: int = 1

    @property
    def num_domains(self) -> int:
        """Return the number of state-domain chunks."""

        return len(self.chunk_sizes)

    @property
    def domain_size(self) -> int:
        """Return the global size of the decomposed axis."""

        return self.state_shape[self.axis]

    @property
    def offsets(self) -> tuple[int, ...]:
        """Return chunk start offsets along the decomposed axis."""

        offsets: list[int] = []
        start = 0
        for size in self.chunk_sizes:
            offsets.append(start)
            start += size
        return tuple(offsets)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the decomposition plan."""

        return asdict(self)


@dataclass(frozen=True)
class NonlinearDomainIdentityReport:
    """Numerical identity report for a decomposed nonlinear prototype step."""

    plan: NonlinearDomainDecompositionPlan
    atol: float
    rtol: float
    max_abs_error: float
    max_rel_error: float
    identity_passed: bool
    decomposed_path_enabled: bool
    claim_scope: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the identity report."""

        data = asdict(self)
        data["plan"] = self.plan.to_dict()
        return data


@dataclass(frozen=True)
class NonlinearParallelStrategy:
    """Readiness contract for one nonlinear parallelization candidate."""

    name: NonlinearParallelStrategyName
    readiness: ParallelReadiness
    independent_work: bool
    changes_solver_layout: bool
    identity_gates: tuple[str, ...]
    physics_gates: tuple[str, ...]
    profiler_gates: tuple[str, ...]
    notes: str

    @property
    def release_ready(self) -> bool:
        """Whether this strategy is allowed for production-facing execution."""

        return self.readiness == "release_ready"

    @property
    def diagnostic_only(self) -> bool:
        """Whether this strategy is limited to correctness/profiling artifacts."""

        return self.readiness == "diagnostic"

    @property
    def blocked(self) -> bool:
        """Whether this strategy is unavailable until required gates exist."""

        return self.readiness == "blocked"

    @property
    def required_gates(self) -> tuple[str, ...]:
        """All identity, physics, and profiler gates required by this policy."""

        return self.identity_gates + self.physics_gates + self.profiler_gates

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the contract."""

        return asdict(self)


_STRATEGIES: tuple[NonlinearParallelStrategy, ...] = (
    NonlinearParallelStrategy(
        name="independent_ky_scan",
        readiness="release_ready",
        independent_work=True,
        changes_solver_layout=False,
        identity_gates=(
            "serial_vs_parallel_ky_scan_growth_rate_identity",
            "serial_vs_parallel_ky_scan_eigenfunction_norm_identity",
        ),
        physics_gates=("linear_ky_scan_reference_physics_gate",),
        profiler_gates=("bounded_independent_ky_scan_scaling_profile",),
        notes="Independent ky tasks preserve solver state layout and ordering.",
    ),
    NonlinearParallelStrategy(
        name="uq_ensemble",
        readiness="release_ready",
        independent_work=True,
        changes_solver_layout=False,
        identity_gates=(
            "serial_vs_parallel_uq_observable_identity",
            "serial_vs_parallel_uq_covariance_identity",
        ),
        physics_gates=("uq_member_physics_gate",),
        profiler_gates=("bounded_uq_ensemble_scaling_profile",),
        notes="Independent UQ members can use batch/thread/process scheduling.",
    ),
    NonlinearParallelStrategy(
        name="whole_state_kx_ky",
        readiness="diagnostic",
        independent_work=False,
        changes_solver_layout=True,
        identity_gates=(
            "whole_state_kx_ky_final_state_identity",
            "whole_state_kx_ky_final_field_identity",
            "whole_state_kx_ky_final_rhs_identity",
        ),
        physics_gates=("nonlinear_window_diagnostic_identity_gate",),
        profiler_gates=("matched_cpu_gpu_whole_state_scaling_profile",),
        notes="Current pjit whole-state kx/ky sharding is a correctness/profiler artifact, not a speedup claim.",
    ),
    NonlinearParallelStrategy(
        name="velocity_species_hermite",
        readiness="diagnostic",
        independent_work=False,
        changes_solver_layout=True,
        identity_gates=(
            "hermite_ghost_exchange_identity",
            "velocity_field_reduce_broadcast_identity",
            "velocity_species_linear_rhs_identity",
        ),
        physics_gates=(
            "species_moment_reduction_physics_gate",
            "nonlinear_fixed_step_identity_gate",
        ),
        profiler_gates=("matched_velocity_species_hermite_scaling_profile",),
        notes="GX-inspired production candidate; promotion requires end-to-end nonlinear identity gates.",
    ),
    NonlinearParallelStrategy(
        name="fft_axis_domain",
        readiness="blocked",
        independent_work=False,
        changes_solver_layout=True,
        identity_gates=(
            "distributed_fft_forward_inverse_identity",
            "distributed_fft_nonlinear_bracket_identity",
            "distributed_fft_field_solve_identity",
        ),
        physics_gates=("fft_axis_nonlinear_window_physics_gate",),
        profiler_gates=("distributed_fft_scaling_profile",),
        notes="Blocked until distributed FFT identity gates exist.",
    ),
)

_STRATEGY_BY_NAME: dict[NonlinearParallelStrategyName, NonlinearParallelStrategy] = {
    strategy.name: strategy for strategy in _STRATEGIES
}


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
    chunk_sizes = tuple(base + (1 if idx < remainder else 0) for idx in range(int(num_domains)))
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
    values = jnp.arange(size, dtype=jnp.float32).reshape(tuple(int(item) for item in shape))
    scaled = values / jnp.asarray(max(size - 1, 1), dtype=values.dtype)
    return scaled + 0.125j * jnp.cos(2.0 * jnp.pi * scaled)


def _prototype_nonlinear_step_axis0(state: jax.Array, dt: float) -> jax.Array:
    left = jnp.roll(state, shift=1, axis=0)
    right = jnp.roll(state, shift=-1, axis=0)
    centered_gradient = 0.5 * (right - left)
    laplacian = right - 2.0 * state + left
    nonlinear_damping = state * jnp.real(jnp.conj(state) * state)
    rhs = 0.03125 * laplacian - 0.015625 * nonlinear_damping + 0.0625j * centered_gradient
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

    if tuple(state.shape) != plan.state_shape:
        raise ValueError("state shape does not match decomposition plan")

    moved = jnp.moveaxis(state, plan.axis, 0)
    domain_size = plan.domain_size
    chunks = []
    for offset, chunk_size in zip(plan.offsets, plan.chunk_sizes, strict=True):
        indices = (jnp.arange(offset - plan.halo, offset + chunk_size + plan.halo) % domain_size)
        local_state = jnp.take(moved, indices, axis=0)
        local_step = _prototype_nonlinear_step_axis0(local_state, dt)
        chunks.append(jax.lax.dynamic_slice_in_dim(local_step, plan.halo, chunk_size, axis=0))
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

    abs_error = jnp.abs(decomposed_state - serial_state)
    scale = jnp.maximum(jnp.abs(serial_state), jnp.asarray(atol, dtype=jnp.real(abs_error).dtype))
    rel_error = abs_error / scale
    max_abs_error = float(jnp.max(abs_error))
    max_rel_error = float(jnp.max(rel_error))
    identity_passed = bool(max_abs_error <= float(atol) and max_rel_error <= float(rtol))
    return NonlinearDomainIdentityReport(
        plan=plan,
        atol=float(atol),
        rtol=float(rtol),
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        identity_passed=identity_passed,
        decomposed_path_enabled=identity_passed,
        claim_scope=(
            "diagnostic nonlinear state-domain identity gate only; "
            "no production routing or speedup claim"
        ),
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
    report = nonlinear_domain_identity_report(serial, decomposed, plan, atol=atol, rtol=rtol)
    gated_state = decomposed if report.decomposed_path_enabled else serial
    return gated_state, report


def nonlinear_parallel_strategies() -> tuple[NonlinearParallelStrategy, ...]:
    """Return all nonlinear parallelization strategy contracts."""

    return _STRATEGIES


def nonlinear_parallel_strategy(
    name: NonlinearParallelStrategyName,
) -> NonlinearParallelStrategy:
    """Return the contract for a named nonlinear parallelization strategy."""

    try:
        return _STRATEGY_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown nonlinear parallelization strategy {name!r}"
        ) from exc


def classify_nonlinear_parallel_strategy(
    name: NonlinearParallelStrategyName,
) -> ParallelReadiness:
    """Return the release-readiness classification for a strategy."""

    return nonlinear_parallel_strategy(name).readiness


def release_ready_nonlinear_parallel_strategies() -> tuple[
    NonlinearParallelStrategy, ...
]:
    """Return production-facing strategies that do not alter solver layout."""

    return tuple(strategy for strategy in _STRATEGIES if strategy.release_ready)


__all__ = [
    "NonlinearDomainDecompositionPlan",
    "NonlinearDomainIdentityReport",
    "NonlinearParallelStrategy",
    "NonlinearParallelStrategyName",
    "ParallelReadiness",
    "build_nonlinear_domain_decomposition_plan",
    "classify_nonlinear_parallel_strategy",
    "deterministic_nonlinear_domain_state",
    "nonlinear_domain_identity_report",
    "nonlinear_domain_parallel_identity_gate",
    "nonlinear_parallel_strategies",
    "nonlinear_parallel_strategy",
    "prototype_nonlinear_domain_decomposed_step",
    "prototype_nonlinear_domain_serial_step",
    "release_ready_nonlinear_parallel_strategies",
]
