"""Planning contract and identity gates for nonlinear parallelization strategies.

The production-facing paths in this module remain policy metadata. The small
state-domain utilities below are conservative diagnostic prototypes: they only
enable a decomposed nonlinear state update after direct numerical identity
against the serial reference on the same deterministic operation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np


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

    @property
    def chunk_bounds(self) -> tuple[tuple[int, int], ...]:
        """Return half-open ``(start, stop)`` bounds for owned chunk cells."""

        return tuple(
            (offset, offset + size)
            for offset, size in zip(self.offsets, self.chunk_sizes, strict=True)
        )

    @property
    def boundary_indices(self) -> tuple[int, ...]:
        """Return global cells that touch a decomposed halo interface."""

        if (
            not self.state_shape
            or not (0 <= int(self.axis) < len(self.state_shape))
            or len(self.chunk_sizes) <= 1
        ):
            return ()
        domain_size = int(self.state_shape[int(self.axis)])
        if domain_size <= 0:
            return ()

        indices: set[int] = set()
        for offset in self.offsets:
            indices.add((offset - 1) % domain_size)
            indices.add(offset % domain_size)
        return tuple(sorted(indices))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the decomposition plan."""

        return asdict(self)

    def decomposition_metadata(self) -> dict[str, Any]:
        """Return derived metadata for diagnostic decomposition artifacts."""

        return {
            **self.to_dict(),
            "num_domains": self.num_domains,
            "domain_size": self.domain_size,
            "offsets": self.offsets,
            "chunk_bounds": self.chunk_bounds,
            "boundary_indices": self.boundary_indices,
        }


@dataclass(frozen=True)
class NonlinearDomainIdentityReport:
    """Numerical identity report for a decomposed nonlinear prototype step."""

    gate_name: str
    plan: NonlinearDomainDecompositionPlan
    atol: float
    rtol: float
    max_abs_error: float
    max_rel_error: float
    plan_valid: bool
    blocked_reasons: tuple[str, ...]
    identity_passed: bool
    decomposed_path_enabled: bool
    claim_scope: str
    boundary_max_abs_error: float = 0.0
    boundary_max_rel_error: float = 0.0
    boundary_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the identity report."""

        data = asdict(self)
        data["plan"] = self.plan.to_dict()
        return data


@dataclass(frozen=True)
class NonlinearDomainTransportWindowReport:
    """Transport-window identity report for the nonlinear domain prototype."""

    gate_name: str
    plan: NonlinearDomainDecompositionPlan
    steps: int
    dt: float
    atol: float
    rtol: float
    max_abs_state_error: float
    max_rel_state_error: float
    max_abs_boundary_error: float
    max_rel_boundary_error: float
    mass_trace_max_abs_error: float
    mass_trace_max_rel_error: float
    free_energy_trace_max_abs_error: float
    free_energy_trace_max_rel_error: float
    flux_proxy_trace_max_abs_error: float
    flux_proxy_trace_max_rel_error: float
    serial_mass_drift: float
    decomposed_mass_drift: float
    serial_free_energy_drift: float
    decomposed_free_energy_drift: float
    plan_valid: bool
    blocked_reasons: tuple[str, ...]
    identity_passed: bool
    decomposed_path_enabled: bool
    claim_scope: str
    boundary_indices: tuple[int, ...] = ()
    serial_mass_trace: tuple[float, ...] = ()
    decomposed_mass_trace: tuple[float, ...] = ()
    serial_free_energy_trace: tuple[float, ...] = ()
    decomposed_free_energy_trace: tuple[float, ...] = ()
    serial_flux_proxy_trace: tuple[float, ...] = ()
    decomposed_flux_proxy_trace: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the transport-window report."""

        data = asdict(self)
        data["plan"] = self.plan.to_dict()
        return data


_NONLINEAR_DOMAIN_GATE_NAME = "nonlinear_domain_local_stencil_identity"
_NONLINEAR_DOMAIN_TRANSPORT_GATE_NAME = "nonlinear_domain_transport_window_identity"
_NONLINEAR_DOMAIN_CLAIM_SCOPE = (
    "diagnostic nonlinear state-domain identity gate only; "
    "bounded local-stencil prototype with no production routing or speedup claim"
)
_NONLINEAR_DOMAIN_TRANSPORT_CLAIM_SCOPE = (
    "diagnostic nonlinear state-domain transport-window identity gate only; "
    "serial-vs-halo-decomposed state, boundary, mass, free-energy, and flux-proxy "
    "traces with no production routing or speedup claim"
)


def _nonlinear_domain_plan_blockers(
    plan: NonlinearDomainDecompositionPlan,
) -> tuple[str, ...]:
    blockers: list[str] = []

    if not plan.state_shape:
        blockers.append("state_shape_empty")
        axis_is_valid = False
    else:
        axis_is_valid = 0 <= int(plan.axis) < len(plan.state_shape)

    if any(int(size) <= 0 for size in plan.state_shape):
        blockers.append("state_shape_non_positive")
    if not axis_is_valid:
        blockers.append("axis_not_canonical")
    if int(plan.halo) != 1:
        blockers.append("unsupported_halo")
    if not plan.chunk_sizes:
        blockers.append("chunk_sizes_empty")
    if any(int(size) <= 0 for size in plan.chunk_sizes):
        blockers.append("chunk_size_non_positive")
    if axis_is_valid and plan.chunk_sizes:
        domain_size = int(plan.state_shape[int(plan.axis)])
        if sum(int(size) for size in plan.chunk_sizes) != domain_size:
            blockers.append("chunk_sizes_do_not_cover_axis")

    return tuple(blockers)


def _nonlinear_domain_identity_blockers(
    serial_state: jax.Array,
    decomposed_state: jax.Array,
    plan: NonlinearDomainDecompositionPlan,
) -> tuple[str, ...]:
    blockers = list(_nonlinear_domain_plan_blockers(plan))
    serial_shape = tuple(int(size) for size in serial_state.shape)
    decomposed_shape = tuple(int(size) for size in decomposed_state.shape)

    if serial_shape != plan.state_shape:
        blockers.append("serial_shape_does_not_match_plan")
    if decomposed_shape != serial_shape:
        blockers.append("decomposed_shape_does_not_match_serial")

    return tuple(blockers)


@dataclass(frozen=True)
class NonlinearSpectralCommunicationReport:
    """Numerical identity report for nonlinear spectral communication layouts."""

    state_shape: tuple[int, int, int, int, int]
    y_chunks: tuple[int, ...]
    x_chunks: tuple[int, ...]
    atol: float
    rtol: float
    fft_max_abs_error: float
    fft_max_rel_error: float
    bracket_max_abs_error: float
    bracket_max_rel_error: float
    field_max_abs_error: float
    field_max_rel_error: float
    identity_passed: bool
    decomposed_path_enabled: bool
    claim_scope: str
    blocked_reasons: tuple[str, ...] = ()
    y_offsets: tuple[int, ...] = ()
    x_offsets: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the communication report."""

        return asdict(self)


@dataclass(frozen=True)
class NonlinearSpectralRHSIdentityReport:
    """Numerical identity report for logical-shard nonlinear spectral RHS."""

    state_shape: tuple[int, int, int, int, int]
    y_chunks: tuple[int, ...]
    x_chunks: tuple[int, ...]
    y_offsets: tuple[int, ...]
    x_offsets: tuple[int, ...]
    tile_bounds: tuple[tuple[int, int, int, int], ...]
    atol: float
    rtol: float
    reconstruction_max_abs_error: float
    reconstruction_max_rel_error: float
    field_max_abs_error: float
    field_max_rel_error: float
    bracket_max_abs_error: float
    bracket_max_rel_error: float
    rhs_max_abs_error: float
    rhs_max_rel_error: float
    identity_passed: bool
    decomposed_path_enabled: bool
    claim_scope: str
    blocked_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the RHS identity report."""

        return asdict(self)


@dataclass(frozen=True)
class NonlinearSpectralIntegratorIdentityReport:
    """Multi-step identity report for logical-shard nonlinear spectral routing."""

    state_shape: tuple[int, int, int, int, int]
    y_chunks: tuple[int, ...]
    x_chunks: tuple[int, ...]
    y_offsets: tuple[int, ...]
    x_offsets: tuple[int, ...]
    tile_bounds: tuple[tuple[int, int, int, int], ...]
    steps: int
    dt: float
    atol: float
    rtol: float
    final_state_max_abs_error: float
    final_state_max_rel_error: float
    free_energy_trace_max_abs_error: float
    free_energy_trace_max_rel_error: float
    field_energy_trace_max_abs_error: float
    field_energy_trace_max_rel_error: float
    flux_proxy_trace_max_abs_error: float
    flux_proxy_trace_max_rel_error: float
    serial_free_energy_drift: float
    logical_free_energy_drift: float
    identity_passed: bool
    decomposed_path_enabled: bool
    claim_scope: str
    blocked_reasons: tuple[str, ...] = ()
    serial_free_energy_trace: tuple[float, ...] = ()
    logical_free_energy_trace: tuple[float, ...] = ()
    serial_field_energy_trace: tuple[float, ...] = ()
    logical_field_energy_trace: tuple[float, ...] = ()
    serial_flux_proxy_trace: tuple[float, ...] = ()
    logical_flux_proxy_trace: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the integrator report."""

        return asdict(self)


@dataclass(frozen=True)
class NonlinearSpectralDomainWorkModel:
    """Communication/work model for the logical nonlinear spectral-domain route."""

    state_shape: tuple[int, int, int, int, int]
    y_chunks: tuple[int, ...]
    x_chunks: tuple[int, ...]
    y_offsets: tuple[int, ...]
    x_offsets: tuple[int, ...]
    tile_bounds: tuple[tuple[int, int, int, int], ...]
    num_tiles: int
    state_elements: int
    field_elements: int
    owned_state_elements_per_step: int
    state_allgather_elements_per_step: int
    bracket_allgather_elements_per_step: int
    field_broadcast_elements_per_step: int
    total_communication_elements_per_step: int
    communication_to_owned_work_ratio: float
    parallel_efficiency_ceiling: float
    max_communication_to_owned_work_ratio: float
    production_speedup_feasible: bool
    feasibility_blockers: tuple[str, ...]
    claim_scope: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the work model."""

        return asdict(self)


@dataclass(frozen=True)
class NonlinearSpectralPencilWorkModel:
    """Communication/work model for a pencil-FFT nonlinear bracket route."""

    state_shape: tuple[int, int, int, int, int]
    y_chunks: tuple[int, ...]
    x_chunks: tuple[int, ...]
    y_offsets: tuple[int, ...]
    x_offsets: tuple[int, ...]
    num_tiles: int
    state_elements: int
    field_elements: int
    transform_payload_elements_per_step: int
    pencil_transpose_elements_per_step: int
    global_reconstruction_elements_per_step: int
    approximate_fft_work_units_per_step: float
    communication_to_fft_work_ratio: float
    parallel_efficiency_ceiling: float
    predicted_speedup_ceiling: float
    max_communication_to_fft_work_ratio: float
    min_predicted_speedup: float
    production_speedup_feasible: bool
    feasibility_blockers: tuple[str, ...]
    claim_scope: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the pencil work model."""

        return asdict(self)


@dataclass(frozen=True)
class NonlinearSpectralPencilRHSIdentityReport:
    """Numerical identity report for the pencil-FFT nonlinear spectral RHS."""

    state_shape: tuple[int, int, int, int, int]
    y_chunks: tuple[int, ...]
    x_chunks: tuple[int, ...]
    y_offsets: tuple[int, ...]
    x_offsets: tuple[int, ...]
    atol: float
    rtol: float
    field_max_abs_error: float
    field_max_rel_error: float
    bracket_max_abs_error: float
    bracket_max_rel_error: float
    rhs_max_abs_error: float
    rhs_max_rel_error: float
    identity_passed: bool
    decomposed_path_enabled: bool
    work_model: NonlinearSpectralPencilWorkModel
    claim_scope: str
    blocked_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the RHS identity report."""

        data = asdict(self)
        data["work_model"] = self.work_model.to_dict()
        return data


@dataclass(frozen=True)
class NonlinearSpectralPencilTransportWindowReport:
    """Multi-step transport-window identity report for the pencil route."""

    state_shape: tuple[int, int, int, int, int]
    y_chunks: tuple[int, ...]
    x_chunks: tuple[int, ...]
    y_offsets: tuple[int, ...]
    x_offsets: tuple[int, ...]
    steps: int
    dt: float
    atol: float
    rtol: float
    final_state_max_abs_error: float
    final_state_max_rel_error: float
    free_energy_trace_max_abs_error: float
    free_energy_trace_max_rel_error: float
    field_energy_trace_max_abs_error: float
    field_energy_trace_max_rel_error: float
    physical_flux_trace_max_abs_error: float
    physical_flux_trace_max_rel_error: float
    bracket_rms_trace_max_abs_error: float
    bracket_rms_trace_max_rel_error: float
    serial_free_energy_drift: float
    pencil_free_energy_drift: float
    identity_passed: bool
    decomposed_path_enabled: bool
    work_model: NonlinearSpectralPencilWorkModel
    claim_scope: str
    blocked_reasons: tuple[str, ...] = ()
    serial_free_energy_trace: tuple[float, ...] = ()
    pencil_free_energy_trace: tuple[float, ...] = ()
    serial_field_energy_trace: tuple[float, ...] = ()
    pencil_field_energy_trace: tuple[float, ...] = ()
    serial_physical_flux_trace: tuple[float, ...] = ()
    pencil_physical_flux_trace: tuple[float, ...] = ()
    serial_bracket_rms_trace: tuple[float, ...] = ()
    pencil_bracket_rms_trace: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the transport report."""

        data = asdict(self)
        data["work_model"] = self.work_model.to_dict()
        return data


@dataclass(frozen=True)
class NonlinearSpectralDevicePencilRHSIdentityReport:
    """Identity report for a device-sharded fused pencil nonlinear RHS."""

    state_shape: tuple[int, int, int, int, int]
    sharded_axis: str
    axis_name: str
    requested_device_count: int
    active_device_count: int
    atol: float
    rtol: float
    rhs_max_abs_error: float
    rhs_max_rel_error: float
    identity_passed: bool
    device_sharding_active: bool
    decomposed_path_enabled: bool
    claim_scope: str
    blocked_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the identity report."""

        return asdict(self)


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
        physics_gates=(
            "nonlinear_window_diagnostic_identity_gate",
            "nonlinear_domain_transport_window_identity",
        ),
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
        readiness="diagnostic",
        independent_work=False,
        changes_solver_layout=True,
        identity_gates=(
            "distributed_fft_forward_inverse_identity",
            "distributed_fft_nonlinear_bracket_identity",
            "distributed_fft_field_solve_identity",
            "logical_sharded_nonlinear_spectral_rhs_identity",
            "logical_sharded_nonlinear_spectral_integrator_identity",
            "pencil_fft_fused_nonlinear_rhs_identity",
            "pencil_fft_physical_transport_window_identity",
            "device_z_pencil_fused_nonlinear_rhs_identity",
        ),
        physics_gates=("fft_axis_nonlinear_window_physics_gate",),
        profiler_gates=("distributed_fft_scaling_profile",),
        notes=(
            "Diagnostic split/reassemble spectral communication, logical spectral "
            "RHS, and pencil-FFT fused-bracket identity gates exist; production "
            "promotion still requires device-level transport-window routing, "
            "conservation, transport-window, and profiler speedup gates."
        ),
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

    plan_blockers = _nonlinear_domain_plan_blockers(plan)
    if plan_blockers:
        raise ValueError(
            "invalid nonlinear domain decomposition plan: "
            + ", ".join(plan_blockers)
        )
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
    report = nonlinear_domain_identity_report(serial, decomposed, plan, atol=atol, rtol=rtol)
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
    scale = jnp.maximum(jnp.abs(reference_arr), jnp.asarray(floor, dtype=reference_arr.dtype))
    max_rel = float(jnp.max(abs_error / scale))
    return max_abs, max_rel


def _trace_drift(trace: tuple[float, ...]) -> float:
    if len(trace) < 2:
        return 0.0
    return float(trace[-1] - trace[0])


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
            blocked_reasons=tuple(blocked_reasons),
            identity_passed=False,
            decomposed_path_enabled=False,
            claim_scope=_NONLINEAR_DOMAIN_TRANSPORT_CLAIM_SCOPE,
            boundary_indices=plan.boundary_indices if plan_valid else (),
        )

    serial_state = state
    decomposed_state = state
    serial_traces: dict[str, list[float]] = {
        "mass": [],
        "free_energy": [],
        "flux_proxy": [],
    }
    decomposed_traces: dict[str, list[float]] = {
        "mass": [],
        "free_energy": [],
        "flux_proxy": [],
    }
    _append_transport_observables(serial_traces, serial_state, plan)
    _append_transport_observables(decomposed_traces, decomposed_state, plan)
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
        _append_transport_observables(serial_traces, serial_state, plan)
        _append_transport_observables(decomposed_traces, decomposed_state, plan)

    state_report = nonlinear_domain_identity_report(
        serial_state,
        decomposed_state,
        plan,
        atol=atol,
        rtol=rtol,
    )
    serial_mass_trace = tuple(serial_traces["mass"])
    decomposed_mass_trace = tuple(decomposed_traces["mass"])
    serial_free_energy_trace = tuple(serial_traces["free_energy"])
    decomposed_free_energy_trace = tuple(decomposed_traces["free_energy"])
    serial_flux_proxy_trace = tuple(serial_traces["flux_proxy"])
    decomposed_flux_proxy_trace = tuple(decomposed_traces["flux_proxy"])
    mass_abs, mass_rel = _relative_trace_error(
        serial_mass_trace,
        decomposed_mass_trace,
        floor=atol,
    )
    free_energy_abs, free_energy_rel = _relative_trace_error(
        serial_free_energy_trace,
        decomposed_free_energy_trace,
        floor=atol,
    )
    flux_proxy_abs, flux_proxy_rel = _relative_trace_error(
        serial_flux_proxy_trace,
        decomposed_flux_proxy_trace,
        floor=atol,
    )
    identity_passed = bool(
        state_report.identity_passed
        and mass_abs <= float(atol)
        and mass_rel <= float(rtol)
        and free_energy_abs <= float(atol)
        and free_energy_rel <= float(rtol)
        and flux_proxy_abs <= float(atol)
        and flux_proxy_rel <= float(rtol)
    )

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
        mass_trace_max_abs_error=mass_abs,
        mass_trace_max_rel_error=mass_rel,
        free_energy_trace_max_abs_error=free_energy_abs,
        free_energy_trace_max_rel_error=free_energy_rel,
        flux_proxy_trace_max_abs_error=flux_proxy_abs,
        flux_proxy_trace_max_rel_error=flux_proxy_rel,
        serial_mass_drift=_trace_drift(serial_mass_trace),
        decomposed_mass_drift=_trace_drift(decomposed_mass_trace),
        serial_free_energy_drift=_trace_drift(serial_free_energy_trace),
        decomposed_free_energy_drift=_trace_drift(decomposed_free_energy_trace),
        plan_valid=plan_valid,
        blocked_reasons=tuple(blocked_reasons),
        identity_passed=identity_passed,
        decomposed_path_enabled=identity_passed,
        claim_scope=_NONLINEAR_DOMAIN_TRANSPORT_CLAIM_SCOPE,
        boundary_indices=state_report.boundary_indices,
        serial_mass_trace=serial_mass_trace,
        decomposed_mass_trace=decomposed_mass_trace,
        serial_free_energy_trace=serial_free_energy_trace,
        decomposed_free_energy_trace=decomposed_free_energy_trace,
        serial_flux_proxy_trace=serial_flux_proxy_trace,
        decomposed_flux_proxy_trace=decomposed_flux_proxy_trace,
    )


def _validate_spectral_state_shape(
    shape: tuple[int, ...],
) -> tuple[int, int, int, int, int]:
    if len(shape) != 5:
        raise ValueError("spectral state shape must be (Nl, Nm, Ny, Nx, Nz)")
    nl, nm, ny, nx, nz = (int(item) for item in shape)
    if min(nl, nm, ny, nx, nz) <= 0:
        raise ValueError("spectral state dimensions must be positive")
    if ny < 2 or nx < 2:
        raise ValueError("spectral communication gate requires Ny and Nx >= 2")
    return (nl, nm, ny, nx, nz)


def deterministic_nonlinear_spectral_state(
    shape: tuple[int, int, int, int, int] = (2, 3, 6, 4, 2),
) -> jax.Array:
    """Return deterministic complex spectral coefficients for communication gates.

    The layout is ``(Nl, Nm, Ny, Nx, Nz)`` with the FFT axes in ``(Ny, Nx)``.
    """

    nl, nm, ny, nx, nz = _validate_spectral_state_shape(tuple(shape))
    laguerre = jnp.arange(nl, dtype=jnp.float32)[:, None, None, None, None]
    hermite = jnp.arange(nm, dtype=jnp.float32)[None, :, None, None, None]
    y = jnp.arange(ny, dtype=jnp.float32)[None, None, :, None, None]
    x = jnp.arange(nx, dtype=jnp.float32)[None, None, None, :, None]
    z = jnp.arange(nz, dtype=jnp.float32)[None, None, None, None, :]
    phase = (
        0.41 * (laguerre + 1.0)
        + 0.29 * (hermite + 1.0)
        + 0.17 * y
        + 0.23 * x
        + 0.31 * (z + 1.0)
    )
    envelope = 1.0 / (1.0 + laguerre + hermite + 0.25 * y + 0.5 * x + z)
    real_part = envelope * jnp.sin(phase)
    imag_part = 0.5 * envelope * jnp.cos(1.7 * phase)
    return real_part.astype(jnp.float32) + 1j * imag_part.astype(jnp.float32)


def _validate_chunks(axis_size: int, chunks: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    if not chunks:
        raise ValueError(f"{name} must contain at least one chunk")
    normalized = tuple(int(item) for item in chunks)
    if any(item <= 0 for item in normalized):
        raise ValueError(f"{name} entries must be positive")
    if sum(normalized) != int(axis_size):
        raise ValueError(f"{name} must sum to the decomposed axis size")
    return normalized


def _chunk_offsets(chunks: tuple[int, ...]) -> tuple[int, ...]:
    offsets: list[int] = []
    start = 0
    for chunk in chunks:
        offsets.append(start)
        start += int(chunk)
    return tuple(offsets)


def _split_reassemble(arr: jax.Array, *, axis: int, chunks: tuple[int, ...]) -> jax.Array:
    canonical_axis = axis % arr.ndim
    normalized_chunks = _validate_chunks(
        int(arr.shape[canonical_axis]),
        chunks,
        name="chunks",
    )
    split_points = []
    offset = 0
    for chunk in normalized_chunks[:-1]:
        offset += chunk
        split_points.append(offset)
    return jnp.concatenate(jnp.split(arr, split_points, axis=canonical_axis), axis=canonical_axis)


def _spectral_layout_round_trip(
    arr: jax.Array,
    *,
    y_axis: int,
    x_axis: int,
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
) -> jax.Array:
    """Simulate the split/transposed/reassembled layout changes used by FFTs."""

    y_axis = y_axis % arr.ndim
    x_axis = x_axis % arr.ndim
    communicated = _split_reassemble(arr, axis=y_axis, chunks=y_chunks)
    transposed = jnp.swapaxes(communicated, y_axis, x_axis)
    reassembled = _split_reassemble(transposed, axis=x_axis, chunks=y_chunks)
    reassembled = _split_reassemble(reassembled, axis=y_axis, chunks=x_chunks)
    return jnp.swapaxes(reassembled, y_axis, x_axis)


def _spectral_tile_bounds(
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
) -> tuple[tuple[int, int, int, int], ...]:
    y_offsets = _chunk_offsets(y_chunks)
    x_offsets = _chunk_offsets(x_chunks)
    return tuple(
        (y_start, y_start + y_size, x_start, x_start + x_size)
        for y_start, y_size in zip(y_offsets, y_chunks, strict=True)
        for x_start, x_size in zip(x_offsets, x_chunks, strict=True)
    )


def nonlinear_spectral_domain_work_model(
    state_shape: tuple[int, int, int, int, int],
    *,
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    max_communication_to_owned_work_ratio: float = 0.5,
) -> NonlinearSpectralDomainWorkModel:
    """Estimate communication pressure for the current logical spectral route.

    The current diagnostic route reconstructs global spectral state/bracket
    arrays before returning owned output tiles. That is useful for identity
    gating, but it implies allgather/broadcast traffic that can dominate the
    owned tile work. This model is a conservative fail-closed screen for that
    route; it is not a performance prediction for a future distributed FFT.
    """

    nl, nm, ny, nx, nz = _validate_spectral_state_shape(tuple(state_shape))
    normalized_y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    normalized_x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")
    num_tiles = len(normalized_y_chunks) * len(normalized_x_chunks)
    state_elements = int(nl * nm * ny * nx * nz)
    field_elements = int(ny * nx * nz)
    communication_factor = max(num_tiles - 1, 0)
    state_allgather_elements = int(state_elements * communication_factor)
    bracket_allgather_elements = int(state_elements * communication_factor)
    field_broadcast_elements = int(field_elements * communication_factor)
    total_communication_elements = (
        state_allgather_elements
        + bracket_allgather_elements
        + field_broadcast_elements
    )
    owned_state_elements = state_elements
    ratio = (
        float(total_communication_elements) / float(owned_state_elements)
        if owned_state_elements > 0
        else float("inf")
    )
    efficiency_ceiling = 1.0 / (1.0 + ratio) if ratio >= 0.0 else 0.0

    blockers: list[str] = []
    if num_tiles < 2:
        blockers.append("single_tile_no_domain_decomposition")
    if ratio > float(max_communication_to_owned_work_ratio):
        blockers.append("global_reconstruction_communication_dominates_owned_work")
    production_speedup_feasible = bool(not blockers)

    return NonlinearSpectralDomainWorkModel(
        state_shape=(nl, nm, ny, nx, nz),
        y_chunks=normalized_y_chunks,
        x_chunks=normalized_x_chunks,
        y_offsets=_chunk_offsets(normalized_y_chunks),
        x_offsets=_chunk_offsets(normalized_x_chunks),
        tile_bounds=_spectral_tile_bounds(normalized_y_chunks, normalized_x_chunks),
        num_tiles=num_tiles,
        state_elements=state_elements,
        field_elements=field_elements,
        owned_state_elements_per_step=owned_state_elements,
        state_allgather_elements_per_step=state_allgather_elements,
        bracket_allgather_elements_per_step=bracket_allgather_elements,
        field_broadcast_elements_per_step=field_broadcast_elements,
        total_communication_elements_per_step=total_communication_elements,
        communication_to_owned_work_ratio=ratio,
        parallel_efficiency_ceiling=efficiency_ceiling,
        max_communication_to_owned_work_ratio=float(max_communication_to_owned_work_ratio),
        production_speedup_feasible=production_speedup_feasible,
        feasibility_blockers=tuple(blockers),
        claim_scope=(
            "diagnostic communication/work model for the current global-reconstruction "
            "logical spectral route; not a distributed FFT performance claim"
        ),
    )


def nonlinear_spectral_pencil_work_model(
    state_shape: tuple[int, int, int, int, int],
    *,
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    max_communication_to_fft_work_ratio: float = 0.35,
    min_predicted_speedup: float = 1.5,
) -> NonlinearSpectralPencilWorkModel:
    """Estimate communication pressure for a pencil-FFT bracket route.

    The pencil route avoids global state/bracket reconstruction. Its remaining
    distributed cost is the all-to-all transpose traffic needed by axis-wise 2D
    FFTs plus a field-reduction/broadcast. This model is intentionally simple
    and fail-closed: it must pass before any strong-scaling timing is treated as
    a meaningful candidate for production promotion.
    """

    nl, nm, ny, nx, nz = _validate_spectral_state_shape(tuple(state_shape))
    normalized_y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    normalized_x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")
    num_tiles = len(normalized_y_chunks) * len(normalized_x_chunks)
    state_elements = int(nl * nm * ny * nx * nz)
    field_elements = int(ny * nx * nz)

    # Fused bracket payload: inverse FFTs for two state gradients, inverse FFTs
    # for two field gradients, and one forward FFT of the bracket.
    transform_payload = int(3 * state_elements + 2 * field_elements)
    sent_fraction = 1.0 - (1.0 / float(num_tiles)) if num_tiles > 0 else 1.0
    pencil_transpose_elements = int(round(transform_payload * sent_fraction))
    fft_extent = max(int(ny * nx), 2)
    approximate_fft_work = float(transform_payload) * math.log2(float(fft_extent))
    ratio = (
        float(pencil_transpose_elements) / approximate_fft_work
        if approximate_fft_work > 0.0
        else float("inf")
    )
    efficiency_ceiling = 1.0 / (1.0 + ratio) if ratio >= 0.0 else 0.0
    predicted_speedup = float(num_tiles) * efficiency_ceiling

    blockers: list[str] = []
    if num_tiles < 2:
        blockers.append("single_tile_no_domain_decomposition")
    if ratio > float(max_communication_to_fft_work_ratio):
        blockers.append("pencil_transpose_communication_dominates_fft_work")
    if predicted_speedup < float(min_predicted_speedup):
        blockers.append("predicted_speedup_below_gate")

    return NonlinearSpectralPencilWorkModel(
        state_shape=(nl, nm, ny, nx, nz),
        y_chunks=normalized_y_chunks,
        x_chunks=normalized_x_chunks,
        y_offsets=_chunk_offsets(normalized_y_chunks),
        x_offsets=_chunk_offsets(normalized_x_chunks),
        num_tiles=num_tiles,
        state_elements=state_elements,
        field_elements=field_elements,
        transform_payload_elements_per_step=transform_payload,
        pencil_transpose_elements_per_step=pencil_transpose_elements,
        global_reconstruction_elements_per_step=0,
        approximate_fft_work_units_per_step=approximate_fft_work,
        communication_to_fft_work_ratio=ratio,
        parallel_efficiency_ceiling=efficiency_ceiling,
        predicted_speedup_ceiling=predicted_speedup,
        max_communication_to_fft_work_ratio=float(max_communication_to_fft_work_ratio),
        min_predicted_speedup=float(min_predicted_speedup),
        production_speedup_feasible=bool(not blockers),
        feasibility_blockers=tuple(blockers),
        claim_scope=(
            "communication/work model for a pencil-FFT nonlinear bracket route "
            "with explicit transpose stages and no global reconstruction; not a "
            "runtime speedup claim without identity and profiler artifacts"
        ),
    )


def _normalize_spectral_tile_bounds(
    tile_bounds: tuple[tuple[int, int, int, int], ...],
) -> tuple[tuple[int, int, int, int], ...]:
    """Return validated fixed-width tile bounds for mypy and runtime checks."""

    normalized: list[tuple[int, int, int, int]] = []
    for item in tile_bounds:
        if len(item) != 4:
            raise ValueError("each spectral tile bound must contain four integers")
        y_start, y_stop, x_start, x_stop = (int(value) for value in item)
        normalized.append((y_start, y_stop, x_start, x_stop))
    return tuple(normalized)


def _logical_spectral_tiles(
    arr: jax.Array,
    *,
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    y_axis: int,
    x_axis: int,
) -> tuple[jax.Array, ...]:
    canonical_y_axis = y_axis % arr.ndim
    canonical_x_axis = x_axis % arr.ndim
    normalized_y_chunks = _validate_chunks(
        int(arr.shape[canonical_y_axis]),
        y_chunks,
        name="y_chunks",
    )
    normalized_x_chunks = _validate_chunks(
        int(arr.shape[canonical_x_axis]),
        x_chunks,
        name="x_chunks",
    )

    tiles: list[jax.Array] = []
    for y_start, y_stop, x_start, x_stop in _spectral_tile_bounds(
        normalized_y_chunks,
        normalized_x_chunks,
    ):
        y_tile = jax.lax.dynamic_slice_in_dim(
            arr,
            y_start,
            y_stop - y_start,
            axis=canonical_y_axis,
        )
        tiles.append(
            jax.lax.dynamic_slice_in_dim(
                y_tile,
                x_start,
                x_stop - x_start,
                axis=canonical_x_axis,
            )
        )
    return tuple(tiles)


def _reconstruct_logical_spectral_tiles(
    tiles: tuple[jax.Array, ...],
    *,
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    y_axis: int,
    x_axis: int,
) -> jax.Array:
    if len(tiles) != len(y_chunks) * len(x_chunks):
        raise ValueError("tile count must match y_chunks by x_chunks")
    if not tiles:
        raise ValueError("at least one tile is required")

    canonical_y_axis = y_axis % tiles[0].ndim
    canonical_x_axis = x_axis % tiles[0].ndim
    rows = []
    tile_index = 0
    for _y_chunk in y_chunks:
        row_tiles = []
        for _x_chunk in x_chunks:
            row_tiles.append(tiles[tile_index])
            tile_index += 1
        rows.append(jnp.concatenate(row_tiles, axis=canonical_x_axis))
    return jnp.concatenate(rows, axis=canonical_y_axis)


def _spectral_wave_numbers(ny: int, nx: int, dtype: Any) -> tuple[jax.Array, jax.Array]:
    ky = jnp.fft.fftfreq(ny, d=1.0 / float(ny)).astype(dtype)
    kx = jnp.fft.fftfreq(nx, d=1.0 / float(nx)).astype(dtype)
    return ky, kx


def _field_from_spectral_density(density_hat: jax.Array) -> jax.Array:
    ny, nx, _nz = (int(item) for item in density_hat.shape)
    real_dtype = jnp.real(density_hat).dtype
    ky, kx = _spectral_wave_numbers(ny, nx, real_dtype)
    kperp2 = ky[:, None, None] ** 2 + kx[None, :, None] ** 2
    phi_hat = density_hat / (1.0 + kperp2)
    return phi_hat.at[0, 0, :].set(0.0)


def _field_from_state(state_hat: jax.Array) -> jax.Array:
    density_hat = jnp.sum(state_hat[:, 0, :, :, :], axis=0)
    return _field_from_spectral_density(density_hat)


def _spectral_bracket(state_hat: jax.Array, phi_hat: jax.Array) -> jax.Array:
    _nl, _nm, ny, nx, _nz = _validate_spectral_state_shape(tuple(state_hat.shape))
    real_dtype = jnp.real(state_hat).dtype
    ky, kx = _spectral_wave_numbers(ny, nx, real_dtype)
    ky_state = ky[None, None, :, None, None]
    kx_state = kx[None, None, None, :, None]
    ky_field = ky[:, None, None]
    kx_field = kx[None, :, None]

    state_dx = jnp.fft.ifft2(1j * kx_state * state_hat, axes=(-3, -2))
    state_dy = jnp.fft.ifft2(1j * ky_state * state_hat, axes=(-3, -2))
    phi_dx = jnp.fft.ifft2(1j * kx_field * phi_hat, axes=(0, 1))
    phi_dy = jnp.fft.ifft2(1j * ky_field * phi_hat, axes=(0, 1))
    bracket_xy = phi_dx[None, None, :, :, :] * state_dy - phi_dy[
        None, None, :, :, :
    ] * state_dx
    return jnp.fft.fft2(bracket_xy, axes=(-3, -2))


def _pencil_ifft2(arr: jax.Array, *, y_axis: int, x_axis: int) -> jax.Array:
    """Return a 2D inverse FFT through explicit x-then-y pencil stages."""

    y_axis = y_axis % arr.ndim
    x_axis = x_axis % arr.ndim
    x_transformed = jnp.fft.ifft(arr, axis=x_axis)
    transposed = jnp.swapaxes(x_transformed, y_axis, x_axis)
    y_transformed = jnp.fft.ifft(transposed, axis=x_axis)
    return jnp.swapaxes(y_transformed, y_axis, x_axis)


def _pencil_fft2(arr: jax.Array, *, y_axis: int, x_axis: int) -> jax.Array:
    """Return a 2D forward FFT through explicit x-then-y pencil stages."""

    y_axis = y_axis % arr.ndim
    x_axis = x_axis % arr.ndim
    x_transformed = jnp.fft.fft(arr, axis=x_axis)
    transposed = jnp.swapaxes(x_transformed, y_axis, x_axis)
    y_transformed = jnp.fft.fft(transposed, axis=x_axis)
    return jnp.swapaxes(y_transformed, y_axis, x_axis)


def _pencil_spectral_bracket(state_hat: jax.Array, phi_hat: jax.Array) -> jax.Array:
    """Return the pseudo-spectral bracket using pencil FFT staging.

    This function is the local algorithmic route that a distributed pencil FFT
    implementation should follow: stack derivative operands, transform through
    explicit axis-transpose stages, multiply in physical space, and transform
    the bracket back without first reconstructing logical output tiles.
    """

    _nl, _nm, ny, nx, _nz = _validate_spectral_state_shape(tuple(state_hat.shape))
    real_dtype = jnp.real(state_hat).dtype
    ky, kx = _spectral_wave_numbers(ny, nx, real_dtype)
    ky_state = ky[None, None, :, None, None]
    kx_state = kx[None, None, None, :, None]
    ky_field = ky[:, None, None]
    kx_field = kx[None, :, None]

    state_grad_hat = jnp.stack(
        [1j * kx_state * state_hat, 1j * ky_state * state_hat],
        axis=0,
    )
    state_grad_xy = _pencil_ifft2(state_grad_hat, y_axis=-3, x_axis=-2)
    state_dx = state_grad_xy[0]
    state_dy = state_grad_xy[1]

    field_grad_hat = jnp.stack(
        [1j * kx_field * phi_hat, 1j * ky_field * phi_hat],
        axis=0,
    )
    field_grad_xy = _pencil_ifft2(field_grad_hat, y_axis=1, x_axis=2)
    phi_dx = field_grad_xy[0]
    phi_dy = field_grad_xy[1]

    bracket_xy = phi_dx[None, None, :, :, :] * state_dy - phi_dy[
        None, None, :, :, :
    ] * state_dx
    return _pencil_fft2(bracket_xy, y_axis=-3, x_axis=-2)


def _spectral_rhs_from_bracket(bracket_hat: jax.Array) -> jax.Array:
    """Return the ExB advection contribution used by the identity micro-route."""

    return -bracket_hat


def _serial_nonlinear_spectral_rhs(state_hat: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    field = _field_from_state(state_hat)
    bracket = _spectral_bracket(state_hat, field)
    rhs = _spectral_rhs_from_bracket(bracket)
    return field, bracket, rhs


def _pencil_nonlinear_spectral_rhs(state_hat: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    field = _field_from_state(state_hat)
    bracket = _pencil_spectral_bracket(state_hat, field)
    rhs = _spectral_rhs_from_bracket(bracket)
    return field, bracket, rhs


def _max_abs_rel_error(
    reference: jax.Array,
    candidate: jax.Array,
    *,
    atol: float,
) -> tuple[float, float]:
    if tuple(reference.shape) != tuple(candidate.shape):
        return float("inf"), float("inf")
    abs_error = jnp.abs(candidate - reference)
    scale = jnp.maximum(jnp.abs(reference), jnp.asarray(atol, dtype=jnp.real(abs_error).dtype))
    rel_error = abs_error / scale
    return float(jnp.max(abs_error)), float(jnp.max(rel_error))


def _nonlinear_spectral_report_blockers(
    serial_fft_roundtrip: jax.Array,
    communicated_fft_roundtrip: jax.Array,
    serial_bracket: jax.Array,
    communicated_bracket: jax.Array,
    serial_field: jax.Array,
    communicated_field: jax.Array,
    *,
    state_shape: tuple[int, ...],
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
) -> tuple[str, ...]:
    blockers: list[str] = []

    try:
        normalized_state_shape = _validate_spectral_state_shape(tuple(state_shape))
    except ValueError:
        normalized_state_shape = None
        blockers.append("state_shape_invalid")

    if normalized_state_shape is not None:
        _nl, _nm, ny, nx, nz = normalized_state_shape
        try:
            _validate_chunks(ny, y_chunks, name="y_chunks")
        except ValueError:
            blockers.append("y_chunks_invalid")
        try:
            _validate_chunks(nx, x_chunks, name="x_chunks")
        except ValueError:
            blockers.append("x_chunks_invalid")

        expected_field_shape = (ny, nx, nz)
        state_arrays = (
            ("serial_fft_roundtrip", serial_fft_roundtrip),
            ("communicated_fft_roundtrip", communicated_fft_roundtrip),
            ("serial_bracket", serial_bracket),
            ("communicated_bracket", communicated_bracket),
        )
        field_arrays = (
            ("serial_field", serial_field),
            ("communicated_field", communicated_field),
        )
        for name, arr in state_arrays:
            if tuple(arr.shape) != normalized_state_shape:
                blockers.append(f"{name}_shape_mismatch")
        for name, arr in field_arrays:
            if tuple(arr.shape) != expected_field_shape:
                blockers.append(f"{name}_shape_mismatch")

    return tuple(blockers)


def nonlinear_spectral_communication_identity_report(
    serial_fft_roundtrip: jax.Array,
    communicated_fft_roundtrip: jax.Array,
    serial_bracket: jax.Array,
    communicated_bracket: jax.Array,
    serial_field: jax.Array,
    communicated_field: jax.Array,
    *,
    state_shape: tuple[int, int, int, int, int],
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> NonlinearSpectralCommunicationReport:
    """Compare spectral communication outputs and fail closed on mismatches."""

    blocked_reasons = _nonlinear_spectral_report_blockers(
        serial_fft_roundtrip,
        communicated_fft_roundtrip,
        serial_bracket,
        communicated_bracket,
        serial_field,
        communicated_field,
        state_shape=state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    fft_abs, fft_rel = _max_abs_rel_error(
        serial_fft_roundtrip,
        communicated_fft_roundtrip,
        atol=atol,
    )
    bracket_abs, bracket_rel = _max_abs_rel_error(
        serial_bracket,
        communicated_bracket,
        atol=atol,
    )
    field_abs, field_rel = _max_abs_rel_error(
        serial_field,
        communicated_field,
        atol=atol,
    )
    identity_passed = bool(
        not blocked_reasons
        and fft_abs <= float(atol)
        and fft_rel <= float(rtol)
        and bracket_abs <= float(atol)
        and bracket_rel <= float(rtol)
        and field_abs <= float(atol)
        and field_rel <= float(rtol)
    )
    return NonlinearSpectralCommunicationReport(
        state_shape=state_shape,
        y_chunks=tuple(int(item) for item in y_chunks),
        x_chunks=tuple(int(item) for item in x_chunks),
        y_offsets=_chunk_offsets(y_chunks),
        x_offsets=_chunk_offsets(x_chunks),
        atol=float(atol),
        rtol=float(rtol),
        fft_max_abs_error=fft_abs,
        fft_max_rel_error=fft_rel,
        bracket_max_abs_error=bracket_abs,
        bracket_max_rel_error=bracket_rel,
        field_max_abs_error=field_abs,
        field_max_rel_error=field_rel,
        identity_passed=identity_passed,
        decomposed_path_enabled=identity_passed,
        claim_scope=(
            "diagnostic spectral communication identity gate only; "
            "split/reassemble layout simulation with no production routing or speedup claim"
        ),
        blocked_reasons=blocked_reasons,
    )


def nonlinear_spectral_communication_identity_gate(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> NonlinearSpectralCommunicationReport:
    """Validate FFT, bracket, and field layout identity under split/reassemble."""

    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    _nl, _nm, ny, nx, _nz = state_shape
    y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")

    communicated_state = _spectral_layout_round_trip(
        state_hat,
        y_axis=-3,
        x_axis=-2,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    serial_fft = jnp.fft.fft2(jnp.fft.ifft2(state_hat, axes=(-3, -2)), axes=(-3, -2))
    communicated_fft = jnp.fft.fft2(
        jnp.fft.ifft2(communicated_state, axes=(-3, -2)),
        axes=(-3, -2),
    )
    serial_field = _field_from_state(state_hat)
    communicated_field = _field_from_state(communicated_state)
    serial_bracket = _spectral_bracket(state_hat, serial_field)
    communicated_bracket = _spectral_bracket(communicated_state, communicated_field)

    return nonlinear_spectral_communication_identity_report(
        serial_fft,
        communicated_fft,
        serial_bracket,
        communicated_bracket,
        serial_field,
        communicated_field,
        state_shape=state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        atol=atol,
        rtol=rtol,
    )


def _nonlinear_spectral_rhs_report_blockers(
    serial_reconstruction: jax.Array,
    logical_reconstruction: jax.Array,
    serial_field: jax.Array,
    logical_field: jax.Array,
    serial_bracket: jax.Array,
    logical_bracket: jax.Array,
    serial_rhs: jax.Array,
    logical_rhs: jax.Array,
    *,
    state_shape: tuple[int, ...],
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    tile_bounds: tuple[tuple[int, int, int, int], ...],
) -> tuple[str, ...]:
    blockers: list[str] = []

    try:
        normalized_state_shape = _validate_spectral_state_shape(tuple(state_shape))
    except ValueError:
        normalized_state_shape = None
        blockers.append("state_shape_invalid")

    normalized_y_chunks: tuple[int, ...] | None = None
    normalized_x_chunks: tuple[int, ...] | None = None
    if normalized_state_shape is not None:
        _nl, _nm, ny, nx, nz = normalized_state_shape
        try:
            normalized_y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
        except ValueError:
            blockers.append("y_chunks_invalid")
        try:
            normalized_x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")
        except ValueError:
            blockers.append("x_chunks_invalid")

        expected_field_shape = (ny, nx, nz)
        state_arrays = (
            ("serial_reconstruction", serial_reconstruction),
            ("logical_reconstruction", logical_reconstruction),
            ("serial_bracket", serial_bracket),
            ("logical_bracket", logical_bracket),
            ("serial_rhs", serial_rhs),
            ("logical_rhs", logical_rhs),
        )
        field_arrays = (
            ("serial_field", serial_field),
            ("logical_field", logical_field),
        )
        for name, arr in state_arrays:
            if tuple(arr.shape) != normalized_state_shape:
                blockers.append(f"{name}_shape_mismatch")
        for name, arr in field_arrays:
            if tuple(arr.shape) != expected_field_shape:
                blockers.append(f"{name}_shape_mismatch")

    if normalized_y_chunks is not None and normalized_x_chunks is not None:
        if tile_bounds != _spectral_tile_bounds(normalized_y_chunks, normalized_x_chunks):
            blockers.append("tile_bounds_not_row_major")

    return tuple(blockers)


def nonlinear_spectral_rhs_identity_report(
    serial_reconstruction: jax.Array,
    logical_reconstruction: jax.Array,
    serial_field: jax.Array,
    logical_field: jax.Array,
    serial_bracket: jax.Array,
    logical_bracket: jax.Array,
    serial_rhs: jax.Array,
    logical_rhs: jax.Array,
    *,
    state_shape: tuple[int, int, int, int, int],
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    tile_bounds: tuple[tuple[int, int, int, int], ...] | None = None,
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> NonlinearSpectralRHSIdentityReport:
    """Compare serial and logical-shard spectral RHS outputs fail-closed."""

    normalized_y_chunks = tuple(int(item) for item in y_chunks)
    normalized_x_chunks = tuple(int(item) for item in x_chunks)
    effective_tile_bounds = (
        _spectral_tile_bounds(normalized_y_chunks, normalized_x_chunks)
        if tile_bounds is None
        else _normalize_spectral_tile_bounds(tile_bounds)
    )
    blocked_reasons = _nonlinear_spectral_rhs_report_blockers(
        serial_reconstruction,
        logical_reconstruction,
        serial_field,
        logical_field,
        serial_bracket,
        logical_bracket,
        serial_rhs,
        logical_rhs,
        state_shape=state_shape,
        y_chunks=normalized_y_chunks,
        x_chunks=normalized_x_chunks,
        tile_bounds=effective_tile_bounds,
    )
    reconstruction_abs, reconstruction_rel = _max_abs_rel_error(
        serial_reconstruction,
        logical_reconstruction,
        atol=atol,
    )
    field_abs, field_rel = _max_abs_rel_error(
        serial_field,
        logical_field,
        atol=atol,
    )
    bracket_abs, bracket_rel = _max_abs_rel_error(
        serial_bracket,
        logical_bracket,
        atol=atol,
    )
    rhs_abs, rhs_rel = _max_abs_rel_error(
        serial_rhs,
        logical_rhs,
        atol=atol,
    )
    identity_passed = bool(
        not blocked_reasons
        and reconstruction_abs <= float(atol)
        and reconstruction_rel <= float(rtol)
        and field_abs <= float(atol)
        and field_rel <= float(rtol)
        and bracket_abs <= float(atol)
        and bracket_rel <= float(rtol)
        and rhs_abs <= float(atol)
        and rhs_rel <= float(rtol)
    )
    return NonlinearSpectralRHSIdentityReport(
        state_shape=state_shape,
        y_chunks=normalized_y_chunks,
        x_chunks=normalized_x_chunks,
        y_offsets=_chunk_offsets(normalized_y_chunks),
        x_offsets=_chunk_offsets(normalized_x_chunks),
        tile_bounds=effective_tile_bounds,
        atol=float(atol),
        rtol=float(rtol),
        reconstruction_max_abs_error=reconstruction_abs,
        reconstruction_max_rel_error=reconstruction_rel,
        field_max_abs_error=field_abs,
        field_max_rel_error=field_rel,
        bracket_max_abs_error=bracket_abs,
        bracket_max_rel_error=bracket_rel,
        rhs_max_abs_error=rhs_abs,
        rhs_max_rel_error=rhs_rel,
        identity_passed=identity_passed,
        decomposed_path_enabled=identity_passed,
        claim_scope=(
            "diagnostic nonlinear spectral RHS identity gate only; "
            "logical output-tile reconstruction with existing bracket contribution "
            "and no production routing or speedup claim"
        ),
        blocked_reasons=blocked_reasons,
    )


def _logical_sharded_nonlinear_spectral_rhs(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    state_tiles = _logical_spectral_tiles(
        state_hat,
        y_axis=-3,
        x_axis=-2,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    reconstructed_state = _reconstruct_logical_spectral_tiles(
        state_tiles,
        y_axis=-3,
        x_axis=-2,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    field = _field_from_state(reconstructed_state)
    bracket = _spectral_bracket(reconstructed_state, field)
    bracket_tiles = _logical_spectral_tiles(
        bracket,
        y_axis=-3,
        x_axis=-2,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    logical_bracket = _reconstruct_logical_spectral_tiles(
        bracket_tiles,
        y_axis=-3,
        x_axis=-2,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    rhs_tiles = tuple(_spectral_rhs_from_bracket(tile) for tile in bracket_tiles)
    logical_rhs = _reconstruct_logical_spectral_tiles(
        rhs_tiles,
        y_axis=-3,
        x_axis=-2,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    return reconstructed_state, field, logical_bracket, logical_rhs


def nonlinear_spectral_rhs_identity_gate(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> NonlinearSpectralRHSIdentityReport:
    """Validate serial-vs-logical-shard nonlinear spectral RHS identity.

    This diagnostic route owns and reassembles spectral ``(y, x)`` output tiles
    in row-major order. It deliberately does not install distributed FFT
    runtime routing or make a speedup claim.
    """

    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    _nl, _nm, ny, nx, _nz = state_shape
    y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")

    serial_field, serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(state_hat)
    (
        logical_reconstruction,
        logical_field,
        logical_bracket,
        logical_rhs,
    ) = _logical_sharded_nonlinear_spectral_rhs(
        state_hat,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )

    return nonlinear_spectral_rhs_identity_report(
        state_hat,
        logical_reconstruction,
        serial_field,
        logical_field,
        serial_bracket,
        logical_bracket,
        serial_rhs,
        logical_rhs,
        state_shape=state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        tile_bounds=_spectral_tile_bounds(y_chunks, x_chunks),
        atol=atol,
        rtol=rtol,
    )


def logical_decomposed_nonlinear_spectral_rhs(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> tuple[jax.Array, NonlinearSpectralRHSIdentityReport]:
    """Return the logical-shard nonlinear spectral RHS after identity gating.

    The returned RHS uses the logical decomposed route only when it is exactly
    equivalent to the serial reference under the provided tolerances. Otherwise
    the serial RHS is returned and ``decomposed_path_enabled`` is false. This is
    still a local diagnostic route, not a distributed runtime implementation.
    """

    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    _nl, _nm, ny, nx, _nz = state_shape
    y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")
    serial_field, serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(state_hat)
    (
        logical_reconstruction,
        logical_field,
        logical_bracket,
        logical_rhs,
    ) = _logical_sharded_nonlinear_spectral_rhs(
        state_hat,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    report = nonlinear_spectral_rhs_identity_report(
        state_hat,
        logical_reconstruction,
        serial_field,
        logical_field,
        serial_bracket,
        logical_bracket,
        serial_rhs,
        logical_rhs,
        state_shape=state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        tile_bounds=_spectral_tile_bounds(y_chunks, x_chunks),
        atol=atol,
        rtol=rtol,
    )
    gated_rhs = logical_rhs if report.decomposed_path_enabled else serial_rhs
    return gated_rhs, report


def nonlinear_spectral_pencil_rhs_identity_gate(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    atol: float = 5.0e-6,
    rtol: float = 1.0e-5,
    max_communication_to_fft_work_ratio: float = 0.35,
    min_predicted_speedup: float = 1.5,
) -> NonlinearSpectralPencilRHSIdentityReport:
    """Validate serial-vs-pencil nonlinear spectral RHS identity.

    The pencil route uses stacked derivative transforms and explicit axis
    transposes, avoiding the logical tile reconstruction used by the older
    diagnostic route.  Passing this report only enables the routed path when
    the communication model also predicts plausible speedup.
    """

    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    _nl, _nm, ny, nx, _nz = state_shape
    y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")
    work_model = nonlinear_spectral_pencil_work_model(
        state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        max_communication_to_fft_work_ratio=max_communication_to_fft_work_ratio,
        min_predicted_speedup=min_predicted_speedup,
    )

    serial_field, serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(state_hat)
    pencil_field, pencil_bracket, pencil_rhs = _pencil_nonlinear_spectral_rhs(state_hat)
    field_abs, field_rel = _max_abs_rel_error(serial_field, pencil_field, atol=atol)
    bracket_abs, bracket_rel = _max_abs_rel_error(serial_bracket, pencil_bracket, atol=atol)
    rhs_abs, rhs_rel = _max_abs_rel_error(serial_rhs, pencil_rhs, atol=atol)
    identity_passed = bool(
        field_abs <= float(atol)
        and field_rel <= float(rtol)
        and bracket_abs <= float(atol)
        and bracket_rel <= float(rtol)
        and rhs_abs <= float(atol)
        and rhs_rel <= float(rtol)
    )
    decomposed_path_enabled = bool(identity_passed and work_model.production_speedup_feasible)
    blocked_reasons: list[str] = []
    if not identity_passed:
        blocked_reasons.append("pencil_rhs_identity_failed")
    blocked_reasons.extend(work_model.feasibility_blockers)

    return NonlinearSpectralPencilRHSIdentityReport(
        state_shape=state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        y_offsets=_chunk_offsets(y_chunks),
        x_offsets=_chunk_offsets(x_chunks),
        atol=float(atol),
        rtol=float(rtol),
        field_max_abs_error=field_abs,
        field_max_rel_error=field_rel,
        bracket_max_abs_error=bracket_abs,
        bracket_max_rel_error=bracket_rel,
        rhs_max_abs_error=rhs_abs,
        rhs_max_rel_error=rhs_rel,
        identity_passed=identity_passed,
        decomposed_path_enabled=decomposed_path_enabled,
        work_model=work_model,
        claim_scope=(
            "diagnostic pencil-FFT nonlinear spectral RHS identity gate; "
            "axis-wise FFT staging with no global reconstruction, but no "
            "production speedup claim without profiler-backed scaling"
        ),
        blocked_reasons=tuple(blocked_reasons),
    )


def pencil_decomposed_nonlinear_spectral_rhs(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    atol: float = 5.0e-6,
    rtol: float = 1.0e-5,
) -> tuple[jax.Array, NonlinearSpectralPencilRHSIdentityReport]:
    """Return the pencil nonlinear spectral RHS after identity/model gating."""

    _serial_field, _serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(state_hat)
    _pencil_field, _pencil_bracket, pencil_rhs = _pencil_nonlinear_spectral_rhs(state_hat)
    report = nonlinear_spectral_pencil_rhs_identity_gate(
        state_hat,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        atol=atol,
        rtol=rtol,
    )
    gated_rhs = pencil_rhs if report.decomposed_path_enabled else serial_rhs
    return gated_rhs, report


def _device_z_sharding_for_spectral_state(
    state_hat: jax.Array,
    *,
    devices: Sequence[Any] | None,
    axis_name: str,
) -> tuple[Any | None, Any | None, tuple[str, ...], int, int]:
    """Return a z-axis sharding for the fused-bracket route, or blockers.

    The nonlinear pseudo-spectral bracket transforms only the ``(ky, kx)``
    axes. Sharding over ``z`` therefore keeps the FFTs local to each device and
    avoids the global tile reconstruction that blocked the older logical route.
    """

    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    *_, nz = state_shape
    device_tuple = tuple(devices) if devices is not None else tuple(jax.devices())
    requested_device_count = len(device_tuple)
    blockers: list[str] = []
    if requested_device_count < 2:
        blockers.append("requires_at_least_two_devices")
        return None, None, tuple(blockers), requested_device_count, 0

    active_device_count = requested_device_count
    if nz % active_device_count != 0:
        blockers.append("z_extent_not_divisible_by_device_count")
        return None, None, tuple(blockers), requested_device_count, 0

    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    mesh = Mesh(np.asarray(device_tuple[:active_device_count]), (axis_name,))
    sharding = NamedSharding(mesh, PartitionSpec(None, None, None, None, axis_name))
    return mesh, sharding, (), requested_device_count, active_device_count


def device_z_pencil_nonlinear_spectral_rhs(
    state_hat: jax.Array,
    *,
    devices: Sequence[Any] | None = None,
    axis_name: str = "z",
    atol: float = 5.0e-6,
    rtol: float = 1.0e-4,
) -> tuple[jax.Array, NonlinearSpectralDevicePencilRHSIdentityReport]:
    """Return the z-sharded fused pencil nonlinear RHS after identity gating.

    This is the first real device-sharded nonlinear spectral route in this
    module. It shards over the field-line ``z`` axis so that the FFT axes remain
    local on every device. The function falls back to the serial RHS unless a
    multi-device sharding exists and the sharded fused-bracket RHS matches the
    serial reference within the requested tolerances.
    """

    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    mesh, sharding, blockers, requested_count, active_count = (
        _device_z_sharding_for_spectral_state(
            state_hat,
            devices=devices,
            axis_name=axis_name,
        )
    )
    _serial_field, _serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(state_hat)
    if blockers or mesh is None or sharding is None:
        report = NonlinearSpectralDevicePencilRHSIdentityReport(
            state_shape=state_shape,
            sharded_axis="z",
            axis_name=str(axis_name),
            requested_device_count=int(requested_count),
            active_device_count=int(active_count),
            atol=float(atol),
            rtol=float(rtol),
            rhs_max_abs_error=float("inf"),
            rhs_max_rel_error=float("inf"),
            identity_passed=False,
            device_sharding_active=False,
            decomposed_path_enabled=False,
            claim_scope=(
                "device z-sharded fused pencil nonlinear RHS gate; skipped because "
                "the requested local device topology cannot support the z shard"
            ),
            blocked_reasons=blockers,
        )
        return serial_rhs, report

    def _rhs_only(local_state: jax.Array) -> jax.Array:
        return _pencil_nonlinear_spectral_rhs(local_state)[2]

    with mesh:
        sharded_rhs_fn = jax.jit(
            _rhs_only,
            in_shardings=sharding,
            out_shardings=sharding,
        )
        sharded_state = jax.device_put(state_hat, sharding)
        candidate_rhs = sharded_rhs_fn(sharded_state)

    rhs_abs, rhs_rel = _max_abs_rel_error(serial_rhs, candidate_rhs, atol=atol)
    identity_passed = bool(rhs_abs <= float(atol) and rhs_rel <= float(rtol))
    blockers_list: list[str] = []
    if not identity_passed:
        blockers_list.append("device_z_pencil_rhs_identity_failed")

    report = NonlinearSpectralDevicePencilRHSIdentityReport(
        state_shape=state_shape,
        sharded_axis="z",
        axis_name=str(axis_name),
        requested_device_count=int(requested_count),
        active_device_count=int(active_count),
        atol=float(atol),
        rtol=float(rtol),
        rhs_max_abs_error=rhs_abs,
        rhs_max_rel_error=rhs_rel,
        identity_passed=identity_passed,
        device_sharding_active=True,
        decomposed_path_enabled=identity_passed,
        claim_scope=(
            "device z-sharded fused pencil nonlinear RHS identity gate; FFT axes "
            "remain local per device and no global spectral reconstruction is used, "
            "but no speedup claim is allowed without matched profiler gates"
        ),
        blocked_reasons=tuple(blockers_list),
    )
    gated_rhs = candidate_rhs if report.decomposed_path_enabled else serial_rhs
    return gated_rhs, report


def _spectral_physical_transport_observables(
    state_hat: jax.Array,
    bracket_hat: jax.Array,
) -> tuple[float, float, float, float]:
    """Return physical-space transport-window observables for the micro-route."""

    _nl, _nm, ny, nx, _nz = _validate_spectral_state_shape(tuple(state_hat.shape))
    real_dtype = jnp.real(state_hat).dtype
    ky, _kx = _spectral_wave_numbers(ny, nx, real_dtype)
    field = _field_from_state(state_hat)
    density_hat = jnp.sum(state_hat[:, 0, :, :, :], axis=0)
    density_xy = _pencil_ifft2(density_hat, y_axis=0, x_axis=1)
    phi_y = _pencil_ifft2(1j * ky[:, None, None] * field, y_axis=0, x_axis=1)
    physical_flux = float(jnp.mean(jnp.abs(jnp.real(jnp.conj(density_xy) * (-phi_y)))))
    free_energy = float(jnp.sum(jnp.abs(state_hat) ** 2))
    field_energy = float(jnp.sum(jnp.abs(field) ** 2))
    bracket_rms = float(jnp.sqrt(jnp.mean(jnp.abs(bracket_hat) ** 2)))
    return free_energy, field_energy, physical_flux, bracket_rms


def _append_spectral_physical_observables(
    traces: dict[str, list[float]],
    state_hat: jax.Array,
    bracket_hat: jax.Array,
) -> None:
    free_energy, field_energy, physical_flux, bracket_rms = (
        _spectral_physical_transport_observables(state_hat, bracket_hat)
    )
    traces["free_energy"].append(free_energy)
    traces["field_energy"].append(field_energy)
    traces["physical_flux"].append(physical_flux)
    traces["bracket_rms"].append(bracket_rms)


def nonlinear_spectral_pencil_transport_window_identity_gate(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    dt: float = 0.005,
    steps: int = 4,
    atol: float = 5.0e-6,
    rtol: float = 1.0e-5,
    max_communication_to_fft_work_ratio: float = 0.35,
    min_predicted_speedup: float = 1.5,
) -> NonlinearSpectralPencilTransportWindowReport:
    """Validate a serial-vs-pencil nonlinear transport window.

    The trace includes physical-space density/``E_r`` transport proxy values in
    addition to free energy, field energy, bracket RMS, and final state identity.
    This is a routing identity gate for the pseudo-spectral nonlinear kernel,
    not a calibrated turbulent heat-flux validation.
    """

    if int(steps) < 1:
        raise ValueError("steps must be at least one")
    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    _nl, _nm, ny, nx, _nz = state_shape
    y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")
    work_model = nonlinear_spectral_pencil_work_model(
        state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        max_communication_to_fft_work_ratio=max_communication_to_fft_work_ratio,
        min_predicted_speedup=min_predicted_speedup,
    )

    serial_state = state_hat
    pencil_state = state_hat
    serial_traces: dict[str, list[float]] = {
        "free_energy": [],
        "field_energy": [],
        "physical_flux": [],
        "bracket_rms": [],
    }
    pencil_traces: dict[str, list[float]] = {
        "free_energy": [],
        "field_energy": [],
        "physical_flux": [],
        "bracket_rms": [],
    }
    _serial_field, serial_bracket, _serial_rhs = _serial_nonlinear_spectral_rhs(serial_state)
    _pencil_field, pencil_bracket, _pencil_rhs = _pencil_nonlinear_spectral_rhs(pencil_state)
    _append_spectral_physical_observables(serial_traces, serial_state, serial_bracket)
    _append_spectral_physical_observables(pencil_traces, pencil_state, pencil_bracket)

    dt_array = jnp.asarray(float(dt), dtype=jnp.real(state_hat).dtype)
    for _ in range(int(steps)):
        _serial_field, serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(
            serial_state,
        )
        _pencil_field, pencil_bracket, pencil_rhs = _pencil_nonlinear_spectral_rhs(
            pencil_state,
        )
        serial_state = serial_state + dt_array * serial_rhs
        pencil_state = pencil_state + dt_array * pencil_rhs
        _serial_field, serial_bracket, _serial_rhs = _serial_nonlinear_spectral_rhs(
            serial_state,
        )
        _pencil_field, pencil_bracket, _pencil_rhs = _pencil_nonlinear_spectral_rhs(
            pencil_state,
        )
        _append_spectral_physical_observables(serial_traces, serial_state, serial_bracket)
        _append_spectral_physical_observables(pencil_traces, pencil_state, pencil_bracket)

    state_abs, state_rel = _max_abs_rel_error(serial_state, pencil_state, atol=atol)
    serial_free = tuple(serial_traces["free_energy"])
    pencil_free = tuple(pencil_traces["free_energy"])
    serial_field_energy = tuple(serial_traces["field_energy"])
    pencil_field_energy = tuple(pencil_traces["field_energy"])
    serial_physical_flux = tuple(serial_traces["physical_flux"])
    pencil_physical_flux = tuple(pencil_traces["physical_flux"])
    serial_bracket_rms = tuple(serial_traces["bracket_rms"])
    pencil_bracket_rms = tuple(pencil_traces["bracket_rms"])
    free_abs, free_rel = _relative_trace_error(serial_free, pencil_free, floor=atol)
    field_abs, field_rel = _relative_trace_error(
        serial_field_energy,
        pencil_field_energy,
        floor=atol,
    )
    flux_abs, flux_rel = _relative_trace_error(
        serial_physical_flux,
        pencil_physical_flux,
        floor=atol,
    )
    bracket_abs, bracket_rel = _relative_trace_error(
        serial_bracket_rms,
        pencil_bracket_rms,
        floor=atol,
    )
    identity_passed = bool(
        state_abs <= float(atol)
        and state_rel <= float(rtol)
        and free_abs <= float(atol)
        and free_rel <= float(rtol)
        and field_abs <= float(atol)
        and field_rel <= float(rtol)
        and flux_abs <= float(atol)
        and flux_rel <= float(rtol)
        and bracket_abs <= float(atol)
        and bracket_rel <= float(rtol)
    )
    decomposed_path_enabled = bool(identity_passed and work_model.production_speedup_feasible)
    blocked_reasons: list[str] = []
    if not identity_passed:
        blocked_reasons.append("pencil_transport_window_identity_failed")
    blocked_reasons.extend(work_model.feasibility_blockers)

    return NonlinearSpectralPencilTransportWindowReport(
        state_shape=state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        y_offsets=_chunk_offsets(y_chunks),
        x_offsets=_chunk_offsets(x_chunks),
        steps=int(steps),
        dt=float(dt),
        atol=float(atol),
        rtol=float(rtol),
        final_state_max_abs_error=state_abs,
        final_state_max_rel_error=state_rel,
        free_energy_trace_max_abs_error=free_abs,
        free_energy_trace_max_rel_error=free_rel,
        field_energy_trace_max_abs_error=field_abs,
        field_energy_trace_max_rel_error=field_rel,
        physical_flux_trace_max_abs_error=flux_abs,
        physical_flux_trace_max_rel_error=flux_rel,
        bracket_rms_trace_max_abs_error=bracket_abs,
        bracket_rms_trace_max_rel_error=bracket_rel,
        serial_free_energy_drift=_trace_drift(serial_free),
        pencil_free_energy_drift=_trace_drift(pencil_free),
        identity_passed=identity_passed,
        decomposed_path_enabled=decomposed_path_enabled,
        work_model=work_model,
        claim_scope=(
            "diagnostic serial-vs-pencil nonlinear physical-space transport-window "
            "identity gate; includes a density-times-radial-E-field transport "
            "proxy, but is not an absolute nonlinear turbulent heat-flux claim"
        ),
        blocked_reasons=tuple(blocked_reasons),
        serial_free_energy_trace=serial_free,
        pencil_free_energy_trace=pencil_free,
        serial_field_energy_trace=serial_field_energy,
        pencil_field_energy_trace=pencil_field_energy,
        serial_physical_flux_trace=serial_physical_flux,
        pencil_physical_flux_trace=pencil_physical_flux,
        serial_bracket_rms_trace=serial_bracket_rms,
        pencil_bracket_rms_trace=pencil_bracket_rms,
    )


def _spectral_integrator_observables(state_hat: jax.Array) -> tuple[float, float, float]:
    field = _field_from_state(state_hat)
    free_energy = float(jnp.sum(jnp.abs(state_hat) ** 2))
    field_energy = float(jnp.sum(jnp.abs(field) ** 2))
    bracket = _spectral_bracket(state_hat, field)
    flux_proxy = float(jnp.mean(jnp.abs(_spectral_rhs_from_bracket(bracket))))
    return free_energy, field_energy, flux_proxy


def _append_spectral_observables(
    traces: dict[str, list[float]],
    state_hat: jax.Array,
) -> None:
    free_energy, field_energy, flux_proxy = _spectral_integrator_observables(state_hat)
    traces["free_energy"].append(free_energy)
    traces["field_energy"].append(field_energy)
    traces["flux_proxy"].append(flux_proxy)


def nonlinear_spectral_integrator_identity_gate(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    dt: float = 0.005,
    steps: int = 4,
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> NonlinearSpectralIntegratorIdentityReport:
    """Validate serial-vs-logical-shard nonlinear spectral integration.

    This gate compares a fixed-step explicit-Euler micro-integration using the
    same nonlinear spectral RHS on serial and logical tiled routes. It verifies
    final-state identity and per-step free-energy, field-energy, and flux-proxy
    traces. Passing this gate is necessary but not sufficient for production
    nonlinear domain parallelization.
    """

    if int(steps) < 1:
        raise ValueError("steps must be at least one")
    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    _nl, _nm, ny, nx, _nz = state_shape
    y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")

    serial_state = state_hat
    logical_state = state_hat
    serial_traces: dict[str, list[float]] = {
        "free_energy": [],
        "field_energy": [],
        "flux_proxy": [],
    }
    logical_traces: dict[str, list[float]] = {
        "free_energy": [],
        "field_energy": [],
        "flux_proxy": [],
    }
    _append_spectral_observables(serial_traces, serial_state)
    _append_spectral_observables(logical_traces, logical_state)
    dt_array = jnp.asarray(float(dt), dtype=jnp.real(state_hat).dtype)
    blocked_reasons: list[str] = []

    for _ in range(int(steps)):
        _serial_field, _serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(serial_state)
        logical_rhs, rhs_report = logical_decomposed_nonlinear_spectral_rhs(
            logical_state,
            y_chunks=y_chunks,
            x_chunks=x_chunks,
            atol=atol,
            rtol=rtol,
        )
        blocked_reasons.extend(rhs_report.blocked_reasons)
        if not rhs_report.identity_passed:
            blocked_reasons.append("per_step_rhs_identity_failed")
        serial_state = serial_state + dt_array * serial_rhs
        logical_state = logical_state + dt_array * logical_rhs
        _append_spectral_observables(serial_traces, serial_state)
        _append_spectral_observables(logical_traces, logical_state)

    state_abs, state_rel = _max_abs_rel_error(serial_state, logical_state, atol=atol)
    serial_free = tuple(serial_traces["free_energy"])
    logical_free = tuple(logical_traces["free_energy"])
    serial_field_energy = tuple(serial_traces["field_energy"])
    logical_field_energy = tuple(logical_traces["field_energy"])
    serial_flux = tuple(serial_traces["flux_proxy"])
    logical_flux = tuple(logical_traces["flux_proxy"])
    free_abs, free_rel = _relative_trace_error(serial_free, logical_free, floor=atol)
    field_abs, field_rel = _relative_trace_error(
        serial_field_energy,
        logical_field_energy,
        floor=atol,
    )
    flux_abs, flux_rel = _relative_trace_error(serial_flux, logical_flux, floor=atol)
    unique_blockers = tuple(sorted(set(blocked_reasons)))
    identity_passed = bool(
        not unique_blockers
        and state_abs <= float(atol)
        and state_rel <= float(rtol)
        and free_abs <= float(atol)
        and free_rel <= float(rtol)
        and field_abs <= float(atol)
        and field_rel <= float(rtol)
        and flux_abs <= float(atol)
        and flux_rel <= float(rtol)
    )
    return NonlinearSpectralIntegratorIdentityReport(
        state_shape=state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        y_offsets=_chunk_offsets(y_chunks),
        x_offsets=_chunk_offsets(x_chunks),
        tile_bounds=_spectral_tile_bounds(y_chunks, x_chunks),
        steps=int(steps),
        dt=float(dt),
        atol=float(atol),
        rtol=float(rtol),
        final_state_max_abs_error=state_abs,
        final_state_max_rel_error=state_rel,
        free_energy_trace_max_abs_error=free_abs,
        free_energy_trace_max_rel_error=free_rel,
        field_energy_trace_max_abs_error=field_abs,
        field_energy_trace_max_rel_error=field_rel,
        flux_proxy_trace_max_abs_error=flux_abs,
        flux_proxy_trace_max_rel_error=flux_rel,
        serial_free_energy_drift=_trace_drift(serial_free),
        logical_free_energy_drift=_trace_drift(logical_free),
        identity_passed=identity_passed,
        decomposed_path_enabled=identity_passed,
        claim_scope=(
            "diagnostic nonlinear spectral integrator identity gate only; "
            "fixed-step serial-vs-logical-shard RHS routing with final-state "
            "and transport-proxy trace identity, no production distributed FFT "
            "routing or speedup claim"
        ),
        blocked_reasons=unique_blockers,
        serial_free_energy_trace=serial_free,
        logical_free_energy_trace=logical_free,
        serial_field_energy_trace=serial_field_energy,
        logical_field_energy_trace=logical_field_energy,
        serial_flux_proxy_trace=serial_flux,
        logical_flux_proxy_trace=logical_flux,
    )


def integrate_logical_decomposed_nonlinear_spectral(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    dt: float = 0.005,
    steps: int = 4,
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> tuple[jax.Array, NonlinearSpectralIntegratorIdentityReport]:
    """Integrate with the logical decomposed spectral route after identity gating.

    This is the callable route behind the nonlinear spectral integrator identity
    artifact.  Each step requests the logical tiled RHS; if the local RHS gate
    fails, that step uses the serial RHS instead.  The returned report is the
    independent serial-vs-routed identity gate for the full fixed-step window.
    Passing this function's report is required before timing the route, but it
    is still not a production distributed-FFT implementation or speedup claim.
    """

    if int(steps) < 1:
        raise ValueError("steps must be at least one")
    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    _nl, _nm, ny, nx, _nz = state_shape
    y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")

    routed_state = state_hat
    dt_array = jnp.asarray(float(dt), dtype=jnp.real(state_hat).dtype)
    for _ in range(int(steps)):
        logical_rhs, rhs_report = logical_decomposed_nonlinear_spectral_rhs(
            routed_state,
            y_chunks=y_chunks,
            x_chunks=x_chunks,
            atol=atol,
            rtol=rtol,
        )
        if rhs_report.decomposed_path_enabled:
            step_rhs = logical_rhs
        else:
            _field, _bracket, step_rhs = _serial_nonlinear_spectral_rhs(routed_state)
        routed_state = routed_state + dt_array * step_rhs

    report = nonlinear_spectral_integrator_identity_gate(
        state_hat,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        dt=dt,
        steps=steps,
        atol=atol,
        rtol=rtol,
    )
    if report.decomposed_path_enabled:
        return routed_state, report

    serial_state = state_hat
    for _ in range(int(steps)):
        _field, _bracket, serial_rhs = _serial_nonlinear_spectral_rhs(serial_state)
        serial_state = serial_state + dt_array * serial_rhs
    return serial_state, report


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
    "NonlinearDomainTransportWindowReport",
    "NonlinearParallelStrategy",
    "NonlinearParallelStrategyName",
    "NonlinearSpectralCommunicationReport",
    "NonlinearSpectralDevicePencilRHSIdentityReport",
    "NonlinearSpectralDomainWorkModel",
    "NonlinearSpectralIntegratorIdentityReport",
    "NonlinearSpectralPencilRHSIdentityReport",
    "NonlinearSpectralPencilTransportWindowReport",
    "NonlinearSpectralPencilWorkModel",
    "NonlinearSpectralRHSIdentityReport",
    "ParallelReadiness",
    "build_nonlinear_domain_decomposition_plan",
    "classify_nonlinear_parallel_strategy",
    "deterministic_nonlinear_domain_state",
    "deterministic_nonlinear_spectral_state",
    "device_z_pencil_nonlinear_spectral_rhs",
    "integrate_logical_decomposed_nonlinear_spectral",
    "nonlinear_domain_identity_report",
    "nonlinear_domain_parallel_identity_gate",
    "nonlinear_domain_transport_window_identity_gate",
    "nonlinear_parallel_strategies",
    "nonlinear_parallel_strategy",
    "nonlinear_spectral_communication_identity_gate",
    "nonlinear_spectral_communication_identity_report",
    "nonlinear_spectral_domain_work_model",
    "logical_decomposed_nonlinear_spectral_rhs",
    "nonlinear_spectral_integrator_identity_gate",
    "nonlinear_spectral_pencil_rhs_identity_gate",
    "nonlinear_spectral_pencil_transport_window_identity_gate",
    "nonlinear_spectral_pencil_work_model",
    "nonlinear_spectral_rhs_identity_gate",
    "nonlinear_spectral_rhs_identity_report",
    "pencil_decomposed_nonlinear_spectral_rhs",
    "prototype_nonlinear_domain_decomposed_step",
    "prototype_nonlinear_domain_serial_step",
    "release_ready_nonlinear_parallel_strategies",
]
