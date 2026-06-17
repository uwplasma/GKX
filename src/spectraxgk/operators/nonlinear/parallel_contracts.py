"""Contracts and report DTOs for nonlinear parallelization policy.

This module is intentionally free of numerical kernels.  It owns immutable
contracts, JSON-friendly reports, and fail-closed policy helpers used by the
``spectraxgk.operators.nonlinear.parallel`` facade.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import jax


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
class NonlinearSpectralDevicePencilTransportWindowReport:
    """Multi-step identity report for device-z-sharded pencil routing."""

    state_shape: tuple[int, int, int, int, int]
    sharded_axis: str
    axis_name: str
    requested_device_count: int
    active_device_count: int
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
    device_free_energy_drift: float
    identity_passed: bool
    device_sharding_active: bool
    decomposed_path_enabled: bool
    claim_scope: str
    blocked_reasons: tuple[str, ...] = ()
    serial_free_energy_trace: tuple[float, ...] = ()
    device_free_energy_trace: tuple[float, ...] = ()
    serial_field_energy_trace: tuple[float, ...] = ()
    device_field_energy_trace: tuple[float, ...] = ()
    serial_physical_flux_trace: tuple[float, ...] = ()
    device_physical_flux_trace: tuple[float, ...] = ()
    serial_bracket_rms_trace: tuple[float, ...] = ()
    device_bracket_rms_trace: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the transport report."""

        return asdict(self)


@dataclass(frozen=True)
class NonlinearSpectralDevicePencilFFTBatchModel:
    """cuFFT batch-pressure preflight model for the device-z pencil route."""

    state_shape: tuple[int, int, int, int, int]
    device_count: int
    local_z_extent: int
    max_fft_axis_extent: int
    max_fft_batch_count: int
    unchunked_fft_batch_count: int
    suggested_z_chunk_size: int | None
    effective_z_chunk_size: int
    chunked_fft_batch_count: int
    chunking_required: bool
    chunking_active: bool
    disable_gpu_preallocation_recommended: bool
    profiling_candidate: bool
    feasibility_blockers: tuple[str, ...]
    claim_scope: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the batch model."""

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
        notes="Velocity-space production candidate; promotion requires end-to-end nonlinear identity gates.",
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
            "device_z_pencil_physical_transport_window_identity",
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


__all__ = [
    "NonlinearDomainDecompositionPlan",
    "NonlinearDomainIdentityReport",
    "NonlinearDomainTransportWindowReport",
    "NonlinearParallelStrategy",
    "NonlinearParallelStrategyName",
    "NonlinearSpectralCommunicationReport",
    "NonlinearSpectralDevicePencilFFTBatchModel",
    "NonlinearSpectralDevicePencilRHSIdentityReport",
    "NonlinearSpectralDevicePencilTransportWindowReport",
    "NonlinearSpectralDomainWorkModel",
    "NonlinearSpectralIntegratorIdentityReport",
    "NonlinearSpectralPencilRHSIdentityReport",
    "NonlinearSpectralPencilTransportWindowReport",
    "NonlinearSpectralPencilWorkModel",
    "NonlinearSpectralRHSIdentityReport",
    "ParallelReadiness",
]
