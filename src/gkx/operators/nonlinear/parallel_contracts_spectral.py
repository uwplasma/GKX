"""Spectral report and work-model contracts for nonlinear parallel diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


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


__all__ = [
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
]
