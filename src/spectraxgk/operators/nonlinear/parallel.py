"""Public facade for nonlinear parallelization contracts and identity gates.

The production-facing paths in this module remain policy metadata. Numerical
routes are conservative diagnostic local-stencil routes: they only enable decomposed
updates after direct numerical identity against serial reference operations.
"""

from __future__ import annotations


from spectraxgk.operators.nonlinear.parallel_contracts_domain import (
    NonlinearDomainDecompositionPlan,
    NonlinearDomainIdentityReport,
    NonlinearDomainTransportWindowReport,
)
from spectraxgk.operators.nonlinear.parallel_contracts_spectral import (
    NonlinearSpectralCommunicationReport,
    NonlinearSpectralDevicePencilFFTBatchModel,
    NonlinearSpectralDevicePencilRHSIdentityReport,
    NonlinearSpectralDevicePencilTransportWindowReport,
    NonlinearSpectralDomainWorkModel,
    NonlinearSpectralIntegratorIdentityReport,
    NonlinearSpectralPencilRHSIdentityReport,
    NonlinearSpectralPencilTransportWindowReport,
    NonlinearSpectralPencilWorkModel,
    NonlinearSpectralRHSIdentityReport,
)
from spectraxgk.operators.nonlinear.parallel_contracts_strategy import (
    _STRATEGIES,
    _STRATEGY_BY_NAME,
    NonlinearParallelStrategy,
    NonlinearParallelStrategyName,
    ParallelReadiness,
)


from spectraxgk.operators.nonlinear.domain_decomposition import (
    build_nonlinear_domain_decomposition_plan,
    deterministic_nonlinear_domain_state,
    nonlinear_domain_identity_report,
    nonlinear_domain_parallel_identity_gate,
    nonlinear_domain_transport_window_identity_gate,
    local_stencil_nonlinear_domain_decomposed_step,
    local_stencil_nonlinear_domain_serial_step,
)


from spectraxgk.operators.nonlinear.spectral_core import (
    _chunk_offsets as _chunk_offsets,
    _field_from_spectral_density as _field_from_spectral_density,
    _field_from_state as _field_from_state,
    _host_max_abs_rel_error as _host_max_abs_rel_error,
    _host_staged_array_for_sharding as _host_staged_array_for_sharding,
    _logical_spectral_tiles as _logical_spectral_tiles,
    _max_abs_rel_error as _max_abs_rel_error,
    _normalize_spectral_tile_bounds as _normalize_spectral_tile_bounds,
    _pencil_fft2 as _pencil_fft2,
    _pencil_ifft2 as _pencil_ifft2,
    _pencil_nonlinear_spectral_rhs as _pencil_nonlinear_spectral_rhs,
    _pencil_nonlinear_spectral_rhs_z_chunked as _pencil_nonlinear_spectral_rhs_z_chunked,
    _pencil_spectral_bracket as _pencil_spectral_bracket,
    _pencil_spectral_bracket_z_chunked as _pencil_spectral_bracket_z_chunked,
    _reconstruct_logical_spectral_tiles as _reconstruct_logical_spectral_tiles,
    _serial_nonlinear_spectral_rhs as _serial_nonlinear_spectral_rhs,
    _spectral_bracket as _spectral_bracket,
    _spectral_layout_round_trip as _spectral_layout_round_trip,
    _spectral_rhs_from_bracket as _spectral_rhs_from_bracket,
    _spectral_tile_bounds as _spectral_tile_bounds,
    _spectral_wave_numbers as _spectral_wave_numbers,
    _split_reassemble as _split_reassemble,
    _validate_chunks as _validate_chunks,
    _validate_spectral_state_shape as _validate_spectral_state_shape,
    _within_abs_or_rel_tolerance as _within_abs_or_rel_tolerance,
    deterministic_nonlinear_spectral_state,
    device_z_pencil_fft_batch_pressure_model,
    nonlinear_spectral_domain_work_model,
    nonlinear_spectral_pencil_work_model,
)

from spectraxgk.operators.nonlinear.spectral_identity_integrator import (
    _append_spectral_observables as _append_spectral_observables,
    _spectral_integrator_observables as _spectral_integrator_observables,
    integrate_logical_decomposed_nonlinear_spectral,
    nonlinear_spectral_integrator_identity_gate,
    nonlinear_spectral_pencil_transport_window_identity_gate,
)
from spectraxgk.operators.nonlinear.spectral_identity_reports import (
    _nonlinear_spectral_report_blockers as _nonlinear_spectral_report_blockers,
    _nonlinear_spectral_rhs_report_blockers as _nonlinear_spectral_rhs_report_blockers,
    nonlinear_spectral_communication_identity_gate,
    nonlinear_spectral_communication_identity_report,
    nonlinear_spectral_rhs_identity_report,
)
from spectraxgk.operators.nonlinear.spectral_identity_rhs import (
    _logical_sharded_nonlinear_spectral_rhs as _logical_sharded_nonlinear_spectral_rhs,
    logical_decomposed_nonlinear_spectral_rhs,
    nonlinear_spectral_pencil_rhs_identity_gate,
    nonlinear_spectral_rhs_identity_gate,
    pencil_decomposed_nonlinear_spectral_rhs,
)

from spectraxgk.operators.nonlinear.device_z import (
    _append_spectral_physical_observable_vector as _append_spectral_physical_observable_vector,
    _append_spectral_physical_observables as _append_spectral_physical_observables,
    _device_z_pencil_shard_map_observables_fn as _device_z_pencil_shard_map_observables_fn,
    _device_z_pencil_shard_map_rhs_fn as _device_z_pencil_shard_map_rhs_fn,
    _device_z_sharding_for_spectral_state as _device_z_sharding_for_spectral_state,
    _spectral_physical_transport_observable_sums as _spectral_physical_transport_observable_sums,
    _spectral_physical_transport_observable_vector_from_sums as _spectral_physical_transport_observable_vector_from_sums,
    device_z_pencil_nonlinear_spectral_rhs,
    device_z_pencil_nonlinear_spectral_transport_window_identity_gate,
)




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
    "build_nonlinear_domain_decomposition_plan",
    "classify_nonlinear_parallel_strategy",
    "deterministic_nonlinear_domain_state",
    "deterministic_nonlinear_spectral_state",
    "device_z_pencil_fft_batch_pressure_model",
    "device_z_pencil_nonlinear_spectral_rhs",
    "device_z_pencil_nonlinear_spectral_transport_window_identity_gate",
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
    "local_stencil_nonlinear_domain_decomposed_step",
    "local_stencil_nonlinear_domain_serial_step",
    "release_ready_nonlinear_parallel_strategies",
]
