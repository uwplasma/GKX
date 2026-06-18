"""Public facade for nonlinear parallelization contracts and identity gates.

The production-facing paths in this module remain policy metadata. Numerical
routes are conservative diagnostic prototypes: they only enable decomposed
updates after direct numerical identity against serial reference operations.
"""

from __future__ import annotations


import jax
import jax.numpy as jnp

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
    _relative_trace_error,
    _trace_drift,
    build_nonlinear_domain_decomposition_plan,
    deterministic_nonlinear_domain_state,
    nonlinear_domain_identity_report,
    nonlinear_domain_parallel_identity_gate,
    nonlinear_domain_transport_window_identity_gate,
    prototype_nonlinear_domain_decomposed_step,
    prototype_nonlinear_domain_serial_step,
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
    nonlinear_spectral_rhs_identity_gate,
)

from spectraxgk.operators.nonlinear.device_z import (
    _append_spectral_physical_observable_vector as _append_spectral_physical_observable_vector,
    _append_spectral_physical_observables as _append_spectral_physical_observables,
    _device_z_pencil_shard_map_observables_fn as _device_z_pencil_shard_map_observables_fn,
    _device_z_pencil_shard_map_rhs_fn as _device_z_pencil_shard_map_rhs_fn,
    _device_z_sharding_for_spectral_state as _device_z_sharding_for_spectral_state,
    _spectral_physical_transport_observable_sums as _spectral_physical_transport_observable_sums,
    _spectral_physical_transport_observable_vector_from_sums as _spectral_physical_transport_observable_vector_from_sums,
    _spectral_physical_transport_observables as _spectral_physical_transport_observables,
    device_z_pencil_nonlinear_spectral_rhs,
    device_z_pencil_nonlinear_spectral_transport_window_identity_gate,
)


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
    bracket_abs, bracket_rel = _max_abs_rel_error(
        serial_bracket, pencil_bracket, atol=atol
    )
    rhs_abs, rhs_rel = _max_abs_rel_error(serial_rhs, pencil_rhs, atol=atol)
    identity_passed = bool(
        field_abs <= float(atol)
        and field_rel <= float(rtol)
        and bracket_abs <= float(atol)
        and bracket_rel <= float(rtol)
        and rhs_abs <= float(atol)
        and rhs_rel <= float(rtol)
    )
    decomposed_path_enabled = bool(
        identity_passed and work_model.production_speedup_feasible
    )
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

    _serial_field, _serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(
        state_hat
    )
    _pencil_field, _pencil_bracket, pencil_rhs = _pencil_nonlinear_spectral_rhs(
        state_hat
    )
    report = nonlinear_spectral_pencil_rhs_identity_gate(
        state_hat,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        atol=atol,
        rtol=rtol,
    )
    gated_rhs = pencil_rhs if report.decomposed_path_enabled else serial_rhs
    return gated_rhs, report












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
    _serial_field, serial_bracket, _serial_rhs = _serial_nonlinear_spectral_rhs(
        serial_state
    )
    _pencil_field, pencil_bracket, _pencil_rhs = _pencil_nonlinear_spectral_rhs(
        pencil_state
    )
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
        _append_spectral_physical_observables(
            serial_traces, serial_state, serial_bracket
        )
        _append_spectral_physical_observables(
            pencil_traces, pencil_state, pencil_bracket
        )

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
    decomposed_path_enabled = bool(
        identity_passed and work_model.production_speedup_feasible
    )
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
    "prototype_nonlinear_domain_decomposed_step",
    "prototype_nonlinear_domain_serial_step",
    "release_ready_nonlinear_parallel_strategies",
]
