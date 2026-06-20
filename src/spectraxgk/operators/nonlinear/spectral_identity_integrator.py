"""Fixed-window nonlinear spectral integrator identity gates."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from spectraxgk.operators.nonlinear.domain_decomposition import (
    _relative_trace_error,
    _trace_drift,
)
from spectraxgk.operators.nonlinear.parallel_contracts_spectral import (
    NonlinearSpectralIntegratorIdentityReport,
    NonlinearSpectralPencilTransportWindowReport,
    NonlinearSpectralPencilWorkModel,
)
from spectraxgk.operators.nonlinear.spectral_core import (
    _chunk_offsets,
    _field_from_state,
    _max_abs_rel_error,
    _pencil_nonlinear_spectral_rhs,
    _serial_nonlinear_spectral_rhs,
    _spectral_bracket,
    _spectral_rhs_from_bracket,
    _spectral_tile_bounds,
    _validate_chunks,
    _validate_spectral_state_shape,
    nonlinear_spectral_pencil_work_model,
)
from spectraxgk.operators.nonlinear.device_z import (
    _append_spectral_physical_observables,
)
from spectraxgk.operators.nonlinear.spectral_identity_rhs import (
    logical_decomposed_nonlinear_spectral_rhs,
)


@dataclass(frozen=True)
class _SpectralIntegratorTraces:
    free_energy: tuple[float, ...]
    field_energy: tuple[float, ...]
    flux_proxy: tuple[float, ...]


@dataclass(frozen=True)
class _SpectralIntegratorSamples:
    serial_state: jax.Array
    logical_state: jax.Array
    serial_traces: _SpectralIntegratorTraces
    logical_traces: _SpectralIntegratorTraces
    blocked_reasons: tuple[str, ...]


@dataclass(frozen=True)
class _SpectralIntegratorTraceErrors:
    free_abs: float
    free_rel: float
    field_abs: float
    field_rel: float
    flux_abs: float
    flux_rel: float


@dataclass(frozen=True)
class _SpectralPhysicalTraces:
    free_energy: tuple[float, ...]
    field_energy: tuple[float, ...]
    physical_flux: tuple[float, ...]
    bracket_rms: tuple[float, ...]


@dataclass(frozen=True)
class _SpectralPhysicalSamples:
    serial_state: jax.Array
    pencil_state: jax.Array
    serial_traces: _SpectralPhysicalTraces
    pencil_traces: _SpectralPhysicalTraces


@dataclass(frozen=True)
class _SpectralPhysicalTraceErrors:
    free_abs: float
    free_rel: float
    field_abs: float
    field_rel: float
    flux_abs: float
    flux_rel: float
    bracket_abs: float
    bracket_rel: float


def _spectral_integrator_observables(
    state_hat: jax.Array,
) -> tuple[float, float, float]:
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


def _new_spectral_integrator_traces() -> dict[str, list[float]]:
    return {
        "free_energy": [],
        "field_energy": [],
        "flux_proxy": [],
    }


def _freeze_spectral_integrator_traces(
    traces: dict[str, list[float]],
) -> _SpectralIntegratorTraces:
    return _SpectralIntegratorTraces(
        free_energy=tuple(traces["free_energy"]),
        field_energy=tuple(traces["field_energy"]),
        flux_proxy=tuple(traces["flux_proxy"]),
    )


def _run_logical_spectral_window(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    dt: float,
    steps: int,
    atol: float,
    rtol: float,
) -> _SpectralIntegratorSamples:
    serial_state = state_hat
    logical_state = state_hat
    serial_trace_lists = _new_spectral_integrator_traces()
    logical_trace_lists = _new_spectral_integrator_traces()
    _append_spectral_observables(serial_trace_lists, serial_state)
    _append_spectral_observables(logical_trace_lists, logical_state)
    dt_array = jnp.asarray(float(dt), dtype=jnp.real(state_hat).dtype)
    blocked_reasons: list[str] = []

    for _ in range(int(steps)):
        _serial_field, _serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(
            serial_state
        )
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
        _append_spectral_observables(serial_trace_lists, serial_state)
        _append_spectral_observables(logical_trace_lists, logical_state)

    return _SpectralIntegratorSamples(
        serial_state=serial_state,
        logical_state=logical_state,
        serial_traces=_freeze_spectral_integrator_traces(serial_trace_lists),
        logical_traces=_freeze_spectral_integrator_traces(logical_trace_lists),
        blocked_reasons=tuple(sorted(set(blocked_reasons))),
    )


def _spectral_integrator_trace_errors(
    serial: _SpectralIntegratorTraces,
    logical: _SpectralIntegratorTraces,
    *,
    floor: float,
) -> _SpectralIntegratorTraceErrors:
    free_abs, free_rel = _relative_trace_error(
        serial.free_energy,
        logical.free_energy,
        floor=floor,
    )
    field_abs, field_rel = _relative_trace_error(
        serial.field_energy,
        logical.field_energy,
        floor=floor,
    )
    flux_abs, flux_rel = _relative_trace_error(
        serial.flux_proxy,
        logical.flux_proxy,
        floor=floor,
    )
    return _SpectralIntegratorTraceErrors(
        free_abs=free_abs,
        free_rel=free_rel,
        field_abs=field_abs,
        field_rel=field_rel,
        flux_abs=flux_abs,
        flux_rel=flux_rel,
    )


def _spectral_integrator_identity_passed(
    *,
    state_abs: float,
    state_rel: float,
    errors: _SpectralIntegratorTraceErrors,
    blocked_reasons: tuple[str, ...],
    atol: float,
    rtol: float,
) -> bool:
    return bool(
        not blocked_reasons
        and state_abs <= float(atol)
        and state_rel <= float(rtol)
        and errors.free_abs <= float(atol)
        and errors.free_rel <= float(rtol)
        and errors.field_abs <= float(atol)
        and errors.field_rel <= float(rtol)
        and errors.flux_abs <= float(atol)
        and errors.flux_rel <= float(rtol)
    )


def _spectral_integrator_report(
    state_shape: tuple[int, int, int, int, int],
    samples: _SpectralIntegratorSamples,
    errors: _SpectralIntegratorTraceErrors,
    *,
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    steps: int,
    dt: float,
    atol: float,
    rtol: float,
) -> NonlinearSpectralIntegratorIdentityReport:
    state_abs, state_rel = _max_abs_rel_error(
        samples.serial_state,
        samples.logical_state,
        atol=atol,
    )
    identity_passed = _spectral_integrator_identity_passed(
        state_abs=state_abs,
        state_rel=state_rel,
        errors=errors,
        blocked_reasons=samples.blocked_reasons,
        atol=atol,
        rtol=rtol,
    )
    serial = samples.serial_traces
    logical = samples.logical_traces
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
        free_energy_trace_max_abs_error=errors.free_abs,
        free_energy_trace_max_rel_error=errors.free_rel,
        field_energy_trace_max_abs_error=errors.field_abs,
        field_energy_trace_max_rel_error=errors.field_rel,
        flux_proxy_trace_max_abs_error=errors.flux_abs,
        flux_proxy_trace_max_rel_error=errors.flux_rel,
        serial_free_energy_drift=_trace_drift(serial.free_energy),
        logical_free_energy_drift=_trace_drift(logical.free_energy),
        identity_passed=identity_passed,
        decomposed_path_enabled=identity_passed,
        claim_scope=(
            "diagnostic nonlinear spectral integrator identity gate only; "
            "fixed-step serial-vs-logical-shard RHS routing with final-state "
            "and transport-proxy trace identity, no production distributed FFT "
            "routing or speedup claim"
        ),
        blocked_reasons=samples.blocked_reasons,
        serial_free_energy_trace=serial.free_energy,
        logical_free_energy_trace=logical.free_energy,
        serial_field_energy_trace=serial.field_energy,
        logical_field_energy_trace=logical.field_energy,
        serial_flux_proxy_trace=serial.flux_proxy,
        logical_flux_proxy_trace=logical.flux_proxy,
    )


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

    samples = _run_logical_spectral_window(
        state_hat,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        dt=dt,
        steps=steps,
        atol=atol,
        rtol=rtol,
    )
    errors = _spectral_integrator_trace_errors(
        samples.serial_traces,
        samples.logical_traces,
        floor=atol,
    )
    return _spectral_integrator_report(
        state_shape,
        samples,
        errors,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        steps=steps,
        dt=dt,
        atol=atol,
        rtol=rtol,
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


def _new_spectral_physical_traces() -> dict[str, list[float]]:
    return {
        "free_energy": [],
        "field_energy": [],
        "physical_flux": [],
        "bracket_rms": [],
    }


def _freeze_spectral_physical_traces(
    traces: dict[str, list[float]],
) -> _SpectralPhysicalTraces:
    return _SpectralPhysicalTraces(
        free_energy=tuple(traces["free_energy"]),
        field_energy=tuple(traces["field_energy"]),
        physical_flux=tuple(traces["physical_flux"]),
        bracket_rms=tuple(traces["bracket_rms"]),
    )


def _append_serial_and_pencil_physical_observables(
    serial_traces: dict[str, list[float]],
    pencil_traces: dict[str, list[float]],
    serial_state: jax.Array,
    pencil_state: jax.Array,
) -> None:
    _serial_field, serial_bracket, _serial_rhs = _serial_nonlinear_spectral_rhs(
        serial_state
    )
    _pencil_field, pencil_bracket, _pencil_rhs = _pencil_nonlinear_spectral_rhs(
        pencil_state
    )
    _append_spectral_physical_observables(serial_traces, serial_state, serial_bracket)
    _append_spectral_physical_observables(pencil_traces, pencil_state, pencil_bracket)


def _run_pencil_transport_window(
    state_hat: jax.Array,
    *,
    dt: float,
    steps: int,
) -> _SpectralPhysicalSamples:
    serial_state = state_hat
    pencil_state = state_hat
    serial_trace_lists = _new_spectral_physical_traces()
    pencil_trace_lists = _new_spectral_physical_traces()
    _append_serial_and_pencil_physical_observables(
        serial_trace_lists,
        pencil_trace_lists,
        serial_state,
        pencil_state,
    )

    dt_array = jnp.asarray(float(dt), dtype=jnp.real(state_hat).dtype)
    for _ in range(int(steps)):
        _serial_field, _serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(
            serial_state,
        )
        _pencil_field, _pencil_bracket, pencil_rhs = _pencil_nonlinear_spectral_rhs(
            pencil_state,
        )
        serial_state = serial_state + dt_array * serial_rhs
        pencil_state = pencil_state + dt_array * pencil_rhs
        _append_serial_and_pencil_physical_observables(
            serial_trace_lists,
            pencil_trace_lists,
            serial_state,
            pencil_state,
        )

    return _SpectralPhysicalSamples(
        serial_state=serial_state,
        pencil_state=pencil_state,
        serial_traces=_freeze_spectral_physical_traces(serial_trace_lists),
        pencil_traces=_freeze_spectral_physical_traces(pencil_trace_lists),
    )


def _spectral_physical_trace_errors(
    serial: _SpectralPhysicalTraces,
    pencil: _SpectralPhysicalTraces,
    *,
    floor: float,
) -> _SpectralPhysicalTraceErrors:
    free_abs, free_rel = _relative_trace_error(
        serial.free_energy,
        pencil.free_energy,
        floor=floor,
    )
    field_abs, field_rel = _relative_trace_error(
        serial.field_energy,
        pencil.field_energy,
        floor=floor,
    )
    flux_abs, flux_rel = _relative_trace_error(
        serial.physical_flux,
        pencil.physical_flux,
        floor=floor,
    )
    bracket_abs, bracket_rel = _relative_trace_error(
        serial.bracket_rms,
        pencil.bracket_rms,
        floor=floor,
    )
    return _SpectralPhysicalTraceErrors(
        free_abs=free_abs,
        free_rel=free_rel,
        field_abs=field_abs,
        field_rel=field_rel,
        flux_abs=flux_abs,
        flux_rel=flux_rel,
        bracket_abs=bracket_abs,
        bracket_rel=bracket_rel,
    )


def _pencil_transport_identity_passed(
    *,
    state_abs: float,
    state_rel: float,
    errors: _SpectralPhysicalTraceErrors,
    atol: float,
    rtol: float,
) -> bool:
    return bool(
        state_abs <= float(atol)
        and state_rel <= float(rtol)
        and errors.free_abs <= float(atol)
        and errors.free_rel <= float(rtol)
        and errors.field_abs <= float(atol)
        and errors.field_rel <= float(rtol)
        and errors.flux_abs <= float(atol)
        and errors.flux_rel <= float(rtol)
        and errors.bracket_abs <= float(atol)
        and errors.bracket_rel <= float(rtol)
    )


def _pencil_transport_blockers(
    *,
    identity_passed: bool,
    work_model: NonlinearSpectralPencilWorkModel,
) -> tuple[str, ...]:
    blocked_reasons: list[str] = []
    if not identity_passed:
        blocked_reasons.append("pencil_transport_window_identity_failed")
    blocked_reasons.extend(work_model.feasibility_blockers)
    return tuple(blocked_reasons)


def _pencil_transport_window_report(
    state_shape: tuple[int, int, int, int, int],
    samples: _SpectralPhysicalSamples,
    errors: _SpectralPhysicalTraceErrors,
    *,
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    work_model: NonlinearSpectralPencilWorkModel,
    steps: int,
    dt: float,
    atol: float,
    rtol: float,
) -> NonlinearSpectralPencilTransportWindowReport:
    state_abs, state_rel = _max_abs_rel_error(
        samples.serial_state,
        samples.pencil_state,
        atol=atol,
    )
    identity_passed = _pencil_transport_identity_passed(
        state_abs=state_abs,
        state_rel=state_rel,
        errors=errors,
        atol=atol,
        rtol=rtol,
    )
    decomposed_path_enabled = bool(
        identity_passed and work_model.production_speedup_feasible
    )
    serial = samples.serial_traces
    pencil = samples.pencil_traces
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
        free_energy_trace_max_abs_error=errors.free_abs,
        free_energy_trace_max_rel_error=errors.free_rel,
        field_energy_trace_max_abs_error=errors.field_abs,
        field_energy_trace_max_rel_error=errors.field_rel,
        physical_flux_trace_max_abs_error=errors.flux_abs,
        physical_flux_trace_max_rel_error=errors.flux_rel,
        bracket_rms_trace_max_abs_error=errors.bracket_abs,
        bracket_rms_trace_max_rel_error=errors.bracket_rel,
        serial_free_energy_drift=_trace_drift(serial.free_energy),
        pencil_free_energy_drift=_trace_drift(pencil.free_energy),
        identity_passed=identity_passed,
        decomposed_path_enabled=decomposed_path_enabled,
        work_model=work_model,
        claim_scope=(
            "diagnostic serial-vs-pencil nonlinear physical-space transport-window "
            "identity gate; includes a density-times-radial-E-field transport "
            "proxy, but is not an absolute nonlinear turbulent heat-flux claim"
        ),
        blocked_reasons=_pencil_transport_blockers(
            identity_passed=identity_passed,
            work_model=work_model,
        ),
        serial_free_energy_trace=serial.free_energy,
        pencil_free_energy_trace=pencil.free_energy,
        serial_field_energy_trace=serial.field_energy,
        pencil_field_energy_trace=pencil.field_energy,
        serial_physical_flux_trace=serial.physical_flux,
        pencil_physical_flux_trace=pencil.physical_flux,
        serial_bracket_rms_trace=serial.bracket_rms,
        pencil_bracket_rms_trace=pencil.bracket_rms,
    )


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
    """Validate a serial-vs-pencil nonlinear transport window."""

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
    samples = _run_pencil_transport_window(state_hat, dt=dt, steps=steps)
    errors = _spectral_physical_trace_errors(
        samples.serial_traces,
        samples.pencil_traces,
        floor=atol,
    )
    return _pencil_transport_window_report(
        state_shape,
        samples,
        errors,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        work_model=work_model,
        steps=steps,
        dt=dt,
        atol=atol,
        rtol=rtol,
    )


__all__ = [
    "_append_spectral_observables",
    "_spectral_integrator_observables",
    "integrate_logical_decomposed_nonlinear_spectral",
    "nonlinear_spectral_pencil_transport_window_identity_gate",
    "nonlinear_spectral_integrator_identity_gate",
]
