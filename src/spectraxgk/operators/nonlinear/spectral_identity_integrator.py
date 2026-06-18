"""Fixed-window nonlinear spectral integrator identity gates."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectraxgk.operators.nonlinear.domain_decomposition import (
    _relative_trace_error,
    _trace_drift,
)
from spectraxgk.operators.nonlinear.parallel_contracts_spectral import (
    NonlinearSpectralIntegratorIdentityReport,
)
from spectraxgk.operators.nonlinear.spectral_core import (
    _chunk_offsets,
    _field_from_state,
    _max_abs_rel_error,
    _serial_nonlinear_spectral_rhs,
    _spectral_bracket,
    _spectral_rhs_from_bracket,
    _spectral_tile_bounds,
    _validate_chunks,
    _validate_spectral_state_shape,
)
from spectraxgk.operators.nonlinear.spectral_identity_rhs import (
    logical_decomposed_nonlinear_spectral_rhs,
)


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


__all__ = [
    "_append_spectral_observables",
    "_spectral_integrator_observables",
    "integrate_logical_decomposed_nonlinear_spectral",
    "nonlinear_spectral_integrator_identity_gate",
]
