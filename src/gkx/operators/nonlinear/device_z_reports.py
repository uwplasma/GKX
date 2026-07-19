"""Fail-closed reports for device-z nonlinear spectral routing."""

from __future__ import annotations

from typing import Mapping, Sequence

from gkx.operators.nonlinear.domain_decomposition import (
    _relative_trace_error,
    _trace_drift,
)
from gkx.operators.nonlinear.parallel_contracts_spectral import (
    NonlinearSpectralDevicePencilRHSIdentityReport,
    NonlinearSpectralDevicePencilTransportWindowReport,
)
from gkx.operators.nonlinear.spectral_core import _within_abs_or_rel_tolerance

_TRANSPORT_TRACE_KEYS = (
    "free_energy",
    "field_energy",
    "physical_flux",
    "bracket_rms",
)


def _new_transport_trace_dict() -> dict[str, list[float]]:
    """Return empty scalar traces used by nonlinear transport-window gates."""

    return {key: [] for key in _TRANSPORT_TRACE_KEYS}


def _transport_trace_tuples(
    traces: dict[str, list[float]],
) -> dict[str, tuple[float, ...]]:
    """Freeze mutable transport traces for report assembly."""

    return {key: tuple(traces[key]) for key in _TRANSPORT_TRACE_KEYS}


def _transport_trace_error_pairs(
    serial: dict[str, tuple[float, ...]],
    device: dict[str, tuple[float, ...]],
    *,
    floor: float,
) -> dict[str, tuple[float, float]]:
    """Return max absolute/relative errors for each transport trace."""

    return {
        key: _relative_trace_error(serial[key], device[key], floor=floor)
        for key in _TRANSPORT_TRACE_KEYS
    }


def _device_z_transport_identity_passed(
    *,
    state_abs: float,
    state_rel: float,
    trace_errors: Mapping[str, tuple[float, float]],
    atol: float,
    rtol: float,
) -> bool:
    """Return whether final state and all transport traces pass identity gates."""

    tolerances = [
        (state_abs, state_rel),
        *(trace_errors[key] for key in _TRANSPORT_TRACE_KEYS),
    ]
    return bool(
        all(
            _within_abs_or_rel_tolerance(abs_err, rel_err, atol=atol, rtol=rtol)
            for abs_err, rel_err in tolerances
        )
    )


def _blocked_device_z_transport_window_report(
    *,
    state_shape: tuple[int, int, int, int, int],
    axis_name: str,
    requested_count: int,
    active_count: int,
    steps: int,
    dt: float,
    atol: float,
    rtol: float,
    blocked_reasons: Sequence[str],
    serial_traces: dict[str, list[float]],
) -> NonlinearSpectralDevicePencilTransportWindowReport:
    """Return a fail-closed device-z transport-window report."""

    serial = _transport_trace_tuples(serial_traces)
    return NonlinearSpectralDevicePencilTransportWindowReport(
        state_shape=state_shape,
        sharded_axis="z",
        axis_name=str(axis_name),
        requested_device_count=int(requested_count),
        active_device_count=int(active_count),
        steps=int(steps),
        dt=float(dt),
        atol=float(atol),
        rtol=float(rtol),
        final_state_max_abs_error=float("inf"),
        final_state_max_rel_error=float("inf"),
        free_energy_trace_max_abs_error=float("inf"),
        free_energy_trace_max_rel_error=float("inf"),
        field_energy_trace_max_abs_error=float("inf"),
        field_energy_trace_max_rel_error=float("inf"),
        physical_flux_trace_max_abs_error=float("inf"),
        physical_flux_trace_max_rel_error=float("inf"),
        bracket_rms_trace_max_abs_error=float("inf"),
        bracket_rms_trace_max_rel_error=float("inf"),
        serial_free_energy_drift=_trace_drift(serial["free_energy"]),
        device_free_energy_drift=0.0,
        identity_passed=False,
        device_sharding_active=False,
        decomposed_path_enabled=False,
        claim_scope=(
            "device z-sharded shard_map nonlinear transport-window gate; "
            "skipped because the requested local device topology cannot "
            "support the z shard"
        ),
        blocked_reasons=tuple(blocked_reasons),
        serial_free_energy_trace=serial["free_energy"],
        serial_field_energy_trace=serial["field_energy"],
        serial_physical_flux_trace=serial["physical_flux"],
        serial_bracket_rms_trace=serial["bracket_rms"],
    )


def _blocked_device_z_rhs_report(
    *,
    state_shape: tuple[int, int, int, int, int],
    axis_name: str,
    requested_count: int,
    active_count: int,
    atol: float,
    rtol: float,
    blocked_reasons: Sequence[str],
) -> NonlinearSpectralDevicePencilRHSIdentityReport:
    """Return a fail-closed device-z RHS identity report."""

    return NonlinearSpectralDevicePencilRHSIdentityReport(
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
        blocked_reasons=tuple(blocked_reasons),
    )


def _device_z_rhs_identity_report(
    *,
    state_shape: tuple[int, int, int, int, int],
    axis_name: str,
    requested_count: int,
    active_count: int,
    atol: float,
    rtol: float,
    rhs_abs: float,
    rhs_rel: float,
) -> NonlinearSpectralDevicePencilRHSIdentityReport:
    """Return the passed/blocked device-z RHS identity report."""

    identity_passed = bool(rhs_abs <= float(atol) and rhs_rel <= float(rtol))
    blocked_reasons = () if identity_passed else ("device_z_pencil_rhs_identity_failed",)
    return NonlinearSpectralDevicePencilRHSIdentityReport(
        state_shape=state_shape,
        sharded_axis="z",
        axis_name=str(axis_name),
        requested_device_count=int(requested_count),
        active_device_count=int(active_count),
        atol=float(atol),
        rtol=float(rtol),
        rhs_max_abs_error=float(rhs_abs),
        rhs_max_rel_error=float(rhs_rel),
        identity_passed=identity_passed,
        device_sharding_active=True,
        decomposed_path_enabled=identity_passed,
        claim_scope=(
            "device z-sharded shard_map fused pencil nonlinear RHS identity gate; "
            "FFT axes remain local per device and no global spectral reconstruction "
            "is used, host-gathered RHS identity is required, and no speedup claim "
            "is allowed without matched profiler gates"
        ),
        blocked_reasons=blocked_reasons,
    )


def _device_z_transport_window_report(
    *,
    state_shape: tuple[int, int, int, int, int],
    axis_name: str,
    requested_count: int,
    active_count: int,
    steps: int,
    dt: float,
    atol: float,
    rtol: float,
    state_abs: float,
    state_rel: float,
    serial_trace_values: dict[str, tuple[float, ...]],
    device_trace_values: dict[str, tuple[float, ...]],
    blocked_reasons: Sequence[str],
) -> NonlinearSpectralDevicePencilTransportWindowReport:
    """Return the passed/blocked device-z transport-window report."""

    trace_errors = _transport_trace_error_pairs(
        serial_trace_values,
        device_trace_values,
        floor=atol,
    )
    free_abs, free_rel = trace_errors["free_energy"]
    field_abs, field_rel = trace_errors["field_energy"]
    flux_abs, flux_rel = trace_errors["physical_flux"]
    bracket_abs, bracket_rel = trace_errors["bracket_rms"]
    identity_passed = _device_z_transport_identity_passed(
        state_abs=state_abs,
        state_rel=state_rel,
        trace_errors=trace_errors,
        atol=atol,
        rtol=rtol,
    )
    report_blockers = list(blocked_reasons)
    if not identity_passed:
        report_blockers.append("device_z_pencil_transport_window_identity_failed")

    return NonlinearSpectralDevicePencilTransportWindowReport(
        state_shape=state_shape,
        sharded_axis="z",
        axis_name=str(axis_name),
        requested_device_count=int(requested_count),
        active_device_count=int(active_count),
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
        serial_free_energy_drift=_trace_drift(serial_trace_values["free_energy"]),
        device_free_energy_drift=_trace_drift(device_trace_values["free_energy"]),
        identity_passed=identity_passed,
        device_sharding_active=True,
        decomposed_path_enabled=identity_passed,
        claim_scope=(
            "device z-sharded shard_map nonlinear transport-window identity gate; "
            "compares serial and sharded final state plus free-energy, field-energy, "
            "physical-flux, and bracket-RMS traces before any profiler-backed "
            "speedup claim is allowed"
        ),
        blocked_reasons=tuple(sorted(set(report_blockers))),
        serial_free_energy_trace=serial_trace_values["free_energy"],
        device_free_energy_trace=device_trace_values["free_energy"],
        serial_field_energy_trace=serial_trace_values["field_energy"],
        device_field_energy_trace=device_trace_values["field_energy"],
        serial_physical_flux_trace=serial_trace_values["physical_flux"],
        device_physical_flux_trace=device_trace_values["physical_flux"],
        serial_bracket_rms_trace=serial_trace_values["bracket_rms"],
        device_bracket_rms_trace=device_trace_values["bracket_rms"],
    )


__all__ = [
    "_blocked_device_z_rhs_report",
    "_blocked_device_z_transport_window_report",
    "_device_z_rhs_identity_report",
    "_device_z_transport_identity_passed",
    "_device_z_transport_window_report",
    "_new_transport_trace_dict",
    "_transport_trace_error_pairs",
    "_transport_trace_tuples",
]
