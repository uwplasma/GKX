"""Device-z nonlinear spectral parallel routes and identity gates."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.operators.nonlinear.parallel_contracts import (
    NonlinearSpectralDevicePencilRHSIdentityReport,
    NonlinearSpectralDevicePencilTransportWindowReport,
)
from spectraxgk.operators.nonlinear.domain_decomposition import _relative_trace_error, _trace_drift
from spectraxgk.operators.nonlinear.spectral_core import (
    _field_from_state,
    _host_max_abs_rel_error,
    _host_staged_array_for_sharding,
    _pencil_ifft2,
    _pencil_nonlinear_spectral_rhs,
    _pencil_nonlinear_spectral_rhs_z_chunked,
    _serial_nonlinear_spectral_rhs,
    _spectral_wave_numbers,
    _validate_spectral_state_shape,
    _within_abs_or_rel_tolerance,
)

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

def _device_z_pencil_shard_map_rhs_fn(  # pragma: no cover - exercised by profile artifacts.
    mesh: Any,
    *,
    axis_name: str,
    z_chunk_size: int | None = None,
) -> Any:
    """Return a jitted shard-map RHS for z-sharded spectral states."""

    from jax.sharding import PartitionSpec

    def _local_rhs(local_state: jax.Array) -> jax.Array:
        if z_chunk_size is not None:
            return _pencil_nonlinear_spectral_rhs_z_chunked(
                local_state,
                z_chunk_size=int(z_chunk_size),
            )[2]
        return _pencil_nonlinear_spectral_rhs(local_state)[2]

    state_spec = PartitionSpec(None, None, None, None, axis_name)
    return jax.jit(
        jax.shard_map(
            _local_rhs,
            mesh=mesh,
            in_specs=state_spec,
            out_specs=state_spec,
            check_vma=False,
        )
    )

def _spectral_physical_transport_observable_sums(
    state_hat: jax.Array,
    bracket_hat: jax.Array,
) -> jax.Array:
    """Return additive physical-space observable sums for local z slabs."""

    _nl, _nm, ny, nx, _nz = _validate_spectral_state_shape(tuple(state_hat.shape))
    real_dtype = jnp.real(state_hat).dtype
    ky, _kx = _spectral_wave_numbers(ny, nx, real_dtype)
    field = _field_from_state(state_hat)
    density_hat = jnp.sum(state_hat[:, 0, :, :, :], axis=0)
    density_xy = _pencil_ifft2(density_hat, y_axis=0, x_axis=1)
    phi_y = _pencil_ifft2(1j * ky[:, None, None] * field, y_axis=0, x_axis=1)
    flux_density = jnp.abs(jnp.real(jnp.conj(density_xy) * (-phi_y)))
    bracket_abs2 = jnp.abs(bracket_hat) ** 2
    return jnp.asarray(
        [
            jnp.sum(jnp.abs(state_hat) ** 2),
            jnp.sum(jnp.abs(field) ** 2),
            jnp.sum(flux_density),
            jnp.asarray(flux_density.size, dtype=real_dtype),
            jnp.sum(bracket_abs2),
            jnp.asarray(bracket_abs2.size, dtype=real_dtype),
        ],
        dtype=real_dtype,
    )

def _spectral_physical_transport_observable_vector_from_sums(
    sums: jax.Array,
) -> jax.Array:
    """Convert additive observable sums into ``[Wg, Wphi, Q, bracket_rms]``."""

    real_dtype = sums.dtype
    flux_count = jnp.maximum(sums[3], jnp.asarray(1.0, dtype=real_dtype))
    bracket_count = jnp.maximum(sums[5], jnp.asarray(1.0, dtype=real_dtype))
    return jnp.asarray(
        [
            sums[0],
            sums[1],
            sums[2] / flux_count,
            jnp.sqrt(sums[4] / bracket_count),
        ],
        dtype=real_dtype,
    )

def _device_z_pencil_shard_map_observables_fn(  # pragma: no cover - exercised by profile artifacts.
    mesh: Any,
    *,
    axis_name: str,
    z_chunk_size: int | None = None,
) -> Any:
    """Return a shard-map scalar observable reducer for z-sharded states."""

    from jax.sharding import PartitionSpec

    def _local_observables(local_state: jax.Array) -> jax.Array:
        if z_chunk_size is not None:
            _field, bracket, _rhs = _pencil_nonlinear_spectral_rhs_z_chunked(
                local_state,
                z_chunk_size=int(z_chunk_size),
            )
        else:
            _field, bracket, _rhs = _pencil_nonlinear_spectral_rhs(local_state)
        local_sums = _spectral_physical_transport_observable_sums(local_state, bracket)
        global_sums = jax.lax.psum(local_sums, axis_name)
        return _spectral_physical_transport_observable_vector_from_sums(global_sums)

    state_spec = PartitionSpec(None, None, None, None, axis_name)
    return jax.jit(
        jax.shard_map(
            _local_observables,
            mesh=mesh,
            in_specs=state_spec,
            out_specs=PartitionSpec(),
            check_vma=False,
        )
    )

def device_z_pencil_nonlinear_spectral_rhs(
    state_hat: jax.Array,
    *,
    devices: Sequence[Any] | None = None,
    axis_name: str = "z",
    z_chunk_size: int | None = None,
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
    _serial_field, _serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(
        state_hat
    )
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

    with mesh:  # pragma: no cover - exercised by CPU/GPU profile artifacts.
        sharded_rhs_fn = _device_z_pencil_shard_map_rhs_fn(
            mesh,
            axis_name=axis_name,
            z_chunk_size=z_chunk_size,
        )
        sharded_state = jax.device_put(
            _host_staged_array_for_sharding(state_hat),
            sharding,
        )
        candidate_rhs = sharded_rhs_fn(sharded_state)

    rhs_abs, rhs_rel = _host_max_abs_rel_error(serial_rhs, candidate_rhs, atol=atol)
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
            "device z-sharded shard_map fused pencil nonlinear RHS identity gate; "
            "FFT axes remain local per device and no global spectral reconstruction "
            "is used, host-gathered RHS identity is required, and no speedup claim "
            "is allowed without matched profiler gates"
        ),
        blocked_reasons=tuple(blockers_list),
    )
    gated_rhs = candidate_rhs if report.decomposed_path_enabled else serial_rhs
    return gated_rhs, report

def device_z_pencil_nonlinear_spectral_transport_window_identity_gate(
    state_hat: jax.Array,
    *,
    devices: Sequence[Any] | None = None,
    axis_name: str = "z",
    z_chunk_size: int | None = None,
    dt: float = 0.005,
    steps: int = 4,
    atol: float = 5.0e-6,
    rtol: float = 1.0e-4,
    observable_mode: Literal["host_gather", "sharded_reduce"] = "host_gather",
) -> NonlinearSpectralDevicePencilTransportWindowReport:
    """Validate a multi-step serial-vs-device-z-sharded nonlinear window.

    The route advances the same explicit fixed-step micro-window with the
    serial nonlinear RHS and the shard-map z-pencil RHS. It compares the final
    state and physical-space scalar traces, including a density-times-radial
    electric-field flux proxy and bracket RMS. Passing this gate is still not a
    turbulent heat-flux validation; it only permits timing the decomposed
    device route on the same deterministic operator.
    """

    if int(steps) < 1:
        raise ValueError("steps must be at least one")
    if observable_mode not in {"host_gather", "sharded_reduce"}:
        raise ValueError("observable_mode must be 'host_gather' or 'sharded_reduce'")
    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    mesh, sharding, blockers, requested_count, active_count = (
        _device_z_sharding_for_spectral_state(
            state_hat,
            devices=devices,
            axis_name=axis_name,
        )
    )

    serial_traces: dict[str, list[float]] = {
        "free_energy": [],
        "field_energy": [],
        "physical_flux": [],
        "bracket_rms": [],
    }
    device_traces: dict[str, list[float]] = {
        "free_energy": [],
        "field_energy": [],
        "physical_flux": [],
        "bracket_rms": [],
    }

    serial_state = state_hat
    _serial_field, serial_bracket, _serial_rhs = _serial_nonlinear_spectral_rhs(
        serial_state
    )
    _append_spectral_physical_observables(serial_traces, serial_state, serial_bracket)

    blocked_reasons = list(blockers)
    if blockers or mesh is None or sharding is None:
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
            serial_free_energy_drift=_trace_drift(tuple(serial_traces["free_energy"])),
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
            serial_free_energy_trace=tuple(serial_traces["free_energy"]),
            serial_field_energy_trace=tuple(serial_traces["field_energy"]),
            serial_physical_flux_trace=tuple(serial_traces["physical_flux"]),
            serial_bracket_rms_trace=tuple(serial_traces["bracket_rms"]),
        )

    dt_array = jnp.asarray(float(dt), dtype=jnp.real(state_hat).dtype)
    with mesh:  # pragma: no cover - exercised by CPU/GPU profile artifacts.
        sharded_rhs_fn = _device_z_pencil_shard_map_rhs_fn(
            mesh,
            axis_name=axis_name,
            z_chunk_size=z_chunk_size,
        )
        sharded_observables_fn = _device_z_pencil_shard_map_observables_fn(
            mesh,
            axis_name=axis_name,
            z_chunk_size=z_chunk_size,
        )
        device_state = jax.device_put(
            _host_staged_array_for_sharding(state_hat),
            sharding,
        )
        if observable_mode == "sharded_reduce":
            _append_spectral_physical_observable_vector(
                device_traces,
                sharded_observables_fn(device_state),
            )
        else:
            device_state_for_observables = jnp.asarray(jax.device_get(device_state))
            _device_field, device_bracket, _device_rhs = _serial_nonlinear_spectral_rhs(
                device_state_for_observables,
            )
            _append_spectral_physical_observables(
                device_traces,
                device_state_for_observables,
                device_bracket,
            )

        for _ in range(int(steps)):
            _serial_field, _serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(
                serial_state,
            )
            serial_state = serial_state + dt_array * serial_rhs
            device_rhs = sharded_rhs_fn(device_state)
            device_state = device_state + dt_array * device_rhs

            _serial_field, serial_bracket, _serial_rhs = _serial_nonlinear_spectral_rhs(
                serial_state,
            )
            _append_spectral_physical_observables(
                serial_traces,
                serial_state,
                serial_bracket,
            )
            if observable_mode == "sharded_reduce":
                _append_spectral_physical_observable_vector(
                    device_traces,
                    sharded_observables_fn(device_state),
                )
            else:
                device_state_for_observables = jnp.asarray(jax.device_get(device_state))
                _device_field, device_bracket, _device_rhs = (
                    _serial_nonlinear_spectral_rhs(
                        device_state_for_observables,
                    )
                )
                _append_spectral_physical_observables(
                    device_traces,
                    device_state_for_observables,
                    device_bracket,
                )

    device_final_state = jnp.asarray(jax.device_get(device_state))
    state_abs, state_rel = _host_max_abs_rel_error(
        serial_state,
        device_final_state,
        atol=atol,
    )
    serial_free = tuple(serial_traces["free_energy"])
    device_free = tuple(device_traces["free_energy"])
    serial_field_energy = tuple(serial_traces["field_energy"])
    device_field_energy = tuple(device_traces["field_energy"])
    serial_physical_flux = tuple(serial_traces["physical_flux"])
    device_physical_flux = tuple(device_traces["physical_flux"])
    serial_bracket_rms = tuple(serial_traces["bracket_rms"])
    device_bracket_rms = tuple(device_traces["bracket_rms"])
    free_abs, free_rel = _relative_trace_error(serial_free, device_free, floor=atol)
    field_abs, field_rel = _relative_trace_error(
        serial_field_energy,
        device_field_energy,
        floor=atol,
    )
    flux_abs, flux_rel = _relative_trace_error(
        serial_physical_flux,
        device_physical_flux,
        floor=atol,
    )
    bracket_abs, bracket_rel = _relative_trace_error(
        serial_bracket_rms,
        device_bracket_rms,
        floor=atol,
    )
    identity_passed = bool(
        _within_abs_or_rel_tolerance(state_abs, state_rel, atol=atol, rtol=rtol)
        and _within_abs_or_rel_tolerance(free_abs, free_rel, atol=atol, rtol=rtol)
        and _within_abs_or_rel_tolerance(field_abs, field_rel, atol=atol, rtol=rtol)
        and _within_abs_or_rel_tolerance(flux_abs, flux_rel, atol=atol, rtol=rtol)
        and _within_abs_or_rel_tolerance(bracket_abs, bracket_rel, atol=atol, rtol=rtol)
    )
    if not identity_passed:
        blocked_reasons.append("device_z_pencil_transport_window_identity_failed")

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
        serial_free_energy_drift=_trace_drift(serial_free),
        device_free_energy_drift=_trace_drift(device_free),
        identity_passed=identity_passed,
        device_sharding_active=True,
        decomposed_path_enabled=identity_passed,
        claim_scope=(
            "device z-sharded shard_map nonlinear transport-window identity gate; "
            "compares serial and sharded final state plus free-energy, field-energy, "
            "physical-flux, and bracket-RMS traces before any profiler-backed "
            "speedup claim is allowed"
        ),
        blocked_reasons=tuple(sorted(set(blocked_reasons))),
        serial_free_energy_trace=serial_free,
        device_free_energy_trace=device_free,
        serial_field_energy_trace=serial_field_energy,
        device_field_energy_trace=device_field_energy,
        serial_physical_flux_trace=serial_physical_flux,
        device_physical_flux_trace=device_physical_flux,
        serial_bracket_rms_trace=serial_bracket_rms,
        device_bracket_rms_trace=device_bracket_rms,
    )

def _spectral_physical_transport_observables(
    state_hat: jax.Array,
    bracket_hat: jax.Array,
) -> tuple[float, float, float, float]:
    """Return physical-space transport-window observables for the micro-route."""

    values = _spectral_physical_transport_observable_vector_from_sums(
        _spectral_physical_transport_observable_sums(state_hat, bracket_hat)
    )
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
    )

def _append_spectral_physical_observable_vector(
    traces: dict[str, list[float]],
    values: jax.Array,
) -> None:
    """Append ``[Wg, Wphi, Q, bracket_rms]`` scalar observables."""

    traces["free_energy"].append(float(values[0]))
    traces["field_energy"].append(float(values[1]))
    traces["physical_flux"].append(float(values[2]))
    traces["bracket_rms"].append(float(values[3]))

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

__all__ = [
    "_append_spectral_physical_observable_vector",
    "_append_spectral_physical_observables",
    "_device_z_pencil_shard_map_observables_fn",
    "_device_z_pencil_shard_map_rhs_fn",
    "_device_z_sharding_for_spectral_state",
    "_spectral_physical_transport_observable_sums",
    "_spectral_physical_transport_observable_vector_from_sums",
    "_spectral_physical_transport_observables",
    "device_z_pencil_nonlinear_spectral_rhs",
    "device_z_pencil_nonlinear_spectral_transport_window_identity_gate",
]
