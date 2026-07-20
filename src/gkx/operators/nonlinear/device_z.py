"""Device-z nonlinear spectral parallel routes and identity gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from gkx.operators.nonlinear.parallel_contracts_spectral import (
    NonlinearSpectralDevicePencilRHSIdentityReport,
    NonlinearSpectralDevicePencilTransportWindowReport,
)
from gkx.operators.nonlinear.device_z_reports import (
    _blocked_device_z_rhs_report,
    _blocked_device_z_transport_window_report,
    _device_z_rhs_identity_report,
    _device_z_transport_identity_passed,
    _device_z_transport_window_report,
    _new_transport_trace_dict,
    _transport_trace_error_pairs,
    _transport_trace_tuples,
)
from gkx.operators.nonlinear.spectral_core import (
    _field_from_state,
    _host_max_abs_rel_error,
    _host_staged_array_for_sharding,
    _pencil_ifft2,
    _pencil_nonlinear_spectral_rhs,
    _pencil_nonlinear_spectral_rhs_z_chunked,
    _serial_nonlinear_spectral_rhs,
    _spectral_wave_numbers,
    _validate_spectral_state_shape,
)


@dataclass(frozen=True)
class _DeviceZShardingSetup:
    state_shape: tuple[int, int, int, int, int]
    mesh: Any | None
    sharding: Any | None
    blockers: tuple[str, ...]
    requested_count: int
    active_count: int


@dataclass(frozen=True)
class _DeviceZTransportSamples:
    serial_state: jax.Array
    device_state: jax.Array
    serial_traces: dict[str, tuple[float, ...]]
    device_traces: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class _DeviceZComputeStates:
    serial_state: jax.Array
    device_state: jax.Array


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


def _device_z_sharding_setup(
    state_hat: jax.Array,
    *,
    devices: Sequence[Any] | None,
    axis_name: str,
) -> _DeviceZShardingSetup:
    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    mesh, sharding, blockers, requested_count, active_count = (
        _device_z_sharding_for_spectral_state(
            state_hat,
            devices=devices,
            axis_name=axis_name,
        )
    )
    return _DeviceZShardingSetup(
        state_shape=state_shape,
        mesh=mesh,
        sharding=sharding,
        blockers=blockers,
        requested_count=requested_count,
        active_count=active_count,
    )


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

def _append_device_z_transport_observables(
    traces: dict[str, list[float]],
    state_hat: jax.Array,
    *,
    observable_mode: Literal["host_gather", "sharded_reduce"],
    sharded_observables_fn: Any | None,
) -> None:
    """Append device-z transport observables using the selected reduction route."""

    if observable_mode == "sharded_reduce":
        if sharded_observables_fn is None:
            raise ValueError("sharded observable reducer is required")
        _append_spectral_physical_observable_vector(
            traces,
            sharded_observables_fn(state_hat),
        )
        return

    state_for_observables = jnp.asarray(jax.device_get(state_hat))
    _field, bracket, _rhs = _serial_nonlinear_spectral_rhs(state_for_observables)
    _append_spectral_physical_observables(
        traces,
        state_for_observables,
        bracket,
    )


def _serial_nonlinear_rhs(state_hat: jax.Array) -> jax.Array:
    _serial_field, _serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(
        state_hat
    )
    return serial_rhs


def _blocked_device_z_rhs_if_needed(
    setup: _DeviceZShardingSetup,
    *,
    axis_name: str,
    atol: float,
    rtol: float,
) -> NonlinearSpectralDevicePencilRHSIdentityReport | None:
    if not setup.blockers and setup.mesh is not None and setup.sharding is not None:
        return None
    return _blocked_device_z_rhs_report(
        state_shape=setup.state_shape,
        axis_name=axis_name,
        requested_count=setup.requested_count,
        active_count=setup.active_count,
        atol=atol,
        rtol=rtol,
        blocked_reasons=setup.blockers,
    )


def _run_device_z_sharded_rhs(
    state_hat: jax.Array,
    setup: _DeviceZShardingSetup,
    *,
    axis_name: str,
    z_chunk_size: int | None,
) -> jax.Array:
    if setup.mesh is None or setup.sharding is None:
        raise ValueError("device-z sharding setup is not active")
    with setup.mesh:  # pragma: no cover - exercised by CPU/GPU profile artifacts.
        sharded_rhs_fn = _device_z_pencil_shard_map_rhs_fn(
            setup.mesh,
            axis_name=axis_name,
            z_chunk_size=z_chunk_size,
        )
        sharded_state = jax.device_put(
            _host_staged_array_for_sharding(state_hat),
            setup.sharding,
        )
        return sharded_rhs_fn(sharded_state)


def _device_z_rhs_report(
    setup: _DeviceZShardingSetup,
    serial_rhs: jax.Array,
    candidate_rhs: jax.Array,
    *,
    axis_name: str,
    atol: float,
    rtol: float,
) -> NonlinearSpectralDevicePencilRHSIdentityReport:
    rhs_abs, rhs_rel = _host_max_abs_rel_error(serial_rhs, candidate_rhs, atol=atol)
    return _device_z_rhs_identity_report(
        state_shape=setup.state_shape,
        axis_name=axis_name,
        requested_count=setup.requested_count,
        active_count=setup.active_count,
        atol=atol,
        rtol=rtol,
        rhs_abs=rhs_abs,
        rhs_rel=rhs_rel,
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

    setup = _device_z_sharding_setup(state_hat, devices=devices, axis_name=axis_name)
    serial_rhs = _serial_nonlinear_rhs(state_hat)
    if report := _blocked_device_z_rhs_if_needed(
        setup,
        axis_name=axis_name,
        atol=atol,
        rtol=rtol,
    ):
        return serial_rhs, report

    candidate_rhs = _run_device_z_sharded_rhs(
        state_hat,
        setup,
        axis_name=axis_name,
        z_chunk_size=z_chunk_size,
    )
    report = _device_z_rhs_report(
        setup,
        serial_rhs,
        candidate_rhs,
        axis_name=axis_name,
        atol=atol,
        rtol=rtol,
    )
    gated_rhs = candidate_rhs if report.decomposed_path_enabled else serial_rhs
    return gated_rhs, report


def _validate_device_z_transport_window(
    *,
    steps: int,
    observable_mode: Literal["host_gather", "sharded_reduce"],
) -> None:
    if int(steps) < 1:
        raise ValueError("steps must be at least one")
    if observable_mode not in {"host_gather", "sharded_reduce"}:
        raise ValueError("observable_mode must be 'host_gather' or 'sharded_reduce'")


def _append_serial_transport_observables(
    traces: dict[str, list[float]],
    serial_state: jax.Array,
) -> None:
    _serial_field, serial_bracket, _serial_rhs = _serial_nonlinear_spectral_rhs(
        serial_state
    )
    _append_spectral_physical_observables(traces, serial_state, serial_bracket)


def _initial_serial_transport_traces(state_hat: jax.Array) -> dict[str, list[float]]:
    serial_traces = _new_transport_trace_dict()
    _append_serial_transport_observables(serial_traces, state_hat)
    return serial_traces


def _blocked_device_z_transport_if_needed(
    setup: _DeviceZShardingSetup,
    *,
    axis_name: str,
    steps: int,
    dt: float,
    atol: float,
    rtol: float,
    serial_traces: dict[str, list[float]],
) -> NonlinearSpectralDevicePencilTransportWindowReport | None:
    if not setup.blockers and setup.mesh is not None and setup.sharding is not None:
        return None
    return _blocked_device_z_transport_window_report(
        state_shape=setup.state_shape,
        axis_name=axis_name,
        requested_count=setup.requested_count,
        active_count=setup.active_count,
        steps=steps,
        dt=dt,
        atol=atol,
        rtol=rtol,
        blocked_reasons=list(setup.blockers),
        serial_traces=serial_traces,
    )


def _run_device_z_transport_window(
    state_hat: jax.Array,
    setup: _DeviceZShardingSetup,
    *,
    axis_name: str,
    z_chunk_size: int | None,
    dt: float,
    steps: int,
    observable_mode: Literal["host_gather", "sharded_reduce"],
) -> _DeviceZTransportSamples:
    if setup.mesh is None or setup.sharding is None:
        raise ValueError("device-z sharding setup is not active")

    compute_states = _run_device_z_compute_window_states(
        state_hat,
        setup,
        axis_name=axis_name,
        z_chunk_size=z_chunk_size,
        dt=dt,
        steps=steps,
    )
    serial_traces = _initial_serial_transport_traces(state_hat)
    device_traces = _new_transport_trace_dict()
    serial_state = state_hat
    dt_array = jnp.asarray(float(dt), dtype=jnp.real(state_hat).dtype)
    with setup.mesh:  # pragma: no cover - exercised by CPU/GPU profile artifacts.
        sharded_rhs_fn = _device_z_pencil_shard_map_rhs_fn(
            setup.mesh,
            axis_name=axis_name,
            z_chunk_size=z_chunk_size,
        )
        sharded_observables_fn = _device_z_pencil_shard_map_observables_fn(
            setup.mesh,
            axis_name=axis_name,
            z_chunk_size=z_chunk_size,
        )
        device_state = jax.device_put(
            _host_staged_array_for_sharding(state_hat),
            setup.sharding,
        )
        _append_device_z_transport_observables(
            device_traces,
            device_state,
            observable_mode=observable_mode,
            sharded_observables_fn=sharded_observables_fn,
        )
        for _ in range(int(steps)):
            serial_state = serial_state + dt_array * _serial_nonlinear_rhs(serial_state)
            device_state = device_state + dt_array * sharded_rhs_fn(device_state)
            _append_serial_transport_observables(serial_traces, serial_state)
            _append_device_z_transport_observables(
                device_traces,
                device_state,
                observable_mode=observable_mode,
                sharded_observables_fn=sharded_observables_fn,
            )

    return _DeviceZTransportSamples(
        serial_state=compute_states.serial_state,
        device_state=compute_states.device_state,
        serial_traces=_transport_trace_tuples(serial_traces),
        device_traces=_transport_trace_tuples(device_traces),
    )


def _serial_transport_compute_state(
    state_hat: jax.Array,
    *,
    dt: float,
    steps: int,
) -> jax.Array:
    """Return the compute-only serial fixed-window state used for timing."""

    dt_array = jnp.asarray(float(dt), dtype=jnp.real(state_hat).dtype)

    def _route(item: jax.Array) -> jax.Array:
        out = item
        for _ in range(int(steps)):
            out = out + dt_array * _serial_nonlinear_rhs(out)
        return out

    return jax.jit(_route)(state_hat)


def _run_device_z_compute_window_states(
    state_hat: jax.Array,
    setup: _DeviceZShardingSetup,
    *,
    axis_name: str,
    z_chunk_size: int | None,
    dt: float,
    steps: int,
) -> _DeviceZComputeStates:
    """Return final states from the compute-only serial and z-sharded routes.

    The transport-window gate also collects scalar traces with host-visible
    instrumentation. Final-state identity must match the compute-only route
    that the profiler times, otherwise per-step diagnostics can create a
    different numerical path than the speedup candidate.
    """

    if setup.mesh is None or setup.sharding is None:
        raise ValueError("device-z sharding setup is not active")
    dt_array = jnp.asarray(float(dt), dtype=jnp.real(state_hat).dtype)
    serial_state = _serial_transport_compute_state(
        state_hat,
        dt=dt,
        steps=steps,
    )
    with setup.mesh:  # pragma: no cover - exercised by CPU/GPU profile artifacts.
        sharded_rhs_fn = _device_z_pencil_shard_map_rhs_fn(
            setup.mesh,
            axis_name=axis_name,
            z_chunk_size=z_chunk_size,
        )
        sharded_state = jax.device_put(
            _host_staged_array_for_sharding(state_hat),
            setup.sharding,
        )

        def _route(item: jax.Array) -> jax.Array:
            out = item
            for _ in range(int(steps)):
                out = out + dt_array * sharded_rhs_fn(out)
            return out

        device_state = jax.jit(_route)(sharded_state)
    return _DeviceZComputeStates(
        serial_state=serial_state,
        device_state=jnp.asarray(jax.device_get(device_state)),
    )


def _device_z_transport_window_report_from_samples(
    setup: _DeviceZShardingSetup,
    samples: _DeviceZTransportSamples,
    *,
    axis_name: str,
    steps: int,
    dt: float,
    atol: float,
    rtol: float,
) -> NonlinearSpectralDevicePencilTransportWindowReport:
    state_abs, state_rel = _host_max_abs_rel_error(
        samples.serial_state,
        samples.device_state,
        atol=atol,
    )
    return _device_z_transport_window_report(
        state_shape=setup.state_shape,
        axis_name=axis_name,
        requested_count=setup.requested_count,
        active_count=setup.active_count,
        steps=steps,
        dt=dt,
        atol=atol,
        rtol=rtol,
        state_abs=state_abs,
        state_rel=state_rel,
        serial_trace_values=samples.serial_traces,
        device_trace_values=samples.device_traces,
        blocked_reasons=list(setup.blockers),
    )


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

    _validate_device_z_transport_window(steps=steps, observable_mode=observable_mode)
    setup = _device_z_sharding_setup(state_hat, devices=devices, axis_name=axis_name)
    serial_traces = _initial_serial_transport_traces(state_hat)
    if report := _blocked_device_z_transport_if_needed(
        setup,
        axis_name=axis_name,
        steps=steps,
        dt=dt,
        atol=atol,
        rtol=rtol,
        serial_traces=serial_traces,
    ):
        return report

    samples = _run_device_z_transport_window(
        state_hat,
        setup,
        axis_name=axis_name,
        z_chunk_size=z_chunk_size,
        dt=dt,
        steps=steps,
        observable_mode=observable_mode,
    )
    return _device_z_transport_window_report_from_samples(
        setup,
        samples,
        axis_name=axis_name,
        steps=steps,
        dt=dt,
        atol=atol,
        rtol=rtol,
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
    values = _spectral_physical_transport_observable_vector_from_sums(
        _spectral_physical_transport_observable_sums(state_hat, bracket_hat)
    )
    _append_spectral_physical_observable_vector(traces, values)

__all__ = [
    "_append_device_z_transport_observables",
    "_append_spectral_physical_observable_vector",
    "_append_spectral_physical_observables",
    "_blocked_device_z_rhs_report",
    "_blocked_device_z_transport_window_report",
    "_device_z_rhs_identity_report",
    "_device_z_pencil_shard_map_observables_fn",
    "_device_z_pencil_shard_map_rhs_fn",
    "_device_z_sharding_for_spectral_state",
    "_device_z_transport_identity_passed",
    "_device_z_transport_window_report",
    "_new_transport_trace_dict",
    "_run_device_z_compute_window_states",
    "_spectral_physical_transport_observable_sums",
    "_spectral_physical_transport_observable_vector_from_sums",
    "_serial_transport_compute_state",
    "_transport_trace_error_pairs",
    "_transport_trace_tuples",
    "device_z_pencil_nonlinear_spectral_rhs",
    "device_z_pencil_nonlinear_spectral_transport_window_identity_gate",
]
