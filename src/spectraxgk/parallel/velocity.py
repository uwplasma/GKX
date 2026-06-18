"""Velocity-space decomposition plans for production parallelization."""

from __future__ import annotations

from typing import Any, Sequence

from spectraxgk.parallel import velocity_streaming as _velocity_streaming
from spectraxgk.parallel.velocity_drive import (
    diamagnetic_drive_reference,
    diamagnetic_drive_shard_map,
    electrostatic_phi_reference,
    electrostatic_phi_shard_map,
)
from spectraxgk.parallel.velocity_hermite import (
    hermite_neighbor_reference,
    hermite_neighbor_shard_map,
    hermite_shift_reference,
    hermite_shift_shard_map,
    velocity_field_reduce_reference,
    velocity_field_reduce_shard_map,
)
from spectraxgk.parallel.velocity_plan import (
    VelocityShardingPlan,
    _AXIS_ALIASES,  # noqa: F401 - private compatibility hook
    _slice_axis,  # noqa: F401 - private compatibility hook
    _state_dims,  # noqa: F401 - private compatibility hook
    build_velocity_sharding_plan,
)
from spectraxgk.parallel.velocity_streaming import (
    curvature_gradb_drift_reference,
    curvature_gradb_drift_shard_map,
    mirror_drift_reference,
    mirror_drift_shard_map,
)

_hermite_ladder_coefficients = _velocity_streaming._hermite_ladder_coefficients
_broadcast_vth = _velocity_streaming._broadcast_vth


def _sync_streaming_hooks() -> None:
    _velocity_streaming._hermite_ladder_coefficients = _hermite_ladder_coefficients
    _velocity_streaming._broadcast_vth = _broadcast_vth


def hermite_streaming_ladder_reference(state: Any, *, vth: Any = 1.0) -> Any:
    """Return the full-array Hermite streaming ladder contribution."""

    _sync_streaming_hooks()
    return _velocity_streaming.hermite_streaming_ladder_reference(state, vth=vth)


def hermite_streaming_ladder_shard_map(
    state: Any,
    plan: VelocityShardingPlan,
    *,
    vth: Any = 1.0,
    devices: Sequence[Any] | None = None,
    axis_name: str = "m",
) -> Any:
    """Return a shard-map Hermite streaming ladder contribution."""

    _sync_streaming_hooks()
    return _velocity_streaming.hermite_streaming_ladder_shard_map(
        state,
        plan,
        vth=vth,
        devices=devices,
        axis_name=axis_name,
    )


def periodic_streaming_reference(state: Any, *, kz: Any, vth: Any = 1.0) -> Any:
    """Return periodic parallel streaming using full-array operations."""

    _sync_streaming_hooks()
    return _velocity_streaming.periodic_streaming_reference(state, kz=kz, vth=vth)


def periodic_streaming_shard_map(
    state: Any,
    plan: VelocityShardingPlan,
    *,
    kz: Any,
    vth: Any = 1.0,
    devices: Sequence[Any] | None = None,
    axis_name: str = "m",
) -> Any:
    """Return periodic parallel streaming through the Hermite shard-map path."""

    _sync_streaming_hooks()
    return _velocity_streaming.periodic_streaming_shard_map(
        state,
        plan,
        kz=kz,
        vth=vth,
        devices=devices,
        axis_name=axis_name,
    )


__all__ = [
    "VelocityShardingPlan",
    "build_velocity_sharding_plan",
    "curvature_gradb_drift_reference",
    "curvature_gradb_drift_shard_map",
    "diamagnetic_drive_reference",
    "diamagnetic_drive_shard_map",
    "electrostatic_phi_reference",
    "electrostatic_phi_shard_map",
    "hermite_neighbor_reference",
    "hermite_neighbor_shard_map",
    "hermite_shift_reference",
    "hermite_shift_shard_map",
    "hermite_streaming_ladder_reference",
    "hermite_streaming_ladder_shard_map",
    "mirror_drift_reference",
    "mirror_drift_shard_map",
    "periodic_streaming_reference",
    "periodic_streaming_shard_map",
    "velocity_field_reduce_reference",
    "velocity_field_reduce_shard_map",
]
