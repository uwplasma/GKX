"""Hermite-axis exchange and velocity-field reduction kernels."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from spectraxgk.parallel.velocity_plan import (
    VelocityShardingPlan,
    _AXIS_ALIASES,
    _slice_axis,
    _state_dims,
)


def hermite_neighbor_reference(state: Any) -> tuple[Any, Any]:
    """Return full-array lower/upper Hermite-neighbor states.

    The Hermite streaming ladder couples moment ``m`` to ``m-1`` and ``m+1``.
    Physical boundaries outside ``[0, Nm-1]`` are zeros. The returned arrays
    have the same shape as ``state`` and provide the lower and upper neighbor
    values for every Hermite index.
    """

    import jax.numpy as jnp

    arr = jnp.asarray(state)
    dims = _state_dims(arr.ndim)
    m_axis = dims.index("m")
    zero_ghost = jnp.zeros_like(_slice_axis(arr, m_axis, 0, 1))
    lower = jnp.concatenate([zero_ghost, _slice_axis(arr, m_axis, 0, -1)], axis=m_axis)
    upper = jnp.concatenate(
        [_slice_axis(arr, m_axis, 1, None), zero_ghost], axis=m_axis
    )
    return lower, upper


def hermite_shift_reference(state: Any, *, offset: int) -> Any:
    """Shift a state along the Hermite axis with zero physical boundaries."""

    import jax.numpy as jnp

    from spectraxgk.operators.linear.streaming import shift_axis

    arr = jnp.asarray(state)
    dims = _state_dims(arr.ndim)
    return shift_axis(arr, int(offset), axis=dims.index("m"))


def hermite_neighbor_shard_map(
    state: Any,
    plan: VelocityShardingPlan,
    *,
    devices: Sequence[Any] | None = None,
    axis_name: str = "m",
) -> tuple[Any, Any]:
    """Exchange nearest Hermite neighbors with ``jax.shard_map``.

    This is a communication-kernel identity primitive, not a production
    nonlinear solver path. It currently supports one-dimensional Hermite
    decomposition plans. More complex species-Hermite meshes should first add a
    separate field-reduction and broadcast gate.
    """

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    arr = jnp.asarray(state)
    if tuple(arr.shape) != tuple(plan.state_shape):
        raise ValueError(
            "state shape does not match the supplied velocity sharding plan"
        )
    dims = _state_dims(arr.ndim)
    m_axis = dims.index("m")
    m_chunks = int(plan.chunks.get("m", 1))
    if m_chunks < 1:
        raise ValueError("Hermite chunk count must be >= 1")
    active_non_hermite = tuple(axis for axis in plan.active_axes if axis != "m")
    if active_non_hermite:
        raise NotImplementedError(
            "Hermite shard-map exchange currently supports only an active 'm' axis"
        )
    if m_chunks == 1:
        return hermite_neighbor_reference(arr)
    if int(arr.shape[m_axis]) % m_chunks != 0:
        raise ValueError("Hermite dimension must divide evenly across Hermite chunks")

    device_list = list(devices) if devices is not None else list(jax.devices())
    if len(device_list) < m_chunks:
        raise ValueError("not enough devices for the requested Hermite decomposition")

    mesh = Mesh(np.asarray(device_list[:m_chunks]), (axis_name,))
    spec_list: list[str | None] = [None] * arr.ndim
    spec_list[m_axis] = axis_name
    spec = PartitionSpec(*spec_list)
    sharding = NamedSharding(mesh, spec)
    prev_pairs = tuple((idx, idx + 1) for idx in range(m_chunks - 1))
    next_pairs = tuple((idx, idx - 1) for idx in range(1, m_chunks))

    def exchange(local):
        first = _slice_axis(local, m_axis, 0, 1)
        last = _slice_axis(local, m_axis, -1, None)
        prev_boundary = jax.lax.ppermute(last, axis_name, prev_pairs)
        next_boundary = jax.lax.ppermute(first, axis_name, next_pairs)
        lower = jnp.concatenate(
            [prev_boundary, _slice_axis(local, m_axis, 0, -1)], axis=m_axis
        )
        upper = jnp.concatenate(
            [_slice_axis(local, m_axis, 1, None), next_boundary], axis=m_axis
        )
        return lower, upper

    mapped = jax.shard_map(
        exchange,
        mesh=mesh,
        in_specs=spec,
        out_specs=(spec, spec),
        axis_names={axis_name},
    )
    return mapped(jax.device_put(arr, sharding))


def hermite_shift_shard_map(
    state: Any,
    plan: VelocityShardingPlan,
    *,
    offset: int,
    devices: Sequence[Any] | None = None,
    axis_name: str = "m",
) -> Any:
    """Shift a Hermite-sharded state by ``offset`` moments with shard exchange."""

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    shift = int(offset)
    arr = jnp.asarray(state)
    if tuple(arr.shape) != tuple(plan.state_shape):
        raise ValueError(
            "state shape does not match the supplied velocity sharding plan"
        )
    if shift == 0:
        return arr
    dims = _state_dims(arr.ndim)
    m_axis = dims.index("m")
    m_chunks = int(plan.chunks.get("m", 1))
    if abs(shift) >= int(arr.shape[m_axis]):
        return jnp.zeros_like(arr)
    active_non_hermite = tuple(axis for axis in plan.active_axes if axis != "m")
    if active_non_hermite:
        raise NotImplementedError(
            "Hermite shard-map shift currently supports only an active 'm' axis"
        )
    if m_chunks == 1:
        return hermite_shift_reference(arr, offset=shift)
    if int(arr.shape[m_axis]) % m_chunks != 0:
        raise ValueError("Hermite dimension must divide evenly across Hermite chunks")
    local_m = int(arr.shape[m_axis]) // m_chunks
    depth = abs(shift)
    if depth > local_m:
        raise NotImplementedError(
            "Hermite shard-map shift currently requires abs(offset) <= local shard size"
        )

    device_list = list(devices) if devices is not None else list(jax.devices())
    if len(device_list) < m_chunks:
        raise ValueError("not enough devices for the requested Hermite decomposition")

    mesh = Mesh(np.asarray(device_list[:m_chunks]), (axis_name,))
    spec_list: list[str | None] = [None] * arr.ndim
    spec_list[m_axis] = axis_name
    spec = PartitionSpec(*spec_list)
    sharding = NamedSharding(mesh, spec)
    prev_pairs = tuple((idx, idx + 1) for idx in range(m_chunks - 1))
    next_pairs = tuple((idx, idx - 1) for idx in range(1, m_chunks))

    def exchange(local):
        if shift < 0:
            boundary = _slice_axis(local, m_axis, -depth, None)
            received = jax.lax.ppermute(boundary, axis_name, prev_pairs)
            return jnp.concatenate(
                [received, _slice_axis(local, m_axis, 0, -depth)], axis=m_axis
            )
        boundary = _slice_axis(local, m_axis, 0, depth)
        received = jax.lax.ppermute(boundary, axis_name, next_pairs)
        return jnp.concatenate(
            [_slice_axis(local, m_axis, depth, None), received], axis=m_axis
        )

    mapped = jax.shard_map(
        exchange,
        mesh=mesh,
        in_specs=spec,
        out_specs=spec,
        axis_names={axis_name},
    )
    return mapped(jax.device_put(arr, sharding))


def velocity_field_reduce_reference(state: Any, *, axis: str = "m") -> Any:
    """Return the full-array velocity-axis reduction used by field solves."""

    import jax.numpy as jnp

    arr = jnp.asarray(state)
    dims = _state_dims(arr.ndim)
    key = _AXIS_ALIASES.get(str(axis).strip().lower().replace("-", "_"))
    if key is None:
        raise ValueError(f"Unknown reduction axis '{axis}'")
    if key not in dims:
        raise ValueError(f"axis '{axis}' is not present in a {arr.ndim}D state")
    return jnp.sum(arr, axis=dims.index(key))


def velocity_field_reduce_shard_map(
    state: Any,
    plan: VelocityShardingPlan,
    *,
    axis: str = "m",
    devices: Sequence[Any] | None = None,
    axis_name: str = "m",
) -> Any:
    """Reduce one velocity axis across a shard-map mesh and broadcast it."""

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    arr = jnp.asarray(state)
    if tuple(arr.shape) != tuple(plan.state_shape):
        raise ValueError(
            "state shape does not match the supplied velocity sharding plan"
        )
    dims = _state_dims(arr.ndim)
    key = _AXIS_ALIASES.get(str(axis).strip().lower().replace("-", "_"))
    if key is None:
        raise ValueError(f"Unknown reduction axis '{axis}'")
    if key not in dims:
        raise ValueError(f"axis '{axis}' is not present in a {arr.ndim}D state")
    reduce_axis = dims.index(key)
    chunks = int(plan.chunks.get(key, 1))
    active_other_axes = tuple(
        active_axis for active_axis in plan.active_axes if active_axis != key
    )
    if active_other_axes:
        raise NotImplementedError(
            "field-reduction shard-map gate supports one active reduction axis"
        )
    if chunks == 1:
        return velocity_field_reduce_reference(arr, axis=axis)
    if int(arr.shape[reduce_axis]) % chunks != 0:
        raise ValueError(f"{key} dimension must divide evenly across {key} chunks")

    device_list = list(devices) if devices is not None else list(jax.devices())
    if len(device_list) < chunks:
        raise ValueError("not enough devices for the requested Hermite decomposition")

    mesh = Mesh(np.asarray(device_list[:chunks]), (axis_name,))
    spec_list: list[str | None] = [None] * arr.ndim
    spec_list[reduce_axis] = axis_name
    input_spec = PartitionSpec(*spec_list)
    output_spec = PartitionSpec(*[None for _ in range(arr.ndim - 1)])
    sharding = NamedSharding(mesh, input_spec)

    def reduce(local):
        local_sum = jnp.sum(local, axis=reduce_axis)
        return jax.lax.psum(local_sum, axis_name)

    mapped = jax.shard_map(
        reduce,
        mesh=mesh,
        in_specs=input_spec,
        out_specs=output_spec,
        axis_names={axis_name},
    )
    return mapped(jax.device_put(arr, sharding))


__all__ = [
    "hermite_neighbor_reference",
    "hermite_neighbor_shard_map",
    "hermite_shift_reference",
    "hermite_shift_shard_map",
    "velocity_field_reduce_reference",
    "velocity_field_reduce_shard_map",
]
