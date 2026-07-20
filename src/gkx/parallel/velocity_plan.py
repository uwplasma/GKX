"""Velocity-space decomposition metadata and planning helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


_AXIS_ALIASES = {
    "species": "s",
    "s": "s",
    "laguerre": "l",
    "l": "l",
    "hermite": "m",
    "m": "m",
    "ky": "ky",
    "kx": "kx",
    "z": "z",
}


@dataclass(frozen=True)
class VelocityShardingPlan:
    """JSON-friendly plan for decomposing a packed GK state over devices."""

    state_shape: tuple[int, ...]
    dims: tuple[str, ...]
    num_devices: int
    chunks: dict[str, int]
    shard_shape: tuple[int, ...]
    active_axes: tuple[str, ...]
    hermite_ghost_depth: int
    needs_hermite_exchange: bool
    needs_field_reduction: bool
    field_reduction_axes: tuple[str, ...]
    communication_pattern: str
    load_balance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _state_dims(ndim: int) -> tuple[str, ...]:
    if ndim == 5:
        return ("l", "m", "ky", "kx", "z")
    if ndim == 6:
        return ("s", "l", "m", "ky", "kx", "z")
    raise ValueError("state_shape must have 5 or 6 dimensions")


def _normalize_axes(
    axes: Sequence[str] | None, *, has_species: bool
) -> tuple[str, ...]:
    raw = (
        axes
        if axes is not None
        else (
            ("species", "hermite", "laguerre")
            if has_species
            else ("hermite", "laguerre")
        )
    )
    normalized: list[str] = []
    for axis in raw:
        key = str(axis).strip().lower().replace("-", "_")
        if key not in _AXIS_ALIASES:
            raise ValueError(f"Unknown velocity-sharding axis '{axis}'")
        dim = _AXIS_ALIASES[key]
        if dim == "s" and not has_species:
            raise ValueError(
                "species sharding requires a 6D state with an explicit species axis"
            )
        if dim not in normalized:
            normalized.append(dim)
    return tuple(normalized)


def _largest_factor_at_most(total: int, limit: int) -> int:
    for factor in range(min(int(total), int(limit)), 0, -1):
        if total % factor == 0:
            return factor
    return 1


def _axis_chunks(
    dim_sizes: dict[str, int], num_devices: int, axes: tuple[str, ...]
) -> dict[str, int]:
    remaining = int(num_devices)
    chunks = {dim: 1 for dim in dim_sizes}
    for axis in axes:
        factor = _largest_factor_at_most(remaining, dim_sizes[axis])
        chunks[axis] = factor
        remaining //= factor
        if remaining == 1:
            break
    if remaining != 1:
        raise ValueError(
            "num_devices could not be factored over the requested velocity axes; "
            "choose a device count that divides available species/Hermite/Laguerre extents"
        )
    return chunks


def _chunked_axis_size(size: int, chunks: int) -> int:
    return int(np.ceil(int(size) / int(chunks)))


def _slice_axis(array: Any, axis: int, start: int | None, stop: int | None) -> Any:
    index = [slice(None)] * array.ndim
    index[int(axis)] = slice(start, stop)
    return array[tuple(index)]


def build_velocity_sharding_plan(
    state_shape: Sequence[int],
    *,
    num_devices: int,
    axes: Sequence[str] | None = None,
    hermite_ghost_depth: int = 1,
) -> VelocityShardingPlan:
    """Build a species/Hermite velocity-space decomposition plan.

    The plan is metadata only. It does not move arrays or claim speedup. It
    records which axes should be split, where Hermite ghost exchange is needed,
    and which velocity axes require field-solve reductions/broadcasts before a
    production ``shard_map`` implementation is allowed to use the layout.
    """

    shape = tuple(int(x) for x in state_shape)
    if any(size < 1 for size in shape):
        raise ValueError("all state_shape entries must be >= 1")
    devices = int(num_devices)
    if devices < 1:
        raise ValueError("num_devices must be >= 1")
    ghost_depth = int(hermite_ghost_depth)
    if ghost_depth < 0:
        raise ValueError("hermite_ghost_depth must be >= 0")

    dims = _state_dims(len(shape))
    dim_sizes = dict(zip(dims, shape, strict=True))
    axis_order = _normalize_axes(axes, has_species="s" in dims)
    chunks = _axis_chunks(dim_sizes, devices, axis_order)
    active_axes = tuple(dim for dim in dims if chunks[dim] > 1)
    shard_shape = tuple(_chunked_axis_size(dim_sizes[dim], chunks[dim]) for dim in dims)

    needs_hermite_exchange = bool(chunks.get("m", 1) > 1 and ghost_depth > 0)
    field_reduction_axes = tuple(
        axis for axis in ("s", "l", "m") if chunks.get(axis, 1) > 1
    )
    needs_field_reduction = bool(field_reduction_axes)
    total_shard_slots = int(np.prod([chunks[dim] for dim in dims], dtype=int))
    load_balance = float(devices / total_shard_slots) if total_shard_slots else 0.0
    communication = "none"
    if needs_hermite_exchange and needs_field_reduction:
        communication = "hermite_ghost_exchange+field_reduce_broadcast"
    elif needs_hermite_exchange:
        communication = "hermite_ghost_exchange"
    elif needs_field_reduction:
        communication = "field_reduce_broadcast"

    return VelocityShardingPlan(
        state_shape=shape,
        dims=dims,
        num_devices=devices,
        chunks=chunks,
        shard_shape=shard_shape,
        active_axes=active_axes,
        hermite_ghost_depth=ghost_depth,
        needs_hermite_exchange=needs_hermite_exchange,
        needs_field_reduction=needs_field_reduction,
        field_reduction_axes=field_reduction_axes,
        communication_pattern=communication,
        load_balance=load_balance,
    )


__all__ = ["VelocityShardingPlan", "build_velocity_sharding_plan"]
