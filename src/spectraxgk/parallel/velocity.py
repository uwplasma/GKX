"""Velocity-space decomposition plans for production parallelization."""

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

    from spectraxgk.terms.operators import shift_axis

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
    if key != "m":
        raise NotImplementedError(
            "field-reduction shard-map gate currently supports only the Hermite axis"
        )

    m_axis = dims.index("m")
    m_chunks = int(plan.chunks.get("m", 1))
    active_non_hermite = tuple(
        active_axis for active_axis in plan.active_axes if active_axis != "m"
    )
    if active_non_hermite:
        raise NotImplementedError(
            "field-reduction shard-map gate currently supports only an active 'm' axis"
        )
    if m_chunks == 1:
        return velocity_field_reduce_reference(arr, axis=axis)
    if int(arr.shape[m_axis]) % m_chunks != 0:
        raise ValueError("Hermite dimension must divide evenly across Hermite chunks")

    device_list = list(devices) if devices is not None else list(jax.devices())
    if len(device_list) < m_chunks:
        raise ValueError("not enough devices for the requested Hermite decomposition")

    mesh = Mesh(np.asarray(device_list[:m_chunks]), (axis_name,))
    spec_list: list[str | None] = [None] * arr.ndim
    spec_list[m_axis] = axis_name
    input_spec = PartitionSpec(*spec_list)
    output_spec = PartitionSpec(*[None for _ in range(arr.ndim - 1)])
    sharding = NamedSharding(mesh, input_spec)

    def reduce(local):
        local_sum = jnp.sum(local, axis=m_axis)
        return jax.lax.psum(local_sum, axis_name)

    mapped = jax.shard_map(
        reduce,
        mesh=mesh,
        in_specs=input_spec,
        out_specs=output_spec,
        axis_names={axis_name},
    )
    return mapped(jax.device_put(arr, sharding))


def _hermite_ladder_coefficients(state: Any) -> tuple[Any, Any, int]:
    import jax.numpy as jnp

    from spectraxgk.core.velocity import hermite_ladder_coeffs

    arr = jnp.asarray(state)
    dims = _state_dims(arr.ndim)
    m_axis = dims.index("m")
    nm = int(arr.shape[m_axis])
    sqrt_p, sqrt_m = hermite_ladder_coeffs(nm - 1)
    shape = [1] * arr.ndim
    shape[m_axis] = nm
    return (
        sqrt_p.reshape(shape).astype(arr.dtype),
        sqrt_m.reshape(shape).astype(arr.dtype),
        m_axis,
    )


def _broadcast_vth(vth: Any, ndim: int, *, dtype: Any) -> Any:
    import jax.numpy as jnp

    vth_arr = jnp.asarray(vth, dtype=dtype)
    if vth_arr.ndim == 0:
        return vth_arr
    shape = [1] * int(ndim)
    shape[0] = int(vth_arr.shape[0])
    return vth_arr.reshape(shape)


def hermite_streaming_ladder_reference(state: Any, *, vth: Any = 1.0) -> Any:
    """Return the full-array Hermite streaming ladder contribution."""

    import jax.numpy as jnp

    arr = jnp.asarray(state)
    lower, upper = hermite_neighbor_reference(arr)
    sqrt_p, sqrt_m, _m_axis = _hermite_ladder_coefficients(arr)
    real_dtype = jnp.real(arr).dtype
    return _broadcast_vth(vth, arr.ndim, dtype=real_dtype) * (
        sqrt_p * upper + sqrt_m * lower
    )


def hermite_streaming_ladder_shard_map(
    state: Any,
    plan: VelocityShardingPlan,
    *,
    vth: Any = 1.0,
    devices: Sequence[Any] | None = None,
    axis_name: str = "m",
) -> Any:
    """Return a shard-map Hermite streaming ladder contribution."""

    import jax.numpy as jnp

    arr = jnp.asarray(state)
    lower, upper = hermite_neighbor_shard_map(
        arr, plan, devices=devices, axis_name=axis_name
    )
    sqrt_p, sqrt_m, _m_axis = _hermite_ladder_coefficients(arr)
    real_dtype = jnp.real(arr).dtype
    return _broadcast_vth(vth, arr.ndim, dtype=real_dtype) * (
        sqrt_p * upper + sqrt_m * lower
    )


def periodic_streaming_reference(state: Any, *, kz: Any, vth: Any = 1.0) -> Any:
    """Return periodic parallel streaming using full-array operations."""

    import jax.numpy as jnp

    from spectraxgk.terms.operators import grad_z_periodic

    arr = jnp.asarray(state)
    dstate_dz = grad_z_periodic(arr, kz=kz)
    return hermite_streaming_ladder_reference(dstate_dz, vth=vth)


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

    import jax.numpy as jnp

    from spectraxgk.terms.operators import grad_z_periodic

    arr = jnp.asarray(state)
    dstate_dz = grad_z_periodic(arr, kz=kz)
    return hermite_streaming_ladder_shard_map(
        dstate_dz, plan, vth=vth, devices=devices, axis_name=axis_name
    )


def mirror_drift_reference(
    H: Any,
    *,
    vth: Any,
    bgrad: Any,
    ell: Any,
    sqrt_m: Any,
    sqrt_m_p1: Any,
    weight: Any = 1.0,
) -> Any:
    """Return the mirror-drift contribution with full-array Hermite shifts."""

    import jax.numpy as jnp

    from spectraxgk.terms.operators import shift_axis

    arr = jnp.asarray(H)
    dims = _state_dims(arr.ndim)
    axis_l = dims.index("l")
    ell_p1 = ell + 1.0
    H_m_p1 = hermite_shift_reference(arr, offset=1)
    H_m_m1 = hermite_shift_reference(arr, offset=-1)
    mirror_term = (
        -sqrt_m_p1 * ell_p1 * H_m_p1
        - sqrt_m_p1 * ell * shift_axis(H_m_p1, -1, axis=axis_l)
        + sqrt_m * ell * H_m_m1
        + sqrt_m * ell_p1 * shift_axis(H_m_m1, 1, axis=axis_l)
    )
    bgrad_shape = [1] * arr.ndim
    bgrad_shape[-1] = int(jnp.asarray(bgrad).shape[-1])
    vth_arr = jnp.asarray(vth, dtype=jnp.real(arr).dtype)
    if arr.ndim == 6:
        vth_arr = vth_arr.reshape((vth_arr.reshape(-1).shape[0], 1, 1, 1, 1, 1))
    else:
        vth_arr = vth_arr.reshape(-1)[0]
    return (
        -jnp.asarray(weight, dtype=jnp.real(arr).dtype)
        * vth_arr
        * jnp.asarray(bgrad).reshape(bgrad_shape)
        * mirror_term
    )


def mirror_drift_shard_map(
    H: Any,
    plan: VelocityShardingPlan,
    *,
    vth: Any,
    bgrad: Any,
    ell: Any,
    sqrt_m: Any,
    sqrt_m_p1: Any,
    weight: Any = 1.0,
    devices: Sequence[Any] | None = None,
    axis_name: str = "m",
) -> Any:
    """Return the mirror-drift contribution using Hermite shard exchange."""

    import jax.numpy as jnp

    from spectraxgk.terms.operators import shift_axis

    arr = jnp.asarray(H)
    dims = _state_dims(arr.ndim)
    axis_l = dims.index("l")
    ell_p1 = ell + 1.0
    H_m_p1 = hermite_shift_shard_map(
        arr, plan, offset=1, devices=devices, axis_name=axis_name
    )
    H_m_m1 = hermite_shift_shard_map(
        arr, plan, offset=-1, devices=devices, axis_name=axis_name
    )
    mirror_term = (
        -sqrt_m_p1 * ell_p1 * H_m_p1
        - sqrt_m_p1 * ell * shift_axis(H_m_p1, -1, axis=axis_l)
        + sqrt_m * ell * H_m_m1
        + sqrt_m * ell_p1 * shift_axis(H_m_m1, 1, axis=axis_l)
    )
    bgrad_shape = [1] * arr.ndim
    bgrad_shape[-1] = int(jnp.asarray(bgrad).shape[-1])
    vth_arr = jnp.asarray(vth, dtype=jnp.real(arr).dtype)
    if arr.ndim == 6:
        vth_arr = vth_arr.reshape((vth_arr.reshape(-1).shape[0], 1, 1, 1, 1, 1))
    else:
        vth_arr = vth_arr.reshape(-1)[0]
    return (
        -jnp.asarray(weight, dtype=jnp.real(arr).dtype)
        * vth_arr
        * jnp.asarray(bgrad).reshape(bgrad_shape)
        * mirror_term
    )


def curvature_gradb_drift_reference(
    H: Any,
    *,
    tz: Any,
    omega_d_scale: Any,
    cv_d: Any,
    gb_d: Any,
    ell: Any,
    m: Any,
    weight_curv: Any = 1.0,
    weight_gradb: Any = 1.0,
) -> Any:
    """Return curvature and grad-B drift contributions with full-array shifts."""

    import jax.numpy as jnp

    from spectraxgk.terms.operators import shift_axis

    arr = jnp.asarray(H)
    dims = _state_dims(arr.ndim)
    axis_l = dims.index("l")
    H_m_p2 = hermite_shift_reference(arr, offset=2)
    H_m_m2 = hermite_shift_reference(arr, offset=-2)
    curv_term = (
        jnp.sqrt((m + 1.0) * (m + 2.0)) * H_m_p2
        + (2.0 * m + 1.0) * arr
        + jnp.sqrt(m * (m - 1.0)) * H_m_m2
    )
    gradb_term = (
        (ell + 1.0) * shift_axis(arr, 1, axis=axis_l)
        + (2.0 * ell + 1.0) * arr
        + ell * shift_axis(arr, -1, axis=axis_l)
    )
    imag = jnp.asarray(1j, dtype=arr.dtype)
    real_dtype = jnp.real(arr).dtype
    if arr.ndim == 6:
        tz_s = jnp.asarray(tz, dtype=real_dtype).reshape((-1, 1, 1, 1, 1, 1))
        cv = jnp.asarray(cv_d, dtype=real_dtype)[None, None, None, ...]
        gb = jnp.asarray(gb_d, dtype=real_dtype)[None, None, None, ...]
    else:
        tz_s = jnp.asarray(tz, dtype=real_dtype).reshape(-1)[0]
        cv = jnp.asarray(cv_d, dtype=real_dtype)[None, None, ...]
        gb = jnp.asarray(gb_d, dtype=real_dtype)[None, None, ...]
    icv = imag * tz_s * jnp.asarray(omega_d_scale, dtype=real_dtype) * cv
    igb = imag * tz_s * jnp.asarray(omega_d_scale, dtype=real_dtype) * gb
    return (
        -jnp.asarray(weight_curv, dtype=real_dtype) * icv * curv_term
        - jnp.asarray(weight_gradb, dtype=real_dtype) * igb * gradb_term
    )


def curvature_gradb_drift_shard_map(
    H: Any,
    plan: VelocityShardingPlan,
    *,
    tz: Any,
    omega_d_scale: Any,
    cv_d: Any,
    gb_d: Any,
    ell: Any,
    m: Any,
    weight_curv: Any = 1.0,
    weight_gradb: Any = 1.0,
    devices: Sequence[Any] | None = None,
    axis_name: str = "m",
) -> Any:
    """Return curvature and grad-B drift contributions using Hermite exchange."""

    import jax.numpy as jnp

    from spectraxgk.terms.operators import shift_axis

    arr = jnp.asarray(H)
    dims = _state_dims(arr.ndim)
    axis_l = dims.index("l")
    H_m_p2 = hermite_shift_shard_map(
        arr, plan, offset=2, devices=devices, axis_name=axis_name
    )
    H_m_m2 = hermite_shift_shard_map(
        arr, plan, offset=-2, devices=devices, axis_name=axis_name
    )
    curv_term = (
        jnp.sqrt((m + 1.0) * (m + 2.0)) * H_m_p2
        + (2.0 * m + 1.0) * arr
        + jnp.sqrt(m * (m - 1.0)) * H_m_m2
    )
    gradb_term = (
        (ell + 1.0) * shift_axis(arr, 1, axis=axis_l)
        + (2.0 * ell + 1.0) * arr
        + ell * shift_axis(arr, -1, axis=axis_l)
    )
    imag = jnp.asarray(1j, dtype=arr.dtype)
    real_dtype = jnp.real(arr).dtype
    if arr.ndim == 6:
        tz_s = jnp.asarray(tz, dtype=real_dtype).reshape((-1, 1, 1, 1, 1, 1))
        cv = jnp.asarray(cv_d, dtype=real_dtype)[None, None, None, ...]
        gb = jnp.asarray(gb_d, dtype=real_dtype)[None, None, None, ...]
    else:
        tz_s = jnp.asarray(tz, dtype=real_dtype).reshape(-1)[0]
        cv = jnp.asarray(cv_d, dtype=real_dtype)[None, None, ...]
        gb = jnp.asarray(gb_d, dtype=real_dtype)[None, None, ...]
    icv = imag * tz_s * jnp.asarray(omega_d_scale, dtype=real_dtype) * cv
    igb = imag * tz_s * jnp.asarray(omega_d_scale, dtype=real_dtype) * gb
    return (
        -jnp.asarray(weight_curv, dtype=real_dtype) * icv * curv_term
        - jnp.asarray(weight_gradb, dtype=real_dtype) * igb * gradb_term
    )


def _diamagnetic_drive_from_global_m(
    *,
    state: Any,
    global_m: Any,
    phi: Any,
    Jl: Any,
    l4: Any,
    tprim: Any,
    fprim: Any,
    omega_star_scale: Any,
    ky: Any,
    weight: Any,
) -> Any:
    import jax.numpy as jnp

    from spectraxgk.terms.operators import shift_axis

    arr = jnp.asarray(state)
    real_dtype = jnp.real(arr).dtype
    jl = jnp.asarray(Jl)
    if jl.ndim == 5:
        jl = jl[0]
    if jl.ndim != 4:
        raise ValueError("Jl must have shape (Nl, Ny, Nx, Nz) or (1, Nl, Ny, Nx, Nz)")
    jl_m1 = shift_axis(jl, -1, axis=0)
    jl_p1 = shift_axis(jl, 1, axis=0)
    ell = jnp.asarray(l4, dtype=real_dtype).reshape((jl.shape[0], 1, 1, 1))
    tprim_s = jnp.asarray(tprim, dtype=real_dtype).reshape(-1)[0]
    fprim_s = jnp.asarray(fprim, dtype=real_dtype).reshape(-1)[0]
    omega_star = (
        jnp.asarray(1j, dtype=arr.dtype)
        * jnp.asarray(omega_star_scale, dtype=real_dtype)
        * jnp.asarray(ky, dtype=real_dtype)
    )
    omega_star_s = omega_star.reshape((1, omega_star.shape[0], 1, 1))
    phi_arr = jnp.asarray(phi, dtype=arr.dtype)
    drive_m0 = (
        omega_star_s
        * phi_arr
        * (
            jl_m1 * (ell * tprim_s)
            + jl * (fprim_s + 2.0 * ell * tprim_s)
            + jl_p1 * ((ell + 1.0) * tprim_s)
        )
    )
    drive = (global_m == 0).astype(arr.dtype) * drive_m0[:, None, ...]
    if int(arr.shape[1]) > 0:
        drive_m2 = (
            omega_star_s
            * phi_arr
            * jl
            * (tprim_s / jnp.sqrt(jnp.asarray(2.0, dtype=real_dtype)))
        )
        drive = drive + (global_m == 2).astype(arr.dtype) * drive_m2[:, None, ...]
    return jnp.asarray(weight, dtype=real_dtype) * drive


def diamagnetic_drive_reference(
    state: Any,
    *,
    phi: Any,
    Jl: Any,
    l4: Any,
    tprim: Any,
    fprim: Any,
    omega_star_scale: Any,
    ky: Any,
    weight: Any = 1.0,
) -> Any:
    """Return the single-species electrostatic diamagnetic drive."""

    import jax.numpy as jnp

    arr = jnp.asarray(state)
    if arr.ndim != 5:
        raise NotImplementedError(
            "diamagnetic_drive_reference currently supports single-species 5D states"
        )
    global_m = jnp.arange(arr.shape[1], dtype=jnp.int32).reshape(
        (1, arr.shape[1], 1, 1, 1)
    )
    return _diamagnetic_drive_from_global_m(
        state=arr,
        global_m=global_m,
        phi=phi,
        Jl=Jl,
        l4=l4,
        tprim=tprim,
        fprim=fprim,
        omega_star_scale=omega_star_scale,
        ky=ky,
        weight=weight,
    )


def diamagnetic_drive_shard_map(
    state: Any,
    plan: VelocityShardingPlan,
    *,
    phi: Any,
    Jl: Any,
    l4: Any,
    tprim: Any,
    fprim: Any,
    omega_star_scale: Any,
    ky: Any,
    weight: Any = 1.0,
    devices: Sequence[Any] | None = None,
    axis_name: str = "m",
) -> Any:
    """Return the diamagnetic drive through a Hermite-sharded local map."""

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    arr = jnp.asarray(state)
    if arr.ndim != 5:
        raise NotImplementedError(
            "diamagnetic_drive_shard_map currently supports single-species 5D states"
        )
    if tuple(arr.shape) != tuple(plan.state_shape):
        raise ValueError(
            "state shape does not match the supplied velocity sharding plan"
        )
    dims = _state_dims(arr.ndim)
    m_axis = dims.index("m")
    m_chunks = int(plan.chunks.get("m", 1))
    active_non_hermite = tuple(
        active_axis for active_axis in plan.active_axes if active_axis != "m"
    )
    if active_non_hermite:
        raise NotImplementedError(
            "diamagnetic drive gate currently supports only an active 'm' axis"
        )
    if m_chunks == 1:
        return diamagnetic_drive_reference(
            arr,
            phi=phi,
            Jl=Jl,
            l4=l4,
            tprim=tprim,
            fprim=fprim,
            omega_star_scale=omega_star_scale,
            ky=ky,
            weight=weight,
        )
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
    local_m = int(arr.shape[m_axis]) // m_chunks
    local_m_index = jnp.arange(local_m, dtype=jnp.int32).reshape((1, local_m, 1, 1, 1))

    def drive(local):
        global_m = jax.lax.axis_index(axis_name) * local_m + local_m_index
        return _diamagnetic_drive_from_global_m(
            state=local,
            global_m=global_m,
            phi=phi,
            Jl=Jl,
            l4=l4,
            tprim=tprim,
            fprim=fprim,
            omega_star_scale=omega_star_scale,
            ky=ky,
            weight=weight,
        )

    mapped = jax.shard_map(
        drive,
        mesh=mesh,
        in_specs=spec,
        out_specs=spec,
        axis_names={axis_name},
    )
    return mapped(jax.device_put(arr, sharding))


def electrostatic_phi_reference(
    state: Any,
    *,
    Jl: Any,
    tau_e: Any,
    charge: Any = 1.0,
    density: Any = 1.0,
    tz: Any = 1.0,
    mask0: Any | None = None,
) -> Any:
    """Return single-species electrostatic phi from a full 5D state."""

    import jax.numpy as jnp

    arr = jnp.asarray(state)
    if arr.ndim != 5:
        raise NotImplementedError(
            "electrostatic_phi_reference currently supports single-species 5D states"
        )
    jl = jnp.asarray(Jl)
    if jl.ndim == 5:
        jl = jl[0]
    if jl.ndim != 4:
        raise ValueError("Jl must have shape (Nl, Ny, Nx, Nz) or (1, Nl, Ny, Nx, Nz)")
    real_dtype = jnp.real(arr).dtype
    charge_s = jnp.asarray(charge, dtype=real_dtype).reshape(-1)[0]
    density_s = jnp.asarray(density, dtype=real_dtype).reshape(-1)[0]
    tz_s = jnp.asarray(tz, dtype=real_dtype).reshape(-1)[0]
    zt = jnp.where(tz_s == 0.0, 0.0, 1.0 / tz_s)
    nbar = density_s * charge_s * jnp.sum(jl * arr[:, 0, ...], axis=0)
    g0 = jnp.sum(jl * jl, axis=0)
    qneut = density_s * charge_s * zt * (1.0 - g0)
    den_safe = jnp.where(
        jnp.asarray(tau_e, dtype=real_dtype) + qneut == 0.0,
        jnp.inf,
        jnp.asarray(tau_e, dtype=real_dtype) + qneut,
    )
    phi = nbar / den_safe
    if mask0 is not None:
        phi = jnp.where(jnp.asarray(mask0), 0.0, phi)
    return phi


def electrostatic_phi_shard_map(
    state: Any,
    plan: VelocityShardingPlan,
    *,
    Jl: Any,
    tau_e: Any,
    charge: Any = 1.0,
    density: Any = 1.0,
    tz: Any = 1.0,
    mask0: Any | None = None,
    devices: Sequence[Any] | None = None,
    axis_name: str = "m",
) -> Any:
    """Solve electrostatic phi using a Hermite-sharded density reduction."""

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    arr = jnp.asarray(state)
    if arr.ndim != 5:
        raise NotImplementedError(
            "electrostatic_phi_shard_map currently supports single-species 5D states"
        )
    if tuple(arr.shape) != tuple(plan.state_shape):
        raise ValueError(
            "state shape does not match the supplied velocity sharding plan"
        )
    dims = _state_dims(arr.ndim)
    m_axis = dims.index("m")
    m_chunks = int(plan.chunks.get("m", 1))
    active_non_hermite = tuple(
        active_axis for active_axis in plan.active_axes if active_axis != "m"
    )
    if active_non_hermite:
        raise NotImplementedError(
            "electrostatic field-reduction gate currently supports only an active 'm' axis"
        )
    if m_chunks == 1:
        return electrostatic_phi_reference(
            arr,
            Jl=Jl,
            tau_e=tau_e,
            charge=charge,
            density=density,
            tz=tz,
            mask0=mask0,
        )
    if int(arr.shape[m_axis]) % m_chunks != 0:
        raise ValueError("Hermite dimension must divide evenly across Hermite chunks")

    jl = jnp.asarray(Jl)
    if jl.ndim == 5:
        jl = jl[0]
    if jl.ndim != 4:
        raise ValueError("Jl must have shape (Nl, Ny, Nx, Nz) or (1, Nl, Ny, Nx, Nz)")
    real_dtype = jnp.real(arr).dtype
    charge_s = jnp.asarray(charge, dtype=real_dtype).reshape(-1)[0]
    density_s = jnp.asarray(density, dtype=real_dtype).reshape(-1)[0]
    tz_s = jnp.asarray(tz, dtype=real_dtype).reshape(-1)[0]
    zt = jnp.where(tz_s == 0.0, 0.0, 1.0 / tz_s)
    tau = jnp.asarray(tau_e, dtype=real_dtype)
    g0 = jnp.sum(jl * jl, axis=0)
    qneut = density_s * charge_s * zt * (1.0 - g0)
    den_safe = jnp.where(tau + qneut == 0.0, jnp.inf, tau + qneut)

    device_list = list(devices) if devices is not None else list(jax.devices())
    if len(device_list) < m_chunks:
        raise ValueError("not enough devices for the requested Hermite decomposition")

    mesh = Mesh(np.asarray(device_list[:m_chunks]), (axis_name,))
    spec_list: list[str | None] = [None] * arr.ndim
    spec_list[m_axis] = axis_name
    input_spec = PartitionSpec(*spec_list)
    output_spec = PartitionSpec(*[None for _ in range(arr.ndim - 2)])
    sharding = NamedSharding(mesh, input_spec)
    local_m = int(arr.shape[m_axis]) // m_chunks
    local_m_index = jnp.arange(local_m, dtype=jnp.int32).reshape((1, local_m, 1, 1, 1))

    def local_density(local):
        global_m = jax.lax.axis_index(axis_name) * local_m + local_m_index
        m0 = (global_m == 0).astype(local.dtype)
        local_gm0 = jnp.sum(local * m0, axis=m_axis)
        local_nbar = density_s * charge_s * jnp.sum(jl * local_gm0, axis=0)
        return jax.lax.psum(local_nbar, axis_name)

    mapped = jax.shard_map(
        local_density,
        mesh=mesh,
        in_specs=input_spec,
        out_specs=output_spec,
        axis_names={axis_name},
    )
    nbar = mapped(jax.device_put(arr, sharding))
    phi = nbar / den_safe
    if mask0 is not None:
        phi = jnp.where(jnp.asarray(mask0), 0.0, phi)
    return phi


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
