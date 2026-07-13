"""Velocity-parallel streaming and magnetic-drift microkernels."""

from __future__ import annotations

from typing import Any, Sequence

from spectraxgk.parallel.velocity_hermite import (
    hermite_neighbor_reference,
    hermite_neighbor_shard_map,
    hermite_shift_reference,
    hermite_shift_shard_map,
)
from spectraxgk.parallel.velocity_plan import VelocityShardingPlan, _state_dims


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

    from spectraxgk.operators.linear.streaming import grad_z_periodic

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

    from spectraxgk.operators.linear.streaming import grad_z_periodic

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

    from spectraxgk.operators.linear.streaming import shift_axis

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

    from spectraxgk.operators.linear.streaming import shift_axis

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

    from spectraxgk.operators.linear.streaming import shift_axis

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

    from spectraxgk.operators.linear.streaming import shift_axis

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


__all__ = [
    "curvature_gradb_drift_reference",
    "curvature_gradb_drift_shard_map",
    "hermite_streaming_ladder_reference",
    "hermite_streaming_ladder_shard_map",
    "mirror_drift_reference",
    "mirror_drift_shard_map",
    "periodic_streaming_reference",
    "periodic_streaming_shard_map",
]
