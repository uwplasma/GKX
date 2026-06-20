"""Velocity-sharded electrostatic slice routes for the linear RHS."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.moments import build_H
from spectraxgk.operators.linear.params import LinearParams, LinearTerms
from spectraxgk.solvers.linear.parallel_common import (
    _is_electrostatic_slice_terms,
    _resolve_parallel_devices,
)
from spectraxgk.solvers.linear.parallel_streaming import (
    _streaming_electrostatic_from_phi_velocity_sharded,
)

_FUSED_ELECTROSTATIC_SLICE_KERNEL_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}


def _fused_electrostatic_route_setup(
    arr: jnp.ndarray,
    *,
    plan: Any,
    devices: Any,
    axis_name: str,
) -> SimpleNamespace:
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    dims = ("l", "m", "ky", "kx", "z")
    m_axis = dims.index("m")
    m_chunks = int(plan.chunks.get("m", 1))
    if m_chunks <= 1:
        raise ValueError("fused Hermite route requires more than one Hermite chunk")
    if int(arr.shape[m_axis]) % m_chunks != 0:
        raise ValueError("Hermite dimension must divide evenly across Hermite chunks")
    active_non_hermite = tuple(axis for axis in plan.active_axes if axis != "m")
    if active_non_hermite:
        raise NotImplementedError(
            "fused electrostatic slice route currently supports only an active 'm' axis"
        )

    device_list = list(devices)
    if len(device_list) < m_chunks:
        raise ValueError("not enough devices for the requested Hermite decomposition")

    mesh = Mesh(np.asarray(device_list[:m_chunks]), (axis_name,))
    spec_list: list[str | None] = [None] * arr.ndim
    spec_list[m_axis] = axis_name
    state_spec = PartitionSpec(*spec_list)
    return SimpleNamespace(
        m_axis=m_axis,
        m_chunks=m_chunks,
        local_m=int(arr.shape[m_axis]) // m_chunks,
        device_list=device_list,
        mesh=mesh,
        state_spec=state_spec,
        phi_spec=PartitionSpec(None, None, None),
        sharding=NamedSharding(mesh, state_spec),
        prev_pairs=tuple((idx, idx + 1) for idx in range(m_chunks - 1)),
        next_pairs=tuple((idx, idx - 1) for idx in range(1, m_chunks)),
    )


def _fused_electrostatic_constants(
    arr: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_weights: LinearTerms,
    *,
    local_m: int,
) -> SimpleNamespace:
    from spectraxgk.terms.operators import (
        grad_z_periodic as operator_grad_z_periodic,
        shift_axis as operator_shift_axis,
    )

    real_dtype = jnp.real(arr).dtype
    jl = jnp.asarray(cache.Jl)
    if jl.ndim == 5:
        jl = jl[0]
    charge_s = jnp.asarray(params.charge_sign, dtype=real_dtype).reshape(-1)[0]
    density_s = jnp.asarray(params.density, dtype=real_dtype).reshape(-1)[0]
    tau = jnp.asarray(params.tau_e, dtype=real_dtype)
    tz_s = jnp.asarray(params.tz, dtype=real_dtype).reshape(-1)[0]
    zt = jnp.where(tz_s == 0.0, 0.0, 1.0 / tz_s)
    g0 = jnp.sum(jl * jl, axis=0)
    denominator = tau + density_s * charge_s * zt * (1.0 - g0)
    ell = jnp.arange(arr.shape[0], dtype=real_dtype).reshape((arr.shape[0], 1, 1, 1, 1))
    return SimpleNamespace(
        real_dtype=real_dtype,
        jl=jl,
        charge_s=charge_s,
        density_s=density_s,
        tau=tau,
        tz_s=tz_s,
        zt=zt,
        vth_s=jnp.asarray(params.vth, dtype=real_dtype).reshape(-1)[0],
        den_safe=jnp.where(denominator == 0.0, jnp.inf, denominator),
        mask0=None if cache.mask0 is None else jnp.asarray(cache.mask0),
        ell=ell,
        ell_p1=ell + 1.0,
        bgrad=jnp.asarray(cache.bgrad, dtype=real_dtype).reshape(
            (1, 1, 1, 1, int(jnp.asarray(cache.bgrad).shape[-1]))
        ),
        cv=jnp.asarray(cache.cv_d, dtype=real_dtype).reshape(
            (1, 1) + tuple(jnp.asarray(cache.cv_d).shape)
        ),
        gb=jnp.asarray(cache.gb_d, dtype=real_dtype).reshape(
            (1, 1) + tuple(jnp.asarray(cache.gb_d).shape)
        ),
        omega_d_scale=jnp.asarray(params.omega_d_scale, dtype=real_dtype),
        kpar_scale=jnp.asarray(params.kpar_scale, dtype=real_dtype),
        imag=jnp.asarray(1j, dtype=arr.dtype),
        omega_star_s=(
            jnp.asarray(1j, dtype=arr.dtype)
            * jnp.asarray(params.omega_star_scale, dtype=real_dtype)
            * jnp.asarray(cache.ky, dtype=real_dtype)
        ).reshape((1, int(jnp.asarray(cache.ky).shape[0]), 1, 1)),
        tprim_s=jnp.asarray(params.R_over_LTi, dtype=real_dtype).reshape(-1)[0],
        fprim_s=jnp.asarray(params.R_over_Ln, dtype=real_dtype).reshape(-1)[0],
        jl_m1=operator_shift_axis(jl, -1, axis=0),
        jl_p1=operator_shift_axis(jl, 1, axis=0),
        l4=jnp.asarray(cache.l4, dtype=real_dtype).reshape((arr.shape[0], 1, 1, 1)),
        w_streaming=jnp.asarray(term_weights.streaming, dtype=real_dtype),
        w_mirror=jnp.asarray(term_weights.mirror, dtype=real_dtype),
        w_curv=jnp.asarray(term_weights.curvature, dtype=real_dtype),
        w_gradb=jnp.asarray(term_weights.gradb, dtype=real_dtype),
        w_diamag=jnp.asarray(term_weights.diamagnetic, dtype=real_dtype),
        local_m_index=jnp.arange(local_m, dtype=jnp.int32).reshape((1, local_m, 1, 1, 1)),
        grad_z=operator_grad_z_periodic,
        shift_axis=operator_shift_axis,
        kz=cache.kz,
    )


def _fused_hermite_shift(
    local: jnp.ndarray,
    *,
    offset: int,
    axis_name: str,
    prev_pairs: tuple[tuple[int, int], ...],
    next_pairs: tuple[tuple[int, int], ...],
) -> jnp.ndarray:
    depth = abs(int(offset))
    if depth == 0:
        return local
    if offset < 0:
        boundary = local[:, -depth:, ...]
        received = jax.lax.ppermute(boundary, axis_name, prev_pairs)
        return jnp.concatenate([received, local[:, :-depth, ...]], axis=1)
    boundary = local[:, :depth, ...]
    received = jax.lax.ppermute(boundary, axis_name, next_pairs)
    return jnp.concatenate([local[:, depth:, ...], received], axis=1)


def _fused_mode_coordinates(
    *,
    axis_name: str,
    local_m: int,
    local_m_index: jnp.ndarray,
    real_dtype: Any,
    local_dtype: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    global_m = jax.lax.axis_index(axis_name) * local_m + local_m_index
    return global_m, global_m.astype(real_dtype), (global_m == 0).astype(local_dtype)


def _fused_electrostatic_phi(
    local: jnp.ndarray,
    data: SimpleNamespace,
    *,
    m0: jnp.ndarray,
    axis_name: str,
) -> jnp.ndarray:
    local_gm0 = jnp.sum(local * m0, axis=1)
    local_nbar = data.density_s * data.charge_s * jnp.sum(data.jl * local_gm0, axis=0)
    phi = jax.lax.psum(local_nbar, axis_name) / data.den_safe
    return phi if data.mask0 is None else jnp.where(data.mask0, 0.0, phi)


def _fused_streaming_term(
    local: jnp.ndarray,
    phi: jnp.ndarray,
    data: SimpleNamespace,
    route: SimpleNamespace,
    *,
    global_m: jnp.ndarray,
    global_m_real: jnp.ndarray,
    axis_name: str,
) -> jnp.ndarray:
    dlocal_dz = data.grad_z(local, kz=data.kz)
    lower = _fused_hermite_shift(
        dlocal_dz,
        offset=-1,
        axis_name=axis_name,
        prev_pairs=route.prev_pairs,
        next_pairs=route.next_pairs,
    )
    upper = _fused_hermite_shift(
        dlocal_dz,
        offset=1,
        axis_name=axis_name,
        prev_pairs=route.prev_pairs,
        next_pairs=route.next_pairs,
    )
    streaming = -data.vth_s * (
        jnp.sqrt(global_m_real + 1.0) * upper + jnp.sqrt(global_m_real) * lower
    )
    field_drive_m1 = (global_m == 1).astype(local.dtype) * (
        -data.zt * data.vth_s * data.jl * phi
    )[:, None, ...]
    return streaming + data.kpar_scale * data.grad_z(field_drive_m1, kz=data.kz)


def _fused_distribution_with_field(
    local: jnp.ndarray,
    phi: jnp.ndarray,
    data: SimpleNamespace,
    *,
    m0: jnp.ndarray,
) -> jnp.ndarray:
    return local + m0 * (data.zt * data.jl * phi)[:, None, ...]


def _fused_mirror_term(
    h: jnp.ndarray,
    data: SimpleNamespace,
    route: SimpleNamespace,
    *,
    global_m_real: jnp.ndarray,
    axis_name: str,
) -> jnp.ndarray:
    h_m_p1 = _fused_hermite_shift(
        h,
        offset=1,
        axis_name=axis_name,
        prev_pairs=route.prev_pairs,
        next_pairs=route.next_pairs,
    )
    h_m_m1 = _fused_hermite_shift(
        h,
        offset=-1,
        axis_name=axis_name,
        prev_pairs=route.prev_pairs,
        next_pairs=route.next_pairs,
    )
    mirror_term = (
        -jnp.sqrt(global_m_real + 1.0) * data.ell_p1 * h_m_p1
        - jnp.sqrt(global_m_real + 1.0)
        * data.ell
        * data.shift_axis(h_m_p1, -1, axis=0)
        + jnp.sqrt(global_m_real) * data.ell * h_m_m1
        + jnp.sqrt(global_m_real) * data.ell_p1 * data.shift_axis(h_m_m1, 1, axis=0)
    )
    return -data.vth_s * data.bgrad * mirror_term


def _fused_curvature_gradb_terms(
    h: jnp.ndarray,
    data: SimpleNamespace,
    route: SimpleNamespace,
    *,
    global_m_real: jnp.ndarray,
    axis_name: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    h_m_p2 = _fused_hermite_shift(
        h,
        offset=2,
        axis_name=axis_name,
        prev_pairs=route.prev_pairs,
        next_pairs=route.next_pairs,
    )
    h_m_m2 = _fused_hermite_shift(
        h,
        offset=-2,
        axis_name=axis_name,
        prev_pairs=route.prev_pairs,
        next_pairs=route.next_pairs,
    )
    curv_term = (
        jnp.sqrt((global_m_real + 1.0) * (global_m_real + 2.0)) * h_m_p2
        + (2.0 * global_m_real + 1.0) * h
        + jnp.sqrt(global_m_real * (global_m_real - 1.0)) * h_m_m2
    )
    gradb_term = (
        (data.ell + 1.0) * data.shift_axis(h, 1, axis=0)
        + (2.0 * data.ell + 1.0) * h
        + data.ell * data.shift_axis(h, -1, axis=0)
    )
    curvature = -(data.imag * data.tz_s * data.omega_d_scale * data.cv) * curv_term
    gradb = -(data.imag * data.tz_s * data.omega_d_scale * data.gb) * gradb_term
    return curvature, gradb


def _fused_diamagnetic_term(
    phi: jnp.ndarray,
    data: SimpleNamespace,
    *,
    global_m: jnp.ndarray,
    local_dtype: Any,
) -> jnp.ndarray:
    drive_m0 = (
        data.omega_star_s
        * phi
        * (
            data.jl_m1 * (data.l4 * data.tprim_s)
            + data.jl * (data.fprim_s + 2.0 * data.l4 * data.tprim_s)
            + data.jl_p1 * ((data.l4 + 1.0) * data.tprim_s)
        )
    )
    drive_m2 = (
        data.omega_star_s
        * phi
        * data.jl
        * (data.tprim_s / jnp.sqrt(jnp.asarray(2.0, dtype=data.real_dtype)))
    )
    diamagnetic = (global_m == 0).astype(local_dtype) * drive_m0[:, None, ...]
    return diamagnetic + (global_m == 2).astype(local_dtype) * drive_m2[:, None, ...]


def _fused_electrostatic_cache_key(
    arr: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_weights: LinearTerms,
    route: SimpleNamespace,
    axis_name: str,
) -> tuple[Any, ...]:
    return (
        "electrostatic_linear_slices_fused",
        tuple(int(x) for x in arr.shape),
        str(arr.dtype),
        id(cache),
        id(params),
        float(term_weights.streaming),
        float(term_weights.mirror),
        float(term_weights.curvature),
        float(term_weights.gradb),
        float(term_weights.diamagnetic),
        tuple(str(device) for device in route.device_list[: route.m_chunks]),
        axis_name,
    )


def _cached_fused_electrostatic_kernel(
    fused: Any,
    route: SimpleNamespace,
    cache_key: tuple[Any, ...],
    *,
    axis_name: str,
) -> tuple[Any, Any]:
    cached = _FUSED_ELECTROSTATIC_SLICE_KERNEL_CACHE.get(cache_key)
    if cached is None:
        mapped = jax.jit(
            jax.shard_map(
                fused,
                mesh=route.mesh,
                in_specs=route.state_spec,
                out_specs=(route.state_spec, route.phi_spec),
                axis_names={axis_name},
            )
        )
        cached = (mapped, route.sharding)
        _FUSED_ELECTROSTATIC_SLICE_KERNEL_CACHE[cache_key] = cached
    return cached


def _linear_rhs_electrostatic_slices_velocity_sharded_fused(
    arr: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_weights: LinearTerms,
    *,
    plan: Any,
    devices: Any,
    axis_name: str = "m",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fuse the current single-species periodic electrostatic shard-map route."""

    route = _fused_electrostatic_route_setup(
        arr,
        plan=plan,
        devices=devices,
        axis_name=axis_name,
    )
    data = _fused_electrostatic_constants(
        arr,
        cache,
        params,
        term_weights,
        local_m=route.local_m,
    )

    def fused(local):
        global_m, global_m_real, m0 = _fused_mode_coordinates(
            axis_name=axis_name,
            local_m=route.local_m,
            local_m_index=data.local_m_index,
            real_dtype=data.real_dtype,
            local_dtype=local.dtype,
        )
        phi = _fused_electrostatic_phi(local, data, m0=m0, axis_name=axis_name)
        streaming = _fused_streaming_term(
            local,
            phi,
            data,
            route,
            global_m=global_m,
            global_m_real=global_m_real,
            axis_name=axis_name,
        )
        h = _fused_distribution_with_field(local, phi, data, m0=m0)
        mirror = _fused_mirror_term(
            h,
            data,
            route,
            global_m_real=global_m_real,
            axis_name=axis_name,
        )
        curvature, gradb = _fused_curvature_gradb_terms(
            h,
            data,
            route,
            global_m_real=global_m_real,
            axis_name=axis_name,
        )
        diamagnetic = _fused_diamagnetic_term(
            phi,
            data,
            global_m=global_m,
            local_dtype=local.dtype,
        )
        rhs = (
            data.w_streaming * streaming
            + data.w_mirror * mirror
            + data.w_curv * curvature
            + data.w_gradb * gradb
        )
        return rhs + data.w_diamag * diamagnetic, phi

    cache_key = _fused_electrostatic_cache_key(
        arr,
        cache,
        params,
        term_weights,
        route,
        axis_name,
    )
    mapped, sharding = _cached_fused_electrostatic_kernel(
        fused,
        route,
        cache_key,
        axis_name=axis_name,
    )
    return mapped(jax.device_put(arr, sharding))


def _linear_rhs_electrostatic_slices_velocity_sharded_serial(
    arr: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_weights: LinearTerms,
    *,
    plan: Any,
    device_list: list[Any],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    from spectraxgk.parallel.velocity import (
        curvature_gradb_drift_shard_map,
        diamagnetic_drive_shard_map,
        electrostatic_phi_shard_map,
        mirror_drift_shard_map,
    )

    real_dtype = jnp.real(arr).dtype
    phi = electrostatic_phi_shard_map(
        arr,
        plan,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
        devices=device_list,
    )
    dG = jnp.zeros_like(arr)
    if float(term_weights.streaming) != 0.0:
        streaming = _streaming_electrostatic_from_phi_velocity_sharded(
            arr,
            cache,
            params,
            phi=phi,
            plan=plan,
            devices=device_list,
        )
        dG = dG + jnp.asarray(term_weights.streaming, dtype=real_dtype) * streaming
    H = build_H(arr, cache.Jl, phi, jnp.asarray([params.tz], dtype=real_dtype))
    if float(term_weights.mirror) != 0.0:
        dG = dG + mirror_drift_shard_map(
            H,
            plan,
            vth=jnp.asarray([params.vth], dtype=real_dtype),
            bgrad=cache.bgrad,
            ell=cache.l,
            sqrt_m=cache.sqrt_m,
            sqrt_m_p1=cache.sqrt_m_p1,
            weight=jnp.asarray(term_weights.mirror, dtype=real_dtype),
            devices=device_list,
        )
    if float(term_weights.curvature) != 0.0 or float(term_weights.gradb) != 0.0:
        dG = dG + curvature_gradb_drift_shard_map(
            H,
            plan,
            tz=jnp.asarray([params.tz], dtype=real_dtype),
            omega_d_scale=params.omega_d_scale,
            cv_d=cache.cv_d,
            gb_d=cache.gb_d,
            ell=cache.l,
            m=cache.m,
            weight_curv=jnp.asarray(term_weights.curvature, dtype=real_dtype),
            weight_gradb=jnp.asarray(term_weights.gradb, dtype=real_dtype),
            devices=device_list,
        )
    if float(term_weights.diamagnetic) != 0.0:
        dG = dG + diamagnetic_drive_shard_map(
            arr,
            plan,
            phi=phi,
            Jl=cache.Jl,
            l4=cache.l4,
            tprim=params.R_over_LTi,
            fprim=params.R_over_Ln,
            omega_star_scale=params.omega_star_scale,
            ky=cache.ky,
            weight=jnp.asarray(term_weights.diamagnetic, dtype=real_dtype),
            devices=device_list,
        )
    return dG, phi


def linear_rhs_electrostatic_slices_velocity_sharded(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    num_devices: int | None = None,
    devices: Any | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute gated electrostatic streaming, drift, and diamagnetic slices."""

    from spectraxgk.parallel.velocity import (
        build_velocity_sharding_plan,
    )

    term_weights = terms if terms is not None else LinearTerms()
    if not _is_electrostatic_slice_terms(term_weights):
        raise NotImplementedError(
            "electrostatic slice route allows only electrostatic linear terms"
        )
    arr = jnp.asarray(G)
    if arr.ndim != 5:
        raise NotImplementedError(
            "velocity-sharded electrostatic slice route currently supports single-species 5D states"
        )
    if bool(getattr(cache, "use_twist_shift", False)):
        raise NotImplementedError(
            "velocity-sharded electrostatic slice route currently requires a periodic z grid"
        )

    device_list = _resolve_parallel_devices(num_devices=num_devices, devices=devices)
    plan = build_velocity_sharding_plan(
        arr.shape, num_devices=len(device_list), axes=("hermite",)
    )
    if len(device_list) > 1:
        return _linear_rhs_electrostatic_slices_velocity_sharded_fused(
            arr,
            cache,
            params,
            term_weights,
            plan=plan,
            devices=device_list,
        )
    return _linear_rhs_electrostatic_slices_velocity_sharded_serial(
        arr,
        cache,
        params,
        term_weights,
        plan=plan,
        device_list=device_list,
    )


__all__ = [
    "_FUSED_ELECTROSTATIC_SLICE_KERNEL_CACHE",
    "_linear_rhs_electrostatic_slices_velocity_sharded_fused",
    "linear_rhs_electrostatic_slices_velocity_sharded",
]
