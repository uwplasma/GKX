"""Velocity-parallel linear RHS helpers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.moments import build_H
from spectraxgk.operators.linear.params import LinearParams, LinearTerms, _as_species_array

__all__ = [
    "_FUSED_ELECTROSTATIC_SLICE_KERNEL_CACHE",
    "_electrostatic_streaming_field_rhs",
    "_is_electrostatic_field_terms",
    "_is_electrostatic_slice_terms",
    "_is_streaming_only_terms",
    "_linear_rhs_electrostatic_slices_velocity_sharded_fused",
    "_resolve_parallel_devices",
    "_streaming_electrostatic_from_phi_velocity_sharded",
    "linear_rhs_electrostatic_slices_velocity_sharded",
    "linear_rhs_parallel_cached",
    "linear_rhs_streaming_electrostatic_velocity_sharded",
    "linear_rhs_streaming_velocity_sharded",
]

_FUSED_ELECTROSTATIC_SLICE_KERNEL_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}


def _is_streaming_only_terms(terms: LinearTerms | None) -> bool:
    term_weights = terms if terms is not None else LinearTerms()
    return (
        float(term_weights.streaming) == 1.0
        and float(term_weights.mirror) == 0.0
        and float(term_weights.curvature) == 0.0
        and float(term_weights.gradb) == 0.0
        and float(term_weights.diamagnetic) == 0.0
        and float(term_weights.collisions) == 0.0
        and float(term_weights.hypercollisions) == 0.0
        and float(term_weights.hyperdiffusion) == 0.0
        and float(term_weights.end_damping) == 0.0
        and float(term_weights.apar) == 0.0
        and float(term_weights.bpar) == 0.0
    )


def _is_electrostatic_slice_terms(terms: LinearTerms | None) -> bool:
    term_weights = terms if terms is not None else LinearTerms()
    return (
        float(term_weights.collisions) == 0.0
        and float(term_weights.hypercollisions) == 0.0
        and float(term_weights.hyperdiffusion) == 0.0
        and float(term_weights.end_damping) == 0.0
        and float(term_weights.apar) == 0.0
        and float(term_weights.bpar) == 0.0
    )


def _is_electrostatic_field_terms(terms: LinearTerms | None) -> bool:
    term_weights = terms if terms is not None else LinearTerms()
    return float(term_weights.apar) == 0.0 and float(term_weights.bpar) == 0.0


def _resolve_parallel_devices(
    *, num_devices: int | None = None, devices: Any | None = None
) -> list[Any]:
    """Return an explicit device list for opt-in parallel diagnostics."""

    if devices is None:
        device_list = list(jax.devices())
        if num_devices is not None:
            device_count = int(num_devices)
            if device_count < 1:
                raise ValueError("num_devices must be >= 1")
            if len(device_list) < device_count:
                raise ValueError(
                    f"requested {device_count} devices, but only {len(device_list)} are available"
                )
            device_list = device_list[:device_count]
    else:
        device_list = list(devices)
        if num_devices is not None and int(num_devices) != len(device_list):
            raise ValueError("num_devices must match the explicit devices list length")
    if not device_list:
        raise ValueError("at least one device is required")
    return device_list


def linear_rhs_streaming_velocity_sharded(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    num_devices: int | None = None,
    devices: Any | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the streaming-only linear RHS with the Hermite shard-map path.

    This diagnostic route is intentionally narrower than
    :func:`linear_rhs_cached`: it covers the velocity-space streaming operator
    only and returns a zero electrostatic potential. It is used to gate the
    future production velocity decomposition before field solves, drifts,
    collisions, and nonlinear terms are exposed through the runtime path.
    """

    from spectraxgk.parallel.velocity import (
        build_velocity_sharding_plan,
        periodic_streaming_shard_map,
    )

    arr = jnp.asarray(G)
    if arr.ndim not in (5, 6):
        raise ValueError(
            "G must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
        )

    device_list = _resolve_parallel_devices(num_devices=num_devices, devices=devices)
    plan = build_velocity_sharding_plan(
        arr.shape, num_devices=len(device_list), axes=("hermite",)
    )
    dG = -periodic_streaming_shard_map(
        arr, plan, kz=cache.kz, vth=params.vth, devices=device_list
    )
    phi = jnp.zeros(arr.shape[-3:], dtype=arr.dtype)
    return dG, phi


def _streaming_electrostatic_from_phi_velocity_sharded(
    arr: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    phi: jnp.ndarray,
    plan: Any,
    devices: Any,
) -> jnp.ndarray:
    """Apply electrostatic streaming with a precomputed electrostatic field."""

    from spectraxgk.terms.operators import grad_z_periodic as operator_grad_z_periodic
    from spectraxgk.parallel.velocity import periodic_streaming_shard_map

    particle_streaming = -periodic_streaming_shard_map(
        arr, plan, kz=cache.kz, vth=params.vth, devices=devices
    )
    real_dtype = jnp.real(arr).dtype
    G6 = arr[None, ...]
    tz = _as_species_array(params.tz, 1, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, 1, "vth").astype(real_dtype)
    field_rhs = _electrostatic_streaming_field_rhs(
        G6, phi=phi, Jl=cache.Jl, tz=tz, vth=vth
    )
    field_streaming = jnp.asarray(
        params.kpar_scale, dtype=real_dtype
    ) * operator_grad_z_periodic(field_rhs, kz=cache.kz)
    return particle_streaming + field_streaming[0]


def _electrostatic_streaming_field_rhs(
    G6: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    Jl: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
) -> jnp.ndarray:
    """Build the pre-derivative electrostatic streaming field term."""

    Nm = G6.shape[2]
    m_idx = jnp.arange(Nm, dtype=jnp.int32)[None, None, :, None, None, None]
    zt = jnp.where(tz == 0.0, 0.0, 1.0 / tz)
    zt5 = zt[:, None, None, None, None]
    vth5 = vth[:, None, None, None, None]
    phi_s = phi[None, None, ...]
    drive_m1 = -zt5 * vth5 * Jl * phi_s
    return (m_idx == 1).astype(G6.dtype) * drive_m1[:, :, None, ...]


def linear_rhs_streaming_electrostatic_velocity_sharded(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    num_devices: int | None = None,
    devices: Any | None = None,
    use_custom_vjp: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute electrostatic streaming RHS with Hermite-sharded particle streaming.

    This route solves ``phi`` with the production electrostatic field solve,
    applies the Hermite velocity-sharded particle-streaming operator, and adds
    the reference-aligned electrostatic streaming field term. It is limited to periodic
    field-line grids and excludes electromagnetic fields by construction.
    """

    from spectraxgk.parallel.velocity import (
        build_velocity_sharding_plan,
        electrostatic_phi_shard_map,
    )

    arr = jnp.asarray(G)
    if arr.ndim != 5:
        raise NotImplementedError(
            "velocity-sharded electrostatic streaming currently supports single-species 5D states"
        )
    if bool(getattr(cache, "use_twist_shift", False)):
        raise NotImplementedError(
            "velocity-sharded electrostatic streaming currently requires a periodic z grid"
        )

    device_list = _resolve_parallel_devices(num_devices=num_devices, devices=devices)
    plan = build_velocity_sharding_plan(
        arr.shape, num_devices=len(device_list), axes=("hermite",)
    )
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
    return _streaming_electrostatic_from_phi_velocity_sharded(
        arr, cache, params, phi=phi, plan=plan, devices=device_list
    ), phi


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

    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from spectraxgk.terms.operators import (
        grad_z_periodic as operator_grad_z_periodic,
        shift_axis as operator_shift_axis,
    )

    dims = ("l", "m", "ky", "kx", "z")
    m_axis = dims.index("m")
    m_chunks = int(plan.chunks.get("m", 1))
    if m_chunks <= 1:
        raise ValueError("fused Hermite route requires more than one Hermite chunk")
    if int(arr.shape[m_axis]) % m_chunks != 0:
        raise ValueError("Hermite dimension must divide evenly across Hermite chunks")
    active_non_hermite = tuple(
        active_axis for active_axis in plan.active_axes if active_axis != "m"
    )
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
    phi_spec = PartitionSpec(None, None, None)
    sharding = NamedSharding(mesh, state_spec)
    local_m = int(arr.shape[m_axis]) // m_chunks
    local_m_index = jnp.arange(local_m, dtype=jnp.int32).reshape((1, local_m, 1, 1, 1))
    prev_pairs = tuple((idx, idx + 1) for idx in range(m_chunks - 1))
    next_pairs = tuple((idx, idx - 1) for idx in range(1, m_chunks))

    real_dtype = jnp.real(arr).dtype
    jl = jnp.asarray(cache.Jl)
    if jl.ndim == 5:
        jl = jl[0]
    charge_s = jnp.asarray(params.charge_sign, dtype=real_dtype).reshape(-1)[0]
    density_s = jnp.asarray(params.density, dtype=real_dtype).reshape(-1)[0]
    tau = jnp.asarray(params.tau_e, dtype=real_dtype)
    tz_s = jnp.asarray(params.tz, dtype=real_dtype).reshape(-1)[0]
    zt = jnp.where(tz_s == 0.0, 0.0, 1.0 / tz_s)
    vth_s = jnp.asarray(params.vth, dtype=real_dtype).reshape(-1)[0]
    g0 = jnp.sum(jl * jl, axis=0)
    den_safe = jnp.where(
        tau + density_s * charge_s * zt * (1.0 - g0) == 0.0,
        jnp.inf,
        tau + density_s * charge_s * zt * (1.0 - g0),
    )
    mask0 = None if cache.mask0 is None else jnp.asarray(cache.mask0)
    ell = jnp.arange(arr.shape[0], dtype=real_dtype).reshape((arr.shape[0], 1, 1, 1, 1))
    ell_p1 = ell + 1.0
    bgrad = jnp.asarray(cache.bgrad, dtype=real_dtype).reshape(
        (1, 1, 1, 1, int(jnp.asarray(cache.bgrad).shape[-1]))
    )
    cv = jnp.asarray(cache.cv_d, dtype=real_dtype).reshape(
        (1, 1) + tuple(jnp.asarray(cache.cv_d).shape)
    )
    gb = jnp.asarray(cache.gb_d, dtype=real_dtype).reshape(
        (1, 1) + tuple(jnp.asarray(cache.gb_d).shape)
    )
    omega_d_scale = jnp.asarray(params.omega_d_scale, dtype=real_dtype)
    kpar_scale = jnp.asarray(params.kpar_scale, dtype=real_dtype)
    imag = jnp.asarray(1j, dtype=arr.dtype)
    omega_star = (
        imag
        * jnp.asarray(params.omega_star_scale, dtype=real_dtype)
        * jnp.asarray(cache.ky, dtype=real_dtype)
    )
    omega_star_s = omega_star.reshape((1, omega_star.shape[0], 1, 1))
    tprim_s = jnp.asarray(params.R_over_LTi, dtype=real_dtype).reshape(-1)[0]
    fprim_s = jnp.asarray(params.R_over_Ln, dtype=real_dtype).reshape(-1)[0]
    jl_m1 = operator_shift_axis(jl, -1, axis=0)
    jl_p1 = operator_shift_axis(jl, 1, axis=0)
    l4 = jnp.asarray(cache.l4, dtype=real_dtype).reshape((arr.shape[0], 1, 1, 1))
    w_streaming = jnp.asarray(term_weights.streaming, dtype=real_dtype)
    w_mirror = jnp.asarray(term_weights.mirror, dtype=real_dtype)
    w_curv = jnp.asarray(term_weights.curvature, dtype=real_dtype)
    w_gradb = jnp.asarray(term_weights.gradb, dtype=real_dtype)
    w_diamag = jnp.asarray(term_weights.diamagnetic, dtype=real_dtype)

    def shift_m(local, *, offset: int):
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

    def fused(local):
        global_m = jax.lax.axis_index(axis_name) * local_m + local_m_index
        global_m_real = global_m.astype(real_dtype)
        m0 = (global_m == 0).astype(local.dtype)
        local_gm0 = jnp.sum(local * m0, axis=1)
        local_nbar = density_s * charge_s * jnp.sum(jl * local_gm0, axis=0)
        phi = jax.lax.psum(local_nbar, axis_name) / den_safe
        if mask0 is not None:
            phi = jnp.where(mask0, 0.0, phi)

        dlocal_dz = operator_grad_z_periodic(local, kz=cache.kz)
        lower = shift_m(dlocal_dz, offset=-1)
        upper = shift_m(dlocal_dz, offset=1)
        streaming = -vth_s * (
            jnp.sqrt(global_m_real + 1.0) * upper + jnp.sqrt(global_m_real) * lower
        )
        field_drive_m1 = (global_m == 1).astype(local.dtype) * (-zt * vth_s * jl * phi)[
            :, None, ...
        ]
        streaming = streaming + kpar_scale * operator_grad_z_periodic(
            field_drive_m1, kz=cache.kz
        )

        h = local + (global_m == 0).astype(local.dtype) * (zt * jl * phi)[:, None, ...]
        h_m_p1 = shift_m(h, offset=1)
        h_m_m1 = shift_m(h, offset=-1)
        mirror_term = (
            -jnp.sqrt(global_m_real + 1.0) * ell_p1 * h_m_p1
            - jnp.sqrt(global_m_real + 1.0)
            * ell
            * operator_shift_axis(h_m_p1, -1, axis=0)
            + jnp.sqrt(global_m_real) * ell * h_m_m1
            + jnp.sqrt(global_m_real) * ell_p1 * operator_shift_axis(h_m_m1, 1, axis=0)
        )
        mirror = -vth_s * bgrad * mirror_term

        h_m_p2 = shift_m(h, offset=2)
        h_m_m2 = shift_m(h, offset=-2)
        curv_term = (
            jnp.sqrt((global_m_real + 1.0) * (global_m_real + 2.0)) * h_m_p2
            + (2.0 * global_m_real + 1.0) * h
            + jnp.sqrt(global_m_real * (global_m_real - 1.0)) * h_m_m2
        )
        gradb_term = (
            (ell + 1.0) * operator_shift_axis(h, 1, axis=0)
            + (2.0 * ell + 1.0) * h
            + ell * operator_shift_axis(h, -1, axis=0)
        )
        curvature = -(imag * tz_s * omega_d_scale * cv) * curv_term
        gradb = -(imag * tz_s * omega_d_scale * gb) * gradb_term

        drive_m0 = (
            omega_star_s
            * phi
            * (
                jl_m1 * (l4 * tprim_s)
                + jl * (fprim_s + 2.0 * l4 * tprim_s)
                + jl_p1 * ((l4 + 1.0) * tprim_s)
            )
        )
        drive_m2 = (
            omega_star_s
            * phi
            * jl
            * (tprim_s / jnp.sqrt(jnp.asarray(2.0, dtype=real_dtype)))
        )
        diamagnetic = (global_m == 0).astype(local.dtype) * drive_m0[:, None, ...]
        diamagnetic = (
            diamagnetic + (global_m == 2).astype(local.dtype) * drive_m2[:, None, ...]
        )

        rhs = (
            w_streaming * streaming
            + w_mirror * mirror
            + w_curv * curvature
            + w_gradb * gradb
        )
        rhs = rhs + w_diamag * diamagnetic
        return rhs, phi

    cache_key = (
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
        tuple(str(device) for device in device_list[:m_chunks]),
        axis_name,
    )
    cached = _FUSED_ELECTROSTATIC_SLICE_KERNEL_CACHE.get(cache_key)
    if cached is None:
        mapped = jax.jit(
            jax.shard_map(
                fused,
                mesh=mesh,
                in_specs=state_spec,
                out_specs=(state_spec, phi_spec),
                axis_names={axis_name},
            )
        )
        cached = (mapped, sharding)
        _FUSED_ELECTROSTATIC_SLICE_KERNEL_CACHE[cache_key] = cached
    else:
        mapped, sharding = cached
    return mapped(jax.device_put(arr, sharding))


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
        curvature_gradb_drift_shard_map,
        diamagnetic_drive_shard_map,
        electrostatic_phi_shard_map,
        mirror_drift_shard_map,
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


def linear_rhs_parallel_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    parallel: Any | None = None,
    use_jit: bool = True,
    use_custom_vjp: bool = True,
    dt: jnp.ndarray | float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute linear RHS with an explicit, disabled-by-default parallel route.

    ``parallel=None`` and ``parallel.strategy="serial"`` are exact aliases for
    :func:`linear_rhs_cached`. The non-serial velocity routes are opt-in,
    Hermite-axis-only identity gates. ``backend="auto"`` selects the most
    complete currently gated electrostatic route when the term set is eligible;
    otherwise callers must request a narrower explicit backend.
    """

    from spectraxgk.operators.linear.rhs import linear_rhs_cached

    if (
        parallel is None
        or str(getattr(parallel, "strategy", "serial")).lower() == "serial"
    ):
        return linear_rhs_cached(
            G,
            cache,
            params,
            terms=terms,
            use_jit=use_jit,
            use_custom_vjp=use_custom_vjp,
            dt=dt,
        )

    strategy = str(getattr(parallel, "strategy", "serial")).lower().replace("-", "_")
    backend = str(getattr(parallel, "backend", "auto")).lower().replace("-", "_")
    axis = str(getattr(parallel, "axis", "hermite")).lower().replace("-", "_")
    if strategy == "velocity" and backend == "auto":
        if axis not in {"m", "hermite"}:
            raise NotImplementedError(
                "velocity sharding currently supports only the Hermite axis"
            )
        if _is_electrostatic_slice_terms(terms):
            backend = "electrostatic_linear_slices"
        else:
            raise NotImplementedError(
                "backend='auto' can only select gated electrostatic velocity routes; "
                "disable collision/EM/end-damping terms or request an explicit backend"
            )
    if strategy == "velocity" and backend in {
        "streaming_only",
        "linear_streaming_only",
    }:
        if axis not in {"m", "hermite"}:
            raise NotImplementedError(
                "streaming-only velocity sharding currently supports only the Hermite axis"
            )
        if not _is_streaming_only_terms(terms):
            raise NotImplementedError(
                "velocity streaming route requires streaming-only LinearTerms"
            )
        return linear_rhs_streaming_velocity_sharded(
            G,
            cache,
            params,
            num_devices=getattr(parallel, "num_devices", None),
        )
    if strategy == "velocity" and backend in {
        "streaming_electrostatic",
        "linear_streaming_electrostatic",
    }:
        if axis not in {"m", "hermite"}:
            raise NotImplementedError(
                "electrostatic streaming velocity sharding currently supports only the Hermite axis"
            )
        if not _is_streaming_only_terms(terms):
            raise NotImplementedError(
                "electrostatic velocity streaming route requires streaming-only LinearTerms"
            )
        return linear_rhs_streaming_electrostatic_velocity_sharded(
            G,
            cache,
            params,
            num_devices=getattr(parallel, "num_devices", None),
            use_custom_vjp=use_custom_vjp,
        )
    if strategy == "velocity" and backend in {
        "electrostatic_linear_slices",
        "linear_electrostatic_slices",
    }:
        if axis not in {"m", "hermite"}:
            raise NotImplementedError(
                "electrostatic slice velocity sharding currently supports only the Hermite axis"
            )
        if not _is_electrostatic_slice_terms(terms):
            raise NotImplementedError(
                "electrostatic slice route requires collision/EM terms to be disabled"
            )
        return linear_rhs_electrostatic_slices_velocity_sharded(
            G,
            cache,
            params,
            terms=terms,
            num_devices=getattr(parallel, "num_devices", None),
        )

    raise NotImplementedError(
        "parallel linear RHS currently supports only strategy='velocity' with gated electrostatic backends"
    )
