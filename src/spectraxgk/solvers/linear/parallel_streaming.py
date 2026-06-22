"""Velocity-sharded streaming routes for the linear RHS."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.params import LinearParams, _as_species_array
from spectraxgk.solvers.linear.parallel_common import _resolve_parallel_devices


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

    from spectraxgk.parallel.velocity import periodic_streaming_shard_map
    from spectraxgk.terms.operators import grad_z_periodic as operator_grad_z_periodic

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
    the benchmark-compatible electrostatic streaming field term. It is limited to periodic
    field-line grids and excludes electromagnetic fields by construction.
    """

    from spectraxgk.parallel.velocity import (
        build_velocity_sharding_plan,
        electrostatic_phi_shard_map,
    )

    del use_custom_vjp
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


__all__ = [
    "_electrostatic_streaming_field_rhs",
    "_streaming_electrostatic_from_phi_velocity_sharded",
    "linear_rhs_streaming_electrostatic_velocity_sharded",
    "linear_rhs_streaming_velocity_sharded",
]
