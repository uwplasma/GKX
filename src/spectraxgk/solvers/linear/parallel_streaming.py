"""Velocity-sharded streaming routes for the linear RHS."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.params import LinearParams, _as_species_array
from spectraxgk.solvers.linear.parallel_common import _resolve_parallel_devices


def linear_rhs_streaming_electrostatic_species_hermite_sharded(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    species_chunks: int = 2,
    hermite_chunks: int = 2,
    devices: Any | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute electrostatic streaming on a species-by-Hermite device mesh."""

    import jax
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    from spectraxgk.operators.linear.params import _as_species_array
    from spectraxgk.terms.operators import grad_z_periodic

    arr = jnp.asarray(G)
    if arr.ndim != 6:
        raise ValueError("mixed species-Hermite streaming requires a 6D state")
    if bool(getattr(cache, "use_twist_shift", False)):
        raise NotImplementedError(
            "mixed species-Hermite streaming currently requires a periodic z grid"
        )
    ns = int(arr.shape[0])
    nm = int(arr.shape[2])
    s_chunks = int(species_chunks)
    m_chunks = int(hermite_chunks)
    if s_chunks != ns:
        raise ValueError("mixed streaming currently requires one species per mesh row")
    if m_chunks < 2 or nm % m_chunks != 0:
        raise ValueError("Hermite chunks must be at least two and divide Nm evenly")
    device_list = _resolve_parallel_devices(
        num_devices=s_chunks * m_chunks, devices=devices
    )
    mesh = Mesh(
        np.asarray(device_list).reshape((s_chunks, m_chunks)),
        ("species", "m"),
    )
    state_spec = PartitionSpec("species", None, "m", None, None, None)
    jl_spec = PartitionSpec("species", None, None, None, None)
    species_spec = PartitionSpec("species")
    phi_spec = PartitionSpec(None, None, None)
    state_sharding = NamedSharding(mesh, state_spec)
    jl_sharding = NamedSharding(mesh, jl_spec)
    species_sharding = NamedSharding(mesh, species_spec)
    real_dtype = jnp.real(arr).dtype
    charge = _as_species_array(params.charge_sign, ns, "charge_sign").astype(real_dtype)
    density = _as_species_array(params.density, ns, "density").astype(real_dtype)
    tz = _as_species_array(params.tz, ns, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)
    local_m = nm // m_chunks
    lower_pairs = tuple((index, index + 1) for index in range(m_chunks - 1))
    upper_pairs = tuple((index, index - 1) for index in range(1, m_chunks))

    def mixed_streaming(local_state, local_jl, charge_s, density_s, tz_s, vth_s):
        m_start = jax.lax.axis_index("m") * local_m
        global_m = m_start + jnp.arange(local_m, dtype=jnp.int32)
        m_shape = (1, 1, local_m, 1, 1, 1)
        m_real = global_m.astype(real_dtype).reshape(m_shape)
        m0 = (global_m == 0).astype(local_state.dtype).reshape(m_shape)
        m1 = (global_m == 1).astype(local_state.dtype).reshape(m_shape)

        gm0 = jnp.sum(local_state * m0, axis=2)
        weight = density_s[:, None, None, None] * charge_s[:, None, None, None]
        local_nbar = jnp.sum(weight * jnp.sum(local_jl * gm0, axis=1), axis=0)
        nbar = jax.lax.psum(local_nbar, ("species", "m"))
        g0 = jnp.sum(local_jl * local_jl, axis=1)
        zt_s = jnp.where(tz_s == 0.0, 0.0, 1.0 / tz_s)
        local_qneut = jnp.sum(weight * zt_s[:, None, None, None] * (1.0 - g0), axis=0)
        qneut = jax.lax.psum(local_qneut, "species")
        tau_e = jnp.asarray(params.tau_e, dtype=real_dtype)
        denominator = tau_e + qneut
        denominator_safe = jnp.where(denominator == 0.0, jnp.inf, denominator)
        jacobian = jnp.asarray(cache.jacobian, dtype=real_dtype)
        jac = jacobian[None, None, :]
        average_numerator = jnp.sum(
            jnp.where(jac == 0.0, 0.0, nbar / denominator_safe * jac), axis=-1
        )
        average_denominator = jnp.sum(jacobian * qneut / denominator_safe, axis=-1)
        average_denominator_safe = jnp.where(
            average_denominator == 0.0, jnp.inf, average_denominator
        )
        ky0 = (cache.ky == 0.0)[:, None]
        finite_kx = (jnp.arange(average_numerator.shape[1]) > 0)[None, :]
        phi_average = jnp.where(
            ky0 & finite_kx,
            average_numerator / average_denominator_safe,
            0.0,
        )
        phi = jax.lax.cond(
            jnp.any(tau_e > 0.0),
            lambda _: (nbar + tau_e * phi_average[..., None]) / denominator_safe,
            lambda _: nbar / denominator_safe,
            operand=None,
        )
        phi = jnp.where(cache.mask0, 0.0, phi)

        lower_boundary = local_state[:, :, -1:, ...]
        lower_received = jax.lax.ppermute(lower_boundary, "m", lower_pairs)
        lower = jnp.concatenate([lower_received, local_state[:, :, :-1, ...]], axis=2)
        upper_boundary = local_state[:, :, :1, ...]
        upper_received = jax.lax.ppermute(upper_boundary, "m", upper_pairs)
        upper = jnp.concatenate([local_state[:, :, 1:, ...], upper_received], axis=2)
        vth6 = vth_s[:, None, None, None, None, None]
        ladder = -vth6 * (jnp.sqrt(m_real + 1.0) * upper + jnp.sqrt(m_real) * lower)
        field_drive = (
            -zt_s[:, None, None, None, None]
            * vth_s[:, None, None, None, None]
            * local_jl
            * phi[None, None, ...]
        )
        pre_derivative = ladder + m1 * field_drive[:, :, None, ...]
        rhs = jnp.asarray(params.kpar_scale, dtype=real_dtype) * grad_z_periodic(
            pre_derivative, kz=cache.kz
        )
        return rhs, phi

    mapped = jax.shard_map(
        mixed_streaming,
        mesh=mesh,
        in_specs=(
            state_spec,
            jl_spec,
            species_spec,
            species_spec,
            species_spec,
            species_spec,
        ),
        out_specs=(state_spec, phi_spec),
        axis_names={"species", "m"},
    )
    return mapped(
        jax.device_put(arr, state_sharding),
        jax.device_put(cache.Jl, jl_sharding),
        jax.device_put(charge, species_sharding),
        jax.device_put(density, species_sharding),
        jax.device_put(tz, species_sharding),
        jax.device_put(vth, species_sharding),
    )


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
    "linear_rhs_streaming_electrostatic_species_hermite_sharded",
    "linear_rhs_streaming_velocity_sharded",
]
