"""Electrostatic field reduction and diamagnetic-drive microkernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from spectraxgk.parallel.velocity_plan import VelocityShardingPlan, _state_dims


@dataclass(frozen=True)
class _HermiteShardContext:
    mesh: Any
    spec: Any
    sharding: Any
    local_m: int
    local_m_index: Any


def _single_species_state_and_plan(
    state: Any,
    plan: VelocityShardingPlan,
    *,
    caller: str,
    active_axis_message: str,
) -> tuple[Any, int, int]:
    """Validate the common single-species Hermite-sharding contract."""

    import jax.numpy as jnp

    arr = jnp.asarray(state)
    if arr.ndim != 5:
        raise NotImplementedError(
            f"{caller} currently supports single-species 5D states"
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
        raise NotImplementedError(active_axis_message)
    if int(arr.shape[m_axis]) % m_chunks != 0:
        raise ValueError("Hermite dimension must divide evenly across Hermite chunks")
    return arr, m_axis, m_chunks


def _normalise_single_species_jl(Jl: Any) -> Any:
    import jax.numpy as jnp

    jl = jnp.asarray(Jl)
    if jl.ndim == 5:
        jl = jl[0]
    if jl.ndim != 4:
        raise ValueError("Jl must have shape (Nl, Ny, Nx, Nz) or (1, Nl, Ny, Nx, Nz)")
    return jl


def _normalise_species_jl(Jl: Any, *, species: int) -> Any:
    import jax.numpy as jnp

    jl = jnp.asarray(Jl)
    if jl.ndim != 5 or int(jl.shape[0]) != species:
        raise ValueError("Jl must have shape (Ns, Nl, Ny, Nx, Nz)")
    return jl


def _normalise_single_species_b(b: Any) -> Any:
    import jax.numpy as jnp

    value = jnp.asarray(b)
    if value.ndim == 4:
        value = value[0]
    if value.ndim != 3:
        raise ValueError("b must have shape (Ny, Nx, Nz) or (1, Ny, Nx, Nz)")
    return value


def _hermite_shard_context(
    arr: Any,
    *,
    m_axis: int,
    m_chunks: int,
    devices: Sequence[Any] | None,
    axis_name: str,
) -> _HermiteShardContext:
    """Build the mesh, sharding, and local Hermite indices for shard_map."""

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

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
    return _HermiteShardContext(
        mesh=mesh,
        spec=spec,
        sharding=sharding,
        local_m=local_m,
        local_m_index=local_m_index,
    )


def _diamagnetic_drive_from_global_m(
    *,
    state: Any,
    global_m: Any,
    phi: Any,
    Jl: Any,
    b: Any,
    l4: Any,
    tprim: Any,
    fprim: Any,
    omega_star_scale: Any,
    ky: Any,
    weight: Any,
) -> Any:
    import jax.numpy as jnp

    from spectraxgk.core.velocity import laguerre_gyroaverage_neighbors

    arr = jnp.asarray(state)
    real_dtype = jnp.real(arr).dtype
    jl = _normalise_single_species_jl(Jl)
    b_arr = _normalise_single_species_b(b)
    jl_m1, jl_p1 = laguerre_gyroaverage_neighbors(jl, b_arr, axis=0)
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
    b: Any,
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
        b=b,
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
    b: Any,
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

    arr, m_axis, m_chunks = _single_species_state_and_plan(
        state,
        plan,
        caller="diamagnetic_drive_shard_map",
        active_axis_message=(
            "diamagnetic drive gate currently supports only an active 'm' axis"
        ),
    )
    if m_chunks == 1:
        return diamagnetic_drive_reference(
            arr,
            phi=phi,
            Jl=Jl,
            b=b,
            l4=l4,
            tprim=tprim,
            fprim=fprim,
            omega_star_scale=omega_star_scale,
            ky=ky,
            weight=weight,
        )
    shard_ctx = _hermite_shard_context(
        arr,
        m_axis=m_axis,
        m_chunks=m_chunks,
        devices=devices,
        axis_name=axis_name,
    )

    def drive(local):
        global_m = (
            jax.lax.axis_index(axis_name) * shard_ctx.local_m + shard_ctx.local_m_index
        )
        return _diamagnetic_drive_from_global_m(
            state=local,
            global_m=global_m,
            phi=phi,
            Jl=Jl,
            b=b,
            l4=l4,
            tprim=tprim,
            fprim=fprim,
            omega_star_scale=omega_star_scale,
            ky=ky,
            weight=weight,
        )

    mapped = jax.shard_map(
        drive,
        mesh=shard_ctx.mesh,
        in_specs=shard_ctx.spec,
        out_specs=shard_ctx.spec,
        axis_names={axis_name},
    )
    return mapped(jax.device_put(arr, shard_ctx.sharding))


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
    """Return electrostatic phi from a full single- or multi-species state."""

    import jax.numpy as jnp

    arr = jnp.asarray(state)
    if arr.ndim not in (5, 6):
        raise ValueError("state must have shape (l,m,ky,kx,z) or (s,l,m,ky,kx,z)")
    multi_species = arr.ndim == 6
    species = int(arr.shape[0]) if multi_species else 1
    jl = (
        _normalise_species_jl(Jl, species=species)
        if multi_species
        else _normalise_single_species_jl(Jl)[None, ...]
    )
    species_state = arr if multi_species else arr[None, ...]
    real_dtype = jnp.real(arr).dtype
    charge_s = jnp.broadcast_to(jnp.asarray(charge, dtype=real_dtype), (species,))
    density_s = jnp.broadcast_to(jnp.asarray(density, dtype=real_dtype), (species,))
    tz_s = jnp.broadcast_to(jnp.asarray(tz, dtype=real_dtype), (species,))
    zt = jnp.where(tz_s == 0.0, 0.0, 1.0 / tz_s)
    nbar = jnp.sum(
        density_s[:, None, None, None]
        * charge_s[:, None, None, None]
        * jnp.sum(jl * species_state[:, :, 0, ...], axis=1),
        axis=0,
    )
    g0 = jnp.sum(jl * jl, axis=1)
    qneut = jnp.sum(
        density_s[:, None, None, None]
        * charge_s[:, None, None, None]
        * zt[:, None, None, None]
        * (1.0 - g0),
        axis=0,
    )
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
    """Solve electrostatic phi using a species- or Hermite-sharded reduction."""

    import jax
    import jax.numpy as jnp

    arr = jnp.asarray(state)
    if tuple(arr.shape) != tuple(plan.state_shape):
        raise ValueError(
            "state shape does not match the supplied velocity sharding plan"
        )
    if arr.ndim == 6:
        active_other_axes = tuple(axis for axis in plan.active_axes if axis != "s")
        if active_other_axes:
            raise NotImplementedError(
                "multi-species electrostatic reduction supports only an active 's' axis"
            )
        s_chunks = int(plan.chunks.get("s", 1))
        if int(arr.shape[0]) % s_chunks != 0:
            raise ValueError(
                "species dimension must divide evenly across species chunks"
            )
        if s_chunks == 1:
            return electrostatic_phi_reference(
                arr,
                Jl=Jl,
                tau_e=tau_e,
                charge=charge,
                density=density,
                tz=tz,
                mask0=mask0,
            )
        species = int(arr.shape[0])
        jl = _normalise_species_jl(Jl, species=species)
        real_dtype = jnp.real(arr).dtype
        charge_s = jnp.broadcast_to(jnp.asarray(charge, dtype=real_dtype), (species,))
        density_s = jnp.broadcast_to(jnp.asarray(density, dtype=real_dtype), (species,))
        tz_s = jnp.broadcast_to(jnp.asarray(tz, dtype=real_dtype), (species,))
        zt = jnp.where(tz_s == 0.0, 0.0, 1.0 / tz_s)
        device_list = list(devices) if devices is not None else list(jax.devices())
        if len(device_list) < s_chunks:
            raise ValueError(
                "not enough devices for the requested species decomposition"
            )
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        mesh = Mesh(np.asarray(device_list[:s_chunks]), (axis_name,))
        state_spec = PartitionSpec(axis_name, None, None, None, None, None)
        jl_spec = PartitionSpec(axis_name, None, None, None, None)
        vector_spec = PartitionSpec(axis_name)
        output_spec = PartitionSpec(None, None, None)
        state_sharding = NamedSharding(mesh, state_spec)
        jl_sharding = NamedSharding(mesh, jl_spec)
        vector_sharding = NamedSharding(mesh, vector_spec)

        def local_moments(local_state, local_jl, local_charge, local_density, local_zt):
            local_weight = (
                local_density[:, None, None, None] * local_charge[:, None, None, None]
            )
            local_nbar = jnp.sum(
                local_weight * jnp.sum(local_jl * local_state[:, :, 0, ...], axis=1),
                axis=0,
            )
            local_g0 = jnp.sum(local_jl * local_jl, axis=1)
            local_qneut = jnp.sum(
                local_weight * local_zt[:, None, None, None] * (1.0 - local_g0), axis=0
            )
            return (
                jax.lax.psum(local_nbar, axis_name),
                jax.lax.psum(local_qneut, axis_name),
            )

        species_mapped = jax.shard_map(
            local_moments,
            mesh=mesh,
            in_specs=(state_spec, jl_spec, vector_spec, vector_spec, vector_spec),
            out_specs=(output_spec, output_spec),
            axis_names={axis_name},
        )
        nbar, qneut = species_mapped(
            jax.device_put(arr, state_sharding),
            jax.device_put(jl, jl_sharding),
            jax.device_put(charge_s, vector_sharding),
            jax.device_put(density_s, vector_sharding),
            jax.device_put(zt, vector_sharding),
        )
        denominator = jnp.asarray(tau_e, dtype=real_dtype) + qneut
        phi = nbar / jnp.where(denominator == 0.0, jnp.inf, denominator)
        return phi if mask0 is None else jnp.where(jnp.asarray(mask0), 0.0, phi)

    arr, m_axis, m_chunks = _single_species_state_and_plan(
        arr,
        plan,
        caller="electrostatic_phi_shard_map",
        active_axis_message="electrostatic field reduction supports only an active 'm' axis",
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

    jl = _normalise_single_species_jl(Jl)
    real_dtype = jnp.real(arr).dtype
    charge_s = jnp.asarray(charge, dtype=real_dtype).reshape(-1)[0]
    density_s = jnp.asarray(density, dtype=real_dtype).reshape(-1)[0]
    tz_s = jnp.asarray(tz, dtype=real_dtype).reshape(-1)[0]
    zt = jnp.where(tz_s == 0.0, 0.0, 1.0 / tz_s)
    tau = jnp.asarray(tau_e, dtype=real_dtype)
    g0 = jnp.sum(jl * jl, axis=0)
    qneut = density_s * charge_s * zt * (1.0 - g0)
    den_safe = jnp.where(tau + qneut == 0.0, jnp.inf, tau + qneut)

    shard_ctx = _hermite_shard_context(
        arr,
        m_axis=m_axis,
        m_chunks=m_chunks,
        devices=devices,
        axis_name=axis_name,
    )
    input_spec = shard_ctx.spec
    from jax.sharding import PartitionSpec

    output_spec = PartitionSpec(*[None for _ in range(arr.ndim - 2)])

    def local_density(local):
        global_m = (
            jax.lax.axis_index(axis_name) * shard_ctx.local_m + shard_ctx.local_m_index
        )
        m0 = (global_m == 0).astype(local.dtype)
        local_gm0 = jnp.sum(local * m0, axis=m_axis)
        local_nbar = density_s * charge_s * jnp.sum(jl * local_gm0, axis=0)
        return jax.lax.psum(local_nbar, axis_name)

    hermite_mapped = jax.shard_map(
        local_density,
        mesh=shard_ctx.mesh,
        in_specs=input_spec,
        out_specs=output_spec,
        axis_names={axis_name},
    )
    nbar = hermite_mapped(jax.device_put(arr, shard_ctx.sharding))
    phi = nbar / den_safe
    if mask0 is not None:
        phi = jnp.where(jnp.asarray(mask0), 0.0, phi)
    return phi


__all__ = [
    "diamagnetic_drive_reference",
    "diamagnetic_drive_shard_map",
    "electrostatic_phi_reference",
    "electrostatic_phi_shard_map",
]
