"""Helpers for multi-device sharding of GK state arrays."""

from __future__ import annotations

from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def _mesh_from_devices(
    devices: Iterable[jax.Device] | None,
    axis_name: str,
) -> Mesh | None:
    device_list = list(devices) if devices is not None else list(jax.devices())
    if len(device_list) < 2:
        return None
    return Mesh(np.array(device_list), (axis_name,))


def resolve_state_sharding(
    G0: jnp.ndarray,
    spec: str | None,
    *,
    axis_name: str = "d",
    devices: Iterable[jax.Device] | None = None,
) -> NamedSharding | None:
    """Return a NamedSharding for the packed state, or None if disabled.

    Parameters
    ----------
    G0 : jnp.ndarray
        Initial state array with shape (Nl, Nm, Ny, Nx, Nz) or
        (Ns, Nl, Nm, Ny, Nx, Nz).
    spec : str | None
        Sharding directive. Allowed values:
        - None / "none" / "off": disable sharding
        - "auto" or "ky": shard along ky (recommended default)
        - "kx", "z", "l", "m", "species": shard along the named axis
    axis_name : str
        Mesh axis name for the sharded dimension.
    devices : Iterable[jax.Device] | None
        Optional explicit device list (useful for tests).
    """

    if spec is None:
        return None
    key = str(spec).strip().lower()
    if key in {"", "none", "off", "false", "0"}:
        return None
    if key == "auto":
        key = "ky"

    axis_map = {
        "ky": "ky",
        "kx": "kx",
        "z": "z",
        "l": "l",
        "m": "m",
        "species": "s",
        "s": "s",
    }
    if key not in axis_map:
        raise ValueError(
            "state_sharding must be one of 'auto', 'ky', 'kx', 'z', 'l', 'm', 'species', or 'none'"
        )

    mesh = _mesh_from_devices(devices, axis_name)
    if mesh is None:
        return None

    if G0.ndim == 5:
        dims = ["l", "m", "ky", "kx", "z"]
    elif G0.ndim == 6:
        dims = ["s", "l", "m", "ky", "kx", "z"]
    else:
        raise ValueError("G0 must have 5 or 6 dimensions for sharding")

    target_dim = axis_map[key]
    if target_dim not in dims:
        raise ValueError(f"Cannot shard along '{target_dim}' for state with dims {dims}")

    spec_list: list[str | None] = [None] * len(dims)
    spec_list[dims.index(target_dim)] = axis_name
    return NamedSharding(mesh, PartitionSpec(*spec_list))


SPECIES_MESH_AXIS = "s"
HERMITE_MESH_AXIS = "m"


def build_species_hermite_mesh(
    plan,
    *,
    devices: Iterable[jax.Device] | None = None,
) -> Mesh:
    """Return the 2-D ``(species, hermite)`` device mesh a plan asks for.

    Species vary slowest so that a species-only mesh keeps each species'
    devices adjacent, which makes the halo-free configuration the natural one
    on a two-GPU box.
    """

    device_list = list(devices) if devices is not None else list(jax.devices())
    ns_chunks = int(plan.chunks.get(SPECIES_MESH_AXIS, 1))
    nm_chunks = int(plan.chunks.get(HERMITE_MESH_AXIS, 1))
    needed = ns_chunks * nm_chunks
    if len(device_list) < needed:
        raise ValueError(
            f"the ({ns_chunks}, {nm_chunks}) species-Hermite mesh needs {needed} "
            f"devices, but {len(device_list)} are visible"
        )
    grid = np.array(device_list[:needed], dtype=object).reshape(ns_chunks, nm_chunks)
    return Mesh(grid, (SPECIES_MESH_AXIS, HERMITE_MESH_AXIS))


def species_hermite_state_spec(ndim: int) -> PartitionSpec:
    """Return ``P('s', None, 'm', None, None, None)`` for a packed state.

    ``(ky, kx)``, Laguerre and ``z`` stay replicated: every FFT in the bracket
    and in parallel streaming runs over one of them, so sharding any of them
    would put a distributed transpose inside the RHS.
    """

    if ndim == 6:
        return PartitionSpec(SPECIES_MESH_AXIS, None, HERMITE_MESH_AXIS, None, None, None)
    if ndim == 5:
        return PartitionSpec(None, HERMITE_MESH_AXIS, None, None, None)
    raise ValueError("packed state must have 5 or 6 dimensions")


def resolve_species_hermite_sharding(
    G0: jnp.ndarray,
    plan,
    *,
    devices: Iterable[jax.Device] | None = None,
    mesh: Mesh | None = None,
) -> NamedSharding:
    """Return the ``NamedSharding`` that places a packed state on the mesh."""

    resolved = mesh if mesh is not None else build_species_hermite_mesh(
        plan, devices=devices
    )
    return NamedSharding(resolved, species_hermite_state_spec(int(G0.ndim)))


__all__ = [
    "HERMITE_MESH_AXIS",
    "SPECIES_MESH_AXIS",
    "build_species_hermite_mesh",
    "resolve_species_hermite_sharding",
    "resolve_state_sharding",
    "species_hermite_state_spec",
]
