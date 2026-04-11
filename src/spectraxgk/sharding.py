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
