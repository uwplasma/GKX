"""Production parallelization helpers for independent scan and ensemble work."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np


def split_evenly(values: np.ndarray, n_parts: int) -> list[np.ndarray]:
    """Split an array into nonempty, nearly equal chunks along axis zero."""

    arr = np.asarray(values)
    parts = int(n_parts)
    if parts < 1:
        raise ValueError("n_parts must be >= 1")
    if arr.shape[0] == 0:
        return []
    return [chunk for chunk in np.array_split(arr, min(parts, arr.shape[0]), axis=0) if chunk.shape[0] > 0]


def pad_to_multiple(values: jnp.ndarray, multiple: int) -> tuple[jnp.ndarray, int]:
    """Pad axis zero by edge repetition so its length is divisible by ``multiple``."""

    arr = jnp.asarray(values)
    n = int(arr.shape[0])
    m = int(multiple)
    if m < 1:
        raise ValueError("multiple must be >= 1")
    if n == 0:
        raise ValueError("cannot pad an empty batch")
    remainder = n % m
    if remainder == 0:
        return arr, n
    pad = m - remainder
    tail = jnp.repeat(arr[-1:], pad, axis=0)
    return jnp.concatenate([arr, tail], axis=0), n


def batch_map(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    values: jnp.ndarray | np.ndarray,
    *,
    batch_size: int | None = None,
    devices: Iterable[jax.Device] | None = None,
) -> jnp.ndarray:
    """Map ``fn`` over independent inputs with optional multi-device batching.

    This helper is intended for embarrassingly parallel physics workloads such
    as linear ``k_y`` scans, parameter sweeps, and UQ ensembles. It preserves
    numerical identity with ``jax.vmap(fn)(values)`` while allowing the leading
    batch axis to be distributed over available devices when more than one
    device is supplied.
    """

    arr = jnp.asarray(values)
    if arr.shape[0] == 0:
        raise ValueError("values must contain at least one item")
    chunk_size = int(arr.shape[0] if batch_size is None else batch_size)
    if chunk_size < 1:
        raise ValueError("batch_size must be >= 1")

    device_list = list(devices) if devices is not None else list(jax.devices())
    if len(device_list) < 2:
        outputs = [jax.vmap(fn)(chunk) for chunk in jnp.array_split(arr, int(np.ceil(arr.shape[0] / chunk_size)), axis=0)]
        return jnp.concatenate(outputs, axis=0)

    ndev = len(device_list)
    per_device = max(1, int(np.ceil(chunk_size / ndev)))
    pmapped = jax.pmap(lambda shard: jax.vmap(fn)(shard), devices=device_list)
    outputs = []
    for chunk in jnp.array_split(arr, int(np.ceil(arr.shape[0] / chunk_size)), axis=0):
        padded, original_n = pad_to_multiple(chunk, ndev * per_device)
        sharded = padded.reshape((ndev, per_device) + tuple(padded.shape[1:]))
        mapped = pmapped(sharded).reshape((ndev * per_device,) + tuple(jax.eval_shape(fn, padded[0]).shape))
        outputs.append(mapped[:original_n])
    return jnp.concatenate(outputs, axis=0)


def ky_scan_batches(ky_values: np.ndarray, *, n_batches: int) -> list[np.ndarray]:
    """Return balanced ``k_y`` chunks for independent linear-scan execution."""

    ky = np.asarray(ky_values, dtype=float)
    if ky.ndim != 1:
        raise ValueError("ky_values must be one-dimensional")
    return split_evenly(ky, n_batches)


__all__ = ["batch_map", "ky_scan_batches", "pad_to_multiple", "split_evenly"]
