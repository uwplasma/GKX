"""Production parallelization helpers for independent scan and ensemble work."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

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


def _concat_batch_outputs(outputs: list[Any]) -> Any:
    """Concatenate a sequence of batched array or pytree outputs."""

    if not outputs:
        raise ValueError("cannot concatenate an empty batch output list")
    return jax.tree_util.tree_map(lambda *parts: jnp.concatenate(parts, axis=0), *outputs)


def batch_map(
    fn: Callable[[jnp.ndarray], Any],
    values: jnp.ndarray | np.ndarray,
    *,
    batch_size: int | None = None,
    devices: Iterable[jax.Device] | None = None,
) -> Any:
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
        return _concat_batch_outputs(outputs)

    ndev = len(device_list)
    per_device = max(1, int(np.ceil(chunk_size / ndev)))
    pmapped = jax.pmap(lambda shard: jax.vmap(fn)(shard), devices=device_list)
    outputs = []
    for chunk in jnp.array_split(arr, int(np.ceil(arr.shape[0] / chunk_size)), axis=0):
        padded, original_n = pad_to_multiple(chunk, ndev * per_device)
        sharded = padded.reshape((ndev, per_device) + tuple(padded.shape[1:]))
        mapped = pmapped(sharded)
        outputs.append(
            jax.tree_util.tree_map(
                lambda leaf: jnp.asarray(leaf).reshape((ndev * per_device,) + tuple(jnp.asarray(leaf).shape[2:]))[
                    :original_n
                ],
                mapped,
            )
        )
    return _concat_batch_outputs(outputs)


def ky_scan_batches(ky_values: np.ndarray, *, n_batches: int) -> list[np.ndarray]:
    """Return balanced ``k_y`` chunks for independent linear-scan execution."""

    ky = np.asarray(ky_values, dtype=float)
    if ky.ndim != 1:
        raise ValueError("ky_values must be one-dimensional")
    return split_evenly(ky, n_batches)


def independent_map(
    fn: Callable[[Any], Any],
    values: Iterable[Any],
    *,
    workers: int = 1,
    executor: str = "thread",
) -> list[Any]:
    """Map independent Python tasks while preserving serial result ordering.

    ``batch_map`` handles JAX-array workloads. This helper covers file-backed
    calibration, finite-difference, and UQ tasks whose individual units are
    independent Python calls. The acceptance contract is numerical identity
    with ``[fn(value) for value in values]``; timing is secondary.
    """

    items = list(values)
    n_workers = int(workers)
    if n_workers < 1:
        raise ValueError("workers must be >= 1")
    executor_key = str(executor).strip().lower()
    if executor_key not in {"thread", "threads", "process", "processes"}:
        raise ValueError("executor must be 'thread' or 'process'")
    if not items:
        return []
    if n_workers == 1 or len(items) == 1:
        return [fn(item) for item in items]

    max_workers = min(n_workers, len(items))
    if executor_key in {"thread", "threads"}:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(fn, items))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(fn, items))


__all__ = ["batch_map", "independent_map", "ky_scan_batches", "pad_to_multiple", "split_evenly"]
