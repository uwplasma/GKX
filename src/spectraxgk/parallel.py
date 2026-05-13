"""Production parallelization helpers for independent scan and ensemble work."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ParallelIdentityReport:
    """Numerical-identity report for an independent parallel execution path."""

    kind: str
    backend: str
    requested_workers: int
    actual_workers: int
    problem_size: int
    identity_passed: bool
    max_abs_error: float
    max_rel_error: float
    atol: float
    rtol: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report for artifacts and CI gates."""

        return asdict(self)


@dataclass(frozen=True)
class IndependentWorkerMetadata:
    """Resolved worker metadata for ordered independent Python tasks."""

    requested_workers: int
    actual_workers: int
    problem_size: int
    executor: str
    parallel_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable worker metadata payload."""

        return asdict(self)


class IndependentMapExecutionError(RuntimeError):
    """Worker failure annotated with independent-map execution metadata."""

    def __init__(
        self,
        index: int,
        executor: str,
        actual_workers: int,
        original_type: str,
        original_message: str,
    ) -> None:
        self.index = int(index)
        self.executor = str(executor)
        self.actual_workers = int(actual_workers)
        self.original_type = str(original_type)
        self.original_message = str(original_message)
        super().__init__(
            "independent_map task "
            f"{self.index} failed with executor='{self.executor}' "
            f"and actual_workers={self.actual_workers}: "
            f"{self.original_type}: {self.original_message}"
        )

    def __reduce__(self) -> tuple[type[IndependentMapExecutionError], tuple[Any, ...]]:
        """Keep process-pool transport picklable on worker exceptions."""

        return (
            type(self),
            (
                self.index,
                self.executor,
                self.actual_workers,
                self.original_type,
                self.original_message,
            ),
        )


def _tree_error_stats(reference: Any, observed: Any) -> tuple[float, float]:
    """Return max absolute and relative errors for matching pytrees."""

    ref_leaves, ref_tree = jax.tree_util.tree_flatten(reference)
    obs_leaves, obs_tree = jax.tree_util.tree_flatten(observed)
    if repr(ref_tree) != repr(obs_tree):
        raise ValueError("reference and observed pytrees have different structures")
    if not ref_leaves:
        return 0.0, 0.0

    max_abs = 0.0
    max_rel = 0.0
    for ref_leaf, obs_leaf in zip(ref_leaves, obs_leaves, strict=True):
        ref = np.asarray(ref_leaf)
        obs = np.asarray(obs_leaf)
        if ref.shape != obs.shape:
            raise ValueError(
                f"reference and observed leaf shapes differ: {ref.shape} != {obs.shape}"
            )
        delta = np.abs(obs - ref)
        abs_err = float(np.max(delta)) if delta.size else 0.0
        scale = float(np.max(np.abs(ref))) if ref.size else 0.0
        rel_err = abs_err / max(scale, np.finfo(float).tiny)
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
    return max_abs, max_rel


def parallel_identity_report(
    reference: Any,
    observed: Any,
    *,
    kind: str,
    problem_size: int,
    requested_workers: int,
    actual_workers: int | None = None,
    backend: str | None = None,
    atol: float = 1e-12,
    rtol: float = 1e-10,
    metadata: dict[str, Any] | None = None,
) -> ParallelIdentityReport:
    """Build a numerical-identity report for serial-vs-parallel outputs."""

    requested = int(requested_workers)
    actual = int(requested if actual_workers is None else actual_workers)
    size = int(problem_size)
    tolerance_atol = float(atol)
    tolerance_rtol = float(rtol)
    if requested < 1:
        raise ValueError("requested_workers must be >= 1")
    if actual < 1 or actual > requested:
        raise ValueError("actual_workers must be in [1, requested_workers]")
    if size < 1:
        raise ValueError("problem_size must be >= 1")
    if tolerance_atol < 0.0 or tolerance_rtol < 0.0:
        raise ValueError("atol and rtol must be non-negative")

    max_abs, max_rel = _tree_error_stats(reference, observed)
    passed = bool(max_abs <= tolerance_atol or max_rel <= tolerance_rtol)
    return ParallelIdentityReport(
        kind=str(kind),
        backend=str(backend or jax.default_backend()),
        requested_workers=requested,
        actual_workers=actual,
        problem_size=size,
        identity_passed=passed,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        atol=tolerance_atol,
        rtol=tolerance_rtol,
        metadata=dict(metadata or {}),
    )


def _normalize_independent_executor(executor: str) -> str:
    executor_key = str(executor).strip().lower()
    if executor_key in {"thread", "threads"}:
        return "thread"
    if executor_key in {"process", "processes"}:
        return "process"
    raise ValueError("executor must be 'thread' or 'process'")


def independent_worker_metadata(
    problem_size: int,
    *,
    workers: int = 1,
    executor: str = "thread",
) -> IndependentWorkerMetadata:
    """Resolve independent-task worker counts and normalized executor metadata."""

    size = int(problem_size)
    requested = int(workers)
    if size < 0:
        raise ValueError("problem_size must be non-negative")
    if requested < 1:
        raise ValueError("workers must be >= 1")
    executor_key = _normalize_independent_executor(executor)
    actual = min(requested, size) if size else 0
    return IndependentWorkerMetadata(
        requested_workers=requested,
        actual_workers=actual,
        problem_size=size,
        executor=executor_key,
        parallel_enabled=actual > 1,
    )


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


def batch_map_identity_report(
    fn: Callable[[jnp.ndarray], Any],
    values: jnp.ndarray | np.ndarray,
    *,
    batch_size: int | None = None,
    devices: Iterable[jax.Device] | None = None,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> ParallelIdentityReport:
    """Compare ``batch_map`` against ``vmap`` and return a CI-ready gate report."""

    arr = jnp.asarray(values)
    if arr.shape[0] == 0:
        raise ValueError("values must contain at least one item")
    device_list = list(devices) if devices is not None else list(jax.devices())
    requested = max(1, len(device_list))
    observed = batch_map(fn, arr, batch_size=batch_size, devices=device_list)
    reference = jax.vmap(fn)(arr)
    return parallel_identity_report(
        reference,
        observed,
        kind="batch_map_serial_identity",
        problem_size=int(arr.shape[0]),
        requested_workers=requested,
        actual_workers=min(requested, int(arr.shape[0])),
        backend=jax.default_backend(),
        atol=atol,
        rtol=rtol,
        metadata={
            "batch_size": None if batch_size is None else int(batch_size),
            "tree": str(jax.tree_util.tree_structure(reference)),
        },
    )


def ky_scan_batches(ky_values: np.ndarray, *, n_batches: int) -> list[np.ndarray]:
    """Return balanced ``k_y`` chunks for independent linear-scan execution."""

    ky = np.asarray(ky_values, dtype=float)
    if ky.ndim != 1:
        raise ValueError("ky_values must be one-dimensional")
    return split_evenly(ky, n_batches)


def _run_indexed_independent_task(
    task: tuple[Callable[[Any], Any], int, Any, str, int],
) -> tuple[int, Any]:
    fn, index, item, executor, actual_workers = task
    try:
        return index, fn(item)
    except Exception as exc:
        raise IndependentMapExecutionError(
            index,
            executor,
            actual_workers,
            type(exc).__name__,
            str(exc),
        ) from exc


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
    worker_metadata = independent_worker_metadata(
        len(items),
        workers=workers,
        executor=executor,
    )
    if not items:
        return []
    if worker_metadata.actual_workers == 1:
        return [fn(item) for item in items]

    tasks = (
        (fn, index, item, worker_metadata.executor, worker_metadata.actual_workers)
        for index, item in enumerate(items)
    )
    executor_cls = (
        ThreadPoolExecutor
        if worker_metadata.executor == "thread"
        else ProcessPoolExecutor
    )
    try:
        with executor_cls(max_workers=worker_metadata.actual_workers) as pool:
            indexed_results = list(pool.map(_run_indexed_independent_task, tasks))
    except IndependentMapExecutionError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "independent_map executor "
            f"'{worker_metadata.executor}' failed before completing "
            f"{worker_metadata.problem_size} task(s) with "
            f"actual_workers={worker_metadata.actual_workers}: {exc}"
        ) from exc

    indices = [index for index, _ in indexed_results]
    expected_indices = list(range(len(items)))
    if indices != expected_indices:
        raise RuntimeError(
            "independent_map executor returned results out of serial order: "
            f"{indices} != {expected_indices}"
        )
    return [result for _, result in indexed_results]


def independent_map_identity_report(
    fn: Callable[[Any], Any],
    values: Iterable[Any],
    *,
    workers: int = 1,
    executor: str = "thread",
    atol: float = 1e-12,
    rtol: float = 1e-10,
    metadata: dict[str, Any] | None = None,
) -> ParallelIdentityReport:
    """Compare ``independent_map`` against a serial list-comprehension run."""

    items = list(values)
    worker_metadata = independent_worker_metadata(
        len(items),
        workers=workers,
        executor=executor,
    )
    if worker_metadata.problem_size < 1:
        raise ValueError("values must contain at least one item")

    reference = [fn(item) for item in items]
    observed = independent_map(
        fn,
        items,
        workers=worker_metadata.requested_workers,
        executor=worker_metadata.executor,
    )
    report_metadata = dict(metadata or {})
    report_metadata.update(
        {
            "executor": worker_metadata.executor,
            "parallel_enabled": worker_metadata.parallel_enabled,
            "worker_metadata": worker_metadata.to_dict(),
            "tree": str(jax.tree_util.tree_structure(reference)),
        }
    )
    return parallel_identity_report(
        reference,
        observed,
        kind="independent_map_serial_identity",
        problem_size=worker_metadata.problem_size,
        requested_workers=worker_metadata.requested_workers,
        actual_workers=worker_metadata.actual_workers,
        backend=f"python:{worker_metadata.executor}",
        atol=atol,
        rtol=rtol,
        metadata=report_metadata,
    )


__all__ = [
    "IndependentMapExecutionError",
    "IndependentWorkerMetadata",
    "ParallelIdentityReport",
    "batch_map",
    "batch_map_identity_report",
    "independent_map",
    "independent_map_identity_report",
    "independent_worker_metadata",
    "ky_scan_batches",
    "pad_to_multiple",
    "parallel_identity_report",
    "split_evenly",
]
