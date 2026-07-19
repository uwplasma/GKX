"""Ordered Python-task parallelism for UQ and optimization ensembles."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import jax

from gkx.parallel.identity import (
    ParallelIdentityReport,
    parallel_identity_report,
)


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


@dataclass(frozen=True)
class IndependentEnsembleProvenanceReport:
    """End-to-end provenance gate for independent UQ/optimization ensembles."""

    kind: str
    workload: str
    executor: str
    requested_workers: int
    actual_workers: int
    problem_size: int
    passed: bool
    identity_passed: bool
    ordering_passed: bool
    worker_clipping_passed: bool
    reconstruction_identity_passed: bool
    exception_metadata_passed: bool
    serial_indices: tuple[int, ...]
    parallel_indices: tuple[int, ...]
    reconstructed_indices: tuple[int, ...]
    identity_report: ParallelIdentityReport
    reconstruction_report: dict[str, Any]
    exception_metadata: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable provenance payload."""

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


@dataclass(frozen=True)
class _IndexedPayloads:
    payloads: tuple[tuple[int, Any], ...]
    indices: tuple[int, ...]
    results: tuple[Any, ...]


@dataclass(frozen=True)
class _EnsembleReconstruction:
    contract: Any
    reconstructed_indices: tuple[int, ...]
    report: Any


@dataclass(frozen=True)
class _EnsembleGateChecks:
    ordering_passed: bool
    worker_clipping_passed: bool
    reconstruction_passed: bool
    exception_passed: bool
    exception_metadata: dict[str, Any]


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


def _run_provenance_indexed_task(
    task: tuple[Callable[[Any], Any], int, Any],
) -> tuple[int, Any]:
    fn, index, item = task
    return int(index), fn(item)


def _independent_exception_probe(value: str) -> str:
    if value == "fail":
        raise ValueError("provenance probe failure")
    return value


def _probe_exception_metadata(
    *,
    requested_workers: int,
    executor: str,
) -> tuple[bool, dict[str, Any]]:
    probe_workers = max(2, int(requested_workers))
    executor_key = _normalize_independent_executor(executor)
    expected_actual = 2
    try:
        independent_map(
            _independent_exception_probe,
            ["ok", "fail"],
            workers=probe_workers,
            executor=executor_key,
        )
    except IndependentMapExecutionError as exc:
        metadata = {
            "index": exc.index,
            "executor": exc.executor,
            "actual_workers": exc.actual_workers,
            "original_type": exc.original_type,
            "original_message": exc.original_message,
            "probe_workers": probe_workers,
        }
        passed = bool(
            exc.index == 1
            and exc.executor == executor_key
            and exc.actual_workers == expected_actual
            and exc.original_type == "ValueError"
            and "provenance probe failure" in exc.original_message
        )
        metadata["passed"] = passed
        return passed, metadata
    except Exception as exc:  # pragma: no cover - defensive fail-closed path
        return False, {
            "passed": False,
            "unexpected_type": type(exc).__name__,
            "unexpected_message": str(exc),
            "probe_workers": probe_workers,
            "executor": executor_key,
        }
    return False, {
        "passed": False,
        "missing_exception": True,
        "probe_workers": probe_workers,
        "executor": executor_key,
    }


def _validate_ensemble_workload(workload: str) -> None:
    if workload not in {"uq_ensemble", "optimization_ensemble"}:
        raise ValueError("workload must be 'uq_ensemble' or 'optimization_ensemble'")


def _indexed_payloads(payloads: Iterable[tuple[int, Any]]) -> _IndexedPayloads:
    frozen = tuple(payloads)
    return _IndexedPayloads(
        payloads=frozen,
        indices=tuple(index for index, _ in frozen),
        results=tuple(result for _, result in frozen),
    )


def _serial_ensemble_payloads(
    fn: Callable[[Any], Any],
    items: tuple[Any, ...],
) -> _IndexedPayloads:
    return _indexed_payloads((index, fn(item)) for index, item in enumerate(items))


def _parallel_ensemble_payloads(
    fn: Callable[[Any], Any],
    items: tuple[Any, ...],
    worker_metadata: IndependentWorkerMetadata,
) -> _IndexedPayloads:
    return _indexed_payloads(
        independent_map(
            _run_provenance_indexed_task,
            ((fn, index, item) for index, item in enumerate(items)),
            workers=worker_metadata.requested_workers,
            executor=worker_metadata.executor,
        )
    )


def _ensemble_reconstruction(
    parallel: _IndexedPayloads,
    worker_metadata: IndependentWorkerMetadata,
    *,
    workload: str,
) -> _EnsembleReconstruction:
    from gkx.parallel.decomposition import (
        build_independent_portfolio_decomposition,
        reconstruct_serial,
        serial_reconstruction_identity_report,
        shard_sequence,
    )

    contract = build_independent_portfolio_decomposition(
        worker_metadata.problem_size,
        requested_shards=worker_metadata.requested_workers,
        workload=workload,  # type: ignore[arg-type]
    )
    shard_payloads = shard_sequence(parallel.payloads, contract)
    reconstructed_payloads = reconstruct_serial(contract, shard_payloads)
    return _EnsembleReconstruction(
        contract=contract,
        reconstructed_indices=tuple(index for index, _ in reconstructed_payloads),
        report=serial_reconstruction_identity_report(parallel.indices, contract),
    )


def _ensemble_identity_report(
    serial: _IndexedPayloads,
    parallel: _IndexedPayloads,
    worker_metadata: IndependentWorkerMetadata,
    *,
    workload: str,
    atol: float,
    rtol: float,
) -> ParallelIdentityReport:
    return parallel_identity_report(
        list(serial.results),
        list(parallel.results),
        kind="independent_ensemble_serial_identity",
        problem_size=worker_metadata.problem_size,
        requested_workers=worker_metadata.requested_workers,
        actual_workers=worker_metadata.actual_workers,
        backend=f"python:{worker_metadata.executor}",
        atol=atol,
        rtol=rtol,
        metadata={
            "executor": worker_metadata.executor,
            "worker_metadata": worker_metadata.to_dict(),
            "workload": workload,
            "tree": str(jax.tree_util.tree_structure(list(serial.results))),
        },
    )


def _ensemble_gate_checks(
    serial: _IndexedPayloads,
    parallel: _IndexedPayloads,
    reconstruction: _EnsembleReconstruction,
    worker_metadata: IndependentWorkerMetadata,
) -> _EnsembleGateChecks:
    exception_passed, exception_metadata = _probe_exception_metadata(
        requested_workers=worker_metadata.requested_workers,
        executor=worker_metadata.executor,
    )
    return _EnsembleGateChecks(
        ordering_passed=bool(
            serial.indices == parallel.indices == reconstruction.reconstructed_indices
        ),
        worker_clipping_passed=bool(
            worker_metadata.actual_workers
            == min(worker_metadata.requested_workers, worker_metadata.problem_size)
        ),
        reconstruction_passed=bool(
            reconstruction.report.identity_passed
            and reconstruction.reconstructed_indices == parallel.indices
        ),
        exception_passed=exception_passed,
        exception_metadata=exception_metadata,
    )


def _ensemble_report_metadata(
    metadata: dict[str, Any] | None,
    contract: Any,
) -> dict[str, Any]:
    report_metadata = dict(metadata or {})
    report_metadata.update(
        {
            "claim": (
                "independent ensemble batching preserves serial result ordering "
                "and does not change solver layout"
            ),
            "contract": contract.to_dict(),
        }
    )
    return report_metadata


def _pack_ensemble_provenance_report(
    *,
    workload: str,
    worker_metadata: IndependentWorkerMetadata,
    serial: _IndexedPayloads,
    parallel: _IndexedPayloads,
    reconstruction: _EnsembleReconstruction,
    identity: ParallelIdentityReport,
    checks: _EnsembleGateChecks,
    metadata: dict[str, Any],
) -> IndependentEnsembleProvenanceReport:
    passed = bool(
        identity.identity_passed
        and checks.ordering_passed
        and checks.worker_clipping_passed
        and checks.reconstruction_passed
        and checks.exception_passed
    )
    return IndependentEnsembleProvenanceReport(
        kind="independent_ensemble_provenance_gate",
        workload=workload,
        executor=worker_metadata.executor,
        requested_workers=worker_metadata.requested_workers,
        actual_workers=worker_metadata.actual_workers,
        problem_size=worker_metadata.problem_size,
        passed=passed,
        identity_passed=identity.identity_passed,
        ordering_passed=checks.ordering_passed,
        worker_clipping_passed=checks.worker_clipping_passed,
        reconstruction_identity_passed=checks.reconstruction_passed,
        exception_metadata_passed=checks.exception_passed,
        serial_indices=serial.indices,
        parallel_indices=parallel.indices,
        reconstructed_indices=reconstruction.reconstructed_indices,
        identity_report=identity,
        reconstruction_report=reconstruction.report.to_dict(),
        exception_metadata=checks.exception_metadata,
        metadata=metadata,
    )


def _independent_map_tasks(
    fn: Callable[[Any], Any],
    items: tuple[Any, ...],
    worker_metadata: IndependentWorkerMetadata,
) -> tuple[tuple[Callable[[Any], Any], int, Any, str, int], ...]:
    return tuple(
        (fn, index, item, worker_metadata.executor, worker_metadata.actual_workers)
        for index, item in enumerate(items)
    )


def _executor_class(executor: str) -> type[ThreadPoolExecutor] | type[ProcessPoolExecutor]:
    return ThreadPoolExecutor if executor == "thread" else ProcessPoolExecutor


def _run_parallel_indexed_tasks(
    tasks: tuple[tuple[Callable[[Any], Any], int, Any, str, int], ...],
    worker_metadata: IndependentWorkerMetadata,
) -> list[tuple[int, Any]]:
    try:
        with _executor_class(worker_metadata.executor)(
            max_workers=worker_metadata.actual_workers
        ) as pool:
            return list(pool.map(_run_indexed_independent_task, tasks))
    except IndependentMapExecutionError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "independent_map executor "
            f"'{worker_metadata.executor}' failed before completing "
            f"{worker_metadata.problem_size} task(s) with "
            f"actual_workers={worker_metadata.actual_workers}: {exc}"
        ) from exc


def _ordered_results(indexed_results: list[tuple[int, Any]], size: int) -> list[Any]:
    indices = [index for index, _ in indexed_results]
    expected_indices = list(range(size))
    if indices != expected_indices:
        raise RuntimeError(
            "independent_map executor returned results out of serial order: "
            f"{indices} != {expected_indices}"
        )
    return [result for _, result in indexed_results]


def _independent_identity_metadata(
    metadata: dict[str, Any] | None,
    worker_metadata: IndependentWorkerMetadata,
    reference: list[Any],
) -> dict[str, Any]:
    report_metadata = dict(metadata or {})
    report_metadata.update(
        {
            "executor": worker_metadata.executor,
            "parallel_enabled": worker_metadata.parallel_enabled,
            "worker_metadata": worker_metadata.to_dict(),
            "tree": str(jax.tree_util.tree_structure(reference)),
        }
    )
    return report_metadata


def independent_ensemble_provenance_gate(
    fn: Callable[[Any], Any],
    values: Iterable[Any],
    *,
    workers: int = 1,
    executor: str = "thread",
    workload: str = "uq_ensemble",
    atol: float = 1e-12,
    rtol: float = 1e-10,
    metadata: dict[str, Any] | None = None,
) -> IndependentEnsembleProvenanceReport:
    """Verify independent UQ/optimization ensemble batching provenance.

    The gate intentionally runs ``fn`` serially and through ``independent_map``.
    It verifies result identity, serial result ordering, worker clipping,
    deterministic shard reconstruction, and failure metadata for the same
    independent-map executor family.
    """

    _validate_ensemble_workload(workload)

    items = tuple(values)
    worker_metadata = independent_worker_metadata(
        len(items),
        workers=workers,
        executor=executor,
    )
    if worker_metadata.problem_size < 1:
        raise ValueError("values must contain at least one item")

    serial_payloads = _serial_ensemble_payloads(fn, items)
    parallel_payloads = _parallel_ensemble_payloads(fn, items, worker_metadata)
    reconstruction = _ensemble_reconstruction(
        parallel_payloads,
        worker_metadata,
        workload=workload,
    )
    identity = _ensemble_identity_report(
        serial_payloads,
        parallel_payloads,
        worker_metadata,
        workload=workload,
        atol=atol,
        rtol=rtol,
    )
    checks = _ensemble_gate_checks(
        serial_payloads,
        parallel_payloads,
        reconstruction,
        worker_metadata,
    )
    report_metadata = _ensemble_report_metadata(metadata, reconstruction.contract)
    return _pack_ensemble_provenance_report(
        workload=workload,
        worker_metadata=worker_metadata,
        serial=serial_payloads,
        parallel=parallel_payloads,
        reconstruction=reconstruction,
        identity=identity,
        checks=checks,
        metadata=report_metadata,
    )


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

    items = tuple(values)
    worker_metadata = independent_worker_metadata(
        len(items),
        workers=workers,
        executor=executor,
    )
    if not items:
        return []
    if worker_metadata.actual_workers == 1:
        return [fn(item) for item in items]

    indexed_results = _run_parallel_indexed_tasks(
        _independent_map_tasks(fn, items, worker_metadata),
        worker_metadata,
    )
    return _ordered_results(indexed_results, len(items))


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

    items = tuple(values)
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
        metadata=_independent_identity_metadata(metadata, worker_metadata, reference),
    )


__all__ = [
    "IndependentEnsembleProvenanceReport",
    "IndependentMapExecutionError",
    "IndependentWorkerMetadata",
    "independent_ensemble_provenance_gate",
    "independent_map",
    "independent_map_identity_report",
    "independent_worker_metadata",
]
