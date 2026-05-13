"""Deterministic decomposition contracts for parallel work portfolios.

The helpers in this module describe partitioning and reconstruction contracts.
They do not route solver execution, alter nonlinear state layout, or make
speedup claims.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeVar


IndependentWorkload = Literal["independent_ky_scan", "uq_ensemble"]
DiagnosticWorkload = Literal["diagnostic_nonlinear_domain"]
DecompositionWorkload = IndependentWorkload | DiagnosticWorkload
ClaimLevel = Literal[
    "production_independent_batching",
    "diagnostic_nonlinear_domain_partition",
]

_INDEPENDENT_WORKLOADS: frozenset[str] = frozenset(
    {"independent_ky_scan", "uq_ensemble"}
)

T = TypeVar("T")


@dataclass(frozen=True)
class ShardAssignment:
    """A deterministic contiguous assignment of serial indices to one shard."""

    shard_id: int
    start: int
    stop: int
    indices: tuple[int, ...]
    label: str

    @property
    def size(self) -> int:
        """Number of serial items assigned to this shard."""

        return len(self.indices)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the assignment."""

        return asdict(self)


@dataclass(frozen=True)
class DecompositionContract:
    """Claim-scoped shard assignment contract for a parallelization path."""

    workload: DecompositionWorkload
    claim_level: ClaimLevel
    claim_label: str
    n_items: int
    requested_shards: int
    actual_shards: int
    shards: tuple[ShardAssignment, ...]
    independent_work: bool
    changes_solver_layout: bool
    state_shape: tuple[int, ...] | None = None
    axis: int | None = None

    @property
    def production_independent_batching(self) -> bool:
        """Whether this contract is for production independent-work batching."""

        return self.claim_level == "production_independent_batching"

    @property
    def diagnostic_nonlinear_partition(self) -> bool:
        """Whether this contract is diagnostic nonlinear-domain metadata."""

        return self.claim_level == "diagnostic_nonlinear_domain_partition"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the contract."""

        return asdict(self)


@dataclass(frozen=True)
class ReconstructionIdentityReport:
    """Serial reconstruction identity report for a decomposition contract."""

    workload: DecompositionWorkload
    claim_level: ClaimLevel
    claim_label: str
    n_items: int
    requested_shards: int
    actual_shards: int
    identity_passed: bool
    expected_indices: tuple[int, ...]
    reconstructed_indices: tuple[int, ...]
    missing_indices: tuple[int, ...]
    duplicate_indices: tuple[int, ...]
    out_of_range_indices: tuple[int, ...]
    out_of_order: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the report."""

        return asdict(self)


def _validate_count(name: str, value: int, *, allow_zero: bool) -> int:
    count = int(value)
    minimum = 0 if allow_zero else 1
    if count < minimum:
        qualifier = "non-negative" if allow_zero else ">= 1"
        raise ValueError(f"{name} must be {qualifier}")
    return count


def _balanced_assignments(
    *,
    workload: DecompositionWorkload,
    n_items: int,
    requested_shards: int,
    label_prefix: str,
) -> tuple[ShardAssignment, ...]:
    n = _validate_count("n_items", n_items, allow_zero=True)
    requested = _validate_count("requested_shards", requested_shards, allow_zero=False)
    if n == 0:
        return ()

    actual = min(requested, n)
    base, remainder = divmod(n, actual)
    assignments: list[ShardAssignment] = []
    start = 0
    for shard_id in range(actual):
        size = base + (1 if shard_id < remainder else 0)
        stop = start + size
        indices = tuple(range(start, stop))
        label = f"{label_prefix}:shard_{shard_id:03d}:items_{start:06d}_{stop:06d}"
        assignments.append(
            ShardAssignment(
                shard_id=shard_id,
                start=start,
                stop=stop,
                indices=indices,
                label=label,
            )
        )
        start = stop
    if start != n:  # pragma: no cover - defensive invariant check
        raise AssertionError(f"{workload} assignments did not cover all items")
    return tuple(assignments)


def _independent_claim_label(workload: IndependentWorkload) -> str:
    if workload == "independent_ky_scan":
        portfolio = "independent ky scan"
    elif workload == "uq_ensemble":
        portfolio = "independent UQ ensemble"
    else:  # pragma: no cover - protected by caller validation
        raise ValueError(f"unknown independent workload: {workload}")
    return (
        f"production independent batching contract for {portfolio}; "
        "serial ordering and reconstruction identity only; "
        "not a nonlinear state-domain decomposition speedup claim"
    )


def build_independent_portfolio_decomposition(
    n_items: int,
    *,
    requested_shards: int,
    workload: IndependentWorkload,
) -> DecompositionContract:
    """Build a production independent-work decomposition contract.

    The assignment is deterministic, balanced, contiguous, and contains no
    empty shards. It covers release-ready independent portfolios only:
    ``independent_ky_scan`` and ``uq_ensemble``.
    """

    if workload not in _INDEPENDENT_WORKLOADS:
        raise ValueError("workload must be 'independent_ky_scan' or 'uq_ensemble'")
    n = _validate_count("n_items", n_items, allow_zero=True)
    requested = _validate_count(
        "requested_shards",
        requested_shards,
        allow_zero=False,
    )
    shards = _balanced_assignments(
        workload=workload,
        n_items=n,
        requested_shards=requested,
        label_prefix=workload,
    )
    return DecompositionContract(
        workload=workload,
        claim_level="production_independent_batching",
        claim_label=_independent_claim_label(workload),
        n_items=n,
        requested_shards=requested,
        actual_shards=len(shards),
        shards=shards,
        independent_work=True,
        changes_solver_layout=False,
    )


def _diagnostic_claim_label() -> str:
    return (
        "diagnostic nonlinear state-domain partition contract; "
        "serial split/reassemble identity only; "
        "no production routing or speedup claim"
    )


def build_diagnostic_nonlinear_domain_decomposition(
    state_shape: Iterable[int],
    *,
    axis: int,
    requested_shards: int,
) -> DecompositionContract:
    """Build a diagnostic nonlinear-domain partition contract.

    This metadata describes split/reassemble coverage along one state axis.
    It is intentionally not a production nonlinear route and does not claim
    nonlinear speedup.
    """

    shape = tuple(int(size) for size in state_shape)
    if not shape:
        raise ValueError("state_shape must contain at least one axis")
    if any(size <= 0 for size in shape):
        raise ValueError("state_shape entries must be positive")
    canonical_axis = int(axis) % len(shape)
    requested = _validate_count(
        "requested_shards",
        requested_shards,
        allow_zero=False,
    )
    domain_size = shape[canonical_axis]
    shards = _balanced_assignments(
        workload="diagnostic_nonlinear_domain",
        n_items=domain_size,
        requested_shards=requested,
        label_prefix=f"diagnostic_nonlinear_domain:axis_{canonical_axis}",
    )
    return DecompositionContract(
        workload="diagnostic_nonlinear_domain",
        claim_level="diagnostic_nonlinear_domain_partition",
        claim_label=_diagnostic_claim_label(),
        n_items=domain_size,
        requested_shards=requested,
        actual_shards=len(shards),
        shards=shards,
        independent_work=False,
        changes_solver_layout=True,
        state_shape=shape,
        axis=canonical_axis,
    )


def shard_sequence(
    values: Sequence[T],
    contract: DecompositionContract,
) -> tuple[tuple[T, ...], ...]:
    """Return values grouped according to a decomposition contract."""

    items = tuple(values)
    if len(items) != contract.n_items:
        raise ValueError("values length must match contract.n_items")
    return tuple(tuple(items[index] for index in shard.indices) for shard in contract.shards)


def reconstruct_serial(
    contract: DecompositionContract,
    shard_values: Sequence[Sequence[T]],
) -> tuple[T, ...]:
    """Reassemble shard values into serial index order."""

    if len(shard_values) != contract.actual_shards:
        raise ValueError("shard_values length must match contract.actual_shards")

    reconstructed: list[Any] = [None] * contract.n_items
    filled = [False] * contract.n_items
    for shard, values in zip(contract.shards, shard_values, strict=True):
        shard_tuple = tuple(values)
        if len(shard_tuple) != shard.size:
            raise ValueError("each shard value group must match its assignment size")
        for index, value in zip(shard.indices, shard_tuple, strict=True):
            if index < 0 or index >= contract.n_items:
                raise ValueError("shard assignment index out of range")
            if filled[index]:
                raise ValueError("shard assignments contain duplicate indices")
            reconstructed[index] = value
            filled[index] = True

    if not all(filled):
        raise ValueError("shard assignments do not cover all serial indices")
    return tuple(reconstructed)


def _coverage(contract: DecompositionContract) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    bool,
]:
    expected = tuple(range(contract.n_items))
    reconstructed = tuple(index for shard in contract.shards for index in shard.indices)
    counts = Counter(reconstructed)
    missing = tuple(index for index in expected if counts[index] == 0)
    duplicates = tuple(index for index, count in sorted(counts.items()) if count > 1)
    out_of_range = tuple(
        index for index in reconstructed if index < 0 or index >= contract.n_items
    )
    out_of_order = reconstructed != expected
    return expected, reconstructed, missing, duplicates, out_of_range, out_of_order


def _default_equal(left: T, right: T) -> bool:
    if left is right:
        return True
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def serial_reconstruction_identity_report(
    values: Sequence[T],
    contract: DecompositionContract,
    *,
    equal: Callable[[T, T], bool] | None = None,
) -> ReconstructionIdentityReport:
    """Check that contract sharding reassembles exactly to serial order."""

    items = tuple(values)
    shards = shard_sequence(items, contract)
    reconstructed_values = reconstruct_serial(contract, shards)
    expected_indices, reconstructed_indices, missing, duplicates, out_of_range, out_of_order = _coverage(
        contract
    )
    comparator = equal or _default_equal
    values_match = len(reconstructed_values) == len(items) and all(
        comparator(left, right)
        for left, right in zip(reconstructed_values, items, strict=True)
    )
    identity_passed = bool(
        values_match
        and not missing
        and not duplicates
        and not out_of_range
        and not out_of_order
    )
    return ReconstructionIdentityReport(
        workload=contract.workload,
        claim_level=contract.claim_level,
        claim_label=contract.claim_label,
        n_items=contract.n_items,
        requested_shards=contract.requested_shards,
        actual_shards=contract.actual_shards,
        identity_passed=identity_passed,
        expected_indices=expected_indices,
        reconstructed_indices=reconstructed_indices,
        missing_indices=missing,
        duplicate_indices=duplicates,
        out_of_range_indices=out_of_range,
        out_of_order=out_of_order,
    )


__all__ = [
    "ClaimLevel",
    "DecompositionContract",
    "DecompositionWorkload",
    "DiagnosticWorkload",
    "IndependentWorkload",
    "ReconstructionIdentityReport",
    "ShardAssignment",
    "build_diagnostic_nonlinear_domain_decomposition",
    "build_independent_portfolio_decomposition",
    "reconstruct_serial",
    "serial_reconstruction_identity_report",
    "shard_sequence",
]
