"""Numerical-identity reports for parallel execution paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import jax
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


__all__ = ["ParallelIdentityReport", "parallel_identity_report"]
