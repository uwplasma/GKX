"""Reduced-objective prelaunch gates for nonlinear stellarator audits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from spectraxgk.validation.stellarator.transport_policies import (
    VMECJAXNonlinearAuditPolicy,
    VMECJAXReducedPrelaunchPolicy,
    _finite_float_or_none,
)
from spectraxgk.validation.stellarator.transport_samples import (
    transport_objective_sample_summary,
)


@dataclass(frozen=True)
class _PrelaunchMetrics:
    baseline: float | None
    candidate: float | None
    failed_reference: float | None
    relative_reduction: float | None
    threshold: float
    threshold_sources: dict[str, float | None]


@dataclass(frozen=True)
class _CrossSampleStatus:
    available: bool
    passed: bool | None
    rows: list[dict[str, Any]]


def _sample_statistics_summary(
    sample_statistics: Mapping[str, Any] | None,
    *,
    policy: VMECJAXReducedPrelaunchPolicy,
    role: str,
) -> dict[str, Any]:
    """Return a gate row for deterministic reduced-metric sample dispersion."""

    if not isinstance(sample_statistics, Mapping):
        return {
            "role": role,
            "available": False,
            "weighted_mean": None,
            "weighted_standard_error": None,
            "cross_sample_sem_rel": None,
            "passed": None,
            "blockers": [],
        }
    weighted_mean = _finite_float_or_none(sample_statistics.get("weighted_mean"))
    weighted_sem = _finite_float_or_none(
        sample_statistics.get("weighted_standard_error")
    )
    blockers: list[str] = []
    sem_rel = None
    if weighted_mean is None:
        blockers.append(f"{role}_missing_cross_sample_weighted_mean")
    if weighted_sem is None:
        blockers.append(f"{role}_missing_cross_sample_weighted_sem")
    if weighted_mean is not None and weighted_sem is not None:
        sem_rel = float(abs(weighted_sem) / max(abs(weighted_mean), 1.0e-300))
        if sem_rel > float(policy.maximum_cross_sample_sem_rel):
            blockers.append(f"{role}_cross_sample_sem_rel_too_large")
    return {
        "role": role,
        "available": True,
        "weighted_mean": weighted_mean,
        "weighted_standard_error": weighted_sem,
        "cross_sample_sem_rel": sem_rel,
        "passed": not blockers,
        "blockers": blockers,
    }


def _prelaunch_policies(
    *,
    policy: VMECJAXReducedPrelaunchPolicy | None,
    nonlinear_policy: VMECJAXNonlinearAuditPolicy | None,
) -> tuple[VMECJAXReducedPrelaunchPolicy, VMECJAXNonlinearAuditPolicy]:
    return (
        policy or VMECJAXReducedPrelaunchPolicy(),
        nonlinear_policy or VMECJAXNonlinearAuditPolicy(),
    )


def _relative_reduction(
    baseline: float | None,
    candidate: float | None,
) -> float | None:
    if baseline is None or candidate is None:
        return None
    return float((baseline - candidate) / max(abs(baseline), 1.0e-300))


def _threshold_sources(
    *,
    failed_reference: float | None,
    policy: VMECJAXReducedPrelaunchPolicy,
) -> tuple[float, dict[str, float | None]]:
    threshold = float(policy.minimum_relative_reduction)
    sources: dict[str, float | None] = {
        "policy_minimum_relative_reduction": float(policy.minimum_relative_reduction),
        "failed_reference_relative_reduction": failed_reference,
        "failed_reference_safety_factor": float(policy.failed_reference_safety_factor),
        "failed_reference_threshold": None,
    }
    if failed_reference is not None:
        failed_threshold = float(policy.failed_reference_safety_factor) * failed_reference
        sources["failed_reference_threshold"] = failed_threshold
        threshold = max(threshold, failed_threshold)
    return threshold, sources


def _prelaunch_metrics(
    *,
    baseline_metric: float,
    candidate_metric: float,
    failed_reference_relative_reduction: float | None,
    policy: VMECJAXReducedPrelaunchPolicy,
) -> _PrelaunchMetrics:
    baseline = _finite_float_or_none(baseline_metric)
    candidate = _finite_float_or_none(candidate_metric)
    failed_reference = _finite_float_or_none(failed_reference_relative_reduction)
    relative_reduction = _relative_reduction(baseline, candidate)
    threshold, threshold_sources = _threshold_sources(
        failed_reference=failed_reference,
        policy=policy,
    )
    return _PrelaunchMetrics(
        baseline=baseline,
        candidate=candidate,
        failed_reference=failed_reference,
        relative_reduction=relative_reduction,
        threshold=threshold,
        threshold_sources=threshold_sources,
    )


def _metric_blockers(metrics: _PrelaunchMetrics) -> list[str]:
    blockers: list[str] = []
    if metrics.baseline is None:
        blockers.append("missing_baseline_reduced_metric")
    if metrics.candidate is None:
        blockers.append("missing_candidate_reduced_metric")
    if metrics.relative_reduction is None:
        blockers.append("missing_relative_reduced_reduction")
    elif metrics.relative_reduction < metrics.threshold:
        blockers.append("insufficient_reduced_margin_for_nonlinear_audit")
    return blockers


def _cross_sample_status(
    *,
    baseline_sample_statistics: Mapping[str, Any] | None,
    candidate_sample_statistics: Mapping[str, Any] | None,
    policy: VMECJAXReducedPrelaunchPolicy,
) -> _CrossSampleStatus:
    rows = [
        _sample_statistics_summary(
            baseline_sample_statistics,
            policy=policy,
            role="baseline",
        ),
        _sample_statistics_summary(
            candidate_sample_statistics,
            policy=policy,
            role="candidate",
        ),
    ]
    available = all(bool(row["available"]) for row in rows)
    passed = None if not available else all(bool(row["passed"]) for row in rows)
    return _CrossSampleStatus(available=available, passed=passed, rows=rows)


def _cross_sample_blockers(status: _CrossSampleStatus) -> list[str]:
    if not status.available or status.passed is not False:
        return []
    return [str(item) for row in status.rows for item in row["blockers"]]


def _sample_coverage_blockers(
    sample_summary: Mapping[str, Any],
    *,
    policy: VMECJAXReducedPrelaunchPolicy,
) -> list[str]:
    if bool(policy.require_sample_coverage) and not bool(sample_summary["passed"]):
        return [str(item) for item in sample_summary["blockers"]]
    return []


def _prelaunch_gates(
    *,
    metrics: _PrelaunchMetrics,
    sample_summary: Mapping[str, Any],
    cross_sample: _CrossSampleStatus,
    nonlinear_policy: VMECJAXNonlinearAuditPolicy,
    policy: VMECJAXReducedPrelaunchPolicy,
    blockers: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "metric": "reduced_margin_for_nonlinear_audit",
            "passed": "insufficient_reduced_margin_for_nonlinear_audit"
            not in blockers
            and "missing_relative_reduced_reduction" not in blockers,
            "value": metrics.relative_reduction,
            "threshold": metrics.threshold,
        },
        {
            "metric": "multi_sample_objective_coverage",
            "passed": bool(sample_summary["passed"]),
            "value": int(sample_summary["sample_count"]),
            "threshold": int(nonlinear_policy.minimum_sample_count),
        },
        {
            "metric": "reduced_cross_sample_dispersion",
            "passed": cross_sample.passed,
            "value": [
                row["cross_sample_sem_rel"]
                for row in cross_sample.rows
                if row["cross_sample_sem_rel"] is not None
            ],
            "threshold": float(policy.maximum_cross_sample_sem_rel),
        },
    ]


def _cross_sample_payload(status: _CrossSampleStatus) -> dict[str, Any]:
    return {
        "available": status.available,
        "passed": status.passed,
        "rows": status.rows,
        "claim_scope": (
            "deterministic spread over the reduced surface/field-line/ky "
            "objective grid; not stochastic nonlinear heat-flux uncertainty"
        ),
    }


def _next_action(blockers: list[str]) -> str:
    if not blockers:
        return "launch replicated long-window nonlinear audit only with baseline/candidate ensembles"
    return (
        "do not launch an expensive nonlinear audit; increase reduced-objective "
        "margin or broaden the objective before spending GPU time"
    )


def _prelaunch_payload(
    *,
    policy: VMECJAXReducedPrelaunchPolicy,
    nonlinear_policy: VMECJAXNonlinearAuditPolicy,
    metrics: _PrelaunchMetrics,
    sample_summary: Mapping[str, Any],
    cross_sample: _CrossSampleStatus,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "kind": "vmec_jax_reduced_nonlinear_audit_prelaunch_report",
        "claim_scope": (
            "reduced-objective prelaunch guard only; passing this gate permits a "
            "replicated nonlinear audit but does not promote a turbulence claim"
        ),
        "policy": policy.to_dict(),
        "nonlinear_policy": nonlinear_policy.to_dict(),
        "metric_key": str(policy.metric_key),
        "baseline_metric": metrics.baseline,
        "candidate_metric": metrics.candidate,
        "relative_reduced_reduction": metrics.relative_reduction,
        "required_relative_reduced_reduction": metrics.threshold,
        "threshold_sources": metrics.threshold_sources,
        "objective_sample_summary": sample_summary,
        "reduced_cross_sample_statistics": _cross_sample_payload(cross_sample),
        "passed": not blockers,
        "blockers": blockers,
        "gates": _prelaunch_gates(
            metrics=metrics,
            sample_summary=sample_summary,
            cross_sample=cross_sample,
            nonlinear_policy=nonlinear_policy,
            policy=policy,
            blockers=blockers,
        ),
        "next_action": _next_action(blockers),
    }


def build_reduced_nonlinear_audit_prelaunch_report(
    *,
    baseline_metric: float,
    candidate_metric: float,
    objective_sample_set: Any = None,
    baseline_sample_statistics: Mapping[str, Any] | None = None,
    candidate_sample_statistics: Mapping[str, Any] | None = None,
    failed_reference_relative_reduction: float | None = None,
    policy: VMECJAXReducedPrelaunchPolicy | None = None,
    nonlinear_policy: VMECJAXNonlinearAuditPolicy | None = None,
) -> dict[str, Any]:
    """Gate reduced transport candidates before launching nonlinear audits.

    This is intentionally conservative.  A reduced nonlinear-window improvement
    should exceed both an absolute release threshold and, when available, a
    safety factor above a known failed-transfer reference before spending
    another long GPU campaign.
    """

    policy, nonlinear_policy = _prelaunch_policies(
        policy=policy,
        nonlinear_policy=nonlinear_policy,
    )
    metrics = _prelaunch_metrics(
        baseline_metric=baseline_metric,
        candidate_metric=candidate_metric,
        failed_reference_relative_reduction=failed_reference_relative_reduction,
        policy=policy,
    )
    sample_summary = transport_objective_sample_summary(
        objective_sample_set,
        policy=nonlinear_policy,
    )
    cross_sample = _cross_sample_status(
        baseline_sample_statistics=baseline_sample_statistics,
        candidate_sample_statistics=candidate_sample_statistics,
        policy=policy,
    )
    blockers = [
        *_metric_blockers(metrics),
        *_sample_coverage_blockers(sample_summary, policy=policy),
        *_cross_sample_blockers(cross_sample),
    ]
    return _prelaunch_payload(
        policy=policy,
        nonlinear_policy=nonlinear_policy,
        metrics=metrics,
        sample_summary=sample_summary,
        cross_sample=cross_sample,
        blockers=blockers,
    )


__all__ = [
    "_sample_statistics_summary",
    "build_reduced_nonlinear_audit_prelaunch_report",
]
