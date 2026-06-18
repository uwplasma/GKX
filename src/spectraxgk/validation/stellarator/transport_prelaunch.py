"""Reduced-objective prelaunch gates for nonlinear stellarator audits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from spectraxgk.validation.stellarator.transport_policies import (
    VMECJAXNonlinearAuditPolicy,
    VMECJAXReducedPrelaunchPolicy,
    _finite_float_or_none,
)
from spectraxgk.validation.stellarator.transport_samples import (
    transport_objective_sample_summary,
)

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

    policy = policy or VMECJAXReducedPrelaunchPolicy()
    nonlinear_policy = nonlinear_policy or VMECJAXNonlinearAuditPolicy()
    baseline = _finite_float_or_none(baseline_metric)
    candidate = _finite_float_or_none(candidate_metric)
    failed_reference = _finite_float_or_none(failed_reference_relative_reduction)
    blockers: list[str] = []
    relative_reduction = None
    if baseline is None:
        blockers.append("missing_baseline_reduced_metric")
    if candidate is None:
        blockers.append("missing_candidate_reduced_metric")
    if baseline is not None and candidate is not None:
        relative_reduction = float((baseline - candidate) / max(abs(baseline), 1.0e-300))

    threshold = float(policy.minimum_relative_reduction)
    threshold_sources = {
        "policy_minimum_relative_reduction": float(policy.minimum_relative_reduction),
        "failed_reference_relative_reduction": failed_reference,
        "failed_reference_safety_factor": float(policy.failed_reference_safety_factor),
        "failed_reference_threshold": None,
    }
    if failed_reference is not None:
        failed_threshold = float(policy.failed_reference_safety_factor) * failed_reference
        threshold_sources["failed_reference_threshold"] = failed_threshold
        threshold = max(threshold, failed_threshold)

    if relative_reduction is None:
        blockers.append("missing_relative_reduced_reduction")
    elif relative_reduction < threshold:
        blockers.append("insufficient_reduced_margin_for_nonlinear_audit")

    sample_summary = transport_objective_sample_summary(
        objective_sample_set,
        policy=nonlinear_policy,
    )
    if bool(policy.require_sample_coverage) and not bool(sample_summary["passed"]):
        blockers.extend(str(item) for item in sample_summary["blockers"])
    cross_sample_rows = [
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
    cross_sample_available = all(bool(row["available"]) for row in cross_sample_rows)
    cross_sample_passed = (
        None
        if not cross_sample_available
        else all(bool(row["passed"]) for row in cross_sample_rows)
    )
    if cross_sample_available and cross_sample_passed is False:
        for row in cross_sample_rows:
            blockers.extend(str(item) for item in row["blockers"])

    return {
        "kind": "vmec_jax_reduced_nonlinear_audit_prelaunch_report",
        "claim_scope": (
            "reduced-objective prelaunch guard only; passing this gate permits a "
            "replicated nonlinear audit but does not promote a turbulence claim"
        ),
        "policy": policy.to_dict(),
        "nonlinear_policy": nonlinear_policy.to_dict(),
        "metric_key": str(policy.metric_key),
        "baseline_metric": baseline,
        "candidate_metric": candidate,
        "relative_reduced_reduction": relative_reduction,
        "required_relative_reduced_reduction": threshold,
        "threshold_sources": threshold_sources,
        "objective_sample_summary": sample_summary,
        "reduced_cross_sample_statistics": {
            "available": cross_sample_available,
            "passed": cross_sample_passed,
            "rows": cross_sample_rows,
            "claim_scope": (
                "deterministic spread over the reduced surface/field-line/ky "
                "objective grid; not stochastic nonlinear heat-flux uncertainty"
            ),
        },
        "passed": not blockers,
        "blockers": blockers,
        "gates": [
            {
                "metric": "reduced_margin_for_nonlinear_audit",
                "passed": "insufficient_reduced_margin_for_nonlinear_audit" not in blockers
                and "missing_relative_reduced_reduction" not in blockers,
                "value": relative_reduction,
                "threshold": threshold,
            },
            {
                "metric": "multi_sample_objective_coverage",
                "passed": bool(sample_summary["passed"]),
                "value": int(sample_summary["sample_count"]),
                "threshold": int(nonlinear_policy.minimum_sample_count),
            },
            {
                "metric": "reduced_cross_sample_dispersion",
                "passed": cross_sample_passed,
                "value": [
                    row["cross_sample_sem_rel"]
                    for row in cross_sample_rows
                    if row["cross_sample_sem_rel"] is not None
                ],
                "threshold": float(policy.maximum_cross_sample_sem_rel),
            },
        ],
        "next_action": (
            "launch replicated long-window nonlinear audit only with baseline/candidate ensembles"
            if not blockers
            else (
                "do not launch an expensive nonlinear audit; increase reduced-objective "
                "margin or broaden the objective before spending GPU time"
            )
        ),
    }




__all__ = [
    "_sample_statistics_summary",
    "build_reduced_nonlinear_audit_prelaunch_report",
]
