"""Matched nonlinear audit redesign gates for stellarator transport claims."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from spectraxgk.validation.stellarator.transport_policies import (
    VMECJAXNonlinearAuditPolicy,
    _finite_float_or_none,
)
from spectraxgk.validation.stellarator.transport_samples import (
    transport_objective_sample_summary,
)


@dataclass(frozen=True)
class _MatchedNonlinearAuditMetrics:
    relative_reduction: float | None
    z_score: float | None
    baseline_passed: bool
    candidate_passed: bool
    comparison_passed: bool


def _matched_nonlinear_audit_metrics(
    matched_comparison: Mapping[str, Any],
) -> _MatchedNonlinearAuditMetrics:
    stats = matched_comparison.get("statistics", {})
    if not isinstance(stats, Mapping):
        stats = {}
    baseline = matched_comparison.get("baseline", {})
    candidate = matched_comparison.get("candidate", {})
    return _MatchedNonlinearAuditMetrics(
        relative_reduction=_finite_float_or_none(stats.get("relative_reduction")),
        z_score=_finite_float_or_none(stats.get("uncertainty_z_score")),
        baseline_passed=bool(baseline.get("passed", False))
        if isinstance(baseline, Mapping)
        else False,
        candidate_passed=bool(candidate.get("passed", False))
        if isinstance(candidate, Mapping)
        else False,
        comparison_passed=bool(matched_comparison.get("passed", False)),
    )


def _nonlinear_audit_blockers(
    metrics: _MatchedNonlinearAuditMetrics,
    policy: VMECJAXNonlinearAuditPolicy,
) -> list[str]:
    blockers: list[str] = []
    if not metrics.baseline_passed:
        blockers.append("baseline_ensemble_failed")
    if not metrics.candidate_passed:
        blockers.append("candidate_ensemble_failed")
    if metrics.relative_reduction is None:
        blockers.append("missing_relative_reduction")
    elif metrics.relative_reduction < float(policy.minimum_relative_reduction):
        blockers.append("insufficient_matched_reduction")
    if metrics.z_score is None:
        blockers.append("missing_uncertainty_z_score")
    elif metrics.z_score < float(policy.minimum_uncertainty_z_score):
        blockers.append("insufficient_uncertainty_separation")
    if not metrics.comparison_passed:
        blockers.append("matched_comparison_not_passed")
    return blockers


def _recommended_transport_sample_set(
    policy: VMECJAXNonlinearAuditPolicy,
) -> dict[str, Any]:
    return {
        "surfaces": [float(item) for item in policy.recommended_surfaces],
        "alphas": [float(item) for item in policy.recommended_alphas],
        "ky_values": [float(item) for item in policy.recommended_ky_values],
        "sample_count": (
            len(policy.recommended_surfaces)
            * len(policy.recommended_alphas)
            * len(policy.recommended_ky_values)
        ),
    }


def _nonlinear_audit_gate_rows(
    *,
    metrics: _MatchedNonlinearAuditMetrics,
    nonlinear_blockers: list[str],
    sample_summary: Mapping[str, Any],
    policy: VMECJAXNonlinearAuditPolicy,
) -> list[dict[str, Any]]:
    return [
        {
            "metric": "matched_replicated_late_window_reduction",
            "passed": "insufficient_matched_reduction" not in nonlinear_blockers
            and "missing_relative_reduction" not in nonlinear_blockers,
            "value": metrics.relative_reduction,
            "threshold": float(policy.minimum_relative_reduction),
        },
        {
            "metric": "uncertainty_separated_reduction",
            "passed": "insufficient_uncertainty_separation" not in nonlinear_blockers
            and "missing_uncertainty_z_score" not in nonlinear_blockers,
            "value": metrics.z_score,
            "threshold": float(policy.minimum_uncertainty_z_score),
        },
        {
            "metric": "multi_sample_objective_coverage",
            "passed": bool(sample_summary["passed"]),
            "value": int(sample_summary["sample_count"]),
            "threshold": int(policy.minimum_sample_count),
        },
    ]


def _nonlinear_audit_next_action(promoted: bool) -> str:
    if promoted:
        return "candidate may be used as nonlinear turbulent-flux optimization evidence"
    return (
        "redesign the reduced transport objective with the recommended multi-surface, "
        "multi-field-line, multi-ky sample set; rerun projected admission; then repeat "
        "the matched long-window nonlinear audit"
    )


def build_nonlinear_audit_redesign_report(
    matched_comparison: Mapping[str, Any],
    *,
    objective_sample_set: Any = None,
    policy: VMECJAXNonlinearAuditPolicy | None = None,
) -> dict[str, Any]:
    """Decide whether a matched nonlinear audit promotes or redesigns a candidate.

    This is the fail-closed bridge between reduced VMEC-JAX transport admission
    and expensive long-window nonlinear evidence.  A candidate is promoted only
    if the matched replicated nonlinear comparison passes, has a positive
    uncertainty-separated reduction, and the reduced objective used enough
    surface/field-line/``k_y`` samples to avoid a single-point overfit.
    """

    policy = policy or VMECJAXNonlinearAuditPolicy()
    metrics = _matched_nonlinear_audit_metrics(matched_comparison)
    nonlinear_blockers = _nonlinear_audit_blockers(metrics, policy)
    sample_summary = transport_objective_sample_summary(objective_sample_set, policy=policy)
    all_blockers = nonlinear_blockers + list(sample_summary["blockers"])
    promoted = not all_blockers
    return {
        "kind": "vmec_jax_nonlinear_transport_audit_redesign_report",
        "policy": policy.to_dict(),
        "matched_comparison_case": matched_comparison.get("case"),
        "matched_comparison_passed": metrics.comparison_passed,
        "nonlinear_audit_promoted": promoted,
        "requires_objective_redesign": not promoted,
        "nonlinear_audit_blockers": nonlinear_blockers,
        "objective_sample_summary": sample_summary,
        "blockers": all_blockers,
        "recommended_sample_set": _recommended_transport_sample_set(policy),
        "gates": _nonlinear_audit_gate_rows(
            metrics=metrics,
            nonlinear_blockers=nonlinear_blockers,
            sample_summary=sample_summary,
            policy=policy,
        ),
        "next_action": _nonlinear_audit_next_action(promoted),
    }





__all__ = ["build_nonlinear_audit_redesign_report"]
