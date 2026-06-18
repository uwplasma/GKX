"""Matched nonlinear audit redesign gates for stellarator transport claims."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from spectraxgk.validation.stellarator.transport_policies import (
    VMECJAXNonlinearAuditPolicy,
    _finite_float_or_none,
)
from spectraxgk.validation.stellarator.transport_samples import (
    transport_objective_sample_summary,
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
    stats = matched_comparison.get("statistics", {})
    if not isinstance(stats, Mapping):
        stats = {}
    relative_reduction = _finite_float_or_none(stats.get("relative_reduction"))
    z_score = _finite_float_or_none(stats.get("uncertainty_z_score"))
    baseline = matched_comparison.get("baseline", {})
    candidate = matched_comparison.get("candidate", {})
    baseline_passed = bool(baseline.get("passed", False)) if isinstance(baseline, Mapping) else False
    candidate_passed = bool(candidate.get("passed", False)) if isinstance(candidate, Mapping) else False
    comparison_passed = bool(matched_comparison.get("passed", False))
    nonlinear_blockers: list[str] = []
    if not baseline_passed:
        nonlinear_blockers.append("baseline_ensemble_failed")
    if not candidate_passed:
        nonlinear_blockers.append("candidate_ensemble_failed")
    if relative_reduction is None:
        nonlinear_blockers.append("missing_relative_reduction")
    elif relative_reduction < float(policy.minimum_relative_reduction):
        nonlinear_blockers.append("insufficient_matched_reduction")
    if z_score is None:
        nonlinear_blockers.append("missing_uncertainty_z_score")
    elif z_score < float(policy.minimum_uncertainty_z_score):
        nonlinear_blockers.append("insufficient_uncertainty_separation")
    if not comparison_passed:
        nonlinear_blockers.append("matched_comparison_not_passed")

    sample_summary = transport_objective_sample_summary(objective_sample_set, policy=policy)
    all_blockers = nonlinear_blockers + list(sample_summary["blockers"])
    promoted = not all_blockers
    recommended = {
        "surfaces": [float(item) for item in policy.recommended_surfaces],
        "alphas": [float(item) for item in policy.recommended_alphas],
        "ky_values": [float(item) for item in policy.recommended_ky_values],
        "sample_count": (
            len(policy.recommended_surfaces)
            * len(policy.recommended_alphas)
            * len(policy.recommended_ky_values)
        ),
    }
    return {
        "kind": "vmec_jax_nonlinear_transport_audit_redesign_report",
        "policy": policy.to_dict(),
        "matched_comparison_case": matched_comparison.get("case"),
        "matched_comparison_passed": comparison_passed,
        "nonlinear_audit_promoted": promoted,
        "requires_objective_redesign": not promoted,
        "nonlinear_audit_blockers": nonlinear_blockers,
        "objective_sample_summary": sample_summary,
        "blockers": all_blockers,
        "recommended_sample_set": recommended,
        "gates": [
            {
                "metric": "matched_replicated_late_window_reduction",
                "passed": "insufficient_matched_reduction" not in nonlinear_blockers
                and "missing_relative_reduction" not in nonlinear_blockers,
                "value": relative_reduction,
                "threshold": float(policy.minimum_relative_reduction),
            },
            {
                "metric": "uncertainty_separated_reduction",
                "passed": "insufficient_uncertainty_separation" not in nonlinear_blockers
                and "missing_uncertainty_z_score" not in nonlinear_blockers,
                "value": z_score,
                "threshold": float(policy.minimum_uncertainty_z_score),
            },
            {
                "metric": "multi_sample_objective_coverage",
                "passed": bool(sample_summary["passed"]),
                "value": int(sample_summary["sample_count"]),
                "threshold": int(policy.minimum_sample_count),
            },
        ],
        "next_action": (
            "candidate may be used as nonlinear turbulent-flux optimization evidence"
            if promoted
            else (
                "redesign the reduced transport objective with the recommended multi-surface, "
                "multi-field-line, multi-ky sample set; rerun projected admission; then repeat "
                "the matched long-window nonlinear audit"
            )
        ),
    }





__all__ = ["build_nonlinear_audit_redesign_report"]
