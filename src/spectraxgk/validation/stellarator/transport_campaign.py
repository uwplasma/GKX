"""Campaign admission gates for nonlinear stellarator transport optimization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from spectraxgk.validation.stellarator.transport_policies import (
    VMECJAXNonlinearCampaignPolicy,
    _finite_float_or_none,
)

def build_nonlinear_campaign_admission_report(
    *,
    reduced_prelaunch_report: Mapping[str, Any],
    landscape_admission_report: Mapping[str, Any],
    policy: VMECJAXNonlinearCampaignPolicy | None = None,
) -> dict[str, Any]:
    """Gate the next nonlinear optimizer campaign from existing evidence.

    This report intentionally promotes only a *campaign launch*.  It requires a
    reduced prelaunch pass and an uncertainty-separated replicated nonlinear
    landscape point.  It does not convert that point into a general
    multi-coefficient turbulent-flux optimization result.
    """

    policy = policy or VMECJAXNonlinearCampaignPolicy()
    blockers: list[str] = []
    gates: list[dict[str, Any]] = []

    prelaunch_passed = bool(reduced_prelaunch_report.get("passed", False))
    gates.append(
        {
            "metric": "reduced_prelaunch_gate",
            "passed": prelaunch_passed,
            "detail": reduced_prelaunch_report.get("blockers", []),
        }
    )
    if bool(policy.require_reduced_prelaunch_passed) and not prelaunch_passed:
        blockers.append("reduced_prelaunch_gate_failed")

    sample_summary = reduced_prelaunch_report.get("objective_sample_summary")
    sample_passed = (
        bool(sample_summary.get("passed", False))
        if isinstance(sample_summary, Mapping)
        else False
    )
    gates.append(
        {
            "metric": "reduced_objective_sample_coverage",
            "passed": sample_passed,
            "value": (
                int(sample_summary["sample_count"])
                if isinstance(sample_summary, Mapping)
                and sample_summary.get("sample_count") is not None
                else None
            ),
            "detail": (
                sample_summary.get("blockers", [])
                if isinstance(sample_summary, Mapping)
                else "missing objective_sample_summary"
            ),
        }
    )
    if not sample_passed:
        blockers.append("reduced_objective_sample_coverage_failed")

    cross_sample = reduced_prelaunch_report.get("reduced_cross_sample_statistics")
    cross_sample_available = (
        bool(cross_sample.get("available", False))
        if isinstance(cross_sample, Mapping)
        else False
    )
    cross_sample_passed = (
        bool(cross_sample.get("passed", False))
        if isinstance(cross_sample, Mapping)
        and cross_sample.get("passed") is not None
        else None
    )
    gates.append(
        {
            "metric": "reduced_cross_sample_dispersion",
            "passed": cross_sample_passed,
            "detail": (
                cross_sample.get("rows", [])
                if isinstance(cross_sample, Mapping)
                else "missing reduced_cross_sample_statistics"
            ),
        }
    )
    if bool(policy.require_reduced_cross_sample_gate):
        if not cross_sample_available:
            blockers.append("reduced_cross_sample_statistics_missing")
        elif cross_sample_passed is not True:
            blockers.append("reduced_cross_sample_dispersion_failed")

    landscape_passed = bool(landscape_admission_report.get("passed", False))
    gates.append(
        {
            "metric": "replicated_landscape_admission",
            "passed": landscape_passed,
            "detail": landscape_admission_report.get("next_action"),
        }
    )
    if bool(policy.require_landscape_admission_passed) and not landscape_passed:
        blockers.append("replicated_landscape_admission_failed")

    selected = landscape_admission_report.get("selected_candidate")
    selected_map: Mapping[str, Any] = selected if isinstance(selected, Mapping) else {}
    if not selected_map:
        blockers.append("missing_selected_landscape_candidate")

    relative_reduction = _finite_float_or_none(selected_map.get("relative_reduction"))
    z_score = _finite_float_or_none(selected_map.get("uncertainty_z_score"))
    sem_rel = _finite_float_or_none(selected_map.get("combined_sem_rel"))
    n_reports = int(selected_map.get("n_reports", 0) or 0)
    candidate_gates = [
        (
            "landscape_relative_reduction",
            relative_reduction is not None
            and relative_reduction
            >= float(policy.minimum_landscape_relative_reduction),
            relative_reduction,
            float(policy.minimum_landscape_relative_reduction),
            "selected_landscape_reduction_too_small",
        ),
        (
            "landscape_uncertainty_separation",
            z_score is not None
            and z_score >= float(policy.minimum_landscape_uncertainty_z_score),
            z_score,
            float(policy.minimum_landscape_uncertainty_z_score),
            "selected_landscape_uncertainty_separation_too_small",
        ),
        (
            "landscape_candidate_sem_rel",
            sem_rel is not None and sem_rel <= float(policy.maximum_landscape_sem_rel),
            sem_rel,
            float(policy.maximum_landscape_sem_rel),
            "selected_landscape_sem_rel_too_large",
        ),
        (
            "landscape_candidate_replicates",
            n_reports >= int(policy.minimum_landscape_replicate_count),
            n_reports,
            int(policy.minimum_landscape_replicate_count),
            "selected_landscape_insufficient_replicates",
        ),
    ]
    for metric, passed, value, threshold, blocker in candidate_gates:
        gates.append(
            {
                "metric": metric,
                "passed": bool(passed),
                "value": value,
                "threshold": threshold,
            }
        )
        if not passed:
            blockers.append(blocker)

    admitted = not blockers
    return {
        "kind": "vmec_jax_nonlinear_campaign_admission_report",
        "claim_scope": (
            "next nonlinear optimizer-campaign admission only; not a production "
            "multi-coefficient turbulent-flux optimization claim"
        ),
        "policy": policy.to_dict(),
        "passed": admitted,
        "campaign_admitted": admitted,
        "blockers": blockers,
        "gates": gates,
        "selected_landscape_candidate": dict(selected_map) if selected_map else None,
        "next_action": (
            "launch a bounded multi-control optimizer campaign from the admitted landscape direction, "
            "with matched baseline/candidate t=[350,700] replicated nonlinear audits before promotion"
            if admitted
            else (
                "do not launch a broader nonlinear optimizer campaign; fix the reduced gate, "
                "cross-sample dispersion, or replicated landscape uncertainty first"
            )
        ),
    }





__all__ = ["build_nonlinear_campaign_admission_report"]
