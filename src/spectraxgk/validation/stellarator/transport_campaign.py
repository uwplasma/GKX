"""Campaign admission gates for nonlinear stellarator transport optimization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from spectraxgk.validation.stellarator.transport_policies import (
    VMECJAXNonlinearCampaignPolicy,
    _finite_float_or_none,
)

def _reduced_prelaunch_gate(
    reduced_prelaunch_report: Mapping[str, Any],
    policy: VMECJAXNonlinearCampaignPolicy,
) -> tuple[dict[str, Any], list[str]]:
    prelaunch_passed = bool(reduced_prelaunch_report.get("passed", False))
    blockers: list[str] = []
    if bool(policy.require_reduced_prelaunch_passed) and not prelaunch_passed:
        blockers.append("reduced_prelaunch_gate_failed")
    return (
        {
            "metric": "reduced_prelaunch_gate",
            "passed": prelaunch_passed,
            "detail": reduced_prelaunch_report.get("blockers", []),
        },
        blockers,
    )


def _reduced_objective_sample_gate(
    reduced_prelaunch_report: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    sample_summary = reduced_prelaunch_report.get("objective_sample_summary")
    sample_map = sample_summary if isinstance(sample_summary, Mapping) else None
    sample_passed = bool(sample_map.get("passed", False)) if sample_map else False
    blockers = [] if sample_passed else ["reduced_objective_sample_coverage_failed"]
    return (
        {
            "metric": "reduced_objective_sample_coverage",
            "passed": sample_passed,
            "value": (
                int(sample_map["sample_count"])
                if sample_map is not None and sample_map.get("sample_count") is not None
                else None
            ),
            "detail": (
                sample_map.get("blockers", [])
                if sample_map is not None
                else "missing objective_sample_summary"
            ),
        },
        blockers,
    )


def _reduced_cross_sample_gate(
    reduced_prelaunch_report: Mapping[str, Any],
    policy: VMECJAXNonlinearCampaignPolicy,
) -> tuple[dict[str, Any], list[str]]:
    cross_sample = reduced_prelaunch_report.get("reduced_cross_sample_statistics")
    cross_map = cross_sample if isinstance(cross_sample, Mapping) else None
    cross_sample_available = bool(cross_map.get("available", False)) if cross_map else False
    cross_sample_passed = (
        bool(cross_map.get("passed", False))
        if cross_map is not None and cross_map.get("passed") is not None
        else None
    )
    blockers: list[str] = []
    if bool(policy.require_reduced_cross_sample_gate):
        if not cross_sample_available:
            blockers.append("reduced_cross_sample_statistics_missing")
        elif cross_sample_passed is not True:
            blockers.append("reduced_cross_sample_dispersion_failed")
    return (
        {
            "metric": "reduced_cross_sample_dispersion",
            "passed": cross_sample_passed,
            "detail": (
                cross_map.get("rows", [])
                if cross_map is not None
                else "missing reduced_cross_sample_statistics"
            ),
        },
        blockers,
    )


def _landscape_admission_gate(
    landscape_admission_report: Mapping[str, Any],
    policy: VMECJAXNonlinearCampaignPolicy,
) -> tuple[dict[str, Any], list[str]]:
    landscape_passed = bool(landscape_admission_report.get("passed", False))
    blockers: list[str] = []
    if bool(policy.require_landscape_admission_passed) and not landscape_passed:
        blockers.append("replicated_landscape_admission_failed")
    return (
        {
            "metric": "replicated_landscape_admission",
            "passed": landscape_passed,
            "detail": landscape_admission_report.get("next_action"),
        },
        blockers,
    )


def _selected_landscape_candidate(
    landscape_admission_report: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[str]]:
    selected = landscape_admission_report.get("selected_candidate")
    selected_map: Mapping[str, Any] = selected if isinstance(selected, Mapping) else {}
    return selected_map, ([] if selected_map else ["missing_selected_landscape_candidate"])


def _candidate_gate_specs(
    selected_map: Mapping[str, Any],
    policy: VMECJAXNonlinearCampaignPolicy,
) -> list[tuple[str, bool, Any, Any, str]]:
    relative_reduction = _finite_float_or_none(selected_map.get("relative_reduction"))
    z_score = _finite_float_or_none(selected_map.get("uncertainty_z_score"))
    sem_rel = _finite_float_or_none(selected_map.get("combined_sem_rel"))
    n_reports = int(selected_map.get("n_reports", 0) or 0)
    return [
        (
            "landscape_relative_reduction",
            relative_reduction is not None
            and relative_reduction >= float(policy.minimum_landscape_relative_reduction),
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


def _candidate_gates_and_blockers(
    selected_map: Mapping[str, Any],
    policy: VMECJAXNonlinearCampaignPolicy,
) -> tuple[list[dict[str, Any]], list[str]]:
    gates: list[dict[str, Any]] = []
    blockers: list[str] = []
    for metric, passed, value, threshold, blocker in _candidate_gate_specs(
        selected_map, policy
    ):
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
    return gates, blockers


def _campaign_next_action(admitted: bool) -> str:
    if admitted:
        return (
            "launch a bounded multi-control optimizer campaign from the admitted landscape direction, "
            "with matched baseline/candidate t=[350,700] replicated nonlinear audits before promotion"
        )
    return (
        "do not launch a broader nonlinear optimizer campaign; fix the reduced gate, "
        "cross-sample dispersion, or replicated landscape uncertainty first"
    )


def _pack_campaign_admission_report(
    *,
    policy: VMECJAXNonlinearCampaignPolicy,
    blockers: list[str],
    gates: list[dict[str, Any]],
    selected_map: Mapping[str, Any],
) -> dict[str, Any]:
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
        "next_action": _campaign_next_action(admitted),
    }


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
    gates: list[dict[str, Any]] = []
    blockers: list[str] = []
    for gate, gate_blockers in (
        _reduced_prelaunch_gate(reduced_prelaunch_report, policy),
        _reduced_objective_sample_gate(reduced_prelaunch_report),
        _reduced_cross_sample_gate(reduced_prelaunch_report, policy),
        _landscape_admission_gate(landscape_admission_report, policy),
    ):
        gates.append(gate)
        blockers.extend(gate_blockers)
    selected_map, selected_blockers = _selected_landscape_candidate(
        landscape_admission_report
    )
    blockers.extend(selected_blockers)
    candidate_gates, candidate_blockers = _candidate_gates_and_blockers(
        selected_map, policy
    )
    gates.extend(candidate_gates)
    blockers.extend(candidate_blockers)
    return _pack_campaign_admission_report(
        policy=policy,
        blockers=blockers,
        gates=gates,
        selected_map=selected_map,
    )






__all__ = ["build_nonlinear_campaign_admission_report"]
