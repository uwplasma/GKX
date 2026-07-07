"""Stellarator nonlinear-transport admission and redesign reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from spectraxgk.objectives.vmec_transport_admission import (
    VMECJAXNonlinearAuditPolicy,
    VMECJAXNonlinearCampaignPolicy,
    VMECJAXReducedPrelaunchPolicy,
    _finite_float_or_none,
    transport_objective_sample_summary,
)

# ---- replicated landscape admission ----
@dataclass(frozen=True)
class _LandscapeReductionMetrics:
    relative_reduction: float | None
    uncertainty_z_score: float | None


def _ensemble_statistics(ensemble: Mapping[str, Any]) -> dict[str, Any]:
    stats = ensemble.get("statistics", {})
    if not isinstance(stats, Mapping):
        stats = {}
    return {
        "passed": bool(ensemble.get("passed", False)),
        "ensemble_mean": _finite_float_or_none(stats.get("ensemble_mean")),
        "combined_sem": _finite_float_or_none(stats.get("combined_sem")),
        "combined_sem_rel": _finite_float_or_none(stats.get("combined_sem_rel")),
        "n_reports": int(stats.get("n_reports", 0) or 0),
        "case": ensemble.get("case"),
    }


def _ensemble_blockers(
    stats: Mapping[str, Any],
    *,
    role: str,
    policy: VMECJAXNonlinearAuditPolicy,
) -> list[str]:
    blockers: list[str] = []
    if not bool(stats.get("passed")):
        blockers.append(f"{role}_ensemble_failed")
    if stats.get("ensemble_mean") is None:
        blockers.append(f"{role}_missing_ensemble_mean")
    if stats.get("combined_sem") is None:
        blockers.append(f"{role}_missing_combined_sem")
    sem_rel = stats.get("combined_sem_rel")
    if sem_rel is None:
        blockers.append(f"{role}_missing_combined_sem_rel")
    elif float(sem_rel) > float(policy.maximum_combined_sem_rel):
        blockers.append(f"{role}_combined_sem_rel_too_large")
    if int(stats.get("n_reports", 0) or 0) < int(policy.minimum_replicate_count):
        blockers.append(f"{role}_insufficient_replicates")
    return blockers


def _candidate_landscape_labels(
    candidate_ensembles: Sequence[Mapping[str, Any]],
    candidate_labels: Sequence[str] | None,
) -> tuple[str, ...]:
    labels = (
        tuple(str(item) for item in candidate_labels)
        if candidate_labels is not None
        else tuple(f"candidate_{index}" for index, _ in enumerate(candidate_ensembles))
    )
    if len(labels) != len(candidate_ensembles):
        raise ValueError("candidate_labels must have the same length as candidate_ensembles")
    return labels


def _candidate_reduction_metrics(
    *,
    baseline_mean: float | None,
    candidate_mean: float | None,
    baseline_sem: float | None,
    candidate_sem: float | None,
    blockers: list[str],
    policy: VMECJAXNonlinearAuditPolicy,
) -> _LandscapeReductionMetrics:
    relative_reduction = None
    uncertainty_z_score = None
    if baseline_mean is not None and candidate_mean is not None:
        relative_reduction = float(
            (baseline_mean - candidate_mean) / max(abs(baseline_mean), 1.0e-300)
        )
        if relative_reduction < float(policy.minimum_relative_reduction):
            blockers.append("insufficient_relative_reduction")
    else:
        blockers.append("missing_relative_reduction")
    if (
        baseline_mean is not None
        and candidate_mean is not None
        and baseline_sem is not None
        and candidate_sem is not None
    ):
        combined_uncertainty = float(np.hypot(baseline_sem, candidate_sem))
        uncertainty_z_score = float(
            (baseline_mean - candidate_mean) / max(combined_uncertainty, 1.0e-300)
        )
        if uncertainty_z_score < float(policy.minimum_uncertainty_z_score):
            blockers.append("insufficient_uncertainty_separation")
    else:
        blockers.append("missing_uncertainty_z_score")
    return _LandscapeReductionMetrics(relative_reduction, uncertainty_z_score)


def _candidate_landscape_row(
    *,
    label: str,
    ensemble: Mapping[str, Any],
    baseline_stats: Mapping[str, Any],
    baseline_blockers: list[str],
    policy: VMECJAXNonlinearAuditPolicy,
) -> dict[str, Any]:
    stats = _ensemble_statistics(ensemble)
    blockers = list(baseline_blockers)
    blockers.extend(_ensemble_blockers(stats, role="candidate", policy=policy))
    candidate_mean = cast(float | None, stats.get("ensemble_mean"))
    candidate_sem = cast(float | None, stats.get("combined_sem"))
    metrics = _candidate_reduction_metrics(
        baseline_mean=cast(float | None, baseline_stats.get("ensemble_mean")),
        candidate_mean=candidate_mean,
        baseline_sem=cast(float | None, baseline_stats.get("combined_sem")),
        candidate_sem=candidate_sem,
        blockers=blockers,
        policy=policy,
    )
    return {
        "label": label,
        "case": stats.get("case"),
        "ensemble_mean": candidate_mean,
        "combined_sem": candidate_sem,
        "combined_sem_rel": stats.get("combined_sem_rel"),
        "n_reports": stats.get("n_reports"),
        "relative_reduction": metrics.relative_reduction,
        "uncertainty_z_score": metrics.uncertainty_z_score,
        "admission_blockers": blockers,
        "admitted": not blockers,
    }


def _select_landscape_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    admitted = [row for row in rows if not row.get("admission_blockers")]
    if not admitted:
        return None
    return max(
        admitted,
        key=lambda row: (
            float(row.get("relative_reduction") or 0.0),
            float(row.get("uncertainty_z_score") or 0.0),
        ),
    )


def _landscape_next_action(selected: Mapping[str, Any] | None) -> str:
    if selected is not None:
        return "use selected direction for the next uncertainty-aware optimizer admission step"
    return (
        "do not launch a broader optimizer from this landscape without more resolved "
        "nonlinear evidence"
    )


def build_nonlinear_landscape_admission_report(
    baseline_ensemble: Mapping[str, Any],
    candidate_ensembles: Sequence[Mapping[str, Any]],
    *,
    candidate_labels: Sequence[str] | None = None,
    policy: VMECJAXNonlinearAuditPolicy | None = None,
) -> dict[str, Any]:
    """Select an uncertainty-separated nonlinear candidate from a landscape.

    This gate is for boundary-coefficient or line-search landscapes where a
    small number of selected points have replicated late-window nonlinear
    ensembles.  It does not validate multi-coefficient global optimization by
    itself; it only answers whether any supplied candidate has a statistically
    resolved lower heat flux than the supplied baseline ensemble.
    """

    policy = policy or VMECJAXNonlinearAuditPolicy()
    labels = _candidate_landscape_labels(candidate_ensembles, candidate_labels)
    baseline_stats = _ensemble_statistics(baseline_ensemble)
    baseline_blockers = _ensemble_blockers(baseline_stats, role="baseline", policy=policy)
    rows = [
        _candidate_landscape_row(
            label=label,
            ensemble=ensemble,
            baseline_stats=baseline_stats,
            baseline_blockers=baseline_blockers,
            policy=policy,
        )
        for label, ensemble in zip(labels, candidate_ensembles, strict=True)
    ]
    selected = _select_landscape_candidate(rows)
    return {
        "kind": "nonlinear_landscape_admission_report",
        "claim_scope": (
            "selected replicated nonlinear landscape admission; not a multi-coefficient "
            "or multi-flux-tube turbulent-optimization claim"
        ),
        "policy": policy.to_dict(),
        "baseline": baseline_stats,
        "candidates": rows,
        "selected_candidate": selected,
        "passed": selected is not None,
        "next_action": _landscape_next_action(selected),
    }




# ---- reduced prelaunch gates ----
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


# ---- campaign admission gates ----
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






# ---- matched nonlinear audit redesign ----
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






__all__ = [
    "build_nonlinear_audit_redesign_report",
    "build_nonlinear_campaign_admission_report",
    "build_nonlinear_landscape_admission_report",
    "build_reduced_nonlinear_audit_prelaunch_report",
]
