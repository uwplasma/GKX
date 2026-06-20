"""Replicated nonlinear landscape admission reports for stellarator transport."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from spectraxgk.validation.stellarator.transport_policies import (
    VMECJAXNonlinearAuditPolicy,
    _finite_float_or_none,
)

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
    labels = (
        tuple(str(item) for item in candidate_labels)
        if candidate_labels is not None
        else tuple(f"candidate_{index}" for index, _ in enumerate(candidate_ensembles))
    )
    if len(labels) != len(candidate_ensembles):
        raise ValueError("candidate_labels must have the same length as candidate_ensembles")
    baseline_stats = _ensemble_statistics(baseline_ensemble)
    baseline_blockers = _ensemble_blockers(baseline_stats, role="baseline", policy=policy)
    rows: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    baseline_mean = cast(float | None, baseline_stats.get("ensemble_mean"))
    baseline_sem = cast(float | None, baseline_stats.get("combined_sem"))
    for label, ensemble in zip(labels, candidate_ensembles, strict=True):
        stats = _ensemble_statistics(ensemble)
        blockers = list(baseline_blockers)
        blockers.extend(_ensemble_blockers(stats, role="candidate", policy=policy))
        candidate_mean = cast(float | None, stats.get("ensemble_mean"))
        candidate_sem = cast(float | None, stats.get("combined_sem"))
        relative_reduction = None
        uncertainty_z_score = None
        if baseline_mean is not None and candidate_mean is not None:
            relative_reduction = float((baseline_mean - candidate_mean) / max(abs(baseline_mean), 1.0e-300))
            if relative_reduction < float(policy.minimum_relative_reduction):
                blockers.append("insufficient_relative_reduction")
        else:
            blockers.append("missing_relative_reduction")
        if baseline_mean is not None and candidate_mean is not None and baseline_sem is not None and candidate_sem is not None:
            combined_uncertainty = float(np.hypot(baseline_sem, candidate_sem))
            uncertainty_z_score = float((baseline_mean - candidate_mean) / max(combined_uncertainty, 1.0e-300))
            if uncertainty_z_score < float(policy.minimum_uncertainty_z_score):
                blockers.append("insufficient_uncertainty_separation")
        else:
            blockers.append("missing_uncertainty_z_score")
        row = {
            "label": label,
            "case": stats.get("case"),
            "ensemble_mean": candidate_mean,
            "combined_sem": candidate_sem,
            "combined_sem_rel": stats.get("combined_sem_rel"),
            "n_reports": stats.get("n_reports"),
            "relative_reduction": relative_reduction,
            "uncertainty_z_score": uncertainty_z_score,
            "admission_blockers": blockers,
            "admitted": not blockers,
        }
        rows.append(row)
        if not blockers:
            admitted.append(row)
    selected = None
    if admitted:
        selected = max(
            admitted,
            key=lambda row: (
                float(row.get("relative_reduction") or 0.0),
                float(row.get("uncertainty_z_score") or 0.0),
            ),
        )
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
        "next_action": (
            "use selected direction for the next uncertainty-aware optimizer admission step"
            if selected is not None
            else "do not launch a broader optimizer from this landscape without more resolved nonlinear evidence"
        ),
    }




__all__ = [
    "_ensemble_blockers",
    "_ensemble_statistics",
    "build_nonlinear_landscape_admission_report",
]
