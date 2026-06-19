"""Screening reports for nonlinear turbulence-gradient campaign planning."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence
import math

from spectraxgk.validation.nonlinear_gradient.evidence_classification import (
    classify_gradient_artifact,
)
from spectraxgk.validation.nonlinear_gradient.evidence_core import (
    NonlinearTurbulenceGradientCandidateRankingConfig,
    NonlinearTurbulenceGradientEvidenceConfig,
    _finite_float,
    _json_number,
)
from spectraxgk.validation.nonlinear_gradient.evidence_brackets import (
    _bracket_sweep_recommendation,
    _bracket_sweep_row,
    _delta_key,
    _paired_same_sign_fraction,
    _paired_uncertainty_rel,
    nonlinear_turbulence_gradient_bracket_sweep_report,
)
from spectraxgk.validation.nonlinear_gradient.evidence_scoring import _metric_margin



def _candidate_next_action(
    *,
    passed: bool,
    response_margin: float,
    asymmetry_margin: float,
    uncertainty_margin: float,
    condition_margin: float,
) -> str:
    if passed:
        return (
            "promote only after the source campaign provenance is frozen in docs and CI"
        )
    if response_margin < 1.0:
        return (
            "abandon or enlarge the perturbation only if locality remains bounded; "
            "the heat-flux response is not resolved above the minimum response gate"
        )
    if asymmetry_margin < 1.0 and uncertainty_margin >= 1.0:
        return (
            "repeat with a smaller bracket or nearby control; uncertainty is adequate "
            "but the finite-difference response is nonlocal"
        )
    if uncertainty_margin < 1.0 and asymmetry_margin >= 1.0:
        return (
            "keep the local direction but increase statistical power: longer windows, "
            "more replicas, or a checked amplitude bracket"
        )
    if condition_margin < 1.0:
        return "choose a better-conditioned observable/control pair before rerunning"
    return (
        "do not relax gates; move to an overdetermined least-squares/profile-gradient "
        "campaign with multiple controls and matched long-window replicas"
    )


def nonlinear_turbulence_gradient_candidate_ranking_report(
    artifacts: Sequence[dict[str, Any]],
    *,
    paths: Sequence[str | None] | None = None,
    labels: Sequence[str | None] | None = None,
    config: NonlinearTurbulenceGradientCandidateRankingConfig | None = None,
) -> dict[str, Any]:
    """Rank nonlinear turbulence-gradient candidates without promoting failures.

    The ranking is a planning aid, not a replacement for the production gate.
    It scores each candidate by the weakest normalized evidence margin across
    response, locality, conditioning, and uncertainty.  This makes the next
    campaign choice explicit: candidates with complementary failures should
    move to a profile-gradient or overdetermined least-squares design instead
    of repeating a single boundary coefficient indefinitely.
    """

    cfg = config or NonlinearTurbulenceGradientCandidateRankingConfig()
    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")

    rows: list[dict[str, Any]] = []
    for index, (artifact, path, label) in enumerate(
        zip(artifacts, path_list, label_list)
    ):
        evidence_cfg = NonlinearTurbulenceGradientEvidenceConfig(
            max_gradient_uncertainty_rel=cfg.max_gradient_uncertainty_rel,
            max_fd_asymmetry_rel=cfg.max_fd_asymmetry_rel,
            max_fd_condition_number=cfg.max_fd_condition_number,
            min_fd_response_fraction=cfg.min_fd_response_fraction,
            value_floor=cfg.value_floor,
        )
        classified = classify_gradient_artifact(
            artifact, path=path, config=evidence_cfg
        )
        conditioning = classified.get("conditioning", {})
        if not isinstance(conditioning, dict):
            conditioning = {}
        response_fraction = _finite_float(conditioning.get("response_fraction"))
        fd_asymmetry_rel = _finite_float(conditioning.get("fd_asymmetry_rel"))
        fd_condition_number = _finite_float(conditioning.get("fd_condition_number"))
        gradient_uncertainty_rel = _finite_float(
            conditioning.get("gradient_uncertainty_rel")
        )

        response_margin = _metric_margin(
            response_fraction,
            target=cfg.min_fd_response_fraction,
            sense="min",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        )
        asymmetry_margin = _metric_margin(
            fd_asymmetry_rel,
            target=cfg.max_fd_asymmetry_rel,
            sense="max",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        )
        condition_margin = _metric_margin(
            fd_condition_number,
            target=cfg.max_fd_condition_number,
            sense="max",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        )
        uncertainty_margin = _metric_margin(
            gradient_uncertainty_rel,
            target=cfg.max_gradient_uncertainty_rel,
            sense="max",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        )
        margins = {
            "response": response_margin,
            "locality": asymmetry_margin,
            "conditioning": condition_margin,
            "uncertainty": uncertainty_margin,
        }
        weakest_margin = min(margins.values())
        geometric_score = (
            math.prod(max(value, 0.0) for value in margins.values()) ** 0.25
        )
        if not bool(classified.get("explicit_production_scope", False)):
            geometric_score *= 0.5
        passed = bool(
            classified.get("qualifies_for_production_turbulence_gradient", False)
        )
        parameter_name = str(
            artifact.get("parameter_name") or label or path or f"candidate_{index}"
        )
        failed_gates = [
            str(gate.get("metric", ""))
            for gate in classified.get("gates", [])
            if isinstance(gate, dict) and not bool(gate.get("passed", False))
        ]
        rows.append(
            {
                "rank": None,
                "index": index,
                "label": str(label or parameter_name),
                "path": path,
                "parameter_name": parameter_name,
                "passed": passed,
                "evidence_class": classified.get("evidence_class"),
                "failed_gates": failed_gates,
                "metrics": {
                    "central_gradient": conditioning.get("central_gradient"),
                    "response_fraction": conditioning.get("response_fraction"),
                    "fd_asymmetry_rel": conditioning.get("fd_asymmetry_rel"),
                    "fd_condition_number": conditioning.get("fd_condition_number"),
                    "gradient_uncertainty_rel": conditioning.get(
                        "gradient_uncertainty_rel"
                    ),
                },
                "margins": margins,
                "weakest_margin": _json_number(weakest_margin),
                "score": _json_number(geometric_score),
                "next_action": _candidate_next_action(
                    passed=passed,
                    response_margin=response_margin,
                    asymmetry_margin=asymmetry_margin,
                    uncertainty_margin=uncertainty_margin,
                    condition_margin=condition_margin,
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            bool(row["passed"]),
            float(row["weakest_margin"] or 0.0),
            float(row["score"] or 0.0),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    passed_rows = [row for row in rows if bool(row["passed"])]
    local_but_noisy = [
        row
        for row in rows
        if float(row["margins"]["locality"]) >= 1.0
        and float(row["margins"]["uncertainty"]) < 1.0
    ]
    quiet_but_nonlocal = [
        row
        for row in rows
        if float(row["margins"]["uncertainty"]) >= 1.0
        and float(row["margins"]["locality"]) < 1.0
    ]
    overdetermined_followup = cfg.campaign_context == "overdetermined_followup"
    if passed_rows:
        recommendation = (
            "one or more candidates passes the production evidence gates; freeze "
            "the provenance and promote only the passed artifact"
        )
    elif overdetermined_followup and local_but_noisy and quiet_but_nonlocal:
        recommendation = (
            "the overdetermined follow-up completed with no promotable candidate; "
            "keep the nonlinear-gradient claim fail-closed, target the best local "
            "but noisy control with additional independent replicas or variance "
            "reduction only if the cost is justified, and replace or shrink the "
            "nonlocal controls before another production campaign"
        )
    elif overdetermined_followup and local_but_noisy:
        recommendation = (
            "the overdetermined follow-up found local but statistically unresolved "
            "candidates; keep the claim fail-closed and add independent replicas "
            "or a lower-variance observable before promotion"
        )
    elif overdetermined_followup and quiet_but_nonlocal:
        recommendation = (
            "the overdetermined follow-up found statistically quiet but nonlocal "
            "candidates; keep the claim fail-closed and shrink the perturbation "
            "or choose more local controls before adding replicas"
        )
    elif local_but_noisy and quiet_but_nonlocal:
        recommendation = (
            "use an overdetermined least-squares/profile-gradient campaign: current "
            "single-control candidates have complementary locality and uncertainty failures"
        )
    elif local_but_noisy:
        recommendation = "extend statistical power for the best local direction before changing controls"
    elif quiet_but_nonlocal:
        recommendation = "reduce bracket size or choose a nearby/local control before adding replicas"
    else:
        recommendation = (
            "screen new profile-gradient or objective-gradient controls; current candidates "
            "do not isolate a promotable response"
        )

    return {
        "kind": "nonlinear_turbulence_gradient_candidate_ranking",
        "claim_level": "campaign_planning_not_gradient_evidence",
        "passed": bool(passed_rows),
        "promotion_ready_candidate_count": len(passed_rows),
        "best_candidate": rows[0] if rows else None,
        "recommendation": recommendation,
        "config": asdict(cfg),
        "candidates": rows,
    }



__all__ = [
    "_bracket_sweep_recommendation",
    "_bracket_sweep_row",
    "_candidate_next_action",
    "_delta_key",
    "_metric_margin",
    "_paired_same_sign_fraction",
    "_paired_uncertainty_rel",
    "nonlinear_turbulence_gradient_bracket_sweep_report",
    "nonlinear_turbulence_gradient_candidate_ranking_report",
]
