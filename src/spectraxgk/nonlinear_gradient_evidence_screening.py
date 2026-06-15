"""Screening reports for nonlinear turbulence-gradient campaign planning."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence
import math

from spectraxgk.nonlinear_gradient_evidence_classification import (
    classify_gradient_artifact,
)
from spectraxgk.nonlinear_gradient_evidence_core import (
    NonlinearTurbulenceGradientBracketSweepConfig,
    NonlinearTurbulenceGradientCandidateRankingConfig,
    NonlinearTurbulenceGradientEvidenceConfig,
    _finite_float,
    _json_number,
)


def _metric_margin(
    value: float | None,
    *,
    target: float,
    sense: str,
    cap: float,
    value_floor: float,
) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    finite_value = float(value)
    if sense == "min":
        margin = finite_value / max(float(target), float(value_floor))
    elif sense == "max":
        margin = float(target) / max(abs(finite_value), float(value_floor))
    else:  # pragma: no cover - guarded by internal call sites.
        raise ValueError(f"unsupported margin sense: {sense}")
    return max(0.0, min(float(cap), margin))


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


def _paired_uncertainty_rel(artifact: dict[str, Any]) -> float | None:
    diagnostics = artifact.get("paired_replicate_diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    return _finite_float(diagnostics.get("central_gradient_uncertainty_rel"))


def _paired_same_sign_fraction(artifact: dict[str, Any]) -> float | None:
    diagnostics = artifact.get("paired_replicate_diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    return _finite_float(diagnostics.get("same_sign_fraction"))


def _bracket_sweep_row(
    artifact: dict[str, Any],
    *,
    label: str | None,
    path: str | None,
    config: NonlinearTurbulenceGradientBracketSweepConfig,
) -> dict[str, Any]:
    evidence_cfg = NonlinearTurbulenceGradientEvidenceConfig(
        max_gradient_uncertainty_rel=config.max_gradient_uncertainty_rel,
        max_fd_asymmetry_rel=config.max_fd_asymmetry_rel,
        max_fd_condition_number=config.max_fd_condition_number,
        min_fd_response_fraction=config.min_fd_response_fraction,
        value_floor=config.value_floor,
    )
    classified = classify_gradient_artifact(artifact, path=path, config=evidence_cfg)
    conditioning = classified.get("conditioning")
    if not isinstance(conditioning, dict):
        conditioning = {}
    delta = _finite_float(artifact.get("delta_parameter"))
    response_fraction = _finite_float(conditioning.get("response_fraction"))
    fd_asymmetry_rel = _finite_float(conditioning.get("fd_asymmetry_rel"))
    fd_condition_number = _finite_float(conditioning.get("fd_condition_number"))
    gradient_uncertainty_rel = _finite_float(
        conditioning.get("gradient_uncertainty_rel")
    )
    paired_uncertainty_rel = _paired_uncertainty_rel(artifact)
    paired_same_sign = _paired_same_sign_fraction(artifact)
    response_margin = _metric_margin(
        response_fraction,
        target=config.min_fd_response_fraction,
        sense="min",
        cap=config.score_cap,
        value_floor=config.value_floor,
    )
    locality_margin = _metric_margin(
        fd_asymmetry_rel,
        target=config.max_fd_asymmetry_rel,
        sense="max",
        cap=config.score_cap,
        value_floor=config.value_floor,
    )
    uncertainty_margin = _metric_margin(
        gradient_uncertainty_rel,
        target=config.max_gradient_uncertainty_rel,
        sense="max",
        cap=config.score_cap,
        value_floor=config.value_floor,
    )
    condition_margin = _metric_margin(
        fd_condition_number,
        target=config.max_fd_condition_number,
        sense="max",
        cap=config.score_cap,
        value_floor=config.value_floor,
    )
    margins = {
        "response": response_margin,
        "locality": locality_margin,
        "uncertainty": uncertainty_margin,
        "conditioning": condition_margin,
    }
    repeated_bracket_stable = (
        paired_uncertainty_rel is not None
        and paired_uncertainty_rel <= float(config.max_repeated_bracket_uncertainty_rel)
        and paired_same_sign is not None
        and paired_same_sign >= float(config.min_repeated_bracket_same_sign_fraction)
    )
    return {
        "label": str(label or artifact.get("parameter_name") or path or ""),
        "path": path,
        "parameter_name": str(artifact.get("parameter_name", "")),
        "delta_parameter": _json_number(delta),
        "passed": bool(
            classified.get("qualifies_for_production_turbulence_gradient", False)
        ),
        "metrics": {
            "central_gradient": conditioning.get("central_gradient"),
            "response_fraction": conditioning.get("response_fraction"),
            "fd_asymmetry_rel": conditioning.get("fd_asymmetry_rel"),
            "fd_condition_number": conditioning.get("fd_condition_number"),
            "gradient_uncertainty_rel": conditioning.get("gradient_uncertainty_rel"),
            "paired_gradient_uncertainty_rel": _json_number(paired_uncertainty_rel),
            "paired_same_sign_fraction": _json_number(paired_same_sign),
            "repeated_bracket_stable": repeated_bracket_stable,
        },
        "margins": margins,
        "weakest_margin": _json_number(min(margins.values())),
        "score": _json_number(
            math.prod(max(value, 0.0) for value in margins.values()) ** 0.25
        ),
        "failed_gates": [
            str(gate.get("metric", ""))
            for gate in classified.get("gates", [])
            if isinstance(gate, dict) and not bool(gate.get("passed", False))
        ],
    }


def _delta_key(row: dict[str, Any]) -> float:
    delta = _finite_float(row.get("delta_parameter"))
    if delta is None:
        return math.inf
    return float(delta)


def _bracket_sweep_recommendation(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "run at least two matched plus/minus perturbation amplitudes before claiming bracket locality"
    parameter_names = {
        str(row.get("parameter_name", "")) for row in rows if row.get("parameter_name")
    }
    if len(parameter_names) > 1:
        return (
            "mixed controls were supplied to a same-control bracket sweep; split the "
            "artifacts by control or use the nonlinear turbulence-gradient candidate "
            "ranking/overdetermined campaign planner"
        )
    passed_rows = [row for row in rows if bool(row.get("passed", False))]
    if passed_rows:
        best = min(passed_rows, key=_delta_key)
        return (
            "promote only the passed same-control bracket after freezing provenance; "
            f"smallest passing delta is {best.get('delta_parameter')}"
        )

    response_ok = [row for row in rows if float(row["margins"]["response"]) >= 1.0]
    response_ok_gradients = [
        _finite_float(row.get("metrics", {}).get("central_gradient"))
        for row in response_ok
        if isinstance(row.get("metrics"), dict)
    ]
    response_ok_signs = {
        math.copysign(1.0, float(value))
        for value in response_ok_gradients
        if value is not None and value != 0.0
    }
    if len(response_ok_signs) > 1:
        return (
            "same-control resolved brackets change central-gradient sign; do not add "
            "replicas at one amplitude, and move to a locality/amplitude sweep with "
            "stricter provenance or a smoother composite profile-gradient direction"
        )
    local_rows = [
        row for row in response_ok if float(row["margins"]["locality"]) >= 1.0
    ]
    quiet_rows = [
        row for row in response_ok if float(row["margins"]["uncertainty"]) >= 1.0
    ]
    repeated_unstable = [
        row
        for row in rows
        if row["metrics"].get("paired_gradient_uncertainty_rel") is not None
        and not bool(row["metrics"].get("repeated_bracket_stable", False))
    ]
    if local_rows and not quiet_rows:
        if repeated_unstable:
            return (
                "do not add replicas at the same bracket yet; matched-pair diagnostics "
                "show seed-level instability, so run a perturbation-amplitude/locality "
                "sweep or switch to a smoother composite profile-gradient direction"
            )
        return (
            "locality is acceptable but uncertainty is not; add statistical power only "
            "after a second nearby perturbation amplitude confirms the same gradient sign"
        )
    if quiet_rows and not local_rows:
        return (
            "uncertainty is acceptable only for nonlocal brackets; shrink the perturbation "
            "or choose a more local control before adding replicas"
        )
    if response_ok and local_rows and quiet_rows:
        return (
            "the numerical bracket margins are resolved, local, and quiet, but no input "
            "artifact has production long-window scope; rerun or re-export with matched "
            "post-transient provenance before considering promotion"
        )
    if response_ok and not local_rows and not quiet_rows:
        return (
            "the response is detectable but neither local nor statistically resolved; "
            "prefer an overdetermined/profile-gradient campaign over more single-control runs"
        )
    return (
        "the heat-flux response is not resolved at the tested amplitudes; abandon this "
        "control or enlarge the perturbation only if a locality sweep remains bounded"
    )


def nonlinear_turbulence_gradient_bracket_sweep_report(
    artifacts: Sequence[dict[str, Any]],
    *,
    labels: Sequence[str | None] | None = None,
    paths: Sequence[str | None] | None = None,
    config: NonlinearTurbulenceGradientBracketSweepConfig | None = None,
) -> dict[str, Any]:
    """Summarize a same-control perturbation-amplitude sweep.

    This is a planning/claim-boundary utility.  It does not promote nonlinear
    turbulence-gradient evidence unless an input finite-difference artifact
    already passes the production long-window gate.  Its main purpose is to
    decide whether the next expensive campaign should add replicas at the same
    bracket, change the perturbation amplitude, or move to an overdetermined
    profile-gradient direction.
    """

    cfg = config or NonlinearTurbulenceGradientBracketSweepConfig()
    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")

    rows = [
        _bracket_sweep_row(artifact, label=label, path=path, config=cfg)
        for artifact, label, path in zip(artifacts, label_list, path_list)
    ]
    rows.sort(key=_delta_key)
    parameter_names = sorted(
        {row["parameter_name"] for row in rows if row["parameter_name"]}
    )
    same_control = len(parameter_names) <= 1
    passed_rows = [row for row in rows if bool(row.get("passed", False))]
    return {
        "kind": "nonlinear_turbulence_gradient_bracket_sweep",
        "claim_level": "same_control_bracket_locality_planning_not_gradient_promotion",
        "passed": bool(passed_rows) and same_control,
        "promotion_ready_bracket_count": len(passed_rows) if same_control else 0,
        "same_control_gate": {
            "passed": same_control,
            "parameter_names": parameter_names,
        },
        "parameter_names": parameter_names,
        "recommendation": _bracket_sweep_recommendation(rows),
        "config": asdict(cfg),
        "brackets": rows,
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
