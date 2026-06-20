"""Matched-replicate follow-up plans for nonlinear turbulence-gradient audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
import math

from spectraxgk.validation.nonlinear_gradient.followup_core import (
    STATE_TO_RUN_STATE,
    NonlinearGradientFollowupConfig,
    _artifact_passed,
    _json_number,
    _metric,
    _replicate_count,
    _seed_numbers,
)


def _required_replicates(
    *,
    current_count: int | None,
    uncertainty_rel: float,
    config: NonlinearGradientFollowupConfig,
) -> int | None:
    if current_count is None or current_count <= 0:
        return None
    target = max(float(config.max_gradient_uncertainty_rel), float(config.value_floor))
    scale = (float(uncertainty_rel) / target) ** 2 * float(config.sem_safety_factor)
    return max(current_count + 1, int(math.ceil(current_count * scale)))


def _planned_matched_runs(
    *,
    source_ensembles: Mapping[str, Any],
    extra_replicates_per_state: int,
    config: NonlinearGradientFollowupConfig,
) -> list[dict[str, Any]]:
    if extra_replicates_per_state <= 0:
        return []
    seeds = _seed_numbers(source_ensembles)
    next_seed = max(seeds) + 1 if seeds else 31
    runs: list[dict[str, Any]] = []
    for offset in range(extra_replicates_per_state):
        seed = next_seed + offset
        for source_state, run_state in STATE_TO_RUN_STATE.items():
            if source_state not in source_ensembles:
                continue
            runs.append(
                {
                    "state": run_state,
                    "variant_axis": "seed",
                    "variant_label": f"seed{seed}",
                    "seed": seed,
                    "timestep": float(config.default_nominal_timestep),
                    "reason": (
                        "independent nominal-timestep replicate to reduce "
                        "central finite-difference gradient uncertainty"
                    ),
                }
            )
    return runs


@dataclass(frozen=True)
class _FollowupMetrics:
    response_fraction: float | None
    asymmetry_rel: float | None
    uncertainty_rel: float | None


@dataclass(frozen=True)
class _FollowupDecision:
    action: str
    recommendation: str
    required_replicates: int | None
    extra_replicates: int
    planned_runs: list[dict[str, Any]]


def _validated_followup_config(
    config: NonlinearGradientFollowupConfig | None,
) -> NonlinearGradientFollowupConfig:
    cfg = config or NonlinearGradientFollowupConfig()
    if cfg.max_extra_replicates_per_state < 0:
        raise ValueError("max_extra_replicates_per_state must be non-negative")
    if cfg.sem_safety_factor <= 0.0:
        raise ValueError("sem_safety_factor must be positive")
    if cfg.max_gradient_uncertainty_rel <= 0.0:
        raise ValueError("max_gradient_uncertainty_rel must be positive")
    return cfg


def _normalized_optional_labels(
    *,
    artifacts: Sequence[Mapping[str, Any]],
    values: Sequence[str | None] | None,
    name: str,
) -> list[str | None]:
    value_list = list(values or [None] * len(artifacts))
    if len(value_list) != len(artifacts):
        raise ValueError(f"{name} length must match artifacts")
    return value_list


def _source_ensembles(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    source_ensembles_raw = artifact.get("source_ensembles")
    return source_ensembles_raw if isinstance(source_ensembles_raw, Mapping) else {}


def _candidate_label(
    artifact: Mapping[str, Any],
    *,
    path: str | None,
    label: str | None,
    index: int,
) -> str:
    return str(label or artifact.get("parameter_name") or path or index)


def _candidate_metrics(artifact: Mapping[str, Any]) -> _FollowupMetrics:
    return _FollowupMetrics(
        response_fraction=_metric(artifact, "response_fraction"),
        asymmetry_rel=_metric(artifact, "fd_asymmetry_rel", "asymmetry_rel"),
        uncertainty_rel=_metric(
            artifact,
            "gradient_uncertainty_rel",
            "gradient_relative_uncertainty",
        ),
    )


def _metric_acceptance(
    metrics: _FollowupMetrics,
    cfg: NonlinearGradientFollowupConfig,
) -> tuple[bool, bool, bool]:
    response_ok = metrics.response_fraction is not None and metrics.response_fraction >= float(
        cfg.min_fd_response_fraction
    )
    locality_ok = metrics.asymmetry_rel is not None and metrics.asymmetry_rel <= float(
        cfg.max_fd_asymmetry_rel
    )
    uncertainty_ok = metrics.uncertainty_rel is not None and metrics.uncertainty_rel <= float(
        cfg.max_gradient_uncertainty_rel
    )
    return response_ok, locality_ok, uncertainty_ok


def _no_runs_decision(action: str, recommendation: str) -> _FollowupDecision:
    return _FollowupDecision(
        action=action,
        recommendation=recommendation,
        required_replicates=None,
        extra_replicates=0,
        planned_runs=[],
    )


def _replicate_decision(
    *,
    source_ensembles: Mapping[str, Any],
    current_replicates: int | None,
    uncertainty_rel: float,
    cfg: NonlinearGradientFollowupConfig,
) -> _FollowupDecision:
    required_replicates = _required_replicates(
        current_count=current_replicates,
        uncertainty_rel=uncertainty_rel,
        config=cfg,
    )
    if required_replicates is None:
        return _no_runs_decision(
            "recover_replicate_metadata",
            (
                "uncertainty is marginal but the artifact lacks replicate counts; "
                "recover ensemble metadata before launching runs"
            ),
        )
    extra_replicates = max(0, required_replicates - int(current_replicates or 0))
    extra_replicates = min(extra_replicates, int(cfg.max_extra_replicates_per_state))
    return _FollowupDecision(
        action="add_matched_nominal_seed_replicates",
        recommendation=(
            "response and locality pass, but uncertainty is marginal; add only "
            "the matched independent replicas needed by the 1/sqrt(N) estimate"
        ),
        required_replicates=required_replicates,
        extra_replicates=extra_replicates,
        planned_runs=_planned_matched_runs(
            source_ensembles=source_ensembles,
            extra_replicates_per_state=extra_replicates,
            config=cfg,
        ),
    )


def _followup_decision(
    *,
    passed: bool,
    metrics: _FollowupMetrics,
    source_ensembles: Mapping[str, Any],
    current_replicates: int | None,
    cfg: NonlinearGradientFollowupConfig,
) -> _FollowupDecision:
    response_ok, locality_ok, uncertainty_ok = _metric_acceptance(metrics, cfg)
    if passed:
        return _no_runs_decision(
            "freeze_promoted_candidate",
            "candidate already passes production gates; freeze provenance",
        )
    if not response_ok:
        return _no_runs_decision(
            "replace_control_or_increase_checked_bracket",
            (
                "finite-difference response is not resolved; change the control or "
                "perform a checked bracket sweep before adding replicas"
            ),
        )
    if not locality_ok:
        return _no_runs_decision(
            "shrink_bracket_or_replace_control",
            (
                "finite-difference bracket is nonlocal; shrink the perturbation or "
                "replace the control before adding replicas"
            ),
        )
    if not uncertainty_ok and metrics.uncertainty_rel is not None:
        return _replicate_decision(
            source_ensembles=source_ensembles,
            current_replicates=current_replicates,
            uncertainty_rel=metrics.uncertainty_rel,
            cfg=cfg,
        )
    return _no_runs_decision(
        "no_followup_needed",
        "all scalar gates are satisfied, but artifact was not marked passed",
    )


def _annotated_planned_runs(
    planned_runs: Sequence[Mapping[str, Any]],
    *,
    index: int,
    candidate_label: str,
) -> list[dict[str, Any]]:
    return [
        {
            **run,
            "candidate_index": index,
            "candidate_label": candidate_label,
        }
        for run in planned_runs
    ]


def _candidate_action_row(
    *,
    index: int,
    artifact: Mapping[str, Any],
    path: str | None,
    label: str | None,
    cfg: NonlinearGradientFollowupConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_ensembles = _source_ensembles(artifact)
    metrics = _candidate_metrics(artifact)
    current_replicates = _replicate_count(source_ensembles)
    passed = _artifact_passed(artifact)
    decision = _followup_decision(
        passed=passed,
        metrics=metrics,
        source_ensembles=source_ensembles,
        current_replicates=current_replicates,
        cfg=cfg,
    )
    candidate_label = _candidate_label(
        artifact,
        path=path,
        label=label,
        index=index,
    )
    return (
        {
            "index": index,
            "label": candidate_label,
            "path": path,
            "parameter_name": str(artifact.get("parameter_name") or ""),
            "passed": passed,
            "action": decision.action,
            "recommendation": decision.recommendation,
            "metrics": {
                "response_fraction": _json_number(metrics.response_fraction),
                "fd_asymmetry_rel": _json_number(metrics.asymmetry_rel),
                "gradient_uncertainty_rel": _json_number(metrics.uncertainty_rel),
            },
            "current_replicates_per_state": current_replicates,
            "estimated_required_replicates_per_state": decision.required_replicates,
            "extra_replicates_per_state": decision.extra_replicates,
            "planned_run_count": len(decision.planned_runs),
            "planned_runs": decision.planned_runs,
        },
        _annotated_planned_runs(
            decision.planned_runs,
            index=index,
            candidate_label=candidate_label,
        ),
    )


def _candidate_action_groups(
    candidate_actions: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    promoted = [
        row for row in candidate_actions if row["action"] == "freeze_promoted_candidate"
    ]
    runnable = [row for row in candidate_actions if row["planned_run_count"]]
    nonlocal_rows = [
        row
        for row in candidate_actions
        if row["action"] == "shrink_bracket_or_replace_control"
    ]
    unresolved_rows = [
        row
        for row in candidate_actions
        if row["action"] == "replace_control_or_increase_checked_bracket"
    ]
    return promoted, runnable, nonlocal_rows, unresolved_rows


def _next_followup_action(
    *,
    promoted: Sequence[Mapping[str, Any]],
    runnable: Sequence[Mapping[str, Any]],
    nonlocal_rows: Sequence[Mapping[str, Any]],
    unresolved_rows: Sequence[Mapping[str, Any]],
) -> str:
    if promoted:
        return "freeze the promoted candidate and do not launch more follow-up runs"
    if runnable:
        return "launch the bounded matched-replicate follow-up for the listed local noisy candidate"
    if nonlocal_rows:
        return "run a smaller-bracket or replacement-control sweep before adding replicas"
    if unresolved_rows:
        return "choose controls with a resolved heat-flux response before adding replicas"
    return "inspect artifacts; no safe production follow-up was inferred"


def _pack_followup_plan(
    *,
    case: str,
    cfg: NonlinearGradientFollowupConfig,
    candidate_actions: list[dict[str, Any]],
    planned_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    promoted, runnable, nonlocal_rows, unresolved_rows = _candidate_action_groups(
        candidate_actions
    )
    next_action = _next_followup_action(
        promoted=promoted,
        runnable=runnable,
        nonlocal_rows=nonlocal_rows,
        unresolved_rows=unresolved_rows,
    )
    return {
        "kind": "nonlinear_turbulence_gradient_followup_plan",
        "claim_level": "campaign_planning_not_gradient_evidence",
        "case": case,
        "passed": bool(promoted),
        "next_action": next_action,
        "config": asdict(cfg),
        "summary": {
            "candidate_count": len(candidate_actions),
            "promoted_candidate_count": len(promoted),
            "runnable_followup_candidate_count": len(runnable),
            "planned_run_count": len(planned_runs),
        },
        "candidate_actions": candidate_actions,
        "planned_runs": planned_runs,
    }


def nonlinear_gradient_followup_plan(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    paths: Sequence[str | None] | None = None,
    labels: Sequence[str | None] | None = None,
    case: str = "nonlinear_turbulence_gradient_followup",
    config: NonlinearGradientFollowupConfig | None = None,
) -> dict[str, Any]:
    """Build a bounded, fail-closed follow-up plan from gradient artifacts."""

    cfg = _validated_followup_config(config)
    path_list = _normalized_optional_labels(
        artifacts=artifacts,
        values=paths,
        name="paths",
    )
    label_list = _normalized_optional_labels(
        artifacts=artifacts,
        values=labels,
        name="labels",
    )
    candidate_actions: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    for index, (artifact, path, label) in enumerate(
        zip(artifacts, path_list, label_list)
    ):
        row, annotated_runs = _candidate_action_row(
            index=index,
            artifact=artifact,
            path=path,
            label=label,
            cfg=cfg,
        )
        candidate_actions.append(row)
        all_runs.extend(annotated_runs)
    return _pack_followup_plan(
        case=case,
        cfg=cfg,
        candidate_actions=candidate_actions,
        planned_runs=all_runs,
    )


__all__ = [
    "_planned_matched_runs",
    "_required_replicates",
    "nonlinear_gradient_followup_plan",
]
