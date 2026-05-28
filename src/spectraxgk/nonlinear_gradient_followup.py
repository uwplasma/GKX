"""Targeted follow-up plans for nonlinear turbulence-gradient audits.

This module turns failed long-window central finite-difference artifacts into a
bounded run prescription.  It is deliberately conservative: extra replicas are
recommended only when the finite-difference response is resolved and local, but
the propagated gradient uncertainty is slightly too large.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
import math
import re


STATE_TO_RUN_STATE = {
    "baseline": "baseline",
    "plus": "plus_delta",
    "minus": "minus_delta",
}


@dataclass(frozen=True)
class NonlinearGradientFollowupConfig:
    """Acceptance and cost controls for the follow-up planner."""

    max_gradient_uncertainty_rel: float = 0.50
    max_fd_asymmetry_rel: float = 0.50
    min_fd_response_fraction: float = 0.03
    sem_safety_factor: float = 1.10
    max_extra_replicates_per_state: int = 4
    default_nominal_timestep: float = 0.05
    value_floor: float = 1.0e-12


@dataclass(frozen=True)
class NonlinearGradientCandidateDesignConfig:
    """Conditioning limits for the next nonlinear-gradient campaign design."""

    max_gradient_uncertainty_rel: float = 0.50
    max_fd_asymmetry_rel: float = 0.50
    min_fd_response_fraction: float = 0.03
    sem_safety_factor: float = 1.10
    max_extra_replicates_per_state: int = 4
    max_checked_bracket_scale: float = 1.50
    locality_safety_factor: float = 0.95
    value_floor: float = 1.0e-12


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _json_number(value: Any) -> float | int | None:
    number = _finite_float(value)
    if number is None:
        return None
    if isinstance(value, int):
        return value
    return float(number)


def _metric(payload: Mapping[str, Any], *names: str) -> float | None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}
    conditioning = payload.get("conditioning")
    if isinstance(conditioning, Mapping):
        sources: tuple[Mapping[str, Any], ...] = (metrics, conditioning, payload)
    else:
        sources = (metrics, payload)
    for source in sources:
        for name in names:
            value = _finite_float(source.get(name))
            if value is not None:
                return value
    return None


def _artifact_passed(payload: Mapping[str, Any]) -> bool:
    if bool(payload.get("passed", False)):
        return True
    nested = payload.get("nonlinear_turbulence_gradient_gate")
    return isinstance(nested, Mapping) and bool(nested.get("passed", False))


def _label_from_row(row: Mapping[str, Any]) -> str | None:
    for key in ("variant_label", "source_artifact", "summary_artifact", "path"):
        value = row.get(key)
        if not isinstance(value, str):
            continue
        match = re.search(r"(seed[0-9]+|dt[0-9]+(?:p[0-9]+)?)", value)
        if match:
            return match.group(1)
    return None


def _seed_numbers(source_ensembles: Mapping[str, Any]) -> list[int]:
    seeds: list[int] = []
    for raw in source_ensembles.values():
        if not isinstance(raw, Mapping):
            continue
        rows = raw.get("rows")
        if not isinstance(rows, Sequence):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            label = _label_from_row(row)
            if label is None:
                continue
            match = re.fullmatch(r"seed([0-9]+)", label)
            if match:
                seeds.append(int(match.group(1)))
    return seeds


def _replicate_count(source_ensembles: Mapping[str, Any]) -> int | None:
    counts: list[int] = []
    for state in ("minus", "baseline", "plus"):
        raw = source_ensembles.get(state)
        if not isinstance(raw, Mapping):
            continue
        count = _finite_int(raw.get("n_reports"))
        if count is None:
            rows = raw.get("rows")
            count = len(rows) if isinstance(rows, Sequence) else None
        if count is not None:
            counts.append(count)
    return min(counts) if counts else None


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


def _required_replicates_for_scaled_bracket(
    *,
    current_count: int | None,
    uncertainty_rel: float | None,
    bracket_scale: float,
    config: NonlinearGradientCandidateDesignConfig,
) -> int | None:
    if current_count is None or current_count <= 0:
        return None
    if uncertainty_rel is None or bracket_scale <= 0.0:
        return None
    target = max(float(config.max_gradient_uncertainty_rel), float(config.value_floor))
    scale = (float(uncertainty_rel) / (target * float(bracket_scale))) ** 2
    scale *= float(config.sem_safety_factor)
    return max(current_count, int(math.ceil(current_count * scale)))


def _design_row(
    artifact: Mapping[str, Any],
    *,
    index: int,
    path: str | None,
    label: str | None,
    config: NonlinearGradientCandidateDesignConfig,
) -> dict[str, Any]:
    source_ensembles_raw = artifact.get("source_ensembles")
    source_ensembles = source_ensembles_raw if isinstance(source_ensembles_raw, Mapping) else {}
    response_fraction = _metric(artifact, "response_fraction")
    asymmetry_rel = _metric(artifact, "fd_asymmetry_rel", "asymmetry_rel")
    uncertainty_rel = _metric(
        artifact,
        "gradient_uncertainty_rel",
        "gradient_relative_uncertainty",
    )
    current_replicates = _replicate_count(source_ensembles)
    passed = _artifact_passed(artifact)

    response_ok = response_fraction is not None and response_fraction >= config.min_fd_response_fraction
    locality_ok = asymmetry_rel is not None and asymmetry_rel <= config.max_fd_asymmetry_rel
    uncertainty_ok = uncertainty_rel is not None and uncertainty_rel <= config.max_gradient_uncertainty_rel

    uncertainty_required_scale = None
    if uncertainty_rel is not None:
        uncertainty_required_scale = max(1.0, uncertainty_rel / max(config.max_gradient_uncertainty_rel, config.value_floor))

    locality_scale_limit = None
    if asymmetry_rel is not None and asymmetry_rel > 0.0:
        locality_scale_limit = (
            config.locality_safety_factor * config.max_fd_asymmetry_rel / asymmetry_rel
        )
    elif asymmetry_rel == 0.0:
        locality_scale_limit = float("inf")

    usable_bracket_scale = 1.0
    if locality_scale_limit is not None:
        usable_bracket_scale = max(1.0, min(config.max_checked_bracket_scale, locality_scale_limit))
    elif uncertainty_required_scale is not None:
        usable_bracket_scale = min(config.max_checked_bracket_scale, uncertainty_required_scale)

    bracket_only_feasible = (
        bool(response_ok)
        and bool(locality_ok)
        and uncertainty_required_scale is not None
        and locality_scale_limit is not None
        and uncertainty_required_scale <= min(config.max_checked_bracket_scale, locality_scale_limit)
    )
    required_replicates_no_bracket = _required_replicates_for_scaled_bracket(
        current_count=current_replicates,
        uncertainty_rel=uncertainty_rel,
        bracket_scale=1.0,
        config=config,
    )
    required_replicates_at_local_limit = _required_replicates_for_scaled_bracket(
        current_count=current_replicates,
        uncertainty_rel=uncertainty_rel,
        bracket_scale=usable_bracket_scale,
        config=config,
    )
    extra_replicates_at_local_limit = None
    if required_replicates_at_local_limit is not None and current_replicates is not None:
        extra_replicates_at_local_limit = max(0, required_replicates_at_local_limit - current_replicates)

    if passed:
        action = "freeze_promoted_candidate"
        recommendation = "candidate already passes production gates; freeze provenance"
    elif not response_ok:
        action = "increase_checked_bracket_or_replace_control"
        recommendation = (
            "the finite-difference response is below the resolved-response gate; "
            "run a checked bracket sweep or replace this control before adding replicas"
        )
    elif not locality_ok:
        action = "shrink_or_replace_nonlocal_control"
        recommendation = (
            "the finite-difference bracket is nonlocal; shrink the bracket or "
            "replace the control before spending replicas"
        )
    elif uncertainty_ok:
        action = "inspect_pass_flag"
        recommendation = "scalar gates pass but the artifact did not promote; inspect metadata and provenance"
    elif bracket_only_feasible:
        action = "run_checked_larger_bracket"
        recommendation = (
            "a bounded larger bracket can in principle resolve uncertainty while "
            "staying below the locality limit; run a short locality/response sweep first"
        )
    elif extra_replicates_at_local_limit is not None and extra_replicates_at_local_limit <= config.max_extra_replicates_per_state:
        action = "add_limited_replicates_with_locality_cap"
        recommendation = (
            "combine the largest locality-safe bracket with a bounded number of matched replicas"
        )
    else:
        action = "design_better_conditioned_control_or_variance_reduction"
        recommendation = (
            "bracket-only and bounded-replica fixes are not efficient; design a better-conditioned "
            "composite direction, variance-reduced observable, or checked response-larger bracket"
        )

    return {
        "index": index,
        "label": str(label or artifact.get("parameter_name") or path or index),
        "path": path,
        "parameter_name": str(artifact.get("parameter_name") or ""),
        "passed": passed,
        "action": action,
        "recommendation": recommendation,
        "metrics": {
            "response_fraction": _json_number(response_fraction),
            "fd_asymmetry_rel": _json_number(asymmetry_rel),
            "gradient_uncertainty_rel": _json_number(uncertainty_rel),
        },
        "gate_status": {
            "response_ok": bool(response_ok),
            "locality_ok": bool(locality_ok),
            "uncertainty_ok": bool(uncertainty_ok),
        },
        "current_replicates_per_state": current_replicates,
        "uncertainty_required_bracket_scale": _json_number(uncertainty_required_scale),
        "locality_safe_bracket_scale_limit": _json_number(locality_scale_limit),
        "usable_bracket_scale_for_estimate": _json_number(usable_bracket_scale),
        "bracket_only_feasible": bool(bracket_only_feasible),
        "estimated_required_replicates_no_bracket": required_replicates_no_bracket,
        "estimated_required_replicates_at_locality_limit": required_replicates_at_local_limit,
        "estimated_extra_replicates_at_locality_limit": extra_replicates_at_local_limit,
    }


def nonlinear_gradient_candidate_design_report(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    paths: Sequence[str | None] | None = None,
    labels: Sequence[str | None] | None = None,
    case: str = "nonlinear_turbulence_gradient_candidate_design",
    config: NonlinearGradientCandidateDesignConfig | None = None,
) -> dict[str, Any]:
    """Return the next-campaign design implied by failed production gates.

    The report estimates whether a failed central-FD candidate can be rescued by
    a larger bracket, by bounded extra replicas, or whether the next campaign
    should instead change the control/observable.  The estimates use the usual
    ``1/sqrt(N)`` SEM scaling and the local finite-difference assumption that
    response grows approximately linearly with bracket size before the asymmetry
    gate is hit.
    """

    cfg = config or NonlinearGradientCandidateDesignConfig()
    if cfg.max_gradient_uncertainty_rel <= 0.0:
        raise ValueError("max_gradient_uncertainty_rel must be positive")
    if cfg.max_fd_asymmetry_rel <= 0.0:
        raise ValueError("max_fd_asymmetry_rel must be positive")
    if cfg.min_fd_response_fraction <= 0.0:
        raise ValueError("min_fd_response_fraction must be positive")
    if cfg.sem_safety_factor <= 0.0:
        raise ValueError("sem_safety_factor must be positive")
    if cfg.max_extra_replicates_per_state < 0:
        raise ValueError("max_extra_replicates_per_state must be non-negative")
    if cfg.max_checked_bracket_scale < 1.0:
        raise ValueError("max_checked_bracket_scale must be at least one")
    if cfg.locality_safety_factor <= 0.0:
        raise ValueError("locality_safety_factor must be positive")

    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")

    candidates = [
        _design_row(artifact, index=index, path=path, label=label, config=cfg)
        for index, (artifact, path, label) in enumerate(zip(artifacts, path_list, label_list))
    ]
    promoted = [row for row in candidates if row["action"] == "freeze_promoted_candidate"]
    bracket_ready = [row for row in candidates if row["action"] == "run_checked_larger_bracket"]
    replica_ready = [row for row in candidates if row["action"] == "add_limited_replicates_with_locality_cap"]
    replacement = [
        row
        for row in candidates
        if row["action"] in {
            "design_better_conditioned_control_or_variance_reduction",
            "increase_checked_bracket_or_replace_control",
            "shrink_or_replace_nonlocal_control",
        }
    ]

    if promoted:
        next_action = "freeze promoted candidate provenance"
    elif bracket_ready:
        next_action = "run a bounded bracket/locality sweep before new long windows"
    elif replica_ready:
        next_action = "combine locality-capped bracket scale with bounded matched replicas"
    elif replacement:
        next_action = "design a better-conditioned control or variance-reduced observable before more GPU replicas"
    else:
        next_action = "inspect candidate metadata before designing new runs"

    return {
        "kind": "nonlinear_turbulence_gradient_candidate_design_report",
        "claim_level": "campaign_design_not_gradient_evidence",
        "case": case,
        "passed": bool(promoted),
        "next_action": next_action,
        "config": asdict(cfg),
        "summary": {
            "candidate_count": len(candidates),
            "promoted_candidate_count": len(promoted),
            "bracket_ready_count": len(bracket_ready),
            "replica_ready_count": len(replica_ready),
            "replacement_or_variance_reduction_count": len(replacement),
        },
        "candidates": candidates,
    }


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


def nonlinear_gradient_followup_plan(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    paths: Sequence[str | None] | None = None,
    labels: Sequence[str | None] | None = None,
    case: str = "nonlinear_turbulence_gradient_followup",
    config: NonlinearGradientFollowupConfig | None = None,
) -> dict[str, Any]:
    """Build a bounded, fail-closed follow-up plan from gradient artifacts."""

    cfg = config or NonlinearGradientFollowupConfig()
    if cfg.max_extra_replicates_per_state < 0:
        raise ValueError("max_extra_replicates_per_state must be non-negative")
    if cfg.sem_safety_factor <= 0.0:
        raise ValueError("sem_safety_factor must be positive")
    if cfg.max_gradient_uncertainty_rel <= 0.0:
        raise ValueError("max_gradient_uncertainty_rel must be positive")

    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")

    candidate_actions: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    for index, (artifact, path, label) in enumerate(zip(artifacts, path_list, label_list)):
        source_ensembles_raw = artifact.get("source_ensembles")
        source_ensembles = (
            source_ensembles_raw if isinstance(source_ensembles_raw, Mapping) else {}
        )
        response_fraction = _metric(artifact, "response_fraction")
        asymmetry_rel = _metric(artifact, "fd_asymmetry_rel", "asymmetry_rel")
        uncertainty_rel = _metric(
            artifact,
            "gradient_uncertainty_rel",
            "gradient_relative_uncertainty",
        )
        current_replicates = _replicate_count(source_ensembles)
        passed = _artifact_passed(artifact)
        response_ok = (
            response_fraction is not None
            and response_fraction >= float(cfg.min_fd_response_fraction)
        )
        locality_ok = (
            asymmetry_rel is not None
            and asymmetry_rel <= float(cfg.max_fd_asymmetry_rel)
        )
        uncertainty_ok = (
            uncertainty_rel is not None
            and uncertainty_rel <= float(cfg.max_gradient_uncertainty_rel)
        )
        action = "inspect_artifact"
        recommendation = "missing required metrics; inspect the artifact before spending GPU time"
        required_replicates = None
        extra_replicates = 0
        planned_runs: list[dict[str, Any]] = []

        if passed:
            action = "freeze_promoted_candidate"
            recommendation = "candidate already passes production gates; freeze provenance"
        elif not response_ok:
            action = "replace_control_or_increase_checked_bracket"
            recommendation = (
                "finite-difference response is not resolved; change the control or "
                "perform a checked bracket sweep before adding replicas"
            )
        elif not locality_ok:
            action = "shrink_bracket_or_replace_control"
            recommendation = (
                "finite-difference bracket is nonlocal; shrink the perturbation or "
                "replace the control before adding replicas"
            )
        elif not uncertainty_ok and uncertainty_rel is not None:
            required_replicates = _required_replicates(
                current_count=current_replicates,
                uncertainty_rel=uncertainty_rel,
                config=cfg,
            )
            if required_replicates is None:
                action = "recover_replicate_metadata"
                recommendation = (
                    "uncertainty is marginal but the artifact lacks replicate counts; "
                    "recover ensemble metadata before launching runs"
                )
            else:
                extra_replicates = max(0, required_replicates - int(current_replicates or 0))
                extra_replicates = min(extra_replicates, int(cfg.max_extra_replicates_per_state))
                action = "add_matched_nominal_seed_replicates"
                recommendation = (
                    "response and locality pass, but uncertainty is marginal; add only "
                    "the matched independent replicas needed by the 1/sqrt(N) estimate"
                )
                planned_runs = _planned_matched_runs(
                    source_ensembles=source_ensembles,
                    extra_replicates_per_state=extra_replicates,
                    config=cfg,
                )
        else:
            action = "no_followup_needed"
            recommendation = "all scalar gates are satisfied, but artifact was not marked passed"

        all_runs.extend(
            {
                **run,
                "candidate_index": index,
                "candidate_label": str(label or artifact.get("parameter_name") or path or index),
            }
            for run in planned_runs
        )
        candidate_actions.append(
            {
                "index": index,
                "label": str(label or artifact.get("parameter_name") or path or index),
                "path": path,
                "parameter_name": str(artifact.get("parameter_name") or ""),
                "passed": passed,
                "action": action,
                "recommendation": recommendation,
                "metrics": {
                    "response_fraction": _json_number(response_fraction),
                    "fd_asymmetry_rel": _json_number(asymmetry_rel),
                    "gradient_uncertainty_rel": _json_number(uncertainty_rel),
                },
                "current_replicates_per_state": current_replicates,
                "estimated_required_replicates_per_state": required_replicates,
                "extra_replicates_per_state": extra_replicates,
                "planned_run_count": len(planned_runs),
                "planned_runs": planned_runs,
            }
        )

    promoted = [row for row in candidate_actions if row["action"] == "freeze_promoted_candidate"]
    runnable = [row for row in candidate_actions if row["planned_run_count"]]
    nonlocal_rows = [row for row in candidate_actions if row["action"] == "shrink_bracket_or_replace_control"]
    unresolved_rows = [
        row
        for row in candidate_actions
        if row["action"] == "replace_control_or_increase_checked_bracket"
    ]
    if promoted:
        next_action = "freeze the promoted candidate and do not launch more follow-up runs"
    elif runnable:
        next_action = "launch the bounded matched-replicate follow-up for the listed local noisy candidate"
    elif nonlocal_rows:
        next_action = "run a smaller-bracket or replacement-control sweep before adding replicas"
    elif unresolved_rows:
        next_action = "choose controls with a resolved heat-flux response before adding replicas"
    else:
        next_action = "inspect artifacts; no safe production follow-up was inferred"

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
            "planned_run_count": len(all_runs),
        },
        "candidate_actions": candidate_actions,
        "planned_runs": all_runs,
    }
