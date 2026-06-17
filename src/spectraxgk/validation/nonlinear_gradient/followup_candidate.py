"""Candidate-design reports for nonlinear turbulence-gradient follow-up campaigns."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence
import math

from spectraxgk.validation.nonlinear_gradient.followup_core import (
    NonlinearGradientCandidateDesignConfig,
    _artifact_passed,
    _ensemble_state_variance_report,
    _json_number,
    _metric,
    _replicate_count,
)


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
    source_ensembles = (
        source_ensembles_raw if isinstance(source_ensembles_raw, Mapping) else {}
    )
    variance_report = _ensemble_state_variance_report(source_ensembles, config=config)
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
        and response_fraction >= config.min_fd_response_fraction
    )
    locality_ok = (
        asymmetry_rel is not None and asymmetry_rel <= config.max_fd_asymmetry_rel
    )
    uncertainty_ok = (
        uncertainty_rel is not None
        and uncertainty_rel <= config.max_gradient_uncertainty_rel
    )

    uncertainty_required_scale = None
    if uncertainty_rel is not None:
        uncertainty_required_scale = max(
            1.0,
            uncertainty_rel
            / max(config.max_gradient_uncertainty_rel, config.value_floor),
        )

    locality_scale_limit = None
    if asymmetry_rel is not None and asymmetry_rel > 0.0:
        locality_scale_limit = (
            config.locality_safety_factor * config.max_fd_asymmetry_rel / asymmetry_rel
        )
    elif asymmetry_rel == 0.0:
        locality_scale_limit = float("inf")

    usable_bracket_scale = 1.0
    if locality_scale_limit is not None:
        usable_bracket_scale = max(
            1.0, min(config.max_checked_bracket_scale, locality_scale_limit)
        )
    elif uncertainty_required_scale is not None:
        usable_bracket_scale = min(
            config.max_checked_bracket_scale, uncertainty_required_scale
        )

    bracket_only_feasible = (
        bool(response_ok)
        and bool(locality_ok)
        and uncertainty_required_scale is not None
        and locality_scale_limit is not None
        and uncertainty_required_scale
        <= min(config.max_checked_bracket_scale, locality_scale_limit)
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
    if (
        required_replicates_at_local_limit is not None
        and current_replicates is not None
    ):
        extra_replicates_at_local_limit = max(
            0, required_replicates_at_local_limit - current_replicates
        )

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
    elif variance_report["failed_spread_states"]:
        action = "design_variance_reduction_for_limiting_state"
        recommendation = str(variance_report["recommendation"])
    elif bracket_only_feasible:
        action = "run_checked_larger_bracket"
        recommendation = (
            "a bounded larger bracket can in principle resolve uncertainty while "
            "staying below the locality limit; run a short locality/response sweep first"
        )
    elif (
        extra_replicates_at_local_limit is not None
        and extra_replicates_at_local_limit <= config.max_extra_replicates_per_state
    ):
        action = "add_limited_replicates_with_locality_cap"
        recommendation = "combine the largest locality-safe bracket with a bounded number of matched replicas"
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
        "variance_reduction": variance_report,
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
    if cfg.max_window_mean_rel_spread <= 0.0:
        raise ValueError("max_window_mean_rel_spread must be positive")
    if cfg.max_window_sem_rel <= 0.0:
        raise ValueError("max_window_sem_rel must be positive")
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
        for index, (artifact, path, label) in enumerate(
            zip(artifacts, path_list, label_list)
        )
    ]
    promoted = [
        row for row in candidates if row["action"] == "freeze_promoted_candidate"
    ]
    bracket_ready = [
        row for row in candidates if row["action"] == "run_checked_larger_bracket"
    ]
    replica_ready = [
        row
        for row in candidates
        if row["action"] == "add_limited_replicates_with_locality_cap"
    ]
    variance_limited = [
        row
        for row in candidates
        if row["action"] == "design_variance_reduction_for_limiting_state"
    ]
    replacement = [
        row
        for row in candidates
        if row["action"]
        in {
            "design_better_conditioned_control_or_variance_reduction",
            "design_variance_reduction_for_limiting_state",
            "increase_checked_bracket_or_replace_control",
            "shrink_or_replace_nonlocal_control",
        }
    ]

    if promoted:
        next_action = "freeze promoted candidate provenance"
    elif bracket_ready:
        next_action = "run a bounded bracket/locality sweep before new long windows"
    elif variance_limited:
        next_action = "target paired-seed or control-variate variance reduction for limiting states"
    elif replica_ready:
        next_action = (
            "combine locality-capped bracket scale with bounded matched replicas"
        )
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
            "variance_limited_count": len(variance_limited),
            "replacement_or_variance_reduction_count": len(replacement),
        },
        "candidates": candidates,
    }


__all__ = [
    "_design_row",
    "_required_replicates_for_scaled_bracket",
    "nonlinear_gradient_candidate_design_report",
]
