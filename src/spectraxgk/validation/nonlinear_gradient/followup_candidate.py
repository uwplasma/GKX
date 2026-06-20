"""Candidate-design reports for nonlinear turbulence-gradient follow-up campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class _CandidateMetrics:
    response_fraction: float | None
    asymmetry_rel: float | None
    uncertainty_rel: float | None
    current_replicates: int | None
    variance_report: dict[str, Any]
    passed: bool


@dataclass(frozen=True)
class _CandidateGateStatus:
    response_ok: bool
    locality_ok: bool
    uncertainty_ok: bool


@dataclass(frozen=True)
class _CandidateBracketEstimates:
    uncertainty_required_scale: float | None
    locality_scale_limit: float | None
    usable_bracket_scale: float
    bracket_only_feasible: bool
    required_replicates_no_bracket: int | None
    required_replicates_at_local_limit: int | None
    extra_replicates_at_local_limit: int | None


@dataclass(frozen=True)
class _CandidateDecision:
    action: str
    recommendation: str


@dataclass(frozen=True)
class _CandidateGroups:
    promoted: list[Mapping[str, Any]]
    bracket_ready: list[Mapping[str, Any]]
    replica_ready: list[Mapping[str, Any]]
    variance_limited: list[Mapping[str, Any]]
    replacement: list[Mapping[str, Any]]


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


def _candidate_metrics(
    artifact: Mapping[str, Any],
    *,
    config: NonlinearGradientCandidateDesignConfig,
) -> _CandidateMetrics:
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
    return _CandidateMetrics(
        response_fraction=response_fraction,
        asymmetry_rel=asymmetry_rel,
        uncertainty_rel=uncertainty_rel,
        current_replicates=_replicate_count(source_ensembles),
        variance_report=variance_report,
        passed=_artifact_passed(artifact),
    )


def _candidate_gate_status(
    metrics: _CandidateMetrics,
    *,
    config: NonlinearGradientCandidateDesignConfig,
) -> _CandidateGateStatus:
    return _CandidateGateStatus(
        response_ok=(
            metrics.response_fraction is not None
            and metrics.response_fraction >= config.min_fd_response_fraction
        ),
        locality_ok=(
            metrics.asymmetry_rel is not None
            and metrics.asymmetry_rel <= config.max_fd_asymmetry_rel
        ),
        uncertainty_ok=(
            metrics.uncertainty_rel is not None
            and metrics.uncertainty_rel <= config.max_gradient_uncertainty_rel
        ),
    )


def _uncertainty_required_scale(
    uncertainty_rel: float | None,
    *,
    config: NonlinearGradientCandidateDesignConfig,
) -> float | None:
    if uncertainty_rel is None:
        return None
    return max(
        1.0,
        uncertainty_rel / max(config.max_gradient_uncertainty_rel, config.value_floor),
    )


def _locality_scale_limit(
    asymmetry_rel: float | None,
    *,
    config: NonlinearGradientCandidateDesignConfig,
) -> float | None:
    if asymmetry_rel is not None and asymmetry_rel > 0.0:
        return (
            config.locality_safety_factor * config.max_fd_asymmetry_rel / asymmetry_rel
        )
    if asymmetry_rel == 0.0:
        return float("inf")
    return None


def _usable_bracket_scale(
    *,
    locality_scale_limit: float | None,
    uncertainty_required_scale: float | None,
    config: NonlinearGradientCandidateDesignConfig,
) -> float:
    usable_bracket_scale = 1.0
    if locality_scale_limit is not None:
        usable_bracket_scale = max(
            1.0, min(config.max_checked_bracket_scale, locality_scale_limit)
        )
    elif uncertainty_required_scale is not None:
        usable_bracket_scale = min(
            config.max_checked_bracket_scale, uncertainty_required_scale
        )
    return usable_bracket_scale


def _required_replicate_estimates(
    metrics: _CandidateMetrics,
    *,
    usable_bracket_scale: float,
    config: NonlinearGradientCandidateDesignConfig,
) -> tuple[int | None, int | None, int | None]:
    no_bracket = _required_replicates_for_scaled_bracket(
        current_count=metrics.current_replicates,
        uncertainty_rel=metrics.uncertainty_rel,
        bracket_scale=1.0,
        config=config,
    )
    at_local_limit = _required_replicates_for_scaled_bracket(
        current_count=metrics.current_replicates,
        uncertainty_rel=metrics.uncertainty_rel,
        bracket_scale=usable_bracket_scale,
        config=config,
    )
    extra = None
    if at_local_limit is not None and metrics.current_replicates is not None:
        extra = max(0, at_local_limit - metrics.current_replicates)
    return no_bracket, at_local_limit, extra


def _candidate_bracket_estimates(
    metrics: _CandidateMetrics,
    gates: _CandidateGateStatus,
    *,
    config: NonlinearGradientCandidateDesignConfig,
) -> _CandidateBracketEstimates:
    uncertainty_required_scale = _uncertainty_required_scale(
        metrics.uncertainty_rel, config=config
    )
    locality_scale_limit = _locality_scale_limit(
        metrics.asymmetry_rel, config=config
    )
    usable_bracket_scale = _usable_bracket_scale(
        locality_scale_limit=locality_scale_limit,
        uncertainty_required_scale=uncertainty_required_scale,
        config=config,
    )
    bracket_only_feasible = (
        bool(gates.response_ok)
        and bool(gates.locality_ok)
        and uncertainty_required_scale is not None
        and locality_scale_limit is not None
        and uncertainty_required_scale
        <= min(config.max_checked_bracket_scale, locality_scale_limit)
    )
    no_bracket, at_local_limit, extra = _required_replicate_estimates(
        metrics, usable_bracket_scale=usable_bracket_scale, config=config
    )
    return _CandidateBracketEstimates(
        uncertainty_required_scale=uncertainty_required_scale,
        locality_scale_limit=locality_scale_limit,
        usable_bracket_scale=usable_bracket_scale,
        bracket_only_feasible=bracket_only_feasible,
        required_replicates_no_bracket=no_bracket,
        required_replicates_at_local_limit=at_local_limit,
        extra_replicates_at_local_limit=extra,
    )


def _limited_replicates_are_feasible(
    estimates: _CandidateBracketEstimates,
    *,
    config: NonlinearGradientCandidateDesignConfig,
) -> bool:
    return (
        estimates.extra_replicates_at_local_limit is not None
        and estimates.extra_replicates_at_local_limit
        <= config.max_extra_replicates_per_state
    )


def _candidate_decision(
    metrics: _CandidateMetrics,
    gates: _CandidateGateStatus,
    estimates: _CandidateBracketEstimates,
    *,
    config: NonlinearGradientCandidateDesignConfig,
) -> _CandidateDecision:
    if metrics.passed:
        return _CandidateDecision(
            "freeze_promoted_candidate",
            "candidate already passes production gates; freeze provenance",
        )
    if not gates.response_ok:
        return _CandidateDecision(
            "increase_checked_bracket_or_replace_control",
            "the finite-difference response is below the resolved-response gate; "
            "run a checked bracket sweep or replace this control before adding replicas",
        )
    if not gates.locality_ok:
        return _CandidateDecision(
            "shrink_or_replace_nonlocal_control",
            "the finite-difference bracket is nonlocal; shrink the bracket or "
            "replace the control before spending replicas",
        )
    if gates.uncertainty_ok:
        return _CandidateDecision(
            "inspect_pass_flag",
            "scalar gates pass but the artifact did not promote; inspect metadata and provenance",
        )
    if metrics.variance_report["failed_spread_states"]:
        return _CandidateDecision(
            "design_variance_reduction_for_limiting_state",
            str(metrics.variance_report["recommendation"]),
        )
    if estimates.bracket_only_feasible:
        return _CandidateDecision(
            "run_checked_larger_bracket",
            "a bounded larger bracket can in principle resolve uncertainty while "
            "staying below the locality limit; run a short locality/response sweep first",
        )
    if _limited_replicates_are_feasible(estimates, config=config):
        return _CandidateDecision(
            "add_limited_replicates_with_locality_cap",
            "combine the largest locality-safe bracket with a bounded number of matched replicas",
    )
    return _CandidateDecision(
        "design_better_conditioned_control_or_variance_reduction",
        "bracket-only and bounded-replica fixes are not efficient; design a better-conditioned "
        "composite direction, variance-reduced observable, or checked response-larger bracket",
    )


def _candidate_label(
    artifact: Mapping[str, Any],
    *,
    index: int,
    path: str | None,
    label: str | None,
) -> str:
    return str(label or artifact.get("parameter_name") or path or index)


def _candidate_metric_payload(metrics: _CandidateMetrics) -> dict[str, Any]:
    return {
        "response_fraction": _json_number(metrics.response_fraction),
        "fd_asymmetry_rel": _json_number(metrics.asymmetry_rel),
        "gradient_uncertainty_rel": _json_number(metrics.uncertainty_rel),
    }


def _candidate_gate_payload(gates: _CandidateGateStatus) -> dict[str, bool]:
    return {
        "response_ok": bool(gates.response_ok),
        "locality_ok": bool(gates.locality_ok),
        "uncertainty_ok": bool(gates.uncertainty_ok),
    }


def _candidate_estimate_payload(
    estimates: _CandidateBracketEstimates,
) -> dict[str, Any]:
    return {
        "uncertainty_required_bracket_scale": _json_number(
            estimates.uncertainty_required_scale
        ),
        "locality_safe_bracket_scale_limit": _json_number(
            estimates.locality_scale_limit
        ),
        "usable_bracket_scale_for_estimate": _json_number(
            estimates.usable_bracket_scale
        ),
        "bracket_only_feasible": bool(estimates.bracket_only_feasible),
        "estimated_required_replicates_no_bracket": (
            estimates.required_replicates_no_bracket
        ),
        "estimated_required_replicates_at_locality_limit": (
            estimates.required_replicates_at_local_limit
        ),
        "estimated_extra_replicates_at_locality_limit": (
            estimates.extra_replicates_at_local_limit
        ),
    }


def _candidate_payload(
    artifact: Mapping[str, Any],
    *,
    index: int,
    path: str | None,
    label: str | None,
    metrics: _CandidateMetrics,
    gates: _CandidateGateStatus,
    estimates: _CandidateBracketEstimates,
    decision: _CandidateDecision,
) -> dict[str, Any]:
    return {
        "index": index,
        "label": _candidate_label(artifact, index=index, path=path, label=label),
        "path": path,
        "parameter_name": str(artifact.get("parameter_name") or ""),
        "passed": metrics.passed,
        "action": decision.action,
        "recommendation": decision.recommendation,
        "metrics": _candidate_metric_payload(metrics),
        "gate_status": _candidate_gate_payload(gates),
        "variance_reduction": metrics.variance_report,
        "current_replicates_per_state": metrics.current_replicates,
        **_candidate_estimate_payload(estimates),
    }


def _design_row(
    artifact: Mapping[str, Any],
    *,
    index: int,
    path: str | None,
    label: str | None,
    config: NonlinearGradientCandidateDesignConfig,
) -> dict[str, Any]:
    metrics = _candidate_metrics(artifact, config=config)
    gates = _candidate_gate_status(metrics, config=config)
    estimates = _candidate_bracket_estimates(metrics, gates, config=config)
    decision = _candidate_decision(metrics, gates, estimates, config=config)
    return _candidate_payload(
        artifact,
        index=index,
        path=path,
        label=label,
        metrics=metrics,
        gates=gates,
        estimates=estimates,
        decision=decision,
    )


def _validated_candidate_design_config(
    config: NonlinearGradientCandidateDesignConfig | None,
) -> NonlinearGradientCandidateDesignConfig:
    cfg = config or NonlinearGradientCandidateDesignConfig()
    positive_checks = (
        ("max_gradient_uncertainty_rel", cfg.max_gradient_uncertainty_rel),
        ("max_fd_asymmetry_rel", cfg.max_fd_asymmetry_rel),
        ("max_window_mean_rel_spread", cfg.max_window_mean_rel_spread),
        ("max_window_sem_rel", cfg.max_window_sem_rel),
        ("min_fd_response_fraction", cfg.min_fd_response_fraction),
        ("sem_safety_factor", cfg.sem_safety_factor),
        ("locality_safety_factor", cfg.locality_safety_factor),
    )
    for name, value in positive_checks:
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if cfg.max_extra_replicates_per_state < 0:
        raise ValueError("max_extra_replicates_per_state must be non-negative")
    if cfg.max_checked_bracket_scale < 1.0:
        raise ValueError("max_checked_bracket_scale must be at least one")
    return cfg


def _metadata_lists(
    *,
    artifacts: Sequence[Mapping[str, Any]],
    paths: Sequence[str | None] | None,
    labels: Sequence[str | None] | None,
) -> tuple[list[str | None], list[str | None]]:
    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")
    return path_list, label_list


def _candidate_rows(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    path_list: Sequence[str | None],
    label_list: Sequence[str | None],
    config: NonlinearGradientCandidateDesignConfig,
) -> list[dict[str, Any]]:
    return [
        _design_row(artifact, index=index, path=path, label=label, config=config)
        for index, (artifact, path, label) in enumerate(
            zip(artifacts, path_list, label_list)
        )
    ]


def _rows_with_action(
    candidates: Sequence[Mapping[str, Any]],
    action: str,
) -> list[Mapping[str, Any]]:
    return [row for row in candidates if row["action"] == action]


def _replacement_or_variance_rows(
    candidates: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    replacement_actions = {
        "design_better_conditioned_control_or_variance_reduction",
        "design_variance_reduction_for_limiting_state",
        "increase_checked_bracket_or_replace_control",
        "shrink_or_replace_nonlocal_control",
    }
    return [row for row in candidates if row["action"] in replacement_actions]


def _candidate_groups(candidates: Sequence[Mapping[str, Any]]) -> _CandidateGroups:
    return _CandidateGroups(
        promoted=_rows_with_action(candidates, "freeze_promoted_candidate"),
        bracket_ready=_rows_with_action(candidates, "run_checked_larger_bracket"),
        replica_ready=_rows_with_action(
            candidates, "add_limited_replicates_with_locality_cap"
        ),
        variance_limited=_rows_with_action(
            candidates, "design_variance_reduction_for_limiting_state"
        ),
        replacement=_replacement_or_variance_rows(candidates),
    )


def _next_action_from_groups(
    groups: _CandidateGroups,
) -> str:
    if groups.promoted:
        return "freeze promoted candidate provenance"
    if groups.bracket_ready:
        return "run a bounded bracket/locality sweep before new long windows"
    if groups.variance_limited:
        return (
            "target paired-seed or control-variate variance reduction for limiting states"
        )
    if groups.replica_ready:
        return "combine locality-capped bracket scale with bounded matched replicas"
    if groups.replacement:
        return "design a better-conditioned control or variance-reduced observable before more GPU replicas"
    return "inspect candidate metadata before designing new runs"


def _summary_from_groups(
    *,
    candidates: Sequence[Mapping[str, Any]],
    groups: _CandidateGroups,
) -> dict[str, int]:
    return {
        "candidate_count": len(candidates),
        "promoted_candidate_count": len(groups.promoted),
        "bracket_ready_count": len(groups.bracket_ready),
        "replica_ready_count": len(groups.replica_ready),
        "variance_limited_count": len(groups.variance_limited),
        "replacement_or_variance_reduction_count": len(groups.replacement),
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

    cfg = _validated_candidate_design_config(config)
    path_list, label_list = _metadata_lists(
        artifacts=artifacts, paths=paths, labels=labels
    )
    candidates = _candidate_rows(
        artifacts, path_list=path_list, label_list=label_list, config=cfg
    )
    groups = _candidate_groups(candidates)

    return {
        "kind": "nonlinear_turbulence_gradient_candidate_design_report",
        "claim_level": "campaign_design_not_gradient_evidence",
        "case": case,
        "passed": bool(groups.promoted),
        "next_action": _next_action_from_groups(groups),
        "config": asdict(cfg),
        "summary": _summary_from_groups(candidates=candidates, groups=groups),
        "candidates": candidates,
    }


__all__ = [
    "_design_row",
    "_required_replicates_for_scaled_bracket",
    "nonlinear_gradient_candidate_design_report",
]
