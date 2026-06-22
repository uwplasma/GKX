"""Variance-reduction follow-up reports for nonlinear-gradient audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
import math

from spectraxgk.validation.nonlinear_gradient.followup_core import (
    NonlinearGradientCandidateDesignConfig,
    NonlinearGradientControlMeanGateConfig,
    NonlinearGradientControlVariateCampaignConfig,
    NonlinearGradientVarianceReductionConfig,
    _control_variate_candidate,
    _ensemble_state_variance_report,
    _finite_float,
    _finite_int,
    _json_number,
    _mean_and_sem,
    _state_means_by_label,
)


def _control_variate_candidate_sort_key(row: Mapping[str, Any]) -> tuple[float, float]:
    uncertainty = _finite_float(row.get("adjusted_response_uncertainty_rel"))
    reduction = _finite_float(row.get("sem_reduction_fraction"))
    return (
        uncertainty if uncertainty is not None else float("inf"),
        -(reduction if reduction is not None else float("-inf")),
    )


def _control_variate_candidates(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates_raw = report.get("control_variate_candidates")
    if not isinstance(candidates_raw, Sequence):
        return []
    return [row for row in candidates_raw if isinstance(row, Mapping)]


def _select_control_variate_candidate(
    report: Mapping[str, Any],
    candidate_name: str | None,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], Mapping[str, Any] | None]:
    summary_raw = report.get("summary")
    summary = summary_raw if isinstance(summary_raw, Mapping) else {}
    requested = candidate_name or str(summary.get("best_control_variate") or "")
    candidates = _control_variate_candidates(report)
    candidate = (
        next((row for row in candidates if str(row.get("name")) == requested), None)
        if requested
        else None
    )
    if candidate is None and candidates:
        candidate = min(candidates, key=_control_variate_candidate_sort_key)
    return summary, candidates, candidate


def _validate_variance_reduction_config(
    cfg: NonlinearGradientVarianceReductionConfig,
) -> None:
    if cfg.max_paired_response_uncertainty_rel <= 0.0:
        raise ValueError("max_paired_response_uncertainty_rel must be positive")
    if cfg.max_control_variate_uncertainty_rel <= 0.0:
        raise ValueError("max_control_variate_uncertainty_rel must be positive")
    if cfg.min_control_variate_sem_reduction < 0.0:
        raise ValueError("min_control_variate_sem_reduction must be non-negative")
    if cfg.sem_safety_factor <= 0.0:
        raise ValueError("sem_safety_factor must be positive")
    if cfg.min_common_pairs < 1:
        raise ValueError("min_common_pairs must be positive")
    if cfg.max_extra_paired_seeds < 0:
        raise ValueError("max_extra_paired_seeds must be non-negative")


def _source_ensemble_mapping(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    source_ensembles_raw = artifact.get("source_ensembles")
    return source_ensembles_raw if isinstance(source_ensembles_raw, Mapping) else {}


def _paired_variance_rows(
    plus: Mapping[str, float],
    minus: Mapping[str, float],
    baseline: Mapping[str, float],
) -> tuple[list[str], list[str], list[dict[str, Any]], list[float]]:
    common_labels = sorted(set(plus).intersection(minus))
    common_with_baseline = sorted(set(common_labels).intersection(baseline))
    pair_rows: list[dict[str, Any]] = []
    paired_differences: list[float] = []
    for item in common_labels:
        diff = plus[item] - minus[item]
        paired_differences.append(diff)
        row: dict[str, Any] = {
            "label": item,
            "plus_mean": _json_number(plus[item]),
            "minus_mean": _json_number(minus[item]),
            "plus_minus_difference": _json_number(diff),
        }
        if item in baseline:
            row["baseline_mean"] = _json_number(baseline[item])
            row["plus_baseline_difference"] = _json_number(plus[item] - baseline[item])
            row["baseline_minus_difference"] = _json_number(baseline[item] - minus[item])
        pair_rows.append(row)
    return common_labels, common_with_baseline, pair_rows, paired_differences


def _paired_uncertainty_rel(
    paired_mean: float | None,
    paired_sem: float | None,
    *,
    value_floor: float,
) -> float | None:
    if paired_mean is None or paired_sem is None:
        return None
    return abs(paired_sem) / max(abs(paired_mean), value_floor)


def _variance_control_candidates(
    *,
    common_with_baseline: Sequence[str],
    plus: Mapping[str, float],
    minus: Mapping[str, float],
    baseline: Mapping[str, float],
    paired_mean: float | None,
    paired_sem: float | None,
    cfg: NonlinearGradientVarianceReductionConfig,
) -> list[dict[str, Any]]:
    if not common_with_baseline:
        return []
    response_for_baseline = [plus[item] - minus[item] for item in common_with_baseline]
    baseline_control = [baseline[item] for item in common_with_baseline]
    midpoint_control = [0.5 * (plus[item] + minus[item]) for item in common_with_baseline]
    return [
        _control_variate_candidate(
            name="baseline_transport_common_mode",
            response_samples=response_for_baseline,
            control_samples=baseline_control,
            response_mean=paired_mean,
            raw_sem=paired_sem,
            config=cfg,
        ),
        _control_variate_candidate(
            name="plus_minus_midpoint_common_mode",
            response_samples=response_for_baseline,
            control_samples=midpoint_control,
            response_mean=paired_mean,
            raw_sem=paired_sem,
            config=cfg,
        ),
    ]


def _apparently_useful_control_candidates(
    control_candidates: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in control_candidates
        if "control_variate_uncertainty_above_gate" not in row.get("blockers", [])
        and "control_variate_sem_reduction_too_small" not in row.get("blockers", [])
    ]


def _required_extra_pairs(
    *,
    common_pair_count: int,
    paired_uncertainty_rel: float | None,
    cfg: NonlinearGradientVarianceReductionConfig,
) -> tuple[int | None, int | None]:
    if paired_uncertainty_rel is None:
        return None, None
    scale = (paired_uncertainty_rel / cfg.max_paired_response_uncertainty_rel) ** 2
    scale *= cfg.sem_safety_factor
    required_pairs = max(common_pair_count, int(math.ceil(common_pair_count * scale)))
    return required_pairs, max(0, required_pairs - common_pair_count)


def _variance_followup_action(
    *,
    common_pair_count: int,
    paired_uncertainty_rel: float | None,
    apparent_candidates: Sequence[Mapping[str, Any]],
    control_candidates: Sequence[Mapping[str, Any]],
    extra_pairs: int | None,
    cfg: NonlinearGradientVarianceReductionConfig,
) -> tuple[str, str]:
    if common_pair_count < cfg.min_common_pairs:
        return (
            "recover_or_add_matched_seed_pairs",
            "common plus/minus seed labels are insufficient for paired finite differences",
        )
    if paired_uncertainty_rel is None:
        return (
            "add_matched_seed_pairs",
            "paired response SEM cannot be estimated from fewer than two finite pairs",
        )
    if paired_uncertainty_rel <= cfg.max_paired_response_uncertainty_rel:
        return (
            "use_paired_seed_response_estimator",
            "paired seed response uncertainty is within the target gate",
        )
    if apparent_candidates and cfg.require_known_control_mean:
        return (
            "estimate_control_mean_or_redesign_observable",
            "a common-mode control variate reduces residual scatter, but its expectation is not "
            "independently known; estimate the control mean or redesign the observable before "
            "using it as a production uncertainty reducer",
        )
    if any(row.get("admissible") for row in control_candidates):
        return (
            "use_control_variate_response_estimator",
            "control-variate response uncertainty is within the target gate",
        )
    if extra_pairs is not None and extra_pairs <= cfg.max_extra_paired_seeds:
        return (
            "add_matched_paired_seed_replicates",
            "add bounded matched plus/minus seed pairs before changing the observable",
        )
    return (
        "design_control_variate_or_new_observable",
        "paired seed differences reduce common noise but are still too uncertain; "
        "design a control-variate observable or better-conditioned response before more GPU time",
    )


def _pack_variance_reduction_plan(
    *,
    artifact: Mapping[str, Any],
    path: str | None,
    label: str | None,
    case: str,
    cfg: NonlinearGradientVarianceReductionConfig,
    variance: Mapping[str, Any],
    action: str,
    recommendation: str,
    common_pair_count: int,
    common_with_baseline_count: int,
    paired_mean: float | None,
    paired_sem: float | None,
    paired_uncertainty_rel: float | None,
    required_pairs: int | None,
    extra_pairs: int | None,
    best_control_variate: Mapping[str, Any] | None,
    control_candidates: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "kind": "nonlinear_turbulence_gradient_variance_reduction_plan",
        "claim_level": "campaign_design_not_gradient_evidence",
        "case": case,
        "path": path,
        "label": str(label or artifact.get("parameter_name") or path or case),
        "passed": action
        in {
            "use_paired_seed_response_estimator",
            "use_control_variate_response_estimator",
        },
        "action": action,
        "recommendation": recommendation,
        "config": asdict(cfg),
        "variance_reduction": variance,
        "summary": {
            "common_pair_count": common_pair_count,
            "common_with_baseline_count": common_with_baseline_count,
            "paired_response_mean": _json_number(paired_mean),
            "paired_response_sem": _json_number(paired_sem),
            "paired_response_uncertainty_rel": _json_number(paired_uncertainty_rel),
            "required_pair_count": required_pairs,
            "extra_pair_count": extra_pairs,
            "best_control_variate": (
                None if best_control_variate is None else str(best_control_variate.get("name"))
            ),
        },
        "control_variate_candidates": list(control_candidates),
        "pair_rows": list(pair_rows),
    }


@dataclass(frozen=True)
class _VarianceReductionState:
    variance: Mapping[str, Any]
    common_labels: list[str]
    common_with_baseline: list[str]
    pair_rows: list[dict[str, Any]]
    paired_mean: float | None
    paired_sem: float | None
    paired_uncertainty_rel: float | None
    control_candidates: list[dict[str, Any]]
    best_control_variate: Mapping[str, Any] | None
    required_pairs: int | None
    extra_pairs: int | None


def _variance_reduction_state(
    artifact: Mapping[str, Any],
    cfg: NonlinearGradientVarianceReductionConfig,
) -> _VarianceReductionState:
    source_ensembles = _source_ensemble_mapping(artifact)
    variance = _ensemble_state_variance_report(
        source_ensembles,
        config=NonlinearGradientCandidateDesignConfig(
            max_window_mean_rel_spread=0.15,
            max_window_sem_rel=0.25,
        ),
    )
    plus = _state_means_by_label(source_ensembles, "plus")
    minus = _state_means_by_label(source_ensembles, "minus")
    baseline = _state_means_by_label(source_ensembles, "baseline")
    common_labels, common_with_baseline, pair_rows, paired_differences = (
        _paired_variance_rows(plus, minus, baseline)
    )
    paired_mean, paired_sem = _mean_and_sem(paired_differences)
    paired_uncertainty_rel = _paired_uncertainty_rel(
        paired_mean, paired_sem, value_floor=cfg.value_floor
    )
    control_candidates = _variance_control_candidates(
        common_with_baseline=common_with_baseline,
        plus=plus,
        minus=minus,
        baseline=baseline,
        paired_mean=paired_mean,
        paired_sem=paired_sem,
        cfg=cfg,
    )
    best_control_variate = (
        min(control_candidates, key=_control_variate_candidate_sort_key)
        if control_candidates
        else None
    )
    required_pairs, extra_pairs = _required_extra_pairs(
        common_pair_count=len(common_labels),
        paired_uncertainty_rel=paired_uncertainty_rel,
        cfg=cfg,
    )
    return _VarianceReductionState(
        variance=variance,
        common_labels=common_labels,
        common_with_baseline=common_with_baseline,
        pair_rows=pair_rows,
        paired_mean=paired_mean,
        paired_sem=paired_sem,
        paired_uncertainty_rel=paired_uncertainty_rel,
        control_candidates=control_candidates,
        best_control_variate=best_control_variate,
        required_pairs=required_pairs,
        extra_pairs=extra_pairs,
    )


def _variance_reduction_decision(
    state: _VarianceReductionState,
    cfg: NonlinearGradientVarianceReductionConfig,
) -> tuple[str, str]:
    return _variance_followup_action(
        common_pair_count=len(state.common_labels),
        paired_uncertainty_rel=state.paired_uncertainty_rel,
        apparent_candidates=_apparently_useful_control_candidates(
            state.control_candidates
        ),
        control_candidates=state.control_candidates,
        extra_pairs=state.extra_pairs,
        cfg=cfg,
    )


def nonlinear_gradient_variance_reduction_plan(
    artifact: Mapping[str, Any],
    *,
    path: str | None = None,
    label: str | None = None,
    case: str = "nonlinear_turbulence_gradient_variance_reduction_plan",
    config: NonlinearGradientVarianceReductionConfig | None = None,
) -> dict[str, Any]:
    """Plan paired-seed/control-variate follow-up for a failed central-FD artifact.

    The plan uses common seed/timestep labels across ``plus`` and ``minus``
    ensembles to estimate the uncertainty of paired finite-difference
    responses.  It is a campaign-design artifact, not nonlinear-gradient
    evidence.
    """

    cfg = config or NonlinearGradientVarianceReductionConfig()
    _validate_variance_reduction_config(cfg)
    state = _variance_reduction_state(artifact, cfg)
    action, recommendation = _variance_reduction_decision(state, cfg)
    return _pack_variance_reduction_plan(
        artifact=artifact,
        path=path,
        label=label,
        case=case,
        cfg=cfg,
        variance=state.variance,
        action=action,
        recommendation=recommendation,
        common_pair_count=len(state.common_labels),
        common_with_baseline_count=len(state.common_with_baseline),
        paired_mean=state.paired_mean,
        paired_sem=state.paired_sem,
        paired_uncertainty_rel=state.paired_uncertainty_rel,
        required_pairs=state.required_pairs,
        extra_pairs=state.extra_pairs,
        best_control_variate=state.best_control_variate,
        control_candidates=state.control_candidates,
        pair_rows=state.pair_rows,
    )


def _validate_control_variate_campaign_config(
    cfg: NonlinearGradientControlVariateCampaignConfig,
) -> None:
    if cfg.target_response_uncertainty_rel <= 0.0:
        raise ValueError("target_response_uncertainty_rel must be positive")
    if cfg.sem_safety_factor <= 0.0:
        raise ValueError("sem_safety_factor must be positive")
    if cfg.min_control_mean_pairs < 1:
        raise ValueError("min_control_mean_pairs must be positive")
    if cfg.max_control_mean_pairs < cfg.min_control_mean_pairs:
        raise ValueError("max_control_mean_pairs must be at least min_control_mean_pairs")
    if cfg.first_new_seed < 0:
        raise ValueError("first_new_seed must be non-negative")


def _campaign_candidate_values(
    summary: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    candidate_pair_count = _finite_int(None if candidate is None else candidate.get("n_samples"))
    current_pair_count = _finite_int(
        summary.get("common_with_baseline_count") or summary.get("common_pair_count")
    )
    return {
        "response_mean": _finite_float(summary.get("paired_response_mean")),
        "raw_response_uncertainty_rel": _finite_float(
            summary.get("paired_response_uncertainty_rel")
        ),
        "candidate_name": None if candidate is None else str(candidate.get("name")),
        "beta": _finite_float(None if candidate is None else candidate.get("beta")),
        "residual_sem": _finite_float(
            None if candidate is None else candidate.get("adjusted_response_sem")
        ),
        "residual_uncertainty_rel": _finite_float(
            None if candidate is None else candidate.get("adjusted_response_uncertainty_rel")
        ),
        "control_sample_std": _finite_float(
            None if candidate is None else candidate.get("control_sample_std")
        ),
        "control_sample_sem": _finite_float(
            None if candidate is None else candidate.get("control_sample_sem")
        ),
        "current_pair_count": candidate_pair_count or current_pair_count,
        "candidate_blockers": (
            list(candidate.get("blockers", [])) if isinstance(candidate, Mapping) else []
        ),
    }


def _campaign_initial_blockers(
    candidate: Mapping[str, Any] | None,
    values: Mapping[str, Any],
    *,
    value_floor: float,
) -> list[str]:
    blockers: list[str] = []
    if candidate is None:
        blockers.append("no_control_variate_candidate")
    response_mean = values["response_mean"]
    if response_mean is None or abs(response_mean) <= value_floor:
        blockers.append("degenerate_response_mean")
    if values["beta"] is None:
        blockers.append("missing_control_variate_beta")
    if values["residual_sem"] is None:
        blockers.append("missing_residual_sem")
    control_sample_std = values["control_sample_std"]
    if control_sample_std is None or control_sample_std <= value_floor:
        blockers.append("missing_or_degenerate_control_sample_std")
    current_pair_count = values["current_pair_count"]
    if current_pair_count is None or current_pair_count < 2:
        blockers.append("insufficient_existing_pairs")
    hard_candidate_blockers = [
        str(item)
        for item in values["candidate_blockers"]
        if str(item) != "control_mean_not_independently_known"
    ]
    blockers.extend(f"candidate_{item}" for item in hard_candidate_blockers)
    return blockers


def _campaign_prediction(
    values: Mapping[str, Any],
    blockers: list[str],
    *,
    cfg: NonlinearGradientControlVariateCampaignConfig,
) -> dict[str, Any]:
    prediction: dict[str, Any] = {
        "target_abs_sem": None,
        "independent_control_pairs": None,
        "predicted_control_mean_sem": None,
        "predicted_control_contribution_sem": None,
        "predicted_combined_sem": None,
        "predicted_combined_uncertainty_rel": None,
        "residual_margin_sem": None,
    }
    if blockers:
        return prediction
    response_mean = float(values["response_mean"])
    beta = float(values["beta"])
    residual_sem = float(values["residual_sem"])
    control_sample_std = float(values["control_sample_std"])
    target_abs_sem = cfg.target_response_uncertainty_rel * max(
        abs(response_mean), cfg.value_floor
    )
    residual_margin_squared = target_abs_sem * target_abs_sem - residual_sem * residual_sem
    prediction["target_abs_sem"] = target_abs_sem
    if residual_margin_squared <= 0.0:
        blockers.append("residual_sem_already_exceeds_target")
        return prediction
    residual_margin_sem = math.sqrt(residual_margin_squared)
    required = math.ceil(
        cfg.sem_safety_factor
        * (abs(beta) * control_sample_std / max(residual_margin_sem, cfg.value_floor)) ** 2
    )
    independent_control_pairs = max(cfg.min_control_mean_pairs, int(required))
    predicted_control_mean_sem = control_sample_std / math.sqrt(independent_control_pairs)
    predicted_control_contribution_sem = abs(beta) * predicted_control_mean_sem
    predicted_combined_sem = math.sqrt(
        residual_sem * residual_sem + predicted_control_contribution_sem**2
    )
    predicted_combined_uncertainty_rel = predicted_combined_sem / max(
        abs(response_mean), cfg.value_floor
    )
    prediction.update(
        {
            "independent_control_pairs": independent_control_pairs,
            "predicted_control_mean_sem": predicted_control_mean_sem,
            "predicted_control_contribution_sem": predicted_control_contribution_sem,
            "predicted_combined_sem": predicted_combined_sem,
            "predicted_combined_uncertainty_rel": predicted_combined_uncertainty_rel,
            "residual_margin_sem": residual_margin_sem,
        }
    )
    if independent_control_pairs > cfg.max_control_mean_pairs:
        blockers.append("control_mean_pair_budget_exceeded")
    if predicted_combined_uncertainty_rel > cfg.target_response_uncertainty_rel:
        blockers.append("predicted_uncertainty_above_target")
    return prediction


def _planned_control_variate_pairs(
    independent_control_pairs: int | None,
    *,
    first_new_seed: int,
) -> list[dict[str, Any]]:
    if independent_control_pairs is None:
        return []
    planned_pairs = []
    for offset in range(independent_control_pairs):
        seed = first_new_seed + offset
        planned_pairs.append(
            {
                "pair_index": offset + 1,
                "variant_label": f"seed{seed}",
                "plus_state": "plus_delta",
                "minus_state": "minus_delta",
                "control_observable": "0.5 * (Q_plus + Q_minus)",
                "response_observable": "Q_plus - Q_minus",
            }
        )
    return planned_pairs


def _pack_control_variate_campaign_plan(
    *,
    variance_report: Mapping[str, Any],
    case: str,
    cfg: NonlinearGradientControlVariateCampaignConfig,
    values: Mapping[str, Any],
    blockers: Sequence[str],
    prediction: Mapping[str, Any],
    planned_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    action = (
        "launch_independent_control_mean_campaign"
        if not blockers
        else "redesign_observable_or_raise_control_mean_budget"
    )
    independent_control_pairs = prediction["independent_control_pairs"]
    return {
        "kind": "nonlinear_turbulence_gradient_control_variate_campaign_plan",
        "claim_level": "pre_run_campaign_design_not_gradient_evidence",
        "case": case,
        "passed": action == "launch_independent_control_mean_campaign",
        "action": action,
        "config": asdict(cfg),
        "source_variance_report_case": variance_report.get("case"),
        "candidate_name": values["candidate_name"],
        "candidate_blockers": values["candidate_blockers"],
        "blockers": list(blockers),
        "summary": {
            "raw_response_uncertainty_rel": _json_number(
                values["raw_response_uncertainty_rel"]
            ),
            "residual_uncertainty_rel": _json_number(
                values["residual_uncertainty_rel"]
            ),
            "predicted_combined_uncertainty_rel": _json_number(
                prediction["predicted_combined_uncertainty_rel"]
            ),
            "paired_response_mean": _json_number(values["response_mean"]),
            "target_abs_sem": _json_number(prediction["target_abs_sem"]),
            "residual_sem": _json_number(values["residual_sem"]),
            "residual_margin_sem": _json_number(prediction["residual_margin_sem"]),
            "control_sample_sem": _json_number(values["control_sample_sem"]),
            "control_sample_std": _json_number(values["control_sample_std"]),
            "control_variate_beta": _json_number(values["beta"]),
            "current_common_pair_count": values["current_pair_count"],
            "required_independent_control_mean_pairs": independent_control_pairs,
            "planned_new_run_count": None
            if independent_control_pairs is None
            else 2 * independent_control_pairs,
            "predicted_control_mean_sem": _json_number(
                prediction["predicted_control_mean_sem"]
            ),
            "predicted_control_contribution_sem": _json_number(
                prediction["predicted_control_contribution_sem"]
            ),
            "predicted_combined_sem": _json_number(prediction["predicted_combined_sem"]),
        },
        "planned_pairs": list(planned_pairs),
        "postprocess_contract": {
            "control_mean_estimator": "mean over independent 0.5 * (Q_plus + Q_minus) matched pairs",
            "response_estimator": (
                "apply the screened beta to matched response samples using the independently "
                "estimated control mean; include residual SEM and beta^2 Var(control_mean)"
            ),
            "promotion_rule": (
                "do not promote until output gates, replicated-window gates, control-mean "
                "uncertainty, and combined response uncertainty all pass"
            ),
        },
    }


def nonlinear_gradient_control_variate_campaign_plan(
    variance_report: Mapping[str, Any],
    *,
    case: str = "nonlinear_turbulence_gradient_control_variate_campaign",
    candidate_name: str | None = None,
    config: NonlinearGradientControlVariateCampaignConfig | None = None,
) -> dict[str, Any]:
    """Design an independent control-mean campaign from a variance runbook.

    The input is the output of :func:`nonlinear_gradient_variance_reduction_plan`.
    The planner is intentionally fail-closed: a sample-centered control variate
    can motivate new runs, but the campaign is only considered launch-ready when
    an independent control-mean estimate can bring the combined uncertainty
    under the target gate within the configured run budget.
    """

    cfg = config or NonlinearGradientControlVariateCampaignConfig()
    _validate_control_variate_campaign_config(cfg)
    summary, _candidates, candidate = _select_control_variate_candidate(
        variance_report, candidate_name
    )
    values = _campaign_candidate_values(summary, candidate)
    blockers = _campaign_initial_blockers(candidate, values, value_floor=cfg.value_floor)
    prediction = _campaign_prediction(values, blockers, cfg=cfg)
    planned_pairs = _planned_control_variate_pairs(
        prediction["independent_control_pairs"], first_new_seed=cfg.first_new_seed
    )
    return _pack_control_variate_campaign_plan(
        variance_report=variance_report,
        case=case,
        cfg=cfg,
        values=values,
        blockers=blockers,
        prediction=prediction,
        planned_pairs=planned_pairs,
    )


def _validate_control_mean_gate_config(cfg: NonlinearGradientControlMeanGateConfig) -> None:
    if cfg.target_response_uncertainty_rel <= 0.0:
        raise ValueError("target_response_uncertainty_rel must be positive")
    if cfg.min_control_mean_pairs < 1:
        raise ValueError("min_control_mean_pairs must be positive")


def _control_mean_candidate_values(
    summary: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "response_mean": _finite_float(summary.get("paired_response_mean")),
        "beta": _finite_float(None if candidate is None else candidate.get("beta")),
        "residual_sem": _finite_float(
            None if candidate is None else candidate.get("adjusted_response_sem")
        ),
        "residual_uncertainty_rel": _finite_float(
            None if candidate is None else candidate.get("adjusted_response_uncertainty_rel")
        ),
        "candidate_name": None if candidate is None else str(candidate.get("name")),
    }


def _control_mean_initial_blockers(
    *,
    candidate: Mapping[str, Any] | None,
    values: Mapping[str, Any],
    plus_ensemble: Mapping[str, Any],
    minus_ensemble: Mapping[str, Any],
    cfg: NonlinearGradientControlMeanGateConfig,
) -> list[str]:
    blockers: list[str] = []
    response_mean = values["response_mean"]
    if candidate is None:
        blockers.append("no_control_variate_candidate")
    if response_mean is None or abs(response_mean) <= cfg.value_floor:
        blockers.append("degenerate_response_mean")
    if values["beta"] is None:
        blockers.append("missing_control_variate_beta")
    if values["residual_sem"] is None:
        blockers.append("missing_residual_sem")
    if cfg.require_state_ensembles_passed:
        if not bool(plus_ensemble.get("passed", False)):
            blockers.append("plus_control_ensemble_failed")
        if not bool(minus_ensemble.get("passed", False)):
            blockers.append("minus_control_ensemble_failed")
    return blockers


def _control_mean_samples(
    plus_ensemble: Mapping[str, Any],
    minus_ensemble: Mapping[str, Any],
) -> tuple[list[str], Mapping[str, float], Mapping[str, float], list[float], list[float]]:
    source = {"plus": plus_ensemble, "minus": minus_ensemble}
    plus = _state_means_by_label(source, "plus")
    minus = _state_means_by_label(source, "minus")
    common_labels = sorted(set(plus).intersection(minus))
    control_samples = [0.5 * (plus[item] + minus[item]) for item in common_labels]
    response_samples = [plus[item] - minus[item] for item in common_labels]
    return common_labels, plus, minus, control_samples, response_samples


def _control_mean_uncertainty(
    *,
    values: Mapping[str, Any],
    control_sem: float | None,
    blockers: list[str],
    cfg: NonlinearGradientControlMeanGateConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "control_contribution_sem": None,
        "combined_sem": None,
        "combined_uncertainty_rel": None,
    }
    if blockers:
        return result
    beta = float(values["beta"])
    residual_sem = float(values["residual_sem"])
    response_mean = float(values["response_mean"])
    assert control_sem is not None
    control_contribution_sem = abs(beta) * control_sem
    combined_sem = math.sqrt(
        residual_sem * residual_sem + control_contribution_sem * control_contribution_sem
    )
    combined_uncertainty_rel = combined_sem / max(abs(response_mean), cfg.value_floor)
    result.update(
        {
            "control_contribution_sem": control_contribution_sem,
            "combined_sem": combined_sem,
            "combined_uncertainty_rel": combined_uncertainty_rel,
        }
    )
    if combined_uncertainty_rel > cfg.target_response_uncertainty_rel:
        blockers.append("combined_response_uncertainty_above_target")
    return result


def _control_mean_pair_rows(
    common_labels: Sequence[str],
    plus: Mapping[str, float],
    minus: Mapping[str, float],
) -> list[dict[str, Any]]:
    return [
        {
            "label": item,
            "plus_mean": _json_number(plus[item]),
            "minus_mean": _json_number(minus[item]),
            "control_mean_sample": _json_number(0.5 * (plus[item] + minus[item])),
            "response_sample": _json_number(plus[item] - minus[item]),
        }
        for item in common_labels
    ]


def _pack_control_mean_gate(
    *,
    variance_report: Mapping[str, Any],
    plus_path: str | None,
    minus_path: str | None,
    case: str,
    cfg: NonlinearGradientControlMeanGateConfig,
    values: Mapping[str, Any],
    blockers: Sequence[str],
    common_pair_count: int,
    control_mean: float | None,
    control_sem: float | None,
    response_mean_independent: float | None,
    response_sem_independent: float | None,
    uncertainty: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "kind": "nonlinear_turbulence_gradient_control_mean_gate",
        "claim_level": "independent_control_mean_uncertainty_gate_not_gradient_promotion",
        "case": case,
        "passed": not blockers,
        "candidate_name": values["candidate_name"],
        "blockers": list(blockers),
        "config": asdict(cfg),
        "source_variance_report_case": variance_report.get("case"),
        "plus_path": plus_path,
        "minus_path": minus_path,
        "summary": {
            "common_pair_count": common_pair_count,
            "paired_response_mean": _json_number(values["response_mean"]),
            "residual_sem": _json_number(values["residual_sem"]),
            "residual_uncertainty_rel": _json_number(values["residual_uncertainty_rel"]),
            "control_mean": _json_number(control_mean),
            "control_mean_sem": _json_number(control_sem),
            "control_contribution_sem": _json_number(
                uncertainty["control_contribution_sem"]
            ),
            "combined_response_sem": _json_number(uncertainty["combined_sem"]),
            "combined_response_uncertainty_rel": _json_number(
                uncertainty["combined_uncertainty_rel"]
            ),
            "independent_response_mean": _json_number(response_mean_independent),
            "independent_response_sem": _json_number(response_sem_independent),
            "control_variate_beta": _json_number(values["beta"]),
        },
        "pair_rows": list(pair_rows),
    }


def nonlinear_gradient_control_mean_gate(
    variance_report: Mapping[str, Any],
    *,
    plus_ensemble: Mapping[str, Any],
    minus_ensemble: Mapping[str, Any],
    plus_path: str | None = None,
    minus_path: str | None = None,
    case: str = "nonlinear_turbulence_gradient_control_mean_gate",
    candidate_name: str | None = None,
    config: NonlinearGradientControlMeanGateConfig | None = None,
) -> dict[str, Any]:
    """Evaluate an independent control-mean estimate for a screened CV response."""

    cfg = config or NonlinearGradientControlMeanGateConfig()
    _validate_control_mean_gate_config(cfg)
    summary, _candidates, candidate = _select_control_variate_candidate(
        variance_report, candidate_name
    )
    values = _control_mean_candidate_values(summary, candidate)
    blockers = _control_mean_initial_blockers(
        candidate=candidate,
        values=values,
        plus_ensemble=plus_ensemble,
        minus_ensemble=minus_ensemble,
        cfg=cfg,
    )
    common_labels, plus, minus, control_samples, response_samples = _control_mean_samples(
        plus_ensemble, minus_ensemble
    )
    control_mean, control_sem = _mean_and_sem(control_samples)
    response_mean_independent, response_sem_independent = _mean_and_sem(response_samples)
    if len(common_labels) < cfg.min_control_mean_pairs:
        blockers.append("insufficient_control_mean_pairs")
    if control_sem is None:
        blockers.append("control_mean_sem_unavailable")
    uncertainty = _control_mean_uncertainty(
        values=values,
        control_sem=control_sem,
        blockers=blockers,
        cfg=cfg,
    )
    return _pack_control_mean_gate(
        variance_report=variance_report,
        plus_path=plus_path,
        minus_path=minus_path,
        case=case,
        cfg=cfg,
        values=values,
        blockers=blockers,
        common_pair_count=len(common_labels),
        control_mean=control_mean,
        control_sem=control_sem,
        response_mean_independent=response_mean_independent,
        response_sem_independent=response_sem_independent,
        uncertainty=uncertainty,
        pair_rows=_control_mean_pair_rows(common_labels, plus, minus),
    )


__all__ = [
    "nonlinear_gradient_control_mean_gate",
    "nonlinear_gradient_control_variate_campaign_plan",
    "nonlinear_gradient_variance_reduction_plan",
]
