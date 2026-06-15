"""Variance-reduction follow-up reports for nonlinear-gradient audits."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence
import math

from spectraxgk.nonlinear_gradient_followup_core import (
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

    source_ensembles_raw = artifact.get("source_ensembles")
    source_ensembles = source_ensembles_raw if isinstance(source_ensembles_raw, Mapping) else {}
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

    paired_mean, paired_sem = _mean_and_sem(paired_differences)
    paired_uncertainty_rel = None
    if paired_mean is not None and paired_sem is not None:
        paired_uncertainty_rel = abs(paired_sem) / max(abs(paired_mean), cfg.value_floor)

    control_candidates: list[dict[str, Any]] = []
    if common_with_baseline:
        response_for_baseline = [plus[item] - minus[item] for item in common_with_baseline]
        baseline_control = [baseline[item] for item in common_with_baseline]
        midpoint_control = [0.5 * (plus[item] + minus[item]) for item in common_with_baseline]
        control_candidates.append(
            _control_variate_candidate(
                name="baseline_transport_common_mode",
                response_samples=response_for_baseline,
                control_samples=baseline_control,
                response_mean=paired_mean,
                raw_sem=paired_sem,
                config=cfg,
            )
        )
        control_candidates.append(
            _control_variate_candidate(
                name="plus_minus_midpoint_common_mode",
                response_samples=response_for_baseline,
                control_samples=midpoint_control,
                response_mean=paired_mean,
                raw_sem=paired_sem,
                config=cfg,
            )
        )

    apparent_candidates = [
        row
        for row in control_candidates
        if "control_variate_uncertainty_above_gate" not in row.get("blockers", [])
        and "control_variate_sem_reduction_too_small" not in row.get("blockers", [])
    ]
    best_control_variate = None
    if control_candidates:
        def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[float, float]:
            uncertainty = _finite_float(row.get("adjusted_response_uncertainty_rel"))
            reduction = _finite_float(row.get("sem_reduction_fraction"))
            return (
                uncertainty if uncertainty is not None else float("inf"),
                -(reduction if reduction is not None else float("-inf")),
            )

        best_control_variate = min(
            control_candidates,
            key=_candidate_sort_key,
        )

    required_pairs = None
    extra_pairs = None
    if paired_uncertainty_rel is not None:
        scale = (paired_uncertainty_rel / cfg.max_paired_response_uncertainty_rel) ** 2
        scale *= cfg.sem_safety_factor
        required_pairs = max(len(common_labels), int(math.ceil(len(common_labels) * scale)))
        extra_pairs = max(0, required_pairs - len(common_labels))

    if len(common_labels) < cfg.min_common_pairs:
        action = "recover_or_add_matched_seed_pairs"
        recommendation = "common plus/minus seed labels are insufficient for paired finite differences"
    elif paired_uncertainty_rel is None:
        action = "add_matched_seed_pairs"
        recommendation = "paired response SEM cannot be estimated from fewer than two finite pairs"
    elif paired_uncertainty_rel <= cfg.max_paired_response_uncertainty_rel:
        action = "use_paired_seed_response_estimator"
        recommendation = "paired seed response uncertainty is within the target gate"
    elif apparent_candidates and cfg.require_known_control_mean:
        action = "estimate_control_mean_or_redesign_observable"
        recommendation = (
            "a common-mode control variate reduces residual scatter, but its expectation is not "
            "independently known; estimate the control mean or redesign the observable before "
            "using it as a production uncertainty reducer"
        )
    elif any(row.get("admissible") for row in control_candidates):
        action = "use_control_variate_response_estimator"
        recommendation = "control-variate response uncertainty is within the target gate"
    elif extra_pairs is not None and extra_pairs <= cfg.max_extra_paired_seeds:
        action = "add_matched_paired_seed_replicates"
        recommendation = "add bounded matched plus/minus seed pairs before changing the observable"
    else:
        action = "design_control_variate_or_new_observable"
        recommendation = (
            "paired seed differences reduce common noise but are still too uncertain; "
            "design a control-variate observable or better-conditioned response before more GPU time"
        )

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
            "common_pair_count": len(common_labels),
            "common_with_baseline_count": len(common_with_baseline),
            "paired_response_mean": _json_number(paired_mean),
            "paired_response_sem": _json_number(paired_sem),
            "paired_response_uncertainty_rel": _json_number(paired_uncertainty_rel),
            "required_pair_count": required_pairs,
            "extra_pair_count": extra_pairs,
            "best_control_variate": (
                None if best_control_variate is None else str(best_control_variate.get("name"))
            ),
        },
        "control_variate_candidates": control_candidates,
        "pair_rows": pair_rows,
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

    summary_raw = variance_report.get("summary")
    summary = summary_raw if isinstance(summary_raw, Mapping) else {}
    response_mean = _finite_float(summary.get("paired_response_mean"))
    raw_response_uncertainty_rel = _finite_float(summary.get("paired_response_uncertainty_rel"))
    requested_candidate = candidate_name or str(summary.get("best_control_variate") or "")
    candidates_raw = variance_report.get("control_variate_candidates")
    candidates = [row for row in candidates_raw if isinstance(row, Mapping)] if isinstance(candidates_raw, Sequence) else []
    if requested_candidate:
        candidate = next((row for row in candidates if str(row.get("name")) == requested_candidate), None)
    else:
        candidate = None
    if candidate is None and candidates:
        def _sort_key(row: Mapping[str, Any]) -> tuple[float, float]:
            uncertainty = _finite_float(row.get("adjusted_response_uncertainty_rel"))
            reduction = _finite_float(row.get("sem_reduction_fraction"))
            return (
                uncertainty if uncertainty is not None else float("inf"),
                -(reduction if reduction is not None else float("-inf")),
            )

        candidate = min(candidates, key=_sort_key)

    blockers: list[str] = []
    if candidate is None:
        blockers.append("no_control_variate_candidate")
    candidate_name_out = None if candidate is None else str(candidate.get("name"))
    beta = _finite_float(None if candidate is None else candidate.get("beta"))
    residual_sem = _finite_float(None if candidate is None else candidate.get("adjusted_response_sem"))
    residual_uncertainty_rel = _finite_float(
        None if candidate is None else candidate.get("adjusted_response_uncertainty_rel")
    )
    control_sample_std = _finite_float(None if candidate is None else candidate.get("control_sample_std"))
    control_sample_sem = _finite_float(None if candidate is None else candidate.get("control_sample_sem"))
    current_pair_count = _finite_int(summary.get("common_with_baseline_count") or summary.get("common_pair_count"))
    candidate_pair_count = _finite_int(None if candidate is None else candidate.get("n_samples"))
    current_pair_count = candidate_pair_count or current_pair_count
    candidate_blockers = list(candidate.get("blockers", [])) if isinstance(candidate, Mapping) else []

    if response_mean is None or abs(response_mean) <= cfg.value_floor:
        blockers.append("degenerate_response_mean")
    if beta is None:
        blockers.append("missing_control_variate_beta")
    if residual_sem is None:
        blockers.append("missing_residual_sem")
    if control_sample_std is None or control_sample_std <= cfg.value_floor:
        blockers.append("missing_or_degenerate_control_sample_std")
    if current_pair_count is None or current_pair_count < 2:
        blockers.append("insufficient_existing_pairs")
    hard_candidate_blockers = [
        str(item)
        for item in candidate_blockers
        if str(item) != "control_mean_not_independently_known"
    ]
    if hard_candidate_blockers:
        blockers.extend(f"candidate_{item}" for item in hard_candidate_blockers)

    target_abs_sem = None
    independent_control_pairs = None
    predicted_control_mean_sem = None
    predicted_control_contribution_sem = None
    predicted_combined_sem = None
    predicted_combined_uncertainty_rel = None
    residual_margin_sem = None
    if not blockers:
        assert response_mean is not None
        assert beta is not None
        assert residual_sem is not None
        assert control_sample_std is not None
        target_abs_sem = cfg.target_response_uncertainty_rel * max(abs(response_mean), cfg.value_floor)
        residual_margin_squared = target_abs_sem * target_abs_sem - residual_sem * residual_sem
        if residual_margin_squared <= 0.0:
            blockers.append("residual_sem_already_exceeds_target")
        else:
            residual_margin_sem = math.sqrt(residual_margin_squared)
            required = math.ceil(
                cfg.sem_safety_factor
                * (abs(beta) * control_sample_std / max(residual_margin_sem, cfg.value_floor)) ** 2
            )
            independent_control_pairs = max(cfg.min_control_mean_pairs, int(required))
            predicted_control_mean_sem = control_sample_std / math.sqrt(independent_control_pairs)
            predicted_control_contribution_sem = abs(beta) * predicted_control_mean_sem
            predicted_combined_sem = math.sqrt(residual_sem * residual_sem + predicted_control_contribution_sem**2)
            predicted_combined_uncertainty_rel = predicted_combined_sem / max(abs(response_mean), cfg.value_floor)
            if independent_control_pairs > cfg.max_control_mean_pairs:
                blockers.append("control_mean_pair_budget_exceeded")
            if predicted_combined_uncertainty_rel > cfg.target_response_uncertainty_rel:
                blockers.append("predicted_uncertainty_above_target")

    action = (
        "launch_independent_control_mean_campaign"
        if not blockers
        else "redesign_observable_or_raise_control_mean_budget"
    )
    planned_pairs = []
    if independent_control_pairs is not None:
        for offset in range(independent_control_pairs):
            seed = cfg.first_new_seed + offset
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

    return {
        "kind": "nonlinear_turbulence_gradient_control_variate_campaign_plan",
        "claim_level": "pre_run_campaign_design_not_gradient_evidence",
        "case": case,
        "passed": action == "launch_independent_control_mean_campaign",
        "action": action,
        "config": asdict(cfg),
        "source_variance_report_case": variance_report.get("case"),
        "candidate_name": candidate_name_out,
        "candidate_blockers": candidate_blockers,
        "blockers": blockers,
        "summary": {
            "raw_response_uncertainty_rel": _json_number(raw_response_uncertainty_rel),
            "residual_uncertainty_rel": _json_number(residual_uncertainty_rel),
            "predicted_combined_uncertainty_rel": _json_number(predicted_combined_uncertainty_rel),
            "paired_response_mean": _json_number(response_mean),
            "target_abs_sem": _json_number(target_abs_sem),
            "residual_sem": _json_number(residual_sem),
            "residual_margin_sem": _json_number(residual_margin_sem),
            "control_sample_sem": _json_number(control_sample_sem),
            "control_sample_std": _json_number(control_sample_std),
            "control_variate_beta": _json_number(beta),
            "current_common_pair_count": current_pair_count,
            "required_independent_control_mean_pairs": independent_control_pairs,
            "planned_new_run_count": None if independent_control_pairs is None else 2 * independent_control_pairs,
            "predicted_control_mean_sem": _json_number(predicted_control_mean_sem),
            "predicted_control_contribution_sem": _json_number(predicted_control_contribution_sem),
            "predicted_combined_sem": _json_number(predicted_combined_sem),
        },
        "planned_pairs": planned_pairs,
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
    if cfg.target_response_uncertainty_rel <= 0.0:
        raise ValueError("target_response_uncertainty_rel must be positive")
    if cfg.min_control_mean_pairs < 1:
        raise ValueError("min_control_mean_pairs must be positive")

    summary_raw = variance_report.get("summary")
    summary = summary_raw if isinstance(summary_raw, Mapping) else {}
    requested_candidate = candidate_name or str(summary.get("best_control_variate") or "")
    candidates_raw = variance_report.get("control_variate_candidates")
    candidates = [row for row in candidates_raw if isinstance(row, Mapping)] if isinstance(candidates_raw, Sequence) else []
    candidate = (
        next((row for row in candidates if str(row.get("name")) == requested_candidate), None)
        if requested_candidate
        else None
    )
    if candidate is None and candidates:
        candidate = min(
            candidates,
            key=lambda row: (
                _finite_float(row.get("adjusted_response_uncertainty_rel")) or float("inf"),
                -(_finite_float(row.get("sem_reduction_fraction")) or float("-inf")),
            ),
        )

    blockers: list[str] = []
    response_mean = _finite_float(summary.get("paired_response_mean"))
    beta = _finite_float(None if candidate is None else candidate.get("beta"))
    residual_sem = _finite_float(None if candidate is None else candidate.get("adjusted_response_sem"))
    residual_uncertainty_rel = _finite_float(
        None if candidate is None else candidate.get("adjusted_response_uncertainty_rel")
    )
    if candidate is None:
        blockers.append("no_control_variate_candidate")
    if response_mean is None or abs(response_mean) <= cfg.value_floor:
        blockers.append("degenerate_response_mean")
    if beta is None:
        blockers.append("missing_control_variate_beta")
    if residual_sem is None:
        blockers.append("missing_residual_sem")
    if cfg.require_state_ensembles_passed:
        if not bool(plus_ensemble.get("passed", False)):
            blockers.append("plus_control_ensemble_failed")
        if not bool(minus_ensemble.get("passed", False)):
            blockers.append("minus_control_ensemble_failed")

    source = {"plus": plus_ensemble, "minus": minus_ensemble}
    plus = _state_means_by_label(source, "plus")
    minus = _state_means_by_label(source, "minus")
    common_labels = sorted(set(plus).intersection(minus))
    control_samples = [0.5 * (plus[item] + minus[item]) for item in common_labels]
    response_samples = [plus[item] - minus[item] for item in common_labels]
    control_mean, control_sem = _mean_and_sem(control_samples)
    response_mean_independent, response_sem_independent = _mean_and_sem(response_samples)
    if len(common_labels) < cfg.min_control_mean_pairs:
        blockers.append("insufficient_control_mean_pairs")
    if control_sem is None:
        blockers.append("control_mean_sem_unavailable")

    control_contribution_sem = None
    combined_sem = None
    combined_uncertainty_rel = None
    if not blockers:
        assert beta is not None
        assert residual_sem is not None
        assert response_mean is not None
        assert control_sem is not None
        control_contribution_sem = abs(beta) * control_sem
        combined_sem = math.sqrt(residual_sem * residual_sem + control_contribution_sem * control_contribution_sem)
        combined_uncertainty_rel = combined_sem / max(abs(response_mean), cfg.value_floor)
        if combined_uncertainty_rel > cfg.target_response_uncertainty_rel:
            blockers.append("combined_response_uncertainty_above_target")

    pair_rows = [
        {
            "label": item,
            "plus_mean": _json_number(plus[item]),
            "minus_mean": _json_number(minus[item]),
            "control_mean_sample": _json_number(0.5 * (plus[item] + minus[item])),
            "response_sample": _json_number(plus[item] - minus[item]),
        }
        for item in common_labels
    ]

    return {
        "kind": "nonlinear_turbulence_gradient_control_mean_gate",
        "claim_level": "independent_control_mean_uncertainty_gate_not_gradient_promotion",
        "case": case,
        "passed": not blockers,
        "candidate_name": None if candidate is None else str(candidate.get("name")),
        "blockers": blockers,
        "config": asdict(cfg),
        "source_variance_report_case": variance_report.get("case"),
        "plus_path": plus_path,
        "minus_path": minus_path,
        "summary": {
            "common_pair_count": len(common_labels),
            "paired_response_mean": _json_number(response_mean),
            "residual_sem": _json_number(residual_sem),
            "residual_uncertainty_rel": _json_number(residual_uncertainty_rel),
            "control_mean": _json_number(control_mean),
            "control_mean_sem": _json_number(control_sem),
            "control_contribution_sem": _json_number(control_contribution_sem),
            "combined_response_sem": _json_number(combined_sem),
            "combined_response_uncertainty_rel": _json_number(combined_uncertainty_rel),
            "independent_response_mean": _json_number(response_mean_independent),
            "independent_response_sem": _json_number(response_sem_independent),
            "control_variate_beta": _json_number(beta),
        },
        "pair_rows": pair_rows,
    }

__all__ = [
    "nonlinear_gradient_control_mean_gate",
    "nonlinear_gradient_control_variate_campaign_plan",
    "nonlinear_gradient_variance_reduction_plan",
]
