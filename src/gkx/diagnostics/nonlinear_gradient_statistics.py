"""Replicated nonlinear-gradient uncertainty and control-variate gates.

These pure functions quantify matched plus/minus transport responses and an
independent control-mean correction. They never launch simulations and are safe
to use from validation tools, tests, and Python optimization workflows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class NonlinearGradientCandidateDesignConfig:
    """Conditioning limits for the next nonlinear-gradient campaign design."""

    max_gradient_uncertainty_rel: float = 0.50
    max_fd_asymmetry_rel: float = 0.50
    max_window_mean_rel_spread: float = 0.15
    max_window_sem_rel: float = 0.25
    min_fd_response_fraction: float = 0.03
    sem_safety_factor: float = 1.10
    max_extra_replicates_per_state: int = 4
    max_checked_bracket_scale: float = 1.50
    locality_safety_factor: float = 0.95
    value_floor: float = 1.0e-12


@dataclass(frozen=True)
class NonlinearGradientVarianceReductionConfig:
    """Controls for paired-seed/control-variate nonlinear-gradient planning."""

    max_paired_response_uncertainty_rel: float = 0.50
    max_control_variate_uncertainty_rel: float = 0.50
    min_control_variate_sem_reduction: float = 0.25
    require_known_control_mean: bool = True
    sem_safety_factor: float = 1.10
    min_common_pairs: int = 2
    max_extra_paired_seeds: int = 4
    value_floor: float = 1.0e-12


@dataclass(frozen=True)
class NonlinearGradientControlMeanGateConfig:
    """Acceptance limits for an independent control-mean estimate."""

    target_response_uncertainty_rel: float = 0.50
    min_control_mean_pairs: int = 4
    require_state_ensembles_passed: bool = True
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


def _label_from_row(row: Mapping[str, Any]) -> str | None:
    variant = row.get("variant")
    if isinstance(variant, Mapping):
        seed = variant.get("seed")
        if seed is not None:
            try:
                return f"seed{int(seed)}"
            except (TypeError, ValueError):
                pass
    for key in ("variant_label", "source_artifact", "summary_artifact", "path"):
        value = row.get(key)
        if not isinstance(value, str):
            continue
        label_source = Path(value).name if key != "variant_label" else value
        matches = re.findall(r"(seed[0-9]+|dt[0-9]+(?:p[0-9]+)?)", label_source)
        if matches:
            return matches[-1]
    return None


def _ensemble_stats_value(raw: Mapping[str, Any], name: str) -> float | None:
    statistics = raw.get("statistics")
    if isinstance(statistics, Mapping):
        value = _finite_float(statistics.get(name))
        if value is not None:
            return value
    return _finite_float(raw.get(name))


def _ensemble_state_variance_report(
    source_ensembles: Mapping[str, Any],
    *,
    config: NonlinearGradientCandidateDesignConfig,
) -> dict[str, Any]:
    """Summarize which state limits a finite-difference variance-reduction plan."""

    rows: list[dict[str, Any]] = []
    for state in ("baseline", "plus", "minus"):
        raw = source_ensembles.get(state)
        if not isinstance(raw, Mapping):
            continue
        statistics = raw.get("statistics")
        n_reports = _finite_int(raw.get("n_reports"))
        if n_reports is None and isinstance(statistics, Mapping):
            n_reports = _finite_int(statistics.get("n_reports"))
        mean_rel_spread = _ensemble_stats_value(raw, "mean_rel_spread")
        combined_sem_rel = _ensemble_stats_value(raw, "combined_sem_rel")
        rows.append(
            {
                "state": state,
                "passed": bool(raw.get("passed", False)),
                "n_reports": n_reports,
                "mean_rel_spread": _json_number(mean_rel_spread),
                "combined_sem_rel": _json_number(combined_sem_rel),
                "spread_gate_passed": (
                    None
                    if mean_rel_spread is None
                    else bool(mean_rel_spread <= config.max_window_mean_rel_spread)
                ),
                "sem_gate_passed": (
                    None
                    if combined_sem_rel is None
                    else bool(combined_sem_rel <= config.max_window_sem_rel)
                ),
            }
        )

    finite_spreads = [
        (str(row["state"]), float(row["mean_rel_spread"]))
        for row in rows
        if row.get("mean_rel_spread") is not None
    ]
    limiting_state = None
    max_mean_rel_spread = None
    if finite_spreads:
        limiting_state, max_mean_rel_spread = max(
            finite_spreads, key=lambda item: item[1]
        )
    failed_spread_states = [
        str(row["state"]) for row in rows if row.get("spread_gate_passed") is False
    ]
    failed_sem_states = [
        str(row["state"]) for row in rows if row.get("sem_gate_passed") is False
    ]
    recommendation = "no replicated-window variance limiter identified"
    if failed_spread_states:
        recommendation = (
            "target paired-seed or control-variate variance reduction for "
            f"{', '.join(failed_spread_states)} before adding blind replicas"
        )
    elif failed_sem_states:
        recommendation = (
            "target additional matched replicas for "
            f"{', '.join(failed_sem_states)} before changing the bracket"
        )
    elif rows:
        recommendation = "state ensembles pass spread/SEM gates; focus on finite-difference conditioning"

    return {
        "state_rows": rows,
        "limiting_state": limiting_state,
        "max_mean_rel_spread": _json_number(max_mean_rel_spread),
        "failed_spread_states": failed_spread_states,
        "failed_sem_states": failed_sem_states,
        "recommendation": recommendation,
    }


def _state_means_by_label(
    source_ensembles: Mapping[str, Any], state: str
) -> dict[str, float]:
    raw = source_ensembles.get(state)
    if not isinstance(raw, Mapping):
        return {}
    rows = raw.get("rows")
    if not isinstance(rows, Sequence):
        return {}
    out: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        label = _label_from_row(row)
        value = _finite_float(row.get("late_mean"))
        if label is not None and value is not None:
            out[label] = value
    return out


def _mean_and_sem(values: Sequence[float]) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None, None
    mean = sum(finite) / len(finite)
    if len(finite) < 2:
        return mean, None
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return mean, math.sqrt(variance / len(finite))


def _sample_covariance(
    x_values: Sequence[float], y_values: Sequence[float]
) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / (
        len(x_values) - 1
    )


def _control_variate_candidate(
    *,
    name: str,
    response_samples: Sequence[float],
    control_samples: Sequence[float],
    response_mean: float | None,
    raw_sem: float | None,
    config: NonlinearGradientVarianceReductionConfig,
) -> dict[str, Any]:
    if len(response_samples) != len(control_samples) or len(response_samples) < 3:
        return {
            "name": name,
            "admissible": False,
            "blockers": ["insufficient_common_samples"],
        }
    control_mean, control_sem = _mean_and_sem(control_samples)
    response_variance = _sample_covariance(response_samples, response_samples)
    control_variance = _sample_covariance(control_samples, control_samples)
    covariance = _sample_covariance(response_samples, control_samples)
    blockers: list[str] = []
    if (
        response_mean is None
        or raw_sem is None
        or response_variance is None
        or control_variance is None
        or covariance is None
        or control_mean is None
        or control_variance <= config.value_floor
    ):
        blockers.append("degenerate_control_or_response")
        return {
            "name": name,
            "admissible": False,
            "blockers": blockers,
        }

    beta = covariance / control_variance
    adjusted_samples = [
        response - beta * (control - control_mean)
        for response, control in zip(response_samples, control_samples)
    ]
    adjusted_mean, adjusted_sem = _mean_and_sem(adjusted_samples)
    sample_count = len(response_samples)
    adjusted_uncertainty_rel = None
    sem_reduction = None
    if adjusted_sem is not None:
        adjusted_uncertainty_rel = abs(adjusted_sem) / max(
            abs(response_mean), config.value_floor
        )
        sem_reduction = 1.0 - adjusted_sem / max(raw_sem, config.value_floor)
    correlation = covariance / math.sqrt(
        max(response_variance * control_variance, config.value_floor)
    )

    if (
        adjusted_uncertainty_rel is None
        or adjusted_uncertainty_rel > config.max_control_variate_uncertainty_rel
    ):
        blockers.append("control_variate_uncertainty_above_gate")
    if (
        sem_reduction is None
        or sem_reduction < config.min_control_variate_sem_reduction
    ):
        blockers.append("control_variate_sem_reduction_too_small")
    if config.require_known_control_mean:
        blockers.append("control_mean_not_independently_known")

    return {
        "name": name,
        "admissible": not blockers,
        "blockers": blockers,
        "n_samples": len(response_samples),
        "beta": _json_number(beta),
        "correlation": _json_number(correlation),
        "control_mean_sample": _json_number(control_mean),
        "control_sample_sem": _json_number(control_sem),
        "control_sample_std": _json_number(
            None if control_sem is None else control_sem * math.sqrt(sample_count)
        ),
        "adjusted_response_mean": _json_number(adjusted_mean),
        "adjusted_response_sem": _json_number(adjusted_sem),
        "adjusted_response_sample_std": _json_number(
            None if adjusted_sem is None else adjusted_sem * math.sqrt(sample_count)
        ),
        "adjusted_response_uncertainty_rel": _json_number(adjusted_uncertainty_rel),
        "sem_reduction_fraction": _json_number(sem_reduction),
        "requires_independent_control_mean": bool(config.require_known_control_mean),
    }


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
            row["baseline_minus_difference"] = _json_number(
                baseline[item] - minus[item]
            )
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
    midpoint_control = [
        0.5 * (plus[item] + minus[item]) for item in common_with_baseline
    ]
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
                None
                if best_control_variate is None
                else str(best_control_variate.get("name"))
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


def _validate_control_mean_gate_config(
    cfg: NonlinearGradientControlMeanGateConfig,
) -> None:
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
            None
            if candidate is None
            else candidate.get("adjusted_response_uncertainty_rel")
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
) -> tuple[
    list[str], Mapping[str, float], Mapping[str, float], list[float], list[float]
]:
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
        residual_sem * residual_sem
        + control_contribution_sem * control_contribution_sem
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
            "residual_uncertainty_rel": _json_number(
                values["residual_uncertainty_rel"]
            ),
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
    common_labels, plus, minus, control_samples, response_samples = (
        _control_mean_samples(plus_ensemble, minus_ensemble)
    )
    control_mean, control_sem = _mean_and_sem(control_samples)
    response_mean_independent, response_sem_independent = _mean_and_sem(
        response_samples
    )
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
    "NonlinearGradientControlMeanGateConfig",
    "NonlinearGradientVarianceReductionConfig",
    "nonlinear_gradient_control_mean_gate",
    "nonlinear_gradient_variance_reduction_plan",
]
