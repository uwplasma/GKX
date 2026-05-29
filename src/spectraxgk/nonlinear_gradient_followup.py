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
class NonlinearGradientControlVariateCampaignConfig:
    """Controls for an independent control-mean campaign."""

    target_response_uncertainty_rel: float = 0.50
    sem_safety_factor: float = 1.10
    min_control_mean_pairs: int = 4
    max_control_mean_pairs: int = 32
    first_new_seed: int = 34
    value_floor: float = 1.0e-12


@dataclass(frozen=True)
class NonlinearGradientControlMeanGateConfig:
    """Acceptance limits for an independent control-mean estimate."""

    target_response_uncertainty_rel: float = 0.50
    min_control_mean_pairs: int = 4
    require_state_ensembles_passed: bool = True
    value_floor: float = 1.0e-12


@dataclass(frozen=True)
class NonlinearGradientCompositeControlConfig:
    """Controls for constructing the next composite nonlinear-gradient direction."""

    max_gradient_uncertainty_rel: float = 1.00
    max_fd_asymmetry_rel: float = 0.50
    min_fd_response_fraction: float = 0.03
    min_same_sign_fraction: float = 0.80
    min_controls: int = 2
    default_relative_delta: float = 0.02
    max_weight_abs: float = 1.0
    value_floor: float = 1.0e-12


@dataclass(frozen=True)
class NonlinearGradientQLSeedScreenConfig:
    """Admission limits for quasilinear-seeded nonlinear-gradient controls."""

    target_objectives: tuple[str, ...] = (
        "mixing_length_heat_flux_proxy",
        "linear_heat_flux_weight",
        "gamma",
    )
    primary_objective: str = "mixing_length_heat_flux_proxy"
    min_distinct_controls: int = 2
    min_cases_per_control: int = 2
    min_sign_consistency: float = 0.75
    max_objective_rel_error: float = 0.02
    min_abs_sensitivity: float = 1.0e-12
    require_artifact_passed: bool = False


@dataclass(frozen=True)
class NonlinearGradientStateControlRunbookConfig:
    """Admission limits for mapping VMEC-state controls to launchable inputs."""

    min_mapped_controls: int = 2
    max_mapping_condition_number: float = 1.0e6
    max_mapping_relative_residual: float = 0.10
    default_relative_delta: float = 0.02
    require_mapping_passed: bool = True


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


def _nested_metric(payload: Mapping[str, Any], source_name: str, *names: str) -> float | None:
    source = payload.get(source_name)
    if not isinstance(source, Mapping):
        return None
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


def _coefficient_label_from_parameter(parameter: Any) -> str | None:
    if not isinstance(parameter, str):
        return None
    match = re.fullmatch(r"\s*(rbc|rbs|zbc|zbs)_([+-]?\d+)_([+-]?\d+)\s*", parameter, re.IGNORECASE)
    if match is None:
        return None
    family, m_value, n_value = match.groups()
    return f"{family.upper()}({int(m_value)},{int(n_value)})"


def _state_control_family(parameter_indices: Any) -> str | None:
    if not isinstance(parameter_indices, Mapping) or not parameter_indices:
        return None
    for family in ("Rcos", "Rsin", "Zcos", "Zsin", "RBC", "RBS", "ZBC", "ZBS"):
        if family in parameter_indices:
            return family
    return str(next(iter(parameter_indices)))


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
        limiting_state, max_mean_rel_spread = max(finite_spreads, key=lambda item: item[1])
    failed_spread_states = [
        str(row["state"])
        for row in rows
        if row.get("spread_gate_passed") is False
    ]
    failed_sem_states = [
        str(row["state"])
        for row in rows
        if row.get("sem_gate_passed") is False
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


def _state_means_by_label(source_ensembles: Mapping[str, Any], state: str) -> dict[str, float]:
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


def _sample_covariance(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / (len(x_values) - 1)


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
        adjusted_uncertainty_rel = abs(adjusted_sem) / max(abs(response_mean), config.value_floor)
        sem_reduction = 1.0 - adjusted_sem / max(raw_sem, config.value_floor)
    correlation = covariance / math.sqrt(max(response_variance * control_variance, config.value_floor))

    if adjusted_uncertainty_rel is None or adjusted_uncertainty_rel > config.max_control_variate_uncertainty_rel:
        blockers.append("control_variate_uncertainty_above_gate")
    if sem_reduction is None or sem_reduction < config.min_control_variate_sem_reduction:
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
        "control_sample_std": _json_number(None if control_sem is None else control_sem * math.sqrt(sample_count)),
        "adjusted_response_mean": _json_number(adjusted_mean),
        "adjusted_response_sem": _json_number(adjusted_sem),
        "adjusted_response_sample_std": _json_number(
            None if adjusted_sem is None else adjusted_sem * math.sqrt(sample_count)
        ),
        "adjusted_response_uncertainty_rel": _json_number(adjusted_uncertainty_rel),
        "sem_reduction_fraction": _json_number(sem_reduction),
        "requires_independent_control_mean": bool(config.require_known_control_mean),
    }


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
    elif variance_report["failed_spread_states"]:
        action = "design_variance_reduction_for_limiting_state"
        recommendation = str(variance_report["recommendation"])
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
        for index, (artifact, path, label) in enumerate(zip(artifacts, path_list, label_list))
    ]
    promoted = [row for row in candidates if row["action"] == "freeze_promoted_candidate"]
    bracket_ready = [row for row in candidates if row["action"] == "run_checked_larger_bracket"]
    replica_ready = [row for row in candidates if row["action"] == "add_limited_replicates_with_locality_cap"]
    variance_limited = [row for row in candidates if row["action"] == "design_variance_reduction_for_limiting_state"]
    replacement = [
        row
        for row in candidates
        if row["action"] in {
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
            "variance_limited_count": len(variance_limited),
            "replacement_or_variance_reduction_count": len(replacement),
        },
        "candidates": candidates,
    }


def _composite_control_row(
    artifact: Mapping[str, Any],
    *,
    index: int,
    path: str | None,
    label: str | None,
    config: NonlinearGradientCompositeControlConfig,
) -> dict[str, Any]:
    parameter_name = str(artifact.get("parameter_name") or label or path or f"candidate_{index}")
    coefficient = _coefficient_label_from_parameter(artifact.get("parameter_name"))
    response_fraction = _metric(artifact, "response_fraction")
    asymmetry_rel = _metric(artifact, "fd_asymmetry_rel", "asymmetry_rel")
    uncertainty_rel = _metric(
        artifact,
        "gradient_uncertainty_rel",
        "gradient_relative_uncertainty",
    )
    central_gradient = _metric(
        artifact,
        "central_gradient",
        "central_fd_dq_dparameter",
        "central_fd_dq_dtprim",
    )
    paired_uncertainty_rel = _nested_metric(
        artifact,
        "paired_replicate_diagnostics",
        "central_gradient_uncertainty_rel",
    )
    same_sign_fraction = _nested_metric(
        artifact,
        "paired_replicate_diagnostics",
        "same_sign_fraction",
    )
    response_ok = response_fraction is not None and response_fraction >= config.min_fd_response_fraction
    locality_ok = asymmetry_rel is not None and asymmetry_rel <= config.max_fd_asymmetry_rel
    uncertainty_ok = uncertainty_rel is not None and uncertainty_rel <= config.max_gradient_uncertainty_rel
    same_sign_ok = same_sign_fraction is None or same_sign_fraction >= config.min_same_sign_fraction
    gradient_ok = central_gradient is not None and abs(central_gradient) > config.value_floor
    coefficient_ok = coefficient is not None
    admissible = bool(
        coefficient_ok
        and gradient_ok
        and response_ok
        and locality_ok
        and uncertainty_ok
        and same_sign_ok
    )

    blockers: list[str] = []
    if not coefficient_ok:
        blockers.append("parameter_not_vmec_boundary_coefficient")
    if not gradient_ok:
        blockers.append("missing_or_zero_central_gradient")
    if not response_ok:
        blockers.append("unresolved_heat_flux_response")
    if not locality_ok:
        blockers.append("nonlocal_finite_difference_bracket")
    if not uncertainty_ok:
        blockers.append("gradient_uncertainty_too_large")
    if not same_sign_ok:
        blockers.append("paired_replicate_sign_not_robust")

    descent_gradient = None if central_gradient is None else -float(central_gradient)
    return {
        "index": index,
        "label": str(label or parameter_name),
        "path": path,
        "parameter_name": parameter_name,
        "coefficient": coefficient,
        "admissible_for_composite_direction": admissible,
        "blockers": blockers,
        "metrics": {
            "central_gradient": _json_number(central_gradient),
            "descent_gradient": _json_number(descent_gradient),
            "response_fraction": _json_number(response_fraction),
            "fd_asymmetry_rel": _json_number(asymmetry_rel),
            "gradient_uncertainty_rel": _json_number(uncertainty_rel),
            "paired_gradient_uncertainty_rel": _json_number(paired_uncertainty_rel),
            "same_sign_fraction": _json_number(same_sign_fraction),
        },
        "gate_status": {
            "coefficient_ok": coefficient_ok,
            "gradient_ok": gradient_ok,
            "response_ok": bool(response_ok),
            "locality_ok": bool(locality_ok),
            "uncertainty_ok": bool(uncertainty_ok),
            "same_sign_ok": bool(same_sign_ok),
        },
    }


def nonlinear_gradient_composite_control_report(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    paths: Sequence[str | None] | None = None,
    labels: Sequence[str | None] | None = None,
    case: str = "nonlinear_turbulence_gradient_composite_control_design",
    config: NonlinearGradientCompositeControlConfig | None = None,
) -> dict[str, Any]:
    """Design a normalized VMEC-boundary direction from resolved FD candidates.

    This is a launch-planning gate, not nonlinear-gradient evidence.  The
    returned controls are the steepest-descent direction in the subspace of
    candidates that already pass locality, response, uncertainty, coefficient,
    and paired-sign checks. If fewer than ``min_controls`` survive, the report
    fails closed and provides exact blockers instead of producing a misleading
    multi-coefficient launch recommendation.
    """

    cfg = config or NonlinearGradientCompositeControlConfig()
    if cfg.max_gradient_uncertainty_rel <= 0.0:
        raise ValueError("max_gradient_uncertainty_rel must be positive")
    if cfg.max_fd_asymmetry_rel <= 0.0:
        raise ValueError("max_fd_asymmetry_rel must be positive")
    if cfg.min_fd_response_fraction <= 0.0:
        raise ValueError("min_fd_response_fraction must be positive")
    if cfg.min_same_sign_fraction <= 0.0 or cfg.min_same_sign_fraction > 1.0:
        raise ValueError("min_same_sign_fraction must be in (0, 1]")
    if cfg.min_controls < 1:
        raise ValueError("min_controls must be at least one")
    if cfg.default_relative_delta <= 0.0:
        raise ValueError("default_relative_delta must be positive")
    if cfg.max_weight_abs <= 0.0:
        raise ValueError("max_weight_abs must be positive")

    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")

    rows = [
        _composite_control_row(artifact, index=index, path=path, label=label, config=cfg)
        for index, (artifact, path, label) in enumerate(zip(artifacts, path_list, label_list))
    ]
    admissible_rows = [row for row in rows if bool(row["admissible_for_composite_direction"])]
    max_abs_descent = max(
        (abs(float(row["metrics"]["descent_gradient"])) for row in admissible_rows),
        default=0.0,
    )
    controls: list[dict[str, Any]] = []
    if max_abs_descent > cfg.value_floor:
        for row in admissible_rows:
            descent = float(row["metrics"]["descent_gradient"])
            weight = cfg.max_weight_abs * descent / max_abs_descent
            controls.append(
                {
                    "parameter_name": row["parameter_name"],
                    "coefficient": row["coefficient"],
                    "weight": _json_number(weight),
                    "control_argument": f"{row['coefficient']}:{weight:.12g}",
                    "source_label": row["label"],
                    "source_path": row["path"],
                }
            )

    launch_ready = len(controls) >= cfg.min_controls
    if launch_ready:
        control_args = " ".join(f"--control {control['control_argument']}" for control in controls)
        command_template = (
            "python tools/write_vmec_boundary_profile_perturbation_inputs.py "
            "--baseline-input <input.vmec> "
            "--out-dir docs/_static/<case>_composite_direction "
            f"--case {case} "
            f"{control_args} "
            f"--relative-delta {cfg.default_relative_delta:.12g}"
        )
        next_action = "launch a checked VMEC profile-direction bracket sweep before long nonlinear windows"
    elif controls:
        command_template = None
        next_action = (
            "only one admissible control remains; screen additional local/resolved controls "
            "or explicitly run a single-control bracket check before a long campaign"
        )
    else:
        command_template = None
        next_action = "no admissible controls; screen new VMEC-boundary directions before nonlinear GPU runs"

    return {
        "kind": "nonlinear_turbulence_gradient_composite_control_design",
        "claim_level": "composite_control_launch_plan_not_gradient_evidence",
        "case": case,
        "passed": bool(launch_ready),
        "next_action": next_action,
        "config": asdict(cfg),
        "summary": {
            "candidate_count": len(rows),
            "admissible_control_count": len(controls),
            "required_control_count": cfg.min_controls,
            "launch_ready": bool(launch_ready),
        },
        "controls": controls,
        "write_profile_direction_command_template": command_template,
        "candidates": rows,
    }


def _ql_seed_rows(
    artifact: Mapping[str, Any],
    *,
    index: int,
    path: str | None,
    label: str | None,
    config: NonlinearGradientQLSeedScreenConfig,
) -> list[dict[str, Any]]:
    objective_gates = artifact.get("objective_gates")
    if not isinstance(objective_gates, Sequence):
        return []
    artifact_passed = bool(artifact.get("passed", False))
    case_name = str(artifact.get("case_name") or label or path or f"artifact_{index}")
    parameter_indices = artifact.get("parameter_indices")
    source_family = _state_control_family(parameter_indices)
    rows: list[dict[str, Any]] = []
    for gate_index, gate in enumerate(objective_gates):
        if not isinstance(gate, Mapping):
            continue
        objective = str(gate.get("objective") or "")
        if objective not in config.target_objectives:
            continue
        parameter = str(gate.get("parameter") or "")
        implicit = _finite_float(gate.get("implicit"))
        finite_difference = _finite_float(gate.get("finite_difference"))
        rel_error = _finite_float(gate.get("rel_error"))
        gate_passed = bool(gate.get("passed", False))
        sensitivity_resolved = implicit is not None and abs(implicit) >= config.min_abs_sensitivity
        rel_error_ok = rel_error is not None and rel_error <= config.max_objective_rel_error
        accepted = bool(
            parameter
            and sensitivity_resolved
            and rel_error_ok
            and gate_passed
            and (artifact_passed or not config.require_artifact_passed)
        )
        blockers: list[str] = []
        if not parameter:
            blockers.append("missing_parameter_name")
        if not sensitivity_resolved:
            blockers.append("unresolved_objective_sensitivity")
        if not rel_error_ok:
            blockers.append("ad_fd_relative_error_too_large")
        if not gate_passed:
            blockers.append("objective_gate_failed")
        if config.require_artifact_passed and not artifact_passed:
            blockers.append("source_artifact_failed")
        direction = None if implicit is None else -math.copysign(1.0, implicit)
        rows.append(
            {
                "artifact_index": index,
                "gate_index": gate_index,
                "label": str(label or case_name),
                "path": path,
                "case_name": case_name,
                "source_kind": str(artifact.get("kind", "")),
                "source_artifact_passed": artifact_passed,
                "state_parameter": parameter,
                "state_control_family": source_family,
                "parameter_indices": parameter_indices if isinstance(parameter_indices, Mapping) else None,
                "objective": objective,
                "accepted_objective_gate": accepted,
                "blockers": blockers,
                "metrics": {
                    "implicit_sensitivity": _json_number(implicit),
                    "finite_difference_sensitivity": _json_number(finite_difference),
                    "relative_error": _json_number(rel_error),
                    "descent_direction_sign": _json_number(direction),
                },
            }
        )
    return rows


def _sign_consistency(values: Sequence[float], *, value_floor: float) -> tuple[float | None, float | None]:
    signs = [math.copysign(1.0, value) for value in values if abs(value) > value_floor]
    if not signs:
        return None, None
    positive = sum(1 for sign in signs if sign > 0.0)
    negative = len(signs) - positive
    dominant = 1.0 if positive >= negative else -1.0
    return dominant, max(positive, negative) / len(signs)


def nonlinear_gradient_ql_seed_screen_report(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    paths: Sequence[str | None] | None = None,
    labels: Sequence[str | None] | None = None,
    case: str = "nonlinear_turbulence_gradient_ql_seed_screen",
    config: NonlinearGradientQLSeedScreenConfig | None = None,
) -> dict[str, Any]:
    """Screen QL/linear sensitivity artifacts before nonlinear-gradient runs.

    The report groups full-chain VMEC/Boozer sensitivity rows by state
    parameter and admits a control only when the primary objective sensitivity
    is AD/FD-consistent, resolved, sign-consistent across enough artifacts, and
    tied to a distinct VMEC-state control.  The output is deliberately a
    planning artifact: VMEC-state controls are not assumed to be patchable
    ``RBC/ZBS`` input-file coefficients.
    """

    cfg = config or NonlinearGradientQLSeedScreenConfig()
    if not cfg.target_objectives:
        raise ValueError("target_objectives must be non-empty")
    if cfg.primary_objective not in cfg.target_objectives:
        raise ValueError("primary_objective must be included in target_objectives")
    if cfg.min_distinct_controls < 1:
        raise ValueError("min_distinct_controls must be at least one")
    if cfg.min_cases_per_control < 1:
        raise ValueError("min_cases_per_control must be at least one")
    if cfg.min_sign_consistency <= 0.0 or cfg.min_sign_consistency > 1.0:
        raise ValueError("min_sign_consistency must be in (0, 1]")
    if cfg.max_objective_rel_error < 0.0:
        raise ValueError("max_objective_rel_error must be non-negative")
    if cfg.min_abs_sensitivity <= 0.0:
        raise ValueError("min_abs_sensitivity must be positive")

    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")

    objective_rows: list[dict[str, Any]] = []
    for index, (artifact, path, label) in enumerate(zip(artifacts, path_list, label_list)):
        objective_rows.extend(_ql_seed_rows(artifact, index=index, path=path, label=label, config=cfg))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in objective_rows:
        if row["objective"] == cfg.primary_objective:
            grouped.setdefault(str(row["state_parameter"]), []).append(row)

    controls: list[dict[str, Any]] = []
    for parameter, rows in sorted(grouped.items()):
        accepted_rows = [row for row in rows if bool(row["accepted_objective_gate"])]
        sensitivities = [
            float(row["metrics"]["implicit_sensitivity"])
            for row in accepted_rows
            if row["metrics"]["implicit_sensitivity"] is not None
        ]
        dominant_sign, sign_fraction = _sign_consistency(
            sensitivities,
            value_floor=cfg.min_abs_sensitivity,
        )
        n_cases = len({str(row["case_name"]) for row in accepted_rows})
        enough_cases = n_cases >= cfg.min_cases_per_control
        sign_ok = sign_fraction is not None and sign_fraction >= cfg.min_sign_consistency
        admitted = bool(enough_cases and sign_ok)
        blockers: list[str] = []
        if not accepted_rows:
            blockers.append("no_accepted_primary_objective_rows")
        if not enough_cases:
            blockers.append("insufficient_case_coverage")
        if not sign_ok:
            blockers.append("cross_artifact_sign_not_consistent")
        direction = None if dominant_sign is None else -dominant_sign
        mean_abs_sensitivity = None
        if sensitivities:
            mean_abs_sensitivity = sum(abs(value) for value in sensitivities) / len(sensitivities)
        controls.append(
            {
                "state_parameter": parameter,
                "state_control_family": accepted_rows[0].get("state_control_family") if accepted_rows else None,
                "admitted_for_nonlinear_screen": admitted,
                "blockers": blockers,
                "primary_objective": cfg.primary_objective,
                "n_accepted_rows": len(accepted_rows),
                "n_cases": n_cases,
                "dominant_sensitivity_sign": _json_number(dominant_sign),
                "descent_direction_sign": _json_number(direction),
                "sign_consistency_fraction": _json_number(sign_fraction),
                "mean_abs_sensitivity": _json_number(mean_abs_sensitivity),
                "state_control_argument": None
                if direction is None
                else f"{parameter}:{direction:.12g}",
                "source_rows": [
                    {
                        "case_name": row["case_name"],
                        "path": row["path"],
                        "source_artifact_passed": row["source_artifact_passed"],
                        "implicit_sensitivity": row["metrics"]["implicit_sensitivity"],
                        "relative_error": row["metrics"]["relative_error"],
                    }
                    for row in accepted_rows
                ],
            }
        )

    admitted_controls = [row for row in controls if bool(row["admitted_for_nonlinear_screen"])]
    passed = len(admitted_controls) >= cfg.min_distinct_controls
    if passed:
        next_action = "build checked short-bracket nonlinear-gradient screens for admitted VMEC-state controls"
    elif controls:
        next_action = (
            "generate additional QL/linear sensitivity artifacts for distinct VMEC-state controls "
            "before nonlinear GPU campaigns"
        )
    else:
        next_action = "no usable QL/linear sensitivity rows; generate full-chain VMEC/Boozer gradient artifacts first"

    return {
        "kind": "nonlinear_turbulence_gradient_ql_seed_screen",
        "claim_level": "ql_seeded_control_screen_not_nonlinear_gradient_evidence",
        "case": case,
        "passed": bool(passed),
        "next_action": next_action,
        "config": asdict(cfg),
        "summary": {
            "artifact_count": len(artifacts),
            "objective_row_count": len(objective_rows),
            "control_count": len(controls),
            "admitted_control_count": len(admitted_controls),
            "required_distinct_controls": cfg.min_distinct_controls,
        },
        "admitted_controls": admitted_controls,
        "controls": controls,
        "objective_rows": objective_rows,
        "scope_note": (
            "Rows describe vmec_jax state controls. They are not direct VMEC "
            "input-file RBC/ZBS coefficients unless a separate mapping artifact "
            "establishes that relation."
        ),
    }


def _mapping_control_rows(mapping_artifacts: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    rows_by_parameter: dict[str, dict[str, Any]] = {}
    for artifact_index, artifact in enumerate(mapping_artifacts):
        artifact_passed = bool(artifact.get("passed", False))
        raw_rows = artifact.get("controls")
        if not isinstance(raw_rows, Sequence):
            raw_rows = artifact.get("mappings")
        if not isinstance(raw_rows, Sequence):
            continue
        for row_index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, Mapping):
                continue
            parameter = raw_row.get("state_parameter")
            if not isinstance(parameter, str) or not parameter:
                parameter = raw_row.get("state_control")
            if not isinstance(parameter, str) or not parameter:
                continue
            candidate = {
                "artifact_index": artifact_index,
                "row_index": row_index,
                "state_parameter": parameter,
                "input_control_argument": raw_row.get("input_control_argument"),
                "input_direction": raw_row.get("input_direction"),
                "input_parameter": raw_row.get("input_parameter"),
                "passed": bool(raw_row.get("passed", False)) and artifact_passed,
                "row_passed": bool(raw_row.get("passed", False)),
                "artifact_passed": artifact_passed,
                "condition_number": _json_number(
                    _metric(raw_row, "condition_number", "mapping_condition_number")
                ),
                "relative_residual": _json_number(
                    _metric(raw_row, "relative_residual", "mapping_relative_residual")
                ),
                "dominant_response_sign": _json_number(
                    _metric(raw_row, "dominant_response_sign", "response_sign")
                ),
                "source_kind": str(artifact.get("kind", "")),
                "source_case": artifact.get("case") or artifact.get("case_name"),
            }
            current = rows_by_parameter.get(parameter)
            if current is None or (
                bool(candidate["passed"]) and not bool(current.get("passed", False))
            ):
                rows_by_parameter[parameter] = candidate
    return rows_by_parameter


def nonlinear_gradient_state_control_runbook_report(
    ql_seed_screen: Mapping[str, Any],
    *,
    mapping_artifacts: Sequence[Mapping[str, Any]] = (),
    case: str = "nonlinear_gradient_state_control_runbook",
    config: NonlinearGradientStateControlRunbookConfig | None = None,
) -> dict[str, Any]:
    """Build a fail-closed launch runbook for VMEC-state nonlinear-gradient controls.

    The QL seed screen operates on internal ``vmec_jax`` state coordinates.
    Nonlinear campaigns, however, need perturbable input directions that can be
    written to VMEC inputs and re-equilibrated.  This report joins the admitted
    state controls to an explicit state-to-input mapping artifact and refuses to
    produce launch commands until the mapping is conditioned and complete.
    """

    cfg = config or NonlinearGradientStateControlRunbookConfig()
    if cfg.min_mapped_controls < 1:
        raise ValueError("min_mapped_controls must be at least one")
    if cfg.max_mapping_condition_number <= 0.0:
        raise ValueError("max_mapping_condition_number must be positive")
    if cfg.max_mapping_relative_residual < 0.0:
        raise ValueError("max_mapping_relative_residual must be non-negative")
    if cfg.default_relative_delta <= 0.0:
        raise ValueError("default_relative_delta must be positive")

    admitted_raw = ql_seed_screen.get("admitted_controls")
    admitted = admitted_raw if isinstance(admitted_raw, Sequence) else ()
    ql_kind = ql_seed_screen.get("kind")
    ql_screen_usable = (
        ql_kind == "nonlinear_turbulence_gradient_ql_seed_screen"
        and bool(ql_seed_screen.get("passed", False))
    )
    mapping_by_parameter = _mapping_control_rows(mapping_artifacts)
    control_rows: list[dict[str, Any]] = []
    mapped_controls: list[dict[str, Any]] = []
    for raw_control in admitted:
        if not isinstance(raw_control, Mapping):
            continue
        parameter = raw_control.get("state_parameter")
        if not isinstance(parameter, str) or not parameter:
            continue
        mapping = mapping_by_parameter.get(parameter)
        blockers: list[str] = []
        if ql_kind != "nonlinear_turbulence_gradient_ql_seed_screen":
            blockers.append("invalid_ql_seed_screen_kind")
        if not bool(ql_seed_screen.get("passed", False)):
            blockers.append("ql_seed_screen_failed")
        if mapping is None:
            blockers.append("missing_state_to_input_mapping")
            mapping_passed = False
            condition_number = None
            relative_residual = None
        else:
            condition_number = _finite_float(mapping.get("condition_number"))
            relative_residual = _finite_float(mapping.get("relative_residual"))
            mapping_passed = bool(mapping.get("passed", False)) or not cfg.require_mapping_passed
            if cfg.require_mapping_passed and not bool(mapping.get("artifact_passed", False)):
                blockers.append("mapping_artifact_failed")
            if cfg.require_mapping_passed and not bool(mapping.get("passed", False)):
                if "mapping_artifact_failed" not in blockers:
                    blockers.append("mapping_artifact_failed")
            if condition_number is None:
                blockers.append("missing_mapping_condition_number")
            elif condition_number > cfg.max_mapping_condition_number:
                blockers.append("mapping_condition_number_too_large")
            if relative_residual is None:
                blockers.append("missing_mapping_relative_residual")
            elif relative_residual > cfg.max_mapping_relative_residual:
                blockers.append("mapping_relative_residual_too_large")

        input_control = None if mapping is None else mapping.get("input_control_argument")
        if mapping is not None and not input_control:
            blockers.append("missing_input_control_argument")
        mapped = bool(ql_screen_usable and mapping is not None and mapping_passed and not blockers and input_control)
        row = {
            "state_parameter": parameter,
            "state_control_argument": raw_control.get("state_control_argument"),
            "descent_direction_sign": raw_control.get("descent_direction_sign"),
            "mapping_ready": mapped,
            "blockers": blockers,
            "input_control_argument": input_control,
            "input_direction": None if mapping is None else mapping.get("input_direction"),
            "input_parameter": None if mapping is None else mapping.get("input_parameter"),
            "condition_number": _json_number(condition_number),
            "relative_residual": _json_number(relative_residual),
            "short_bracket_command_fragment": None
            if not mapped
            else (
                f"--control {input_control} "
                f"--relative-delta {cfg.default_relative_delta:.12g}"
            ),
        }
        if mapped:
            mapped_controls.append(row)
        control_rows.append(row)

    passed = len(mapped_controls) >= cfg.min_mapped_controls
    if passed:
        next_action = (
            "write checked short-bracket nonlinear-gradient launch manifests for mapped VMEC input directions"
        )
    elif control_rows:
        next_action = (
            "build a VMEC-state-to-input mapping artifact before launching nonlinear-gradient campaigns"
        )
    else:
        next_action = "run the QL seed screen first; no admitted VMEC-state controls were provided"

    return {
        "kind": "nonlinear_gradient_state_control_runbook",
        "claim_level": "state_to_input_mapping_gate_not_nonlinear_gradient_evidence",
        "case": case,
        "passed": bool(passed),
        "next_action": next_action,
        "config": asdict(cfg),
        "summary": {
            "admitted_state_control_count": len(control_rows),
            "mapped_control_count": len(mapped_controls),
            "required_mapped_control_count": cfg.min_mapped_controls,
            "mapping_artifact_count": len(mapping_artifacts),
            "ql_seed_screen_usable": bool(ql_screen_usable),
        },
        "mapped_controls": mapped_controls,
        "controls": control_rows,
        "mapping_protocol": [
            "select perturbable VMEC input coefficients or profile directions",
            "solve baseline/plus/minus equilibria with vmec_jax",
            "measure the induced VMEC-state response in the admitted state-control basis",
            "accept the mapping only if the local Jacobian is conditioned and residual-bounded",
            "only then launch checked short-bracket nonlinear-gradient screens",
        ],
        "scope_note": (
            "A passed QL seed screen is upstream evidence only. This runbook "
            "requires an explicit state-to-input mapping before any long-window "
            "nonlinear turbulence-gradient or optimization claim."
        ),
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
