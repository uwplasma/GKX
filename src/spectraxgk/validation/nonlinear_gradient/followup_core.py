"""Shared configuration and statistics helpers for nonlinear-gradient follow-up.

This module is intentionally side-effect free. Public campaign planners remain
in ``spectraxgk.validation.nonlinear_gradient.followup`` while reusable config, JSON
sanitization, replicate parsing, and variance-reduction helpers live here for
unit testing and future planner splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


__all__ = [
    "STATE_TO_RUN_STATE",
    "NonlinearGradientCandidateDesignConfig",
    "NonlinearGradientCompositeControlConfig",
    "NonlinearGradientControlMeanGateConfig",
    "NonlinearGradientControlVariateCampaignConfig",
    "NonlinearGradientFollowupConfig",
    "NonlinearGradientQLSeedScreenConfig",
    "NonlinearGradientStateControlRunbookConfig",
    "NonlinearGradientVarianceReductionConfig",
    "_artifact_passed",
    "_coefficient_label_from_parameter",
    "_control_variate_candidate",
    "_ensemble_state_variance_report",
    "_ensemble_stats_value",
    "_finite_float",
    "_finite_int",
    "_json_number",
    "_label_from_row",
    "_mean_and_sem",
    "_metric",
    "_nested_metric",
    "_replicate_count",
    "_sample_covariance",
    "_seed_numbers",
    "_state_control_family",
    "_state_means_by_label",
]
