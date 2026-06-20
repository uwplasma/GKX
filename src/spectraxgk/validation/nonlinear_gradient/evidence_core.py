"""Core helpers for nonlinear turbulence-gradient evidence gates.

The public :mod:`spectraxgk.validation.nonlinear_gradient.evidence` facade owns artifact
report assembly.  This module keeps the small, deterministic claim-boundary
pieces separate so they can be tested and reused without importing the full
reporting layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence
import math
import re

NON_PRODUCTION_SCOPE_MARKERS = (
    "startup",
    "plumbing",
    "reduced",
    "estimator",
    "smooth_logistic",
    "mixing_length",
    "not_transport",
    "not transport",
    "not_production",
    "not production",
    "not_simulation_claim",
    "not simulation claim",
    "feasibility",
    "pilot",
    "pending",
)

PRODUCTION_SCOPE_MARKERS = (
    "production_long_window",
    "production long-window",
    "long_window_nonlinear_turbulence_gradient",
    "long-window nonlinear turbulence gradient",
    "production nonlinear window gradient",
)


@dataclass(frozen=True)
class NonlinearTurbulenceGradientEvidenceConfig:
    """Acceptance limits for production nonlinear turbulence-gradient evidence."""

    min_window_reports: int = 2
    max_window_mean_rel_spread: float = 0.15
    max_window_combined_sem_rel: float = 0.25
    max_gradient_uncertainty_rel: float = 0.50
    max_fd_asymmetry_rel: float = 0.50
    max_fd_condition_number: float = 1.0e8
    min_fd_response_fraction: float = 0.03
    value_floor: float = 1.0e-12


@dataclass(frozen=True)
class NonlinearTurbulenceGradientGapConfig:
    """Default campaign shape required before promoting turbulence gradients."""

    case_slug: str = "optimized_equilibrium_turbulence_gradient"
    parameter_name: str = "vmec_state_control_or_profile_gradient"
    perturbation_fraction: float = 0.05
    t_start: float = 0.0
    analysis_tmin: float = 350.0
    analysis_tmax: float = 700.0
    minimum_tmax: float = 700.0
    minimum_grid: str = "n64x64x64x40x40"
    replicate_labels: tuple[str, ...] = ("seed31", "seed32", "dt0p04")


@dataclass(frozen=True)
class NonlinearTurbulenceGradientFiniteDifferenceConfig:
    """Acceptance limits for paired long-window finite-difference gradients."""

    min_window_reports: int = 2
    max_window_mean_rel_spread: float = 0.15
    max_window_combined_sem_rel: float = 0.25
    max_gradient_uncertainty_rel: float = 0.50
    max_fd_asymmetry_rel: float = 0.50
    max_fd_condition_number: float = 1.0e8
    min_fd_response_fraction: float = 0.03
    value_floor: float = 1.0e-12


@dataclass(frozen=True)
class NonlinearTurbulenceGradientCandidateRankingConfig:
    """Scoring limits used to rank failed nonlinear-gradient control candidates."""

    max_gradient_uncertainty_rel: float = 0.50
    max_fd_asymmetry_rel: float = 0.50
    max_fd_condition_number: float = 1.0e8
    min_fd_response_fraction: float = 0.03
    score_cap: float = 2.0
    value_floor: float = 1.0e-12
    campaign_context: str = "single_control_screen"


@dataclass(frozen=True)
class NonlinearTurbulenceGradientBracketSweepConfig:
    """Decision limits for same-control perturbation-amplitude sweeps."""

    max_gradient_uncertainty_rel: float = 0.50
    max_fd_asymmetry_rel: float = 0.50
    max_fd_condition_number: float = 1.0e8
    min_fd_response_fraction: float = 0.03
    max_repeated_bracket_uncertainty_rel: float = 0.75
    min_repeated_bracket_same_sign_fraction: float = 0.80
    score_cap: float = 2.0
    value_floor: float = 1.0e-12


@dataclass(frozen=True)
class _GradientConditioningMetrics:
    """Canonical metrics extracted from one nonlinear-gradient evidence artifact."""

    derivative: float | None
    response_fraction: float | None
    asymmetry: float | None
    condition_number: float | None
    uncertainty_rel: float | None


def _json_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if isinstance(value, int):
        return value
    return number


def _finite_float(value: Any) -> float | None:
    number = _json_number(value)
    return None if number is None else float(number)


def _gate(metric: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"metric": metric, "passed": bool(passed), "detail": str(detail)}


def _artifact_passed(payload: dict[str, Any]) -> bool:
    if bool(payload.get("passed", False)):
        return True
    for key in ("gate_report", "promotion_gate"):
        nested = payload.get(key)
        if isinstance(nested, dict) and bool(nested.get("passed", False)):
            return True
    return False


def _ensemble_statistics_row(payload: dict[str, Any], *, path: str | None = None) -> dict[str, Any]:
    statistics = payload.get("statistics")
    if not isinstance(statistics, dict):
        statistics = {}
    return {
        "path": path,
        "kind": str(payload.get("kind", "")),
        "case": str(payload.get("case", "")),
        "passed": _artifact_passed(payload),
        "ensemble_mean": _json_number(statistics.get("ensemble_mean")),
        "combined_sem": _json_number(statistics.get("combined_sem")),
        "combined_sem_rel": _json_number(statistics.get("combined_sem_rel")),
        "mean_rel_spread": _json_number(statistics.get("mean_rel_spread")),
        "n_reports": _json_number(statistics.get("n_reports")),
        "statistics": statistics,
        "rows": payload.get("rows", []) if isinstance(payload.get("rows"), list) else [],
    }


def _replicate_label_from_row(row: dict[str, Any]) -> str | None:
    for key in ("variant_label", "source_artifact", "summary_artifact", "path"):
        value = row.get(key)
        if not isinstance(value, str):
            continue
        match = re.search(r"(seed[0-9]+|dt[0-9]+(?:p[0-9]+)?)", value)
        if match:
            return match.group(1)
    return None


def _late_mean_by_replicate(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    rows = row.get("rows")
    if not isinstance(rows, list):
        return out
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        label = _replicate_label_from_row(entry)
        value = _finite_float(entry.get("late_mean"))
        if label is None or value is None:
            continue
        out[label] = float(value)
    return out


def _paired_replicate_fd_diagnostics(
    *,
    rows: dict[str, dict[str, Any]],
    delta: float,
    value_floor: float,
) -> dict[str, Any]:
    minus_by_label = _late_mean_by_replicate(rows["minus"])
    baseline_by_label = _late_mean_by_replicate(rows["baseline"])
    plus_by_label = _late_mean_by_replicate(rows["plus"])
    labels = sorted(set(minus_by_label) & set(plus_by_label))
    pair_rows: list[dict[str, Any]] = []
    gradients: list[float] = []
    responses: list[float] = []
    for label in labels:
        minus_value = minus_by_label[label]
        plus_value = plus_by_label[label]
        response = plus_value - minus_value
        gradient = response / (2.0 * delta)
        pair = {
            "label": label,
            "minus_late_mean": _json_number(minus_value),
            "plus_late_mean": _json_number(plus_value),
            "response": _json_number(response),
            "central_gradient": _json_number(gradient),
        }
        if label in baseline_by_label:
            baseline_value = baseline_by_label[label]
            forward_gradient = (plus_value - baseline_value) / delta
            backward_gradient = (baseline_value - minus_value) / delta
            pair["baseline_late_mean"] = _json_number(baseline_value)
            pair["forward_gradient"] = _json_number(forward_gradient)
            pair["backward_gradient"] = _json_number(backward_gradient)
            pair["fd_asymmetry_rel"] = _json_number(
                abs(forward_gradient - backward_gradient)
                / max(abs(gradient), float(value_floor))
            )
        pair_rows.append(pair)
        gradients.append(float(gradient))
        responses.append(float(response))

    gradient_mean = math.nan
    gradient_sample_sem = math.nan
    gradient_uncertainty_rel = math.nan
    same_sign_fraction = math.nan
    if gradients:
        gradient_mean = float(sum(gradients) / len(gradients))
        signs = [math.copysign(1.0, value) for value in gradients if value != 0.0]
        if signs:
            positive = sum(1 for value in signs if value > 0.0)
            negative = len(signs) - positive
            same_sign_fraction = max(positive, negative) / len(signs)
        if len(gradients) >= 2:
            variance = sum((value - gradient_mean) ** 2 for value in gradients) / (
                len(gradients) - 1
            )
            gradient_sample_sem = math.sqrt(variance / len(gradients))
            gradient_uncertainty_rel = gradient_sample_sem / max(
                abs(gradient_mean),
                float(value_floor),
            )

    return {
        "claim_level": "diagnostic_only_not_a_production_gate",
        "common_plus_minus_labels": labels,
        "common_all_state_labels": sorted(
            set(minus_by_label) & set(baseline_by_label) & set(plus_by_label)
        ),
        "n_pairs": len(pair_rows),
        "paired_rows": pair_rows,
        "central_gradient_mean": _json_number(gradient_mean),
        "central_gradient_sample_sem": _json_number(gradient_sample_sem),
        "central_gradient_uncertainty_rel": _json_number(gradient_uncertainty_rel),
        "same_sign_fraction": _json_number(same_sign_fraction),
        "mean_response": _json_number(
            sum(responses) / len(responses) if responses else math.nan
        ),
    }


def _claim_text(payload: dict[str, Any]) -> str:
    parts = [
        str(payload.get(key, ""))
        for key in (
            "kind",
            "claim_level",
            "claim_scope",
            "case",
            "notes",
            "next_action",
        )
    ]
    return " ".join(parts).lower()


def _scope_blockers(payload: dict[str, Any]) -> list[str]:
    text = _claim_text(payload)
    blockers = [marker for marker in NON_PRODUCTION_SCOPE_MARKERS if marker in text]
    if payload.get("transport_average_gate") is False:
        blockers.append("transport_average_gate_false")
    return sorted(set(blockers))


def _explicit_production_scope(payload: dict[str, Any]) -> bool:
    if bool(payload.get("production_nonlinear_window_gradient_gate", False)):
        return True
    if bool(payload.get("nonlinear_turbulence_gradient_gate", False)):
        return True
    text = _claim_text(payload)
    return any(marker in text for marker in PRODUCTION_SCOPE_MARKERS)


def _nested_dict(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _first_finite(payloads: Iterable[dict[str, Any]], keys: Sequence[str]) -> float | None:
    for payload in payloads:
        for key in keys:
            value = _finite_float(payload.get(key))
            if value is not None:
                return value
    return None


def _objective_gate_values(payload: dict[str, Any]) -> list[float]:
    values: list[float] = []
    raw = payload.get("objective_gates")
    if not isinstance(raw, list):
        return values
    for row in raw:
        if not isinstance(row, dict):
            continue
        for key in ("finite_difference", "implicit"):
            value = _finite_float(row.get(key))
            if value is not None:
                values.append(value)
    return values


def _gradient_conditioning_candidates(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return nested dictionaries that may carry nonlinear-gradient metrics."""

    return (
        _nested_dict(payload, "gradient", "gradient_summary"),
        _nested_dict(payload, "metrics"),
        _nested_dict(
            payload,
            "conditioning",
            "conditioning_gate",
            "finite_difference_conditioning",
            "gradient_conditioning",
        ),
        _nested_dict(
            payload,
            "uncertainty",
            "uncertainty_gate",
            "gradient_uncertainty",
        ),
        payload,
    )


def _extract_gradient_conditioning_metrics(
    payload: dict[str, Any],
) -> _GradientConditioningMetrics:
    """Extract the canonical gate metrics from known artifact schema variants."""

    candidates = _gradient_conditioning_candidates(payload)
    derivative = _first_finite(
        candidates,
        (
            "central",
            "central_gradient",
            "central_fd",
            "central_fd_dq_dparameter",
            "central_fd_dq_dtprim",
            "finite_difference",
            "gradient",
        ),
    )
    objective_values = _objective_gate_values(payload)
    if derivative is None and objective_values:
        derivative = objective_values[0]

    return _GradientConditioningMetrics(
        derivative=derivative,
        response_fraction=_first_finite(
            candidates,
            (
                "response_fraction",
                "resolved_response_fraction",
                "fd_response_fraction",
            ),
        ),
        asymmetry=_first_finite(
            candidates,
            (
                "asymmetry_rel",
                "derivative_asymmetry",
                "fd_asymmetry_rel",
            ),
        ),
        condition_number=_first_finite(
            candidates,
            (
                "condition_number",
                "fd_condition_number",
                "sensitivity_condition_number",
            ),
        ),
        uncertainty_rel=_first_finite(
            candidates,
            (
                "gradient_sem_rel",
                "sem_rel",
                "gradient_uncertainty_rel",
                "gradient_relative_uncertainty",
                "relative_uncertainty",
            ),
        ),
    )


def _gradient_conditioning_gates(
    metrics: _GradientConditioningMetrics,
    *,
    config: NonlinearTurbulenceGradientEvidenceConfig,
) -> list[dict[str, Any]]:
    """Build production nonlinear-gradient conditioning gate rows."""

    return [
        _gate(
            "finite_gradient_estimate",
            metrics.derivative is not None,
            f"central_gradient={metrics.derivative}",
        ),
        _gate(
            "fd_response_resolved",
            metrics.response_fraction is not None
            and metrics.response_fraction >= float(config.min_fd_response_fraction),
            "response_fraction={value} min={gate}".format(
                value=metrics.response_fraction,
                gate=config.min_fd_response_fraction,
            ),
        ),
        _gate(
            "fd_asymmetry_bounded",
            metrics.asymmetry is not None
            and metrics.asymmetry <= float(config.max_fd_asymmetry_rel),
            "fd_asymmetry_rel={value} max={gate}".format(
                value=metrics.asymmetry,
                gate=config.max_fd_asymmetry_rel,
            ),
        ),
        _gate(
            "fd_condition_number_bounded",
            metrics.condition_number is not None
            and metrics.condition_number <= float(config.max_fd_condition_number),
            "condition_number={value} max={gate}".format(
                value=metrics.condition_number,
                gate=config.max_fd_condition_number,
            ),
        ),
        _gate(
            "gradient_uncertainty_bounded",
            metrics.uncertainty_rel is not None
            and metrics.uncertainty_rel <= float(config.max_gradient_uncertainty_rel),
            "gradient_uncertainty_rel={value} max={gate}".format(
                value=metrics.uncertainty_rel,
                gate=config.max_gradient_uncertainty_rel,
            ),
        ),
    ]


def _gradient_conditioning_payload(
    metrics: _GradientConditioningMetrics,
    gates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pack nonlinear-gradient metric extraction and gates into the public schema."""

    return {
        "central_gradient": _json_number(metrics.derivative),
        "response_fraction": _json_number(metrics.response_fraction),
        "fd_asymmetry_rel": _json_number(metrics.asymmetry),
        "fd_condition_number": _json_number(metrics.condition_number),
        "gradient_uncertainty_rel": _json_number(metrics.uncertainty_rel),
        "gates": gates,
        "passed": all(bool(gate["passed"]) for gate in gates),
    }


def _gradient_conditioning_summary(
    payload: dict[str, Any],
    *,
    config: NonlinearTurbulenceGradientEvidenceConfig,
) -> dict[str, Any]:
    metrics = _extract_gradient_conditioning_metrics(payload)
    gates = _gradient_conditioning_gates(metrics, config=config)
    return _gradient_conditioning_payload(metrics, gates)


__all__ = [
    "NON_PRODUCTION_SCOPE_MARKERS",
    "PRODUCTION_SCOPE_MARKERS",
    "NonlinearTurbulenceGradientBracketSweepConfig",
    "NonlinearTurbulenceGradientCandidateRankingConfig",
    "NonlinearTurbulenceGradientEvidenceConfig",
    "NonlinearTurbulenceGradientFiniteDifferenceConfig",
    "NonlinearTurbulenceGradientGapConfig",
    "_artifact_passed",
    "_claim_text",
    "_ensemble_statistics_row",
    "_explicit_production_scope",
    "_finite_float",
    "_first_finite",
    "_gate",
    "_gradient_conditioning_summary",
    "_json_number",
    "_late_mean_by_replicate",
    "_nested_dict",
    "_objective_gate_values",
    "_paired_replicate_fd_diagnostics",
    "_replicate_label_from_row",
    "_scope_blockers",
]
