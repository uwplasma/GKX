"""Nonlinear turbulent-gradient evidence and claim-boundary diagnostics.

This module contains pure JSON/dictionary diagnostics used to decide whether
long-window nonlinear transport-gradient artifacts support production claims.
It does not launch simulations. Campaign design and follow-up planning live in
``tools.campaigns.nonlinear_gradient_followup``.
"""

from __future__ import annotations

# ---- shared parsing and gate primitives ----
"""Core helpers for nonlinear turbulence-gradient evidence gates.

The public :mod:`spectraxgk.diagnostics.nonlinear_gradient_evidence` facade owns artifact
report assembly.  This module keeps the small, deterministic claim-boundary
pieces separate so they can be tested and reused without importing the full
reporting layer.
"""


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

# ---- artifact classification helpers ----
"""Classification helpers for nonlinear turbulence-gradient evidence artifacts."""


from typing import Any



def classify_gradient_artifact(
    payload: dict[str, Any],
    *,
    path: str | None = None,
    config: NonlinearTurbulenceGradientEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Classify a gradient/FD artifact without promoting ambiguous evidence."""

    cfg = config or NonlinearTurbulenceGradientEvidenceConfig()
    kind = str(payload.get("kind", ""))
    blockers = _scope_blockers(payload)
    explicit_production = _explicit_production_scope(payload)
    conditioning = _gradient_conditioning_summary(payload, config=cfg)
    passed = _artifact_passed(payload)
    production_scope = bool(explicit_production and not blockers)
    qualifies = bool(passed and production_scope and conditioning["passed"])

    if blockers:
        evidence_class = "startup_or_reduced_window_fd_not_production"
    elif explicit_production:
        evidence_class = "production_long_window_turbulence_gradient_candidate"
    else:
        evidence_class = "unscoped_gradient_or_fd_artifact_not_production"

    gates = [
        _gate("artifact_passed", passed, f"kind={kind}"),
        _gate(
            "explicit_production_long_window_scope",
            production_scope,
            "explicit_production_scope={scope} scope_blockers={blockers}".format(
                scope=explicit_production,
                blockers=blockers,
            ),
        ),
        *conditioning["gates"],
    ]
    return {
        "path": path,
        "kind": kind,
        "claim_level": str(payload.get("claim_level", "")),
        "claim_scope": str(payload.get("claim_scope", "")),
        "evidence_class": evidence_class,
        "artifact_passed": passed,
        "explicit_production_scope": explicit_production,
        "scope_blockers": blockers,
        "conditioning": {
            key: value for key, value in conditioning.items() if key not in {"gates"}
        },
        "gates": gates,
        "qualifies_for_production_turbulence_gradient": qualifies,
    }


__all__ = ["classify_gradient_artifact"]

# ---- replicated window summaries ----
"""Replicated nonlinear-window evidence summaries for turbulence-gradient gates."""


from typing import Any, Sequence

from spectraxgk.diagnostics.transport_windows import (
    NonlinearWindowEnsembleConfig,
)
from spectraxgk.diagnostics.transport_windows import (
    nonlinear_window_ensemble_report,
)
from spectraxgk.diagnostics.transport_windows import (
    nonlinear_window_stats_promotion_ready,
)


def _ensemble_row(
    payload: dict[str, Any],
    *,
    path: str | None,
    source: str,
    config: NonlinearTurbulenceGradientEvidenceConfig,
) -> dict[str, Any]:
    statistics = payload.get("statistics")
    if not isinstance(statistics, dict):
        statistics = {}
    n_reports = _finite_float(statistics.get("n_reports"))
    combined_sem_rel = _finite_float(statistics.get("combined_sem_rel"))
    mean_rel_spread = _finite_float(statistics.get("mean_rel_spread"))
    passed = _artifact_passed(payload)
    qualifies = bool(
        passed
        and n_reports is not None
        and int(n_reports) >= int(config.min_window_reports)
        and combined_sem_rel is not None
        and combined_sem_rel <= float(config.max_window_combined_sem_rel)
        and mean_rel_spread is not None
        and mean_rel_spread <= float(config.max_window_mean_rel_spread)
    )
    return {
        "path": path,
        "source": source,
        "kind": str(payload.get("kind", "")),
        "passed": passed,
        "n_reports": None if n_reports is None else int(n_reports),
        "combined_sem_rel": _json_number(combined_sem_rel),
        "mean_rel_spread": _json_number(mean_rel_spread),
        "qualifies_for_replicated_long_window_uncertainty": qualifies,
        "statistics": statistics,
    }


def _single_window_row(
    payload: dict[str, Any],
    *,
    path: str | None,
    kind: str,
) -> dict[str, Any]:
    """Return a row for one convergence-window summary."""

    ready, failures = nonlinear_window_stats_promotion_ready(payload)
    return {
        "path": path,
        "kind": kind,
        "case": str(payload.get("case", "")),
        "passed": _artifact_passed(payload),
        "promotion_ready": ready,
        "failures": failures,
    }


def _unsupported_window_row(
    payload: dict[str, Any],
    *,
    path: str | None,
    kind: str,
) -> dict[str, Any]:
    """Return a non-qualifying row for an unsupported window artifact."""

    return {
        "path": path,
        "source": "unsupported_window_artifact",
        "kind": kind,
        "passed": _artifact_passed(payload),
        "qualifies_for_replicated_long_window_uncertainty": False,
    }


def _collect_window_artifact_rows(
    window_artifacts: Sequence[dict[str, Any]],
    path_list: Sequence[str | None],
    *,
    config: NonlinearTurbulenceGradientEvidenceConfig,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str | None],
]:
    """Classify input window artifacts into ensemble and convergence rows."""

    rows: list[dict[str, Any]] = []
    convergence_reports: list[dict[str, Any]] = []
    convergence_paths: list[str | None] = []
    single_window_rows: list[dict[str, Any]] = []

    for payload, path in zip(window_artifacts, path_list):
        kind = str(payload.get("kind", ""))
        if kind == "nonlinear_window_ensemble_report":
            rows.append(
                _ensemble_row(
                    payload,
                    path=path,
                    source="input_ensemble",
                    config=config,
                )
            )
        elif kind == "nonlinear_window_convergence_report":
            single_window_rows.append(
                _single_window_row(payload, path=path, kind=kind)
            )
            convergence_reports.append(payload)
            convergence_paths.append(path)
        else:
            rows.append(_unsupported_window_row(payload, path=path, kind=kind))
    return rows, single_window_rows, convergence_reports, convergence_paths


def _derived_ensemble_row(
    convergence_reports: Sequence[dict[str, Any]],
    convergence_paths: Sequence[str | None],
    *,
    config: NonlinearTurbulenceGradientEvidenceConfig,
) -> dict[str, Any] | None:
    """Build a replicated-window ensemble row from individual windows."""

    if len(convergence_reports) < int(config.min_window_reports):
        return None
    derived_payload = nonlinear_window_ensemble_report(
        convergence_reports,
        case="derived_long_window_replicate_evidence",
        comparison="derived_from_supplied_window_summaries",
        config=NonlinearWindowEnsembleConfig(
            min_reports=config.min_window_reports,
            max_mean_rel_spread=config.max_window_mean_rel_spread,
            max_combined_sem_rel=config.max_window_combined_sem_rel,
            value_floor=config.value_floor,
            require_individual_passed=True,
        ),
    )
    row = _ensemble_row(
        derived_payload,
        path=None,
        source="derived_from_window_summaries",
        config=config,
    )
    row["input_paths"] = list(convergence_paths)
    return row


def _qualifying_window_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ensemble rows that qualify as replicated long-window evidence."""

    return [
        row
        for row in rows
        if bool(row.get("qualifies_for_replicated_long_window_uncertainty", False))
    ]


def _window_evidence_gates(
    qualifying_rows: Sequence[dict[str, Any]],
    *,
    config: NonlinearTurbulenceGradientEvidenceConfig,
) -> list[dict[str, Any]]:
    """Return the gate list for replicated long-window uncertainty evidence."""

    return [
        _gate(
            "replicated_long_window_uncertainty",
            bool(qualifying_rows),
            "qualifying_ensembles={count} min_window_reports={min_reports}".format(
                count=len(qualifying_rows),
                min_reports=config.min_window_reports,
            ),
        )
    ]


def summarize_window_evidence(
    window_artifacts: Sequence[dict[str, Any]],
    *,
    paths: Sequence[str | None] | None = None,
    config: NonlinearTurbulenceGradientEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Summarize replicated long-window uncertainty evidence.

    Existing ``nonlinear_window_ensemble_report`` artifacts are consumed
    directly.  If only individual ``nonlinear_window_convergence_report``
    summaries are supplied, a derived ensemble is built from those summaries
    using the configured uncertainty limits.
    """

    cfg = config or NonlinearTurbulenceGradientEvidenceConfig()
    path_list = list(paths or [None] * len(window_artifacts))
    if len(path_list) != len(window_artifacts):
        raise ValueError("paths length must match window_artifacts length")

    rows, single_window_rows, convergence_reports, convergence_paths = (
        _collect_window_artifact_rows(window_artifacts, path_list, config=cfg)
    )
    derived_ensemble = _derived_ensemble_row(
        convergence_reports,
        convergence_paths,
        config=cfg,
    )
    if derived_ensemble is not None:
        rows.append(derived_ensemble)

    qualifying_rows = _qualifying_window_rows(rows)
    gates = _window_evidence_gates(qualifying_rows, config=cfg)
    return {
        "passed": bool(qualifying_rows),
        "gates": gates,
        "ensemble_rows": rows,
        "single_window_rows": single_window_rows,
        "derived_ensemble": derived_ensemble,
    }


__all__ = ["_ensemble_row", "summarize_window_evidence"]

# ---- finite-difference evidence reports ----
"""Finite-difference turbulence-gradient evidence reports."""


from dataclasses import asdict, dataclass
from typing import Any
import math



@dataclass(frozen=True)
class _FiniteDifferenceMetrics:
    central_gradient: float
    forward_gradient: float
    backward_gradient: float
    response: float
    response_fraction: float
    fd_asymmetry_rel: float
    fd_condition_number: float
    gradient_uncertainty: float
    gradient_uncertainty_rel: float


def _validated_fd_inputs(
    config: NonlinearTurbulenceGradientFiniteDifferenceConfig | None,
    delta_parameter: float,
) -> tuple[NonlinearTurbulenceGradientFiniteDifferenceConfig, float]:
    cfg = config or NonlinearTurbulenceGradientFiniteDifferenceConfig()
    delta = float(delta_parameter)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta_parameter must be finite and positive")
    return cfg, delta


def _window_rows(
    *,
    baseline: dict[str, Any],
    plus: dict[str, Any],
    minus: dict[str, Any],
    baseline_path: str | None,
    plus_path: str | None,
    minus_path: str | None,
) -> dict[str, dict[str, Any]]:
    return {
        "minus": _ensemble_statistics_row(minus, path=minus_path),
        "baseline": _ensemble_statistics_row(baseline, path=baseline_path),
        "plus": _ensemble_statistics_row(plus, path=plus_path),
    }


def _window_mean_sem(
    rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    means = {
        name: _finite_float(row.get("ensemble_mean")) for name, row in rows.items()
    }
    sems = {name: _finite_float(row.get("combined_sem")) for name, row in rows.items()}
    return means, sems


def _required_float(values: dict[str, float | None], key: str) -> float:
    value = values[key]
    assert value is not None
    return float(value)


def _fd_transport_response(
    *,
    means: dict[str, float | None],
    delta: float,
    value_floor: float,
) -> tuple[float, float, float, float, float, float, float]:
    finite_means = all(value is not None for value in means.values())
    if not finite_means:
        return (math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
    minus_mean = _required_float(means, "minus")
    baseline_mean = _required_float(means, "baseline")
    plus_mean = _required_float(means, "plus")
    central_gradient = (plus_mean - minus_mean) / (2.0 * delta)
    forward_gradient = (plus_mean - baseline_mean) / delta
    backward_gradient = (baseline_mean - minus_mean) / delta
    response = abs(plus_mean - minus_mean)
    response_fraction = response / max(abs(baseline_mean), value_floor)
    fd_asymmetry_rel = abs(forward_gradient - backward_gradient) / max(
        abs(central_gradient),
        value_floor,
    )
    fd_condition_number = (abs(plus_mean) + abs(minus_mean)) / max(
        response,
        value_floor,
    )
    return (
        central_gradient,
        forward_gradient,
        backward_gradient,
        response,
        response_fraction,
        fd_asymmetry_rel,
        fd_condition_number,
    )


def _fd_gradient_uncertainty(
    *,
    sems: dict[str, float | None],
    central_gradient: float,
    delta: float,
    value_floor: float,
) -> tuple[float, float]:
    finite_sems = all(value is not None for value in sems.values())
    if not finite_sems:
        return math.nan, math.nan
    gradient_uncertainty = math.sqrt(
        _required_float(sems, "plus") ** 2 + _required_float(sems, "minus") ** 2
    ) / (2.0 * delta)
    gradient_uncertainty_rel = gradient_uncertainty / max(
        abs(central_gradient) if math.isfinite(central_gradient) else 0.0,
        value_floor,
    )
    return gradient_uncertainty, gradient_uncertainty_rel


def _finite_difference_metrics(
    *,
    means: dict[str, float | None],
    sems: dict[str, float | None],
    delta: float,
    cfg: NonlinearTurbulenceGradientFiniteDifferenceConfig,
) -> _FiniteDifferenceMetrics:
    value_floor = float(cfg.value_floor)
    (
        central_gradient,
        forward_gradient,
        backward_gradient,
        response,
        response_fraction,
        fd_asymmetry_rel,
        fd_condition_number,
    ) = _fd_transport_response(
        means=means,
        delta=delta,
        value_floor=value_floor,
    )
    gradient_uncertainty, gradient_uncertainty_rel = _fd_gradient_uncertainty(
        sems=sems,
        central_gradient=central_gradient,
        delta=delta,
        value_floor=value_floor,
    )
    return _FiniteDifferenceMetrics(
        central_gradient=central_gradient,
        forward_gradient=forward_gradient,
        backward_gradient=backward_gradient,
        response=response,
        response_fraction=response_fraction,
        fd_asymmetry_rel=fd_asymmetry_rel,
        fd_condition_number=fd_condition_number,
        gradient_uncertainty=gradient_uncertainty,
        gradient_uncertainty_rel=gradient_uncertainty_rel,
    )


def _source_ensemble_gates(
    rows: dict[str, dict[str, Any]],
    cfg: NonlinearTurbulenceGradientFiniteDifferenceConfig,
) -> list[dict[str, Any]]:
    source_gates: list[dict[str, Any]] = []
    for name, row in rows.items():
        n_reports = _finite_float(row.get("n_reports"))
        source_gates.extend(
            [
                _gate(
                    f"{name}_ensemble_kind",
                    row.get("kind") == "nonlinear_window_ensemble_report",
                    f"kind={row.get('kind')}",
                ),
                _gate(
                    f"{name}_ensemble_passed",
                    bool(row["passed"]),
                    f"path={row.get('path')}",
                ),
                _gate(
                    f"{name}_ensemble_replicated",
                    n_reports is not None and n_reports >= int(cfg.min_window_reports),
                    f"n_reports={n_reports} min={cfg.min_window_reports}",
                ),
            ]
        )
    return source_gates


def _window_quality_gates(
    rows: dict[str, dict[str, Any]],
    cfg: NonlinearTurbulenceGradientFiniteDifferenceConfig,
) -> list[dict[str, Any]]:
    window_gates: list[dict[str, Any]] = []
    for name, row in rows.items():
        mean_rel_spread = _finite_float(row.get("mean_rel_spread"))
        combined_sem_rel = _finite_float(row.get("combined_sem_rel"))
        window_gates.extend(
            [
                _gate(
                    f"{name}_window_mean_spread",
                    mean_rel_spread is not None
                    and mean_rel_spread <= float(cfg.max_window_mean_rel_spread),
                    f"mean_rel_spread={mean_rel_spread} max={cfg.max_window_mean_rel_spread}",
                ),
                _gate(
                    f"{name}_window_sem",
                    combined_sem_rel is not None
                    and combined_sem_rel <= float(cfg.max_window_combined_sem_rel),
                    f"combined_sem_rel={combined_sem_rel} max={cfg.max_window_combined_sem_rel}",
                ),
            ]
        )
    return window_gates


def _gradient_resolution_gates(
    *,
    means: dict[str, float | None],
    sems: dict[str, float | None],
    metrics: _FiniteDifferenceMetrics,
    cfg: NonlinearTurbulenceGradientFiniteDifferenceConfig,
) -> list[dict[str, Any]]:
    finite_means = all(value is not None for value in means.values())
    finite_sems = all(value is not None for value in sems.values())
    gradient_gates = [
        _gate("finite_window_means", finite_means, f"means={means}"),
        _gate("finite_window_uncertainties", finite_sems, f"combined_sem={sems}"),
        _gate(
            "fd_response_resolved",
            math.isfinite(metrics.response_fraction)
            and metrics.response_fraction >= float(cfg.min_fd_response_fraction),
            f"response_fraction={metrics.response_fraction} min={cfg.min_fd_response_fraction}",
        ),
        _gate(
            "fd_asymmetry_bounded",
            math.isfinite(metrics.fd_asymmetry_rel)
            and metrics.fd_asymmetry_rel <= float(cfg.max_fd_asymmetry_rel),
            f"fd_asymmetry_rel={metrics.fd_asymmetry_rel} max={cfg.max_fd_asymmetry_rel}",
        ),
        _gate(
            "fd_condition_number_bounded",
            math.isfinite(metrics.fd_condition_number)
            and metrics.fd_condition_number <= float(cfg.max_fd_condition_number),
            f"fd_condition_number={metrics.fd_condition_number} max={cfg.max_fd_condition_number}",
        ),
        _gate(
            "gradient_uncertainty_bounded",
            math.isfinite(metrics.gradient_uncertainty_rel)
            and metrics.gradient_uncertainty_rel
            <= float(cfg.max_gradient_uncertainty_rel),
            f"gradient_uncertainty_rel={metrics.gradient_uncertainty_rel} max={cfg.max_gradient_uncertainty_rel}",
        ),
    ]
    return gradient_gates


def _fd_metrics_payload(
    metrics: _FiniteDifferenceMetrics,
    *,
    means: dict[str, float | None],
    sems: dict[str, float | None],
) -> dict[str, Any]:
    return {
        "central_gradient": _json_number(metrics.central_gradient),
        "forward_gradient": _json_number(metrics.forward_gradient),
        "backward_gradient": _json_number(metrics.backward_gradient),
        "response": _json_number(metrics.response),
        "response_fraction": _json_number(metrics.response_fraction),
        "fd_asymmetry_rel": _json_number(metrics.fd_asymmetry_rel),
        "asymmetry_rel": _json_number(metrics.fd_asymmetry_rel),
        "fd_condition_number": _json_number(metrics.fd_condition_number),
        "condition_number": _json_number(metrics.fd_condition_number),
        "gradient_uncertainty": _json_number(metrics.gradient_uncertainty),
        "gradient_uncertainty_rel": _json_number(metrics.gradient_uncertainty_rel),
        "gradient_relative_uncertainty": _json_number(
            metrics.gradient_uncertainty_rel
        ),
        "baseline_window_mean": means["baseline"],
        "plus_window_mean": means["plus"],
        "minus_window_mean": means["minus"],
        "baseline_window_sem": sems["baseline"],
        "plus_window_sem": sems["plus"],
        "minus_window_sem": sems["minus"],
    }


def _finite_difference_gates(
    *,
    rows: dict[str, dict[str, Any]],
    means: dict[str, float | None],
    sems: dict[str, float | None],
    metrics: _FiniteDifferenceMetrics,
    cfg: NonlinearTurbulenceGradientFiniteDifferenceConfig,
) -> list[dict[str, Any]]:
    return [
        *_source_ensemble_gates(rows, cfg),
        *_window_quality_gates(rows, cfg),
        *_gradient_resolution_gates(
            means=means,
            sems=sems,
            metrics=metrics,
            cfg=cfg,
        ),
    ]


def _pack_finite_difference_report(
    *,
    parameter_name: str,
    delta: float,
    rows: dict[str, dict[str, Any]],
    means: dict[str, float | None],
    sems: dict[str, float | None],
    metrics: _FiniteDifferenceMetrics,
    gates: list[dict[str, Any]],
    cfg: NonlinearTurbulenceGradientFiniteDifferenceConfig,
) -> dict[str, Any]:
    passed = all(bool(gate["passed"]) for gate in gates)
    return {
        "kind": "nonlinear_turbulence_gradient_central_fd_gate",
        "claim_level": "production_long_window_nonlinear_turbulence_gradient_candidate",
        "claim_scope": (
            "production_long_window nonlinear turbulence gradient from matched replicated "
            "post-transient heat-flux windows"
        ),
        "parameter_name": str(parameter_name),
        "delta_parameter": delta,
        "passed": passed,
        "production_nonlinear_window_gradient_gate": passed,
        "nonlinear_turbulence_gradient_gate": passed,
        "metrics": _fd_metrics_payload(metrics, means=means, sems=sems),
        "source_ensembles": rows,
        "paired_replicate_diagnostics": _paired_replicate_fd_diagnostics(
            rows=rows,
            delta=delta,
            value_floor=float(cfg.value_floor),
        ),
        "config": asdict(cfg),
        "gates": gates,
        "blockers": [gate["metric"] for gate in gates if not bool(gate["passed"])],
    }


def nonlinear_turbulence_gradient_finite_difference_report(
    *,
    baseline: dict[str, Any],
    plus: dict[str, Any],
    minus: dict[str, Any],
    delta_parameter: float,
    parameter_name: str,
    baseline_path: str | None = None,
    plus_path: str | None = None,
    minus_path: str | None = None,
    config: NonlinearTurbulenceGradientFiniteDifferenceConfig | None = None,
) -> dict[str, Any]:
    """Build a production long-window central finite-difference gradient gate.

    Inputs must be replicated ``nonlinear_window_ensemble_report`` payloads for
    the same nonlinear case and analysis window, differing only by the perturbed
    parameter.  The report computes the central finite-difference heat-flux
    gradient and checks that the response is resolved above ensemble
    uncertainty before allowing any turbulence-gradient claim.
    """

    cfg, delta = _validated_fd_inputs(config, delta_parameter)
    rows = _window_rows(
        baseline=baseline,
        plus=plus,
        minus=minus,
        baseline_path=baseline_path,
        plus_path=plus_path,
        minus_path=minus_path,
    )
    means, sems = _window_mean_sem(rows)
    metrics = _finite_difference_metrics(
        means=means,
        sems=sems,
        delta=delta,
        cfg=cfg,
    )
    gates = _finite_difference_gates(
        rows=rows,
        means=means,
        sems=sems,
        metrics=metrics,
        cfg=cfg,
    )
    return _pack_finite_difference_report(
        parameter_name=parameter_name,
        delta=delta,
        rows=rows,
        means=means,
        sems=sems,
        metrics=metrics,
        gates=gates,
        cfg=cfg,
    )


__all__ = ["nonlinear_turbulence_gradient_finite_difference_report"]

# ---- candidate scoring helpers ----
"""Shared score-margin helpers for nonlinear-gradient screening reports."""


import math


def _metric_margin(
    value: float | None,
    *,
    target: float,
    sense: str,
    cap: float,
    value_floor: float,
) -> float:
    """Return a capped normalized evidence margin for one gate metric."""

    if value is None or not math.isfinite(float(value)):
        return 0.0
    finite_value = float(value)
    if sense == "min":
        margin = finite_value / max(float(target), float(value_floor))
    elif sense == "max":
        margin = float(target) / max(abs(finite_value), float(value_floor))
    else:  # pragma: no cover - guarded by internal call sites.
        raise ValueError(f"unsupported margin sense: {sense}")
    return max(0.0, min(float(cap), margin))


__all__ = ["_metric_margin"]

# ---- bracket sweep reports ----
"""Same-control bracket-sweep reports for nonlinear-gradient evidence."""


from dataclasses import asdict, dataclass
from typing import Any, Sequence
import math



def _paired_uncertainty_rel(artifact: dict[str, Any]) -> float | None:
    diagnostics = artifact.get("paired_replicate_diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    return _finite_float(diagnostics.get("central_gradient_uncertainty_rel"))


def _paired_same_sign_fraction(artifact: dict[str, Any]) -> float | None:
    diagnostics = artifact.get("paired_replicate_diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    return _finite_float(diagnostics.get("same_sign_fraction"))


@dataclass(frozen=True)
class _BracketConditioningMetrics:
    central_gradient: Any
    response_fraction: float | None
    fd_asymmetry_rel: float | None
    fd_condition_number: float | None
    gradient_uncertainty_rel: float | None
    paired_uncertainty_rel: float | None
    paired_same_sign_fraction: float | None


def _bracket_evidence_config(
    config: NonlinearTurbulenceGradientBracketSweepConfig,
) -> NonlinearTurbulenceGradientEvidenceConfig:
    return NonlinearTurbulenceGradientEvidenceConfig(
        max_gradient_uncertainty_rel=config.max_gradient_uncertainty_rel,
        max_fd_asymmetry_rel=config.max_fd_asymmetry_rel,
        max_fd_condition_number=config.max_fd_condition_number,
        min_fd_response_fraction=config.min_fd_response_fraction,
        value_floor=config.value_floor,
    )


def _bracket_conditioning_metrics(
    artifact: dict[str, Any],
    classified: dict[str, Any],
) -> _BracketConditioningMetrics:
    conditioning = classified.get("conditioning")
    if not isinstance(conditioning, dict):
        conditioning = {}
    return _BracketConditioningMetrics(
        central_gradient=conditioning.get("central_gradient"),
        response_fraction=_finite_float(conditioning.get("response_fraction")),
        fd_asymmetry_rel=_finite_float(conditioning.get("fd_asymmetry_rel")),
        fd_condition_number=_finite_float(conditioning.get("fd_condition_number")),
        gradient_uncertainty_rel=_finite_float(
            conditioning.get("gradient_uncertainty_rel")
        ),
        paired_uncertainty_rel=_paired_uncertainty_rel(artifact),
        paired_same_sign_fraction=_paired_same_sign_fraction(artifact),
    )


def _bracket_margin_scores(
    metrics: _BracketConditioningMetrics,
    config: NonlinearTurbulenceGradientBracketSweepConfig,
) -> dict[str, float]:
    return {
        "response": _metric_margin(
            metrics.response_fraction,
            target=config.min_fd_response_fraction,
            sense="min",
            cap=config.score_cap,
            value_floor=config.value_floor,
        ),
        "locality": _metric_margin(
            metrics.fd_asymmetry_rel,
            target=config.max_fd_asymmetry_rel,
            sense="max",
            cap=config.score_cap,
            value_floor=config.value_floor,
        ),
        "uncertainty": _metric_margin(
            metrics.gradient_uncertainty_rel,
            target=config.max_gradient_uncertainty_rel,
            sense="max",
            cap=config.score_cap,
            value_floor=config.value_floor,
        ),
        "conditioning": _metric_margin(
            metrics.fd_condition_number,
            target=config.max_fd_condition_number,
            sense="max",
            cap=config.score_cap,
            value_floor=config.value_floor,
        ),
    }


def _repeated_bracket_stable(
    metrics: _BracketConditioningMetrics,
    config: NonlinearTurbulenceGradientBracketSweepConfig,
) -> bool:
    return (
        metrics.paired_uncertainty_rel is not None
        and metrics.paired_uncertainty_rel
        <= float(config.max_repeated_bracket_uncertainty_rel)
        and metrics.paired_same_sign_fraction is not None
        and metrics.paired_same_sign_fraction
        >= float(config.min_repeated_bracket_same_sign_fraction)
    )


def _failed_bracket_gate_names(classified: dict[str, Any]) -> list[str]:
    return [
        str(gate.get("metric", ""))
        for gate in classified.get("gates", [])
        if isinstance(gate, dict) and not bool(gate.get("passed", False))
    ]


def _bracket_sweep_row(
    artifact: dict[str, Any],
    *,
    label: str | None,
    path: str | None,
    config: NonlinearTurbulenceGradientBracketSweepConfig,
) -> dict[str, Any]:
    classified = classify_gradient_artifact(
        artifact,
        path=path,
        config=_bracket_evidence_config(config),
    )
    metrics = _bracket_conditioning_metrics(artifact, classified)
    delta = _finite_float(artifact.get("delta_parameter"))
    margins = _bracket_margin_scores(metrics, config)
    return {
        "label": str(label or artifact.get("parameter_name") or path or ""),
        "path": path,
        "parameter_name": str(artifact.get("parameter_name", "")),
        "delta_parameter": _json_number(delta),
        "passed": bool(
            classified.get("qualifies_for_production_turbulence_gradient", False)
        ),
        "metrics": {
            "central_gradient": metrics.central_gradient,
            "response_fraction": metrics.response_fraction,
            "fd_asymmetry_rel": metrics.fd_asymmetry_rel,
            "fd_condition_number": metrics.fd_condition_number,
            "gradient_uncertainty_rel": metrics.gradient_uncertainty_rel,
            "paired_gradient_uncertainty_rel": _json_number(
                metrics.paired_uncertainty_rel
            ),
            "paired_same_sign_fraction": _json_number(
                metrics.paired_same_sign_fraction
            ),
            "repeated_bracket_stable": _repeated_bracket_stable(metrics, config),
        },
        "margins": margins,
        "weakest_margin": _json_number(min(margins.values())),
        "score": _json_number(
            math.prod(max(value, 0.0) for value in margins.values()) ** 0.25
        ),
        "failed_gates": _failed_bracket_gate_names(classified),
    }


def _delta_key(row: dict[str, Any]) -> float:
    delta = _finite_float(row.get("delta_parameter"))
    if delta is None:
        return math.inf
    return float(delta)


def _bracket_parameter_names(rows: Sequence[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("parameter_name", "")) for row in rows if row.get("parameter_name")
    }


def _response_ok_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if float(row["margins"]["response"]) >= 1.0]


def _response_ok_signs(rows: Sequence[dict[str, Any]]) -> set[float]:
    gradients = [
        _finite_float(row.get("metrics", {}).get("central_gradient"))
        for row in rows
        if isinstance(row.get("metrics"), dict)
    ]
    return {
        math.copysign(1.0, float(value))
        for value in gradients
        if value is not None and value != 0.0
    }


def _rows_with_margin(
    rows: Sequence[dict[str, Any]],
    margin_name: str,
) -> list[dict[str, Any]]:
    return [row for row in rows if float(row["margins"][margin_name]) >= 1.0]


def _repeated_unstable_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["metrics"].get("paired_gradient_uncertainty_rel") is not None
        and not bool(row["metrics"].get("repeated_bracket_stable", False))
    ]


def _initial_bracket_sweep_recommendation(rows: Sequence[dict[str, Any]]) -> str | None:
    if not rows:
        return "run at least two matched plus/minus perturbation amplitudes before claiming bracket locality"
    if len(_bracket_parameter_names(rows)) > 1:
        return (
            "mixed controls were supplied to a same-control bracket sweep; split the "
            "artifacts by control or use the nonlinear turbulence-gradient candidate "
            "ranking/overdetermined campaign planner"
        )
    passed_rows = [row for row in rows if bool(row.get("passed", False))]
    if passed_rows:
        best = min(passed_rows, key=_delta_key)
        return (
            "promote only the passed same-control bracket after freezing provenance; "
            f"smallest passing delta is {best.get('delta_parameter')}"
        )
    return None


def _resolved_bracket_sweep_recommendation(
    *,
    response_ok: Sequence[dict[str, Any]],
    local_rows: Sequence[dict[str, Any]],
    quiet_rows: Sequence[dict[str, Any]],
    repeated_unstable: Sequence[dict[str, Any]],
) -> str:
    if local_rows and not quiet_rows:
        if repeated_unstable:
            return (
                "do not add replicas at the same bracket yet; matched-pair diagnostics "
                "show seed-level instability, so run a perturbation-amplitude/locality "
                "sweep or switch to a smoother composite profile-gradient direction"
            )
        return (
            "locality is acceptable but uncertainty is not; add statistical power only "
            "after a second nearby perturbation amplitude confirms the same gradient sign"
        )
    if quiet_rows and not local_rows:
        return (
            "uncertainty is acceptable only for nonlocal brackets; shrink the perturbation "
            "or choose a more local control before adding replicas"
        )
    if response_ok and local_rows and quiet_rows:
        return (
            "the numerical bracket margins are resolved, local, and quiet, but no input "
            "artifact has production long-window scope; rerun or re-export with matched "
            "post-transient provenance before considering promotion"
        )
    if response_ok and not local_rows and not quiet_rows:
        return (
            "the response is detectable but neither local nor statistically resolved; "
            "prefer an overdetermined/profile-gradient campaign over more single-control runs"
        )
    return (
        "the heat-flux response is not resolved at the tested amplitudes; abandon this "
        "control or enlarge the perturbation only if a locality sweep remains bounded"
    )


def _bracket_sweep_recommendation(rows: Sequence[dict[str, Any]]) -> str:
    initial = _initial_bracket_sweep_recommendation(rows)
    if initial is not None:
        return initial
    response_ok = _response_ok_rows(rows)
    if len(_response_ok_signs(response_ok)) > 1:
        return (
            "same-control resolved brackets change central-gradient sign; do not add "
            "replicas at one amplitude, and move to a locality/amplitude sweep with "
            "stricter provenance or a smoother composite profile-gradient direction"
        )
    return _resolved_bracket_sweep_recommendation(
        response_ok=response_ok,
        local_rows=_rows_with_margin(response_ok, "locality"),
        quiet_rows=_rows_with_margin(response_ok, "uncertainty"),
        repeated_unstable=_repeated_unstable_rows(rows),
    )


def nonlinear_turbulence_gradient_bracket_sweep_report(
    artifacts: Sequence[dict[str, Any]],
    *,
    labels: Sequence[str | None] | None = None,
    paths: Sequence[str | None] | None = None,
    config: NonlinearTurbulenceGradientBracketSweepConfig | None = None,
) -> dict[str, Any]:
    """Summarize a same-control perturbation-amplitude sweep.

    This is a planning/claim-boundary utility.  It does not promote nonlinear
    turbulence-gradient evidence unless an input finite-difference artifact
    already passes the production long-window gate.  Its main purpose is to
    decide whether the next expensive campaign should add replicas at the same
    bracket, change the perturbation amplitude, or move to an overdetermined
    profile-gradient direction.
    """

    cfg = config or NonlinearTurbulenceGradientBracketSweepConfig()
    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")

    rows = [
        _bracket_sweep_row(artifact, label=label, path=path, config=cfg)
        for artifact, label, path in zip(artifacts, label_list, path_list)
    ]
    rows.sort(key=_delta_key)
    parameter_names = sorted(
        {row["parameter_name"] for row in rows if row["parameter_name"]}
    )
    same_control = len(parameter_names) <= 1
    passed_rows = [row for row in rows if bool(row.get("passed", False))]
    return {
        "kind": "nonlinear_turbulence_gradient_bracket_sweep",
        "claim_level": "same_control_bracket_locality_planning_not_gradient_promotion",
        "passed": bool(passed_rows) and same_control,
        "promotion_ready_bracket_count": len(passed_rows) if same_control else 0,
        "same_control_gate": {
            "passed": same_control,
            "parameter_names": parameter_names,
        },
        "parameter_names": parameter_names,
        "recommendation": _bracket_sweep_recommendation(rows),
        "config": asdict(cfg),
        "brackets": rows,
    }


__all__ = [
    "_bracket_sweep_recommendation",
    "_bracket_sweep_row",
    "_delta_key",
    "_paired_same_sign_fraction",
    "_paired_uncertainty_rel",
    "nonlinear_turbulence_gradient_bracket_sweep_report",
]

# ---- screening reports ----
"""Screening reports for nonlinear turbulence-gradient campaign planning."""


from dataclasses import asdict
from typing import Any, Sequence
import math




def _candidate_next_action(
    *,
    passed: bool,
    response_margin: float,
    asymmetry_margin: float,
    uncertainty_margin: float,
    condition_margin: float,
) -> str:
    if passed:
        return (
            "promote only after the source campaign provenance is frozen in docs and CI"
        )
    if response_margin < 1.0:
        return (
            "abandon or enlarge the perturbation only if locality remains bounded; "
            "the heat-flux response is not resolved above the minimum response gate"
        )
    if asymmetry_margin < 1.0 and uncertainty_margin >= 1.0:
        return (
            "repeat with a smaller bracket or nearby control; uncertainty is adequate "
            "but the finite-difference response is nonlocal"
        )
    if uncertainty_margin < 1.0 and asymmetry_margin >= 1.0:
        return (
            "keep the local direction but increase statistical power: longer windows, "
            "more replicas, or a checked amplitude bracket"
        )
    if condition_margin < 1.0:
        return "choose a better-conditioned observable/control pair before rerunning"
    return (
        "do not relax gates; move to an overdetermined least-squares/profile-gradient "
        "campaign with multiple controls and matched long-window replicas"
    )


def _ranking_evidence_config(
    cfg: NonlinearTurbulenceGradientCandidateRankingConfig,
) -> NonlinearTurbulenceGradientEvidenceConfig:
    return NonlinearTurbulenceGradientEvidenceConfig(
        max_gradient_uncertainty_rel=cfg.max_gradient_uncertainty_rel,
        max_fd_asymmetry_rel=cfg.max_fd_asymmetry_rel,
        max_fd_condition_number=cfg.max_fd_condition_number,
        min_fd_response_fraction=cfg.min_fd_response_fraction,
        value_floor=cfg.value_floor,
    )


def _conditioning_metrics(classified: dict[str, Any]) -> dict[str, Any]:
    conditioning = classified.get("conditioning", {})
    return conditioning if isinstance(conditioning, dict) else {}


def _candidate_margins(
    conditioning: dict[str, Any],
    cfg: NonlinearTurbulenceGradientCandidateRankingConfig,
) -> dict[str, float]:
    response_fraction = _finite_float(conditioning.get("response_fraction"))
    fd_asymmetry_rel = _finite_float(conditioning.get("fd_asymmetry_rel"))
    fd_condition_number = _finite_float(conditioning.get("fd_condition_number"))
    gradient_uncertainty_rel = _finite_float(conditioning.get("gradient_uncertainty_rel"))
    return {
        "response": _metric_margin(
            response_fraction,
            target=cfg.min_fd_response_fraction,
            sense="min",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        ),
        "locality": _metric_margin(
            fd_asymmetry_rel,
            target=cfg.max_fd_asymmetry_rel,
            sense="max",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        ),
        "conditioning": _metric_margin(
            fd_condition_number,
            target=cfg.max_fd_condition_number,
            sense="max",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        ),
        "uncertainty": _metric_margin(
            gradient_uncertainty_rel,
            target=cfg.max_gradient_uncertainty_rel,
            sense="max",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        ),
    }


def _candidate_failed_gates(classified: dict[str, Any]) -> list[str]:
    return [
        str(gate.get("metric", ""))
        for gate in classified.get("gates", [])
        if isinstance(gate, dict) and not bool(gate.get("passed", False))
    ]


def _candidate_score(
    margins: dict[str, float],
    *,
    explicit_production_scope: bool,
) -> tuple[float, float]:
    weakest_margin = min(margins.values())
    geometric_score = math.prod(max(value, 0.0) for value in margins.values()) ** 0.25
    if not explicit_production_scope:
        geometric_score *= 0.5
    return weakest_margin, geometric_score


def _candidate_metric_payload(conditioning: dict[str, Any]) -> dict[str, Any]:
    return {
        "central_gradient": conditioning.get("central_gradient"),
        "response_fraction": conditioning.get("response_fraction"),
        "fd_asymmetry_rel": conditioning.get("fd_asymmetry_rel"),
        "fd_condition_number": conditioning.get("fd_condition_number"),
        "gradient_uncertainty_rel": conditioning.get("gradient_uncertainty_rel"),
    }


def _candidate_ranking_row(
    *,
    artifact: dict[str, Any],
    path: str | None,
    label: str | None,
    index: int,
    cfg: NonlinearTurbulenceGradientCandidateRankingConfig,
) -> dict[str, Any]:
    classified = classify_gradient_artifact(
        artifact,
        path=path,
        config=_ranking_evidence_config(cfg),
    )
    conditioning = _conditioning_metrics(classified)
    margins = _candidate_margins(conditioning, cfg)
    weakest_margin, geometric_score = _candidate_score(
        margins,
        explicit_production_scope=bool(classified.get("explicit_production_scope", False)),
    )
    passed = bool(classified.get("qualifies_for_production_turbulence_gradient", False))
    parameter_name = str(artifact.get("parameter_name") or label or path or f"candidate_{index}")
    return {
        "rank": None,
        "index": index,
        "label": str(label or parameter_name),
        "path": path,
        "parameter_name": parameter_name,
        "passed": passed,
        "evidence_class": classified.get("evidence_class"),
        "failed_gates": _candidate_failed_gates(classified),
        "metrics": _candidate_metric_payload(conditioning),
        "margins": margins,
        "weakest_margin": _json_number(weakest_margin),
        "score": _json_number(geometric_score),
        "next_action": _candidate_next_action(
            passed=passed,
            response_margin=margins["response"],
            asymmetry_margin=margins["locality"],
            uncertainty_margin=margins["uncertainty"],
            condition_margin=margins["conditioning"],
        ),
    }


def _rank_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows.sort(
        key=lambda row: (
            bool(row["passed"]),
            float(row["weakest_margin"] or 0.0),
            float(row["score"] or 0.0),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _candidate_followup_groups(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    passed_rows = [row for row in rows if bool(row["passed"])]
    local_but_noisy = [
        row
        for row in rows
        if float(row["margins"]["locality"]) >= 1.0
        and float(row["margins"]["uncertainty"]) < 1.0
    ]
    quiet_but_nonlocal = [
        row
        for row in rows
        if float(row["margins"]["uncertainty"]) >= 1.0
        and float(row["margins"]["locality"]) < 1.0
    ]
    return passed_rows, local_but_noisy, quiet_but_nonlocal


def _ranking_recommendation(
    *,
    passed_rows: list[dict[str, Any]],
    local_but_noisy: list[dict[str, Any]],
    quiet_but_nonlocal: list[dict[str, Any]],
    overdetermined_followup: bool,
) -> str:
    if passed_rows:
        return (
            "one or more candidates passes the production evidence gates; freeze "
            "the provenance and promote only the passed artifact"
        )
    if overdetermined_followup and local_but_noisy and quiet_but_nonlocal:
        return (
            "the overdetermined follow-up completed with no promotable candidate; "
            "keep the nonlinear-gradient claim fail-closed, target the best local "
            "but noisy control with additional independent replicas or variance "
            "reduction only if the cost is justified, and replace or shrink the "
            "nonlocal controls before another production campaign"
        )
    if overdetermined_followup and local_but_noisy:
        return (
            "the overdetermined follow-up found local but statistically unresolved "
            "candidates; keep the claim fail-closed and add independent replicas "
            "or a lower-variance observable before promotion"
        )
    if overdetermined_followup and quiet_but_nonlocal:
        return (
            "the overdetermined follow-up found statistically quiet but nonlocal "
            "candidates; keep the claim fail-closed and shrink the perturbation "
            "or choose more local controls before adding replicas"
        )
    if local_but_noisy and quiet_but_nonlocal:
        return (
            "use an overdetermined least-squares/profile-gradient campaign: current "
            "single-control candidates have complementary locality and uncertainty failures"
        )
    if local_but_noisy:
        return "extend statistical power for the best local direction before changing controls"
    if quiet_but_nonlocal:
        return "reduce bracket size or choose a nearby/local control before adding replicas"
    return (
        "screen new profile-gradient or objective-gradient controls; current candidates "
        "do not isolate a promotable response"
    )


def _pack_candidate_ranking_report(
    *,
    rows: list[dict[str, Any]],
    passed_rows: list[dict[str, Any]],
    recommendation: str,
    cfg: NonlinearTurbulenceGradientCandidateRankingConfig,
) -> dict[str, Any]:
    return {
        "kind": "nonlinear_turbulence_gradient_candidate_ranking",
        "claim_level": "campaign_planning_not_gradient_evidence",
        "passed": bool(passed_rows),
        "promotion_ready_candidate_count": len(passed_rows),
        "best_candidate": rows[0] if rows else None,
        "recommendation": recommendation,
        "config": asdict(cfg),
        "candidates": rows,
    }


def nonlinear_turbulence_gradient_candidate_ranking_report(
    artifacts: Sequence[dict[str, Any]],
    *,
    paths: Sequence[str | None] | None = None,
    labels: Sequence[str | None] | None = None,
    config: NonlinearTurbulenceGradientCandidateRankingConfig | None = None,
) -> dict[str, Any]:
    """Rank nonlinear turbulence-gradient candidates without promoting failures.

    The ranking is a planning aid, not a replacement for the production gate.
    It scores each candidate by the weakest normalized evidence margin across
    response, locality, conditioning, and uncertainty.  This makes the next
    campaign choice explicit: candidates with complementary failures should
    move to a profile-gradient or overdetermined least-squares design instead
    of repeating a single boundary coefficient indefinitely.
    """

    cfg = config or NonlinearTurbulenceGradientCandidateRankingConfig()
    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")

    rows = _rank_candidate_rows(
        [
            _candidate_ranking_row(
                artifact=artifact,
                path=path,
                label=label,
                index=index,
                cfg=cfg,
            )
            for index, (artifact, path, label) in enumerate(zip(artifacts, path_list, label_list))
        ]
    )
    passed_rows, local_but_noisy, quiet_but_nonlocal = _candidate_followup_groups(rows)
    recommendation = _ranking_recommendation(
        passed_rows=passed_rows,
        local_but_noisy=local_but_noisy,
        quiet_but_nonlocal=quiet_but_nonlocal,
        overdetermined_followup=cfg.campaign_context == "overdetermined_followup",
    )
    return _pack_candidate_ranking_report(
        rows=rows,
        passed_rows=passed_rows,
        recommendation=recommendation,
        cfg=cfg,
    )



__all__ = [
    "_bracket_sweep_recommendation",
    "_bracket_sweep_row",
    "_candidate_next_action",
    "_delta_key",
    "_metric_margin",
    "_paired_same_sign_fraction",
    "_paired_uncertainty_rel",
    "nonlinear_turbulence_gradient_bracket_sweep_report",
    "nonlinear_turbulence_gradient_candidate_ranking_report",
]

# ---- evidence-gap reports ----
"""Gap and production-report orchestration for nonlinear gradient evidence."""


from dataclasses import asdict, dataclass
from typing import Any, Sequence



@dataclass(frozen=True)
class _EvidenceGapContext:
    cfg: NonlinearTurbulenceGradientEvidenceConfig
    gap_cfg: NonlinearTurbulenceGradientGapConfig
    passed: bool
    blockers: list[str]
    gradient: dict[str, Any]
    windows: dict[str, Any]
    qualifying_windows: list[dict[str, Any]]
    failed_gradient_gates: list[dict[str, str]]
    has_production_candidate: bool


def _required_run_rows(
    config: NonlinearTurbulenceGradientGapConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state, multiplier in (
        ("minus_delta", 1.0 - config.perturbation_fraction),
        ("baseline", 1.0),
        ("plus_delta", 1.0 + config.perturbation_fraction),
    ):
        rows.append(
            {
                "state": state,
                "parameter_name": config.parameter_name,
                "parameter_multiplier": multiplier,
                "replicates": list(config.replicate_labels),
                "required_output": (
                    "docs/_static/{case}_{state}_replicates/"
                    "{case}_{state}_t{tmax:g}_ensemble_gate.json"
                ).format(
                    case=config.case_slug,
                    state=state,
                    tmax=config.analysis_tmax,
                ),
                "run_contract": {
                    "same_numerics_except_parameter": True,
                    "t_start": config.t_start,
                    "minimum_tmax": config.minimum_tmax,
                    "analysis_window": [config.analysis_tmin, config.analysis_tmax],
                    "minimum_grid": config.minimum_grid,
                },
            }
        )
    return rows


def _gap_configs(
    *,
    config: NonlinearTurbulenceGradientEvidenceConfig | None,
    gap_config: NonlinearTurbulenceGradientGapConfig | None,
) -> tuple[
    NonlinearTurbulenceGradientEvidenceConfig,
    NonlinearTurbulenceGradientGapConfig,
]:
    return (
        config or NonlinearTurbulenceGradientEvidenceConfig(),
        gap_config or NonlinearTurbulenceGradientGapConfig(),
    )


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _qualifying_gap_window_rows(windows: dict[str, Any]) -> list[dict[str, Any]]:
    rows = windows.get("ensemble_rows", [])
    if not isinstance(rows, Sequence):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict)
        and bool(row.get("qualifies_for_replicated_long_window_uncertainty", False))
    ]


def _failed_gradient_gates(gradient: dict[str, Any]) -> list[dict[str, str]]:
    gates = gradient.get("gates", [])
    if not isinstance(gates, Sequence):
        return []
    return [
        {
            "metric": str(gate.get("metric", "")),
            "detail": str(gate.get("detail", "")),
        }
        for gate in gates
        if isinstance(gate, dict) and not bool(gate.get("passed", False))
    ]


def _gap_context(
    evidence_report: dict[str, Any],
    *,
    cfg: NonlinearTurbulenceGradientEvidenceConfig,
    gap_cfg: NonlinearTurbulenceGradientGapConfig,
) -> _EvidenceGapContext:
    gradient = _dict_or_empty(evidence_report.get("gradient_artifact"))
    windows = _dict_or_empty(evidence_report.get("window_evidence"))
    failed_gradient_gates = _failed_gradient_gates(gradient)
    gradient_class = str(gradient.get("evidence_class", ""))
    return _EvidenceGapContext(
        cfg=cfg,
        gap_cfg=gap_cfg,
        passed=bool(evidence_report.get("passed", False)),
        blockers=[str(item) for item in evidence_report.get("blockers", [])],
        gradient=gradient,
        windows=windows,
        qualifying_windows=_qualifying_gap_window_rows(windows),
        failed_gradient_gates=failed_gradient_gates,
        has_production_candidate=(
            gradient_class == "production_long_window_turbulence_gradient_candidate"
        ),
    )


def _production_gradient_missing_row(ctx: _EvidenceGapContext) -> dict[str, Any]:
    if ctx.has_production_candidate:
        return {
            "blocker": "production_gradient_artifact",
            "needed": (
                "the current matched long-window production-candidate "
                "finite-difference artifact must pass all recorded "
                "response, asymmetry, conditioning, and propagated "
                "gradient-uncertainty gates"
            ),
            "current_artifact_class": ctx.gradient.get("evidence_class"),
            "current_artifact_path": ctx.gradient.get("path"),
            "current_failed_gates": ctx.failed_gradient_gates,
        }
    return {
        "blocker": "production_gradient_artifact",
        "needed": (
            "central finite-difference or adjoint/VJP artifact computed "
            "from matched long post-transient nonlinear heat-flux windows"
        ),
        "current_artifact_class": ctx.gradient.get("evidence_class"),
        "current_artifact_path": ctx.gradient.get("path"),
    }


def _replicated_window_missing_row(ctx: _EvidenceGapContext) -> dict[str, Any]:
    return {
        "blocker": "replicated_long_window_uncertainty",
        "needed": (
            "at least one baseline/plus/minus campaign with replicated "
            "post-transient transport-window ensemble gates"
        ),
        "qualifying_window_ensembles": len(ctx.qualifying_windows),
    }


def _missing_evidence_rows(ctx: _EvidenceGapContext) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    if "production_gradient_artifact" in ctx.blockers:
        missing.append(_production_gradient_missing_row(ctx))
    if "replicated_long_window_uncertainty" in ctx.blockers:
        missing.append(_replicated_window_missing_row(ctx))
    return missing


def _finite_difference_audit_contract(
    *,
    cfg: NonlinearTurbulenceGradientEvidenceConfig,
    gap_cfg: NonlinearTurbulenceGradientGapConfig,
) -> dict[str, Any]:
    return {
        "required_output": (
            "docs/_static/{case}_{parameter}_central_fd_gradient_gate.json"
        ).format(
            case=gap_cfg.case_slug,
            parameter=gap_cfg.parameter_name,
        ),
        "formula": "dQ/dp = (mean(Q_plus) - mean(Q_minus)) / (2 * delta_p)",
        "required_metrics": [
            "central_gradient",
            "response_fraction",
            "fd_asymmetry_rel",
            "fd_condition_number",
            "gradient_uncertainty_rel",
            "baseline_window_mean",
            "plus_window_mean",
            "minus_window_mean",
            "baseline_window_sem",
            "plus_window_sem",
            "minus_window_sem",
        ],
        "acceptance_gates": {
            "production_nonlinear_window_gradient_gate": True,
            "response_fraction_min": cfg.min_fd_response_fraction,
            "fd_asymmetry_rel_max": cfg.max_fd_asymmetry_rel,
            "fd_condition_number_max": cfg.max_fd_condition_number,
            "gradient_uncertainty_rel_max": cfg.max_gradient_uncertainty_rel,
            "window_mean_rel_spread_max": cfg.max_window_mean_rel_spread,
            "window_combined_sem_rel_max": cfg.max_window_combined_sem_rel,
        },
        "fallback_if_marginal": (
            "repeat the paired campaign with a second perturbation fraction "
            "or longer analysis_tmax; do not promote if the response is not "
            "resolved above the transport-window uncertainty."
        ),
    }


def _claim_level(ctx: _EvidenceGapContext) -> str:
    if ctx.has_production_candidate and not ctx.passed:
        return "fail_closed_production_candidate_gradient_gate_not_resolved"
    return "fail_closed_missing_campaign_plan_not_gradient_evidence"


def _required_campaign(
    *,
    gap_cfg: NonlinearTurbulenceGradientGapConfig,
    finite_difference_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_slug": gap_cfg.case_slug,
        "parameter_name": gap_cfg.parameter_name,
        "perturbation_fraction": gap_cfg.perturbation_fraction,
        "required_runs": _required_run_rows(gap_cfg),
        "finite_difference_audit": finite_difference_audit,
    }


def _gradient_evidence_requirements() -> list[str]:
    return [
        "run baseline, plus-delta, and minus-delta nonlinear simulations with identical numerical settings except the perturbed parameter",
        "use the same seed/timestep replicate labels for all three parameter states",
        "discard the startup transient and average only over the declared post-transient analysis window",
        "build passed ensemble gates for baseline, plus, and minus states before computing the gradient",
        "record finite-difference response, asymmetry, condition number, and gradient uncertainty in the production gradient artifact",
    ]


def nonlinear_turbulence_gradient_evidence_gap_report(
    evidence_report: dict[str, Any],
    *,
    config: NonlinearTurbulenceGradientEvidenceConfig | None = None,
    gap_config: NonlinearTurbulenceGradientGapConfig | None = None,
) -> dict[str, Any]:
    """Return the fail-closed run campaign needed to close gradient evidence.

    The report is deliberately prescriptive: it requires paired plus/minus
    long-window nonlinear runs with the same seeds, timestep variant, grid, and
    post-transient analysis window before a finite-difference turbulence
    gradient can be promoted.  It does not infer a gradient from standalone
    replicated transport windows.
    """

    cfg, gap_cfg = _gap_configs(config=config, gap_config=gap_config)
    ctx = _gap_context(evidence_report, cfg=cfg, gap_cfg=gap_cfg)
    finite_difference_audit = _finite_difference_audit_contract(
        cfg=cfg, gap_cfg=gap_cfg
    )
    return {
        "kind": "nonlinear_turbulence_gradient_evidence_gap_report",
        "claim_level": _claim_level(ctx),
        "passed": ctx.passed,
        "promotion_blocked": not ctx.passed,
        "blockers": ctx.blockers,
        "missing_evidence": _missing_evidence_rows(ctx),
        "current_gradient_candidate_present": ctx.has_production_candidate,
        "current_gradient_failed_gates": ctx.failed_gradient_gates,
        "current_window_evidence_passed": bool(ctx.windows.get("passed", False)),
        "qualifying_window_ensemble_count": len(ctx.qualifying_windows),
        "required_campaign": _required_campaign(
            gap_cfg=gap_cfg, finite_difference_audit=finite_difference_audit
        ),
        "requirements": _gradient_evidence_requirements(),
        "notes": (
            "Standalone passed transport windows are necessary but not sufficient: "
            "production turbulence-gradient evidence requires paired parameter "
            "perturbations tied to the same post-transient averaging protocol."
        ),
    }


def nonlinear_turbulence_gradient_evidence_report(
    gradient_artifact: dict[str, Any],
    *,
    window_artifacts: Sequence[dict[str, Any]] = (),
    gradient_path: str | None = None,
    window_paths: Sequence[str | None] | None = None,
    config: NonlinearTurbulenceGradientEvidenceConfig | None = None,
    gap_config: NonlinearTurbulenceGradientGapConfig | None = None,
) -> dict[str, Any]:
    """Return a fail-closed production nonlinear gradient evidence report."""

    cfg = config or NonlinearTurbulenceGradientEvidenceConfig()
    gradient = classify_gradient_artifact(
        gradient_artifact,
        path=gradient_path,
        config=cfg,
    )
    windows = summarize_window_evidence(
        list(window_artifacts),
        paths=window_paths,
        config=cfg,
    )
    gates = [
        _gate(
            "production_gradient_artifact",
            bool(gradient["qualifies_for_production_turbulence_gradient"]),
            str(gradient["evidence_class"]),
        ),
        *windows["gates"],
    ]
    passed = all(bool(gate["passed"]) for gate in gates)
    blockers = [str(gate["metric"]) for gate in gates if not bool(gate["passed"])]
    report = {
        "kind": "nonlinear_turbulence_gradient_evidence_report",
        "claim_level": "fail_closed_claim_boundary_for_long_window_nonlinear_turbulence_gradient_evidence",
        "passed": passed,
        "production_nonlinear_window_gradient_gate": passed,
        "blockers": blockers,
        "requirements": [
            "gradient artifact must explicitly claim production long-window nonlinear turbulence-gradient scope",
            "startup/reduced-window finite-difference or estimator artifacts are recorded but never promoted",
            "finite-difference response, asymmetry, and condition number must be recorded and within gates",
            "gradient uncertainty must be recorded and within gate",
            "replicated post-transient nonlinear-window summaries must pass ensemble uncertainty gates",
        ],
        "config": asdict(cfg),
        "gates": gates,
        "gradient_artifact": gradient,
        "window_evidence": windows,
        "notes": (
            "This checker distinguishes claim boundaries only.  Passing it means "
            "the supplied artifacts meet the recorded evidence contract; it does "
            "not run or certify new nonlinear simulations."
        ),
    }
    report["evidence_gap"] = nonlinear_turbulence_gradient_evidence_gap_report(
        report,
        config=cfg,
        gap_config=gap_config,
    )
    return report


__all__ = [
    "_required_run_rows",
    "nonlinear_turbulence_gradient_evidence_gap_report",
    "nonlinear_turbulence_gradient_evidence_report",
]

# ---- production evidence entry points ----
"""Claim-boundary gates for nonlinear turbulence-gradient evidence.

This module is intentionally data-only.  It does not run nonlinear solves and
does not infer production turbulence-gradient support from startup finite
differences, reduced nonlinear-window estimators, or single late-window
summaries.  The default behavior is fail-closed unless an artifact explicitly
records production long-window gradient scope, finite-difference conditioning,
gradient uncertainty, and replicated nonlinear-window uncertainty evidence.
"""


from pathlib import Path
from typing import Any
import json



def load_json_artifact(path: str | Path) -> dict[str, Any]:
    """Load a JSON object artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


__all__ = [
    "NON_PRODUCTION_SCOPE_MARKERS",
    "NonlinearTurbulenceGradientBracketSweepConfig",
    "NonlinearTurbulenceGradientCandidateRankingConfig",
    "NonlinearTurbulenceGradientEvidenceConfig",
    "NonlinearTurbulenceGradientFiniteDifferenceConfig",
    "NonlinearTurbulenceGradientGapConfig",
    "classify_gradient_artifact",
    "load_json_artifact",
    "nonlinear_turbulence_gradient_bracket_sweep_report",
    "nonlinear_turbulence_gradient_candidate_ranking_report",
    "nonlinear_turbulence_gradient_evidence_gap_report",
    "nonlinear_turbulence_gradient_evidence_report",
    "nonlinear_turbulence_gradient_finite_difference_report",
    "summarize_window_evidence",
]
