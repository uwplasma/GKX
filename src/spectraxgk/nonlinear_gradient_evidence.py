"""Claim-boundary gates for nonlinear turbulence-gradient evidence.

This module is intentionally data-only.  It does not run nonlinear solves and
does not infer production turbulence-gradient support from startup finite
differences, reduced nonlinear-window estimators, or single late-window
summaries.  The default behavior is fail-closed unless an artifact explicitly
records production long-window gradient scope, finite-difference conditioning,
gradient uncertainty, and replicated nonlinear-window uncertainty evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import json
import math
import re

from spectraxgk.quasilinear_window import (
    NonlinearWindowEnsembleConfig,
    nonlinear_window_ensemble_report,
    nonlinear_window_stats_promotion_ready,
)


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


def _gradient_conditioning_summary(
    payload: dict[str, Any],
    *,
    config: NonlinearTurbulenceGradientEvidenceConfig,
) -> dict[str, Any]:
    gradient = _nested_dict(payload, "gradient", "gradient_summary")
    metrics = _nested_dict(payload, "metrics")
    conditioning = _nested_dict(
        payload,
        "conditioning",
        "conditioning_gate",
        "finite_difference_conditioning",
        "gradient_conditioning",
    )
    uncertainty = _nested_dict(
        payload,
        "uncertainty",
        "uncertainty_gate",
        "gradient_uncertainty",
    )
    candidates = (gradient, metrics, conditioning, uncertainty, payload)

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

    response_fraction = _first_finite(
        candidates,
        (
            "response_fraction",
            "resolved_response_fraction",
            "fd_response_fraction",
        ),
    )
    asymmetry = _first_finite(
        candidates,
        (
            "asymmetry_rel",
            "derivative_asymmetry",
            "fd_asymmetry_rel",
        ),
    )
    condition_number = _first_finite(
        candidates,
        (
            "condition_number",
            "fd_condition_number",
            "sensitivity_condition_number",
        ),
    )
    uncertainty_rel = _first_finite(
        candidates,
        (
            "gradient_sem_rel",
            "sem_rel",
            "gradient_uncertainty_rel",
            "gradient_relative_uncertainty",
            "relative_uncertainty",
        ),
    )

    gates = [
        _gate(
            "finite_gradient_estimate",
            derivative is not None,
            f"central_gradient={derivative}",
        ),
        _gate(
            "fd_response_resolved",
            response_fraction is not None
            and response_fraction >= float(config.min_fd_response_fraction),
            "response_fraction={value} min={gate}".format(
                value=response_fraction,
                gate=config.min_fd_response_fraction,
            ),
        ),
        _gate(
            "fd_asymmetry_bounded",
            asymmetry is not None and asymmetry <= float(config.max_fd_asymmetry_rel),
            "fd_asymmetry_rel={value} max={gate}".format(
                value=asymmetry,
                gate=config.max_fd_asymmetry_rel,
            ),
        ),
        _gate(
            "fd_condition_number_bounded",
            condition_number is not None
            and condition_number <= float(config.max_fd_condition_number),
            "condition_number={value} max={gate}".format(
                value=condition_number,
                gate=config.max_fd_condition_number,
            ),
        ),
        _gate(
            "gradient_uncertainty_bounded",
            uncertainty_rel is not None
            and uncertainty_rel <= float(config.max_gradient_uncertainty_rel),
            "gradient_uncertainty_rel={value} max={gate}".format(
                value=uncertainty_rel,
                gate=config.max_gradient_uncertainty_rel,
            ),
        ),
    ]
    return {
        "central_gradient": _json_number(derivative),
        "response_fraction": _json_number(response_fraction),
        "fd_asymmetry_rel": _json_number(asymmetry),
        "fd_condition_number": _json_number(condition_number),
        "gradient_uncertainty_rel": _json_number(uncertainty_rel),
        "gates": gates,
        "passed": all(bool(gate["passed"]) for gate in gates),
    }


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
            key: value
            for key, value in conditioning.items()
            if key not in {"gates"}
        },
        "gates": gates,
        "qualifies_for_production_turbulence_gradient": qualifies,
    }


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

    rows: list[dict[str, Any]] = []
    convergence_reports: list[dict[str, Any]] = []
    convergence_paths: list[str | None] = []
    single_window_rows: list[dict[str, Any]] = []

    for payload, path in zip(window_artifacts, path_list):
        kind = str(payload.get("kind", ""))
        if kind == "nonlinear_window_ensemble_report":
            rows.append(_ensemble_row(payload, path=path, source="input_ensemble", config=cfg))
        elif kind == "nonlinear_window_convergence_report":
            ready, failures = nonlinear_window_stats_promotion_ready(payload)
            single_window_rows.append(
                {
                    "path": path,
                    "kind": kind,
                    "case": str(payload.get("case", "")),
                    "passed": _artifact_passed(payload),
                    "promotion_ready": ready,
                    "failures": failures,
                }
            )
            convergence_reports.append(payload)
            convergence_paths.append(path)
        else:
            rows.append(
                {
                    "path": path,
                    "source": "unsupported_window_artifact",
                    "kind": kind,
                    "passed": _artifact_passed(payload),
                    "qualifies_for_replicated_long_window_uncertainty": False,
                }
            )

    derived_ensemble = None
    if len(convergence_reports) >= int(cfg.min_window_reports):
        derived_payload = nonlinear_window_ensemble_report(
            convergence_reports,
            case="derived_long_window_replicate_evidence",
            comparison="derived_from_supplied_window_summaries",
            config=NonlinearWindowEnsembleConfig(
                min_reports=cfg.min_window_reports,
                max_mean_rel_spread=cfg.max_window_mean_rel_spread,
                max_combined_sem_rel=cfg.max_window_combined_sem_rel,
                value_floor=cfg.value_floor,
                require_individual_passed=True,
            ),
        )
        derived_ensemble = _ensemble_row(
            derived_payload,
            path=None,
            source="derived_from_window_summaries",
            config=cfg,
        )
        derived_ensemble["input_paths"] = convergence_paths
        rows.append(derived_ensemble)

    qualifying_rows = [
        row
        for row in rows
        if bool(row.get("qualifies_for_replicated_long_window_uncertainty", False))
    ]
    gates = [
        _gate(
            "replicated_long_window_uncertainty",
            bool(qualifying_rows),
            "qualifying_ensembles={count} min_window_reports={min_reports}".format(
                count=len(qualifying_rows),
                min_reports=cfg.min_window_reports,
            ),
        )
    ]
    return {
        "passed": bool(qualifying_rows),
        "gates": gates,
        "ensemble_rows": rows,
        "single_window_rows": single_window_rows,
        "derived_ensemble": derived_ensemble,
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

    cfg = config or NonlinearTurbulenceGradientFiniteDifferenceConfig()
    delta = float(delta_parameter)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta_parameter must be finite and positive")

    rows = {
        "minus": _ensemble_statistics_row(minus, path=minus_path),
        "baseline": _ensemble_statistics_row(baseline, path=baseline_path),
        "plus": _ensemble_statistics_row(plus, path=plus_path),
    }
    means = {name: _finite_float(row.get("ensemble_mean")) for name, row in rows.items()}
    sems = {name: _finite_float(row.get("combined_sem")) for name, row in rows.items()}
    finite_means = all(value is not None for value in means.values())
    finite_sems = all(value is not None for value in sems.values())

    if finite_means:
        assert means["minus"] is not None
        assert means["baseline"] is not None
        assert means["plus"] is not None
        minus_mean = float(means["minus"])
        baseline_mean = float(means["baseline"])
        plus_mean = float(means["plus"])
        central_gradient = (plus_mean - minus_mean) / (2.0 * delta)
        forward_gradient = (plus_mean - baseline_mean) / delta
        backward_gradient = (baseline_mean - minus_mean) / delta
        response = abs(plus_mean - minus_mean)
        response_fraction = response / max(abs(baseline_mean), float(cfg.value_floor))
        fd_asymmetry_rel = abs(forward_gradient - backward_gradient) / max(
            abs(central_gradient),
            float(cfg.value_floor),
        )
        fd_condition_number = (abs(plus_mean) + abs(minus_mean)) / max(
            response,
            float(cfg.value_floor),
        )
    else:
        central_gradient = math.nan
        forward_gradient = math.nan
        backward_gradient = math.nan
        response = math.nan
        response_fraction = math.nan
        fd_asymmetry_rel = math.nan
        fd_condition_number = math.nan

    if finite_sems:
        assert sems["plus"] is not None
        assert sems["minus"] is not None
        gradient_uncertainty = math.sqrt(float(sems["plus"]) ** 2 + float(sems["minus"]) ** 2) / (
            2.0 * delta
        )
        gradient_uncertainty_rel = gradient_uncertainty / max(
            abs(central_gradient) if math.isfinite(central_gradient) else 0.0,
            float(cfg.value_floor),
        )
    else:
        gradient_uncertainty = math.nan
        gradient_uncertainty_rel = math.nan

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
                _gate(f"{name}_ensemble_passed", bool(row["passed"]), f"path={row.get('path')}"),
                _gate(
                    f"{name}_ensemble_replicated",
                    n_reports is not None and n_reports >= int(cfg.min_window_reports),
                    f"n_reports={n_reports} min={cfg.min_window_reports}",
                ),
            ]
        )
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

    gradient_gates = [
        _gate("finite_window_means", finite_means, f"means={means}"),
        _gate("finite_window_uncertainties", finite_sems, f"combined_sem={sems}"),
        _gate(
            "fd_response_resolved",
            math.isfinite(response_fraction)
            and response_fraction >= float(cfg.min_fd_response_fraction),
            f"response_fraction={response_fraction} min={cfg.min_fd_response_fraction}",
        ),
        _gate(
            "fd_asymmetry_bounded",
            math.isfinite(fd_asymmetry_rel)
            and fd_asymmetry_rel <= float(cfg.max_fd_asymmetry_rel),
            f"fd_asymmetry_rel={fd_asymmetry_rel} max={cfg.max_fd_asymmetry_rel}",
        ),
        _gate(
            "fd_condition_number_bounded",
            math.isfinite(fd_condition_number)
            and fd_condition_number <= float(cfg.max_fd_condition_number),
            f"fd_condition_number={fd_condition_number} max={cfg.max_fd_condition_number}",
        ),
        _gate(
            "gradient_uncertainty_bounded",
            math.isfinite(gradient_uncertainty_rel)
            and gradient_uncertainty_rel <= float(cfg.max_gradient_uncertainty_rel),
            f"gradient_uncertainty_rel={gradient_uncertainty_rel} max={cfg.max_gradient_uncertainty_rel}",
        ),
    ]
    gates = [*source_gates, *window_gates, *gradient_gates]
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
        "metrics": {
            "central_gradient": _json_number(central_gradient),
            "forward_gradient": _json_number(forward_gradient),
            "backward_gradient": _json_number(backward_gradient),
            "response": _json_number(response),
            "response_fraction": _json_number(response_fraction),
            "fd_asymmetry_rel": _json_number(fd_asymmetry_rel),
            "asymmetry_rel": _json_number(fd_asymmetry_rel),
            "fd_condition_number": _json_number(fd_condition_number),
            "condition_number": _json_number(fd_condition_number),
            "gradient_uncertainty": _json_number(gradient_uncertainty),
            "gradient_uncertainty_rel": _json_number(gradient_uncertainty_rel),
            "gradient_relative_uncertainty": _json_number(gradient_uncertainty_rel),
            "baseline_window_mean": means["baseline"],
            "plus_window_mean": means["plus"],
            "minus_window_mean": means["minus"],
            "baseline_window_sem": sems["baseline"],
            "plus_window_sem": sems["plus"],
            "minus_window_sem": sems["minus"],
        },
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


def _metric_margin(
    value: float | None,
    *,
    target: float,
    sense: str,
    cap: float,
    value_floor: float,
) -> float:
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


def _candidate_next_action(
    *,
    passed: bool,
    response_margin: float,
    asymmetry_margin: float,
    uncertainty_margin: float,
    condition_margin: float,
) -> str:
    if passed:
        return "promote only after the source campaign provenance is frozen in docs and CI"
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

    rows: list[dict[str, Any]] = []
    for index, (artifact, path, label) in enumerate(zip(artifacts, path_list, label_list)):
        evidence_cfg = NonlinearTurbulenceGradientEvidenceConfig(
            max_gradient_uncertainty_rel=cfg.max_gradient_uncertainty_rel,
            max_fd_asymmetry_rel=cfg.max_fd_asymmetry_rel,
            max_fd_condition_number=cfg.max_fd_condition_number,
            min_fd_response_fraction=cfg.min_fd_response_fraction,
            value_floor=cfg.value_floor,
        )
        classified = classify_gradient_artifact(artifact, path=path, config=evidence_cfg)
        conditioning = classified.get("conditioning", {})
        if not isinstance(conditioning, dict):
            conditioning = {}
        response_fraction = _finite_float(conditioning.get("response_fraction"))
        fd_asymmetry_rel = _finite_float(conditioning.get("fd_asymmetry_rel"))
        fd_condition_number = _finite_float(conditioning.get("fd_condition_number"))
        gradient_uncertainty_rel = _finite_float(conditioning.get("gradient_uncertainty_rel"))

        response_margin = _metric_margin(
            response_fraction,
            target=cfg.min_fd_response_fraction,
            sense="min",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        )
        asymmetry_margin = _metric_margin(
            fd_asymmetry_rel,
            target=cfg.max_fd_asymmetry_rel,
            sense="max",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        )
        condition_margin = _metric_margin(
            fd_condition_number,
            target=cfg.max_fd_condition_number,
            sense="max",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        )
        uncertainty_margin = _metric_margin(
            gradient_uncertainty_rel,
            target=cfg.max_gradient_uncertainty_rel,
            sense="max",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        )
        margins = {
            "response": response_margin,
            "locality": asymmetry_margin,
            "conditioning": condition_margin,
            "uncertainty": uncertainty_margin,
        }
        weakest_margin = min(margins.values())
        geometric_score = math.prod(max(value, 0.0) for value in margins.values()) ** 0.25
        if not bool(classified.get("explicit_production_scope", False)):
            geometric_score *= 0.5
        passed = bool(classified.get("qualifies_for_production_turbulence_gradient", False))
        parameter_name = str(artifact.get("parameter_name") or label or path or f"candidate_{index}")
        failed_gates = [
            str(gate.get("metric", ""))
            for gate in classified.get("gates", [])
            if isinstance(gate, dict) and not bool(gate.get("passed", False))
        ]
        rows.append(
            {
                "rank": None,
                "index": index,
                "label": str(label or parameter_name),
                "path": path,
                "parameter_name": parameter_name,
                "passed": passed,
                "evidence_class": classified.get("evidence_class"),
                "failed_gates": failed_gates,
                "metrics": {
                    "central_gradient": conditioning.get("central_gradient"),
                    "response_fraction": conditioning.get("response_fraction"),
                    "fd_asymmetry_rel": conditioning.get("fd_asymmetry_rel"),
                    "fd_condition_number": conditioning.get("fd_condition_number"),
                    "gradient_uncertainty_rel": conditioning.get("gradient_uncertainty_rel"),
                },
                "margins": margins,
                "weakest_margin": _json_number(weakest_margin),
                "score": _json_number(geometric_score),
                "next_action": _candidate_next_action(
                    passed=passed,
                    response_margin=response_margin,
                    asymmetry_margin=asymmetry_margin,
                    uncertainty_margin=uncertainty_margin,
                    condition_margin=condition_margin,
                ),
            }
        )

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

    passed_rows = [row for row in rows if bool(row["passed"])]
    local_but_noisy = [
        row
        for row in rows
        if float(row["margins"]["locality"]) >= 1.0 and float(row["margins"]["uncertainty"]) < 1.0
    ]
    quiet_but_nonlocal = [
        row
        for row in rows
        if float(row["margins"]["uncertainty"]) >= 1.0 and float(row["margins"]["locality"]) < 1.0
    ]
    if passed_rows:
        recommendation = (
            "one or more candidates passes the production evidence gates; freeze "
            "the provenance and promote only the passed artifact"
        )
    elif local_but_noisy and quiet_but_nonlocal:
        recommendation = (
            "use an overdetermined least-squares/profile-gradient campaign: current "
            "single-control candidates have complementary locality and uncertainty failures"
        )
    elif local_but_noisy:
        recommendation = (
            "extend statistical power for the best local direction before changing controls"
        )
    elif quiet_but_nonlocal:
        recommendation = (
            "reduce bracket size or choose a nearby/local control before adding replicas"
        )
    else:
        recommendation = (
            "screen new profile-gradient or objective-gradient controls; current candidates "
            "do not isolate a promotable response"
        )

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


def _required_run_rows(config: NonlinearTurbulenceGradientGapConfig) -> list[dict[str, Any]]:
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

    cfg = config or NonlinearTurbulenceGradientEvidenceConfig()
    gap_cfg = gap_config or NonlinearTurbulenceGradientGapConfig()
    passed = bool(evidence_report.get("passed", False))
    blockers = [str(item) for item in evidence_report.get("blockers", [])]
    gradient = evidence_report.get("gradient_artifact")
    if not isinstance(gradient, dict):
        gradient = {}
    windows = evidence_report.get("window_evidence")
    if not isinstance(windows, dict):
        windows = {}
    qualifying_windows = [
        row
        for row in windows.get("ensemble_rows", [])
        if isinstance(row, dict)
        and bool(row.get("qualifies_for_replicated_long_window_uncertainty", False))
    ]
    failed_gradient_gates = [
        {
            "metric": str(gate.get("metric", "")),
            "detail": str(gate.get("detail", "")),
        }
        for gate in gradient.get("gates", [])
        if isinstance(gate, dict) and not bool(gate.get("passed", False))
    ]
    gradient_class = str(gradient.get("evidence_class", ""))
    has_production_candidate = (
        gradient_class == "production_long_window_turbulence_gradient_candidate"
    )
    missing: list[dict[str, Any]] = []
    if "production_gradient_artifact" in blockers:
        if has_production_candidate:
            missing.append(
                {
                    "blocker": "production_gradient_artifact",
                    "needed": (
                        "the current matched long-window production-candidate "
                        "finite-difference artifact must pass all recorded "
                        "response, asymmetry, conditioning, and propagated "
                        "gradient-uncertainty gates"
                    ),
                    "current_artifact_class": gradient.get("evidence_class"),
                    "current_artifact_path": gradient.get("path"),
                    "current_failed_gates": failed_gradient_gates,
                }
            )
        else:
            missing.append(
                {
                    "blocker": "production_gradient_artifact",
                    "needed": (
                        "central finite-difference or adjoint/VJP artifact computed "
                        "from matched long post-transient nonlinear heat-flux windows"
                    ),
                    "current_artifact_class": gradient.get("evidence_class"),
                    "current_artifact_path": gradient.get("path"),
                }
            )
    if "replicated_long_window_uncertainty" in blockers:
        missing.append(
            {
                "blocker": "replicated_long_window_uncertainty",
                "needed": (
                    "at least one baseline/plus/minus campaign with replicated "
                    "post-transient transport-window ensemble gates"
                ),
                "qualifying_window_ensembles": len(qualifying_windows),
            }
        )

    finite_difference_audit = {
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
    return {
        "kind": "nonlinear_turbulence_gradient_evidence_gap_report",
        "claim_level": (
            "fail_closed_production_candidate_gradient_gate_not_resolved"
            if has_production_candidate and not passed
            else "fail_closed_missing_campaign_plan_not_gradient_evidence"
        ),
        "passed": passed,
        "promotion_blocked": not passed,
        "blockers": blockers,
        "missing_evidence": missing,
        "current_gradient_candidate_present": has_production_candidate,
        "current_gradient_failed_gates": failed_gradient_gates,
        "current_window_evidence_passed": bool(windows.get("passed", False)),
        "qualifying_window_ensemble_count": len(qualifying_windows),
        "required_campaign": {
            "case_slug": gap_cfg.case_slug,
            "parameter_name": gap_cfg.parameter_name,
            "perturbation_fraction": gap_cfg.perturbation_fraction,
            "required_runs": _required_run_rows(gap_cfg),
            "finite_difference_audit": finite_difference_audit,
        },
        "requirements": [
            "run baseline, plus-delta, and minus-delta nonlinear simulations with identical numerical settings except the perturbed parameter",
            "use the same seed/timestep replicate labels for all three parameter states",
            "discard the startup transient and average only over the declared post-transient analysis window",
            "build passed ensemble gates for baseline, plus, and minus states before computing the gradient",
            "record finite-difference response, asymmetry, condition number, and gradient uncertainty in the production gradient artifact",
        ],
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


def load_json_artifact(path: str | Path) -> dict[str, Any]:
    """Load a JSON object artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


__all__ = [
    "NON_PRODUCTION_SCOPE_MARKERS",
    "NonlinearTurbulenceGradientCandidateRankingConfig",
    "NonlinearTurbulenceGradientEvidenceConfig",
    "NonlinearTurbulenceGradientFiniteDifferenceConfig",
    "NonlinearTurbulenceGradientGapConfig",
    "classify_gradient_artifact",
    "load_json_artifact",
    "nonlinear_turbulence_gradient_candidate_ranking_report",
    "nonlinear_turbulence_gradient_evidence_gap_report",
    "nonlinear_turbulence_gradient_evidence_report",
    "nonlinear_turbulence_gradient_finite_difference_report",
    "summarize_window_evidence",
]
