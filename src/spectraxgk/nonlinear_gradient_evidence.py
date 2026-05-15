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


def _json_number(value: Any) -> float | int | None:
    if value is None:
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
    if payload.get("production_nonlinear_window_gradient_gate") is False:
        blockers.append("production_nonlinear_window_gradient_gate_false")
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


def nonlinear_turbulence_gradient_evidence_report(
    gradient_artifact: dict[str, Any],
    *,
    window_artifacts: Sequence[dict[str, Any]] = (),
    gradient_path: str | None = None,
    window_paths: Sequence[str | None] | None = None,
    config: NonlinearTurbulenceGradientEvidenceConfig | None = None,
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
    return {
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


def load_json_artifact(path: str | Path) -> dict[str, Any]:
    """Load a JSON object artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


__all__ = [
    "NON_PRODUCTION_SCOPE_MARKERS",
    "NonlinearTurbulenceGradientEvidenceConfig",
    "classify_gradient_artifact",
    "load_json_artifact",
    "nonlinear_turbulence_gradient_evidence_report",
    "summarize_window_evidence",
]
