"""Input normalization and summaries for quasilinear model-selection gates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ABSOLUTE_FLUX_PROMOTED_CLAIM = "calibrated_absolute_flux"

_OPTIMIZED_EQUILIBRIUM_MARKERS = (
    "optimized_equilibrium",
    "optimized-equilibrium",
    "post_optimization",
    "post-optimization",
)


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _gate(metric: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"metric": metric, "passed": bool(passed), "detail": str(detail)}


def _as_dict(payload: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    path = Path(payload)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    out = dict(data)
    out.setdefault("source_artifact", str(path))
    return out


def _ensure_path_payload(name: str, payload: object) -> str | Path:
    if not isinstance(payload, (str, Path)):
        raise TypeError(f"{name} must be a path, got {type(payload).__name__}")
    return payload


def _accepted_candidates(gate: dict[str, Any]) -> list[str]:
    raw = gate.get("accepted_candidates", gate.get("accepted_rules", []))
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def _required_candidate_metrics(
    candidate: dict[str, Any],
    candidate_gate: dict[str, Any],
    *,
    required_candidate: str,
    transport_gate: float | None,
    interval_coverage_gate: float | None,
) -> dict[str, Any]:
    """Collect the metric bundle used by required-candidate promotion gates."""

    candidates = candidate.get("candidates", {})
    required_payload = (
        candidates.get(required_candidate, {}) if isinstance(candidates, dict) else {}
    )
    if not isinstance(required_payload, dict):
        required_payload = {}

    transport_threshold = (
        _finite_float(candidate_gate.get("transport_mean_relative_error_gate"))
        if transport_gate is None
        else float(transport_gate)
    )
    coverage_threshold = (
        _finite_float(candidate_gate.get("interval_coverage_gate"))
        if interval_coverage_gate is None
        else float(interval_coverage_gate)
    )
    return {
        "accepted": _accepted_candidates(candidate_gate),
        "required_payload": required_payload,
        "candidate_error": _finite_float(
            required_payload.get("mean_abs_relative_error")
        ),
        "candidate_coverage": _finite_float(
            required_payload.get("prediction_interval_coverage")
        ),
        "transport_threshold": transport_threshold,
        "coverage_threshold": coverage_threshold,
        "null_error": _finite_float(
            candidate_gate.get("null_training_mean_mean_abs_relative_error")
        ),
        "linear_error": _finite_float(
            candidate_gate.get("linear_weight_mean_abs_relative_error")
        ),
        "promotion_eligible": bool(required_payload.get("promotion_eligible", True)),
    }


def _calibration_summary(report: dict[str, Any]) -> dict[str, Any]:
    by_split = report.get("by_split", {})
    holdout = by_split.get("holdout", {}) if isinstance(by_split, dict) else {}
    holdout_error = (
        holdout.get("mean_abs_relative_error")
        if isinstance(holdout, dict)
        else None
    )
    return {
        "artifact": report.get("source_artifact"),
        "kind": report.get("kind", "unknown"),
        "claim_level": report.get("claim_level"),
        "passed": bool(report.get("passed", False)),
        "holdout_mean_abs_relative_error": _finite_float(holdout_error),
    }


def _claim_text(report: dict[str, Any]) -> str:
    fields = (
        "kind",
        "case",
        "claim_level",
        "claim_scope",
        "notes",
        "next_action",
        "source_artifact",
    )
    return " ".join(str(report.get(field, "")) for field in fields).lower()


def _nested_gate_passed(report: dict[str, Any], key: str) -> bool:
    gate = report.get(key)
    return isinstance(gate, dict) and bool(gate.get("passed", False))


def _claims_universal_absolute_flux(report: dict[str, Any]) -> bool:
    claim = str(report.get("claim_level", ""))
    if claim == ABSOLUTE_FLUX_PROMOTED_CLAIM:
        return True
    for key in (
        "absolute_flux_promoted",
        "universal_absolute_flux_promoted",
        "runtime_absolute_flux_predictor",
    ):
        if bool(report.get(key, False)):
            return True
    return False


def _optimized_equilibrium_rows(report: dict[str, Any]) -> list[object]:
    rows = report.get("optimized_equilibrium_artifacts")
    return rows if isinstance(rows, list) else []


def _qualifying_optimized_count(
    report: dict[str, Any], optimized_rows: list[object]
) -> int:
    summary = report.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    return int(
        _finite_float(summary.get("qualifying_optimized_equilibrium_ensembles"))
        or sum(
            1
            for row in optimized_rows
            if isinstance(row, dict)
            and bool(row.get("qualifies_for_production_optimization", False))
        )
        or 0
    )


def _optimized_row_has_marker(row: object) -> bool:
    if not isinstance(row, dict):
        return False
    return bool(row.get("optimized_equilibrium_marker", False)) or any(
        marker in str(row.get(field, "")).lower()
        for field in ("path", "case")
        for marker in _OPTIMIZED_EQUILIBRIUM_MARKERS
    )


def _has_optimized_equilibrium_marker(
    report: dict[str, Any], optimized_rows: list[object]
) -> bool:
    if any(marker in _claim_text(report) for marker in _OPTIMIZED_EQUILIBRIUM_MARKERS):
        return True
    return any(_optimized_row_has_marker(row) for row in optimized_rows)


def _optimized_equilibrium_audit_passed(
    report: dict[str, Any],
    *,
    production_guard: bool,
    production_promoted: bool,
    promotion_gate_passed: bool,
    gate_report_passed: bool,
    optimized_marker: bool,
    qualifying_optimized_count: int,
) -> bool:
    if production_guard:
        return bool(
            production_promoted
            and promotion_gate_passed
            and qualifying_optimized_count > 0
        )
    return bool(
        (
            bool(report.get("passed", False))
            or promotion_gate_passed
            or gate_report_passed
        )
        and optimized_marker
    )


def _optimized_equilibrium_audit_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize optimized-equilibrium nonlinear evidence without broad promotion."""

    kind = str(report.get("kind", ""))
    claim_level = str(report.get("claim_level", ""))
    path = report.get("source_artifact")
    optimized_rows = _optimized_equilibrium_rows(report)
    qualifying_optimized_count = _qualifying_optimized_count(report, optimized_rows)
    optimized_marker = _has_optimized_equilibrium_marker(report, optimized_rows)
    production_guard = kind == "production_nonlinear_turbulent_flux_optimization_guard"
    production_promoted = bool(
        report.get("production_nonlinear_optimization_promoted", False)
    )
    promotion_gate_passed = _nested_gate_passed(report, "promotion_gate")
    gate_report_passed = _nested_gate_passed(report, "gate_report")
    top_level_passed = bool(report.get("passed", False))
    claims_universal = _claims_universal_absolute_flux(report)
    audit_passed = _optimized_equilibrium_audit_passed(
        report,
        production_guard=production_guard,
        production_promoted=production_promoted,
        promotion_gate_passed=promotion_gate_passed,
        gate_report_passed=gate_report_passed,
        optimized_marker=optimized_marker,
        qualifying_optimized_count=qualifying_optimized_count,
    )
    supports_scoped = bool(audit_passed and optimized_marker and not claims_universal)
    blockers: list[str] = []
    if not optimized_marker:
        blockers.append("missing_optimized_equilibrium_marker")
    if not audit_passed:
        blockers.append("optimized_equilibrium_audit_not_passed")
    if claims_universal:
        blockers.append("universal_absolute_flux_overclaim")

    return {
        "artifact": path,
        "kind": kind,
        "claim_level": claim_level,
        "passed": top_level_passed,
        "promotion_gate_passed": promotion_gate_passed,
        "production_nonlinear_optimization_promoted": production_promoted,
        "optimized_equilibrium_marker": optimized_marker,
        "qualifying_optimized_equilibrium_ensembles": qualifying_optimized_count,
        "claims_universal_absolute_flux": claims_universal,
        "supports_scoped_optimized_equilibrium_transport": supports_scoped,
        "blockers": blockers,
    }


__all__ = ["ABSOLUTE_FLUX_PROMOTED_CLAIM"]
