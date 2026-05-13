"""Claim-boundary checks for quasilinear model-selection artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REQUIRED_CANDIDATE = "spectral_envelope_ridge"


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


def build_quasilinear_model_selection_status(
    *,
    dataset_sufficiency: dict[str, Any] | str | Path,
    candidate_uncertainty: dict[str, Any] | str | Path,
    calibration_reports: Iterable[dict[str, Any] | str | Path] = (),
    required_candidate: str = DEFAULT_REQUIRED_CANDIDATE,
    transport_gate: float | None = None,
    interval_coverage_gate: float | None = None,
) -> dict[str, Any]:
    """Combine quasilinear model-selection gates into one claim ledger.

    The status is intentionally narrower than an absolute-flux calibration
    report. It passes only when the dataset-volume gate and uncertainty gate
    support the selected reduced candidate while all simple train/holdout
    calibration reports remain unpromoted. This lets documentation state a
    positive model-selection result without implying a runtime absolute-flux
    predictor.
    """

    dataset = _as_dict(dataset_sufficiency)
    candidate = _as_dict(candidate_uncertainty)
    reports = [_as_dict(report) for report in calibration_reports]

    dataset_gate = (
        dataset.get("promotion_gate", {})
        if isinstance(dataset.get("promotion_gate", {}), dict)
        else {}
    )
    candidate_gate = (
        candidate.get("promotion_gate", {})
        if isinstance(candidate.get("promotion_gate", {}), dict)
        else {}
    )
    candidates = candidate.get("candidates", {})
    required_payload = (
        candidates.get(required_candidate, {}) if isinstance(candidates, dict) else {}
    )
    if not isinstance(required_payload, dict):
        required_payload = {}

    accepted = _accepted_candidates(candidate_gate)
    candidate_error = _finite_float(required_payload.get("mean_abs_relative_error"))
    candidate_coverage = _finite_float(
        required_payload.get("prediction_interval_coverage")
    )
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
    null_error = _finite_float(
        candidate_gate.get("null_training_mean_mean_abs_relative_error")
    )
    linear_error = _finite_float(
        candidate_gate.get("linear_weight_mean_abs_relative_error")
    )
    promotion_eligible = bool(required_payload.get("promotion_eligible", True))

    summaries = [_calibration_summary(report) for report in reports]
    promoted_absolute_reports = [
        row
        for row in summaries
        if row["claim_level"] == "calibrated_absolute_flux" and bool(row["passed"])
    ]
    calibration_reports_missing_holdout_metrics = [
        row for row in summaries if row["holdout_mean_abs_relative_error"] is None
    ]

    gates = [
        _gate(
            "dataset_sufficiency_passed",
            bool(dataset_gate.get("passed", False)),
            f"blockers={dataset_gate.get('blockers', [])}",
        ),
        _gate(
            "candidate_uncertainty_passed",
            bool(candidate_gate.get("passed", False)),
            f"accepted={accepted}",
        ),
        _gate(
            "required_candidate_accepted",
            required_candidate in accepted,
            f"required={required_candidate} accepted={accepted}",
        ),
        _gate(
            "required_candidate_eligible",
            promotion_eligible,
            f"eligibility_failures={required_payload.get('eligibility_failures', [])}",
        ),
    ]
    if transport_threshold is not None:
        gates.append(
            _gate(
                "required_candidate_transport_error",
                candidate_error is not None and candidate_error <= transport_threshold,
                f"mean_abs_relative_error={candidate_error} gate={transport_threshold}",
            )
        )
    else:
        gates.append(
            _gate(
                "required_candidate_transport_error",
                False,
                "missing transport_mean_relative_error_gate",
            )
        )
    if coverage_threshold is not None:
        gates.append(
            _gate(
                "required_candidate_interval_coverage",
                candidate_coverage is not None
                and candidate_coverage >= coverage_threshold,
                f"coverage={candidate_coverage} gate={coverage_threshold}",
            )
        )
    else:
        gates.append(
            _gate(
                "required_candidate_interval_coverage",
                False,
                "missing interval_coverage_gate",
            )
        )
    gates.extend(
        [
            _gate(
                "required_candidate_beats_training_mean_null",
                candidate_error is not None
                and null_error is not None
                and candidate_error < null_error,
                f"candidate={candidate_error} null={null_error}",
            ),
            _gate(
                "required_candidate_beats_linear_weight",
                candidate_error is not None
                and linear_error is not None
                and candidate_error < linear_error,
                f"candidate={candidate_error} linear_weight={linear_error}",
            ),
            _gate(
                "absolute_flux_not_promoted",
                not promoted_absolute_reports,
                f"promoted_reports={len(promoted_absolute_reports)}",
            ),
            _gate(
                "calibration_reports_have_holdout_metrics",
                not calibration_reports_missing_holdout_metrics,
                "missing_holdout_metrics="
                f"{len(calibration_reports_missing_holdout_metrics)}",
            ),
        ]
    )

    passed = all(bool(gate["passed"]) for gate in gates)
    blockers = [gate["metric"] for gate in gates if not bool(gate["passed"])]
    return {
        "kind": "quasilinear_model_selection_status",
        "claim_level": (
            "scoped_candidate_model_selection_not_runtime_absolute_flux"
            if passed
            else "model_selection_or_scope_incomplete"
        ),
        "passed": passed,
        "required_candidate": str(required_candidate),
        "accepted_candidates": accepted,
        "promotion_gate": {
            "passed": passed,
            "blockers": blockers,
            "requires_dataset_sufficiency": True,
            "requires_uncertainty_skill": True,
            "requires_no_absolute_flux_promotion": True,
        },
        "metrics": {
            "candidate_mean_abs_relative_error": candidate_error,
            "candidate_prediction_interval_coverage": candidate_coverage,
            "transport_mean_relative_error_gate": transport_threshold,
            "interval_coverage_gate": coverage_threshold,
            "null_training_mean_mean_abs_relative_error": null_error,
            "linear_weight_mean_abs_relative_error": linear_error,
        },
        "gate_report": {
            "case": "quasilinear_model_selection",
            "passed": passed,
            "max_abs_error": 0.0 if passed else 1.0,
            "max_rel_error": 0.0 if passed else 1.0,
            "gates": gates,
        },
        "calibration_reports": summaries,
        "notes": (
            "A passed status promotes only the scoped model-selection result. "
            "It does not promote a runtime/TOML absolute-flux predictor or a "
            "universal nonlinear transport model."
        ),
    }


def build_quasilinear_model_selection_status_from_paths(
    *,
    dataset_sufficiency: str | Path,
    candidate_uncertainty: str | Path,
    calibration_reports: Iterable[str | Path],
    required_candidate: str = DEFAULT_REQUIRED_CANDIDATE,
) -> dict[str, Any]:
    """Path-based wrapper for artifact scripts and CI checks."""

    calibration_report_paths = tuple(
        _ensure_path_payload(f"calibration_reports[{idx}]", report)
        for idx, report in enumerate(calibration_reports)
    )
    return build_quasilinear_model_selection_status(
        dataset_sufficiency=_ensure_path_payload(
            "dataset_sufficiency", dataset_sufficiency
        ),
        candidate_uncertainty=_ensure_path_payload(
            "candidate_uncertainty", candidate_uncertainty
        ),
        calibration_reports=calibration_report_paths,
        required_candidate=required_candidate,
    )


__all__ = [
    "DEFAULT_REQUIRED_CANDIDATE",
    "build_quasilinear_model_selection_status",
    "build_quasilinear_model_selection_status_from_paths",
]
