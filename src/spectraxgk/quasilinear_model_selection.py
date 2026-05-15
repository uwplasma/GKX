"""Claim-boundary checks for quasilinear model-selection artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REQUIRED_CANDIDATE = "spectral_envelope_ridge"
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


def _optimized_equilibrium_audit_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize optimized-equilibrium nonlinear evidence without broad promotion."""

    kind = str(report.get("kind", ""))
    claim_level = str(report.get("claim_level", ""))
    path = report.get("source_artifact")
    text = _claim_text(report)
    optimized_rows = report.get("optimized_equilibrium_artifacts")
    optimized_rows = optimized_rows if isinstance(optimized_rows, list) else []
    summary = report.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    qualifying_optimized_count = int(
        _finite_float(summary.get("qualifying_optimized_equilibrium_ensembles"))
        or sum(
            1
            for row in optimized_rows
            if isinstance(row, dict)
            and bool(row.get("qualifies_for_production_optimization", False))
        )
        or 0
    )
    optimized_marker = any(marker in text for marker in _OPTIMIZED_EQUILIBRIUM_MARKERS)
    if optimized_rows:
        optimized_marker = optimized_marker or any(
            isinstance(row, dict)
            and (
                bool(row.get("optimized_equilibrium_marker", False))
                or any(
                    marker in str(row.get("path", "")).lower()
                    for marker in _OPTIMIZED_EQUILIBRIUM_MARKERS
                )
                or any(
                    marker in str(row.get("case", "")).lower()
                    for marker in _OPTIMIZED_EQUILIBRIUM_MARKERS
                )
            )
            for row in optimized_rows
        )

    production_guard = kind == "production_nonlinear_turbulent_flux_optimization_guard"
    production_promoted = bool(
        report.get("production_nonlinear_optimization_promoted", False)
    )
    promotion_gate_passed = _nested_gate_passed(report, "promotion_gate")
    gate_report_passed = _nested_gate_passed(report, "gate_report")
    top_level_passed = bool(report.get("passed", False))
    claims_universal = _claims_universal_absolute_flux(report)

    if production_guard:
        audit_passed = bool(
            production_promoted
            and promotion_gate_passed
            and qualifying_optimized_count > 0
        )
    else:
        audit_passed = bool(
            (top_level_passed or promotion_gate_passed or gate_report_passed)
            and optimized_marker
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


def build_quasilinear_model_selection_status(
    *,
    dataset_sufficiency: dict[str, Any] | str | Path,
    candidate_uncertainty: dict[str, Any] | str | Path,
    calibration_reports: Iterable[dict[str, Any] | str | Path] = (),
    optimized_equilibrium_nonlinear_audits: Iterable[
        dict[str, Any] | str | Path
    ] = (),
    required_candidate: str = DEFAULT_REQUIRED_CANDIDATE,
    transport_gate: float | None = None,
    interval_coverage_gate: float | None = None,
    require_optimized_equilibrium_nonlinear_audit: bool = False,
) -> dict[str, Any]:
    """Combine quasilinear model-selection gates into one claim ledger.

    The status is intentionally narrower than an absolute-flux calibration
    report. It passes only when the dataset-volume gate and uncertainty gate
    support the selected reduced candidate while all simple train/holdout
    calibration reports remain unpromoted. This lets documentation state a
    positive model-selection result without implying a runtime absolute-flux
    predictor. Optional optimized-equilibrium nonlinear audit artifacts can
    strengthen the scoped evidence ledger, but they never promote a universal
    absolute-flux claim.
    """

    dataset = _as_dict(dataset_sufficiency)
    candidate = _as_dict(candidate_uncertainty)
    reports = [_as_dict(report) for report in calibration_reports]
    optimized_audits = [
        _as_dict(report) for report in optimized_equilibrium_nonlinear_audits
    ]

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
        if row["claim_level"] == ABSOLUTE_FLUX_PROMOTED_CLAIM and bool(row["passed"])
    ]
    calibration_reports_missing_holdout_metrics = [
        row for row in summaries if row["holdout_mean_abs_relative_error"] is None
    ]
    optimized_audit_summaries = [
        _optimized_equilibrium_audit_summary(report) for report in optimized_audits
    ]
    qualifying_optimized_audits = [
        row
        for row in optimized_audit_summaries
        if bool(row["supports_scoped_optimized_equilibrium_transport"])
    ]
    optimized_audits_claiming_universal_absolute_flux = [
        row
        for row in optimized_audit_summaries
        if bool(row["claims_universal_absolute_flux"])
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
    if optimized_audit_summaries or require_optimized_equilibrium_nonlinear_audit:
        gates.extend(
            [
                _gate(
                    "optimized_equilibrium_nonlinear_audit_present",
                    bool(optimized_audit_summaries),
                    f"audits={len(optimized_audit_summaries)}",
                ),
                _gate(
                    "optimized_equilibrium_nonlinear_audit_qualified",
                    bool(qualifying_optimized_audits),
                    "qualifying_audits="
                    f"{len(qualifying_optimized_audits)}",
                ),
                _gate(
                    "optimized_equilibrium_nonlinear_audit_scope_limited",
                    not optimized_audits_claiming_universal_absolute_flux,
                    "universal_absolute_flux_overclaims="
                    f"{len(optimized_audits_claiming_universal_absolute_flux)}",
                ),
            ]
        )

    passed = all(bool(gate["passed"]) for gate in gates)
    blockers = [gate["metric"] for gate in gates if not bool(gate["passed"])]
    scoped_optimized_evidence = bool(qualifying_optimized_audits)
    return {
        "kind": "quasilinear_model_selection_status",
        "claim_level": (
            "scoped_candidate_model_selection_with_optimized_equilibrium_nonlinear_audit_not_universal_absolute_flux"
            if passed and scoped_optimized_evidence
            else
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
            "requires_optimized_equilibrium_nonlinear_audit": bool(
                require_optimized_equilibrium_nonlinear_audit
            ),
        },
        "absolute_flux_promotion": {
            "universal_absolute_flux_promoted": False,
            "runtime_absolute_flux_predictor_promoted": False,
            "scoped_model_selection_promoted": passed,
            "scoped_optimized_equilibrium_nonlinear_audit_supported": (
                scoped_optimized_evidence
            ),
            "honest_status": (
                "scoped_candidate_with_audited_optimized_equilibrium_evidence_not_universal_absolute_flux"
                if passed and scoped_optimized_evidence
                else "scoped_candidate_only_not_absolute_flux"
                if passed
                else "not_promoted"
            ),
            "blockers": blockers,
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
        "optimized_equilibrium_nonlinear_audits": optimized_audit_summaries,
        "notes": (
            "A passed status promotes only the scoped model-selection result. "
            "Optimized-equilibrium nonlinear audits, when supplied, can support "
            "only that audited equilibrium. The status does not promote a "
            "runtime/TOML absolute-flux predictor or a universal nonlinear "
            "transport model."
        ),
    }


def build_quasilinear_model_selection_status_from_paths(
    *,
    dataset_sufficiency: str | Path,
    candidate_uncertainty: str | Path,
    calibration_reports: Iterable[str | Path],
    optimized_equilibrium_nonlinear_audits: Iterable[str | Path] = (),
    required_candidate: str = DEFAULT_REQUIRED_CANDIDATE,
    require_optimized_equilibrium_nonlinear_audit: bool = False,
) -> dict[str, Any]:
    """Path-based wrapper for artifact scripts and CI checks."""

    calibration_report_paths = tuple(
        _ensure_path_payload(f"calibration_reports[{idx}]", report)
        for idx, report in enumerate(calibration_reports)
    )
    optimized_audit_paths = tuple(
        _ensure_path_payload(
            f"optimized_equilibrium_nonlinear_audits[{idx}]", report
        )
        for idx, report in enumerate(optimized_equilibrium_nonlinear_audits)
    )
    return build_quasilinear_model_selection_status(
        dataset_sufficiency=_ensure_path_payload(
            "dataset_sufficiency", dataset_sufficiency
        ),
        candidate_uncertainty=_ensure_path_payload(
            "candidate_uncertainty", candidate_uncertainty
        ),
        calibration_reports=calibration_report_paths,
        optimized_equilibrium_nonlinear_audits=optimized_audit_paths,
        required_candidate=required_candidate,
        require_optimized_equilibrium_nonlinear_audit=(
            require_optimized_equilibrium_nonlinear_audit
        ),
    )


__all__ = [
    "ABSOLUTE_FLUX_PROMOTED_CLAIM",
    "DEFAULT_REQUIRED_CANDIDATE",
    "build_quasilinear_model_selection_status",
    "build_quasilinear_model_selection_status_from_paths",
]
