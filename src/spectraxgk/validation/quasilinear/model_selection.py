"""Claim-boundary checks for quasilinear model-selection artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from spectraxgk.validation.quasilinear.model_selection_inputs import (
    ABSOLUTE_FLUX_PROMOTED_CLAIM,
    _as_dict,
    _calibration_summary,
    _ensure_path_payload,
    _gate,
    _optimized_equilibrium_audit_summary,
    _required_candidate_metrics,
)

DEFAULT_REQUIRED_CANDIDATE = "spectral_envelope_ridge"



def _promotion_gate(payload: dict[str, Any]) -> dict[str, Any]:
    gate = payload.get("promotion_gate", {})
    return gate if isinstance(gate, dict) else {}


def _calibration_gate_context(
    reports: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = [_calibration_summary(report) for report in reports]
    promoted = [
        row
        for row in summaries
        if row["claim_level"] == ABSOLUTE_FLUX_PROMOTED_CLAIM and bool(row["passed"])
    ]
    missing_holdout = [
        row for row in summaries if row["holdout_mean_abs_relative_error"] is None
    ]
    return summaries, promoted, missing_holdout


def _optimized_audit_gate_context(
    audits: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = [_optimized_equilibrium_audit_summary(report) for report in audits]
    qualifying = [
        row
        for row in summaries
        if bool(row["supports_scoped_optimized_equilibrium_transport"])
    ]
    overclaims = [row for row in summaries if bool(row["claims_universal_absolute_flux"])]
    return summaries, qualifying, overclaims


def _required_candidate_gate_rows(
    *,
    dataset_gate: dict[str, Any],
    candidate_gate: dict[str, Any],
    candidate_metrics: dict[str, Any],
    required_candidate: str,
) -> list[dict[str, Any]]:
    accepted = candidate_metrics["accepted"]
    required_payload = candidate_metrics["required_payload"]
    candidate_error = candidate_metrics["candidate_error"]
    candidate_coverage = candidate_metrics["candidate_coverage"]
    transport_threshold = candidate_metrics["transport_threshold"]
    coverage_threshold = candidate_metrics["coverage_threshold"]
    null_error = candidate_metrics["null_error"]
    linear_error = candidate_metrics["linear_error"]
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
            bool(candidate_metrics["promotion_eligible"]),
            f"eligibility_failures={required_payload.get('eligibility_failures', [])}",
        ),
    ]
    gates.append(
        _gate(
            "required_candidate_transport_error",
            candidate_error is not None
            and transport_threshold is not None
            and candidate_error <= transport_threshold,
            f"mean_abs_relative_error={candidate_error} gate={transport_threshold}"
            if transport_threshold is not None
            else "missing transport_mean_relative_error_gate",
        )
    )
    gates.append(
        _gate(
            "required_candidate_interval_coverage",
            candidate_coverage is not None
            and coverage_threshold is not None
            and candidate_coverage >= coverage_threshold,
            f"coverage={candidate_coverage} gate={coverage_threshold}"
            if coverage_threshold is not None
            else "missing interval_coverage_gate",
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
        ]
    )
    return gates


def _claim_boundary_gate_rows(
    *,
    promoted_absolute_reports: list[dict[str, Any]],
    calibration_reports_missing_holdout_metrics: list[dict[str, Any]],
    optimized_audit_summaries: list[dict[str, Any]],
    qualifying_optimized_audits: list[dict[str, Any]],
    optimized_audits_claiming_universal_absolute_flux: list[dict[str, Any]],
    require_optimized_equilibrium_nonlinear_audit: bool,
) -> list[dict[str, Any]]:
    gates = [
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
                    "qualifying_audits=" f"{len(qualifying_optimized_audits)}",
                ),
                _gate(
                    "optimized_equilibrium_nonlinear_audit_scope_limited",
                    not optimized_audits_claiming_universal_absolute_flux,
                    "universal_absolute_flux_overclaims="
                    f"{len(optimized_audits_claiming_universal_absolute_flux)}",
                ),
            ]
        )
    return gates


def _claim_level(passed: bool, scoped_optimized_evidence: bool) -> str:
    if passed and scoped_optimized_evidence:
        return "scoped_candidate_model_selection_with_optimized_equilibrium_nonlinear_audit_not_universal_absolute_flux"
    if passed:
        return "scoped_candidate_model_selection_not_runtime_absolute_flux"
    return "model_selection_or_scope_incomplete"


def _absolute_flux_promotion_status(
    *, passed: bool, scoped_optimized_evidence: bool, blockers: list[str]
) -> dict[str, Any]:
    if passed and scoped_optimized_evidence:
        honest_status = (
            "scoped_candidate_with_audited_optimized_equilibrium_evidence_not_universal_absolute_flux"
        )
    elif passed:
        honest_status = "scoped_candidate_only_not_absolute_flux"
    else:
        honest_status = "not_promoted"
    return {
        "universal_absolute_flux_promoted": False,
        "runtime_absolute_flux_predictor_promoted": False,
        "scoped_model_selection_promoted": passed,
        "scoped_optimized_equilibrium_nonlinear_audit_supported": (
            scoped_optimized_evidence
        ),
        "honest_status": honest_status,
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
    dataset_gate = _promotion_gate(dataset)
    candidate_gate = _promotion_gate(candidate)
    candidate_metrics = _required_candidate_metrics(
        candidate,
        candidate_gate,
        required_candidate=required_candidate,
        transport_gate=transport_gate,
        interval_coverage_gate=interval_coverage_gate,
    )
    summaries, promoted_reports, missing_holdout = _calibration_gate_context(reports)
    audit_summaries, qualifying_audits, universal_overclaims = (
        _optimized_audit_gate_context(optimized_audits)
    )
    gates = _required_candidate_gate_rows(
        dataset_gate=dataset_gate,
        candidate_gate=candidate_gate,
        candidate_metrics=candidate_metrics,
        required_candidate=required_candidate,
    )
    gates.extend(
        _claim_boundary_gate_rows(
            promoted_absolute_reports=promoted_reports,
            calibration_reports_missing_holdout_metrics=missing_holdout,
            optimized_audit_summaries=audit_summaries,
            qualifying_optimized_audits=qualifying_audits,
            optimized_audits_claiming_universal_absolute_flux=universal_overclaims,
            require_optimized_equilibrium_nonlinear_audit=(
                require_optimized_equilibrium_nonlinear_audit
            ),
        )
    )

    passed = all(bool(gate["passed"]) for gate in gates)
    blockers = [gate["metric"] for gate in gates if not bool(gate["passed"])]
    scoped_optimized_evidence = bool(qualifying_audits)
    return {
        "kind": "quasilinear_model_selection_status",
        "claim_level": _claim_level(passed, scoped_optimized_evidence),
        "passed": passed,
        "required_candidate": str(required_candidate),
        "accepted_candidates": candidate_metrics["accepted"],
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
        "absolute_flux_promotion": _absolute_flux_promotion_status(
            passed=passed,
            scoped_optimized_evidence=scoped_optimized_evidence,
            blockers=blockers,
        ),
        "metrics": {
            "candidate_mean_abs_relative_error": candidate_metrics["candidate_error"],
            "candidate_prediction_interval_coverage": candidate_metrics[
                "candidate_coverage"
            ],
            "transport_mean_relative_error_gate": candidate_metrics[
                "transport_threshold"
            ],
            "interval_coverage_gate": candidate_metrics["coverage_threshold"],
            "null_training_mean_mean_abs_relative_error": candidate_metrics[
                "null_error"
            ],
            "linear_weight_mean_abs_relative_error": candidate_metrics[
                "linear_error"
            ],
        },
        "gate_report": {
            "case": "quasilinear_model_selection",
            "passed": passed,
            "max_abs_error": 0.0 if passed else 1.0,
            "max_rel_error": 0.0 if passed else 1.0,
            "gates": gates,
        },
        "calibration_reports": summaries,
        "optimized_equilibrium_nonlinear_audits": audit_summaries,
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
