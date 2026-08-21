"""Tests for quasilinear model-selection claim-boundary utilities."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest
import gkx
from gkx.diagnostics import quasilinear_model_selection as model_selection_inputs
from gkx.diagnostics.quasilinear_model_selection import (
    _required_candidate_metrics,
    build_quasilinear_model_selection_status,
    build_quasilinear_model_selection_status_from_paths,
)


def _dataset_payload(*, passed: bool = True) -> dict:
    return {
        "kind": "quasilinear_dataset_sufficiency",
        "promotion_gate": {"passed": passed, "blockers": [] if passed else ["data"]},
    }


def _candidate_payload(*, accepted: bool = True) -> dict:
    accepted_candidates = ["spectral_envelope_ridge"] if accepted else []
    return {
        "kind": "quasilinear_candidate_uncertainty_report",
        "promotion_gate": {
            "passed": accepted,
            "accepted_candidates": accepted_candidates,
            "transport_mean_relative_error_gate": 0.35,
            "interval_coverage_gate": 0.75,
            "null_training_mean_mean_abs_relative_error": 0.82,
            "linear_weight_mean_abs_relative_error": 0.93,
        },
        "candidates": {
            "spectral_envelope_ridge": {
                "mean_abs_relative_error": 0.24,
                "prediction_interval_coverage": 0.86,
                "promotion_eligible": True,
                "eligibility_failures": [],
            }
        },
    }


def _calibration_report(*, promoted: bool = False) -> dict:
    return {
        "kind": "quasilinear_calibration_report",
        "claim_level": (
            "calibrated_absolute_flux" if promoted else "calibration_dataset"
        ),
        "passed": promoted,
        "by_split": {
            "holdout": {
                "n": 5,
                "mean_abs_relative_error": 0.2 if promoted else 2.5,
            }
        },
    }


def _optimized_equilibrium_audit(*, universal_overclaim: bool = False) -> dict:
    return {
        "kind": "production_nonlinear_turbulent_flux_optimization_guard",
        "claim_level": (
            "calibrated_absolute_flux"
            if universal_overclaim
            else "production_nonlinear_optimization_promoted_by_replicated_transport_windows"
        ),
        "passed": True,
        "production_nonlinear_optimization_promoted": True,
        "promotion_gate": {"passed": True, "blockers": []},
        "summary": {
            "qualifying_optimized_equilibrium_ensembles": 1,
            "production_nonlinear_optimization_ready": 1,
        },
        "optimized_equilibrium_artifacts": [
            {
                "path": "optimized_equilibrium_final.json",
                "case": "optimized_equilibrium_final",
                "optimized_equilibrium_marker": True,
                "qualifies_for_production_optimization": True,
            }
        ],
    }


def test_model_selection_status_promotes_scoped_candidate_not_absolute_flux() -> None:
    status = build_quasilinear_model_selection_status(
        dataset_sufficiency=_dataset_payload(),
        candidate_uncertainty=_candidate_payload(),
        calibration_reports=[_calibration_report()],
    )

    assert (
        gkx.build_quasilinear_model_selection_status
        is build_quasilinear_model_selection_status
    )
    assert status["passed"] is True
    assert (
        status["claim_level"]
        == "scoped_candidate_model_selection_not_runtime_absolute_flux"
    )
    assert status["promotion_gate"]["blockers"] == []
    assert status["metrics"]["candidate_mean_abs_relative_error"] == 0.24
    assert status["calibration_reports"][0]["claim_level"] == "calibration_dataset"
    assert (
        status["absolute_flux_promotion"]["universal_absolute_flux_promoted"] is False
    )


def test_model_selection_status_fails_closed_for_overclaims_or_missing_skill() -> None:
    promoted = build_quasilinear_model_selection_status(
        dataset_sufficiency=_dataset_payload(),
        candidate_uncertainty=_candidate_payload(),
        calibration_reports=[_calibration_report(promoted=True)],
    )
    assert promoted["passed"] is False
    assert "absolute_flux_not_promoted" in promoted["promotion_gate"]["blockers"]

    missing_candidate = build_quasilinear_model_selection_status(
        dataset_sufficiency=_dataset_payload(),
        candidate_uncertainty=_candidate_payload(accepted=False),
        calibration_reports=[_calibration_report()],
    )
    assert missing_candidate["passed"] is False
    assert (
        "required_candidate_accepted" in missing_candidate["promotion_gate"]["blockers"]
    )
    assert (
        "candidate_uncertainty_passed"
        in missing_candidate["promotion_gate"]["blockers"]
    )


def test_model_selection_status_path_wrapper_preserves_source_artifacts(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.json"
    candidate = tmp_path / "candidate.json"
    calibration = tmp_path / "calibration.json"
    dataset.write_text(json.dumps(_dataset_payload()), encoding="utf-8")
    candidate.write_text(json.dumps(_candidate_payload()), encoding="utf-8")
    calibration.write_text(json.dumps(_calibration_report()), encoding="utf-8")

    status = build_quasilinear_model_selection_status_from_paths(
        dataset_sufficiency=dataset,
        candidate_uncertainty=candidate,
        calibration_reports=[calibration],
    )

    assert status["passed"] is True
    assert status["calibration_reports"][0]["artifact"] == str(calibration)
    assert (
        gkx.build_quasilinear_model_selection_status_from_paths
        is build_quasilinear_model_selection_status_from_paths
    )


def test_model_selection_can_include_optimized_equilibrium_audit_without_universal_flux_promotion() -> (
    None
):
    status = build_quasilinear_model_selection_status(
        dataset_sufficiency=_dataset_payload(),
        candidate_uncertainty=_candidate_payload(),
        calibration_reports=[_calibration_report()],
        optimized_equilibrium_nonlinear_audits=[_optimized_equilibrium_audit()],
        require_optimized_equilibrium_nonlinear_audit=True,
    )

    assert status["passed"] is True
    assert (
        status["claim_level"]
        == "scoped_candidate_model_selection_with_optimized_equilibrium_nonlinear_audit_not_universal_absolute_flux"
    )
    assert (
        status["absolute_flux_promotion"]["honest_status"]
        == "scoped_candidate_with_audited_optimized_equilibrium_evidence_not_universal_absolute_flux"
    )
    assert (
        status["absolute_flux_promotion"]["universal_absolute_flux_promoted"] is False
    )
    assert (
        status["absolute_flux_promotion"][
            "scoped_optimized_equilibrium_nonlinear_audit_supported"
        ]
        is True
    )
    assert (
        status["optimized_equilibrium_nonlinear_audits"][0][
            "supports_scoped_optimized_equilibrium_transport"
        ]
        is True
    )


def test_model_selection_fails_closed_for_missing_required_optimized_audit() -> None:
    status = build_quasilinear_model_selection_status(
        dataset_sufficiency=_dataset_payload(),
        candidate_uncertainty=_candidate_payload(),
        calibration_reports=[_calibration_report()],
        require_optimized_equilibrium_nonlinear_audit=True,
    )

    assert status["passed"] is False
    assert (
        "optimized_equilibrium_nonlinear_audit_present"
        in status["promotion_gate"]["blockers"]
    )
    assert (
        "optimized_equilibrium_nonlinear_audit_qualified"
        in status["promotion_gate"]["blockers"]
    )


def test_model_selection_rejects_optimized_audit_universal_absolute_flux_overclaim() -> (
    None
):
    status = build_quasilinear_model_selection_status(
        dataset_sufficiency=_dataset_payload(),
        candidate_uncertainty=_candidate_payload(),
        calibration_reports=[_calibration_report()],
        optimized_equilibrium_nonlinear_audits=[
            _optimized_equilibrium_audit(universal_overclaim=True)
        ],
    )

    assert status["passed"] is False
    assert (
        "optimized_equilibrium_nonlinear_audit_scope_limited"
        in status["promotion_gate"]["blockers"]
    )
    assert (
        status["optimized_equilibrium_nonlinear_audits"][0][
            "claims_universal_absolute_flux"
        ]
        is True
    )


# Additional fail-closed model-selection contracts.
def _selection_extra_dataset_payload() -> dict[str, Any]:
    return {
        "kind": "quasilinear_dataset_sufficiency",
        "promotion_gate": {"passed": True, "blockers": []},
    }


def _selection_extra_candidate_payload() -> dict[str, Any]:
    return {
        "kind": "quasilinear_candidate_uncertainty_report",
        "promotion_gate": {
            "passed": True,
            "accepted_candidates": ["spectral_envelope_ridge"],
            "transport_mean_relative_error_gate": 0.35,
            "interval_coverage_gate": 0.75,
            "null_training_mean_mean_abs_relative_error": 0.82,
            "linear_weight_mean_abs_relative_error": 0.93,
        },
        "candidates": {
            "spectral_envelope_ridge": {
                "mean_abs_relative_error": 0.24,
                "prediction_interval_coverage": 0.86,
                "promotion_eligible": True,
                "eligibility_failures": [],
            }
        },
    }


def _selection_extra_calibration_report(
    *,
    claim_level: str = "calibration_dataset",
    passed: object = False,
    include_by_split: bool = True,
) -> dict[str, Any]:
    report = {
        "kind": "quasilinear_calibration_report",
        "claim_level": claim_level,
        "passed": passed,
    }
    if include_by_split:
        report["by_split"] = {"holdout": {"n": 3, "mean_abs_relative_error": 1.8}}
    return report


def _selection_extra_optimized_equilibrium_audit(*, passed: bool = True) -> dict[str, Any]:
    return {
        "kind": "production_nonlinear_turbulent_flux_optimization_guard",
        "claim_level": "production_nonlinear_optimization_promoted_by_replicated_transport_windows",
        "passed": True,
        "production_nonlinear_optimization_promoted": passed,
        "promotion_gate": {"passed": passed, "blockers": [] if passed else ["audit"]},
        "summary": {
            "qualifying_optimized_equilibrium_ensembles": 1 if passed else 0,
        },
        "optimized_equilibrium_artifacts": [
            {
                "path": "optimized_equilibrium_post_optimization_gate.json",
                "optimized_equilibrium_marker": True,
                "qualifies_for_production_optimization": passed,
            }
        ],
    }


def _selection_extra_write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_model_selection_input_helpers_have_single_canonical_owner() -> None:
    assert inspect.getmodule(_required_candidate_metrics) is model_selection_inputs
    assert (
        inspect.getmodule(model_selection_inputs._optimized_equilibrium_audit_summary)
        is model_selection_inputs
    )


def test_required_candidate_metrics_normalize_thresholds_and_payloads() -> None:
    candidate = _selection_extra_candidate_payload()
    metrics = _required_candidate_metrics(
        candidate,
        candidate["promotion_gate"],
        required_candidate="spectral_envelope_ridge",
        transport_gate=0.3,
        interval_coverage_gate=0.8,
    )

    assert metrics["accepted"] == ["spectral_envelope_ridge"]
    assert metrics["candidate_error"] == 0.24
    assert metrics["candidate_coverage"] == 0.86
    assert metrics["transport_threshold"] == 0.3
    assert metrics["coverage_threshold"] == 0.8
    assert metrics["null_error"] == 0.82
    assert metrics["linear_error"] == 0.93
    assert metrics["promotion_eligible"] is True

    malformed = {
        "candidates": {"spectral_envelope_ridge": "not-a-dict"},
    }
    gate = {
        "accepted_rules": ["spectral_envelope_ridge"],
        "transport_mean_relative_error_gate": "0.35",
        "interval_coverage_gate": "0.75",
    }
    missing_metrics = _required_candidate_metrics(
        malformed,
        gate,
        required_candidate="spectral_envelope_ridge",
        transport_gate=None,
        interval_coverage_gate=None,
    )

    assert missing_metrics["accepted"] == ["spectral_envelope_ridge"]
    assert missing_metrics["required_payload"] == {}
    assert missing_metrics["candidate_error"] is None
    assert missing_metrics["candidate_coverage"] is None
    assert missing_metrics["transport_threshold"] == 0.35
    assert missing_metrics["coverage_threshold"] == 0.75
    assert missing_metrics["promotion_eligible"] is True


def test_model_selection_path_inputs_reject_malformed_json_payloads(
    tmp_path: Path,
) -> None:
    dataset = _selection_extra_write_json(tmp_path / "dataset.json", _selection_extra_dataset_payload())
    candidate = _selection_extra_write_json(tmp_path / "candidate.json", _selection_extra_candidate_payload())
    calibration = _selection_extra_write_json(tmp_path / "calibration.json", _selection_extra_calibration_report())

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError):
        build_quasilinear_model_selection_status_from_paths(
            dataset_sufficiency=bad_json,
            candidate_uncertainty=candidate,
            calibration_reports=[calibration],
        )

    array_payload = _selection_extra_write_json(tmp_path / "array.json", [])
    with pytest.raises(ValueError, match="must contain a JSON object"):
        build_quasilinear_model_selection_status_from_paths(
            dataset_sufficiency=dataset,
            candidate_uncertainty=array_payload,
            calibration_reports=[calibration],
        )


def test_model_selection_fails_closed_when_required_candidate_metrics_are_missing() -> (
    None
):
    candidate = _selection_extra_candidate_payload()
    candidate["candidates"] = {"spectral_envelope_ridge": {"promotion_eligible": True}}

    status = build_quasilinear_model_selection_status(
        dataset_sufficiency=_selection_extra_dataset_payload(),
        candidate_uncertainty=candidate,
        calibration_reports=[_selection_extra_calibration_report()],
    )

    assert status["passed"] is False
    assert status["metrics"]["candidate_mean_abs_relative_error"] is None
    assert status["metrics"]["candidate_prediction_interval_coverage"] is None
    assert {
        "required_candidate_transport_error",
        "required_candidate_interval_coverage",
        "required_candidate_beats_training_mean_null",
        "required_candidate_beats_linear_weight",
    }.issubset(status["promotion_gate"]["blockers"])


def test_model_selection_fails_closed_on_absolute_flux_overclaim_without_holdout() -> (
    None
):
    status = build_quasilinear_model_selection_status(
        dataset_sufficiency=_selection_extra_dataset_payload(),
        candidate_uncertainty=_selection_extra_candidate_payload(),
        calibration_reports=[
            _selection_extra_calibration_report(
                claim_level="calibrated_absolute_flux",
                passed="yes",
                include_by_split=False,
            )
        ],
    )

    assert status["passed"] is False
    assert "absolute_flux_not_promoted" in status["promotion_gate"]["blockers"]
    assert status["calibration_reports"][0]["holdout_mean_abs_relative_error"] is None


def test_model_selection_fails_closed_for_selection_extra_calibration_report_missing_by_split() -> None:
    status = build_quasilinear_model_selection_status(
        dataset_sufficiency=_selection_extra_dataset_payload(),
        candidate_uncertainty=_selection_extra_candidate_payload(),
        calibration_reports=[_selection_extra_calibration_report(include_by_split=False)],
    )

    assert status["passed"] is False
    assert (
        "calibration_reports_have_holdout_metrics"
        in status["promotion_gate"]["blockers"]
    )
    assert status["calibration_reports"][0]["holdout_mean_abs_relative_error"] is None


def test_model_selection_path_wrapper_requires_path_payloads(tmp_path: Path) -> None:
    candidate = _selection_extra_write_json(tmp_path / "candidate.json", _selection_extra_candidate_payload())
    calibration = _selection_extra_write_json(tmp_path / "calibration.json", _selection_extra_calibration_report())

    with pytest.raises(TypeError, match="dataset_sufficiency must be a path"):
        build_quasilinear_model_selection_status_from_paths(
            dataset_sufficiency=_selection_extra_dataset_payload(),  # type: ignore[arg-type]
            candidate_uncertainty=candidate,
            calibration_reports=[calibration],
        )

    dataset = _selection_extra_write_json(tmp_path / "dataset.json", _selection_extra_dataset_payload())
    with pytest.raises(TypeError, match=r"calibration_reports\[0\] must be a path"):
        build_quasilinear_model_selection_status_from_paths(
            dataset_sufficiency=dataset,
            candidate_uncertainty=candidate,
            calibration_reports=[_selection_extra_calibration_report()],  # type: ignore[list-item]
        )

    with pytest.raises(
        TypeError, match=r"optimized_equilibrium_nonlinear_audits\[0\] must be a path"
    ):
        build_quasilinear_model_selection_status_from_paths(
            dataset_sufficiency=dataset,
            candidate_uncertainty=candidate,
            calibration_reports=[calibration],
            optimized_equilibrium_nonlinear_audits=[_selection_extra_optimized_equilibrium_audit()],  # type: ignore[list-item]
        )


def test_model_selection_path_wrapper_accepts_selection_extra_optimized_equilibrium_audit(
    tmp_path: Path,
) -> None:
    dataset = _selection_extra_write_json(tmp_path / "dataset.json", _selection_extra_dataset_payload())
    candidate = _selection_extra_write_json(tmp_path / "candidate.json", _selection_extra_candidate_payload())
    calibration = _selection_extra_write_json(tmp_path / "calibration.json", _selection_extra_calibration_report())
    audit = _selection_extra_write_json(
        tmp_path / "optimized_audit.json", _selection_extra_optimized_equilibrium_audit()
    )

    status = build_quasilinear_model_selection_status_from_paths(
        dataset_sufficiency=dataset,
        candidate_uncertainty=candidate,
        calibration_reports=[calibration],
        optimized_equilibrium_nonlinear_audits=[audit],
        require_optimized_equilibrium_nonlinear_audit=True,
    )

    assert status["passed"] is True
    assert status["optimized_equilibrium_nonlinear_audits"][0]["artifact"] == str(audit)


def test_model_selection_fails_closed_for_supplied_failed_optimized_audit() -> None:
    status = build_quasilinear_model_selection_status(
        dataset_sufficiency=_selection_extra_dataset_payload(),
        candidate_uncertainty=_selection_extra_candidate_payload(),
        calibration_reports=[_selection_extra_calibration_report()],
        optimized_equilibrium_nonlinear_audits=[
            _selection_extra_optimized_equilibrium_audit(passed=False)
        ],
    )

    assert status["passed"] is False
    assert (
        "optimized_equilibrium_nonlinear_audit_qualified"
        in status["promotion_gate"]["blockers"]
    )
    assert (
        "optimized_equilibrium_audit_not_passed"
        in status["optimized_equilibrium_nonlinear_audits"][0]["blockers"]
    )


def test_model_selection_helpers_are_exported_at_package_top_level() -> None:
    import gkx as sgk

    status = sgk.build_quasilinear_model_selection_status(
        dataset_sufficiency=_selection_extra_dataset_payload(),
        candidate_uncertainty=_selection_extra_candidate_payload(),
        calibration_reports=[_selection_extra_calibration_report()],
    )

    assert status["passed"] is True
    assert status["required_candidate"] == "spectral_envelope_ridge"
