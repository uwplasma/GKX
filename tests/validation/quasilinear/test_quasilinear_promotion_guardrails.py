"""Tests for quasilinear absolute-flux promotion guardrails."""

from __future__ import annotations

import json
import os
from pathlib import Path

from support.paths import REPO_ROOT, load_release_tool
import subprocess
import sys


def _load_tool_module():
    return load_release_tool("check_quasilinear_promotion_guardrails")


def _write_doc(path: Path, text: str | None = None) -> None:
    path.write_text(
        text
        or (
            "This diagnostic is not a runtime/TOML absolute-flux predictor. "
            "Absolute-flux prediction not promoted.\n"
        ),
        encoding="utf-8",
    )


def _window_stats(*, passed: bool = True) -> dict:
    return {
        "kind": "nonlinear_window_convergence_report",
        "passed": passed,
        "statistics": {
            "late_mean": 1.0,
            "sem": 0.02,
            "block_bootstrap_sem": 0.02,
            "running_mean_rel_drift": 0.01,
        },
        "window": {
            "input_tmin": None,
            "transient_fraction": 0.5,
            "transient_cutoff": 50.0,
            "late_tmin": 50.0,
            "late_tmax": 100.0,
            "n_finite_late": 64,
        },
        "provenance": {"source_artifact": "tools_out/nonlinear_trace.csv"},
        "gate_report": {"passed": passed},
    }


def _calibration_report(
    *, claim_level: str, passed: bool, holdout_error: float
) -> dict:
    return {
        "kind": "quasilinear_calibration_report",
        "claim_level": claim_level,
        "passed": passed,
        "holdout_mean_rel_gate": 0.35,
        "metadata": {"calibration_policy": "one constant train/holdout"},
        "by_split": {
            "train": {"n": 1, "mean_abs_relative_error": 0.0},
            "holdout": {"n": 1, "mean_abs_relative_error": holdout_error},
        },
        "points": [
            {
                "case": "train",
                "split": "train",
                "geometry": "cyclone",
                "electron_model": "adiabatic",
                "saturation_rule": "mixing_length",
                "nonlinear_artifact": "tools_out/train.csv",
                "quasilinear_artifact": "docs/_static/train_spectrum.csv",
                "predicted_heat_flux": 1.0,
                "raw_predicted_heat_flux": 0.5,
                "calibration_scale": 2.0,
                "observed_heat_flux": 1.0,
                "observed_heat_flux_std": 0.1,
                "nonlinear_window_stats": _window_stats(),
            },
            {
                "case": "holdout",
                "split": "holdout",
                "geometry": "miller",
                "electron_model": "adiabatic",
                "saturation_rule": "mixing_length",
                "nonlinear_artifact": "tools_out/holdout.csv",
                "quasilinear_artifact": "docs/_static/holdout_spectrum.csv",
                "predicted_heat_flux": 1.0,
                "raw_predicted_heat_flux": 0.5,
                "calibration_scale": 2.0,
                "observed_heat_flux": 1.0,
                "observed_heat_flux_std": 0.2,
                "nonlinear_window_stats": _window_stats(),
            },
        ],
    }


def _release_contract(
    *,
    claim_level: str = "scoped_candidate_model_selection_not_runtime_flux_predictor",
    absolute_flux_promoted: bool = False,
    include_guardrail_artifact: bool = True,
) -> dict:
    artifacts = [
        "docs/_static/quasilinear_validated_calibration_inputs.json",
        "docs/_static/quasilinear_candidate_uncertainty.json",
    ]
    if include_guardrail_artifact:
        artifacts.append("docs/_static/quasilinear_promotion_guardrails.json")
    return {
        "kind": "gkx_1_7_frozen_release_contract",
        "release_lanes": [
            {
                "lane": "Quasilinear diagnostics and saturation-model selection",
                "status": "closed",
                "claim_level": claim_level,
                "primary_artifacts": artifacts,
                "key_metrics": {
                    "absolute_flux_promoted": absolute_flux_promoted,
                    "uq_candidate_promotion_passed": False,
                    "dataset_sufficiency_promotion_passed": False,
                    "accepted_uq_candidates": [],
                },
            }
        ],
    }


def _candidate_uncertainty_sidecar() -> dict:
    return {
        "kind": "quasilinear_candidate_uncertainty_report",
        "claim_level": "candidate_model_development_not_runtime_option",
        "passed": False,
        "notes": (
            "Candidate retained only as a scoped rank-screening near miss, "
            "not a runtime/TOML absolute-flux predictor."
        ),
        "null_training_mean_baseline": {"mean_abs_relative_error": 0.79},
        "candidates": {
            "linear_weight": {
                "mean_abs_relative_error": 0.85,
                "promotion_eligible": True,
            },
            "linear_state_ridge": {
                "mean_abs_relative_error": 1.1,
                "promotion_eligible": False,
                "eligibility_failures": ["insufficient_train_to_parameter_ratio"],
            },
            "spectral_envelope_ridge": {
                "mean_abs_relative_error": 0.377,
                "promotion_eligible": True,
            },
        },
        "promotion_gate": {
            "passed": False,
            "accepted_candidates": [],
            "requires_beating_linear_weight_baseline": True,
            "requires_beating_training_mean_null": True,
            "transport_mean_relative_error_gate": 0.35,
        },
    }


def test_promoted_absolute_flux_requires_passed_holdout_gate_and_window_stats(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    payload = _calibration_report(
        claim_level="calibrated_absolute_flux",
        passed=True,
        holdout_error=0.7,
    )
    payload["points"][1]["observed_heat_flux_std"] = float("nan")
    payload["points"][1]["nonlinear_window_stats"] = _window_stats(passed=False)
    report.write_text(json.dumps(payload), encoding="utf-8")
    doc = tmp_path / "doc.rst"
    _write_doc(doc)

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is False
    failed = {
        gate["metric"]: gate["detail"]
        for gate in audit["gate_report"]["gates"]
        if not gate["passed"]
    }
    assert "train_holdout_point_metadata" in failed
    assert "promoted_holdout_gate" in failed
    assert "promoted_holdout_window_convergence" in failed


def test_promoted_absolute_flux_requires_converged_holdout_window_metadata(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    payload = _calibration_report(
        claim_level="calibrated_absolute_flux",
        passed=True,
        holdout_error=0.1,
    )
    del payload["points"][1]["nonlinear_window_stats"]
    report.write_text(json.dumps(payload), encoding="utf-8")
    doc = tmp_path / "doc.rst"
    _write_doc(doc)

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is False
    failed_metrics = {
        gate["metric"] for gate in audit["gate_report"]["gates"] if not gate["passed"]
    }
    assert "promoted_holdout_window_convergence" in failed_metrics


def test_unpromoted_report_with_finite_metadata_passes_synthetic_guardrail(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _calibration_report(
                claim_level="calibration_dataset",
                passed=False,
                holdout_error=4.0,
            )
        ),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(doc)

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is True
    assert audit["calibration_reports"][0]["n_train"] == 1
    assert audit["calibration_reports"][0]["n_holdout"] == 1


def test_docs_without_nonpromotion_marker_fail_scope_check(tmp_path: Path) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _calibration_report(
                claim_level="calibration_dataset",
                passed=False,
                holdout_error=4.0,
            )
        ),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(doc, "This section describes a calibrated absolute-flux predictor.\n")

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is False
    failed_metrics = {
        gate["metric"] for gate in audit["gate_report"]["gates"] if not gate["passed"]
    }
    assert f"doc_scope_marker:{doc}" in failed_metrics
    assert f"doc_no_absolute_flux_overclaim:{doc}" in failed_metrics


def test_wrapped_negative_absolute_flux_phrase_is_not_overclaim(tmp_path: Path) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _calibration_report(
                claim_level="calibration_dataset",
                passed=False,
                holdout_error=4.0,
            )
        ),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(
        doc,
        (
            "This result is scoped model-development evidence, not a\n"
            "runtime/TOML absolute-flux predictor. Absolute-flux prediction not "
            "promoted.\n"
        ),
    )

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is True
    doc_rows = {row["doc"]: row for row in audit["doc_checks"]}
    assert doc_rows[str(doc)]["overclaim_lines"] == []


def test_release_contract_ql_lane_requires_scoped_nonabsolute_candidate(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "release_contract.json"
    payload = _release_contract(
        claim_level="calibrated_absolute_flux",
        absolute_flux_promoted=True,
    )
    payload["release_lanes"][0]["key_metrics"]["uq_candidate_promotion_passed"] = True
    payload["release_lanes"][0]["key_metrics"]["accepted_uq_candidates"] = [
        "spectral_envelope_ridge"
    ]
    report.write_text(json.dumps(payload), encoding="utf-8")
    doc = tmp_path / "doc.rst"
    _write_doc(doc)

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is False
    failed_metrics = {
        gate["metric"] for gate in audit["gate_report"]["gates"] if not gate["passed"]
    }
    assert "release_contract_ql_not_absolute_flux" in failed_metrics
    assert "release_contract_ql_closed_scope_is_non_absolute" in failed_metrics
    assert "release_contract_ql_candidate_scope_not_runtime" in failed_metrics


def test_release_contract_ql_lane_requires_guardrail_artifact(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "release_contract.json"
    report.write_text(
        json.dumps(_release_contract(include_guardrail_artifact=False)),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(doc)

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is False
    failed_metrics = {
        gate["metric"] for gate in audit["gate_report"]["gates"] if not gate["passed"]
    }
    assert "release_contract_ql_guardrail_artifact_listed" in failed_metrics


def test_release_contract_ql_lane_passes_when_candidate_is_scoped(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "release_contract.json"
    report.write_text(json.dumps(_release_contract()), encoding="utf-8")
    doc = tmp_path / "doc.rst"
    _write_doc(doc)

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is True
    assert audit["summary"]["n_release_contracts"] == 1
    assert audit["release_contracts"][0]["ql_status"] == "closed"


def test_manuscript_figure_audit_requires_json_sidecar_and_index_entry(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _calibration_report(
                claim_level="calibration_dataset",
                passed=False,
                holdout_error=4.0,
            )
        ),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(doc)
    figure_base = tmp_path / "docs/_static/quasilinear_candidate_uncertainty"
    figure_base.parent.mkdir(parents=True)
    figure_base.with_suffix(".png").write_bytes(b"not a real png for metadata test")
    index = tmp_path / "manuscript_figures.rst"
    index.write_text(
        (
            f"current artifact base: ``{figure_base.with_suffix('.png')}`` "
            "with PDF companion only. "
            "No runtime/TOML absolute-flux predictor; absolute-flux runtime "
            "promotion remains blocked.\n"
        ),
        encoding="utf-8",
    )

    audit = mod.build_guardrail_audit(
        [str(report)],
        [doc],
        [figure_base],
        index,
    )

    assert audit["passed"] is False
    failed_metrics = {
        gate["metric"] for gate in audit["gate_report"]["gates"] if not gate["passed"]
    }
    assert f"ql_figure_json_sidecar_exists:{figure_base}" in failed_metrics
    assert f"ql_figure_index_mentions_json_sidecar:{figure_base}" in failed_metrics


def test_manuscript_figure_audit_requires_explicit_failed_baselines(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _calibration_report(
                claim_level="calibration_dataset",
                passed=False,
                holdout_error=4.0,
            )
        ),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(doc)
    figure_base = tmp_path / "docs/_static/quasilinear_candidate_uncertainty"
    figure_base.parent.mkdir(parents=True)
    figure_base.with_suffix(".png").write_bytes(b"not a real png for metadata test")
    sidecar = _candidate_uncertainty_sidecar()
    sidecar["promotion_gate"]["accepted_candidates"] = [
        "spectral_envelope_ridge",
        "linear_weight",
    ]
    figure_base.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    index = tmp_path / "manuscript_figures.rst"
    index.write_text(
        (
            f"current artifact base: ``{figure_base.with_suffix('.png')}`` "
            f"with JSON companion ``{figure_base.with_suffix('.json')}``. "
            "No runtime/TOML absolute-flux predictor; absolute-flux runtime "
            "promotion remains blocked.\n"
        ),
        encoding="utf-8",
    )

    audit = mod.build_guardrail_audit(
        [str(report)],
        [doc],
        [figure_base],
        index,
    )

    assert audit["passed"] is False
    failed_metrics = {
        gate["metric"] for gate in audit["gate_report"]["gates"] if not gate["passed"]
    }
    assert f"ql_figure_failed_baselines_explicit:{figure_base}" in failed_metrics


def test_manuscript_figure_audit_accepts_scoped_candidate_sidecar(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _calibration_report(
                claim_level="calibration_dataset",
                passed=False,
                holdout_error=4.0,
            )
        ),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(doc)
    figure_base = tmp_path / "docs/_static/quasilinear_candidate_uncertainty"
    figure_base.parent.mkdir(parents=True)
    figure_base.with_suffix(".png").write_bytes(b"not a real png for metadata test")
    figure_base.with_suffix(".json").write_text(
        json.dumps(_candidate_uncertainty_sidecar()),
        encoding="utf-8",
    )
    index = tmp_path / "manuscript_figures.rst"
    index.write_text(
        (
            f"current artifact base: ``{figure_base.with_suffix('.png')}`` "
            f"with JSON companion ``{figure_base.with_suffix('.json')}``. "
            "No runtime/TOML absolute-flux predictor; absolute-flux runtime "
            "promotion remains blocked.\n"
        ),
        encoding="utf-8",
    )

    audit = mod.build_guardrail_audit(
        [str(report)],
        [doc],
        [figure_base],
        index,
    )

    assert audit["passed"] is True
    assert audit["summary"]["n_manuscript_figure_checks"] == 1
    assert audit["manuscript_figure_provenance"][0]["claim_scoped"] is True
    assert audit["manuscript_figure_provenance"][0]["failed_baselines_explicit"] is True


def test_dataset_sufficiency_failure_can_be_downstream_skill_not_data_volume(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    doc = tmp_path / "doc.rst"
    _write_doc(doc)
    figure_base = tmp_path / "docs/_static/quasilinear_dataset_sufficiency"
    figure_base.parent.mkdir(parents=True)
    figure_base.with_suffix(".png").write_bytes(b"not a real png for metadata test")
    figure_base.with_suffix(".json").write_text(
        json.dumps(
            {
                "kind": "quasilinear_dataset_sufficiency",
                "claim_level": "scoped_low_parameter_candidate_promotion_not_runtime_option",
                "notes": "Dataset guard, not a runtime/TOML absolute-flux predictor.",
                "candidate_requirements": [
                    {
                        "candidate": "linear_state_ridge",
                        "data_volume_passed": True,
                    }
                ],
                "downstream_gates": {
                    "saturation_rule_sweep": {
                        "passed": False,
                        "accepted": [],
                    }
                },
                "promotion_gate": {
                    "passed": False,
                    "blockers": ["downstream_candidate_skill_gates_not_passed"],
                    "requires_downstream_candidate_skill_gates": True,
                },
                "input_validation": {
                    "passed": True,
                    "cases": [{"case": "holdout", "required": True, "passed": True}],
                },
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "manuscript_figures.rst"
    index.write_text(
        (
            f"current artifact base: ``{figure_base.with_suffix('.png')}`` "
            f"with JSON companion ``{figure_base.with_suffix('.json')}``. "
            "No runtime/TOML absolute-flux predictor; absolute-flux runtime "
            "promotion remains blocked.\n"
        ),
        encoding="utf-8",
    )

    audit = mod.build_guardrail_audit(
        [str(figure_base.with_suffix(".json"))],
        [doc],
        [figure_base],
        index,
    )

    failed_metrics = {
        gate["metric"] for gate in audit["gate_report"]["gates"] if not gate["passed"]
    }
    assert f"ql_figure_failed_baselines_explicit:{figure_base}" not in failed_metrics


def test_tracked_quasilinear_promotion_guardrails_pass() -> None:
    mod = _load_tool_module()

    audit = mod.build_guardrail_audit(
        list(mod.DEFAULT_REPORT_PATTERNS),
        [str(path) for path in mod.DEFAULT_DOCS],
    )

    assert audit["passed"] is True
    assert audit["summary"]["n_calibration_reports"] == 4
    assert audit["summary"]["n_input_validation_reports"] >= 4
    assert audit["summary"]["n_promotion_gate_reports"] >= 4
    assert audit["summary"]["n_release_contracts"] == 1
    assert audit["summary"]["n_doc_checks"] == 4
    assert audit["summary"]["n_manuscript_figure_checks"] == len(
        mod.DEFAULT_MANUSCRIPT_FIGURE_BASES
    )


def test_guardrail_script_runs_before_editable_install(tmp_path: Path) -> None:
    root = REPO_ROOT
    out = tmp_path / "guardrails.json"
    env = {**os.environ, "PYTHONPATH": ""}

    completed = subprocess.run(
        [
            sys.executable,
            "tools/release/check_quasilinear_promotion_guardrails.py",
            "--out-json",
            str(out),
        ],
        cwd=root,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "quasilinear_promotion_guardrails_passed=True" in completed.stdout
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is True
