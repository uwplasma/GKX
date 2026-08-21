"""Release, manifest, coverage, and artifact hygiene gates."""

from __future__ import annotations


# ---- quasilinear calibration-input provenance gates ----

"""Tests for quasilinear calibration input validation gates."""


from support.paths import load_release_tool
import json
from pathlib import Path


def _load_quasilinear_tool_module():
    return load_release_tool("check_quasilinear_promotion_guardrails")


def _write_report(path: Path, artifact: str, *, split: str = "holdout") -> None:
    payload = {
        "kind": "quasilinear_calibration_report",
        "points": [
            {
                "case": "synthetic",
                "split": split,
                "predicted_heat_flux": 1.0,
                "observed_heat_flux": 1.1,
                "saturation_rule": "linear_weight",
                "nonlinear_artifact": artifact,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_audit_passes_when_required_point_matches_passed_gate(tmp_path: Path) -> None:
    mod = _load_quasilinear_tool_module()
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "case": "synthetic_nonlinear_window",
                "gkx": "tools_out/synthetic.csv",
                "gate_report": {
                    "case": "synthetic_nonlinear_window",
                    "passed": True,
                    "gates": [],
                },
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, "tools_out/synthetic.csv")

    paths = mod.write_audit(
        [report],
        gate_patterns=[str(gate)],
        out_json=tmp_path / "audit.json",
        no_plot=True,
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert (
        payload["reports"][0]["points"][0]["reason"] == "matched passed nonlinear gate"
    )


def test_audit_passes_when_required_point_cites_passed_gate_sidecar(
    tmp_path: Path,
) -> None:
    mod = _load_quasilinear_tool_module()
    gate = tmp_path / "ensemble_gate.json"
    gate.write_text(
        json.dumps(
            {
                "case": "replicated_nonlinear_window",
                "kind": "nonlinear_window_ensemble_report",
                "promotion_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, gate.as_posix())

    paths = mod.write_audit(
        [report],
        gate_patterns=[str(gate)],
        out_json=tmp_path / "audit.json",
        no_plot=True,
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    point = payload["reports"][0]["points"][0]
    assert payload["passed"] is True
    assert point["reason"] == "matched passed nonlinear gate"
    assert point["matched_gate"]["artifact"] == gate.as_posix()


def test_default_gate_glob_recurses_into_nested_holdout_artifacts(
    tmp_path: Path,
) -> None:
    mod = _load_quasilinear_tool_module()
    gate = tmp_path / "docs/_static/nested_holdouts/case/ensemble_gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_text(
        json.dumps(
            {
                "case": "nested_replicated_ensemble",
                "kind": "nonlinear_window_ensemble_report",
                "claim_level": "replicated_nonlinear_window_uncertainty_gate_not_simulation_claim",
                "passed": True,
                "promotion_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, gate.as_posix())
    old_default = mod.DEFAULT_GATE_GLOB
    mod.DEFAULT_GATE_GLOB = str(tmp_path / "docs/_static/**/*.json")
    try:
        paths = mod.write_audit(
            [report], out_json=tmp_path / "audit.json", no_plot=True
        )
    finally:
        mod.DEFAULT_GATE_GLOB = old_default

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    point = payload["reports"][0]["points"][0]
    assert payload["passed"] is True
    assert point["matched_gate"]["artifact"].endswith(
        "nested_holdouts/case/ensemble_gate.json"
    )


def test_audit_normalizes_absolute_artifact_paths_from_other_checkouts(
    tmp_path: Path,
) -> None:
    mod = _load_quasilinear_tool_module()
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "case": "synthetic_nonlinear_window",
                "gkx": "tools_out/synthetic.csv",
                "gate_report": {
                    "case": "synthetic_nonlinear_window",
                    "passed": True,
                    "gates": [],
                },
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, "/Users/example/local/GKX/tools_out/synthetic.csv")

    paths = mod.write_audit(
        [report],
        gate_patterns=[str(gate)],
        out_json=tmp_path / "audit.json",
        no_plot=True,
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    point = payload["reports"][0]["points"][0]
    assert payload["passed"] is True
    assert point["nonlinear_artifact"] == "tools_out/synthetic.csv"
    assert point["reason"] == "matched passed nonlinear gate"


def test_audit_fails_when_required_point_uses_failed_gate(tmp_path: Path) -> None:
    mod = _load_quasilinear_tool_module()
    gate = tmp_path / "external_gate.json"
    gate.write_text(
        json.dumps(
            {
                "case": "external_cth_like",
                "promotion_gate": {"passed": False},
                "runs": [
                    {
                        "csv": "docs/_static/external_vmec_cth_like_nonlinear_t150_pilot.traces.csv"
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(
        report, "docs/_static/external_vmec_cth_like_nonlinear_t150_pilot.traces.csv"
    )

    paths = mod.write_audit(
        [report],
        gate_patterns=[str(gate)],
        out_json=tmp_path / "audit.json",
        no_plot=True,
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert (
        payload["reports"][0]["points"][0]["reason"]
        == "matching nonlinear gate is negative evidence for calibration admission"
    )
    assert payload["n_negative_evidence"] == 1


def test_audit_records_qh_gate_with_unacceptable_claim_as_negative_evidence(
    tmp_path: Path,
) -> None:
    mod = _load_quasilinear_tool_module()
    gate = tmp_path / "external_qh_gate.json"
    gate.write_text(
        json.dumps(
            {
                "case": "nfp4 QH external VMEC nonlinear high-grid convergence",
                "claim_level": "finite_high_grid_long_nonlinear_feasibility_not_yet_transport_validation",
                "gate_report": {
                    "case": "nfp4 QH external VMEC nonlinear high-grid convergence",
                    "passed": True,
                    "gates": [],
                },
                "kind": "external_vmec_nonlinear_grid_convergence_gate",
                "promotion_gate": {"passed": True},
                "runs": [
                    {
                        "csv": "docs/_static/external_vmec_qh_nonlinear_t150_n64_pilot.traces.csv"
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(
        report,
        "docs/_static/external_vmec_qh_nonlinear_t150_n64_pilot.traces.csv",
    )

    paths = mod.write_audit(
        [report],
        gate_patterns=[str(gate)],
        out_json=tmp_path / "audit.json",
        no_plot=True,
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    point = payload["reports"][0]["points"][0]
    assert payload["passed"] is False
    assert point["passed"] is False
    assert (
        point["reason"]
        == "matching nonlinear gate is negative evidence for calibration admission"
    )
    assert point["matched_gate"]["raw_gate_passed"] is True
    assert point["matched_gate"]["promotion_gate_passed"] is True
    assert point["matched_gate"]["claim_level_acceptable"] is False
    assert point["matched_gate"]["admission_blockers"] == ["claim_level_not_acceptable"]
    assert (
        payload["negative_evidence"][0]["case"]
        == "nfp4 QH external VMEC nonlinear high-grid convergence"
    )


def test_audit_fails_when_required_point_has_no_gate(tmp_path: Path) -> None:
    mod = _load_quasilinear_tool_module()
    report = tmp_path / "report.json"
    _write_report(report, "tools_out/missing.csv")

    paths = mod.write_audit(
        [report], gate_patterns=[], out_json=tmp_path / "audit.json", no_plot=True
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert (
        payload["reports"][0]["points"][0]["reason"]
        == "no matching nonlinear validation/convergence gate"
    )


def test_audit_accepts_nested_high_grid_admission_input_artifact(
    tmp_path: Path,
) -> None:
    mod = _load_quasilinear_tool_module()
    gate = tmp_path / "high_grid_admission.json"
    gate.write_text(
        json.dumps(
            {
                "kind": "external_vmec_high_grid_admission_gate",
                "case": "synthetic high-grid admission",
                "claim_level": "passed_high_grid_transport_holdout_admission_under_coarse_grid_exclusion",
                "inputs": {
                    "replicate_ensemble_gate": "docs/_static/replicate/ensemble_gate.json",
                },
                "promotion_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, "docs/_static/replicate/ensemble_gate.json")

    paths = mod.write_audit(
        [report],
        gate_patterns=[str(gate)],
        out_json=tmp_path / "audit.json",
        no_plot=True,
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    point = payload["reports"][0]["points"][0]
    assert payload["passed"] is True
    assert point["matched_gate"]["case"] == "synthetic high-grid admission"


def test_audit_prefers_external_admission_gate_over_raw_nested_ensemble(
    tmp_path: Path,
) -> None:
    mod = _load_quasilinear_tool_module()
    raw = tmp_path / "docs/_static/external_vmec_holdouts/case/ensemble_gate.json"
    admission = tmp_path / "aa_admission.json"
    artifact = raw.as_posix()
    raw.parent.mkdir(parents=True)
    raw.write_text(
        json.dumps(
            {
                "case": "synthetic_external_vmec_ensemble",
                "kind": "nonlinear_window_ensemble_report",
                "claim_level": "replicated_nonlinear_window_uncertainty_gate_not_simulation_claim",
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    admission.write_text(
        json.dumps(
            {
                "case": "synthetic external admission",
                "kind": "external_vmec_replicate_admission_gate",
                "claim_level": "passed_replicated_external_vmec_transport_holdout_under_explicit_spread_gate",
                "inputs": {"replicate_ensemble_gate": artifact},
                "promotion_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, artifact)

    paths = mod.write_audit(
        [report],
        gate_patterns=[str(raw), str(admission)],
        out_json=tmp_path / "audit.json",
        no_plot=True,
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    point = payload["reports"][0]["points"][0]
    assert payload["passed"] is True
    assert point["matched_gate"]["kind"] == "external_vmec_replicate_admission_gate"
    assert point["matched_gate"]["claim_level_acceptable"] is True


def test_audit_ignores_non_required_audit_split_without_gate(tmp_path: Path) -> None:
    mod = _load_quasilinear_tool_module()
    report = tmp_path / "report.json"
    _write_report(report, "tools_out/missing.csv", split="audit")

    paths = mod.write_audit(
        [report], gate_patterns=[], out_json=tmp_path / "audit.json", no_plot=True
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["reports"][0]["points"][0]["reason"] == "not required split"


def test_tracked_quasilinear_train_holdout_reports_use_passed_nonlinear_gates() -> None:
    mod = _load_quasilinear_tool_module()
    root = Path(__file__).resolve().parents[2]
    reports = [
        root / "docs/_static/quasilinear_cyclone_miller_train_holdout_report.json",
        root / "docs/_static/quasilinear_hsx_train_holdout_report.json",
        root / "docs/_static/quasilinear_w7x_train_holdout_report.json",
        root / "docs/_static/quasilinear_stellarator_train_holdout_report.json",
    ]

    payload = mod.audit_calibration_inputs(reports)

    assert payload["passed"] is True
    required_rows = [
        point
        for report in payload["reports"]
        for point in report["points"]
        if point["required"]
    ]
    assert len(required_rows) == 20
    assert all(point["matched_gate"] is not None for point in required_rows)
    matched_cases = {point["matched_gate"]["case"] for point in required_rows}
    assert matched_cases == {
        "cyclone_nonlinear_long_window",
        "cyclone_miller_nonlinear_window",
        "hsx_nonlinear_window",
        "w7x_nonlinear_window",
        "D-shaped tokamak external VMEC nonlinear t250 high-grid convergence",
        "ITERModel external VMEC nonlinear t350 high-grid convergence",
        "updown_asym_external_vmec_t450",
        "circular_external_vmec_t450",
        "CTH-like external VMEC modified-protocol high-grid admission",
        "Shaped tokamak pressure external VMEC dt=0.04 high-grid transport holdout admission",
        "qp_diag_nfp2_m4_final_t250_n64_seed_timestep_ensemble_gate",
        "solovev_reference_repair_dt002_amp1em5_n48_t250",
    }
    external_rows = [
        point
        for point in required_rows
        if "external_vmec" in str(point["matched_gate"]["artifact"])
    ]
    assert [point["case"] for point in external_rows] == [
        "dshape_external_vmec_t250_window",
        "itermodel_external_vmec_t350_window",
        "updown_asym_external_vmec_t450_window",
        "circular_external_vmec_t450_window",
        "cth_like_external_vmec_t700_high_grid_window",
        "shaped_tokamak_pressure_external_vmec_t650_high_grid_window",
        "solovev_reference_repair_dt002_amp1em5_n48_t250",
    ]


# ---- test_check_release_readiness.py ----


import pytest

from tools.release.check_release_readiness import TECHNICAL_COMPLETION_TARGET
from tools.release.check_release_readiness import (
    ReleaseReadinessError,
    build_frozen_output_fingerprint,
    check_release_readiness,
)


def _write_release_ready_tree(root: Path) -> None:
    (root / "src" / "gkx").mkdir(parents=True)
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "docs" / "_static").mkdir(parents=True)
    (root / "benchmarks" / "references").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        """
[project]
name = "gkx"
version = "1.2.3"

[project.scripts]
gkx = "gkx.cli:main"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "src" / "gkx" / "_version.py").write_text(
        '__version__ = "1.2.3"\n',
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "ci.yml").write_text(
        "\n".join(
            [
                "wide-coverage-shards",
                "coverage-wide-shard-manifest.json",
                "--require-shard-data",
                "--coverage-xml coverage-wide.xml",
                "--enforce-package-coverage",
                "codecov/codecov-action",
                "tools/release/check_parallel_scaling_artifacts.py",
                "tools/release/check_package_architecture_manifest.py",
                "tools/release/check_parallel_scaling_artifacts.py --performance-manifest-only",
                "tools/release/check_quasilinear_promotion_guardrails.py",
                "tools/release/check_vmec_boozer_gates.py differentiability-claim",
                "tools/artifacts/build_parallelization_completion_status.py",
                "tools/release/check_release_readiness.py technical-status",
                "tools/release/check_release_readiness.py",
                "rm -rf build dist",
            ]
        ),
        encoding="utf-8",
    )
    (root / "codecov.yml").write_text(
        """
codecov:
  notify:
    after_n_builds: 2
    wait_for_ci: true

coverage:
  status:
    project:
      default:
        target: 95%
        threshold: 0.5%
        flags:
          - wide-package
    patch:
      default:
        informational: true
""".lstrip(),
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "release.yml").write_text(
        "name: Release\n"
        "tools/release/check_release_readiness.py version\n"
        "tools/release/check_repository_size_manifest.py\n"
        "tools/release/check_repository_size_manifest.py release-artifacts\n"
        "tools/release/check_package_architecture_manifest.py\n"
        "tools/release/check_parallel_scaling_artifacts.py --performance-manifest-only\n"
        "tools/release/check_parallel_scaling_artifacts.py\n"
        "tools/release/check_quasilinear_promotion_guardrails.py\n"
        "tools/release/check_vmec_boozer_gates.py differentiability-claim\n"
        "tools/artifacts/build_parallelization_completion_status.py\n"
        "tools/release/check_release_readiness.py technical-status\n"
        "tools/release/check_release_readiness.py\n"
        "rm -rf build dist\n"
        "gh-action-pypi-publish\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "Install with pip install gkx, run gkx. License: MIT.\n",
        encoding="utf-8",
    )
    for artifact in (
        "runtime_memory_benchmark.png",
        "runtime_memory_summary_ship_refresh.json",
        "runtime_memory_results_ship_refresh.csv",
        "validation_gate_index.json",
        "validation_coverage_manifest_summary.json",
        "quasilinear_promotion_guardrails.json",
        "vmec_boozer_differentiability_claim_guard.json",
        "vmec_boozer_shaped_pressure_solver_frequency_gradient_gate.json",
        "vmec_boozer_shaped_pressure_quasilinear_gradient_gate.json",
        "vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json",
        "technical_release_status.json",
        "independent_ky_scan_scaling_large.json",
        "quasilinear_uq_ensemble_scaling_large.json",
        "parallelization_completion_status.json",
        "nonlinear_sharding_strong_scaling_large.json",
        "nonlinear_domain_parallel_identity_gate.json",
        "nonlinear_spectral_communication_identity_gate.json",
        "vmec_boundary_transport_landscape_admission.json",
        "vmec_boundary_transport_prelaunch_gate.json",
        "nonlinear_campaign_admission_report.json",
        "strict_qa_top12_edge_prelaunch_gate.json",
    ):
        (root / "docs" / "_static" / artifact).write_text("{}", encoding="utf-8")
    (root / "docs" / "_static" / "technical_release_status.json").write_text(
        """
{
  "failed_required": [],
  "kind": "gkx_technical_release_status",
  "lanes": {},
  "passed": true,
  "target_percent": 98.0,
  "technical_release_completion_percent": 100.0
}
""".lstrip(),
        encoding="utf-8",
    )
    (root / "benchmarks" / "references" / "gkx_1_7_release_contract.json").write_text(
        """
{
  "kind": "gkx_1_7_frozen_release_contract",
  "optimization_policy": {
    "prelaunch_gates": [
      {
        "label": "replicated landscape admission",
        "path": "docs/_static/vmec_boundary_transport_landscape_admission.json",
        "passed": true,
        "expected_raw_passed": true,
        "raw_passed": true,
        "sample_count": 12.0,
        "blockers": []
      },
      {
        "label": "selected reduced prelaunch",
        "path": "docs/_static/vmec_boundary_transport_prelaunch_gate.json",
        "passed": true,
        "expected_raw_passed": true,
        "raw_passed": true,
        "sample_count": 18.0,
        "blockers": []
      },
      {
        "label": "weak reduced-margin reference",
        "path": "docs/_static/strict_qa_top12_edge_prelaunch_gate.json",
        "passed": true,
        "expected_raw_passed": false,
        "raw_passed": false,
        "sample_count": 18.0,
        "blockers": ["insufficient_reduced_margin_for_nonlinear_audit"]
      },
      {
        "label": "next nonlinear campaign admission",
        "path": "docs/_static/nonlinear_campaign_admission_report.json",
        "passed": true,
        "expected_raw_passed": true,
        "raw_passed": true,
        "sample_count": 18.0,
        "blockers": []
      }
    ],
    "summary": {
      "qa_baseline_gate_passed": true,
      "quasilinear_model_selection_passed": false,
      "simple_quasilinear_absolute_flux_promoted": false,
      "long_window_nonlinear_audit_passed": true,
      "nonlinear_prelaunch_policy_ready": true,
      "nonlinear_campaign_admission_ready": true,
      "negative_reference_blocks_weak_margin": true,
      "claim_evidence_level": "scoped_matched_replicated_nonlinear_audit",
      "claim_promotion_blockers": [
        "quasilinear_model_selection_not_promoted",
        "simple_quasilinear_absolute_flux_not_promoted"
      ]
    }
  },
  "performance": {
    "representative_refresh": {
      "path": "benchmarks/references/gkx_2_representative_performance_refresh.json",
      "correctness_passed": true,
      "cpu_rows_admitted": 2,
      "gpu_rows_admitted": 0,
      "gpu_rows_blocked": 2,
      "performance_claim_updated": false
    },
    "row_count": 1,
    "rows": [{"case": "test", "backend": "cpu", "status": "success"}]
  },
  "public_api": {
    "count": 1,
    "exports": [{"name": "solve", "module": "gkx", "symbol": "solve"}]
  },
  "release_lanes": [
    {
      "claim_level": "release_claim",
      "lane": "CI/release hygiene and status automation",
      "status": "closed"
    },
    {
      "claim_level": "deferred_out_of_release_scope",
      "lane": "Future physics extension",
      "status": "deferred"
    }
  ]
}
""".lstrip(),
        encoding="utf-8",
    )
    contract_path = root / "benchmarks" / "references" / "gkx_1_7_release_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    frozen_path = "docs/_static/validation_gate_index.json"
    contract["baseline"] = {"git_tag": "v1.7.0"}
    contract["frozen_output_fingerprints"] = {
        "algorithm": "test contract",
        "entries": [
            {
                "label": "validation gate index",
                **build_frozen_output_fingerprint(root, frozen_path),
            }
        ],
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    (root / "benchmarks" / "references" / "gkx_2_representative_performance_refresh.json").write_text(
        json.dumps(
            {
                "kind": "gkx_representative_performance_refresh",
                "summary": {
                    "correctness_passed": True,
                    "cpu_rows_admitted": 2,
                    "gpu_rows_admitted": 0,
                    "gpu_rows_blocked": 2,
                    "performance_claim_updated": False,
                },
                "workloads": [
                    {
                        "case": case,
                        "correctness": {"cpu_finite": True, "gpu_finite": True},
                        "cpu": {"admitted": True},
                        "gpu": {
                            "admitted": False,
                            "blocker": "contended test device",
                        },
                    }
                    for case in ("linear", "nonlinear")
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "docs" / "_static" / "w7x_tem_extension_status.json").write_text(
        """
{
  "claim_scope": "extension_tracking",
  "kind": "w7x_tem_extension_status",
  "lanes": [
    {
      "claim_level": "partial_extension_not_release_claim",
      "lane": "W7-X TEM extension",
      "status": "partial"
    }
  ],
  "summary": {"n_partial": 1}
}
""".lstrip(),
        encoding="utf-8",
    )


def _replace_release_optimization_policy(root: Path, policy: dict[str, object]) -> None:
    contract = root / "benchmarks" / "references" / "gkx_1_7_release_contract.json"
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["optimization_policy"] = policy
    contract.write_text(json.dumps(payload), encoding="utf-8")


def test_release_readiness_accepts_ci_release_docs_and_artifact_contracts(
    tmp_path: Path,
) -> None:
    _write_release_ready_tree(tmp_path)

    report = check_release_readiness(tmp_path)

    assert report["passed"] is True
    assert report["project"]["name"] == "gkx"
    assert report["project"]["scripts"] == ["gkx"]
    assert report["version"]["project_version"] == "1.2.3"
    assert (
        report["release_target"]["technical_completion_fraction"]
        == TECHNICAL_COMPLETION_TARGET
    )
    assert report["lane_status"]["passed"] is True
    assert report["lane_status"]["frozen_outputs"]["count"] == 1
    assert report["lane_status"]["active_fraction_closed"] == 1.0
    assert report["lane_status"]["release_scoped_open_or_blocked"] == 0
    assert report["technical_status"]["passed"] is True
    assert report["technical_status"]["completion_percent"] >= 98.0
    assert report["optimization_status"]["passed"] is True
    assert (
        report["optimization_status"]["summary"]["nonlinear_prelaunch_policy_ready"]
        is True
    )
    assert report["lane_status"]["status_artifacts"][
        "benchmarks/references/gkx_1_7_release_contract.json"
    ]["status_counts"] == {"closed": 1, "deferred": 1}


def test_release_readiness_rejects_changed_frozen_output(tmp_path: Path) -> None:
    _write_release_ready_tree(tmp_path)
    (tmp_path / "docs" / "_static" / "validation_gate_index.json").write_text(
        '{"changed": 1}', encoding="utf-8"
    )

    with pytest.raises(
        ReleaseReadinessError, match="frozen numerical output fingerprints changed"
    ):
        check_release_readiness(tmp_path)


def test_release_readiness_rejects_false_performance_promotion(tmp_path: Path) -> None:
    _write_release_ready_tree(tmp_path)
    refresh = (
        tmp_path
        / "benchmarks"
        / "references"
        / "gkx_2_representative_performance_refresh.json"
    )
    payload = json.loads(refresh.read_text(encoding="utf-8"))
    payload["summary"]["performance_claim_updated"] = True
    refresh.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReleaseReadinessError, match="performance_claim_updated"):
        check_release_readiness(tmp_path)


def test_release_readiness_rejects_missing_ci_guardrails(tmp_path: Path) -> None:
    _write_release_ready_tree(tmp_path)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text(
        "wide-coverage-shards\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseReadinessError, match="ci.yml missing release checks"):
        check_release_readiness(tmp_path)


def test_release_readiness_rejects_missing_codecov_status_policy(
    tmp_path: Path,
) -> None:
    _write_release_ready_tree(tmp_path)
    (tmp_path / "codecov.yml").write_text(
        """
coverage:
  status:
    project:
      default:
        target: 95%
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ReleaseReadinessError,
        match="codecov.yml missing wide-coverage status policy",
    ):
        check_release_readiness(tmp_path)


def test_release_readiness_rejects_missing_release_guardrails(tmp_path: Path) -> None:
    _write_release_ready_tree(tmp_path)
    (tmp_path / ".github" / "workflows" / "release.yml").write_text(
        "name: Release\n"
        "tools/release/check_release_readiness.py version\n"
        "gh-action-pypi-publish\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ReleaseReadinessError, match="release.yml missing publish/version checks"
    ):
        check_release_readiness(tmp_path)


def test_release_readiness_rejects_unclean_distribution_build(tmp_path: Path) -> None:
    _write_release_ready_tree(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "release.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace("rm -rf build dist\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(
        ReleaseReadinessError, match="release.yml missing publish/version checks"
    ):
        check_release_readiness(tmp_path)


def test_release_readiness_rejects_below_target_release_completion(
    tmp_path: Path,
) -> None:
    _write_release_ready_tree(tmp_path)
    contract = tmp_path / "benchmarks" / "references" / "gkx_1_7_release_contract.json"
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["release_lanes"] = [
        {
            "claim_level": "release_claim",
            "lane": "CI/release hygiene and status automation",
            "status": "partial",
        }
    ]
    contract.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ReleaseReadinessError,
        match="release-scoped technical completion below target",
    ):
        check_release_readiness(tmp_path)


def test_release_readiness_rejects_failed_technical_status(tmp_path: Path) -> None:
    _write_release_ready_tree(tmp_path)
    (tmp_path / "docs" / "_static" / "technical_release_status.json").write_text(
        """
{
  "failed_required": ["docs_release_hygiene: release scope"],
  "kind": "gkx_technical_release_status",
  "lanes": {},
  "passed": false,
  "target_percent": 98.0,
  "technical_release_completion_percent": 92.0
}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ReleaseReadinessError,
        match="technical release status below target",
    ):
        check_release_readiness(tmp_path)


def test_release_readiness_rejects_missing_optimization_prelaunch_policy(
    tmp_path: Path,
) -> None:
    _write_release_ready_tree(tmp_path)
    _replace_release_optimization_policy(
        tmp_path,
        {
            "prelaunch_gates": [],
            "summary": {
                "qa_baseline_gate_passed": True,
                "quasilinear_model_selection_passed": False,
                "simple_quasilinear_absolute_flux_promoted": False,
                "long_window_nonlinear_audit_passed": True,
                "nonlinear_prelaunch_policy_ready": False,
                "nonlinear_campaign_admission_ready": True,
                "negative_reference_blocks_weak_margin": False,
            },
        },
    )

    with pytest.raises(
        ReleaseReadinessError,
        match="optimization status prelaunch/claim-boundary flags failed",
    ):
        check_release_readiness(tmp_path)


def test_release_readiness_requires_explicit_optimization_status_booleans(
    tmp_path: Path,
) -> None:
    _write_release_ready_tree(tmp_path)
    _replace_release_optimization_policy(
        tmp_path,
        {
            "prelaunch_gates": [
                {"label": "landscape", "passed": True},
                {"label": "positive", "passed": True},
                {"label": "negative", "passed": True},
            ],
            "summary": {
                "qa_baseline_gate_passed": True,
                "quasilinear_model_selection_passed": False,
                "long_window_nonlinear_audit_passed": True,
                "nonlinear_prelaunch_policy_ready": True,
                "negative_reference_blocks_weak_margin": True,
            },
        },
    )

    with pytest.raises(
        ReleaseReadinessError,
        match="optimization status prelaunch/claim-boundary flags failed",
    ):
        check_release_readiness(tmp_path)


def test_release_readiness_rejects_stale_prelaunch_gate_rows(
    tmp_path: Path,
) -> None:
    _write_release_ready_tree(tmp_path)
    _replace_release_optimization_policy(
        tmp_path,
        {
            "prelaunch_gates": [
                {
                    "label": "replicated landscape admission",
                    "path": "docs/_static/vmec_boundary_transport_landscape_admission.json",
                    "passed": True,
                    "expected_raw_passed": True,
                    "raw_passed": True,
                    "sample_count": 12.0,
                    "blockers": [],
                },
                {
                    "label": "selected reduced prelaunch",
                    "path": "docs/_static/vmec_boundary_transport_prelaunch_gate.json",
                    "passed": True,
                    "expected_raw_passed": True,
                    "raw_passed": True,
                    "sample_count": 1.0,
                    "blockers": [],
                },
                {
                    "label": "weak reduced-margin reference",
                    "path": "docs/_static/strict_qa_top12_edge_prelaunch_gate.json",
                    "passed": True,
                    "expected_raw_passed": True,
                    "raw_passed": True,
                    "sample_count": 18.0,
                    "blockers": [],
                },
                {
                    "label": "next nonlinear campaign admission",
                    "path": "docs/_static/nonlinear_campaign_admission_report.json",
                    "passed": True,
                    "expected_raw_passed": True,
                    "raw_passed": True,
                    "sample_count": 18.0,
                    "blockers": [],
                },
            ],
            "summary": {
                "qa_baseline_gate_passed": True,
                "quasilinear_model_selection_passed": False,
                "simple_quasilinear_absolute_flux_promoted": False,
                "long_window_nonlinear_audit_passed": True,
                "nonlinear_prelaunch_policy_ready": True,
                "nonlinear_campaign_admission_ready": True,
                "negative_reference_blocks_weak_margin": True,
                "claim_evidence_level": "scoped_matched_replicated_nonlinear_audit",
                "claim_promotion_blockers": [
                    "quasilinear_model_selection_not_promoted",
                    "simple_quasilinear_absolute_flux_not_promoted",
                ],
            },
        },
    )

    with pytest.raises(
        ReleaseReadinessError,
        match="optimization status prelaunch/claim-boundary flags failed",
    ):
        check_release_readiness(tmp_path)


# ---- test_release_hygiene_gates.py ----

import hashlib
import subprocess
import sys
import textwrap

import yaml

from tools.release.check_release_readiness import (
    LANES,
    ReleaseVersionError,
    build_technical_release_status,
    default_tag_from_github_env,
    fetch_pypi_versions,
    normalize_tag,
    read_project_version,
    read_source_version,
    validate_release_version,
)


def _write_version_files(
    root: Path, *, project: str = "1.2.3", source: str = "1.2.3"
) -> None:
    (root / "src" / "gkx").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "gkx"
            version = "{project}"
            """
        ).strip(),
        encoding="utf-8",
    )
    (root / "src" / "gkx" / "_version.py").write_text(
        f'__version__ = "{source}"\n',
        encoding="utf-8",
    )


def test_release_version_accepts_matching_project_source_and_tag(
    tmp_path: Path,
) -> None:
    _write_version_files(tmp_path, project="2.0.1", source="2.0.1")

    report = validate_release_version(
        root=tmp_path,
        tag="refs/tags/v2.0.1",
        require_tag=True,
        pypi_versions={"1.5.0", "2.0.0"},
    )

    assert report["project_version"] == "2.0.1"
    assert report["source_version"] == "2.0.1"
    assert report["tag"] == "v2.0.1"
    assert report["checked_pypi"] is True


def test_release_version_rejects_source_pyproject_mismatch(tmp_path: Path) -> None:
    _write_version_files(tmp_path, project="2.0.1", source="2.0.0")

    with pytest.raises(ReleaseVersionError, match="_version.py"):
        validate_release_version(root=tmp_path)


def test_release_version_rejects_wrong_or_missing_tag(tmp_path: Path) -> None:
    _write_version_files(tmp_path, project="2.0.1", source="2.0.1")

    with pytest.raises(ReleaseVersionError, match="expected 'v2.0.1'"):
        validate_release_version(root=tmp_path, tag="v2.0.0", require_tag=True)
    with pytest.raises(ReleaseVersionError, match="requires a tag"):
        validate_release_version(root=tmp_path, tag=None, require_tag=True)


def test_release_version_rejects_duplicate_pypi_version(tmp_path: Path) -> None:
    _write_version_files(tmp_path, project="2.0.1", source="2.0.1")

    with pytest.raises(ReleaseVersionError, match="already exists on PyPI"):
        validate_release_version(
            root=tmp_path, tag="v2.0.1", require_tag=True, pypi_versions={"2.0.1"}
        )


def test_fetch_pypi_versions_treats_unpublished_project_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A never-published project answers 404 and must not fail the first release."""

    import urllib.error
    import urllib.request

    def raise_not_found(url: str, timeout: float = 0.0) -> None:
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", raise_not_found)

    assert fetch_pypi_versions("gkx") == set()


def test_fetch_pypi_versions_propagates_non_404_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real PyPI outages must not be mistaken for an unpublished project."""

    import urllib.error
    import urllib.request

    def raise_server_error(url: str, timeout: float = 0.0) -> None:
        raise urllib.error.HTTPError(url, 503, "Service Unavailable", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", raise_server_error)

    with pytest.raises(urllib.error.HTTPError):
        fetch_pypi_versions("gkx")


def test_release_version_readers_and_tag_normalization(tmp_path: Path) -> None:
    _write_version_files(tmp_path, project="3.4.5", source="3.4.5")

    assert read_project_version(tmp_path) == "3.4.5"
    assert read_source_version(tmp_path) == "3.4.5"
    assert normalize_tag("refs/tags/v3.4.5") == "v3.4.5"
    assert normalize_tag("") is None


def test_default_tag_from_github_env_ignores_branch_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_REF_TYPE", "branch")

    assert default_tag_from_github_env() is None

    monkeypatch.setenv("GITHUB_REF_NAME", "v2.0.1")
    monkeypatch.setenv("GITHUB_REF_TYPE", "tag")

    assert default_tag_from_github_env() == "v2.0.1"


def test_ci_quick_test_matrix_references_existing_paths() -> None:
    """Keep hardcoded CI pytest and coverage shards internally consistent."""

    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github" / "workflows" / "ci.yml").read_text())
    shards = workflow["jobs"]["quick-tests"]["strategy"]["matrix"]["shard"]

    missing: list[str] = []
    for shard in shards:
        for entry in str(shard["files"]).split():
            if not (root / entry).exists():
                missing.append(f"{shard['name']}: {entry}")

    assert missing == []

    wide_job = workflow["jobs"]["wide-coverage-shards"]
    wide_shards = wide_job["strategy"]["matrix"]["shard"]
    shard_count = len(wide_shards)
    assert wide_shards == list(range(1, shard_count + 1))

    shard_step = next(
        step
        for step in wide_job["steps"]
        if "Wide package coverage" in step.get("name", "")
    )
    combine_step = next(
        step
        for step in workflow["jobs"]["wide-coverage"]["steps"]
        if step.get("name") == "Combine wide package coverage"
    )
    assert f"/{shard_count}" in shard_step["name"]
    assert f"--shards {shard_count}" in shard_step["run"]
    assert f"--shards {shard_count}" in combine_step["run"]


def _release_artifact_manifest(
    tmp_path: Path, *, sha: str, size: int, action: str = "move_to_release"
) -> Path:
    release_fields = (
        '\nrelease_tag = "v-test"\nrelease_url = "https://example.test/download/panel.png"'
        if action == "move_to_release"
        else ""
    )
    manifest = tmp_path / "release_artifacts.toml"
    manifest.write_text(
        textwrap.dedent(
            f"""
            [policy]
            release_series = "test"
            default_destination = "GitHub Releases"
            status = "planned"

            [[artifacts]]
            path = "panel.png"
            size_bytes = {size}
            sha256 = "{sha}"
            action = "{action}"
            artifact_type = "panel"
            release_asset_name = "panel.png"
            reason = "test panel"
            preview_strategy = "test preview"
            replay_command = "python make_panel.py"
            {release_fields}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_repository_hygiene_import_does_not_require_pillow() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "release"
        / "check_repository_size_manifest.py"
    )
    code = """
import importlib.abc
import runpy
import sys

class BlockPillow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "PIL" or fullname.startswith("PIL."):
            raise ModuleNotFoundError("Pillow intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockPillow())
runpy.run_path(sys.argv[1], run_name="repository_hygiene_import_test")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_release_artifact_manifest_validates_size_and_sha(tmp_path: Path) -> None:
    mod = load_release_tool("check_repository_size_manifest")
    payload = b"panel"
    (tmp_path / "panel.png").write_bytes(payload)
    manifest = _release_artifact_manifest(
        tmp_path, sha=hashlib.sha256(payload).hexdigest(), size=len(payload)
    )

    report = mod.check_release_artifact_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is True
    assert report["move_to_release_bytes"] == len(payload)


def test_release_artifact_manifest_accepts_kept_preview_action(tmp_path: Path) -> None:
    mod = load_release_tool("check_repository_size_manifest")
    payload = b"preview"
    (tmp_path / "panel.png").write_bytes(payload)
    manifest = _release_artifact_manifest(
        tmp_path,
        sha=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        action="keep_preview_in_repo",
    )

    report = mod.check_release_artifact_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is True
    assert report["move_to_release_bytes"] == 0
    assert report["artifacts"][0]["action"] == "keep_preview_in_repo"


def test_release_artifact_manifest_fails_on_sha_mismatch(tmp_path: Path) -> None:
    mod = load_release_tool("check_repository_size_manifest")
    payload = b"panel"
    (tmp_path / "panel.png").write_bytes(payload)
    manifest = _release_artifact_manifest(tmp_path, sha="0" * 64, size=len(payload))

    report = mod.check_release_artifact_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is False
    assert any("sha256" in failure for failure in report["failures"])


def test_release_artifact_manifest_accepts_uploaded_release_asset(
    tmp_path: Path,
) -> None:
    mod = load_release_tool("check_repository_size_manifest")
    payload = b"panel"
    manifest = _release_artifact_manifest(
        tmp_path, sha=hashlib.sha256(payload).hexdigest(), size=len(payload)
    )

    report = mod.check_release_artifact_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is True
    assert report["move_to_release_bytes"] == len(payload)
    assert report["artifacts"][0]["exists"] is False
    assert report["artifacts"][0]["release_tag"] == "v-test"
    assert report["artifacts"][0]["release_url"].endswith("/panel.png")


def test_release_artifact_manifest_requires_url_for_missing_moved_asset(
    tmp_path: Path,
) -> None:
    mod = load_release_tool("check_repository_size_manifest")
    payload = b"panel"
    manifest = _release_artifact_manifest(
        tmp_path, sha=hashlib.sha256(payload).hexdigest(), size=len(payload)
    )
    text = manifest.read_text(encoding="utf-8")
    text = "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith(("release_tag", "release_url"))
    )
    manifest.write_text(text + "\n", encoding="utf-8")

    report = mod.check_release_artifact_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is False
    assert any("does not exist" in failure for failure in report["failures"])


def _init_size_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "small.txt").write_text("small\n", encoding="utf-8")
    (tmp_path / "large.bin").write_bytes(b"0" * 64)
    subprocess.run(["git", "add", "small.txt", "large.bin"], cwd=tmp_path, check=True)


def test_repository_size_manifest_passes_for_allowed_large_file(tmp_path: Path) -> None:
    mod = load_release_tool("check_repository_size_manifest")
    _init_size_repo(tmp_path)
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        textwrap.dedent(
            """
            [policy]
            max_tracked_total_bytes = 1000
            max_unlisted_tracked_file_bytes = 32

            [[allowed_large_files]]
            path = "large.bin"
            max_bytes = 128
            reason = "test fixture"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    report = mod.check_repository_size_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is True
    assert report["unlisted_large_files"] == []


def test_repository_size_manifest_fails_for_unlisted_large_file(tmp_path: Path) -> None:
    mod = load_release_tool("check_repository_size_manifest")
    _init_size_repo(tmp_path)
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        "[policy]\nmax_tracked_total_bytes = 1000\nmax_unlisted_tracked_file_bytes = 32\n",
        encoding="utf-8",
    )

    report = mod.check_repository_size_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is False
    assert report["unlisted_large_files"] == [{"path": "large.bin", "bytes": 64}]
    assert any("large.bin" in failure for failure in report["failures"])


def test_repository_size_report_separates_tracked_and_local_roots(
    tmp_path: Path,
) -> None:
    mod = load_release_tool("check_repository_size_manifest")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "docs" / "_static").mkdir(parents=True)
    (tmp_path / "tools_out").mkdir()
    (tmp_path / "src" / "small.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "docs" / "_static" / "panel.png").write_bytes(b"0" * 128)
    (tmp_path / "tools_out" / "scratch.nc").write_bytes(b"1" * 256)
    subprocess.run(
        ["git", "add", "src/small.py", "docs/_static/panel.png"],
        cwd=tmp_path,
        check=True,
    )

    report = mod.build_repository_size_report(tmp_path, top_n=1)

    assert report["kind"] == "repository_size_audit"
    assert report["tracked_file_count"] == 2
    assert report["largest_tracked_files"][0]["path"] == "docs/_static/panel.png"
    assert report["tracked_by_category"]["docs/_static"] == 128
    local = {row["path"]: row for row in report["local_artifact_roots"]}
    assert local["tools_out"]["bytes"] == 256


def _write_minimal_release_status_tree(root: Path) -> None:
    text_by_path: dict[Path, list[str]] = {}
    for checks in LANES.values():
        for check in checks:
            path = root / check.path
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() in {".png", ".pdf"}:
                path.write_bytes(b"artifact")
                continue
            text_by_path.setdefault(path, []).append(check.snippet or "present")
    for path, snippets in text_by_path.items():
        path.write_text("\n".join(snippets) + "\n", encoding="utf-8")


def test_technical_release_status_passes_complete_evidence_tree(tmp_path: Path) -> None:
    _write_minimal_release_status_tree(tmp_path)

    report = build_technical_release_status(tmp_path)

    assert report["passed"] is True
    assert report["technical_release_completion_percent"] == 100.0
    assert not report["failed_required"]
    assert set(report["lanes"]) == set(LANES)


def test_technical_release_status_reports_missing_required_evidence(
    tmp_path: Path,
) -> None:
    _write_minimal_release_status_tree(tmp_path)
    (tmp_path / "docs" / "parallelization.rst").unlink()

    report = build_technical_release_status(tmp_path)

    assert report["passed"] is False
    assert report["technical_release_completion_percent"] < 100.0
    assert any(
        "parallelization_release_surface" in item for item in report["failed_required"]
    )


# ---- test_release_manifests.py ----

import re

from support.paths import REPO_ROOT

from gkx.utils import tomlcompat as tomllib
from tools.release.check_package_architecture_manifest import (
    validate_architecture_policy,
)


ROOT = REPO_ROOT
LARGE_MODULE_DIRECT_ROW_MIN_SOURCE_LINES = 2_000
PUBLIC_PACKAGE_API_INIT_EXCEPTIONS = {
    "gkx.api",
    "gkx.geometry",
    "gkx.operators",
    "gkx.operators.linear",
}


def _load_performance_manifest_tool():
    return load_release_tool("check_parallel_scaling_artifacts")


def _load_validation_coverage_tool():
    return load_release_tool("check_validation_coverage_manifest")


def _architecture_manifest(*, allowed: list[str]) -> dict[str, object]:
    return {
        "metadata": {
            "schema_version": 1,
            "title": "test architecture policy",
            "layout_authority": "plan.md",
            "status": "active",
        },
        "root_prefix_policy": {
            "blocked_prefixes": ["runtime_", "nonlinear_"],
            "allowed_root_prefix_modules": allowed,
        },
        "package_policy": {
            "required_domain_packages": ["gkx.operators"],
            "required_docs": ["plan.md"],
        },
    }


def _architecture_manifest_with_topology(
    *, count_path: str, baseline: int, target: int
) -> dict[str, object]:
    data = _architecture_manifest(allowed=[])
    data["topology_policy"] = {
        "mode": "no_regression_until_target",
        "description": "test topology policy",
        "counts": [
            {
                "name": "test_python_files",
                "path": count_path,
                "pattern": "*.py",
                "recursive": True,
                "baseline": baseline,
                "target": target,
            }
        ],
    }
    return data


def _architecture_manifest_with_complexity(
    *, baseline: int, target: int
) -> dict[str, object]:
    data = _architecture_manifest(allowed=[])
    data["complexity_policy"] = {
        "mode": "no_regression_until_target",
        "description": "test complexity policy",
        "default_max_lines": 3,
        "public_facade_max_lines": 2,
        "public_facades": ["facade.py"],
        "exceptions": [
            {
                "path": "facade.py",
                "baseline_lines": baseline,
                "target_lines": target,
                "reason": "test facade migration",
            }
        ],
    }
    return data


def _architecture_manifest_with_line_budget(
    *, count_path: str, baseline: int, target: int
) -> dict[str, object]:
    data = _architecture_manifest(allowed=[])
    data["line_budget_policy"] = {
        "mode": "no_regression_until_target",
        "description": "test aggregate line budget",
        "counts": [
            {
                "name": "test_python_lines",
                "path": count_path,
                "pattern": "*.py",
                "recursive": True,
                "baseline": baseline,
                "target": target,
            }
        ],
    }
    return data


def _performance_manifest_text(
    *, tool: str, artifact: str, status: str = "active"
) -> str:
    return f"""
[metadata]
schema_version = 1

[[lanes]]
name = "lane"
owner = "owner"
status = "{status}"
priority = "high"
platforms = ["cpu"]
cases = ["case"]
profiling_tools = ["{tool}"]
metrics = ["runtime_s"]
artifact_paths = ["{artifact}"]
bottleneck_hypotheses = ["hypothesis"]
optimization_actions = ["action"]
gates = ["gate"]
"""


def _write_package(tmp_path: Path, *modules: str) -> None:
    package = tmp_path / "src" / "gkx"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("# package\n", encoding="utf-8")
    for module in modules:
        module_path = tmp_path / "src" / Path(*module.split(".")).with_suffix(".py")
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text("# source\n", encoding="utf-8")


def _write_fast_inputs(tmp_path: Path) -> None:
    test = tmp_path / "tests" / "test_runtime.py"
    test.parent.mkdir(parents=True, exist_ok=True)
    test.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}\n", encoding="utf-8")


def _coverage_row(module: str, owned_modules: list[str] | None = None) -> str:
    owned = ""
    if owned_modules is not None:
        body = "\n".join(f'  "{owned_module}",' for owned_module in owned_modules)
        owned = f"owned_modules = [\n{body}\n]\n"
    return f"""
[[modules]]
module = "{module}"
path = "src/{module.replace(".", "/")}.py"
{owned}owner_lane = "runtime lane"
status = "active"
coverage_priority = "high"
coverage_target_percent = 95.0
reference_anchors = ["reference"]
physics_contracts = ["physics"]
numerics_contracts = ["numerics"]
fast_tests = ["tests/test_runtime.py"]
artifact_paths = ["docs/_static/gate.json"]
next_tests = ["next"]
"""


def _coverage_manifest(*rows: str) -> str:
    return """
[metadata]
package_coverage_target_percent = 95.0

[coverage_inventory]
require_all_package_modules_owned = true
excluded_modules = ["gkx.__init__"]
""" + "".join(rows)


def _validate_tmp_coverage_manifest(tmp_path: Path, manifest_text: str):
    mod = _load_validation_coverage_tool()
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(manifest_text, encoding="utf-8")
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        return mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def _repository_manifest_sets() -> tuple[set[str], set[str], set[str]]:
    mod = _load_validation_coverage_tool()
    data = mod.load_manifest()
    summary = mod.validate_manifest(data)
    direct_modules = {row["module"] for row in summary["rows"]}
    owned_modules = {
        owned_module
        for modules in summary["owned_modules_by_owner"].values()
        for owned_module in modules
    }
    excluded_modules = set(data["coverage_inventory"]["excluded_modules"])
    return direct_modules, owned_modules, excluded_modules


def _documented_public_api_modules() -> set[str]:
    api_reference = (ROOT / "docs" / "api.rst").read_text(encoding="utf-8")
    return set(
        re.findall(
            r"^\.\. automodule:: (gkx(?:\.[A-Za-z_]\w*)*)\s*$",
            api_reference,
            flags=re.MULTILINE,
        )
    )


def _manifest_candidates_for_api_module(module: str) -> set[str]:
    source_base = ROOT / "src" / Path(*module.split("."))
    candidates: set[str] = set()
    if source_base.with_suffix(".py").exists():
        candidates.add(module)
    if (source_base / "__init__.py").exists():
        candidates.add(f"{module}.__init__")
    return candidates


def _source_module_name(path: Path) -> str:
    return ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)


def _source_line_count(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def test_validate_architecture_policy_accepts_manifested_root_facade(tmp_path):
    source_root = tmp_path / "gkx"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "nonlinear_removed_helper.py").write_text("", encoding="utf-8")

    summary = validate_architecture_policy(
        _architecture_manifest(allowed=["gkx.nonlinear_removed_helper"]),
        source_root=source_root,
        check_paths=False,
    )

    assert summary["n_current_root_prefix_modules"] == 1
    assert summary["n_allowed_root_prefix_modules"] == 1


def test_validate_architecture_policy_rejects_new_root_prefix_module(tmp_path):
    source_root = tmp_path / "gkx"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "runtime_extra.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="root-level prefix modules"):
        validate_architecture_policy(
            _architecture_manifest(allowed=[]),
            source_root=source_root,
            check_paths=False,
        )


def test_validate_architecture_policy_reports_topology_gap(tmp_path):
    source_root = tmp_path / "gkx"
    count_root = tmp_path / "counted"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    count_root.mkdir()
    for index in range(3):
        (count_root / f"module_{index}.py").write_text("", encoding="utf-8")

    summary = validate_architecture_policy(
        _architecture_manifest_with_topology(
            count_path=str(count_root), baseline=5, target=2
        ),
        source_root=source_root,
        check_paths=False,
    )

    row = summary["topology_counts"][0]
    assert row["count"] == 3
    assert row["baseline"] == 5
    assert row["target"] == 2
    assert row["remaining_to_target"] == 1
    assert row["target_met"] is False
    assert summary["topology_targets_met"] is False


def test_validate_architecture_policy_rejects_topology_regression(tmp_path):
    source_root = tmp_path / "gkx"
    count_root = tmp_path / "counted"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    count_root.mkdir()
    for index in range(3):
        (count_root / f"module_{index}.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="above baseline"):
        validate_architecture_policy(
            _architecture_manifest_with_topology(
                count_path=str(count_root), baseline=2, target=1
            ),
            source_root=source_root,
            check_paths=False,
        )


def test_validate_architecture_policy_can_require_topology_targets(tmp_path):
    source_root = tmp_path / "gkx"
    count_root = tmp_path / "counted"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    count_root.mkdir()
    for index in range(2):
        (count_root / f"module_{index}.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="target not met"):
        validate_architecture_policy(
            _architecture_manifest_with_topology(
                count_path=str(count_root), baseline=3, target=1
            ),
            source_root=source_root,
            check_paths=False,
            require_topology_targets=True,
        )

    summary = validate_architecture_policy(
        _architecture_manifest_with_topology(
            count_path=str(count_root), baseline=3, target=2
        ),
        source_root=source_root,
        check_paths=False,
        require_topology_targets=True,
    )
    assert summary["topology_targets_met"] is True


def test_validate_architecture_policy_tracks_aggregate_line_budget(tmp_path):
    source_root = tmp_path / "gkx"
    count_root = tmp_path / "counted"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    count_root.mkdir()
    (count_root / "a.py").write_text("a\nb\n", encoding="utf-8")
    (count_root / "b.py").write_text("c\n", encoding="utf-8")

    summary = validate_architecture_policy(
        _architecture_manifest_with_line_budget(
            count_path=str(count_root), baseline=5, target=2
        ),
        source_root=source_root,
        check_paths=False,
    )

    row = summary["line_budget_counts"][0]
    assert row["files"] == 2
    assert row["lines"] == 3
    assert row["remaining_to_target"] == 1
    assert summary["line_budget_targets_met"] is False

    with pytest.raises(ValueError, match="line-budget target not met"):
        validate_architecture_policy(
            _architecture_manifest_with_line_budget(
                count_path=str(count_root), baseline=5, target=2
            ),
            source_root=source_root,
            check_paths=False,
            require_line_targets=True,
        )


def test_validate_architecture_policy_rejects_line_budget_regression(tmp_path):
    source_root = tmp_path / "gkx"
    count_root = tmp_path / "counted"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    count_root.mkdir()
    (count_root / "a.py").write_text("a\nb\nc\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line count regressed"):
        validate_architecture_policy(
            _architecture_manifest_with_line_budget(
                count_path=str(count_root), baseline=2, target=1
            ),
            source_root=source_root,
            check_paths=False,
        )


def test_validate_architecture_policy_tracks_complexity_exceptions(tmp_path):
    source_root = tmp_path / "gkx"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "facade.py").write_text("a\nb\nc\nd\n", encoding="utf-8")

    summary = validate_architecture_policy(
        _architecture_manifest_with_complexity(baseline=5, target=2),
        source_root=source_root,
        check_paths=False,
    )

    row = summary["complexity_exceptions"][0]
    assert row["path"] == "facade.py"
    assert row["lines"] == 4
    assert row["remaining_to_target"] == 2
    assert summary["complexity_targets_met"] is False

    with pytest.raises(ValueError, match="complexity target not met"):
        validate_architecture_policy(
            _architecture_manifest_with_complexity(baseline=5, target=2),
            source_root=source_root,
            check_paths=False,
            require_complexity_targets=True,
        )


def test_validate_architecture_policy_rejects_unowned_complexity_growth(tmp_path):
    source_root = tmp_path / "gkx"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "facade.py").write_text("a\nb\nc\n", encoding="utf-8")
    (source_root / "new_hotspot.py").write_text("a\nb\nc\nd\n", encoding="utf-8")

    with pytest.raises(ValueError, match="without reviewed exceptions"):
        validate_architecture_policy(
            _architecture_manifest_with_complexity(baseline=5, target=2),
            source_root=source_root,
            check_paths=False,
        )


def test_package_architecture_inventory_classifies_repository_areas() -> None:
    mod = load_release_tool("check_package_architecture_manifest")

    role, action, notes = mod._role_and_action(
        Path("src/gkx/operators/nonlinear/rhs.py")
    )
    tool_role, tool_action, tool_notes = mod._role_and_action(
        Path("tools/artifacts/build_linear_validation_artifacts.py")
    )
    summary = mod._summary(
        [
            mod.InventoryRow(
                path="src/gkx/operators/nonlinear/rhs.py",
                area="src/gkx/operators",
                role=role,
                action=action,
                suffix=".py",
                bytes=12,
                lines=1,
                notes=notes,
            ),
            mod.InventoryRow(
                path="tools/artifacts/build_linear_validation_artifacts.py",
                area="tools/artifacts",
                role=tool_role,
                action=tool_action,
                suffix=".py",
                bytes=8,
                lines=1,
                notes=tool_notes,
            ),
        ]
    )

    assert role == "promoted library code"
    assert action == "merge-to-2.0-domain"
    assert "target=gkx.model" in notes
    assert tool_role == "artifact builder"
    assert tool_action == "keep-or-merge"
    assert summary["merge-to-2.0-domain"] == {"files": 1, "bytes": 12}
    assert summary["keep-or-merge"] == {"files": 1, "bytes": 8}


def test_benchmark_capability_matrix_is_complete_and_fail_closed() -> None:
    with (ROOT / "benchmarks" / "capability_matrix.toml").open("rb") as stream:
        payload = tomllib.load(stream)

    metadata = payload["metadata"]
    rows = payload["capabilities"]
    by_id = {row["id"]: row for row in rows}
    allowed_statuses = {
        "validated",
        "validated_scoped",
        "validated_limited_model",
        "planned",
        "planned_research_lane",
        "blocked",
        "not_shipped",
    }

    assert metadata["comparison_code"] == "GX"
    assert metadata["comparison_revision"]
    assert metadata["comparison_source_fingerprint"].startswith("sha256:")
    assert metadata["office_instrumented_source_fingerprint"].startswith("sha256:")
    assert (
        metadata["comparison_source_fingerprint"]
        != metadata["office_instrumented_source_fingerprint"]
    )
    assert "validated_clean_rebuild" in metadata["office_binary_status"]
    assert "OpenMPI 4.1.6" in metadata["office_binary_status"]
    assert "HDF5 1.14.5" in metadata["office_binary_status"]
    assert "Cyclone" in metadata["office_runtime_probe"]
    assert "2145 steps" in metadata["office_runtime_probe"]
    assert len(by_id) == len(rows) >= 15
    assert {row["status"] for row in rows} <= allowed_statuses
    assert all(row["gkx_owner"] and row["evidence"] for row in rows)
    assert by_id["nonlinear_multi_device_domain_decomposition"]["status"] == "blocked"
    assert (
        by_id["conserving_lenard_bernstein_dougherty_like_collisions"]["status"]
        == "validated_limited_model"
    )
    assert (
        by_id["linearized_sugama_or_coulomb_collisions"]["status"]
        == "planned_research_lane"
    )
    assert (
        by_id["jax_autodiff_and_implicit_gradients"]["group"]
        == "differentiable_extension"
    )
    assert by_id["species_hermite_multi_device_decomposition"]["status"] == "planned"
    assert by_id["equilibrium_exb_flow_shear"]["status"] == "planned_research_lane"
    assert by_id["specialized_reduced_equation_sets"]["status"] == "not_shipped"

    required = payload["matched_comparison_contract"]["required_fields"]
    assert len(required) == len(set(required)) >= 10
    assert "fit_or_transport_window" in required
    assert len(payload["matched_comparison_contract"]["fail_closed_rules"]) >= 3


def test_validate_architecture_policy_rejects_stale_allowlist(tmp_path):
    source_root = tmp_path / "gkx"
    (source_root / "operators").mkdir(parents=True)
    (source_root / "operators" / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="allowlist contains modules"):
        validate_architecture_policy(
            _architecture_manifest(allowed=["gkx.nonlinear_removed_helper"]),
            source_root=source_root,
            check_paths=False,
        )


def test_repository_performance_manifest_is_well_formed() -> None:
    mod = _load_performance_manifest_tool()
    summary = mod.validate_manifest(mod.load_manifest())

    assert summary["n_lanes"] >= 5
    active = set(summary["high_priority_active"])
    assert "cold_start_compile" in active
    assert "nonlinear_warm_throughput" in active
    rows = {row["name"]: row for row in summary["rows"]}
    assert rows["end_to_end_runtime_memory"]["n_tools"] >= 2
    assert rows["parallel_scaling"]["priority"] == "medium"


def test_performance_manifest_main_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_performance_manifest_tool()
    out_json = tmp_path / "summary.json"

    assert mod.run_performance_manifest(["--out-json", str(out_json)]) == 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["n_lanes"] >= 5
    assert "memory_efficiency" in {row["name"] for row in payload["rows"]}


def test_performance_manifest_rejects_missing_tool(tmp_path: Path) -> None:
    mod = _load_performance_manifest_tool()
    artifact = tmp_path / "docs" / "_static" / "runtime.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("artifact\n", encoding="utf-8")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _performance_manifest_text(
            tool="tools/missing.py", artifact="docs/_static/runtime.png"
        ),
        encoding="utf-8",
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="profiling tool does not exist"):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_performance_manifest_accepts_benchmark_performance_driver(
    tmp_path: Path,
) -> None:
    mod = _load_performance_manifest_tool()
    tool = tmp_path / "benchmarks" / "performance" / "benchmark_runtime_memory.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("# benchmark\n", encoding="utf-8")
    artifact = tmp_path / "docs" / "_static" / "runtime.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("artifact\n", encoding="utf-8")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _performance_manifest_text(
            tool="benchmarks/performance/benchmark_runtime_memory.py",
            artifact="docs/_static/runtime.png",
        ),
        encoding="utf-8",
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        summary = mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root

    assert summary["rows"][0]["n_tools"] == 1


def test_performance_manifest_rejects_unowned_driver_path(tmp_path: Path) -> None:
    mod = _load_performance_manifest_tool()
    tool = tmp_path / "scripts" / "benchmark.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("# benchmark\n", encoding="utf-8")
    artifact = tmp_path / "docs" / "_static" / "runtime.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("artifact\n", encoding="utf-8")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _performance_manifest_text(
            tool="scripts/benchmark.py", artifact="docs/_static/runtime.png"
        ),
        encoding="utf-8",
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(
            ValueError,
            match=r"tools/ or benchmarks/performance/",
        ):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_performance_manifest_rejects_invalid_status(tmp_path: Path) -> None:
    mod = _load_performance_manifest_tool()
    tool = tmp_path / "tools" / "profile.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("# tool\n", encoding="utf-8")
    artifact = tmp_path / "docs" / "_static" / "runtime.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("artifact\n", encoding="utf-8")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _performance_manifest_text(
            tool="tools/profile.py",
            artifact="docs/_static/runtime.png",
            status="halfway",
        ),
        encoding="utf-8",
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="invalid status"):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_documented_public_api_modules_have_manifest_tracking() -> None:
    direct_modules, owned_modules, excluded_modules = _repository_manifest_sets()
    tracked_modules = direct_modules | owned_modules | excluded_modules
    public_modules = _documented_public_api_modules()

    missing_source = sorted(
        module
        for module in public_modules
        if not _manifest_candidates_for_api_module(module)
    )
    missing_manifest = {
        module: sorted(candidates)
        for module in sorted(public_modules)
        if (candidates := _manifest_candidates_for_api_module(module))
        and candidates.isdisjoint(tracked_modules)
    }
    excluded_package_api = {
        module
        for module in public_modules
        if f"{module}.__init__" in _manifest_candidates_for_api_module(module)
        and f"{module}.__init__" in excluded_modules
    }

    assert not missing_source
    assert not missing_manifest
    assert excluded_package_api <= PUBLIC_PACKAGE_API_INIT_EXCEPTIONS


def test_large_modules_have_direct_manifest_rows() -> None:
    direct_modules, _, _ = _repository_manifest_sets()
    large_modules_without_direct_rows: dict[str, int] = {}
    for path in (ROOT / "src" / "gkx").rglob("*.py"):
        if path.name == "__init__.py":
            continue
        source_lines = _source_line_count(path)
        module = _source_module_name(path)
        if (
            source_lines >= LARGE_MODULE_DIRECT_ROW_MIN_SOURCE_LINES
            and module not in direct_modules
        ):
            large_modules_without_direct_rows[module] = source_lines

    assert not large_modules_without_direct_rows


def test_manifest_accepts_owned_refactor_modules(tmp_path: Path) -> None:
    _write_package(
        tmp_path, "gkx.runtime", "gkx.workflows.runtime.config"
    )
    _write_fast_inputs(tmp_path)

    summary = _validate_tmp_coverage_manifest(
        tmp_path,
        _coverage_manifest(
            _coverage_row(
                "gkx.runtime",
                owned_modules=["gkx.workflows.runtime.config"],
            )
        ),
    )

    assert summary["n_direct_modules"] == 1
    assert summary["n_owned_modules"] == 1
    assert summary["n_excluded_modules"] == 1
    assert summary["owned_modules_by_owner"]["gkx.runtime"] == [
        "gkx.workflows.runtime.config"
    ]


def test_manifest_rejects_unowned_package_modules(tmp_path: Path) -> None:
    _write_package(
        tmp_path, "gkx.runtime", "gkx.workflows.runtime.config"
    )
    _write_fast_inputs(tmp_path)

    with pytest.raises(ValueError, match="package modules lack coverage ownership"):
        _validate_tmp_coverage_manifest(
            tmp_path, _coverage_manifest(_coverage_row("gkx.runtime"))
        )


def test_manifest_rejects_duplicate_owned_modules(tmp_path: Path) -> None:
    _write_package(
        tmp_path,
        "gkx.runtime",
        "gkx.linear",
        "gkx.workflows.runtime.config",
    )
    _write_fast_inputs(tmp_path)

    manifest = _coverage_manifest(
        _coverage_row(
            "gkx.runtime", owned_modules=["gkx.workflows.runtime.config"]
        ),
        _coverage_row(
            "gkx.linear", owned_modules=["gkx.workflows.runtime.config"]
        ),
    )
    with pytest.raises(ValueError, match="duplicate coverage ownership"):
        _validate_tmp_coverage_manifest(tmp_path, manifest)


def test_manifest_rejects_direct_rows_listed_as_owned_modules(tmp_path: Path) -> None:
    _write_package(tmp_path, "gkx.runtime", "gkx.linear")
    _write_fast_inputs(tmp_path)

    manifest = _coverage_manifest(
        _coverage_row("gkx.runtime", owned_modules=["gkx.linear"]),
        _coverage_row("gkx.linear"),
    )
    with pytest.raises(
        ValueError, match="direct manifest rows must not be listed as owned modules"
    ):
        _validate_tmp_coverage_manifest(tmp_path, manifest)


# ---- test_release_scope_docs.py ----


ROOT = Path(__file__).resolve().parents[2]


def _compact(path: str) -> str:
    return " ".join((ROOT / path).read_text(encoding="utf-8").split())


REQUIRED_PHRASES = {
    "docs/release_scope.rst": (
        "a scoped model-development and optimization-screening result",
        "No runtime/TOML absolute-flux predictor",
        "Solovev and shaped-pressure stress outliers outside the scoped claim",
        "W7-X TEM / kinetic-electron validation",
        "W7-X long-window zonal recurrence/damping closure",
        "selected QA optimized-equilibrium audit is the current scoped exception",
    ),
    "docs/verification_matrix.rst": (
        "Closed as scoped model-development result / failed promotion gate",
        "does not promote a runtime/TOML absolute-flux predictor",
        "W7-X zonal long-window recurrence/damping and W7-X TEM / kinetic-electron validation remain outside",
        "Production nonlinear optimization is promoted only for the selected optimized-equilibrium audit",
    ),
    "README.md": (
        "not a runtime/TOML absolute-flux predictor",
        "declared Solovev and shaped-pressure stress outliers",
        "W7-X zonal long-window recurrence/damping and W7-X TEM / kinetic-electron extensions are deferred",
        "converged post-transient heat-flux windows",
        "Sensitivity sweeps can use the same deterministic independent-work reconstruction, but they need a dedicated",
    ),
    "docs/performance.rst": (
        "Sensitivity sweeps are covered by",
        "before any speedup claim is promoted",
        "Communication-aware nonlinear domain decomposition remains",
    ),
    "docs/parallelization.rst": (
        "It is not a production nonlinear domain",
        "whole-state nonlinear sharding speedup",
    ),
    "docs/examples.rst": (
        "opt-in electrostatic linear-RHS identity artifact",
        "publication speedup claim",
    ),
}

FORBIDDEN_PHRASES = (
    "is a runtime/TOML absolute-flux predictor",
    "promotes a runtime/TOML absolute-flux predictor",
    "runtime/TOML absolute-flux predictor is accepted",
    "universal nonlinear transport model is promoted",
    "W7-X TEM / kinetic-electron validation is closed",
    "W7-X zonal long-window recurrence/damping closure is closed",
    "production nonlinear heat-flux stellarator optimization is release-ready",
    "nonlinear production optimization is release-ready",
    "optimized-equilibrium nonlinear heat-flux validation is closed",
    "production parallelization path for linear scans, quasilinear studies, sensitivity sweeps, and UQ ensembles",
    "production parallelization path for linear scans, quasilinear studies, sensitivity sweeps",
    "current production-parallelization identity artifact",
    "production nonlinear sharding speedup",
    "production nonlinear domain-decomposition speedup claim is closed",
    "broad multi-GPU nonlinear speedup claim",
)

COMPARISON_CODE_PATTERN = re.compile(
    r"\bGX\b|\bgx\b|gx_|_gx|GX-reference|comparison-code"
)
# The gate keeps the comparison code out of the physics, not out of the two
# files that exist to read its output. ``gkx --plot`` accepts a bundle written
# by another code so a cross-code comparison is one command rather than a
# bespoke script, and a reader cannot describe what it parses without naming
# it. Confining that to a reader plus the registry that selects it is the
# point: the figure code never learns which codes exist, and naming one
# anywhere else is still a violation.
COMPARISON_ALLOWED_SOURCE_PREFIXES: tuple[Path, ...] = (
    Path("src/gkx/artifacts/foreign_output.py"),
    Path("src/gkx/artifacts/gx_output.py"),
)


def test_claim_scope_pages_keep_required_quasilinear_boundaries() -> None:
    missing: list[str] = []
    for path, phrases in REQUIRED_PHRASES.items():
        text = _compact(path)
        missing.extend(f"{path}: {phrase}" for phrase in phrases if phrase not in text)

    assert not missing


def test_readme_python_quickstart_imports_exist() -> None:
    """Keep the concise README example on the installed public import surface."""

    from gkx import (
        CycloneBaseCase,
        LinearParams,
        integrate_linear_from_config,
    )
    from gkx.core.grid import build_spectral_grid
    from gkx.geometry import SAlphaGeometry

    assert CycloneBaseCase is not None
    assert LinearParams is not None
    assert integrate_linear_from_config is not None
    assert build_spectral_grid is not None
    assert SAlphaGeometry is not None


def test_claim_scope_pages_avoid_promoted_unscoped_claims() -> None:
    violations: list[str] = []
    for path in REQUIRED_PHRASES:
        text = _compact(path)
        violations.extend(
            f"{path}: {phrase}" for phrase in FORBIDDEN_PHRASES if phrase in text
        )

    assert not violations


def test_core_source_avoids_comparison_code_terminology_outside_benchmarks() -> None:
    violations: list[str] = []
    source_root = ROOT / "src" / "gkx"
    for path in source_root.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(
            rel.is_relative_to(prefix) for prefix in COMPARISON_ALLOWED_SOURCE_PREFIXES
        ):
            continue
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if COMPARISON_CODE_PATTERN.search(line):
                violations.append(f"{rel}:{line_no}: {line.strip()}")

    assert not violations


# ---- test_run_test_gates.py fast ----

from pathlib import Path

from tools.release import run_test_gates


def test_discover_test_files_returns_recursive_tests(tmp_path: Path) -> None:
    (tmp_path / "test_b.py").write_text("", encoding="utf-8")
    (tmp_path / "test_a.py").write_text("", encoding="utf-8")
    (tmp_path / "helper.py").write_text("", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "test_nested.py").write_text("", encoding="utf-8")

    assert [
        path.relative_to(tmp_path)
        for path in run_test_gates.discover_test_files(tmp_path)
    ] == [
        Path("nested/test_nested.py"),
        Path("test_a.py"),
        Path("test_b.py"),
    ]


def test_run_test_gates_fast_relative_test_dir_resolves_under_repository_root() -> None:
    resolved = run_test_gates._resolve_test_dir(Path("tests"))

    assert resolved.is_absolute()
    assert resolved.name == "tests"
    assert run_test_gates.discover_test_files(Path("tests"))


def test_run_tests_uses_bounded_pytest_invocations(monkeypatch, tmp_path: Path) -> None:
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_ok(): assert True\n", encoding="utf-8")
    calls: list[tuple[list[str], float]] = []

    def _fake_run(cmd, *, cwd, check, timeout):
        del cwd, check
        calls.append((list(cmd), float(timeout)))

    monkeypatch.setattr(run_test_gates.subprocess, "run", _fake_run)
    code, results = run_test_gates.run_tests(
        [test_file],
        per_file_timeout_s=12.0,
        total_timeout_s=30.0,
        pytest_args=["-k", "sample"],
    )

    assert code == 0
    assert results[0][1] == "ok"
    assert calls[0][0][0:4] == [run_test_gates.sys.executable, "-m", "pytest", "-q"]
    assert calls[0][0][-3:] == ["-k", "sample", str(test_file)]
    assert calls[0][1] <= 12.0


def test_run_tests_returns_124_on_timeout(monkeypatch, tmp_path: Path) -> None:
    test_file = tmp_path / "test_timeout.py"
    test_file.write_text("def test_slow(): assert True\n", encoding="utf-8")

    def _fake_run(cmd, *, cwd, check, timeout):
        del cwd, check
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(run_test_gates.subprocess, "run", _fake_run)
    code, results = run_test_gates.run_tests(
        [test_file],
        per_file_timeout_s=1.0,
        total_timeout_s=30.0,
    )

    assert code == 124
    assert results[0][1] == "timeout"


def test_run_tests_treats_pytest_no_tests_collected_as_skip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "test_integration_only.py"
    test_file.write_text(
        "import pytest\npytestmark = pytest.mark.integration\n",
        encoding="utf-8",
    )

    def _fake_run(cmd, *, cwd, check, timeout):
        del cwd, check, timeout
        raise subprocess.CalledProcessError(5, cmd)

    monkeypatch.setattr(run_test_gates.subprocess, "run", _fake_run)
    code, results = run_test_gates.run_tests(
        [test_file],
        per_file_timeout_s=1.0,
        total_timeout_s=30.0,
    )

    assert code == 0
    assert results[0][1] == "skipped(no_tests_collected)"


def test_run_tests_marks_remaining_files_after_total_timeout(
    monkeypatch, tmp_path: Path
) -> None:
    files = [tmp_path / "test_one.py", tmp_path / "test_two.py"]
    for path in files:
        path.write_text("def test_ok(): assert True\n", encoding="utf-8")
    monotonic_values = iter([0.0, 0.0, 0.1, 0.1, 0.2, 2.0])

    def _fake_monotonic() -> float:
        return next(monotonic_values)

    def _fake_run(cmd, *, cwd, check, timeout):
        del cmd, cwd, check, timeout

    monkeypatch.setattr(run_test_gates.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(run_test_gates.subprocess, "run", _fake_run)
    code, results = run_test_gates.run_tests(
        files,
        per_file_timeout_s=10.0,
        total_timeout_s=1.0,
    )

    assert code == 124
    assert results[0][1] == "ok"
    assert results[1][1] == "not_run(total_timeout)"


# ---- test_run_test_gates.py wide-coverage ----

from pathlib import Path


from tools.release.run_test_gates import (
    WIDE_COVERAGE_LOGICAL_CPU_DEVICES,
    WIDE_COVERAGE_NODE_BATCHES,
    build_coverage_shard_report,
    _resolve_test_dir,
    discover_test_files,
    split_contiguous,
    validate_coverage_shard_report,
    split_shards,
    wide_coverage_environment,
    wide_coverage_shard_batches,
    write_json,
)


def test_split_shards_is_round_robin_and_complete() -> None:
    files = [Path(f"tests/test_{idx}.py") for idx in range(7)]
    shards = split_shards(files, 3)

    assert shards == [files[0::3], files[1::3], files[2::3]]
    assert sorted(path for shard in shards for path in shard) == files


def test_split_shards_isolates_known_high_cost_tests() -> None:
    expensive = [
        Path("tests/unit/solvers/test_diffrax_integrators_core.py"),
        Path("tests/integration/runtime/test_runtime_runner.py"),
        Path("tests/unit/nonlinear/test_nonlinear.py"),
        Path("tests/unit/nonlinear/test_nonlinear_helpers_extra.py"),
        Path("tests/unit/parallel/test_parallel_linear_velocity.py"),
    ]
    files = expensive + [Path(f"tests/test_light_{idx}.py") for idx in range(12)]
    shards = split_shards(files, 6)

    expensive_by_shard = [
        [path.name for path in shard if path in expensive] for shard in shards
    ]
    assert sorted(name for shard in expensive_by_shard for name in shard) == [
        "test_diffrax_integrators_core.py",
        "test_nonlinear.py",
        "test_nonlinear_helpers_extra.py",
        "test_parallel_linear_velocity.py",
        "test_runtime_runner.py",
    ]
    assert all(len(shard_names) <= 1 for shard_names in expensive_by_shard)
    assert all(
        len(shard) == 1 for shard in shards if any(path in expensive for path in shard)
    )


def test_split_shards_rejects_nonpositive_count() -> None:
    with pytest.raises(ValueError, match="nshards"):
        split_shards([Path("tests/test_a.py")], 0)


def test_split_contiguous_is_complete_balanced_and_bounded() -> None:
    items = [f"test_{idx}" for idx in range(7)]

    chunks = split_contiguous(items, 3)

    assert chunks == [items[:3], items[3:5], items[5:]]
    assert [item for chunk in chunks for item in chunk] == items
    assert split_contiguous(items[:2], 4) == [[items[0]], [items[1]]]
    assert split_contiguous([], 2) == []
    with pytest.raises(ValueError, match="nchunks"):
        split_contiguous(items, 0)


def test_wide_coverage_parallel_owner_requests_four_logical_cpu_devices(
    monkeypatch,
) -> None:
    monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_enable_fast_math=false")
    env = wide_coverage_environment(
        [Path("tests/unit/parallel/test_parallel_linear_velocity.py")]
    )

    assert env is not None
    assert env["JAX_PLATFORMS"] == "cpu"
    assert "--xla_force_host_platform_device_count=4" in env["XLA_FLAGS"]
    assert "--xla_cpu_enable_fast_math=false" in env["XLA_FLAGS"]
    assert wide_coverage_environment([Path("tests/test_light.py")]) is None


def test_wide_coverage_logical_cpu_owners_run_whole_not_by_test_name() -> None:
    """Device-gated owners must run entire, not as a chosen subset.

    Selecting part of such a file leaves every unselected device gate unrun
    while the shard still reports success, so a device-count regression stays
    invisible. Node batching selects by node ID and is also wrong here for a
    second reason: these tests share one JAX compilation cache, so separate
    processes re-pay the compiles the single command pays once.
    """

    for name in WIDE_COVERAGE_LOGICAL_CPU_DEVICES:
        assert name not in WIDE_COVERAGE_NODE_BATCHES


def test_wide_coverage_shard_batches_cover_every_target_once(monkeypatch) -> None:
    from tools.release.run_test_gates import REPO_ROOT

    plain = [REPO_ROOT / "tests/test_a.py", REPO_ROOT / "tests/test_b.py"]

    assert wide_coverage_shard_batches(plain, pytest_args=[]) == [
        ["tests/test_a.py", "tests/test_b.py"]
    ]

    owner = REPO_ROOT / "tests/unit/nonlinear/test_nonlinear_helpers_extra.py"
    nodeids = [f"{owner}::test_{idx}" for idx in range(13)]
    monkeypatch.setattr(
        "tools.release.run_test_gates.collect_pytest_nodeids",
        lambda path, pytest_args: nodeids,
    )
    batches = wide_coverage_shard_batches([owner], pytest_args=[])

    assert len(batches) == WIDE_COVERAGE_NODE_BATCHES[owner.name]
    assert [nodeid for batch in batches for nodeid in batch] == nodeids
    with pytest.raises(SystemExit, match="isolated shard"):
        wide_coverage_shard_batches(
            [owner, REPO_ROOT / "tests/test_a.py"], pytest_args=[]
        )


def test_discover_test_files_returns_sorted_recursive_tests(tmp_path: Path) -> None:
    (tmp_path / "test_b.py").write_text("", encoding="utf-8")
    (tmp_path / "test_a.py").write_text("", encoding="utf-8")
    (tmp_path / "helper.py").write_text("", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "test_nested.py").write_text("", encoding="utf-8")

    assert [path.relative_to(tmp_path) for path in discover_test_files(tmp_path)] == [
        Path("nested/test_nested.py"),
        Path("test_a.py"),
        Path("test_b.py"),
    ]


def test_wide_coverage_relative_test_dir_resolves_under_repository_root() -> None:
    resolved = _resolve_test_dir(Path("tests"))

    assert resolved.is_absolute()
    assert resolved.name == "tests"
    assert discover_test_files(Path("tests"))


def test_coverage_shard_report_tracks_labeled_data(tmp_path: Path) -> None:
    (tmp_path / ".coverage.shard-1.0").write_text("data", encoding="utf-8")
    (tmp_path / ".coverage.shard-2.0").write_text("data", encoding="utf-8")
    (tmp_path / ".coverage.local").write_text("data", encoding="utf-8")

    report = build_coverage_shard_report(tmp_path, 3)

    assert report["coverage_data_file_count"] == 3
    assert report["labeled_shards"] == {
        "1": [".coverage.shard-1.0"],
        "2": [".coverage.shard-2.0"],
    }
    assert report["unlabeled_coverage_data_files"] == [".coverage.local"]
    assert report["missing_labeled_shards"] == [3]
    failures = validate_coverage_shard_report(report, require_labeled_shards=True)
    assert "missing labeled coverage data for shards: [3]" in failures


def test_coverage_shard_report_rejects_empty_and_out_of_range_data(
    tmp_path: Path,
) -> None:
    (tmp_path / ".coverage.shard-1.0").write_text("data", encoding="utf-8")
    (tmp_path / ".coverage.shard-4.0").write_text("data", encoding="utf-8")
    (tmp_path / "EMPTY_SHARD_2").write_text("empty shard\n", encoding="utf-8")

    report = build_coverage_shard_report(tmp_path, 3)
    failures = validate_coverage_shard_report(report, require_labeled_shards=True)

    assert "empty shard markers found: ['EMPTY_SHARD_2']" in failures
    assert (
        "out-of-range labeled coverage data files found: ['.coverage.shard-4.0']"
        in failures
    )
    assert "missing labeled coverage data for shards: [2, 3]" in failures


def test_coverage_shard_report_requires_some_coverage_data(tmp_path: Path) -> None:
    report = build_coverage_shard_report(tmp_path, 2)

    assert validate_coverage_shard_report(report, require_labeled_shards=False) == [
        "no coverage.py data files were found"
    ]


def test_write_json_creates_parent_directory(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "manifest.json"

    write_json(out, {"b": 2, "a": 1})

    assert out.read_text(encoding="utf-8").splitlines() == [
        "{",
        '  "a": 1,',
        '  "b": 2',
        "}",
    ]


# ---- test_validation_coverage_manifest.py ----

from pathlib import Path


def _load_validation_tool_module():
    return load_release_tool("check_validation_coverage_manifest")


def _manifest_text(
    *,
    source: str,
    test: str,
    artifact: str,
    module: str = "gkx.runtime",
    status: str = "active",
) -> str:
    return f"""
[metadata]
package_coverage_target_percent = 95.0

[coverage_inventory]
require_all_package_modules_owned = true
excluded_modules = ["gkx.__init__"]

[[modules]]
module = "{module}"
path = "{source}"
owner_lane = "runtime lane"
status = "{status}"
coverage_priority = "high"
coverage_target_percent = 95.0
reference_anchors = ["reference"]
physics_contracts = ["physics"]
numerics_contracts = ["numerics"]
fast_tests = ["{test}"]
artifact_paths = ["{artifact}"]
next_tests = ["next"]
"""


def _write_minimal_package(tmp_path: Path, *modules: str) -> None:
    package = tmp_path / "src" / "gkx"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("# package\n")
    for module in modules:
        assert module.startswith("gkx.")
        module_path = tmp_path / "src" / Path(*module.split(".")).with_suffix(".py")
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text("# source\n")


def test_repository_validation_manifest_is_well_formed() -> None:
    mod = _load_validation_tool_module()
    summary = mod.validate_manifest(mod.load_manifest())

    assert summary["package_coverage_target_percent"] == 95.0
    assert summary["n_modules"] >= 10
    assert summary["n_package_modules"] == (
        summary["n_direct_modules"]
        + summary["n_owned_modules"]
        + summary["n_excluded_modules"]
    )
    rows = {row["module"]: row for row in summary["rows"]}
    assert rows["gkx.terms.assembly"]["coverage_target_percent"] == 95.0
    assert rows["gkx.runtime"]["n_owned_modules"] >= 5
    assert rows["gkx.diagnostics.validation_gates"]["n_physics_contracts"] >= 2
    assert (
        rows["gkx.objectives.gradient_gates"]["coverage_target_percent"] == 95.0
    )
    assert (
        rows["gkx.objectives.vmec_boozer_gradients"]["n_numerics_contracts"] >= 2
    )

    assert (
        rows["gkx.operators.linear.cache_builder"]["coverage_target_percent"]
        == 95.0
    )
    assert rows["gkx.operators.linear.cache_builder"]["n_owned_modules"] == 2
    assert rows["gkx.operators.linear.moments"]["n_numerics_contracts"] >= 2
    assert rows["gkx.operators.linear.params"]["n_physics_contracts"] >= 2
    assert rows["gkx.operators.linear.linked"]["n_owned_modules"] == 0
    assert rows["gkx.solvers.linear.parallel"]["coverage_target_percent"] == 95.0
    assert rows["gkx.operators.nonlinear.rhs"]["coverage_target_percent"] == 95.0
    assert rows["gkx.operators.nonlinear.rhs"]["n_numerics_contracts"] >= 2
    assert (
        rows["gkx.operators.nonlinear.diagnostic_state"][
            "coverage_target_percent"
        ]
        == 95.0
    )
    assert (
        rows["gkx.operators.nonlinear.diagnostic_state"]["n_physics_contracts"]
        >= 2
    )
    spectral_core = rows["gkx.operators.nonlinear.spectral_core"]
    assert spectral_core["coverage_target_percent"] == 95.0
    assert spectral_core["n_owned_modules"] >= 4
    assert spectral_core["n_numerics_contracts"] >= 2
    assert spectral_core["n_physics_contracts"] >= 2
    assert (
        rows["gkx.solvers.nonlinear.explicit"]["coverage_target_percent"] == 95.0
    )
    assert rows["gkx.solvers.nonlinear.explicit"]["n_numerics_contracts"] >= 2
    assert rows["gkx.solvers.nonlinear.imex"]["coverage_target_percent"] == 95.0
    assert rows["gkx.solvers.nonlinear.imex"]["n_physics_contracts"] >= 2
    assert "gkx.solvers.nonlinear.state_integration" in summary["high_priority_open"]


def test_validation_manifest_main_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_validation_tool_module()
    out_json = tmp_path / "summary.json"

    assert mod.main(["--out-json", str(out_json)]) == 0

    payload = json.loads(out_json.read_text())
    assert payload["n_modules"] >= 10
    assert payload["package_coverage_target_percent"] == 95.0


def test_validation_manifest_rejects_missing_fast_test(tmp_path: Path) -> None:
    mod = _load_validation_tool_module()
    _write_minimal_package(tmp_path, "gkx.runtime")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            source="src/gkx/runtime.py",
            test="tests/missing.py",
            artifact="docs/_static/gate.json",
        )
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="fast test does not exist"):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_validation_manifest_rejects_invalid_status(tmp_path: Path) -> None:
    mod = _load_validation_tool_module()
    _write_minimal_package(tmp_path, "gkx.runtime")
    test = tmp_path / "tests" / "test_runtime.py"
    test.parent.mkdir()
    test.write_text("def test_placeholder():\n    assert True\n")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            source="src/gkx/runtime.py",
            test="tests/test_runtime.py",
            artifact="docs/_static/gate.json",
            status="halfway",
        )
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="invalid status"):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_validation_manifest_rejects_duplicate_manifest_list_entries(
    tmp_path: Path,
) -> None:
    mod = _load_validation_tool_module()
    _write_minimal_package(tmp_path, "gkx.runtime", "gkx.config")
    test = tmp_path / "tests" / "test_runtime.py"
    test.parent.mkdir()
    test.write_text("def test_placeholder():\n    assert True\n")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        """
[metadata]
package_coverage_target_percent = 95.0

[coverage_inventory]
require_all_package_modules_owned = true
excluded_modules = ["gkx.__init__"]

[[modules]]
module = "gkx.runtime"
path = "src/gkx/runtime.py"
owned_modules = ["gkx.config", "gkx.config"]
owner_lane = "runtime lane"
status = "active"
coverage_priority = "high"
coverage_target_percent = 95.0
reference_anchors = ["reference"]
physics_contracts = ["physics"]
numerics_contracts = ["numerics"]
fast_tests = ["tests/test_runtime.py"]
artifact_paths = ["docs/_static/gate.json"]
next_tests = ["next"]
""".strip()
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(
            ValueError, match="owned_modules contains duplicate entries"
        ):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_validation_manifest_rejects_directory_fast_test(tmp_path: Path) -> None:
    mod = _load_validation_tool_module()
    _write_minimal_package(tmp_path, "gkx.runtime")
    test_dir = tmp_path / "tests" / "runtime_cases"
    test_dir.mkdir(parents=True)
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            source="src/gkx/runtime.py",
            test="tests/runtime_cases",
            artifact="docs/_static/gate.json",
        )
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="fast test must be a file"):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_validation_manifest_accepts_nested_fast_test_seen_by_wide_gate(
    tmp_path: Path,
) -> None:
    mod = _load_validation_tool_module()
    _write_minimal_package(tmp_path, "gkx.runtime")
    test = tmp_path / "tests" / "runtime" / "test_runtime.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_placeholder():\n    assert True\n")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            source="src/gkx/runtime.py",
            test="tests/runtime/test_runtime.py",
            artifact="docs/_static/gate.json",
        )
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        summary = mod.validate_manifest(mod.load_manifest(manifest))
        assert summary["n_modules"] == 1
    finally:
        mod.REPO_ROOT = old_root


def test_validation_manifest_rejects_non_pytest_fast_test_name(tmp_path: Path) -> None:
    mod = _load_validation_tool_module()
    _write_minimal_package(tmp_path, "gkx.runtime")
    test = tmp_path / "tests" / "runtime_cases.py"
    test.parent.mkdir()
    test.write_text("def test_placeholder():\n    assert True\n")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            source="src/gkx/runtime.py",
            test="tests/runtime_cases.py",
            artifact="docs/_static/gate.json",
        )
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match=r"tests/\*\*/test_\*\.py"):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_validation_manifest_attaches_measured_package_coverage(tmp_path: Path) -> None:
    mod = _load_validation_tool_module()
    _write_minimal_package(tmp_path, "gkx.runtime")
    test = tmp_path / "tests" / "test_runtime.py"
    test.parent.mkdir()
    test.write_text("def test_placeholder():\n    assert True\n")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            source="src/gkx/runtime.py",
            test="tests/test_runtime.py",
            artifact="docs/_static/gate.json",
        )
    )
    coverage_xml = tmp_path / "coverage.xml"
    coverage_xml.write_text(
        """
<coverage line-rate="0.96">
  <packages>
    <package name="gkx">
      <classes>
        <class filename="src/gkx/runtime.py" line-rate="0.97" />
      </classes>
    </package>
  </packages>
</coverage>
""".strip()
    )

    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        summary = mod.validate_manifest(
            mod.load_manifest(manifest),
            coverage_xml=coverage_xml,
            enforce_package_coverage=True,
        )
    finally:
        mod.REPO_ROOT = old_root

    measured = summary["coverage_xml_summary"]
    assert measured["package_coverage_passed"] is True
    assert measured["package_coverage_percent"] == pytest.approx(96.0)
    assert measured["n_modules_below_target"] == 0
    assert measured["module_rows"][0]["coverage_percent"] == pytest.approx(97.0)


def test_validation_manifest_rejects_duplicate_coverage_xml_module_entries(
    tmp_path: Path,
) -> None:
    mod = _load_validation_tool_module()
    _write_minimal_package(tmp_path, "gkx.runtime")
    test = tmp_path / "tests" / "test_runtime.py"
    test.parent.mkdir()
    test.write_text("def test_placeholder():\n    assert True\n")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            source="src/gkx/runtime.py",
            test="tests/test_runtime.py",
            artifact="docs/_static/gate.json",
        )
    )
    coverage_xml = tmp_path / "coverage.xml"
    coverage_xml.write_text(
        """
<coverage line-rate="0.96">
  <packages>
    <package name="gkx">
      <classes>
        <class filename="src/gkx/runtime.py" line-rate="0.97" />
        <class filename="gkx/runtime.py" line-rate="0.50" />
      </classes>
    </package>
  </packages>
</coverage>
""".strip()
    )

    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(
            ValueError, match="duplicate coverage entry for gkx.runtime"
        ):
            mod.validate_manifest(
                mod.load_manifest(manifest), coverage_xml=coverage_xml
            )
    finally:
        mod.REPO_ROOT = old_root


def test_validation_manifest_rejects_package_coverage_below_target(
    tmp_path: Path,
) -> None:
    mod = _load_validation_tool_module()
    _write_minimal_package(tmp_path, "gkx.runtime")
    test = tmp_path / "tests" / "test_runtime.py"
    test.parent.mkdir()
    test.write_text("def test_placeholder():\n    assert True\n")
    artifact = tmp_path / "docs" / "_static" / "gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(
            source="src/gkx/runtime.py",
            test="tests/test_runtime.py",
            artifact="docs/_static/gate.json",
        )
    )
    coverage_xml = tmp_path / "coverage.xml"
    coverage_xml.write_text(
        """
<coverage line-rate="0.949">
  <packages>
    <package name="gkx">
      <classes>
        <class filename="gkx/runtime.py" line-rate="1.0" />
      </classes>
    </package>
  </packages>
</coverage>
""".strip()
    )

    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="package coverage below manifest target"):
            mod.validate_manifest(
                mod.load_manifest(manifest),
                coverage_xml=coverage_xml,
                enforce_package_coverage=True,
            )
    finally:
        mod.REPO_ROOT = old_root


# ---- interpreter-floor portability gates ----

"""Tests that the declared ``requires-python`` floor is actually reachable.

``tomllib`` is standard library only from Python 3.11, so a bare ``import
tomllib`` makes the importing module uncollectable on the declared 3.10 floor.
Every CI job ran 3.11, so such imports accumulated unseen until the suite was
run on a stock 3.10 box and died during collection. The fallback now lives in
exactly one module and these gates keep it there.
"""

import os
from pathlib import Path

FLOOR_REPO_ROOT = Path(__file__).resolve().parents[2]
TOML_SHIM = "src/gkx/utils/tomlcompat.py"
TOML_SHIM_IMPORT = "from gkx.utils import tomlcompat as tomllib"
# Gates the repo-hygiene job runs before anything is pip-installed.
UNINSTALLED_RELEASE_GATES = (
    "tools/release/check_repository_size_manifest.py",
    "tools/release/check_package_architecture_manifest.py",
    "tools/release/check_parallel_scaling_artifacts.py",
    "tools/release/check_release_readiness.py",
    "tools/release/check_validation_coverage_manifest.py",
)


def _tracked_python_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=FLOOR_REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in proc.stdout.splitlines() if line]


def _uninstalled_run(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the way repo-hygiene does: no installed gkx distribution.

    From 3.11 that is exactly ``-S``, which drops site-packages entirely and so
    proves the gate reaches the shim through ``src`` and needs nothing but the
    standard library. On 3.10 the shim legitimately needs ``tomli`` -- the
    dependency pyproject declares under ``python_version < '3.11'`` -- and that
    lives in site-packages, so ``-S`` there would assert something false.
    """

    flags = ["-S"] if sys.version_info >= (3, 11) else []
    return subprocess.run(
        [sys.executable, *flags, *args],
        cwd=FLOOR_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
        env={"PATH": os.environ.get("PATH", "")},
    )


def test_tomllib_fallback_lives_only_in_the_shim() -> None:
    """No other module may import tomllib/tomli, directly or behind its own guard."""

    offenders: dict[str, list[int]] = {}
    for relative in _tracked_python_files():
        if relative == TOML_SHIM:
            continue
        text = (FLOOR_REPO_ROOT / relative).read_text(encoding="utf-8")
        hits = [
            number
            for number, line in enumerate(text.splitlines(), start=1)
            if line.split("#")[0].strip().split(" as ")[0]
            in {"import tomllib", "import tomli"}
        ]
        if hits:
            offenders[relative] = hits

    assert not offenders, (
        "these modules import tomllib/tomli directly and so cannot be imported on "
        f"the declared Python floor; use `{TOML_SHIM_IMPORT}` instead: {offenders}"
    )


def test_toml_shim_stays_cheap_to_import() -> None:
    """The uninstalled release gates reach the shim through ``src``, so keep it light.

    Importing it must not drag in the solver stack, and on 3.11+ it must need
    nothing outside the standard library at all.
    """

    probe = (
        "import sys; sys.path.insert(0, 'src');"
        " from gkx.utils import tomlcompat;"
        " assert tomlcompat.loads('a = 1') == {'a': 1};"
        " heavy = {'jax', 'jaxlib', 'numpy', 'scipy', 'matplotlib', 'netCDF4'}"
        " & set(sys.modules);"
        " assert not heavy, heavy"
    )
    proc = _uninstalled_run(["-c", probe])
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("relative", UNINSTALLED_RELEASE_GATES)
def test_repo_hygiene_gates_run_without_an_install(relative: str) -> None:
    """repo-hygiene runs these before any ``pip install``; keep that working."""

    proc = _uninstalled_run([relative, "--help"])
    assert proc.returncode == 0, proc.stderr


# ---- traced-constant hygiene gate ----


def _cached_factories_building_device_arrays(root: Path) -> list[str]:
    """Report ``lru_cache``d functions that build a ``jnp`` array in their body.

    A cached factory hands one object to every caller. An array built with
    ``jnp`` while filling that cache belongs to whichever trace happened to be
    active first, so the next trace that reuses the key gets a constant from a
    scope that has closed -- an ``UnexpectedTracerError`` with no line of its
    own to blame. This is the shape that hid under the compressed real-FFT
    projector until #61 made its index array host data.

    Arrays built inside a closure the factory *returns* are fine: those are
    constructed once per call of the closure, in the caller's own trace. The
    walk therefore stops at nested ``def``/``lambda`` bodies, which is what
    keeps the gate quiet enough to be worth having.
    """

    import ast

    def is_cache_decorator(node: ast.AST) -> bool:
        target = node.func if isinstance(node, ast.Call) else node
        if isinstance(target, ast.Attribute):
            return target.attr in {"lru_cache", "cache"}
        return isinstance(target, ast.Name) and target.id in {"lru_cache", "cache"}

    def own_body(fn: ast.AST):
        stack = list(fn.body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            yield node
            stack.extend(ast.iter_child_nodes(node))

    findings: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(is_cache_decorator(d) for d in fn.decorator_list):
                continue
            for node in own_body(fn):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "jnp"
                ):
                    findings.append(
                        f"{path.name}:{node.lineno} {fn.name} materializes "
                        f"jnp.{node.attr} while filling its cache"
                    )
    return findings


def test_no_cached_factory_materializes_a_device_array() -> None:
    """No cached factory leaks one trace's device constant into the next."""

    findings = _cached_factories_building_device_arrays(REPO_ROOT / "src" / "gkx")
    assert not findings, "\n".join(findings)


def test_cached_factory_gate_sees_the_shape_it_is_written_for(tmp_path) -> None:
    """The gate flags the escaping constant and spares the returned closure.

    A gate nobody can make fail is not a gate; a gate that fires on every
    cached factory mentioning ``jnp`` anywhere would be switched off within a
    week. Run the real scanner over both forms side by side so neither half of
    that claim can rot.
    """

    (tmp_path / "probe.py").write_text(
        "from functools import lru_cache\n"
        "import jax.numpy as jnp\n"
        "\n"
        "@lru_cache(maxsize=4)\n"
        "def escaping(n):\n"
        "    index = jnp.arange(n)\n"
        "    return lambda state: state[..., index]\n"
        "\n"
        "@lru_cache(maxsize=4)\n"
        "def contained(n):\n"
        "    def project(state):\n"
        "        return state + jnp.arange(n)\n"
        "    return project\n"
        "\n"
        "def uncached(n):\n"
        "    return jnp.arange(n)\n",
        encoding="utf-8",
    )
    findings = _cached_factories_building_device_arrays(tmp_path)
    assert len(findings) == 1, findings
    assert "escaping materializes jnp.arange" in findings[0]


# ---- [time] run_to audit for shipped nonlinear configs ----

"""Every shipped deck that reaches the nonlinear runtime declares its horizon.

``[time] run_to`` defaults to ``"saturation"``: a diagnosed nonlinear run ends
when its post-spin-up heat-flux window converges, with ``t_max`` only as a hard
cap. That default arrived after most of these decks were written, and it is not
neutral for a deck whose heat flux is not the observable -- a zero-gradient
zonal-response benchmark was silently truncated at ``t = 7.66`` of a requested
60, reporting ``gamma = NaN``, because a flux that never leaves zero satisfies
every convergence test built on it.

The tables below are the audit. A deck is either in ``_RUN_TO_REQUIRED``, and
must then carry that ``run_to`` in its own ``[time]`` block, or in
``_RUN_TO_DEFAULT_CLEARED``, whose value records the measured first-chunk stop
decision that cleared it. The gate's real work is the completeness check: a new
deck reaching the nonlinear runtime fails here until somebody measures it,
which is the step that was skipped when the default changed.

Measurement protocol for the cleared decks: run the deck's own configuration
through ``run_runtime_nonlinear`` for its first saturation chunk -- 512 steps or
the deck's whole budget, whichever is smaller, one diagnostic sample per step,
which is what ``_run_chunked_diagnostics`` does -- and evaluate
``saturation_stop_decision`` on the resulting trace. That reproduces the first
decision the chunk loop would make, which is where a premature stop lands.
"""

from support.paths import REPO_ROOT as RUN_TO_REPO_ROOT

# Decks that must pin run_to themselves, and the value they must pin.
_RUN_TO_REQUIRED = {
    "benchmarks/runtime_miller_zonal_response.toml": "t_max",
    "benchmarks/runtime_w7x_zonal_response_vmec.toml": "t_max",
    "benchmarks/runtime_secondary_slab.toml": "t_max",
}

# Decks cleared to run under the default, with the measurement that cleared
# them: the first-chunk stop decision, and the gate that refused it.
_RUN_TO_DEFAULT_CLEARED: dict[str, str] = {}

# Decks that reach the nonlinear runtime and have NOT been measured yet. This
# is debt, not clearance: the audit that produced the tables above was stopped
# partway, and every deck here is a turbulence case where saturation stopping
# is the intended behaviour, so the open question is only whether it fires
# early. Naming them keeps the completeness check doing its real work -- a
# deck that appears from now on still fails until somebody measures it -- while
# not claiming measurements nobody took. Empty this by moving each entry into
# _RUN_TO_DEFAULT_CLEARED with its first-chunk stop decision.
_RUN_TO_AUDIT_PENDING = {
    "examples/common_input.toml",
    "examples/nonlinear/axisymmetric/runtime_circular_vmec_nonlinear.toml",
    "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml",
    "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml",
    "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_short.toml",
    "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_t400.toml",
    "examples/nonlinear/axisymmetric/runtime_etg_nonlinear.toml",
    "examples/nonlinear/axisymmetric/runtime_kbm_nonlinear.toml",
    "examples/nonlinear/axisymmetric/runtime_kbm_nonlinear_seed.toml",
    "examples/nonlinear/axisymmetric/runtime_kbm_nonlinear_short.toml",
    "examples/nonlinear/axisymmetric/runtime_kbm_nonlinear_short_lockin.toml",
    "examples/nonlinear/axisymmetric/runtime_kbm_nonlinear_t100.toml",
    "examples/nonlinear/axisymmetric/runtime_kbm_nonlinear_t100_nx4ny8_dt9e4.toml",
    "examples/nonlinear/non-axisymmetric/runtime_hsx_nonlinear_vmec_geometry.toml",
    "examples/nonlinear/non-axisymmetric/runtime_w7x_nonlinear_imported_geometry.toml",
    "examples/nonlinear/non-axisymmetric/runtime_w7x_nonlinear_vmec_geometry.toml",
}

# Decks that ship as linear but are promoted into the nonlinear runtime by a
# shipped driver, so their [time] block still sets a nonlinear stop policy.
_RUN_TO_PROMOTED = {
    "benchmarks/runtime_secondary_slab.toml": (
        "benchmarks/secondary_slab_workflow.py, via build_secondary_stage2_config"
    ),
}

# Files under the audited roots that are not GKX runtime decks at all. Listed
# rather than skipped: a deck that stops parsing would otherwise leave the
# audit silently by the same door.
_NOT_A_GKX_RUNTIME_DECK = {
    "examples/nonlinear/non-axisymmetric/reference_hsx_nonlinear_adiabatic_electrons.toml": (
        "reference-code input deck, [Dimensions]/[Physics] schema, not a GKX runtime TOML"
    ),
}


def _shipped_toml_paths() -> list[str]:
    """Repo-relative paths of every TOML under the audited roots."""

    paths: list[str] = []
    for root in ("benchmarks", "examples"):
        for path in sorted((RUN_TO_REPO_ROOT / root).rglob("*.toml")):
            paths.append(str(path.relative_to(RUN_TO_REPO_ROOT)))
    return paths


def _decks_reaching_the_nonlinear_runtime() -> tuple[set[str], set[str]]:
    """Split the shipped decks into (reaches nonlinear runtime, unparseable).

    A deck reaches the nonlinear runtime when it turns the nonlinear term on or
    when it is not a linear run -- the second half is what catches the
    initial-value relaxation benchmarks, which carry ``nonlinear = false`` and
    still go through ``run_runtime_nonlinear`` because that is the only route
    with streamed diagnostics. Decks a shipped driver promotes are added by
    name, since nothing in their own contents says so.
    """

    from gkx.workflows.runtime.toml import load_runtime_from_toml

    reaches: set[str] = set()
    unparseable: set[str] = set()
    for relative in _shipped_toml_paths():
        try:
            cfg, _raw = load_runtime_from_toml(RUN_TO_REPO_ROOT / relative)
        except Exception:
            unparseable.add(relative)
            continue
        if bool(cfg.physics.nonlinear) or not bool(cfg.physics.linear):
            reaches.add(relative)
    return reaches | set(_RUN_TO_PROMOTED), unparseable


def _deck_run_to(relative: str) -> str | None:
    """The ``[time] run_to`` the deck itself carries, or None if it leaves it."""

    from gkx.utils import tomlcompat as tomllib

    data = tomllib.loads((RUN_TO_REPO_ROOT / relative).read_text(encoding="utf-8"))
    value = data.get("time", {}).get("run_to")
    return None if value is None else str(value).strip().lower()


def test_every_shipped_nonlinear_deck_is_covered_by_the_run_to_audit() -> None:
    """No deck reaches the nonlinear runtime without an audited stop policy."""

    reaches, unparseable = _decks_reaching_the_nonlinear_runtime()
    audited = (
        set(_RUN_TO_REQUIRED) | set(_RUN_TO_DEFAULT_CLEARED) | _RUN_TO_AUDIT_PENDING
    )

    assert unparseable == set(_NOT_A_GKX_RUNTIME_DECK), (
        "a shipped TOML changed parseability; audit it or record why it is not "
        f"a GKX runtime deck: {sorted(unparseable ^ set(_NOT_A_GKX_RUNTIME_DECK))}"
    )
    assert reaches - audited == set(), (
        "these decks reach the nonlinear runtime and are not in the run_to "
        "audit. Measure the first-chunk saturation decision (see this module's "
        "protocol) and add them to _RUN_TO_REQUIRED or "
        f"_RUN_TO_DEFAULT_CLEARED: {sorted(reaches - audited)}"
    )
    assert audited - reaches == set(), (
        "these decks are audited but no longer reach the nonlinear runtime; "
        f"drop them from the audit: {sorted(audited - reaches)}"
    )


def test_decks_that_must_pin_run_to_pin_it_in_their_own_time_block() -> None:
    """A required stop policy has to be in the deck, not in a caller."""

    for relative, expected in sorted(_RUN_TO_REQUIRED.items()):
        assert _deck_run_to(relative) == expected, (
            f"{relative} must set [time] run_to = \"{expected}\"; its heat flux "
            "is not the observable it exists to produce, so the default "
            "saturation stop would end it on a quantity nobody asked for"
        )


def test_decks_cleared_for_the_default_carry_their_measurement() -> None:
    """A deck cleared to run under the default records what cleared it.

    The evidence has to name the gate that refused the first-chunk stop, in the
    vocabulary ``saturation_stop_decision`` reports, so that "measured, looked
    fine" cannot pass for a measurement.
    """

    refusal_gates = {
        "flux_indistinguishable_from_zero",
        "tau_ac_unresolved",
        "window_below_min_window",
        "rel_sem_above_threshold",
        "window_not_stationary",
        "guard_not_stationary",
        "trace_shorter_than_min_samples",
        "post_spinup_window_too_short",
    }
    for relative, evidence in sorted(_RUN_TO_DEFAULT_CLEARED.items()):
        assert _deck_run_to(relative) is None, (
            f"{relative} now pins run_to; move it to _RUN_TO_REQUIRED"
        )
        named = refusal_gates.intersection(evidence.split())
        assert named, (
            f"{relative} is cleared without naming the gate that refused its "
            f"first-chunk stop; one of {sorted(refusal_gates)} must appear"
        )


def test_run_to_audit_discovery_sees_a_deck_that_only_the_flags_reveal() -> None:
    """The classifier catches the relaxation shape, not just nonlinear = true.

    The zonal-response benchmarks are the reason this gate exists and they ship
    ``nonlinear = false``. A classifier keyed on that flag alone would have
    passed every one of them.
    """

    reaches, _ = _decks_reaching_the_nonlinear_runtime()

    assert "benchmarks/runtime_miller_zonal_response.toml" in reaches
    assert "benchmarks/runtime_w7x_zonal_response_vmec.toml" in reaches
    # Promoted by a driver: nothing in the deck itself says nonlinear.
    assert "benchmarks/runtime_secondary_slab.toml" in reaches
    # A genuinely linear deck stays out, or the audit becomes noise.
    assert "examples/linear/axisymmetric/cyclone.toml" not in reaches
    assert "benchmarks/collisional_zonal_response.toml" not in reaches


# ---- GX parity: measured build-reproducibility floors ----

"""A parity case may not be gated below the spread of its own reference.

The linear parity matrix reports differences against reference-code output. For
one case, ``kbm_miller``, that reference moves by 0.16-0.20 percent between two
legitimate builds of the same reference commit -- the only difference being
``-prec-sqrt=true``, IEEE-correct single-precision ``sqrtf`` -- while the
run-to-run noise floor is exactly zero, the same binary reproducing its output
bitwise. Every other case in the matrix is bit-identical between those builds
or moves by at most 0.07 percent.

So a sub-0.1-percent gate on that case would be gating the compiler, and the
manifest records the resolution of the instrument as
``build_reproducibility_floor``. This gate keeps that number from being quietly
tightened back below what was measured, and keeps floors from being invented
for cases where nothing was measured.
"""

_PARITY_MANIFEST = RUN_TO_REPO_ROOT / "tools" / "gx_parity_matrix_manifest.toml"
_PARITY_BUILDER = (
    RUN_TO_REPO_ROOT / "tools" / "comparison" / "build_gx_parity_matrix.py"
)
# Largest relative move measured between the two builds: 0.204% in omega at
# ky = 0.2. A floor at or below that would not cover the measurement it exists
# to record. See the branch-only roadmap's gx_precsqrt note.
_KBM_TWO_BUILD_SPREAD = 0.00204


def _parity_cases() -> list[dict]:
    from gkx.utils import tomlcompat as tomllib

    return tomllib.loads(_PARITY_MANIFEST.read_text(encoding="utf-8"))["case"]


def test_kbm_miller_parity_floor_covers_its_measured_two_build_spread() -> None:
    """The one case with a measured floor keeps a floor that covers it."""

    cases = {case["key"]: case for case in _parity_cases()}
    floor = cases["kbm_miller"].get("build_reproducibility_floor")

    assert floor is not None, (
        "kbm_miller must declare build_reproducibility_floor; without it the "
        "reported differences invite a gate tighter than the reference's own "
        "build-to-build spread"
    )
    assert float(floor) > _KBM_TWO_BUILD_SPREAD, (
        f"kbm_miller floor {floor} does not cover the measured two-build "
        f"spread {_KBM_TWO_BUILD_SPREAD}; that spread is compiler arithmetic, "
        "not physics, so a gate under it cannot be met by any build choice"
    )


def test_parity_floors_are_declared_only_where_they_were_measured() -> None:
    """A floor is a measurement, so it needs a reason beside it."""

    cases = _parity_cases()
    declared = {
        case["key"]
        for case in cases
        if case.get("build_reproducibility_floor") is not None
    }

    assert declared == {"kbm_miller"}, (
        "build_reproducibility_floor is a claim that a case's reference moves "
        "between builds. Add one only with the measurement that shows it, and "
        f"the comment recording that measurement: {sorted(declared)}"
    )
    manifest_text = _PARITY_MANIFEST.read_text(encoding="utf-8")
    reason_block = manifest_text.split("build_reproducibility_floor = ")[0]
    assert "prec-sqrt" in reason_block, (
        "the reason for the kbm_miller floor must stay beside the number"
    )


def test_parity_builder_reads_the_declared_floor() -> None:
    """A manifest key nothing reads is documentation pretending to be a gate."""

    source = _PARITY_BUILDER.read_text(encoding="utf-8")

    assert 'case.get("build_reproducibility_floor")' in source
    assert '"build_reproducibility_floor": floor' in source
    assert '"within_build_reproducibility_floor"' in source
