"""Contracts for nonlinear transport-window artifact and gate utilities."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from support.paths import REPO_ROOT, load_release_tool
from gkx.diagnostics.transport_windows import (
    NonlinearWindowConvergenceConfig,
    nonlinear_window_convergence_report,
)
from gkx.diagnostics.validation_gates import matched_nonlinear_transport_report

ROOT = REPO_ROOT
OUTPUT_TARGET_SCRIPT = ROOT / "tools" / "release" / "check_nonlinear_transport_gates.py"
output_target = load_release_tool("check_nonlinear_transport_gates")
window_ensemble = load_release_tool("check_nonlinear_transport_gates")
window_readiness = window_ensemble
FLOW_SHEAR_GATE = ROOT / "docs" / "_static" / "flow_shear_fixed_step_response_gate.json"


def _touch_bundle(output: Path) -> None:
    stem = (
        output.name[: -len(".out.nc")]
        if output.name.endswith(".out.nc")
        else output.stem
    )
    base = output.with_name(stem)
    for suffix in ("out.nc", "restart.nc", "big.nc"):
        Path(f"{base}.{suffix}").write_text("stub\n", encoding="utf-8")


def test_output_target_checker_accepts_near_horizon_and_rejects_partial_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "run.out.nc"
    _touch_bundle(output)

    monkeypatch.setattr(output_target, "_read_output_tmax", lambda _path: 1499.927)
    accepted = output_target.build_target_time_report(
        output=output, target_time=1500.0, time_tolerance=0.1
    )
    assert accepted["bundle_complete"] is True
    assert accepted["target_time_confirmed"] is True

    monkeypatch.setattr(output_target, "_read_output_tmax", lambda _path: 400.0)
    rejected = output_target.build_target_time_report(
        output=output, target_time=1500.0, time_tolerance=0.1
    )
    assert rejected["bundle_complete"] is True
    assert rejected["target_time_confirmed"] is False


def test_output_target_checker_cli_and_direct_help_contracts(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "run.out.nc"
    _touch_bundle(output)
    monkeypatch.setattr(output_target, "_read_output_tmax", lambda _path: 19.95)

    assert (
        output_target.main(
            [
                "target-time",
                "--output",
                str(output),
                "--target-time",
                "20",
                "--time-tolerance",
                "0.1",
                "--quiet",
            ]
        )
        == 0
    )
    assert (
        output_target.main(
            [
                "target-time",
                "--output",
                str(output),
                "--target-time",
                "20",
                "--time-tolerance",
                "0.01",
                "--quiet",
            ]
        )
        == 1
    )

    result = subprocess.run(
        [sys.executable, str(OUTPUT_TARGET_SCRIPT), "target-time", "--help"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "--target-time" in result.stdout


def _window_report(offset: float, *, case: str) -> dict[str, object]:
    t = np.linspace(0.0, 200.0, 201)
    heat = 4.0 + offset + 0.04 * np.sin(2.0 * np.pi * t / 10.0)
    return nonlinear_window_convergence_report(
        t,
        heat,
        case=case,
        source_artifact=f"{case}.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
            min_blocks=4,
            max_running_mean_rel_drift=0.02,
            max_sem_rel=0.02,
        ),
    )


def test_matched_transport_requires_converged_windows_and_resolved_reduction() -> None:
    baseline = _window_report(0.5, case="baseline")
    treatment = _window_report(0.0, case="flow_shear")
    report = matched_nonlinear_transport_report(
        baseline,
        treatment,
        case="flow_shear_transport",
        treatment_name="gamma_e_0p01",
        min_relative_reduction=0.05,
        min_uncertainty_z_score=2.0,
    )
    assert report["passed"] is True
    assert report["statistics"]["relative_reduction"] > 0.1
    assert report["statistics"]["uncertainty_z_score"] > 2.0

    drifting_t = np.linspace(0.0, 200.0, 201)
    drifting_heat = 4.0 + 0.01 * drifting_t
    drifting = nonlinear_window_convergence_report(
        drifting_t,
        drifting_heat,
        case="drifting_treatment",
        source_artifact="drifting_treatment.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
            min_blocks=4,
            max_running_mean_rel_drift=0.02,
            max_terminal_mean_rel_delta=0.02,
        ),
    )
    rejected = matched_nonlinear_transport_report(baseline, drifting)
    assert rejected["passed"] is False
    assert rejected["windows_ready"] is False
    failed = {gate["metric"] for gate in rejected["gates"] if not gate["passed"]}
    assert "treatment_window_passed" in failed


def test_matched_transport_cli_writes_fail_closed_report(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    treatment = tmp_path / "treatment.json"
    baseline.write_text(
        json.dumps(_window_report(0.5, case="baseline")), encoding="utf-8"
    )
    treatment.write_text(
        json.dumps(_window_report(0.0, case="flow_shear")), encoding="utf-8"
    )
    output = tmp_path / "matched.json"

    rc = window_ensemble.main(
        [
            "matched-windows",
            "--baseline",
            str(baseline),
            "--treatment",
            str(treatment),
            "--out-json",
            str(output),
            "--min-relative-reduction",
            "0.05",
            "--min-uncertainty-z-score",
            "2.0",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["kind"] == "matched_nonlinear_transport_comparison"
    assert payload["passed"] is True


def test_fixed_step_flow_shear_artifact_preserves_negative_evidence() -> None:
    payload = json.loads(FLOW_SHEAR_GATE.read_text(encoding="utf-8"))

    assert payload["passed"] is False
    assert payload["conclusion"]["input_file_exposure_allowed"] is False
    assert payload["configuration"]["time"]["analysis_window"] == [240.0, 300.0]

    internal = payload["gkx_gk"]
    comparison = payload["comparison"]
    assert internal["baseline_window"]["passed"] is False
    assert internal["treatment_window"]["passed"] is False
    assert internal["matched"]["statistics"]["relative_reduction"] < 0.0
    assert comparison["baseline_window"]["passed"] is True
    assert comparison["treatment_window"]["passed"] is True
    assert comparison["matched"]["statistics"]["relative_reduction"] < -0.20
    assert comparison["matched"]["statistics"]["uncertainty_z_score"] < -2.0


def test_nonlinear_window_ensemble_tool_writes_json_png_and_fails_closed(
    tmp_path: Path,
) -> None:
    reports = []
    for idx, offset in enumerate((-0.02, 0.0, 0.02)):
        path = tmp_path / f"seed_{idx}.json"
        path.write_text(
            json.dumps(_window_report(offset, case=f"seed_{idx}")), encoding="utf-8"
        )
        reports.append(path)

    out_json = tmp_path / "ensemble.json"
    out_png = tmp_path / "ensemble.png"
    rc = window_ensemble.main(
        [
            "ensemble",
            *[str(path) for path in reports],
            "--out-json",
            str(out_json),
            "--out-png",
            str(out_png),
            "--case",
            "seed_replicates",
            "--comparison",
            "random_seed_replicates",
            "--min-reports",
            "3",
            "--max-mean-rel-spread",
            "0.02",
            "--max-combined-sem-rel",
            "0.02",
        ]
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 0
    assert out_png.exists()
    assert payload["passed"] is True
    assert payload["comparison"] == "random_seed_replicates"
    assert payload["statistics"]["n_reports"] == 3

    paths = []
    for idx, offset in enumerate((0.0, 2.0)):
        path = tmp_path / f"dt_{idx}.json"
        path.write_text(
            json.dumps(_window_report(offset, case=f"dt_{idx}")), encoding="utf-8"
        )
        paths.append(path)

    failed_json = tmp_path / "ensemble_failed.json"
    rc = window_ensemble.main(
        [
            "ensemble",
            *[str(path) for path in paths],
            "--out-json",
            str(failed_json),
            "--max-mean-rel-spread",
            "0.05",
        ]
    )
    failed_payload = json.loads(failed_json.read_text(encoding="utf-8"))
    failed = {gate["metric"] for gate in failed_payload["gates"] if not gate["passed"]}
    assert rc == 1
    assert "mean_relative_spread" in failed


def _write_trace(path: Path, offset: float = 0.0) -> None:
    t = np.linspace(0.0, 100.0, 101)
    heat = 5.0 + offset + 0.02 * np.sin(2.0 * np.pi * t / 10.0)
    lines = ["t,heat_flux"]
    lines.extend(f"{time:.12g},{value:.12g}" for time, value in zip(t, heat))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(
    path: Path,
    trace: Path,
    *,
    case: str = "case_a",
    seed: int | None = None,
    timestep: float | None = None,
) -> Path:
    payload: dict[str, object] = {
        "kind": "nonlinear_window_summary",
        "case": case,
        "gkx": trace.name,
        "tmin": 50.0,
        "tmax": 100.0,
        "promotion_gate": {"passed": True},
    }
    if seed is not None:
        payload["seed"] = seed
    if timestep is not None:
        payload["dt"] = timestep
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_readiness_tool_writes_reports_and_requires_seed_timestep_replicates(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.csv"
    _write_trace(trace)
    summary = _write_summary(tmp_path / "summary.json", trace)
    out_json = tmp_path / "manifest.json"
    reports_dir = tmp_path / "reports"

    rc = window_readiness.main(
        [
            "readiness",
            str(summary),
            "--out-json",
            str(out_json),
            "--reports-dir",
            str(reports_dir),
        ]
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["passed"] is False
    assert (reports_dir / "summary.convergence.json").exists()
    assert payload["observed_artifacts"][0]["promotion_ready"] is True
    missing_axes = {item["variant_axis"] for item in payload["missing_artifacts"]}
    assert missing_axes == {"seed", "timestep"}
    assert all(item["missing_count"] == 2 for item in payload["missing_artifacts"])

    summaries = []
    for idx, (seed, timestep, offset) in enumerate(
        ((11, 0.02, -0.01), (22, 0.01, 0.01))
    ):
        trace = tmp_path / f"trace_{idx}.csv"
        _write_trace(trace, offset=offset)
        summaries.append(
            _write_summary(
                tmp_path / f"summary_{idx}.json",
                trace,
                seed=seed,
                timestep=timestep,
            )
        )
    passed_json = tmp_path / "manifest_passed.json"
    rc = window_readiness.main(
        [
            "readiness",
            *[str(path) for path in summaries],
            "--out-json",
            str(passed_json),
        ]
    )
    passed_payload = json.loads(passed_json.read_text(encoding="utf-8"))
    assert rc == 0
    assert passed_payload["passed"] is True
    assert passed_payload["missing_artifacts"] == []
    assert (
        passed_payload["cases"][0]["variant_axes"]["seed"]["observed_distinct_count"]
        == 2
    )
    assert (
        passed_payload["cases"][0]["variant_axes"]["timestep"][
            "observed_distinct_count"
        ]
        == 2
    )
