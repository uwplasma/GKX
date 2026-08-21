"""Contracts for grouped nonlinear transport release gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from support.paths import REPO_ROOT, load_release_tool


ROOT = REPO_ROOT
SCRIPT = ROOT / "tools" / "release" / "check_nonlinear_transport_gates.py"
mod = load_release_tool("check_nonlinear_transport_gates")


def _write_manifest(
    tmp_path: Path, outputs: list[Path], *, include_dt: bool = False
) -> Path:
    config = {"window": {"tmin": 10.0, "tmax": 20.0}}
    if include_dt:
        config.update({"dt": 0.05, "dt_variants": [0.04]})
    manifest = {
        "kind": "matched_nonlinear_transport_matrix_campaign",
        "config": config,
        "samples": [
            {
                "sample_id": "s0p45_a0_ky0p1",
                "surface_torflux": 0.45,
                "alpha": 0.0,
                "ky": 0.1,
                "states": {
                    "baseline": {"label": "base", "final_outputs": [str(outputs[0])]},
                    "candidate": {"label": "cand", "final_outputs": [str(outputs[1])]},
                },
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _touch_bundle(output: Path) -> None:
    stem = (
        output.name[: -len(".out.nc")]
        if output.name.endswith(".out.nc")
        else output.stem
    )
    base = output.with_name(stem)
    for suffix in ("out.nc", "restart.nc", "big.nc"):
        Path(f"{base}.{suffix}").write_text("stub\n", encoding="utf-8")


def test_progress_requires_target_time_even_when_bundle_exists(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "base.out.nc"
    cand = tmp_path / "cand.out.nc"
    _touch_bundle(base)
    _touch_bundle(cand)
    manifest = _write_manifest(tmp_path, [base, cand])
    monkeypatch.setattr(mod, "_read_output_tmax", lambda _path: 19.0)

    report = mod.build_matrix_progress_report(matrix_manifest=manifest)

    assert report["summary"]["expected_outputs"] == 2
    assert report["summary"]["complete_bundles"] == 2
    assert report["summary"]["target_time_confirmed"] == 0
    assert report["summary"]["ready_for_postprocess"] is False


def test_progress_passes_when_all_bundles_reach_target_time(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "base.out.nc"
    cand = tmp_path / "cand.out.nc"
    _touch_bundle(base)
    _touch_bundle(cand)
    manifest = _write_manifest(tmp_path, [base, cand])
    monkeypatch.setattr(mod, "_read_output_tmax", lambda _path: 20.0)

    report = mod.build_matrix_progress_report(matrix_manifest=manifest)

    assert report["summary"]["complete_bundles"] == 2
    assert report["summary"]["target_time_confirmed"] == 2
    assert report["summary"]["ready_for_postprocess"] is True
    assert all(row["bundle_complete"] for row in report["rows"])
    assert all(row["target_time_confirmed"] for row in report["rows"])


def test_progress_accepts_fixed_step_output_within_manifest_dt_tolerance(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "base.out.nc"
    cand = tmp_path / "cand.out.nc"
    _touch_bundle(base)
    _touch_bundle(cand)
    manifest = _write_manifest(tmp_path, [base, cand], include_dt=True)
    monkeypatch.setattr(mod, "_read_output_tmax", lambda _path: 19.927)

    report = mod.build_matrix_progress_report(matrix_manifest=manifest)

    assert report["time_tolerance"] == 0.1
    assert report["summary"]["target_time_confirmed"] == 2
    assert report["summary"]["ready_for_postprocess"] is True


def test_progress_keeps_checkpoint_below_dt_tolerance_incomplete(
    tmp_path: Path, monkeypatch
) -> None:
    base = tmp_path / "base.out.nc"
    cand = tmp_path / "cand.out.nc"
    _touch_bundle(base)
    _touch_bundle(cand)
    manifest = _write_manifest(tmp_path, [base, cand], include_dt=True)
    monkeypatch.setattr(mod, "_read_output_tmax", lambda _path: 19.85)

    report = mod.build_matrix_progress_report(matrix_manifest=manifest)

    assert report["time_tolerance"] == 0.1
    assert report["summary"]["target_time_confirmed"] == 0
    assert report["summary"]["ready_for_postprocess"] is False


def test_progress_cli_uses_manifest_dt_tolerance_by_default(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    base = tmp_path / "base.out.nc"
    cand = tmp_path / "cand.out.nc"
    _touch_bundle(base)
    _touch_bundle(cand)
    manifest = _write_manifest(tmp_path, [base, cand], include_dt=True)
    out_json = tmp_path / "progress.json"
    monkeypatch.setattr(mod, "_read_output_tmax", lambda _path: 19.927)

    rc = mod.main(["matrix-progress", "--matrix-manifest", str(manifest), "--out-json", str(out_json)])
    stdout = capsys.readouterr().out
    report = json.loads(out_json.read_text(encoding="utf-8"))

    assert rc == 0
    assert '"ready_for_postprocess": true' in stdout.lower()
    assert report["time_tolerance"] == 0.1
    assert report["summary"]["target_time_confirmed"] == 2


def test_skip_time_check_does_not_read_output_time(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "base.out.nc"
    cand = tmp_path / "cand.out.nc"
    _touch_bundle(base)
    _touch_bundle(cand)
    manifest = _write_manifest(tmp_path, [base, cand], include_dt=True)

    def fail_if_called(_path):
        raise AssertionError("skip_time_check should not read NetCDF times")

    monkeypatch.setattr(mod, "_read_output_tmax", fail_if_called)
    report = mod.build_matrix_progress_report(matrix_manifest=manifest, skip_time_check=True)

    assert report["skip_time_check"] is True
    assert report["summary"]["complete_bundles"] == 2
    assert report["summary"]["target_time_confirmed"] == 0
    assert report["summary"]["ready_for_postprocess"] is False
    assert report["summary"]["time_check_skipped"] is True
    assert all(not row["target_time_confirmed"] for row in report["rows"])
    assert all(row["output_tmax"] is None for row in report["rows"])


def _write_nonlinear_output(
    path: Path,
    *,
    time: np.ndarray | None = None,
    heat: np.ndarray | None = None,
    include_heat: bool = True,
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    t = np.asarray([0.0, 5.0, 10.0] if time is None else time, dtype=float)
    q = np.asarray([1.0, 2.0, 3.0] if heat is None else heat, dtype=float)
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("time", t.size)
        root.createDimension("s", 1)
        grids = root.createGroup("Grids")
        grids.createVariable("time", "f8", ("time",))[:] = t
        diagnostics = root.createGroup("Diagnostics")
        if include_heat:
            diagnostics.createVariable("HeatFlux_st", "f8", ("time", "s"))[:, :] = q[
                :, None
            ]


def test_validate_output_accepts_grouped_runtime_netcdf(tmp_path: Path) -> None:
    out = tmp_path / "run.out.nc"
    _write_nonlinear_output(out)

    row = mod.validate_output(
        out, min_samples=3, tmin=5.0, tmax=10.0, min_window_samples=2
    )

    assert row["passed"] is True
    assert row["samples"] == 3
    assert row["window"]["mean_heat_flux"] == pytest.approx(2.5)


def test_validate_output_fails_closed_for_missing_diagnostics(tmp_path: Path) -> None:
    out = tmp_path / "restart_like.out.nc"
    _write_nonlinear_output(out, include_heat=False)

    row = mod.validate_output(out, min_samples=2)

    assert row["passed"] is False
    assert any("HeatFlux_st" in failure for failure in row["failures"])


def test_check_outputs_reports_required_window_failures(tmp_path: Path) -> None:
    out = tmp_path / "short.out.nc"
    _write_nonlinear_output(out, time=np.asarray([0.0, 5.0, 10.0]), heat=np.ones(3))

    payload = mod.check_outputs(
        [out],
        heat_flux_variable=mod.DEFAULT_HEAT_FLUX_VARIABLE,
        min_samples=2,
        tmin=20.0,
        tmax=30.0,
        tmax_atol=None,
        min_window_samples=2,
        min_abs_window_mean=None,
    )

    assert payload["passed"] is False
    assert payload["summary"] == {"outputs": 1, "passed": 0, "failed": 1}
    assert {
        "does_not_reach_required_tmax",
        "too_few_window_samples",
    }.issubset(set(payload["rows"][0]["failures"]))


def test_validate_output_tolerates_fixed_step_time_roundoff(tmp_path: Path) -> None:
    out = tmp_path / "rounded.out.nc"
    _write_nonlinear_output(
        out,
        time=np.asarray([896.0, 898.0, 899.99]),
        heat=np.asarray([2.0, 2.1, 2.2]),
    )

    row = mod.validate_output(out, min_samples=3, tmin=896.0, tmax=900.0)

    assert row["passed"] is True
    assert row["tmax_atol"] == pytest.approx(0.25 * np.median([2.0, 1.99]))


def test_validate_output_can_enforce_strict_tmax_tolerance(tmp_path: Path) -> None:
    out = tmp_path / "rounded.out.nc"
    _write_nonlinear_output(out, time=np.asarray([0.0, 2.0, 899.99]), heat=np.ones(3))

    row = mod.validate_output(out, min_samples=3, tmax=900.0, tmax_atol=1.0e-4)

    assert row["passed"] is False
    assert "does_not_reach_required_tmax" in row["failures"]


def test_main_writes_json_and_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run.out.nc"
    report = tmp_path / "report.json"
    _write_nonlinear_output(out)

    rc = mod.main(["runtime-outputs", str(out), "--json-out", str(report), "--min-samples", "3"])

    assert rc == 0
    assert report.exists()
    assert '"failed": 0' in capsys.readouterr().out


def _matrix_report(
    path: Path,
    *,
    passed: bool,
    total: int = 18,
    completed: int = 18,
    passed_samples: int = 18,
    mean_reduction: float = 0.03,
) -> Path:
    pass_fraction = passed_samples / total if total else 0.0
    payload = {
        "kind": "matched_nonlinear_transport_matrix_report",
        "passed": passed,
        "summary": {
            "total_samples": total,
            "completed_samples": completed,
            "passed_samples": passed_samples,
            "pass_fraction": pass_fraction,
            "mean_relative_reduction": mean_reduction,
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _excluded_comparison(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "matched_nonlinear_transport_comparison",
                "passed": False,
                "statistics": {
                    "relative_reduction": -0.004,
                    "uncertainty_z_score": -0.2,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_portfolio_selects_best_passing_broad_matrix(tmp_path: Path) -> None:
    accepted = _matrix_report(
        tmp_path / "accepted.json", passed=True, mean_reduction=0.025
    )
    projected = _matrix_report(
        tmp_path / "projected.json", passed=True, mean_reduction=0.04
    )
    strict = _excluded_comparison(tmp_path / "strict_growth.json")

    report = mod.build_transport_matrix_portfolio_report(
        matrix_reports={"accepted_qa_ess": accepted, "projected_0p001": projected},
        excluded_comparisons={"strict_growth": strict},
    )

    assert report["passed"] is True
    assert report["selected_family"] == "projected_0p001"
    assert report["selected_report"]["summary"]["mean_relative_reduction"] == 0.04
    assert report["excluded_comparisons"][0]["label"] == "strict_growth"
    assert "excluded" in report["excluded_comparisons"][0]["note"]


def test_portfolio_blocks_missing_or_failed_broad_matrices(tmp_path: Path) -> None:
    failed = _matrix_report(
        tmp_path / "failed.json",
        passed=False,
        passed_samples=12,
        mean_reduction=0.01,
    )

    report = mod.build_transport_matrix_portfolio_report(
        matrix_reports={
            "accepted_qa_ess": failed,
            "projected_0p001": tmp_path / "missing.json",
        },
        excluded_comparisons={},
    )

    assert report["passed"] is False
    assert report["selected_family"] is None
    assert "no candidate family passed" in report["blockers"][0]
    rows = {row["label"]: row for row in report["matrix_reports"]}
    assert rows["accepted_qa_ess"]["qualifies_for_broad_promotion"] is False
    assert rows["projected_0p001"]["exists"] is False


def test_portfolio_cli_writes_report_and_figure(tmp_path: Path) -> None:
    accepted = _matrix_report(
        tmp_path / "accepted.json", passed=True, mean_reduction=0.025
    )
    out_json = tmp_path / "portfolio.json"
    out_png = tmp_path / "portfolio.png"

    rc = mod.main(
        [
            "matrix-portfolio",
            "--matrix-report",
            f"accepted_qa_ess={accepted}",
            "--out-json",
            str(out_json),
            "--out-figure",
            str(out_png),
            "--fail-on-blocked",
        ]
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["passed"] is True
    assert payload["selected_family"] == "accepted_qa_ess"
    assert out_png.exists()
