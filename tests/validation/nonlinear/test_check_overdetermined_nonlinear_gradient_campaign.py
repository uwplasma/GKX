from __future__ import annotations

import json
from pathlib import Path

from support.paths import REPO_ROOT, load_release_tool

import pytest


ROOT = REPO_ROOT
SCRIPT = (
    ROOT / "tools" / "release" / "check_nonlinear_optimization_gates.py"
)


def _load_tool_module():
    return load_release_tool("check_nonlinear_optimization_gates")


def _write_runtime_output(path: Path, *, time_max: float) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    path.parent.mkdir(parents=True, exist_ok=True)
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("time", 2)
        grids = root.createGroup("Grids")
        grids.createDimension("time", 2)
        times = grids.createVariable("time", "f8", ("time",))
        times[:] = [0.0, time_max]


def _manifest(
    tmp_path: Path,
    *,
    with_nested: bool = False,
    with_runtime: bool = False,
    fd_passed: bool = False,
    required_tmax: float | None = None,
    runtime_tmax: float | None = None,
) -> dict[str, object]:
    work = tmp_path / "work"
    control_dir = work / "zbs_1_0"
    control_dir.mkdir(parents=True)
    inputs = {}
    wouts = {}
    for state in ("baseline", "plus_delta", "minus_delta"):
        input_path = control_dir / f"input.{state}"
        wout_path = control_dir / f"wout_{state}.nc"
        input_path.write_text(f"{state}\n", encoding="utf-8")
        wout_path.write_bytes(f"{state}-wout".encode())
        inputs[state] = str(input_path)
        wouts[state] = str(wout_path)

    nested_manifest = (
        tmp_path / "nonlinear" / "zbs_1_0" / "gradient_campaign_manifest.json"
    )
    outputs = [
        tmp_path / "nonlinear" / "out" / f"{state}.out.nc"
        for state in ("baseline", "plus_delta", "minus_delta")
    ]
    if with_nested:
        nested_manifest.parent.mkdir(parents=True)
        nested_manifest.write_text(
            json.dumps(
                {
                    "kind": "nonlinear_turbulence_gradient_campaign_manifest",
                    "state_ensemble_commands": {
                        state: {"expected_outputs": [str(outputs[index])]}
                        for index, state in enumerate(
                            ("baseline", "plus_delta", "minus_delta")
                        )
                    },
                }
            ),
            encoding="utf-8",
        )
    if with_runtime:
        for output in outputs:
            if runtime_tmax is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"netcdf")
            else:
                _write_runtime_output(output, time_max=runtime_tmax)

    fd = tmp_path / "fd.json"
    fd.write_text(
        json.dumps(
            {
                "kind": "nonlinear_turbulence_gradient_central_fd_gate",
                "passed": fd_passed,
                "blockers": [] if fd_passed else ["gradient_uncertainty_bounded"],
                "metrics": {"gradient_uncertainty_rel": 0.2 if fd_passed else 0.9},
            }
        ),
        encoding="utf-8",
    )
    ranking = tmp_path / "ranking.json"
    ranking.write_text(
        json.dumps(
            {
                "passed": fd_passed,
                "recommendation": "promote" if fd_passed else "blocked",
                "best_candidate": {"label": "zbs_1_0"} if fd_passed else None,
            }
        ),
        encoding="utf-8",
    )
    manifest: dict[str, object] = {
        "kind": "overdetermined_nonlinear_turbulence_gradient_campaign_manifest",
        "case": "qa_profile",
        "controls": [
            {
                "coefficient": "ZBS(1,0)",
                "coefficient_slug": "zbs_1_0",
                "case": "qa_profile_zbs_1_0",
                "state_input_files": inputs,
                "expected_wout_files": wouts,
                "expected_nonlinear_campaign_manifest": str(nested_manifest),
                "expected_fd_artifact": str(fd),
                "vmec_run_commands": {"baseline": "vmex input.baseline"},
                "nonlinear_campaign_command_after_vmec_runs": (
                    f"python tools/campaigns/write_nonlinear_turbulence_gradient_campaign.py --out-dir {nested_manifest.parent}"
                ),
            }
        ],
        "promotion_contract": {"candidate_ranking_json": str(ranking)},
    }
    if required_tmax is not None:
        manifest["run_contract"] = {"analysis_window": [0.0, required_tmax]}
    return manifest


def test_overdetermined_status_blocks_before_nested_manifest(tmp_path: Path) -> None:
    mod = _load_tool_module()

    report = mod.overdetermined_campaign_status_report(_manifest(tmp_path))

    assert report["passed"] is False
    assert report["summary"]["ready_for_runtime_count"] == 0
    assert (
        "missing_nested_nonlinear_campaign_manifest"
        in report["controls"][0]["blockers"]
    )
    assert report["next_actions"] == [
        "run nonlinear_campaign_command_after_vmec_runs for each VMEC-complete control"
    ]


def test_overdetermined_status_tracks_runtime_and_fd_progression(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()

    runtime_blocked = mod.overdetermined_campaign_status_report(
        _manifest(tmp_path, with_nested=True)
    )
    promoted = mod.overdetermined_campaign_status_report(
        _manifest(
            tmp_path / "promoted", with_nested=True, with_runtime=True, fd_passed=True
        )
    )

    assert runtime_blocked["controls"][0]["ready_for_runtime"] is True
    assert runtime_blocked["controls"][0]["runtime_output_status"]["missing_count"] == 3
    assert (
        "run direct full-horizon nonlinear tasks" in runtime_blocked["next_actions"][0]
    )
    assert promoted["passed"] is True
    assert promoted["summary"]["central_fd_promoted_count"] == 1
    assert promoted["ranking_status"]["passed"] is True


def test_overdetermined_status_uses_ranking_recommendation_after_failed_final_gates(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()

    report = mod.overdetermined_campaign_status_report(
        _manifest(tmp_path, with_nested=True, with_runtime=True, fd_passed=False)
    )

    assert report["passed"] is False
    assert report["summary"]["runtime_complete_count"] == 1
    assert report["summary"]["central_fd_promoted_count"] == 0
    assert report["next_actions"] == ["blocked"]


def test_overdetermined_status_requires_full_runtime_time_coverage(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()

    partial = mod.overdetermined_campaign_status_report(
        _manifest(
            tmp_path / "partial",
            with_nested=True,
            with_runtime=True,
            fd_passed=True,
            required_tmax=900.0,
            runtime_tmax=500.0,
        )
    )
    complete = mod.overdetermined_campaign_status_report(
        _manifest(
            tmp_path / "complete",
            with_nested=True,
            with_runtime=True,
            fd_passed=True,
            required_tmax=900.0,
            runtime_tmax=900.0,
        )
    )
    rounded_final = mod.overdetermined_campaign_status_report(
        _manifest(
            tmp_path / "rounded",
            with_nested=True,
            with_runtime=True,
            fd_passed=True,
            required_tmax=900.0,
            runtime_tmax=899.93,
        )
    )

    assert partial["passed"] is False
    assert partial["controls"][0]["runtime_output_status"]["missing_count"] == 0
    assert partial["controls"][0]["runtime_output_status"]["incomplete_count"] == 3
    assert "incomplete_runtime_outputs" in partial["controls"][0]["blockers"]
    assert complete["passed"] is True
    assert rounded_final["passed"] is True
    assert complete["controls"][0]["runtime_output_status"]["complete_count"] == 3


def test_overdetermined_status_cli_writes_json_and_fail_code(tmp_path: Path) -> None:
    mod = _load_tool_module()
    manifest = tmp_path / "manifest.json"
    out = tmp_path / "status.json"
    manifest.write_text(json.dumps(_manifest(tmp_path)), encoding="utf-8")

    assert mod.main(["overdetermined-gradient", str(manifest), "--out-json", str(out)]) == 0
    assert mod.main(["overdetermined-gradient", str(manifest), "--out-json", str(out), "--fail-on-blocked"]) == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["kind"] == "overdetermined_nonlinear_gradient_campaign_status"


def test_overdetermined_status_rejects_wrong_manifest_kind(tmp_path: Path) -> None:
    mod = _load_tool_module()

    with pytest.raises(ValueError, match="overdetermined"):
        mod.overdetermined_campaign_status_report({"kind": "other", "controls": []})
