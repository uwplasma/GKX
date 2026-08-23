"""Contracts for nonlinear transport-window artifact and gate utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from support.paths import REPO_ROOT, load_release_tool, load_tool_script
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


def test_saturation_campaign_prints_cumulative_runtime_progress(capsys) -> None:
    campaign = load_tool_script("campaigns", "nonlinear_saturated_state")

    campaign._campaign_progress(
        "completed nonlinear chunk 2: t=25/100 progress= 25.0% elapsed=01:00"
    )

    assert capsys.readouterr().out == (
        "[gkx] completed nonlinear chunk 2: "
        "t=25/100 progress= 25.0% elapsed=01:00\n"
    )


def test_saturation_campaign_trace_omits_dealiased_zero_modes() -> None:
    campaign = load_tool_script("campaigns", "nonlinear_saturated_state")
    full_kx = np.arange(12, dtype=float)
    full_ky = np.arange(12, dtype=float)
    resolved = type(
        "Resolved",
        (),
        {
            "Phi2_kxt": np.arange(24).reshape(2, 12),
            "HeatFlux_kxst": np.arange(48).reshape(2, 2, 12),
            "Phi2_kyt": np.arange(24).reshape(2, 12),
            "HeatFlux_kyst": np.arange(48).reshape(2, 2, 12),
        },
    )()

    payload = campaign._trace_spectral_payload(
        resolved, kx_full=full_kx, ky_full=full_ky
    )

    assert payload["kx"].shape == (7,)
    assert payload["ky"].shape == (4,)
    assert payload["Phi2_kxt"].shape == (2, 7)
    assert payload["HeatFlux_kxst"].shape == (2, 2, 7)
    assert payload["Phi2_kyt"].shape == (2, 4)
    assert payload["HeatFlux_kyst"].shape == (2, 2, 4)


def test_saturation_campaign_requires_and_records_its_checkout_source(
    tmp_path: Path,
) -> None:
    campaign = load_tool_script("campaigns", "nonlinear_saturated_state")
    provenance = campaign._campaign_source_provenance(
        ROOT / "src" / "gkx" / "__init__.py"
    )
    encoded = campaign._npz_source_provenance(provenance)

    assert provenance["repository_root"] == str(ROOT)
    assert provenance["git_commit"]
    assert campaign._gkx_source_tree_matches(
        ROOT, provenance["git_commit"], provenance["git_commit"]
    )
    assert encoded["gkx_git_commit"].dtype.kind == "U"
    assert encoded["gkx_git_dirty"].dtype.kind in "iu"
    with pytest.raises(SystemExit, match="PYTHONPATH=src"):
        campaign._campaign_source_provenance(tmp_path / "gkx" / "__init__.py")


def test_saturation_campaign_does_not_duplicate_requested_npz_trace(
    tmp_path: Path,
) -> None:
    campaign = load_tool_script("campaigns", "nonlinear_saturated_state")
    values = np.arange(3, dtype=float)
    trace = tmp_path / "trace.npz"
    np.savez_compressed(trace, time=values)

    addressed = campaign._summary_trace_payload(
        values, values + 1, values + 2, values + 3, trace_path=trace
    )
    inline = campaign._summary_trace_payload(
        values, values + 1, values + 2, values + 3, trace_path=None
    )

    assert "trace" not in addressed
    assert addressed["trace_artifact"]["bytes"] == trace.stat().st_size
    assert (
        addressed["trace_artifact"]["sha256"]
        == hashlib.sha256(trace.read_bytes()).hexdigest()
    )
    assert len(inline["trace"]) == 3


def test_saturation_campaign_locks_output_paths_between_processes(
    tmp_path: Path,
) -> None:
    campaign = load_tool_script("campaigns", "nonlinear_saturated_state")
    target = tmp_path / "campaign.npz"

    first = campaign._campaign_output_locks((target, None, target))
    try:
        assert len(first) == 1
        assert "pid=" in (tmp_path / "campaign.npz.lock").read_text()
        with pytest.raises(SystemExit, match="campaign output is locked"):
            campaign._campaign_output_locks((target,))
    finally:
        for handle in first:
            handle.close()

    second = campaign._campaign_output_locks((target,))
    for handle in second:
        handle.close()


def test_saturation_campaign_cannot_promote_a_continuation_segment() -> None:
    campaign = load_tool_script("campaigns", "nonlinear_saturated_state")
    report = {"saturated": True, "reasons": []}

    full = campaign._scope_saturation_report(report, continuation=False)
    segment = campaign._scope_saturation_report(report, continuation=True)

    assert full == {**report, "history_scope": "full_run"}
    assert segment["history_scope"] == "continuation_segment"
    assert segment["segment_saturated"] is True
    assert segment["saturated"] is False
    assert segment["reasons"] == ["prior_history_not_in_report"]
    assert report == {"saturated": True, "reasons": []}


def test_saturation_campaign_records_resolved_timestep_policy() -> None:
    campaign = load_tool_script("campaigns", "nonlinear_saturated_state")
    time_cfg = type(
        "TimeConfig",
        (),
        {
            "fixed_dt": False,
            "dt": 0.1,
            "dt_max": None,
            "cfl": 0.5,
            "method": "rk3",
        },
    )()

    policy = campaign._resolved_timestep_policy(time_cfg)
    encoded = campaign._npz_timestep_identity(policy)

    assert policy == {
        "fixed_dt": False,
        "dt": 0.1,
        "dt_max": None,
        "cfl": 0.5,
        "method": "rk3",
    }
    assert encoded["time_dt_max"].item() == "None"
    assert encoded["time_cfl"].item() == pytest.approx(0.5)


def test_saturation_policy_replay_requires_a_persistent_fixed_window() -> None:
    replay = load_tool_script("campaigns", "nonlinear_saturated_state")
    time = np.arange(0.0, 201.0, 0.5)
    phase = 2.0 * np.pi * time / 10.0
    report = replay.replay_policy(
        time,
        10.0 + 0.15 * np.sin(phase),
        2.0 + 0.03 * np.sin(phase + 0.3),
        100.0 + np.sin(phase + 0.7),
        policy=replay.ReplayPolicy(window=50.0, persistence=20.0),
    )

    assert report["stopped"] is True
    assert report["first_stop"]["persistence_start"] == pytest.approx(50.0)
    assert report["first_stop"]["checkpoint_time"] == pytest.approx(70.0)
    assert report["first_stop"]["decision"]["window_tmin"] == pytest.approx(20.0)
    assert report["first_stop"]["decision"]["statistics"]["heat_flux"]["stationary"]


def test_saturation_policy_replay_requires_clean_contiguous_source_traces(
    tmp_path: Path,
) -> None:
    replay = load_tool_script("campaigns", "nonlinear_saturated_state")

    def write_trace(
        path: Path,
        time: np.ndarray,
        *,
        previous_t_end: float,
        case: str = "qa",
        dirty: int = 0,
    ) -> None:
        np.savez_compressed(
            path,
            time=time,
            heat_flux=np.ones(time.size),
            Wphi=np.ones(time.size),
            Wg=np.ones(time.size),
            gkx_git_commit=np.asarray("abc123"),
            gkx_git_dirty=np.asarray(dirty),
            previous_t_end=np.asarray(previous_t_end),
            campaign_identity_schema=np.asarray("gkx_nonlinear_campaign_v1"),
            case=np.asarray(case),
        )

    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    write_trace(first, np.arange(0.0, 10.0), previous_t_end=0.0)
    write_trace(second, np.arange(10.0, 20.0), previous_t_end=9.0)

    arrays, sources = replay._load_replay_traces([first, second])
    assert arrays[0].tolist() == list(np.arange(20.0))
    assert len(sources) == 2
    assert sources[0]["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()

    write_trace(second, np.arange(10.0, 20.0), previous_t_end=9.0, case="qi")
    with pytest.raises(ValueError, match="campaign identity differs"):
        replay._load_replay_traces([first, second])

    write_trace(second, np.arange(10.0, 20.0), previous_t_end=8.0)
    with pytest.raises(ValueError, match="not a contiguous continuation"):
        replay._load_replay_traces([first, second])

    write_trace(second, np.arange(10.0, 20.0), previous_t_end=9.0, dirty=1)
    with pytest.raises(ValueError, match="not pinned to a clean GKX commit"):
        replay._load_replay_traces([first, second])

    write_trace(second, np.arange(10.0, 20.0), previous_t_end=9.0)
    for path in (first, second):
        with np.load(path, allow_pickle=False) as archive:
            legacy = {name: archive[name] for name in archive.files}
        legacy.pop("campaign_identity_schema")
        np.savez_compressed(path, **legacy)
    with pytest.raises(ValueError, match="legacy continuation replay requires"):
        replay._load_replay_traces([first, second])

    def write_summary(path: Path, trace_path: Path, *, case: str = "qa") -> None:
        with np.load(trace_path, allow_pickle=False) as archive:
            samples = [
                {"t": t, "heat_flux": q, "Wphi": wphi, "Wg": wg}
                for t, q, wphi, wg in zip(
                    archive["time"],
                    archive["heat_flux"],
                    archive["Wphi"],
                    archive["Wg"],
                )
            ]
            previous_t_end = float(archive["previous_t_end"])
        path.write_text(
            json.dumps(
                {
                    "source_provenance": {
                        "git_commit": "abc123",
                        "git_dirty": False,
                    },
                    "previous_t_end": previous_t_end,
                    "case": case,
                    "grid": {"Nx": 2, "Ny": 2, "Nz": 2},
                    "geometry_override": {"vmec_file": "equilibrium.nc"},
                    "random_seed": 31,
                    "alpha": 0.0,
                    "npol": 1.0,
                    "trace": samples,
                }
            ),
            encoding="utf-8",
        )

    first_summary = tmp_path / "first.json"
    second_summary = tmp_path / "second.json"
    write_summary(first_summary, first)
    write_summary(second_summary, second)
    replay._load_replay_traces([first, second], [first_summary, second_summary])

    write_summary(second_summary, second, case="qi")
    with pytest.raises(ValueError, match="campaign identity differs"):
        replay._load_replay_traces([first, second], [first_summary, second_summary])


def test_saturation_campaign_rejects_a_mixed_continuation_state(tmp_path: Path) -> None:
    campaign = load_tool_script("campaigns", "nonlinear_saturated_state")
    provenance = campaign._campaign_source_provenance(
        ROOT / "src" / "gkx" / "__init__.py"
    )
    provenance["git_dirty"] = False
    identity = {
        name: np.asarray(value)
        for name, value in {
            "campaign_identity_schema": "gkx_nonlinear_campaign_v1",
            "case": "qa.toml",
            "input_sha256": "deck",
            "vmec_sha256": "equilibrium",
            "Nx": 2,
            "Ny": 2,
            "Nz": 2,
            "Nl": 1,
            "Nm": 1,
            "random_seed": 31,
            "alpha": "0.0",
            "npol": "1.0",
        }.items()
    }
    state = tmp_path / "state.npz"
    np.savez_compressed(
        state,
        state=np.zeros((1, 1, 1, 2, 2, 2)),
        t_end=10.0,
        **identity,
        **campaign._npz_source_provenance(provenance),
    )

    loaded, t_end = campaign._load_continuation_state(
        state,
        expected_shape=(1, 1, 1, 2, 2, 2),
        expected_identity=identity,
        source_provenance=provenance,
    )
    assert loaded.shape == (1, 1, 1, 2, 2, 2)
    assert t_end == 10.0

    v2_expected = dict(identity)
    v2_expected.update(
        campaign_identity_schema=np.asarray("gkx_nonlinear_campaign_v2"),
        time_fixed_dt=np.asarray(False),
        time_dt=np.asarray(0.1),
        time_dt_max=np.asarray("None"),
        time_cfl=np.asarray(1.0),
        time_method=np.asarray("rk3"),
    )
    campaign._load_continuation_state(
        state,
        expected_shape=(1, 1, 1, 2, 2, 2),
        expected_identity=v2_expected,
        source_provenance=provenance,
    )

    identity["case"] = np.asarray("qi.toml")
    with pytest.raises(SystemExit, match="campaign identity does not match"):
        campaign._load_continuation_state(
            state,
            expected_shape=(1, 1, 1, 2, 2, 2),
            expected_identity=identity,
            source_provenance=provenance,
        )


def test_saturation_campaign_rejects_a_timestep_mismatched_state(
    tmp_path: Path,
) -> None:
    campaign = load_tool_script("campaigns", "nonlinear_saturated_state")
    provenance = campaign._campaign_source_provenance(
        ROOT / "src" / "gkx" / "__init__.py"
    )
    provenance["git_dirty"] = False
    identity = {
        name: np.asarray(value)
        for name, value in {
            "campaign_identity_schema": "gkx_nonlinear_campaign_v2",
            "case": "qa.toml",
            "input_sha256": "deck",
            "vmec_sha256": "equilibrium",
            "Nx": 2,
            "Ny": 2,
            "Nz": 2,
            "Nl": 1,
            "Nm": 1,
            "random_seed": 31,
            "alpha": "0.0",
            "npol": "1.0",
            "time_fixed_dt": False,
            "time_dt": 0.1,
            "time_dt_max": "None",
            "time_cfl": 1.0,
            "time_method": "rk3",
        }.items()
    }
    state = tmp_path / "state.npz"
    np.savez_compressed(
        state,
        state=np.zeros((1, 1, 1, 2, 2, 2)),
        t_end=10.0,
        **identity,
        **campaign._npz_source_provenance(provenance),
    )

    expected = dict(identity)
    expected["time_cfl"] = np.asarray(0.5)
    with pytest.raises(SystemExit, match="campaign identity does not match"):
        campaign._load_continuation_state(
            state,
            expected_shape=(1, 1, 1, 2, 2, 2),
            expected_identity=expected,
            source_provenance=provenance,
        )


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
