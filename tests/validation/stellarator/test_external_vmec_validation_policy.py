from __future__ import annotations

import json
from pathlib import Path

from support.paths import load_release_tool


def _load_admission_tool_module():
    return load_release_tool("check_vmec_boozer_gates")


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _full_grid_gate(
    tmp_path: Path, *, extra_failed_metric: str | None = None, passed: bool = False
) -> Path:
    failed_metrics = [
        "common_window_pairwise_heat_flux_symmetric_relative_difference",
        "least_window_pairwise_heat_flux_symmetric_relative_difference",
    ]
    if extra_failed_metric is not None:
        failed_metrics.append(extra_failed_metric)
    gates = [
        {"metric": metric, "passed": metric not in failed_metrics}
        for metric in [
            "common_window_max_relative_slope_per_time",
            "common_window_pairwise_heat_flux_symmetric_relative_difference",
            "least_window_pairwise_heat_flux_symmetric_relative_difference",
        ]
    ]
    return _write_json(
        tmp_path / "full_grid.json",
        {
            "kind": "external_vmec_nonlinear_grid_convergence_gate",
            "passed": passed,
            "runs": [{"label": "n48"}, {"label": "n64"}, {"label": "n80"}],
            "gate_report": {"passed": passed, "gates": gates},
        },
    )


def _high_grid_gate(
    tmp_path: Path, name: str, *, passed: bool = True, common: float = 0.05
) -> Path:
    return _write_json(
        tmp_path / f"{name}.json",
        {
            "kind": "external_vmec_nonlinear_grid_convergence_gate",
            "passed": passed,
            "thresholds": {"max_pairwise_relative_difference": 0.15},
            "common_window": {
                "max_pairwise_heat_flux_symmetric_relative_difference": common
            },
            "least_windows": {
                "max_pairwise_heat_flux_symmetric_relative_difference": 0.04
            },
            "runs": [{"label": "n64"}, {"label": "n80"}],
            "gate_report": {
                "passed": passed,
                "gates": [{"metric": "pairwise", "passed": passed}],
            },
        },
    )


def _time_horizon_gate(tmp_path: Path, *, passed: bool = True) -> Path:
    return _write_json(
        tmp_path / "time_horizon.json",
        {
            "kind": "external_vmec_time_horizon_gate",
            "passed": passed,
            "thresholds": {"max_relative_change": 0.15},
            "common_window_time_horizon_relative_change": 0.02,
            "least_window_time_horizon_relative_change": 0.03,
            "gate_report": {
                "passed": passed,
                "gates": [{"metric": "horizon", "passed": passed}],
            },
        },
    )


def _replicate_gate(
    tmp_path: Path, *, passed: bool = True, spread: float = 0.04
) -> Path:
    return _write_json(
        tmp_path / "replicate.json",
        {
            "kind": "nonlinear_window_ensemble_report",
            "passed": passed,
            "config": {
                "max_mean_rel_spread": 0.15,
                "max_combined_sem_rel": 0.25,
            },
            "statistics": {
                "n_reports": 4,
                "n_finite_means": 4,
                "ensemble_mean": 9.5,
                "mean_rel_spread": spread,
                "combined_sem_rel": 0.05,
            },
        },
    )


def _build_admission_payload(tmp_path: Path, **overrides: Path):
    mod = _load_admission_tool_module()
    return mod.build_high_grid_admission_payload(
        full_grid_gate_path=overrides.get("full_grid") or _full_grid_gate(tmp_path),
        high_grid_gate_paths=[
            overrides.get("high_grid_a") or _high_grid_gate(tmp_path, "t250"),
            overrides.get("high_grid_b") or _high_grid_gate(tmp_path, "t350"),
        ],
        time_horizon_gate_path=overrides.get("time_horizon")
        or _time_horizon_gate(tmp_path),
        replicate_ensemble_path=overrides.get("replicate") or _replicate_gate(tmp_path),
        excluded_grid_labels=["n48"],
        retained_grid_labels=["n64", "n80"],
        case="synthetic high-grid admission",
    )


def test_high_grid_admission_passes_with_coarse_exclusion_and_replicates(
    tmp_path: Path,
) -> None:
    payload = _build_admission_payload(tmp_path)

    assert payload["kind"] == "external_vmec_high_grid_admission_gate"
    assert payload["passed"] is True
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["blockers"] == []
    assert (
        payload["policy"]["calibration_use"] == "eligible_as_scoped_high_grid_holdout"
    )
    assert (
        payload["claim_level"]
        == "passed_high_grid_transport_holdout_admission_under_coarse_grid_exclusion"
    )


def test_high_grid_admission_fails_unexpected_full_grid_failure(tmp_path: Path) -> None:
    payload = _build_admission_payload(
        tmp_path,
        full_grid=_full_grid_gate(
            tmp_path,
            extra_failed_metric="common_window_max_relative_slope_per_time",
        ),
    )

    assert payload["passed"] is False
    assert (
        "full_grid_failure_limited_to_grid_difference"
        in payload["promotion_gate"]["blockers"]
    )


def test_high_grid_admission_fails_when_high_grid_pair_does_not_pass(
    tmp_path: Path,
) -> None:
    payload = _build_admission_payload(
        tmp_path, high_grid_b=_high_grid_gate(tmp_path, "t350", passed=False)
    )

    assert payload["passed"] is False
    assert "high_grid_gate_failure_count" in payload["promotion_gate"]["blockers"]


def test_high_grid_admission_fails_replicate_spread(tmp_path: Path) -> None:
    payload = _build_admission_payload(
        tmp_path, replicate=_replicate_gate(tmp_path, spread=0.22)
    )

    assert payload["passed"] is False
    assert "replicate_mean_relative_spread" in payload["promotion_gate"]["blockers"]


def test_high_grid_admission_cli_writes_json(tmp_path: Path) -> None:
    mod = _load_admission_tool_module()
    out = tmp_path / "admission.json"
    rc = mod.main(
        [
            "high-grid-admission",
            "--full-grid-gate",
            str(_full_grid_gate(tmp_path)),
            "--high-grid-gate",
            str(_high_grid_gate(tmp_path, "t250")),
            "--high-grid-gate",
            str(_high_grid_gate(tmp_path, "t350")),
            "--time-horizon-gate",
            str(_time_horizon_gate(tmp_path)),
            "--replicate-ensemble",
            str(_replicate_gate(tmp_path)),
            "--excluded-grid-label",
            "n48",
            "--retained-grid-label",
            "n64",
            "--retained-grid-label",
            "n80",
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["passed"] is True
