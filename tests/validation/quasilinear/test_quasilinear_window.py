from __future__ import annotations

import json
import os
from pathlib import Path

from support.paths import REPO_ROOT, load_release_tool, load_tool_script
import subprocess
import sys

import numpy as np
import pytest

from gkx.diagnostics.saturation import (
    SaturationStopConfig,
    saturation_stop_decision,
    sokal_autocorrelation_time,
)
from gkx.diagnostics.transport_windows import (
    NonlinearWindowConvergenceConfig,
    NonlinearWindowEnsembleConfig,
    nonlinear_window_convergence_from_csv,
    nonlinear_window_convergence_from_summary,
    nonlinear_window_convergence_report,
    nonlinear_window_ensemble_report,
    nonlinear_window_stats_promotion_ready,
)


def _load_tool_module():
    return load_release_tool("check_nonlinear_transport_gates")


def test_autocorrelation_campaign_uses_runtime_effective_sample_count(
    tmp_path: Path,
) -> None:
    campaign = load_tool_script("campaigns", "heat_flux_autocorrelation")
    rng = np.random.default_rng(31)
    time = np.arange(400, dtype=float) * 0.2
    flux = np.empty(time.size)
    flux[0] = 4.0
    for index in range(1, flux.size):
        flux[index] = 4.0 + 0.9 * (flux[index - 1] - 4.0) + rng.normal(0.0, 0.1)
    trace = tmp_path / "trace.csv"
    np.savetxt(
        trace,
        np.column_stack((time, flux)),
        delimiter=",",
        header="t,heat_flux",
        comments="",
    )

    report = campaign.analyse(trace)

    assert report is not None
    n = report["samples_in_window"]
    expected = min(
        float(n),
        n * report["output_dt"] / (2.0 * report["tau_ac"]),
    )
    assert report["independent_samples"] == pytest.approx(expected)


def _saturated_trace() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 200.0, 201)
    heat = 4.0 + 0.04 * np.sin(2.0 * np.pi * t / 10.0)
    return t, heat


def test_converged_saturated_transport_window_passes_with_finite_uncertainty() -> None:
    t, heat = _saturated_trace()

    report = nonlinear_window_convergence_report(
        t,
        heat,
        case="synthetic_saturated_itg",
        source_artifact="synthetic.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
            min_blocks=4,
            max_running_mean_rel_drift=0.02,
            max_sem_rel=0.02,
        ),
    )

    assert report["passed"] is True
    assert report["statistics"]["late_mean"] == pytest.approx(4.0, abs=5.0e-3)
    assert np.isfinite(report["statistics"]["block_bootstrap_sem"])
    assert np.isfinite(report["statistics"]["sem"])
    assert report["statistics"]["terminal_mean_rel_delta"] < 0.02
    assert report["window"]["transient_cutoff"] == pytest.approx(100.0)
    ready, failures = nonlinear_window_stats_promotion_ready(report)
    assert ready is True
    assert failures == []


def test_window_report_supports_explicit_bounds_and_deterministic_blocks() -> None:
    t, heat = _saturated_trace()

    report = nonlinear_window_convergence_report(
        t,
        heat,
        case="bounded_saturated_itg",
        source_artifact="bounded.csv",
        config=NonlinearWindowConvergenceConfig(
            tmin=50.0,
            tmax=150.0,
            transient_fraction=0.25,
            block_size=8,
            bootstrap_samples=0,
            min_samples=40,
            min_blocks=4,
            max_running_mean_rel_drift=0.03,
            max_sem_rel=0.03,
        ),
    )

    assert report["passed"] is False
    assert report["window"]["selected_tmin"] == pytest.approx(50.0)
    assert report["window"]["selected_tmax"] == pytest.approx(150.0)
    assert report["statistics"]["block_size"] == 8
    assert report["statistics"]["block_bootstrap_sem"] is None
    failed = {gate["metric"] for gate in report["gates"] if not gate["passed"]}
    assert failed == {"block_bootstrap_sem"}
    # With no bootstrap samples, the diagnostic SEM falls back to sample/block SEM,
    # but promotion still fails closed because bootstrap uncertainty was requested.
    assert np.isfinite(report["statistics"]["sem"])


def test_transient_only_trace_fails_running_mean_gate() -> None:
    t = np.linspace(0.0, 120.0, 121)
    heat = 0.05 * t

    report = nonlinear_window_convergence_report(
        t,
        heat,
        case="ramping_transient",
        source_artifact="ramp.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=32,
            max_running_mean_rel_drift=0.05,
            max_sem_rel=1.0,
        ),
    )

    failed = {gate["metric"] for gate in report["gates"] if not gate["passed"]}
    assert report["passed"] is False
    assert "running_mean_drift" in failed


def test_terminal_subwindow_gate_blocks_cancelled_running_mean_drift() -> None:
    t = np.arange(96.0)
    heat = np.ones_like(t)
    heat[48:] = np.concatenate(
        [
            np.ones(24),
            np.zeros(12),
            2.0 * np.ones(12),
        ]
    )

    report = nonlinear_window_convergence_report(
        t,
        heat,
        case="terminal_drift_hidden_by_half_means",
        source_artifact="terminal.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=48,
            min_blocks=4,
            max_running_mean_rel_drift=0.01,
            terminal_fraction=0.25,
            min_terminal_samples=8,
            max_terminal_mean_rel_delta=0.20,
            max_sem_rel=10.0,
        ),
    )

    failed = {gate["metric"] for gate in report["gates"] if not gate["passed"]}
    assert report["passed"] is False
    assert "terminal_mean_agreement" in failed
    assert "running_mean_drift" not in failed
    assert report["statistics"]["terminal_n_samples"] == 12
    assert report["statistics"]["terminal_mean_rel_delta"] == pytest.approx(1.0)


def test_nonlinear_window_ensemble_gate_accepts_seed_replicates() -> None:
    import gkx as sgk

    assert sgk.NonlinearWindowEnsembleConfig is NonlinearWindowEnsembleConfig
    assert sgk.nonlinear_window_ensemble_report is nonlinear_window_ensemble_report

    t, heat = _saturated_trace()
    reports = [
        nonlinear_window_convergence_report(
            t,
            heat + offset,
            case=f"seed_{idx}",
            source_artifact=f"seed_{idx}.csv",
            config=NonlinearWindowConvergenceConfig(
                transient_fraction=0.5,
                min_samples=64,
                min_blocks=4,
                max_running_mean_rel_drift=0.02,
                max_sem_rel=0.02,
            ),
        )
        for idx, offset in enumerate((-0.02, 0.0, 0.02))
    ]

    report = nonlinear_window_ensemble_report(
        reports,
        case="seed_uncertainty_gate",
        comparison="random_seed_replicates",
        config=NonlinearWindowEnsembleConfig(
            min_reports=3,
            max_mean_rel_spread=0.02,
            max_combined_sem_rel=0.02,
        ),
    )

    assert report["passed"] is True
    assert report["statistics"]["n_reports"] == 3
    assert report["statistics"]["mean_rel_spread"] == pytest.approx(0.01)
    assert report["rows"][0]["source_artifact"] == "seed_0.csv"
    assert {gate["metric"] for gate in report["gates"] if not gate["passed"]} == set()


def test_nonlinear_window_ensemble_gate_blocks_spread_and_failed_inputs() -> None:
    t, heat = _saturated_trace()
    good = nonlinear_window_convergence_report(
        t,
        heat,
        case="good_seed",
        source_artifact="good.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
            max_running_mean_rel_drift=0.02,
            max_sem_rel=0.02,
        ),
    )
    drifted = nonlinear_window_convergence_report(
        t,
        heat + 2.0,
        case="drifted_seed",
        source_artifact="drifted.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
            max_running_mean_rel_drift=0.02,
            max_sem_rel=0.02,
        ),
    )
    failed = dict(good)
    failed["case"] = "failed_seed"
    failed["passed"] = False
    failed["gate_report"] = {"passed": False}

    report = nonlinear_window_ensemble_report(
        [good, drifted, failed],
        config=NonlinearWindowEnsembleConfig(
            min_reports=3,
            max_mean_rel_spread=0.05,
            max_combined_sem_rel=0.02,
        ),
    )

    failed_metrics = {gate["metric"] for gate in report["gates"] if not gate["passed"]}
    assert report["passed"] is False
    assert "individual_windows_passed" in failed_metrics
    assert "mean_relative_spread" in failed_metrics
    assert report["rows"][2]["promotion_ready"] is False


def test_small_window_and_nan_late_window_fail() -> None:
    t = np.linspace(0.0, 10.0, 11)
    heat = np.ones_like(t)
    small = nonlinear_window_convergence_report(
        t,
        heat,
        case="small_window",
        source_artifact="small.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=16,
            min_blocks=4,
        ),
    )
    small_failed = {gate["metric"] for gate in small["gates"] if not gate["passed"]}
    assert "finite_sample_count" in small_failed

    t2, heat2 = _saturated_trace()
    heat2[-3] = np.nan
    nan_report = nonlinear_window_convergence_report(
        t2,
        heat2,
        case="nan_late_window",
        source_artifact="nan.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
        ),
    )
    nan_failed = {gate["metric"] for gate in nan_report["gates"] if not gate["passed"]}
    assert nan_report["passed"] is False
    assert "finite_late_window" in nan_failed


def test_nonlinear_window_convergence_subcommand_writes_json(tmp_path: Path) -> None:
    mod = _load_tool_module()
    t, heat = _saturated_trace()
    csv = tmp_path / "trace.csv"
    csv.write_text(
        "t,heat_flux\n"
        + "\n".join(f"{ti:.8g},{qi:.12g}" for ti, qi in zip(t, heat, strict=True))
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "window.json"

    assert (
        mod.main(
            [
                "convergence",
                "--csv",
                str(csv),
                "--out-json",
                str(out),
                "--min-samples",
                "64",
                "--max-sem-rel",
                "0.02",
                "--max-running-mean-rel-drift",
                "0.02",
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["provenance"]["source_artifact"] == str(csv)


def test_nonlinear_window_script_imports_before_editable_install() -> None:
    root = REPO_ROOT
    env = {**os.environ, "PYTHONPATH": ""}

    completed = subprocess.run(
        [
            sys.executable,
            "tools/release/check_nonlinear_transport_gates.py",
            "convergence",
            "--help",
        ],
        cwd=root,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Check nonlinear late-window convergence metadata" in completed.stdout


def test_nonlinear_window_config_and_input_validation_are_fail_closed() -> None:
    t = np.linspace(0.0, 10.0, 11)
    heat = np.ones_like(t)

    bad_configs = [
        (NonlinearWindowConvergenceConfig(tmin=np.nan), "tmin must be finite"),
        (NonlinearWindowConvergenceConfig(tmax=np.inf), "tmax must be finite"),
        (NonlinearWindowConvergenceConfig(tmin=5.0, tmax=5.0), "tmin must be less"),
        (
            NonlinearWindowConvergenceConfig(transient_fraction=1.0),
            "transient_fraction",
        ),
        (NonlinearWindowConvergenceConfig(min_samples=1), "min_samples"),
        (NonlinearWindowConvergenceConfig(min_blocks=1), "min_blocks"),
        (NonlinearWindowConvergenceConfig(block_size=0), "block_size"),
        (NonlinearWindowConvergenceConfig(bootstrap_samples=-1), "bootstrap_samples"),
        (
            NonlinearWindowConvergenceConfig(max_running_mean_rel_drift=-1.0),
            "running_mean",
        ),
        (NonlinearWindowConvergenceConfig(terminal_fraction=0.0), "terminal_fraction"),
        (NonlinearWindowConvergenceConfig(min_terminal_samples=0), "min_terminal"),
        (
            NonlinearWindowConvergenceConfig(max_terminal_mean_rel_delta=-1.0),
            "terminal_mean",
        ),
        (NonlinearWindowConvergenceConfig(max_sem_rel=-1.0), "max_sem_rel"),
        (NonlinearWindowConvergenceConfig(value_floor=0.0), "value_floor"),
    ]
    for config, message in bad_configs:
        with pytest.raises(ValueError, match=message):
            nonlinear_window_convergence_report(t, heat, config=config)

    with pytest.raises(ValueError, match="same length"):
        nonlinear_window_convergence_report([0.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="must not be empty"):
        nonlinear_window_convergence_report([], [])
    with pytest.raises(ValueError, match="time contains non-finite"):
        nonlinear_window_convergence_report([0.0, np.nan], [1.0, 1.0])
    with pytest.raises(ValueError, match="selected nonlinear window is empty"):
        nonlinear_window_convergence_report([1.0], [1.0])

    bad_ensemble_configs = [
        (NonlinearWindowEnsembleConfig(min_reports=1), "min_reports"),
        (NonlinearWindowEnsembleConfig(max_mean_rel_spread=-1.0), "mean_rel_spread"),
        (NonlinearWindowEnsembleConfig(max_combined_sem_rel=-1.0), "combined_sem"),
        (NonlinearWindowEnsembleConfig(value_floor=0.0), "value_floor"),
    ]
    for config, message in bad_ensemble_configs:
        with pytest.raises(ValueError, match=message):
            nonlinear_window_ensemble_report([], config=config)
    with pytest.raises(TypeError, match="report dictionaries"):
        nonlinear_window_ensemble_report(
            [object()],  # type: ignore[list-item]
            config=NonlinearWindowEnsembleConfig(min_reports=2),
        )


def test_nonlinear_window_csv_and_summary_loaders_cover_artifact_contracts(
    tmp_path: Path,
) -> None:
    csv = tmp_path / "diagnostics.csv"
    t, heat = _saturated_trace()
    csv.write_text(
        "time,q_i\n"
        + "\n".join(f"{ti:.8g},{qi:.12g}" for ti, qi in zip(t, heat, strict=True))
        + "\n",
        encoding="utf-8",
    )
    config = NonlinearWindowConvergenceConfig(
        transient_fraction=0.5,
        min_samples=64,
        max_running_mean_rel_drift=0.02,
        max_sem_rel=0.02,
    )

    from_csv = nonlinear_window_convergence_from_csv(
        csv,
        time_column="time",
        value_column="q_i",
        case="csv_case",
        config=config,
        summary_artifact="summary.json",
    )

    assert from_csv["passed"] is True
    assert from_csv["case"] == "csv_case"
    assert from_csv["observable"] == "q_i"
    assert from_csv["provenance"]["summary_artifact"] == "summary.json"

    summary = tmp_path / "window_summary.json"
    summary.write_text(
        json.dumps(
            {
                "case": "summary_case",
                "gkx": "diagnostics.csv",
                "tmin": 50.0,
                "tmax": 200.0,
            }
        ),
        encoding="utf-8",
    )
    from_summary = nonlinear_window_convergence_from_summary(
        summary,
        time_column="time",
        value_column="q_i",
        config=config,
    )

    assert from_summary["case"] == "summary_case"
    assert from_summary["provenance"]["summary_artifact"] == str(summary)
    assert from_summary["provenance"]["source_artifact"].endswith("diagnostics.csv")

    missing_column = tmp_path / "missing_column.csv"
    missing_column.write_text("time,other\n0.0,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="observable column"):
        nonlinear_window_convergence_from_csv(
            missing_column,
            time_column="time",
            value_column="q_i",
        )

    bad_summary = tmp_path / "bad_summary.json"
    bad_summary.write_text(json.dumps({"other": "diagnostics.csv"}), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostics source"):
        nonlinear_window_convergence_from_summary(bad_summary)

    txt_summary = tmp_path / "txt_summary.json"
    txt_summary.write_text(json.dumps({"gkx": "trace.txt"}), encoding="utf-8")
    (tmp_path / "trace.txt").write_text("not,csv\n", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="diagnostics CSV"):
        nonlinear_window_convergence_from_summary(txt_summary)


def test_nonlinear_window_promotion_ready_reports_all_missing_contracts() -> None:
    ready, failures = nonlinear_window_stats_promotion_ready(None)
    assert ready is False
    assert failures == ["missing nonlinear_window_stats object"]

    ready, failures = nonlinear_window_stats_promotion_ready(
        {
            "kind": "wrong",
            "passed": False,
            "provenance": {},
            "statistics": {"late_mean": 1.0},
            "window": {"transient_fraction": 0.0, "n_finite_late": 0},
            "gate_report": {"passed": False},
        }
    )

    assert ready is False
    assert "unexpected nonlinear_window_stats kind" in failures
    assert "nonlinear window convergence report did not pass" in failures
    assert "missing nonlinear source_artifact provenance" in failures
    assert "missing/non-finite statistics.sem" in failures
    assert "missing/non-finite statistics.terminal_mean_rel_delta" in failures
    assert "missing/non-finite window.late_tmin" in failures
    assert "missing declared transient cutoff policy" in failures
    assert "window has no finite late samples" in failures
    assert "missing passed gate_report" in failures


def test_nonlinear_window_promotion_rejects_string_pass_flags() -> None:
    t, heat = _saturated_trace()
    report = nonlinear_window_convergence_report(
        t,
        heat,
        source_artifact="persisted.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
            max_running_mean_rel_drift=0.02,
            max_sem_rel=0.02,
        ),
    )
    report["passed"] = "true"
    report["gate_report"]["passed"] = "true"

    ready, failures = nonlinear_window_stats_promotion_ready(report)

    assert ready is False
    assert "nonlinear window convergence report did not pass" in failures
    assert "missing passed gate_report" in failures


def test_nonlinear_window_ensemble_promotion_rejects_string_pass_flags() -> None:
    t, heat = _saturated_trace()
    windows = [
        nonlinear_window_convergence_report(
            t,
            heat + offset,
            source_artifact=f"seed_{index}.csv",
            config=NonlinearWindowConvergenceConfig(
                transient_fraction=0.5,
                min_samples=64,
                max_running_mean_rel_drift=0.02,
                max_sem_rel=0.02,
            ),
        )
        for index, offset in enumerate((-0.01, 0.01))
    ]
    report = nonlinear_window_ensemble_report(windows)
    report["passed"] = "true"
    report["gate_report"]["passed"] = "true"
    report["rows"][0]["promotion_ready"] = "true"

    ready, failures = nonlinear_window_stats_promotion_ready(report)

    assert ready is False
    assert "nonlinear window ensemble report did not pass" in failures
    assert "missing passed ensemble gate_report" in failures
    assert "not all ensemble rows are promotion-ready" in failures


def _spinup_then_plateau(
    plateau: float = 5.0, *, drift_per_time: float = 0.0, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Exponential growth to an overshoot, then a correlated noisy plateau."""

    rng = np.random.default_rng(seed)
    t = np.arange(6000) * 0.05
    noise = np.zeros(t.size)
    for i in range(1, t.size):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0.0, 0.25)
    growth = np.minimum(1.0e-3 * np.exp(0.35 * t), 2.0 * plateau)
    saturated = plateau + drift_per_time * t + noise
    return t, np.where(t < 30.0, growth, saturated)


def test_saturation_stop_decision_stops_converged_trace_and_excludes_spinup() -> None:
    t, heat = _spinup_then_plateau()

    decision = saturation_stop_decision(
        t, heat, guard=heat**2, config=SaturationStopConfig(rel_sem=0.05)
    )

    assert decision["saturated"] is True
    assert decision["reasons"] == []
    # The exponential spin-up (heat < median until it nears the plateau) must
    # not pollute the window: the mean lands on the plateau, not on the
    # overshoot, and the window starts after the early growth phase.
    assert decision["window_tmin"] > 20.0
    assert decision["mean"] == pytest.approx(5.0, abs=0.5)
    assert decision["sem"] > 0.0
    assert decision["rel_sem"] <= 0.05
    assert decision["tau_ac_resolved"] is True
    assert decision["window_span"] >= decision["min_window"]


def test_saturation_stop_decision_rejects_non_stationary_trace() -> None:
    t, heat = _spinup_then_plateau(drift_per_time=0.02)

    decision = saturation_stop_decision(t, heat)

    assert decision["saturated"] is False
    assert "window_not_stationary" in decision["reasons"]


def test_saturation_stop_decision_rejects_unresolved_or_short_traces() -> None:
    t = np.arange(2000) * 0.05
    growing = 1.0e-3 * np.exp(0.1 * t)

    decision = saturation_stop_decision(t, growing)
    assert decision["saturated"] is False
    assert decision["reasons"]

    short = saturation_stop_decision(t[:8], growing[:8])
    assert short["saturated"] is False
    assert short["reasons"] == ["trace_shorter_than_min_samples"]
    assert short["mean"] is None


def test_saturation_stop_decision_guard_blocks_drifting_field_energy() -> None:
    t, heat = _spinup_then_plateau()
    _, drifting_guard = _spinup_then_plateau(drift_per_time=0.05, seed=11)

    decision = saturation_stop_decision(t, heat, guard=drifting_guard)

    assert decision["saturated"] is False
    assert "guard_not_stationary" in decision["reasons"]
    assert decision["guard_stationary"] is False


def test_saturation_stop_decision_guard_blocks_drifting_free_energy() -> None:
    t, heat = _spinup_then_plateau()
    _, drifting_wg = _spinup_then_plateau(drift_per_time=0.05, seed=11)

    decision = saturation_stop_decision(t, heat, free_energy_guard=drifting_wg)

    assert decision["saturated"] is False
    assert "Wg_guard_not_stationary" in decision["reasons"]
    assert decision["Wg_guard_stationary"] is False


def test_saturation_stop_decision_honors_min_window_override() -> None:
    t, heat = _spinup_then_plateau()

    decision = saturation_stop_decision(
        t, heat, config=SaturationStopConfig(rel_sem=0.05, min_window=1.0e6)
    )

    assert decision["saturated"] is False
    assert "window_below_min_window" in decision["reasons"]
    assert decision["min_window"] == pytest.approx(1.0e6)


def test_saturation_stop_decision_refuses_a_trace_that_never_left_zero() -> None:
    """A flux that is identically zero has no saturated mean to find.

    The relative SEM divides by a floor when the mean is zero, so without a
    signal gate every other criterion passes on a dead trace and the run stops
    in its first chunk. A zonal-response case, whose heat flux is zero by
    construction, was truncated at t=7.7 of a requested 60 that way.
    """

    time = np.linspace(0.0, 60.0, 600)
    decision = saturation_stop_decision(time, np.zeros_like(time))

    assert decision["saturated"] is False
    assert "flux_indistinguishable_from_zero" in decision["reasons"]


def test_saturation_stop_decision_still_stops_a_small_but_real_flux() -> None:
    """The signal gate must not reject a converged run for being faint.

    The fluctuation here is correlated, because that is what separates a faint
    real flux from a flat one. This test originally used white noise about the
    same mean; that trace is now refused, on purpose, for having no resolved
    correlation time -- see the stationary-from-t=0 test below. Faintness is
    still not a reason to refuse: three parts in a billion saturates.
    """

    rng = np.random.default_rng(0)
    time = np.linspace(0.0, 60.0, 600)
    noise = np.zeros(time.size)
    for index in range(1, time.size):
        noise[index] = 0.9 * noise[index - 1] + rng.normal(0.0, 1.0)
    flux = 3.0e-9 * (1.0 + 0.03 * noise / np.std(noise))

    decision = saturation_stop_decision(time, flux)

    assert decision["saturated"] is True
    assert decision["reasons"] == []
    assert decision["mean"] == pytest.approx(3.0e-9, rel=0.05)
    assert decision["tau_ac"] > 0.0


def test_saturation_stop_decision_refuses_a_flux_stationary_from_t_zero() -> None:
    """A flux that never left its starting value is not a saturated flux.

    The zero-flux guard is an absolute threshold, so it only covers traces that
    happen to sit below it. A run whose heat flux is flat from the first sample
    but a few decades above that floor passed every other gate and stopped in
    its first chunk, at any amplitude: the correlation time of such a trace
    crosses zero at lag one, which made ``tau_ac`` exactly zero, which made the
    derived ``min_window = 10 tau_ac`` zero as well. The window-length
    requirement therefore vanished precisely on the traces carrying the least
    information, and the relative SEM of a long stretch of uncorrelated samples
    is tiny.

    Requiring only ``tau_ac > 0`` does not close it. The lag-one sample
    autocorrelation of uncorrelated noise is positive about half the time, and
    each such realization integrates to a small positive ``tau_ac`` that reads
    as resolved: before the sampling-interval floor, 190 of the 400 draws below
    saturated. So this sweeps realizations rather than checking one, and checks
    two amplitudes because the defect is scale-free while the zero-flux guard
    is not -- scaling a trace cannot change its autocorrelation, so the two
    levels must give identical verdicts.
    """

    time = np.linspace(0.0, 60.0, 600)
    sampling_interval = float(np.median(np.diff(time)))
    saturated_draws = {1.0e-8: 0, 1.0e2: 0}
    for seed in range(400):
        shape = 1.0 + 0.01 * np.random.default_rng(seed).standard_normal(time.size)
        for level in saturated_draws:
            decision = saturation_stop_decision(time, level * shape)
            saturated_draws[level] += bool(decision["saturated"])
            assert decision["tau_ac"] <= sampling_interval, (seed, level)

    assert saturated_draws == {1.0e-8: 0, 1.0e2: 0}, saturated_draws


def test_saturation_stop_decision_still_stops_a_run_that_started_saturated(
) -> None:
    """A warm-started run has no growth phase and must still be allowed to stop.

    This is the case that rules out the obvious alternative fix. Requiring the
    trace to have risen above its own early-time level before a plateau counts
    would also refuse the flat traces above -- but it would refuse these too,
    and these are the runs the stop policy is worth the most on: a run seeded
    from a converged neighbour begins at its saturated flux by construction and
    never grows. Its late/early ratio is 1.02 against 65 for a run that spun up
    from a small perturbation, so no threshold on growth separates it from a
    dead trace. What separates them is that this one has a correlation time.
    """

    time = np.linspace(0.0, 60.0, 600)
    stopped = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        noise = np.zeros(time.size)
        for index in range(1, time.size):
            noise[index] = 0.7 * noise[index - 1] + rng.normal(0.0, 1.0)
        flux = 5.0 * (1.0 + 0.05 * noise / np.std(noise))
        stopped += bool(saturation_stop_decision(time, flux)["saturated"])

    # Not all 40: the half-window stationarity test rejects a minority of
    # realizations on its own, which is its job and predates this gate.
    assert stopped >= 30, stopped


def test_sokal_reports_a_constant_trace_as_unresolved() -> None:
    """A constant signal has no correlation time, so it must not report one.

    Callers read "resolved" as ``cut < rho.size``. The degenerate return placed
    the cut at lag zero, which reads as resolved for any non-empty ``rho`` --
    the opposite of what a zero-variance trace has shown them.
    """

    tau, cut, rho = sokal_autocorrelation_time(np.zeros(64), 0.05)

    assert tau == 0.0
    assert cut >= rho.size
