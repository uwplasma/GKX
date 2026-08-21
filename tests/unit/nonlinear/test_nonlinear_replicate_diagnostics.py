from __future__ import annotations

from gkx.diagnostics.nonlinear_replicates import (
    nonlinear_replicate_spread_report,
)


def test_replicate_spread_report_identifies_mixed_seed_timestep_spread() -> None:
    report = nonlinear_replicate_spread_report(
        [
            {
                "case": "qa_ess_nonlinear_gradient_plus_delta_t900_ensemble",
                "passed": False,
                "statistics": {
                    "ensemble_mean": 10.0,
                    "mean_rel_spread": 0.30,
                    "combined_sem_rel": 0.04,
                },
                "config": {"max_mean_rel_spread": 0.15},
                "rows": [
                    {
                        "index": 0,
                        "late_mean": 10.0,
                        "sem": 0.2,
                        "source_artifact": "case_seed31_heat_flux_trace.csv",
                        "passed": True,
                        "promotion_ready": True,
                    },
                    {
                        "index": 1,
                        "late_mean": 11.5,
                        "sem": 0.2,
                        "source_artifact": "case_seed32_heat_flux_trace.csv",
                        "passed": True,
                        "promotion_ready": True,
                    },
                    {
                        "index": 2,
                        "late_mean": 8.5,
                        "sem": 0.2,
                        "source_artifact": "case_dt0p04_heat_flux_trace.csv",
                        "passed": True,
                        "promotion_ready": True,
                    },
                ],
            }
        ]
    )

    assert report["passed"] is False
    assert report["summary"]["failed_states"] == ["plus_delta"]
    assert report["state_rows"][0]["classification"] == "mixed_seed_timestep_spread"
    assert report["state_rows"][0]["high_variant_axis"] == "seed"
    assert report["state_rows"][0]["low_variant_axis"] == "timestep"
    assert (
        "Do not add same-bracket replicas blindly"
        in report["state_rows"][0]["recommendation"]
    )


def test_replicate_spread_report_passes_with_small_seed_spread() -> None:
    report = nonlinear_replicate_spread_report(
        [
            {
                "case": "qa_ess_nonlinear_gradient_baseline_t900_ensemble",
                "passed": True,
                "statistics": {
                    "ensemble_mean": 10.0,
                    "mean_rel_spread": 0.02,
                    "combined_sem_rel": 0.04,
                },
                "rows": [
                    {
                        "index": 0,
                        "late_mean": 9.9,
                        "source_artifact": "case_seed31_heat_flux_trace.csv",
                        "passed": True,
                        "promotion_ready": True,
                    },
                    {
                        "index": 1,
                        "late_mean": 10.1,
                        "source_artifact": "case_seed32_heat_flux_trace.csv",
                        "passed": True,
                        "promotion_ready": True,
                    },
                ],
            }
        ]
    )

    assert report["passed"] is True
    assert report["state_rows"][0]["classification"] == "passed_replicate_spread_gate"


def test_replicate_spread_report_preserves_joint_seed_timestep_labels() -> None:
    report = nonlinear_replicate_spread_report(
        [
            {
                "case": "qa_ess_nonlinear_gradient_plus_delta_t900_ensemble",
                "passed": False,
                "statistics": {"ensemble_mean": 10.0, "mean_rel_spread": 0.30},
                "rows": [
                    {
                        "index": 0,
                        "late_mean": 11.5,
                        "source_artifact": "case_seed32_dt0p04_heat_flux_trace.csv",
                    },
                    {
                        "index": 1,
                        "late_mean": 8.5,
                        "source_artifact": "case_seed22_dt0p05_heat_flux_trace.csv",
                    },
                ],
            }
        ]
    )

    assert report["replicate_rows"][0]["variant_label"] == "seed32_dt0p04"
    assert report["replicate_rows"][0]["variant_axis"] == "seed_timestep"
