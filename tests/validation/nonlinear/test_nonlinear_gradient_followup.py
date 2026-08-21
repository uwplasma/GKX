from __future__ import annotations

import json

from support.paths import REPO_ROOT

import pytest

from gkx.diagnostics.nonlinear_gradient_statistics import (
    NonlinearGradientControlMeanGateConfig,
    NonlinearGradientVarianceReductionConfig,
    nonlinear_gradient_control_mean_gate,
    nonlinear_gradient_variance_reduction_plan,
)

ROOT = REPO_ROOT


def _ensemble(
    state: str, means: tuple[float, float, float] = (1.0, 1.1, 0.9)
) -> dict[str, object]:
    return {
        "kind": "nonlinear_window_ensemble_report",
        "passed": True,
        "n_reports": 3,
        "rows": [
            {
                "late_mean": value,
                "source_artifact": f"{state}_nonlinear_t900_n64_{label}_heat_flux_trace.csv",
                "summary_artifact": f"{state}_nonlinear_t900_n64_{label}_transport_window.json",
            }
            for value, label in zip(means, ("seed31", "seed32", "dt0p04"))
        ],
    }


def _artifact(
    *,
    response: float = 0.08,
    asymmetry: float = 0.30,
    uncertainty: float = 0.56,
    passed: bool = False,
) -> dict[str, object]:
    return {
        "kind": "nonlinear_turbulence_gradient_central_fd_gate",
        "claim_level": "production_long_window_nonlinear_turbulence_gradient_candidate",
        "parameter_name": "rbc_1_1",
        "passed": passed,
        "metrics": {
            "response_fraction": response,
            "fd_asymmetry_rel": asymmetry,
            "gradient_uncertainty_rel": uncertainty,
        },
        "source_ensembles": {
            "baseline": _ensemble("baseline"),
            "plus": _ensemble("plus"),
            "minus": _ensemble("minus"),
        },
    }


def test_variance_reduction_plan_quantifies_paired_seed_response() -> None:
    artifact = _artifact(response=0.032, asymmetry=0.044, uncertainty=1.81)
    source_ensembles = artifact["source_ensembles"]
    assert isinstance(source_ensembles, dict)
    values = {
        "baseline": {"seed31": 15.4, "seed32": 16.9, "seed33": 17.2, "dt0p04": 15.7},
        "plus": {"seed31": 18.1, "seed32": 15.6, "seed33": 15.5, "dt0p04": 14.9},
        "minus": {"seed31": 17.1, "seed32": 16.2, "seed33": 16.5, "dt0p04": 16.4},
    }
    for state, labeled_values in values.items():
        source_ensembles[state] = {
            "passed": state != "plus",
            "statistics": {
                "n_reports": 4,
                "mean_rel_spread": 0.196 if state == "plus" else 0.05,
                "combined_sem_rel": 0.05,
            },
            "rows": [
                {
                    "late_mean": value,
                    "source_artifact": f"{state}_nonlinear_t900_n64_{label}_heat_flux_trace.csv",
                }
                for label, value in labeled_values.items()
            ],
        }

    report = nonlinear_gradient_variance_reduction_plan(
        artifact,
        config=NonlinearGradientVarianceReductionConfig(max_extra_paired_seeds=1),
    )

    assert report["passed"] is False
    assert report["action"] == "estimate_control_mean_or_redesign_observable"
    assert report["summary"]["common_pair_count"] == 4
    assert report["summary"]["common_with_baseline_count"] == 4
    assert report["summary"]["paired_response_uncertainty_rel"] > 0.5
    assert (
        report["summary"]["best_control_variate"] == "plus_minus_midpoint_common_mode"
    )
    midpoint = report["control_variate_candidates"][1]
    assert midpoint["adjusted_response_uncertainty_rel"] < 0.5
    assert midpoint["control_sample_std"] > 0.0
    assert midpoint["adjusted_response_sample_std"] > 0.0
    assert "control_mean_not_independently_known" in midpoint["blockers"]
    assert report["variance_reduction"]["limiting_state"] == "plus"
    assert report["pair_rows"][0]["label"] == "dt0p04"

    allowed = nonlinear_gradient_variance_reduction_plan(
        artifact,
        config=NonlinearGradientVarianceReductionConfig(
            require_known_control_mean=False
        ),
    )
    assert allowed["action"] == "use_control_variate_response_estimator"
    assert allowed["passed"] is True

    validation_cases = [
        (
            "max_paired_response_uncertainty_rel",
            NonlinearGradientVarianceReductionConfig(
                max_paired_response_uncertainty_rel=0.0
            ),
        ),
        (
            "max_control_variate_uncertainty_rel",
            NonlinearGradientVarianceReductionConfig(
                max_control_variate_uncertainty_rel=0.0
            ),
        ),
        (
            "min_control_variate_sem_reduction",
            NonlinearGradientVarianceReductionConfig(
                min_control_variate_sem_reduction=-0.1
            ),
        ),
        (
            "sem_safety_factor",
            NonlinearGradientVarianceReductionConfig(sem_safety_factor=0.0),
        ),
        (
            "min_common_pairs",
            NonlinearGradientVarianceReductionConfig(min_common_pairs=0),
        ),
        (
            "max_extra_paired_seeds",
            NonlinearGradientVarianceReductionConfig(max_extra_paired_seeds=-1),
        ),
    ]
    for message, config in validation_cases:
        with pytest.raises(ValueError, match=message):
            nonlinear_gradient_variance_reduction_plan(artifact, config=config)


def _control_ensemble(state: str, *, passed: bool = True) -> dict[str, object]:
    rows = []
    for idx, seed in enumerate(range(34, 55)):
        control = 16.28 + 0.02 * ((idx % 3) - 1)
        response = -0.52 + 0.01 * ((idx % 5) - 2)
        if state == "plus":
            value = control + 0.5 * response
        else:
            value = control - 0.5 * response
        rows.append(
            {
                "late_mean": value,
                "source_artifact": f"{state}_seed{seed}_heat_flux_trace.csv",
                "summary_artifact": f"{state}_seed{seed}_transport_window.json",
                "variant": {"seed": seed, "timestep": 0.05},
            }
        )
    return {
        "kind": "nonlinear_window_ensemble_report",
        "passed": passed,
        "n_reports": len(rows),
        "rows": rows,
        "statistics": {
            "n_reports": len(rows),
            "ensemble_mean": sum(float(row["late_mean"]) for row in rows) / len(rows),
            "combined_sem": 0.02,
            "combined_sem_rel": 0.001,
            "mean_rel_spread": 0.002,
        },
    }


def test_control_mean_gate_combines_independent_control_uncertainty() -> None:
    artifact = json.loads(
        (
            ROOT
            / "docs"
            / "_static"
            / "qa_ess_zbs10_rel7p5_nonlinear_gradient_zbs_1_0_central_fd_gradient_gate.json"
        ).read_text(encoding="utf-8")
    )
    variance = nonlinear_gradient_variance_reduction_plan(artifact)

    gate = nonlinear_gradient_control_mean_gate(
        variance,
        plus_ensemble=_control_ensemble("plus"),
        minus_ensemble=_control_ensemble("minus"),
        plus_path="plus.json",
        minus_path="minus.json",
    )

    assert gate["passed"] is True
    assert gate["candidate_name"] == "plus_minus_midpoint_common_mode"
    assert gate["summary"]["common_pair_count"] == 21
    assert gate["summary"]["combined_response_uncertainty_rel"] < 0.5
    assert gate["pair_rows"][0]["label"] == "seed34"

    blocked = nonlinear_gradient_control_mean_gate(
        variance,
        plus_ensemble=_control_ensemble("plus", passed=False),
        minus_ensemble=_control_ensemble("minus"),
    )
    assert blocked["passed"] is False
    assert "plus_control_ensemble_failed" in blocked["blockers"]

    validation_cases = [
        (
            "target_response_uncertainty_rel",
            NonlinearGradientControlMeanGateConfig(target_response_uncertainty_rel=0.0),
        ),
        (
            "min_control_mean_pairs",
            NonlinearGradientControlMeanGateConfig(min_control_mean_pairs=0),
        ),
    ]
    for message, config in validation_cases:
        with pytest.raises(ValueError, match=message):
            nonlinear_gradient_control_mean_gate(
                variance,
                plus_ensemble=_control_ensemble("plus"),
                minus_ensemble=_control_ensemble("minus"),
                config=config,
            )


def test_control_mean_gate_matches_seed_from_artifact_basename() -> None:
    artifact = json.loads(
        (
            ROOT
            / "docs"
            / "_static"
            / "qa_ess_zbs10_rel7p5_nonlinear_gradient_zbs_1_0_central_fd_gradient_gate.json"
        ).read_text(encoding="utf-8")
    )
    variance = nonlinear_gradient_variance_reduction_plan(artifact)
    plus = _control_ensemble("plus")
    minus = _control_ensemble("minus")
    for ensemble in (plus, minus):
        for row in ensemble["rows"]:
            row.pop("variant", None)
            row["source_artifact"] = f"/tmp/interim_seed34_42/{row['source_artifact']}"
            row["summary_artifact"] = (
                f"/tmp/interim_seed34_42/{row['summary_artifact']}"
            )

    gate = nonlinear_gradient_control_mean_gate(
        variance,
        plus_ensemble=plus,
        minus_ensemble=minus,
    )

    assert gate["passed"] is True
    assert gate["summary"]["common_pair_count"] == 21
    assert gate["pair_rows"][1]["label"] == "seed35"
