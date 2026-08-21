from __future__ import annotations

import json
from pathlib import Path

from support.paths import load_release_tool

import numpy as np
import pytest

import gkx.diagnostics.nonlinear_gradient_evidence as evidence
import gkx.diagnostics.metadata as evidence_metadata
import gkx.diagnostics.nonlinear_replicates as evidence_replicates
import gkx.diagnostics.transport as evidence_transport
from gkx.diagnostics.nonlinear_gradient_evidence import (
    NonlinearTurbulenceGradientBracketSweepConfig,
    NonlinearTurbulenceGradientCandidateRankingConfig,
    NonlinearTurbulenceGradientFiniteDifferenceConfig,
    NonlinearTurbulenceGradientGapConfig,
    classify_gradient_artifact,
    load_json_artifact,
    nonlinear_turbulence_gradient_bracket_sweep_report,
    nonlinear_turbulence_gradient_candidate_ranking_report,
    nonlinear_turbulence_gradient_evidence_gap_report,
    nonlinear_turbulence_gradient_evidence_report,
    nonlinear_turbulence_gradient_finite_difference_report,
    summarize_window_evidence,
)
from gkx.diagnostics.transport_windows import (
    NonlinearWindowConvergenceConfig,
    nonlinear_window_convergence_report,
)


def test_nonlinear_gradient_evidence_facade_reexports_core_contracts() -> None:
    assert (
        evidence.NON_PRODUCTION_SCOPE_MARKERS
        is evidence_metadata.NON_PRODUCTION_SCOPE_MARKERS
    )
    assert (
        evidence.NonlinearTurbulenceGradientEvidenceConfig
        is evidence_metadata.NonlinearTurbulenceGradientEvidenceConfig
    )
    assert (
        evidence.NonlinearTurbulenceGradientFiniteDifferenceConfig
        is evidence_metadata.NonlinearTurbulenceGradientFiniteDifferenceConfig
    )
    assert (
        evidence.classify_gradient_artifact
        is evidence_metadata.classify_gradient_artifact
    )


def test_nonlinear_gradient_evidence_exports_actual_owner_contracts() -> None:
    assert (
        evidence.classify_gradient_artifact
        is evidence_metadata.classify_gradient_artifact
    )
    assert evidence._metric_margin is evidence_metadata._metric_margin
    assert evidence._bracket_sweep_row is evidence_metadata._bracket_sweep_row
    assert (
        evidence.summarize_window_evidence
        is evidence_replicates.summarize_window_evidence
    )
    assert (
        evidence.nonlinear_turbulence_gradient_finite_difference_report
        is evidence_transport.nonlinear_turbulence_gradient_finite_difference_report
    )
    for name in (
        "nonlinear_turbulence_gradient_bracket_sweep_report",
        "nonlinear_turbulence_gradient_candidate_ranking_report",
        "nonlinear_turbulence_gradient_evidence_gap_report",
        "nonlinear_turbulence_gradient_evidence_report",
    ):
        assert name in evidence.__all__
        assert callable(getattr(evidence, name))


def test_consolidated_diagnostic_owners_have_complete_export_contracts() -> None:
    assert {
        "ArrayLike",
        "ResolvedDiagnostics",
        "SimulationDiagnostics",
        "NonlinearTurbulenceGradientEvidenceConfig",
        "_artifact_passed",
        "_bracket_sweep_row",
        "_explicit_true",
        "_metric_margin",
        "classify_gradient_artifact",
    } <= set(evidence_metadata.__all__)
    assert {
        "heat_flux_species",
        "heat_flux_total",
        "particle_flux_species",
        "particle_flux_total",
        "turbulent_heating_species",
        "turbulent_heating_total",
        "nonlinear_turbulence_gradient_finite_difference_report",
    } <= set(evidence_transport.__all__)


def _load_tool_module():
    return load_release_tool("check_nonlinear_optimization_gates")


def _window_report(offset: float, *, case: str) -> dict[str, object]:
    t = np.linspace(0.0, 240.0, 241)
    heat = 8.0 + offset + 0.02 * np.sin(2.0 * np.pi * t / 12.0)
    return nonlinear_window_convergence_report(
        t,
        heat,
        case=case,
        source_artifact=f"{case}.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
            min_blocks=4,
            max_running_mean_rel_drift=0.01,
            max_terminal_mean_rel_delta=0.01,
            max_sem_rel=0.02,
        ),
    )


def _production_gradient() -> dict[str, object]:
    return {
        "kind": "nonlinear_turbulence_gradient_finite_difference_audit",
        "claim_level": "production_long_window_nonlinear_turbulence_gradient_evidence",
        "passed": True,
        "production_nonlinear_window_gradient_gate": True,
        "gradient": {
            "central": 1.25,
            "response_fraction": 0.08,
            "asymmetry_rel": 0.12,
        },
        "conditioning": {
            "condition_number": 42.0,
        },
        "uncertainty": {
            "gradient_sem_rel": 0.18,
        },
    }


def _ensemble(
    mean: float,
    sem: float = 0.02,
    *,
    passed: bool = True,
    n_reports: int = 3,
    rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": "nonlinear_window_ensemble_report",
        "case": "matched_replicated_window",
        "passed": passed,
        "statistics": {
            "ensemble_mean": float(mean),
            "combined_sem": float(sem),
            "combined_sem_rel": abs(float(sem) / float(mean)),
            "mean_rel_spread": 0.02,
            "n_reports": n_reports,
        },
    }
    if rows is not None:
        payload["rows"] = rows
    return payload


def test_startup_fd_artifact_is_recorded_but_not_promoted() -> None:
    artifact = {
        "kind": "nonlinear_startup_window_finite_difference_audit",
        "claim_level": "startup_transient_nonlinear_plumbing_fd_audit_not_transport_average",
        "passed": True,
        "transport_average_gate": False,
        "production_nonlinear_window_gradient_gate": False,
        "metrics": {
            "central_fd_dq_dtprim": 2.0,
            "response_fraction": 0.2,
            "derivative_asymmetry": 0.0,
        },
    }

    row = classify_gradient_artifact(artifact)

    assert row["artifact_passed"] is True
    assert row["qualifies_for_production_turbulence_gradient"] is False
    assert row["evidence_class"] == "startup_or_reduced_window_fd_not_production"
    assert "startup" in row["scope_blockers"]
    assert "transport_average_gate_false" in row["scope_blockers"]


def test_gradient_classifier_rejects_string_pass_flags() -> None:
    artifact = _production_gradient()
    artifact["passed"] = "true"
    artifact["gate_report"] = {"passed": "true"}

    row = classify_gradient_artifact(artifact)

    assert row["artifact_passed"] is False
    assert row["qualifies_for_production_turbulence_gradient"] is False
    assert "artifact_passed" in {
        gate["metric"] for gate in row["gates"] if not gate["passed"]
    }


def test_reduced_estimator_gradient_does_not_promote_even_with_replicates() -> None:
    reduced_gradient = {
        "kind": "mode21_vmec_boozer_nonlinear_window_gradient_gate",
        "claim_scope": "reduced nonlinear-window estimator gradient",
        "passed": True,
        "production_nonlinear_window_gradient_gate": False,
        "objective_gates": [
            {
                "objective": "nonlinear_window_heat_flux_mean",
                "finite_difference": 1.0,
                "implicit": 1.0,
            }
        ],
        "conditioning": {
            "condition_number": 10.0,
            "response_fraction": 0.1,
            "asymmetry_rel": 0.1,
        },
        "uncertainty": {
            "gradient_sem_rel": 0.1,
        },
    }
    windows = [
        _window_report(-0.01, case="seed_1"),
        _window_report(0.01, case="seed_2"),
    ]

    report = nonlinear_turbulence_gradient_evidence_report(
        reduced_gradient,
        window_artifacts=windows,
    )

    assert report["passed"] is False
    assert report["window_evidence"]["passed"] is True
    assert (
        report["gradient_artifact"]["qualifies_for_production_turbulence_gradient"]
        is False
    )
    assert report["blockers"] == ["production_gradient_artifact"]
    assert report["evidence_gap"]["promotion_blocked"] is True
    assert report["evidence_gap"]["current_window_evidence_passed"] is True
    assert (
        report["evidence_gap"]["required_campaign"]["required_runs"][0]["state"]
        == "minus_delta"
    )
    assert (
        report["evidence_gap"]["required_campaign"]["required_runs"][1]["state"]
        == "baseline"
    )
    assert (
        report["evidence_gap"]["required_campaign"]["required_runs"][2]["state"]
        == "plus_delta"
    )
    audit = report["evidence_gap"]["required_campaign"]["finite_difference_audit"]
    assert (
        audit["acceptance_gates"]["production_nonlinear_window_gradient_gate"] is True
    )
    assert "central_gradient" in audit["required_metrics"]


def test_production_gradient_can_use_derived_replicated_window_summaries() -> None:
    windows = [
        _window_report(-0.02, case="seed_1"),
        _window_report(0.0, case="seed_2"),
        _window_report(0.02, case="dt_variant"),
    ]

    report = nonlinear_turbulence_gradient_evidence_report(
        _production_gradient(),
        window_artifacts=windows,
    )

    assert report["passed"] is True
    assert report["production_nonlinear_window_gradient_gate"] is True
    assert report["evidence_gap"]["passed"] is True
    assert report["evidence_gap"]["promotion_blocked"] is False
    assert (
        report["window_evidence"]["derived_ensemble"]["source"]
        == "derived_from_window_summaries"
    )
    assert (
        report["gradient_artifact"]["conditioning"]["gradient_uncertainty_rel"] == 0.18
    )


def test_gap_report_names_custom_paired_parameter_campaign() -> None:
    report = {
        "passed": False,
        "blockers": ["production_gradient_artifact"],
        "gradient_artifact": {
            "path": "gradient.json",
            "evidence_class": "startup_or_reduced_window_fd_not_production",
        },
        "window_evidence": {
            "passed": True,
            "ensemble_rows": [
                {"qualifies_for_replicated_long_window_uncertainty": True},
            ],
        },
    }

    gap = nonlinear_turbulence_gradient_evidence_gap_report(
        report,
        gap_config=NonlinearTurbulenceGradientGapConfig(
            case_slug="qa_ess_dqi",
            parameter_name="rbc_1_0",
            perturbation_fraction=0.02,
            analysis_tmin=500.0,
            analysis_tmax=900.0,
            minimum_tmax=900.0,
            minimum_grid="n96x96x64x48x48",
            replicate_labels=("seed41", "seed42", "dt0p04"),
        ),
    )

    assert gap["passed"] is False
    assert gap["promotion_blocked"] is True
    assert gap["qualifying_window_ensemble_count"] == 1
    assert gap["missing_evidence"][0]["current_artifact_path"] == "gradient.json"
    required_runs = gap["required_campaign"]["required_runs"]
    assert [row["state"] for row in required_runs] == [
        "minus_delta",
        "baseline",
        "plus_delta",
    ]
    assert required_runs[0]["parameter_multiplier"] == pytest.approx(0.98)
    assert required_runs[2]["parameter_multiplier"] == pytest.approx(1.02)
    assert required_runs[0]["run_contract"]["analysis_window"] == [500.0, 900.0]
    assert required_runs[0]["run_contract"]["minimum_grid"] == "n96x96x64x48x48"
    assert required_runs[0]["replicates"] == ["seed41", "seed42", "dt0p04"]


def test_production_gradient_fails_closed_without_uncertainty() -> None:
    gradient = _production_gradient()
    gradient.pop("uncertainty")
    windows = [
        _window_report(-0.02, case="seed_1"),
        _window_report(0.02, case="seed_2"),
    ]

    report = nonlinear_turbulence_gradient_evidence_report(
        gradient,
        window_artifacts=windows,
    )

    gradient_gates = {
        gate["metric"]: gate["passed"] for gate in report["gradient_artifact"]["gates"]
    }
    assert report["passed"] is False
    assert gradient_gates["gradient_uncertainty_bounded"] is False


def test_unscoped_nested_passed_artifact_with_bad_numbers_stays_blocked() -> None:
    artifact = {
        "kind": "long_window_gradient_candidate",
        "gate_report": {"passed": True},
        "objective_gates": [
            "not-a-row",
            {"finite_difference": "not-a-number", "implicit": "inf"},
        ],
        "gradient": {"central": "not-a-number"},
        "metrics": {"response_fraction": float("nan"), "derivative_asymmetry": "bad"},
        "conditioning": {"condition_number": "bad"},
        "uncertainty": {"sem_rel": None},
    }

    row = classify_gradient_artifact(artifact)

    assert row["artifact_passed"] is True
    assert row["evidence_class"] == "unscoped_gradient_or_fd_artifact_not_production"
    assert row["qualifies_for_production_turbulence_gradient"] is False
    assert row["conditioning"]["central_gradient"] is None
    assert {gate["metric"]: gate["passed"] for gate in row["gates"]}[
        "finite_gradient_estimate"
    ] is False


def test_explicit_nonlinear_turbulence_gradient_flag_can_promote_scope() -> None:
    artifact = _production_gradient()
    artifact.pop("production_nonlinear_window_gradient_gate")
    artifact["nonlinear_turbulence_gradient_gate"] = True

    row = classify_gradient_artifact(artifact)

    assert row["explicit_production_scope"] is True
    assert (
        row["evidence_class"] == "production_long_window_turbulence_gradient_candidate"
    )
    assert row["qualifies_for_production_turbulence_gradient"] is True


def test_canonical_gradient_uncertainty_rel_is_accepted_for_classification() -> None:
    artifact = {
        "kind": "external_nonlinear_turbulence_gradient_gate",
        "claim_level": "production_long_window_nonlinear_turbulence_gradient_evidence",
        "passed": True,
        "production_nonlinear_window_gradient_gate": True,
        "metrics": {
            "central_gradient": 1.0,
            "response_fraction": 0.1,
            "fd_asymmetry_rel": 0.1,
            "fd_condition_number": 10.0,
            "gradient_uncertainty_rel": 0.2,
        },
    }

    row = classify_gradient_artifact(artifact)

    assert row["conditioning"]["gradient_uncertainty_rel"] == pytest.approx(0.2)
    assert row["qualifies_for_production_turbulence_gradient"] is True


def test_long_window_fd_gate_promotes_only_resolved_replicated_gradient() -> None:
    report = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(9.0),
        baseline=_ensemble(10.0),
        plus=_ensemble(11.0),
        delta_parameter=0.05,
        parameter_name="rbc_1_0",
    )

    assert report["passed"] is True
    assert report["metrics"]["central_gradient"] == pytest.approx(20.0)
    assert report["metrics"]["response_fraction"] == pytest.approx(0.2)
    assert report["production_nonlinear_window_gradient_gate"] is True
    classified = classify_gradient_artifact(report)
    assert classified["qualifies_for_production_turbulence_gradient"] is True
    evidence = nonlinear_turbulence_gradient_evidence_report(
        report,
        window_artifacts=[_ensemble(10.0), _ensemble(10.1)],
    )
    assert evidence["passed"] is True


def test_long_window_fd_gate_reports_paired_replicate_diagnostics() -> None:
    rows_baseline = [
        {"source_artifact": "case_seed31_heat_flux_trace.csv", "late_mean": 10.0},
        {"source_artifact": "case_seed32_heat_flux_trace.csv", "late_mean": 10.2},
    ]
    rows_minus = [
        {"source_artifact": "case_seed31_heat_flux_trace.csv", "late_mean": 9.0},
        {"source_artifact": "case_seed32_heat_flux_trace.csv", "late_mean": 9.3},
    ]
    rows_plus = [
        {"source_artifact": "case_seed31_heat_flux_trace.csv", "late_mean": 11.0},
        {"source_artifact": "case_seed32_heat_flux_trace.csv", "late_mean": 10.7},
    ]

    report = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(9.15, rows=rows_minus),
        baseline=_ensemble(10.1, rows=rows_baseline),
        plus=_ensemble(10.85, rows=rows_plus),
        delta_parameter=0.05,
        parameter_name="rbc_1_0",
    )

    diagnostics = report["paired_replicate_diagnostics"]
    assert diagnostics["claim_level"] == "diagnostic_only_not_a_production_gate"
    assert diagnostics["common_plus_minus_labels"] == ["seed31", "seed32"]
    assert diagnostics["common_all_state_labels"] == ["seed31", "seed32"]
    assert diagnostics["n_pairs"] == 2
    assert diagnostics["central_gradient_mean"] == pytest.approx(17.0)
    assert diagnostics["paired_rows"][0]["central_gradient"] == pytest.approx(20.0)
    assert diagnostics["paired_rows"][1]["central_gradient"] == pytest.approx(14.0)
    assert diagnostics["same_sign_fraction"] == pytest.approx(1.0)


def test_long_window_fd_gate_fails_when_response_is_buried_in_uncertainty() -> None:
    report = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(9.99, sem=0.5),
        baseline=_ensemble(10.0, sem=0.5),
        plus=_ensemble(10.01, sem=0.5),
        delta_parameter=0.05,
        parameter_name="rbc_1_0",
        config=NonlinearTurbulenceGradientFiniteDifferenceConfig(
            min_fd_response_fraction=0.03,
            max_gradient_uncertainty_rel=0.5,
        ),
    )

    gates = {gate["metric"]: gate["passed"] for gate in report["gates"]}
    assert report["passed"] is False
    assert gates["fd_response_resolved"] is False
    assert gates["gradient_uncertainty_bounded"] is False
    classified = classify_gradient_artifact(report)
    assert (
        classified["evidence_class"]
        == "production_long_window_turbulence_gradient_candidate"
    )
    assert classified["qualifies_for_production_turbulence_gradient"] is False


def test_long_window_fd_gate_requires_ensemble_artifacts() -> None:
    baseline = _ensemble(10.0)
    plus = _ensemble(11.0)
    minus = _ensemble(9.0)
    baseline["kind"] = "nonlinear_window_convergence_report"
    plus["kind"] = "nonlinear_window_convergence_report"
    minus["kind"] = "nonlinear_window_convergence_report"

    report = nonlinear_turbulence_gradient_finite_difference_report(
        minus=minus,
        baseline=baseline,
        plus=plus,
        delta_parameter=0.05,
        parameter_name="rbc_1_0",
    )

    assert report["passed"] is False
    assert report["production_nonlinear_window_gradient_gate"] is False
    assert {
        "minus_ensemble_kind",
        "baseline_ensemble_kind",
        "plus_ensemble_kind",
    }.issubset(set(report["blockers"]))
    classified = classify_gradient_artifact(report)
    assert classified["qualifies_for_production_turbulence_gradient"] is False


def test_long_window_fd_gate_requires_replicated_source_ensembles() -> None:
    report = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(9.0, n_reports=1),
        baseline=_ensemble(10.0, n_reports=1),
        plus=_ensemble(11.0, n_reports=1),
        delta_parameter=0.05,
        parameter_name="rbc_1_0",
    )

    assert report["passed"] is False
    assert report["production_nonlinear_window_gradient_gate"] is False
    assert {
        "minus_ensemble_replicated",
        "baseline_ensemble_replicated",
        "plus_ensemble_replicated",
    }.issubset(set(report["blockers"]))


def test_long_window_fd_gate_fails_closed_for_nonfinite_window_statistics() -> None:
    baseline = _ensemble(10.0)
    plus = _ensemble(11.0)
    minus = _ensemble(9.0)
    plus["statistics"]["ensemble_mean"] = "not-a-number"
    plus["statistics"]["combined_sem"] = "not-a-number"

    report = nonlinear_turbulence_gradient_finite_difference_report(
        minus=minus,
        baseline=baseline,
        plus=plus,
        delta_parameter=0.05,
        parameter_name="rbc_1_0",
    )

    assert report["passed"] is False
    assert report["production_nonlinear_window_gradient_gate"] is False
    assert report["nonlinear_turbulence_gradient_gate"] is False
    assert report["metrics"]["central_gradient"] is None
    assert report["metrics"]["gradient_uncertainty_rel"] is None
    assert {"finite_window_means", "finite_window_uncertainties"}.issubset(
        set(report["blockers"])
    )
    classified = classify_gradient_artifact(report)
    assert classified["qualifies_for_production_turbulence_gradient"] is False


def test_long_window_fd_gate_rejects_nonpositive_delta() -> None:
    with pytest.raises(ValueError, match="delta_parameter must be finite and positive"):
        nonlinear_turbulence_gradient_finite_difference_report(
            minus=_ensemble(9.0),
            baseline=_ensemble(10.0),
            plus=_ensemble(11.0),
            delta_parameter=0.0,
            parameter_name="rbc_1_0",
        )


def test_candidate_ranking_selects_profile_gradient_when_failures_are_complementary() -> (
    None
):
    local_but_noisy = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(16.94, sem=0.67),
        baseline=_ensemble(16.31, sem=0.45),
        plus=_ensemble(15.82, sem=0.53),
        delta_parameter=0.0015890833407568477,
        parameter_name="zbs_1_0",
    )
    quiet_but_nonlocal = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(14.67, sem=0.27),
        baseline=_ensemble(16.31, sem=0.45),
        plus=_ensemble(17.13, sem=0.48),
        delta_parameter=0.004699871690217756,
        parameter_name="zbs_1_1",
    )

    report = nonlinear_turbulence_gradient_candidate_ranking_report(
        [local_but_noisy, quiet_but_nonlocal],
        labels=["local_but_noisy", "quiet_but_nonlocal"],
    )

    assert report["passed"] is False
    assert "least-squares/profile-gradient" in report["recommendation"]
    assert report["candidates"][0]["rank"] == 1
    assert report["candidates"][0]["score"] >= report["candidates"][1]["score"]
    assert any(
        "more replicas" in row["next_action"] or "smaller bracket" in row["next_action"]
        for row in report["candidates"]
    )


def test_candidate_ranking_context_reports_completed_overdetermined_followup() -> None:
    local_but_noisy = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(16.08, sem=0.59),
        baseline=_ensemble(15.52, sem=0.45),
        plus=_ensemble(14.74, sem=0.46),
        delta_parameter=0.00301141510243,
        parameter_name="rbc_1_1",
    )
    quiet_but_nonlocal = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(14.67, sem=0.27),
        baseline=_ensemble(16.31, sem=0.45),
        plus=_ensemble(17.13, sem=0.48),
        delta_parameter=0.004699871690217756,
        parameter_name="zbs_1_1",
    )

    report = nonlinear_turbulence_gradient_candidate_ranking_report(
        [local_but_noisy, quiet_but_nonlocal],
        config=NonlinearTurbulenceGradientCandidateRankingConfig(
            campaign_context="overdetermined_followup"
        ),
    )

    assert report["passed"] is False
    assert "overdetermined follow-up completed" in report["recommendation"]
    assert "keep the nonlinear-gradient claim fail-closed" in report["recommendation"]


def test_candidate_ranking_can_promote_a_passing_production_candidate() -> None:
    passing = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(9.0, sem=0.02),
        baseline=_ensemble(10.0, sem=0.02),
        plus=_ensemble(11.0, sem=0.02),
        delta_parameter=0.05,
        parameter_name="profile_gradient",
    )

    report = nonlinear_turbulence_gradient_candidate_ranking_report(
        [passing],
        config=NonlinearTurbulenceGradientCandidateRankingConfig(),
    )

    assert report["passed"] is True
    assert report["promotion_ready_candidate_count"] == 1
    assert report["best_candidate"]["parameter_name"] == "profile_gradient"


def test_candidate_ranking_handles_metadata_errors_and_empty_screens() -> None:
    assert nonlinear_turbulence_gradient_candidate_ranking_report([])[
        "recommendation"
    ].startswith("screen new profile-gradient")

    with pytest.raises(ValueError, match="paths length"):
        nonlinear_turbulence_gradient_candidate_ranking_report(
            [_production_gradient()],
            paths=["first.json", "second.json"],
        )
    with pytest.raises(ValueError, match="labels length"):
        nonlinear_turbulence_gradient_candidate_ranking_report(
            [_production_gradient()],
            labels=["first", "second"],
        )


def test_candidate_ranking_distinguishes_single_failure_modes() -> None:
    local_but_noisy = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(9.0, sem=5.0),
        baseline=_ensemble(10.0, sem=5.0),
        plus=_ensemble(11.0, sem=5.0),
        delta_parameter=0.05,
        parameter_name="local_but_noisy",
    )
    quiet_but_nonlocal = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(9.0, sem=0.02),
        baseline=_ensemble(9.6, sem=0.02),
        plus=_ensemble(11.0, sem=0.02),
        delta_parameter=0.05,
        parameter_name="quiet_but_nonlocal",
        config=NonlinearTurbulenceGradientFiniteDifferenceConfig(
            max_fd_asymmetry_rel=0.5,
        ),
    )

    noisy_report = nonlinear_turbulence_gradient_candidate_ranking_report(
        [local_but_noisy]
    )
    nonlocal_report = nonlinear_turbulence_gradient_candidate_ranking_report(
        [quiet_but_nonlocal]
    )

    assert "extend statistical power" in noisy_report["recommendation"]
    assert "reduce bracket size" in nonlocal_report["recommendation"]
    assert "more replicas" in noisy_report["candidates"][0]["next_action"]
    assert "smaller bracket" in nonlocal_report["candidates"][0]["next_action"]


def test_candidate_ranking_flags_bad_conditioning_and_unscoped_artifacts() -> None:
    ill_conditioned = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(0.09, sem=1.0e-4),
        baseline=_ensemble(0.10, sem=1.0e-4),
        plus=_ensemble(0.11, sem=1.0e-4),
        delta_parameter=0.05,
        parameter_name="near_null_response",
    )
    unscoped = {
        "kind": "legacy_gradient_summary",
        "passed": True,
        "metrics": {
            "central_gradient": 1.0,
            "response_fraction": 0.1,
            "fd_asymmetry_rel": 0.1,
            "fd_condition_number": 2.0,
            "gradient_uncertainty_rel": 0.1,
        },
    }

    report = nonlinear_turbulence_gradient_candidate_ranking_report(
        [ill_conditioned, unscoped],
        config=NonlinearTurbulenceGradientCandidateRankingConfig(
            max_fd_condition_number=5.0,
        ),
    )

    assert report["passed"] is False
    by_name = {row["parameter_name"]: row for row in report["candidates"]}
    assert by_name["near_null_response"]["next_action"].startswith(
        "choose a better-conditioned"
    )
    assert (
        by_name["candidate_1"]["evidence_class"]
        == "unscoped_gradient_or_fd_artifact_not_production"
    )
    assert by_name["candidate_1"]["failed_gates"] == [
        "explicit_production_long_window_scope"
    ]


def test_bracket_sweep_blocks_repeating_unstable_same_bracket() -> None:
    rows_baseline = [
        {"source_artifact": "case_seed31.csv", "late_mean": 15.5},
        {"source_artifact": "case_seed32.csv", "late_mean": 15.15},
        {"source_artifact": "case_seed33.csv", "late_mean": 15.55},
    ]
    rows_minus = [
        {"source_artifact": "case_seed31.csv", "late_mean": 15.0},
        {"source_artifact": "case_seed32.csv", "late_mean": 14.8},
        {"source_artifact": "case_seed33.csv", "late_mean": 15.4},
    ]
    rows_plus = [
        {"source_artifact": "case_seed31.csv", "late_mean": 16.0},
        {"source_artifact": "case_seed32.csv", "late_mean": 15.5},
        {"source_artifact": "case_seed33.csv", "late_mean": 15.7},
    ]
    artifact = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(15.0667, sem=0.5, rows=rows_minus),
        baseline=_ensemble(15.4, sem=0.5, rows=rows_baseline),
        plus=_ensemble(15.7333, sem=0.5, rows=rows_plus),
        delta_parameter=0.05,
        parameter_name="zbs_1_0",
        config=NonlinearTurbulenceGradientFiniteDifferenceConfig(
            min_fd_response_fraction=0.03,
            max_gradient_uncertainty_rel=0.5,
            max_fd_asymmetry_rel=1.0,
        ),
    )

    report = nonlinear_turbulence_gradient_bracket_sweep_report(
        [artifact],
        labels=["zbs_1_0_rel5_seed3"],
        config=NonlinearTurbulenceGradientBracketSweepConfig(
            max_fd_asymmetry_rel=1.0,
            max_repeated_bracket_uncertainty_rel=0.2,
            min_repeated_bracket_same_sign_fraction=1.0,
        ),
    )

    assert report["passed"] is False
    assert "do not add replicas at the same bracket" in report["recommendation"]
    bracket = report["brackets"][0]
    assert bracket["metrics"]["paired_gradient_uncertainty_rel"] is not None
    assert bracket["metrics"]["repeated_bracket_stable"] is False


def test_bracket_sweep_flags_resolved_gradient_sign_change() -> None:
    negative = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(16.9, sem=0.8),
        baseline=_ensemble(16.3, sem=0.8),
        plus=_ensemble(15.8, sem=0.8),
        delta_parameter=0.0016,
        parameter_name="zbs_1_0",
    )
    positive = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(15.8, sem=0.8),
        baseline=_ensemble(16.0, sem=0.8),
        plus=_ensemble(16.4, sem=0.8),
        delta_parameter=0.0010,
        parameter_name="zbs_1_0",
        config=NonlinearTurbulenceGradientFiniteDifferenceConfig(
            max_fd_asymmetry_rel=1.0,
        ),
    )

    report = nonlinear_turbulence_gradient_bracket_sweep_report(
        [negative, positive],
        config=NonlinearTurbulenceGradientBracketSweepConfig(
            max_fd_asymmetry_rel=1.0,
        ),
    )

    assert report["passed"] is False
    assert "change central-gradient sign" in report["recommendation"]


def test_bracket_sweep_promotes_only_passing_long_window_bracket() -> None:
    small = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(9.9, sem=0.4),
        baseline=_ensemble(10.0, sem=0.4),
        plus=_ensemble(10.1, sem=0.4),
        delta_parameter=0.01,
        parameter_name="profile_gradient",
    )
    passing = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(9.0, sem=0.02),
        baseline=_ensemble(10.0, sem=0.02),
        plus=_ensemble(11.0, sem=0.02),
        delta_parameter=0.05,
        parameter_name="profile_gradient",
    )

    report = nonlinear_turbulence_gradient_bracket_sweep_report(
        [passing, small],
        labels=["rel5", "rel1"],
    )

    assert report["passed"] is True
    assert report["promotion_ready_bracket_count"] == 1
    assert [row["delta_parameter"] for row in report["brackets"]] == [0.01, 0.05]
    assert "smallest passing delta is 0.05" in report["recommendation"]


def test_bracket_sweep_fail_closed_metadata_and_empty_inputs() -> None:
    report = nonlinear_turbulence_gradient_bracket_sweep_report([])

    assert report["passed"] is False
    assert report["promotion_ready_bracket_count"] == 0
    assert (
        "at least two matched plus/minus perturbation amplitudes"
        in report["recommendation"]
    )

    with pytest.raises(ValueError, match="paths length"):
        nonlinear_turbulence_gradient_bracket_sweep_report(
            [_production_gradient()],
            paths=["first.json", "second.json"],
        )
    with pytest.raises(ValueError, match="labels length"):
        nonlinear_turbulence_gradient_bracket_sweep_report(
            [_production_gradient()],
            labels=["first", "second"],
        )


def test_bracket_sweep_recommends_shrinking_nonlocal_quiet_bracket() -> None:
    quiet_but_nonlocal = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(9.0, sem=0.02),
        baseline=_ensemble(9.6, sem=0.02),
        plus=_ensemble(11.0, sem=0.02),
        delta_parameter=0.05,
        parameter_name="rbc_1_1",
        config=NonlinearTurbulenceGradientFiniteDifferenceConfig(
            max_fd_asymmetry_rel=0.5,
        ),
    )

    report = nonlinear_turbulence_gradient_bracket_sweep_report([quiet_but_nonlocal])

    assert report["passed"] is False
    assert report["brackets"][0]["margins"]["uncertainty"] >= 1.0
    assert report["brackets"][0]["margins"]["locality"] < 1.0
    assert "shrink the perturbation" in report["recommendation"]


def test_bracket_sweep_recommends_profile_gradient_for_noisy_nonlocal_response() -> (
    None
):
    noisy_and_nonlocal = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(9.0, sem=5.0),
        baseline=_ensemble(9.6, sem=5.0),
        plus=_ensemble(11.0, sem=5.0),
        delta_parameter=0.05,
        parameter_name="zbs_1_0",
        config=NonlinearTurbulenceGradientFiniteDifferenceConfig(
            max_fd_asymmetry_rel=0.5,
        ),
    )

    report = nonlinear_turbulence_gradient_bracket_sweep_report([noisy_and_nonlocal])

    assert report["passed"] is False
    assert report["brackets"][0]["margins"]["response"] >= 1.0
    assert report["brackets"][0]["margins"]["locality"] < 1.0
    assert report["brackets"][0]["margins"]["uncertainty"] < 1.0
    assert "neither local nor statistically resolved" in report["recommendation"]


def test_bracket_sweep_recommends_abandoning_unresolved_single_control() -> None:
    unresolved = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(10.0, sem=0.02),
        baseline=_ensemble(10.0, sem=0.02),
        plus=_ensemble(10.01, sem=0.02),
        delta_parameter=0.05,
        parameter_name="weak_single_control",
    )

    report = nonlinear_turbulence_gradient_bracket_sweep_report([unresolved])

    assert report["passed"] is False
    assert report["brackets"][0]["margins"]["response"] < 1.0
    assert "response is not resolved" in report["recommendation"]


def test_bracket_sweep_fail_closed_for_legacy_missing_delta_artifact() -> None:
    report = nonlinear_turbulence_gradient_bracket_sweep_report(
        [
            {
                "kind": "legacy_gradient_summary",
                "passed": True,
                "parameter_name": "legacy_control",
                "metrics": {
                    "central_gradient": 1.0,
                    "response_fraction": 0.1,
                    "fd_asymmetry_rel": 0.1,
                    "fd_condition_number": 2.0,
                    "gradient_uncertainty_rel": 0.1,
                },
            }
        ]
    )

    assert report["passed"] is False
    assert report["brackets"][0]["delta_parameter"] is None
    assert (
        "explicit_production_long_window_scope" in report["brackets"][0]["failed_gates"]
    )
    assert "production long-window scope" in report["recommendation"]


def test_bracket_sweep_rejects_mixed_controls() -> None:
    zbs = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(9.0, sem=0.02),
        baseline=_ensemble(10.0, sem=0.02),
        plus=_ensemble(11.0, sem=0.02),
        delta_parameter=0.05,
        parameter_name="zbs_1_0",
    )
    rbc = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(8.8, sem=0.02),
        baseline=_ensemble(10.0, sem=0.02),
        plus=_ensemble(11.2, sem=0.02),
        delta_parameter=0.05,
        parameter_name="rbc_1_1",
    )

    report = nonlinear_turbulence_gradient_bracket_sweep_report([zbs, rbc])

    assert report["passed"] is False
    assert report["promotion_ready_bracket_count"] == 0
    assert report["same_control_gate"]["passed"] is False
    assert report["same_control_gate"]["parameter_names"] == ["rbc_1_1", "zbs_1_0"]
    assert "mixed controls" in report["recommendation"]


def test_gap_report_distinguishes_failed_production_candidate_from_missing_campaign() -> (
    None
):
    fd_report = nonlinear_turbulence_gradient_finite_difference_report(
        minus=_ensemble(9.9, sem=0.5),
        baseline=_ensemble(10.0, sem=0.5),
        plus=_ensemble(10.2, sem=0.5),
        delta_parameter=0.05,
        parameter_name="rbc_1_0",
        config=NonlinearTurbulenceGradientFiniteDifferenceConfig(
            min_fd_response_fraction=0.01,
            max_gradient_uncertainty_rel=0.05,
        ),
    )

    evidence = nonlinear_turbulence_gradient_evidence_report(
        fd_report,
        window_artifacts=[_ensemble(10.0), _ensemble(10.1)],
    )
    gap = evidence["evidence_gap"]

    assert evidence["passed"] is False
    assert (
        gap["claim_level"]
        == "fail_closed_production_candidate_gradient_gate_not_resolved"
    )
    assert gap["current_gradient_candidate_present"] is True
    assert gap["missing_evidence"][0]["current_artifact_class"] == (
        "production_long_window_turbulence_gradient_candidate"
    )
    failed_metrics = {
        row["metric"] for row in gap["missing_evidence"][0]["current_failed_gates"]
    }
    assert "artifact_passed" in failed_metrics
    assert "gradient_uncertainty_bounded" in failed_metrics


def test_gap_report_handles_malformed_evidence_report_fail_closed() -> None:
    gap = nonlinear_turbulence_gradient_evidence_gap_report(
        {
            "passed": False,
            "blockers": [
                "production_gradient_artifact",
                "replicated_long_window_uncertainty",
            ],
            "gradient_artifact": "not-a-dict",
            "window_evidence": "not-a-dict",
        }
    )

    assert gap["passed"] is False
    assert gap["current_gradient_candidate_present"] is False
    assert gap["missing_evidence"][0]["current_artifact_class"] is None
    assert gap["missing_evidence"][1]["qualifying_window_ensembles"] == 0


def test_window_evidence_handles_input_ensembles_unsupported_rows_and_path_mismatch() -> (
    None
):
    ensemble = {
        "kind": "nonlinear_window_ensemble_report",
        "passed": True,
        "statistics": {
            "n_reports": 3,
            "combined_sem_rel": 0.04,
            "mean_rel_spread": 0.05,
        },
    }
    malformed_ensemble = {
        "kind": "nonlinear_window_ensemble_report",
        "passed": True,
        "statistics": ["not", "a", "dict"],
    }
    unsupported = {
        "kind": "single_trace_debug_artifact",
        "promotion_gate": {"passed": True},
    }

    summary = summarize_window_evidence(
        [ensemble, malformed_ensemble, unsupported],
        paths=["ensemble.json", "bad_ensemble.json", "debug.json"],
    )

    assert summary["passed"] is True
    rows = summary["ensemble_rows"]
    assert rows[0]["source"] == "input_ensemble"
    assert rows[0]["qualifies_for_replicated_long_window_uncertainty"] is True
    assert rows[1]["statistics"] == {}
    assert rows[2]["source"] == "unsupported_window_artifact"
    assert rows[2]["passed"] is True

    with pytest.raises(ValueError, match="paths length"):
        summarize_window_evidence([ensemble], paths=["ensemble.json", "extra.json"])


def test_load_json_artifact_rejects_non_object_payload(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain a JSON object"):
        load_json_artifact(path)


def test_cli_writes_report_and_can_fail_on_blocked(tmp_path: Path) -> None:
    mod = _load_tool_module()
    gradient_path = tmp_path / "gradient.json"
    gradient_path.write_text(
        json.dumps(
            {
                "kind": "vmec_boozer_nonlinear_startup_finite_difference_audit",
                "claim_level": "vmec_boozer_geometry_perturbed_startup_plumbing_fd_audit_not_transport_average",
                "passed": True,
                "transport_average_gate": False,
                "production_nonlinear_window_gradient_gate": False,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "evidence.json"
    gap_out = tmp_path / "gap.json"

    rc = mod.main(
        [
            "gradient-evidence",
            "--gradient-artifact",
            str(gradient_path),
            "--json-out",
            str(out),
            "--gap-json-out",
            str(gap_out),
            "--gap-case-slug",
            "qa_ess_gradient",
            "--gradient-parameter-name",
            "rbc_1_0",
            "--fail-on-blocked",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    gap = json.loads(gap_out.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["passed"] is False
    assert payload["blockers"] == [
        "production_gradient_artifact",
        "replicated_long_window_uncertainty",
    ]
    assert gap["promotion_blocked"] is True
    assert gap["required_campaign"]["case_slug"] == "qa_ess_gradient"
    assert gap["required_campaign"]["parameter_name"] == "rbc_1_0"
