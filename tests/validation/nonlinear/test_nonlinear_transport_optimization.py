from __future__ import annotations

import json
from pathlib import Path

from support.paths import REPO_ROOT, load_release_tool

import pytest

import gkx
from gkx.diagnostics.nonlinear_transport_optimization import (
    ProductionNonlinearOptimizationGuardConfig,
    matched_optimized_transport_report,
    optimized_equilibrium_transport_report,
    production_nonlinear_optimization_guard_report,
    reduced_artifact_scope_report,
    replicated_transport_ensemble_report,
)


ROOT = REPO_ROOT


def _tool_module():
    mod = load_release_tool("check_nonlinear_optimization_gates")
    return mod


def _optimization_payload(
    *,
    production_claim: bool = False,
    top_level_production_claim: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {"objective_kind": "nonlinear_heat_flux"}
    if production_claim:
        row["claim_level"] = "production nonlinear turbulent transport optimization"
    payload: dict[str, object] = {
        "parameter_names": ["a"],
        "observable_names": ["nonlinear_heat_flux_mean"],
        "results": [{"objective_kind": "growth"}, row],
    }
    if top_level_production_claim:
        payload["production_nonlinear_optimization_claim"] = True
    return payload


def _startup_payload() -> dict[str, object]:
    return {
        "kind": "nonlinear_startup_window_finite_difference_audit",
        "passed": True,
        "claim_level": "startup_transient_nonlinear_plumbing_fd_audit_not_transport_average",
        "transport_average_gate": False,
        "production_nonlinear_window_gradient_gate": False,
    }


def _ensemble_payload(*, case: str = "holdout", mean: float = 4.0) -> dict[str, object]:
    return {
        "kind": "nonlinear_window_ensemble_report",
        "case": case,
        "claim_level": "replicated_nonlinear_window_uncertainty_gate_not_simulation_claim",
        "passed": True,
        "gate_report": {"passed": True},
        "statistics": {
            "n_reports": 3,
            "ensemble_mean": mean,
            "mean_rel_spread": 0.04,
            "combined_sem_rel": 0.03,
        },
        "rows": [
            {
                "source_artifact": f"{case}_seed31_heat_flux_trace.csv",
                "summary_artifact": f"{case}_seed31_transport_window.json",
            },
            {
                "source_artifact": f"{case}_seed32_heat_flux_trace.csv",
                "summary_artifact": f"{case}_seed32_transport_window.json",
            },
            {
                "source_artifact": f"{case}_dt0p04_heat_flux_trace.csv",
                "summary_artifact": f"{case}_dt0p04_transport_window.json",
            },
        ],
    }


def _matched_audit_payload(
    *,
    relative_reduction: float = 0.184,
    sigma: float = 7.8,
    passed: bool = True,
) -> dict[str, object]:
    return {
        "kind": "baseline_optimized_nonlinear_transport_audit",
        "case": "matched_baseline_to_optimized",
        "claim_level": "matched_baseline_to_optimized_replicated_nonlinear_transport_audit",
        "passed": passed,
        "comparison": {
            "relative_reduction": relative_reduction,
            "uncertainty_separation_sigma": sigma,
        },
        "baseline_ensemble": {"qualifies": True},
        "optimized_ensemble": {"qualifies": True},
        "selected_optimized_audit": {"passed": True},
        "gates": [
            {"metric": "baseline_replicated_ensemble_qualified", "passed": True},
            {"metric": "optimized_replicated_ensemble_qualified", "passed": True},
            {"metric": "selected_optimized_equilibrium_audit", "passed": True},
        ],
    }


def _strict_matched_comparison_payload(
    *,
    relative_reduction: float = 0.006,
    sigma: float = 0.26,
    passed: bool = False,
) -> dict[str, object]:
    return {
        "kind": "matched_nonlinear_transport_comparison",
        "case": "strict_t1500_growth_comparison",
        "claim_level": "matched_replicated_late_window_transport_comparison",
        "passed": passed,
        "baseline": {"passed": True, "raw_passed": True, "ensemble_mean": 11.58},
        "candidate": {"passed": True, "raw_passed": True, "ensemble_mean": 11.51},
        "statistics": {
            "relative_reduction": relative_reduction,
            "uncertainty_z_score": sigma,
        },
        "gates": [
            {"metric": "baseline_ensemble_passed", "passed": True},
            {"metric": "candidate_ensemble_passed", "passed": True},
            {
                "metric": "relative_transport_reduction",
                "passed": relative_reduction >= 0.04,
            },
        ],
    }


def test_production_nonlinear_guard_is_release_safe_but_blocks_optimization_promotion() -> (
    None
):
    report = production_nonlinear_optimization_guard_report(
        optimization_artifact=_optimization_payload(),
        optimization_artifact_path="optimization.json",
        reduced_artifacts={"startup.json": _startup_payload()},
        replicated_ensemble_artifacts={
            "dshape.json": _ensemble_payload(case="dshape"),
            "circular.json": _ensemble_payload(case="circular", mean=3.8),
        },
    )

    assert report["passed"] is True
    assert report["safe_to_release"] is True
    assert report["production_nonlinear_optimization_promoted"] is False
    assert report["promotion_gate"]["blockers"] == [
        "optimized_equilibrium_replicated_transport_window",
        "matched_baseline_to_optimized_transport_reduction",
    ]
    assert report["summary"]["qualifying_replicated_holdout_ensembles"] == 2


def test_production_nonlinear_guard_rejects_reduced_optimizer_overclaim() -> None:
    for payload in (
        _optimization_payload(production_claim=True),
        _optimization_payload(top_level_production_claim=True),
    ):
        report = production_nonlinear_optimization_guard_report(
            optimization_artifact=payload,
            optimization_artifact_path="optimization.json",
            reduced_artifacts={"startup.json": _startup_payload()},
            replicated_ensemble_artifacts={
                "dshape.json": _ensemble_payload(case="dshape"),
                "circular.json": _ensemble_payload(case="circular", mean=3.8),
            },
        )

        assert report["safe_to_release"] is False
        assert "reduced_optimizer_not_promoted" in report["safety_gate"]["blockers"]


def test_production_nonlinear_guard_blocks_optimized_window_without_matched_reduction() -> (
    None
):
    report = production_nonlinear_optimization_guard_report(
        optimization_artifact=_optimization_payload(),
        optimization_artifact_path="optimization.json",
        reduced_artifacts={"startup.json": _startup_payload()},
        replicated_ensemble_artifacts={
            "dshape.json": _ensemble_payload(case="dshape"),
            "circular.json": _ensemble_payload(case="circular", mean=3.8),
        },
        optimized_equilibrium_artifacts={
            "optimized_equilibrium_final.json": _ensemble_payload(
                case="optimized_equilibrium_final", mean=2.6
            )
        },
    )

    assert report["safe_to_release"] is True
    assert report["production_nonlinear_optimization_promoted"] is False
    assert report["promotion_gate"]["blockers"] == [
        "optimized_equilibrium_replicated_transport_window",
        "matched_baseline_to_optimized_transport_reduction",
    ]


def test_production_nonlinear_guard_requires_three_matched_optimized_audits() -> None:
    one_audit = production_nonlinear_optimization_guard_report(
        optimization_artifact=_optimization_payload(),
        optimization_artifact_path="optimization.json",
        reduced_artifacts={"startup.json": _startup_payload()},
        replicated_ensemble_artifacts={
            "dshape.json": _ensemble_payload(case="dshape"),
            "circular.json": _ensemble_payload(case="circular", mean=3.8),
        },
        optimized_equilibrium_artifacts={
            "optimized_equilibrium_final.json": _ensemble_payload(
                case="optimized_equilibrium_final", mean=2.6
            )
        },
        matched_optimized_transport_artifacts={
            "matched_optimized_audit.json": _matched_audit_payload()
        },
    )

    assert one_audit["safe_to_release"] is True
    assert one_audit["production_nonlinear_optimization_promoted"] is False
    assert one_audit["summary"]["qualifying_matched_optimized_transport_audits"] == 1
    assert one_audit["summary"]["total_matched_optimized_transport_audits"] == 1
    assert (
        one_audit["evidence_gap"]["required_additional_matched_optimized_audits"] == 2
    )
    assert one_audit["promotion_gate"]["blockers"] == [
        "optimized_equilibrium_replicated_transport_window",
        "matched_baseline_to_optimized_transport_reduction",
    ]


def test_production_nonlinear_guard_promotes_only_with_three_matched_audits() -> None:
    report = production_nonlinear_optimization_guard_report(
        optimization_artifact=_optimization_payload(),
        optimization_artifact_path="optimization.json",
        reduced_artifacts={"startup.json": _startup_payload()},
        replicated_ensemble_artifacts={
            "dshape.json": _ensemble_payload(case="dshape"),
            "circular.json": _ensemble_payload(case="circular", mean=3.8),
        },
        optimized_equilibrium_artifacts={
            "optimized_equilibrium_final.json": _ensemble_payload(
                case="optimized_equilibrium_final", mean=2.6
            ),
            "optimized_equilibrium_second.json": _ensemble_payload(
                case="optimized_equilibrium_second", mean=2.7
            ),
            "optimized_equilibrium_third.json": _ensemble_payload(
                case="optimized_equilibrium_third", mean=2.8
            ),
        },
        matched_optimized_transport_artifacts={
            "matched_optimized_audit.json": _matched_audit_payload(),
            "matched_optimized_audit_second.json": _matched_audit_payload(
                relative_reduction=0.11,
                sigma=2.2,
            ),
            "matched_optimized_audit_third.json": _matched_audit_payload(
                relative_reduction=0.08,
                sigma=1.6,
            ),
        },
    )

    assert report["safe_to_release"] is True
    assert report["production_nonlinear_optimization_promoted"] is True
    assert report["promotion_gate"]["blockers"] == []
    assert report["summary"]["qualifying_matched_optimized_transport_audits"] == 3
    assert report["summary"]["qualifying_optimized_equilibrium_ensembles"] == 3


def test_production_nonlinear_guard_tool_writes_artifacts(tmp_path: Path) -> None:
    mod = _tool_module()
    optimization = tmp_path / "optimization.json"
    startup = tmp_path / "startup.json"
    dshape = tmp_path / "dshape.json"
    circular = tmp_path / "circular.json"
    optimization.write_text(json.dumps(_optimization_payload()), encoding="utf-8")
    startup.write_text(json.dumps(_startup_payload()), encoding="utf-8")
    dshape.write_text(json.dumps(_ensemble_payload(case="dshape")), encoding="utf-8")
    circular.write_text(
        json.dumps(_ensemble_payload(case="circular", mean=3.8)), encoding="utf-8"
    )
    out_json = tmp_path / "guard.json"
    out_png = tmp_path / "guard.png"

    rc = mod.main(
        [
            "production-guard",
            "--optimization-artifact",
            str(optimization),
            "--reduced-artifact",
            str(startup),
            "--replicated-ensemble",
            str(dshape),
            "--replicated-ensemble",
            str(circular),
            "--out-json",
            str(out_json),
            "--out-png",
            str(out_png),
            "--fail-on-unsafe",
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 0
    assert out_png.exists()
    assert payload["safe_to_release"] is True
    assert payload["production_nonlinear_optimization_promoted"] is True
    assert payload["summary"]["qualifying_matched_optimized_transport_audits"] >= 3
    assert payload["summary"]["qualifying_optimized_equilibrium_ensembles"] >= 3
    assert payload["summary"]["total_matched_optimized_transport_audits"] >= 6
    assert payload["summary"]["failed_matched_optimized_transport_audits"] >= 3
    assert payload["promotion_gate"]["blockers"] == []


def test_matched_optimized_transport_report_requires_reduction_and_uncertainty() -> (
    None
):
    good = matched_optimized_transport_report(
        "matched.json",
        _matched_audit_payload(),
    )
    weak = matched_optimized_transport_report(
        "weak.json",
        _matched_audit_payload(relative_reduction=0.01, sigma=0.2),
    )

    assert good["qualifies_for_production_optimization"] is True
    assert weak["qualifies_for_production_optimization"] is False
    assert "insufficient_matched_optimized_reduction" in weak["blockers"]
    assert "insufficient_matched_optimized_uncertainty_separation" in weak["blockers"]
    assert (
        gkx.matched_optimized_transport_report
        is matched_optimized_transport_report
    )


def test_matched_transport_explicit_zero_uncertainty_overrides_fallback() -> None:
    payload = _matched_audit_payload()
    comparison = payload["comparison"]
    assert isinstance(comparison, dict)
    comparison["uncertainty_separation_sigma"] = 0.0
    payload["statistics"] = {"uncertainty_z_score": 8.0}

    report = matched_optimized_transport_report("zero_sigma.json", payload)

    assert report["uncertainty_separation_sigma"] == 0.0
    assert report["uncertainty_separation_ok"] is False
    assert report["qualifies_for_production_optimization"] is False


def test_strict_matched_comparison_schema_is_counted_as_negative_evidence() -> None:
    report = matched_optimized_transport_report(
        "strict_t1500.json",
        _strict_matched_comparison_payload(),
    )

    assert report["baseline_ensemble_qualified"] is True
    assert report["optimized_ensemble_qualified"] is True
    assert report["selected_optimized_audit_closed"] is True
    assert report["qualifies_for_production_optimization"] is False
    assert "insufficient_matched_optimized_reduction" in report["blockers"]
    assert "insufficient_matched_optimized_uncertainty_separation" in report["blockers"]


def test_strict_t1500_candidate_ensembles_count_as_optimized_trace_evidence() -> None:
    report = optimized_equilibrium_transport_report(
        "docs/_static/vmec_qa_t1500_replicates/growth_from_strict_baseline_t1500_ensemble_gate.json",
        _ensemble_payload(
            case="vmec_qa_full_sweep_growth_from_strict_baseline_t1500_replicated_nonlinear_window"
        ),
    )

    assert report["qualifies_as_long_post_transient_replicate"] is True
    assert report["optimized_equilibrium_marker"] is True
    assert report["qualifies_for_production_optimization"] is True


def test_replicated_transport_report_fails_closed_on_unscoped_or_noisy_payloads() -> (
    None
):
    noisy = _ensemble_payload(mean=3.0)
    noisy["claim_level"] = "replicated nonlinear window without required scope"
    noisy["statistics"] = {
        "n_reports": 1,
        "ensemble_mean": 0.0,
        "mean_rel_spread": 0.6,
        "combined_sem_rel": 0.8,
    }

    report = replicated_transport_ensemble_report(
        "noisy.json",
        noisy,
        config=ProductionNonlinearOptimizationGuardConfig(
            min_reports_per_ensemble=2,
            max_mean_rel_spread=0.15,
            max_combined_sem_rel=0.25,
        ),
    )

    assert report["qualifies_as_long_post_transient_replicate"] is False
    assert report["claim_scoped_as_replicated_holdout"] is False
    assert report["finite_transport_mean"] is False
    assert report["mean_rel_spread_ok"] is False
    assert report["combined_sem_rel_ok"] is False
    assert report["report_count_ok"] is False


def test_replicated_transport_report_requires_seed_and_timestep_provenance() -> None:
    aggregate_only = _ensemble_payload(case="optimized", mean=2.5)
    aggregate_only.pop("rows")

    report = replicated_transport_ensemble_report("aggregate.json", aggregate_only)

    assert report["qualifies_as_long_post_transient_replicate"] is False
    assert report["seed_timestep_provenance_ok"] is False
    assert report["seed_timestep_provenance"]["seed_values"] == []
    assert report["seed_timestep_provenance"]["timestep_values"] == []


@pytest.mark.parametrize("invalid_count", ["three", 2.5, -3, float("nan")])
def test_replicated_transport_report_rejects_corrupt_pass_and_count_fields(
    invalid_count,
) -> None:
    payload = _ensemble_payload(case="corrupt", mean=2.5)
    payload["passed"] = "true"
    payload["gate_report"]["passed"] = "true"
    payload["statistics"]["n_reports"] = invalid_count

    report = replicated_transport_ensemble_report("corrupt.json", payload)

    assert report["passed"] is False
    assert report["n_reports"] == 0
    assert report["report_count_ok"] is False
    assert report["qualifies_as_long_post_transient_replicate"] is False


def test_matched_transport_report_rejects_string_gate_flags() -> None:
    payload = _matched_audit_payload()
    payload["passed"] = "true"
    payload["baseline_ensemble"]["qualifies"] = "true"
    payload["optimized_ensemble"]["qualifies"] = "true"
    payload["selected_optimized_audit"]["passed"] = "true"
    for gate in payload["gates"]:
        gate["passed"] = "true"

    report = matched_optimized_transport_report("corrupt.json", payload)

    assert report["passed"] is False
    assert report["baseline_ensemble_qualified"] is False
    assert report["optimized_ensemble_qualified"] is False
    assert report["selected_optimized_audit_closed"] is False
    assert report["qualifies_for_production_optimization"] is False


def test_optimized_equilibrium_marker_and_reduced_scope_reports_are_fail_closed() -> (
    None
):
    optimized = optimized_equilibrium_transport_report(
        "post_optimization_transport.json",
        _ensemble_payload(case="final"),
    )
    nonoptimized = optimized_equilibrium_transport_report(
        "baseline.json", _ensemble_payload(case="baseline")
    )
    reduced = reduced_artifact_scope_report("startup.json", _startup_payload())
    unsafe = reduced_artifact_scope_report(
        "unsafe.json",
        {
            "kind": "optimized_equilibrium_nonlinear_transport_window",
            "passed": True,
            "production_nonlinear_optimization_claim": True,
            "transport_average_gate": False,
        },
    )

    assert optimized["optimized_equilibrium_marker"] is True
    assert optimized["qualifies_for_production_optimization"] is True
    assert nonoptimized["optimized_equilibrium_marker"] is False
    assert nonoptimized["qualifies_for_production_optimization"] is False
    assert reduced["safely_blocked_from_production"] is True
    assert unsafe["claims_production"] is True
    assert unsafe["safely_blocked_from_production"] is False


def test_production_nonlinear_guard_config_validation() -> None:
    invalid_configs = [
        ProductionNonlinearOptimizationGuardConfig(min_replicated_ensembles=0),
        ProductionNonlinearOptimizationGuardConfig(min_reports_per_ensemble=1),
        ProductionNonlinearOptimizationGuardConfig(max_mean_rel_spread=-1.0),
        ProductionNonlinearOptimizationGuardConfig(max_combined_sem_rel=-1.0),
        ProductionNonlinearOptimizationGuardConfig(
            min_optimized_equilibrium_ensembles=0
        ),
        ProductionNonlinearOptimizationGuardConfig(min_matched_optimized_audits=0),
        ProductionNonlinearOptimizationGuardConfig(min_seed_variants=0),
        ProductionNonlinearOptimizationGuardConfig(min_timestep_variants=0),
        ProductionNonlinearOptimizationGuardConfig(
            min_matched_optimized_relative_reduction=-1.0
        ),
        ProductionNonlinearOptimizationGuardConfig(
            min_matched_optimized_uncertainty_sigma=-1.0
        ),
        ProductionNonlinearOptimizationGuardConfig(value_floor=0.0),
    ]

    for cfg in invalid_configs:
        with pytest.raises(ValueError):
            cfg.validate()
