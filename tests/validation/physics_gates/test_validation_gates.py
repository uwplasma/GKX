from __future__ import annotations

import json

import numpy as np
import pytest

import gkx
import gkx.diagnostics.validation_gates as validation_gates
from gkx.diagnostics.analysis import (
    BranchContinuationMetrics,
    LateTimeLinearMetrics,
    NonlinearHeatFluxConvergenceMetrics,
    NonlinearWindowMetrics,
    ObservedOrderMetrics,
)
from gkx.diagnostics.modes import EigenfunctionComparisonMetrics
from gkx.diagnostics.validation_gates import (
    GateReport,
    ScalarGateResult,
    ZonalFlowResponseMetrics,
    branch_continuity_gate_report,
    eigenfunction_gate_report,
    evaluate_scalar_gate,
    gate_report,
    gate_report_to_dict,
    linear_metrics_gate_report,
    nonlinear_heat_flux_convergence_gate_report,
    nonlinear_window_gate_report,
    observed_order_gate_report,
    zonal_response_gate_report,
)
from gkx.diagnostics.zonal_validation import zonal_flow_response_metrics


def test_validation_gate_facade_points_to_focused_modules() -> None:
    import gkx.diagnostics.analysis as metric_analysis
    import gkx.diagnostics.modes as mode_analysis
    import gkx.diagnostics.validation_gates as gates

    assert LateTimeLinearMetrics is metric_analysis.LateTimeLinearMetrics
    assert NonlinearWindowMetrics is metric_analysis.NonlinearWindowMetrics
    assert EigenfunctionComparisonMetrics is mode_analysis.EigenfunctionComparisonMetrics
    assert gates.evaluate_scalar_gate is evaluate_scalar_gate
    assert gates.gate_report_to_dict is gate_report_to_dict
    assert gates.observed_order_gate_report is observed_order_gate_report


def test_validation_gate_primitives_are_public_and_owned_by_diagnostics() -> None:
    assert gkx.evaluate_scalar_gate is evaluate_scalar_gate
    metrics = zonal_flow_response_metrics(
        np.linspace(0.0, 2.0, 8), np.linspace(1.0, 0.6, 8)
    )
    assert isinstance(metrics, ZonalFlowResponseMetrics)
    assert validation_gates.observed_order_gate_report is observed_order_gate_report
    assert validation_gates.branch_continuity_gate_report is branch_continuity_gate_report
    assert (
        validation_gates.nonlinear_heat_flux_convergence_gate_report
        is nonlinear_heat_flux_convergence_gate_report
    )


def test_scalar_gate_and_json_report_are_strict_and_serializable() -> None:
    passed = evaluate_scalar_gate("gamma", 1.01, 1.0, atol=0.0, rtol=0.02)
    failed = evaluate_scalar_gate("omega", 0.7, 1.0, atol=0.0, rtol=0.02)
    near_zero = evaluate_scalar_gate(
        "zonal_residual", 1.0e-4, 0.0, atol=2.0e-4, rtol=0.0
    )
    report = gate_report("case", "reference", [passed, failed, near_zero])
    payload = gate_report_to_dict(report)

    assert isinstance(passed, ScalarGateResult)
    assert isinstance(report, GateReport)
    assert report.passed is False
    assert payload["gates"][0]["metric"] == "gamma"
    assert payload["gates"][1]["passed"] is False
    assert payload["gates"][2]["rel_error"] is None
    json.dumps(payload, allow_nan=False)

    with pytest.raises(ValueError):
        gate_report("empty", "reference", [])
    with pytest.raises(ValueError):
        evaluate_scalar_gate("bad", 1.0, 1.0, atol=-1.0, rtol=0.0)


def test_scalar_gate_thresholds_are_inclusive_and_nonfinite_values_fail() -> None:
    exact_combined = evaluate_scalar_gate(
        "combined_tol", 1.25, 1.0, atol=0.05, rtol=0.20
    )
    just_over = evaluate_scalar_gate("combined_tol", 1.2501, 1.0, atol=0.05, rtol=0.20)
    exact_zero_ref = evaluate_scalar_gate(
        "zero_ref", -2.0e-4, 0.0, atol=2.0e-4, rtol=0.0
    )
    just_over_zero_ref = evaluate_scalar_gate(
        "zero_ref", 2.01e-4, 0.0, atol=2.0e-4, rtol=0.0
    )

    assert exact_combined.passed is True
    assert just_over.passed is False
    assert exact_zero_ref.passed is True
    assert just_over_zero_ref.passed is False

    nonfinite_report = gate_report(
        "nonfinite",
        "synthetic",
        (
            evaluate_scalar_gate("nan_observed", np.nan, 1.0, atol=1.0, rtol=0.0),
            evaluate_scalar_gate("inf_observed", np.inf, 1.0, atol=1.0, rtol=0.0),
            evaluate_scalar_gate("inf_reference", 1.0, np.inf, atol=1.0, rtol=0.0),
        ),
    )
    payload = gate_report_to_dict(nonfinite_report)

    assert nonfinite_report.passed is False
    assert np.isinf(nonfinite_report.max_abs_error)
    assert all(gate.passed is False for gate in nonfinite_report.gates)
    assert payload["max_abs_error"] is None
    assert payload["gates"][0]["observed"] is None
    assert payload["gates"][1]["observed"] is None
    assert payload["gates"][2]["reference"] is None
    json.dumps(payload, allow_nan=False)


def test_family_gate_thresholds_are_inclusive_at_documented_bounds() -> None:
    eigen = eigenfunction_gate_report(
        EigenfunctionComparisonMetrics(overlap=0.95, relative_l2=0.25, phase_shift=0.0),
        case="mode",
        source="synthetic",
        min_overlap=0.95,
        max_relative_l2=0.25,
    )
    order = observed_order_gate_report(
        ObservedOrderMetrics(
            step_sizes=np.array([0.4, 0.2, 0.1]),
            errors=np.array([4.0e-3, 2.0e-3, 1.0e-3]),
            orders=np.array([1.5, 1.5]),
            asymptotic_order=2.0,
        ),
        case="order",
        source="synthetic",
        min_asymptotic_order=2.0,
        min_pairwise_order=1.5,
        max_final_error=1.0e-3,
    )
    branch = branch_continuity_gate_report(
        BranchContinuationMetrics(
            ky=np.array([0.1, 0.2, 0.3]),
            gamma=np.array([0.1, 0.15, 0.2]),
            omega=np.array([1.0, 1.1, 1.2]),
            rel_gamma_jumps=np.array([0.5, 0.25]),
            rel_omega_jumps=np.array([0.25, 0.1]),
            max_rel_gamma_jump=0.5,
            max_rel_omega_jump=0.25,
            min_successive_overlap=0.95,
        ),
        case="branch",
        source="synthetic",
        max_rel_gamma_jump=0.5,
        max_rel_omega_jump=0.25,
        min_successive_overlap=0.95,
    )

    assert eigen.passed is True
    assert order.passed is True
    assert branch.passed is True


def test_validation_gate_family_helpers_cover_physics_observables() -> None:
    linear = LateTimeLinearMetrics(
        gamma_fit=1.0,
        omega_fit=2.0,
        gamma_tail_mean=1.0,
        omega_tail_mean=2.0,
        gamma_tail_std=0.01,
        omega_tail_std=0.02,
        tmin=1.0,
        tmax=2.0,
        nsamples=10,
        signal_source="mode",
    )
    nonlinear = NonlinearWindowMetrics(
        tmin=1.0,
        tmax=2.0,
        nsamples=10,
        heat_flux_mean=1.0,
        heat_flux_std=0.1,
        heat_flux_rms=1.05,
        wphi_mean=2.0,
        wphi_std=0.2,
        wg_mean=3.0,
        wg_std=0.3,
        phi_mode_envelope_mean=4.0,
        phi_mode_envelope_std=0.4,
        phi_mode_envelope_max=4.5,
    )
    nonlinear_convergence = NonlinearHeatFluxConvergenceMetrics(
        tmin=10.0,
        tmax=20.0,
        nsamples=12,
        heat_flux_mean=1.0,
        heat_flux_std=0.02,
        heat_flux_cv=0.02,
        heat_flux_rms=1.0002,
        terminal_tmin=15.0,
        terminal_tmax=20.0,
        terminal_nsamples=6,
        terminal_heat_flux_mean=1.01,
        mean_rel_delta=0.01,
        trend=0.02,
        abs_trend=0.02,
        start_fraction=0.5,
        terminal_fraction=0.5,
    )
    zonal = ZonalFlowResponseMetrics(
        initial_level=1.0,
        initial_policy="first_abs",
        residual_level=0.2,
        residual_std=0.01,
        response_rms=0.3,
        gam_frequency=2.0,
        gam_damping_rate=0.1,
        damping_method="branchwise_extrema",
        frequency_method="hilbert_phase",
        peak_count=4,
        peak_fit_count=4,
        tmin=0.0,
        tmax=10.0,
        fit_tmin=0.0,
        fit_tmax=5.0,
        peak_times=np.array([1.0, 2.0]),
        peak_envelope=np.array([0.5, 0.4]),
        max_peak_times=np.array([1.0]),
        max_peak_values=np.array([0.5]),
        min_peak_times=np.array([2.0]),
        min_peak_values=np.array([-0.4]),
    )

    assert (
        linear_metrics_gate_report(linear, linear, case="linear", source="self").passed
        is True
    )
    assert (
        nonlinear_window_gate_report(
            nonlinear, nonlinear, case="nonlinear", source="self"
        ).passed
        is True
    )
    assert (
        nonlinear_heat_flux_convergence_gate_report(
            nonlinear_convergence,
            case="nonlinear_convergence",
            source="self",
            max_mean_rel_delta=0.02,
            max_cv=0.03,
            max_abs_trend=0.03,
            min_samples=12,
        ).passed
        is True
    )
    assert (
        zonal_response_gate_report(
            zonal,
            zonal,
            case="zonal",
            source="self",
            residual_atol=0.0,
            frequency_atol=0.0,
            damping_atol=0.0,
        ).passed
        is True
    )
    assert (
        eigenfunction_gate_report(
            EigenfunctionComparisonMetrics(
                overlap=0.99, relative_l2=0.01, phase_shift=0.0
            ),
            case="mode",
            source="self",
        ).passed
        is True
    )


def test_order_and_branch_gates_preserve_open_lane_failures() -> None:
    observed = ObservedOrderMetrics(
        step_sizes=np.array([0.4, 0.2, 0.1]),
        errors=np.array([0.01, 0.02, 0.002]),
        orders=np.array([-1.0, 3.32192809]),
        asymptotic_order=3.32192809,
    )
    order_report = observed_order_gate_report(
        observed,
        case="nonmonotone",
        source="synthetic",
        min_asymptotic_order=1.0,
        min_pairwise_order=0.0,
    )
    assert order_report.passed is False
    assert order_report.gates[1].metric == "min_pairwise_order_deficit"

    branch = BranchContinuationMetrics(
        ky=np.array([0.1, 0.2]),
        gamma=np.array([0.1, 0.3]),
        omega=np.array([1.0, 1.1]),
        rel_gamma_jumps=np.array([0.666]),
        rel_omega_jumps=np.array([0.091]),
        max_rel_gamma_jump=0.666,
        max_rel_omega_jump=0.091,
        min_successive_overlap=None,
    )
    branch_report = branch_continuity_gate_report(
        branch,
        case="branch",
        source="synthetic",
        max_rel_gamma_jump=0.5,
        max_rel_omega_jump=0.5,
        min_successive_overlap=0.95,
    )
    assert branch_report.passed is False
    assert branch_report.gates[-1].metric == "successive_overlap_deficit"


def test_nonlinear_window_gate_optional_envelope_policy_is_explicit() -> None:
    reference = NonlinearWindowMetrics(
        tmin=1.0,
        tmax=2.0,
        nsamples=8,
        heat_flux_mean=1.0,
        heat_flux_std=0.1,
        heat_flux_rms=1.1,
        wphi_mean=2.0,
        wphi_std=0.2,
        wg_mean=3.0,
        wg_std=0.3,
        phi_mode_envelope_mean=1.0,
        phi_mode_envelope_std=0.1,
        phi_mode_envelope_max=1.2,
    )
    envelope_mismatch = NonlinearWindowMetrics(
        tmin=1.0,
        tmax=2.0,
        nsamples=8,
        heat_flux_mean=1.0,
        heat_flux_std=0.1,
        heat_flux_rms=1.1,
        wphi_mean=2.0,
        wphi_std=0.2,
        wg_mean=3.0,
        wg_std=0.3,
        phi_mode_envelope_mean=2.0,
        phi_mode_envelope_std=0.1,
        phi_mode_envelope_max=2.2,
    )
    unresolved_mode = NonlinearWindowMetrics(
        tmin=1.0,
        tmax=2.0,
        nsamples=8,
        heat_flux_mean=1.0,
        heat_flux_std=0.1,
        heat_flux_rms=1.1,
        wphi_mean=2.0,
        wphi_std=0.2,
        wg_mean=3.0,
        wg_std=0.3,
        phi_mode_envelope_mean=None,
        phi_mode_envelope_std=None,
        phi_mode_envelope_max=None,
    )

    envelope_report = nonlinear_window_gate_report(
        envelope_mismatch,
        reference,
        case="window",
        source="synthetic",
        rtol=0.1,
    )
    excluded_report = nonlinear_window_gate_report(
        envelope_mismatch,
        reference,
        case="window",
        source="synthetic",
        rtol=0.1,
        include_envelope=False,
    )
    unresolved_report = nonlinear_window_gate_report(
        unresolved_mode,
        reference,
        case="window",
        source="synthetic",
        rtol=0.1,
    )

    assert envelope_report.passed is False
    assert envelope_report.gates[-1].metric == "phi_mode_envelope_mean"
    assert excluded_report.passed is True
    assert [gate.metric for gate in excluded_report.gates] == [
        "heat_flux_mean",
        "heat_flux_rms",
        "wphi_mean",
        "wg_mean",
    ]
    assert unresolved_report.passed is True
    assert len(unresolved_report.gates) == 4


def test_validation_gate_threshold_guards_are_fail_closed() -> None:
    convergence = NonlinearHeatFluxConvergenceMetrics(
        tmin=10.0,
        tmax=20.0,
        nsamples=8,
        heat_flux_mean=1.0,
        heat_flux_std=0.1,
        heat_flux_cv=0.1,
        heat_flux_rms=1.01,
        terminal_tmin=15.0,
        terminal_tmax=20.0,
        terminal_nsamples=4,
        terminal_heat_flux_mean=1.02,
        mean_rel_delta=0.02,
        trend=0.03,
        abs_trend=0.03,
        start_fraction=0.5,
        terminal_fraction=0.5,
    )
    order = ObservedOrderMetrics(
        step_sizes=np.array([0.4, 0.2]),
        errors=np.array([0.02, 0.01]),
        orders=np.array([1.0]),
        asymptotic_order=1.0,
    )
    branch = BranchContinuationMetrics(
        ky=np.array([0.1, 0.2]),
        gamma=np.array([0.1, 0.2]),
        omega=np.array([1.0, 1.1]),
        rel_gamma_jumps=np.array([0.5]),
        rel_omega_jumps=np.array([0.1]),
        max_rel_gamma_jump=0.5,
        max_rel_omega_jump=0.1,
        min_successive_overlap=0.9,
    )

    with pytest.raises(ValueError, match="non-negative"):
        nonlinear_heat_flux_convergence_gate_report(
            convergence,
            case="heat",
            source="synthetic",
            max_cv=-0.1,
        )
    with pytest.raises(ValueError, match="min_samples"):
        nonlinear_heat_flux_convergence_gate_report(
            convergence,
            case="heat",
            source="synthetic",
            min_samples=0,
        )
    with pytest.raises(ValueError, match="min_overlap"):
        eigenfunction_gate_report(
            EigenfunctionComparisonMetrics(
                overlap=0.9,
                relative_l2=0.1,
                phase_shift=0.0,
            ),
            case="mode",
            source="synthetic",
            min_overlap=1.01,
        )
    with pytest.raises(ValueError, match="relative_l2"):
        eigenfunction_gate_report(
            EigenfunctionComparisonMetrics(
                overlap=0.9,
                relative_l2=0.1,
                phase_shift=0.0,
            ),
            case="mode",
            source="synthetic",
            max_relative_l2=-0.1,
        )
    with pytest.raises(ValueError, match="min_asymptotic_order"):
        observed_order_gate_report(
            order,
            case="order",
            source="synthetic",
            min_asymptotic_order=-1.0,
        )
    with pytest.raises(ValueError, match="min_pairwise_order"):
        observed_order_gate_report(
            order,
            case="order",
            source="synthetic",
            min_asymptotic_order=1.0,
            min_pairwise_order=-1.0,
        )
    with pytest.raises(ValueError, match="max_final_error"):
        observed_order_gate_report(
            order,
            case="order",
            source="synthetic",
            min_asymptotic_order=1.0,
            max_final_error=-1.0,
        )
    with pytest.raises(ValueError, match="maximum relative jumps"):
        branch_continuity_gate_report(
            branch,
            case="branch",
            source="synthetic",
            max_rel_gamma_jump=-0.1,
            max_rel_omega_jump=0.2,
        )
    with pytest.raises(ValueError, match="min_successive_overlap"):
        branch_continuity_gate_report(
            branch,
            case="branch",
            source="synthetic",
            max_rel_gamma_jump=1.0,
            max_rel_omega_jump=1.0,
            min_successive_overlap=-0.1,
        )
