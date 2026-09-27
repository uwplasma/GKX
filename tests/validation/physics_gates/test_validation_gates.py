"""Physics gates: the validation-gate machinery and the zonal-response metrics it gates.

These are the measurement and pass/fail primitives that every tracked
comparison runs through -- scalar and family gate reports with their inclusive
thresholds and fail-closed guards, and the W7-X/Merlo zonal-response trace
loaders, reference tables, and GAM residual/damping/frequency estimators whose
numbers those gates read.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import gkx
from gkx.diagnostics.analysis import (
    BranchContinuationMetrics,
    LateTimeLinearMetrics,
    NonlinearHeatFluxConvergenceMetrics,
    NonlinearWindowMetrics,
    ObservedOrderMetrics,
)
from gkx.diagnostics.modes import EigenfunctionComparisonMetrics
import gkx.diagnostics.validation_gates as validation_gates
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
from gkx.diagnostics.zonal_validation import (
    zonal_flow_response_metrics,
)


def test_validation_gate_facade_points_to_focused_modules() -> None:
    import gkx.diagnostics.analysis as metric_analysis
    import gkx.diagnostics.modes as mode_analysis
    import gkx.diagnostics.validation_gates as gates

    assert LateTimeLinearMetrics is metric_analysis.LateTimeLinearMetrics
    assert NonlinearWindowMetrics is metric_analysis.NonlinearWindowMetrics
    assert (
        EigenfunctionComparisonMetrics is mode_analysis.EigenfunctionComparisonMetrics
    )
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
    assert (
        validation_gates.branch_continuity_gate_report is branch_continuity_gate_report
    )
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
        # Explicit: the convergence gate divides the standard error by the
        # INDEPENDENT sample count, and a metrics object built by hand defaults
        # it to zero so it cannot pass a statistical gate by omission.
        n_eff=12.0,
        tau_ac=0.0,
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


# ---- from test_zonal_validation.py ----


GAMMA_GATE_ATOL_R0_OVER_VI = 0.03


MERLO_R0 = 2.77778


def _merlo_like_gam_trace(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """A damped GAM on a residual, plus the ripple that broke the old estimator.

    The second term is a weakly damped, higher-frequency ripple -- what
    velocity-space recurrence actually adds to a collisionless Hermite trace. It
    puts shallow extra extrema inside the fit window, and a four-extrema
    log-linear fit resolves them at one output cadence and not at another. That
    is the mechanism that moved the shipped gamma_GAM by 52 percent when only
    the diagnostic sample spacing changed, at identical physics.
    """

    t = np.linspace(0.0, 60.0, n_samples)
    gam = 0.2 + np.exp(-0.066 * t) * np.cos(0.845 * t)
    ripple = 0.08 * np.exp(-0.02 * t) * np.cos(4.2 * t + 2.1)
    return t, gam + ripple


def _gamma_r0_over_vi(t: np.ndarray, y: np.ndarray, mode: str) -> float:
    metrics = zonal_flow_response_metrics(
        t,
        y,
        tail_fraction=0.3,
        initial_policy="first_abs",
        peak_fit_max_peaks=4,
        damping_fit_mode=mode,
        frequency_fit_mode="hilbert_phase",
        fit_window_tmax=30.0,
    )
    return -float(metrics.gam_damping_rate) * MERLO_R0


def test_period_rms_damping_is_independent_of_the_diagnostic_output_cadence() -> None:
    """The gated GAM damping must not move when only the output cadence changes.

    Output cadence carries no physics: the same trajectory written out twice as
    often has to give the same damping rate. The retired branchwise-extrema fit
    did not -- it moved by more than its own gate tolerance, so its PASS was a
    property of ``sample_stride`` rather than of the solver. This pins that the
    period-RMS envelope estimator is flat across a 4x cadence span while the
    extrema fit is not.
    """

    cadences = [(4801, "0.0125"), (2401, "0.025"), (1201, "0.05")]
    envelope = []
    extrema = []
    for n_samples, _label in cadences:
        t, y = _merlo_like_gam_trace(n_samples)
        envelope.append(_gamma_r0_over_vi(t, y, "period_rms_envelope"))
        extrema.append(_gamma_r0_over_vi(t, y, "branchwise_extrema"))

    envelope_spread = max(envelope) - min(envelope)
    extrema_spread = max(extrema) - min(extrema)

    assert envelope_spread < 0.1 * GAMMA_GATE_ATOL_R0_OVER_VI
    assert extrema_spread > GAMMA_GATE_ATOL_R0_OVER_VI
    assert all(np.isfinite(envelope))


def test_period_rms_damping_recovers_a_known_decay_rate() -> None:
    """The estimator has to be right, not only stable."""

    t = np.linspace(0.0, 60.0, 4801)
    for truth in (0.03, 0.06, 0.10):
        response = 0.2 + np.exp(-truth * t) * np.cos(0.8 * t)
        metrics = zonal_flow_response_metrics(
            t,
            response,
            initial_policy="first_abs",
            damping_fit_mode="period_rms_envelope",
            frequency_fit_mode="hilbert_phase",
            fit_window_tmax=30.0,
        )
        # A sliding one-period RMS reads a decaying sinusoid about 1-2 percent
        # low, because one window of a decaying signal is not one window of a
        # stationary one. That bias is deterministic and far inside the gate.
        assert metrics.gam_damping_rate == pytest.approx(truth, rel=0.03)
        assert metrics.damping_fit_tmax > metrics.damping_fit_tmin


def test_period_rms_damping_ignores_a_slowly_drifting_offset() -> None:
    """A drifting oscillation centre must not leak into the damping rate.

    The residual subtraction uses one number for the whole trace, so at early
    times the oscillation is not centred on it. Branch-wise extrema fits inherit
    that offset; a sliding one-period mean removes it wherever it sits.
    """

    t = np.linspace(0.0, 60.0, 4801)
    oscillation = np.exp(-0.06 * t) * np.cos(0.8 * t)
    flat = zonal_flow_response_metrics(
        t,
        0.2 + oscillation,
        initial_policy="first_abs",
        damping_fit_mode="period_rms_envelope",
        frequency_fit_mode="hilbert_phase",
        fit_window_tmax=30.0,
    )
    drifting = zonal_flow_response_metrics(
        t,
        0.2 + 0.08 * np.exp(-0.05 * t) + oscillation,
        initial_policy="first_abs",
        damping_fit_mode="period_rms_envelope",
        frequency_fit_mode="hilbert_phase",
        fit_window_tmax=30.0,
    )

    shift = abs(drifting.gam_damping_rate - flat.gam_damping_rate) * MERLO_R0
    assert shift < 0.1 * GAMMA_GATE_ATOL_R0_OVER_VI


# ---- the no-argument demo reports its own eigenvalue ----------------------
#
# The demo is the first thing most people run, so the number it prints is the
# first evidence anyone has about whether this code works. It is a coarse
# configuration on purpose (Nl = 7, Nm = 14), which is fine: coarse has a right
# answer too, and that answer is the eigenvalue of the operator the demo builds.
# What is not fine is printing a growth rate that disagrees with it. At the
# original dt = 0.03 over 500 steps the demo printed 0.089982 against a
# certified 0.103263, 12.9 percent low, and said so in four warnings that a
# newcomer has no way to weigh.
#
# This gate runs the demo's own settings twice -- time integration, as the demo
# does, and the certified Krylov eigensolver -- and requires them to agree. It
# is the check that would have caught the drift, and it fails if anyone shortens
# the horizon or raises the step back over the CFL bound.


def _demo_case(tmp_path):
    """Write and load the demo's deck exactly as the demo itself does."""

    from gkx.cli import default_demo_toml_text

    deck = tmp_path / "demo.toml"
    deck.write_text(default_demo_toml_text(), encoding="utf-8")
    return deck


def test_the_demo_reports_the_eigenvalue_of_the_case_it_builds(tmp_path) -> None:
    from gkx.cli import DEFAULT_DEMO_SETTINGS
    from gkx.runtime import run_runtime_linear
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    settings = DEFAULT_DEMO_SETTINGS
    cfg, _data = load_runtime_from_toml(_demo_case(tmp_path))

    shared = dict(
        ky_target=float(settings["ky"]),
        Nl=int(settings["Nl"]),
        Nm=int(settings["Nm"]),
    )
    integrated = run_runtime_linear(
        cfg,
        solver="time",
        method=str(settings["method"]),
        dt=float(settings["dt"]),
        steps=int(settings["steps"]),
        sample_stride=int(settings["sample_stride"]),
        **shared,
    )
    certified = run_runtime_linear(cfg, solver="krylov", **shared)

    assert certified.gamma > 0.0, "the demo case should be unstable"
    relative = abs(integrated.gamma - certified.gamma) / abs(certified.gamma)
    assert relative < 0.01, (
        f"the demo's time-integrated gamma {integrated.gamma:.6f} disagrees with "
        f"the certified eigenvalue {certified.gamma:.6f} of the same case by "
        f"{100 * relative:.1f}%. The demo is the first number anyone sees from "
        "this code; if its horizon is too short or its step is over the CFL "
        "bound, it prints a biased growth rate. Restore dt and steps rather "
        "than widening this tolerance."
    )
