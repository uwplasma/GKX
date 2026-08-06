from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gkx.diagnostics.analysis import ModeSelection
from gkx.benchmarking.shared import (
    LinearRunResult,
    LinearScanResult,
)
from gkx.workflows.linear import run_linear_scan, run_scan_and_mode
from gkx.diagnostics.modes import (
    compare_eigenfunctions,
    load_eigenfunction_reference_bundle,
    normalize_eigenfunction,
    phase_align_eigenfunction,
    save_eigenfunction_reference_bundle,
)
from gkx.artifacts.io import load_diagnostic_time_series
from gkx.artifacts.spectral_layout import infer_triple_dealiased_ny
from gkx.diagnostics.analysis import (
    BranchContinuationMetrics,
    LateTimeLinearMetrics,
    NonlinearHeatFluxConvergenceMetrics,
    NonlinearWindowMetrics,
    branch_continuity_metrics,
    estimate_observed_order,
    late_time_linear_metrics,
    nonlinear_heat_flux_convergence_metrics,
    windowed_nonlinear_metrics,
)
from gkx.diagnostics.growth_windows import (
    _analytic_signal,
    _explicit_time_window,
    _leading_window,
    late_time_window,
)
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
from gkx.diagnostics.modes import EigenfunctionComparisonMetrics
from gkx.diagnostics import SimulationDiagnostics
from gkx.diagnostics.zonal_validation import zonal_flow_response_metrics
from gkx.runtime import RuntimeLinearResult, RuntimeNonlinearResult


def test_normalize_eigenfunction_uses_nearest_zero() -> None:
    eig = np.array([2.0 + 0.0j, 4.0 + 0.0j, 8.0 + 0.0j])
    z = np.array([-1.0, 0.1, 0.9])

    out = normalize_eigenfunction(eig, z)

    np.testing.assert_allclose(out, eig / eig[1])


def test_normalize_eigenfunction_leaves_zero_scale_unchanged() -> None:
    eig = np.array([1.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j])
    z = np.array([-1.0, 0.0, 1.0])

    out = normalize_eigenfunction(eig, z)

    np.testing.assert_allclose(out, eig)


def test_phase_align_and_compare_eigenfunctions() -> None:
    ref = np.array([1.0 + 0.0j, 0.5 + 0.2j, -0.2 + 0.1j])
    trial = ref * np.exp(1j * 0.37)

    aligned, phase_shift = phase_align_eigenfunction(trial, ref)
    np.testing.assert_allclose(aligned, ref, atol=1.0e-12)
    assert phase_shift == pytest.approx(-0.37, abs=1.0e-12)

    metrics = compare_eigenfunctions(trial, ref)
    assert metrics.overlap == pytest.approx(1.0, abs=1.0e-12)
    assert metrics.relative_l2 == pytest.approx(0.0, abs=1.0e-12)


def test_phase_align_validates_shape_and_handles_zero_overlap() -> None:
    with pytest.raises(ValueError, match="same shape"):
        phase_align_eigenfunction(np.ones(2), np.ones(3))

    aligned, phase_shift = phase_align_eigenfunction(
        np.array([1.0 + 0.0j, 0.0j]), np.array([0.0j, 1.0 + 0.0j])
    )
    np.testing.assert_allclose(aligned, np.array([1.0 + 0.0j, 0.0j]))
    assert phase_shift == pytest.approx(0.0)


def test_compare_eigenfunctions_handles_shape_and_zero_norm() -> None:
    with pytest.raises(ValueError):
        compare_eigenfunctions(np.ones(3), np.ones(4))

    metrics = compare_eigenfunctions(
        np.zeros(3, dtype=np.complex128), np.ones(3, dtype=np.complex128)
    )
    assert np.isnan(metrics.overlap)
    assert np.isnan(metrics.relative_l2)


def test_eigenfunction_reference_bundle_roundtrip(tmp_path) -> None:
    theta = np.linspace(-2.0, 2.0, 7)
    mode = np.exp(1j * theta)
    path = tmp_path / "reference_mode.npz"

    out = save_eigenfunction_reference_bundle(
        path,
        theta=theta,
        mode=mode,
        source="GX",
        case="kbm_linear",
        metadata={"ky": 0.2, "note": "frozen"},
    )
    bundle = load_eigenfunction_reference_bundle(out)

    assert out == path
    np.testing.assert_allclose(bundle.theta, theta)
    np.testing.assert_allclose(bundle.mode, mode)
    assert bundle.source == "GX"
    assert bundle.case == "kbm_linear"
    assert bundle.metadata == {"ky": 0.2, "note": "frozen"}


def test_late_time_window_returns_tail_bounds() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    tmin, tmax = late_time_window(t, tail_fraction=0.4)

    assert tmin == pytest.approx(3.0)
    assert tmax == pytest.approx(4.0)


def test_benchmarking_window_helpers_respect_bounds_and_validation() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    mask, tmin, tmax = _leading_window(t, 0.4)
    np.testing.assert_array_equal(mask, np.array([True, True, False, False, False]))
    assert tmin == pytest.approx(0.0)
    assert tmax == pytest.approx(1.0)

    mask_explicit, window_tmin, window_tmax = _explicit_time_window(
        t, tmin=1.2, tmax=3.1
    )
    np.testing.assert_array_equal(
        mask_explicit, np.array([False, False, True, True, False])
    )
    assert window_tmin == pytest.approx(2.0)
    assert window_tmax == pytest.approx(3.0)

    with pytest.raises(ValueError):
        _leading_window(t.reshape(1, -1), 0.5)
    with pytest.raises(ValueError):
        _leading_window(np.array([]), 0.5)
    with pytest.raises(ValueError):
        _leading_window(t, 0.0)
    with pytest.raises(ValueError):
        _explicit_time_window(t, tmin=5.0, tmax=6.0)


def test_analytic_signal_recovers_quadrature_for_periodic_cosine() -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    signal = np.cos(3.0 * t)

    analytic = _analytic_signal(signal)

    np.testing.assert_allclose(np.real(analytic), signal, atol=1.0e-12)
    np.testing.assert_allclose(np.imag(analytic), np.sin(3.0 * t), atol=1.0e-12)
    np.testing.assert_allclose(np.abs(analytic), 1.0, atol=1.0e-12)

    with pytest.raises(ValueError):
        _analytic_signal(np.array([]))
    with pytest.raises(ValueError):
        _analytic_signal(np.ones((2, 2)))


def test_infer_triple_dealiased_ny_matches_gx_grid_convention() -> None:
    assert infer_triple_dealiased_ny(5) == 13
    assert infer_triple_dealiased_ny(9) == 25
    with pytest.raises(ValueError):
        infer_triple_dealiased_ny(1)


def test_scalar_gate_reports_near_zero_and_failure_modes() -> None:
    gate = evaluate_scalar_gate(
        "omega", 1.0e-4, 0.0, atol=2.0e-4, rtol=0.0, units="v_t/R"
    )

    assert isinstance(gate, ScalarGateResult)
    assert gate.passed is True
    assert gate.rel_error == float("inf")
    assert gate.units == "v_t/R"

    failed = evaluate_scalar_gate("gamma", 1.3, 1.0, atol=0.0, rtol=0.1)
    assert failed.passed is False
    assert failed.abs_error == pytest.approx(0.3)
    assert failed.rel_error == pytest.approx(0.3)

    with pytest.raises(ValueError):
        evaluate_scalar_gate("bad", 1.0, 1.0, atol=-1.0, rtol=0.0)
    with pytest.raises(ValueError):
        evaluate_scalar_gate("bad", 1.0, 1.0, atol=0.0, rtol=-1.0)


def test_gate_report_is_json_ready_and_requires_gates() -> None:
    passed = evaluate_scalar_gate("gamma", 1.01, 1.0, atol=0.0, rtol=0.02)
    failed = evaluate_scalar_gate("omega", 0.7, 1.0, atol=0.0, rtol=0.02)
    report = gate_report("cyclone_linear", "GX", [passed, failed])

    assert isinstance(report, GateReport)
    assert report.passed is False
    assert report.max_abs_error == pytest.approx(0.3)
    as_dict = gate_report_to_dict(report)
    assert as_dict["case"] == "cyclone_linear"
    assert as_dict["source"] == "GX"
    assert as_dict["passed"] is False
    assert len(as_dict["gates"]) == 2

    with pytest.raises(ValueError):
        gate_report("empty", "none", [])


def test_linear_metrics_gate_report_uses_growth_and_frequency() -> None:
    ref = LateTimeLinearMetrics(
        gamma_fit=0.1,
        omega_fit=0.3,
        gamma_tail_mean=0.1,
        omega_tail_mean=0.3,
        gamma_tail_std=0.0,
        omega_tail_std=0.0,
        tmin=5.0,
        tmax=10.0,
        nsamples=20,
        signal_source="reference",
    )
    obs = LateTimeLinearMetrics(
        gamma_fit=0.104,
        omega_fit=0.298,
        gamma_tail_mean=0.104,
        omega_tail_mean=0.298,
        gamma_tail_std=0.001,
        omega_tail_std=0.002,
        tmin=5.0,
        tmax=10.0,
        nsamples=20,
        signal_source="gkx",
    )

    report = linear_metrics_gate_report(
        obs, ref, case="cyclone_linear", source="GX", gamma_rtol=0.05, omega_rtol=0.01
    )

    assert report.passed is True
    assert [gate.metric for gate in report.gates] == ["gamma_fit", "omega_fit"]


def test_nonlinear_and_zonal_gate_reports_cover_publication_metrics() -> None:
    ref_nonlin = NonlinearWindowMetrics(
        tmin=20.0,
        tmax=50.0,
        nsamples=12,
        heat_flux_mean=4.0,
        heat_flux_std=0.4,
        heat_flux_rms=4.1,
        wphi_mean=2.0,
        wphi_std=0.2,
        wg_mean=3.0,
        wg_std=0.3,
        phi_mode_envelope_mean=0.5,
        phi_mode_envelope_std=0.05,
        phi_mode_envelope_max=0.7,
    )
    obs_nonlin = NonlinearWindowMetrics(
        tmin=20.0,
        tmax=50.0,
        nsamples=12,
        heat_flux_mean=4.2,
        heat_flux_std=0.5,
        heat_flux_rms=4.25,
        wphi_mean=2.1,
        wphi_std=0.25,
        wg_mean=3.1,
        wg_std=0.35,
        phi_mode_envelope_mean=0.52,
        phi_mode_envelope_std=0.05,
        phi_mode_envelope_max=0.72,
    )

    nonlin_report = nonlinear_window_gate_report(
        obs_nonlin,
        ref_nonlin,
        case="w7x_nonlinear",
        source="GX",
        rtol=0.1,
    )
    assert nonlin_report.passed is True
    assert "heat_flux_mean" in {gate.metric for gate in nonlin_report.gates}
    assert "phi_mode_envelope_mean" in {gate.metric for gate in nonlin_report.gates}

    ref_zonal = ZonalFlowResponseMetrics(
        initial_level=1.0,
        initial_policy="first_abs",
        residual_level=0.19,
        residual_std=0.01,
        response_rms=0.2,
        gam_frequency=2.24,
        gam_damping_rate=0.17,
        damping_method="branchwise_extrema",
        frequency_method="hilbert_phase",
        peak_count=6,
        peak_fit_count=4,
        tmin=30.0,
        tmax=60.0,
        fit_tmin=0.0,
        fit_tmax=30.0,
        peak_times=np.array([1.0, 2.0]),
        peak_envelope=np.array([0.5, 0.4]),
        max_peak_times=np.array([1.0]),
        max_peak_values=np.array([0.5]),
        min_peak_times=np.array([2.0]),
        min_peak_values=np.array([-0.4]),
    )
    obs_zonal = ZonalFlowResponseMetrics(
        initial_level=1.0,
        initial_policy="first_abs",
        residual_level=0.192,
        residual_std=0.012,
        response_rms=0.21,
        gam_frequency=2.20,
        gam_damping_rate=0.176,
        damping_method="branchwise_extrema",
        frequency_method="hilbert_phase",
        peak_count=6,
        peak_fit_count=4,
        tmin=30.0,
        tmax=60.0,
        fit_tmin=0.0,
        fit_tmax=30.0,
        peak_times=np.array([1.0, 2.0]),
        peak_envelope=np.array([0.5, 0.4]),
        max_peak_times=np.array([1.0]),
        max_peak_values=np.array([0.5]),
        min_peak_times=np.array([2.0]),
        min_peak_values=np.array([-0.4]),
    )

    zonal_report = zonal_response_gate_report(
        obs_zonal,
        ref_zonal,
        case="merlo_case_iii",
        source="Merlo et al.",
        residual_atol=0.01,
        frequency_atol=0.1,
        damping_atol=0.02,
    )
    assert zonal_report.passed is True
    assert [gate.metric for gate in zonal_report.gates] == [
        "residual_level",
        "gam_frequency",
        "gam_damping_rate",
    ]


def test_eigenfunction_gate_report_handles_open_and_closed_artifacts() -> None:
    closed = eigenfunction_gate_report(
        EigenfunctionComparisonMetrics(overlap=0.97, relative_l2=0.12, phase_shift=0.4),
        case="kbm_eigenfunction",
        source="GX",
        min_overlap=0.95,
        max_relative_l2=0.25,
    )
    assert closed.passed is True
    assert [gate.metric for gate in closed.gates] == [
        "eigenfunction_overlap",
        "eigenfunction_relative_l2",
    ]

    open_report = eigenfunction_gate_report(
        EigenfunctionComparisonMetrics(overlap=0.63, relative_l2=0.79, phase_shift=0.0),
        case="kbm_eigenfunction",
        source="GX",
        min_overlap=0.95,
        max_relative_l2=0.25,
    )
    assert open_report.passed is False
    assert open_report.gates[0].passed is False
    assert open_report.gates[1].passed is False

    with pytest.raises(ValueError):
        eigenfunction_gate_report(
            EigenfunctionComparisonMetrics(
                overlap=1.0, relative_l2=0.0, phase_shift=0.0
            ),
            case="bad",
            source="GX",
            min_overlap=1.2,
        )
    with pytest.raises(ValueError):
        eigenfunction_gate_report(
            EigenfunctionComparisonMetrics(
                overlap=1.0, relative_l2=0.0, phase_shift=0.0
            ),
            case="bad",
            source="GX",
            max_relative_l2=-1.0,
        )


def test_zonal_flow_response_metrics_recover_residual_and_gam_envelope() -> None:
    t = np.linspace(0.0, 30.0, 3001)
    response = 0.2 + np.exp(-0.1 * t) * np.cos(2.0 * t)

    metrics = zonal_flow_response_metrics(
        t, response, tail_fraction=0.25, initial_fraction=0.05
    )

    assert metrics.initial_policy == "window_abs_mean"
    assert metrics.residual_level * metrics.initial_level == pytest.approx(
        0.2, abs=0.05
    )
    assert metrics.gam_frequency == pytest.approx(2.0, rel=0.1)
    assert metrics.gam_damping_rate == pytest.approx(0.1, rel=0.2)
    assert metrics.peak_count >= 3


def test_zonal_flow_response_metrics_support_first_sample_rh_normalization() -> None:
    t = np.linspace(0.0, 10.0, 101)
    response = 5.0 * (0.2 + 0.8 * np.exp(-0.7 * t) * np.cos(1.5 * t))

    metrics = zonal_flow_response_metrics(
        t,
        response,
        tail_fraction=0.2,
        initial_fraction=0.2,
        initial_policy="first-abs",
    )

    assert metrics.initial_policy == "first_abs"
    assert metrics.initial_level == pytest.approx(abs(response[0]))
    assert metrics.residual_level == pytest.approx(0.2, abs=0.03)


def test_zonal_flow_response_metrics_support_external_initial_level_override() -> None:
    t = np.linspace(0.0, 12.0, 241)
    response = 0.6 + np.exp(-0.7 * t) * np.cos(1.5 * t)

    metrics = zonal_flow_response_metrics(
        t,
        response,
        tail_fraction=0.25,
        initial_policy="first_abs",
        initial_level_override=3.0,
    )

    assert metrics.initial_policy == "first_abs"
    assert metrics.initial_level == pytest.approx(3.0)
    assert metrics.residual_level == pytest.approx(0.2, abs=0.02)


def test_zonal_flow_response_metrics_can_limit_damping_fit_to_early_peaks() -> None:
    t = np.linspace(0.0, 24.0, 2401)
    envelope = np.exp(-0.22 * np.minimum(t, 9.0)) * np.exp(
        0.08 * np.maximum(t - 9.0, 0.0)
    )
    response = 0.2 + envelope * np.cos(2.2 * t)

    metrics_all = zonal_flow_response_metrics(
        t, response, tail_fraction=0.25, initial_policy="first_abs"
    )
    metrics_early = zonal_flow_response_metrics(
        t,
        response,
        tail_fraction=0.25,
        initial_policy="first_abs",
        peak_fit_max_peaks=5,
    )

    assert metrics_all.peak_count >= metrics_early.peak_fit_count
    assert metrics_early.peak_fit_count == 5
    assert metrics_early.gam_damping_rate > metrics_all.gam_damping_rate
    assert metrics_early.gam_damping_rate == pytest.approx(0.22, abs=0.08)


def test_zonal_flow_response_metrics_support_branchwise_merlo_style_fits() -> None:
    t = np.linspace(0.0, 60.0, 6001)
    base = 0.2 + np.exp(-0.06 * t) * np.cos(0.8 * t)
    recurrence = np.where(
        t > 30.0,
        0.18 * (1.0 - np.exp(-0.12 * (t - 30.0))) * np.cos(0.8 * t + 0.2),
        0.0,
    )
    response = base + recurrence

    metrics = zonal_flow_response_metrics(
        t,
        response,
        initial_policy="first_abs",
        damping_fit_mode="branchwise_extrema",
        frequency_fit_mode="hilbert_phase",
        fit_window_tmax=30.0,
        peak_fit_max_peaks=4,
    )

    assert metrics.damping_method == "branchwise_extrema"
    assert metrics.frequency_method == "hilbert_phase"
    assert metrics.gam_damping_rate == pytest.approx(0.06, abs=0.01)
    assert metrics.gam_frequency == pytest.approx(0.8, abs=0.05)
    assert metrics.peak_fit_count == 7
    assert metrics.fit_tmax == pytest.approx(30.0, abs=0.1)


def test_zonal_flow_response_metrics_validate_input_and_handle_nonoscillatory_signal() -> (
    None
):
    with pytest.raises(ValueError):
        zonal_flow_response_metrics(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        zonal_flow_response_metrics(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0])
        )
    with pytest.raises(ValueError):
        zonal_flow_response_metrics(
            np.arange(5.0), np.ones(5), initial_policy="unknown"
        )
    with pytest.raises(ValueError):
        zonal_flow_response_metrics(np.arange(5.0), np.ones(5), peak_fit_max_peaks=0)
    with pytest.raises(ValueError):
        zonal_flow_response_metrics(
            np.arange(5.0), np.ones(5), damping_fit_mode="unknown"
        )
    with pytest.raises(ValueError):
        zonal_flow_response_metrics(
            np.arange(5.0), np.ones(5), frequency_fit_mode="unknown"
        )
    with pytest.raises(ValueError):
        zonal_flow_response_metrics(
            np.arange(5.0), np.ones(5), hilbert_trim_fraction=0.5
        )
    with pytest.raises(ValueError):
        zonal_flow_response_metrics(
            np.arange(5.0), np.ones(5), initial_level_override=0.0
        )

    t = np.linspace(0.0, 5.0, 101)
    response = np.exp(-t)
    metrics = zonal_flow_response_metrics(t, response)
    assert np.isnan(metrics.gam_frequency)
    assert np.isnan(metrics.gam_damping_rate)


def test_zonal_flow_response_metrics_rejects_insufficient_finite_samples() -> None:
    with pytest.raises(ValueError, match="four finite samples"):
        zonal_flow_response_metrics(
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, np.nan, 0.8, np.nan, 0.6]),
        )


def test_load_diagnostic_time_series_reads_gx_style_netcdf(tmp_path) -> None:
    import netCDF4 as nc

    path = tmp_path / "diag.out.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", 4)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = np.array(
            [0.0, 1.0, 2.0, 3.0]
        )
        diag.createVariable("Phi2_zonal_t", "f8", ("time",))[:] = np.array(
            [1.0, 0.7, 0.5, 0.4]
        )

    series = load_diagnostic_time_series(path, variable="Phi2_zonal_t")

    assert np.allclose(series.t, [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(series.values, [1.0, 0.7, 0.5, 0.4])
    assert series.variable == "Phi2_zonal_t"


def test_load_diagnostic_time_series_rejects_missing_variable(tmp_path) -> None:
    import netCDF4 as nc

    path = tmp_path / "diag.out.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", 2)
        grids = ds.createGroup("Grids")
        ds.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        load_diagnostic_time_series(path, variable="Phi2_zonal_t")


def test_load_diagnostic_time_series_extracts_complex_kx_trace_with_phase_alignment(
    tmp_path,
) -> None:
    import netCDF4 as nc

    path = tmp_path / "diag.out.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", 3)
        ds.createDimension("kx", 2)
        ds.createDimension("ri", 2)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = np.array([0.0, 1.0, 2.0])
        raw = np.array(
            [
                [[0.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.5]],
                [[0.0, 0.0], [0.0, -0.25]],
            ],
            dtype=float,
        )
        diag.createVariable("Phi_zonal_mode_kxt", "f8", ("time", "kx", "ri"))[:] = raw

    series = load_diagnostic_time_series(
        path,
        variable="Phi_zonal_mode_kxt",
        kx_index=1,
        component="real",
        align_phase=True,
    )

    assert np.allclose(series.t, [0.0, 1.0, 2.0])
    assert np.allclose(series.values, [1.0, 0.5, -0.25])


def test_load_diagnostic_time_series_covers_components_and_validation(tmp_path) -> None:
    import netCDF4 as nc

    path = tmp_path / "diagnostics.out.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", 3)
        ds.createDimension("kx", 2)
        ds.createDimension("ri", 2)
        ds.createDimension("extra", 2)
        ds.createVariable("time", "f8", ("time",))[:] = np.array([0.0, 1.0, 2.0])
        diag = ds.createGroup("Diagnostics")
        diag.createVariable("Real2D", "f8", ("time", "kx"))[:, :] = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        )
        diag.createVariable("Real3D", "f8", ("time", "kx", "extra"))[:, :, :] = np.ones(
            (3, 2, 2)
        )
        raw = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, -2.0]],
                [[4.0, 0.0], [3.0, 4.0]],
            ],
            dtype=float,
        )
        diag.createVariable("ComplexMode", "f8", ("time", "kx", "ri"))[:, :, :] = raw

    real_abs = load_diagnostic_time_series(
        path, variable="Real2D", kx_index=1, component="abs"
    )
    np.testing.assert_allclose(real_abs.values, [2.0, 4.0, 6.0])
    complex_series = load_diagnostic_time_series(
        path, variable="ComplexMode", kx_index=1, component="complex"
    )
    np.testing.assert_allclose(complex_series.values, [1.0j, -2.0j, 3.0 + 4.0j])
    imag = load_diagnostic_time_series(
        path, variable="ComplexMode", kx_index=1, component="imag"
    )
    np.testing.assert_allclose(imag.values, [1.0, -2.0, 4.0])
    magnitude = load_diagnostic_time_series(
        path, variable="ComplexMode", kx_index=1, component="abs"
    )
    np.testing.assert_allclose(magnitude.values, [1.0, 2.0, 5.0])

    with pytest.raises(ValueError, match="requires kx_index"):
        load_diagnostic_time_series(path, variable="Real2D")
    with pytest.raises(ValueError, match="1D time series"):
        load_diagnostic_time_series(path, variable="Real3D")
    with pytest.raises(ValueError, match="real diagnostics"):
        load_diagnostic_time_series(
            path, variable="Real2D", kx_index=0, component="imag"
        )
    with pytest.raises(ValueError, match="component"):
        load_diagnostic_time_series(
            path, variable="ComplexMode", kx_index=0, component="phase"
        )

    missing_group = tmp_path / "missing_group.out.nc"
    with nc.Dataset(missing_group, "w") as ds:
        ds.createDimension("time", 1)
        ds.createVariable("time", "f8", ("time",))[:] = np.array([0.0])
    with pytest.raises(ValueError, match="missing NetCDF group"):
        load_diagnostic_time_series(missing_group, variable="Real2D")

    missing_time = tmp_path / "missing_time.out.nc"
    with nc.Dataset(missing_time, "w") as ds:
        ds.createDimension("time", 1)
        diag = ds.createGroup("Diagnostics")
        diag.createVariable("Real1D", "f8", ("time",))[:] = np.array([1.0])
    with pytest.raises(ValueError, match="missing time variable"):
        load_diagnostic_time_series(missing_time, variable="Real1D")


def test_run_linear_scan_applies_resolution_and_krylov_policies() -> None:
    calls: list[dict[str, object]] = []

    def fake_run_linear_fn(**kwargs):
        calls.append(kwargs)
        ky = float(kwargs["ky_target"])
        return SimpleNamespace(gamma=ky + 1.0, omega=-ky, ky=ky + 0.01)

    result = run_linear_scan(
        ky_values=np.array([0.1, 0.3]),
        run_linear_fn=fake_run_linear_fn,
        cfg=object(),
        Nl=2,
        Nm=3,
        dt=np.array([0.01, 0.02]),
        steps=np.array([10, 20]),
        method="rk4",
        solver="time",
        krylov_cfg="base",
        window_kw={"window_fraction": 0.5},
        tmin=np.array([1.0, 2.0]),
        tmax=np.array([3.0, 4.0]),
        auto_window=False,
        run_kwargs={"tag": "ok"},
        resolution_policy=lambda ky: (4, 5) if ky < 0.2 else (6, 7),
        krylov_policy=lambda ky: f"kcfg-{ky:.1f}",
    )

    assert isinstance(result, LinearScanResult)
    np.testing.assert_allclose(result.ky, [0.11, 0.31])
    np.testing.assert_allclose(result.gamma, [1.1, 1.3])
    np.testing.assert_allclose(result.omega, [-0.1, -0.3])
    assert calls[0]["Nl"] == 4
    assert calls[0]["Nm"] == 5
    assert calls[0]["krylov_cfg"] == "kcfg-0.1"
    assert calls[0]["tmin"] == 1.0
    assert calls[1]["Nl"] == 6
    assert calls[1]["Nm"] == 7
    assert calls[1]["krylov_cfg"] == "kcfg-0.3"
    assert calls[1]["tmax"] == 4.0
    assert calls[1]["tag"] == "ok"


def test_run_linear_scan_accepts_an_empty_scan() -> None:
    result = run_linear_scan(
        ky_values=np.asarray([], dtype=float),
        run_linear_fn=lambda **_kwargs: None,
        cfg=SimpleNamespace(),
        Nl=4,
        Nm=8,
        dt=0.1,
        steps=2,
        method="rk2",
        solver="time",
        krylov_cfg=None,
        window_kw={},
    )

    assert result.ky.shape == (0,)
    assert result.gamma.shape == (0,)
    assert result.omega.shape == (0,)


def test_run_scan_and_mode_uses_selected_ky_and_fit_window(monkeypatch) -> None:
    selection = ModeSelection(ky_index=0, kx_index=0, z_index=1)
    run = LinearRunResult(
        t=np.array([0.0, 1.0, 2.0]),
        phi_t=np.ones((3, 1, 1, 3), dtype=np.complex128),
        gamma=0.4,
        omega=-0.2,
        ky=0.3,
        selection=selection,
    )
    calls: list[dict[str, object]] = []

    def fake_linear_fn(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setattr(
        "gkx.workflows.linear.extract_mode_time_series",
        lambda phi_t, sel, method: np.array([1.0 + 0.0j, 2.0 + 0.0j, 4.0 + 0.0j]),
    )
    monkeypatch.setattr(
        "gkx.workflows.linear.fit_growth_rate_auto",
        lambda t, signal, **kwargs: (0.5, -0.1, 0.25, 1.75),
    )
    monkeypatch.setattr(
        "gkx.workflows.linear.extract_eigenfunction",
        lambda phi_t, t, selection, z, method, tmin, tmax: np.array([1.0, 2.0, 3.0]),
    )
    monkeypatch.setattr(
        "gkx.workflows.linear.build_spectral_grid",
        lambda _grid: SimpleNamespace(z=np.array([-1.0, 0.0, 1.0])),
    )
    cfg = SimpleNamespace(grid=object())

    result = run_scan_and_mode(
        ky_values=np.array([0.1, 0.3]),
        linear_fn=fake_linear_fn,
        cfg=cfg,
        Nl=2,
        Nm=2,
        dt=np.array([0.1, 0.2]),
        steps=np.array([5, 6]),
        method="rk2",
        solver="time",
        mode_solver="krylov",
        krylov_cfg="kcfg",
        window_kw={"window_fraction": 0.5},
        resolution_policy=lambda ky: (3, 4) if ky < 0.2 else (5, 6),
    )

    assert result.ky_selected == 0.3
    np.testing.assert_allclose(result.scan.gamma, [0.4, 0.4])
    np.testing.assert_allclose(result.eigenfunction, [1.0, 2.0, 3.0])
    assert result.tmin == 0.25
    assert result.tmax == 1.75
    assert calls[0]["solver"] == "time"
    assert calls[1]["solver"] == "time"
    assert calls[2]["solver"] == "krylov"
    assert calls[2]["Nl"] == 5
    assert calls[2]["Nm"] == 6
    assert calls[2]["ky_target"] == 0.3


def test_run_scan_and_mode_short_trace_skips_fit(monkeypatch) -> None:
    run = LinearRunResult(
        t=np.array([0.0]),
        phi_t=np.ones((1, 1, 1, 2), dtype=np.complex128),
        gamma=0.2,
        omega=0.1,
        ky=0.2,
        selection=ModeSelection(ky_index=0, kx_index=0),
    )

    monkeypatch.setattr(
        "gkx.workflows.linear.build_spectral_grid",
        lambda _grid: SimpleNamespace(z=np.array([-1.0, 1.0])),
    )
    monkeypatch.setattr(
        "gkx.workflows.linear.extract_eigenfunction",
        lambda *args, **kwargs: np.array([1.0, -1.0]),
    )

    result = run_scan_and_mode(
        ky_values=np.array([0.2]),
        linear_fn=lambda **kwargs: run,
        cfg=SimpleNamespace(grid=object()),
        Nl=1,
        Nm=1,
        dt=0.1,
        steps=2,
        method="rk2",
        solver="time",
        mode_solver="time",
        krylov_cfg=None,
        window_kw={"window_fraction": 0.5},
        select_ky=lambda scan: float(scan.ky[0]),
    )

    assert result.tmin is None
    assert result.tmax is None
    np.testing.assert_allclose(result.eigenfunction, [1.0, -1.0])


def test_late_time_linear_metrics_from_linear_run_result() -> None:
    gamma = 0.35
    omega = -0.18
    t = np.linspace(0.0, 4.0, 9)
    z_profile = np.array([1.0, 0.5 - 0.25j])
    signal = np.exp((gamma - 1j * omega) * t)
    phi_t = signal[:, None, None, None] * z_profile[None, None, None, :]
    run = LinearRunResult(
        t=t,
        phi_t=phi_t,
        gamma=gamma,
        omega=omega,
        ky=0.3,
        selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
        gamma_t=np.full_like(t, gamma, dtype=float),
        omega_t=np.full_like(t, omega, dtype=float),
    )

    metrics = late_time_linear_metrics(run, tail_fraction=0.5)

    assert metrics.signal_source == "phi_t:project"
    assert metrics.nsamples == 5
    assert metrics.gamma_fit == pytest.approx(gamma, rel=1.0e-3)
    assert metrics.omega_fit == pytest.approx(omega, rel=1.0e-3)
    assert metrics.gamma_tail_mean == pytest.approx(gamma)
    assert metrics.omega_tail_mean == pytest.approx(omega)
    assert metrics.tmin == pytest.approx(2.0)
    assert metrics.tmax == pytest.approx(4.0)


def test_late_time_linear_metrics_runtime_signal_and_scalar_fallback() -> None:
    gamma = 0.22
    omega = -0.07
    t = np.linspace(0.0, 2.0, 5)
    signal = np.exp((gamma - 1j * omega) * t)
    runtime = RuntimeLinearResult(
        ky=0.2,
        gamma=gamma,
        omega=omega,
        selection=ModeSelection(ky_index=0, kx_index=0),
        t=t,
        signal=signal,
    )

    metrics = late_time_linear_metrics(runtime, tail_fraction=0.4)
    assert metrics.signal_source == "signal"
    assert metrics.gamma_fit == pytest.approx(gamma, rel=1.0e-3)
    assert metrics.omega_fit == pytest.approx(omega, rel=1.0e-3)
    assert metrics.nsamples == 2

    scalar_only = late_time_linear_metrics(SimpleNamespace(gamma=0.1, omega=-0.2))
    assert scalar_only.signal_source == "scalar"
    assert scalar_only.gamma_fit == pytest.approx(0.1)
    assert scalar_only.omega_fit == pytest.approx(-0.2)
    assert scalar_only.nsamples == 1


def test_windowed_nonlinear_metrics_from_runtime_result() -> None:
    diagnostics = SimulationDiagnostics(
        t=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        dt_t=np.full(5, 0.1),
        dt_mean=np.full(5, 0.1),
        gamma_t=np.zeros(5),
        omega_t=np.zeros(5),
        Wg_t=np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
        Wphi_t=np.array([0.5, 0.75, 1.0, 1.25, 1.5]),
        Wapar_t=np.zeros(5),
        heat_flux_t=np.array([0.0, 0.2, 0.4, 0.6, 0.8]),
        particle_flux_t=np.zeros(5),
        energy_t=np.zeros(5),
        phi_mode_t=np.array([0.0, 1.0 + 0.0j, 1.0 + 1.0j, 2.0 + 0.0j, 2.0 + 1.0j]),
    )
    result = RuntimeNonlinearResult(
        t=np.asarray(diagnostics.t), diagnostics=diagnostics
    )

    metrics = windowed_nonlinear_metrics(result, start_fraction=0.6)

    assert metrics.nsamples == 2
    assert metrics.tmin == pytest.approx(3.0)
    assert metrics.tmax == pytest.approx(4.0)
    assert metrics.heat_flux_mean == pytest.approx(0.7)
    assert metrics.wphi_mean == pytest.approx(1.375)
    assert metrics.wg_mean == pytest.approx(2.75)
    assert metrics.phi_mode_envelope_max == pytest.approx(np.sqrt(5.0))


def test_windowed_nonlinear_metrics_rejects_missing_or_empty_diagnostics() -> None:
    with pytest.raises(ValueError):
        windowed_nonlinear_metrics(
            RuntimeNonlinearResult(t=np.array([]), diagnostics=None)
        )

    bad = SimulationDiagnostics(
        t=np.array([0.0, 1.0]),
        dt_t=np.full(2, 0.1),
        dt_mean=np.full(2, 0.1),
        gamma_t=np.zeros(2),
        omega_t=np.zeros(2),
        Wg_t=np.array([np.nan, np.nan]),
        Wphi_t=np.array([1.0, 2.0]),
        Wapar_t=np.zeros(2),
        heat_flux_t=np.array([1.0, 2.0]),
        particle_flux_t=np.zeros(2),
        energy_t=np.zeros(2),
    )
    with pytest.raises(ValueError):
        windowed_nonlinear_metrics(bad)


def test_late_time_linear_metrics_without_signal_uses_tail_stats() -> None:
    t = np.linspace(0.0, 4.0, 5)
    result = SimpleNamespace(
        t=t,
        gamma=0.2,
        omega=-0.1,
        gamma_t=np.array([np.nan, 0.1, 0.2, 0.3, 0.4]),
        omega_t=np.array([np.nan, -0.2, -0.3, -0.4, -0.5]),
    )

    metrics = late_time_linear_metrics(result, tail_fraction=0.4)

    assert metrics.signal_source == "scalar"
    assert metrics.gamma_fit == pytest.approx(0.2)
    assert metrics.omega_fit == pytest.approx(-0.1)
    assert metrics.gamma_tail_mean == pytest.approx(0.35)
    assert metrics.omega_tail_mean == pytest.approx(-0.45)
    assert metrics.gamma_tail_std == pytest.approx(0.05)
    assert metrics.omega_tail_std == pytest.approx(0.05)


def test_late_time_linear_and_windowed_nonlinear_metrics_validate_inputs() -> None:
    with pytest.raises(ValueError):
        late_time_linear_metrics(
            SimpleNamespace(t=np.array([0.0, 1.0]), gamma=0.1, omega=0.2),
            tail_fraction=0.0,
        )
    with pytest.raises(ValueError):
        late_time_linear_metrics(
            SimpleNamespace(t=np.array([[0.0, 1.0]]), gamma=0.1, omega=0.2)
        )
    with pytest.raises(ValueError):
        late_time_linear_metrics(SimpleNamespace(t=np.array([]), gamma=0.1, omega=0.2))

    bad_t = SimulationDiagnostics(
        t=np.array([[0.0, 1.0]]),
        dt_t=np.full(2, 0.1),
        dt_mean=np.full(2, 0.1),
        gamma_t=np.zeros(2),
        omega_t=np.zeros(2),
        Wg_t=np.ones(2),
        Wphi_t=np.ones(2),
        Wapar_t=np.zeros(2),
        heat_flux_t=np.ones(2),
        particle_flux_t=np.zeros(2),
        energy_t=np.zeros(2),
    )
    with pytest.raises(ValueError):
        windowed_nonlinear_metrics(bad_t)
    with pytest.raises(ValueError):
        windowed_nonlinear_metrics(
            SimpleNamespace(
                diagnostics=SimulationDiagnostics(
                    t=np.array([0.0, 1.0]),
                    dt_t=np.full(2, 0.1),
                    dt_mean=np.full(2, 0.1),
                    gamma_t=np.zeros(2),
                    omega_t=np.zeros(2),
                    Wg_t=np.ones(2),
                    Wphi_t=np.ones(2),
                    Wapar_t=np.zeros(2),
                    heat_flux_t=np.ones(2),
                    particle_flux_t=np.zeros(2),
                    energy_t=np.zeros(2),
                )
            ),
            start_fraction=1.0,
        )


def test_windowed_nonlinear_metrics_ignores_nonfinite_phi_envelope_and_keeps_window_stats() -> (
    None
):
    diagnostics = SimulationDiagnostics(
        t=np.array([0.0, 1.0, 2.0, 3.0]),
        dt_t=np.full(4, 0.1),
        dt_mean=np.full(4, 0.1),
        gamma_t=np.zeros(4),
        omega_t=np.zeros(4),
        Wg_t=np.array([0.0, 1.0, 2.0, 3.0]),
        Wphi_t=np.array([0.0, 0.5, 1.0, 1.5]),
        Wapar_t=np.zeros(4),
        heat_flux_t=np.array([0.0, 0.2, 0.4, 0.6]),
        particle_flux_t=np.zeros(4),
        energy_t=np.zeros(4),
        phi_mode_t=np.array([np.nan + 0.0j, 1.0 + 0.0j, np.nan + 0.0j, 2.0 + 0.0j]),
    )

    metrics = windowed_nonlinear_metrics(diagnostics, start_fraction=0.5)

    assert metrics.nsamples == 2
    assert metrics.heat_flux_mean == pytest.approx(0.5)
    assert metrics.wphi_mean == pytest.approx(1.25)
    assert metrics.wg_mean == pytest.approx(2.5)
    assert metrics.phi_mode_envelope_mean == pytest.approx(2.0)
    assert metrics.phi_mode_envelope_std == pytest.approx(0.0)
    assert metrics.phi_mode_envelope_max == pytest.approx(2.0)


def test_nonlinear_window_convergence_metrics_pass_stable_post_transient_average() -> (
    None
):
    t = np.linspace(0.0, 19.0, 20)
    heat_flux = 2.0 + 0.02 * np.sin(np.arange(t.size))

    metrics = nonlinear_heat_flux_convergence_metrics(
        t,
        heat_flux,
        start_fraction=0.5,
        terminal_fraction=0.5,
    )
    report = nonlinear_heat_flux_convergence_gate_report(
        metrics,
        case="synthetic_nonlinear",
        source="unit-test",
        max_mean_rel_delta=0.03,
        max_cv=0.02,
        max_abs_trend=0.03,
        min_samples=8,
    )

    assert isinstance(metrics, NonlinearHeatFluxConvergenceMetrics)
    assert metrics.nsamples == 10
    assert metrics.terminal_nsamples == 5
    assert metrics.tmin == pytest.approx(10.0)
    assert metrics.terminal_tmin == pytest.approx(15.0)
    assert metrics.heat_flux_mean == pytest.approx(np.mean(heat_flux[10:]))
    assert metrics.mean_rel_delta < 0.01
    assert report.passed is True
    assert [gate.metric for gate in report.gates] == [
        "heat_flux_terminal_mean_rel_delta",
        "heat_flux_window_cv",
        "heat_flux_window_abs_trend",
        "heat_flux_window_sample_deficit",
    ]


def test_nonlinear_window_convergence_gate_rejects_drifting_reduced_window_proxy() -> (
    None
):
    t = np.linspace(0.0, 19.0, 20)
    heat_flux = np.where(t < 10.0, 4.0, 2.0 + 0.08 * (t - 10.0))

    metrics = nonlinear_heat_flux_convergence_metrics(t, heat_flux, start_fraction=0.5)
    report = nonlinear_heat_flux_convergence_gate_report(
        metrics,
        case="drifting_nonlinear",
        source="unit-test",
        max_mean_rel_delta=0.03,
        max_cv=0.02,
        max_abs_trend=0.03,
        min_samples=12,
    )

    assert metrics.mean_rel_delta > 0.05
    assert metrics.abs_trend > 0.25
    assert report.passed is False
    failed = {gate.metric for gate in report.gates if not gate.passed}
    assert failed == {
        "heat_flux_terminal_mean_rel_delta",
        "heat_flux_window_cv",
        "heat_flux_window_abs_trend",
        "heat_flux_window_sample_deficit",
    }


def test_nonlinear_window_convergence_metrics_validate_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        nonlinear_heat_flux_convergence_metrics(np.array([0.0, 1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="start_fraction"):
        nonlinear_heat_flux_convergence_metrics(
            np.arange(3.0), np.ones(3), start_fraction=1.0
        )
    with pytest.raises(ValueError, match="terminal_fraction"):
        nonlinear_heat_flux_convergence_metrics(
            np.arange(3.0), np.ones(3), terminal_fraction=0.0
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        nonlinear_heat_flux_convergence_metrics(np.array([0.0, 0.0, 1.0]), np.ones(3))
    with pytest.raises(ValueError, match="finite paired sample"):
        nonlinear_heat_flux_convergence_metrics(
            np.array([np.nan, np.inf]), np.array([1.0, 2.0])
        )


def test_estimate_observed_order_returns_asymptotic_pairwise_orders() -> None:
    step_sizes = np.array([0.4, 0.2, 0.1, 0.05])
    errors = 3.0 * step_sizes**2

    metrics = estimate_observed_order(step_sizes, errors)

    np.testing.assert_allclose(metrics.orders, [2.0, 2.0, 2.0], atol=1.0e-12)
    assert metrics.asymptotic_order == pytest.approx(2.0)

    with pytest.raises(ValueError):
        estimate_observed_order(np.array([0.1]), np.array([0.01]))
    with pytest.raises(ValueError):
        estimate_observed_order(np.array([0.2, 0.2]), np.array([0.1, 0.025]))
    with pytest.raises(ValueError):
        estimate_observed_order(np.array([0.2, np.nan]), np.array([0.1, 0.025]))
    with pytest.raises(ValueError):
        estimate_observed_order(np.array([0.2, -0.1]), np.array([0.1, 0.025]))
    with pytest.raises(ValueError):
        estimate_observed_order(np.array([0.2, 0.1]), np.array([0.1, 0.0]))


def test_observed_order_gate_report_tracks_rate_and_final_error() -> None:
    metrics = estimate_observed_order(
        np.array([0.4, 0.2, 0.1]), 2.0 * np.array([0.4, 0.2, 0.1]) ** 2
    )

    report = observed_order_gate_report(
        metrics,
        case="rk2_manufactured",
        source="closed-form",
        min_asymptotic_order=1.95,
        min_pairwise_order=1.95,
        max_final_error=0.03,
    )

    assert report.passed is True
    assert [gate.metric for gate in report.gates] == [
        "observed_order_deficit",
        "min_pairwise_order_deficit",
        "final_error",
    ]

    failed = observed_order_gate_report(
        metrics,
        case="rk2_manufactured",
        source="closed-form",
        min_asymptotic_order=2.5,
        min_pairwise_order=2.5,
        max_final_error=0.01,
    )
    assert failed.passed is False
    nonmonotone = observed_order_gate_report(
        estimate_observed_order(
            np.array([0.4, 0.2, 0.1]), np.array([0.01, 0.02, 0.002])
        ),
        case="nonmonotone",
        source="synthetic",
        min_asymptotic_order=1.0,
        min_pairwise_order=0.0,
    )
    assert nonmonotone.passed is False

    with pytest.raises(ValueError):
        observed_order_gate_report(
            metrics, case="bad", source="closed-form", min_asymptotic_order=-1.0
        )
    with pytest.raises(ValueError):
        observed_order_gate_report(
            metrics,
            case="bad",
            source="closed-form",
            min_asymptotic_order=1.0,
            max_final_error=-1.0,
        )
    with pytest.raises(ValueError):
        observed_order_gate_report(
            metrics,
            case="bad",
            source="closed-form",
            min_asymptotic_order=1.0,
            min_pairwise_order=-1.0,
        )


def test_branch_continuity_metrics_and_gate_report() -> None:
    metrics = branch_continuity_metrics(
        ky=np.array([0.1, 0.2, 0.3]),
        gamma=np.array([0.10, 0.105, 0.110]),
        omega=np.array([-0.30, -0.31, -0.32]),
        successive_overlap=np.array([0.98, 0.97]),
    )

    assert isinstance(metrics, BranchContinuationMetrics)
    assert metrics.max_rel_gamma_jump < 0.06
    assert metrics.max_rel_omega_jump < 0.04
    assert metrics.min_successive_overlap == pytest.approx(0.97)

    report = branch_continuity_gate_report(
        metrics,
        case="kbm_branch",
        source="candidate table",
        max_rel_gamma_jump=0.1,
        max_rel_omega_jump=0.1,
        min_successive_overlap=0.95,
    )
    assert report.passed is True

    jump = branch_continuity_metrics(
        ky=np.array([0.1, 0.2, 0.3]),
        gamma=np.array([0.10, 0.40, 0.11]),
        omega=np.array([-0.30, 0.80, -0.32]),
        successive_overlap=np.array([0.7, 0.6]),
    )
    failed = branch_continuity_gate_report(
        jump,
        case="kbm_branch",
        source="candidate table",
        max_rel_gamma_jump=0.1,
        max_rel_omega_jump=0.1,
        min_successive_overlap=0.95,
    )
    assert failed.passed is False

    with pytest.raises(ValueError):
        branch_continuity_metrics(np.array([0.1]), np.array([0.1]), np.array([0.2]))
    with pytest.raises(ValueError):
        branch_continuity_metrics(
            np.array([[0.1, 0.2]]), np.array([0.1, 0.2]), np.array([0.2, 0.3])
        )
    with pytest.raises(ValueError):
        branch_continuity_metrics(
            np.array([0.1, 0.2]), np.array([0.1]), np.array([0.2, 0.3])
        )
    with pytest.raises(ValueError):
        branch_continuity_metrics(
            np.array([0.1, 0.2]), np.array([0.1, np.nan]), np.array([0.2, 0.3])
        )
    with pytest.raises(ValueError):
        branch_continuity_metrics(
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
            np.array([0.2, 0.3]),
            floor_fraction=-1.0e-3,
        )
    with pytest.raises(ValueError):
        branch_continuity_metrics(
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
            np.array([0.2, 0.3]),
            successive_overlap=np.array([0.9, 0.8]),
        )
    with pytest.raises(ValueError):
        branch_continuity_metrics(
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
            np.array([0.2, 0.3]),
            successive_overlap=np.array([np.nan]),
        )
    with pytest.raises(ValueError):
        branch_continuity_gate_report(
            metrics,
            case="bad",
            source="candidate table",
            max_rel_gamma_jump=-1.0,
            max_rel_omega_jump=0.1,
        )
