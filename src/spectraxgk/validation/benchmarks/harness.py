"""High-level benchmark helpers for scans and eigenfunction extraction."""

from __future__ import annotations

from typing import Any

from spectraxgk.core.grid import build_spectral_grid
from spectraxgk.diagnostics.analysis import (
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
)
from spectraxgk.validation.benchmarks import harness_metrics as _harness_metrics
from spectraxgk.validation.benchmarks import harness_scan as _harness_scan
from spectraxgk.diagnostics.modes import (
    compare_eigenfunctions,
    load_eigenfunction_reference_bundle,
    normalize_eigenfunction,
    phase_align_eigenfunction,
    save_eigenfunction_reference_bundle,
)
from spectraxgk.validation.benchmarks.harness_scan import ScanAndModeResult
from spectraxgk.diagnostics.validation_gates import (
    _analytic_signal,
    _explicit_time_window,
    _leading_window,
    BranchContinuationMetrics,
    DiagnosticTimeSeries,
    EigenfunctionComparisonMetrics,
    EigenfunctionReferenceBundle,
    GateReport as GateReport,
    infer_triple_dealiased_ny,
    late_time_window,
    load_diagnostic_time_series,
    LateTimeLinearMetrics,
    NonlinearHeatFluxConvergenceMetrics,
    NonlinearWindowMetrics,
    ObservedOrderMetrics,
    ScalarGateResult as ScalarGateResult,
    ZonalFlowResponseMetrics,
    branch_continuity_gate_report as branch_continuity_gate_report,
    eigenfunction_gate_report as eigenfunction_gate_report,
    evaluate_scalar_gate as evaluate_scalar_gate,
    gate_report as gate_report,
    gate_report_to_dict as gate_report_to_dict,
    linear_metrics_gate_report as linear_metrics_gate_report,
    nonlinear_heat_flux_convergence_gate_report as nonlinear_heat_flux_convergence_gate_report,
    nonlinear_window_gate_report as nonlinear_window_gate_report,
    observed_order_gate_report as observed_order_gate_report,
    zonal_response_gate_report as zonal_response_gate_report,
)


def _sync_metric_hooks() -> None:
    _harness_metrics.extract_mode_time_series = extract_mode_time_series
    _harness_metrics.fit_growth_rate = fit_growth_rate


def _sync_scan_hooks() -> None:
    _harness_scan.build_spectral_grid = build_spectral_grid
    _harness_scan.extract_eigenfunction = extract_eigenfunction
    _harness_scan.extract_mode_time_series = extract_mode_time_series
    _harness_scan.fit_growth_rate_auto = fit_growth_rate_auto


def zonal_flow_response_metrics(*args: Any, **kwargs: Any) -> ZonalFlowResponseMetrics:
    """Estimate residual level and GAM envelope metrics from a zonal response."""

    return _harness_metrics.zonal_flow_response_metrics(*args, **kwargs)


def late_time_linear_metrics(*args: Any, **kwargs: Any) -> LateTimeLinearMetrics:
    """Return late-time growth/frequency metrics from a linear result."""

    _sync_metric_hooks()
    return _harness_metrics.late_time_linear_metrics(*args, **kwargs)


def windowed_nonlinear_metrics(*args: Any, **kwargs: Any) -> NonlinearWindowMetrics:
    """Return late-window transport metrics from nonlinear diagnostics."""

    return _harness_metrics.windowed_nonlinear_metrics(*args, **kwargs)


def nonlinear_heat_flux_convergence_metrics(
    *args: Any, **kwargs: Any
) -> NonlinearHeatFluxConvergenceMetrics:
    """Summarize post-transient heat-flux average stability."""

    return _harness_metrics.nonlinear_heat_flux_convergence_metrics(*args, **kwargs)


def estimate_observed_order(*args: Any, **kwargs: Any) -> ObservedOrderMetrics:
    """Estimate observed order from step-size refinements."""

    return _harness_metrics.estimate_observed_order(*args, **kwargs)


def branch_continuity_metrics(*args: Any, **kwargs: Any) -> BranchContinuationMetrics:
    """Compute branch-continuity diagnostics for a linear scan."""

    return _harness_metrics.branch_continuity_metrics(*args, **kwargs)


def run_linear_scan(*args: Any, **kwargs: Any):
    """Run a linear scan over ky values."""

    return _harness_scan.run_linear_scan(*args, **kwargs)


def run_scan_and_mode(*args: Any, **kwargs: Any) -> ScanAndModeResult:
    """Run a scan and extract a representative eigenfunction."""

    _sync_scan_hooks()
    return _harness_scan.run_scan_and_mode(*args, **kwargs)


__all__ = [
    "_analytic_signal",
    "_explicit_time_window",
    "_leading_window",
    "BranchContinuationMetrics",
    "DiagnosticTimeSeries",
    "EigenfunctionComparisonMetrics",
    "EigenfunctionReferenceBundle",
    "GateReport",
    "LateTimeLinearMetrics",
    "NonlinearHeatFluxConvergenceMetrics",
    "NonlinearWindowMetrics",
    "ObservedOrderMetrics",
    "ScalarGateResult",
    "ScanAndModeResult",
    "ZonalFlowResponseMetrics",
    "branch_continuity_gate_report",
    "branch_continuity_metrics",
    "compare_eigenfunctions",
    "eigenfunction_gate_report",
    "estimate_observed_order",
    "evaluate_scalar_gate",
    "gate_report",
    "gate_report_to_dict",
    "infer_triple_dealiased_ny",
    "late_time_linear_metrics",
    "late_time_window",
    "linear_metrics_gate_report",
    "load_diagnostic_time_series",
    "load_eigenfunction_reference_bundle",
    "nonlinear_heat_flux_convergence_gate_report",
    "nonlinear_heat_flux_convergence_metrics",
    "nonlinear_window_gate_report",
    "normalize_eigenfunction",
    "observed_order_gate_report",
    "phase_align_eigenfunction",
    "run_linear_scan",
    "run_scan_and_mode",
    "save_eigenfunction_reference_bundle",
    "windowed_nonlinear_metrics",
    "zonal_flow_response_metrics",
    "zonal_response_gate_report",
]
