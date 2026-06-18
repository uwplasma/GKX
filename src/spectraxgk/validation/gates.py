"""Public validation gate primitives for benchmark and manuscript artifacts."""

from __future__ import annotations

from spectraxgk.validation.gate_reports import (
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
from spectraxgk.validation.gate_types import (
    BranchContinuationMetrics,
    DiagnosticTimeSeries,
    EigenfunctionComparisonMetrics,
    EigenfunctionReferenceBundle,
    GateReport,
    LateTimeLinearMetrics,
    NonlinearHeatFluxConvergenceMetrics,
    NonlinearWindowMetrics,
    ObservedOrderMetrics,
    ScalarGateResult,
    ZonalFlowResponseMetrics,
)

__all__ = [
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
    "ZonalFlowResponseMetrics",
    "branch_continuity_gate_report",
    "eigenfunction_gate_report",
    "evaluate_scalar_gate",
    "gate_report",
    "gate_report_to_dict",
    "linear_metrics_gate_report",
    "nonlinear_heat_flux_convergence_gate_report",
    "nonlinear_window_gate_report",
    "observed_order_gate_report",
    "zonal_response_gate_report",
]
