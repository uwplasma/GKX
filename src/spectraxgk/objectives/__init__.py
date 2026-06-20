"""Differentiable solver and stellarator objective helpers."""

from __future__ import annotations

from spectraxgk.objectives.qa_low_turbulence import (
    QA_LOW_TURBULENCE_DESIGN_NAMES,
    QA_LOW_TURBULENCE_OBSERVABLE_NAMES,
    QALowTurbulenceConfig,
    QALowTurbulenceResult,
    default_qa_low_turbulence_initial_params,
    optimize_qa_low_turbulence,
    qa_low_turbulence_comparison_payload,
    qa_low_turbulence_heat_flux_trace,
    qa_low_turbulence_observable_sensitivity_report,
    qa_low_turbulence_observable_vector,
    qa_low_turbulence_observables,
    qa_low_turbulence_objective,
    qa_low_turbulence_residual_names,
    qa_low_turbulence_residual_vector,
    qa_low_turbulence_window_metrics,
)

__all__ = [
    "QA_LOW_TURBULENCE_DESIGN_NAMES",
    "QA_LOW_TURBULENCE_OBSERVABLE_NAMES",
    "QALowTurbulenceConfig",
    "QALowTurbulenceResult",
    "default_qa_low_turbulence_initial_params",
    "optimize_qa_low_turbulence",
    "qa_low_turbulence_comparison_payload",
    "qa_low_turbulence_heat_flux_trace",
    "qa_low_turbulence_observable_sensitivity_report",
    "qa_low_turbulence_observable_vector",
    "qa_low_turbulence_observables",
    "qa_low_turbulence_objective",
    "qa_low_turbulence_residual_names",
    "qa_low_turbulence_residual_vector",
    "qa_low_turbulence_window_metrics",
]
