"""Public facade for reduced QA low-turbulence comparison tools.

The implementation is split into contracts, reduced-model observables,
residual/sensitivity gates, the optimizer loop, and artifact payload builders.
This module preserves the historical import surface for examples and tests.
"""

from __future__ import annotations

from spectraxgk.objectives.qa_low_turbulence_artifacts import (
    _fixed_trace_payload as _fixed_trace_payload,
    _long_window_convergence_gate as _long_window_convergence_gate,
    _scan_density_gradient as _scan_density_gradient,
    qa_low_turbulence_comparison_payload,
    reduced_boundary_surface,
    reduced_lcfs_bmag,
)
from spectraxgk.objectives.qa_low_turbulence_contracts import (
    QA_LOW_TURBULENCE_DESIGN_NAMES,
    QA_LOW_TURBULENCE_OBSERVABLE_NAMES,
    QALowTurbulenceConfig,
    QALowTurbulenceResult,
)
from spectraxgk.objectives.qa_low_turbulence_model import (
    _fd_gate_tolerances as _fd_gate_tolerances,
    _qa_low_turbulence_core as _qa_low_turbulence_core,
    default_qa_low_turbulence_initial_params,
    qa_low_turbulence_heat_flux_trace,
    qa_low_turbulence_observable_vector,
    qa_low_turbulence_observables,
    qa_low_turbulence_window_metrics,
)
from spectraxgk.objectives.qa_low_turbulence_optimizer import optimize_qa_low_turbulence
from spectraxgk.objectives.qa_low_turbulence_residuals import (
    qa_low_turbulence_objective,
    qa_low_turbulence_observable_sensitivity_report,
    qa_low_turbulence_residual_names,
    qa_low_turbulence_residual_vector,
)


__all__ = [
    "QA_LOW_TURBULENCE_DESIGN_NAMES",
    "QA_LOW_TURBULENCE_OBSERVABLE_NAMES",
    "QALowTurbulenceConfig",
    "QALowTurbulenceResult",
    "_fd_gate_tolerances",
    "_fixed_trace_payload",
    "_long_window_convergence_gate",
    "_qa_low_turbulence_core",
    "_scan_density_gradient",
    "default_qa_low_turbulence_initial_params",
    "optimize_qa_low_turbulence",
    "qa_low_turbulence_comparison_payload",
    "qa_low_turbulence_heat_flux_trace",
    "qa_low_turbulence_objective",
    "qa_low_turbulence_observable_sensitivity_report",
    "qa_low_turbulence_observable_vector",
    "qa_low_turbulence_observables",
    "qa_low_turbulence_residual_names",
    "qa_low_turbulence_residual_vector",
    "qa_low_turbulence_window_metrics",
    "reduced_boundary_surface",
    "reduced_lcfs_bmag",
]
