"""Compatibility facade for solver-ready and VMEC/Boozer gradient gates."""

from __future__ import annotations

from spectraxgk.solver_ready_gradient_gates import (
    _linear_eigenpair_quasilinear_features,
    linear_solver_geometry_gradient_report,
    solver_objective_branch_gradient_report,
)
from spectraxgk.solver_vmec_boozer_gradient_gates import (
    VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES,
    VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES,
    VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES,
    _mode21_vmec_boozer_linear_context,
    _mode21_vmec_boozer_quasilinear_features,
    mode21_vmec_boozer_linear_frequency_gradient_report,
    mode21_vmec_boozer_nonlinear_window_gradient_report,
    mode21_vmec_boozer_quasilinear_gradient_report,
)

__all__ = [
    "VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES",
    "VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES",
    "VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES",
    "_linear_eigenpair_quasilinear_features",
    "linear_solver_geometry_gradient_report",
    "solver_objective_branch_gradient_report",
    "_mode21_vmec_boozer_linear_context",
    "_mode21_vmec_boozer_quasilinear_features",
    "mode21_vmec_boozer_linear_frequency_gradient_report",
    "mode21_vmec_boozer_nonlinear_window_gradient_report",
    "mode21_vmec_boozer_quasilinear_gradient_report",
]
