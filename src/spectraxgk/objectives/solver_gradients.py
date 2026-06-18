"""Public solver-objective differentiability facade.

Implementation is split across focused objective modules. This file preserves
historical imports for scripts, docs, tools, and top-level ``spectraxgk`` API
exports while keeping validation and VMEC/Boozer wrappers testable at their
owner modules.
"""

from __future__ import annotations

from spectraxgk.objectives.core import (
    SOLVER_OBJECTIVE_NAMES,
    SolverScalarObjective,
    solver_growth_rate_from_geometry,
    solver_linear_operator_matrix_from_geometry,
    solver_objective_vector_from_geometry,
    solver_scalar_objective_from_vector,
)
from spectraxgk.objectives.eigen import (
    dominant_eigenvalue_branch_locality_report,
    dominant_real_eigenvalue,
)
from spectraxgk.objectives.geometry import (
    SOLVER_GEOMETRY_PARAMETER_NAMES,
    TINY_OBJECTIVE_NAMES,
    _objective_gate_rows,
    default_solver_geometry_design_params,
    solver_ready_geometry_mapping,
    tiny_differentiable_objective_gradient_report,
)
from spectraxgk.objectives.nonlinear_window import (
    _reduced_nonlinear_window_metrics_from_linear_observables,
)
from spectraxgk.objectives.sampling import (
    _aggregate_sample_metadata,
    _aggregate_weights,
    _float_tuple,
    _int_tuple,
    _ky_sample_axis,
    _surface_index_tuple,
    _surface_sample_axis,
    solver_grid_options_from_ky_values,
)
from spectraxgk.objectives.solver_gradient_reports import (
    _mode21_vmec_boozer_linear_context,
    _mode21_vmec_boozer_quasilinear_features,
    linear_solver_geometry_gradient_report,
    mode21_vmec_boozer_linear_frequency_gradient_report,
    mode21_vmec_boozer_nonlinear_window_gradient_report,
    mode21_vmec_boozer_quasilinear_gradient_report,
    solver_objective_branch_gradient_report,
)
from spectraxgk.objectives.solver_vmec import (
    _load_vmec_jax_example_state_bundle,
    _report_float,
    _split_vmec_boozer_objective_kwargs,
    vmec_boozer_aggregate_line_search_holdout_report,
    vmec_boozer_aggregate_scalar_objective_finite_difference_report,
    vmec_boozer_aggregate_scalar_objective_from_state,
    vmec_boozer_aggregate_scalar_objective_line_search_report,
    vmec_boozer_scalar_objective_finite_difference_report,
    vmec_boozer_scalar_objective_from_state,
    vmec_boozer_scalar_objective_line_search_report,
    vmec_boozer_solver_objective_table_from_state,
    vmec_boozer_solver_objective_table_with_metadata_from_state,
    vmec_boozer_solver_objective_vector_from_state,
)
from spectraxgk.objectives.vmec_boozer_gradients import (
    VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES,
    VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES,
    VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES,
)
from spectraxgk.objectives.vmec_state import (
    VMEC_BOOZER_STATE_PARAMETER_FAMILIES,
    VMEC_BOOZER_STATE_PARAMETER_NAMES,
    _replace_vmec_boozer_state_coefficient,
    _vmec_boozer_state_array,
    _vmec_boozer_state_parameter_name,
)

__all__ = [
    "_aggregate_sample_metadata",
    "_aggregate_weights",
    "_float_tuple",
    "_int_tuple",
    "_ky_sample_axis",
    "_load_vmec_jax_example_state_bundle",
    "_mode21_vmec_boozer_linear_context",
    "_mode21_vmec_boozer_quasilinear_features",
    "_objective_gate_rows",
    "_reduced_nonlinear_window_metrics_from_linear_observables",
    "_replace_vmec_boozer_state_coefficient",
    "_report_float",
    "_split_vmec_boozer_objective_kwargs",
    "_surface_index_tuple",
    "_surface_sample_axis",
    "_vmec_boozer_state_array",
    "_vmec_boozer_state_parameter_name",
    "SOLVER_GEOMETRY_PARAMETER_NAMES",
    "SOLVER_OBJECTIVE_NAMES",
    "SolverScalarObjective",
    "TINY_OBJECTIVE_NAMES",
    "VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES",
    "VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES",
    "VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES",
    "VMEC_BOOZER_STATE_PARAMETER_FAMILIES",
    "VMEC_BOOZER_STATE_PARAMETER_NAMES",
    "default_solver_geometry_design_params",
    "dominant_eigenvalue_branch_locality_report",
    "dominant_real_eigenvalue",
    "linear_solver_geometry_gradient_report",
    "mode21_vmec_boozer_linear_frequency_gradient_report",
    "mode21_vmec_boozer_nonlinear_window_gradient_report",
    "mode21_vmec_boozer_quasilinear_gradient_report",
    "solver_growth_rate_from_geometry",
    "solver_linear_operator_matrix_from_geometry",
    "solver_objective_branch_gradient_report",
    "solver_objective_vector_from_geometry",
    "solver_scalar_objective_from_vector",
    "solver_grid_options_from_ky_values",
    "solver_ready_geometry_mapping",
    "tiny_differentiable_objective_gradient_report",
    "vmec_boozer_aggregate_line_search_holdout_report",
    "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
    "vmec_boozer_aggregate_scalar_objective_from_state",
    "vmec_boozer_aggregate_scalar_objective_line_search_report",
    "vmec_boozer_scalar_objective_finite_difference_report",
    "vmec_boozer_scalar_objective_from_state",
    "vmec_boozer_scalar_objective_line_search_report",
    "vmec_boozer_solver_objective_table_from_state",
    "vmec_boozer_solver_objective_table_with_metadata_from_state",
    "vmec_boozer_solver_objective_vector_from_state",
]
