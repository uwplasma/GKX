"""Compatibility facade for VMEC/Boozer objective gates."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from spectraxgk.solver_objective_core import (
    SolverScalarObjective,
    solver_scalar_objective_from_vector,
)
from spectraxgk.solver_vmec_boozer_fd_gates import (
    _load_vmec_jax_example_state_bundle as _load_vmec_jax_example_state_bundle_impl,
    _report_float as _report_float_impl,
    vmec_boozer_aggregate_scalar_objective_finite_difference_report as _vmec_boozer_aggregate_scalar_objective_finite_difference_report_impl,
    vmec_boozer_scalar_objective_finite_difference_report as _vmec_boozer_scalar_objective_finite_difference_report_impl,
)
from spectraxgk.solver_vmec_boozer_line_search_gates import (
    vmec_boozer_aggregate_line_search_holdout_report as _vmec_boozer_aggregate_line_search_holdout_report_impl,
    vmec_boozer_aggregate_scalar_objective_line_search_report as _vmec_boozer_aggregate_scalar_objective_line_search_report_impl,
    vmec_boozer_scalar_objective_line_search_report as _vmec_boozer_scalar_objective_line_search_report_impl,
)
from spectraxgk.solver_vmec_boozer_objectives import (
    vmec_boozer_solver_objective_table_with_metadata_from_state,
    vmec_boozer_solver_objective_vector_from_state,
)
from spectraxgk.solver_vmec_state import (
    _replace_vmec_boozer_state_coefficient,
    _vmec_boozer_state_array,
    _vmec_boozer_state_parameter_name,
)


def _report_float(report: dict[str, object], key: str) -> float:
    return _report_float_impl(report, key)


def _load_vmec_jax_example_state_bundle(
    case_name: str,
) -> dict[str, Any]:  # pragma: no cover
    return _load_vmec_jax_example_state_bundle_impl(case_name)


def vmec_boozer_scalar_objective_finite_difference_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    objective: SolverScalarObjective = "growth",
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    base_delta: float = 0.0,
    perturbation_step: float = 1.0e-7,
    response_atol: float = 0.0,
    max_curvature_ratio: float = 5.0,
    **kwargs: Any,
) -> dict[str, object]:
    load_state_bundle_fn = kwargs.pop(
        "_load_state_bundle_fn", _load_vmec_jax_example_state_bundle
    )
    state_array_fn = kwargs.pop("_state_array_fn", _vmec_boozer_state_array)
    replace_state_coefficient_fn = kwargs.pop(
        "_replace_state_coefficient_fn", _replace_vmec_boozer_state_coefficient
    )
    parameter_name_fn = kwargs.pop(
        "_parameter_name_fn", _vmec_boozer_state_parameter_name
    )
    vector_fn = kwargs.pop("_vector_fn", vmec_boozer_solver_objective_vector_from_state)
    scalar_selector_fn = kwargs.pop(
        "_scalar_selector_fn", solver_scalar_objective_from_vector
    )
    return _vmec_boozer_scalar_objective_finite_difference_report_impl(
        case_name=case_name,
        objective=objective,
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        base_delta=base_delta,
        perturbation_step=perturbation_step,
        response_atol=response_atol,
        max_curvature_ratio=max_curvature_ratio,
        _load_state_bundle_fn=load_state_bundle_fn,
        _state_array_fn=state_array_fn,
        _replace_state_coefficient_fn=replace_state_coefficient_fn,
        _parameter_name_fn=parameter_name_fn,
        _vector_fn=vector_fn,
        _scalar_selector_fn=scalar_selector_fn,
        **kwargs,
    )


def vmec_boozer_aggregate_scalar_objective_finite_difference_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    objective: SolverScalarObjective = "growth",
    reduction: Literal["mean", "weighted_mean", "max"] = "mean",
    weights: tuple[float, ...] | list[float] | np.ndarray | None = None,
    surface_indices: int | None | tuple[int | None, ...] | list[int | None] = (None,),
    torflux_values: float | tuple[float, ...] | list[float] | None = None,
    alphas: float | tuple[float, ...] | list[float] = (0.0,),
    selected_ky_indices: int | tuple[int, ...] | list[int] = (1,),
    ky_values: float | tuple[float, ...] | list[float] | None = None,
    ky_base: float | None = None,
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    base_delta: float = 0.0,
    perturbation_step: float = 1.0e-7,
    response_atol: float = 0.0,
    max_curvature_ratio: float = 5.0,
    **kwargs: Any,
) -> dict[str, object]:
    load_state_bundle_fn = kwargs.pop(
        "_load_state_bundle_fn", _load_vmec_jax_example_state_bundle
    )
    state_array_fn = kwargs.pop("_state_array_fn", _vmec_boozer_state_array)
    replace_state_coefficient_fn = kwargs.pop(
        "_replace_state_coefficient_fn", _replace_vmec_boozer_state_coefficient
    )
    parameter_name_fn = kwargs.pop(
        "_parameter_name_fn", _vmec_boozer_state_parameter_name
    )
    table_with_metadata_fn = kwargs.pop(
        "_table_with_metadata_fn",
        vmec_boozer_solver_objective_table_with_metadata_from_state,
    )
    scalar_selector_fn = kwargs.pop(
        "_scalar_selector_fn", solver_scalar_objective_from_vector
    )
    return _vmec_boozer_aggregate_scalar_objective_finite_difference_report_impl(
        case_name=case_name,
        objective=objective,
        reduction=reduction,
        weights=weights,
        surface_indices=surface_indices,
        torflux_values=torflux_values,
        alphas=alphas,
        selected_ky_indices=selected_ky_indices,
        ky_values=ky_values,
        ky_base=ky_base,
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        base_delta=base_delta,
        perturbation_step=perturbation_step,
        response_atol=response_atol,
        max_curvature_ratio=max_curvature_ratio,
        _load_state_bundle_fn=load_state_bundle_fn,
        _state_array_fn=state_array_fn,
        _replace_state_coefficient_fn=replace_state_coefficient_fn,
        _parameter_name_fn=parameter_name_fn,
        _table_with_metadata_fn=table_with_metadata_fn,
        _scalar_selector_fn=scalar_selector_fn,
        **kwargs,
    )


def vmec_boozer_aggregate_scalar_objective_line_search_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    objective: SolverScalarObjective = "growth",
    reduction: Literal["mean", "weighted_mean", "max"] = "mean",
    weights: tuple[float, ...] | list[float] | np.ndarray | None = None,
    surface_indices: int | None | tuple[int | None, ...] | list[int | None] = (None,),
    alphas: float | tuple[float, ...] | list[float] = (0.0,),
    selected_ky_indices: int | tuple[int, ...] | list[int] = (1,),
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    initial_delta: float = 0.0,
    perturbation_step: float = 1.0e-7,
    update_step: float = 1.0e-8,
    max_steps: int = 3,
    min_improvement: float = 0.0,
    response_atol: float = 0.0,
    max_curvature_ratio: float = 5.0,
    **kwargs: Any,
) -> dict[str, object]:
    finite_difference_report_fn = kwargs.pop(
        "_finite_difference_report_fn",
        vmec_boozer_aggregate_scalar_objective_finite_difference_report,
    )
    return _vmec_boozer_aggregate_scalar_objective_line_search_report_impl(
        case_name=case_name,
        objective=objective,
        reduction=reduction,
        weights=weights,
        surface_indices=surface_indices,
        alphas=alphas,
        selected_ky_indices=selected_ky_indices,
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        initial_delta=initial_delta,
        perturbation_step=perturbation_step,
        update_step=update_step,
        max_steps=max_steps,
        min_improvement=min_improvement,
        response_atol=response_atol,
        max_curvature_ratio=max_curvature_ratio,
        _finite_difference_report_fn=finite_difference_report_fn,
        **kwargs,
    )


def vmec_boozer_aggregate_line_search_holdout_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    objective: SolverScalarObjective = "growth",
    reduction: Literal["mean", "weighted_mean", "max"] = "mean",
    training_weights: tuple[float, ...] | list[float] | np.ndarray | None = None,
    holdout_weights: tuple[float, ...] | list[float] | np.ndarray | None = None,
    training_surface_indices: int | None | tuple[int | None, ...] | list[int | None] = (
        None,
    ),
    training_alphas: float | tuple[float, ...] | list[float] = (0.0,),
    training_selected_ky_indices: int | tuple[int, ...] | list[int] = (1,),
    holdout_surface_indices: int | None | tuple[int | None, ...] | list[int | None] = (
        None,
    ),
    holdout_alphas: float | tuple[float, ...] | list[float] = (0.0,),
    holdout_selected_ky_indices: int | tuple[int, ...] | list[int] = (2,),
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    initial_delta: float = 0.0,
    perturbation_step: float = 1.0e-7,
    update_step: float = 1.0e-8,
    max_steps: int = 3,
    min_improvement: float = 0.0,
    min_holdout_improvement: float = 0.0,
    response_atol: float = 0.0,
    max_curvature_ratio: float = 5.0,
    **kwargs: Any,
) -> dict[str, object]:
    line_search_report_fn = kwargs.pop(
        "_line_search_report_fn",
        vmec_boozer_aggregate_scalar_objective_line_search_report,
    )
    finite_difference_report_fn = kwargs.pop(
        "_finite_difference_report_fn",
        vmec_boozer_aggregate_scalar_objective_finite_difference_report,
    )
    return _vmec_boozer_aggregate_line_search_holdout_report_impl(
        case_name=case_name,
        objective=objective,
        reduction=reduction,
        training_weights=training_weights,
        holdout_weights=holdout_weights,
        training_surface_indices=training_surface_indices,
        training_alphas=training_alphas,
        training_selected_ky_indices=training_selected_ky_indices,
        holdout_surface_indices=holdout_surface_indices,
        holdout_alphas=holdout_alphas,
        holdout_selected_ky_indices=holdout_selected_ky_indices,
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        initial_delta=initial_delta,
        perturbation_step=perturbation_step,
        update_step=update_step,
        max_steps=max_steps,
        min_improvement=min_improvement,
        min_holdout_improvement=min_holdout_improvement,
        response_atol=response_atol,
        max_curvature_ratio=max_curvature_ratio,
        _line_search_report_fn=line_search_report_fn,
        _finite_difference_report_fn=finite_difference_report_fn,
        **kwargs,
    )


def vmec_boozer_scalar_objective_line_search_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    objective: SolverScalarObjective = "growth",
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    initial_delta: float = 0.0,
    perturbation_step: float = 1.0e-7,
    update_step: float = 1.0e-8,
    max_steps: int = 3,
    min_improvement: float = 0.0,
    response_atol: float = 0.0,
    max_curvature_ratio: float = 5.0,
    **kwargs: Any,
) -> dict[str, object]:
    finite_difference_report_fn = kwargs.pop(
        "_finite_difference_report_fn",
        vmec_boozer_scalar_objective_finite_difference_report,
    )
    return _vmec_boozer_scalar_objective_line_search_report_impl(
        case_name=case_name,
        objective=objective,
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        initial_delta=initial_delta,
        perturbation_step=perturbation_step,
        update_step=update_step,
        max_steps=max_steps,
        min_improvement=min_improvement,
        response_atol=response_atol,
        max_curvature_ratio=max_curvature_ratio,
        _finite_difference_report_fn=finite_difference_report_fn,
        **kwargs,
    )


__all__ = [
    "_load_vmec_jax_example_state_bundle",
    "_report_float",
    "vmec_boozer_aggregate_line_search_holdout_report",
    "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
    "vmec_boozer_aggregate_scalar_objective_line_search_report",
    "vmec_boozer_scalar_objective_finite_difference_report",
    "vmec_boozer_scalar_objective_line_search_report",
]
