"""Public VMEC/Boozer solver-objective wrappers.

These functions keep dependency injection seams for validation tests while the
lower-level math lives in ``vmec_boozer*`` owner modules.
"""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.vmec_boozer_core import flux_tube_geometry_from_vmec_boozer_state
from spectraxgk.objectives.core import (
    SolverScalarObjective,
    solver_objective_vector_from_geometry,
    solver_scalar_objective_from_vector,
)
from spectraxgk.objectives.vmec_boozer import (
    _split_vmec_boozer_objective_kwargs as _split_vmec_boozer_objective_kwargs_impl,
    vmec_boozer_aggregate_scalar_objective_from_state as _vmec_boozer_aggregate_scalar_objective_from_state_impl,
    vmec_boozer_scalar_objective_from_state as _vmec_boozer_scalar_objective_from_state_impl,
    vmec_boozer_solver_objective_table_from_state as _vmec_boozer_solver_objective_table_from_state_impl,
    vmec_boozer_solver_objective_table_with_metadata_from_state as _vmec_boozer_solver_objective_table_with_metadata_from_state_impl,
    vmec_boozer_solver_objective_vector_from_state as _vmec_boozer_solver_objective_vector_from_state_impl,
)
from spectraxgk.objectives.vmec_boozer_fd import (
    _load_vmex_example_state_bundle as _load_vmex_example_state_bundle_impl,
    _report_float as _report_float_impl,
    vmec_boozer_aggregate_scalar_objective_finite_difference_report as _vmec_boozer_aggregate_scalar_objective_finite_difference_report_impl,
    vmec_boozer_scalar_objective_finite_difference_report as _vmec_boozer_scalar_objective_finite_difference_report_impl,
)
from spectraxgk.objectives.vmec_boozer_line_search import (
    vmec_boozer_aggregate_line_search_holdout_report as _vmec_boozer_aggregate_line_search_holdout_report_impl,
    vmec_boozer_aggregate_scalar_objective_line_search_report as _vmec_boozer_aggregate_scalar_objective_line_search_report_impl,
    vmec_boozer_scalar_objective_line_search_report as _vmec_boozer_scalar_objective_line_search_report_impl,
)
from spectraxgk.geometry.vmec_state_controls import (
    _replace_vmec_boozer_state_coefficient,
    _vmec_boozer_state_array,
    _vmec_boozer_state_parameter_name,
)

def _report_float(report: dict[str, object], key: str) -> float:
    return _report_float_impl(report, key)


def _split_vmec_boozer_objective_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _split_vmec_boozer_objective_kwargs_impl(kwargs)


def vmec_boozer_solver_objective_vector_from_state(  # pragma: no cover
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    **kwargs: Any,
) -> jnp.ndarray:
    """Evaluate solver objectives from the in-memory VMEC/Boozer bridge."""

    return _vmec_boozer_solver_objective_vector_from_state_impl(
        state,
        static,
        indata,
        wout,
        geometry_fn=flux_tube_geometry_from_vmec_boozer_state,
        objective_vector_fn=solver_objective_vector_from_geometry,
        **kwargs,
    )


def vmec_boozer_solver_objective_table_from_state(  # pragma: no cover
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    *,
    surface_indices: int | None | tuple[int | None, ...] | list[int | None] = (None,),
    torflux_values: float | tuple[float, ...] | list[float] | None = None,
    alphas: float | tuple[float, ...] | list[float] = (0.0,),
    selected_ky_indices: int | tuple[int, ...] | list[int] = (1,),
    ky_values: float | tuple[float, ...] | list[float] | None = None,
    ky_base: float | None = None,
    **kwargs: Any,
) -> jnp.ndarray:
    """Evaluate solver objectives over a surface/field-line/``k_y`` table."""

    return _vmec_boozer_solver_objective_table_from_state_impl(
        state,
        static,
        indata,
        wout,
        surface_indices=surface_indices,
        torflux_values=torflux_values,
        alphas=alphas,
        selected_ky_indices=selected_ky_indices,
        ky_values=ky_values,
        ky_base=ky_base,
        table_with_metadata_fn=vmec_boozer_solver_objective_table_with_metadata_from_state,
        **kwargs,
    )


def vmec_boozer_solver_objective_table_with_metadata_from_state(  # pragma: no cover
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    *,
    surface_indices: int | None | tuple[int | None, ...] | list[int | None] = (None,),
    torflux_values: float | tuple[float, ...] | list[float] | None = None,
    alphas: float | tuple[float, ...] | list[float] = (0.0,),
    selected_ky_indices: int | tuple[int, ...] | list[int] = (1,),
    ky_values: float | tuple[float, ...] | list[float] | None = None,
    ky_base: float | None = None,
    **kwargs: Any,
) -> tuple[jnp.ndarray, list[dict[str, object]]]:
    """Evaluate VMEC/Boozer objective rows and return sample metadata."""

    return _vmec_boozer_solver_objective_table_with_metadata_from_state_impl(
        state,
        static,
        indata,
        wout,
        surface_indices=surface_indices,
        torflux_values=torflux_values,
        alphas=alphas,
        selected_ky_indices=selected_ky_indices,
        ky_values=ky_values,
        ky_base=ky_base,
        geometry_fn=flux_tube_geometry_from_vmec_boozer_state,
        objective_vector_fn=solver_objective_vector_from_geometry,
        **kwargs,
    )


def vmec_boozer_aggregate_scalar_objective_from_state(  # pragma: no cover
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    *,
    objective: SolverScalarObjective = "growth",
    reduction: Literal["mean", "weighted_mean", "max"] = "mean",
    weights: tuple[float, ...] | list[float] | np.ndarray | None = None,
    surface_indices: int | None | tuple[int | None, ...] | list[int | None] = (None,),
    alphas: float | tuple[float, ...] | list[float] = (0.0,),
    selected_ky_indices: int | tuple[int, ...] | list[int] = (1,),
    **kwargs: Any,
) -> jnp.ndarray:
    """Reduce a VMEC/Boozer multi-point objective table to one scalar."""

    return _vmec_boozer_aggregate_scalar_objective_from_state_impl(
        state,
        static,
        indata,
        wout,
        objective=objective,
        reduction=reduction,
        weights=weights,
        surface_indices=surface_indices,
        alphas=alphas,
        selected_ky_indices=selected_ky_indices,
        table_fn=vmec_boozer_solver_objective_table_from_state,
        **kwargs,
    )


def vmec_boozer_scalar_objective_from_state(  # pragma: no cover
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    *,
    objective: SolverScalarObjective = "growth",
    **kwargs: Any,
) -> jnp.ndarray:
    """Evaluate one scalar optimization objective on the VMEC/Boozer path."""

    return _vmec_boozer_scalar_objective_from_state_impl(
        state,
        static,
        indata,
        wout,
        objective=objective,
        vector_fn=vmec_boozer_solver_objective_vector_from_state,
        **kwargs,
    )


def _load_vmex_example_state_bundle(
    case_name: str,
) -> dict[str, Any]:  # pragma: no cover
    return _load_vmex_example_state_bundle_impl(case_name)


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
    """Finite-difference a scalar objective through a VMEC state coefficient."""

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
        _load_state_bundle_fn=_load_vmex_example_state_bundle,
        _state_array_fn=_vmec_boozer_state_array,
        _replace_state_coefficient_fn=_replace_vmec_boozer_state_coefficient,
        _parameter_name_fn=_vmec_boozer_state_parameter_name,
        _vector_fn=vmec_boozer_solver_objective_vector_from_state,
        _scalar_selector_fn=solver_scalar_objective_from_vector,
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
    """Finite-difference a multi-surface/multi-``k_y`` aggregate objective."""

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
        _load_state_bundle_fn=_load_vmex_example_state_bundle,
        _state_array_fn=_vmec_boozer_state_array,
        _replace_state_coefficient_fn=_replace_vmec_boozer_state_coefficient,
        _parameter_name_fn=_vmec_boozer_state_parameter_name,
        _table_with_metadata_fn=vmec_boozer_solver_objective_table_with_metadata_from_state,
        _scalar_selector_fn=solver_scalar_objective_from_vector,
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
    """Run a curvature-gated line search for an aggregate VMEC objective."""

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
        _finite_difference_report_fn=vmec_boozer_aggregate_scalar_objective_finite_difference_report,
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
    """Audit a training aggregate update against held-out aggregate samples."""

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
        _line_search_report_fn=vmec_boozer_aggregate_scalar_objective_line_search_report,
        _finite_difference_report_fn=vmec_boozer_aggregate_scalar_objective_finite_difference_report,
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
    """Run a curvature-gated one-parameter VMEC/Boozer objective line search."""

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
        _finite_difference_report_fn=vmec_boozer_scalar_objective_finite_difference_report,
        **kwargs,
    )

__all__ = [
    "_load_vmex_example_state_bundle",
    "_report_float",
    "_split_vmec_boozer_objective_kwargs",
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
