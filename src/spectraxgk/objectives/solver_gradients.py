"""Production-adjacent solver-objective geometry-gradient gates.

These helpers validate gradients of actual SPECTRAX-GK linear-RHS observables
with respect to solver-ready geometry arrays.  They are deliberately stricter
than reduced optimization proxies, but still narrower than a full
``vmec_jax -> booz_xform_jax -> solver`` optimization claim.
"""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.vmec_boozer_core import flux_tube_geometry_from_vmec_boozer_state
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
from spectraxgk.objectives.gradient_gates import (
    linear_solver_geometry_gradient_report as _linear_solver_geometry_gradient_report_impl,
    solver_objective_branch_gradient_report as _solver_objective_branch_gradient_report_impl,
)
from spectraxgk.objectives.vmec_boozer_gradients import (
    VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES,
    VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES,
    VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES,
    _mode21_vmec_boozer_linear_context as _mode21_vmec_boozer_linear_context_impl,
    _mode21_vmec_boozer_quasilinear_features as _mode21_vmec_boozer_quasilinear_features_impl,
    mode21_vmec_boozer_linear_frequency_gradient_report as _mode21_vmec_boozer_linear_frequency_gradient_report_impl,
    mode21_vmec_boozer_nonlinear_window_gradient_report as _mode21_vmec_boozer_nonlinear_window_gradient_report_impl,
    mode21_vmec_boozer_quasilinear_gradient_report as _mode21_vmec_boozer_quasilinear_gradient_report_impl,
)
from spectraxgk.objectives.vmec_boozer_fd import (
    _load_vmec_jax_example_state_bundle as _load_vmec_jax_example_state_bundle_impl,
    _report_float as _report_float_impl,
    vmec_boozer_aggregate_scalar_objective_finite_difference_report as _vmec_boozer_aggregate_scalar_objective_finite_difference_report_impl,
    vmec_boozer_scalar_objective_finite_difference_report as _vmec_boozer_scalar_objective_finite_difference_report_impl,
)
from spectraxgk.objectives.vmec_boozer_line_search import (
    vmec_boozer_aggregate_line_search_holdout_report as _vmec_boozer_aggregate_line_search_holdout_report_impl,
    vmec_boozer_aggregate_scalar_objective_line_search_report as _vmec_boozer_aggregate_scalar_objective_line_search_report_impl,
    vmec_boozer_scalar_objective_line_search_report as _vmec_boozer_scalar_objective_line_search_report_impl,
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
from spectraxgk.objectives.vmec_boozer import (
    _split_vmec_boozer_objective_kwargs as _split_vmec_boozer_objective_kwargs_impl,
)
from spectraxgk.objectives.vmec_boozer import (
    vmec_boozer_aggregate_scalar_objective_from_state as _vmec_boozer_aggregate_scalar_objective_from_state_impl,
)
from spectraxgk.objectives.vmec_boozer import (
    vmec_boozer_scalar_objective_from_state as _vmec_boozer_scalar_objective_from_state_impl,
)
from spectraxgk.objectives.vmec_boozer import (
    vmec_boozer_solver_objective_table_from_state as _vmec_boozer_solver_objective_table_from_state_impl,
)
from spectraxgk.objectives.vmec_boozer import (
    vmec_boozer_solver_objective_table_with_metadata_from_state as _vmec_boozer_solver_objective_table_with_metadata_from_state_impl,
)
from spectraxgk.objectives.vmec_boozer import (
    vmec_boozer_solver_objective_vector_from_state as _vmec_boozer_solver_objective_vector_from_state_impl,
)
from spectraxgk.objectives.vmec_state import (
    VMEC_BOOZER_STATE_PARAMETER_FAMILIES,
    VMEC_BOOZER_STATE_PARAMETER_NAMES,
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
        _load_state_bundle_fn=_load_vmec_jax_example_state_bundle,
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
        _load_state_bundle_fn=_load_vmec_jax_example_state_bundle,
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


def solver_objective_branch_gradient_report(
    params: jnp.ndarray | np.ndarray | None = None,
    *,
    fd_step: float = 1.0e-3,
    rtol: float = 1.0e-1,
    atol: float = 2.0e-3,
    gap_floor: float = 1.0e-6,
    n_laguerre: int = 2,
    n_hermite: int = 1,
) -> dict[str, object]:
    return _solver_objective_branch_gradient_report_impl(
        params=params,
        fd_step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        _quasilinear_features_fn=_mode21_vmec_boozer_quasilinear_features,
        _objective_vector_fn=solver_objective_vector_from_geometry,
    )


def linear_solver_geometry_gradient_report(
    params: jnp.ndarray | np.ndarray | None = None,
    *,
    fd_step: float = 1.0e-3,
    rtol: float = 1.0e-1,
    atol: float = 2.0e-3,
    gap_floor: float = 1.0e-6,
) -> dict[str, object]:
    return _linear_solver_geometry_gradient_report_impl(
        params=params,
        fd_step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )


def _mode21_vmec_boozer_linear_context(  # pragma: no cover
    *,
    case_name: str,
    radial_index: int | None,
    mode_index: int,
    parameter_family: str,
    surface_index: int | None,
    ntheta: int,
    mboz: int,
    nboz: int,
    surface_stencil_width: int | None,
    n_laguerre: int,
    n_hermite: int,
) -> dict[str, Any]:
    return _mode21_vmec_boozer_linear_context_impl(
        case_name=case_name,
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        surface_index=surface_index,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        surface_stencil_width=surface_stencil_width,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
    )


def _mode21_vmec_boozer_quasilinear_features(
    eigenvalue: jnp.ndarray,
    eigenvector: jnp.ndarray,
    x: jnp.ndarray,
    context: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return _mode21_vmec_boozer_quasilinear_features_impl(
        eigenvalue,
        eigenvector,
        x,
        context,
    )


def mode21_vmec_boozer_linear_frequency_gradient_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    surface_index: int | None = None,
    fd_step: float = 1.0e-6,
    rtol: float = 5.0e-2,
    atol: float = 2.0e-2,
    gap_floor: float = 1.0e-8,
    ntheta: int = 4,
    mboz: int = 21,
    nboz: int = 21,
    surface_stencil_width: int | None = None,
) -> dict[str, object]:
    return _mode21_vmec_boozer_linear_frequency_gradient_report_impl(
        case_name=case_name,
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        surface_index=surface_index,
        fd_step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        surface_stencil_width=surface_stencil_width,
        _linear_context_fn=_mode21_vmec_boozer_linear_context,
    )


def mode21_vmec_boozer_quasilinear_gradient_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    surface_index: int | None = None,
    fd_step: float = 1.0e-6,
    rtol: float = 2.0e-2,
    atol: float = 5.0e-2,
    gap_floor: float = 1.0e-8,
    ntheta: int = 4,
    mboz: int = 21,
    nboz: int = 21,
    surface_stencil_width: int | None = None,
) -> dict[str, object]:
    return _mode21_vmec_boozer_quasilinear_gradient_report_impl(
        case_name=case_name,
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        surface_index=surface_index,
        fd_step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        surface_stencil_width=surface_stencil_width,
        _linear_context_fn=_mode21_vmec_boozer_linear_context,
        _quasilinear_features_fn=_mode21_vmec_boozer_quasilinear_features,
    )


def mode21_vmec_boozer_nonlinear_window_gradient_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    surface_index: int | None = None,
    fd_step: float = 1.0e-6,
    rtol: float = 7.5e-2,
    atol: float = 5.0e-2,
    gap_floor: float = 1.0e-8,
    ntheta: int = 4,
    mboz: int = 21,
    nboz: int = 21,
    surface_stencil_width: int | None = None,
    nonlinear_dt: float = 0.18,
    nonlinear_steps: int = 96,
    tail_fraction: float = 0.30,
) -> dict[str, object]:
    return _mode21_vmec_boozer_nonlinear_window_gradient_report_impl(
        case_name=case_name,
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        surface_index=surface_index,
        fd_step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        surface_stencil_width=surface_stencil_width,
        nonlinear_dt=nonlinear_dt,
        nonlinear_steps=nonlinear_steps,
        tail_fraction=tail_fraction,
        _linear_context_fn=_mode21_vmec_boozer_linear_context,
        _quasilinear_features_fn=_mode21_vmec_boozer_quasilinear_features,
        _window_metrics_fn=_reduced_nonlinear_window_metrics_from_linear_observables,
    )


__all__ = [
    "_aggregate_sample_metadata",
    "_aggregate_weights",
    "_float_tuple",
    "_int_tuple",
    "_ky_sample_axis",
    "_objective_gate_rows",
    "_surface_index_tuple",
    "_surface_sample_axis",
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
