"""Solver-gradient report wrappers for promoted differentiability gates."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.objectives.core import (
    solver_objective_vector_from_geometry,
)
from spectraxgk.objectives.gradient_gates import (
    linear_solver_geometry_gradient_report as _linear_solver_geometry_gradient_report_impl,
    solver_objective_branch_gradient_report as _solver_objective_branch_gradient_report_impl,
)
from spectraxgk.objectives.vmec_boozer_gradients import (
    _reduced_nonlinear_window_metrics_from_linear_observables,
)
from spectraxgk.objectives.vmec_boozer_gradients import (
    _mode21_vmec_boozer_linear_context as _mode21_vmec_boozer_linear_context_impl,
    _mode21_vmec_boozer_quasilinear_features as _mode21_vmec_boozer_quasilinear_features_impl,
    mode21_vmec_boozer_linear_frequency_gradient_report as _mode21_vmec_boozer_linear_frequency_gradient_report_impl,
    mode21_vmec_boozer_nonlinear_window_gradient_report as _mode21_vmec_boozer_nonlinear_window_gradient_report_impl,
    mode21_vmec_boozer_quasilinear_gradient_report as _mode21_vmec_boozer_quasilinear_gradient_report_impl,
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
    "_mode21_vmec_boozer_linear_context",
    "_mode21_vmec_boozer_quasilinear_features",
    "linear_solver_geometry_gradient_report",
    "mode21_vmec_boozer_linear_frequency_gradient_report",
    "mode21_vmec_boozer_nonlinear_window_gradient_report",
    "mode21_vmec_boozer_quasilinear_gradient_report",
    "solver_objective_branch_gradient_report",
]
