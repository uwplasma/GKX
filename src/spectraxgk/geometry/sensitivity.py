"""Geometry sensitivity, inverse-design, and local UQ reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.validation.autodiff import covariance_diagnostics
from spectraxgk.geometry.autodiff_checks import (
    _sensitivity_conditioning_metadata,
    finite_difference_jacobian,
    observable_gradient_validation_report,
)
from spectraxgk.geometry.backend_discovery import (
    _jax_float_dtype,
    discover_differentiable_geometry_backends,
)
from spectraxgk.geometry.flux_tube_contract import (
    _GEOMETRY_OBSERVABLE_NAMES,
    flux_tube_geometry_from_mapping,
    flux_tube_geometry_observables,
)


@dataclass(frozen=True)
class _GeometryInverseDesignProblem:
    """Validated inputs for a local geometry inverse-design solve."""

    params: jnp.ndarray
    target: jnp.ndarray
    indices_np: np.ndarray
    indices: jnp.ndarray


def _prepare_geometry_inverse_design_problem(
    initial_params: jnp.ndarray,
    target_observables: jnp.ndarray,
    observable_indices: Sequence[int] | None,
    *,
    max_steps: int,
    damping: float,
) -> _GeometryInverseDesignProblem:
    """Validate inverse-design inputs and construct selected observable indices."""

    params = jnp.asarray(initial_params, dtype=_jax_float_dtype())
    if params.ndim != 1:
        raise ValueError("initial_params must be one-dimensional")
    if int(max_steps) < 0:
        raise ValueError("max_steps must be non-negative")
    if float(damping) < 0.0:
        raise ValueError("damping must be non-negative")

    if observable_indices is None:
        indices_np = np.arange(len(_GEOMETRY_OBSERVABLE_NAMES), dtype=int)
    else:
        indices_np = np.asarray(list(observable_indices), dtype=int)
    if indices_np.ndim != 1 or indices_np.size == 0:
        raise ValueError(
            "observable_indices must be a non-empty one-dimensional sequence"
        )
    if np.any(indices_np < 0) or np.any(indices_np >= len(_GEOMETRY_OBSERVABLE_NAMES)):
        raise ValueError("observable_indices contains an out-of-range observable index")

    target = jnp.asarray(target_observables, dtype=params.dtype)
    if target.ndim != 1 or int(target.shape[0]) != int(indices_np.size):
        raise ValueError("target_observables length must match observable_indices")
    return _GeometryInverseDesignProblem(
        params=params,
        target=target,
        indices_np=indices_np,
        indices=jnp.asarray(indices_np, dtype=jnp.int32),
    )


def _geometry_observable_fn(
    mapping_fn: Any,
    indices: jnp.ndarray,
    *,
    source_model: str,
) -> Any:
    """Build the selected solver-geometry observable map used by AD/FD checks."""

    def observable_fn(x: jnp.ndarray) -> jnp.ndarray:
        geom = flux_tube_geometry_from_mapping(
            mapping_fn(x),
            source_model=source_model,
            validate_finite=False,
        )
        return flux_tube_geometry_observables(geom)[indices]

    return observable_fn


def _run_geometry_inverse_design_iterations(
    observable_fn: Any,
    params: jnp.ndarray,
    target: jnp.ndarray,
    *,
    max_steps: int,
    damping: float,
) -> tuple[jnp.ndarray, jnp.ndarray, list[dict[str, object]]]:
    """Run the small damped Gauss-Newton inverse-design loop."""

    history: list[dict[str, object]] = []
    p = params
    residual = observable_fn(p) - target
    for step in range(int(max_steps) + 1):
        obs = observable_fn(p)
        residual = obs - target
        objective = 0.5 * jnp.dot(residual, residual)
        history.append(
            {
                "step": int(step),
                "params": np.asarray(p).tolist(),
                "observables": np.asarray(obs).tolist(),
                "objective": float(objective),
                "residual_norm": float(jnp.linalg.norm(residual)),
            }
        )
        if step == int(max_steps):
            break
        jac = jax.jacfwd(observable_fn)(p)
        normal = jac.T @ jac + float(damping) * jnp.eye(int(p.shape[0]), dtype=p.dtype)
        delta = jnp.linalg.solve(normal, jac.T @ residual)
        p = p - delta
    return p, residual, history


def geometry_sensitivity_report(
    mapping_fn: Any,
    params: jnp.ndarray,
    *,
    fd_step: float = 1.0e-4,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
    source_model: str = "vmec_jax:in-memory",
) -> dict[str, object]:
    """Validate geometry-observable sensitivities by AD and finite differences.

    ``mapping_fn(params)`` must return the solver-ready field-line mapping
    accepted by :func:`flux_tube_geometry_from_mapping`. The report is strict
    JSON friendly so examples and CI gates can preserve the derivative
    contract without depending on large VMEC solves.
    """

    p = jnp.asarray(params, dtype=jnp.float64)
    if p.ndim != 1:
        raise ValueError("params must be one-dimensional")

    def observable_fn(x: jnp.ndarray) -> jnp.ndarray:
        geom = flux_tube_geometry_from_mapping(
            mapping_fn(x),
            source_model=source_model,
            validate_finite=False,
        )
        return flux_tube_geometry_observables(geom)

    param_names = tuple(f"param_{idx}" for idx in range(int(p.shape[0])))

    report = observable_gradient_validation_report(
        observable_fn,
        p,
        fd_step=float(fd_step),
        rtol=float(rtol),
        atol=float(atol),
        observable_names=_GEOMETRY_OBSERVABLE_NAMES,
        param_names=param_names,
        relative_floor=1.0e-12,
        report_kind="geometry_sensitivity_ad_fd_gate",
    )
    report["source_model"] = str(source_model)
    return report


def _geometry_inverse_design_selected_names(indices_np: np.ndarray) -> list[str]:
    """Return stable names for the selected geometry observables."""

    return [str(_GEOMETRY_OBSERVABLE_NAMES[int(i)]) for i in indices_np]


def _geometry_inverse_design_derivative_report(
    observable_fn: Any,
    params: jnp.ndarray,
    residual: jnp.ndarray,
    *,
    fd_step: float,
    regularization: float,
    observable_names: list[str],
    param_names: tuple[str, ...],
) -> dict[str, object]:
    """Return AD/FD Jacobian, conditioning, and covariance diagnostics."""

    jac_ad = jax.jacfwd(observable_fn)(params)
    jac_fd = finite_difference_jacobian(observable_fn, params, step=fd_step)
    diff = jac_ad - jac_fd
    scale = jnp.maximum(jnp.abs(jac_fd), 1.0e-12)
    return {
        "jacobian_ad": np.asarray(jac_ad).tolist(),
        "jacobian_fd": np.asarray(jac_fd).tolist(),
        "max_abs_ad_fd_error": float(np.max(np.abs(np.asarray(diff)))),
        "max_rel_ad_fd_error": float(np.max(np.abs(np.asarray(diff) / np.asarray(scale)))),
        "conditioning": _sensitivity_conditioning_metadata(
            jac_ad,
            jac_fd,
            params,
            fd_step=float(fd_step),
            observable_names=observable_names,
            param_names=param_names,
            relative_floor=1.0e-12,
        ),
        "uq": covariance_diagnostics(
            np.asarray(jac_ad),
            np.asarray(residual),
            regularization=regularization,
        ),
    }


def _pack_geometry_inverse_design_report(
    problem: _GeometryInverseDesignProblem,
    observable_fn: Any,
    final_params: jnp.ndarray,
    residual: jnp.ndarray,
    history: list[dict[str, object]],
    derivative_report: dict[str, object],
    *,
    fd_step: float,
    damping: float,
    regularization: float,
    source_model: str,
) -> dict[str, object]:
    """Pack the public inverse-design report schema."""

    observable_names = _geometry_inverse_design_selected_names(problem.indices_np)
    payload: dict[str, object] = {
        "observable_names": observable_names,
        "initial_params": np.asarray(problem.params).tolist(),
        "final_params": np.asarray(final_params).tolist(),
        "target_observables": np.asarray(problem.target).tolist(),
        "final_observables": np.asarray(observable_fn(final_params)).tolist(),
        "final_residual": np.asarray(residual).tolist(),
        "final_residual_norm": float(jnp.linalg.norm(residual)),
        "history": history,
        "fd_step": float(fd_step),
        "damping": float(damping),
        "regularization": float(regularization),
        "source_model": str(source_model),
        "backend_info": discover_differentiable_geometry_backends(),
    }
    payload.update(derivative_report)
    return payload


def geometry_inverse_design_report(
    mapping_fn: Any,
    initial_params: jnp.ndarray,
    target_observables: jnp.ndarray,
    *,
    observable_indices: Sequence[int] | None = None,
    max_steps: int = 8,
    damping: float = 1.0e-8,
    fd_step: float = 1.0e-4,
    regularization: float = 1.0e-8,
    source_model: str = "vmec_jax:in-memory",
) -> dict[str, object]:
    """Run a small Gauss-Newton geometry inverse-design validation.

    ``mapping_fn(params)`` must be the same solver-ready field-line mapping
    accepted by :func:`flux_tube_geometry_from_mapping`. The routine is meant
    for differentiable ``vmec_jax`` / ``booz_xform_jax`` workflows: it keeps
    the optimization, sensitivity check, and local UQ covariance in one
    JSON-friendly report so examples can validate the full AD contract without
    depending on a long equilibrium solve in CI.
    """

    problem = _prepare_geometry_inverse_design_problem(
        initial_params,
        target_observables,
        observable_indices,
        max_steps=max_steps,
        damping=damping,
    )
    observable_fn = _geometry_observable_fn(
        mapping_fn,
        problem.indices,
        source_model=source_model,
    )
    p, residual, history = _run_geometry_inverse_design_iterations(
        observable_fn,
        problem.params,
        problem.target,
        max_steps=max_steps,
        damping=damping,
    )

    observable_names = _geometry_inverse_design_selected_names(problem.indices_np)
    derivative_report = _geometry_inverse_design_derivative_report(
        observable_fn,
        p,
        residual,
        fd_step=fd_step,
        regularization=regularization,
        observable_names=observable_names,
        param_names=tuple(f"param_{idx}" for idx in range(int(problem.params.shape[0]))),
    )
    return _pack_geometry_inverse_design_report(
        problem,
        observable_fn,
        p,
        residual,
        history,
        derivative_report,
        fd_step=fd_step,
        damping=damping,
        regularization=regularization,
        source_model=source_model,
    )


__all__ = [
    "geometry_inverse_design_report",
    "geometry_sensitivity_report",
]
