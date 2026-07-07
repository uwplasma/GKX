"""Residual, covariance, and AD/FD gates for reduced stellarator ITG objectives."""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.objectives.stellarator_contracts import (
    PARAMETER_NAMES,
    StellaratorITGOptimizationConfig,
    StellaratorObjectiveKind,
)
from spectraxgk.objectives.stellarator_reduced import (
    _qa_core_features,
    _validate_params,
    nonlinear_heat_flux_trace,
    nonlinear_heat_flux_window_metrics,
)
from spectraxgk.objectives.autodiff_validation import (
    autodiff_finite_difference_report,
    covariance_diagnostics,
)

_RESIDUAL_CONDITION_NUMBER_LIMIT = 1.0e4


def _precision_gate_tolerances(fd_step: float) -> tuple[float, float, float]:
    """Return FD tolerances that are strict in x64 and stable in float32."""

    if bool(getattr(jax.config, "jax_enable_x64", False)):
        return float(fd_step), 5.0e-3, 6.0e-4
    return max(float(fd_step), 5.0e-3), 5.0e-2, 6.0e-3


def _residual_precision_gate_tolerances(fd_step: float) -> tuple[float, float, float]:
    """Return residual-Jacobian FD tolerances that stay local near zero residuals."""

    if bool(getattr(jax.config, "jax_enable_x64", False)):
        return float(fd_step), 5.0e-3, 6.0e-4
    return min(float(fd_step), 1.0e-4), 5.0e-2, 6.0e-3


def _conditioning_gate_from_covariance(
    covariance: dict[str, Any],
    *,
    min_rank: int,
    condition_number_limit: float,
) -> dict[str, Any]:
    """Return a pass/fail gate for Gauss-Newton residual conditioning."""

    singular = np.asarray(covariance.get("jacobian_singular_values", ()), dtype=float)
    rank = int(covariance.get("sensitivity_map_rank", 0))
    condition_number = float(covariance.get("jacobian_condition_number", float("inf")))
    finite_singular_values = bool(singular.size > 0 and np.all(np.isfinite(singular)))
    finite_condition = bool(np.isfinite(condition_number))
    smallest = float(singular[-1]) if finite_singular_values else 0.0
    limit = float(condition_number_limit)
    if int(min_rank) < 1:
        raise ValueError("min_rank must be >= 1")
    if limit <= 0.0:
        raise ValueError("condition_number_limit must be positive")
    passed = bool(
        finite_singular_values
        and finite_condition
        and rank >= int(min_rank)
        and condition_number <= limit
        and smallest > 0.0
    )
    return {
        "passed": passed,
        "finite_singular_values": finite_singular_values,
        "finite_condition_number": finite_condition,
        "sensitivity_map_rank": rank,
        "min_rank": int(min_rank),
        "rank_deficiency": int(max(int(min_rank) - rank, 0)),
        "jacobian_condition_number": condition_number,
        "condition_number_limit": limit,
        "smallest_singular_value": smallest,
    }


def stellarator_itg_objective(
    params: jnp.ndarray | Sequence[float],
    kind: StellaratorObjectiveKind,
    config: StellaratorITGOptimizationConfig | None = None,
) -> jnp.ndarray:
    """Return the scalar constrained QA + ITG objective for one optimization."""

    residual = stellarator_itg_objective_residual_vector(params, kind, config)
    return jnp.dot(residual, residual)


def stellarator_itg_objective_residual_names(
    kind: StellaratorObjectiveKind,
) -> tuple[str, ...]:
    """Return stable residual names for the weighted QA + ITG objective."""

    if kind not in ("growth", "quasilinear_flux", "nonlinear_heat_flux"):
        raise ValueError(f"unknown stellarator objective kind {kind!r}")
    return (
        "aspect_constraint",
        "iota_constraint",
        "qa_constraint",
        *(f"regularization_{name}" for name in PARAMETER_NAMES),
        f"{kind}_transport_objective",
    )


def stellarator_itg_objective_residual_vector(
    params: jnp.ndarray | Sequence[float],
    kind: StellaratorObjectiveKind,
    config: StellaratorITGOptimizationConfig | None = None,
) -> jnp.ndarray:
    """Return weighted residuals whose squared norm is the optimization objective.

    This is the correct local residual map for Gauss-Newton covariance and
    identifiability diagnostics. Using the initial-to-final observable
    displacement would overstate uncertainty because it measures optimizer
    travel rather than the residual left at the optimized point.
    """

    cfg = config or StellaratorITGOptimizationConfig()
    p = _validate_params(params)
    obs = _qa_core_features(p, cfg)
    dtype = p.dtype
    aspect_res = jnp.sqrt(jnp.asarray(cfg.aspect_weight, dtype=dtype)) * (
        (obs["aspect"] - cfg.target_aspect) / cfg.target_aspect
    )
    iota_res = jnp.sqrt(jnp.asarray(cfg.iota_weight, dtype=dtype)) * (
        obs["mean_iota"] - cfg.target_iota
    )
    qa_res = jnp.sqrt(jnp.asarray(cfg.qa_weight, dtype=dtype)) * obs["qa_residual"]
    reg_res = jnp.sqrt(jnp.asarray(cfg.regularization, dtype=dtype)) * p
    if kind == "growth":
        turbulence = obs["growth_rate"]
    elif kind == "quasilinear_flux":
        turbulence = obs["quasilinear_heat_flux"]
    elif kind == "nonlinear_heat_flux":
        times, heat_flux = nonlinear_heat_flux_trace(p, cfg)
        turbulence = nonlinear_heat_flux_window_metrics(
            times,
            heat_flux,
            tail_fraction=cfg.nonlinear_tail_fraction,
        )["mean"]
    else:
        raise ValueError(f"unknown stellarator objective kind {kind!r}")
    turbulence_res = jnp.sqrt(
        jnp.maximum(
            jnp.asarray(cfg.turbulence_weight, dtype=dtype) * turbulence,
            jnp.asarray(0.0, dtype=dtype),
        )
    )
    return jnp.concatenate(
        [
            jnp.asarray([aspect_res, iota_res, qa_res], dtype=dtype),
            reg_res,
            jnp.asarray([turbulence_res], dtype=dtype),
        ]
    )


def stellarator_itg_residual_sensitivity_report(
    params: jnp.ndarray | Sequence[float],
    kind: StellaratorObjectiveKind,
    config: StellaratorITGOptimizationConfig | None = None,
    *,
    step: float | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    min_rank: int = len(PARAMETER_NAMES),
    condition_number_limit: float = _RESIDUAL_CONDITION_NUMBER_LIMIT,
    covariance_regularization: float = 1.0e-8,
    finite_difference_workers: int = 1,
    finite_difference_executor: str = "thread",
) -> dict[str, Any]:
    """Check residual-Jacobian AD/FD parity and local conditioning.

    The scalar objective gradient can pass even when the residual sensitivity
    map is rank-deficient. This gate validates the full weighted residual map
    used by Gauss-Newton covariance and UQ diagnostics.
    """

    cfg = config or StellaratorITGOptimizationConfig()
    p = _validate_params(params)
    default_step, default_rtol, default_atol = _residual_precision_gate_tolerances(
        cfg.fd_step
    )
    fd_step = default_step if step is None else float(step)
    fd_rtol = default_rtol if rtol is None else float(rtol)
    fd_atol = default_atol if atol is None else float(atol)

    def residual_fn(x: jnp.ndarray) -> jnp.ndarray:
        return stellarator_itg_objective_residual_vector(x, kind, cfg)

    residual_gate = autodiff_finite_difference_report(
        residual_fn,
        p,
        step=fd_step,
        rtol=fd_rtol,
        atol=fd_atol,
        workers=finite_difference_workers,
        parallel_executor=finite_difference_executor,
    )
    jac = np.asarray(residual_gate["jacobian_ad"], dtype=float)
    residual = np.asarray(residual_fn(p), dtype=float)
    covariance = covariance_diagnostics(
        jac, residual, regularization=covariance_regularization
    )
    covariance["source"] = "weighted_objective_residual"
    covariance["residual_names"] = list(stellarator_itg_objective_residual_names(kind))
    conditioning_gate = _conditioning_gate_from_covariance(
        covariance,
        min_rank=min_rank,
        condition_number_limit=condition_number_limit,
    )
    covariance["conditioning_gate"] = conditioning_gate
    return {
        "kind": "stellarator_itg_residual_sensitivity_report",
        "objective_kind": kind,
        "passed": bool(residual_gate["passed"] and conditioning_gate["passed"]),
        "parameter_names": list(PARAMETER_NAMES),
        "residual_names": list(stellarator_itg_objective_residual_names(kind)),
        "finite_difference_gate": residual_gate,
        "conditioning_gate": conditioning_gate,
        "covariance": covariance,
    }


__all__ = [
    "_RESIDUAL_CONDITION_NUMBER_LIMIT",
    "_precision_gate_tolerances",
    "stellarator_itg_objective",
    "stellarator_itg_objective_residual_names",
    "stellarator_itg_objective_residual_vector",
    "stellarator_itg_residual_sensitivity_report",
]
