"""Autodiff, finite-difference, conditioning, and covariance gates for portfolios."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.objectives.portfolio_contracts import (
    PortfolioReduction,
    _floating_dtype,
    _objective_rows,
    _parameter_vector,
    aggregate_objective_portfolio,
    validate_objective_portfolio_contract,
)
from spectraxgk.objectives.autodiff_validation import autodiff_finite_difference_report, covariance_diagnostics


@dataclass(frozen=True)
class _PortfolioWeightOptions:
    sample_weights: Any | None
    surface_weights: Any | None
    alpha_weights: Any | None
    ky_weights: Any | None
    objective_weights: Any | None
    reduction: PortfolioReduction


@dataclass(frozen=True)
class _PortfolioSensitivityFunctions:
    row_table: Callable[[jnp.ndarray], jnp.ndarray]
    scalar: Callable[[jnp.ndarray], jnp.ndarray]
    row_vector: Callable[[jnp.ndarray], jnp.ndarray]


def _conditioning_gate(
    jacobian: np.ndarray,
    *,
    min_rank: int | None,
    condition_number_limit: float,
) -> dict[str, object]:
    jac = np.asarray(jacobian, dtype=float)
    if jac.ndim != 2:
        raise ValueError("jacobian must be a two-dimensional array")
    limit = float(condition_number_limit)
    if limit <= 0.0:
        raise ValueError("condition_number_limit must be positive")
    expected_rank = int(min_rank) if min_rank is not None else int(min(jac.shape))
    if expected_rank < 1:
        raise ValueError("min_rank must be >= 1")
    if expected_rank > int(min(jac.shape)):
        raise ValueError("min_rank cannot exceed min(jacobian.shape)")

    finite = bool(np.all(np.isfinite(jac)))
    singular_values = np.linalg.svd(jac, compute_uv=False) if finite else np.asarray([], dtype=float)
    rank = int(np.linalg.matrix_rank(jac)) if finite else 0
    if singular_values.size == 0 or float(singular_values[-1]) <= 0.0:
        condition_number = float("inf")
        smallest = 0.0
    else:
        smallest = float(singular_values[-1])
        condition_number = float(singular_values[0] / singular_values[-1])
    passed = bool(finite and rank >= expected_rank and condition_number <= limit)
    return {
        "passed": passed,
        "finite_jacobian": finite,
        "sensitivity_map_rank": rank,
        "min_rank": expected_rank,
        "rank_deficiency": int(max(expected_rank - rank, 0)),
        "jacobian_condition_number": condition_number,
        "condition_number_limit": limit,
        "smallest_singular_value": smallest,
        "singular_values": singular_values.tolist(),
    }


def _portfolio_weight_options(
    *,
    sample_weights: Any | None,
    surface_weights: Any | None,
    alpha_weights: Any | None,
    ky_weights: Any | None,
    objective_weights: Any | None,
    reduction: PortfolioReduction,
) -> _PortfolioWeightOptions:
    return _PortfolioWeightOptions(
        sample_weights=sample_weights,
        surface_weights=surface_weights,
        alpha_weights=alpha_weights,
        ky_weights=ky_weights,
        objective_weights=objective_weights,
        reduction=reduction,
    )


def _portfolio_sensitivity_functions(
    objective_row_fn: Callable[[jnp.ndarray], Any],
    *,
    weights: _PortfolioWeightOptions,
) -> _PortfolioSensitivityFunctions:
    def row_table(x: jnp.ndarray) -> jnp.ndarray:
        rows = _objective_rows(objective_row_fn(x))
        return rows.astype(_floating_dtype(x, rows))

    def scalar_fn(x: jnp.ndarray) -> jnp.ndarray:
        return aggregate_objective_portfolio(
            row_table(x),
            sample_weights=weights.sample_weights,
            surface_weights=weights.surface_weights,
            alpha_weights=weights.alpha_weights,
            ky_weights=weights.ky_weights,
            objective_weights=weights.objective_weights,
            reduction=weights.reduction,
            validate=True,
        )

    def row_vector_fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.ravel(row_table(x))

    return _PortfolioSensitivityFunctions(
        row_table=row_table,
        scalar=scalar_fn,
        row_vector=row_vector_fn,
    )


def _validate_portfolio_base_contract(
    base_rows: jnp.ndarray,
    *,
    weights: _PortfolioWeightOptions,
) -> Any:
    return validate_objective_portfolio_contract(
        base_rows,
        sample_weights=weights.sample_weights,
        surface_weights=weights.surface_weights,
        alpha_weights=weights.alpha_weights,
        ky_weights=weights.ky_weights,
        objective_weights=weights.objective_weights,
        reduction=weights.reduction,
    )


def _autodiff_gate(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    p: jnp.ndarray,
    *,
    step: float,
    rtol: float,
    atol: float,
    workers: int,
    parallel_executor: str,
) -> dict[str, Any]:
    return autodiff_finite_difference_report(
        fn,
        p,
        step=step,
        rtol=rtol,
        atol=atol,
        workers=workers,
        parallel_executor=parallel_executor,
    )


def _portfolio_gradient_gates(
    *,
    funcs: _PortfolioSensitivityFunctions,
    p: jnp.ndarray,
    step: float,
    rtol: float,
    atol: float,
    workers: int,
    parallel_executor: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scalar_gate = _autodiff_gate(
        funcs.scalar,
        p,
        step=step,
        rtol=rtol,
        atol=atol,
        workers=workers,
        parallel_executor=parallel_executor,
    )
    row_gate = _autodiff_gate(
        funcs.row_vector,
        p,
        step=step,
        rtol=rtol,
        atol=atol,
        workers=workers,
        parallel_executor=parallel_executor,
    )
    return scalar_gate, row_gate


def _portfolio_conditioning_and_covariance(
    *,
    row_vector: jnp.ndarray,
    row_jacobian_gate: Mapping[str, Any],
    min_rank: int | None,
    condition_number_limit: float,
    covariance_regularization: float,
) -> tuple[dict[str, object], dict[str, Any]]:
    row_jacobian = np.asarray(row_jacobian_gate["jacobian_ad"], dtype=float)
    row_residual = np.asarray(row_vector, dtype=float)
    conditioning_gate = _conditioning_gate(
        row_jacobian,
        min_rank=min_rank,
        condition_number_limit=condition_number_limit,
    )
    covariance = covariance_diagnostics(
        row_jacobian,
        row_residual,
        regularization=covariance_regularization,
    )
    covariance["source"] = "objective_portfolio_rows"
    return conditioning_gate, covariance


def _portfolio_sensitivity_payload(
    *,
    p: jnp.ndarray,
    funcs: _PortfolioSensitivityFunctions,
    contract: Any,
    scalar_gradient_gate: Mapping[str, Any],
    row_jacobian_gate: Mapping[str, Any],
    conditioning_gate: Mapping[str, Any],
    covariance: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "kind": "objective_portfolio_sensitivity_report",
        "passed": bool(
            scalar_gradient_gate["passed"]
            and row_jacobian_gate["passed"]
            and conditioning_gate["passed"]
        ),
        "portfolio_contract": contract.to_dict(),
        "parameter_count": int(p.size),
        "base_value": float(funcs.scalar(p)),
        "base_row_norm": float(jnp.linalg.norm(funcs.row_vector(p))),
        "scalar_gradient_gate": dict(scalar_gradient_gate),
        "row_jacobian_gate": dict(row_jacobian_gate),
        "conditioning_gate": dict(conditioning_gate),
        "covariance": dict(covariance),
    }


def objective_portfolio_sensitivity_report(
    objective_row_fn: Callable[[jnp.ndarray], Any],
    params: Any,
    *,
    sample_weights: Any | None = None,
    surface_weights: Any | None = None,
    alpha_weights: Any | None = None,
    ky_weights: Any | None = None,
    objective_weights: Any | None = None,
    reduction: PortfolioReduction = "weighted_mean",
    step: float = 1.0e-4,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
    min_rank: int | None = None,
    condition_number_limit: float = 1.0e8,
    covariance_regularization: float = 1.0e-9,
    workers: int = 1,
    parallel_executor: str = "thread",
) -> dict[str, object]:
    """AD/FD and conditioning report for a reduced objective-row portfolio.

    ``objective_row_fn`` is the backend boundary: production callers can wire a
    VMEC/Boozer/quasilinear row builder into this gate while tests can use a
    cheap fixture. The report checks both the final scalar reduction and the
    unreduced row sensitivity map so a passing scalar gradient cannot hide a
    rank-deficient or badly conditioned objective table.
    """

    p = _parameter_vector(params)
    weights = _portfolio_weight_options(
        sample_weights=sample_weights,
        surface_weights=surface_weights,
        alpha_weights=alpha_weights,
        ky_weights=ky_weights,
        objective_weights=objective_weights,
        reduction=reduction,
    )
    funcs = _portfolio_sensitivity_functions(objective_row_fn, weights=weights)
    base_rows = funcs.row_table(p)
    contract = _validate_portfolio_base_contract(base_rows, weights=weights)
    scalar_gradient_gate, row_jacobian_gate = _portfolio_gradient_gates(
        funcs=funcs,
        p=p,
        step=step,
        rtol=rtol,
        atol=atol,
        workers=workers,
        parallel_executor=parallel_executor,
    )
    conditioning_gate, covariance = _portfolio_conditioning_and_covariance(
        row_vector=funcs.row_vector(p),
        row_jacobian_gate=row_jacobian_gate,
        min_rank=min_rank,
        condition_number_limit=condition_number_limit,
        covariance_regularization=covariance_regularization,
    )

    return _portfolio_sensitivity_payload(
        p=p,
        funcs=funcs,
        contract=contract,
        scalar_gradient_gate=scalar_gradient_gate,
        row_jacobian_gate=row_jacobian_gate,
        conditioning_gate=conditioning_gate,
        covariance=covariance,
    )



__all__ = [
    "objective_portfolio_sensitivity_report",
]
