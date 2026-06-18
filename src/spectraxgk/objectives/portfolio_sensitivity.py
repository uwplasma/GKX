"""Autodiff, finite-difference, conditioning, and covariance gates for portfolios."""

from __future__ import annotations

from collections.abc import Callable
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
from spectraxgk.validation.autodiff import autodiff_finite_difference_report, covariance_diagnostics

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

    def row_table(x: jnp.ndarray) -> jnp.ndarray:
        rows = _objective_rows(objective_row_fn(x))
        return rows.astype(_floating_dtype(x, rows))

    base_rows = row_table(p)
    contract = validate_objective_portfolio_contract(
        base_rows,
        sample_weights=sample_weights,
        surface_weights=surface_weights,
        alpha_weights=alpha_weights,
        ky_weights=ky_weights,
        objective_weights=objective_weights,
        reduction=reduction,
    )

    def scalar_fn(x: jnp.ndarray) -> jnp.ndarray:
        return aggregate_objective_portfolio(
            row_table(x),
            sample_weights=sample_weights,
            surface_weights=surface_weights,
            alpha_weights=alpha_weights,
            ky_weights=ky_weights,
            objective_weights=objective_weights,
            reduction=reduction,
            validate=True,
        )

    def row_vector_fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.ravel(row_table(x))

    scalar_gradient_gate = autodiff_finite_difference_report(
        scalar_fn,
        p,
        step=step,
        rtol=rtol,
        atol=atol,
        workers=workers,
        parallel_executor=parallel_executor,
    )
    row_jacobian_gate = autodiff_finite_difference_report(
        row_vector_fn,
        p,
        step=step,
        rtol=rtol,
        atol=atol,
        workers=workers,
        parallel_executor=parallel_executor,
    )
    row_jacobian = np.asarray(row_jacobian_gate["jacobian_ad"], dtype=float)
    row_residual = np.asarray(row_vector_fn(p), dtype=float)
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

    return {
        "kind": "objective_portfolio_sensitivity_report",
        "passed": bool(
            scalar_gradient_gate["passed"]
            and row_jacobian_gate["passed"]
            and conditioning_gate["passed"]
        ),
        "portfolio_contract": contract.to_dict(),
        "parameter_count": int(p.size),
        "base_value": float(scalar_fn(p)),
        "base_row_norm": float(jnp.linalg.norm(row_vector_fn(p))),
        "scalar_gradient_gate": scalar_gradient_gate,
        "row_jacobian_gate": row_jacobian_gate,
        "conditioning_gate": conditioning_gate,
        "covariance": covariance,
    }



__all__ = [
    "objective_portfolio_sensitivity_report",
]
