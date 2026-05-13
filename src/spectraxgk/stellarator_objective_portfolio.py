"""Backend-free reduced objective portfolios for stellarator optimization.

This module only reduces already-evaluated objective rows.  It intentionally
does not import VMEC, Boozer, or solver backends so the same contract can be
used by fast CI fixtures and by production VMEC/Boozer objective drivers after
they have built a per-surface/per-alpha/per-ky objective table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from spectraxgk.autodiff_validation import autodiff_finite_difference_report, covariance_diagnostics


PortfolioReduction = Literal["weighted_mean", "mean", "max"]


@dataclass(frozen=True)
class StellaratorObjectivePortfolioContract:
    """Static shape/weight contract for a reduced objective portfolio."""

    n_surfaces: int
    n_alphas: int
    n_ky: int
    n_objectives: int
    reduction: PortfolioReduction
    uses_sample_weights: bool
    uses_separable_sample_weights: bool
    uses_objective_weights: bool

    @property
    def row_shape(self) -> tuple[int, int, int, int]:
        """Expected objective-table shape ``(surface, alpha, ky, objective)``."""

        return (self.n_surfaces, self.n_alphas, self.n_ky, self.n_objectives)

    @property
    def sample_shape(self) -> tuple[int, int, int]:
        """Expected sample-weight shape ``(surface, alpha, ky)``."""

        return (self.n_surfaces, self.n_alphas, self.n_ky)

    @property
    def n_samples(self) -> int:
        """Number of surface/alpha/ky samples in the portfolio."""

        return self.n_surfaces * self.n_alphas * self.n_ky

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""

        payload = asdict(self)
        payload["row_shape"] = list(self.row_shape)
        payload["sample_shape"] = list(self.sample_shape)
        payload["n_samples"] = int(self.n_samples)
        return payload


def _is_real_numeric_dtype(dtype: jnp.dtype) -> bool:
    return bool(jnp.issubdtype(dtype, jnp.number) and not jnp.issubdtype(dtype, jnp.complexfloating))


def _floating_dtype(*arrays: jnp.ndarray) -> jnp.dtype:
    return jnp.result_type(*(arrays + (jnp.asarray(1.0),)))


def _objective_rows(objective_rows: Any) -> jnp.ndarray:
    rows = jnp.asarray(objective_rows)
    if int(rows.ndim) != 4:
        raise ValueError("objective_rows must have shape (n_surface, n_alpha, n_ky, n_objective)")
    if any(int(size) < 1 for size in rows.shape):
        raise ValueError("objective_rows dimensions must all be positive")
    if not _is_real_numeric_dtype(rows.dtype):
        raise TypeError("objective_rows must be a real numeric array")
    return rows


def _parameter_vector(params: Any) -> jnp.ndarray:
    p = jnp.asarray(params)
    if int(p.ndim) != 1:
        raise ValueError("params must be a one-dimensional vector")
    if int(p.shape[0]) < 1:
        raise ValueError("params must contain at least one parameter")
    if not _is_real_numeric_dtype(p.dtype):
        raise TypeError("params must be a real numeric vector")
    return jnp.asarray(p, dtype=_floating_dtype(p))


def _concrete_numpy_array(value: jnp.ndarray | Any) -> np.ndarray | None:
    try:
        return np.asarray(value, dtype=float)
    except Exception as exc:  # pragma: no cover - exercised by JAX tracers under jit/grad.
        class_name = type(exc).__name__
        if "Tracer" in class_name or "Concretization" in class_name:
            return None
        raise


def _validate_concrete_weights(weights: jnp.ndarray | Any, *, name: str) -> None:
    concrete = _concrete_numpy_array(weights)
    if concrete is None:
        return
    if not np.all(np.isfinite(concrete)):
        raise ValueError(f"{name} must be finite")
    if np.any(concrete < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if float(np.sum(concrete)) <= 0.0:
        raise ValueError(f"{name} must have positive sum")


def _normalized_vector(
    weights: Any | None,
    *,
    size: int,
    name: str,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    if weights is None:
        return jnp.full((int(size),), 1.0 / float(size), dtype=dtype)
    array = jnp.asarray(weights, dtype=dtype)
    if int(array.ndim) != 1 or int(array.shape[0]) != int(size):
        raise ValueError(f"{name} must be a length-{int(size)} vector")
    _validate_concrete_weights(array, name=name)
    return array / jnp.sum(array)


def _normalized_sample_weights(
    objective_rows: jnp.ndarray,
    *,
    sample_weights: Any | None,
    surface_weights: Any | None,
    alpha_weights: Any | None,
    ky_weights: Any | None,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    n_surface, n_alpha, n_ky, _n_objective = (int(size) for size in objective_rows.shape)
    has_sample_weights = sample_weights is not None
    has_axis_weights = any(weight is not None for weight in (surface_weights, alpha_weights, ky_weights))
    if has_sample_weights and has_axis_weights:
        raise ValueError("provide either sample_weights or separable surface/alpha/ky weights, not both")
    if has_sample_weights:
        array = jnp.asarray(sample_weights, dtype=dtype)
        if tuple(int(size) for size in array.shape) != (n_surface, n_alpha, n_ky):
            raise ValueError("sample_weights must have shape (n_surface, n_alpha, n_ky)")
        _validate_concrete_weights(array, name="sample_weights")
        return array / jnp.sum(array)

    surface = _normalized_vector(surface_weights, size=n_surface, name="surface_weights", dtype=dtype)
    alpha = _normalized_vector(alpha_weights, size=n_alpha, name="alpha_weights", dtype=dtype)
    ky = _normalized_vector(ky_weights, size=n_ky, name="ky_weights", dtype=dtype)
    return surface[:, None, None] * alpha[None, :, None] * ky[None, None, :]


def portfolio_sample_weight_tensor(
    objective_rows: Any,
    *,
    sample_weights: Any | None = None,
    surface_weights: Any | None = None,
    alpha_weights: Any | None = None,
    ky_weights: Any | None = None,
) -> jnp.ndarray:
    """Return normalized sample weights with shape ``(surface, alpha, ky)``."""

    rows = _objective_rows(objective_rows)
    dtype = _floating_dtype(rows)
    return _normalized_sample_weights(
        rows,
        sample_weights=sample_weights,
        surface_weights=surface_weights,
        alpha_weights=alpha_weights,
        ky_weights=ky_weights,
        dtype=dtype,
    )


def portfolio_objective_weight_vector(
    objective_rows: Any,
    *,
    objective_weights: Any | None = None,
) -> jnp.ndarray:
    """Return normalized objective-column weights."""

    rows = _objective_rows(objective_rows)
    dtype = _floating_dtype(rows)
    return _normalized_vector(
        objective_weights,
        size=int(rows.shape[-1]),
        name="objective_weights",
        dtype=dtype,
    )


def validate_objective_portfolio_contract(
    objective_rows: Any,
    *,
    sample_weights: Any | None = None,
    surface_weights: Any | None = None,
    alpha_weights: Any | None = None,
    ky_weights: Any | None = None,
    objective_weights: Any | None = None,
    reduction: PortfolioReduction = "weighted_mean",
) -> StellaratorObjectivePortfolioContract:
    """Validate static row/weight contracts and return portfolio metadata.

    Concrete weights must be finite, non-negative, and have positive sum.  Under
    JAX tracing, value-level weight checks are deferred to the caller, but shape
    contracts remain enforced from static array shapes.
    """

    if reduction not in ("weighted_mean", "mean", "max"):
        raise ValueError("reduction must be one of 'weighted_mean', 'mean', or 'max'")
    if reduction == "mean" and any(
        weight is not None
        for weight in (sample_weights, surface_weights, alpha_weights, ky_weights, objective_weights)
    ):
        raise ValueError("mean reduction does not accept weights; use weighted_mean")
    if reduction == "max" and any(
        weight is not None for weight in (sample_weights, surface_weights, alpha_weights, ky_weights)
    ):
        raise ValueError("max reduction does not accept sample weights")

    rows = _objective_rows(objective_rows)
    _ = portfolio_sample_weight_tensor(
        rows,
        sample_weights=sample_weights if reduction == "weighted_mean" else None,
        surface_weights=surface_weights if reduction == "weighted_mean" else None,
        alpha_weights=alpha_weights if reduction == "weighted_mean" else None,
        ky_weights=ky_weights if reduction == "weighted_mean" else None,
    )
    _ = portfolio_objective_weight_vector(
        rows,
        objective_weights=objective_weights if reduction in ("weighted_mean", "max") else None,
    )

    n_surface, n_alpha, n_ky, n_objective = (int(size) for size in rows.shape)
    return StellaratorObjectivePortfolioContract(
        n_surfaces=n_surface,
        n_alphas=n_alpha,
        n_ky=n_ky,
        n_objectives=n_objective,
        reduction=reduction,
        uses_sample_weights=sample_weights is not None and reduction == "weighted_mean",
        uses_separable_sample_weights=any(
            weight is not None for weight in (surface_weights, alpha_weights, ky_weights)
        )
        and reduction == "weighted_mean",
        uses_objective_weights=objective_weights is not None and reduction in ("weighted_mean", "max"),
    )


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


def aggregate_objective_portfolio(
    objective_rows: Any,
    *,
    sample_weights: Any | None = None,
    surface_weights: Any | None = None,
    alpha_weights: Any | None = None,
    ky_weights: Any | None = None,
    objective_weights: Any | None = None,
    reduction: PortfolioReduction = "weighted_mean",
    validate: bool = True,
) -> jnp.ndarray:
    """Reduce a ``(surface, alpha, ky, objective)`` table to one scalar.

    ``weighted_mean`` normalizes both sample and objective weights to unit sum,
    making the scalar invariant to the caller's absolute weight scale.  ``mean``
    is the unweighted mean over every table entry.  ``max`` returns the
    worst-case objective-weighted sample and is intended for diagnostics rather
    than smooth gradient-based optimization.
    """

    if validate:
        validate_objective_portfolio_contract(
            objective_rows,
            sample_weights=sample_weights,
            surface_weights=surface_weights,
            alpha_weights=alpha_weights,
            ky_weights=ky_weights,
            objective_weights=objective_weights,
            reduction=reduction,
        )

    rows = _objective_rows(objective_rows)
    dtype = _floating_dtype(rows)
    values = rows.astype(dtype)

    if reduction == "mean":
        return jnp.mean(values)
    if reduction == "weighted_mean":
        sample = _normalized_sample_weights(
            values,
            sample_weights=sample_weights,
            surface_weights=surface_weights,
            alpha_weights=alpha_weights,
            ky_weights=ky_weights,
            dtype=dtype,
        )
        objective = _normalized_vector(
            objective_weights,
            size=int(values.shape[-1]),
            name="objective_weights",
            dtype=dtype,
        )
        return jnp.sum(values * sample[..., None] * objective)
    if reduction == "max":
        objective = _normalized_vector(
            objective_weights,
            size=int(values.shape[-1]),
            name="objective_weights",
            dtype=dtype,
        )
        return jnp.max(jnp.sum(values * objective, axis=-1))
    raise ValueError("reduction must be one of 'weighted_mean', 'mean', or 'max'")


__all__ = [
    "PortfolioReduction",
    "StellaratorObjectivePortfolioContract",
    "aggregate_objective_portfolio",
    "objective_portfolio_sensitivity_report",
    "portfolio_objective_weight_vector",
    "portfolio_sample_weight_tensor",
    "validate_objective_portfolio_contract",
]
