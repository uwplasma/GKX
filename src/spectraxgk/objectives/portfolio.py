"""Backend-free objective portfolio shape, weight, and reduction contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from spectraxgk.objectives.autodiff_validation import (
    autodiff_finite_difference_report,
    covariance_diagnostics,
)

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
    return bool(
        jnp.issubdtype(dtype, jnp.number)
        and not jnp.issubdtype(dtype, jnp.complexfloating)
    )


def _floating_dtype(*arrays: jnp.ndarray) -> jnp.dtype:
    return jnp.result_type(*(arrays + (jnp.asarray(1.0),)))


def _objective_rows(objective_rows: Any) -> jnp.ndarray:
    rows = jnp.asarray(objective_rows)
    if int(rows.ndim) != 4:
        raise ValueError(
            "objective_rows must have shape (n_surface, n_alpha, n_ky, n_objective)"
        )
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
    except (
        Exception
    ) as exc:  # pragma: no cover - exercised by JAX tracers under jit/grad.
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
    n_surface, n_alpha, n_ky, _n_objective = (
        int(size) for size in objective_rows.shape
    )
    has_sample_weights = sample_weights is not None
    has_axis_weights = any(
        weight is not None for weight in (surface_weights, alpha_weights, ky_weights)
    )
    if has_sample_weights and has_axis_weights:
        raise ValueError(
            "provide either sample_weights or separable surface/alpha/ky weights, not both"
        )
    if has_sample_weights:
        array = jnp.asarray(sample_weights, dtype=dtype)
        if tuple(int(size) for size in array.shape) != (n_surface, n_alpha, n_ky):
            raise ValueError(
                "sample_weights must have shape (n_surface, n_alpha, n_ky)"
            )
        _validate_concrete_weights(array, name="sample_weights")
        return array / jnp.sum(array)

    surface = _normalized_vector(
        surface_weights, size=n_surface, name="surface_weights", dtype=dtype
    )
    alpha = _normalized_vector(
        alpha_weights, size=n_alpha, name="alpha_weights", dtype=dtype
    )
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
    """Validate row/weight shapes and concrete finite non-negative weights."""

    if reduction not in ("weighted_mean", "mean", "max"):
        raise ValueError("reduction must be one of 'weighted_mean', 'mean', or 'max'")
    if reduction == "mean" and any(
        weight is not None
        for weight in (
            sample_weights,
            surface_weights,
            alpha_weights,
            ky_weights,
            objective_weights,
        )
    ):
        raise ValueError("mean reduction does not accept weights; use weighted_mean")
    if reduction == "max" and any(
        weight is not None
        for weight in (sample_weights, surface_weights, alpha_weights, ky_weights)
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
        objective_weights=objective_weights
        if reduction in ("weighted_mean", "max")
        else None,
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
            weight is not None
            for weight in (surface_weights, alpha_weights, ky_weights)
        )
        and reduction == "weighted_mean",
        uses_objective_weights=objective_weights is not None
        and reduction in ("weighted_mean", "max"),
    )


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

    ``weighted_mean`` normalizes sample and objective weights; ``mean`` is
    unweighted, while ``max`` selects the worst objective-weighted sample.
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
    singular_values = (
        np.linalg.svd(jac, compute_uv=False) if finite else np.asarray([], dtype=float)
    )
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
    """Check AD/FD parity and conditioning of scalar and row sensitivities.

    Checking unreduced rows prevents a scalar pass from hiding rank deficiency.
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
    "PortfolioReduction",
    "StellaratorObjectivePortfolioContract",
    "aggregate_objective_portfolio",
    "objective_portfolio_sensitivity_report",
    "portfolio_objective_weight_vector",
    "portfolio_sample_weight_tensor",
    "validate_objective_portfolio_contract",
]
