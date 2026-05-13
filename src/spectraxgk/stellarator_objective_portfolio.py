"""Backend-free reduced objective portfolios for stellarator optimization.

This module only reduces already-evaluated objective rows.  It intentionally
does not import VMEC, Boozer, or solver backends so the same contract can be
used by fast CI fixtures and by production VMEC/Boozer objective drivers after
they have built a per-surface/per-alpha/per-ky objective table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np


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
    "portfolio_objective_weight_vector",
    "portfolio_sample_weight_tensor",
    "validate_objective_portfolio_contract",
]
