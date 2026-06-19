"""Reduced zonal-flow objectives for differentiable stellarator optimization.

This module is intentionally backend-free.  Production callers should build
zonal-response metrics from VMEC/Boozer/SPECTRAX-GK rows, then pass the metric
tensors here for reduction, finite-difference checks, and UQ diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.objectives.portfolio_contracts import (
    PortfolioReduction,
    aggregate_objective_portfolio,
)
from spectraxgk.objectives.portfolio_sensitivity import objective_portfolio_sensitivity_report
from spectraxgk.objectives.zonal_records import (
    MissingDampingPolicy,
    _finite_metric_tensor_from_records,
    _zonal_row_table,
)


ZONAL_FLOW_OBJECTIVE_NAMES = (
    "inverse_residual",
    "damping_rate",
    "growth_over_residual",
    "recurrence_amplitude",
)


@dataclass(frozen=True)
class ZonalFlowObjectiveConfig:
    """Weights and floors for a minimizable zonal-flow objective.

    The objective rewards large residual zonal response by minimizing
    ``1 / residual`` and penalizes collisionless damping, linear growth not
    screened by the residual, and late-time recurrence/envelope amplitude.
    Nonlinear heat-flux suppression remains a separate holdout gate.
    """

    residual_weight: float = 1.0
    damping_weight: float = 1.0
    growth_over_residual_weight: float = 0.0
    recurrence_weight: float = 0.0
    residual_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        weights = (
            float(self.residual_weight),
            float(self.damping_weight),
            float(self.growth_over_residual_weight),
            float(self.recurrence_weight),
        )
        if any((not np.isfinite(weight)) or weight < 0.0 for weight in weights):
            raise ValueError("zonal-flow objective weights must be finite and non-negative")
        if float(sum(weights)) <= 0.0:
            raise ValueError("at least one zonal-flow objective weight must be positive")
        if (not np.isfinite(float(self.residual_floor))) or float(self.residual_floor) <= 0.0:
            raise ValueError("residual_floor must be finite and positive")

    def objective_weights(self) -> jnp.ndarray:
        """Return the normalized objective-column weights used by the reducer."""

        return jnp.asarray(
            [
                self.residual_weight,
                self.damping_weight,
                self.growth_over_residual_weight,
                self.recurrence_weight,
            ],
            dtype=jnp.float32,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""

        payload = asdict(self)
        payload["objective_names"] = list(ZONAL_FLOW_OBJECTIVE_NAMES)
        return payload


def _metric_tensor(value: Any, *, name: str, strictly_positive: bool = False) -> jnp.ndarray:
    array = jnp.asarray(value)
    if int(array.ndim) != 3:
        raise ValueError(f"{name} must have shape (n_surface, n_alpha, n_kx)")
    if any(int(size) < 1 for size in array.shape):
        raise ValueError(f"{name} dimensions must all be positive")
    if jnp.issubdtype(array.dtype, jnp.complexfloating) or not jnp.issubdtype(array.dtype, jnp.number):
        raise TypeError(f"{name} must be a real numeric array")
    try:
        concrete = np.asarray(array, dtype=float)
    except Exception as exc:  # pragma: no cover - triggered only under JAX tracing.
        class_name = type(exc).__name__
        if "Tracer" in class_name or "Concretization" in class_name:
            return array
        raise
    if not np.all(np.isfinite(concrete)):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and np.any(concrete <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def zonal_flow_objective_rows(
    *,
    residual_level: Any,
    damping_rate: Any,
    linear_growth_rate: Any | None = None,
    recurrence_amplitude: Any | None = None,
    config: ZonalFlowObjectiveConfig | None = None,
) -> jnp.ndarray:
    """Return objective rows with shape ``(surface, alpha, kx, objective)``.

    ``residual_level`` is the late-time residual normalized to the initial
    zonal potential.  Larger residuals reduce the first objective column.
    ``damping_rate`` should be positive for decaying GAM/zonal envelopes.
    ``linear_growth_rate`` is optional and encodes a suppression-relevance
    metric: high ITG growth with weak residuals is penalized.  The recurrence
    column should be a non-negative late-envelope or moment-tail amplitude.
    """

    cfg = config or ZonalFlowObjectiveConfig()
    residual = _metric_tensor(residual_level, name="residual_level", strictly_positive=True)
    damping = _metric_tensor(damping_rate, name="damping_rate")
    growth = (
        jnp.zeros_like(residual)
        if linear_growth_rate is None
        else _metric_tensor(linear_growth_rate, name="linear_growth_rate")
    )
    recurrence = (
        jnp.zeros_like(residual)
        if recurrence_amplitude is None
        else _metric_tensor(recurrence_amplitude, name="recurrence_amplitude")
    )
    try:
        residual, damping, growth, recurrence = jnp.broadcast_arrays(residual, damping, growth, recurrence)
    except ValueError as exc:
        raise ValueError("all zonal-flow metric tensors must be broadcast-compatible") from exc

    safe_residual = jnp.maximum(residual, jnp.asarray(float(cfg.residual_floor), dtype=residual.dtype))
    rows = jnp.stack(
        (
            1.0 / safe_residual,
            jnp.maximum(damping, 0.0),
            jnp.maximum(growth, 0.0) / safe_residual,
            jnp.maximum(recurrence, 0.0),
        ),
        axis=-1,
    )
    return rows.astype(jnp.result_type(rows, jnp.asarray(1.0)))


def zonal_flow_reduced_objective(
    *,
    residual_level: Any,
    damping_rate: Any,
    linear_growth_rate: Any | None = None,
    recurrence_amplitude: Any | None = None,
    config: ZonalFlowObjectiveConfig | None = None,
    sample_weights: Any | None = None,
    surface_weights: Any | None = None,
    alpha_weights: Any | None = None,
    ky_weights: Any | None = None,
    reduction: PortfolioReduction = "weighted_mean",
) -> jnp.ndarray:
    """Reduce zonal-flow metric tensors to one differentiable scalar objective."""

    cfg = config or ZonalFlowObjectiveConfig()
    rows = zonal_flow_objective_rows(
        residual_level=residual_level,
        damping_rate=damping_rate,
        linear_growth_rate=linear_growth_rate,
        recurrence_amplitude=recurrence_amplitude,
        config=cfg,
    )
    return aggregate_objective_portfolio(
        rows,
        sample_weights=sample_weights,
        surface_weights=surface_weights,
        alpha_weights=alpha_weights,
        ky_weights=ky_weights,
        objective_weights=cfg.objective_weights(),
        reduction=reduction,
    )


def zonal_flow_objective_artifact_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    config: ZonalFlowObjectiveConfig | None = None,
    surface_keys: Sequence[str] = ("surface", "surface_index", "torflux"),
    alpha_keys: Sequence[str] = ("alpha", "field_line_label"),
    kx_keys: Sequence[str] = ("kx", "kx_target", "kx_rhoi"),
    residual_keys: Sequence[str] = ("residual_level", "spectrax_residual"),
    damping_keys: Sequence[str] = ("damping_rate", "gam_damping_rate"),
    linear_growth_keys: Sequence[str] = ("linear_growth_rate", "growth_rate", "gamma"),
    recurrence_keys: Sequence[str] = (
        "recurrence_amplitude",
        "tail_std_ratio",
        "residual_std",
        "tail_std",
    ),
    missing_damping_policy: MissingDampingPolicy = "fail",
    claim_level: str | None = None,
    source_paths: Sequence[str] | None = None,
    reduction: PortfolioReduction = "weighted_mean",
) -> dict[str, object]:
    """Build a strict JSON-friendly zonal-flow objective artifact.

    The input is a table of validated zonal-response metrics.  Rows are mapped
    onto the shared ``(surface, alpha, kx)`` portfolio tensor used by the
    stellarator objective stack.  Missing damping rates fail by default because
    a promoted zonal-flow optimization claim must know the damping convention.
    Diagnostic artifacts can set ``missing_damping_policy='zero'`` to produce
    rows while carrying an explicit ``promotion_ready=False`` flag.
    """

    cfg = config or ZonalFlowObjectiveConfig()
    (
        surfaces,
        alphas,
        kx_values,
        residual,
        damping,
        growth,
        recurrence,
        normalized,
        missing_damping_count,
        missing_recurrence_count,
    ) = _finite_metric_tensor_from_records(
        records,
        surface_keys=surface_keys,
        alpha_keys=alpha_keys,
        kx_keys=kx_keys,
        residual_keys=residual_keys,
        damping_keys=damping_keys,
        linear_growth_keys=linear_growth_keys,
        recurrence_keys=recurrence_keys,
        missing_damping_policy=missing_damping_policy,
    )

    rows = zonal_flow_objective_rows(
        residual_level=residual,
        damping_rate=damping,
        linear_growth_rate=growth,
        recurrence_amplitude=recurrence,
        config=cfg,
    )
    reduced = zonal_flow_reduced_objective(
        residual_level=residual,
        damping_rate=damping,
        linear_growth_rate=growth,
        recurrence_amplitude=recurrence,
        config=cfg,
        reduction=reduction,
    )
    rows_np = np.asarray(rows, dtype=float)
    row_table = _zonal_row_table(
        normalized_records=normalized,
        surfaces=surfaces,
        alphas=alphas,
        kx_values=kx_values,
        objective_rows=rows_np,
        objective_weights=cfg.objective_weights(),
    )

    promotion_ready = missing_damping_count == 0 and missing_recurrence_count == 0
    payload_claim = claim_level or (
        "promotable_zonal_flow_objective_rows"
        if promotion_ready
        else "diagnostic_zonal_flow_objective_rows_not_promoted_to_optimization_claim"
    )
    return {
        "kind": "zonal_flow_objective_artifact",
        "claim_level": payload_claim,
        "promotion_ready": bool(promotion_ready),
        "objective_names": list(ZONAL_FLOW_OBJECTIVE_NAMES),
        "objective_config": cfg.to_dict(),
        "missing_damping_policy": missing_damping_policy,
        "missing_damping_count": int(missing_damping_count),
        "missing_recurrence_count": int(missing_recurrence_count),
        "axes": {
            "surface": [float(value) for value in surfaces],
            "alpha": [float(value) for value in alphas],
            "kx": [float(value) for value in kx_values],
        },
        "sample_count": int(len(normalized)),
        "metrics": {
            "residual_level": residual.tolist(),
            "damping_rate": damping.tolist(),
            "linear_growth_rate": growth.tolist(),
            "recurrence_amplitude": recurrence.tolist(),
        },
        "objective_rows": rows_np.tolist(),
        "reduced_objective": float(np.asarray(reduced)),
        "row_table": row_table,
        "source_paths": list(source_paths or []),
        "reduction": reduction,
    }


def _metric_mapping_rows(
    metrics: Mapping[str, Any],
    *,
    config: ZonalFlowObjectiveConfig,
) -> jnp.ndarray:
    if "residual_level" not in metrics or "damping_rate" not in metrics:
        raise ValueError("metric_fn must return residual_level and damping_rate")
    return zonal_flow_objective_rows(
        residual_level=metrics["residual_level"],
        damping_rate=metrics["damping_rate"],
        linear_growth_rate=metrics.get("linear_growth_rate"),
        recurrence_amplitude=metrics.get("recurrence_amplitude"),
        config=config,
    )


def zonal_flow_objective_sensitivity_report(
    metric_fn: Callable[[jnp.ndarray], Mapping[str, Any]],
    params: Any,
    *,
    config: ZonalFlowObjectiveConfig | None = None,
    sample_weights: Any | None = None,
    surface_weights: Any | None = None,
    alpha_weights: Any | None = None,
    ky_weights: Any | None = None,
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
    """AD/FD, row-Jacobian, and UQ gate for a zonal-flow optimization map."""

    cfg = config or ZonalFlowObjectiveConfig()

    def row_fn(x: jnp.ndarray) -> jnp.ndarray:
        return _metric_mapping_rows(metric_fn(x), config=cfg)

    report = objective_portfolio_sensitivity_report(
        row_fn,
        params,
        sample_weights=sample_weights,
        surface_weights=surface_weights,
        alpha_weights=alpha_weights,
        ky_weights=ky_weights,
        objective_weights=cfg.objective_weights(),
        reduction=reduction,
        step=step,
        rtol=rtol,
        atol=atol,
        min_rank=min_rank,
        condition_number_limit=condition_number_limit,
        covariance_regularization=covariance_regularization,
        workers=workers,
        parallel_executor=parallel_executor,
    )
    report["kind"] = "zonal_flow_objective_sensitivity_report"
    report["objective_config"] = cfg.to_dict()
    report["objective_names"] = list(ZONAL_FLOW_OBJECTIVE_NAMES)
    report["claim_level"] = (
        "reduced_zonal_flow_objective_gradient_gate_not_nonlinear_turbulence_suppression_claim"
    )
    return report
