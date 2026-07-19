"""Reduced zonal-flow objectives for differentiable stellarator optimization."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from gkx.objectives.portfolio import (
    PortfolioReduction,
    aggregate_objective_portfolio,
    objective_portfolio_sensitivity_report,
)

MissingDampingPolicy = Literal["fail", "zero"]
_NormalizedZonalRecord = dict[str, float]


def _first_present(
    record: Mapping[str, Any], keys: Sequence[str]
) -> tuple[str | None, Any]:
    for key in keys:
        if key in record:
            return key, record[key]
    return None, None


def _optional_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "nan", "none", "null"}:
            return None
        value = stripped
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric when present") from exc
    if not np.isfinite(scalar):
        return None
    return scalar


def _required_float(
    record: Mapping[str, Any],
    keys: Sequence[str],
    *,
    field: str,
    default: float | None = None,
) -> float:
    key, raw = _first_present(record, keys)
    value = _optional_float(default if key is None else raw, field=field)
    if value is None:
        raise ValueError(f"record is missing finite {field}; tried keys {list(keys)}")
    return value


def _optional_metric(
    record: Mapping[str, Any],
    keys: Sequence[str],
    *,
    field: str,
    default: float | None = None,
) -> float | None:
    key, raw = _first_present(record, keys)
    if key is None:
        return default
    return _optional_float(raw, field=field)


def _axis_index(values: list[float]) -> dict[float, int]:
    return {value: index for index, value in enumerate(values)}


def _normalize_zonal_objective_record(
    record: Mapping[str, Any],
    *,
    surface_keys: Sequence[str],
    alpha_keys: Sequence[str],
    kx_keys: Sequence[str],
    residual_keys: Sequence[str],
    damping_keys: Sequence[str],
    linear_growth_keys: Sequence[str],
    recurrence_keys: Sequence[str],
    missing_damping_policy: MissingDampingPolicy,
) -> tuple[_NormalizedZonalRecord, bool, bool]:
    """Normalize one record and report which optional metrics were absent."""

    surface = _optional_metric(record, surface_keys, field="surface", default=0.0)
    alpha = _optional_metric(record, alpha_keys, field="alpha", default=0.0)
    kx = _required_float(record, kx_keys, field="kx")
    residual_value = _required_float(record, residual_keys, field="residual_level")
    if residual_value <= 0.0:
        raise ValueError("residual_level must be strictly positive in every record")

    damping_missing = False
    damping_value = _optional_metric(record, damping_keys, field="damping_rate")
    if damping_value is None:
        damping_missing = True
        if missing_damping_policy == "fail":
            raise ValueError("record is missing finite damping_rate")
        damping_value = 0.0

    growth_value = _optional_metric(
        record,
        linear_growth_keys,
        field="linear_growth_rate",
        default=0.0,
    )
    recurrence_missing = False
    recurrence_value = _optional_metric(
        record,
        recurrence_keys,
        field="recurrence_amplitude",
    )
    if recurrence_value is None:
        recurrence_missing = True
        recurrence_value = 0.0

    return (
        {
            "surface": float(surface if surface is not None else 0.0),
            "alpha": float(alpha if alpha is not None else 0.0),
            "kx": float(kx),
            "residual_level": float(residual_value),
            "damping_rate": float(damping_value),
            "linear_growth_rate": float(
                growth_value if growth_value is not None else 0.0
            ),
            "recurrence_amplitude": float(recurrence_value),
        },
        damping_missing,
        recurrence_missing,
    )


def _normalize_zonal_objective_records(
    records: Iterable[Mapping[str, Any]],
    *,
    surface_keys: Sequence[str],
    alpha_keys: Sequence[str],
    kx_keys: Sequence[str],
    residual_keys: Sequence[str],
    damping_keys: Sequence[str],
    linear_growth_keys: Sequence[str],
    recurrence_keys: Sequence[str],
    missing_damping_policy: MissingDampingPolicy,
) -> tuple[list[_NormalizedZonalRecord], int, int]:
    """Normalize raw zonal objective records and count optional omissions."""

    normalized: list[_NormalizedZonalRecord] = []
    missing_damping_count = 0
    missing_recurrence_count = 0
    for record in records:
        row, damping_missing, recurrence_missing = _normalize_zonal_objective_record(
            record,
            surface_keys=surface_keys,
            alpha_keys=alpha_keys,
            kx_keys=kx_keys,
            residual_keys=residual_keys,
            damping_keys=damping_keys,
            linear_growth_keys=linear_growth_keys,
            recurrence_keys=recurrence_keys,
            missing_damping_policy=missing_damping_policy,
        )
        normalized.append(row)
        missing_damping_count += int(damping_missing)
        missing_recurrence_count += int(recurrence_missing)

    if not normalized:
        raise ValueError("at least one zonal-flow objective record is required")
    return normalized, missing_damping_count, missing_recurrence_count


def _zonal_objective_axes(
    normalized: Sequence[Mapping[str, float]],
) -> tuple[list[float], list[float], list[float]]:
    """Return sorted surface, alpha, and kx axes from normalized records."""

    return (
        sorted({row["surface"] for row in normalized}),
        sorted({row["alpha"] for row in normalized}),
        sorted({row["kx"] for row in normalized}),
    )


def _empty_zonal_metric_tensors(
    surfaces: Sequence[float],
    alphas: Sequence[float],
    kx_values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate metric tensors on the record-defined axis product."""

    shape = (len(surfaces), len(alphas), len(kx_values))
    return (
        np.full(shape, np.nan, dtype=float),
        np.full(shape, np.nan, dtype=float),
        np.full(shape, np.nan, dtype=float),
        np.full(shape, np.nan, dtype=float),
    )


def _fill_zonal_metric_tensors(
    normalized: Sequence[Mapping[str, float]],
    *,
    surfaces: list[float],
    alphas: list[float],
    kx_values: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill metric tensors and reject duplicate grid points."""

    residual_tensor, damping_tensor, growth_tensor, recurrence_tensor = (
        _empty_zonal_metric_tensors(surfaces, alphas, kx_values)
    )
    surface_index = _axis_index(surfaces)
    alpha_index = _axis_index(alphas)
    kx_index = _axis_index(kx_values)
    seen: set[tuple[int, int, int]] = set()
    for row in normalized:
        index = (
            surface_index[row["surface"]],
            alpha_index[row["alpha"]],
            kx_index[row["kx"]],
        )
        if index in seen:
            raise ValueError(
                "duplicate zonal-flow objective record for "
                f"surface={row['surface']}, alpha={row['alpha']}, kx={row['kx']}"
            )
        seen.add(index)
        residual_tensor[index] = row["residual_level"]
        damping_tensor[index] = row["damping_rate"]
        growth_tensor[index] = row["linear_growth_rate"]
        recurrence_tensor[index] = row["recurrence_amplitude"]
    return residual_tensor, damping_tensor, growth_tensor, recurrence_tensor


def _validate_zonal_metric_tensors(
    *,
    residual_tensor: np.ndarray,
    damping_tensor: np.ndarray,
    growth_tensor: np.ndarray,
    recurrence_tensor: np.ndarray,
) -> None:
    """Require complete finite tensors for every zonal objective metric."""

    for name, tensor in (
        ("residual_level", residual_tensor),
        ("damping_rate", damping_tensor),
        ("linear_growth_rate", growth_tensor),
        ("recurrence_amplitude", recurrence_tensor),
    ):
        if not np.all(np.isfinite(tensor)):
            raise ValueError(f"records do not form a complete finite tensor for {name}")


def _finite_metric_tensor_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    surface_keys: Sequence[str],
    alpha_keys: Sequence[str],
    kx_keys: Sequence[str],
    residual_keys: Sequence[str],
    damping_keys: Sequence[str],
    linear_growth_keys: Sequence[str],
    recurrence_keys: Sequence[str],
    missing_damping_policy: MissingDampingPolicy,
) -> tuple[
    list[float],
    list[float],
    list[float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, float]],
    int,
    int,
]:
    if missing_damping_policy not in {"fail", "zero"}:
        raise ValueError("missing_damping_policy must be 'fail' or 'zero'")

    normalized, missing_damping_count, missing_recurrence_count = (
        _normalize_zonal_objective_records(
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
    )
    surfaces, alphas, kx_values = _zonal_objective_axes(normalized)
    residual_tensor, damping_tensor, growth_tensor, recurrence_tensor = (
        _fill_zonal_metric_tensors(
            normalized,
            surfaces=surfaces,
            alphas=alphas,
            kx_values=kx_values,
        )
    )
    _validate_zonal_metric_tensors(
        residual_tensor=residual_tensor,
        damping_tensor=damping_tensor,
        growth_tensor=growth_tensor,
        recurrence_tensor=recurrence_tensor,
    )

    return (
        surfaces,
        alphas,
        kx_values,
        residual_tensor,
        damping_tensor,
        growth_tensor,
        recurrence_tensor,
        normalized,
        missing_damping_count,
        missing_recurrence_count,
    )


def _zonal_row_table(
    *,
    normalized_records: Sequence[Mapping[str, float]],
    surfaces: list[float],
    alphas: list[float],
    kx_values: list[float],
    objective_rows: Any,
    objective_weights: Any,
) -> list[dict[str, float]]:
    rows_np = np.asarray(objective_rows, dtype=float)
    weights = np.asarray(objective_weights, dtype=float)
    normalized_weights = weights / float(np.sum(weights))
    surface_index = _axis_index(surfaces)
    alpha_index = _axis_index(alphas)
    kx_index = _axis_index(kx_values)
    row_table: list[dict[str, float]] = []
    base_keys = (
        "surface",
        "alpha",
        "kx",
        "residual_level",
        "damping_rate",
        "linear_growth_rate",
        "recurrence_amplitude",
    )
    for item in normalized_records:
        objective_row = rows_np[
            surface_index[float(item["surface"])],
            alpha_index[float(item["alpha"])],
            kx_index[float(item["kx"])],
            :,
        ]
        row = {key: float(item[key]) for key in base_keys}
        row.update(
            inverse_residual=float(objective_row[0]),
            growth_over_residual=float(objective_row[2]),
            sample_objective=float(np.dot(objective_row, normalized_weights)),
        )
        row_table.append(row)
    return row_table


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
            raise ValueError(
                "zonal-flow objective weights must be finite and non-negative"
            )
        if float(sum(weights)) <= 0.0:
            raise ValueError(
                "at least one zonal-flow objective weight must be positive"
            )
        if (not np.isfinite(float(self.residual_floor))) or float(
            self.residual_floor
        ) <= 0.0:
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


@dataclass(frozen=True)
class _ZonalRecordTensor:
    surfaces: list[float]
    alphas: list[float]
    kx_values: list[float]
    residual: np.ndarray
    damping: np.ndarray
    growth: np.ndarray
    recurrence: np.ndarray
    normalized_records: list[dict[str, float]]
    missing_damping_count: int
    missing_recurrence_count: int


@dataclass(frozen=True)
class _ZonalObjectiveEvaluation:
    rows_np: np.ndarray
    reduced: jnp.ndarray
    row_table: list[dict[str, float]]


def _metric_tensor(
    value: Any, *, name: str, strictly_positive: bool = False
) -> jnp.ndarray:
    array = jnp.asarray(value)
    if int(array.ndim) != 3:
        raise ValueError(f"{name} must have shape (n_surface, n_alpha, n_kx)")
    if any(int(size) < 1 for size in array.shape):
        raise ValueError(f"{name} dimensions must all be positive")
    if jnp.issubdtype(array.dtype, jnp.complexfloating) or not jnp.issubdtype(
        array.dtype, jnp.number
    ):
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
    residual = _metric_tensor(
        residual_level, name="residual_level", strictly_positive=True
    )
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
        residual, damping, growth, recurrence = jnp.broadcast_arrays(
            residual, damping, growth, recurrence
        )
    except ValueError as exc:
        raise ValueError(
            "all zonal-flow metric tensors must be broadcast-compatible"
        ) from exc

    safe_residual = jnp.maximum(
        residual, jnp.asarray(float(cfg.residual_floor), dtype=residual.dtype)
    )
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


def _zonal_record_tensor(
    records: Iterable[Mapping[str, Any]],
    *,
    surface_keys: Sequence[str],
    alpha_keys: Sequence[str],
    kx_keys: Sequence[str],
    residual_keys: Sequence[str],
    damping_keys: Sequence[str],
    linear_growth_keys: Sequence[str],
    recurrence_keys: Sequence[str],
    missing_damping_policy: MissingDampingPolicy,
) -> _ZonalRecordTensor:
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
    return _ZonalRecordTensor(
        surfaces=surfaces,
        alphas=alphas,
        kx_values=kx_values,
        residual=residual,
        damping=damping,
        growth=growth,
        recurrence=recurrence,
        normalized_records=normalized,
        missing_damping_count=int(missing_damping_count),
        missing_recurrence_count=int(missing_recurrence_count),
    )


def _evaluate_zonal_record_objective(
    tensor: _ZonalRecordTensor,
    *,
    config: ZonalFlowObjectiveConfig,
    reduction: PortfolioReduction,
) -> _ZonalObjectiveEvaluation:
    rows = zonal_flow_objective_rows(
        residual_level=tensor.residual,
        damping_rate=tensor.damping,
        linear_growth_rate=tensor.growth,
        recurrence_amplitude=tensor.recurrence,
        config=config,
    )
    reduced = zonal_flow_reduced_objective(
        residual_level=tensor.residual,
        damping_rate=tensor.damping,
        linear_growth_rate=tensor.growth,
        recurrence_amplitude=tensor.recurrence,
        config=config,
        reduction=reduction,
    )
    rows_np = np.asarray(rows, dtype=float)
    row_table = _zonal_row_table(
        normalized_records=tensor.normalized_records,
        surfaces=tensor.surfaces,
        alphas=tensor.alphas,
        kx_values=tensor.kx_values,
        objective_rows=rows_np,
        objective_weights=config.objective_weights(),
    )
    return _ZonalObjectiveEvaluation(
        rows_np=rows_np,
        reduced=reduced,
        row_table=row_table,
    )


def _zonal_promotion_payload(
    *,
    tensor: _ZonalRecordTensor,
    claim_level: str | None,
) -> dict[str, object]:
    promotion_ready = (
        tensor.missing_damping_count == 0 and tensor.missing_recurrence_count == 0
    )
    payload_claim = claim_level or (
        "promotable_zonal_flow_objective_rows"
        if promotion_ready
        else "diagnostic_zonal_flow_objective_rows_not_promoted_to_optimization_claim"
    )
    return {
        "claim_level": payload_claim,
        "promotion_ready": bool(promotion_ready),
        "missing_damping_count": int(tensor.missing_damping_count),
        "missing_recurrence_count": int(tensor.missing_recurrence_count),
    }


def _zonal_axes_payload(tensor: _ZonalRecordTensor) -> dict[str, list[float]]:
    return {
        "surface": [float(value) for value in tensor.surfaces],
        "alpha": [float(value) for value in tensor.alphas],
        "kx": [float(value) for value in tensor.kx_values],
    }


def _zonal_metrics_payload(tensor: _ZonalRecordTensor) -> dict[str, object]:
    return {
        "residual_level": tensor.residual.tolist(),
        "damping_rate": tensor.damping.tolist(),
        "linear_growth_rate": tensor.growth.tolist(),
        "recurrence_amplitude": tensor.recurrence.tolist(),
    }


def _zonal_objective_artifact_payload(
    *,
    tensor: _ZonalRecordTensor,
    evaluation: _ZonalObjectiveEvaluation,
    config: ZonalFlowObjectiveConfig,
    missing_damping_policy: MissingDampingPolicy,
    claim_level: str | None,
    source_paths: Sequence[str] | None,
    reduction: PortfolioReduction,
) -> dict[str, object]:
    payload = {
        "kind": "zonal_flow_objective_artifact",
        "objective_names": list(ZONAL_FLOW_OBJECTIVE_NAMES),
        "objective_config": config.to_dict(),
        "missing_damping_policy": missing_damping_policy,
        "axes": _zonal_axes_payload(tensor),
        "sample_count": int(len(tensor.normalized_records)),
        "metrics": _zonal_metrics_payload(tensor),
        "objective_rows": evaluation.rows_np.tolist(),
        "reduced_objective": float(np.asarray(evaluation.reduced)),
        "row_table": evaluation.row_table,
        "source_paths": list(source_paths or []),
        "reduction": reduction,
    }
    payload.update(_zonal_promotion_payload(tensor=tensor, claim_level=claim_level))
    return payload


def zonal_flow_objective_artifact_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    config: ZonalFlowObjectiveConfig | None = None,
    surface_keys: Sequence[str] = ("surface", "surface_index", "torflux"),
    alpha_keys: Sequence[str] = ("alpha", "field_line_label"),
    kx_keys: Sequence[str] = ("kx", "kx_target", "kx_rhoi"),
    residual_keys: Sequence[str] = ("residual_level", "gkx_residual"),
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
    tensor = _zonal_record_tensor(
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
    evaluation = _evaluate_zonal_record_objective(
        tensor,
        config=cfg,
        reduction=reduction,
    )
    return _zonal_objective_artifact_payload(
        tensor=tensor,
        evaluation=evaluation,
        config=cfg,
        missing_damping_policy=missing_damping_policy,
        claim_level=claim_level,
        source_paths=source_paths,
        reduction=reduction,
    )


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
