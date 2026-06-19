"""Record normalization helpers for reduced zonal-flow objectives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np


MissingDampingPolicy = Literal["fail", "zero"]


def _first_present(record: Mapping[str, Any], keys: Sequence[str]) -> tuple[str | None, Any]:
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

    normalized: list[dict[str, float]] = []
    missing_damping_count = 0
    missing_recurrence_count = 0
    for record in records:
        surface = _optional_metric(record, surface_keys, field="surface", default=0.0)
        alpha = _optional_metric(record, alpha_keys, field="alpha", default=0.0)
        kx = _required_float(record, kx_keys, field="kx")
        residual_value = _required_float(record, residual_keys, field="residual_level")
        if residual_value <= 0.0:
            raise ValueError("residual_level must be strictly positive in every record")

        damping_value = _optional_metric(record, damping_keys, field="damping_rate")
        if damping_value is None:
            missing_damping_count += 1
            if missing_damping_policy == "fail":
                raise ValueError("record is missing finite damping_rate")
            damping_value = 0.0

        growth_value = _optional_metric(
            record,
            linear_growth_keys,
            field="linear_growth_rate",
            default=0.0,
        )
        recurrence_value = _optional_metric(
            record,
            recurrence_keys,
            field="recurrence_amplitude",
        )
        if recurrence_value is None:
            missing_recurrence_count += 1
            recurrence_value = 0.0

        normalized.append(
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
            }
        )

    if not normalized:
        raise ValueError("at least one zonal-flow objective record is required")

    surfaces = sorted({row["surface"] for row in normalized})
    alphas = sorted({row["alpha"] for row in normalized})
    kx_values = sorted({row["kx"] for row in normalized})
    surface_index = _axis_index(surfaces)
    alpha_index = _axis_index(alphas)
    kx_index = _axis_index(kx_values)
    shape = (len(surfaces), len(alphas), len(kx_values))
    residual_tensor = np.full(shape, np.nan, dtype=float)
    damping_tensor = np.full(shape, np.nan, dtype=float)
    growth_tensor = np.full(shape, np.nan, dtype=float)
    recurrence_tensor = np.full(shape, np.nan, dtype=float)
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

    for name, tensor in (
        ("residual_level", residual_tensor),
        ("damping_rate", damping_tensor),
        ("linear_growth_rate", growth_tensor),
        ("recurrence_amplitude", recurrence_tensor),
    ):
        if not np.all(np.isfinite(tensor)):
            raise ValueError(f"records do not form a complete finite tensor for {name}")

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


__all__ = ["MissingDampingPolicy"]
