"""Record normalization helpers for reduced zonal-flow objectives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np


MissingDampingPolicy = Literal["fail", "zero"]
_NormalizedZonalRecord = dict[str, float]


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
            "linear_growth_rate": float(growth_value if growth_value is not None else 0.0),
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


__all__ = ["MissingDampingPolicy"]
