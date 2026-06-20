"""Shared score-margin helpers for nonlinear-gradient screening reports."""

from __future__ import annotations

import math


def _metric_margin(
    value: float | None,
    *,
    target: float,
    sense: str,
    cap: float,
    value_floor: float,
) -> float:
    """Return a capped normalized evidence margin for one gate metric."""

    if value is None or not math.isfinite(float(value)):
        return 0.0
    finite_value = float(value)
    if sense == "min":
        margin = finite_value / max(float(target), float(value_floor))
    elif sense == "max":
        margin = float(target) / max(abs(finite_value), float(value_floor))
    else:  # pragma: no cover - guarded by internal call sites.
        raise ValueError(f"unsupported margin sense: {sense}")
    return max(0.0, min(float(cap), margin))


__all__ = ["_metric_margin"]
