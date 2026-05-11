"""Finite-value validation for runtime artifact emission."""

from __future__ import annotations

from typing import Any

import numpy as np

from spectraxgk.runtime_diagnostics import validate_finite_gx_diagnostics

_RUNTIME_FIELD_NAMES = ("phi", "apar", "bpar")

__all__ = [
    "validate_finite_array",
    "validate_finite_runtime_result",
]


def validate_finite_array(value: Any, *, label: str) -> None:
    """Raise if an optional artifact array contains NaN or infinite values."""

    if value is None:
        return
    arr = np.asarray(value)
    if arr.size == 0 or np.isfinite(arr).all():
        return
    raise RuntimeError(f"{label} contains non-finite values")


def validate_finite_runtime_result(result: Any, *, label: str) -> None:
    """Validate nonlinear runtime result payloads before artifact writes."""

    if result.diagnostics is not None:
        validate_finite_gx_diagnostics(result.diagnostics, label=label)
    validate_finite_array(result.state, label=f"{label} state")
    fields = result.fields
    if fields is None:
        return
    for name in _RUNTIME_FIELD_NAMES:
        validate_finite_array(getattr(fields, name, None), label=f"{label} {name}")
