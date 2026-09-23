"""Diagnostic data contracts and strict persisted-evidence decoding."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


from gkx.diagnostics_contract import (
    ArrayLike,
    CFL_SCALE_LABELS,
    ResolvedDiagnostics,
    SimulationDiagnostics,
)


NON_PRODUCTION_SCOPE_MARKERS = (
    "startup",
    "plumbing",
    "reduced",
    "estimator",
    "smooth_logistic",
    "mixing_length",
    "not_transport",
    "not transport",
    "not_production",
    "not production",
    "not_simulation_claim",
    "not simulation claim",
    "feasibility",
    "pilot",
    "pending",
)

PRODUCTION_SCOPE_MARKERS = (
    "production_long_window",
    "production long-window",
    "long_window_nonlinear_turbulence_gradient",
    "long-window nonlinear turbulence gradient",
    "production nonlinear window gradient",
)


@dataclass(frozen=True)
class NonlinearTurbulenceGradientEvidenceConfig:
    """Acceptance limits for production nonlinear turbulence-gradient evidence."""

    min_window_reports: int = 2
    max_window_mean_rel_spread: float = 0.15
    max_window_combined_sem_rel: float = 0.25
    max_gradient_uncertainty_rel: float = 0.50
    max_fd_asymmetry_rel: float = 0.50
    max_fd_condition_number: float = 1.0e8
    min_fd_response_fraction: float = 0.03
    value_floor: float = 1.0e-12


def _json_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if isinstance(value, int):
        return value
    return number


def _finite_float(value: Any) -> float | None:
    number = _json_number(value)
    return None if number is None else float(number)


def _nonnegative_int(value: Any) -> int:
    """Decode a persisted nonnegative integer, failing closed to zero."""

    number = _finite_float(value)
    if number is None or number < 0.0 or not number.is_integer():
        return 0
    return int(number)


def _explicit_true(value: Any) -> bool:
    """Accept only explicit Boolean truth from persisted evidence fields."""

    return isinstance(value, (bool, np.bool_)) and bool(value)


def _gate(metric: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"metric": metric, "passed": bool(passed), "detail": str(detail)}


def _artifact_passed(payload: dict[str, Any]) -> bool:
    if _explicit_true(payload.get("passed")):
        return True
    for key in ("gate_report", "promotion_gate"):
        nested = payload.get(key)
        if isinstance(nested, dict) and _explicit_true(nested.get("passed")):
            return True
    return False


# ---- artifact classification helpers ----


# ---- candidate scoring helpers ----


# ---- bracket sweep reports ----


__all__ = [
    "ArrayLike",
    "CFL_SCALE_LABELS",
    "NON_PRODUCTION_SCOPE_MARKERS",
    "PRODUCTION_SCOPE_MARKERS",
    "NonlinearTurbulenceGradientEvidenceConfig",
    "ResolvedDiagnostics",
    "SimulationDiagnostics",
    "_artifact_passed",
    "_explicit_true",
    "_finite_float",
    "_gate",
    "_json_number",
    "_nonnegative_int",
]
