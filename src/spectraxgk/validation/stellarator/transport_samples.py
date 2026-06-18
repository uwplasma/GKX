"""Sample-coverage and reduced transport metric helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from spectraxgk.validation.stellarator.transport_policies import (
    DEFAULT_TRANSPORT_METRIC_KEYS,
    VMECJAXNonlinearAuditPolicy,
    _finite_float_or_none,
)

def candidate_transport_metric(
    candidate: Mapping[str, Any],
    *,
    metric_keys: Sequence[str] = DEFAULT_TRANSPORT_METRIC_KEYS,
) -> dict[str, Any]:
    """Return the first finite transport metric found in a candidate summary."""

    for key in tuple(str(item) for item in metric_keys):
        value = _finite_float_or_none(candidate.get(key))
        if value is not None:
            return {
                "available": True,
                "value": value,
                "source": key,
                "uses_total_objective_proxy": key == "objective_final",
            }
    return {
        "available": False,
        "value": None,
        "source": None,
        "uses_total_objective_proxy": False,
    }


def _finite_sequence(values: Any) -> tuple[float, ...]:
    if values is None:
        return ()
    if isinstance(values, np.ndarray):
        raw_values: Sequence[Any] = values.ravel().tolist()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        raw_values = values
    else:
        raw_values = (values,)
    out: list[float] = []
    for value in raw_values:
        finite = _finite_float_or_none(value)
        if finite is not None:
            out.append(finite)
    return tuple(out)


def _sample_values(sample_set: Any, *names: str) -> tuple[float, ...]:
    if sample_set is None:
        return ()
    for name in names:
        if isinstance(sample_set, Mapping) and name in sample_set:
            values = _finite_sequence(sample_set.get(name))
        else:
            values = _finite_sequence(getattr(sample_set, name, None))
        if values:
            return values
    return ()


def _ky_values_single_grid_compatible(values: Sequence[float]) -> bool:
    """Return whether ``ky`` values can share the current single-``Ly`` grid."""

    if not values:
        return False
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < 1 or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        return False
    base = float(np.min(arr))
    ratios = arr / base
    return bool(np.allclose(ratios, np.rint(ratios), rtol=5.0e-10, atol=5.0e-12))


def transport_objective_sample_summary(
    sample_set: Any,
    *,
    policy: VMECJAXNonlinearAuditPolicy | None = None,
) -> dict[str, Any]:
    """Summarize whether a transport objective has enough sample coverage.

    The nonlinear audit that motivated this gate was a single reduced metric:
    it improved locally but did not transfer to the replicated late-window
    heat-flux mean.  Multi-surface, multi-field-line, and multi-``k_y`` coverage
    is therefore treated as an admission requirement for the next candidate.
    """

    policy = policy or VMECJAXNonlinearAuditPolicy()
    surfaces = _sample_values(sample_set, "surfaces", "torflux_values", "rho_values")
    alphas = _sample_values(sample_set, "alphas", "alpha_values", "field_line_labels")
    ky_values = _sample_values(sample_set, "ky_values", "kys", "ky")
    surface_count = len(set(surfaces))
    alpha_count = len(set(alphas))
    ky_count = len(set(ky_values))
    sample_count = surface_count * alpha_count * ky_count
    blockers: list[str] = []
    if sample_set is None:
        blockers.append("missing_objective_sample_set")
    if surface_count < int(policy.minimum_surface_count):
        blockers.append("insufficient_surface_coverage")
    if alpha_count < int(policy.minimum_alpha_count):
        blockers.append("insufficient_field_line_coverage")
    if ky_count < int(policy.minimum_ky_count):
        blockers.append("insufficient_ky_coverage")
    if ky_count and not _ky_values_single_grid_compatible(ky_values):
        blockers.append("ky_values_not_single_grid_compatible")
    if sample_count < int(policy.minimum_sample_count):
        blockers.append("insufficient_total_sample_count")
    return {
        "surfaces": [float(item) for item in surfaces],
        "alphas": [float(item) for item in alphas],
        "ky_values": [float(item) for item in ky_values],
        "surface_count": surface_count,
        "alpha_count": alpha_count,
        "ky_count": ky_count,
        "sample_count": sample_count,
        "passed": not blockers,
        "blockers": blockers,
    }


__all__ = ["candidate_transport_metric", "transport_objective_sample_summary"]
