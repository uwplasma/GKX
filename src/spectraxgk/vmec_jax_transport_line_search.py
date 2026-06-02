"""Utilities for projected VMEC-JAX transport line searches.

These helpers convert a boundary-gradient diagnostic into a normalized
descent direction and JSON-safe line-search manifests.  The VMEC-JAX-dependent
input writer lives in ``tools/``; this module is deliberately backend-free so
CI can test the admission bookkeeping without launching equilibrium solves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProjectedLineSearchPolicy:
    """Selection policy for projected transport line-search candidates."""

    minimum_relative_improvement: float = 0.0
    lower_is_better: bool = True
    require_gate_passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe policy representation."""

        return {
            "minimum_relative_improvement": float(self.minimum_relative_improvement),
            "lower_is_better": bool(self.lower_is_better),
            "require_gate_passed": bool(self.require_gate_passed),
        }


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def sparse_descent_direction_from_gradient_report(
    report: Mapping[str, Any],
    *,
    parameter_count: int | None = None,
    top_n: int | None = None,
) -> np.ndarray:
    """Return a normalized sparse descent direction from a gradient report.

    The direction is ``-grad`` restricted to the ranked
    ``top_gradient_components``.  This convention makes positive line-search
    steps lower a lower-is-better transport objective to first order.
    """

    count = int(parameter_count or report.get("parameter_count") or 0)
    if count <= 0:
        raise ValueError("parameter_count must be positive")
    components = report.get("top_gradient_components")
    if not isinstance(components, Sequence) or not components:
        raise ValueError("gradient report must contain top_gradient_components")
    limit = len(components) if top_n is None else max(0, int(top_n))
    direction = np.zeros(count, dtype=float)
    for row in components[:limit]:
        if not isinstance(row, Mapping):
            raise ValueError("gradient component rows must be mappings")
        index = int(row["parameter_index"])
        if index < 0 or index >= count:
            raise ValueError(f"gradient component index {index} is outside parameter_count={count}")
        value = _finite_float(row.get("gradient"))
        if value is None:
            raise ValueError(f"gradient component {index} is not finite")
        direction[index] = -value
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("selected gradient components produce a zero descent direction")
    return direction / norm


def projected_line_search_input_manifest(
    report: Mapping[str, Any],
    *,
    steps: Sequence[float],
    top_n: int | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe manifest for projected line-search input generation."""

    direction = sparse_descent_direction_from_gradient_report(report, top_n=top_n)
    step_rows = []
    for step in steps:
        step_f = float(step)
        if not np.isfinite(step_f) or step_f <= 0.0:
            raise ValueError("line-search steps must be finite and positive")
        step_rows.append(
            {
                "step": step_f,
                "parameter_l2_norm": step_f,
                "parameter_linf_norm": float(np.max(np.abs(step_f * direction))),
            }
        )
    return {
        "kind": "vmec_jax_projected_transport_line_search_input_manifest",
        "parameter_count": int(direction.size),
        "top_n": int(len(report.get("top_gradient_components", ())) if top_n is None else top_n),
        "direction_l2_norm": float(np.linalg.norm(direction)),
        "direction_linf_norm": float(np.max(np.abs(direction))),
        "steps": step_rows,
    }


def _candidate_metric(row: Mapping[str, Any]) -> float | None:
    for key in ("transport_metric_final", "transport_objective_final", "spectrax_objective_final"):
        value = _finite_float(row.get(key))
        if value is not None:
            return value
    return None


def select_projected_line_search_candidate(
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    policy: ProjectedLineSearchPolicy | None = None,
) -> dict[str, Any]:
    """Select the best gate-passing projected line-search candidate."""

    policy = policy or ProjectedLineSearchPolicy()
    baseline_metric = _candidate_metric(baseline)
    annotated = []
    admitted = []
    for raw in candidates:
        row = dict(raw)
        metric = _candidate_metric(row)
        blockers: list[str] = []
        if metric is None:
            blockers.append("missing_transport_metric")
        if baseline_metric is None:
            blockers.append("missing_baseline_transport_metric")
        if bool(policy.require_gate_passed) and not bool(row.get("gate_passed")):
            blockers.append("gate_failed")
        improvement = None
        if metric is not None and baseline_metric is not None:
            signed = baseline_metric - metric if policy.lower_is_better else metric - baseline_metric
            improvement = float(signed / max(abs(baseline_metric), 1.0e-300))
            if improvement < float(policy.minimum_relative_improvement):
                blockers.append("insufficient_transport_improvement")
        row["transport_metric"] = metric
        row["relative_transport_improvement"] = improvement
        row["admission_blockers"] = blockers
        row["admitted"] = not blockers
        annotated.append(row)
        if not blockers:
            admitted.append(row)
    selected = None
    if admitted:
        selected = max(
            admitted,
            key=lambda row: (
                float(row.get("relative_transport_improvement") or 0.0),
                float(row.get("step") or 0.0),
            ),
        )
    return {
        "kind": "vmec_jax_projected_transport_line_search_admission",
        "policy": policy.to_dict(),
        "baseline_transport_metric": baseline_metric,
        "candidates": annotated,
        "selected_candidate": selected,
        "passed": selected is not None,
        "next_action": (
            "launch matched long-window nonlinear audits for the selected projected candidate"
            if selected is not None
            else "no projected candidate both improves transport and preserves solved-equilibrium gates"
        ),
    }


__all__ = [
    "ProjectedLineSearchPolicy",
    "projected_line_search_input_manifest",
    "select_projected_line_search_candidate",
    "sparse_descent_direction_from_gradient_report",
]
