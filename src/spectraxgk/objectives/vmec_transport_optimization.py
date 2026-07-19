"""Boundary gradients and projected line searches for VMEC transport objectives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


@dataclass(frozen=True)
class _BoundaryTransportGradientEvaluation:
    """Numerical values extracted from one VMEC transport-gradient evaluation."""

    specs: tuple[Any, ...]
    params: np.ndarray
    residual: np.ndarray
    objective_value: float | None
    gradient: np.ndarray
    residual_finite: bool
    gradient_finite: bool
    finite: bool
    sensitive: bool
    residual_l2: float
    residual_linf: float
    gradient_l2: float
    gradient_linf: float


def boundary_spec_record(spec: Any, *, fallback_index: int) -> dict[str, Any]:
    """Return a JSON-safe summary for a VMEC-JAX boundary parameter spec."""

    name = str(getattr(spec, "name", f"p{fallback_index}"))
    return {
        "name": name,
        "kind": None if not hasattr(spec, "kind") else str(getattr(spec, "kind")),
        "mode_index": None
        if not hasattr(spec, "index")
        else int(getattr(spec, "index")),
        "m": None if not hasattr(spec, "m") else int(getattr(spec, "m")),
        "n": None if not hasattr(spec, "n") else int(getattr(spec, "n")),
    }


def _top_gradient_components(
    gradient: np.ndarray,
    specs: Sequence[Any],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    """Return the largest active boundary-gradient components."""

    grad = np.asarray(gradient, dtype=float).reshape(-1)
    order = np.argsort(-np.abs(grad))
    scale = max(float(np.linalg.norm(grad)), 1.0e-300)
    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(order[: max(0, int(top_n))], start=1):
        spec = specs[int(idx)] if int(idx) < len(specs) else object()
        row = boundary_spec_record(spec, fallback_index=int(idx))
        value = float(grad[int(idx)])
        row.update(
            {
                "rank": rank,
                "parameter_index": int(idx),
                "gradient": value,
                "abs_gradient": abs(value),
                "fraction_of_l2_norm": abs(value) / scale,
            }
        )
        rows.append(row)
    return rows


def _resolve_boundary_transport_params(
    optimizer: Any,
    params: Sequence[float] | np.ndarray | None,
) -> tuple[tuple[Any, ...], np.ndarray]:
    """Return optimizer specs and a finite active-boundary parameter vector."""

    specs = tuple(getattr(optimizer, "_specs", ()))
    if params is None:
        if not specs:
            raise ValueError(
                "params must be provided when optimizer._specs is unavailable"
            )
        params_array = np.zeros(len(specs), dtype=float)
    else:
        params_array = np.asarray(params, dtype=float).reshape(-1)
    if not np.all(np.isfinite(params_array)):
        raise ValueError("params must be finite")
    return specs, params_array


def _transport_gradient_classification(*, finite: bool, sensitive: bool) -> str:
    if sensitive:
        return "sensitive_boundary_transport_objective"
    if not finite:
        return "invalid_nonfinite_transport_gradient"
    return "locally_flat_or_underconditioned_transport_objective"


def _transport_gradient_next_action(*, sensitive: bool) -> str:
    if sensitive:
        return (
            "use a constraint-preserving projected update or constrained line search along the leading "
            "transport-gradient components"
        )
    return (
        "do not launch another blind scalar-weight ladder; change the transport observable, "
        "sample set, or finite-difference scale until the local boundary sensitivity is measurable"
    )


def _evaluate_boundary_transport_gradient(
    optimizer: Any,
    *,
    params_array: np.ndarray,
    specs: tuple[Any, ...],
    sensitivity_atol: float,
) -> _BoundaryTransportGradientEvaluation:
    """Evaluate the residual/objective/gradient contract for a VMEC optimizer."""

    residual = np.asarray(optimizer.residual_fun(params_array), dtype=float).reshape(-1)
    objective_value_raw, gradient_raw = optimizer.objective_and_gradient_fun(
        params_array
    )
    objective_value = _finite_float(objective_value_raw)
    gradient = np.asarray(gradient_raw, dtype=float).reshape(-1)
    if gradient.size != params_array.size:
        raise ValueError(
            f"gradient size {gradient.size} does not match parameter size {params_array.size}"
        )

    residual_finite = bool(np.all(np.isfinite(residual)))
    gradient_finite = bool(np.all(np.isfinite(gradient)))
    finite = bool(objective_value is not None and residual_finite and gradient_finite)
    gradient_l2 = float(np.linalg.norm(gradient)) if gradient_finite else float("nan")
    gradient_linf = (
        float(np.max(np.abs(gradient))) if gradient.size and gradient_finite else 0.0
    )
    residual_l2 = float(np.linalg.norm(residual)) if residual_finite else float("nan")
    residual_linf = (
        float(np.max(np.abs(residual))) if residual.size and residual_finite else 0.0
    )
    return _BoundaryTransportGradientEvaluation(
        specs=specs,
        params=params_array,
        residual=residual,
        objective_value=objective_value,
        gradient=gradient,
        residual_finite=residual_finite,
        gradient_finite=gradient_finite,
        finite=finite,
        sensitive=bool(finite and gradient_l2 > float(sensitivity_atol)),
        residual_l2=residual_l2,
        residual_linf=residual_linf,
        gradient_l2=gradient_l2,
        gradient_linf=gradient_linf,
    )


def _boundary_transport_gradient_report_payload(
    evaluation: _BoundaryTransportGradientEvaluation,
    *,
    label: str,
    top_n: int,
    sensitivity_atol: float,
) -> dict[str, Any]:
    """Pack the JSON-safe base report for a VMEC transport-gradient diagnostic."""

    return {
        "kind": "vmex_transport_gradient_diagnostic",
        "label": str(label),
        "parameter_count": int(evaluation.params.size),
        "residual_count": int(evaluation.residual.size),
        "objective_value": evaluation.objective_value,
        "residual_norm_l2": evaluation.residual_l2,
        "residual_norm_linf": evaluation.residual_linf,
        "gradient_norm_l2": evaluation.gradient_l2,
        "gradient_norm_linf": evaluation.gradient_linf,
        "sensitivity_atol": float(sensitivity_atol),
        "finite": evaluation.finite,
        "transport_sensitivity_detected": evaluation.sensitive,
        "classification": _transport_gradient_classification(
            finite=evaluation.finite,
            sensitive=evaluation.sensitive,
        ),
        "top_gradient_components": _top_gradient_components(
            evaluation.gradient,
            evaluation.specs,
            top_n=int(top_n),
        ),
        "next_action": _transport_gradient_next_action(sensitive=evaluation.sensitive),
    }


def _boundary_transport_jacobian_report(
    optimizer: Any,
    *,
    params_array: np.ndarray,
) -> dict[str, Any]:
    """Pack optional dense residual-Jacobian diagnostics when the optimizer has them."""

    jacobian_fun = getattr(optimizer, "jacobian_fun", None)
    if not callable(jacobian_fun):
        return {
            "available": False,
            "reason": "optimizer_has_no_jacobian_fun",
        }
    jac = np.asarray(jacobian_fun(params_array), dtype=float)
    if jac.ndim == 1:
        jac = jac.reshape(1, -1)
    jac_finite = bool(np.all(np.isfinite(jac)))
    return {
        "available": True,
        "shape": [int(item) for item in jac.shape],
        "finite": jac_finite,
        "frobenius_norm": float(np.linalg.norm(jac)) if jac_finite else None,
        "row_norms": (
            [float(item) for item in np.linalg.norm(jac, axis=1)]
            if jac_finite
            else None
        ),
    }


def build_boundary_transport_gradient_report(
    optimizer: Any,
    *,
    params: Sequence[float] | np.ndarray | None = None,
    label: str = "vmex_transport_gradient",
    top_n: int = 12,
    sensitivity_atol: float = 1.0e-12,
    include_jacobian: bool = False,
) -> dict[str, Any]:
    """Evaluate transport residual and boundary-gradient diagnostics.

    Parameters
    ----------
    optimizer:
        VMEC-JAX-like optimizer exposing ``residual_fun(params)`` and
        ``objective_and_gradient_fun(params)``.  The optional ``_specs`` member
        is used only for readable boundary-coefficient labels.
    params:
        Active boundary-parameter vector.  ``None`` means the zero-increment
        vector aligned with ``optimizer._specs``.
    label:
        Human-readable artifact label.
    top_n:
        Number of largest gradient components to keep.
    sensitivity_atol:
        Absolute L2 threshold below which the boundary transport response is
        classified as locally flat for optimization purposes.
    include_jacobian:
        If true and the optimizer exposes ``jacobian_fun``, include dense
        residual-Jacobian norms.  This can be substantially more expensive than
        the reverse scalar-gradient path.
    """

    specs, params_array = _resolve_boundary_transport_params(optimizer, params)
    evaluation = _evaluate_boundary_transport_gradient(
        optimizer,
        params_array=params_array,
        specs=specs,
        sensitivity_atol=sensitivity_atol,
    )
    report = _boundary_transport_gradient_report_payload(
        evaluation,
        label=label,
        top_n=top_n,
        sensitivity_atol=sensitivity_atol,
    )
    if bool(include_jacobian):
        report["jacobian"] = _boundary_transport_jacobian_report(
            optimizer,
            params_array=params_array,
        )

    return report


def write_boundary_transport_gradient_report(
    report: dict[str, Any], path: str | Path
) -> Path:
    """Write a boundary-gradient diagnostic JSON artifact."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return out


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


def sparse_descent_direction_from_gradient_report(
    report: Mapping[str, Any],
    *,
    parameter_count: int | None = None,
    top_n: int | None = None,
    boundary_chain_collection: Mapping[str, Any] | None = None,
    require_boundary_chain_exact_fd: bool = True,
    require_growth_branch_locality: bool = False,
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
    allowed_indices = (
        None
        if boundary_chain_collection is None
        else set(
            boundary_chain_accepted_parameter_indices(
                boundary_chain_collection,
                require_exact_fd=bool(require_boundary_chain_exact_fd),
                require_growth_branch_locality=bool(require_growth_branch_locality),
            )
        )
    )
    direction = np.zeros(count, dtype=float)
    for row in components[:limit]:
        if not isinstance(row, Mapping):
            raise ValueError("gradient component rows must be mappings")
        index = int(row["parameter_index"])
        if index < 0 or index >= count:
            raise ValueError(
                f"gradient component index {index} is outside parameter_count={count}"
            )
        if allowed_indices is not None and index not in allowed_indices:
            continue
        value = _finite_float(row.get("gradient"))
        if value is None:
            raise ValueError(f"gradient component {index} is not finite")
        direction[index] = -value
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(
            "selected gradient components produce a zero descent direction"
        )
    return direction / norm


def boundary_chain_accepted_parameter_indices(
    collection: Mapping[str, Any],
    *,
    require_exact_fd: bool = True,
    require_growth_branch_locality: bool = False,
) -> tuple[int, ...]:
    """Return parameter indices admitted by a boundary-chain collection gate."""

    rows = collection.get("rows")
    if not isinstance(rows, Sequence):
        raise ValueError("boundary-chain collection must contain rows")
    accepted: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("boundary-chain collection rows must be mappings")
        index = row.get("index")
        if index is None:
            continue
        internal_ok = bool(row.get("frozen_axis_jvp_vjp_consistent"))
        exact_ok = bool(row.get("frozen_axis_matches_exact_fd"))
        convention_ok = bool(row.get("frozen_axis_convention_verified", False))
        branch_ok = (
            bool(row.get("growth_branch_locality_checked"))
            and bool(row.get("growth_branch_locality_passed"))
            if bool(require_growth_branch_locality)
            else True
        )
        derivative_ok = (
            exact_ok if bool(require_exact_fd) else (exact_ok or convention_ok)
        )
        if internal_ok and branch_ok and derivative_ok:
            accepted.append(int(index))
    return tuple(dict.fromkeys(accepted))


def projected_line_search_input_manifest(
    report: Mapping[str, Any],
    *,
    steps: Sequence[float],
    top_n: int | None = None,
    boundary_chain_collection: Mapping[str, Any] | None = None,
    require_boundary_chain_exact_fd: bool = True,
    require_growth_branch_locality: bool = False,
) -> dict[str, Any]:
    """Build a JSON-safe manifest for projected line-search input generation."""

    direction = sparse_descent_direction_from_gradient_report(
        report,
        top_n=top_n,
        boundary_chain_collection=boundary_chain_collection,
        require_boundary_chain_exact_fd=bool(require_boundary_chain_exact_fd),
        require_growth_branch_locality=bool(require_growth_branch_locality),
    )
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
    manifest = {
        "kind": "vmex_projected_transport_line_search_input_manifest",
        "parameter_count": int(direction.size),
        "top_n": int(
            len(report.get("top_gradient_components", ())) if top_n is None else top_n
        ),
        "direction_l2_norm": float(np.linalg.norm(direction)),
        "direction_linf_norm": float(np.max(np.abs(direction))),
        "steps": step_rows,
    }
    if boundary_chain_collection is not None:
        manifest["boundary_chain_filter"] = {
            "enabled": True,
            "require_exact_fd": bool(require_boundary_chain_exact_fd),
            "require_frozen_axis_convention_when_exact_fd_missing": not bool(
                require_boundary_chain_exact_fd
            ),
            "require_growth_branch_locality": bool(require_growth_branch_locality),
            "collection_classification": boundary_chain_collection.get(
                "classification"
            ),
            "accepted_parameter_indices": list(
                boundary_chain_accepted_parameter_indices(
                    boundary_chain_collection,
                    require_exact_fd=bool(require_boundary_chain_exact_fd),
                    require_growth_branch_locality=bool(require_growth_branch_locality),
                )
            ),
        }
    return manifest


def _candidate_metric(row: Mapping[str, Any]) -> float | None:
    for key in (
        "transport_metric_final",
        "transport_objective_final",
        "spectrax_objective_final",
    ):
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
            signed = (
                baseline_metric - metric
                if policy.lower_is_better
                else metric - baseline_metric
            )
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
        "kind": "vmex_projected_transport_line_search_admission",
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
    "boundary_chain_accepted_parameter_indices",
    "boundary_spec_record",
    "build_boundary_transport_gradient_report",
    "projected_line_search_input_manifest",
    "select_projected_line_search_candidate",
    "sparse_descent_direction_from_gradient_report",
    "write_boundary_transport_gradient_report",
]
