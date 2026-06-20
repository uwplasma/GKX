"""Boundary-gradient diagnostics for VMEC-JAX transport objectives.

The helpers here intentionally avoid importing ``vmec_jax`` at module import
time.  They operate on the small optimizer protocol exposed by VMEC-JAX
(``residual_fun``, ``objective_and_gradient_fun``, and ``_specs``), which keeps
SPECTRAX-GK tests fast while letting examples evaluate the real full-chain
transport sensitivity on machines with VMEC-JAX installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

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
        "mode_index": None if not hasattr(spec, "index") else int(getattr(spec, "index")),
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
            raise ValueError("params must be provided when optimizer._specs is unavailable")
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
    objective_value_raw, gradient_raw = optimizer.objective_and_gradient_fun(params_array)
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
    gradient_linf = float(np.max(np.abs(gradient))) if gradient.size and gradient_finite else 0.0
    residual_l2 = float(np.linalg.norm(residual)) if residual_finite else float("nan")
    residual_linf = float(np.max(np.abs(residual))) if residual.size and residual_finite else 0.0
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
        "kind": "vmec_jax_transport_gradient_diagnostic",
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
    label: str = "vmec_jax_transport_gradient",
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


def write_boundary_transport_gradient_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write a boundary-gradient diagnostic JSON artifact."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return out


__all__ = [
    "boundary_spec_record",
    "build_boundary_transport_gradient_report",
    "write_boundary_transport_gradient_report",
]
