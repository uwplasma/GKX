"""Admission policy for VMEC-JAX transport-optimization candidates.

This module is deliberately small and independent of VMEC-JAX internals.  It
operates on JSON-safe candidate summaries produced by the optimization and
ladder tools, then answers the release-critical question: which solved WOUT, if
any, is physically admissible for expensive long-window nonlinear audits?
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np


DEFAULT_TRANSPORT_METRIC_KEYS = (
    "transport_objective_final",
    "spectrax_objective_final",
    "transport_metric_final",
    "objective_final",
)


def _finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


@dataclass(frozen=True)
class VMECJAXTransportAdmissionPolicy:
    """Fail-closed policy for selecting transport-aware VMEC candidates."""

    metric_keys: tuple[str, ...] = DEFAULT_TRANSPORT_METRIC_KEYS
    minimum_relative_improvement: float = 0.0
    lower_is_better: bool = True
    require_authoritative_gate: bool = True
    allow_baseline_fallback: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "metric_keys": list(self.metric_keys),
            "minimum_relative_improvement": float(self.minimum_relative_improvement),
            "lower_is_better": bool(self.lower_is_better),
            "require_authoritative_gate": bool(self.require_authoritative_gate),
            "allow_baseline_fallback": bool(self.allow_baseline_fallback),
        }


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


def _physical_gate_blockers(candidate: Mapping[str, Any], policy: VMECJAXTransportAdmissionPolicy) -> list[str]:
    blockers: list[str] = []
    gate_reported_passed = bool(candidate.get("gate_reported_passed", candidate.get("passed", False)))
    gate_authoritative = bool(candidate.get("gate_is_authoritative", True))
    if bool(policy.require_authoritative_gate) and not gate_authoritative:
        blockers.append("non_authoritative_gate")
    if not gate_reported_passed:
        checks = candidate.get("gate_checks", {})
        if isinstance(checks, Mapping):
            failed = [
                f"gate_{name}"
                for name, passed in checks.items()
                if passed is not None and not bool(passed)
            ]
            blockers.extend(failed or ["gate_failed"])
        else:
            blockers.append("gate_failed")
    if not bool(candidate.get("passed", False)):
        if "gate_failed" not in blockers and not any(item.startswith("gate_") for item in blockers):
            blockers.append("candidate_not_passed")
    return blockers


def _relative_improvement(
    baseline_value: float,
    candidate_value: float,
    *,
    lower_is_better: bool,
) -> float:
    signed = baseline_value - candidate_value if lower_is_better else candidate_value - baseline_value
    scale = max(abs(baseline_value), 1.0e-300)
    return float(signed / scale)


def build_transport_admission_report(
    summaries: Sequence[Mapping[str, Any]],
    *,
    policy: VMECJAXTransportAdmissionPolicy | None = None,
) -> dict[str, Any]:
    """Annotate and select VMEC-JAX transport candidates.

    A transport candidate is admitted only when it passes the physical solved-WOUT
    gate and improves the selected transport metric relative to the admitted
    baseline.  The baseline may be promoted only as a fallback audit target; it
    never counts as a transport-optimization success.
    """

    policy = policy or VMECJAXTransportAdmissionPolicy()
    annotated: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None
    for raw in summaries:
        item = dict(raw)
        metric = candidate_transport_metric(item, metric_keys=policy.metric_keys)
        physical_blockers = _physical_gate_blockers(item, policy)
        item["transport_metric"] = metric
        item["physical_gate_blockers"] = physical_blockers
        item["admission_blockers"] = list(physical_blockers)
        item["relative_transport_improvement"] = None
        item["admitted_for_transport_optimization"] = False
        item["admitted_for_long_window_nonlinear_audit"] = False
        annotated.append(item)
        if baseline is None and bool(item.get("baseline")):
            baseline = item

    if baseline is None:
        baseline = next((item for item in annotated if item.get("transport_weight") is None), None)

    baseline_metric = cast(dict[str, Any], baseline.get("transport_metric")) if baseline else None
    baseline_metric_value = (
        _finite_float_or_none(baseline_metric.get("value"))
        if isinstance(baseline_metric, Mapping)
        else None
    )
    baseline_physical_ok = baseline is not None and not baseline.get("physical_gate_blockers")
    if baseline is not None:
        baseline["admitted_for_long_window_nonlinear_audit"] = bool(baseline_physical_ok)

    admitted_transport: list[dict[str, Any]] = []
    for item in annotated:
        if bool(item.get("baseline")):
            continue
        if item.get("transport_weight") is None:
            item["admission_blockers"].append("missing_transport_weight")
        metric = cast(dict[str, Any], item["transport_metric"])
        candidate_metric_value = _finite_float_or_none(metric.get("value"))
        if candidate_metric_value is None:
            item["admission_blockers"].append("missing_transport_metric")
        if baseline_metric_value is None:
            item["admission_blockers"].append("missing_baseline_transport_metric")
        if candidate_metric_value is not None and baseline_metric_value is not None:
            improvement = _relative_improvement(
                baseline_metric_value,
                candidate_metric_value,
                lower_is_better=bool(policy.lower_is_better),
            )
            item["relative_transport_improvement"] = improvement
            if improvement < float(policy.minimum_relative_improvement):
                item["admission_blockers"].append("insufficient_transport_improvement")
        item["admitted_for_transport_optimization"] = not item["admission_blockers"]
        item["admitted_for_long_window_nonlinear_audit"] = bool(item["admitted_for_transport_optimization"])
        if bool(item["admitted_for_transport_optimization"]):
            admitted_transport.append(item)

    promoted: dict[str, Any] | None
    if admitted_transport:
        promoted = max(
            admitted_transport,
            key=lambda item: (
                float(item.get("transport_weight") or 0.0),
                float(item.get("relative_transport_improvement") or 0.0),
            ),
        )
    elif bool(policy.allow_baseline_fallback) and baseline is not None and bool(baseline_physical_ok):
        promoted = baseline
    else:
        promoted = None

    return {
        "kind": "vmec_jax_transport_admission_report",
        "policy": policy.to_dict(),
        "baseline_label": None if baseline is None else baseline.get("label"),
        "baseline_transport_metric": baseline_metric,
        "candidates": annotated,
        "admitted_transport_candidates": [
            item.get("label") for item in admitted_transport if item.get("label") is not None
        ],
        "transport_candidate_admitted": bool(admitted_transport),
        "promoted_candidate": promoted,
        "passed": promoted is not None,
        "next_action": (
            "launch matched long-window nonlinear audits for the admitted transport candidate"
            if admitted_transport
            else (
                "no transport candidate both preserved physical gates and improved the transport metric; "
                "keep the QA-only baseline and use a constraint-preserving projection/admission method"
            )
        ),
    }


def select_admitted_transport_candidate(
    summaries: Sequence[Mapping[str, Any]],
    *,
    policy: VMECJAXTransportAdmissionPolicy | None = None,
) -> dict[str, Any] | None:
    """Return the promoted candidate from :func:`build_transport_admission_report`."""

    report = build_transport_admission_report(summaries, policy=policy)
    promoted = report.get("promoted_candidate")
    return dict(promoted) if isinstance(promoted, Mapping) else None


__all__ = [
    "DEFAULT_TRANSPORT_METRIC_KEYS",
    "VMECJAXTransportAdmissionPolicy",
    "build_transport_admission_report",
    "candidate_transport_metric",
    "select_admitted_transport_candidate",
]
