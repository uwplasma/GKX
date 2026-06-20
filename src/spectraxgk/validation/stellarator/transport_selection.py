"""Transport-candidate selection and admission reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from spectraxgk.validation.stellarator.transport_policies import (
    VMECJAXTransportAdmissionPolicy,
    _finite_float_or_none,
)
from spectraxgk.validation.stellarator.transport_samples import candidate_transport_metric

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


@dataclass(frozen=True)
class _BaselineTransportState:
    item: dict[str, Any] | None
    metric: dict[str, Any] | None
    metric_value: float | None
    physical_ok: bool


def _annotated_transport_candidate(
    raw: Mapping[str, Any],
    policy: VMECJAXTransportAdmissionPolicy,
) -> dict[str, Any]:
    item = dict(raw)
    physical_blockers = _physical_gate_blockers(item, policy)
    item["transport_metric"] = candidate_transport_metric(item, metric_keys=policy.metric_keys)
    item["physical_gate_blockers"] = physical_blockers
    item["admission_blockers"] = list(physical_blockers)
    item["relative_transport_improvement"] = None
    item["admitted_for_transport_optimization"] = False
    item["admitted_for_long_window_nonlinear_audit"] = False
    return item


def _annotated_transport_candidates(
    summaries: Sequence[Mapping[str, Any]],
    policy: VMECJAXTransportAdmissionPolicy,
) -> list[dict[str, Any]]:
    return [_annotated_transport_candidate(raw, policy) for raw in summaries]


def _select_transport_baseline(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    explicit = next((item for item in candidates if bool(item.get("baseline"))), None)
    if explicit is not None:
        return explicit
    return next((item for item in candidates if item.get("transport_weight") is None), None)


def _baseline_transport_state(
    baseline: dict[str, Any] | None,
) -> _BaselineTransportState:
    metric = cast(dict[str, Any], baseline.get("transport_metric")) if baseline else None
    metric_value = (
        _finite_float_or_none(metric.get("value"))
        if isinstance(metric, Mapping)
        else None
    )
    physical_ok = baseline is not None and not baseline.get("physical_gate_blockers")
    if baseline is not None:
        baseline["admitted_for_long_window_nonlinear_audit"] = bool(physical_ok)
    return _BaselineTransportState(
        item=baseline,
        metric=metric,
        metric_value=metric_value,
        physical_ok=bool(physical_ok),
    )


def _candidate_metric_value(item: Mapping[str, Any]) -> float | None:
    metric = cast(dict[str, Any], item["transport_metric"])
    return _finite_float_or_none(metric.get("value"))


def _append_transport_improvement_gate(
    item: dict[str, Any],
    *,
    candidate_metric_value: float | None,
    baseline_metric_value: float | None,
    policy: VMECJAXTransportAdmissionPolicy,
) -> None:
    if candidate_metric_value is not None and baseline_metric_value is not None:
        improvement = _relative_improvement(
            baseline_metric_value,
            candidate_metric_value,
            lower_is_better=bool(policy.lower_is_better),
        )
        item["relative_transport_improvement"] = improvement
        if improvement < float(policy.minimum_relative_improvement):
            item["admission_blockers"].append("insufficient_transport_improvement")


def _evaluate_transport_admission(
    item: dict[str, Any],
    *,
    baseline: _BaselineTransportState,
    policy: VMECJAXTransportAdmissionPolicy,
) -> bool:
    if bool(item.get("baseline")):
        return False
    if item.get("transport_weight") is None:
        item["admission_blockers"].append("missing_transport_weight")

    candidate_metric_value = _candidate_metric_value(item)
    if candidate_metric_value is None:
        item["admission_blockers"].append("missing_transport_metric")
    if baseline.metric_value is None:
        item["admission_blockers"].append("missing_baseline_transport_metric")
    _append_transport_improvement_gate(
        item,
        candidate_metric_value=candidate_metric_value,
        baseline_metric_value=baseline.metric_value,
        policy=policy,
    )

    admitted = not item["admission_blockers"]
    item["admitted_for_transport_optimization"] = admitted
    item["admitted_for_long_window_nonlinear_audit"] = bool(admitted)
    return bool(admitted)


def _admitted_transport_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    baseline: _BaselineTransportState,
    policy: VMECJAXTransportAdmissionPolicy,
) -> list[dict[str, Any]]:
    admitted: list[dict[str, Any]] = []
    for item in candidates:
        if _evaluate_transport_admission(item, baseline=baseline, policy=policy):
            admitted.append(item)
    return admitted


def _promoted_transport_candidate(
    *,
    admitted_transport: Sequence[dict[str, Any]],
    baseline: _BaselineTransportState,
    policy: VMECJAXTransportAdmissionPolicy,
) -> dict[str, Any] | None:
    if admitted_transport:
        return max(
            admitted_transport,
            key=lambda item: (
                float(item.get("transport_weight") or 0.0),
                float(item.get("relative_transport_improvement") or 0.0),
            ),
        )
    if bool(policy.allow_baseline_fallback) and baseline.item is not None and baseline.physical_ok:
        return baseline.item
    return None


def _transport_next_action(admitted_transport: Sequence[dict[str, Any]]) -> str:
    if admitted_transport:
        return "launch matched long-window nonlinear audits for the admitted transport candidate"
    return (
        "no transport candidate both preserved physical gates and improved the transport metric; "
        "keep the QA-only baseline and use a constraint-preserving projection/admission method"
    )


def _transport_admission_payload(
    *,
    policy: VMECJAXTransportAdmissionPolicy,
    candidates: Sequence[dict[str, Any]],
    baseline: _BaselineTransportState,
    admitted_transport: Sequence[dict[str, Any]],
    promoted: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "kind": "vmec_jax_transport_admission_report",
        "policy": policy.to_dict(),
        "baseline_label": None if baseline.item is None else baseline.item.get("label"),
        "baseline_transport_metric": baseline.metric,
        "candidates": list(candidates),
        "admitted_transport_candidates": [
            item.get("label") for item in admitted_transport if item.get("label") is not None
        ],
        "transport_candidate_admitted": bool(admitted_transport),
        "promoted_candidate": promoted,
        "passed": promoted is not None,
        "next_action": _transport_next_action(admitted_transport),
    }


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
    annotated = _annotated_transport_candidates(summaries, policy)
    baseline = _baseline_transport_state(_select_transport_baseline(annotated))
    admitted_transport = _admitted_transport_candidates(
        annotated,
        baseline=baseline,
        policy=policy,
    )
    promoted = _promoted_transport_candidate(
        admitted_transport=admitted_transport,
        baseline=baseline,
        policy=policy,
    )
    return _transport_admission_payload(
        policy=policy,
        candidates=annotated,
        baseline=baseline,
        admitted_transport=admitted_transport,
        promoted=promoted,
    )


def select_admitted_transport_candidate(
    summaries: Sequence[Mapping[str, Any]],
    *,
    policy: VMECJAXTransportAdmissionPolicy | None = None,
) -> dict[str, Any] | None:
    """Return the promoted candidate from :func:`build_transport_admission_report`."""

    report = build_transport_admission_report(summaries, policy=policy)
    promoted = report.get("promoted_candidate")
    return dict(promoted) if isinstance(promoted, Mapping) else None



__all__ = ["build_transport_admission_report", "select_admitted_transport_candidate"]
