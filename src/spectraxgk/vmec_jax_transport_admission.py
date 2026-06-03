"""Admission policy for VMEC-JAX transport-optimization candidates.

This module is deliberately small and independent of VMEC-JAX internals.  It
operates on JSON-safe candidate summaries produced by the optimization and
ladder tools, then answers the release-critical question: which solved WOUT, if
any, is physically admissible for expensive long-window nonlinear audits?
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import pi
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


@dataclass(frozen=True)
class VMECJAXNonlinearAuditPolicy:
    """Policy for promoting or redesigning VMEC-JAX transport candidates.

    Reduced growth/quasilinear/nonlinear-window objectives are useful only if
    they transfer to late-window nonlinear transport.  This policy encodes the
    minimum replicated-audit evidence and sample coverage required before a
    candidate can be promoted beyond local reduced-metric admission.
    """

    minimum_relative_reduction: float = 0.02
    minimum_uncertainty_z_score: float = 1.0
    minimum_surface_count: int = 3
    minimum_alpha_count: int = 2
    minimum_ky_count: int = 3
    minimum_sample_count: int = 12
    recommended_surfaces: tuple[float, ...] = (0.45, 0.64, 0.78)
    recommended_alphas: tuple[float, ...] = (0.0, pi / 4.0)
    recommended_ky_values: tuple[float, ...] = (0.190, 0.300, 0.476)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "minimum_relative_reduction": float(self.minimum_relative_reduction),
            "minimum_uncertainty_z_score": float(self.minimum_uncertainty_z_score),
            "minimum_surface_count": int(self.minimum_surface_count),
            "minimum_alpha_count": int(self.minimum_alpha_count),
            "minimum_ky_count": int(self.minimum_ky_count),
            "minimum_sample_count": int(self.minimum_sample_count),
            "recommended_surfaces": [float(item) for item in self.recommended_surfaces],
            "recommended_alphas": [float(item) for item in self.recommended_alphas],
            "recommended_ky_values": [float(item) for item in self.recommended_ky_values],
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


def build_nonlinear_audit_redesign_report(
    matched_comparison: Mapping[str, Any],
    *,
    objective_sample_set: Any = None,
    policy: VMECJAXNonlinearAuditPolicy | None = None,
) -> dict[str, Any]:
    """Decide whether a matched nonlinear audit promotes or redesigns a candidate.

    This is the fail-closed bridge between reduced VMEC-JAX transport admission
    and expensive long-window nonlinear evidence.  A candidate is promoted only
    if the matched replicated nonlinear comparison passes, has a positive
    uncertainty-separated reduction, and the reduced objective used enough
    surface/field-line/``k_y`` samples to avoid a single-point overfit.
    """

    policy = policy or VMECJAXNonlinearAuditPolicy()
    stats = matched_comparison.get("statistics", {})
    if not isinstance(stats, Mapping):
        stats = {}
    relative_reduction = _finite_float_or_none(stats.get("relative_reduction"))
    z_score = _finite_float_or_none(stats.get("uncertainty_z_score"))
    baseline = matched_comparison.get("baseline", {})
    candidate = matched_comparison.get("candidate", {})
    baseline_passed = bool(baseline.get("passed", False)) if isinstance(baseline, Mapping) else False
    candidate_passed = bool(candidate.get("passed", False)) if isinstance(candidate, Mapping) else False
    comparison_passed = bool(matched_comparison.get("passed", False))
    nonlinear_blockers: list[str] = []
    if not baseline_passed:
        nonlinear_blockers.append("baseline_ensemble_failed")
    if not candidate_passed:
        nonlinear_blockers.append("candidate_ensemble_failed")
    if relative_reduction is None:
        nonlinear_blockers.append("missing_relative_reduction")
    elif relative_reduction < float(policy.minimum_relative_reduction):
        nonlinear_blockers.append("insufficient_matched_reduction")
    if z_score is None:
        nonlinear_blockers.append("missing_uncertainty_z_score")
    elif z_score < float(policy.minimum_uncertainty_z_score):
        nonlinear_blockers.append("insufficient_uncertainty_separation")
    if not comparison_passed:
        nonlinear_blockers.append("matched_comparison_not_passed")

    sample_summary = transport_objective_sample_summary(objective_sample_set, policy=policy)
    all_blockers = nonlinear_blockers + list(sample_summary["blockers"])
    promoted = not all_blockers
    recommended = {
        "surfaces": [float(item) for item in policy.recommended_surfaces],
        "alphas": [float(item) for item in policy.recommended_alphas],
        "ky_values": [float(item) for item in policy.recommended_ky_values],
        "sample_count": (
            len(policy.recommended_surfaces)
            * len(policy.recommended_alphas)
            * len(policy.recommended_ky_values)
        ),
    }
    return {
        "kind": "vmec_jax_nonlinear_transport_audit_redesign_report",
        "policy": policy.to_dict(),
        "matched_comparison_case": matched_comparison.get("case"),
        "matched_comparison_passed": comparison_passed,
        "nonlinear_audit_promoted": promoted,
        "requires_objective_redesign": not promoted,
        "nonlinear_audit_blockers": nonlinear_blockers,
        "objective_sample_summary": sample_summary,
        "blockers": all_blockers,
        "recommended_sample_set": recommended,
        "gates": [
            {
                "metric": "matched_replicated_late_window_reduction",
                "passed": "insufficient_matched_reduction" not in nonlinear_blockers
                and "missing_relative_reduction" not in nonlinear_blockers,
                "value": relative_reduction,
                "threshold": float(policy.minimum_relative_reduction),
            },
            {
                "metric": "uncertainty_separated_reduction",
                "passed": "insufficient_uncertainty_separation" not in nonlinear_blockers
                and "missing_uncertainty_z_score" not in nonlinear_blockers,
                "value": z_score,
                "threshold": float(policy.minimum_uncertainty_z_score),
            },
            {
                "metric": "multi_sample_objective_coverage",
                "passed": bool(sample_summary["passed"]),
                "value": int(sample_summary["sample_count"]),
                "threshold": int(policy.minimum_sample_count),
            },
        ],
        "next_action": (
            "candidate may be used as nonlinear turbulent-flux optimization evidence"
            if promoted
            else (
                "redesign the reduced transport objective with the recommended multi-surface, "
                "multi-field-line, multi-ky sample set; rerun projected admission; then repeat "
                "the matched long-window nonlinear audit"
            )
        ),
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
    "VMECJAXNonlinearAuditPolicy",
    "VMECJAXTransportAdmissionPolicy",
    "build_nonlinear_audit_redesign_report",
    "build_transport_admission_report",
    "candidate_transport_metric",
    "select_admitted_transport_candidate",
    "transport_objective_sample_summary",
]
