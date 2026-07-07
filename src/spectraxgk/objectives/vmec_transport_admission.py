"""VMEC-JAX transport candidate admission policies and metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import pi
from typing import Any, cast

import numpy as np

# ---- policy dataclasses ----
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
    maximum_combined_sem_rel: float = 0.25
    minimum_replicate_count: int = 3
    minimum_surface_count: int = 3
    minimum_alpha_count: int = 2
    minimum_ky_count: int = 3
    minimum_sample_count: int = 12
    recommended_surfaces: tuple[float, ...] = (0.45, 0.64, 0.78)
    recommended_alphas: tuple[float, ...] = (0.0, pi / 4.0)
    recommended_ky_values: tuple[float, ...] = (0.10, 0.30, 0.50)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "minimum_relative_reduction": float(self.minimum_relative_reduction),
            "minimum_uncertainty_z_score": float(self.minimum_uncertainty_z_score),
            "maximum_combined_sem_rel": float(self.maximum_combined_sem_rel),
            "minimum_replicate_count": int(self.minimum_replicate_count),
            "minimum_surface_count": int(self.minimum_surface_count),
            "minimum_alpha_count": int(self.minimum_alpha_count),
            "minimum_ky_count": int(self.minimum_ky_count),
            "minimum_sample_count": int(self.minimum_sample_count),
            "recommended_surfaces": [float(item) for item in self.recommended_surfaces],
            "recommended_alphas": [float(item) for item in self.recommended_alphas],
            "recommended_ky_values": [float(item) for item in self.recommended_ky_values],
        }


@dataclass(frozen=True)
class VMECJAXReducedPrelaunchPolicy:
    """Fail-closed reduced-objective gate before expensive nonlinear audits."""

    metric_key: str = "nonlinear_window_heat_flux"
    minimum_relative_reduction: float = 0.04
    failed_reference_safety_factor: float = 1.5
    require_sample_coverage: bool = True
    maximum_cross_sample_sem_rel: float = 0.35

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "metric_key": str(self.metric_key),
            "minimum_relative_reduction": float(self.minimum_relative_reduction),
            "failed_reference_safety_factor": float(self.failed_reference_safety_factor),
            "require_sample_coverage": bool(self.require_sample_coverage),
            "maximum_cross_sample_sem_rel": float(self.maximum_cross_sample_sem_rel),
        }


@dataclass(frozen=True)
class VMECJAXNonlinearCampaignPolicy:
    """Admission limits for launching the next nonlinear optimizer campaign.

    This gate sits between a reduced candidate screen and a broader optimizer
    campaign.  Passing it means the next campaign is worth launching; it does
    not promote a production nonlinear turbulent-flux optimization claim.
    """

    minimum_landscape_relative_reduction: float = 0.10
    minimum_landscape_uncertainty_z_score: float = 3.0
    maximum_landscape_sem_rel: float = 0.05
    minimum_landscape_replicate_count: int = 3
    require_reduced_prelaunch_passed: bool = True
    require_reduced_cross_sample_gate: bool = True
    require_landscape_admission_passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "minimum_landscape_relative_reduction": float(
                self.minimum_landscape_relative_reduction
            ),
            "minimum_landscape_uncertainty_z_score": float(
                self.minimum_landscape_uncertainty_z_score
            ),
            "maximum_landscape_sem_rel": float(self.maximum_landscape_sem_rel),
            "minimum_landscape_replicate_count": int(
                self.minimum_landscape_replicate_count
            ),
            "require_reduced_prelaunch_passed": bool(
                self.require_reduced_prelaunch_passed
            ),
            "require_reduced_cross_sample_gate": bool(
                self.require_reduced_cross_sample_gate
            ),
            "require_landscape_admission_passed": bool(
                self.require_landscape_admission_passed
            ),
        }



# ---- sample coverage and metric helpers ----
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


# ---- candidate selection reports ----
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




__all__ = [
    "DEFAULT_TRANSPORT_METRIC_KEYS",
    "VMECJAXNonlinearAuditPolicy",
    "VMECJAXNonlinearCampaignPolicy",
    "VMECJAXReducedPrelaunchPolicy",
    "VMECJAXTransportAdmissionPolicy",
    "build_transport_admission_report",
    "candidate_transport_metric",
    "select_admitted_transport_candidate",
    "transport_objective_sample_summary",
]
