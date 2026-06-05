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


def _ensemble_statistics(ensemble: Mapping[str, Any]) -> dict[str, Any]:
    stats = ensemble.get("statistics", {})
    if not isinstance(stats, Mapping):
        stats = {}
    return {
        "passed": bool(ensemble.get("passed", False)),
        "ensemble_mean": _finite_float_or_none(stats.get("ensemble_mean")),
        "combined_sem": _finite_float_or_none(stats.get("combined_sem")),
        "combined_sem_rel": _finite_float_or_none(stats.get("combined_sem_rel")),
        "n_reports": int(stats.get("n_reports", 0) or 0),
        "case": ensemble.get("case"),
    }


def _sample_statistics_summary(
    sample_statistics: Mapping[str, Any] | None,
    *,
    policy: VMECJAXReducedPrelaunchPolicy,
    role: str,
) -> dict[str, Any]:
    """Return a gate row for deterministic reduced-metric sample dispersion."""

    if not isinstance(sample_statistics, Mapping):
        return {
            "role": role,
            "available": False,
            "weighted_mean": None,
            "weighted_standard_error": None,
            "cross_sample_sem_rel": None,
            "passed": None,
            "blockers": [],
        }
    weighted_mean = _finite_float_or_none(sample_statistics.get("weighted_mean"))
    weighted_sem = _finite_float_or_none(
        sample_statistics.get("weighted_standard_error")
    )
    blockers: list[str] = []
    sem_rel = None
    if weighted_mean is None:
        blockers.append(f"{role}_missing_cross_sample_weighted_mean")
    if weighted_sem is None:
        blockers.append(f"{role}_missing_cross_sample_weighted_sem")
    if weighted_mean is not None and weighted_sem is not None:
        sem_rel = float(abs(weighted_sem) / max(abs(weighted_mean), 1.0e-300))
        if sem_rel > float(policy.maximum_cross_sample_sem_rel):
            blockers.append(f"{role}_cross_sample_sem_rel_too_large")
    return {
        "role": role,
        "available": True,
        "weighted_mean": weighted_mean,
        "weighted_standard_error": weighted_sem,
        "cross_sample_sem_rel": sem_rel,
        "passed": not blockers,
        "blockers": blockers,
    }


def _ensemble_blockers(
    stats: Mapping[str, Any],
    *,
    role: str,
    policy: VMECJAXNonlinearAuditPolicy,
) -> list[str]:
    blockers: list[str] = []
    if not bool(stats.get("passed")):
        blockers.append(f"{role}_ensemble_failed")
    if stats.get("ensemble_mean") is None:
        blockers.append(f"{role}_missing_ensemble_mean")
    if stats.get("combined_sem") is None:
        blockers.append(f"{role}_missing_combined_sem")
    sem_rel = stats.get("combined_sem_rel")
    if sem_rel is None:
        blockers.append(f"{role}_missing_combined_sem_rel")
    elif float(sem_rel) > float(policy.maximum_combined_sem_rel):
        blockers.append(f"{role}_combined_sem_rel_too_large")
    if int(stats.get("n_reports", 0) or 0) < int(policy.minimum_replicate_count):
        blockers.append(f"{role}_insufficient_replicates")
    return blockers


def build_nonlinear_landscape_admission_report(
    baseline_ensemble: Mapping[str, Any],
    candidate_ensembles: Sequence[Mapping[str, Any]],
    *,
    candidate_labels: Sequence[str] | None = None,
    policy: VMECJAXNonlinearAuditPolicy | None = None,
) -> dict[str, Any]:
    """Select an uncertainty-separated nonlinear candidate from a landscape.

    This gate is for boundary-coefficient or line-search landscapes where a
    small number of selected points have replicated late-window nonlinear
    ensembles.  It does not validate multi-coefficient global optimization by
    itself; it only answers whether any supplied candidate has a statistically
    resolved lower heat flux than the supplied baseline ensemble.
    """

    policy = policy or VMECJAXNonlinearAuditPolicy()
    labels = (
        tuple(str(item) for item in candidate_labels)
        if candidate_labels is not None
        else tuple(f"candidate_{index}" for index, _ in enumerate(candidate_ensembles))
    )
    if len(labels) != len(candidate_ensembles):
        raise ValueError("candidate_labels must have the same length as candidate_ensembles")
    baseline_stats = _ensemble_statistics(baseline_ensemble)
    baseline_blockers = _ensemble_blockers(baseline_stats, role="baseline", policy=policy)
    rows: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    baseline_mean = cast(float | None, baseline_stats.get("ensemble_mean"))
    baseline_sem = cast(float | None, baseline_stats.get("combined_sem"))
    for label, ensemble in zip(labels, candidate_ensembles, strict=True):
        stats = _ensemble_statistics(ensemble)
        blockers = list(baseline_blockers)
        blockers.extend(_ensemble_blockers(stats, role="candidate", policy=policy))
        candidate_mean = cast(float | None, stats.get("ensemble_mean"))
        candidate_sem = cast(float | None, stats.get("combined_sem"))
        relative_reduction = None
        uncertainty_z_score = None
        if baseline_mean is not None and candidate_mean is not None:
            relative_reduction = float((baseline_mean - candidate_mean) / max(abs(baseline_mean), 1.0e-300))
            if relative_reduction < float(policy.minimum_relative_reduction):
                blockers.append("insufficient_relative_reduction")
        else:
            blockers.append("missing_relative_reduction")
        if baseline_mean is not None and candidate_mean is not None and baseline_sem is not None and candidate_sem is not None:
            combined_uncertainty = float(np.hypot(baseline_sem, candidate_sem))
            uncertainty_z_score = float((baseline_mean - candidate_mean) / max(combined_uncertainty, 1.0e-300))
            if uncertainty_z_score < float(policy.minimum_uncertainty_z_score):
                blockers.append("insufficient_uncertainty_separation")
        else:
            blockers.append("missing_uncertainty_z_score")
        row = {
            "label": label,
            "case": stats.get("case"),
            "ensemble_mean": candidate_mean,
            "combined_sem": candidate_sem,
            "combined_sem_rel": stats.get("combined_sem_rel"),
            "n_reports": stats.get("n_reports"),
            "relative_reduction": relative_reduction,
            "uncertainty_z_score": uncertainty_z_score,
            "admission_blockers": blockers,
            "admitted": not blockers,
        }
        rows.append(row)
        if not blockers:
            admitted.append(row)
    selected = None
    if admitted:
        selected = max(
            admitted,
            key=lambda row: (
                float(row.get("relative_reduction") or 0.0),
                float(row.get("uncertainty_z_score") or 0.0),
            ),
        )
    return {
        "kind": "nonlinear_landscape_admission_report",
        "claim_scope": (
            "selected replicated nonlinear landscape admission; not a multi-coefficient "
            "or multi-flux-tube turbulent-optimization claim"
        ),
        "policy": policy.to_dict(),
        "baseline": baseline_stats,
        "candidates": rows,
        "selected_candidate": selected,
        "passed": selected is not None,
        "next_action": (
            "use selected direction for the next uncertainty-aware optimizer admission step"
            if selected is not None
            else "do not launch a broader optimizer from this landscape without more resolved nonlinear evidence"
        ),
    }


def build_reduced_nonlinear_audit_prelaunch_report(
    *,
    baseline_metric: float,
    candidate_metric: float,
    objective_sample_set: Any = None,
    baseline_sample_statistics: Mapping[str, Any] | None = None,
    candidate_sample_statistics: Mapping[str, Any] | None = None,
    failed_reference_relative_reduction: float | None = None,
    policy: VMECJAXReducedPrelaunchPolicy | None = None,
    nonlinear_policy: VMECJAXNonlinearAuditPolicy | None = None,
) -> dict[str, Any]:
    """Gate reduced transport candidates before launching nonlinear audits.

    This is intentionally conservative.  A reduced nonlinear-window improvement
    should exceed both an absolute release threshold and, when available, a
    safety factor above a known failed-transfer reference before spending
    another long GPU campaign.
    """

    policy = policy or VMECJAXReducedPrelaunchPolicy()
    nonlinear_policy = nonlinear_policy or VMECJAXNonlinearAuditPolicy()
    baseline = _finite_float_or_none(baseline_metric)
    candidate = _finite_float_or_none(candidate_metric)
    failed_reference = _finite_float_or_none(failed_reference_relative_reduction)
    blockers: list[str] = []
    relative_reduction = None
    if baseline is None:
        blockers.append("missing_baseline_reduced_metric")
    if candidate is None:
        blockers.append("missing_candidate_reduced_metric")
    if baseline is not None and candidate is not None:
        relative_reduction = float((baseline - candidate) / max(abs(baseline), 1.0e-300))

    threshold = float(policy.minimum_relative_reduction)
    threshold_sources = {
        "policy_minimum_relative_reduction": float(policy.minimum_relative_reduction),
        "failed_reference_relative_reduction": failed_reference,
        "failed_reference_safety_factor": float(policy.failed_reference_safety_factor),
        "failed_reference_threshold": None,
    }
    if failed_reference is not None:
        failed_threshold = float(policy.failed_reference_safety_factor) * failed_reference
        threshold_sources["failed_reference_threshold"] = failed_threshold
        threshold = max(threshold, failed_threshold)

    if relative_reduction is None:
        blockers.append("missing_relative_reduced_reduction")
    elif relative_reduction < threshold:
        blockers.append("insufficient_reduced_margin_for_nonlinear_audit")

    sample_summary = transport_objective_sample_summary(
        objective_sample_set,
        policy=nonlinear_policy,
    )
    if bool(policy.require_sample_coverage) and not bool(sample_summary["passed"]):
        blockers.extend(str(item) for item in sample_summary["blockers"])
    cross_sample_rows = [
        _sample_statistics_summary(
            baseline_sample_statistics,
            policy=policy,
            role="baseline",
        ),
        _sample_statistics_summary(
            candidate_sample_statistics,
            policy=policy,
            role="candidate",
        ),
    ]
    cross_sample_available = all(bool(row["available"]) for row in cross_sample_rows)
    cross_sample_passed = (
        None
        if not cross_sample_available
        else all(bool(row["passed"]) for row in cross_sample_rows)
    )
    if cross_sample_available and cross_sample_passed is False:
        for row in cross_sample_rows:
            blockers.extend(str(item) for item in row["blockers"])

    return {
        "kind": "vmec_jax_reduced_nonlinear_audit_prelaunch_report",
        "claim_scope": (
            "reduced-objective prelaunch guard only; passing this gate permits a "
            "replicated nonlinear audit but does not promote a turbulence claim"
        ),
        "policy": policy.to_dict(),
        "nonlinear_policy": nonlinear_policy.to_dict(),
        "metric_key": str(policy.metric_key),
        "baseline_metric": baseline,
        "candidate_metric": candidate,
        "relative_reduced_reduction": relative_reduction,
        "required_relative_reduced_reduction": threshold,
        "threshold_sources": threshold_sources,
        "objective_sample_summary": sample_summary,
        "reduced_cross_sample_statistics": {
            "available": cross_sample_available,
            "passed": cross_sample_passed,
            "rows": cross_sample_rows,
            "claim_scope": (
                "deterministic spread over the reduced surface/field-line/ky "
                "objective grid; not stochastic nonlinear heat-flux uncertainty"
            ),
        },
        "passed": not blockers,
        "blockers": blockers,
        "gates": [
            {
                "metric": "reduced_margin_for_nonlinear_audit",
                "passed": "insufficient_reduced_margin_for_nonlinear_audit" not in blockers
                and "missing_relative_reduced_reduction" not in blockers,
                "value": relative_reduction,
                "threshold": threshold,
            },
            {
                "metric": "multi_sample_objective_coverage",
                "passed": bool(sample_summary["passed"]),
                "value": int(sample_summary["sample_count"]),
                "threshold": int(nonlinear_policy.minimum_sample_count),
            },
            {
                "metric": "reduced_cross_sample_dispersion",
                "passed": cross_sample_passed,
                "value": [
                    row["cross_sample_sem_rel"]
                    for row in cross_sample_rows
                    if row["cross_sample_sem_rel"] is not None
                ],
                "threshold": float(policy.maximum_cross_sample_sem_rel),
            },
        ],
        "next_action": (
            "launch replicated long-window nonlinear audit only with baseline/candidate ensembles"
            if not blockers
            else (
                "do not launch an expensive nonlinear audit; increase reduced-objective "
                "margin or broaden the objective before spending GPU time"
            )
        ),
    }


def build_nonlinear_campaign_admission_report(
    *,
    reduced_prelaunch_report: Mapping[str, Any],
    landscape_admission_report: Mapping[str, Any],
    policy: VMECJAXNonlinearCampaignPolicy | None = None,
) -> dict[str, Any]:
    """Gate the next nonlinear optimizer campaign from existing evidence.

    This report intentionally promotes only a *campaign launch*.  It requires a
    reduced prelaunch pass and an uncertainty-separated replicated nonlinear
    landscape point.  It does not convert that point into a general
    multi-coefficient turbulent-flux optimization result.
    """

    policy = policy or VMECJAXNonlinearCampaignPolicy()
    blockers: list[str] = []
    gates: list[dict[str, Any]] = []

    prelaunch_passed = bool(reduced_prelaunch_report.get("passed", False))
    gates.append(
        {
            "metric": "reduced_prelaunch_gate",
            "passed": prelaunch_passed,
            "detail": reduced_prelaunch_report.get("blockers", []),
        }
    )
    if bool(policy.require_reduced_prelaunch_passed) and not prelaunch_passed:
        blockers.append("reduced_prelaunch_gate_failed")

    sample_summary = reduced_prelaunch_report.get("objective_sample_summary")
    sample_passed = (
        bool(sample_summary.get("passed", False))
        if isinstance(sample_summary, Mapping)
        else False
    )
    gates.append(
        {
            "metric": "reduced_objective_sample_coverage",
            "passed": sample_passed,
            "value": (
                int(sample_summary["sample_count"])
                if isinstance(sample_summary, Mapping)
                and sample_summary.get("sample_count") is not None
                else None
            ),
            "detail": (
                sample_summary.get("blockers", [])
                if isinstance(sample_summary, Mapping)
                else "missing objective_sample_summary"
            ),
        }
    )
    if not sample_passed:
        blockers.append("reduced_objective_sample_coverage_failed")

    cross_sample = reduced_prelaunch_report.get("reduced_cross_sample_statistics")
    cross_sample_available = (
        bool(cross_sample.get("available", False))
        if isinstance(cross_sample, Mapping)
        else False
    )
    cross_sample_passed = (
        bool(cross_sample.get("passed", False))
        if isinstance(cross_sample, Mapping)
        and cross_sample.get("passed") is not None
        else None
    )
    gates.append(
        {
            "metric": "reduced_cross_sample_dispersion",
            "passed": cross_sample_passed,
            "detail": (
                cross_sample.get("rows", [])
                if isinstance(cross_sample, Mapping)
                else "missing reduced_cross_sample_statistics"
            ),
        }
    )
    if bool(policy.require_reduced_cross_sample_gate):
        if not cross_sample_available:
            blockers.append("reduced_cross_sample_statistics_missing")
        elif cross_sample_passed is not True:
            blockers.append("reduced_cross_sample_dispersion_failed")

    landscape_passed = bool(landscape_admission_report.get("passed", False))
    gates.append(
        {
            "metric": "replicated_landscape_admission",
            "passed": landscape_passed,
            "detail": landscape_admission_report.get("next_action"),
        }
    )
    if bool(policy.require_landscape_admission_passed) and not landscape_passed:
        blockers.append("replicated_landscape_admission_failed")

    selected = landscape_admission_report.get("selected_candidate")
    selected_map: Mapping[str, Any] = selected if isinstance(selected, Mapping) else {}
    if not selected_map:
        blockers.append("missing_selected_landscape_candidate")

    relative_reduction = _finite_float_or_none(selected_map.get("relative_reduction"))
    z_score = _finite_float_or_none(selected_map.get("uncertainty_z_score"))
    sem_rel = _finite_float_or_none(selected_map.get("combined_sem_rel"))
    n_reports = int(selected_map.get("n_reports", 0) or 0)
    candidate_gates = [
        (
            "landscape_relative_reduction",
            relative_reduction is not None
            and relative_reduction
            >= float(policy.minimum_landscape_relative_reduction),
            relative_reduction,
            float(policy.minimum_landscape_relative_reduction),
            "selected_landscape_reduction_too_small",
        ),
        (
            "landscape_uncertainty_separation",
            z_score is not None
            and z_score >= float(policy.minimum_landscape_uncertainty_z_score),
            z_score,
            float(policy.minimum_landscape_uncertainty_z_score),
            "selected_landscape_uncertainty_separation_too_small",
        ),
        (
            "landscape_candidate_sem_rel",
            sem_rel is not None and sem_rel <= float(policy.maximum_landscape_sem_rel),
            sem_rel,
            float(policy.maximum_landscape_sem_rel),
            "selected_landscape_sem_rel_too_large",
        ),
        (
            "landscape_candidate_replicates",
            n_reports >= int(policy.minimum_landscape_replicate_count),
            n_reports,
            int(policy.minimum_landscape_replicate_count),
            "selected_landscape_insufficient_replicates",
        ),
    ]
    for metric, passed, value, threshold, blocker in candidate_gates:
        gates.append(
            {
                "metric": metric,
                "passed": bool(passed),
                "value": value,
                "threshold": threshold,
            }
        )
        if not passed:
            blockers.append(blocker)

    admitted = not blockers
    return {
        "kind": "vmec_jax_nonlinear_campaign_admission_report",
        "claim_scope": (
            "next nonlinear optimizer-campaign admission only; not a production "
            "multi-coefficient turbulent-flux optimization claim"
        ),
        "policy": policy.to_dict(),
        "passed": admitted,
        "campaign_admitted": admitted,
        "blockers": blockers,
        "gates": gates,
        "selected_landscape_candidate": dict(selected_map) if selected_map else None,
        "next_action": (
            "launch a bounded multi-control optimizer campaign from the admitted landscape direction, "
            "with matched baseline/candidate t=[350,700] replicated nonlinear audits before promotion"
            if admitted
            else (
                "do not launch a broader nonlinear optimizer campaign; fix the reduced gate, "
                "cross-sample dispersion, or replicated landscape uncertainty first"
            )
        ),
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
    "VMECJAXNonlinearCampaignPolicy",
    "VMECJAXReducedPrelaunchPolicy",
    "VMECJAXTransportAdmissionPolicy",
    "build_nonlinear_landscape_admission_report",
    "build_nonlinear_campaign_admission_report",
    "build_nonlinear_audit_redesign_report",
    "build_reduced_nonlinear_audit_prelaunch_report",
    "build_transport_admission_report",
    "candidate_transport_metric",
    "select_admitted_transport_candidate",
    "transport_objective_sample_summary",
]
