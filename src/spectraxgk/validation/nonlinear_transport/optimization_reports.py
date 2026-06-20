"""Replicated nonlinear transport report extractors for optimization gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from spectraxgk.validation.nonlinear_transport.optimization_policy import (
    ProductionNonlinearOptimizationGuardConfig,
    _artifact_passed,
    _claim_text,
    _ensemble_variant_provenance_report,
    _finite_float,
)


@dataclass(frozen=True)
class _MatchedTransportContext:
    comparison: Mapping[str, Any]
    statistics: Mapping[str, Any]
    selected: Mapping[str, Any]
    baseline: Mapping[str, Any]
    optimized: Mapping[str, Any]
    strict_baseline: Mapping[str, Any]
    strict_candidate: Mapping[str, Any]
    named_gate_passed: dict[str, bool]


@dataclass(frozen=True)
class _MatchedTransportMetrics:
    relative_reduction: float | None
    uncertainty_sigma: float | None


@dataclass(frozen=True)
class _MatchedTransportFlags:
    passed: bool
    baseline_qualified: bool
    optimized_qualified: bool
    selected_closed: bool
    reduction_ok: bool
    uncertainty_ok: bool


def _payload_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _named_gate_status(payload: Mapping[str, Any]) -> dict[str, bool]:
    gates = payload.get("gates")
    gate_rows = (
        [row for row in gates if isinstance(row, Mapping)]
        if isinstance(gates, Sequence)
        else []
    )
    return {str(row.get("metric")): bool(row.get("passed", False)) for row in gate_rows}


def _matched_transport_context(payload: Mapping[str, Any]) -> _MatchedTransportContext:
    return _MatchedTransportContext(
        comparison=_payload_mapping(payload, "comparison"),
        statistics=_payload_mapping(payload, "statistics"),
        selected=_payload_mapping(payload, "selected_optimized_audit"),
        baseline=_payload_mapping(payload, "baseline_ensemble"),
        optimized=_payload_mapping(payload, "optimized_ensemble"),
        strict_baseline=_payload_mapping(payload, "baseline"),
        strict_candidate=_payload_mapping(payload, "candidate"),
        named_gate_passed=_named_gate_status(payload),
    )


def replicated_transport_ensemble_report(
    path: str,
    payload: Mapping[str, Any],
    *,
    config: ProductionNonlinearOptimizationGuardConfig | None = None,
) -> dict[str, Any]:
    """Return quality metadata for a long-window replicated transport ensemble."""

    cfg = config or ProductionNonlinearOptimizationGuardConfig()
    cfg.validate()
    stats = payload.get("statistics")
    stats_map: Mapping[str, Any] = stats if isinstance(stats, Mapping) else {}
    kind = str(payload.get("kind", "")).strip().lower()
    claim = _claim_text(payload)
    n_reports = int(_finite_float(stats_map.get("n_reports")) or 0)
    mean_rel_spread = _finite_float(stats_map.get("mean_rel_spread"))
    combined_sem_rel = _finite_float(stats_map.get("combined_sem_rel"))
    ensemble_mean = _finite_float(stats_map.get("ensemble_mean"))
    is_ensemble = kind == "nonlinear_window_ensemble_report"
    passed = _artifact_passed(payload)
    claim_scoped = (
        "replicated_nonlinear_window" in claim and "not_simulation_claim" in claim
    )
    finite_mean = ensemble_mean is not None and abs(ensemble_mean) >= float(
        cfg.value_floor
    )
    spread_ok = mean_rel_spread is not None and mean_rel_spread <= float(
        cfg.max_mean_rel_spread
    )
    sem_ok = combined_sem_rel is not None and combined_sem_rel <= float(
        cfg.max_combined_sem_rel
    )
    report_count_ok = n_reports >= int(cfg.min_reports_per_ensemble)
    provenance = _ensemble_variant_provenance_report(payload, config=cfg)
    provenance_ok = (not cfg.require_seed_timestep_provenance) or bool(
        provenance["passed"]
    )
    qualifies = bool(
        is_ensemble
        and passed
        and claim_scoped
        and finite_mean
        and spread_ok
        and sem_ok
        and report_count_ok
        and provenance_ok
    )
    return {
        "path": path,
        "kind": str(payload.get("kind", "")),
        "case": str(payload.get("case", "")),
        "claim_level": str(payload.get("claim_level", "")),
        "passed": passed,
        "is_nonlinear_window_ensemble": is_ensemble,
        "claim_scoped_as_replicated_holdout": claim_scoped,
        "n_reports": n_reports,
        "ensemble_mean": ensemble_mean,
        "mean_rel_spread": mean_rel_spread,
        "combined_sem_rel": combined_sem_rel,
        "finite_transport_mean": finite_mean,
        "mean_rel_spread_ok": spread_ok,
        "combined_sem_rel_ok": sem_ok,
        "report_count_ok": report_count_ok,
        "seed_timestep_provenance": provenance,
        "seed_timestep_provenance_ok": provenance_ok,
        "qualifies_as_long_post_transient_replicate": qualifies,
    }


def optimized_equilibrium_transport_report(
    path: str,
    payload: Mapping[str, Any],
    *,
    config: ProductionNonlinearOptimizationGuardConfig | None = None,
) -> dict[str, Any]:
    """Return whether an artifact can promote optimized-equilibrium transport."""

    row = replicated_transport_ensemble_report(path, payload, config=config)
    text = _claim_text(payload) + " " + str(path).lower()
    optimized_marker = (
        "optimized_equilibrium" in text
        or "optimized-equilibrium" in text
        or "post_optimization" in text
        or "post-optimization" in text
        or "growth_from_strict_baseline" in text
        or "quasilinear_from_strict_baseline" in text
        or "nonlinear_window_from_strict_baseline" in text
    )
    row["optimized_equilibrium_marker"] = optimized_marker
    row["qualifies_for_production_optimization"] = bool(
        row["qualifies_as_long_post_transient_replicate"] and optimized_marker
    )
    return row


def _matched_transport_metrics(
    context: _MatchedTransportContext,
) -> _MatchedTransportMetrics:
    relative_reduction = _finite_float(
        context.comparison.get("relative_reduction")
        if "relative_reduction" in context.comparison
        else context.statistics.get("relative_reduction")
    )
    uncertainty_sigma = _finite_float(
        context.comparison.get("uncertainty_separation_sigma")
        or context.comparison.get("uncertainty_z_score")
        or context.statistics.get("uncertainty_separation_sigma")
        or context.statistics.get("uncertainty_z_score")
    )
    return _MatchedTransportMetrics(
        relative_reduction=relative_reduction,
        uncertainty_sigma=uncertainty_sigma,
    )


def _matched_transport_flags(
    *,
    payload: Mapping[str, Any],
    context: _MatchedTransportContext,
    metrics: _MatchedTransportMetrics,
    config: ProductionNonlinearOptimizationGuardConfig,
) -> _MatchedTransportFlags:
    named_gate_passed = context.named_gate_passed
    baseline_qualified = bool(
        context.baseline.get("qualifies", False)
        or context.strict_baseline.get("passed", False)
        or context.strict_baseline.get("raw_passed", False)
        or named_gate_passed.get("baseline_replicated_ensemble_qualified", False)
        or named_gate_passed.get("baseline_ensemble_passed", False)
    )
    optimized_qualified = bool(
        context.optimized.get("qualifies", False)
        or context.strict_candidate.get("passed", False)
        or context.strict_candidate.get("raw_passed", False)
        or named_gate_passed.get("optimized_replicated_ensemble_qualified", False)
        or named_gate_passed.get("candidate_ensemble_passed", False)
    )
    selected_closed = bool(
        context.selected.get("passed", False)
        or named_gate_passed.get("selected_optimized_equilibrium_audit", False)
        or (
            context.strict_baseline
            and context.strict_candidate
            and baseline_qualified
            and optimized_qualified
        )
    )
    reduction_ok = (
        metrics.relative_reduction is not None
        and metrics.relative_reduction
        >= float(config.min_matched_optimized_relative_reduction)
    )
    uncertainty_ok = (
        metrics.uncertainty_sigma is not None
        and metrics.uncertainty_sigma
        >= float(config.min_matched_optimized_uncertainty_sigma)
    )
    return _MatchedTransportFlags(
        passed=_artifact_passed(payload),
        baseline_qualified=baseline_qualified,
        optimized_qualified=optimized_qualified,
        selected_closed=selected_closed,
        reduction_ok=reduction_ok,
        uncertainty_ok=uncertainty_ok,
    )


def _matched_transport_blockers(flags: _MatchedTransportFlags) -> list[str]:
    blockers: list[str] = []
    if not flags.passed:
        blockers.append("matched_optimized_audit_failed")
    if not flags.baseline_qualified:
        blockers.append("baseline_replicated_ensemble_not_qualified")
    if not flags.optimized_qualified:
        blockers.append("optimized_replicated_ensemble_not_qualified")
    if not flags.selected_closed:
        blockers.append("selected_optimized_audit_not_closed")
    if not flags.reduction_ok:
        blockers.append("insufficient_matched_optimized_reduction")
    if not flags.uncertainty_ok:
        blockers.append("insufficient_matched_optimized_uncertainty_separation")
    return blockers


def matched_optimized_transport_report(
    path: str,
    payload: Mapping[str, Any],
    *,
    config: ProductionNonlinearOptimizationGuardConfig | None = None,
) -> dict[str, Any]:
    """Return whether a matched baseline-to-optimized audit promotes transport."""

    cfg = config or ProductionNonlinearOptimizationGuardConfig()
    cfg.validate()
    context = _matched_transport_context(payload)
    metrics = _matched_transport_metrics(context)
    flags = _matched_transport_flags(
        payload=payload,
        context=context,
        metrics=metrics,
        config=cfg,
    )
    blockers = _matched_transport_blockers(flags)
    return {
        "path": path,
        "kind": str(payload.get("kind", "")),
        "case": str(payload.get("case", "")),
        "claim_level": str(payload.get("claim_level", "")),
        "passed": flags.passed,
        "baseline_ensemble_qualified": flags.baseline_qualified,
        "optimized_ensemble_qualified": flags.optimized_qualified,
        "selected_optimized_audit_closed": flags.selected_closed,
        "relative_reduction": metrics.relative_reduction,
        "uncertainty_separation_sigma": metrics.uncertainty_sigma,
        "relative_reduction_ok": flags.reduction_ok,
        "uncertainty_separation_ok": flags.uncertainty_ok,
        "blockers": blockers,
        "qualifies_for_production_optimization": not blockers,
    }


__all__ = [
    "matched_optimized_transport_report",
    "optimized_equilibrium_transport_report",
    "replicated_transport_ensemble_report",
]
