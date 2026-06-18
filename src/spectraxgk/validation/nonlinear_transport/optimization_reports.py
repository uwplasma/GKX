"""Replicated nonlinear transport report extractors for optimization gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from spectraxgk.validation.nonlinear_transport.optimization_policy import (
    ProductionNonlinearOptimizationGuardConfig,
    _artifact_passed,
    _claim_text,
    _ensemble_variant_provenance_report,
    _finite_float,
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


def matched_optimized_transport_report(
    path: str,
    payload: Mapping[str, Any],
    *,
    config: ProductionNonlinearOptimizationGuardConfig | None = None,
) -> dict[str, Any]:
    """Return whether a matched baseline-to-optimized audit promotes transport."""

    cfg = config or ProductionNonlinearOptimizationGuardConfig()
    cfg.validate()
    comparison = payload.get("comparison")
    comparison_map: Mapping[str, Any] = (
        comparison if isinstance(comparison, Mapping) else {}
    )
    statistics = payload.get("statistics")
    statistics_map: Mapping[str, Any] = (
        statistics if isinstance(statistics, Mapping) else {}
    )
    selected = payload.get("selected_optimized_audit")
    selected_map: Mapping[str, Any] = selected if isinstance(selected, Mapping) else {}
    baseline = payload.get("baseline_ensemble")
    baseline_map: Mapping[str, Any] = baseline if isinstance(baseline, Mapping) else {}
    optimized = payload.get("optimized_ensemble")
    optimized_map: Mapping[str, Any] = (
        optimized if isinstance(optimized, Mapping) else {}
    )
    strict_baseline = payload.get("baseline")
    strict_baseline_map: Mapping[str, Any] = (
        strict_baseline if isinstance(strict_baseline, Mapping) else {}
    )
    strict_candidate = payload.get("candidate")
    strict_candidate_map: Mapping[str, Any] = (
        strict_candidate if isinstance(strict_candidate, Mapping) else {}
    )
    relative_reduction = _finite_float(
        comparison_map.get("relative_reduction")
        if "relative_reduction" in comparison_map
        else statistics_map.get("relative_reduction")
    )
    uncertainty_sigma = _finite_float(
        comparison_map.get("uncertainty_separation_sigma")
        or comparison_map.get("uncertainty_z_score")
        or statistics_map.get("uncertainty_separation_sigma")
        or statistics_map.get("uncertainty_z_score")
    )
    gates = payload.get("gates")
    gate_rows = (
        [row for row in gates if isinstance(row, Mapping)]
        if isinstance(gates, Sequence)
        else []
    )
    named_gate_passed = {
        str(row.get("metric")): bool(row.get("passed", False)) for row in gate_rows
    }
    passed = _artifact_passed(payload)
    baseline_qualified = bool(
        baseline_map.get("qualifies", False)
        or strict_baseline_map.get("passed", False)
        or strict_baseline_map.get("raw_passed", False)
        or named_gate_passed.get("baseline_replicated_ensemble_qualified", False)
        or named_gate_passed.get("baseline_ensemble_passed", False)
    )
    optimized_qualified = bool(
        optimized_map.get("qualifies", False)
        or strict_candidate_map.get("passed", False)
        or strict_candidate_map.get("raw_passed", False)
        or named_gate_passed.get("optimized_replicated_ensemble_qualified", False)
        or named_gate_passed.get("candidate_ensemble_passed", False)
    )
    selected_closed = bool(
        selected_map.get("passed", False)
        or named_gate_passed.get("selected_optimized_equilibrium_audit", False)
        or (
            strict_baseline_map
            and strict_candidate_map
            and baseline_qualified
            and optimized_qualified
        )
    )
    reduction_ok = relative_reduction is not None and relative_reduction >= float(
        cfg.min_matched_optimized_relative_reduction
    )
    uncertainty_ok = uncertainty_sigma is not None and uncertainty_sigma >= float(
        cfg.min_matched_optimized_uncertainty_sigma
    )
    blockers: list[str] = []
    if not passed:
        blockers.append("matched_optimized_audit_failed")
    if not baseline_qualified:
        blockers.append("baseline_replicated_ensemble_not_qualified")
    if not optimized_qualified:
        blockers.append("optimized_replicated_ensemble_not_qualified")
    if not selected_closed:
        blockers.append("selected_optimized_audit_not_closed")
    if not reduction_ok:
        blockers.append("insufficient_matched_optimized_reduction")
    if not uncertainty_ok:
        blockers.append("insufficient_matched_optimized_uncertainty_separation")
    return {
        "path": path,
        "kind": str(payload.get("kind", "")),
        "case": str(payload.get("case", "")),
        "claim_level": str(payload.get("claim_level", "")),
        "passed": passed,
        "baseline_ensemble_qualified": baseline_qualified,
        "optimized_ensemble_qualified": optimized_qualified,
        "selected_optimized_audit_closed": selected_closed,
        "relative_reduction": relative_reduction,
        "uncertainty_separation_sigma": uncertainty_sigma,
        "relative_reduction_ok": reduction_ok,
        "uncertainty_separation_ok": uncertainty_ok,
        "blockers": blockers,
        "qualifies_for_production_optimization": not blockers,
    }


__all__ = [
    "matched_optimized_transport_report",
    "optimized_equilibrium_transport_report",
    "replicated_transport_ensemble_report",
]
