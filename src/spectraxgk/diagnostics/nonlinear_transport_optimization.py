"""Nonlinear turbulent-transport optimization promotion diagnostics.

These helpers consume already-generated nonlinear transport artifacts and keep
release-scope diagnostics separate from production turbulent-flux optimization
claims. They are data-only and do not launch simulations.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

_NON_PROMOTABLE_MARKERS = (
    "not_transport",
    "not transport",
    "not_production",
    "not production",
    "not_simulation_claim",
    "not simulation claim",
    "startup",
    "reduced",
    "envelope",
    "plumbing",
    "feasibility",
    "diagnostic",
)

_PRODUCTION_CLAIM_MARKERS = (
    "production nonlinear turbulent",
    "production_nonlinear_turbulent",
    "production nonlinear optimization",
    "production_nonlinear_optimization",
    "optimized-equilibrium nonlinear",
    "optimized_equilibrium_nonlinear",
    "converged nonlinear turbulent",
    "converged_nonlinear_turbulent",
)


@dataclass(frozen=True)
class ProductionNonlinearOptimizationGuardConfig:
    """Strict gate settings for production nonlinear optimization promotion."""

    min_replicated_ensembles: int = 2
    min_reports_per_ensemble: int = 2
    max_mean_rel_spread: float = 0.15
    max_combined_sem_rel: float = 0.25
    require_optimized_equilibrium_transport: bool = True
    require_matched_optimized_transport_audit: bool = True
    min_optimized_equilibrium_ensembles: int = 3
    min_matched_optimized_audits: int = 3
    require_seed_timestep_provenance: bool = True
    min_seed_variants: int = 2
    min_timestep_variants: int = 1
    min_matched_optimized_relative_reduction: float = 0.05
    min_matched_optimized_uncertainty_sigma: float = 1.0
    value_floor: float = 1.0e-12

    def validate(self) -> None:
        """Raise if the guard configuration is inconsistent."""

        for name in (
            "min_replicated_ensembles",
            "min_optimized_equilibrium_ensembles",
            "min_matched_optimized_audits",
            "min_seed_variants",
            "min_timestep_variants",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if int(self.min_reports_per_ensemble) < 2:
            raise ValueError("min_reports_per_ensemble must be at least 2")
        for name in (
            "max_mean_rel_spread",
            "max_combined_sem_rel",
            "min_matched_optimized_relative_reduction",
            "min_matched_optimized_uncertainty_sigma",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if float(self.value_floor) <= 0.0:
            raise ValueError("value_floor must be positive")


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _artifact_passed(payload: Mapping[str, Any]) -> bool:
    if any(bool(payload.get(key, False)) for key in ("passed", "gate_passed")):
        return True
    nested = (payload.get(key) for key in ("gate_report", "promotion_gate"))
    return any(
        isinstance(report, Mapping) and bool(report.get("passed", False))
        for report in nested
    )


def _claim_text(payload: Mapping[str, Any]) -> str:
    fields = (
        "kind",
        "case",
        "comparison",
        "claim_level",
        "claim_scope",
        "notes",
        "next_action",
        "model",
    )
    return " ".join(str(payload.get(field, "")) for field in fields).lower()


def _has_non_promotable_marker(payload: Mapping[str, Any]) -> bool:
    text = _claim_text(payload)
    return any(marker in text for marker in _NON_PROMOTABLE_MARKERS)


def _claims_production(payload: Mapping[str, Any]) -> bool:
    text = _claim_text(payload)
    if any(marker in text for marker in _NON_PROMOTABLE_MARKERS):
        return False
    if any(
        bool(payload.get(key, False))
        for key in (
            "production_transport_claim",
            "production_nonlinear_optimization_claim",
        )
    ):
        return True
    return any(marker in text for marker in _PRODUCTION_CLAIM_MARKERS)


def _gate(metric: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"metric": metric, "passed": bool(passed), "detail": detail}


def _variant_value_from_row(row: Mapping[str, Any], axis: str) -> str | None:
    variant = row.get("variant")
    if isinstance(variant, Mapping) and variant.get(axis) not in (None, ""):
        return str(variant.get(axis))
    for key in ("variant_label", "source_artifact", "summary_artifact", "path"):
        value = row.get(key)
        if not isinstance(value, str):
            continue
        if axis == "seed":
            match = re.search(r"seed[0-9]+", value)
        elif axis == "timestep":
            match = re.search(r"dt[0-9]+(?:p[0-9]+)?", value)
        else:
            match = None
        if match:
            return match.group(0)
    return None


def _ensemble_variant_provenance_report(
    payload: Mapping[str, Any],
    *,
    config: ProductionNonlinearOptimizationGuardConfig,
) -> dict[str, Any]:
    rows = payload.get("rows")
    row_maps = (
        [row for row in rows if isinstance(row, Mapping)]
        if isinstance(rows, Sequence)
        else []
    )
    seed_values = sorted(
        {
            value
            for row in row_maps
            if (value := _variant_value_from_row(row, "seed")) is not None
        }
    )
    timestep_values = sorted(
        {
            value
            for row in row_maps
            if (value := _variant_value_from_row(row, "timestep")) is not None
        }
    )
    seed_passed = len(seed_values) >= int(config.min_seed_variants)
    timestep_passed = len(timestep_values) >= int(config.min_timestep_variants)
    return {
        "required": bool(config.require_seed_timestep_provenance),
        "observed_row_count": len(row_maps),
        "seed_values": seed_values,
        "timestep_values": timestep_values,
        "seed_gate_passed": seed_passed,
        "timestep_gate_passed": timestep_passed,
        "passed": seed_passed and timestep_passed,
        "requirements": {
            "min_seed_variants": int(config.min_seed_variants),
            "min_timestep_variants": int(config.min_timestep_variants),
        },
    }


def optimization_artifact_reduction_scope(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return scope metadata for the differentiable optimization comparison."""

    results = payload.get("results")
    result_rows = (
        [row for row in results if isinstance(row, Mapping)]
        if isinstance(results, Sequence)
        else []
    )
    objective_kinds = [str(row.get("objective_kind", "unknown")) for row in result_rows]
    nonlinear_rows = [
        row for row in result_rows if row.get("objective_kind") == "nonlinear_heat_flux"
    ]
    unsafe_rows = [row for row in result_rows if _claims_production(row)]
    artifact_claims_production = _claims_production(payload)
    return {
        "objective_kinds": objective_kinds,
        "contains_reduced_nonlinear_window_objective": bool(nonlinear_rows),
        "n_results": len(result_rows),
        "n_reduced_nonlinear_rows": len(nonlinear_rows),
        "n_rows_claiming_production": len(unsafe_rows),
        "artifact_claims_production": artifact_claims_production,
        "bounded_reduced_scope": bool(
            nonlinear_rows and not unsafe_rows and not artifact_claims_production
        ),
    }


def reduced_artifact_scope_report(
    path: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return whether a startup/reduced artifact is safely blocked from promotion."""

    transport_average_gate = payload.get("transport_average_gate")
    production_gradient_gate = payload.get("production_nonlinear_window_gradient_gate")
    claims_production = _claims_production(payload)
    blocked_by_transport = transport_average_gate is False
    blocked_by_production_gate = production_gradient_gate is False
    blocked_by_claim = _has_non_promotable_marker(payload)
    safely_blocked = bool(
        (blocked_by_transport or blocked_by_production_gate or blocked_by_claim)
        and not claims_production
    )
    return {
        "path": path,
        "kind": str(payload.get("kind", "")),
        "passed": _artifact_passed(payload),
        "claim_level": str(payload.get("claim_level", "")),
        "transport_average_gate": transport_average_gate,
        "production_nonlinear_window_gradient_gate": production_gradient_gate,
        "claims_production": claims_production,
        "blocked_by_transport_average_gate": blocked_by_transport,
        "blocked_by_production_gradient_gate": blocked_by_production_gate,
        "blocked_by_claim_scope": blocked_by_claim,
        "safely_blocked_from_production": safely_blocked,
    }


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

    cfg = _validated_config(config)
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
    checks = (
        (flags.passed, "matched_optimized_audit_failed"),
        (flags.baseline_qualified, "baseline_replicated_ensemble_not_qualified"),
        (flags.optimized_qualified, "optimized_replicated_ensemble_not_qualified"),
        (flags.selected_closed, "selected_optimized_audit_not_closed"),
        (flags.reduction_ok, "insufficient_matched_optimized_reduction"),
        (flags.uncertainty_ok, "insufficient_matched_optimized_uncertainty_separation"),
    )
    return [blocker for passed, blocker in checks if not passed]


def matched_optimized_transport_report(
    path: str,
    payload: Mapping[str, Any],
    *,
    config: ProductionNonlinearOptimizationGuardConfig | None = None,
) -> dict[str, Any]:
    """Return whether a matched baseline-to-optimized audit promotes transport."""

    cfg = _validated_config(config)
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


@dataclass(frozen=True)
class _GuardArtifactMaps:
    reduced: Mapping[str, Mapping[str, Any]]
    replicated_ensembles: Mapping[str, Mapping[str, Any]]
    optimized_equilibria: Mapping[str, Mapping[str, Any]]
    matched_optimized: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class _GuardRows:
    optimization_scope: dict[str, Any]
    reduced: list[dict[str, Any]]
    ensembles: list[dict[str, Any]]
    optimized: list[dict[str, Any]]
    matched_optimized: list[dict[str, Any]]
    qualifying_ensembles: list[dict[str, Any]]
    qualifying_optimized: list[dict[str, Any]]
    qualifying_matched_optimized: list[dict[str, Any]]
    failed_matched_optimized: list[dict[str, Any]]
    best_matched_reduction: float | None


@dataclass(frozen=True)
class _GuardGateStatus:
    safety_gates: list[dict[str, object]]
    promotion_gates: list[dict[str, object]]
    safety_blockers: list[Any]
    promotion_blockers: list[Any]
    safe_to_release: bool
    promoted: bool


def _empty_optimization_scope() -> dict[str, Any]:
    return {
        "objective_kinds": [],
        "contains_reduced_nonlinear_window_objective": False,
        "n_results": 0,
        "n_reduced_nonlinear_rows": 0,
        "n_rows_claiming_production": 0,
        "artifact_claims_production": False,
        "bounded_reduced_scope": False,
    }


def _optimization_scope(
    optimization_artifact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(optimization_artifact, Mapping):
        return optimization_artifact_reduction_scope(optimization_artifact)
    return _empty_optimization_scope()


def _sorted_report_rows(
    artifacts: Mapping[str, Mapping[str, Any]],
    report_fn: Any,
    *,
    config: ProductionNonlinearOptimizationGuardConfig | None = None,
) -> list[dict[str, Any]]:
    if config is None:
        return [report_fn(path, payload) for path, payload in sorted(artifacts.items())]
    return [
        report_fn(path, payload, config=config)
        for path, payload in sorted(artifacts.items())
    ]


def _qualifying(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [row for row in rows if bool(row[key])]


def _failed_matched_optimized(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(row["path"]),
            "case": str(row["case"]),
            "relative_reduction": row["relative_reduction"],
            "uncertainty_separation_sigma": row["uncertainty_separation_sigma"],
            "blockers": list(row["blockers"]),
        }
        for row in rows
        if not bool(row["qualifies_for_production_optimization"])
    ]


def _best_matched_reduction(rows: list[dict[str, Any]]) -> float | None:
    values = [
        float(row["relative_reduction"])
        for row in rows
        if row["relative_reduction"] is not None
    ]
    return max(values) if values else None


def _safety_gates(
    *,
    optimization_artifact: Mapping[str, Any] | None,
    optimization_scope: Mapping[str, Any],
    optimization_artifact_path: str,
    reduced_rows: list[dict[str, Any]],
    qualifying_ensembles: list[dict[str, Any]],
    cfg: ProductionNonlinearOptimizationGuardConfig,
) -> list[dict[str, object]]:
    rows_claiming_production = int(
        _finite_float(optimization_scope.get("n_rows_claiming_production")) or 0
    )
    artifact_claims_production = bool(
        optimization_scope.get("artifact_claims_production", False)
    )
    return [
        _gate(
            "optimization_artifact_present",
            isinstance(optimization_artifact, Mapping)
            and bool(optimization_scope["objective_kinds"]),
            optimization_artifact_path or "missing optimization artifact",
        ),
        _gate(
            "reduced_optimizer_not_promoted",
            rows_claiming_production == 0 and not artifact_claims_production,
            (
                f"rows_claiming_production={rows_claiming_production} "
                f"artifact_claims_production={artifact_claims_production}"
            ),
        ),
        _gate(
            "startup_or_reduced_artifacts_blocked",
            bool(reduced_rows)
            and all(
                bool(row["safely_blocked_from_production"]) for row in reduced_rows
            ),
            f"reduced_artifacts={len(reduced_rows)}",
        ),
        _gate(
            "replicated_long_window_holdouts_present",
            len(qualifying_ensembles) >= int(cfg.min_replicated_ensembles),
            f"qualifying_ensembles={len(qualifying_ensembles)} min={cfg.min_replicated_ensembles}",
        ),
    ]


def _promotion_gates(
    *,
    qualifying_optimized: list[dict[str, Any]],
    qualifying_matched_optimized: list[dict[str, Any]],
    cfg: ProductionNonlinearOptimizationGuardConfig,
) -> list[dict[str, object]]:
    return [
        _gate(
            "optimized_equilibrium_replicated_transport_window",
            len(qualifying_optimized) >= int(cfg.min_optimized_equilibrium_ensembles)
            or not bool(cfg.require_optimized_equilibrium_transport),
            (
                "; ".join(str(row["path"]) for row in qualifying_optimized)
                if qualifying_optimized
                else (
                    "provide long post-transient replicated nonlinear transport "
                    "windows for at least three independent optimized equilibria"
                )
            ),
        ),
        _gate(
            "matched_baseline_to_optimized_transport_reduction",
            len(qualifying_matched_optimized) >= int(cfg.min_matched_optimized_audits)
            or not bool(cfg.require_matched_optimized_transport_audit),
            (
                "; ".join(str(row["path"]) for row in qualifying_matched_optimized)
                if qualifying_matched_optimized
                else (
                    "provide at least three matched baseline-to-optimized nonlinear "
                    "audits with positive relative reduction and uncertainty separation"
                )
            ),
        ),
    ]


def _guard_summary(
    *,
    qualifying_ensembles: list[dict[str, Any]],
    qualifying_optimized: list[dict[str, Any]],
    qualifying_matched_optimized: list[dict[str, Any]],
    matched_optimized_rows: list[dict[str, Any]],
    failed_matched_optimized: list[dict[str, Any]],
    best_matched_reduction: float | None,
    promoted: bool,
) -> dict[str, Any]:
    return {
        "qualifying_replicated_holdout_ensembles": len(qualifying_ensembles),
        "qualifying_optimized_equilibrium_ensembles": len(qualifying_optimized),
        "qualifying_matched_optimized_transport_audits": len(
            qualifying_matched_optimized
        ),
        "total_matched_optimized_transport_audits": len(matched_optimized_rows),
        "failed_matched_optimized_transport_audits": len(failed_matched_optimized),
        "best_matched_optimized_relative_reduction": best_matched_reduction,
        "production_nonlinear_optimization_ready": int(promoted),
    }


def _evidence_gap(
    *,
    failed_matched_optimized: list[dict[str, Any]],
    qualifying_optimized: list[dict[str, Any]],
    qualifying_matched_optimized: list[dict[str, Any]],
    cfg: ProductionNonlinearOptimizationGuardConfig,
) -> dict[str, Any]:
    return {
        "claim_boundary": (
            "Existing strict matched audits are included as negative evidence. "
            "They do not promote broad nonlinear turbulent-flux optimization unless "
            "they pass the same long-window reduction and uncertainty-separation gates."
        ),
        "failed_matched_optimized_transport_audits": failed_matched_optimized,
        "required_additional_optimized_equilibrium_ensembles": max(
            int(cfg.min_optimized_equilibrium_ensembles) - len(qualifying_optimized),
            0,
        ),
        "required_additional_matched_optimized_audits": max(
            int(cfg.min_matched_optimized_audits) - len(qualifying_matched_optimized),
            0,
        ),
    }


def _validated_config(
    config: ProductionNonlinearOptimizationGuardConfig | None,
) -> ProductionNonlinearOptimizationGuardConfig:
    cfg = config or ProductionNonlinearOptimizationGuardConfig()
    cfg.validate()
    return cfg


def _artifact_maps(
    *,
    reduced_artifacts: Mapping[str, Mapping[str, Any]] | None,
    replicated_ensemble_artifacts: Mapping[str, Mapping[str, Any]] | None,
    optimized_equilibrium_artifacts: Mapping[str, Mapping[str, Any]] | None,
    matched_optimized_transport_artifacts: Mapping[str, Mapping[str, Any]] | None,
) -> _GuardArtifactMaps:
    return _GuardArtifactMaps(
        reduced=reduced_artifacts or {},
        replicated_ensembles=replicated_ensemble_artifacts or {},
        optimized_equilibria=optimized_equilibrium_artifacts or {},
        matched_optimized=matched_optimized_transport_artifacts or {},
    )


def _guard_rows(
    *,
    optimization_artifact: Mapping[str, Any] | None,
    artifacts: _GuardArtifactMaps,
    cfg: ProductionNonlinearOptimizationGuardConfig,
) -> _GuardRows:
    optimization_scope = _optimization_scope(optimization_artifact)
    reduced_rows = _sorted_report_rows(artifacts.reduced, reduced_artifact_scope_report)
    ensemble_rows = _sorted_report_rows(
        artifacts.replicated_ensembles,
        replicated_transport_ensemble_report,
        config=cfg,
    )
    optimized_rows = _sorted_report_rows(
        artifacts.optimized_equilibria,
        optimized_equilibrium_transport_report,
        config=cfg,
    )
    matched_rows = _sorted_report_rows(
        artifacts.matched_optimized,
        matched_optimized_transport_report,
        config=cfg,
    )
    qualifying_ensembles = _qualifying(
        ensemble_rows, "qualifies_as_long_post_transient_replicate"
    )
    qualifying_optimized = _qualifying(
        optimized_rows, "qualifies_for_production_optimization"
    )
    qualifying_matched = _qualifying(
        matched_rows, "qualifies_for_production_optimization"
    )
    return _GuardRows(
        optimization_scope=optimization_scope,
        reduced=reduced_rows,
        ensembles=ensemble_rows,
        optimized=optimized_rows,
        matched_optimized=matched_rows,
        qualifying_ensembles=qualifying_ensembles,
        qualifying_optimized=qualifying_optimized,
        qualifying_matched_optimized=qualifying_matched,
        failed_matched_optimized=_failed_matched_optimized(matched_rows),
        best_matched_reduction=_best_matched_reduction(matched_rows),
    )


def _gate_status(
    *,
    optimization_artifact: Mapping[str, Any] | None,
    optimization_artifact_path: str,
    rows: _GuardRows,
    cfg: ProductionNonlinearOptimizationGuardConfig,
) -> _GuardGateStatus:
    safety_gates = _safety_gates(
        optimization_artifact=optimization_artifact,
        optimization_scope=rows.optimization_scope,
        optimization_artifact_path=optimization_artifact_path,
        reduced_rows=rows.reduced,
        qualifying_ensembles=rows.qualifying_ensembles,
        cfg=cfg,
    )

    def blockers(gates: list[dict[str, object]]) -> list[object]:
        return [gate["metric"] for gate in gates if not bool(gate["passed"])]

    safety_blockers = blockers(safety_gates)
    promotion_gates = _promotion_gates(
        qualifying_optimized=rows.qualifying_optimized,
        qualifying_matched_optimized=rows.qualifying_matched_optimized,
        cfg=cfg,
    )
    promotion_blockers = blockers(promotion_gates)
    safe_to_release = not safety_blockers
    promoted = safe_to_release and not promotion_blockers
    return _GuardGateStatus(
        safety_gates=safety_gates,
        promotion_gates=promotion_gates,
        safety_blockers=safety_blockers,
        promotion_blockers=promotion_blockers,
        safe_to_release=safe_to_release,
        promoted=promoted,
    )


def _claim_level(gates: _GuardGateStatus) -> str:
    return (
        "production_nonlinear_optimization_promoted_by_replicated_transport_windows"
        if gates.promoted
        else "production_nonlinear_optimization_blocked_until_optimized_equilibrium_replicated_transport_windows"
    )


def _safety_gate_payload(gates: _GuardGateStatus) -> dict[str, Any]:
    return {
        "passed": gates.safe_to_release,
        "blockers": gates.safety_blockers,
        "requirements": [
            "differentiable optimization artifacts must not claim production nonlinear turbulent transport",
            "startup and reduced nonlinear-window artifacts must record false production/transport gates",
            "at least two long post-transient replicated nonlinear-window holdout ensembles must pass",
        ],
    }


def _promotion_gate_payload(gates: _GuardGateStatus) -> dict[str, Any]:
    return {
        "passed": gates.promoted,
        "blockers": gates.promotion_blockers,
        "requirements": [
            "optimized equilibrium must have long post-transient replicated nonlinear transport-window audits",
            "at least three matched baseline-to-optimized nonlinear audits must show positive uncertainty-separated reductions",
            "replicates must include independent seed/initial-condition and timestep evidence",
            "optimized-equilibrium transport means must satisfy running-window, block/SEM, spread, and finite-flux gates",
        ],
    }


def _guard_report_payload(
    *,
    optimization_artifact_path: str,
    rows: _GuardRows,
    gates: _GuardGateStatus,
    cfg: ProductionNonlinearOptimizationGuardConfig,
) -> dict[str, Any]:
    return {
        "kind": "production_nonlinear_turbulent_flux_optimization_guard",
        "claim_level": _claim_level(gates),
        "passed": gates.safe_to_release,
        "safe_to_release": gates.safe_to_release,
        "production_nonlinear_optimization_promoted": gates.promoted,
        "optimization_artifact_path": optimization_artifact_path,
        "optimization_scope": rows.optimization_scope,
        "safety_gate": _safety_gate_payload(gates),
        "promotion_gate": _promotion_gate_payload(gates),
        "gates": gates.safety_gates + gates.promotion_gates,
        "reduced_artifacts": rows.reduced,
        "replicated_ensemble_artifacts": rows.ensembles,
        "optimized_equilibrium_artifacts": rows.optimized,
        "matched_optimized_transport_artifacts": rows.matched_optimized,
        "summary": _guard_summary(
            qualifying_ensembles=rows.qualifying_ensembles,
            qualifying_optimized=rows.qualifying_optimized,
            qualifying_matched_optimized=rows.qualifying_matched_optimized,
            matched_optimized_rows=rows.matched_optimized,
            failed_matched_optimized=rows.failed_matched_optimized,
            best_matched_reduction=rows.best_matched_reduction,
            promoted=gates.promoted,
        ),
        "evidence_gap": _evidence_gap(
            failed_matched_optimized=rows.failed_matched_optimized,
            qualifying_optimized=rows.qualifying_optimized,
            qualifying_matched_optimized=rows.qualifying_matched_optimized,
            cfg=cfg,
        ),
        "config": asdict(cfg),
        "notes": (
            "This guard intentionally allows release when reduced/startup nonlinear "
            "artifacts are scoped correctly. Production nonlinear turbulent-flux "
            "optimization is promoted only when optimized-equilibrium long-window "
            "replicate audits exist and pass."
        ),
    }


def production_nonlinear_optimization_guard_report(
    *,
    optimization_artifact: Mapping[str, Any] | None,
    optimization_artifact_path: str = "",
    reduced_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    replicated_ensemble_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    optimized_equilibrium_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    matched_optimized_transport_artifacts: Mapping[str, Mapping[str, Any]]
    | None = None,
    config: ProductionNonlinearOptimizationGuardConfig | None = None,
) -> dict[str, Any]:
    """Build the fail-closed nonlinear turbulent-flux optimization guard.

    The top-level ``passed`` field means the release is safe: reduced/startup
    artifacts are correctly scoped and long-window replicated holdouts are
    present. It does *not* mean production nonlinear optimization is promoted;
    that is reported separately by ``production_nonlinear_optimization_promoted``.
    """

    cfg = _validated_config(config)
    artifacts = _artifact_maps(
        reduced_artifacts=reduced_artifacts,
        replicated_ensemble_artifacts=replicated_ensemble_artifacts,
        optimized_equilibrium_artifacts=optimized_equilibrium_artifacts,
        matched_optimized_transport_artifacts=matched_optimized_transport_artifacts,
    )
    rows = _guard_rows(
        optimization_artifact=optimization_artifact,
        artifacts=artifacts,
        cfg=cfg,
    )
    gates = _gate_status(
        optimization_artifact=optimization_artifact,
        optimization_artifact_path=optimization_artifact_path,
        rows=rows,
        cfg=cfg,
    )
    return _guard_report_payload(
        optimization_artifact_path=optimization_artifact_path,
        rows=rows,
        gates=gates,
        cfg=cfg,
    )


__all__ = (
    "ProductionNonlinearOptimizationGuardConfig",
    "matched_optimized_transport_report",
    "optimization_artifact_reduction_scope",
    "optimized_equilibrium_transport_report",
    "production_nonlinear_optimization_guard_report",
    "reduced_artifact_scope_report",
    "replicated_transport_ensemble_report",
)
