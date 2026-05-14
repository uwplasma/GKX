"""Promotion guardrails for nonlinear turbulent-transport optimization claims.

The reduced stellarator optimization examples are intentionally differentiable
and cheap. Production turbulent-flux optimization is a different claim: it must
be supported by long post-transient nonlinear transport windows, replicated in
seed/initial-condition and timestep, and then repeated on the optimized
equilibrium. This module keeps that distinction executable so startup windows
or reduced nonlinear envelopes cannot silently become production claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
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
    value_floor: float = 1.0e-12

    def validate(self) -> None:
        """Raise if the guard configuration is inconsistent."""

        if int(self.min_replicated_ensembles) < 1:
            raise ValueError("min_replicated_ensembles must be positive")
        if int(self.min_reports_per_ensemble) < 2:
            raise ValueError("min_reports_per_ensemble must be at least 2")
        if float(self.max_mean_rel_spread) < 0.0:
            raise ValueError("max_mean_rel_spread must be non-negative")
        if float(self.max_combined_sem_rel) < 0.0:
            raise ValueError("max_combined_sem_rel must be non-negative")
        if float(self.value_floor) <= 0.0:
            raise ValueError("value_floor must be positive")


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _artifact_passed(payload: Mapping[str, Any]) -> bool:
    if bool(payload.get("passed", False)):
        return True
    if bool(payload.get("gate_passed", False)):
        return True
    for key in ("gate_report", "promotion_gate"):
        nested = payload.get(key)
        if isinstance(nested, Mapping) and bool(nested.get("passed", False)):
            return True
    return False


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
    if bool(payload.get("production_transport_claim", False)):
        return True
    if bool(payload.get("production_nonlinear_optimization_claim", False)):
        return True
    return any(marker in text for marker in _PRODUCTION_CLAIM_MARKERS)


def _gate(metric: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"metric": metric, "passed": bool(passed), "detail": detail}


def optimization_artifact_reduction_scope(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return scope metadata for the differentiable optimization comparison."""

    results = payload.get("results")
    result_rows = [row for row in results if isinstance(row, Mapping)] if isinstance(results, Sequence) else []
    objective_kinds = [str(row.get("objective_kind", "unknown")) for row in result_rows]
    nonlinear_rows = [row for row in result_rows if row.get("objective_kind") == "nonlinear_heat_flux"]
    unsafe_rows = [row for row in result_rows if _claims_production(row)]
    artifact_claims_production = _claims_production(payload)
    return {
        "objective_kinds": objective_kinds,
        "contains_reduced_nonlinear_window_objective": bool(nonlinear_rows),
        "n_results": len(result_rows),
        "n_reduced_nonlinear_rows": len(nonlinear_rows),
        "n_rows_claiming_production": len(unsafe_rows),
        "artifact_claims_production": artifact_claims_production,
        "legacy_reduced_scope": bool(
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
    safely_blocked = bool((blocked_by_transport or blocked_by_production_gate or blocked_by_claim) and not claims_production)
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
    claim_scoped = "replicated_nonlinear_window" in claim and "not_simulation_claim" in claim
    finite_mean = ensemble_mean is not None and abs(ensemble_mean) >= float(cfg.value_floor)
    spread_ok = mean_rel_spread is not None and mean_rel_spread <= float(cfg.max_mean_rel_spread)
    sem_ok = combined_sem_rel is not None and combined_sem_rel <= float(cfg.max_combined_sem_rel)
    report_count_ok = n_reports >= int(cfg.min_reports_per_ensemble)
    qualifies = bool(is_ensemble and passed and claim_scoped and finite_mean and spread_ok and sem_ok and report_count_ok)
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
    )
    row["optimized_equilibrium_marker"] = optimized_marker
    row["qualifies_for_production_optimization"] = bool(
        row["qualifies_as_long_post_transient_replicate"] and optimized_marker
    )
    return row


def production_nonlinear_optimization_guard_report(
    *,
    optimization_artifact: Mapping[str, Any] | None,
    optimization_artifact_path: str = "",
    reduced_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    replicated_ensemble_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    optimized_equilibrium_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    config: ProductionNonlinearOptimizationGuardConfig | None = None,
) -> dict[str, Any]:
    """Build the fail-closed nonlinear turbulent-flux optimization guard.

    The top-level ``passed`` field means the release is safe: reduced/startup
    artifacts are correctly scoped and long-window replicated holdouts are
    present. It does *not* mean production nonlinear optimization is promoted;
    that is reported separately by ``production_nonlinear_optimization_promoted``.
    """

    cfg = config or ProductionNonlinearOptimizationGuardConfig()
    cfg.validate()
    reduced_artifacts = reduced_artifacts or {}
    replicated_ensemble_artifacts = replicated_ensemble_artifacts or {}
    optimized_equilibrium_artifacts = optimized_equilibrium_artifacts or {}

    optimization_scope = (
        optimization_artifact_reduction_scope(optimization_artifact)
        if isinstance(optimization_artifact, Mapping)
        else {
            "objective_kinds": [],
            "contains_reduced_nonlinear_window_objective": False,
            "n_results": 0,
            "n_reduced_nonlinear_rows": 0,
            "n_rows_claiming_production": 0,
            "artifact_claims_production": False,
            "legacy_reduced_scope": False,
        }
    )
    reduced_rows = [
        reduced_artifact_scope_report(path, payload)
        for path, payload in sorted(reduced_artifacts.items())
    ]
    ensemble_rows = [
        replicated_transport_ensemble_report(path, payload, config=cfg)
        for path, payload in sorted(replicated_ensemble_artifacts.items())
    ]
    optimized_rows = [
        optimized_equilibrium_transport_report(path, payload, config=cfg)
        for path, payload in sorted(optimized_equilibrium_artifacts.items())
    ]
    qualifying_ensembles = [
        row for row in ensemble_rows if bool(row["qualifies_as_long_post_transient_replicate"])
    ]
    qualifying_optimized = [
        row for row in optimized_rows if bool(row["qualifies_for_production_optimization"])
    ]
    rows_claiming_production = int(
        _finite_float(optimization_scope.get("n_rows_claiming_production")) or 0
    )
    artifact_claims_production = bool(
        optimization_scope.get("artifact_claims_production", False)
    )

    safety_gates = [
        _gate(
            "optimization_artifact_present",
            isinstance(optimization_artifact, Mapping) and bool(optimization_scope["objective_kinds"]),
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
            bool(reduced_rows) and all(bool(row["safely_blocked_from_production"]) for row in reduced_rows),
            f"reduced_artifacts={len(reduced_rows)}",
        ),
        _gate(
            "replicated_long_window_holdouts_present",
            len(qualifying_ensembles) >= int(cfg.min_replicated_ensembles),
            f"qualifying_ensembles={len(qualifying_ensembles)} min={cfg.min_replicated_ensembles}",
        ),
    ]
    safety_blockers = [gate["metric"] for gate in safety_gates if not bool(gate["passed"])]
    safe_to_release = not safety_blockers

    promotion_gates = [
        _gate(
            "optimized_equilibrium_replicated_transport_window",
            bool(qualifying_optimized) or not bool(cfg.require_optimized_equilibrium_transport),
            (
                "; ".join(str(row["path"]) for row in qualifying_optimized)
                if qualifying_optimized
                else "provide long post-transient replicated nonlinear transport windows for the optimized equilibrium"
            ),
        ),
    ]
    promotion_blockers = [gate["metric"] for gate in promotion_gates if not bool(gate["passed"])]
    promoted = safe_to_release and not promotion_blockers

    return {
        "kind": "production_nonlinear_turbulent_flux_optimization_guard",
        "claim_level": (
            "production_nonlinear_optimization_blocked_until_optimized_equilibrium_replicated_transport_windows"
            if not promoted
            else "production_nonlinear_optimization_promoted_by_replicated_transport_windows"
        ),
        "passed": safe_to_release,
        "safe_to_release": safe_to_release,
        "production_nonlinear_optimization_promoted": promoted,
        "optimization_artifact_path": optimization_artifact_path,
        "optimization_scope": optimization_scope,
        "safety_gate": {
            "passed": safe_to_release,
            "blockers": safety_blockers,
            "requirements": [
                "differentiable optimization artifacts must not claim production nonlinear turbulent transport",
                "startup and reduced nonlinear-window artifacts must record false production/transport gates",
                "at least two long post-transient replicated nonlinear-window holdout ensembles must pass",
            ],
        },
        "promotion_gate": {
            "passed": promoted,
            "blockers": promotion_blockers,
            "requirements": [
                "optimized equilibrium must have long post-transient replicated nonlinear transport-window audits",
                "replicates must include independent seed/initial-condition and timestep evidence",
                "optimized-equilibrium transport means must satisfy running-window, block/SEM, spread, and finite-flux gates",
            ],
        },
        "gates": safety_gates + promotion_gates,
        "reduced_artifacts": reduced_rows,
        "replicated_ensemble_artifacts": ensemble_rows,
        "optimized_equilibrium_artifacts": optimized_rows,
        "summary": {
            "qualifying_replicated_holdout_ensembles": len(qualifying_ensembles),
            "qualifying_optimized_equilibrium_ensembles": len(qualifying_optimized),
            "production_nonlinear_optimization_ready": int(promoted),
        },
        "config": asdict(cfg),
        "notes": (
            "This guard intentionally allows release when reduced/startup nonlinear "
            "artifacts are scoped correctly. Production nonlinear turbulent-flux "
            "optimization is promoted only when optimized-equilibrium long-window "
            "replicate audits exist and pass."
        ),
    }


__all__ = [
    "ProductionNonlinearOptimizationGuardConfig",
    "optimization_artifact_reduction_scope",
    "optimized_equilibrium_transport_report",
    "production_nonlinear_optimization_guard_report",
    "reduced_artifact_scope_report",
    "replicated_transport_ensemble_report",
]
