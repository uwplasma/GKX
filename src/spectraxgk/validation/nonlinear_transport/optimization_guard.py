"""Promotion guardrails for nonlinear turbulent-transport optimization claims.

The reduced stellarator optimization examples are intentionally differentiable
and cheap. Production turbulent-flux optimization is a different claim: it must
be supported by long post-transient nonlinear transport windows, replicated in
seed/initial-condition and timestep, and then repeated on the optimized
equilibrium. This module keeps that distinction executable so startup windows
or reduced nonlinear envelopes cannot silently become production claims.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from spectraxgk.validation.nonlinear_transport.optimization_policy import (
    ProductionNonlinearOptimizationGuardConfig,
    _finite_float,
    _gate,
    optimization_artifact_reduction_scope,
    reduced_artifact_scope_report,
)
from spectraxgk.validation.nonlinear_transport.optimization_reports import (
    matched_optimized_transport_report,
    optimized_equilibrium_transport_report,
    replicated_transport_ensemble_report,
)


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


def _optimization_scope(optimization_artifact: Mapping[str, Any] | None) -> dict[str, Any]:
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
            int(cfg.min_matched_optimized_audits)
            - len(qualifying_matched_optimized),
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
    safety_blockers = [
        gate["metric"] for gate in safety_gates if not bool(gate["passed"])
    ]
    promotion_gates = _promotion_gates(
        qualifying_optimized=rows.qualifying_optimized,
        qualifying_matched_optimized=rows.qualifying_matched_optimized,
        cfg=cfg,
    )
    promotion_blockers = [
        gate["metric"] for gate in promotion_gates if not bool(gate["passed"])
    ]
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
    if gates.promoted:
        return "production_nonlinear_optimization_promoted_by_replicated_transport_windows"
    return "production_nonlinear_optimization_blocked_until_optimized_equilibrium_replicated_transport_windows"


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


__all__ = [
    "ProductionNonlinearOptimizationGuardConfig",
    "matched_optimized_transport_report",
    "optimization_artifact_reduction_scope",
    "optimized_equilibrium_transport_report",
    "production_nonlinear_optimization_guard_report",
    "reduced_artifact_scope_report",
    "replicated_transport_ensemble_report",
]
