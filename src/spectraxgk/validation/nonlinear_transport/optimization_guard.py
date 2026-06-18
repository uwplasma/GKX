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
from dataclasses import asdict
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

    cfg = config or ProductionNonlinearOptimizationGuardConfig()
    cfg.validate()
    reduced_artifacts = reduced_artifacts or {}
    replicated_ensemble_artifacts = replicated_ensemble_artifacts or {}
    optimized_equilibrium_artifacts = optimized_equilibrium_artifacts or {}
    matched_optimized_transport_artifacts = matched_optimized_transport_artifacts or {}

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
            "bounded_reduced_scope": False,
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
    matched_optimized_rows = [
        matched_optimized_transport_report(path, payload, config=cfg)
        for path, payload in sorted(matched_optimized_transport_artifacts.items())
    ]
    qualifying_ensembles = [
        row
        for row in ensemble_rows
        if bool(row["qualifies_as_long_post_transient_replicate"])
    ]
    qualifying_optimized = [
        row
        for row in optimized_rows
        if bool(row["qualifies_for_production_optimization"])
    ]
    qualifying_matched_optimized = [
        row
        for row in matched_optimized_rows
        if bool(row["qualifies_for_production_optimization"])
    ]
    failed_matched_optimized = [
        {
            "path": str(row["path"]),
            "case": str(row["case"]),
            "relative_reduction": row["relative_reduction"],
            "uncertainty_separation_sigma": row["uncertainty_separation_sigma"],
            "blockers": list(row["blockers"]),
        }
        for row in matched_optimized_rows
        if not bool(row["qualifies_for_production_optimization"])
    ]
    reduction_values = [
        float(row["relative_reduction"])
        for row in matched_optimized_rows
        if row["relative_reduction"] is not None
    ]
    best_matched_reduction = max(reduction_values) if reduction_values else None
    rows_claiming_production = int(
        _finite_float(optimization_scope.get("n_rows_claiming_production")) or 0
    )
    artifact_claims_production = bool(
        optimization_scope.get("artifact_claims_production", False)
    )

    safety_gates = [
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
    safety_blockers = [
        gate["metric"] for gate in safety_gates if not bool(gate["passed"])
    ]
    safe_to_release = not safety_blockers

    promotion_gates = [
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
    promotion_blockers = [
        gate["metric"] for gate in promotion_gates if not bool(gate["passed"])
    ]
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
                "at least three matched baseline-to-optimized nonlinear audits must show positive uncertainty-separated reductions",
                "replicates must include independent seed/initial-condition and timestep evidence",
                "optimized-equilibrium transport means must satisfy running-window, block/SEM, spread, and finite-flux gates",
            ],
        },
        "gates": safety_gates + promotion_gates,
        "reduced_artifacts": reduced_rows,
        "replicated_ensemble_artifacts": ensemble_rows,
        "optimized_equilibrium_artifacts": optimized_rows,
        "matched_optimized_transport_artifacts": matched_optimized_rows,
        "summary": {
            "qualifying_replicated_holdout_ensembles": len(qualifying_ensembles),
            "qualifying_optimized_equilibrium_ensembles": len(qualifying_optimized),
            "qualifying_matched_optimized_transport_audits": len(
                qualifying_matched_optimized
            ),
            "total_matched_optimized_transport_audits": len(matched_optimized_rows),
            "failed_matched_optimized_transport_audits": len(failed_matched_optimized),
            "best_matched_optimized_relative_reduction": best_matched_reduction,
            "production_nonlinear_optimization_ready": int(promoted),
        },
        "evidence_gap": {
            "claim_boundary": (
                "Existing strict matched audits are included as negative evidence. "
                "They do not promote broad nonlinear turbulent-flux optimization unless "
                "they pass the same long-window reduction and uncertainty-separation gates."
            ),
            "failed_matched_optimized_transport_audits": failed_matched_optimized,
            "required_additional_optimized_equilibrium_ensembles": max(
                int(cfg.min_optimized_equilibrium_ensembles)
                - len(qualifying_optimized),
                0,
            ),
            "required_additional_matched_optimized_audits": max(
                int(cfg.min_matched_optimized_audits)
                - len(qualifying_matched_optimized),
                0,
            ),
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
    "matched_optimized_transport_report",
    "optimization_artifact_reduction_scope",
    "optimized_equilibrium_transport_report",
    "production_nonlinear_optimization_guard_report",
    "reduced_artifact_scope_report",
    "replicated_transport_ensemble_report",
]
