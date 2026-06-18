"""Policy and scope helpers for nonlinear optimization promotion guards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
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

        if int(self.min_replicated_ensembles) < 1:
            raise ValueError("min_replicated_ensembles must be positive")
        if int(self.min_reports_per_ensemble) < 2:
            raise ValueError("min_reports_per_ensemble must be at least 2")
        if float(self.max_mean_rel_spread) < 0.0:
            raise ValueError("max_mean_rel_spread must be non-negative")
        if float(self.max_combined_sem_rel) < 0.0:
            raise ValueError("max_combined_sem_rel must be non-negative")
        if int(self.min_optimized_equilibrium_ensembles) < 1:
            raise ValueError("min_optimized_equilibrium_ensembles must be positive")
        if int(self.min_matched_optimized_audits) < 1:
            raise ValueError("min_matched_optimized_audits must be positive")
        if int(self.min_seed_variants) < 1:
            raise ValueError("min_seed_variants must be positive")
        if int(self.min_timestep_variants) < 1:
            raise ValueError("min_timestep_variants must be positive")
        if float(self.min_matched_optimized_relative_reduction) < 0.0:
            raise ValueError(
                "min_matched_optimized_relative_reduction must be non-negative"
            )
        if float(self.min_matched_optimized_uncertainty_sigma) < 0.0:
            raise ValueError(
                "min_matched_optimized_uncertainty_sigma must be non-negative"
            )
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


__all__ = [
    "ProductionNonlinearOptimizationGuardConfig",
    "optimization_artifact_reduction_scope",
    "reduced_artifact_scope_report",
]
