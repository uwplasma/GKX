#!/usr/bin/env python3
"""Grouped VMEC/Boozer release claim gates.

Subcommands:
- ``differentiability-claim`` validates scoped differentiable-geometry evidence.
- ``aggregate-holdout`` validates held-out surface/field-line promotion evidence.
- ``high-grid-admission`` validates scoped external-VMEC transport holdouts.
- ``reduced-portfolio`` validates reduced multi-point portfolio artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_OUT = (
    REPO_ROOT / "docs" / "_static" / "vmec_boozer_differentiability_claim_guard.json"
)

DEFAULT_PARITY_MATRIX = "docs/_static/vmec_boozer_parity_matrix.json"
DEFAULT_GRADIENT_MATRIX = "docs/_static/vmec_boozer_gradient_holdout_matrix.json"
DEFAULT_BRIDGE = "docs/_static/differentiable_geometry_bridge.json"
DEFAULT_NONLINEAR_FD_AUDIT = "docs/_static/vmec_boozer_nonlinear_window_fd_audit.json"
DEFAULT_FINITE_BETA_FREQUENCY_GATE = (
    "docs/_static/vmec_boozer_shaped_pressure_solver_frequency_gradient_gate.json"
)
DEFAULT_FINITE_BETA_QUASILINEAR_GATE = (
    "docs/_static/vmec_boozer_shaped_pressure_quasilinear_gradient_gate.json"
)
DEFAULT_FINITE_BETA_NONLINEAR_WINDOW_GATE = (
    "docs/_static/vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json"
)

REQUIRED_GRADIENT_GATE_TYPES = {
    "frequency",
    "quasilinear",
    "nonlinear-window estimator",
}
REQUIRED_OBJECTIVES_BY_GATE_TYPE = {
    "frequency": {
        "gamma",
        "omega",
    },
    "quasilinear": {
        "gamma",
        "omega",
        "kperp_eff2",
        "linear_heat_flux_weight",
        "mixing_length_heat_flux_proxy",
    },
    "nonlinear-window estimator": {
        "gamma",
        "omega",
        "kperp_eff2",
        "linear_heat_flux_weight",
        "mixing_length_heat_flux_proxy",
        "nonlinear_window_heat_flux_mean",
        "nonlinear_window_heat_flux_cv",
        "nonlinear_window_heat_flux_trend",
    },
}
MAX_REL_ERROR_BY_GATE_TYPE = {
    "frequency": 5.0e-2,
    "quasilinear": 2.0e-2,
    "nonlinear-window estimator": 7.5e-2,
}
REQUIRED_SOURCE_SCOPE = "mode21_vmec_boozer_state"
MINIMUM_BOOZER_MODE_COUNT = 21
MINIMUM_GRADIENT_CASES = 2
REQUIRED_PARITY_ROW_FAMILIES = {
    "axisymmetric finite-beta",
    "quasi-helical",
    "quasi-isodynamic",
}


def _read_json(root: Path, rel_path: str) -> dict[str, Any]:
    path = root / rel_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing JSON artifact: {rel_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {rel_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{rel_path}: expected a JSON object")
    return payload


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _finite_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _path_rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _parity_matrix_checks(parity: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    summary = _as_dict(parity.get("summary"))
    rows = _as_rows(parity.get("rows"))
    minimum_mode_count = parity.get("minimum_boozer_mode_count")
    mode_floor_passed = (
        isinstance(minimum_mode_count, (int, float))
        and minimum_mode_count >= MINIMUM_BOOZER_MODE_COUNT
    )
    row_mode_failures = [
        row.get("case_name", "<unknown>")
        for row in rows
        if min(float(row.get("mboz", 0.0)), float(row.get("nboz", 0.0)))
        < MINIMUM_BOOZER_MODE_COUNT
        or row.get("mode_floor_passed") is False
    ]
    direct_open_rows = [
        row.get("case_name", "<unknown>")
        for row in rows
        if row.get("production_parity_passed") is False
        and row.get("status") == "diagnostic_open"
    ]
    hidden_direct_failures = [
        row.get("case_name", "<unknown>")
        for row in rows
        if row.get("production_parity_passed") is False
        and row.get("status") != "diagnostic_open"
    ]
    passed_families = {
        str(row.get("family", "")).strip()
        for row in rows
        if row.get("available") is not False and row.get("equal_arc_all_passed") is True
    }
    finite_beta_pressure_rows = [
        row.get("case_name", "<unknown>")
        for row in rows
        if row.get("equal_arc_all_passed") is True
        and (
            "finite-beta" in str(row.get("family", "")).lower()
            or "pressure" in str(row.get("family", "")).lower()
            or "pressure" in str(row.get("case_name", "")).lower()
        )
    ]
    claim_level = str(parity.get("claim_level", ""))
    claim_scoped = "not_full_transport_gradient_claim" in claim_level
    checks = {
        "all_available": _bool(summary.get("all_available")),
        "all_equal_arc_passed": _bool(summary.get("all_equal_arc_passed")),
        "minimum_boozer_mode_count": minimum_mode_count,
        "mode_floor_passed": mode_floor_passed and not row_mode_failures,
        "n_cases": summary.get("n_cases", len(rows)),
        "n_equal_arc_passed": summary.get("n_equal_arc_passed"),
        "direct_tensor_parity_open_rows": sorted(
            str(item) for item in direct_open_rows
        ),
        "direct_tensor_parity_hidden_failures": sorted(
            str(item) for item in hidden_direct_failures
        ),
        "passed_families": sorted(passed_families),
        "required_families": sorted(REQUIRED_PARITY_ROW_FAMILIES),
        "missing_required_families": sorted(
            REQUIRED_PARITY_ROW_FAMILIES - passed_families
        ),
        "finite_beta_pressure_equal_arc_rows": sorted(
            str(item) for item in finite_beta_pressure_rows
        ),
        "claim_level": claim_level,
        "claim_scoped_not_full_transport_gradient": claim_scoped,
    }
    blockers: list[str] = []
    if not checks["all_available"]:
        blockers.append("parity_matrix_not_all_available")
    if not checks["all_equal_arc_passed"]:
        blockers.append("equal_arc_parity_matrix_failed")
    if not checks["mode_floor_passed"]:
        blockers.append("boozer_mode_floor_failed")
    if hidden_direct_failures:
        blockers.append("direct_tensor_parity_failure_not_marked_diagnostic_open")
    if checks["missing_required_families"]:
        blockers.append("parity_matrix_missing_required_family")
    if not finite_beta_pressure_rows:
        blockers.append("parity_matrix_missing_finite_beta_pressure_row")
    if not claim_scoped:
        blockers.append("parity_matrix_claim_not_scoped")
    return checks, blockers


def _gradient_matrix_checks(
    gradient: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    summary = _as_dict(gradient.get("summary"))
    rows = _as_rows(gradient.get("rows"))
    cases = {str(row.get("case_name")) for row in rows if row.get("case_name")}
    gate_types = {str(row.get("gate_type")) for row in rows if row.get("gate_type")}
    source_scope_failures = [
        row.get("path") or row.get("case_name", "<unknown>")
        for row in rows
        if row.get("source_scope") != REQUIRED_SOURCE_SCOPE
    ]
    mode_failures = [
        row.get("path") or row.get("case_name", "<unknown>")
        for row in rows
        if min(float(row.get("mboz", 0.0)), float(row.get("nboz", 0.0)))
        < MINIMUM_BOOZER_MODE_COUNT
    ]
    failed_rows = [
        row.get("path") or row.get("case_name", "<unknown>")
        for row in rows
        if row.get("passed") is not True
    ]
    gate_types_by_case: dict[str, set[str]] = {case: set() for case in cases}
    missing_gate_types_by_case: dict[str, list[str]] = {}
    missing_objectives: dict[str, list[str]] = {}
    failed_objectives: dict[str, list[str]] = {}
    row_error_failures: dict[str, dict[str, float | None]] = {}
    for row in rows:
        case = str(row.get("case_name", "<unknown>"))
        gate_type = str(row.get("gate_type", "<unknown>"))
        gate_types_by_case.setdefault(case, set()).add(gate_type)
        required_objectives = REQUIRED_OBJECTIVES_BY_GATE_TYPE.get(gate_type, set())
        objectives = _as_dict(row.get("objectives"))
        objective_rel_error = _as_dict(row.get("objective_rel_error"))
        row_id = f"{case}:{gate_type}"
        missing = sorted(required_objectives - set(objectives))
        failed = sorted(
            objective
            for objective in required_objectives
            if objectives.get(objective) is not True
        )
        if missing:
            missing_objectives[row_id] = missing
        if failed:
            failed_objectives[row_id] = failed
        threshold = MAX_REL_ERROR_BY_GATE_TYPE.get(gate_type)
        max_rel_error = _finite_float_or_none(row.get("max_rel_error"))
        if threshold is not None and (
            max_rel_error is None or max_rel_error > threshold
        ):
            row_error_failures[row_id] = {
                "max_rel_error": max_rel_error,
                "threshold": threshold,
            }
        for objective in required_objectives:
            rel_error = _finite_float_or_none(objective_rel_error.get(objective))
            if rel_error is None or (threshold is not None and rel_error > threshold):
                row_error_failures[f"{row_id}:{objective}"] = {
                    "max_rel_error": rel_error,
                    "threshold": threshold,
                }
    for case, case_gate_types in gate_types_by_case.items():
        missing = sorted(REQUIRED_GRADIENT_GATE_TYPES - case_gate_types)
        if missing:
            missing_gate_types_by_case[case] = missing
    claim_level = str(gradient.get("claim_level", ""))
    claim_scoped = "not_production_nonlinear_optimization" in claim_level
    checks = {
        "passed": _bool(gradient.get("passed")),
        "all_gates_passed": _bool(summary.get("all_gates_passed")),
        "all_mode21_source_scope": _bool(summary.get("all_mode21_source_scope")),
        "all_mboz_nboz_at_least_21": _bool(summary.get("all_mboz_nboz_at_least_21")),
        "n_cases": len(cases),
        "cases": sorted(cases),
        "gate_types": sorted(gate_types),
        "required_gate_types": sorted(REQUIRED_GRADIENT_GATE_TYPES),
        "missing_gate_types": sorted(REQUIRED_GRADIENT_GATE_TYPES - gate_types),
        "missing_gate_types_by_case": missing_gate_types_by_case,
        "required_objectives_by_gate_type": {
            gate_type: sorted(objectives)
            for gate_type, objectives in REQUIRED_OBJECTIVES_BY_GATE_TYPE.items()
        },
        "max_rel_error_by_gate_type": MAX_REL_ERROR_BY_GATE_TYPE,
        "missing_objectives": missing_objectives,
        "failed_objectives": failed_objectives,
        "row_error_failures": row_error_failures,
        "source_scope_failures": sorted(str(item) for item in source_scope_failures),
        "mode_failures": sorted(str(item) for item in mode_failures),
        "failed_rows": sorted(str(item) for item in failed_rows),
        "claim_level": claim_level,
        "claim_scoped_not_production_nonlinear_optimization": claim_scoped,
    }
    blockers: list[str] = []
    if not checks["passed"] or not checks["all_gates_passed"] or failed_rows:
        blockers.append("gradient_holdout_matrix_failed")
    if checks["n_cases"] < MINIMUM_GRADIENT_CASES:
        blockers.append("gradient_holdout_case_count_below_two")
    if checks["missing_gate_types"]:
        blockers.append("gradient_holdout_missing_required_gate_types")
    if missing_gate_types_by_case:
        blockers.append("gradient_holdout_missing_required_gate_type_for_case")
    if missing_objectives:
        blockers.append("gradient_holdout_missing_required_objective")
    if failed_objectives:
        blockers.append("gradient_holdout_objective_failed")
    if row_error_failures:
        blockers.append("gradient_holdout_error_threshold_failed")
    if source_scope_failures or not checks["all_mode21_source_scope"]:
        blockers.append("gradient_holdout_wrong_source_scope")
    if mode_failures or not checks["all_mboz_nboz_at_least_21"]:
        blockers.append("gradient_holdout_mode_floor_failed")
    if not claim_scoped:
        blockers.append("gradient_holdout_claim_not_scoped")
    return checks, blockers


def _bridge_checks(bridge: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    direct = _as_dict(bridge.get("vmex_flux_tube_array_parity"))
    status = str(direct.get("status", ""))
    production_parity_passed = direct.get("production_parity_passed")
    interpretation = str(direct.get("interpretation", ""))
    direct_gap_scoped = production_parity_passed is True or (
        production_parity_passed is False
        and status == "diagnostic_open"
        and bool(interpretation)
    )
    equal_arc_keys = (
        "equal_arc_core_passed",
        "equal_arc_derivative_passed",
        "equal_arc_metric_passed",
        "equal_arc_drift_passed",
    )
    equal_arc_checks = {key: direct.get(key) for key in equal_arc_keys if key in direct}
    checks = {
        "direct_status": status,
        "direct_production_parity_passed": production_parity_passed,
        "direct_interpretation_present": bool(interpretation),
        "direct_tensor_gap_explicitly_scoped": direct_gap_scoped,
        "equal_arc_checks": equal_arc_checks,
        "vmex_boozer_available": _bool(
            _as_dict(bridge.get("vmex_boozer_flux_tube")).get("available")
        ),
        "booz_xform_flux_tube_available": _bool(
            _as_dict(bridge.get("booz_xform_flux_tube")).get("available")
        ),
    }
    blockers: list[str] = []
    if not direct_gap_scoped:
        blockers.append("direct_tensor_parity_gap_not_explicitly_scoped")
    if equal_arc_checks and not all(
        value is True for value in equal_arc_checks.values()
    ):
        blockers.append("bridge_equal_arc_subcheck_failed")
    if not checks["vmex_boozer_available"]:
        blockers.append("vmex_boozer_bridge_unavailable")
    if not checks["booz_xform_flux_tube_available"]:
        blockers.append("booz_xform_flux_tube_bridge_unavailable")
    return checks, blockers


def _nonlinear_fd_checks(audit: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    claim_level = str(audit.get("claim_level", ""))
    production_gate = _bool(audit.get("production_nonlinear_window_gradient_gate"))
    transport_average_gate = _bool(audit.get("transport_average_gate"))
    production_path_gate = _bool(
        audit.get("vmec_boozer_production_nonlinear_observable_fd_path_gate")
    )
    startup_gate = _bool(
        audit.get("vmec_boozer_startup_nonlinear_plumbing_fd_path_gate")
    )
    scoped_startup = "not_transport_average" in claim_level
    checks = {
        "passed": _bool(audit.get("passed")),
        "claim_level": claim_level,
        "startup_plumbing_gate": startup_gate,
        "transport_average_gate": transport_average_gate,
        "production_nonlinear_window_gradient_gate": production_gate,
        "production_nonlinear_observable_fd_path_gate": production_path_gate,
        "claim_scoped_not_transport_average": scoped_startup,
    }
    blockers: list[str] = []
    if not checks["passed"]:
        blockers.append("nonlinear_fd_startup_audit_failed")
    if not startup_gate:
        blockers.append("nonlinear_fd_startup_plumbing_gate_failed")
    if production_gate or transport_average_gate or production_path_gate:
        blockers.append("startup_fd_audit_attempts_production_nonlinear_claim")
    if not scoped_startup:
        blockers.append("nonlinear_fd_claim_not_scoped_to_startup")
    return checks, blockers


def _finite_beta_frequency_checks(
    gate: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    case_name = str(gate.get("case_name", ""))
    kind = str(gate.get("kind", ""))
    source_scope = str(gate.get("source_scope", ""))
    objectives = _as_dict(
        {
            row.get("objective"): row.get("passed")
            for row in _as_rows(gate.get("objective_gates"))
        }
    )
    objective_rel_errors = _as_dict(
        {
            row.get("objective"): row.get("rel_error")
            for row in _as_rows(gate.get("objective_gates"))
        }
    )
    required = REQUIRED_OBJECTIVES_BY_GATE_TYPE["frequency"]
    missing = sorted(required - set(objectives))
    failed = sorted(
        objective for objective in required if objectives.get(objective) is not True
    )
    max_rel_error = _finite_float_or_none(
        _as_dict(gate.get("eigenpair_gate")).get("max_rel_error")
    )
    threshold = MAX_REL_ERROR_BY_GATE_TYPE["frequency"]
    objective_error_failures = {
        objective: _finite_float_or_none(objective_rel_errors.get(objective))
        for objective in required
        if _finite_float_or_none(objective_rel_errors.get(objective)) is None
        or float(_finite_float_or_none(objective_rel_errors.get(objective))) > threshold
    }
    checks = {
        "passed": _bool(gate.get("passed")),
        "case_name": case_name,
        "kind": kind,
        "source_scope": source_scope,
        "mboz": gate.get("mboz"),
        "nboz": gate.get("nboz"),
        "surface_stencil_width": gate.get("surface_stencil_width"),
        "linear_frequency_gradient_gate": _bool(
            gate.get("linear_frequency_gradient_gate")
        ),
        "linear_growth_gradient_gate": _bool(gate.get("linear_growth_gradient_gate")),
        "quasilinear_weight_gradient_gate": _bool(
            gate.get("quasilinear_weight_gradient_gate")
        ),
        "nonlinear_window_gradient_gate": _bool(
            gate.get("nonlinear_window_gradient_gate")
        ),
        "required_objectives": sorted(required),
        "missing_objectives": missing,
        "failed_objectives": failed,
        "max_rel_error": max_rel_error,
        "max_rel_error_threshold": threshold,
        "objective_error_failures": objective_error_failures,
    }
    blockers: list[str] = []
    if not checks["passed"] or not checks["linear_frequency_gradient_gate"]:
        blockers.append("finite_beta_frequency_gate_failed")
    if case_name != "shaped_tokamak_pressure":
        blockers.append("finite_beta_frequency_gate_wrong_case")
    if kind != "mode21_vmec_boozer_linear_frequency_gradient_gate":
        blockers.append("finite_beta_frequency_gate_wrong_kind")
    if source_scope != REQUIRED_SOURCE_SCOPE:
        blockers.append("finite_beta_frequency_gate_wrong_source_scope")
    if (
        min(float(gate.get("mboz", 0.0)), float(gate.get("nboz", 0.0)))
        < MINIMUM_BOOZER_MODE_COUNT
    ):
        blockers.append("finite_beta_frequency_gate_mode_floor_failed")
    if missing:
        blockers.append("finite_beta_frequency_gate_missing_objective")
    if failed:
        blockers.append("finite_beta_frequency_gate_objective_failed")
    if max_rel_error is None or max_rel_error > threshold or objective_error_failures:
        blockers.append("finite_beta_frequency_gate_error_threshold_failed")
    if (
        checks["quasilinear_weight_gradient_gate"]
        or checks["nonlinear_window_gradient_gate"]
    ):
        blockers.append("finite_beta_frequency_gate_attempts_transport_gradient_claim")
    return checks, blockers


def _finite_beta_quasilinear_checks(
    gate: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    case_name = str(gate.get("case_name", ""))
    kind = str(gate.get("kind", ""))
    source_scope = str(gate.get("source_scope", ""))
    objectives = _as_dict(
        {
            row.get("objective"): row.get("passed")
            for row in _as_rows(gate.get("objective_gates"))
        }
    )
    objective_rel_errors = _as_dict(
        {
            row.get("objective"): row.get("rel_error")
            for row in _as_rows(gate.get("objective_gates"))
        }
    )
    required = REQUIRED_OBJECTIVES_BY_GATE_TYPE["quasilinear"]
    missing = sorted(required - set(objectives))
    failed = sorted(
        objective for objective in required if objectives.get(objective) is not True
    )
    max_rel_error = _finite_float_or_none(
        _as_dict(gate.get("eigenpair_gate")).get("max_rel_error")
    )
    threshold = MAX_REL_ERROR_BY_GATE_TYPE["quasilinear"]
    objective_error_failures = {
        objective: _finite_float_or_none(objective_rel_errors.get(objective))
        for objective in required
        if _finite_float_or_none(objective_rel_errors.get(objective)) is None
        or float(_finite_float_or_none(objective_rel_errors.get(objective))) > threshold
    }
    checks = {
        "passed": _bool(gate.get("passed")),
        "case_name": case_name,
        "kind": kind,
        "source_scope": source_scope,
        "mboz": gate.get("mboz"),
        "nboz": gate.get("nboz"),
        "surface_stencil_width": gate.get("surface_stencil_width"),
        "linear_frequency_gradient_gate": _bool(
            gate.get("linear_frequency_gradient_gate")
        ),
        "linear_growth_gradient_gate": _bool(gate.get("linear_growth_gradient_gate")),
        "quasilinear_weight_gradient_gate": _bool(
            gate.get("quasilinear_weight_gradient_gate")
        ),
        "nonlinear_window_gradient_gate": _bool(
            gate.get("nonlinear_window_gradient_gate")
        ),
        "required_objectives": sorted(required),
        "missing_objectives": missing,
        "failed_objectives": failed,
        "max_rel_error": max_rel_error,
        "max_rel_error_threshold": threshold,
        "objective_error_failures": objective_error_failures,
    }
    blockers: list[str] = []
    if not checks["passed"] or not checks["quasilinear_weight_gradient_gate"]:
        blockers.append("finite_beta_quasilinear_gate_failed")
    if case_name != "shaped_tokamak_pressure":
        blockers.append("finite_beta_quasilinear_gate_wrong_case")
    if kind != "mode21_vmec_boozer_quasilinear_gradient_gate":
        blockers.append("finite_beta_quasilinear_gate_wrong_kind")
    if source_scope != REQUIRED_SOURCE_SCOPE:
        blockers.append("finite_beta_quasilinear_gate_wrong_source_scope")
    if (
        min(float(gate.get("mboz", 0.0)), float(gate.get("nboz", 0.0)))
        < MINIMUM_BOOZER_MODE_COUNT
    ):
        blockers.append("finite_beta_quasilinear_gate_mode_floor_failed")
    if missing:
        blockers.append("finite_beta_quasilinear_gate_missing_objective")
    if failed:
        blockers.append("finite_beta_quasilinear_gate_objective_failed")
    if max_rel_error is None or max_rel_error > threshold or objective_error_failures:
        blockers.append("finite_beta_quasilinear_gate_error_threshold_failed")
    if checks["nonlinear_window_gradient_gate"]:
        blockers.append(
            "finite_beta_quasilinear_gate_attempts_nonlinear_gradient_claim"
        )
    return checks, blockers


def _finite_beta_nonlinear_window_checks(
    gate: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    case_name = str(gate.get("case_name", ""))
    kind = str(gate.get("kind", ""))
    source_scope = str(gate.get("source_scope", ""))
    objectives = _as_dict(
        {
            row.get("objective"): row.get("passed")
            for row in _as_rows(gate.get("objective_gates"))
        }
    )
    objective_rel_errors = _as_dict(
        {
            row.get("objective"): row.get("rel_error")
            for row in _as_rows(gate.get("objective_gates"))
        }
    )
    required = REQUIRED_OBJECTIVES_BY_GATE_TYPE["nonlinear-window estimator"]
    missing = sorted(required - set(objectives))
    failed = sorted(
        objective for objective in required if objectives.get(objective) is not True
    )
    max_rel_error = _finite_float_or_none(
        _as_dict(gate.get("eigenpair_gate")).get("max_rel_error")
    )
    threshold = MAX_REL_ERROR_BY_GATE_TYPE["nonlinear-window estimator"]
    objective_error_failures = {
        objective: _finite_float_or_none(objective_rel_errors.get(objective))
        for objective in required
        if _finite_float_or_none(objective_rel_errors.get(objective)) is None
        or float(_finite_float_or_none(objective_rel_errors.get(objective))) > threshold
    }
    checks = {
        "passed": _bool(gate.get("passed")),
        "case_name": case_name,
        "kind": kind,
        "source_scope": source_scope,
        "mboz": gate.get("mboz"),
        "nboz": gate.get("nboz"),
        "surface_stencil_width": gate.get("surface_stencil_width"),
        "linear_frequency_gradient_gate": _bool(
            gate.get("linear_frequency_gradient_gate")
        ),
        "linear_growth_gradient_gate": _bool(gate.get("linear_growth_gradient_gate")),
        "quasilinear_weight_gradient_gate": _bool(
            gate.get("quasilinear_weight_gradient_gate")
        ),
        "nonlinear_window_gradient_gate": _bool(
            gate.get("nonlinear_window_gradient_gate")
        ),
        "production_nonlinear_window_gradient_gate": _bool(
            gate.get("production_nonlinear_window_gradient_gate")
        ),
        "required_objectives": sorted(required),
        "missing_objectives": missing,
        "failed_objectives": failed,
        "max_rel_error": max_rel_error,
        "max_rel_error_threshold": threshold,
        "objective_error_failures": objective_error_failures,
    }
    blockers: list[str] = []
    if not checks["passed"] or not checks["nonlinear_window_gradient_gate"]:
        blockers.append("finite_beta_nonlinear_window_gate_failed")
    if case_name != "shaped_tokamak_pressure":
        blockers.append("finite_beta_nonlinear_window_gate_wrong_case")
    if kind != "mode21_vmec_boozer_nonlinear_window_gradient_gate":
        blockers.append("finite_beta_nonlinear_window_gate_wrong_kind")
    if source_scope != REQUIRED_SOURCE_SCOPE:
        blockers.append("finite_beta_nonlinear_window_gate_wrong_source_scope")
    if (
        min(float(gate.get("mboz", 0.0)), float(gate.get("nboz", 0.0)))
        < MINIMUM_BOOZER_MODE_COUNT
    ):
        blockers.append("finite_beta_nonlinear_window_gate_mode_floor_failed")
    if missing:
        blockers.append("finite_beta_nonlinear_window_gate_missing_objective")
    if failed:
        blockers.append("finite_beta_nonlinear_window_gate_objective_failed")
    if max_rel_error is None or max_rel_error > threshold or objective_error_failures:
        blockers.append("finite_beta_nonlinear_window_gate_error_threshold_failed")
    if checks["production_nonlinear_window_gradient_gate"]:
        blockers.append(
            "finite_beta_nonlinear_window_gate_attempts_production_transport_claim"
        )
    return checks, blockers


def build_vmec_boozer_differentiability_claim_guard(
    root: Path = REPO_ROOT,
    *,
    parity_matrix_path: str = DEFAULT_PARITY_MATRIX,
    gradient_matrix_path: str = DEFAULT_GRADIENT_MATRIX,
    bridge_path: str = DEFAULT_BRIDGE,
    nonlinear_fd_audit_path: str = DEFAULT_NONLINEAR_FD_AUDIT,
    finite_beta_frequency_gate_path: str = DEFAULT_FINITE_BETA_FREQUENCY_GATE,
    finite_beta_quasilinear_gate_path: str = DEFAULT_FINITE_BETA_QUASILINEAR_GATE,
    finite_beta_nonlinear_window_gate_path: str = DEFAULT_FINITE_BETA_NONLINEAR_WINDOW_GATE,
) -> dict[str, Any]:
    """Return a machine-checkable VMEC/Boozer differentiability claim guard."""

    root = root.resolve()
    artifacts = {
        "parity_matrix": parity_matrix_path,
        "gradient_holdout_matrix": gradient_matrix_path,
        "differentiable_geometry_bridge": bridge_path,
        "nonlinear_fd_audit": nonlinear_fd_audit_path,
        "finite_beta_frequency_gate": finite_beta_frequency_gate_path,
        "finite_beta_quasilinear_gate": finite_beta_quasilinear_gate_path,
        "finite_beta_nonlinear_window_gate": finite_beta_nonlinear_window_gate_path,
    }
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    try:
        parity = _read_json(root, parity_matrix_path)
        checks["parity_matrix"], parity_blockers = _parity_matrix_checks(parity)
        blockers.extend(parity_blockers)
    except ValueError as exc:
        checks["parity_matrix"] = {"error": str(exc)}
        blockers.append("parity_matrix_unreadable")

    try:
        gradient = _read_json(root, gradient_matrix_path)
        checks["gradient_holdout_matrix"], gradient_blockers = _gradient_matrix_checks(
            gradient
        )
        blockers.extend(gradient_blockers)
    except ValueError as exc:
        checks["gradient_holdout_matrix"] = {"error": str(exc)}
        blockers.append("gradient_holdout_matrix_unreadable")

    try:
        bridge = _read_json(root, bridge_path)
        checks["differentiable_geometry_bridge"], bridge_blockers = _bridge_checks(
            bridge
        )
        blockers.extend(bridge_blockers)
    except ValueError as exc:
        checks["differentiable_geometry_bridge"] = {"error": str(exc)}
        blockers.append("differentiable_geometry_bridge_unreadable")

    try:
        nonlinear_fd = _read_json(root, nonlinear_fd_audit_path)
        checks["nonlinear_fd_audit"], nonlinear_fd_blockers = _nonlinear_fd_checks(
            nonlinear_fd
        )
        blockers.extend(nonlinear_fd_blockers)
    except ValueError as exc:
        checks["nonlinear_fd_audit"] = {"error": str(exc)}
        blockers.append("nonlinear_fd_audit_unreadable")

    try:
        finite_beta_frequency = _read_json(root, finite_beta_frequency_gate_path)
        checks["finite_beta_frequency_gate"], finite_beta_blockers = (
            _finite_beta_frequency_checks(finite_beta_frequency)
        )
        blockers.extend(finite_beta_blockers)
    except ValueError as exc:
        checks["finite_beta_frequency_gate"] = {"error": str(exc)}
        blockers.append("finite_beta_frequency_gate_unreadable")

    try:
        finite_beta_quasilinear = _read_json(root, finite_beta_quasilinear_gate_path)
        checks["finite_beta_quasilinear_gate"], finite_beta_ql_blockers = (
            _finite_beta_quasilinear_checks(finite_beta_quasilinear)
        )
        blockers.extend(finite_beta_ql_blockers)
    except ValueError as exc:
        checks["finite_beta_quasilinear_gate"] = {"error": str(exc)}
        blockers.append("finite_beta_quasilinear_gate_unreadable")

    try:
        finite_beta_nonlinear_window = _read_json(
            root, finite_beta_nonlinear_window_gate_path
        )
        (
            checks["finite_beta_nonlinear_window_gate"],
            finite_beta_nl_blockers,
        ) = _finite_beta_nonlinear_window_checks(finite_beta_nonlinear_window)
        blockers.extend(finite_beta_nl_blockers)
    except ValueError as exc:
        checks["finite_beta_nonlinear_window_gate"] = {"error": str(exc)}
        blockers.append("finite_beta_nonlinear_window_gate_unreadable")

    unique_blockers = sorted(set(blockers))
    return {
        "kind": "vmec_boozer_differentiability_claim_guard",
        "claim_level": (
            "reduced_full_chain_differentiability_guard_not_full_nonlinear_transport_optimization"
        ),
        "passed": not unique_blockers,
        "blockers": unique_blockers,
        "minimum_boozer_mode_count": MINIMUM_BOOZER_MODE_COUNT,
        "minimum_gradient_cases": MINIMUM_GRADIENT_CASES,
        "artifacts": artifacts,
        "checks": checks,
        "scope": (
            "Passes only the release-level reduced differentiability claim: "
            "VMEC/Boozer equal-arc parity and AD/finite-difference gradients for "
            "linear, quasilinear, and nonlinear-window estimator objectives, plus "
            "finite-beta shaped-pressure eigenfrequency, quasilinear, and reduced "
            "nonlinear-window estimator gradients. "
            "Direct VMEC tensor-vs-imported-EIK parity and converged nonlinear "
            "transport-gradient optimization remain explicitly "
            "out of scope."
        ),
        "root": str(root),
    }


def build_differentiability_claim_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--parity-matrix", default=DEFAULT_PARITY_MATRIX)
    parser.add_argument("--gradient-matrix", default=DEFAULT_GRADIENT_MATRIX)
    parser.add_argument("--bridge", default=DEFAULT_BRIDGE)
    parser.add_argument("--nonlinear-fd-audit", default=DEFAULT_NONLINEAR_FD_AUDIT)
    parser.add_argument(
        "--finite-beta-frequency-gate", default=DEFAULT_FINITE_BETA_FREQUENCY_GATE
    )
    parser.add_argument(
        "--finite-beta-quasilinear-gate", default=DEFAULT_FINITE_BETA_QUASILINEAR_GATE
    )
    parser.add_argument(
        "--finite-beta-nonlinear-window-gate",
        default=DEFAULT_FINITE_BETA_NONLINEAR_WINDOW_GATE,
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    return parser


def main_differentiability_claim(argv: list[str] | None = None) -> int:
    args = build_differentiability_claim_parser().parse_args(argv)
    root = args.root.resolve()
    out_json = args.out_json
    if out_json is not None and not out_json.is_absolute():
        out_json = root / out_json
    report = build_vmec_boozer_differentiability_claim_guard(
        root=root,
        parity_matrix_path=args.parity_matrix,
        gradient_matrix_path=args.gradient_matrix,
        bridge_path=args.bridge,
        nonlinear_fd_audit_path=args.nonlinear_fd_audit,
        finite_beta_frequency_gate_path=args.finite_beta_frequency_gate,
        finite_beta_quasilinear_gate_path=args.finite_beta_quasilinear_gate,
        finite_beta_nonlinear_window_gate_path=args.finite_beta_nonlinear_window_gate,
    )
    if out_json is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote {_path_rel(root, out_json)}")
    if not report["passed"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    return 0


# ---- aggregate holdout promotion gate ----

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGGREGATE_ARTIFACT = (
    ROOT / "docs/_static/vmec_boozer_aggregate_objective_gate.json"
)
DEFAULT_LINE_SEARCH_ARTIFACT = (
    ROOT / "docs/_static/vmec_boozer_aggregate_line_search_gate.json"
)
DEFAULT_NONLINEAR_ENSEMBLE_ARTIFACTS: tuple[Path, ...] = ()

NON_PROMOTABLE_CLAIM_MARKERS = (
    "not_transport",
    "not transport",
    "not_production",
    "not production",
    "not a nonlinear",
    "startup",
    "reduced",
    "plumbing",
    "exploratory",
    "feasibility",
    "pending",
    "negative",
)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _artifact_passed(payload: dict[str, Any]) -> bool:
    if bool(payload.get("passed", False)):
        return True
    if bool(payload.get("gate_passed", False)):
        return True
    for key in ("promotion_gate", "gate_report"):
        nested = payload.get(key)
        if isinstance(nested, dict) and bool(nested.get("passed", False)):
            return True
    return False


def _claim_scope_blocks_promotion(payload: dict[str, Any]) -> list[str]:
    claim_text = " ".join(
        str(payload.get(key, ""))
        for key in ("claim_level", "claim_scope", "notes", "next_action")
    ).lower()
    blockers = [
        marker for marker in NON_PROMOTABLE_CLAIM_MARKERS if marker in claim_text
    ]
    if payload.get("transport_average_gate") is False:
        blockers.append("transport_average_gate_false")
    return sorted(set(blockers))


def _samples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("samples", "holdout_samples", "heldout_samples", "validation_samples"):
        raw = payload.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    return []


def _is_nonlinear_window_ensemble(payload: dict[str, Any]) -> bool:
    kind = str(payload.get("kind", "")).strip().lower()
    claim_level = str(payload.get("claim_level", "")).strip().lower()
    if kind == "nonlinear_window_ensemble_readiness_manifest":
        return False
    return kind == "nonlinear_window_ensemble_report" or (
        "replicated_nonlinear_window" in claim_level
        and "manifest_blocks_promotion" not in claim_level
    )


def _is_nonlinear_window_readiness_manifest(payload: dict[str, Any]) -> bool:
    return str(payload.get("kind", "")).strip().lower() == (
        "nonlinear_window_ensemble_readiness_manifest"
    )


def _alpha(sample: dict[str, Any]) -> float | None:
    value = sample.get("alpha")
    if value is None:
        return None
    try:
        alpha = float(value)
    except (TypeError, ValueError):
        return None
    return alpha if math.isfinite(alpha) else None


def _sample_identity(sample: dict[str, Any]) -> tuple[str, str, str]:
    surface = sample.get("surface_index")
    ky = sample.get("selected_ky_index")
    alpha = _alpha(sample)
    alpha_key = "" if alpha is None else f"{alpha:.16g}"
    return (str(surface), alpha_key, str(ky))


def _sample_set(payload: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {_sample_identity(sample) for sample in _samples(payload)}


def _has_heldout_surface_or_field_line(
    training_samples: list[dict[str, Any]],
    holdout_samples: list[dict[str, Any]],
    *,
    alpha_atol: float,
) -> tuple[bool, str]:
    training_surfaces = {sample.get("surface_index") for sample in training_samples}
    training_alphas = [
        alpha for sample in training_samples if (alpha := _alpha(sample)) is not None
    ]
    for sample in holdout_samples:
        surface = sample.get("surface_index")
        if surface is not None and surface not in training_surfaces:
            return True, f"held-out surface_index={surface}"
        alpha = _alpha(sample)
        if alpha is not None and all(
            abs(alpha - item) > alpha_atol for item in training_alphas
        ):
            return True, f"held-out field-line alpha={alpha:.16g}"
    return False, "no passed holdout sample changes surface_index or field-line alpha"


def _gate(metric: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"metric": metric, "passed": bool(passed), "detail": detail}


def check_vmec_boozer_aggregate_holdout_gate(
    *,
    aggregate_artifact: str | Path = DEFAULT_AGGREGATE_ARTIFACT,
    line_search_artifact: str | Path = DEFAULT_LINE_SEARCH_ARTIFACT,
    holdout_artifacts: tuple[str | Path, ...] = (),
    nonlinear_ensemble_artifacts: tuple[
        str | Path, ...
    ] = DEFAULT_NONLINEAR_ENSEMBLE_ARTIFACTS,
    alpha_atol: float = 1.0e-12,
) -> dict[str, Any]:
    """Return a JSON-ready promotion gate for aggregate optimization artifacts."""

    aggregate_path = Path(aggregate_artifact)
    line_search_path = Path(line_search_artifact)
    aggregate = _load_json_object(aggregate_path)
    line_search = _load_json_object(line_search_path)
    training_samples = _samples(aggregate)
    line_search_same_samples = bool(training_samples) and _sample_set(
        aggregate
    ) == _sample_set(line_search)

    holdout_rows: list[dict[str, Any]] = []
    qualifying_holdout_reasons: list[str] = []
    for raw_path in holdout_artifacts:
        path = Path(raw_path)
        payload = _load_json_object(path)
        samples = _samples(payload)
        passed = _artifact_passed(payload)
        scope_blockers = _claim_scope_blocks_promotion(payload)
        has_holdout_sample, reason = _has_heldout_surface_or_field_line(
            training_samples,
            samples,
            alpha_atol=alpha_atol,
        )
        qualifies = bool(passed and not scope_blockers and has_holdout_sample)
        if qualifies:
            qualifying_holdout_reasons.append(f"{_repo_relative(path)}: {reason}")
        holdout_rows.append(
            {
                "path": _repo_relative(path),
                "passed": passed,
                "claim_scope_blockers": scope_blockers,
                "n_samples": len(samples),
                "heldout_surface_or_field_line": has_holdout_sample,
                "heldout_reason": reason,
                "qualifies_for_promotion": qualifies,
            }
        )

    nonlinear_ensemble_rows: list[dict[str, Any]] = []
    qualifying_ensemble_reasons: list[str] = []
    for raw_path in nonlinear_ensemble_artifacts:
        path = Path(raw_path)
        payload = _load_json_object(path)
        passed = _artifact_passed(payload)
        is_ensemble = _is_nonlinear_window_ensemble(payload)
        is_readiness_manifest = _is_nonlinear_window_readiness_manifest(payload)
        promotion_gate = payload.get("promotion_gate")
        readiness_blockers = (
            list(promotion_gate.get("blockers", []))
            if isinstance(promotion_gate, dict)
            else []
        )
        qualifies = bool(passed and is_ensemble)
        if qualifies:
            qualifying_ensemble_reasons.append(_repo_relative(path))
        nonlinear_ensemble_rows.append(
            {
                "path": _repo_relative(path),
                "passed": passed,
                "is_nonlinear_window_ensemble": is_ensemble,
                "is_nonlinear_window_readiness_manifest": is_readiness_manifest,
                "claim_level": str(payload.get("claim_level", "")),
                "readiness_blockers": readiness_blockers,
                "missing_artifacts": payload.get("missing_artifacts", [])
                if is_readiness_manifest
                else [],
                "qualifies_for_production_nonlinear_promotion": qualifies,
            }
        )

    gates = [
        _gate(
            "aggregate_finite_difference_artifact_passed",
            _artifact_passed(aggregate),
            _repo_relative(aggregate_path),
        ),
        _gate(
            "aggregate_line_search_artifact_passed",
            _artifact_passed(line_search),
            _repo_relative(line_search_path),
        ),
        _gate(
            "line_search_reuses_aggregate_sample_set",
            line_search_same_samples,
            "line-search samples must match the aggregate objective samples",
        ),
        _gate(
            "passed_holdout_surface_or_field_line_artifact",
            bool(qualifying_holdout_reasons),
            "; ".join(qualifying_holdout_reasons)
            if qualifying_holdout_reasons
            else "provide a passed production-scope holdout artifact with a new surface_index or alpha",
        ),
        _gate(
            "passed_replicated_nonlinear_window_ensemble",
            bool(qualifying_ensemble_reasons),
            "; ".join(qualifying_ensemble_reasons)
            if qualifying_ensemble_reasons
            else "provide a passed replicated nonlinear-window ensemble artifact before any production nonlinear optimized-equilibrium claim",
        ),
    ]
    blockers = [gate["metric"] for gate in gates if not bool(gate["passed"])]
    passed = not blockers
    return {
        "kind": "vmec_boozer_aggregate_holdout_promotion_gate",
        "claim_level": (
            "aggregate_optimization_promotion_requires_heldout_surface_or_field_line_validation"
        ),
        "passed": passed,
        "promotion_gate": {
            "passed": passed,
            "blockers": blockers,
            "requirements": [
                "aggregate finite-difference artifact passes",
                "aggregate line-search artifact passes on the same sample set",
                "at least one passed production-scope validation artifact covers a held-out surface_index or field-line alpha",
                "at least one passed replicated nonlinear-window ensemble artifact supports the post-transient transport mean and uncertainty",
                "k_y-only holdouts do not satisfy the surface/field-line requirement",
            ],
        },
        "gates": gates,
        "training_sample_summary": {
            "n_samples": len(training_samples),
            "surfaces": sorted(
                {str(sample.get("surface_index")) for sample in training_samples}
            ),
            "alphas": sorted(
                {
                    f"{alpha:.16g}"
                    for sample in training_samples
                    if (alpha := _alpha(sample)) is not None
                }
            ),
            "selected_ky_indices": sorted(
                {str(sample.get("selected_ky_index")) for sample in training_samples}
            ),
        },
        "holdout_artifacts": holdout_rows,
        "nonlinear_ensemble_artifacts": nonlinear_ensemble_rows,
        "notes": (
            "This check gates claim promotion only. Passing aggregate reduced-objective "
            "FD and line-search artifacts proves optimizer plumbing; it does not by "
            "itself validate optimized-equilibrium nonlinear transport. Promotion "
            "requires independent held-out surface or field-line evidence plus "
            "replicated nonlinear-window uncertainty evidence."
        ),
    }


def build_aggregate_holdout_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-artifact", type=Path, default=DEFAULT_AGGREGATE_ARTIFACT
    )
    parser.add_argument(
        "--line-search-artifact", type=Path, default=DEFAULT_LINE_SEARCH_ARTIFACT
    )
    parser.add_argument("--holdout-artifact", action="append", type=Path, default=[])
    parser.add_argument(
        "--nonlinear-ensemble-artifact", action="append", type=Path, default=[]
    )
    parser.add_argument("--alpha-atol", type=float, default=1.0e-12)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return non-zero when the promotion gate is blocked.",
    )
    return parser


def main_aggregate_holdout(argv: list[str] | None = None) -> int:
    args = build_aggregate_holdout_parser().parse_args(argv)
    report = check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=args.aggregate_artifact,
        line_search_artifact=args.line_search_artifact,
        holdout_artifacts=tuple(args.holdout_artifact),
        nonlinear_ensemble_artifacts=tuple(args.nonlinear_ensemble_artifact),
        alpha_atol=args.alpha_atol,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"saved {args.json_out}")
    else:
        print(text)
    if args.fail_on_blocked and not bool(report["passed"]):
        print(
            "VMEC/Boozer aggregate optimization promotion blocked: "
            + ", ".join(report["promotion_gate"]["blockers"]),
            file=sys.stderr,
        )
        return 1
    return 0


# ---- reduced portfolio artifact gate ----

ROOT = Path(__file__).resolve().parents[2]


DEFAULT_REDUCED_PORTFOLIO_ROW_ARTIFACT = (
    ROOT / "docs" / "_static" / "vmec_boozer_aggregate_objective_gate.json"
)
DEFAULT_REDUCED_PORTFOLIO_GRADIENT_ARTIFACTS = (
    ROOT / "docs" / "_static" / "vmec_boozer_quasilinear_gradient_gate.json",
    ROOT / "docs" / "_static" / "vmec_boozer_solver_frequency_gradient_gate.json",
)
DEFAULT_REDUCED_PORTFOLIO_OUT = (
    ROOT / "docs" / "_static" / "vmec_boozer_reduced_portfolio_guard.json"
)


def _json_clean(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _read_json_object_path(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _reduced_portfolio_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


def _reduced_portfolio_guard_helpers():
    src = ROOT / "src"
    for path in (src, ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from gkx.objectives.portfolio_guard import (
        ReducedPortfolioArtifactGuardConfig,
        reduced_portfolio_artifact_guard_report,
    )

    return ReducedPortfolioArtifactGuardConfig, reduced_portfolio_artifact_guard_report


def build_vmec_boozer_reduced_portfolio_guard_payload(
    *,
    row_artifact: str | Path = DEFAULT_REDUCED_PORTFOLIO_ROW_ARTIFACT,
    gradient_artifacts: list[str | Path] | tuple[str | Path, ...] = (
        DEFAULT_REDUCED_PORTFOLIO_GRADIENT_ARTIFACTS
    ),
    min_alphas: int = 2,
    min_ky: int = 2,
    min_objectives: int = 1,
    min_boozer_mode: int = 21,
    value_rtol: float = 1.0e-8,
    value_atol: float = 1.0e-8,
) -> dict[str, object]:
    """Return the VMEC/Boozer reduced-portfolio promotion guard payload."""

    (
        ReducedPortfolioArtifactGuardConfig,
        reduced_portfolio_artifact_guard_report,
    ) = _reduced_portfolio_guard_helpers()
    row_path = _reduced_portfolio_path(row_artifact)
    gradient_paths = [_reduced_portfolio_path(path) for path in gradient_artifacts]
    row_payload = _read_json_object_path(row_path)
    gradient_payloads = [_read_json_object_path(path) for path in gradient_paths]
    config = ReducedPortfolioArtifactGuardConfig(
        min_alphas=int(min_alphas),
        min_ky=int(min_ky),
        min_objectives=int(min_objectives),
        min_boozer_mode=int(min_boozer_mode),
        value_rtol=float(value_rtol),
        value_atol=float(value_atol),
    )
    report = reduced_portfolio_artifact_guard_report(
        row_payload,
        gradient_artifacts=gradient_payloads,
        config=config,
    )
    report["row_artifact"] = _path_rel(ROOT, row_path)
    report["gradient_artifacts"] = [_path_rel(ROOT, path) for path in gradient_paths]
    return report


def write_vmec_boozer_reduced_portfolio_guard_artifact(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_REDUCED_PORTFOLIO_OUT,
) -> str:
    """Write the guard JSON artifact."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(out_path)


def build_reduced_portfolio_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row-artifact", type=Path, default=DEFAULT_REDUCED_PORTFOLIO_ROW_ARTIFACT
    )
    parser.add_argument(
        "--gradient-artifact",
        type=Path,
        action="append",
        default=None,
        help="VMEC/Boozer gradient artifact with finite implicit/FD objective gates.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_REDUCED_PORTFOLIO_OUT)
    parser.add_argument("--min-alphas", type=int, default=2)
    parser.add_argument("--min-ky", type=int, default=2)
    parser.add_argument("--min-objectives", type=int, default=2)
    parser.add_argument("--min-boozer-mode", type=int, default=21)
    parser.add_argument("--value-rtol", type=float, default=1.0e-6)
    parser.add_argument("--value-atol", type=float, default=1.0e-6)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main_reduced_portfolio_guard(argv: list[str] | None = None) -> int:
    args = build_reduced_portfolio_parser().parse_args(argv)
    gradient_artifacts = (
        tuple(args.gradient_artifact)
        if args.gradient_artifact is not None
        else DEFAULT_REDUCED_PORTFOLIO_GRADIENT_ARTIFACTS
    )
    payload = build_vmec_boozer_reduced_portfolio_guard_payload(
        row_artifact=args.row_artifact,
        gradient_artifacts=gradient_artifacts,
        min_alphas=args.min_alphas,
        min_ky=args.min_ky,
        min_objectives=args.min_objectives,
        min_boozer_mode=args.min_boozer_mode,
        value_rtol=args.value_rtol,
        value_atol=args.value_atol,
    )
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
    else:
        print(write_vmec_boozer_reduced_portfolio_guard_artifact(payload, out=args.out))
    return 0 if bool(payload.get("passed", False)) else 1


HIGH_GRID_ADMISSION_DEFAULT_OUT = (
    REPO_ROOT / "docs" / "_static" / "external_vmec_high_grid_admission_gate.json"
)
DEFAULT_ALLOWED_FULL_GRID_FAILURES = {
    "common_window_pairwise_heat_flux_symmetric_relative_difference",
    "least_window_pairwise_heat_flux_symmetric_relative_difference",
}
DEFAULT_MAX_PAIRWISE_RELATIVE_DIFFERENCE = 0.15
DEFAULT_MAX_TIME_HORIZON_RELATIVE_CHANGE = 0.15
DEFAULT_MAX_MEAN_REL_SPREAD = 0.15
DEFAULT_MAX_COMBINED_SEM_REL = 0.25
DEFAULT_MIN_REPLICATES = 3
DEFAULT_VALUE_FLOOR = 1.0e-12


def _repo_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _high_grid_json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _high_grid_json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_high_grid_json_clean(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _load_json(path: Path, *, expected_kind: str | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    if expected_kind is not None and payload.get("kind") != expected_kind:
        raise ValueError(
            f"{path} has kind {payload.get('kind')!r}; expected {expected_kind!r}"
        )
    return payload


def _failed_metrics(payload: dict[str, Any]) -> set[str]:
    report = payload.get("gate_report", {})
    gates = report.get("gates", []) if isinstance(report, dict) else []
    return {
        str(gate.get("metric", "unknown"))
        for gate in gates
        if isinstance(gate, dict) and not bool(gate.get("passed", False))
    }


def _grid_labels(payload: dict[str, Any]) -> list[str]:
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        return []
    return [str(run.get("label", "")) for run in runs if isinstance(run, dict)]


def _threshold(payload: dict[str, Any], key: str, default: float) -> float:
    thresholds = payload.get("thresholds", {})
    if isinstance(thresholds, dict):
        try:
            return float(thresholds.get(key, default))
        except (TypeError, ValueError):
            return float(default)
    return float(default)


def _metric(
    payload: dict[str, Any], path: tuple[str, ...], default: float = float("inf")
) -> float:
    item: Any = payload
    for key in path:
        if not isinstance(item, dict) or key not in item:
            return float(default)
        item = item[key]
    try:
        out = float(item)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def build_high_grid_admission_payload(
    *,
    full_grid_gate_path: Path,
    high_grid_gate_paths: list[Path],
    time_horizon_gate_path: Path,
    replicate_ensemble_path: Path,
    excluded_grid_labels: list[str],
    retained_grid_labels: list[str],
    case: str = "External-VMEC high-grid admission",
    allowed_full_grid_failures: set[str] | None = None,
    min_replicates: int = DEFAULT_MIN_REPLICATES,
    value_floor: float = DEFAULT_VALUE_FLOOR,
) -> dict[str, Any]:
    from gkx.diagnostics.validation_gates import (
        evaluate_scalar_gate,
        gate_report,
        gate_report_to_dict,
    )

    """Return a JSON-ready scoped high-grid admission report."""

    if not high_grid_gate_paths:
        raise ValueError("at least one high-grid gate is required")
    if not excluded_grid_labels:
        raise ValueError("at least one excluded coarse-grid label is required")
    if len(retained_grid_labels) < 2:
        raise ValueError("at least two retained high-grid labels are required")
    if min_replicates < 2:
        raise ValueError("min_replicates must be at least 2")
    if value_floor < 0.0:
        raise ValueError("value_floor must be non-negative")

    allowed_failures = set(
        allowed_full_grid_failures or DEFAULT_ALLOWED_FULL_GRID_FAILURES
    )
    full_grid = _load_json(
        full_grid_gate_path,
        expected_kind="external_vmec_nonlinear_grid_convergence_gate",
    )
    high_grid_gates = [
        _load_json(path, expected_kind="external_vmec_nonlinear_grid_convergence_gate")
        for path in high_grid_gate_paths
    ]
    time_horizon = _load_json(
        time_horizon_gate_path, expected_kind="external_vmec_time_horizon_gate"
    )
    replicate = _load_json(replicate_ensemble_path)
    if replicate.get("kind") not in {
        "nonlinear_window_ensemble_report",
        "nonlinear_window_ensemble_gate",
    }:
        raise ValueError(
            f"{replicate_ensemble_path} is not a nonlinear-window ensemble gate"
        )

    full_labels = set(_grid_labels(full_grid))
    missing_excluded = sorted(set(excluded_grid_labels) - full_labels)
    missing_retained = sorted(set(retained_grid_labels) - full_labels)
    unexpected_full_failures = sorted(_failed_metrics(full_grid) - allowed_failures)
    high_grid_failed_count = sum(
        0 if bool(gate.get("passed", False)) else 1 for gate in high_grid_gates
    )
    high_grid_label_mismatch_count = sum(
        0 if set(_grid_labels(gate)) == set(retained_grid_labels) else 1
        for gate in high_grid_gates
    )

    max_high_grid_common_diff = max(
        _metric(
            gate,
            ("common_window", "max_pairwise_heat_flux_symmetric_relative_difference"),
        )
        for gate in high_grid_gates
    )
    max_high_grid_least_diff = max(
        _metric(
            gate,
            ("least_windows", "max_pairwise_heat_flux_symmetric_relative_difference"),
        )
        for gate in high_grid_gates
    )
    max_pairwise_threshold = min(
        _threshold(
            gate,
            "max_pairwise_relative_difference",
            DEFAULT_MAX_PAIRWISE_RELATIVE_DIFFERENCE,
        )
        for gate in high_grid_gates
    )

    horizon_common_change = _metric(
        time_horizon, ("common_window_time_horizon_relative_change",)
    )
    horizon_least_change = _metric(
        time_horizon, ("least_window_time_horizon_relative_change",)
    )
    horizon_threshold = _threshold(
        time_horizon,
        "max_relative_change",
        DEFAULT_MAX_TIME_HORIZON_RELATIVE_CHANGE,
    )

    statistics = replicate.get("statistics", {})
    config = replicate.get("config", {})
    if not isinstance(statistics, dict):
        statistics = {}
    if not isinstance(config, dict):
        config = {}
    n_reports = int(statistics.get("n_reports", 0) or 0)
    n_finite = int(statistics.get("n_finite_means", 0) or 0)
    ensemble_mean = float(statistics.get("ensemble_mean", float("nan")))
    mean_rel_spread = float(statistics.get("mean_rel_spread", float("inf")))
    combined_sem_rel = float(statistics.get("combined_sem_rel", float("inf")))
    spread_threshold = float(
        config.get("max_mean_rel_spread", DEFAULT_MAX_MEAN_REL_SPREAD)
    )
    sem_threshold = float(
        config.get("max_combined_sem_rel", DEFAULT_MAX_COMBINED_SEM_REL)
    )

    gates = [
        evaluate_scalar_gate(
            "full_grid_gate_failed_before_coarse_exclusion",
            0.0 if not bool(full_grid.get("passed", False)) else 1.0,
            0.0,
            atol=0.0,
            rtol=0.0,
            notes="High-grid admission is only meaningful as an exception to a failed full-grid ladder.",
        ),
        evaluate_scalar_gate(
            "full_grid_failure_limited_to_grid_difference",
            float(len(unexpected_full_failures)),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="metrics",
            notes="The full-grid gate may fail only grid-difference metrics, not stationarity/sample gates.",
        ),
        evaluate_scalar_gate(
            "excluded_coarse_grid_labels_present",
            float(len(missing_excluded)),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="labels",
            notes="Every excluded grid must be present in the failed full-grid ladder.",
        ),
        evaluate_scalar_gate(
            "retained_high_grid_labels_present",
            float(len(missing_retained)),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="labels",
            notes="The retained high grids must also be present in the failed full-grid ladder.",
        ),
        evaluate_scalar_gate(
            "high_grid_gate_failure_count",
            float(high_grid_failed_count),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="gates",
            notes="Every retained high-grid convergence gate used for admission must pass.",
        ),
        evaluate_scalar_gate(
            "high_grid_label_mismatch_count",
            float(high_grid_label_mismatch_count),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="gates",
            notes="Every high-grid gate must use exactly the retained-grid labels.",
        ),
        evaluate_scalar_gate(
            "high_grid_common_window_pairwise_difference",
            max_high_grid_common_diff,
            0.0,
            atol=max_pairwise_threshold,
            rtol=0.0,
            notes="Highest-grid common-window heat-flux means must agree under the production threshold.",
        ),
        evaluate_scalar_gate(
            "high_grid_least_window_pairwise_difference",
            max_high_grid_least_diff,
            0.0,
            atol=max_pairwise_threshold,
            rtol=0.0,
            notes="Highest-grid least-trending-window heat-flux means must agree under the production threshold.",
        ),
        evaluate_scalar_gate(
            "time_horizon_gate_failure_count",
            0.0 if bool(time_horizon.get("passed", False)) else 1.0,
            0.0,
            atol=0.0,
            rtol=0.0,
            units="gates",
            notes="Late high-grid horizons must be stable before coarse-grid exclusion can be considered.",
        ),
        evaluate_scalar_gate(
            "time_horizon_common_window_change",
            horizon_common_change,
            0.0,
            atol=horizon_threshold,
            rtol=0.0,
            notes="High-grid averaged common-window means must be stable across late final times.",
        ),
        evaluate_scalar_gate(
            "time_horizon_least_window_change",
            horizon_least_change,
            0.0,
            atol=horizon_threshold,
            rtol=0.0,
            notes="High-grid averaged least-window means must be stable across late final times.",
        ),
        evaluate_scalar_gate(
            "replicate_ensemble_gate_failure_count",
            0.0 if bool(replicate.get("passed", False)) else 1.0,
            0.0,
            atol=0.0,
            rtol=0.0,
            units="gates",
            notes="A seed/timestep replicated late transport window must pass.",
        ),
        evaluate_scalar_gate(
            "replicate_count_shortfall",
            float(max(0, min_replicates - n_reports)),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="replicates",
            notes="At least three replicated reports are required for this scoped exception.",
        ),
        evaluate_scalar_gate(
            "finite_replicate_count_shortfall",
            float(max(0, min_replicates - n_finite)),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="replicates",
            notes="Replicate means must be finite.",
        ),
        evaluate_scalar_gate(
            "replicate_mean_relative_spread",
            mean_rel_spread,
            0.0,
            atol=spread_threshold,
            rtol=0.0,
            notes="Seed/timestep spread of late-window means.",
        ),
        evaluate_scalar_gate(
            "replicate_combined_sem_relative",
            combined_sem_rel,
            0.0,
            atol=sem_threshold,
            rtol=0.0,
            notes="Combined uncertainty estimate normalized by the ensemble mean.",
        ),
        evaluate_scalar_gate(
            "nonzero_transport_mean_floor",
            0.0
            if math.isfinite(ensemble_mean) and abs(ensemble_mean) >= value_floor
            else 1.0,
            0.0,
            atol=0.0,
            rtol=0.0,
            notes="Admission must not be based on startup/noise-floor heat flux.",
        ),
    ]
    report = gate_report(case, "external_vmec_high_grid_admission_policy", gates)
    blockers = [gate.metric for gate in report.gates if not gate.passed]
    passed = bool(report.passed)
    return _high_grid_json_clean(
        {
            "kind": "external_vmec_high_grid_admission_gate",
            "case": case,
            "passed": passed,
            "gate_index_include": True,
            "claim_level": (
                "passed_high_grid_transport_holdout_admission_under_coarse_grid_exclusion"
                if passed
                else "blocked_high_grid_transport_holdout_admission"
            ),
            "claim_scope": (
                "The case is eligible as a high-grid nonlinear transport holdout with "
                "the listed coarse grids excluded. This is not a universal absolute-flux "
                "prediction or normal full-ladder convergence claim."
            ),
            "inputs": {
                "full_grid_gate": _repo_relative_path(full_grid_gate_path),
                "high_grid_gates": [
                    _repo_relative_path(path) for path in high_grid_gate_paths
                ],
                "time_horizon_gate": _repo_relative_path(time_horizon_gate_path),
                "replicate_ensemble_gate": _repo_relative_path(replicate_ensemble_path),
            },
            "policy": {
                "excluded_grid_labels": list(excluded_grid_labels),
                "retained_grid_labels": list(retained_grid_labels),
                "allowed_full_grid_failure_metrics": sorted(allowed_failures),
                "min_replicates": int(min_replicates),
                "value_floor": float(value_floor),
                "calibration_use": "eligible_as_scoped_high_grid_holdout"
                if passed
                else "blocked",
                "restrictions": [
                    "do not describe as full n48/n64/n80 convergence",
                    "do not use for universal absolute-flux promotion without separate calibration gates",
                    "retain the coarse-grid failure sidecar with the admitted artifact",
                    "rerun if physics, dissipation, flux-tube, or resolution settings change",
                ],
            },
            "summary": {
                "full_grid_failed_metrics": sorted(_failed_metrics(full_grid)),
                "unexpected_full_grid_failed_metrics": unexpected_full_failures,
                "missing_excluded_grid_labels": missing_excluded,
                "missing_retained_grid_labels": missing_retained,
                "max_high_grid_common_difference": max_high_grid_common_diff,
                "max_high_grid_least_difference": max_high_grid_least_diff,
                "high_grid_pairwise_threshold": max_pairwise_threshold,
                "time_horizon_common_change": horizon_common_change,
                "time_horizon_least_change": horizon_least_change,
                "time_horizon_threshold": horizon_threshold,
                "replicate_n_reports": n_reports,
                "replicate_n_finite_means": n_finite,
                "replicate_ensemble_mean": ensemble_mean,
                "replicate_mean_rel_spread": mean_rel_spread,
                "replicate_combined_sem_rel": combined_sem_rel,
                "replicate_mean_rel_spread_threshold": spread_threshold,
                "replicate_combined_sem_rel_threshold": sem_threshold,
            },
            "literature_policy": {
                "summary": (
                    "Nonlinear gyrokinetic heat-flux admission follows the benchmark "
                    "practice of comparing saturated late-time traces, resolution ladders, "
                    "and uncertainty of time averages. Coarse-grid exclusion is allowed "
                    "only when the retained higher grids, horizon stability, and replicated "
                    "transport windows pass fail-closed gates."
                ),
                "anchors": [
                    {
                        "name": "Dimits et al. 2000 Cyclone nonlinear heat-flux benchmark",
                        "url": "https://doi.org/10.1063/1.873896",
                    },
                    {
                        "name": "Gonzalez-Jerez et al. 2022 W7-X stella/GENE benchmark",
                        "url": "https://doi.org/10.1017/S0022377822000393",
                    },
                    {
                        "name": "Mandell et al. 2024 GX nonlinear convergence and benchmark practice",
                        "url": "https://doi.org/10.1017/S0022377822000617",
                    },
                    {
                        "name": "Hoffmann, Frei & Ricci 2023 nonlinear moment/GK convergence study",
                        "url": "https://arxiv.org/abs/2308.01016",
                    },
                    {
                        "name": "Oberparleiter et al. 2016 uncertainty and stopping rule for nonlinear gyrokinetics",
                        "url": "https://doi.org/10.1063/1.4960039",
                    },
                ],
            },
            "promotion_gate": {
                "passed": passed,
                "blockers": blockers,
                "reason": (
                    "high-grid admission policy passed; eligible for scoped holdout metadata"
                    if passed
                    else "high-grid admission policy failed"
                ),
            },
            "gate_report": gate_report_to_dict(report),
        }
    )


def build_high_grid_admission_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-grid-gate", type=Path, required=True)
    parser.add_argument("--high-grid-gate", type=Path, action="append", required=True)
    parser.add_argument("--time-horizon-gate", type=Path, required=True)
    parser.add_argument("--replicate-ensemble", type=Path, required=True)
    parser.add_argument("--excluded-grid-label", action="append", required=True)
    parser.add_argument("--retained-grid-label", action="append", required=True)
    parser.add_argument("--case", default="External-VMEC high-grid admission")
    parser.add_argument("--min-replicates", type=int, default=DEFAULT_MIN_REPLICATES)
    parser.add_argument("--value-floor", type=float, default=DEFAULT_VALUE_FLOOR)
    parser.add_argument("--out", type=Path, default=HIGH_GRID_ADMISSION_DEFAULT_OUT)
    return parser


def main_high_grid_admission(argv: list[str] | None = None) -> int:
    args = build_high_grid_admission_parser().parse_args(argv)
    payload = build_high_grid_admission_payload(
        full_grid_gate_path=args.full_grid_gate,
        high_grid_gate_paths=args.high_grid_gate,
        time_horizon_gate_path=args.time_horizon_gate,
        replicate_ensemble_path=args.replicate_ensemble,
        excluded_grid_labels=args.excluded_grid_label,
        retained_grid_labels=args.retained_grid_label,
        case=args.case,
        min_replicates=args.min_replicates,
        value_floor=args.value_floor,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"out": _repo_relative_path(args.out), "passed": payload["passed"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 1


_VMEC_BOOZER_GATE_COMMANDS = {
    "differentiability-claim": main_differentiability_claim,
    "aggregate-holdout": main_aggregate_holdout,
    "high-grid-admission": main_high_grid_admission,
    "reduced-portfolio": main_reduced_portfolio_guard,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=sorted(_VMEC_BOOZER_GATE_COMMANDS),
        help="VMEC/Boozer release gate to run.",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _VMEC_BOOZER_GATE_COMMANDS[args.command](args.args)


if __name__ == "__main__":
    raise SystemExit(main())
