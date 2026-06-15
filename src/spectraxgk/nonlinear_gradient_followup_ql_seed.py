"""QL/linear seed-screen reports for nonlinear-gradient controls."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence
import math

from spectraxgk.nonlinear_gradient_followup_core import (
    NonlinearGradientQLSeedScreenConfig,
    _finite_float,
    _json_number,
    _state_control_family,
)


def _ql_seed_rows(
    artifact: Mapping[str, Any],
    *,
    index: int,
    path: str | None,
    label: str | None,
    config: NonlinearGradientQLSeedScreenConfig,
) -> list[dict[str, Any]]:
    objective_gates = artifact.get("objective_gates")
    if not isinstance(objective_gates, Sequence):
        return []
    artifact_passed = bool(artifact.get("passed", False))
    case_name = str(artifact.get("case_name") or label or path or f"artifact_{index}")
    parameter_indices = artifact.get("parameter_indices")
    source_family = _state_control_family(parameter_indices)
    rows: list[dict[str, Any]] = []
    for gate_index, gate in enumerate(objective_gates):
        if not isinstance(gate, Mapping):
            continue
        objective = str(gate.get("objective") or "")
        if objective not in config.target_objectives:
            continue
        parameter = str(gate.get("parameter") or "")
        implicit = _finite_float(gate.get("implicit"))
        finite_difference = _finite_float(gate.get("finite_difference"))
        rel_error = _finite_float(gate.get("rel_error"))
        gate_passed = bool(gate.get("passed", False))
        sensitivity_resolved = (
            implicit is not None and abs(implicit) >= config.min_abs_sensitivity
        )
        rel_error_ok = (
            rel_error is not None and rel_error <= config.max_objective_rel_error
        )
        accepted = bool(
            parameter
            and sensitivity_resolved
            and rel_error_ok
            and gate_passed
            and (artifact_passed or not config.require_artifact_passed)
        )
        blockers: list[str] = []
        if not parameter:
            blockers.append("missing_parameter_name")
        if not sensitivity_resolved:
            blockers.append("unresolved_objective_sensitivity")
        if not rel_error_ok:
            blockers.append("ad_fd_relative_error_too_large")
        if not gate_passed:
            blockers.append("objective_gate_failed")
        if config.require_artifact_passed and not artifact_passed:
            blockers.append("source_artifact_failed")
        direction = None if implicit is None else -math.copysign(1.0, implicit)
        rows.append(
            {
                "artifact_index": index,
                "gate_index": gate_index,
                "label": str(label or case_name),
                "path": path,
                "case_name": case_name,
                "source_kind": str(artifact.get("kind", "")),
                "source_artifact_passed": artifact_passed,
                "state_parameter": parameter,
                "state_control_family": source_family,
                "parameter_indices": parameter_indices
                if isinstance(parameter_indices, Mapping)
                else None,
                "objective": objective,
                "accepted_objective_gate": accepted,
                "blockers": blockers,
                "metrics": {
                    "implicit_sensitivity": _json_number(implicit),
                    "finite_difference_sensitivity": _json_number(finite_difference),
                    "relative_error": _json_number(rel_error),
                    "descent_direction_sign": _json_number(direction),
                },
            }
        )
    return rows


def _sign_consistency(
    values: Sequence[float], *, value_floor: float
) -> tuple[float | None, float | None]:
    signs = [math.copysign(1.0, value) for value in values if abs(value) > value_floor]
    if not signs:
        return None, None
    positive = sum(1 for sign in signs if sign > 0.0)
    negative = len(signs) - positive
    dominant = 1.0 if positive >= negative else -1.0
    return dominant, max(positive, negative) / len(signs)


def nonlinear_gradient_ql_seed_screen_report(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    paths: Sequence[str | None] | None = None,
    labels: Sequence[str | None] | None = None,
    case: str = "nonlinear_turbulence_gradient_ql_seed_screen",
    config: NonlinearGradientQLSeedScreenConfig | None = None,
) -> dict[str, Any]:
    """Screen QL/linear sensitivity artifacts before nonlinear-gradient runs.

    The report groups full-chain VMEC/Boozer sensitivity rows by state
    parameter and admits a control only when the primary objective sensitivity
    is AD/FD-consistent, resolved, sign-consistent across enough artifacts, and
    tied to a distinct VMEC-state control.  The output is deliberately a
    planning artifact: VMEC-state controls are not assumed to be patchable
    ``RBC/ZBS`` input-file coefficients.
    """

    cfg = config or NonlinearGradientQLSeedScreenConfig()
    if not cfg.target_objectives:
        raise ValueError("target_objectives must be non-empty")
    if cfg.primary_objective not in cfg.target_objectives:
        raise ValueError("primary_objective must be included in target_objectives")
    if cfg.min_distinct_controls < 1:
        raise ValueError("min_distinct_controls must be at least one")
    if cfg.min_cases_per_control < 1:
        raise ValueError("min_cases_per_control must be at least one")
    if cfg.min_sign_consistency <= 0.0 or cfg.min_sign_consistency > 1.0:
        raise ValueError("min_sign_consistency must be in (0, 1]")
    if cfg.max_objective_rel_error < 0.0:
        raise ValueError("max_objective_rel_error must be non-negative")
    if cfg.min_abs_sensitivity <= 0.0:
        raise ValueError("min_abs_sensitivity must be positive")

    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")

    objective_rows: list[dict[str, Any]] = []
    for index, (artifact, path, label) in enumerate(
        zip(artifacts, path_list, label_list)
    ):
        objective_rows.extend(
            _ql_seed_rows(artifact, index=index, path=path, label=label, config=cfg)
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in objective_rows:
        if row["objective"] == cfg.primary_objective:
            grouped.setdefault(str(row["state_parameter"]), []).append(row)

    controls: list[dict[str, Any]] = []
    for parameter, rows in sorted(grouped.items()):
        accepted_rows = [row for row in rows if bool(row["accepted_objective_gate"])]
        sensitivities = [
            float(row["metrics"]["implicit_sensitivity"])
            for row in accepted_rows
            if row["metrics"]["implicit_sensitivity"] is not None
        ]
        dominant_sign, sign_fraction = _sign_consistency(
            sensitivities,
            value_floor=cfg.min_abs_sensitivity,
        )
        n_cases = len({str(row["case_name"]) for row in accepted_rows})
        enough_cases = n_cases >= cfg.min_cases_per_control
        sign_ok = (
            sign_fraction is not None and sign_fraction >= cfg.min_sign_consistency
        )
        admitted = bool(enough_cases and sign_ok)
        blockers: list[str] = []
        if not accepted_rows:
            blockers.append("no_accepted_primary_objective_rows")
        if not enough_cases:
            blockers.append("insufficient_case_coverage")
        if not sign_ok:
            blockers.append("cross_artifact_sign_not_consistent")
        direction = None if dominant_sign is None else -dominant_sign
        mean_abs_sensitivity = None
        if sensitivities:
            mean_abs_sensitivity = sum(abs(value) for value in sensitivities) / len(
                sensitivities
            )
        controls.append(
            {
                "state_parameter": parameter,
                "state_control_family": accepted_rows[0].get("state_control_family")
                if accepted_rows
                else None,
                "admitted_for_nonlinear_screen": admitted,
                "blockers": blockers,
                "primary_objective": cfg.primary_objective,
                "n_accepted_rows": len(accepted_rows),
                "n_cases": n_cases,
                "dominant_sensitivity_sign": _json_number(dominant_sign),
                "descent_direction_sign": _json_number(direction),
                "sign_consistency_fraction": _json_number(sign_fraction),
                "mean_abs_sensitivity": _json_number(mean_abs_sensitivity),
                "state_control_argument": None
                if direction is None
                else f"{parameter}:{direction:.12g}",
                "source_rows": [
                    {
                        "case_name": row["case_name"],
                        "path": row["path"],
                        "source_artifact_passed": row["source_artifact_passed"],
                        "implicit_sensitivity": row["metrics"]["implicit_sensitivity"],
                        "relative_error": row["metrics"]["relative_error"],
                    }
                    for row in accepted_rows
                ],
            }
        )

    admitted_controls = [
        row for row in controls if bool(row["admitted_for_nonlinear_screen"])
    ]
    passed = len(admitted_controls) >= cfg.min_distinct_controls
    if passed:
        next_action = "build checked short-bracket nonlinear-gradient screens for admitted VMEC-state controls"
    elif controls:
        next_action = (
            "generate additional QL/linear sensitivity artifacts for distinct VMEC-state controls "
            "before nonlinear GPU campaigns"
        )
    else:
        next_action = "no usable QL/linear sensitivity rows; generate full-chain VMEC/Boozer gradient artifacts first"

    return {
        "kind": "nonlinear_turbulence_gradient_ql_seed_screen",
        "claim_level": "ql_seeded_control_screen_not_nonlinear_gradient_evidence",
        "case": case,
        "passed": bool(passed),
        "next_action": next_action,
        "config": asdict(cfg),
        "summary": {
            "artifact_count": len(artifacts),
            "objective_row_count": len(objective_rows),
            "control_count": len(controls),
            "admitted_control_count": len(admitted_controls),
            "required_distinct_controls": cfg.min_distinct_controls,
        },
        "admitted_controls": admitted_controls,
        "controls": controls,
        "objective_rows": objective_rows,
        "scope_note": (
            "Rows describe vmec_jax state controls. They are not direct VMEC "
            "input-file RBC/ZBS coefficients unless a separate mapping artifact "
            "establishes that relation."
        ),
    }


__all__ = [
    "_ql_seed_rows",
    "_sign_consistency",
    "nonlinear_gradient_ql_seed_screen_report",
]
