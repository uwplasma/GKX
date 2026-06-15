"""State-to-input runbook reports for nonlinear-gradient controls."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from spectraxgk.nonlinear_gradient_followup_core import (
    NonlinearGradientStateControlRunbookConfig,
    _finite_float,
    _json_number,
    _metric,
)


def _mapping_control_rows(
    mapping_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows_by_parameter: dict[str, dict[str, Any]] = {}
    for artifact_index, artifact in enumerate(mapping_artifacts):
        artifact_passed = bool(artifact.get("passed", False))
        raw_rows = artifact.get("controls")
        if not isinstance(raw_rows, Sequence):
            raw_rows = artifact.get("mappings")
        if not isinstance(raw_rows, Sequence):
            continue
        for row_index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, Mapping):
                continue
            parameter = raw_row.get("state_parameter")
            if not isinstance(parameter, str) or not parameter:
                parameter = raw_row.get("state_control")
            if not isinstance(parameter, str) or not parameter:
                continue
            candidate = {
                "artifact_index": artifact_index,
                "row_index": row_index,
                "state_parameter": parameter,
                "input_control_argument": raw_row.get("input_control_argument"),
                "input_direction": raw_row.get("input_direction"),
                "input_parameter": raw_row.get("input_parameter"),
                "passed": bool(raw_row.get("passed", False)) and artifact_passed,
                "row_passed": bool(raw_row.get("passed", False)),
                "artifact_passed": artifact_passed,
                "condition_number": _json_number(
                    _metric(raw_row, "condition_number", "mapping_condition_number")
                ),
                "relative_residual": _json_number(
                    _metric(raw_row, "relative_residual", "mapping_relative_residual")
                ),
                "dominant_response_sign": _json_number(
                    _metric(raw_row, "dominant_response_sign", "response_sign")
                ),
                "source_kind": str(artifact.get("kind", "")),
                "source_case": artifact.get("case") or artifact.get("case_name"),
            }
            current = rows_by_parameter.get(parameter)
            if current is None or (
                bool(candidate["passed"]) and not bool(current.get("passed", False))
            ):
                rows_by_parameter[parameter] = candidate
    return rows_by_parameter


def nonlinear_gradient_state_control_runbook_report(
    ql_seed_screen: Mapping[str, Any],
    *,
    mapping_artifacts: Sequence[Mapping[str, Any]] = (),
    case: str = "nonlinear_gradient_state_control_runbook",
    config: NonlinearGradientStateControlRunbookConfig | None = None,
) -> dict[str, Any]:
    """Build a fail-closed launch runbook for VMEC-state nonlinear-gradient controls.

    The QL seed screen operates on internal ``vmec_jax`` state coordinates.
    Nonlinear campaigns, however, need perturbable input directions that can be
    written to VMEC inputs and re-equilibrated.  This report joins the admitted
    state controls to an explicit state-to-input mapping artifact and refuses to
    produce launch commands until the mapping is conditioned and complete.
    """

    cfg = config or NonlinearGradientStateControlRunbookConfig()
    if cfg.min_mapped_controls < 1:
        raise ValueError("min_mapped_controls must be at least one")
    if cfg.max_mapping_condition_number <= 0.0:
        raise ValueError("max_mapping_condition_number must be positive")
    if cfg.max_mapping_relative_residual < 0.0:
        raise ValueError("max_mapping_relative_residual must be non-negative")
    if cfg.default_relative_delta <= 0.0:
        raise ValueError("default_relative_delta must be positive")

    admitted_raw = ql_seed_screen.get("admitted_controls")
    admitted = admitted_raw if isinstance(admitted_raw, Sequence) else ()
    ql_kind = ql_seed_screen.get("kind")
    ql_screen_usable = (
        ql_kind == "nonlinear_turbulence_gradient_ql_seed_screen"
        and bool(ql_seed_screen.get("passed", False))
    )
    mapping_by_parameter = _mapping_control_rows(mapping_artifacts)
    control_rows: list[dict[str, Any]] = []
    mapped_controls: list[dict[str, Any]] = []
    for raw_control in admitted:
        if not isinstance(raw_control, Mapping):
            continue
        parameter = raw_control.get("state_parameter")
        if not isinstance(parameter, str) or not parameter:
            continue
        mapping = mapping_by_parameter.get(parameter)
        blockers: list[str] = []
        if ql_kind != "nonlinear_turbulence_gradient_ql_seed_screen":
            blockers.append("invalid_ql_seed_screen_kind")
        if not bool(ql_seed_screen.get("passed", False)):
            blockers.append("ql_seed_screen_failed")
        if mapping is None:
            blockers.append("missing_state_to_input_mapping")
            mapping_passed = False
            condition_number = None
            relative_residual = None
        else:
            condition_number = _finite_float(mapping.get("condition_number"))
            relative_residual = _finite_float(mapping.get("relative_residual"))
            mapping_passed = (
                bool(mapping.get("passed", False)) or not cfg.require_mapping_passed
            )
            if cfg.require_mapping_passed and not bool(
                mapping.get("artifact_passed", False)
            ):
                blockers.append("mapping_artifact_failed")
            if cfg.require_mapping_passed and not bool(mapping.get("passed", False)):
                if "mapping_artifact_failed" not in blockers:
                    blockers.append("mapping_artifact_failed")
            if condition_number is None:
                blockers.append("missing_mapping_condition_number")
            elif condition_number > cfg.max_mapping_condition_number:
                blockers.append("mapping_condition_number_too_large")
            if relative_residual is None:
                blockers.append("missing_mapping_relative_residual")
            elif relative_residual > cfg.max_mapping_relative_residual:
                blockers.append("mapping_relative_residual_too_large")

        input_control = (
            None if mapping is None else mapping.get("input_control_argument")
        )
        if mapping is not None and not input_control:
            blockers.append("missing_input_control_argument")
        mapped = bool(
            ql_screen_usable
            and mapping is not None
            and mapping_passed
            and not blockers
            and input_control
        )
        row = {
            "state_parameter": parameter,
            "state_control_argument": raw_control.get("state_control_argument"),
            "descent_direction_sign": raw_control.get("descent_direction_sign"),
            "mapping_ready": mapped,
            "blockers": blockers,
            "input_control_argument": input_control,
            "input_direction": None
            if mapping is None
            else mapping.get("input_direction"),
            "input_parameter": None
            if mapping is None
            else mapping.get("input_parameter"),
            "condition_number": _json_number(condition_number),
            "relative_residual": _json_number(relative_residual),
            "short_bracket_command_fragment": None
            if not mapped
            else (
                f"--control {input_control} "
                f"--relative-delta {cfg.default_relative_delta:.12g}"
            ),
        }
        if mapped:
            mapped_controls.append(row)
        control_rows.append(row)

    passed = len(mapped_controls) >= cfg.min_mapped_controls
    if passed:
        next_action = "write checked short-bracket nonlinear-gradient launch manifests for mapped VMEC input directions"
    elif control_rows:
        next_action = "build a VMEC-state-to-input mapping artifact before launching nonlinear-gradient campaigns"
    else:
        next_action = "run the QL seed screen first; no admitted VMEC-state controls were provided"

    return {
        "kind": "nonlinear_gradient_state_control_runbook",
        "claim_level": "state_to_input_mapping_gate_not_nonlinear_gradient_evidence",
        "case": case,
        "passed": bool(passed),
        "next_action": next_action,
        "config": asdict(cfg),
        "summary": {
            "admitted_state_control_count": len(control_rows),
            "mapped_control_count": len(mapped_controls),
            "required_mapped_control_count": cfg.min_mapped_controls,
            "mapping_artifact_count": len(mapping_artifacts),
            "ql_seed_screen_usable": bool(ql_screen_usable),
        },
        "mapped_controls": mapped_controls,
        "controls": control_rows,
        "mapping_protocol": [
            "select perturbable VMEC input coefficients or profile directions",
            "solve baseline/plus/minus equilibria with vmec_jax",
            "measure the induced VMEC-state response in the admitted state-control basis",
            "accept the mapping only if the local Jacobian is conditioned and residual-bounded",
            "only then launch checked short-bracket nonlinear-gradient screens",
        ],
        "scope_note": (
            "A passed QL seed screen is upstream evidence only. This runbook "
            "requires an explicit state-to-input mapping before any long-window "
            "nonlinear turbulence-gradient or optimization claim."
        ),
    }


__all__ = [
    "_mapping_control_rows",
    "nonlinear_gradient_state_control_runbook_report",
]
