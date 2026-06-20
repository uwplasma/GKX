"""State-to-input runbook reports for nonlinear-gradient controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from spectraxgk.validation.nonlinear_gradient.followup_core import (
    NonlinearGradientStateControlRunbookConfig,
    _finite_float,
    _json_number,
    _metric,
)


@dataclass(frozen=True)
class _QLScreenState:
    kind: Any
    passed: bool
    usable: bool
    admitted_controls: Sequence[Any]


@dataclass(frozen=True)
class _MappingQuality:
    mapping_passed: bool
    condition_number: float | None
    relative_residual: float | None
    input_control: Any
    blockers: list[str]


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


def _validated_runbook_config(
    config: NonlinearGradientStateControlRunbookConfig | None,
) -> NonlinearGradientStateControlRunbookConfig:
    cfg = config or NonlinearGradientStateControlRunbookConfig()
    if cfg.min_mapped_controls < 1:
        raise ValueError("min_mapped_controls must be at least one")
    if cfg.max_mapping_condition_number <= 0.0:
        raise ValueError("max_mapping_condition_number must be positive")
    if cfg.max_mapping_relative_residual < 0.0:
        raise ValueError("max_mapping_relative_residual must be non-negative")
    if cfg.default_relative_delta <= 0.0:
        raise ValueError("default_relative_delta must be positive")
    return cfg


def _ql_screen_state(ql_seed_screen: Mapping[str, Any]) -> _QLScreenState:
    admitted_raw = ql_seed_screen.get("admitted_controls")
    admitted = admitted_raw if isinstance(admitted_raw, Sequence) else ()
    ql_kind = ql_seed_screen.get("kind")
    ql_passed = bool(ql_seed_screen.get("passed", False))
    return _QLScreenState(
        kind=ql_kind,
        passed=ql_passed,
        usable=ql_kind == "nonlinear_turbulence_gradient_ql_seed_screen"
        and ql_passed,
        admitted_controls=admitted,
    )


def _ql_blockers(state: _QLScreenState) -> list[str]:
    blockers: list[str] = []
    if state.kind != "nonlinear_turbulence_gradient_ql_seed_screen":
        blockers.append("invalid_ql_seed_screen_kind")
    if not state.passed:
        blockers.append("ql_seed_screen_failed")
    return blockers


def _mapping_blockers(
    mapping: Mapping[str, Any],
    *,
    condition_number: float | None,
    relative_residual: float | None,
    config: NonlinearGradientStateControlRunbookConfig,
) -> list[str]:
    blockers: list[str] = []
    if _required_mapping_gate_failed(mapping, config=config):
        blockers.append("mapping_artifact_failed")
    if condition_number is None:
        blockers.append("missing_mapping_condition_number")
    elif condition_number > config.max_mapping_condition_number:
        blockers.append("mapping_condition_number_too_large")
    if relative_residual is None:
        blockers.append("missing_mapping_relative_residual")
    elif relative_residual > config.max_mapping_relative_residual:
        blockers.append("mapping_relative_residual_too_large")
    return blockers


def _required_mapping_gate_failed(
    mapping: Mapping[str, Any],
    *,
    config: NonlinearGradientStateControlRunbookConfig,
) -> bool:
    return config.require_mapping_passed and (
        not bool(mapping.get("artifact_passed", False))
        or not bool(mapping.get("passed", False))
    )


def _mapping_quality(
    mapping: Mapping[str, Any] | None,
    *,
    config: NonlinearGradientStateControlRunbookConfig,
) -> _MappingQuality:
    if mapping is None:
        return _MappingQuality(False, None, None, None, ["missing_state_to_input_mapping"])
    condition_number = _finite_float(mapping.get("condition_number"))
    relative_residual = _finite_float(mapping.get("relative_residual"))
    blockers = _mapping_blockers(
        mapping,
        condition_number=condition_number,
        relative_residual=relative_residual,
        config=config,
    )
    input_control = mapping.get("input_control_argument")
    if not input_control:
        blockers.append("missing_input_control_argument")
    mapping_passed = bool(mapping.get("passed", False)) or not config.require_mapping_passed
    return _MappingQuality(
        mapping_passed=mapping_passed,
        condition_number=condition_number,
        relative_residual=relative_residual,
        input_control=input_control,
        blockers=blockers,
    )


def _short_bracket_fragment(
    *,
    mapped: bool,
    input_control: Any,
    config: NonlinearGradientStateControlRunbookConfig,
) -> str | None:
    if not mapped:
        return None
    return f"--control {input_control} --relative-delta {config.default_relative_delta:.12g}"


def _runbook_row(
    raw_control: Mapping[str, Any],
    *,
    mapping: Mapping[str, Any] | None,
    ql_state: _QLScreenState,
    config: NonlinearGradientStateControlRunbookConfig,
) -> dict[str, Any]:
    quality = _mapping_quality(mapping, config=config)
    blockers = [*_ql_blockers(ql_state), *quality.blockers]
    mapped = bool(
        ql_state.usable
        and mapping is not None
        and quality.mapping_passed
        and not blockers
        and quality.input_control
    )
    return {
        "state_parameter": raw_control.get("state_parameter"),
        "state_control_argument": raw_control.get("state_control_argument"),
        "descent_direction_sign": raw_control.get("descent_direction_sign"),
        "mapping_ready": mapped,
        "blockers": blockers,
        "input_control_argument": quality.input_control,
        "input_direction": None if mapping is None else mapping.get("input_direction"),
        "input_parameter": None if mapping is None else mapping.get("input_parameter"),
        "condition_number": _json_number(quality.condition_number),
        "relative_residual": _json_number(quality.relative_residual),
        "short_bracket_command_fragment": _short_bracket_fragment(
            mapped=mapped, input_control=quality.input_control, config=config
        ),
    }


def _runbook_rows(
    *,
    ql_state: _QLScreenState,
    mapping_by_parameter: Mapping[str, Mapping[str, Any]],
    config: NonlinearGradientStateControlRunbookConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    control_rows: list[dict[str, Any]] = []
    mapped_controls: list[dict[str, Any]] = []
    for raw_control in ql_state.admitted_controls:
        if not isinstance(raw_control, Mapping):
            continue
        parameter = raw_control.get("state_parameter")
        if not isinstance(parameter, str) or not parameter:
            continue
        row = _runbook_row(
            raw_control,
            mapping=mapping_by_parameter.get(parameter),
            ql_state=ql_state,
            config=config,
        )
        if row["mapping_ready"]:
            mapped_controls.append(row)
        control_rows.append(row)
    return control_rows, mapped_controls


def _next_action(
    *,
    passed: bool,
    control_rows: Sequence[Mapping[str, Any]],
) -> str:
    if passed:
        return "write checked short-bracket nonlinear-gradient launch manifests for mapped VMEC input directions"
    if control_rows:
        return "build a VMEC-state-to-input mapping artifact before launching nonlinear-gradient campaigns"
    return "run the QL seed screen first; no admitted VMEC-state controls were provided"


def _summary(
    *,
    control_rows: Sequence[Mapping[str, Any]],
    mapped_controls: Sequence[Mapping[str, Any]],
    mapping_artifact_count: int,
    ql_state: _QLScreenState,
    config: NonlinearGradientStateControlRunbookConfig,
) -> dict[str, Any]:
    return {
        "admitted_state_control_count": len(control_rows),
        "mapped_control_count": len(mapped_controls),
        "required_mapped_control_count": config.min_mapped_controls,
        "mapping_artifact_count": mapping_artifact_count,
        "ql_seed_screen_usable": bool(ql_state.usable),
    }


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

    cfg = _validated_runbook_config(config)
    ql_state = _ql_screen_state(ql_seed_screen)
    mapping_by_parameter = _mapping_control_rows(mapping_artifacts)
    control_rows, mapped_controls = _runbook_rows(
        ql_state=ql_state,
        mapping_by_parameter=mapping_by_parameter,
        config=cfg,
    )
    passed = len(mapped_controls) >= cfg.min_mapped_controls

    return {
        "kind": "nonlinear_gradient_state_control_runbook",
        "claim_level": "state_to_input_mapping_gate_not_nonlinear_gradient_evidence",
        "case": case,
        "passed": bool(passed),
        "next_action": _next_action(passed=passed, control_rows=control_rows),
        "config": asdict(cfg),
        "summary": _summary(
            control_rows=control_rows,
            mapped_controls=mapped_controls,
            mapping_artifact_count=len(mapping_artifacts),
            ql_state=ql_state,
            config=cfg,
        ),
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
