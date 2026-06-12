"""Fail-closed admission helpers for quasilinear external-VMEC holdouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


ACCEPTABLE_EXTERNAL_VMEC_HOLDOUT_CLAIM_LEVELS = frozenset(
    {
        "passed_grid_convergence_candidate_for_transport_holdout",
        "passed_grid_converged_external_vmec_transport_window",
        "passed_high_grid_transport_holdout_admission_under_coarse_grid_exclusion",
    }
)

EXTERNAL_VMEC_HOLDOUT_GATE_KINDS = frozenset(
    {
        "external_vmec_high_grid_admission_gate",
        "external_vmec_nonlinear_grid_convergence_gate",
        "external_vmec_transport_window_summary",
    }
)


def _contains_external_vmec_marker(values: Iterable[object]) -> bool:
    return any("external_vmec" in str(value).lower() for value in values)


def is_external_vmec_holdout_gate(
    payload: dict[str, Any],
    *,
    artifact: str | Path | None = None,
    artifact_keys: Iterable[str] = (),
) -> bool:
    """Return whether a gate should use external-VMEC holdout admission rules."""

    kind = str(payload.get("kind", ""))
    claim_level = str(payload.get("claim_level", ""))
    if kind in EXTERNAL_VMEC_HOLDOUT_GATE_KINDS:
        return True
    if claim_level in ACCEPTABLE_EXTERNAL_VMEC_HOLDOUT_CLAIM_LEVELS:
        return True
    if claim_level.startswith("negative_grid_convergence_result"):
        return True
    case = str(payload.get("case", ""))
    values = (artifact, case, *tuple(artifact_keys))
    return _contains_external_vmec_marker(value for value in values if value)


def external_vmec_holdout_admission_status(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Return fail-closed calibration/model-selection admission metadata.

    External-VMEC holdout evidence may update quasilinear calibration ledgers
    only when its explicit promotion gate passes and its claim level is one of
    the scoped holdout-admission claim levels. Any other combination is negative
    evidence for admission, even if a lower-level convergence gate reports pass.
    """

    promotion_gate = payload.get("promotion_gate")
    promotion_gate = promotion_gate if isinstance(promotion_gate, dict) else {}
    gate_report = payload.get("gate_report")
    gate_report = gate_report if isinstance(gate_report, dict) else {}
    claim_level = str(payload.get("claim_level", ""))

    promotion_gate_passed = bool(promotion_gate.get("passed", False))
    claim_level_acceptable = (
        claim_level in ACCEPTABLE_EXTERNAL_VMEC_HOLDOUT_CLAIM_LEVELS
    )
    gate_report_passed = (
        bool(gate_report.get("passed", False)) if gate_report else None
    )
    top_level_passed = (
        bool(payload.get("passed", False)) if "passed" in payload else None
    )
    raw_gate_passed = bool(
        promotion_gate_passed or gate_report_passed is True or top_level_passed is True
    )

    blockers: list[str] = []
    if not promotion_gate_passed:
        blockers.append("promotion_gate_not_passed")
    if not claim_level_acceptable:
        blockers.append("claim_level_not_acceptable")
    if gate_report_passed is False:
        blockers.append("gate_report_not_passed")
    if top_level_passed is False:
        blockers.append("payload_not_passed")

    admitted = not blockers
    return {
        "admissible_for_calibration": admitted,
        "promotion_gate_passed": promotion_gate_passed,
        "claim_level": claim_level,
        "claim_level_acceptable": claim_level_acceptable,
        "gate_report_passed": gate_report_passed,
        "top_level_passed": top_level_passed,
        "raw_gate_passed": raw_gate_passed,
        "negative_evidence": not admitted,
        "admission_blockers": blockers,
    }
