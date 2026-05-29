"""Targeted follow-up plans for nonlinear turbulence-gradient audits.

This module turns failed long-window central finite-difference artifacts into a
bounded run prescription.  It is deliberately conservative: extra replicas are
recommended only when the finite-difference response is resolved and local, but
the propagated gradient uncertainty is slightly too large.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
import math
import re


STATE_TO_RUN_STATE = {
    "baseline": "baseline",
    "plus": "plus_delta",
    "minus": "minus_delta",
}


@dataclass(frozen=True)
class NonlinearGradientFollowupConfig:
    """Acceptance and cost controls for the follow-up planner."""

    max_gradient_uncertainty_rel: float = 0.50
    max_fd_asymmetry_rel: float = 0.50
    min_fd_response_fraction: float = 0.03
    sem_safety_factor: float = 1.10
    max_extra_replicates_per_state: int = 4
    default_nominal_timestep: float = 0.05
    value_floor: float = 1.0e-12


@dataclass(frozen=True)
class NonlinearGradientCandidateDesignConfig:
    """Conditioning limits for the next nonlinear-gradient campaign design."""

    max_gradient_uncertainty_rel: float = 0.50
    max_fd_asymmetry_rel: float = 0.50
    min_fd_response_fraction: float = 0.03
    sem_safety_factor: float = 1.10
    max_extra_replicates_per_state: int = 4
    max_checked_bracket_scale: float = 1.50
    locality_safety_factor: float = 0.95
    value_floor: float = 1.0e-12


@dataclass(frozen=True)
class NonlinearGradientCompositeControlConfig:
    """Controls for constructing the next composite nonlinear-gradient direction."""

    max_gradient_uncertainty_rel: float = 1.00
    max_fd_asymmetry_rel: float = 0.50
    min_fd_response_fraction: float = 0.03
    min_same_sign_fraction: float = 0.80
    min_controls: int = 2
    default_relative_delta: float = 0.02
    max_weight_abs: float = 1.0
    value_floor: float = 1.0e-12


@dataclass(frozen=True)
class NonlinearGradientQLSeedScreenConfig:
    """Admission limits for quasilinear-seeded nonlinear-gradient controls."""

    target_objectives: tuple[str, ...] = (
        "mixing_length_heat_flux_proxy",
        "linear_heat_flux_weight",
        "gamma",
    )
    primary_objective: str = "mixing_length_heat_flux_proxy"
    min_distinct_controls: int = 2
    min_cases_per_control: int = 2
    min_sign_consistency: float = 0.75
    max_objective_rel_error: float = 0.02
    min_abs_sensitivity: float = 1.0e-12
    require_artifact_passed: bool = False


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _json_number(value: Any) -> float | int | None:
    number = _finite_float(value)
    if number is None:
        return None
    if isinstance(value, int):
        return value
    return float(number)


def _metric(payload: Mapping[str, Any], *names: str) -> float | None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}
    conditioning = payload.get("conditioning")
    if isinstance(conditioning, Mapping):
        sources: tuple[Mapping[str, Any], ...] = (metrics, conditioning, payload)
    else:
        sources = (metrics, payload)
    for source in sources:
        for name in names:
            value = _finite_float(source.get(name))
            if value is not None:
                return value
    return None


def _nested_metric(payload: Mapping[str, Any], source_name: str, *names: str) -> float | None:
    source = payload.get(source_name)
    if not isinstance(source, Mapping):
        return None
    for name in names:
        value = _finite_float(source.get(name))
        if value is not None:
            return value
    return None


def _artifact_passed(payload: Mapping[str, Any]) -> bool:
    if bool(payload.get("passed", False)):
        return True
    nested = payload.get("nonlinear_turbulence_gradient_gate")
    return isinstance(nested, Mapping) and bool(nested.get("passed", False))


def _coefficient_label_from_parameter(parameter: Any) -> str | None:
    if not isinstance(parameter, str):
        return None
    match = re.fullmatch(r"\s*(rbc|rbs|zbc|zbs)_([+-]?\d+)_([+-]?\d+)\s*", parameter, re.IGNORECASE)
    if match is None:
        return None
    family, m_value, n_value = match.groups()
    return f"{family.upper()}({int(m_value)},{int(n_value)})"


def _state_control_family(parameter_indices: Any) -> str | None:
    if not isinstance(parameter_indices, Mapping) or not parameter_indices:
        return None
    for family in ("Rcos", "Rsin", "Zcos", "Zsin", "RBC", "RBS", "ZBC", "ZBS"):
        if family in parameter_indices:
            return family
    return str(next(iter(parameter_indices)))


def _label_from_row(row: Mapping[str, Any]) -> str | None:
    for key in ("variant_label", "source_artifact", "summary_artifact", "path"):
        value = row.get(key)
        if not isinstance(value, str):
            continue
        match = re.search(r"(seed[0-9]+|dt[0-9]+(?:p[0-9]+)?)", value)
        if match:
            return match.group(1)
    return None


def _seed_numbers(source_ensembles: Mapping[str, Any]) -> list[int]:
    seeds: list[int] = []
    for raw in source_ensembles.values():
        if not isinstance(raw, Mapping):
            continue
        rows = raw.get("rows")
        if not isinstance(rows, Sequence):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            label = _label_from_row(row)
            if label is None:
                continue
            match = re.fullmatch(r"seed([0-9]+)", label)
            if match:
                seeds.append(int(match.group(1)))
    return seeds


def _replicate_count(source_ensembles: Mapping[str, Any]) -> int | None:
    counts: list[int] = []
    for state in ("minus", "baseline", "plus"):
        raw = source_ensembles.get(state)
        if not isinstance(raw, Mapping):
            continue
        count = _finite_int(raw.get("n_reports"))
        if count is None:
            rows = raw.get("rows")
            count = len(rows) if isinstance(rows, Sequence) else None
        if count is not None:
            counts.append(count)
    return min(counts) if counts else None


def _required_replicates(
    *,
    current_count: int | None,
    uncertainty_rel: float,
    config: NonlinearGradientFollowupConfig,
) -> int | None:
    if current_count is None or current_count <= 0:
        return None
    target = max(float(config.max_gradient_uncertainty_rel), float(config.value_floor))
    scale = (float(uncertainty_rel) / target) ** 2 * float(config.sem_safety_factor)
    return max(current_count + 1, int(math.ceil(current_count * scale)))


def _required_replicates_for_scaled_bracket(
    *,
    current_count: int | None,
    uncertainty_rel: float | None,
    bracket_scale: float,
    config: NonlinearGradientCandidateDesignConfig,
) -> int | None:
    if current_count is None or current_count <= 0:
        return None
    if uncertainty_rel is None or bracket_scale <= 0.0:
        return None
    target = max(float(config.max_gradient_uncertainty_rel), float(config.value_floor))
    scale = (float(uncertainty_rel) / (target * float(bracket_scale))) ** 2
    scale *= float(config.sem_safety_factor)
    return max(current_count, int(math.ceil(current_count * scale)))


def _design_row(
    artifact: Mapping[str, Any],
    *,
    index: int,
    path: str | None,
    label: str | None,
    config: NonlinearGradientCandidateDesignConfig,
) -> dict[str, Any]:
    source_ensembles_raw = artifact.get("source_ensembles")
    source_ensembles = source_ensembles_raw if isinstance(source_ensembles_raw, Mapping) else {}
    response_fraction = _metric(artifact, "response_fraction")
    asymmetry_rel = _metric(artifact, "fd_asymmetry_rel", "asymmetry_rel")
    uncertainty_rel = _metric(
        artifact,
        "gradient_uncertainty_rel",
        "gradient_relative_uncertainty",
    )
    current_replicates = _replicate_count(source_ensembles)
    passed = _artifact_passed(artifact)

    response_ok = response_fraction is not None and response_fraction >= config.min_fd_response_fraction
    locality_ok = asymmetry_rel is not None and asymmetry_rel <= config.max_fd_asymmetry_rel
    uncertainty_ok = uncertainty_rel is not None and uncertainty_rel <= config.max_gradient_uncertainty_rel

    uncertainty_required_scale = None
    if uncertainty_rel is not None:
        uncertainty_required_scale = max(1.0, uncertainty_rel / max(config.max_gradient_uncertainty_rel, config.value_floor))

    locality_scale_limit = None
    if asymmetry_rel is not None and asymmetry_rel > 0.0:
        locality_scale_limit = (
            config.locality_safety_factor * config.max_fd_asymmetry_rel / asymmetry_rel
        )
    elif asymmetry_rel == 0.0:
        locality_scale_limit = float("inf")

    usable_bracket_scale = 1.0
    if locality_scale_limit is not None:
        usable_bracket_scale = max(1.0, min(config.max_checked_bracket_scale, locality_scale_limit))
    elif uncertainty_required_scale is not None:
        usable_bracket_scale = min(config.max_checked_bracket_scale, uncertainty_required_scale)

    bracket_only_feasible = (
        bool(response_ok)
        and bool(locality_ok)
        and uncertainty_required_scale is not None
        and locality_scale_limit is not None
        and uncertainty_required_scale <= min(config.max_checked_bracket_scale, locality_scale_limit)
    )
    required_replicates_no_bracket = _required_replicates_for_scaled_bracket(
        current_count=current_replicates,
        uncertainty_rel=uncertainty_rel,
        bracket_scale=1.0,
        config=config,
    )
    required_replicates_at_local_limit = _required_replicates_for_scaled_bracket(
        current_count=current_replicates,
        uncertainty_rel=uncertainty_rel,
        bracket_scale=usable_bracket_scale,
        config=config,
    )
    extra_replicates_at_local_limit = None
    if required_replicates_at_local_limit is not None and current_replicates is not None:
        extra_replicates_at_local_limit = max(0, required_replicates_at_local_limit - current_replicates)

    if passed:
        action = "freeze_promoted_candidate"
        recommendation = "candidate already passes production gates; freeze provenance"
    elif not response_ok:
        action = "increase_checked_bracket_or_replace_control"
        recommendation = (
            "the finite-difference response is below the resolved-response gate; "
            "run a checked bracket sweep or replace this control before adding replicas"
        )
    elif not locality_ok:
        action = "shrink_or_replace_nonlocal_control"
        recommendation = (
            "the finite-difference bracket is nonlocal; shrink the bracket or "
            "replace the control before spending replicas"
        )
    elif uncertainty_ok:
        action = "inspect_pass_flag"
        recommendation = "scalar gates pass but the artifact did not promote; inspect metadata and provenance"
    elif bracket_only_feasible:
        action = "run_checked_larger_bracket"
        recommendation = (
            "a bounded larger bracket can in principle resolve uncertainty while "
            "staying below the locality limit; run a short locality/response sweep first"
        )
    elif extra_replicates_at_local_limit is not None and extra_replicates_at_local_limit <= config.max_extra_replicates_per_state:
        action = "add_limited_replicates_with_locality_cap"
        recommendation = (
            "combine the largest locality-safe bracket with a bounded number of matched replicas"
        )
    else:
        action = "design_better_conditioned_control_or_variance_reduction"
        recommendation = (
            "bracket-only and bounded-replica fixes are not efficient; design a better-conditioned "
            "composite direction, variance-reduced observable, or checked response-larger bracket"
        )

    return {
        "index": index,
        "label": str(label or artifact.get("parameter_name") or path or index),
        "path": path,
        "parameter_name": str(artifact.get("parameter_name") or ""),
        "passed": passed,
        "action": action,
        "recommendation": recommendation,
        "metrics": {
            "response_fraction": _json_number(response_fraction),
            "fd_asymmetry_rel": _json_number(asymmetry_rel),
            "gradient_uncertainty_rel": _json_number(uncertainty_rel),
        },
        "gate_status": {
            "response_ok": bool(response_ok),
            "locality_ok": bool(locality_ok),
            "uncertainty_ok": bool(uncertainty_ok),
        },
        "current_replicates_per_state": current_replicates,
        "uncertainty_required_bracket_scale": _json_number(uncertainty_required_scale),
        "locality_safe_bracket_scale_limit": _json_number(locality_scale_limit),
        "usable_bracket_scale_for_estimate": _json_number(usable_bracket_scale),
        "bracket_only_feasible": bool(bracket_only_feasible),
        "estimated_required_replicates_no_bracket": required_replicates_no_bracket,
        "estimated_required_replicates_at_locality_limit": required_replicates_at_local_limit,
        "estimated_extra_replicates_at_locality_limit": extra_replicates_at_local_limit,
    }


def nonlinear_gradient_candidate_design_report(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    paths: Sequence[str | None] | None = None,
    labels: Sequence[str | None] | None = None,
    case: str = "nonlinear_turbulence_gradient_candidate_design",
    config: NonlinearGradientCandidateDesignConfig | None = None,
) -> dict[str, Any]:
    """Return the next-campaign design implied by failed production gates.

    The report estimates whether a failed central-FD candidate can be rescued by
    a larger bracket, by bounded extra replicas, or whether the next campaign
    should instead change the control/observable.  The estimates use the usual
    ``1/sqrt(N)`` SEM scaling and the local finite-difference assumption that
    response grows approximately linearly with bracket size before the asymmetry
    gate is hit.
    """

    cfg = config or NonlinearGradientCandidateDesignConfig()
    if cfg.max_gradient_uncertainty_rel <= 0.0:
        raise ValueError("max_gradient_uncertainty_rel must be positive")
    if cfg.max_fd_asymmetry_rel <= 0.0:
        raise ValueError("max_fd_asymmetry_rel must be positive")
    if cfg.min_fd_response_fraction <= 0.0:
        raise ValueError("min_fd_response_fraction must be positive")
    if cfg.sem_safety_factor <= 0.0:
        raise ValueError("sem_safety_factor must be positive")
    if cfg.max_extra_replicates_per_state < 0:
        raise ValueError("max_extra_replicates_per_state must be non-negative")
    if cfg.max_checked_bracket_scale < 1.0:
        raise ValueError("max_checked_bracket_scale must be at least one")
    if cfg.locality_safety_factor <= 0.0:
        raise ValueError("locality_safety_factor must be positive")

    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")

    candidates = [
        _design_row(artifact, index=index, path=path, label=label, config=cfg)
        for index, (artifact, path, label) in enumerate(zip(artifacts, path_list, label_list))
    ]
    promoted = [row for row in candidates if row["action"] == "freeze_promoted_candidate"]
    bracket_ready = [row for row in candidates if row["action"] == "run_checked_larger_bracket"]
    replica_ready = [row for row in candidates if row["action"] == "add_limited_replicates_with_locality_cap"]
    replacement = [
        row
        for row in candidates
        if row["action"] in {
            "design_better_conditioned_control_or_variance_reduction",
            "increase_checked_bracket_or_replace_control",
            "shrink_or_replace_nonlocal_control",
        }
    ]

    if promoted:
        next_action = "freeze promoted candidate provenance"
    elif bracket_ready:
        next_action = "run a bounded bracket/locality sweep before new long windows"
    elif replica_ready:
        next_action = "combine locality-capped bracket scale with bounded matched replicas"
    elif replacement:
        next_action = "design a better-conditioned control or variance-reduced observable before more GPU replicas"
    else:
        next_action = "inspect candidate metadata before designing new runs"

    return {
        "kind": "nonlinear_turbulence_gradient_candidate_design_report",
        "claim_level": "campaign_design_not_gradient_evidence",
        "case": case,
        "passed": bool(promoted),
        "next_action": next_action,
        "config": asdict(cfg),
        "summary": {
            "candidate_count": len(candidates),
            "promoted_candidate_count": len(promoted),
            "bracket_ready_count": len(bracket_ready),
            "replica_ready_count": len(replica_ready),
            "replacement_or_variance_reduction_count": len(replacement),
        },
        "candidates": candidates,
    }


def _composite_control_row(
    artifact: Mapping[str, Any],
    *,
    index: int,
    path: str | None,
    label: str | None,
    config: NonlinearGradientCompositeControlConfig,
) -> dict[str, Any]:
    parameter_name = str(artifact.get("parameter_name") or label or path or f"candidate_{index}")
    coefficient = _coefficient_label_from_parameter(artifact.get("parameter_name"))
    response_fraction = _metric(artifact, "response_fraction")
    asymmetry_rel = _metric(artifact, "fd_asymmetry_rel", "asymmetry_rel")
    uncertainty_rel = _metric(
        artifact,
        "gradient_uncertainty_rel",
        "gradient_relative_uncertainty",
    )
    central_gradient = _metric(
        artifact,
        "central_gradient",
        "central_fd_dq_dparameter",
        "central_fd_dq_dtprim",
    )
    paired_uncertainty_rel = _nested_metric(
        artifact,
        "paired_replicate_diagnostics",
        "central_gradient_uncertainty_rel",
    )
    same_sign_fraction = _nested_metric(
        artifact,
        "paired_replicate_diagnostics",
        "same_sign_fraction",
    )
    response_ok = response_fraction is not None and response_fraction >= config.min_fd_response_fraction
    locality_ok = asymmetry_rel is not None and asymmetry_rel <= config.max_fd_asymmetry_rel
    uncertainty_ok = uncertainty_rel is not None and uncertainty_rel <= config.max_gradient_uncertainty_rel
    same_sign_ok = same_sign_fraction is None or same_sign_fraction >= config.min_same_sign_fraction
    gradient_ok = central_gradient is not None and abs(central_gradient) > config.value_floor
    coefficient_ok = coefficient is not None
    admissible = bool(
        coefficient_ok
        and gradient_ok
        and response_ok
        and locality_ok
        and uncertainty_ok
        and same_sign_ok
    )

    blockers: list[str] = []
    if not coefficient_ok:
        blockers.append("parameter_not_vmec_boundary_coefficient")
    if not gradient_ok:
        blockers.append("missing_or_zero_central_gradient")
    if not response_ok:
        blockers.append("unresolved_heat_flux_response")
    if not locality_ok:
        blockers.append("nonlocal_finite_difference_bracket")
    if not uncertainty_ok:
        blockers.append("gradient_uncertainty_too_large")
    if not same_sign_ok:
        blockers.append("paired_replicate_sign_not_robust")

    descent_gradient = None if central_gradient is None else -float(central_gradient)
    return {
        "index": index,
        "label": str(label or parameter_name),
        "path": path,
        "parameter_name": parameter_name,
        "coefficient": coefficient,
        "admissible_for_composite_direction": admissible,
        "blockers": blockers,
        "metrics": {
            "central_gradient": _json_number(central_gradient),
            "descent_gradient": _json_number(descent_gradient),
            "response_fraction": _json_number(response_fraction),
            "fd_asymmetry_rel": _json_number(asymmetry_rel),
            "gradient_uncertainty_rel": _json_number(uncertainty_rel),
            "paired_gradient_uncertainty_rel": _json_number(paired_uncertainty_rel),
            "same_sign_fraction": _json_number(same_sign_fraction),
        },
        "gate_status": {
            "coefficient_ok": coefficient_ok,
            "gradient_ok": gradient_ok,
            "response_ok": bool(response_ok),
            "locality_ok": bool(locality_ok),
            "uncertainty_ok": bool(uncertainty_ok),
            "same_sign_ok": bool(same_sign_ok),
        },
    }


def nonlinear_gradient_composite_control_report(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    paths: Sequence[str | None] | None = None,
    labels: Sequence[str | None] | None = None,
    case: str = "nonlinear_turbulence_gradient_composite_control_design",
    config: NonlinearGradientCompositeControlConfig | None = None,
) -> dict[str, Any]:
    """Design a normalized VMEC-boundary direction from resolved FD candidates.

    This is a launch-planning gate, not nonlinear-gradient evidence.  The
    returned controls are the steepest-descent direction in the subspace of
    candidates that already pass locality, response, uncertainty, coefficient,
    and paired-sign checks. If fewer than ``min_controls`` survive, the report
    fails closed and provides exact blockers instead of producing a misleading
    multi-coefficient launch recommendation.
    """

    cfg = config or NonlinearGradientCompositeControlConfig()
    if cfg.max_gradient_uncertainty_rel <= 0.0:
        raise ValueError("max_gradient_uncertainty_rel must be positive")
    if cfg.max_fd_asymmetry_rel <= 0.0:
        raise ValueError("max_fd_asymmetry_rel must be positive")
    if cfg.min_fd_response_fraction <= 0.0:
        raise ValueError("min_fd_response_fraction must be positive")
    if cfg.min_same_sign_fraction <= 0.0 or cfg.min_same_sign_fraction > 1.0:
        raise ValueError("min_same_sign_fraction must be in (0, 1]")
    if cfg.min_controls < 1:
        raise ValueError("min_controls must be at least one")
    if cfg.default_relative_delta <= 0.0:
        raise ValueError("default_relative_delta must be positive")
    if cfg.max_weight_abs <= 0.0:
        raise ValueError("max_weight_abs must be positive")

    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")

    rows = [
        _composite_control_row(artifact, index=index, path=path, label=label, config=cfg)
        for index, (artifact, path, label) in enumerate(zip(artifacts, path_list, label_list))
    ]
    admissible_rows = [row for row in rows if bool(row["admissible_for_composite_direction"])]
    max_abs_descent = max(
        (abs(float(row["metrics"]["descent_gradient"])) for row in admissible_rows),
        default=0.0,
    )
    controls: list[dict[str, Any]] = []
    if max_abs_descent > cfg.value_floor:
        for row in admissible_rows:
            descent = float(row["metrics"]["descent_gradient"])
            weight = cfg.max_weight_abs * descent / max_abs_descent
            controls.append(
                {
                    "parameter_name": row["parameter_name"],
                    "coefficient": row["coefficient"],
                    "weight": _json_number(weight),
                    "control_argument": f"{row['coefficient']}:{weight:.12g}",
                    "source_label": row["label"],
                    "source_path": row["path"],
                }
            )

    launch_ready = len(controls) >= cfg.min_controls
    if launch_ready:
        control_args = " ".join(f"--control {control['control_argument']}" for control in controls)
        command_template = (
            "python tools/write_vmec_boundary_profile_perturbation_inputs.py "
            "--baseline-input <input.vmec> "
            "--out-dir docs/_static/<case>_composite_direction "
            f"--case {case} "
            f"{control_args} "
            f"--relative-delta {cfg.default_relative_delta:.12g}"
        )
        next_action = "launch a checked VMEC profile-direction bracket sweep before long nonlinear windows"
    elif controls:
        command_template = None
        next_action = (
            "only one admissible control remains; screen additional local/resolved controls "
            "or explicitly run a single-control bracket check before a long campaign"
        )
    else:
        command_template = None
        next_action = "no admissible controls; screen new VMEC-boundary directions before nonlinear GPU runs"

    return {
        "kind": "nonlinear_turbulence_gradient_composite_control_design",
        "claim_level": "composite_control_launch_plan_not_gradient_evidence",
        "case": case,
        "passed": bool(launch_ready),
        "next_action": next_action,
        "config": asdict(cfg),
        "summary": {
            "candidate_count": len(rows),
            "admissible_control_count": len(controls),
            "required_control_count": cfg.min_controls,
            "launch_ready": bool(launch_ready),
        },
        "controls": controls,
        "write_profile_direction_command_template": command_template,
        "candidates": rows,
    }


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
        sensitivity_resolved = implicit is not None and abs(implicit) >= config.min_abs_sensitivity
        rel_error_ok = rel_error is not None and rel_error <= config.max_objective_rel_error
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
                "parameter_indices": parameter_indices if isinstance(parameter_indices, Mapping) else None,
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


def _sign_consistency(values: Sequence[float], *, value_floor: float) -> tuple[float | None, float | None]:
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
    for index, (artifact, path, label) in enumerate(zip(artifacts, path_list, label_list)):
        objective_rows.extend(_ql_seed_rows(artifact, index=index, path=path, label=label, config=cfg))

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
        sign_ok = sign_fraction is not None and sign_fraction >= cfg.min_sign_consistency
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
            mean_abs_sensitivity = sum(abs(value) for value in sensitivities) / len(sensitivities)
        controls.append(
            {
                "state_parameter": parameter,
                "state_control_family": accepted_rows[0].get("state_control_family") if accepted_rows else None,
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

    admitted_controls = [row for row in controls if bool(row["admitted_for_nonlinear_screen"])]
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


def _planned_matched_runs(
    *,
    source_ensembles: Mapping[str, Any],
    extra_replicates_per_state: int,
    config: NonlinearGradientFollowupConfig,
) -> list[dict[str, Any]]:
    if extra_replicates_per_state <= 0:
        return []
    seeds = _seed_numbers(source_ensembles)
    next_seed = max(seeds) + 1 if seeds else 31
    runs: list[dict[str, Any]] = []
    for offset in range(extra_replicates_per_state):
        seed = next_seed + offset
        for source_state, run_state in STATE_TO_RUN_STATE.items():
            if source_state not in source_ensembles:
                continue
            runs.append(
                {
                    "state": run_state,
                    "variant_axis": "seed",
                    "variant_label": f"seed{seed}",
                    "seed": seed,
                    "timestep": float(config.default_nominal_timestep),
                    "reason": (
                        "independent nominal-timestep replicate to reduce "
                        "central finite-difference gradient uncertainty"
                    ),
                }
            )
    return runs


def nonlinear_gradient_followup_plan(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    paths: Sequence[str | None] | None = None,
    labels: Sequence[str | None] | None = None,
    case: str = "nonlinear_turbulence_gradient_followup",
    config: NonlinearGradientFollowupConfig | None = None,
) -> dict[str, Any]:
    """Build a bounded, fail-closed follow-up plan from gradient artifacts."""

    cfg = config or NonlinearGradientFollowupConfig()
    if cfg.max_extra_replicates_per_state < 0:
        raise ValueError("max_extra_replicates_per_state must be non-negative")
    if cfg.sem_safety_factor <= 0.0:
        raise ValueError("sem_safety_factor must be positive")
    if cfg.max_gradient_uncertainty_rel <= 0.0:
        raise ValueError("max_gradient_uncertainty_rel must be positive")

    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")

    candidate_actions: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    for index, (artifact, path, label) in enumerate(zip(artifacts, path_list, label_list)):
        source_ensembles_raw = artifact.get("source_ensembles")
        source_ensembles = (
            source_ensembles_raw if isinstance(source_ensembles_raw, Mapping) else {}
        )
        response_fraction = _metric(artifact, "response_fraction")
        asymmetry_rel = _metric(artifact, "fd_asymmetry_rel", "asymmetry_rel")
        uncertainty_rel = _metric(
            artifact,
            "gradient_uncertainty_rel",
            "gradient_relative_uncertainty",
        )
        current_replicates = _replicate_count(source_ensembles)
        passed = _artifact_passed(artifact)
        response_ok = (
            response_fraction is not None
            and response_fraction >= float(cfg.min_fd_response_fraction)
        )
        locality_ok = (
            asymmetry_rel is not None
            and asymmetry_rel <= float(cfg.max_fd_asymmetry_rel)
        )
        uncertainty_ok = (
            uncertainty_rel is not None
            and uncertainty_rel <= float(cfg.max_gradient_uncertainty_rel)
        )
        action = "inspect_artifact"
        recommendation = "missing required metrics; inspect the artifact before spending GPU time"
        required_replicates = None
        extra_replicates = 0
        planned_runs: list[dict[str, Any]] = []

        if passed:
            action = "freeze_promoted_candidate"
            recommendation = "candidate already passes production gates; freeze provenance"
        elif not response_ok:
            action = "replace_control_or_increase_checked_bracket"
            recommendation = (
                "finite-difference response is not resolved; change the control or "
                "perform a checked bracket sweep before adding replicas"
            )
        elif not locality_ok:
            action = "shrink_bracket_or_replace_control"
            recommendation = (
                "finite-difference bracket is nonlocal; shrink the perturbation or "
                "replace the control before adding replicas"
            )
        elif not uncertainty_ok and uncertainty_rel is not None:
            required_replicates = _required_replicates(
                current_count=current_replicates,
                uncertainty_rel=uncertainty_rel,
                config=cfg,
            )
            if required_replicates is None:
                action = "recover_replicate_metadata"
                recommendation = (
                    "uncertainty is marginal but the artifact lacks replicate counts; "
                    "recover ensemble metadata before launching runs"
                )
            else:
                extra_replicates = max(0, required_replicates - int(current_replicates or 0))
                extra_replicates = min(extra_replicates, int(cfg.max_extra_replicates_per_state))
                action = "add_matched_nominal_seed_replicates"
                recommendation = (
                    "response and locality pass, but uncertainty is marginal; add only "
                    "the matched independent replicas needed by the 1/sqrt(N) estimate"
                )
                planned_runs = _planned_matched_runs(
                    source_ensembles=source_ensembles,
                    extra_replicates_per_state=extra_replicates,
                    config=cfg,
                )
        else:
            action = "no_followup_needed"
            recommendation = "all scalar gates are satisfied, but artifact was not marked passed"

        all_runs.extend(
            {
                **run,
                "candidate_index": index,
                "candidate_label": str(label or artifact.get("parameter_name") or path or index),
            }
            for run in planned_runs
        )
        candidate_actions.append(
            {
                "index": index,
                "label": str(label or artifact.get("parameter_name") or path or index),
                "path": path,
                "parameter_name": str(artifact.get("parameter_name") or ""),
                "passed": passed,
                "action": action,
                "recommendation": recommendation,
                "metrics": {
                    "response_fraction": _json_number(response_fraction),
                    "fd_asymmetry_rel": _json_number(asymmetry_rel),
                    "gradient_uncertainty_rel": _json_number(uncertainty_rel),
                },
                "current_replicates_per_state": current_replicates,
                "estimated_required_replicates_per_state": required_replicates,
                "extra_replicates_per_state": extra_replicates,
                "planned_run_count": len(planned_runs),
                "planned_runs": planned_runs,
            }
        )

    promoted = [row for row in candidate_actions if row["action"] == "freeze_promoted_candidate"]
    runnable = [row for row in candidate_actions if row["planned_run_count"]]
    nonlocal_rows = [row for row in candidate_actions if row["action"] == "shrink_bracket_or_replace_control"]
    unresolved_rows = [
        row
        for row in candidate_actions
        if row["action"] == "replace_control_or_increase_checked_bracket"
    ]
    if promoted:
        next_action = "freeze the promoted candidate and do not launch more follow-up runs"
    elif runnable:
        next_action = "launch the bounded matched-replicate follow-up for the listed local noisy candidate"
    elif nonlocal_rows:
        next_action = "run a smaller-bracket or replacement-control sweep before adding replicas"
    elif unresolved_rows:
        next_action = "choose controls with a resolved heat-flux response before adding replicas"
    else:
        next_action = "inspect artifacts; no safe production follow-up was inferred"

    return {
        "kind": "nonlinear_turbulence_gradient_followup_plan",
        "claim_level": "campaign_planning_not_gradient_evidence",
        "case": case,
        "passed": bool(promoted),
        "next_action": next_action,
        "config": asdict(cfg),
        "summary": {
            "candidate_count": len(candidate_actions),
            "promoted_candidate_count": len(promoted),
            "runnable_followup_candidate_count": len(runnable),
            "planned_run_count": len(all_runs),
        },
        "candidate_actions": candidate_actions,
        "planned_runs": all_runs,
    }
