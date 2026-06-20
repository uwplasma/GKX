"""Composite-control reports for nonlinear turbulence-gradient follow-up."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from spectraxgk.validation.nonlinear_gradient.followup_core import (
    NonlinearGradientCompositeControlConfig,
    _coefficient_label_from_parameter,
    _json_number,
    _metric,
    _nested_metric,
)


@dataclass(frozen=True)
class _CompositeLaunchPlan:
    launch_ready: bool
    command_template: str | None
    next_action: str


def _composite_control_row(
    artifact: Mapping[str, Any],
    *,
    index: int,
    path: str | None,
    label: str | None,
    config: NonlinearGradientCompositeControlConfig,
) -> dict[str, Any]:
    parameter_name = str(
        artifact.get("parameter_name") or label or path or f"candidate_{index}"
    )
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
    response_ok = (
        response_fraction is not None
        and response_fraction >= config.min_fd_response_fraction
    )
    locality_ok = (
        asymmetry_rel is not None and asymmetry_rel <= config.max_fd_asymmetry_rel
    )
    uncertainty_ok = (
        uncertainty_rel is not None
        and uncertainty_rel <= config.max_gradient_uncertainty_rel
    )
    same_sign_ok = (
        same_sign_fraction is None
        or same_sign_fraction >= config.min_same_sign_fraction
    )
    gradient_ok = (
        central_gradient is not None and abs(central_gradient) > config.value_floor
    )
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


def _validate_composite_control_config(
    cfg: NonlinearGradientCompositeControlConfig,
) -> None:
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


def _normalized_composite_metadata(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    paths: Sequence[str | None] | None,
    labels: Sequence[str | None] | None,
) -> tuple[list[str | None], list[str | None]]:
    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")
    return path_list, label_list


def _composite_control_rows(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    paths: Sequence[str | None],
    labels: Sequence[str | None],
    config: NonlinearGradientCompositeControlConfig,
) -> list[dict[str, Any]]:
    return [
        _composite_control_row(
            artifact, index=index, path=path, label=label, config=config
        )
        for index, (artifact, path, label) in enumerate(
            zip(artifacts, paths, labels)
        )
    ]


def _composite_controls(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: NonlinearGradientCompositeControlConfig,
) -> list[dict[str, Any]]:
    admissible_rows = [
        row for row in rows if bool(row["admissible_for_composite_direction"])
    ]
    max_abs_descent = max(
        (abs(float(row["metrics"]["descent_gradient"])) for row in admissible_rows),
        default=0.0,
    )
    controls: list[dict[str, Any]] = []
    if max_abs_descent <= config.value_floor:
        return controls
    for row in admissible_rows:
        descent = float(row["metrics"]["descent_gradient"])
        weight = config.max_weight_abs * descent / max_abs_descent
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
    return controls


def _composite_launch_plan(
    *,
    controls: Sequence[Mapping[str, Any]],
    case: str,
    config: NonlinearGradientCompositeControlConfig,
) -> _CompositeLaunchPlan:
    launch_ready = len(controls) >= config.min_controls
    if launch_ready:
        control_args = " ".join(
            f"--control {control['control_argument']}" for control in controls
        )
        command_template = (
            "python tools/write_vmec_boundary_profile_perturbation_inputs.py "
            "--baseline-input <input.vmec> "
            "--out-dir docs/_static/<case>_composite_direction "
            f"--case {case} "
            f"{control_args} "
            f"--relative-delta {config.default_relative_delta:.12g}"
        )
        return _CompositeLaunchPlan(
            launch_ready=True,
            command_template=command_template,
            next_action=(
                "launch a checked VMEC profile-direction bracket sweep before "
                "long nonlinear windows"
            ),
        )
    if controls:
        return _CompositeLaunchPlan(
            launch_ready=False,
            command_template=None,
            next_action=(
                "only one admissible control remains; screen additional "
                "local/resolved controls or explicitly run a single-control "
                "bracket check before a long campaign"
            ),
        )
    return _CompositeLaunchPlan(
        launch_ready=False,
        command_template=None,
        next_action=(
            "no admissible controls; screen new VMEC-boundary directions before "
            "nonlinear GPU runs"
        ),
    )


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
    _validate_composite_control_config(cfg)
    path_list, label_list = _normalized_composite_metadata(
        artifacts, paths=paths, labels=labels
    )
    rows = _composite_control_rows(
        artifacts, paths=path_list, labels=label_list, config=cfg
    )
    controls = _composite_controls(rows, config=cfg)
    launch_plan = _composite_launch_plan(controls=controls, case=case, config=cfg)

    return {
        "kind": "nonlinear_turbulence_gradient_composite_control_design",
        "claim_level": "composite_control_launch_plan_not_gradient_evidence",
        "case": case,
        "passed": bool(launch_plan.launch_ready),
        "next_action": launch_plan.next_action,
        "config": asdict(cfg),
        "summary": {
            "candidate_count": len(rows),
            "admissible_control_count": len(controls),
            "required_control_count": cfg.min_controls,
            "launch_ready": bool(launch_plan.launch_ready),
        },
        "controls": controls,
        "write_profile_direction_command_template": launch_plan.command_template,
        "candidates": rows,
    }


__all__ = ["_composite_control_row", "nonlinear_gradient_composite_control_report"]
