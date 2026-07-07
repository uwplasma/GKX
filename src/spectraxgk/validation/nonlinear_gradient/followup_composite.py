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


@dataclass(frozen=True)
class _CompositeControlMetrics:
    parameter_name: str
    coefficient: str | None
    response_fraction: float | None
    asymmetry_rel: float | None
    uncertainty_rel: float | None
    central_gradient: float | None
    paired_uncertainty_rel: float | None
    same_sign_fraction: float | None


def _composite_control_metrics(
    artifact: Mapping[str, Any],
    *,
    index: int,
    path: str | None,
    label: str | None,
) -> _CompositeControlMetrics:
    """Extract canonical metrics for one candidate boundary coefficient."""

    parameter_name = str(
        artifact.get("parameter_name") or label or path or f"candidate_{index}"
    )
    return _CompositeControlMetrics(
        parameter_name=parameter_name,
        coefficient=_coefficient_label_from_parameter(artifact.get("parameter_name")),
        response_fraction=_metric(artifact, "response_fraction"),
        asymmetry_rel=_metric(artifact, "fd_asymmetry_rel", "asymmetry_rel"),
        uncertainty_rel=_metric(
            artifact,
            "gradient_uncertainty_rel",
            "gradient_relative_uncertainty",
        ),
        central_gradient=_metric(
            artifact,
            "central_gradient",
            "central_fd_dq_dparameter",
            "central_fd_dq_dtprim",
        ),
        paired_uncertainty_rel=_nested_metric(
            artifact,
            "paired_replicate_diagnostics",
            "central_gradient_uncertainty_rel",
        ),
        same_sign_fraction=_nested_metric(
            artifact,
            "paired_replicate_diagnostics",
            "same_sign_fraction",
        ),
    )


def _composite_control_gate_status(
    metrics: _CompositeControlMetrics,
    *,
    config: NonlinearGradientCompositeControlConfig,
) -> dict[str, bool]:
    """Return per-condition gate flags for a composite-control candidate."""

    return {
        "coefficient_ok": metrics.coefficient is not None,
        "gradient_ok": bool(
            metrics.central_gradient is not None
            and abs(metrics.central_gradient) > config.value_floor
        ),
        "response_ok": bool(
            metrics.response_fraction is not None
            and metrics.response_fraction >= config.min_fd_response_fraction
        ),
        "locality_ok": bool(
            metrics.asymmetry_rel is not None
            and metrics.asymmetry_rel <= config.max_fd_asymmetry_rel
        ),
        "uncertainty_ok": bool(
            metrics.uncertainty_rel is not None
            and metrics.uncertainty_rel <= config.max_gradient_uncertainty_rel
        ),
        "same_sign_ok": bool(
            metrics.same_sign_fraction is None
            or metrics.same_sign_fraction >= config.min_same_sign_fraction
        ),
    }


def _composite_control_blockers(gate_status: Mapping[str, bool]) -> list[str]:
    """Return human-readable blockers for a failed composite-control row."""

    blockers: list[str] = []
    if not gate_status["coefficient_ok"]:
        blockers.append("parameter_not_vmec_boundary_coefficient")
    if not gate_status["gradient_ok"]:
        blockers.append("missing_or_zero_central_gradient")
    if not gate_status["response_ok"]:
        blockers.append("unresolved_heat_flux_response")
    if not gate_status["locality_ok"]:
        blockers.append("nonlocal_finite_difference_bracket")
    if not gate_status["uncertainty_ok"]:
        blockers.append("gradient_uncertainty_too_large")
    if not gate_status["same_sign_ok"]:
        blockers.append("paired_replicate_sign_not_robust")
    return blockers


def _composite_control_metric_payload(
    metrics: _CompositeControlMetrics,
) -> dict[str, Any]:
    """Return JSON-friendly metric payload for one composite-control row."""

    descent_gradient = (
        None if metrics.central_gradient is None else -float(metrics.central_gradient)
    )
    return {
        "central_gradient": _json_number(metrics.central_gradient),
        "descent_gradient": _json_number(descent_gradient),
        "response_fraction": _json_number(metrics.response_fraction),
        "fd_asymmetry_rel": _json_number(metrics.asymmetry_rel),
        "gradient_uncertainty_rel": _json_number(metrics.uncertainty_rel),
        "paired_gradient_uncertainty_rel": _json_number(
            metrics.paired_uncertainty_rel
        ),
        "same_sign_fraction": _json_number(metrics.same_sign_fraction),
    }


def _composite_control_row(
    artifact: Mapping[str, Any],
    *,
    index: int,
    path: str | None,
    label: str | None,
    config: NonlinearGradientCompositeControlConfig,
) -> dict[str, Any]:
    metrics = _composite_control_metrics(
        artifact, index=index, path=path, label=label
    )
    gate_status = _composite_control_gate_status(metrics, config=config)
    admissible = bool(all(gate_status.values()))
    return {
        "index": index,
        "label": str(label or metrics.parameter_name),
        "path": path,
        "parameter_name": metrics.parameter_name,
        "coefficient": metrics.coefficient,
        "admissible_for_composite_direction": admissible,
        "blockers": _composite_control_blockers(gate_status),
        "metrics": _composite_control_metric_payload(metrics),
        "gate_status": gate_status,
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
            "python tools/campaigns/write_vmec_boundary_profile_perturbation_inputs.py "
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
