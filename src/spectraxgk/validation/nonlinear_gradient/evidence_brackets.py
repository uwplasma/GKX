"""Same-control bracket-sweep reports for nonlinear-gradient evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence
import math

from spectraxgk.validation.nonlinear_gradient.evidence_classification import (
    classify_gradient_artifact,
)
from spectraxgk.validation.nonlinear_gradient.evidence_core import (
    NonlinearTurbulenceGradientBracketSweepConfig,
    NonlinearTurbulenceGradientEvidenceConfig,
    _finite_float,
    _json_number,
)
from spectraxgk.validation.nonlinear_gradient.evidence_scoring import _metric_margin


def _paired_uncertainty_rel(artifact: dict[str, Any]) -> float | None:
    diagnostics = artifact.get("paired_replicate_diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    return _finite_float(diagnostics.get("central_gradient_uncertainty_rel"))


def _paired_same_sign_fraction(artifact: dict[str, Any]) -> float | None:
    diagnostics = artifact.get("paired_replicate_diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    return _finite_float(diagnostics.get("same_sign_fraction"))


@dataclass(frozen=True)
class _BracketConditioningMetrics:
    central_gradient: Any
    response_fraction: float | None
    fd_asymmetry_rel: float | None
    fd_condition_number: float | None
    gradient_uncertainty_rel: float | None
    paired_uncertainty_rel: float | None
    paired_same_sign_fraction: float | None


def _bracket_evidence_config(
    config: NonlinearTurbulenceGradientBracketSweepConfig,
) -> NonlinearTurbulenceGradientEvidenceConfig:
    return NonlinearTurbulenceGradientEvidenceConfig(
        max_gradient_uncertainty_rel=config.max_gradient_uncertainty_rel,
        max_fd_asymmetry_rel=config.max_fd_asymmetry_rel,
        max_fd_condition_number=config.max_fd_condition_number,
        min_fd_response_fraction=config.min_fd_response_fraction,
        value_floor=config.value_floor,
    )


def _bracket_conditioning_metrics(
    artifact: dict[str, Any],
    classified: dict[str, Any],
) -> _BracketConditioningMetrics:
    conditioning = classified.get("conditioning")
    if not isinstance(conditioning, dict):
        conditioning = {}
    return _BracketConditioningMetrics(
        central_gradient=conditioning.get("central_gradient"),
        response_fraction=_finite_float(conditioning.get("response_fraction")),
        fd_asymmetry_rel=_finite_float(conditioning.get("fd_asymmetry_rel")),
        fd_condition_number=_finite_float(conditioning.get("fd_condition_number")),
        gradient_uncertainty_rel=_finite_float(
            conditioning.get("gradient_uncertainty_rel")
        ),
        paired_uncertainty_rel=_paired_uncertainty_rel(artifact),
        paired_same_sign_fraction=_paired_same_sign_fraction(artifact),
    )


def _bracket_margin_scores(
    metrics: _BracketConditioningMetrics,
    config: NonlinearTurbulenceGradientBracketSweepConfig,
) -> dict[str, float]:
    return {
        "response": _metric_margin(
            metrics.response_fraction,
            target=config.min_fd_response_fraction,
            sense="min",
            cap=config.score_cap,
            value_floor=config.value_floor,
        ),
        "locality": _metric_margin(
            metrics.fd_asymmetry_rel,
            target=config.max_fd_asymmetry_rel,
            sense="max",
            cap=config.score_cap,
            value_floor=config.value_floor,
        ),
        "uncertainty": _metric_margin(
            metrics.gradient_uncertainty_rel,
            target=config.max_gradient_uncertainty_rel,
            sense="max",
            cap=config.score_cap,
            value_floor=config.value_floor,
        ),
        "conditioning": _metric_margin(
            metrics.fd_condition_number,
            target=config.max_fd_condition_number,
            sense="max",
            cap=config.score_cap,
            value_floor=config.value_floor,
        ),
    }


def _repeated_bracket_stable(
    metrics: _BracketConditioningMetrics,
    config: NonlinearTurbulenceGradientBracketSweepConfig,
) -> bool:
    return (
        metrics.paired_uncertainty_rel is not None
        and metrics.paired_uncertainty_rel
        <= float(config.max_repeated_bracket_uncertainty_rel)
        and metrics.paired_same_sign_fraction is not None
        and metrics.paired_same_sign_fraction
        >= float(config.min_repeated_bracket_same_sign_fraction)
    )


def _failed_bracket_gate_names(classified: dict[str, Any]) -> list[str]:
    return [
        str(gate.get("metric", ""))
        for gate in classified.get("gates", [])
        if isinstance(gate, dict) and not bool(gate.get("passed", False))
    ]


def _bracket_sweep_row(
    artifact: dict[str, Any],
    *,
    label: str | None,
    path: str | None,
    config: NonlinearTurbulenceGradientBracketSweepConfig,
) -> dict[str, Any]:
    classified = classify_gradient_artifact(
        artifact,
        path=path,
        config=_bracket_evidence_config(config),
    )
    metrics = _bracket_conditioning_metrics(artifact, classified)
    delta = _finite_float(artifact.get("delta_parameter"))
    margins = _bracket_margin_scores(metrics, config)
    return {
        "label": str(label or artifact.get("parameter_name") or path or ""),
        "path": path,
        "parameter_name": str(artifact.get("parameter_name", "")),
        "delta_parameter": _json_number(delta),
        "passed": bool(
            classified.get("qualifies_for_production_turbulence_gradient", False)
        ),
        "metrics": {
            "central_gradient": metrics.central_gradient,
            "response_fraction": metrics.response_fraction,
            "fd_asymmetry_rel": metrics.fd_asymmetry_rel,
            "fd_condition_number": metrics.fd_condition_number,
            "gradient_uncertainty_rel": metrics.gradient_uncertainty_rel,
            "paired_gradient_uncertainty_rel": _json_number(
                metrics.paired_uncertainty_rel
            ),
            "paired_same_sign_fraction": _json_number(
                metrics.paired_same_sign_fraction
            ),
            "repeated_bracket_stable": _repeated_bracket_stable(metrics, config),
        },
        "margins": margins,
        "weakest_margin": _json_number(min(margins.values())),
        "score": _json_number(
            math.prod(max(value, 0.0) for value in margins.values()) ** 0.25
        ),
        "failed_gates": _failed_bracket_gate_names(classified),
    }


def _delta_key(row: dict[str, Any]) -> float:
    delta = _finite_float(row.get("delta_parameter"))
    if delta is None:
        return math.inf
    return float(delta)


def _bracket_parameter_names(rows: Sequence[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("parameter_name", "")) for row in rows if row.get("parameter_name")
    }


def _response_ok_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if float(row["margins"]["response"]) >= 1.0]


def _response_ok_signs(rows: Sequence[dict[str, Any]]) -> set[float]:
    gradients = [
        _finite_float(row.get("metrics", {}).get("central_gradient"))
        for row in rows
        if isinstance(row.get("metrics"), dict)
    ]
    return {
        math.copysign(1.0, float(value))
        for value in gradients
        if value is not None and value != 0.0
    }


def _rows_with_margin(
    rows: Sequence[dict[str, Any]],
    margin_name: str,
) -> list[dict[str, Any]]:
    return [row for row in rows if float(row["margins"][margin_name]) >= 1.0]


def _repeated_unstable_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["metrics"].get("paired_gradient_uncertainty_rel") is not None
        and not bool(row["metrics"].get("repeated_bracket_stable", False))
    ]


def _initial_bracket_sweep_recommendation(rows: Sequence[dict[str, Any]]) -> str | None:
    if not rows:
        return "run at least two matched plus/minus perturbation amplitudes before claiming bracket locality"
    if len(_bracket_parameter_names(rows)) > 1:
        return (
            "mixed controls were supplied to a same-control bracket sweep; split the "
            "artifacts by control or use the nonlinear turbulence-gradient candidate "
            "ranking/overdetermined campaign planner"
        )
    passed_rows = [row for row in rows if bool(row.get("passed", False))]
    if passed_rows:
        best = min(passed_rows, key=_delta_key)
        return (
            "promote only the passed same-control bracket after freezing provenance; "
            f"smallest passing delta is {best.get('delta_parameter')}"
        )
    return None


def _resolved_bracket_sweep_recommendation(
    *,
    response_ok: Sequence[dict[str, Any]],
    local_rows: Sequence[dict[str, Any]],
    quiet_rows: Sequence[dict[str, Any]],
    repeated_unstable: Sequence[dict[str, Any]],
) -> str:
    if local_rows and not quiet_rows:
        if repeated_unstable:
            return (
                "do not add replicas at the same bracket yet; matched-pair diagnostics "
                "show seed-level instability, so run a perturbation-amplitude/locality "
                "sweep or switch to a smoother composite profile-gradient direction"
            )
        return (
            "locality is acceptable but uncertainty is not; add statistical power only "
            "after a second nearby perturbation amplitude confirms the same gradient sign"
        )
    if quiet_rows and not local_rows:
        return (
            "uncertainty is acceptable only for nonlocal brackets; shrink the perturbation "
            "or choose a more local control before adding replicas"
        )
    if response_ok and local_rows and quiet_rows:
        return (
            "the numerical bracket margins are resolved, local, and quiet, but no input "
            "artifact has production long-window scope; rerun or re-export with matched "
            "post-transient provenance before considering promotion"
        )
    if response_ok and not local_rows and not quiet_rows:
        return (
            "the response is detectable but neither local nor statistically resolved; "
            "prefer an overdetermined/profile-gradient campaign over more single-control runs"
        )
    return (
        "the heat-flux response is not resolved at the tested amplitudes; abandon this "
        "control or enlarge the perturbation only if a locality sweep remains bounded"
    )


def _bracket_sweep_recommendation(rows: Sequence[dict[str, Any]]) -> str:
    initial = _initial_bracket_sweep_recommendation(rows)
    if initial is not None:
        return initial
    response_ok = _response_ok_rows(rows)
    if len(_response_ok_signs(response_ok)) > 1:
        return (
            "same-control resolved brackets change central-gradient sign; do not add "
            "replicas at one amplitude, and move to a locality/amplitude sweep with "
            "stricter provenance or a smoother composite profile-gradient direction"
        )
    return _resolved_bracket_sweep_recommendation(
        response_ok=response_ok,
        local_rows=_rows_with_margin(response_ok, "locality"),
        quiet_rows=_rows_with_margin(response_ok, "uncertainty"),
        repeated_unstable=_repeated_unstable_rows(rows),
    )


def nonlinear_turbulence_gradient_bracket_sweep_report(
    artifacts: Sequence[dict[str, Any]],
    *,
    labels: Sequence[str | None] | None = None,
    paths: Sequence[str | None] | None = None,
    config: NonlinearTurbulenceGradientBracketSweepConfig | None = None,
) -> dict[str, Any]:
    """Summarize a same-control perturbation-amplitude sweep.

    This is a planning/claim-boundary utility.  It does not promote nonlinear
    turbulence-gradient evidence unless an input finite-difference artifact
    already passes the production long-window gate.  Its main purpose is to
    decide whether the next expensive campaign should add replicas at the same
    bracket, change the perturbation amplitude, or move to an overdetermined
    profile-gradient direction.
    """

    cfg = config or NonlinearTurbulenceGradientBracketSweepConfig()
    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")

    rows = [
        _bracket_sweep_row(artifact, label=label, path=path, config=cfg)
        for artifact, label, path in zip(artifacts, label_list, path_list)
    ]
    rows.sort(key=_delta_key)
    parameter_names = sorted(
        {row["parameter_name"] for row in rows if row["parameter_name"]}
    )
    same_control = len(parameter_names) <= 1
    passed_rows = [row for row in rows if bool(row.get("passed", False))]
    return {
        "kind": "nonlinear_turbulence_gradient_bracket_sweep",
        "claim_level": "same_control_bracket_locality_planning_not_gradient_promotion",
        "passed": bool(passed_rows) and same_control,
        "promotion_ready_bracket_count": len(passed_rows) if same_control else 0,
        "same_control_gate": {
            "passed": same_control,
            "parameter_names": parameter_names,
        },
        "parameter_names": parameter_names,
        "recommendation": _bracket_sweep_recommendation(rows),
        "config": asdict(cfg),
        "brackets": rows,
    }


__all__ = [
    "_bracket_sweep_recommendation",
    "_bracket_sweep_row",
    "_delta_key",
    "_paired_same_sign_fraction",
    "_paired_uncertainty_rel",
    "nonlinear_turbulence_gradient_bracket_sweep_report",
]
