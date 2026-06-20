"""Gap and production-report orchestration for nonlinear gradient evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from spectraxgk.validation.nonlinear_gradient.evidence_classification import (
    classify_gradient_artifact,
)
from spectraxgk.validation.nonlinear_gradient.evidence_core import (
    NonlinearTurbulenceGradientEvidenceConfig,
    NonlinearTurbulenceGradientGapConfig,
    _gate,
)
from spectraxgk.validation.nonlinear_gradient.evidence_windows import summarize_window_evidence


@dataclass(frozen=True)
class _EvidenceGapContext:
    cfg: NonlinearTurbulenceGradientEvidenceConfig
    gap_cfg: NonlinearTurbulenceGradientGapConfig
    passed: bool
    blockers: list[str]
    gradient: dict[str, Any]
    windows: dict[str, Any]
    qualifying_windows: list[dict[str, Any]]
    failed_gradient_gates: list[dict[str, str]]
    has_production_candidate: bool


def _required_run_rows(
    config: NonlinearTurbulenceGradientGapConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state, multiplier in (
        ("minus_delta", 1.0 - config.perturbation_fraction),
        ("baseline", 1.0),
        ("plus_delta", 1.0 + config.perturbation_fraction),
    ):
        rows.append(
            {
                "state": state,
                "parameter_name": config.parameter_name,
                "parameter_multiplier": multiplier,
                "replicates": list(config.replicate_labels),
                "required_output": (
                    "docs/_static/{case}_{state}_replicates/"
                    "{case}_{state}_t{tmax:g}_ensemble_gate.json"
                ).format(
                    case=config.case_slug,
                    state=state,
                    tmax=config.analysis_tmax,
                ),
                "run_contract": {
                    "same_numerics_except_parameter": True,
                    "t_start": config.t_start,
                    "minimum_tmax": config.minimum_tmax,
                    "analysis_window": [config.analysis_tmin, config.analysis_tmax],
                    "minimum_grid": config.minimum_grid,
                },
            }
        )
    return rows


def _gap_configs(
    *,
    config: NonlinearTurbulenceGradientEvidenceConfig | None,
    gap_config: NonlinearTurbulenceGradientGapConfig | None,
) -> tuple[
    NonlinearTurbulenceGradientEvidenceConfig,
    NonlinearTurbulenceGradientGapConfig,
]:
    return (
        config or NonlinearTurbulenceGradientEvidenceConfig(),
        gap_config or NonlinearTurbulenceGradientGapConfig(),
    )


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _qualifying_window_rows(windows: dict[str, Any]) -> list[dict[str, Any]]:
    rows = windows.get("ensemble_rows", [])
    if not isinstance(rows, Sequence):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict)
        and bool(row.get("qualifies_for_replicated_long_window_uncertainty", False))
    ]


def _failed_gradient_gates(gradient: dict[str, Any]) -> list[dict[str, str]]:
    gates = gradient.get("gates", [])
    if not isinstance(gates, Sequence):
        return []
    return [
        {
            "metric": str(gate.get("metric", "")),
            "detail": str(gate.get("detail", "")),
        }
        for gate in gates
        if isinstance(gate, dict) and not bool(gate.get("passed", False))
    ]


def _gap_context(
    evidence_report: dict[str, Any],
    *,
    cfg: NonlinearTurbulenceGradientEvidenceConfig,
    gap_cfg: NonlinearTurbulenceGradientGapConfig,
) -> _EvidenceGapContext:
    gradient = _dict_or_empty(evidence_report.get("gradient_artifact"))
    windows = _dict_or_empty(evidence_report.get("window_evidence"))
    failed_gradient_gates = _failed_gradient_gates(gradient)
    gradient_class = str(gradient.get("evidence_class", ""))
    return _EvidenceGapContext(
        cfg=cfg,
        gap_cfg=gap_cfg,
        passed=bool(evidence_report.get("passed", False)),
        blockers=[str(item) for item in evidence_report.get("blockers", [])],
        gradient=gradient,
        windows=windows,
        qualifying_windows=_qualifying_window_rows(windows),
        failed_gradient_gates=failed_gradient_gates,
        has_production_candidate=(
            gradient_class == "production_long_window_turbulence_gradient_candidate"
        ),
    )


def _production_gradient_missing_row(ctx: _EvidenceGapContext) -> dict[str, Any]:
    if ctx.has_production_candidate:
        return {
            "blocker": "production_gradient_artifact",
            "needed": (
                "the current matched long-window production-candidate "
                "finite-difference artifact must pass all recorded "
                "response, asymmetry, conditioning, and propagated "
                "gradient-uncertainty gates"
            ),
            "current_artifact_class": ctx.gradient.get("evidence_class"),
            "current_artifact_path": ctx.gradient.get("path"),
            "current_failed_gates": ctx.failed_gradient_gates,
        }
    return {
        "blocker": "production_gradient_artifact",
        "needed": (
            "central finite-difference or adjoint/VJP artifact computed "
            "from matched long post-transient nonlinear heat-flux windows"
        ),
        "current_artifact_class": ctx.gradient.get("evidence_class"),
        "current_artifact_path": ctx.gradient.get("path"),
    }


def _replicated_window_missing_row(ctx: _EvidenceGapContext) -> dict[str, Any]:
    return {
        "blocker": "replicated_long_window_uncertainty",
        "needed": (
            "at least one baseline/plus/minus campaign with replicated "
            "post-transient transport-window ensemble gates"
        ),
        "qualifying_window_ensembles": len(ctx.qualifying_windows),
    }


def _missing_evidence_rows(ctx: _EvidenceGapContext) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    if "production_gradient_artifact" in ctx.blockers:
        missing.append(_production_gradient_missing_row(ctx))
    if "replicated_long_window_uncertainty" in ctx.blockers:
        missing.append(_replicated_window_missing_row(ctx))
    return missing


def _finite_difference_audit_contract(
    *,
    cfg: NonlinearTurbulenceGradientEvidenceConfig,
    gap_cfg: NonlinearTurbulenceGradientGapConfig,
) -> dict[str, Any]:
    return {
        "required_output": (
            "docs/_static/{case}_{parameter}_central_fd_gradient_gate.json"
        ).format(
            case=gap_cfg.case_slug,
            parameter=gap_cfg.parameter_name,
        ),
        "formula": "dQ/dp = (mean(Q_plus) - mean(Q_minus)) / (2 * delta_p)",
        "required_metrics": [
            "central_gradient",
            "response_fraction",
            "fd_asymmetry_rel",
            "fd_condition_number",
            "gradient_uncertainty_rel",
            "baseline_window_mean",
            "plus_window_mean",
            "minus_window_mean",
            "baseline_window_sem",
            "plus_window_sem",
            "minus_window_sem",
        ],
        "acceptance_gates": {
            "production_nonlinear_window_gradient_gate": True,
            "response_fraction_min": cfg.min_fd_response_fraction,
            "fd_asymmetry_rel_max": cfg.max_fd_asymmetry_rel,
            "fd_condition_number_max": cfg.max_fd_condition_number,
            "gradient_uncertainty_rel_max": cfg.max_gradient_uncertainty_rel,
            "window_mean_rel_spread_max": cfg.max_window_mean_rel_spread,
            "window_combined_sem_rel_max": cfg.max_window_combined_sem_rel,
        },
        "fallback_if_marginal": (
            "repeat the paired campaign with a second perturbation fraction "
            "or longer analysis_tmax; do not promote if the response is not "
            "resolved above the transport-window uncertainty."
        ),
    }


def _claim_level(ctx: _EvidenceGapContext) -> str:
    if ctx.has_production_candidate and not ctx.passed:
        return "fail_closed_production_candidate_gradient_gate_not_resolved"
    return "fail_closed_missing_campaign_plan_not_gradient_evidence"


def _required_campaign(
    *,
    gap_cfg: NonlinearTurbulenceGradientGapConfig,
    finite_difference_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_slug": gap_cfg.case_slug,
        "parameter_name": gap_cfg.parameter_name,
        "perturbation_fraction": gap_cfg.perturbation_fraction,
        "required_runs": _required_run_rows(gap_cfg),
        "finite_difference_audit": finite_difference_audit,
    }


def _gradient_evidence_requirements() -> list[str]:
    return [
        "run baseline, plus-delta, and minus-delta nonlinear simulations with identical numerical settings except the perturbed parameter",
        "use the same seed/timestep replicate labels for all three parameter states",
        "discard the startup transient and average only over the declared post-transient analysis window",
        "build passed ensemble gates for baseline, plus, and minus states before computing the gradient",
        "record finite-difference response, asymmetry, condition number, and gradient uncertainty in the production gradient artifact",
    ]


def nonlinear_turbulence_gradient_evidence_gap_report(
    evidence_report: dict[str, Any],
    *,
    config: NonlinearTurbulenceGradientEvidenceConfig | None = None,
    gap_config: NonlinearTurbulenceGradientGapConfig | None = None,
) -> dict[str, Any]:
    """Return the fail-closed run campaign needed to close gradient evidence.

    The report is deliberately prescriptive: it requires paired plus/minus
    long-window nonlinear runs with the same seeds, timestep variant, grid, and
    post-transient analysis window before a finite-difference turbulence
    gradient can be promoted.  It does not infer a gradient from standalone
    replicated transport windows.
    """

    cfg, gap_cfg = _gap_configs(config=config, gap_config=gap_config)
    ctx = _gap_context(evidence_report, cfg=cfg, gap_cfg=gap_cfg)
    finite_difference_audit = _finite_difference_audit_contract(
        cfg=cfg, gap_cfg=gap_cfg
    )
    return {
        "kind": "nonlinear_turbulence_gradient_evidence_gap_report",
        "claim_level": _claim_level(ctx),
        "passed": ctx.passed,
        "promotion_blocked": not ctx.passed,
        "blockers": ctx.blockers,
        "missing_evidence": _missing_evidence_rows(ctx),
        "current_gradient_candidate_present": ctx.has_production_candidate,
        "current_gradient_failed_gates": ctx.failed_gradient_gates,
        "current_window_evidence_passed": bool(ctx.windows.get("passed", False)),
        "qualifying_window_ensemble_count": len(ctx.qualifying_windows),
        "required_campaign": _required_campaign(
            gap_cfg=gap_cfg, finite_difference_audit=finite_difference_audit
        ),
        "requirements": _gradient_evidence_requirements(),
        "notes": (
            "Standalone passed transport windows are necessary but not sufficient: "
            "production turbulence-gradient evidence requires paired parameter "
            "perturbations tied to the same post-transient averaging protocol."
        ),
    }


def nonlinear_turbulence_gradient_evidence_report(
    gradient_artifact: dict[str, Any],
    *,
    window_artifacts: Sequence[dict[str, Any]] = (),
    gradient_path: str | None = None,
    window_paths: Sequence[str | None] | None = None,
    config: NonlinearTurbulenceGradientEvidenceConfig | None = None,
    gap_config: NonlinearTurbulenceGradientGapConfig | None = None,
) -> dict[str, Any]:
    """Return a fail-closed production nonlinear gradient evidence report."""

    cfg = config or NonlinearTurbulenceGradientEvidenceConfig()
    gradient = classify_gradient_artifact(
        gradient_artifact,
        path=gradient_path,
        config=cfg,
    )
    windows = summarize_window_evidence(
        list(window_artifacts),
        paths=window_paths,
        config=cfg,
    )
    gates = [
        _gate(
            "production_gradient_artifact",
            bool(gradient["qualifies_for_production_turbulence_gradient"]),
            str(gradient["evidence_class"]),
        ),
        *windows["gates"],
    ]
    passed = all(bool(gate["passed"]) for gate in gates)
    blockers = [str(gate["metric"]) for gate in gates if not bool(gate["passed"])]
    report = {
        "kind": "nonlinear_turbulence_gradient_evidence_report",
        "claim_level": "fail_closed_claim_boundary_for_long_window_nonlinear_turbulence_gradient_evidence",
        "passed": passed,
        "production_nonlinear_window_gradient_gate": passed,
        "blockers": blockers,
        "requirements": [
            "gradient artifact must explicitly claim production long-window nonlinear turbulence-gradient scope",
            "startup/reduced-window finite-difference or estimator artifacts are recorded but never promoted",
            "finite-difference response, asymmetry, and condition number must be recorded and within gates",
            "gradient uncertainty must be recorded and within gate",
            "replicated post-transient nonlinear-window summaries must pass ensemble uncertainty gates",
        ],
        "config": asdict(cfg),
        "gates": gates,
        "gradient_artifact": gradient,
        "window_evidence": windows,
        "notes": (
            "This checker distinguishes claim boundaries only.  Passing it means "
            "the supplied artifacts meet the recorded evidence contract; it does "
            "not run or certify new nonlinear simulations."
        ),
    }
    report["evidence_gap"] = nonlinear_turbulence_gradient_evidence_gap_report(
        report,
        config=cfg,
        gap_config=gap_config,
    )
    return report


__all__ = [
    "_required_run_rows",
    "nonlinear_turbulence_gradient_evidence_gap_report",
    "nonlinear_turbulence_gradient_evidence_report",
]
