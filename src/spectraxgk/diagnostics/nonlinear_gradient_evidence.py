"""Nonlinear turbulent-gradient evidence and claim-boundary diagnostics.

This module contains pure JSON/dictionary diagnostics used to decide whether
long-window nonlinear transport-gradient artifacts support production claims.
It does not launch simulations. Campaign design and follow-up planning live in
``tools.campaigns.nonlinear_gradient_followup``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

from spectraxgk.diagnostics.metadata import (
    NON_PRODUCTION_SCOPE_MARKERS,
    NonlinearTurbulenceGradientBracketSweepConfig,
    NonlinearTurbulenceGradientCandidateRankingConfig,
    NonlinearTurbulenceGradientEvidenceConfig,
    NonlinearTurbulenceGradientFiniteDifferenceConfig,
    NonlinearTurbulenceGradientGapConfig,
    _finite_float,
    _gate,
    _json_number,
    _bracket_sweep_row,
    _metric_margin,
    classify_gradient_artifact,
)
from spectraxgk.diagnostics.nonlinear_replicates import (
    summarize_window_evidence,
)
from spectraxgk.diagnostics.transport import (
    nonlinear_turbulence_gradient_finite_difference_report,
)

# ---- shared parsing and gate primitives ----
"""Core helpers for nonlinear turbulence-gradient evidence gates.

The public :mod:`spectraxgk.diagnostics.nonlinear_gradient_evidence` facade owns artifact
report assembly.  This module keeps the small, deterministic claim-boundary
pieces separate so they can be tested and reused without importing the full
reporting layer.
"""


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
    "nonlinear_turbulence_gradient_bracket_sweep_report",
]

# ---- screening reports ----
"""Screening reports for nonlinear turbulence-gradient campaign planning."""


def _candidate_next_action(
    *,
    passed: bool,
    response_margin: float,
    asymmetry_margin: float,
    uncertainty_margin: float,
    condition_margin: float,
) -> str:
    if passed:
        return (
            "promote only after the source campaign provenance is frozen in docs and CI"
        )
    if response_margin < 1.0:
        return (
            "abandon or enlarge the perturbation only if locality remains bounded; "
            "the heat-flux response is not resolved above the minimum response gate"
        )
    if asymmetry_margin < 1.0 and uncertainty_margin >= 1.0:
        return (
            "repeat with a smaller bracket or nearby control; uncertainty is adequate "
            "but the finite-difference response is nonlocal"
        )
    if uncertainty_margin < 1.0 and asymmetry_margin >= 1.0:
        return (
            "keep the local direction but increase statistical power: longer windows, "
            "more replicas, or a checked amplitude bracket"
        )
    if condition_margin < 1.0:
        return "choose a better-conditioned observable/control pair before rerunning"
    return (
        "do not relax gates; move to an overdetermined least-squares/profile-gradient "
        "campaign with multiple controls and matched long-window replicas"
    )


def _ranking_evidence_config(
    cfg: NonlinearTurbulenceGradientCandidateRankingConfig,
) -> NonlinearTurbulenceGradientEvidenceConfig:
    return NonlinearTurbulenceGradientEvidenceConfig(
        max_gradient_uncertainty_rel=cfg.max_gradient_uncertainty_rel,
        max_fd_asymmetry_rel=cfg.max_fd_asymmetry_rel,
        max_fd_condition_number=cfg.max_fd_condition_number,
        min_fd_response_fraction=cfg.min_fd_response_fraction,
        value_floor=cfg.value_floor,
    )


def _conditioning_metrics(classified: dict[str, Any]) -> dict[str, Any]:
    conditioning = classified.get("conditioning", {})
    return conditioning if isinstance(conditioning, dict) else {}


def _candidate_margins(
    conditioning: dict[str, Any],
    cfg: NonlinearTurbulenceGradientCandidateRankingConfig,
) -> dict[str, float]:
    response_fraction = _finite_float(conditioning.get("response_fraction"))
    fd_asymmetry_rel = _finite_float(conditioning.get("fd_asymmetry_rel"))
    fd_condition_number = _finite_float(conditioning.get("fd_condition_number"))
    gradient_uncertainty_rel = _finite_float(
        conditioning.get("gradient_uncertainty_rel")
    )
    return {
        "response": _metric_margin(
            response_fraction,
            target=cfg.min_fd_response_fraction,
            sense="min",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        ),
        "locality": _metric_margin(
            fd_asymmetry_rel,
            target=cfg.max_fd_asymmetry_rel,
            sense="max",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        ),
        "conditioning": _metric_margin(
            fd_condition_number,
            target=cfg.max_fd_condition_number,
            sense="max",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        ),
        "uncertainty": _metric_margin(
            gradient_uncertainty_rel,
            target=cfg.max_gradient_uncertainty_rel,
            sense="max",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        ),
    }


def _candidate_failed_gates(classified: dict[str, Any]) -> list[str]:
    return [
        str(gate.get("metric", ""))
        for gate in classified.get("gates", [])
        if isinstance(gate, dict) and not bool(gate.get("passed", False))
    ]


def _candidate_score(
    margins: dict[str, float],
    *,
    explicit_production_scope: bool,
) -> tuple[float, float]:
    weakest_margin = min(margins.values())
    geometric_score = math.prod(max(value, 0.0) for value in margins.values()) ** 0.25
    if not explicit_production_scope:
        geometric_score *= 0.5
    return weakest_margin, geometric_score


def _candidate_metric_payload(conditioning: dict[str, Any]) -> dict[str, Any]:
    return {
        "central_gradient": conditioning.get("central_gradient"),
        "response_fraction": conditioning.get("response_fraction"),
        "fd_asymmetry_rel": conditioning.get("fd_asymmetry_rel"),
        "fd_condition_number": conditioning.get("fd_condition_number"),
        "gradient_uncertainty_rel": conditioning.get("gradient_uncertainty_rel"),
    }


def _candidate_ranking_row(
    *,
    artifact: dict[str, Any],
    path: str | None,
    label: str | None,
    index: int,
    cfg: NonlinearTurbulenceGradientCandidateRankingConfig,
) -> dict[str, Any]:
    classified = classify_gradient_artifact(
        artifact,
        path=path,
        config=_ranking_evidence_config(cfg),
    )
    conditioning = _conditioning_metrics(classified)
    margins = _candidate_margins(conditioning, cfg)
    weakest_margin, geometric_score = _candidate_score(
        margins,
        explicit_production_scope=bool(
            classified.get("explicit_production_scope", False)
        ),
    )
    passed = bool(classified.get("qualifies_for_production_turbulence_gradient", False))
    parameter_name = str(
        artifact.get("parameter_name") or label or path or f"candidate_{index}"
    )
    return {
        "rank": None,
        "index": index,
        "label": str(label or parameter_name),
        "path": path,
        "parameter_name": parameter_name,
        "passed": passed,
        "evidence_class": classified.get("evidence_class"),
        "failed_gates": _candidate_failed_gates(classified),
        "metrics": _candidate_metric_payload(conditioning),
        "margins": margins,
        "weakest_margin": _json_number(weakest_margin),
        "score": _json_number(geometric_score),
        "next_action": _candidate_next_action(
            passed=passed,
            response_margin=margins["response"],
            asymmetry_margin=margins["locality"],
            uncertainty_margin=margins["uncertainty"],
            condition_margin=margins["conditioning"],
        ),
    }


def _rank_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows.sort(
        key=lambda row: (
            bool(row["passed"]),
            float(row["weakest_margin"] or 0.0),
            float(row["score"] or 0.0),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _candidate_followup_groups(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    passed_rows = [row for row in rows if bool(row["passed"])]
    local_but_noisy = [
        row
        for row in rows
        if float(row["margins"]["locality"]) >= 1.0
        and float(row["margins"]["uncertainty"]) < 1.0
    ]
    quiet_but_nonlocal = [
        row
        for row in rows
        if float(row["margins"]["uncertainty"]) >= 1.0
        and float(row["margins"]["locality"]) < 1.0
    ]
    return passed_rows, local_but_noisy, quiet_but_nonlocal


def _ranking_recommendation(
    *,
    passed_rows: list[dict[str, Any]],
    local_but_noisy: list[dict[str, Any]],
    quiet_but_nonlocal: list[dict[str, Any]],
    overdetermined_followup: bool,
) -> str:
    if passed_rows:
        return (
            "one or more candidates passes the production evidence gates; freeze "
            "the provenance and promote only the passed artifact"
        )
    if overdetermined_followup and local_but_noisy and quiet_but_nonlocal:
        return (
            "the overdetermined follow-up completed with no promotable candidate; "
            "keep the nonlinear-gradient claim fail-closed, target the best local "
            "but noisy control with additional independent replicas or variance "
            "reduction only if the cost is justified, and replace or shrink the "
            "nonlocal controls before another production campaign"
        )
    if overdetermined_followup and local_but_noisy:
        return (
            "the overdetermined follow-up found local but statistically unresolved "
            "candidates; keep the claim fail-closed and add independent replicas "
            "or a lower-variance observable before promotion"
        )
    if overdetermined_followup and quiet_but_nonlocal:
        return (
            "the overdetermined follow-up found statistically quiet but nonlocal "
            "candidates; keep the claim fail-closed and shrink the perturbation "
            "or choose more local controls before adding replicas"
        )
    if local_but_noisy and quiet_but_nonlocal:
        return (
            "use an overdetermined least-squares/profile-gradient campaign: current "
            "single-control candidates have complementary locality and uncertainty failures"
        )
    if local_but_noisy:
        return "extend statistical power for the best local direction before changing controls"
    if quiet_but_nonlocal:
        return "reduce bracket size or choose a nearby/local control before adding replicas"
    return (
        "screen new profile-gradient or objective-gradient controls; current candidates "
        "do not isolate a promotable response"
    )


def _pack_candidate_ranking_report(
    *,
    rows: list[dict[str, Any]],
    passed_rows: list[dict[str, Any]],
    recommendation: str,
    cfg: NonlinearTurbulenceGradientCandidateRankingConfig,
) -> dict[str, Any]:
    return {
        "kind": "nonlinear_turbulence_gradient_candidate_ranking",
        "claim_level": "campaign_planning_not_gradient_evidence",
        "passed": bool(passed_rows),
        "promotion_ready_candidate_count": len(passed_rows),
        "best_candidate": rows[0] if rows else None,
        "recommendation": recommendation,
        "config": asdict(cfg),
        "candidates": rows,
    }


def nonlinear_turbulence_gradient_candidate_ranking_report(
    artifacts: Sequence[dict[str, Any]],
    *,
    paths: Sequence[str | None] | None = None,
    labels: Sequence[str | None] | None = None,
    config: NonlinearTurbulenceGradientCandidateRankingConfig | None = None,
) -> dict[str, Any]:
    """Rank nonlinear turbulence-gradient candidates without promoting failures.

    The ranking is a planning aid, not a replacement for the production gate.
    It scores each candidate by the weakest normalized evidence margin across
    response, locality, conditioning, and uncertainty.  This makes the next
    campaign choice explicit: candidates with complementary failures should
    move to a profile-gradient or overdetermined least-squares design instead
    of repeating a single boundary coefficient indefinitely.
    """

    cfg = config or NonlinearTurbulenceGradientCandidateRankingConfig()
    path_list = list(paths or [None] * len(artifacts))
    label_list = list(labels or [None] * len(artifacts))
    if len(path_list) != len(artifacts):
        raise ValueError("paths length must match artifacts")
    if len(label_list) != len(artifacts):
        raise ValueError("labels length must match artifacts")

    rows = _rank_candidate_rows(
        [
            _candidate_ranking_row(
                artifact=artifact,
                path=path,
                label=label,
                index=index,
                cfg=cfg,
            )
            for index, (artifact, path, label) in enumerate(
                zip(artifacts, path_list, label_list)
            )
        ]
    )
    passed_rows, local_but_noisy, quiet_but_nonlocal = _candidate_followup_groups(rows)
    recommendation = _ranking_recommendation(
        passed_rows=passed_rows,
        local_but_noisy=local_but_noisy,
        quiet_but_nonlocal=quiet_but_nonlocal,
        overdetermined_followup=cfg.campaign_context == "overdetermined_followup",
    )
    return _pack_candidate_ranking_report(
        rows=rows,
        passed_rows=passed_rows,
        recommendation=recommendation,
        cfg=cfg,
    )


__all__ = [
    "_bracket_sweep_recommendation",
    "_bracket_sweep_row",
    "_candidate_next_action",
    "_delta_key",
    "_metric_margin",
    "nonlinear_turbulence_gradient_bracket_sweep_report",
    "nonlinear_turbulence_gradient_candidate_ranking_report",
]

# ---- evidence-gap reports ----
"""Gap and production-report orchestration for nonlinear gradient evidence."""


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


def _qualifying_gap_window_rows(windows: dict[str, Any]) -> list[dict[str, Any]]:
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
        qualifying_windows=_qualifying_gap_window_rows(windows),
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

# ---- production evidence entry points ----
"""Claim-boundary gates for nonlinear turbulence-gradient evidence.

This module is intentionally data-only.  It does not run nonlinear solves and
does not infer production turbulence-gradient support from startup finite
differences, reduced nonlinear-window estimators, or single late-window
summaries.  The default behavior is fail-closed unless an artifact explicitly
records production long-window gradient scope, finite-difference conditioning,
gradient uncertainty, and replicated nonlinear-window uncertainty evidence.
"""


def load_json_artifact(path: str | Path) -> dict[str, Any]:
    """Load a JSON object artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


__all__ = [
    "NON_PRODUCTION_SCOPE_MARKERS",
    "NonlinearTurbulenceGradientBracketSweepConfig",
    "NonlinearTurbulenceGradientCandidateRankingConfig",
    "NonlinearTurbulenceGradientEvidenceConfig",
    "NonlinearTurbulenceGradientFiniteDifferenceConfig",
    "NonlinearTurbulenceGradientGapConfig",
    "classify_gradient_artifact",
    "load_json_artifact",
    "nonlinear_turbulence_gradient_bracket_sweep_report",
    "nonlinear_turbulence_gradient_candidate_ranking_report",
    "nonlinear_turbulence_gradient_evidence_gap_report",
    "nonlinear_turbulence_gradient_evidence_report",
    "nonlinear_turbulence_gradient_finite_difference_report",
    "summarize_window_evidence",
]
