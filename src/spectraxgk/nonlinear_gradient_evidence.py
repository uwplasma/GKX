"""Claim-boundary gates for nonlinear turbulence-gradient evidence.

This module is intentionally data-only.  It does not run nonlinear solves and
does not infer production turbulence-gradient support from startup finite
differences, reduced nonlinear-window estimators, or single late-window
summaries.  The default behavior is fail-closed unless an artifact explicitly
records production long-window gradient scope, finite-difference conditioning,
gradient uncertainty, and replicated nonlinear-window uncertainty evidence.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence
import json
import math

from spectraxgk.quasilinear_window import (
    NonlinearWindowEnsembleConfig,
    nonlinear_window_ensemble_report,
    nonlinear_window_stats_promotion_ready,
)


from spectraxgk.nonlinear_gradient_evidence_core import (
    NON_PRODUCTION_SCOPE_MARKERS,
    NonlinearTurbulenceGradientBracketSweepConfig,
    NonlinearTurbulenceGradientCandidateRankingConfig,
    NonlinearTurbulenceGradientEvidenceConfig,
    NonlinearTurbulenceGradientFiniteDifferenceConfig,
    NonlinearTurbulenceGradientGapConfig,
    _artifact_passed,
    _ensemble_statistics_row,
    _explicit_production_scope,
    _finite_float,
    _gate,
    _gradient_conditioning_summary,
    _json_number,
    _paired_replicate_fd_diagnostics,
    _scope_blockers,
)


def classify_gradient_artifact(
    payload: dict[str, Any],
    *,
    path: str | None = None,
    config: NonlinearTurbulenceGradientEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Classify a gradient/FD artifact without promoting ambiguous evidence."""

    cfg = config or NonlinearTurbulenceGradientEvidenceConfig()
    kind = str(payload.get("kind", ""))
    blockers = _scope_blockers(payload)
    explicit_production = _explicit_production_scope(payload)
    conditioning = _gradient_conditioning_summary(payload, config=cfg)
    passed = _artifact_passed(payload)
    production_scope = bool(explicit_production and not blockers)
    qualifies = bool(passed and production_scope and conditioning["passed"])

    if blockers:
        evidence_class = "startup_or_reduced_window_fd_not_production"
    elif explicit_production:
        evidence_class = "production_long_window_turbulence_gradient_candidate"
    else:
        evidence_class = "unscoped_gradient_or_fd_artifact_not_production"

    gates = [
        _gate("artifact_passed", passed, f"kind={kind}"),
        _gate(
            "explicit_production_long_window_scope",
            production_scope,
            "explicit_production_scope={scope} scope_blockers={blockers}".format(
                scope=explicit_production,
                blockers=blockers,
            ),
        ),
        *conditioning["gates"],
    ]
    return {
        "path": path,
        "kind": kind,
        "claim_level": str(payload.get("claim_level", "")),
        "claim_scope": str(payload.get("claim_scope", "")),
        "evidence_class": evidence_class,
        "artifact_passed": passed,
        "explicit_production_scope": explicit_production,
        "scope_blockers": blockers,
        "conditioning": {
            key: value
            for key, value in conditioning.items()
            if key not in {"gates"}
        },
        "gates": gates,
        "qualifies_for_production_turbulence_gradient": qualifies,
    }


def _ensemble_row(
    payload: dict[str, Any],
    *,
    path: str | None,
    source: str,
    config: NonlinearTurbulenceGradientEvidenceConfig,
) -> dict[str, Any]:
    statistics = payload.get("statistics")
    if not isinstance(statistics, dict):
        statistics = {}
    n_reports = _finite_float(statistics.get("n_reports"))
    combined_sem_rel = _finite_float(statistics.get("combined_sem_rel"))
    mean_rel_spread = _finite_float(statistics.get("mean_rel_spread"))
    passed = _artifact_passed(payload)
    qualifies = bool(
        passed
        and n_reports is not None
        and int(n_reports) >= int(config.min_window_reports)
        and combined_sem_rel is not None
        and combined_sem_rel <= float(config.max_window_combined_sem_rel)
        and mean_rel_spread is not None
        and mean_rel_spread <= float(config.max_window_mean_rel_spread)
    )
    return {
        "path": path,
        "source": source,
        "kind": str(payload.get("kind", "")),
        "passed": passed,
        "n_reports": None if n_reports is None else int(n_reports),
        "combined_sem_rel": _json_number(combined_sem_rel),
        "mean_rel_spread": _json_number(mean_rel_spread),
        "qualifies_for_replicated_long_window_uncertainty": qualifies,
        "statistics": statistics,
    }


def summarize_window_evidence(
    window_artifacts: Sequence[dict[str, Any]],
    *,
    paths: Sequence[str | None] | None = None,
    config: NonlinearTurbulenceGradientEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Summarize replicated long-window uncertainty evidence.

    Existing ``nonlinear_window_ensemble_report`` artifacts are consumed
    directly.  If only individual ``nonlinear_window_convergence_report``
    summaries are supplied, a derived ensemble is built from those summaries
    using the configured uncertainty limits.
    """

    cfg = config or NonlinearTurbulenceGradientEvidenceConfig()
    path_list = list(paths or [None] * len(window_artifacts))
    if len(path_list) != len(window_artifacts):
        raise ValueError("paths length must match window_artifacts length")

    rows: list[dict[str, Any]] = []
    convergence_reports: list[dict[str, Any]] = []
    convergence_paths: list[str | None] = []
    single_window_rows: list[dict[str, Any]] = []

    for payload, path in zip(window_artifacts, path_list):
        kind = str(payload.get("kind", ""))
        if kind == "nonlinear_window_ensemble_report":
            rows.append(_ensemble_row(payload, path=path, source="input_ensemble", config=cfg))
        elif kind == "nonlinear_window_convergence_report":
            ready, failures = nonlinear_window_stats_promotion_ready(payload)
            single_window_rows.append(
                {
                    "path": path,
                    "kind": kind,
                    "case": str(payload.get("case", "")),
                    "passed": _artifact_passed(payload),
                    "promotion_ready": ready,
                    "failures": failures,
                }
            )
            convergence_reports.append(payload)
            convergence_paths.append(path)
        else:
            rows.append(
                {
                    "path": path,
                    "source": "unsupported_window_artifact",
                    "kind": kind,
                    "passed": _artifact_passed(payload),
                    "qualifies_for_replicated_long_window_uncertainty": False,
                }
            )

    derived_ensemble = None
    if len(convergence_reports) >= int(cfg.min_window_reports):
        derived_payload = nonlinear_window_ensemble_report(
            convergence_reports,
            case="derived_long_window_replicate_evidence",
            comparison="derived_from_supplied_window_summaries",
            config=NonlinearWindowEnsembleConfig(
                min_reports=cfg.min_window_reports,
                max_mean_rel_spread=cfg.max_window_mean_rel_spread,
                max_combined_sem_rel=cfg.max_window_combined_sem_rel,
                value_floor=cfg.value_floor,
                require_individual_passed=True,
            ),
        )
        derived_ensemble = _ensemble_row(
            derived_payload,
            path=None,
            source="derived_from_window_summaries",
            config=cfg,
        )
        derived_ensemble["input_paths"] = convergence_paths
        rows.append(derived_ensemble)

    qualifying_rows = [
        row
        for row in rows
        if bool(row.get("qualifies_for_replicated_long_window_uncertainty", False))
    ]
    gates = [
        _gate(
            "replicated_long_window_uncertainty",
            bool(qualifying_rows),
            "qualifying_ensembles={count} min_window_reports={min_reports}".format(
                count=len(qualifying_rows),
                min_reports=cfg.min_window_reports,
            ),
        )
    ]
    return {
        "passed": bool(qualifying_rows),
        "gates": gates,
        "ensemble_rows": rows,
        "single_window_rows": single_window_rows,
        "derived_ensemble": derived_ensemble,
    }


def nonlinear_turbulence_gradient_finite_difference_report(
    *,
    baseline: dict[str, Any],
    plus: dict[str, Any],
    minus: dict[str, Any],
    delta_parameter: float,
    parameter_name: str,
    baseline_path: str | None = None,
    plus_path: str | None = None,
    minus_path: str | None = None,
    config: NonlinearTurbulenceGradientFiniteDifferenceConfig | None = None,
) -> dict[str, Any]:
    """Build a production long-window central finite-difference gradient gate.

    Inputs must be replicated ``nonlinear_window_ensemble_report`` payloads for
    the same nonlinear case and analysis window, differing only by the perturbed
    parameter.  The report computes the central finite-difference heat-flux
    gradient and checks that the response is resolved above ensemble
    uncertainty before allowing any turbulence-gradient claim.
    """

    cfg = config or NonlinearTurbulenceGradientFiniteDifferenceConfig()
    delta = float(delta_parameter)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta_parameter must be finite and positive")

    rows = {
        "minus": _ensemble_statistics_row(minus, path=minus_path),
        "baseline": _ensemble_statistics_row(baseline, path=baseline_path),
        "plus": _ensemble_statistics_row(plus, path=plus_path),
    }
    means = {name: _finite_float(row.get("ensemble_mean")) for name, row in rows.items()}
    sems = {name: _finite_float(row.get("combined_sem")) for name, row in rows.items()}
    finite_means = all(value is not None for value in means.values())
    finite_sems = all(value is not None for value in sems.values())

    if finite_means:
        assert means["minus"] is not None
        assert means["baseline"] is not None
        assert means["plus"] is not None
        minus_mean = float(means["minus"])
        baseline_mean = float(means["baseline"])
        plus_mean = float(means["plus"])
        central_gradient = (plus_mean - minus_mean) / (2.0 * delta)
        forward_gradient = (plus_mean - baseline_mean) / delta
        backward_gradient = (baseline_mean - minus_mean) / delta
        response = abs(plus_mean - minus_mean)
        response_fraction = response / max(abs(baseline_mean), float(cfg.value_floor))
        fd_asymmetry_rel = abs(forward_gradient - backward_gradient) / max(
            abs(central_gradient),
            float(cfg.value_floor),
        )
        fd_condition_number = (abs(plus_mean) + abs(minus_mean)) / max(
            response,
            float(cfg.value_floor),
        )
    else:
        central_gradient = math.nan
        forward_gradient = math.nan
        backward_gradient = math.nan
        response = math.nan
        response_fraction = math.nan
        fd_asymmetry_rel = math.nan
        fd_condition_number = math.nan

    if finite_sems:
        assert sems["plus"] is not None
        assert sems["minus"] is not None
        gradient_uncertainty = math.sqrt(float(sems["plus"]) ** 2 + float(sems["minus"]) ** 2) / (
            2.0 * delta
        )
        gradient_uncertainty_rel = gradient_uncertainty / max(
            abs(central_gradient) if math.isfinite(central_gradient) else 0.0,
            float(cfg.value_floor),
        )
    else:
        gradient_uncertainty = math.nan
        gradient_uncertainty_rel = math.nan

    source_gates: list[dict[str, Any]] = []
    for name, row in rows.items():
        n_reports = _finite_float(row.get("n_reports"))
        source_gates.extend(
            [
                _gate(
                    f"{name}_ensemble_kind",
                    row.get("kind") == "nonlinear_window_ensemble_report",
                    f"kind={row.get('kind')}",
                ),
                _gate(f"{name}_ensemble_passed", bool(row["passed"]), f"path={row.get('path')}"),
                _gate(
                    f"{name}_ensemble_replicated",
                    n_reports is not None and n_reports >= int(cfg.min_window_reports),
                    f"n_reports={n_reports} min={cfg.min_window_reports}",
                ),
            ]
        )
    window_gates: list[dict[str, Any]] = []
    for name, row in rows.items():
        mean_rel_spread = _finite_float(row.get("mean_rel_spread"))
        combined_sem_rel = _finite_float(row.get("combined_sem_rel"))
        window_gates.extend(
            [
                _gate(
                    f"{name}_window_mean_spread",
                    mean_rel_spread is not None
                    and mean_rel_spread <= float(cfg.max_window_mean_rel_spread),
                    f"mean_rel_spread={mean_rel_spread} max={cfg.max_window_mean_rel_spread}",
                ),
                _gate(
                    f"{name}_window_sem",
                    combined_sem_rel is not None
                    and combined_sem_rel <= float(cfg.max_window_combined_sem_rel),
                    f"combined_sem_rel={combined_sem_rel} max={cfg.max_window_combined_sem_rel}",
                ),
            ]
        )

    gradient_gates = [
        _gate("finite_window_means", finite_means, f"means={means}"),
        _gate("finite_window_uncertainties", finite_sems, f"combined_sem={sems}"),
        _gate(
            "fd_response_resolved",
            math.isfinite(response_fraction)
            and response_fraction >= float(cfg.min_fd_response_fraction),
            f"response_fraction={response_fraction} min={cfg.min_fd_response_fraction}",
        ),
        _gate(
            "fd_asymmetry_bounded",
            math.isfinite(fd_asymmetry_rel)
            and fd_asymmetry_rel <= float(cfg.max_fd_asymmetry_rel),
            f"fd_asymmetry_rel={fd_asymmetry_rel} max={cfg.max_fd_asymmetry_rel}",
        ),
        _gate(
            "fd_condition_number_bounded",
            math.isfinite(fd_condition_number)
            and fd_condition_number <= float(cfg.max_fd_condition_number),
            f"fd_condition_number={fd_condition_number} max={cfg.max_fd_condition_number}",
        ),
        _gate(
            "gradient_uncertainty_bounded",
            math.isfinite(gradient_uncertainty_rel)
            and gradient_uncertainty_rel <= float(cfg.max_gradient_uncertainty_rel),
            f"gradient_uncertainty_rel={gradient_uncertainty_rel} max={cfg.max_gradient_uncertainty_rel}",
        ),
    ]
    gates = [*source_gates, *window_gates, *gradient_gates]
    passed = all(bool(gate["passed"]) for gate in gates)
    return {
        "kind": "nonlinear_turbulence_gradient_central_fd_gate",
        "claim_level": "production_long_window_nonlinear_turbulence_gradient_candidate",
        "claim_scope": (
            "production_long_window nonlinear turbulence gradient from matched replicated "
            "post-transient heat-flux windows"
        ),
        "parameter_name": str(parameter_name),
        "delta_parameter": delta,
        "passed": passed,
        "production_nonlinear_window_gradient_gate": passed,
        "nonlinear_turbulence_gradient_gate": passed,
        "metrics": {
            "central_gradient": _json_number(central_gradient),
            "forward_gradient": _json_number(forward_gradient),
            "backward_gradient": _json_number(backward_gradient),
            "response": _json_number(response),
            "response_fraction": _json_number(response_fraction),
            "fd_asymmetry_rel": _json_number(fd_asymmetry_rel),
            "asymmetry_rel": _json_number(fd_asymmetry_rel),
            "fd_condition_number": _json_number(fd_condition_number),
            "condition_number": _json_number(fd_condition_number),
            "gradient_uncertainty": _json_number(gradient_uncertainty),
            "gradient_uncertainty_rel": _json_number(gradient_uncertainty_rel),
            "gradient_relative_uncertainty": _json_number(gradient_uncertainty_rel),
            "baseline_window_mean": means["baseline"],
            "plus_window_mean": means["plus"],
            "minus_window_mean": means["minus"],
            "baseline_window_sem": sems["baseline"],
            "plus_window_sem": sems["plus"],
            "minus_window_sem": sems["minus"],
        },
        "source_ensembles": rows,
        "paired_replicate_diagnostics": _paired_replicate_fd_diagnostics(
            rows=rows,
            delta=delta,
            value_floor=float(cfg.value_floor),
        ),
        "config": asdict(cfg),
        "gates": gates,
        "blockers": [gate["metric"] for gate in gates if not bool(gate["passed"])],
    }


def _metric_margin(
    value: float | None,
    *,
    target: float,
    sense: str,
    cap: float,
    value_floor: float,
) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    finite_value = float(value)
    if sense == "min":
        margin = finite_value / max(float(target), float(value_floor))
    elif sense == "max":
        margin = float(target) / max(abs(finite_value), float(value_floor))
    else:  # pragma: no cover - guarded by internal call sites.
        raise ValueError(f"unsupported margin sense: {sense}")
    return max(0.0, min(float(cap), margin))


def _candidate_next_action(
    *,
    passed: bool,
    response_margin: float,
    asymmetry_margin: float,
    uncertainty_margin: float,
    condition_margin: float,
) -> str:
    if passed:
        return "promote only after the source campaign provenance is frozen in docs and CI"
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

    rows: list[dict[str, Any]] = []
    for index, (artifact, path, label) in enumerate(zip(artifacts, path_list, label_list)):
        evidence_cfg = NonlinearTurbulenceGradientEvidenceConfig(
            max_gradient_uncertainty_rel=cfg.max_gradient_uncertainty_rel,
            max_fd_asymmetry_rel=cfg.max_fd_asymmetry_rel,
            max_fd_condition_number=cfg.max_fd_condition_number,
            min_fd_response_fraction=cfg.min_fd_response_fraction,
            value_floor=cfg.value_floor,
        )
        classified = classify_gradient_artifact(artifact, path=path, config=evidence_cfg)
        conditioning = classified.get("conditioning", {})
        if not isinstance(conditioning, dict):
            conditioning = {}
        response_fraction = _finite_float(conditioning.get("response_fraction"))
        fd_asymmetry_rel = _finite_float(conditioning.get("fd_asymmetry_rel"))
        fd_condition_number = _finite_float(conditioning.get("fd_condition_number"))
        gradient_uncertainty_rel = _finite_float(conditioning.get("gradient_uncertainty_rel"))

        response_margin = _metric_margin(
            response_fraction,
            target=cfg.min_fd_response_fraction,
            sense="min",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        )
        asymmetry_margin = _metric_margin(
            fd_asymmetry_rel,
            target=cfg.max_fd_asymmetry_rel,
            sense="max",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        )
        condition_margin = _metric_margin(
            fd_condition_number,
            target=cfg.max_fd_condition_number,
            sense="max",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        )
        uncertainty_margin = _metric_margin(
            gradient_uncertainty_rel,
            target=cfg.max_gradient_uncertainty_rel,
            sense="max",
            cap=cfg.score_cap,
            value_floor=cfg.value_floor,
        )
        margins = {
            "response": response_margin,
            "locality": asymmetry_margin,
            "conditioning": condition_margin,
            "uncertainty": uncertainty_margin,
        }
        weakest_margin = min(margins.values())
        geometric_score = math.prod(max(value, 0.0) for value in margins.values()) ** 0.25
        if not bool(classified.get("explicit_production_scope", False)):
            geometric_score *= 0.5
        passed = bool(classified.get("qualifies_for_production_turbulence_gradient", False))
        parameter_name = str(artifact.get("parameter_name") or label or path or f"candidate_{index}")
        failed_gates = [
            str(gate.get("metric", ""))
            for gate in classified.get("gates", [])
            if isinstance(gate, dict) and not bool(gate.get("passed", False))
        ]
        rows.append(
            {
                "rank": None,
                "index": index,
                "label": str(label or parameter_name),
                "path": path,
                "parameter_name": parameter_name,
                "passed": passed,
                "evidence_class": classified.get("evidence_class"),
                "failed_gates": failed_gates,
                "metrics": {
                    "central_gradient": conditioning.get("central_gradient"),
                    "response_fraction": conditioning.get("response_fraction"),
                    "fd_asymmetry_rel": conditioning.get("fd_asymmetry_rel"),
                    "fd_condition_number": conditioning.get("fd_condition_number"),
                    "gradient_uncertainty_rel": conditioning.get("gradient_uncertainty_rel"),
                },
                "margins": margins,
                "weakest_margin": _json_number(weakest_margin),
                "score": _json_number(geometric_score),
                "next_action": _candidate_next_action(
                    passed=passed,
                    response_margin=response_margin,
                    asymmetry_margin=asymmetry_margin,
                    uncertainty_margin=uncertainty_margin,
                    condition_margin=condition_margin,
                ),
            }
        )

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

    passed_rows = [row for row in rows if bool(row["passed"])]
    local_but_noisy = [
        row
        for row in rows
        if float(row["margins"]["locality"]) >= 1.0 and float(row["margins"]["uncertainty"]) < 1.0
    ]
    quiet_but_nonlocal = [
        row
        for row in rows
        if float(row["margins"]["uncertainty"]) >= 1.0 and float(row["margins"]["locality"]) < 1.0
    ]
    overdetermined_followup = cfg.campaign_context == "overdetermined_followup"
    if passed_rows:
        recommendation = (
            "one or more candidates passes the production evidence gates; freeze "
            "the provenance and promote only the passed artifact"
        )
    elif overdetermined_followup and local_but_noisy and quiet_but_nonlocal:
        recommendation = (
            "the overdetermined follow-up completed with no promotable candidate; "
            "keep the nonlinear-gradient claim fail-closed, target the best local "
            "but noisy control with additional independent replicas or variance "
            "reduction only if the cost is justified, and replace or shrink the "
            "nonlocal controls before another production campaign"
        )
    elif overdetermined_followup and local_but_noisy:
        recommendation = (
            "the overdetermined follow-up found local but statistically unresolved "
            "candidates; keep the claim fail-closed and add independent replicas "
            "or a lower-variance observable before promotion"
        )
    elif overdetermined_followup and quiet_but_nonlocal:
        recommendation = (
            "the overdetermined follow-up found statistically quiet but nonlocal "
            "candidates; keep the claim fail-closed and shrink the perturbation "
            "or choose more local controls before adding replicas"
        )
    elif local_but_noisy and quiet_but_nonlocal:
        recommendation = (
            "use an overdetermined least-squares/profile-gradient campaign: current "
            "single-control candidates have complementary locality and uncertainty failures"
        )
    elif local_but_noisy:
        recommendation = (
            "extend statistical power for the best local direction before changing controls"
        )
    elif quiet_but_nonlocal:
        recommendation = (
            "reduce bracket size or choose a nearby/local control before adding replicas"
        )
    else:
        recommendation = (
            "screen new profile-gradient or objective-gradient controls; current candidates "
            "do not isolate a promotable response"
        )

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


def _bracket_sweep_row(
    artifact: dict[str, Any],
    *,
    label: str | None,
    path: str | None,
    config: NonlinearTurbulenceGradientBracketSweepConfig,
) -> dict[str, Any]:
    evidence_cfg = NonlinearTurbulenceGradientEvidenceConfig(
        max_gradient_uncertainty_rel=config.max_gradient_uncertainty_rel,
        max_fd_asymmetry_rel=config.max_fd_asymmetry_rel,
        max_fd_condition_number=config.max_fd_condition_number,
        min_fd_response_fraction=config.min_fd_response_fraction,
        value_floor=config.value_floor,
    )
    classified = classify_gradient_artifact(artifact, path=path, config=evidence_cfg)
    conditioning = classified.get("conditioning")
    if not isinstance(conditioning, dict):
        conditioning = {}
    delta = _finite_float(artifact.get("delta_parameter"))
    response_fraction = _finite_float(conditioning.get("response_fraction"))
    fd_asymmetry_rel = _finite_float(conditioning.get("fd_asymmetry_rel"))
    fd_condition_number = _finite_float(conditioning.get("fd_condition_number"))
    gradient_uncertainty_rel = _finite_float(conditioning.get("gradient_uncertainty_rel"))
    paired_uncertainty_rel = _paired_uncertainty_rel(artifact)
    paired_same_sign = _paired_same_sign_fraction(artifact)
    response_margin = _metric_margin(
        response_fraction,
        target=config.min_fd_response_fraction,
        sense="min",
        cap=config.score_cap,
        value_floor=config.value_floor,
    )
    locality_margin = _metric_margin(
        fd_asymmetry_rel,
        target=config.max_fd_asymmetry_rel,
        sense="max",
        cap=config.score_cap,
        value_floor=config.value_floor,
    )
    uncertainty_margin = _metric_margin(
        gradient_uncertainty_rel,
        target=config.max_gradient_uncertainty_rel,
        sense="max",
        cap=config.score_cap,
        value_floor=config.value_floor,
    )
    condition_margin = _metric_margin(
        fd_condition_number,
        target=config.max_fd_condition_number,
        sense="max",
        cap=config.score_cap,
        value_floor=config.value_floor,
    )
    margins = {
        "response": response_margin,
        "locality": locality_margin,
        "uncertainty": uncertainty_margin,
        "conditioning": condition_margin,
    }
    repeated_bracket_stable = (
        paired_uncertainty_rel is not None
        and paired_uncertainty_rel <= float(config.max_repeated_bracket_uncertainty_rel)
        and paired_same_sign is not None
        and paired_same_sign >= float(config.min_repeated_bracket_same_sign_fraction)
    )
    return {
        "label": str(label or artifact.get("parameter_name") or path or ""),
        "path": path,
        "parameter_name": str(artifact.get("parameter_name", "")),
        "delta_parameter": _json_number(delta),
        "passed": bool(classified.get("qualifies_for_production_turbulence_gradient", False)),
        "metrics": {
            "central_gradient": conditioning.get("central_gradient"),
            "response_fraction": conditioning.get("response_fraction"),
            "fd_asymmetry_rel": conditioning.get("fd_asymmetry_rel"),
            "fd_condition_number": conditioning.get("fd_condition_number"),
            "gradient_uncertainty_rel": conditioning.get("gradient_uncertainty_rel"),
            "paired_gradient_uncertainty_rel": _json_number(paired_uncertainty_rel),
            "paired_same_sign_fraction": _json_number(paired_same_sign),
            "repeated_bracket_stable": repeated_bracket_stable,
        },
        "margins": margins,
        "weakest_margin": _json_number(min(margins.values())),
        "score": _json_number(math.prod(max(value, 0.0) for value in margins.values()) ** 0.25),
        "failed_gates": [
            str(gate.get("metric", ""))
            for gate in classified.get("gates", [])
            if isinstance(gate, dict) and not bool(gate.get("passed", False))
        ],
    }


def _delta_key(row: dict[str, Any]) -> float:
    delta = _finite_float(row.get("delta_parameter"))
    if delta is None:
        return math.inf
    return float(delta)


def _bracket_sweep_recommendation(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "run at least two matched plus/minus perturbation amplitudes before claiming bracket locality"
    parameter_names = {str(row.get("parameter_name", "")) for row in rows if row.get("parameter_name")}
    if len(parameter_names) > 1:
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

    response_ok = [
        row
        for row in rows
        if float(row["margins"]["response"]) >= 1.0
    ]
    response_ok_gradients = [
        _finite_float(row.get("metrics", {}).get("central_gradient"))
        for row in response_ok
        if isinstance(row.get("metrics"), dict)
    ]
    response_ok_signs = {
        math.copysign(1.0, float(value))
        for value in response_ok_gradients
        if value is not None and value != 0.0
    }
    if len(response_ok_signs) > 1:
        return (
            "same-control resolved brackets change central-gradient sign; do not add "
            "replicas at one amplitude, and move to a locality/amplitude sweep with "
            "stricter provenance or a smoother composite profile-gradient direction"
        )
    local_rows = [
        row
        for row in response_ok
        if float(row["margins"]["locality"]) >= 1.0
    ]
    quiet_rows = [
        row
        for row in response_ok
        if float(row["margins"]["uncertainty"]) >= 1.0
    ]
    repeated_unstable = [
        row
        for row in rows
        if row["metrics"].get("paired_gradient_uncertainty_rel") is not None
        and not bool(row["metrics"].get("repeated_bracket_stable", False))
    ]
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
    parameter_names = sorted({row["parameter_name"] for row in rows if row["parameter_name"]})
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


def _required_run_rows(config: NonlinearTurbulenceGradientGapConfig) -> list[dict[str, Any]]:
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

    cfg = config or NonlinearTurbulenceGradientEvidenceConfig()
    gap_cfg = gap_config or NonlinearTurbulenceGradientGapConfig()
    passed = bool(evidence_report.get("passed", False))
    blockers = [str(item) for item in evidence_report.get("blockers", [])]
    gradient = evidence_report.get("gradient_artifact")
    if not isinstance(gradient, dict):
        gradient = {}
    windows = evidence_report.get("window_evidence")
    if not isinstance(windows, dict):
        windows = {}
    qualifying_windows = [
        row
        for row in windows.get("ensemble_rows", [])
        if isinstance(row, dict)
        and bool(row.get("qualifies_for_replicated_long_window_uncertainty", False))
    ]
    failed_gradient_gates = [
        {
            "metric": str(gate.get("metric", "")),
            "detail": str(gate.get("detail", "")),
        }
        for gate in gradient.get("gates", [])
        if isinstance(gate, dict) and not bool(gate.get("passed", False))
    ]
    gradient_class = str(gradient.get("evidence_class", ""))
    has_production_candidate = (
        gradient_class == "production_long_window_turbulence_gradient_candidate"
    )
    missing: list[dict[str, Any]] = []
    if "production_gradient_artifact" in blockers:
        if has_production_candidate:
            missing.append(
                {
                    "blocker": "production_gradient_artifact",
                    "needed": (
                        "the current matched long-window production-candidate "
                        "finite-difference artifact must pass all recorded "
                        "response, asymmetry, conditioning, and propagated "
                        "gradient-uncertainty gates"
                    ),
                    "current_artifact_class": gradient.get("evidence_class"),
                    "current_artifact_path": gradient.get("path"),
                    "current_failed_gates": failed_gradient_gates,
                }
            )
        else:
            missing.append(
                {
                    "blocker": "production_gradient_artifact",
                    "needed": (
                        "central finite-difference or adjoint/VJP artifact computed "
                        "from matched long post-transient nonlinear heat-flux windows"
                    ),
                    "current_artifact_class": gradient.get("evidence_class"),
                    "current_artifact_path": gradient.get("path"),
                }
            )
    if "replicated_long_window_uncertainty" in blockers:
        missing.append(
            {
                "blocker": "replicated_long_window_uncertainty",
                "needed": (
                    "at least one baseline/plus/minus campaign with replicated "
                    "post-transient transport-window ensemble gates"
                ),
                "qualifying_window_ensembles": len(qualifying_windows),
            }
        )

    finite_difference_audit = {
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
    return {
        "kind": "nonlinear_turbulence_gradient_evidence_gap_report",
        "claim_level": (
            "fail_closed_production_candidate_gradient_gate_not_resolved"
            if has_production_candidate and not passed
            else "fail_closed_missing_campaign_plan_not_gradient_evidence"
        ),
        "passed": passed,
        "promotion_blocked": not passed,
        "blockers": blockers,
        "missing_evidence": missing,
        "current_gradient_candidate_present": has_production_candidate,
        "current_gradient_failed_gates": failed_gradient_gates,
        "current_window_evidence_passed": bool(windows.get("passed", False)),
        "qualifying_window_ensemble_count": len(qualifying_windows),
        "required_campaign": {
            "case_slug": gap_cfg.case_slug,
            "parameter_name": gap_cfg.parameter_name,
            "perturbation_fraction": gap_cfg.perturbation_fraction,
            "required_runs": _required_run_rows(gap_cfg),
            "finite_difference_audit": finite_difference_audit,
        },
        "requirements": [
            "run baseline, plus-delta, and minus-delta nonlinear simulations with identical numerical settings except the perturbed parameter",
            "use the same seed/timestep replicate labels for all three parameter states",
            "discard the startup transient and average only over the declared post-transient analysis window",
            "build passed ensemble gates for baseline, plus, and minus states before computing the gradient",
            "record finite-difference response, asymmetry, condition number, and gradient uncertainty in the production gradient artifact",
        ],
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
