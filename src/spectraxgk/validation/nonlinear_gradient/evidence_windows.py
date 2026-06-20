"""Replicated nonlinear-window evidence summaries for turbulence-gradient gates."""

from __future__ import annotations

from typing import Any, Sequence

from spectraxgk.validation.nonlinear_gradient.evidence_core import (
    NonlinearTurbulenceGradientEvidenceConfig,
    _artifact_passed,
    _finite_float,
    _gate,
    _json_number,
)
from spectraxgk.validation.quasilinear.window_config import (
    NonlinearWindowEnsembleConfig,
)
from spectraxgk.validation.quasilinear.window_ensemble import (
    nonlinear_window_ensemble_report,
)
from spectraxgk.validation.quasilinear.window_promotion import (
    nonlinear_window_stats_promotion_ready,
)


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
            rows.append(
                _ensemble_row(payload, path=path, source="input_ensemble", config=cfg)
            )
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


__all__ = ["_ensemble_row", "summarize_window_evidence"]
