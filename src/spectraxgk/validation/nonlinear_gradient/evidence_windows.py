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


def _single_window_row(
    payload: dict[str, Any],
    *,
    path: str | None,
    kind: str,
) -> dict[str, Any]:
    """Return a row for one convergence-window summary."""

    ready, failures = nonlinear_window_stats_promotion_ready(payload)
    return {
        "path": path,
        "kind": kind,
        "case": str(payload.get("case", "")),
        "passed": _artifact_passed(payload),
        "promotion_ready": ready,
        "failures": failures,
    }


def _unsupported_window_row(
    payload: dict[str, Any],
    *,
    path: str | None,
    kind: str,
) -> dict[str, Any]:
    """Return a non-qualifying row for an unsupported window artifact."""

    return {
        "path": path,
        "source": "unsupported_window_artifact",
        "kind": kind,
        "passed": _artifact_passed(payload),
        "qualifies_for_replicated_long_window_uncertainty": False,
    }


def _collect_window_artifact_rows(
    window_artifacts: Sequence[dict[str, Any]],
    path_list: Sequence[str | None],
    *,
    config: NonlinearTurbulenceGradientEvidenceConfig,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str | None],
]:
    """Classify input window artifacts into ensemble and convergence rows."""

    rows: list[dict[str, Any]] = []
    convergence_reports: list[dict[str, Any]] = []
    convergence_paths: list[str | None] = []
    single_window_rows: list[dict[str, Any]] = []

    for payload, path in zip(window_artifacts, path_list):
        kind = str(payload.get("kind", ""))
        if kind == "nonlinear_window_ensemble_report":
            rows.append(
                _ensemble_row(
                    payload,
                    path=path,
                    source="input_ensemble",
                    config=config,
                )
            )
        elif kind == "nonlinear_window_convergence_report":
            single_window_rows.append(
                _single_window_row(payload, path=path, kind=kind)
            )
            convergence_reports.append(payload)
            convergence_paths.append(path)
        else:
            rows.append(_unsupported_window_row(payload, path=path, kind=kind))
    return rows, single_window_rows, convergence_reports, convergence_paths


def _derived_ensemble_row(
    convergence_reports: Sequence[dict[str, Any]],
    convergence_paths: Sequence[str | None],
    *,
    config: NonlinearTurbulenceGradientEvidenceConfig,
) -> dict[str, Any] | None:
    """Build a replicated-window ensemble row from individual windows."""

    if len(convergence_reports) < int(config.min_window_reports):
        return None
    derived_payload = nonlinear_window_ensemble_report(
        convergence_reports,
        case="derived_long_window_replicate_evidence",
        comparison="derived_from_supplied_window_summaries",
        config=NonlinearWindowEnsembleConfig(
            min_reports=config.min_window_reports,
            max_mean_rel_spread=config.max_window_mean_rel_spread,
            max_combined_sem_rel=config.max_window_combined_sem_rel,
            value_floor=config.value_floor,
            require_individual_passed=True,
        ),
    )
    row = _ensemble_row(
        derived_payload,
        path=None,
        source="derived_from_window_summaries",
        config=config,
    )
    row["input_paths"] = list(convergence_paths)
    return row


def _qualifying_window_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ensemble rows that qualify as replicated long-window evidence."""

    return [
        row
        for row in rows
        if bool(row.get("qualifies_for_replicated_long_window_uncertainty", False))
    ]


def _window_evidence_gates(
    qualifying_rows: Sequence[dict[str, Any]],
    *,
    config: NonlinearTurbulenceGradientEvidenceConfig,
) -> list[dict[str, Any]]:
    """Return the gate list for replicated long-window uncertainty evidence."""

    return [
        _gate(
            "replicated_long_window_uncertainty",
            bool(qualifying_rows),
            "qualifying_ensembles={count} min_window_reports={min_reports}".format(
                count=len(qualifying_rows),
                min_reports=config.min_window_reports,
            ),
        )
    ]


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

    rows, single_window_rows, convergence_reports, convergence_paths = (
        _collect_window_artifact_rows(window_artifacts, path_list, config=cfg)
    )
    derived_ensemble = _derived_ensemble_row(
        convergence_reports,
        convergence_paths,
        config=cfg,
    )
    if derived_ensemble is not None:
        rows.append(derived_ensemble)

    qualifying_rows = _qualifying_window_rows(rows)
    gates = _window_evidence_gates(qualifying_rows, config=cfg)
    return {
        "passed": bool(qualifying_rows),
        "gates": gates,
        "ensemble_rows": rows,
        "single_window_rows": single_window_rows,
        "derived_ensemble": derived_ensemble,
    }


__all__ = ["_ensemble_row", "summarize_window_evidence"]
