"""Replicate and ensemble gates for nonlinear transport-window evidence."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence
import math

import numpy as np

from spectraxgk.validation.quasilinear.window_config import (
    NonlinearWindowEnsembleConfig,
    NonlinearWindowEnsembleManifestConfig,
    _validate_ensemble_config,
    _validate_ensemble_manifest_config,
)
from spectraxgk.validation.quasilinear.window_promotion import (
    _report_late_mean,
    _report_sem,
    nonlinear_window_stats_promotion_ready,
)
from spectraxgk.validation.quasilinear.window_statistics import _gate, _json_number


def _ensemble_report_rows(
    reports: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[float], list[float], bool]:
    rows: list[dict[str, Any]] = []
    means: list[float] = []
    sems: list[float] = []
    individual_ready = True
    for idx, report in enumerate(reports):
        if not isinstance(report, dict):
            raise TypeError("reports must contain nonlinear window report dictionaries")
        late_mean = _report_late_mean(report)
        sem = _report_sem(report)
        report_passed = bool(report.get("passed", False))
        ready, failures = nonlinear_window_stats_promotion_ready(report)
        if not bool(report_passed and ready):
            individual_ready = False
        if late_mean is not None:
            means.append(late_mean)
        if sem is not None:
            sems.append(sem)
        provenance = report.get("provenance")
        provenance_dict: dict[str, Any] = provenance if isinstance(provenance, dict) else {}
        rows.append(
            {
                "index": int(idx),
                "case": str(report.get("case", f"report_{idx}")),
                "passed": report_passed,
                "promotion_ready": ready,
                "late_mean": _json_number(late_mean),
                "sem": _json_number(sem),
                "failures": failures,
                "source_artifact": provenance_dict.get("source_artifact"),
                "summary_artifact": provenance_dict.get("summary_artifact"),
            }
        )
    return rows, means, sems, individual_ready


def _finite_sem_candidates(*values: float | None) -> list[float]:
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def _ensemble_statistics(
    means: Sequence[float],
    sems: Sequence[float],
    *,
    cfg: NonlinearWindowEnsembleConfig,
) -> tuple[dict[str, Any], bool, bool]:
    mean_arr = np.asarray(means, dtype=float)
    sem_arr = np.asarray(sems, dtype=float)
    scale = max(
        abs(float(np.mean(mean_arr))) if mean_arr.size else 0.0,
        float(cfg.value_floor),
    )
    mean_spread = float(np.max(mean_arr) - np.min(mean_arr)) if mean_arr.size else None
    mean_rel_spread = None if mean_spread is None else float(mean_spread / scale)
    sample_sem = (
        float(np.std(mean_arr, ddof=1) / np.sqrt(mean_arr.size))
        if mean_arr.size >= 2
        else None
    )
    max_individual_sem = float(np.max(sem_arr)) if sem_arr.size else None
    sem_candidates = _finite_sem_candidates(sample_sem, max_individual_sem)
    combined_sem = max(sem_candidates) if sem_candidates else None
    combined_sem_rel = None if combined_sem is None else float(combined_sem / scale)
    mean_rel_spread_ok = (
        mean_rel_spread is not None
        and math.isfinite(mean_rel_spread)
        and mean_rel_spread <= float(cfg.max_mean_rel_spread)
    )
    combined_sem_rel_ok = (
        combined_sem_rel is not None
        and math.isfinite(combined_sem_rel)
        and combined_sem_rel <= float(cfg.max_combined_sem_rel)
    )
    return (
        {
            "n_finite_means": int(mean_arr.size),
            "ensemble_mean": _json_number(
                float(np.mean(mean_arr)) if mean_arr.size else None
            ),
            "mean_spread": _json_number(mean_spread),
            "mean_rel_spread": _json_number(mean_rel_spread),
            "sample_sem": _json_number(sample_sem),
            "max_individual_sem": _json_number(max_individual_sem),
            "combined_sem": _json_number(combined_sem),
            "combined_sem_rel": _json_number(combined_sem_rel),
        },
        mean_rel_spread_ok,
        combined_sem_rel_ok,
    )


def _ensemble_gates(
    *,
    n_reports: int,
    n_finite_means: int,
    individual_ready: bool,
    mean_rel_spread_ok: bool,
    combined_sem_rel_ok: bool,
    statistics: dict[str, Any],
    cfg: NonlinearWindowEnsembleConfig,
) -> list[dict[str, Any]]:
    return [
        _gate(
            "report_count",
            n_reports >= int(cfg.min_reports),
            f"reports={n_reports} min_reports={cfg.min_reports}",
        ),
        _gate(
            "individual_windows_passed",
            (not cfg.require_individual_passed) or individual_ready,
            f"require_individual_passed={cfg.require_individual_passed}",
        ),
        _gate(
            "finite_late_means",
            n_finite_means == n_reports and n_reports > 0,
            f"finite_means={n_finite_means} reports={n_reports}",
        ),
        _gate(
            "mean_relative_spread",
            mean_rel_spread_ok,
            "mean_rel_spread={value} gate={gate}".format(
                value=statistics["mean_rel_spread"],
                gate=cfg.max_mean_rel_spread,
            ),
        ),
        _gate(
            "combined_sem",
            combined_sem_rel_ok,
            "combined_sem_rel={value} gate={gate}".format(
                value=statistics["combined_sem_rel"],
                gate=cfg.max_combined_sem_rel,
            ),
        ),
    ]


def _pack_ensemble_report(
    *,
    case: str,
    comparison: str,
    rows: list[dict[str, Any]],
    statistics: dict[str, Any],
    gates: list[dict[str, Any]],
    cfg: NonlinearWindowEnsembleConfig,
) -> dict[str, Any]:
    passed = all(bool(gate["passed"]) for gate in gates)
    report_statistics = {"n_reports": len(rows), **statistics}
    return {
        "kind": "nonlinear_window_ensemble_report",
        "claim_level": "replicated_nonlinear_window_uncertainty_gate_not_simulation_claim",
        "case": str(case),
        "comparison": str(comparison),
        "passed": passed,
        "statistics": report_statistics,
        "gates": gates,
        "gate_report": {
            "case": str(case),
            "source": "nonlinear_window_convergence_reports",
            "passed": passed,
            "max_abs_error": 0.0 if passed else 1.0,
            "max_rel_error": 0.0 if passed else 1.0,
            "gates": gates,
        },
        "rows": rows,
        "config": asdict(cfg),
    }


def nonlinear_window_ensemble_report(
    reports: Sequence[dict[str, Any]],
    *,
    case: str = "nonlinear_window_ensemble",
    comparison: str = "replicate_uncertainty",
    config: NonlinearWindowEnsembleConfig | None = None,
) -> dict[str, Any]:
    """Gate repeated nonlinear-window summaries for seed/timestep robustness.

    The input reports are expected to come from
    :func:`nonlinear_window_convergence_report`. This helper does not inspect
    raw time traces; it compares already-gated late-window means and their
    uncertainty metadata so production promotion can require seed, initial
    condition, or timestep robustness without rerunning simulations inside the
    checker.
    """

    cfg = config or NonlinearWindowEnsembleConfig()
    _validate_ensemble_config(cfg)
    rows, means, sems, individual_ready = _ensemble_report_rows(reports)
    statistics, mean_rel_spread_ok, combined_sem_rel_ok = _ensemble_statistics(
        means,
        sems,
        cfg=cfg,
    )
    gates = _ensemble_gates(
        n_reports=len(rows),
        n_finite_means=int(statistics["n_finite_means"]),
        individual_ready=individual_ready,
        mean_rel_spread_ok=mean_rel_spread_ok,
        combined_sem_rel_ok=combined_sem_rel_ok,
        statistics=statistics,
        cfg=cfg,
    )
    return _pack_ensemble_report(
        case=case,
        comparison=comparison,
        rows=rows,
        statistics=statistics,
        gates=gates,
        cfg=cfg,
    )


def _required_manifest_axes(
    cfg: NonlinearWindowEnsembleManifestConfig,
) -> tuple[str, ...]:
    return tuple(str(axis).strip() for axis in cfg.required_variant_axes)


def _manifest_record_row(
    idx: int,
    raw_record: dict[str, Any],
    *,
    required_axes: tuple[str, ...],
) -> tuple[str, dict[str, Any]]:
    report = raw_record.get("report")
    if not isinstance(report, dict):
        raise ValueError("each record must contain a nonlinear-window report")
    report_case = str(
        raw_record.get("case")
        or raw_record.get("ensemble_case")
        or report.get("case")
        or f"case_{idx}"
    )
    raw_variant = raw_record.get("variant")
    variant: dict[str, Any] = raw_variant if isinstance(raw_variant, dict) else {}
    ready, failures = nonlinear_window_stats_promotion_ready(report)
    provenance = report.get("provenance")
    provenance_dict: dict[str, Any] = provenance if isinstance(provenance, dict) else {}
    return report_case, {
        "index": int(idx),
        "case": report_case,
        "summary_artifact": raw_record.get("summary_artifact")
        or provenance_dict.get("summary_artifact"),
        "source_artifact": raw_record.get("source_artifact")
        or provenance_dict.get("source_artifact"),
        "convergence_report_artifact": raw_record.get(
            "convergence_report_artifact"
        ),
        "passed": bool(report.get("passed", False)),
        "promotion_ready": ready,
        "failures": failures,
        "variant": {axis: variant.get(axis) for axis in required_axes},
        "late_mean": _json_number(_report_late_mean(report)),
        "sem": _json_number(_report_sem(report)),
    }


def _manifest_rows_by_case(
    records: Sequence[dict[str, Any]],
    *,
    required_axes: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = {}
    for idx, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise TypeError("records must contain dictionaries")
        report_case, row = _manifest_record_row(
            idx,
            raw_record,
            required_axes=required_axes,
        )
        rows.append(row)
        by_case.setdefault(report_case, []).append(row)
    return rows, by_case


def _manifest_axis_status(
    *,
    report_case: str,
    axis: str,
    ready_rows: Sequence[dict[str, Any]],
    cfg: NonlinearWindowEnsembleManifestConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    values = sorted(
        {
            str(row["variant"].get(axis))
            for row in ready_rows
            if row["variant"].get(axis) not in (None, "")
        }
    )
    missing_count = max(0, int(cfg.min_replicates_per_case) - len(values))
    status = {
        "passed": missing_count == 0,
        "observed_distinct_values": values,
        "observed_distinct_count": len(values),
        "required_distinct_count": int(cfg.min_replicates_per_case),
        "missing_count": missing_count,
    }
    if not missing_count:
        return status, None
    return status, {
        "case": report_case,
        "variant_axis": axis,
        "missing_count": missing_count,
        "observed_distinct_values": values,
        "required_distinct_count": int(cfg.min_replicates_per_case),
        "artifact_hint": (
            f"add {missing_count} passed nonlinear-window convergence "
            f"report(s) for case '{report_case}' with distinct {axis} "
            "metadata and trace provenance"
        ),
        "metadata_requirements": [
            "summary JSON or convergence report with source_artifact provenance",
            f"variant.{axis} or equivalent top-level {axis} metadata",
            "passed nonlinear_window_convergence_report gates",
        ],
    }


def _manifest_case_row(
    report_case: str,
    observed: Sequence[dict[str, Any]],
    *,
    required_axes: tuple[str, ...],
    cfg: NonlinearWindowEnsembleManifestConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ready_rows = [row for row in observed if bool(row["promotion_ready"])]
    per_axis: dict[str, Any] = {}
    missing_artifacts: list[dict[str, Any]] = []
    for axis in required_axes:
        status, missing = _manifest_axis_status(
            report_case=report_case,
            axis=axis,
            ready_rows=ready_rows,
            cfg=cfg,
        )
        per_axis[axis] = status
        if missing is not None:
            missing_artifacts.append(missing)
    return (
        {
            "case": report_case,
            "n_observed_artifacts": len(observed),
            "n_promotion_ready_artifacts": len(ready_rows),
            "observed_summary_artifacts": [
                row["summary_artifact"] for row in observed if row["summary_artifact"]
            ],
            "observed_convergence_report_artifacts": [
                row["convergence_report_artifact"]
                for row in observed
                if row["convergence_report_artifact"]
            ],
            "variant_axes": per_axis,
            "ensemble_gate_runnable": all(
                bool(per_axis[axis]["passed"]) for axis in required_axes
            ),
        },
        missing_artifacts,
    )


def _manifest_case_rows(
    by_case: dict[str, list[dict[str, Any]]],
    *,
    required_axes: tuple[str, ...],
    cfg: NonlinearWindowEnsembleManifestConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    missing_artifacts: list[dict[str, Any]] = []
    for report_case in sorted(by_case):
        case_row, missing = _manifest_case_row(
            report_case,
            by_case[report_case],
            required_axes=required_axes,
            cfg=cfg,
        )
        case_rows.append(case_row)
        missing_artifacts.extend(missing)
    return case_rows, missing_artifacts


def _manifest_gates(
    *,
    rows: Sequence[dict[str, Any]],
    case_rows: Sequence[dict[str, Any]],
    missing_artifacts: Sequence[dict[str, Any]],
    cfg: NonlinearWindowEnsembleManifestConfig,
) -> list[dict[str, Any]]:
    observed_ready = all(bool(row["promotion_ready"]) for row in rows) if rows else False
    axes_passed = not missing_artifacts and bool(case_rows)
    return [
        _gate(
            "observed_window_artifacts_present",
            bool(rows),
            f"observed_artifacts={len(rows)}",
        ),
        _gate(
            "observed_windows_promotion_ready",
            (not cfg.require_observed_windows_ready) or observed_ready,
            f"require_observed_windows_ready={cfg.require_observed_windows_ready}",
        ),
        _gate(
            "seed_and_timestep_replicates_present",
            axes_passed,
            (
                f"missing_artifact_groups={len(missing_artifacts)}"
                if missing_artifacts
                else "all required variant axes have enough passed artifacts"
            ),
        ),
    ]


def _pack_ensemble_artifact_manifest(
    *,
    case: str,
    rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    missing_artifacts: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    cfg: NonlinearWindowEnsembleManifestConfig,
) -> dict[str, Any]:
    passed = all(bool(gate["passed"]) for gate in gates)
    return {
        "kind": "nonlinear_window_ensemble_readiness_manifest",
        "claim_level": (
            "replicated_seed_timestep_artifact_manifest_blocks_promotion_until_ready"
        ),
        "case": str(case),
        "passed": passed,
        "promotion_gate": {
            "passed": passed,
            "blockers": [gate["metric"] for gate in gates if not bool(gate["passed"])],
            "requirements": [
                "every observed late-window report must pass convergence metadata gates",
                "each case must include distinct passed seed-replicate artifacts",
                "each case must include distinct passed timestep-replicate artifacts",
                "only after this manifest passes should the replicated ensemble gate be run",
            ],
        },
        "gates": gates,
        "cases": case_rows,
        "observed_artifacts": rows,
        "missing_artifacts": missing_artifacts,
        "config": asdict(cfg),
    }


def nonlinear_window_ensemble_artifact_manifest(
    records: Sequence[dict[str, Any]],
    *,
    case: str = "nonlinear_window_ensemble_artifact_manifest",
    config: NonlinearWindowEnsembleManifestConfig | None = None,
) -> dict[str, Any]:
    """Return a promotion-blocking manifest for missing ensemble artifacts.

    Each record should contain a ``report`` produced by
    :func:`nonlinear_window_convergence_report`, plus optional ``variant``
    metadata such as ``{"seed": 1, "timestep": 0.02}``. The manifest is
    intentionally conservative: a production nonlinear optimization promotion
    needs distinct passed artifacts for every required variant axis, so a
    single late-window summary is recorded as useful convergence evidence but
    not as replicated-ensemble evidence.
    """

    cfg = config or NonlinearWindowEnsembleManifestConfig()
    _validate_ensemble_manifest_config(cfg)
    required_axes = _required_manifest_axes(cfg)
    rows, by_case = _manifest_rows_by_case(records, required_axes=required_axes)
    case_rows, missing_artifacts = _manifest_case_rows(
        by_case,
        required_axes=required_axes,
        cfg=cfg,
    )
    gates = _manifest_gates(
        rows=rows,
        case_rows=case_rows,
        missing_artifacts=missing_artifacts,
        cfg=cfg,
    )
    return _pack_ensemble_artifact_manifest(
        case=case,
        rows=rows,
        case_rows=case_rows,
        missing_artifacts=missing_artifacts,
        gates=gates,
        cfg=cfg,
    )


__all__ = [
    "nonlinear_window_ensemble_artifact_manifest",
    "nonlinear_window_ensemble_report",
]
