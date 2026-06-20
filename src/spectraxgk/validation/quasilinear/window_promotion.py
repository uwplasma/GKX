"""Promotion-readiness checks for nonlinear transport-window metadata."""

from __future__ import annotations

from typing import Any
import math

from spectraxgk.validation.quasilinear.window_statistics import _finite_number


def nonlinear_window_stats_promotion_ready(
    stats: object,
) -> tuple[bool, list[str]]:
    """Return whether serialized nonlinear window metadata can support promotion."""

    failures: list[str] = []
    if not isinstance(stats, dict):
        return False, ["missing nonlinear_window_stats object"]
    if stats.get("kind") == "nonlinear_window_ensemble_report":
        if not bool(stats.get("passed", False)):
            failures.append("nonlinear window ensemble report did not pass")
        gate_report = stats.get("gate_report")
        if not isinstance(gate_report, dict) or not bool(
            gate_report.get("passed", False)
        ):
            failures.append("missing passed ensemble gate_report")
        statistics = stats.get("statistics")
        if not isinstance(statistics, dict):
            failures.append("missing ensemble statistics object")
            statistics = {}
        for field in ("ensemble_mean", "combined_sem", "combined_sem_rel"):
            if not _finite_number(statistics.get(field)):
                failures.append(f"missing/non-finite statistics.{field}")
        rows = stats.get("rows")
        if not isinstance(rows, list) or not rows:
            failures.append("ensemble report has no rows")
        else:
            ready_rows = [
                row
                for row in rows
                if isinstance(row, dict) and bool(row.get("promotion_ready", False))
            ]
            if len(ready_rows) != len(rows):
                failures.append("not all ensemble rows are promotion-ready")
            if not any(
                isinstance(row, dict) and str(row.get("source_artifact", "")).strip()
                for row in rows
            ):
                failures.append("missing ensemble source_artifact provenance")
        return not failures, failures
    if stats.get("kind") != "nonlinear_window_convergence_report":
        failures.append("unexpected nonlinear_window_stats kind")
    if not bool(stats.get("passed", False)):
        failures.append("nonlinear window convergence report did not pass")
    provenance = stats.get("provenance")
    if (
        not isinstance(provenance, dict)
        or not str(provenance.get("source_artifact", "")).strip()
    ):
        failures.append("missing nonlinear source_artifact provenance")
    statistics = stats.get("statistics")
    if not isinstance(statistics, dict):
        failures.append("missing statistics object")
        statistics = {}
    for field in (
        "late_mean",
        "sem",
        "block_bootstrap_sem",
        "running_mean_rel_drift",
        "terminal_mean_rel_delta",
    ):
        if not _finite_number(statistics.get(field)):
            failures.append(f"missing/non-finite statistics.{field}")
    window = stats.get("window")
    if not isinstance(window, dict):
        failures.append("missing window object")
        window = {}
    for field in ("transient_cutoff", "late_tmin", "late_tmax"):
        if not _finite_number(window.get(field)):
            failures.append(f"missing/non-finite window.{field}")
    raw_transient_fraction = window.get("transient_fraction", 0.0)
    has_declared_cutoff = _finite_number(window.get("input_tmin")) or (
        _finite_number(raw_transient_fraction) and float(raw_transient_fraction) > 0.0
    )
    if not has_declared_cutoff:
        failures.append("missing declared transient cutoff policy")
    n_finite_late = window.get("n_finite_late", 0)
    if not _finite_number(n_finite_late) or int(float(n_finite_late)) <= 0:
        failures.append("window has no finite late samples")
    gate_report = stats.get("gate_report")
    if not isinstance(gate_report, dict) or not bool(gate_report.get("passed", False)):
        failures.append("missing passed gate_report")
    return not failures, failures


def _report_late_mean(report: dict[str, Any]) -> float | None:
    statistics = report.get("statistics")
    if not isinstance(statistics, dict):
        return None
    value = statistics.get("late_mean")
    if value is None:
        return None
    try:
        late_mean = float(value)
    except (TypeError, ValueError):
        return None
    return late_mean if math.isfinite(late_mean) else None


def _report_sem(report: dict[str, Any]) -> float | None:
    statistics = report.get("statistics")
    if not isinstance(statistics, dict):
        return None
    value = statistics.get("sem")
    if value is None:
        return None
    try:
        sem = float(value)
    except (TypeError, ValueError):
        return None
    return sem if math.isfinite(sem) else None


__all__ = ["nonlinear_window_stats_promotion_ready"]
