"""Diagnostics for replicated nonlinear transport-window spread."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any
import math
import re

from spectraxgk.diagnostics.metadata import (
    NonlinearTurbulenceGradientEvidenceConfig,
    _artifact_passed,
)
from spectraxgk.diagnostics.transport_windows import (
    NonlinearWindowEnsembleConfig,
    _gate,
    _json_number as _window_json_number,
    _report_late_mean,
    _report_sem,
    nonlinear_window_ensemble_report,
    nonlinear_window_stats_promotion_ready,
)

_STATE_RE = re.compile(r"(?:^|_)(baseline|plus_delta|minus_delta)(?:_|$)")
_SEED_RE = re.compile(r"seed(\d+)")
_DT_RE = re.compile(r"dt([0-9A-Za-zp]+)")


@dataclass(frozen=True)
class NonlinearReplicateSpreadConfig:
    """Thresholds for classifying replicated nonlinear-window spread."""

    max_mean_rel_spread: float = 0.15
    value_floor: float = 1.0e-12


@dataclass(frozen=True)
class _StateSpreadDiagnostics:
    state: str
    passed: bool
    stats: Mapping[str, Any]
    rows: list[Mapping[str, Any]]
    ensemble_mean: float | None
    scale: float
    mean_rel_spread: float | None
    spread_gate: float
    high_label: str | None
    high_axis: str | None
    low_label: str | None
    low_axis: str | None
    classification: str


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_number(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    return value if math.isfinite(float(value)) else None


def _state_label(ensemble: Mapping[str, Any], index: int) -> str:
    candidates: list[str] = []
    for key in ("state", "case", "comparison"):
        value = ensemble.get(key)
        if value is not None:
            candidates.append(str(value))
    rows = ensemble.get("rows")
    if isinstance(rows, Sequence):
        for row in rows:
            if isinstance(row, Mapping):
                for key in ("source_artifact", "summary_artifact", "case"):
                    value = row.get(key)
                    if value is not None:
                        candidates.append(str(value))
    for candidate in candidates:
        match = _STATE_RE.search(candidate)
        if match is not None:
            return match.group(1)
    return f"state_{index}"


def _variant_label(row: Mapping[str, Any]) -> tuple[str, str]:
    for key in ("variant_label", "label"):
        value = row.get(key)
        if isinstance(value, str) and value:
            axis = str(row.get("variant_axis", "") or "")
            if not axis:
                if value.startswith("seed") and "_dt" in value:
                    axis = "seed_timestep"
                elif value.startswith("seed"):
                    axis = "seed"
                elif value.startswith("dt"):
                    axis = "timestep"
                else:
                    axis = "unknown"
            return value, axis

    candidates = [
        str(value)
        for value in (
            row.get("source_artifact"),
            row.get("summary_artifact"),
            row.get("case"),
        )
        if value is not None
    ]
    for candidate in candidates:
        seed = _SEED_RE.search(candidate)
        dt = _DT_RE.search(candidate)
        if seed is not None and dt is not None:
            return f"seed{seed.group(1)}_dt{dt.group(1)}", "seed_timestep"
        if seed is not None:
            return f"seed{seed.group(1)}", "seed"
        if dt is not None:
            return f"dt{dt.group(1)}", "timestep"
    return f"replicate_{row.get('index', 'unknown')}", "unknown"


def _recommendation(classification: str) -> str:
    if classification == "passed_replicate_spread_gate":
        return "Replicate spread is within the configured gate; no extra replicas are indicated."
    if classification == "mixed_seed_timestep_spread":
        return (
            "Do not add same-bracket replicas blindly. The high and low windows are on different "
            "variant axes, so first disambiguate seed sensitivity from timestep sensitivity or shrink "
            "the finite-difference bracket."
        )
    if classification == "seed_spread_limited":
        return (
            "Seed variability dominates. Add a matched seed at the same parameter state or switch to "
            "paired-seed finite differences before using this bracket for a gradient claim."
        )
    if classification == "timestep_spread_limited":
        return (
            "Timestep sensitivity dominates. Retune the timestep/window convergence before adding "
            "more random seeds at this state."
        )
    return (
        "Replicate spread is not classifiable from the available labels. Preserve the fail-closed "
        "claim boundary and add explicit seed/timestep metadata to the next run manifest."
    )


def _classify_state(
    *,
    passed: bool,
    mean_rel_spread: float | None,
    spread_gate: float,
    high_axis: str | None,
    low_axis: str | None,
) -> str:
    if passed and mean_rel_spread is not None and mean_rel_spread <= spread_gate:
        return "passed_replicate_spread_gate"
    if mean_rel_spread is not None and mean_rel_spread <= spread_gate:
        return "passed_replicate_spread_gate"
    if high_axis is not None and low_axis is not None and high_axis != low_axis:
        return "mixed_seed_timestep_spread"
    if high_axis == "seed" and low_axis == "seed":
        return "seed_spread_limited"
    if high_axis == "timestep" and low_axis == "timestep":
        return "timestep_spread_limited"
    return "spread_limited_unknown_axis"


def _validated_config(
    config: NonlinearReplicateSpreadConfig | None,
) -> NonlinearReplicateSpreadConfig:
    cfg = config or NonlinearReplicateSpreadConfig()
    if cfg.max_mean_rel_spread < 0.0:
        raise ValueError("max_mean_rel_spread must be non-negative")
    if cfg.value_floor <= 0.0:
        raise ValueError("value_floor must be positive")
    return cfg


def _ensemble_rows(ensemble: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows_obj = ensemble.get("rows")
    if not isinstance(rows_obj, Sequence):
        return []
    return [row for row in rows_obj if isinstance(row, Mapping)]


def _finite_late_means(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    return [
        mean
        for row in rows
        if (mean := _finite_float(row.get("late_mean"))) is not None
    ]


def _ensemble_mean_from_stats(
    stats: Mapping[str, Any],
    *,
    finite_means: Sequence[float],
) -> float | None:
    ensemble_mean = _finite_float(stats.get("ensemble_mean"))
    if ensemble_mean is None and finite_means:
        ensemble_mean = sum(finite_means) / len(finite_means)
    return ensemble_mean


def _row_mean_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], float]]:
    pairs: list[tuple[Mapping[str, Any], float]] = []
    for row in rows:
        row_mean = _finite_float(row.get("late_mean"))
        if row_mean is not None:
            pairs.append((row, row_mean))
    return pairs


def _high_low_variant_labels(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str | None, str | None, str | None]:
    pairs = _row_mean_pairs(rows)
    if not pairs:
        return None, None, None, None
    high_row = max(pairs, key=lambda item: item[1])[0]
    low_row = min(pairs, key=lambda item: item[1])[0]
    high_label, high_axis = _variant_label(high_row)
    low_label, low_axis = _variant_label(low_row)
    return high_label, high_axis, low_label, low_axis


def _spread_gate(
    ensemble: Mapping[str, Any],
    stats: Mapping[str, Any],
    *,
    config: NonlinearReplicateSpreadConfig,
) -> float:
    spread_gate = _finite_float(stats.get("max_mean_rel_spread"))
    if spread_gate is None:
        raw_config = ensemble.get("config")
        if isinstance(raw_config, Mapping):
            spread_gate = _finite_float(raw_config.get("max_mean_rel_spread"))
    return float(config.max_mean_rel_spread if spread_gate is None else spread_gate)


def _mean_rel_spread(
    stats: Mapping[str, Any],
    *,
    finite_means: Sequence[float],
    scale: float,
) -> float | None:
    mean_rel_spread = _finite_float(stats.get("mean_rel_spread"))
    if mean_rel_spread is None and finite_means:
        mean_rel_spread = (max(finite_means) - min(finite_means)) / scale
    return mean_rel_spread


def _state_diagnostics(
    ensemble: Mapping[str, Any],
    *,
    state_index: int,
    config: NonlinearReplicateSpreadConfig,
) -> _StateSpreadDiagnostics:
    state = _state_label(ensemble, state_index)
    statistics = ensemble.get("statistics")
    stats = statistics if isinstance(statistics, Mapping) else {}
    rows = _ensemble_rows(ensemble)
    finite_means = _finite_late_means(rows)
    ensemble_mean = _ensemble_mean_from_stats(stats, finite_means=finite_means)
    scale = max(abs(float(ensemble_mean or 0.0)), float(config.value_floor))
    high_label, high_axis, low_label, low_axis = _high_low_variant_labels(rows)
    mean_rel_spread = _mean_rel_spread(stats, finite_means=finite_means, scale=scale)
    spread_gate = _spread_gate(ensemble, stats, config=config)
    passed = bool(ensemble.get("passed", False))
    classification = _classify_state(
        passed=passed,
        mean_rel_spread=mean_rel_spread,
        spread_gate=spread_gate,
        high_axis=high_axis,
        low_axis=low_axis,
    )
    return _StateSpreadDiagnostics(
        state=state,
        passed=passed,
        stats=stats,
        rows=rows,
        ensemble_mean=ensemble_mean,
        scale=scale,
        mean_rel_spread=mean_rel_spread,
        spread_gate=spread_gate,
        high_label=high_label,
        high_axis=high_axis,
        low_label=low_label,
        low_axis=low_axis,
        classification=classification,
    )


def _state_row(diagnostics: _StateSpreadDiagnostics) -> dict[str, Any]:
    return {
        "state": diagnostics.state,
        "passed": diagnostics.passed,
        "classification": diagnostics.classification,
        "recommendation": _recommendation(diagnostics.classification),
        "ensemble_mean": _json_number(diagnostics.ensemble_mean),
        "mean_rel_spread": _json_number(diagnostics.mean_rel_spread),
        "mean_rel_spread_gate": _json_number(diagnostics.spread_gate),
        "combined_sem_rel": _json_number(
            _finite_float(diagnostics.stats.get("combined_sem_rel"))
        ),
        "high_variant_label": diagnostics.high_label,
        "high_variant_axis": diagnostics.high_axis,
        "low_variant_label": diagnostics.low_label,
        "low_variant_axis": diagnostics.low_axis,
    }


def _replicate_row(
    row: Mapping[str, Any],
    *,
    diagnostics: _StateSpreadDiagnostics,
    fallback_index: int,
) -> dict[str, Any]:
    mean = _finite_float(row.get("late_mean"))
    sem = _finite_float(row.get("sem"))
    label, axis = _variant_label(row)
    rel_delta = None
    if mean is not None and diagnostics.ensemble_mean is not None:
        rel_delta = (mean - diagnostics.ensemble_mean) / diagnostics.scale
    window_stats = row.get("window_statistics")
    window = window_stats if isinstance(window_stats, Mapping) else {}
    return {
        "state": diagnostics.state,
        "index": int(row.get("index", fallback_index)),
        "variant_label": label,
        "variant_axis": axis,
        "late_mean": _json_number(mean),
        "sem": _json_number(sem),
        "ensemble_mean": _json_number(diagnostics.ensemble_mean),
        "relative_delta": _json_number(rel_delta),
        "passed": bool(row.get("passed", False)),
        "promotion_ready": bool(row.get("promotion_ready", False)),
        "source_artifact": row.get("source_artifact"),
        "summary_artifact": row.get("summary_artifact"),
        "running_mean_rel_drift": _json_number(
            _finite_float(window.get("running_mean_rel_drift"))
        ),
        "terminal_mean_rel_delta": _json_number(
            _finite_float(window.get("terminal_mean_rel_delta"))
        ),
        "sem_rel": _json_number(_finite_float(window.get("sem_rel"))),
        "n_blocks": _json_number(_finite_float(window.get("n_blocks"))),
    }


def _replicate_rows(
    diagnostics: _StateSpreadDiagnostics,
    *,
    start_index: int,
) -> list[dict[str, Any]]:
    return [
        _replicate_row(row, diagnostics=diagnostics, fallback_index=start_index + i)
        for i, row in enumerate(diagnostics.rows)
    ]


def _summary(
    *,
    ensembles: Sequence[Mapping[str, Any]],
    replicate_rows: Sequence[Mapping[str, Any]],
    failed_states: Sequence[str],
    classifications: Mapping[str, int],
) -> dict[str, Any]:
    passed = not failed_states
    return {
        "n_states": len(ensembles),
        "n_replicates": len(replicate_rows),
        "failed_states": list(failed_states),
        "classifications": dict(classifications),
        "recommendation": (
            "All replicated ensembles are within spread gates."
            if passed
            else "Keep the nonlinear-gradient claim fail-closed and target the failed states first."
        ),
    }


def nonlinear_replicate_spread_report(
    ensembles: Sequence[Mapping[str, Any]],
    *,
    case: str = "nonlinear_replicate_spread_diagnostic",
    config: NonlinearReplicateSpreadConfig | None = None,
) -> dict[str, Any]:
    """Classify which replicate/state drives nonlinear-window ensemble spread.

    Parameters
    ----------
    ensembles:
        Sequence of ensemble JSON payloads, typically produced by
        ``tools/release/check_nonlinear_window_ensemble.py``.
    case:
        Human-readable label for the diagnostic artifact.
    config:
        Spread threshold and numerical floor used for relative deviations.
    """

    cfg = _validated_config(config)
    state_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    failed_states: list[str] = []
    classifications: dict[str, int] = {}

    for state_index, ensemble in enumerate(ensembles):
        diagnostics = _state_diagnostics(ensemble, state_index=state_index, config=cfg)
        classification = diagnostics.classification
        classifications[classification] = classifications.get(classification, 0) + 1
        if classification != "passed_replicate_spread_gate":
            failed_states.append(diagnostics.state)
        state_rows.append(_state_row(diagnostics))
        replicate_rows.extend(
            _replicate_rows(diagnostics, start_index=len(replicate_rows))
        )

    passed = not failed_states
    return {
        "kind": "nonlinear_replicate_spread_diagnostic",
        "claim_level": "replicate_spread_diagnostic_not_simulation_claim",
        "case": str(case),
        "passed": passed,
        "summary": _summary(
            ensembles=ensembles,
            replicate_rows=replicate_rows,
            failed_states=failed_states,
            classifications=classifications,
        ),
        "state_rows": state_rows,
        "replicate_rows": replicate_rows,
        "config": {
            "max_mean_rel_spread": float(cfg.max_mean_rel_spread),
            "value_floor": float(cfg.value_floor),
        },
    }


@dataclass(frozen=True)
class NonlinearWindowEnsembleManifestConfig:
    """Artifact requirements before a replicated nonlinear ensemble can run."""

    min_replicates_per_case: int = 2
    required_variant_axes: tuple[str, ...] = ("seed", "timestep")
    require_observed_windows_ready: bool = True


def _validate_ensemble_manifest_config(
    config: NonlinearWindowEnsembleManifestConfig,
) -> None:
    if int(config.min_replicates_per_case) < 2:
        raise ValueError("min_replicates_per_case must be at least 2")
    axes = tuple(str(axis).strip() for axis in config.required_variant_axes)
    if not axes or any(not axis for axis in axes):
        raise ValueError("required_variant_axes must contain non-empty names")


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
        "convergence_report_artifact": raw_record.get("convergence_report_artifact"),
        "passed": bool(report.get("passed", False)),
        "promotion_ready": ready,
        "failures": failures,
        "variant": {axis: variant.get(axis) for axis in required_axes},
        "late_mean": _window_json_number(_report_late_mean(report)),
        "sem": _window_json_number(_report_sem(report)),
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
    observed_ready = (
        all(bool(row["promotion_ready"]) for row in rows) if rows else False
    )
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


# ---- replicated window summaries ----
"""Replicated nonlinear-window evidence summaries for turbulence-gradient gates."""


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
            single_window_rows.append(_single_window_row(payload, path=path, kind=kind))
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


__all__ = [
    "NonlinearReplicateSpreadConfig",
    "NonlinearWindowEnsembleManifestConfig",
    "nonlinear_replicate_spread_report",
    "summarize_window_evidence",
    "nonlinear_window_ensemble_artifact_manifest",
]
