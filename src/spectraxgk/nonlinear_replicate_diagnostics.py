"""Diagnostics for replicated nonlinear transport-window spread.

The routines here are intentionally data-only. They consume already-generated
ensemble JSON artifacts and classify why a replicated nonlinear window passed
or failed before more GPU time is spent on follow-up runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math
import re


_STATE_RE = re.compile(r"(?:^|_)(baseline|plus_delta|minus_delta)(?:_|$)")
_SEED_RE = re.compile(r"seed(\d+)")
_DT_RE = re.compile(r"dt([0-9A-Za-zp]+)")


@dataclass(frozen=True)
class NonlinearReplicateSpreadConfig:
    """Thresholds for classifying replicated nonlinear-window spread."""

    max_mean_rel_spread: float = 0.15
    value_floor: float = 1.0e-12


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
                if value.startswith("seed"):
                    axis = "seed"
                elif value.startswith("dt"):
                    axis = "timestep"
                else:
                    axis = "unknown"
            return value, axis

    candidates = [
        str(value)
        for value in (row.get("source_artifact"), row.get("summary_artifact"), row.get("case"))
        if value is not None
    ]
    for candidate in candidates:
        seed = _SEED_RE.search(candidate)
        if seed is not None:
            return f"seed{seed.group(1)}", "seed"
        dt = _DT_RE.search(candidate)
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
        ``tools/check_nonlinear_window_ensemble.py``.
    case:
        Human-readable label for the diagnostic artifact.
    config:
        Spread threshold and numerical floor used for relative deviations.
    """

    cfg = config or NonlinearReplicateSpreadConfig()
    if cfg.max_mean_rel_spread < 0.0:
        raise ValueError("max_mean_rel_spread must be non-negative")
    if cfg.value_floor <= 0.0:
        raise ValueError("value_floor must be positive")

    state_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    failed_states: list[str] = []
    classifications: dict[str, int] = {}

    for state_index, ensemble in enumerate(ensembles):
        state = _state_label(ensemble, state_index)
        statistics = ensemble.get("statistics")
        stats = statistics if isinstance(statistics, Mapping) else {}
        rows_obj = ensemble.get("rows")
        rows = [row for row in rows_obj if isinstance(row, Mapping)] if isinstance(rows_obj, Sequence) else []
        means = [_finite_float(row.get("late_mean")) for row in rows]
        finite_means = [mean for mean in means if mean is not None]
        ensemble_mean = _finite_float(stats.get("ensemble_mean"))
        if ensemble_mean is None and finite_means:
            ensemble_mean = sum(finite_means) / len(finite_means)
        scale = max(abs(float(ensemble_mean or 0.0)), float(cfg.value_floor))

        high_row: Mapping[str, Any] | None = None
        low_row: Mapping[str, Any] | None = None
        if rows and finite_means:
            row_mean_pairs: list[tuple[Mapping[str, Any], float]] = []
            for row in rows:
                row_mean = _finite_float(row.get("late_mean"))
                if row_mean is not None:
                    row_mean_pairs.append((row, row_mean))
            high_row = max(row_mean_pairs, key=lambda item: item[1])[0]
            low_row = min(row_mean_pairs, key=lambda item: item[1])[0]

        high_label, high_axis = _variant_label(high_row) if high_row is not None else (None, None)
        low_label, low_axis = _variant_label(low_row) if low_row is not None else (None, None)
        mean_rel_spread = _finite_float(stats.get("mean_rel_spread"))
        if mean_rel_spread is None and finite_means:
            mean_rel_spread = (max(finite_means) - min(finite_means)) / scale
        spread_gate = _finite_float(stats.get("max_mean_rel_spread"))
        if spread_gate is None:
            raw_config = ensemble.get("config")
            if isinstance(raw_config, Mapping):
                spread_gate = _finite_float(raw_config.get("max_mean_rel_spread"))
        if spread_gate is None:
            spread_gate = float(cfg.max_mean_rel_spread)
        passed = bool(ensemble.get("passed", False))
        classification = _classify_state(
            passed=passed,
            mean_rel_spread=mean_rel_spread,
            spread_gate=spread_gate,
            high_axis=high_axis,
            low_axis=low_axis,
        )
        classifications[classification] = classifications.get(classification, 0) + 1
        if classification != "passed_replicate_spread_gate":
            failed_states.append(state)

        state_rows.append(
            {
                "state": state,
                "passed": passed,
                "classification": classification,
                "recommendation": _recommendation(classification),
                "ensemble_mean": _json_number(ensemble_mean),
                "mean_rel_spread": _json_number(mean_rel_spread),
                "mean_rel_spread_gate": _json_number(spread_gate),
                "combined_sem_rel": _json_number(_finite_float(stats.get("combined_sem_rel"))),
                "high_variant_label": high_label,
                "high_variant_axis": high_axis,
                "low_variant_label": low_label,
                "low_variant_axis": low_axis,
            }
        )

        for row in rows:
            mean = _finite_float(row.get("late_mean"))
            sem = _finite_float(row.get("sem"))
            label, axis = _variant_label(row)
            rel_delta = None if mean is None or ensemble_mean is None else (mean - ensemble_mean) / scale
            window_stats = row.get("window_statistics")
            window = window_stats if isinstance(window_stats, Mapping) else {}
            replicate_rows.append(
                {
                    "state": state,
                    "index": int(row.get("index", len(replicate_rows))),
                    "variant_label": label,
                    "variant_axis": axis,
                    "late_mean": _json_number(mean),
                    "sem": _json_number(sem),
                    "ensemble_mean": _json_number(ensemble_mean),
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
            )

    passed = not failed_states
    return {
        "kind": "nonlinear_replicate_spread_diagnostic",
        "claim_level": "replicate_spread_diagnostic_not_simulation_claim",
        "case": str(case),
        "passed": passed,
        "summary": {
            "n_states": len(ensembles),
            "n_replicates": len(replicate_rows),
            "failed_states": failed_states,
            "classifications": classifications,
            "recommendation": (
                "All replicated ensembles are within spread gates."
                if passed
                else "Keep the nonlinear-gradient claim fail-closed and target the failed states first."
            ),
        },
        "state_rows": state_rows,
        "replicate_rows": replicate_rows,
        "config": {
            "max_mean_rel_spread": float(cfg.max_mean_rel_spread),
            "value_floor": float(cfg.value_floor),
        },
    }


__all__ = [
    "NonlinearReplicateSpreadConfig",
    "nonlinear_replicate_spread_report",
]
