"""Nonlinear late-window convergence statistics for quasilinear calibration.

The helpers in this module are intentionally data-only: they operate on time
traces or compact diagnostics artifacts and do not launch nonlinear solves.
They provide the metadata required before a nonlinear holdout can support an
absolute-flux quasilinear promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
import json
import math

import numpy as np


@dataclass(frozen=True)
class NonlinearWindowConvergenceConfig:
    """Gate settings for a nonlinear post-transient transport window."""

    tmin: float | None = None
    tmax: float | None = None
    transient_fraction: float = 0.5
    min_samples: int = 24
    min_blocks: int = 4
    block_size: int | None = None
    bootstrap_samples: int = 256
    bootstrap_seed: int = 0
    max_running_mean_rel_drift: float = 0.15
    terminal_fraction: float = 0.25
    min_terminal_samples: int = 8
    max_terminal_mean_rel_delta: float = 0.10
    max_sem_rel: float = 0.25
    value_floor: float = 1.0e-12
    require_all_finite: bool = True


@dataclass(frozen=True)
class NonlinearWindowEnsembleConfig:
    """Gate settings for replicated nonlinear transport-window summaries."""

    min_reports: int = 2
    max_mean_rel_spread: float = 0.15
    max_combined_sem_rel: float = 0.25
    value_floor: float = 1.0e-12
    require_individual_passed: bool = True


@dataclass(frozen=True)
class NonlinearWindowEnsembleManifestConfig:
    """Artifact requirements before a replicated nonlinear ensemble can run."""

    min_replicates_per_case: int = 2
    required_variant_axes: tuple[str, ...] = ("seed", "timestep")
    require_observed_windows_ready: bool = True


def _json_number(value: float | int | np.generic | None) -> float | int | None:
    if value is None:
        return None
    scalar = value.item() if isinstance(value, np.generic) else value
    if isinstance(scalar, (float, int)) and math.isfinite(float(scalar)):
        return scalar
    return None


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _gate(metric: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"metric": metric, "passed": bool(passed), "detail": detail}


def _validate_config(config: NonlinearWindowConvergenceConfig) -> None:
    if config.tmin is not None and not math.isfinite(float(config.tmin)):
        raise ValueError("tmin must be finite when supplied")
    if config.tmax is not None and not math.isfinite(float(config.tmax)):
        raise ValueError("tmax must be finite when supplied")
    if config.tmin is not None and config.tmax is not None:
        if float(config.tmin) >= float(config.tmax):
            raise ValueError("tmin must be less than tmax")
    if not 0.0 <= float(config.transient_fraction) < 1.0:
        raise ValueError("transient_fraction must be in [0, 1)")
    if int(config.min_samples) < 2:
        raise ValueError("min_samples must be at least 2")
    if int(config.min_blocks) < 2:
        raise ValueError("min_blocks must be at least 2")
    if config.block_size is not None and int(config.block_size) < 1:
        raise ValueError("block_size must be positive when supplied")
    if int(config.bootstrap_samples) < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    if float(config.max_running_mean_rel_drift) < 0.0:
        raise ValueError("max_running_mean_rel_drift must be non-negative")
    if not 0.0 < float(config.terminal_fraction) <= 1.0:
        raise ValueError("terminal_fraction must be in (0, 1]")
    if int(config.min_terminal_samples) < 1:
        raise ValueError("min_terminal_samples must be positive")
    if float(config.max_terminal_mean_rel_delta) < 0.0:
        raise ValueError("max_terminal_mean_rel_delta must be non-negative")
    if float(config.max_sem_rel) < 0.0:
        raise ValueError("max_sem_rel must be non-negative")
    if float(config.value_floor) <= 0.0:
        raise ValueError("value_floor must be positive")


def _validate_ensemble_config(config: NonlinearWindowEnsembleConfig) -> None:
    if int(config.min_reports) < 2:
        raise ValueError("min_reports must be at least 2")
    if float(config.max_mean_rel_spread) < 0.0:
        raise ValueError("max_mean_rel_spread must be non-negative")
    if float(config.max_combined_sem_rel) < 0.0:
        raise ValueError("max_combined_sem_rel must be non-negative")
    if float(config.value_floor) <= 0.0:
        raise ValueError("value_floor must be positive")


def _validate_ensemble_manifest_config(
    config: NonlinearWindowEnsembleManifestConfig,
) -> None:
    if int(config.min_replicates_per_case) < 2:
        raise ValueError("min_replicates_per_case must be at least 2")
    axes = tuple(str(axis).strip() for axis in config.required_variant_axes)
    if not axes or any(not axis for axis in axes):
        raise ValueError("required_variant_axes must contain non-empty names")


def _late_window_bounds(
    t_sorted: np.ndarray, config: NonlinearWindowConvergenceConfig
) -> tuple[float, float, float]:
    raw_tmin = float(np.min(t_sorted))
    raw_tmax = float(np.max(t_sorted))
    selected_tmin = raw_tmin if config.tmin is None else float(config.tmin)
    selected_tmax = raw_tmax if config.tmax is None else float(config.tmax)
    if selected_tmin >= selected_tmax:
        raise ValueError("selected nonlinear window is empty")
    if config.tmin is None:
        cutoff = selected_tmin + float(config.transient_fraction) * (
            selected_tmax - selected_tmin
        )
    else:
        cutoff = selected_tmin
    return cutoff, selected_tmin, selected_tmax


def _contiguous_block_means(values: np.ndarray, block_size: int) -> np.ndarray:
    n_blocks = int(values.size // block_size)
    if n_blocks <= 0:
        return np.asarray([], dtype=float)
    trimmed = values[: n_blocks * block_size]
    return np.mean(trimmed.reshape(n_blocks, block_size), axis=1)


def _default_block_size(n_samples: int, min_blocks: int) -> int:
    target_blocks = max(int(min_blocks), int(np.sqrt(max(n_samples, 1))))
    return max(1, int(n_samples) // target_blocks)


def _bootstrap_sem(block_means: np.ndarray, *, samples: int, seed: int) -> float | None:
    if block_means.size < 2 or int(samples) <= 0:
        return None
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, block_means.size, size=(int(samples), block_means.size))
    boot_means = np.mean(block_means[draws], axis=1)
    return float(np.std(boot_means, ddof=1))


def nonlinear_window_convergence_report(
    time: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    case: str = "nonlinear_window",
    observable: str = "heat_flux",
    source_artifact: str | None = None,
    summary_artifact: str | None = None,
    config: NonlinearWindowConvergenceConfig | None = None,
) -> dict[str, Any]:
    """Return finite late-window statistics and convergence gates.

    The running-mean drift compares the mean of the first and second halves of
    the late window, normalized by the late-window mean scale. The uncertainty
    gate uses the maximum of the sample SEM, contiguous-block SEM, and
    block-bootstrap SEM when available.
    """

    cfg = config or NonlinearWindowConvergenceConfig()
    _validate_config(cfg)
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if t.size != y.size:
        raise ValueError("time and values must have the same length")
    if t.size == 0:
        raise ValueError("time and values must not be empty")
    finite_time = np.isfinite(t)
    if not np.all(finite_time):
        raise ValueError("time contains non-finite samples")
    order = np.argsort(t)
    t_sorted = t[order]
    y_sorted = y[order]
    cutoff, selected_tmin, selected_tmax = _late_window_bounds(t_sorted, cfg)
    late_mask = (t_sorted >= cutoff) & (t_sorted <= selected_tmax)
    late_t = t_sorted[late_mask]
    late_y_raw = y_sorted[late_mask]
    finite_y = np.isfinite(late_y_raw)
    finite_late_y = late_y_raw[finite_y]
    finite_late_t = late_t[finite_y]
    n_late = int(late_y_raw.size)
    n_finite_late = int(finite_late_y.size)
    n_nonfinite_late = int(n_late - n_finite_late)

    stats: dict[str, Any] = {
        "late_mean": None,
        "late_std": None,
        "sample_sem": None,
        "block_sem": None,
        "block_bootstrap_sem": None,
        "sem": None,
        "sem_rel": None,
        "running_mean_drift": None,
        "running_mean_rel_drift": None,
        "first_half_mean": None,
        "second_half_mean": None,
        "terminal_mean": None,
        "terminal_mean_delta": None,
        "terminal_mean_rel_delta": None,
        "terminal_tmin": None,
        "terminal_tmax": None,
        "terminal_n_samples": 0,
        "block_size": None,
        "n_blocks": 0,
        "bootstrap_samples": int(cfg.bootstrap_samples),
        "bootstrap_seed": int(cfg.bootstrap_seed),
    }
    gates: list[dict[str, Any]] = []
    finite_gate = n_late > 0 and n_finite_late > 0
    if bool(cfg.require_all_finite):
        finite_gate = finite_gate and n_nonfinite_late == 0
    gates.append(
        _gate(
            "finite_late_window",
            finite_gate,
            f"finite={n_finite_late} nonfinite={n_nonfinite_late} late={n_late}",
        )
    )
    gates.append(
        _gate(
            "finite_sample_count",
            n_finite_late >= int(cfg.min_samples),
            f"finite_late_samples={n_finite_late} min_samples={cfg.min_samples}",
        )
    )

    if n_finite_late >= 2:
        late_mean = float(np.mean(finite_late_y))
        late_std = float(np.std(finite_late_y, ddof=0))
        sample_sem = float(np.std(finite_late_y, ddof=1) / np.sqrt(n_finite_late))
        scale = max(abs(late_mean), float(cfg.value_floor))
        midpoint = n_finite_late // 2
        first_half = finite_late_y[:midpoint]
        second_half = finite_late_y[midpoint:]
        if first_half.size > 0 and second_half.size > 0:
            first_half_mean = float(np.mean(first_half))
            second_half_mean = float(np.mean(second_half))
            drift = abs(second_half_mean - first_half_mean)
            rel_drift = drift / scale
        else:
            first_half_mean = None
            second_half_mean = None
            drift = None
            rel_drift = None
        terminal_start = min(
            n_finite_late - 1,
            max(0, int(math.floor((1.0 - float(cfg.terminal_fraction)) * n_finite_late))),
        )
        terminal_y = finite_late_y[terminal_start:]
        terminal_t = finite_late_t[terminal_start:]
        terminal_mean = float(np.mean(terminal_y))
        terminal_delta = abs(terminal_mean - late_mean)
        terminal_rel_delta = terminal_delta / scale

        block_size = (
            int(cfg.block_size)
            if cfg.block_size is not None
            else _default_block_size(n_finite_late, int(cfg.min_blocks))
        )
        block_means = _contiguous_block_means(finite_late_y, block_size)
        n_blocks = int(block_means.size)
        block_sem = (
            float(np.std(block_means, ddof=1) / np.sqrt(n_blocks))
            if n_blocks >= 2
            else None
        )
        boot_sem = _bootstrap_sem(
            block_means,
            samples=int(cfg.bootstrap_samples),
            seed=int(cfg.bootstrap_seed),
        )
        sem_candidates = [
            value
            for value in (sample_sem, block_sem, boot_sem)
            if value is not None and math.isfinite(float(value))
        ]
        sem = max(sem_candidates) if sem_candidates else None
        sem_rel = None if sem is None else float(sem / scale)
        stats.update(
            {
                "late_mean": late_mean,
                "late_std": late_std,
                "sample_sem": sample_sem,
                "block_sem": block_sem,
                "block_bootstrap_sem": boot_sem,
                "sem": sem,
                "sem_rel": sem_rel,
                "running_mean_drift": drift,
                "running_mean_rel_drift": rel_drift,
                "first_half_mean": first_half_mean,
                "second_half_mean": second_half_mean,
                "terminal_mean": terminal_mean,
                "terminal_mean_delta": terminal_delta,
                "terminal_mean_rel_delta": terminal_rel_delta,
                "terminal_tmin": float(terminal_t[0]),
                "terminal_tmax": float(terminal_t[-1]),
                "terminal_n_samples": int(terminal_y.size),
                "block_size": int(block_size),
                "n_blocks": n_blocks,
            }
        )

    gates.append(
        _gate(
            "minimum_block_count",
            int(stats["n_blocks"]) >= int(cfg.min_blocks),
            f"n_blocks={stats['n_blocks']} min_blocks={cfg.min_blocks}",
        )
    )
    rel_drift = stats["running_mean_rel_drift"]
    gates.append(
        _gate(
            "running_mean_drift",
            _finite_number(rel_drift)
            and float(rel_drift) <= float(cfg.max_running_mean_rel_drift),
            "running_mean_rel_drift={value} gate={gate}".format(
                value=rel_drift,
                gate=cfg.max_running_mean_rel_drift,
            ),
        )
    )
    gates.append(
        _gate(
            "terminal_sample_count",
            int(stats["terminal_n_samples"]) >= int(cfg.min_terminal_samples),
            "terminal_samples={value} min_terminal_samples={gate}".format(
                value=stats["terminal_n_samples"],
                gate=cfg.min_terminal_samples,
            ),
        )
    )
    terminal_rel_delta = stats["terminal_mean_rel_delta"]
    gates.append(
        _gate(
            "terminal_mean_agreement",
            _finite_number(terminal_rel_delta)
            and float(terminal_rel_delta) <= float(cfg.max_terminal_mean_rel_delta),
            "terminal_mean_rel_delta={value} gate={gate} terminal_fraction={fraction}".format(
                value=terminal_rel_delta,
                gate=cfg.max_terminal_mean_rel_delta,
                fraction=cfg.terminal_fraction,
            ),
        )
    )
    sem_rel = stats["sem_rel"]
    gates.append(
        _gate(
            "block_bootstrap_sem",
            _finite_number(stats["block_bootstrap_sem"])
            and _finite_number(sem_rel)
            and float(sem_rel) <= float(cfg.max_sem_rel),
            f"sem_rel={sem_rel} gate={cfg.max_sem_rel}",
        )
    )

    passed = all(bool(gate["passed"]) for gate in gates)
    window = {
        "input_tmin": _json_number(cfg.tmin),
        "input_tmax": _json_number(cfg.tmax),
        "selected_tmin": selected_tmin,
        "selected_tmax": selected_tmax,
        "transient_fraction": float(cfg.transient_fraction),
        "transient_cutoff": float(cutoff),
        "late_tmin": float(finite_late_t[0]) if finite_late_t.size else None,
        "late_tmax": float(finite_late_t[-1]) if finite_late_t.size else None,
        "n_total": int(t_sorted.size),
        "n_late": n_late,
        "n_finite_late": n_finite_late,
        "n_nonfinite_late": n_nonfinite_late,
    }
    return {
        "kind": "nonlinear_window_convergence_report",
        "claim_level": "nonlinear_holdout_window_metadata_not_simulation_claim",
        "case": str(case),
        "observable": str(observable),
        "passed": passed,
        "window": window,
        "statistics": {key: _json_number(value) for key, value in stats.items()},
        "gates": gates,
        "gate_report": {
            "case": str(case),
            "source": source_artifact or "in_memory_trace",
            "passed": passed,
            "max_abs_error": 0.0 if passed else 1.0,
            "max_rel_error": 0.0 if passed else 1.0,
            "gates": gates,
        },
        "provenance": {
            "source_artifact": source_artifact,
            "summary_artifact": summary_artifact,
            "time_column": "t",
            "observable_column": str(observable),
        },
        "config": asdict(cfg),
    }


def _resolve_summary_artifact(summary_path: Path, source: object) -> Path:
    diag_path = Path(str(source))
    if diag_path.is_absolute():
        return diag_path
    candidates = (
        (summary_path.parent / diag_path).resolve(),
        (summary_path.parent.parent / diag_path).resolve(),
        (Path.cwd() / diag_path).resolve(),
    )
    return next(
        (candidate for candidate in candidates if candidate.exists()), candidates[0]
    )


def nonlinear_window_convergence_from_csv(
    csv_path: str | Path,
    *,
    time_column: str = "t",
    value_column: str = "heat_flux",
    case: str | None = None,
    config: NonlinearWindowConvergenceConfig | None = None,
    summary_artifact: str | None = None,
) -> dict[str, Any]:
    """Build a convergence report from a diagnostics CSV."""

    path = Path(csv_path)
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    names = set(data.dtype.names or ())
    if time_column not in names:
        raise ValueError(f"{path} is missing time column '{time_column}'")
    if value_column not in names:
        raise ValueError(f"{path} is missing observable column '{value_column}'")
    return nonlinear_window_convergence_report(
        np.asarray(data[time_column], dtype=float),
        np.asarray(data[value_column], dtype=float),
        case=str(case or path.stem),
        observable=str(value_column),
        source_artifact=str(path),
        summary_artifact=summary_artifact,
        config=config,
    )


def nonlinear_window_convergence_from_summary(
    summary_json: str | Path,
    *,
    diagnostics_source: str = "spectrax",
    time_column: str = "t",
    value_column: str = "heat_flux",
    case: str | None = None,
    config: NonlinearWindowConvergenceConfig | None = None,
) -> dict[str, Any]:
    """Build a convergence report from a window summary and diagnostics CSV."""

    summary_path = Path(summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source = summary.get(diagnostics_source)
    if source is None:
        raise ValueError(
            f"summary does not contain diagnostics source '{diagnostics_source}'"
        )
    cfg = config or NonlinearWindowConvergenceConfig(
        tmin=summary.get("tmin"),
        tmax=summary.get("tmax"),
    )
    diag_path = _resolve_summary_artifact(summary_path, source)
    if diag_path.suffix.lower() != ".csv":
        raise NotImplementedError(
            "nonlinear window convergence currently reads diagnostics CSV files"
        )
    return nonlinear_window_convergence_from_csv(
        diag_path,
        time_column=time_column,
        value_column=value_column,
        case=str(case or summary.get("case", summary_path.stem)),
        config=cfg,
        summary_artifact=str(summary_path),
    )


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
        if not isinstance(gate_report, dict) or not bool(gate_report.get("passed", False)):
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
            ready_rows = [row for row in rows if isinstance(row, dict) and bool(row.get("promotion_ready", False))]
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
        row_passed = bool(report_passed and ready)
        if not row_passed:
            individual_ready = False
        if late_mean is not None:
            means.append(late_mean)
        if sem is not None:
            sems.append(sem)
        raw_provenance = report.get("provenance")
        provenance: dict[str, Any] = raw_provenance if isinstance(raw_provenance, dict) else {}
        rows.append(
            {
                "index": int(idx),
                "case": str(report.get("case", f"report_{idx}")),
                "passed": report_passed,
                "promotion_ready": ready,
                "late_mean": _json_number(late_mean),
                "sem": _json_number(sem),
                "failures": failures,
                "source_artifact": provenance.get("source_artifact"),
                "summary_artifact": provenance.get("summary_artifact"),
            }
        )

    mean_arr = np.asarray(means, dtype=float)
    sem_arr = np.asarray(sems, dtype=float)
    n_reports = len(rows)
    scale = max(abs(float(np.mean(mean_arr))) if mean_arr.size else 0.0, float(cfg.value_floor))
    mean_spread = float(np.max(mean_arr) - np.min(mean_arr)) if mean_arr.size else None
    mean_rel_spread = None if mean_spread is None else float(mean_spread / scale)
    sample_sem = (
        float(np.std(mean_arr, ddof=1) / np.sqrt(mean_arr.size))
        if mean_arr.size >= 2
        else None
    )
    max_individual_sem = float(np.max(sem_arr)) if sem_arr.size else None
    sem_candidates = [
        value
        for value in (sample_sem, max_individual_sem)
        if value is not None and math.isfinite(float(value))
    ]
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

    gates = [
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
            mean_arr.size == n_reports and n_reports > 0,
            f"finite_means={mean_arr.size} reports={n_reports}",
        ),
        _gate(
            "mean_relative_spread",
            mean_rel_spread_ok,
            "mean_rel_spread={value} gate={gate}".format(
                value=mean_rel_spread,
                gate=cfg.max_mean_rel_spread,
            ),
        ),
        _gate(
            "combined_sem",
            combined_sem_rel_ok,
            "combined_sem_rel={value} gate={gate}".format(
                value=combined_sem_rel,
                gate=cfg.max_combined_sem_rel,
            ),
        ),
    ]
    passed = all(bool(gate["passed"]) for gate in gates)
    return {
        "kind": "nonlinear_window_ensemble_report",
        "claim_level": "replicated_nonlinear_window_uncertainty_gate_not_simulation_claim",
        "case": str(case),
        "comparison": str(comparison),
        "passed": passed,
        "statistics": {
            "n_reports": n_reports,
            "n_finite_means": int(mean_arr.size),
            "ensemble_mean": _json_number(float(np.mean(mean_arr)) if mean_arr.size else None),
            "mean_spread": _json_number(mean_spread),
            "mean_rel_spread": _json_number(mean_rel_spread),
            "sample_sem": _json_number(sample_sem),
            "max_individual_sem": _json_number(max_individual_sem),
            "combined_sem": _json_number(combined_sem),
            "combined_sem_rel": _json_number(combined_sem_rel),
        },
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
    required_axes = tuple(str(axis).strip() for axis in cfg.required_variant_axes)
    rows: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = {}
    for idx, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise TypeError("records must contain dictionaries")
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
        row = {
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
            "late_mean": _json_number(_report_late_mean(report)),
            "sem": _json_number(_report_sem(report)),
        }
        rows.append(row)
        by_case.setdefault(report_case, []).append(row)

    case_rows: list[dict[str, Any]] = []
    missing_artifacts: list[dict[str, Any]] = []
    for report_case in sorted(by_case):
        observed = by_case[report_case]
        ready_rows = [row for row in observed if bool(row["promotion_ready"])]
        per_axis: dict[str, Any] = {}
        for axis in required_axes:
            values = sorted(
                {
                    str(row["variant"].get(axis))
                    for row in ready_rows
                    if row["variant"].get(axis) not in (None, "")
                }
            )
            missing_count = max(0, int(cfg.min_replicates_per_case) - len(values))
            axis_passed = missing_count == 0
            per_axis[axis] = {
                "passed": axis_passed,
                "observed_distinct_values": values,
                "observed_distinct_count": len(values),
                "required_distinct_count": int(cfg.min_replicates_per_case),
                "missing_count": missing_count,
            }
            if missing_count:
                missing_artifacts.append(
                    {
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
                )
        case_rows.append(
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
            }
        )

    observed_ready = all(bool(row["promotion_ready"]) for row in rows) if rows else False
    axes_passed = not missing_artifacts and bool(case_rows)
    gates = [
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


__all__ = [
    "NonlinearWindowConvergenceConfig",
    "NonlinearWindowEnsembleConfig",
    "NonlinearWindowEnsembleManifestConfig",
    "nonlinear_window_ensemble_artifact_manifest",
    "nonlinear_window_ensemble_report",
    "nonlinear_window_convergence_from_csv",
    "nonlinear_window_convergence_from_summary",
    "nonlinear_window_convergence_report",
    "nonlinear_window_stats_promotion_ready",
]
