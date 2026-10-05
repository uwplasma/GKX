"""Late-window transport diagnostics and promotion gates.

This module owns reusable nonlinear transport-window statistics that support
quasilinear calibration, nonlinear transport acceptance, and release gates. It
is intentionally diagnostics-focused: long-run campaign launch policy belongs in
``tools`` and tests, while solver kernels remain in ``operators`` and
``solvers``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
import json
import math

import numpy as np

from gkx.diagnostics.metadata import _explicit_true, _gate


# Consolidated from window_config.py.
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


# Consolidated from window_statistics.py.
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


def _empty_window_statistics(
    config: NonlinearWindowConvergenceConfig,
) -> dict[str, Any]:
    return {
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
        "bootstrap_samples": int(config.bootstrap_samples),
        "bootstrap_seed": int(config.bootstrap_seed),
    }


def _validated_late_window(
    time: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    config: NonlinearWindowConvergenceConfig,
) -> dict[str, Any]:
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if t.size != y.size:
        raise ValueError("time and values must have the same length")
    if t.size == 0:
        raise ValueError("time and values must not be empty")
    if not np.all(np.isfinite(t)):
        raise ValueError("time contains non-finite samples")
    order = np.argsort(t)
    t_sorted = t[order]
    y_sorted = y[order]
    cutoff, selected_tmin, selected_tmax = _late_window_bounds(t_sorted, config)
    late_mask = (t_sorted >= cutoff) & (t_sorted <= selected_tmax)
    late_t = t_sorted[late_mask]
    late_y_raw = y_sorted[late_mask]
    finite_y = np.isfinite(late_y_raw)
    finite_late_y = late_y_raw[finite_y]
    finite_late_t = late_t[finite_y]
    n_late = int(late_y_raw.size)
    return {
        "t_sorted": t_sorted,
        "cutoff": cutoff,
        "selected_tmin": selected_tmin,
        "selected_tmax": selected_tmax,
        "finite_late_t": finite_late_t,
        "finite_late_y": finite_late_y,
        "n_late": n_late,
        "n_finite_late": int(finite_late_y.size),
        "n_nonfinite_late": int(n_late - finite_late_y.size),
    }


def _split_half_drift(values: np.ndarray, scale: float) -> tuple[float | None, ...]:
    midpoint = values.size // 2
    first_half = values[:midpoint]
    second_half = values[midpoint:]
    if first_half.size == 0 or second_half.size == 0:
        return None, None, None, None
    first_half_mean = float(np.mean(first_half))
    second_half_mean = float(np.mean(second_half))
    drift = abs(second_half_mean - first_half_mean)
    return first_half_mean, second_half_mean, drift, drift / scale


def _terminal_window_stats(
    finite_late_t: np.ndarray,
    finite_late_y: np.ndarray,
    late_mean: float,
    scale: float,
    config: NonlinearWindowConvergenceConfig,
) -> dict[str, Any]:
    n_finite_late = int(finite_late_y.size)
    terminal_start = min(
        n_finite_late - 1,
        max(
            0, int(math.floor((1.0 - float(config.terminal_fraction)) * n_finite_late))
        ),
    )
    terminal_y = finite_late_y[terminal_start:]
    terminal_t = finite_late_t[terminal_start:]
    terminal_mean = float(np.mean(terminal_y))
    terminal_delta = abs(terminal_mean - late_mean)
    return {
        "terminal_mean": terminal_mean,
        "terminal_mean_delta": terminal_delta,
        "terminal_mean_rel_delta": terminal_delta / scale,
        "terminal_tmin": float(terminal_t[0]),
        "terminal_tmax": float(terminal_t[-1]),
        "terminal_n_samples": int(terminal_y.size),
    }


def _block_uncertainty_stats(
    finite_late_y: np.ndarray,
    sample_sem: float,
    scale: float,
    config: NonlinearWindowConvergenceConfig,
) -> dict[str, Any]:
    block_size = (
        int(config.block_size)
        if config.block_size is not None
        else _default_block_size(finite_late_y.size, int(config.min_blocks))
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
        samples=int(config.bootstrap_samples),
        seed=int(config.bootstrap_seed),
    )
    sem_candidates = [
        value
        for value in (sample_sem, block_sem, boot_sem)
        if value is not None and math.isfinite(float(value))
    ]
    sem = max(sem_candidates) if sem_candidates else None
    return {
        "block_sem": block_sem,
        "block_bootstrap_sem": boot_sem,
        "sem": sem,
        "sem_rel": None if sem is None else float(sem / scale),
        "block_size": int(block_size),
        "n_blocks": n_blocks,
    }


def _late_window_statistics(
    finite_late_t: np.ndarray,
    finite_late_y: np.ndarray,
    config: NonlinearWindowConvergenceConfig,
) -> dict[str, Any]:
    stats = _empty_window_statistics(config)
    n_finite_late = int(finite_late_y.size)
    if n_finite_late < 2:
        return stats
    late_mean = float(np.mean(finite_late_y))
    late_std = float(np.std(finite_late_y, ddof=0))
    sample_sem = float(np.std(finite_late_y, ddof=1) / np.sqrt(n_finite_late))
    scale = max(abs(late_mean), float(config.value_floor))
    first_mean, second_mean, drift, rel_drift = _split_half_drift(finite_late_y, scale)
    stats.update(
        {
            "late_mean": late_mean,
            "late_std": late_std,
            "sample_sem": sample_sem,
            "running_mean_drift": drift,
            "running_mean_rel_drift": rel_drift,
            "first_half_mean": first_mean,
            "second_half_mean": second_mean,
        }
    )
    stats.update(
        _terminal_window_stats(finite_late_t, finite_late_y, late_mean, scale, config)
    )
    stats.update(_block_uncertainty_stats(finite_late_y, sample_sem, scale, config))
    return stats


def _finite_window_gates(
    late: dict[str, Any], config: NonlinearWindowConvergenceConfig
) -> list[dict[str, Any]]:
    finite_gate = late["n_late"] > 0 and late["n_finite_late"] > 0
    if bool(config.require_all_finite):
        finite_gate = finite_gate and late["n_nonfinite_late"] == 0
    return [
        _gate(
            "finite_late_window",
            finite_gate,
            "finite={finite} nonfinite={nonfinite} late={late}".format(
                finite=late["n_finite_late"],
                nonfinite=late["n_nonfinite_late"],
                late=late["n_late"],
            ),
        ),
        _gate(
            "finite_sample_count",
            late["n_finite_late"] >= int(config.min_samples),
            f"finite_late_samples={late['n_finite_late']} min_samples={config.min_samples}",
        ),
    ]


def _convergence_gate_rows(
    stats: dict[str, Any], config: NonlinearWindowConvergenceConfig
) -> list[dict[str, Any]]:
    rel_drift = stats["running_mean_rel_drift"]
    terminal_rel_delta = stats["terminal_mean_rel_delta"]
    sem_rel = stats["sem_rel"]
    return [
        _gate(
            "minimum_block_count",
            int(stats["n_blocks"]) >= int(config.min_blocks),
            f"n_blocks={stats['n_blocks']} min_blocks={config.min_blocks}",
        ),
        _gate(
            "running_mean_drift",
            _finite_number(rel_drift)
            and float(rel_drift) <= float(config.max_running_mean_rel_drift),
            "running_mean_rel_drift={value} gate={gate}".format(
                value=rel_drift,
                gate=config.max_running_mean_rel_drift,
            ),
        ),
        _gate(
            "terminal_sample_count",
            int(stats["terminal_n_samples"]) >= int(config.min_terminal_samples),
            "terminal_samples={value} min_terminal_samples={gate}".format(
                value=stats["terminal_n_samples"],
                gate=config.min_terminal_samples,
            ),
        ),
        _gate(
            "terminal_mean_agreement",
            _finite_number(terminal_rel_delta)
            and float(terminal_rel_delta) <= float(config.max_terminal_mean_rel_delta),
            "terminal_mean_rel_delta={value} gate={gate} terminal_fraction={fraction}".format(
                value=terminal_rel_delta,
                gate=config.max_terminal_mean_rel_delta,
                fraction=config.terminal_fraction,
            ),
        ),
        _gate(
            "block_bootstrap_sem",
            _finite_number(stats["block_bootstrap_sem"])
            and _finite_number(sem_rel)
            and float(sem_rel) <= float(config.max_sem_rel),
            f"sem_rel={sem_rel} gate={config.max_sem_rel}",
        ),
    ]


def _window_metadata(
    late: dict[str, Any], config: NonlinearWindowConvergenceConfig
) -> dict[str, Any]:
    finite_late_t = late["finite_late_t"]
    return {
        "input_tmin": _json_number(config.tmin),
        "input_tmax": _json_number(config.tmax),
        "selected_tmin": late["selected_tmin"],
        "selected_tmax": late["selected_tmax"],
        "transient_fraction": float(config.transient_fraction),
        "transient_cutoff": float(late["cutoff"]),
        "late_tmin": float(finite_late_t[0]) if finite_late_t.size else None,
        "late_tmax": float(finite_late_t[-1]) if finite_late_t.size else None,
        "n_total": int(late["t_sorted"].size),
        "n_late": late["n_late"],
        "n_finite_late": late["n_finite_late"],
        "n_nonfinite_late": late["n_nonfinite_late"],
    }


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
    late = _validated_late_window(time, values, cfg)
    stats = _late_window_statistics(late["finite_late_t"], late["finite_late_y"], cfg)
    gates = _finite_window_gates(late, cfg) + _convergence_gate_rows(stats, cfg)
    passed = all(bool(gate["passed"]) for gate in gates)
    return {
        "kind": "nonlinear_window_convergence_report",
        "claim_level": "nonlinear_holdout_window_metadata_not_simulation_claim",
        "case": str(case),
        "observable": str(observable),
        "passed": passed,
        "window": _window_metadata(late, cfg),
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


# Consolidated from window_io.py.
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
    diagnostics_source: str = "gkx",
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


# Consolidated from window_promotion.py.
def _append_missing_finite_fields(
    failures: list[str],
    values: dict[str, Any],
    *,
    prefix: str,
    fields: tuple[str, ...],
) -> None:
    """Append missing/non-finite diagnostics with stable message text."""

    for field in fields:
        if not _finite_number(values.get(field)):
            failures.append(f"missing/non-finite {prefix}.{field}")


def _ensemble_window_stats_failures(stats: dict[str, Any]) -> list[str]:
    """Return promotion failures for replicated ensemble-window reports."""

    failures: list[str] = []
    if not _explicit_true(stats.get("passed")):
        failures.append("nonlinear window ensemble report did not pass")
    gate_report = stats.get("gate_report")
    if not isinstance(gate_report, dict) or not _explicit_true(
        gate_report.get("passed")
    ):
        failures.append("missing passed ensemble gate_report")
    statistics = stats.get("statistics")
    if not isinstance(statistics, dict):
        failures.append("missing ensemble statistics object")
        statistics = {}
    _append_missing_finite_fields(
        failures,
        statistics,
        prefix="statistics",
        fields=("ensemble_mean", "combined_sem", "combined_sem_rel"),
    )
    rows = stats.get("rows")
    if not isinstance(rows, list) or not rows:
        failures.append("ensemble report has no rows")
    else:
        ready_rows = [
            row
            for row in rows
            if isinstance(row, dict) and _explicit_true(row.get("promotion_ready"))
        ]
        if len(ready_rows) != len(rows):
            failures.append("not all ensemble rows are promotion-ready")
        if not any(
            isinstance(row, dict) and str(row.get("source_artifact", "")).strip()
            for row in rows
        ):
            failures.append("missing ensemble source_artifact provenance")
    return failures


def _declares_transient_cutoff(window: dict[str, Any]) -> bool:
    raw_transient_fraction = window.get("transient_fraction", 0.0)
    return _finite_number(window.get("input_tmin")) or (
        _finite_number(raw_transient_fraction) and float(raw_transient_fraction) > 0.0
    )


def _convergence_window_stats_failures(stats: dict[str, Any]) -> list[str]:
    """Return promotion failures for single-run convergence-window reports."""

    failures: list[str] = []
    if stats.get("kind") != "nonlinear_window_convergence_report":
        failures.append("unexpected nonlinear_window_stats kind")
    if not _explicit_true(stats.get("passed")):
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
    _append_missing_finite_fields(
        failures,
        statistics,
        prefix="statistics",
        fields=(
            "late_mean",
            "sem",
            "block_bootstrap_sem",
            "running_mean_rel_drift",
            "terminal_mean_rel_delta",
        ),
    )
    window = stats.get("window")
    if not isinstance(window, dict):
        failures.append("missing window object")
        window = {}
    _append_missing_finite_fields(
        failures,
        window,
        prefix="window",
        fields=("transient_cutoff", "late_tmin", "late_tmax"),
    )
    if not _declares_transient_cutoff(window):
        failures.append("missing declared transient cutoff policy")
    n_finite_late = window.get("n_finite_late", 0)
    if not _finite_number(n_finite_late) or int(float(n_finite_late)) <= 0:
        failures.append("window has no finite late samples")
    gate_report = stats.get("gate_report")
    if not isinstance(gate_report, dict) or not _explicit_true(
        gate_report.get("passed")
    ):
        failures.append("missing passed gate_report")
    return failures


def nonlinear_window_stats_promotion_ready(
    stats: object,
) -> tuple[bool, list[str]]:
    """Return whether serialized nonlinear window metadata can support promotion."""

    if not isinstance(stats, dict):
        return False, ["missing nonlinear_window_stats object"]
    if stats.get("kind") == "nonlinear_window_ensemble_report":
        failures = _ensemble_window_stats_failures(stats)
    else:
        failures = _convergence_window_stats_failures(stats)
    return not failures, failures


def _report_statistic(report: dict[str, Any], name: str) -> float | None:
    statistics = report.get("statistics")
    if not isinstance(statistics, dict):
        return None
    value = statistics.get(name)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


# Consolidated from window_ensemble.py.
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
        late_mean = _report_statistic(report, "late_mean")
        sem = _report_statistic(report, "sem")
        report_passed = _explicit_true(report.get("passed"))
        ready, failures = nonlinear_window_stats_promotion_ready(report)
        if not (report_passed and ready):
            individual_ready = False
        if late_mean is not None:
            means.append(late_mean)
        if sem is not None:
            sems.append(sem)
        provenance = report.get("provenance")
        provenance_dict: dict[str, Any] = (
            provenance if isinstance(provenance, dict) else {}
        )
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
    passed = all(bool(gate["passed"]) for gate in gates)
    return {
        "kind": "nonlinear_window_ensemble_report",
        "claim_level": "replicated_nonlinear_window_uncertainty_gate_not_simulation_claim",
        "case": str(case),
        "comparison": str(comparison),
        "passed": passed,
        "statistics": {"n_reports": len(rows), **statistics},
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


__all__ = [
    "NonlinearWindowConvergenceConfig",
    "NonlinearWindowEnsembleConfig",
    "nonlinear_window_convergence_from_csv",
    "nonlinear_window_convergence_from_summary",
    "nonlinear_window_convergence_report",
    "nonlinear_window_ensemble_report",
    "nonlinear_window_stats_promotion_ready",
]
