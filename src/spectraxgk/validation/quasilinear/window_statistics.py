"""Late-window statistics for nonlinear transport diagnostics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence
import math

import numpy as np

from spectraxgk.validation.quasilinear.window_config import (
    NonlinearWindowConvergenceConfig,
    _validate_config,
)


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
            max(
                0, int(math.floor((1.0 - float(cfg.terminal_fraction)) * n_finite_late))
            ),
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


__all__ = ["nonlinear_window_convergence_report"]
