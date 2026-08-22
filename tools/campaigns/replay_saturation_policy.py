"""Replay one frozen saturation policy on source-pinned nonlinear traces.

The policy tests the trailing window at every saved checkpoint. Heat flux must
have a resolved autocorrelation time, a corrected relative SEM below the
declared threshold, and at least min_tau_multiples correlation times. Heat
flux, Wphi, and Wg must each pass the production half-window stationarity gate.
A stop is reported only after the complete decision remains true for the
declared persistence time.

This tool reads compact NPZ traces and runs no simulation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gkx.diagnostics.saturation import (
    _halves_stationary,
    _sokal_window_mean_sem,
)


@dataclass(frozen=True)
class ReplayPolicy:
    window: float = 75.0
    persistence: float = 60.0
    rel_sem: float = 0.05
    min_tau_multiples: float = 10.0
    min_samples: int = 16


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _fixed_window_decision(
    time: np.ndarray,
    heat_flux: np.ndarray,
    wphi: np.ndarray,
    wg: np.ndarray,
    end: int,
    policy: ReplayPolicy,
) -> dict[str, Any]:
    """Score exactly one causal trailing window."""
    if time[end] - time[0] < policy.window:
        return {"passed": False, "reasons": ["trace_shorter_than_window"]}
    start = int(np.searchsorted(time, time[end] - policy.window, side="left"))
    window_time = time[start : end + 1]
    if window_time.size < max(policy.min_samples, 8):
        return {"passed": False, "reasons": ["window_shorter_than_min_samples"]}
    dt = float(np.median(np.diff(window_time)))
    if not np.isfinite(dt) or dt <= 0.0:
        return {"passed": False, "reasons": ["degenerate_window_time_axis"]}

    statistics: dict[str, dict[str, Any]] = {}
    for name, signal in (
        ("heat_flux", heat_flux),
        ("Wphi", wphi),
        ("Wg", wg),
    ):
        values = signal[start : end + 1]
        mean, sem, tau, tau_resolved = _sokal_window_mean_sem(values, dt)
        first, second, halves_sem, stationary = _halves_stationary(values, dt)
        statistics[name] = {
            "mean": mean,
            "sem": sem,
            "tau_ac": tau,
            "tau_ac_resolved": tau_resolved,
            "first_half_mean": first,
            "second_half_mean": second,
            "halves_sem": halves_sem,
            "stationary": stationary,
        }

    flux = statistics["heat_flux"]
    span = float(window_time[-1] - window_time[0])
    relative_sem = float(flux["sem"] / max(abs(flux["mean"]), 1.0e-12))
    gates = {
        "tau_ac_unresolved": bool(flux["tau_ac_resolved"]),
        "window_below_min_tau_multiples": bool(
            span >= policy.min_tau_multiples * flux["tau_ac"]
        ),
        "rel_sem_above_threshold": bool(relative_sem <= policy.rel_sem),
        "heat_flux_not_stationary": bool(flux["stationary"]),
        "Wphi_not_stationary": bool(statistics["Wphi"]["stationary"]),
        "Wg_not_stationary": bool(statistics["Wg"]["stationary"]),
    }
    return {
        "passed": all(gates.values()),
        "reasons": [name for name, passed in gates.items() if not passed],
        "window_tmin": float(window_time[0]),
        "window_tmax": float(window_time[-1]),
        "window_span": span,
        "n_window": int(window_time.size),
        "output_dt": dt,
        "relative_sem": relative_sem,
        "statistics": statistics,
    }


def replay_policy(
    time: np.ndarray | Sequence[float],
    heat_flux: np.ndarray | Sequence[float],
    wphi: np.ndarray | Sequence[float],
    wg: np.ndarray | Sequence[float],
    *,
    policy: ReplayPolicy | None = None,
) -> dict[str, Any]:
    """Return pass islands and the first persistence-qualified causal stop."""
    cfg = ReplayPolicy() if policy is None else policy
    if cfg.window <= 0.0 or cfg.persistence < 0.0:
        raise ValueError("window must be positive and persistence non-negative")
    if cfg.rel_sem <= 0.0 or cfg.min_tau_multiples <= 0.0:
        raise ValueError("rel_sem and min_tau_multiples must be positive")

    arrays = [
        np.asarray(values, dtype=float).reshape(-1)
        for values in (time, heat_flux, wphi, wg)
    ]
    time_axis, flux_axis, wphi_axis, wg_axis = arrays
    if not time_axis.size or any(values.size != time_axis.size for values in arrays):
        raise ValueError(
            "time and diagnostic arrays must have one equal nonzero length"
        )
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("time and diagnostic arrays must be finite")
    if np.any(np.diff(time_axis) <= 0.0):
        raise ValueError("time must be strictly increasing")

    pass_start: int | None = None
    islands: list[dict[str, Any]] = []
    first_stop: dict[str, Any] | None = None
    terminal_decision: dict[str, Any] = {}
    for end in range(time_axis.size):
        decision = _fixed_window_decision(
            time_axis, flux_axis, wphi_axis, wg_axis, end, cfg
        )
        terminal_decision = decision
        if decision["passed"]:
            if pass_start is None:
                pass_start = end
            if (
                first_stop is None
                and time_axis[end] - time_axis[pass_start] >= cfg.persistence
            ):
                first_stop = {
                    "checkpoint_index": end,
                    "checkpoint_time": float(time_axis[end]),
                    "persistence_start": float(time_axis[pass_start]),
                    "persistence_observed": float(
                        time_axis[end] - time_axis[pass_start]
                    ),
                    "decision": decision,
                }
        elif pass_start is not None:
            previous = end - 1
            islands.append(
                {
                    "tmin": float(time_axis[pass_start]),
                    "tmax": float(time_axis[previous]),
                    "duration": float(time_axis[previous] - time_axis[pass_start]),
                }
            )
            pass_start = None
    if pass_start is not None:
        islands.append(
            {
                "tmin": float(time_axis[pass_start]),
                "tmax": float(time_axis[-1]),
                "duration": float(time_axis[-1] - time_axis[pass_start]),
            }
        )

    return {
        "kind": "nonlinear_saturation_policy_replay",
        "claim_level": "causal_post_processing_of_source_pinned_traces",
        "policy": asdict(cfg),
        "stopped": first_stop is not None,
        "first_stop": first_stop,
        "pass_islands": islands,
        "samples": int(time_axis.size),
        "tmin": float(time_axis[0]),
        "tmax": float(time_axis[-1]),
        "terminal_decision": terminal_decision,
    }


def _load_traces(
    paths: Sequence[Path],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    arrays: list[list[np.ndarray]] = [[], [], [], []]
    sources: list[dict[str, Any]] = []
    commit: str | None = None
    previous_time: float | None = None
    for path in paths:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with np.load(path, allow_pickle=False) as archive:
            values = [
                np.asarray(archive[name], dtype=float).reshape(-1)
                for name in ("time", "heat_flux", "Wphi", "Wg")
            ]
            source_commit = str(np.asarray(archive["gkx_git_commit"]).item())
            source_dirty = int(np.asarray(archive["gkx_git_dirty"]).item())
        if not source_commit or source_dirty != 0:
            raise ValueError(f"{path}: trace is not pinned to a clean GKX commit")
        if commit is not None and source_commit != commit:
            raise ValueError(f"{path}: GKX commit differs from preceding trace")
        if previous_time is not None and values[0][0] <= previous_time:
            raise ValueError(f"{path}: trace does not continue after preceding trace")
        commit = source_commit
        previous_time = float(values[0][-1])
        for parts, value in zip(arrays, values):
            parts.append(value)
        sources.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "gkx_git_commit": source_commit,
            }
        )
    return [np.concatenate(parts) for parts in arrays], sources


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, nargs="+", required=True)
    parser.add_argument("--window", type=float, default=75.0)
    parser.add_argument("--persistence", type=float, default=60.0)
    parser.add_argument("--rel-sem", type=float, default=0.05)
    parser.add_argument("--min-tau-multiples", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    arrays, sources = _load_traces(args.trace)
    report = replay_policy(
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        policy=ReplayPolicy(
            window=args.window,
            persistence=args.persistence,
            rel_sem=args.rel_sem,
            min_tau_multiples=args.min_tau_multiples,
        ),
    )
    report["source_traces"] = sources
    implementation_files = [
        Path(__file__).resolve(),
        Path(_sokal_window_mean_sem.__code__.co_filename).resolve(),
    ]
    report["policy_implementation"] = [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in implementation_files
    ]
    encoded = json.dumps(report, indent=2, default=_json_float) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    stop = report["first_stop"]
    print(
        f"stop={stop['checkpoint_time']:.6g}" if stop is not None else "stop=none",
        f"islands={len(report['pass_islands'])}",
        f"tmax={report['tmax']:.6g}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
