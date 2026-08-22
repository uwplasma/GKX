"""Reach a genuinely saturated nonlinear state, using GKX's own production stepper.

The nonlinear heat-flux adjoint needs a
trajectory that has actually saturated. A hand-rolled fixed-step integrator
cannot supply one: the E x B nonlinearity imposes a CFL condition that tightens
as the amplitude grows, so a step size that is stable through the linear phase
goes unstable exactly when the nonlinearity would otherwise bite. Measured, that
looked like a run whose energy grew by 3.3e49 over t=600 and never saturated.

Production solves this with machinery the hand-rolled loop skipped entirely:
CFL-adaptive stepping (``fixed_dt=false``, ``dt_max``, ``cfl``), a state
projector enforcing the reality and fixed-mode constraints, and a dealias mask.
Rather than reimplement any of that, this tool drives ``run_runtime_nonlinear``
-- the same entry point the CLI uses -- and saves the final state.

Saturation is judged by the production stop policy on the heat-flux trace,
with ``Wphi`` and ``Wg`` as stationarity guards. A state that fails is written
out anyway, flagged, so the failure is inspectable rather than silent. A
continuation segment cannot claim full-history saturation by itself. The same
tool can replay a frozen trailing-window/persistence policy on source-pinned
NPZ segments without rerunning GKX.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _campaign_output_locks(paths: Sequence[Path | None]) -> list[Any]:
    """Lock requested artifacts so two campaigns cannot share output paths."""
    handles: list[Any] = []
    targets = sorted({path.resolve() for path in paths if path is not None})
    try:
        for target in targets:
            lock_path = Path(f"{target}.lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = lock_path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.seek(0)
                owner = handle.read().strip() or "another process"
                handle.close()
                raise SystemExit(f"campaign output is locked by {owner}: {target}")
            handle.seek(0)
            handle.truncate()
            handle.write(
                f"pid={os.getpid()} host={os.uname().nodename} "
                f"started_unix={time.time():.6f}\n"
            )
            handle.flush()
            handles.append(handle)
    except BaseException:
        for handle in handles:
            handle.close()
        raise
    return handles


def _git_output(repository: Path, *args: str) -> str | None:
    """Return one Git query without making the campaign depend on Git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _campaign_source_provenance(package_file: str | Path) -> dict[str, object]:
    """Require and describe the checkout source used by this campaign."""
    repository = Path(__file__).resolve().parents[2]
    source_file = Path(package_file).resolve()
    expected_package = (repository / "src" / "gkx").resolve()
    if not source_file.is_relative_to(expected_package):
        raise SystemExit(
            f"campaign imported GKX from {source_file}, expected {expected_package}; "
            "run from this checkout with PYTHONPATH=src"
        )
    dirty = _git_output(repository, "status", "--porcelain", "--untracked-files=no")
    return {
        "source_file": str(source_file),
        "repository_root": str(repository),
        "git_commit": _git_output(repository, "rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty),
    }


def _campaign_progress(message: str) -> None:
    """Print one host-side, cumulative runtime progress update."""

    print(f"[gkx] {message}", flush=True)


def _npz_source_provenance(provenance: dict[str, object]) -> dict[str, np.ndarray]:
    """Encode source provenance without object arrays or pickle."""
    dirty = provenance["git_dirty"]
    return {
        "gkx_source_file": np.asarray(str(provenance["source_file"])),
        "gkx_repository_root": np.asarray(str(provenance["repository_root"])),
        "gkx_git_commit": np.asarray(str(provenance["git_commit"] or "")),
        "gkx_git_dirty": np.asarray(-1 if dirty is None else int(bool(dirty))),
    }


def _summary_trace_payload(
    time_axis: np.ndarray,
    heat_flux: np.ndarray,
    wphi: np.ndarray,
    wg: np.ndarray,
    *,
    trace_path: Path | None,
) -> dict[str, object]:
    """Inline a JSON-only trace or address the requested NPZ by digest."""
    if trace_path is not None:
        with trace_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "trace_artifact": {
                "schema": "nonlinear_saturation_trace_npz_v1",
                "path": str(trace_path),
                "bytes": trace_path.stat().st_size,
                "sha256": digest,
            }
        }
    return {
        "trace": [
            {"t": float(t), "heat_flux": float(q), "Wphi": float(phi), "Wg": float(g)}
            for t, q, phi, g in zip(time_axis, heat_flux, wphi, wg)
        ]
    }


def _scope_saturation_report(
    report: dict[str, Any], *, continuation: bool
) -> dict[str, Any]:
    """Prevent a continuation segment from claiming full-history saturation."""
    scoped = dict(report)
    scoped["history_scope"] = "continuation_segment" if continuation else "full_run"
    if not continuation:
        return scoped
    scoped["segment_saturated"] = bool(scoped["saturated"])
    scoped["saturated"] = False
    reasons = list(scoped.get("reasons", []))
    reasons.append("prior_history_not_in_report")
    scoped["reasons"] = reasons
    return scoped


@dataclasses.dataclass(frozen=True)
class ReplayPolicy:
    window: float = 75.0
    persistence: float = 60.0
    rel_sem: float = 0.05
    min_tau_multiples: float = 10.0
    min_samples: int = 16


_REPLAY_IDENTITY_FIELDS_V1 = (
    "campaign_identity_schema",
    "case",
    "input_sha256",
    "vmec_sha256",
    "Nx",
    "Ny",
    "Nz",
    "Nl",
    "Nm",
    "random_seed",
    "alpha",
    "npol",
    "kx",
    "ky",
)
# A v2 campaign may continue a source-compatible v1 state. The schema has
# already been checked here; compare the physical v1 fields but not the version
# string or the timestep fields that v1 could not record.
_STATE_IDENTITY_FIELDS_V1 = _REPLAY_IDENTITY_FIELDS_V1[1:-2]
_REPLAY_IDENTITY_FIELDS = _REPLAY_IDENTITY_FIELDS_V1[:-2] + (
    "time_fixed_dt",
    "time_dt",
    "time_dt_max",
    "time_cfl",
    "time_method",
    "kx",
    "ky",
)
_STATE_IDENTITY_FIELDS = _REPLAY_IDENTITY_FIELDS[:-2]


def _resolved_timestep_policy(time_cfg: Any) -> dict[str, object]:
    """Return the resolved integration policy in JSON-native types."""
    return {
        "fixed_dt": bool(time_cfg.fixed_dt),
        "dt": float(time_cfg.dt),
        "dt_max": None if time_cfg.dt_max is None else float(time_cfg.dt_max),
        "cfl": float(time_cfg.cfl),
        "method": str(time_cfg.method),
    }


def _npz_timestep_identity(policy: dict[str, object]) -> dict[str, np.ndarray]:
    """Encode the timestep policy without object arrays or pickle."""
    return {
        "time_fixed_dt": np.asarray(policy["fixed_dt"]),
        "time_dt": np.asarray(policy["dt"]),
        "time_dt_max": np.asarray(str(policy["dt_max"])),
        "time_cfl": np.asarray(policy["cfl"]),
        "time_method": np.asarray(policy["method"]),
    }


def replay_policy(
    time_axis: np.ndarray | Sequence[float],
    heat_flux: np.ndarray | Sequence[float],
    wphi: np.ndarray | Sequence[float],
    wg: np.ndarray | Sequence[float],
    *,
    policy: ReplayPolicy | None = None,
) -> dict[str, Any]:
    """Replay one causal trailing-window decision with a persistence hold."""
    from gkx.diagnostics.saturation import (
        _halves_stationary,
        _sokal_window_mean_sem,
    )

    cfg = ReplayPolicy() if policy is None else policy
    if cfg.window <= 0.0 or cfg.persistence < 0.0:
        raise ValueError("window must be positive and persistence non-negative")
    if cfg.rel_sem <= 0.0 or cfg.min_tau_multiples <= 0.0:
        raise ValueError("rel_sem and min_tau_multiples must be positive")
    arrays = [
        np.asarray(values, dtype=float).reshape(-1)
        for values in (time_axis, heat_flux, wphi, wg)
    ]
    time, flux, field_energy, free_energy = arrays
    if not time.size or any(values.size != time.size for values in arrays):
        raise ValueError("time and diagnostics must have one equal nonzero length")
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("time and diagnostics must be finite")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be strictly increasing")

    def score(end: int) -> dict[str, Any]:
        if time[end] - time[0] < cfg.window:
            return {"passed": False, "reasons": ["trace_shorter_than_window"]}
        start = int(np.searchsorted(time, time[end] - cfg.window, side="left"))
        window_time = time[start : end + 1]
        if window_time.size < max(cfg.min_samples, 8):
            return {"passed": False, "reasons": ["window_shorter_than_min_samples"]}
        dt = float(np.median(np.diff(window_time)))
        if not np.isfinite(dt) or dt <= 0.0:
            return {"passed": False, "reasons": ["degenerate_window_time_axis"]}
        statistics: dict[str, dict[str, Any]] = {}
        for name, signal in (
            ("heat_flux", flux),
            ("Wphi", field_energy),
            ("Wg", free_energy),
        ):
            values = signal[start : end + 1]
            mean, sem, tau, resolved = _sokal_window_mean_sem(values, dt)
            first, second, halves_sem, stationary = _halves_stationary(values, dt)
            statistics[name] = {
                "mean": mean,
                "sem": sem,
                "tau_ac": tau,
                "tau_ac_resolved": resolved,
                "first_half_mean": first,
                "second_half_mean": second,
                "halves_sem": halves_sem,
                "stationary": stationary,
            }
        flux_stats = statistics["heat_flux"]
        span = float(window_time[-1] - window_time[0])
        relative_sem = float(flux_stats["sem"] / max(abs(flux_stats["mean"]), 1.0e-12))
        gates = {
            "tau_ac_unresolved": bool(flux_stats["tau_ac_resolved"]),
            "window_below_min_tau_multiples": bool(
                span >= cfg.min_tau_multiples * flux_stats["tau_ac"]
            ),
            "rel_sem_above_threshold": relative_sem <= cfg.rel_sem,
            **{
                f"{name}_not_stationary": bool(stats["stationary"])
                for name, stats in statistics.items()
            },
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

    pass_start: int | None = None
    first_stop: dict[str, Any] | None = None
    islands: list[dict[str, float]] = []
    terminal: dict[str, Any] = {}
    for end in range(time.size):
        terminal = score(end)
        if terminal["passed"]:
            if pass_start is None:
                pass_start = end
            if first_stop is None and time[end] - time[pass_start] >= cfg.persistence:
                first_stop = {
                    "checkpoint_index": end,
                    "checkpoint_time": float(time[end]),
                    "persistence_start": float(time[pass_start]),
                    "persistence_observed": float(time[end] - time[pass_start]),
                    "decision": terminal,
                }
        elif pass_start is not None:
            previous = end - 1
            islands.append(
                {
                    "tmin": float(time[pass_start]),
                    "tmax": float(time[previous]),
                    "duration": float(time[previous] - time[pass_start]),
                }
            )
            pass_start = None
    if pass_start is not None:
        islands.append(
            {
                "tmin": float(time[pass_start]),
                "tmax": float(time[-1]),
                "duration": float(time[-1] - time[pass_start]),
            }
        )
    return {
        "kind": "nonlinear_saturation_policy_replay",
        "claim_level": "causal_post_processing_of_source_pinned_traces",
        "policy": dataclasses.asdict(cfg),
        "stopped": first_stop is not None,
        "first_stop": first_stop,
        "pass_islands": islands,
        "samples": int(time.size),
        "tmin": float(time[0]),
        "tmax": float(time[-1]),
        "terminal_decision": terminal,
    }


def _load_replay_traces(
    paths: Sequence[Path],
    summaries: Sequence[Path] | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Join clean, same-commit continuation traces and record their digests."""
    if summaries is not None and len(summaries) != len(paths):
        raise ValueError("replay summaries must match replay traces one-for-one")
    arrays: list[list[np.ndarray]] = [[], [], [], []]
    sources: list[dict[str, Any]] = []
    commit: str | None = None
    previous_time: float | None = None
    identity: dict[str, np.ndarray] | None = None
    for index, path in enumerate(paths):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with np.load(path, allow_pickle=False) as archive:
            values = [
                np.asarray(archive[name], dtype=float).reshape(-1)
                for name in ("time", "heat_flux", "Wphi", "Wg")
            ]
            source_commit = str(np.asarray(archive["gkx_git_commit"]).item())
            source_dirty = int(np.asarray(archive["gkx_git_dirty"]).item())
            segment_start = (
                float(np.asarray(archive["previous_t_end"]).item())
                if "previous_t_end" in archive
                else None
            )
            segment_identity = {
                name: np.asarray(archive[name])
                for name in _REPLAY_IDENTITY_FIELDS
                if name in archive
            }
        summary_source: dict[str, Any] | None = None
        if summaries is not None:
            summary_path = summaries[index]
            summary_bytes = summary_path.read_bytes()
            summary = json.loads(summary_bytes)
            provenance = summary.get("source_provenance", {})
            if (
                provenance.get("git_commit") != source_commit
                or provenance.get("git_dirty") is not False
                or float(summary.get("previous_t_end", float("nan")))
                != segment_start
            ):
                raise ValueError(f"{summary_path}: summary provenance differs from trace")
            if "trace_artifact" in summary:
                if summary["trace_artifact"].get("sha256") != digest:
                    raise ValueError(f"{summary_path}: summary digest differs from trace")
            elif "trace" in summary:
                for name, values_part in zip(
                    ("t", "heat_flux", "Wphi", "Wg"), values
                ):
                    recorded = np.asarray(
                        [sample[name] for sample in summary["trace"]], dtype=float
                    )
                    if not np.array_equal(recorded, values_part):
                        raise ValueError(
                            f"{summary_path}: summary scalar history differs from trace"
                        )
            else:
                raise ValueError(f"{summary_path}: summary does not address its trace")
            identity_payload = {
                name: summary.get(name)
                for name in (
                    "case",
                    "grid",
                    "geometry_override",
                    "random_seed",
                    "alpha",
                    "npol",
                    "timestep_policy",
                )
            }
            segment_identity["summary_identity"] = np.asarray(
                json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
            )
            summary_source = {
                "path": str(summary_path),
                "bytes": len(summary_bytes),
                "sha256": hashlib.sha256(summary_bytes).hexdigest(),
            }
        if not source_commit or source_dirty != 0:
            raise ValueError(f"{path}: trace is not pinned to a clean GKX commit")
        if commit is not None and source_commit != commit:
            raise ValueError(f"{path}: GKX commit differs from preceding trace")
        if previous_time is None:
            if segment_start not in (None, 0.0):
                raise ValueError(f"{path}: first trace omits prior history")
            identity = segment_identity
        else:
            if "campaign_identity_schema" not in segment_identity and summaries is None:
                raise ValueError(
                    f"{path}: legacy continuation replay requires matching summaries"
                )
            if segment_start is None or not np.isclose(
                segment_start, previous_time, rtol=0.0, atol=1.0e-9
            ):
                raise ValueError(f"{path}: trace is not a contiguous continuation")
            assert identity is not None
            if segment_identity.keys() != identity.keys() or any(
                not np.array_equal(segment_identity[name], value)
                for name, value in identity.items()
            ):
                raise ValueError(f"{path}: campaign identity differs from first trace")
        if not values[0].size or np.any(np.diff(values[0]) <= 0.0):
            raise ValueError(f"{path}: segment time must be strictly increasing")
        if previous_time is not None and values[0][0] <= segment_start:
            raise ValueError(f"{path}: segment does not advance its declared start")
        commit, previous_time = source_commit, float(values[0][-1])
        for parts, value in zip(arrays, values):
            parts.append(value)
        source = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "gkx_git_commit": source_commit,
        }
        if summary_source is not None:
            source["summary"] = summary_source
        sources.append(source)
    return [np.concatenate(parts) for parts in arrays], sources


def _run_policy_replay(args: argparse.Namespace) -> int:
    arrays, sources = _load_replay_traces(args.replay_trace, args.replay_summary)
    report = replay_policy(
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        policy=ReplayPolicy(
            window=args.replay_window,
            persistence=args.replay_persistence,
            rel_sem=args.replay_rel_sem,
            min_tau_multiples=args.replay_min_tau_multiples,
        ),
    )
    from gkx.diagnostics.saturation import _sokal_window_mean_sem

    implementation = [
        Path(__file__).resolve(),
        Path(_sokal_window_mean_sem.__code__.co_filename).resolve(),
    ]
    report["source_traces"] = sources
    report["policy_implementation"] = [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in implementation
    ]
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    stop = report["first_stop"]
    print(
        f"stop={stop['checkpoint_time']:.6g}" if stop is not None else "stop=none",
        f"islands={len(report['pass_islands'])}",
        f"tmax={report['tmax']:.6g}",
    )
    return 0


def _gkx_source_tree_matches(repository: str | Path, left: str, right: str) -> bool:
    """Return whether two commits contain the same installable GKX source."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "diff",
                "--quiet",
                f"{left}..{right}",
                "--",
                "src/gkx",
            ],
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _load_continuation_state(
    path: Path,
    *,
    expected_shape: tuple[int, ...],
    expected_identity: dict[str, np.ndarray],
    source_provenance: dict[str, object],
) -> tuple[np.ndarray, float]:
    """Load only a clean, source-compatible, same-campaign restart state."""
    with np.load(path, allow_pickle=False) as archive:
        state = np.asarray(archive["state"])
        required = {
            "gkx_git_commit",
            "gkx_git_dirty",
            "Nx",
            "Ny",
            "Nz",
            "Nl",
            "Nm",
            "random_seed",
        }
        missing = required.difference(archive.files)
        if missing:
            raise SystemExit(f"continuation state lacks {', '.join(sorted(missing))}")
        commit = str(np.asarray(archive["gkx_git_commit"]).item())
        dirty = int(np.asarray(archive["gkx_git_dirty"]).item())
        if dirty != 0 or not commit or not _gkx_source_tree_matches(
            str(source_provenance["repository_root"]),
            commit,
            str(source_provenance["git_commit"]),
        ):
            raise SystemExit("continuation state is not pinned to compatible GKX source")
        schema = (
            str(np.asarray(archive["campaign_identity_schema"]).item())
            if "campaign_identity_schema" in archive
            else None
        )
        if schema == "gkx_nonlinear_campaign_v1":
            fields = _STATE_IDENTITY_FIELDS_V1
        elif schema == "gkx_nonlinear_campaign_v2":
            fields = _STATE_IDENTITY_FIELDS
        elif schema is None:
            fields = ("Nx", "Ny", "Nz", "Nl", "Nm", "random_seed")
        else:
            raise SystemExit(f"unsupported continuation identity schema {schema!r}")
        if any(
            name not in archive
            or not np.array_equal(np.asarray(archive[name]), expected_identity[name])
            for name in fields
        ):
            raise SystemExit("continuation state campaign identity does not match")
        previous_t_end = float(archive["t_end"]) if "t_end" in archive else 0.0
    if state.shape != expected_shape:
        raise SystemExit(f"continuation shape {state.shape} != {expected_shape}")
    return state, previous_t_end


def _trace_spectral_payload(resolved, *, kx_full, ky_full) -> dict[str, np.ndarray]:
    """Return resolved diagnostics on GKX's physical dealiased output axes."""
    from gkx.artifacts.spectral_layout import (
        _condense_kx_for_output,
        _condense_ky_for_output,
        _dealiased_kx_values,
        _dealiased_ky_values,
    )

    kx = _dealiased_kx_values(np.asarray(kx_full))
    ky = _dealiased_ky_values(np.asarray(ky_full))
    payload = {"kx": kx, "ky": ky}
    if resolved is None:
        return payload
    for name in ("Phi2_kxt", "HeatFlux_kxst"):
        value = getattr(resolved, name)
        if value is not None:
            payload[name] = _condense_kx_for_output(
                value, full_nx=np.size(kx_full), active_nx=kx.size
            )
    for name in ("Phi2_kyt", "HeatFlux_kyst"):
        value = getattr(resolved, name)
        if value is not None:
            payload[name] = _condense_ky_for_output(
                value, full_ny=np.size(ky_full), active_ny=ky.size
            )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--toml",
        type=Path,
        default=Path(
            "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_t400.toml"
        ),
    )
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--nl", type=int, default=None)
    parser.add_argument("--nm", type=int, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--dt-max", type=float, default=None)
    parser.add_argument("--cfl", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--npol", type=float, default=None)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--diagnostics-stride", type=int, default=None)
    parser.add_argument(
        "--run-to",
        choices=("saturation", "t_max"),
        default=None,
        help="override the deck stop policy; use t_max for held-out audits",
    )
    parser.add_argument(
        "--vmec-file",
        type=Path,
        default=None,
        help="override [geometry].vmec_file after loading the deck",
    )
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=None,
        help="optional npz continuation state; the segment cannot claim "
        "full-history saturation without the preceding scalar trace",
    )
    parser.add_argument("--state-out", type=Path, default=None)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--trace-out",
        type=Path,
        default=None,
        help="compact npz with scalar traces and available kx/ky spectra",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--replay-trace",
        type=Path,
        nargs="+",
        help="replay the frozen policy on ordered source-pinned NPZ segments",
    )
    parser.add_argument(
        "--replay-summary",
        type=Path,
        nargs="+",
        help="ordered companion JSON summaries required for legacy continuations",
    )
    parser.add_argument("--replay-window", type=float, default=75.0)
    parser.add_argument("--replay-persistence", type=float, default=60.0)
    parser.add_argument("--replay-rel-sem", type=float, default=0.05)
    parser.add_argument("--replay-min-tau-multiples", type=float, default=10.0)
    args = parser.parse_args()
    args._output_locks = _campaign_output_locks(
        (args.output, args.trace_out, args.state_out)
    )
    if args.replay_trace is not None:
        return _run_policy_replay(args)

    import gkx

    source_provenance = _campaign_source_provenance(gkx.__file__)
    if source_provenance["git_dirty"] is not False:
        raise SystemExit(
            "campaign checkout has tracked changes or unknown Git state; commit "
            "the exact source before producing acceptance evidence"
        )
    print(
        f"GKX source: {source_provenance['source_file']} "
        f"commit={source_provenance['git_commit']} "
        f"dirty={source_provenance['git_dirty']}",
        flush=True,
    )
    from gkx.utils.compilation_cache import enable_persistent_compilation_cache

    cache_dir = enable_persistent_compilation_cache()
    print(f"compilation cache: {cache_dir or 'off'}", flush=True)
    import jax

    jax.config.update("jax_enable_x64", True)
    print(f"devices: {jax.devices()}", flush=True)

    from gkx import build_spectral_grid, run_runtime_nonlinear
    from gkx.diagnostics.saturation import (
        SaturationStopConfig,
        saturation_stop_decision,
    )
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    runtime, raw = load_runtime_from_toml(args.toml)
    grid_override = {
        k: v for k, v in (("Nx", args.nx), ("Ny", args.ny)) if v is not None
    }
    if args.nz is not None:
        grid_override["Nz"] = args.nz
        if runtime.grid.ntheta is not None:
            periods = runtime.grid.zp
            if periods is None:
                periods = (
                    2 * int(runtime.grid.nperiod) - 1
                    if runtime.grid.nperiod is not None
                    else 1
                )
            if args.nz % periods:
                raise SystemExit(
                    f"--nz={args.nz} must be divisible by the field-line period "
                    f"factor {periods}"
                )
            grid_override["ntheta"] = args.nz // periods
    if grid_override:
        runtime = dataclasses.replace(
            runtime, grid=dataclasses.replace(runtime.grid, **grid_override)
        )
    time_override = {}
    if args.t_max is not None:
        time_override["t_max"] = args.t_max
    if args.dt is not None:
        time_override["dt"] = args.dt
    if args.dt_max is not None:
        time_override["dt_max"] = args.dt_max
    if args.cfl is not None:
        time_override["cfl"] = args.cfl
    if args.run_to is not None:
        time_override["run_to"] = args.run_to
    if time_override:
        runtime = dataclasses.replace(
            runtime, time=dataclasses.replace(runtime.time, **time_override)
        )
    geometry_override = {}
    if args.vmec_file is not None:
        geometry_override["vmec_file"] = str(args.vmec_file.resolve())
    if args.alpha is not None:
        geometry_override["alpha"] = args.alpha
    if args.npol is not None:
        geometry_override["npol"] = args.npol
    if geometry_override:
        runtime = dataclasses.replace(
            runtime,
            geometry=dataclasses.replace(
                runtime.geometry,
                **geometry_override,  # type: ignore[arg-type]
            ),
        )
    if args.seed is not None:
        runtime = dataclasses.replace(
            runtime, init=dataclasses.replace(runtime.init, random_seed=args.seed)
        )

    time_cfg = runtime.time
    timestep_policy = _resolved_timestep_policy(time_cfg)
    spectral_grid = build_spectral_grid(runtime.grid)
    grid_shape = {
        "Nx": int(spectral_grid.kx.size),
        "Ny": int(spectral_grid.ky.size),
        "Nz": int(spectral_grid.z.size),
    }
    print(
        f"case: {args.toml.name}  grid={grid_shape}  "
        f"alpha={runtime.geometry.alpha} npol={runtime.geometry.npol}  "
        f"t_max={time_cfg.t_max}  fixed_dt={time_cfg.fixed_dt}  "
        f"dt={time_cfg.dt} dt_max={time_cfg.dt_max} cfl={time_cfg.cfl}",
        flush=True,
    )
    if time_cfg.fixed_dt:
        print(
            "  NOTE: this config uses a fixed step. The nonlinear CFL tightens "
            "with amplitude, so saturation is not guaranteed.",
            flush=True,
        )

    n_laguerre = int(args.nl or raw.get("run", {}).get("Nl", 4))
    n_hermite = int(args.nm or raw.get("run", {}).get("Nm", 8))
    vmec_file = getattr(runtime.geometry, "vmec_file", None)
    vmec_path = None if vmec_file is None else Path(str(vmec_file)).expanduser()
    replay_identity = {
        "campaign_identity_schema": np.asarray("gkx_nonlinear_campaign_v2"),
        "case": np.asarray(args.toml.name),
        "input_sha256": np.asarray(hashlib.sha256(args.toml.read_bytes()).hexdigest()),
        "vmec_sha256": np.asarray(
            hashlib.sha256(vmec_path.read_bytes()).hexdigest()
            if vmec_path is not None and vmec_path.is_file()
            else ""
        ),
        **{name: np.asarray(value) for name, value in grid_shape.items()},
        "Nl": np.asarray(n_laguerre),
        "Nm": np.asarray(n_hermite),
        "random_seed": np.asarray(int(runtime.init.random_seed)),
        "alpha": np.asarray(str(runtime.geometry.alpha)),
        "npol": np.asarray(str(runtime.geometry.npol)),
        **_npz_timestep_identity(timestep_policy),
    }
    previous_t_end = 0.0

    def run(config):
        return run_runtime_nonlinear(
            config,
            Nl=n_laguerre,
            Nm=n_hermite,
            sample_stride=args.sample_stride,
            diagnostics_stride=args.diagnostics_stride,
            return_state=True,
            diagnostics=True,
            show_progress=args.progress,
            status_callback=_campaign_progress if args.progress else None,
        )

    started = time.time()
    if args.initial_state is None:
        result = run(runtime)
    else:
        expected_shape = (
            len(runtime.species),
            n_laguerre,
            n_hermite,
            grid_shape["Ny"],
            grid_shape["Nx"],
            grid_shape["Nz"],
        )
        continuation_state, previous_t_end = _load_continuation_state(
            args.initial_state,
            expected_shape=expected_shape,
            expected_identity=replay_identity,
            source_provenance=source_provenance,
        )
        with tempfile.TemporaryDirectory(prefix="gkx-saturation-continuation-") as temp:
            restart_file = Path(temp) / "restart.bin"
            np.asarray(continuation_state, dtype=np.complex64).tofile(restart_file)
            continuation_runtime = dataclasses.replace(
                runtime,
                init=dataclasses.replace(
                    runtime.init,
                    init_file=str(restart_file),
                    init_file_scale=1.0,
                    init_file_mode="replace",
                    init_amp=0.0,
                ),
            )
            result = run(continuation_runtime)
    elapsed = time.time() - started

    diagnostics = result.diagnostics
    if diagnostics is None:
        raise RuntimeError("run returned no diagnostics; cannot judge saturation")
    times = np.asarray(diagnostics.t, dtype=float)
    flux = np.asarray(diagnostics.heat_flux_t, dtype=float)
    wphi = np.asarray(diagnostics.Wphi_t, dtype=float)
    wg = np.asarray(diagnostics.Wg_t, dtype=float)
    steps_dt = np.asarray(diagnostics.dt_t, dtype=float)
    absolute_times = previous_t_end + times

    stop_config = SaturationStopConfig(
        rel_sem=float(time_cfg.saturation_rel_sem),
        min_window=time_cfg.saturation_min_window,
    )
    report = _scope_saturation_report(
        saturation_stop_decision(
            absolute_times,
            flux,
            guard=wphi,
            free_energy_guard=wg,
            config=stop_config,
        ),
        continuation=args.initial_state is not None,
    )
    print(
        f"\nran {times.size} samples to t={absolute_times[-1]:.6g} in {elapsed:.1f}s",
        flush=True,
    )
    print(
        f"  adaptive dt: min={np.nanmin(steps_dt):.4g} "
        f"median={np.nanmedian(steps_dt):.4g} max={np.nanmax(steps_dt):.4g}",
        flush=True,
    )
    if report.get("tau_ac") is not None:
        tau_ac = float(report["tau_ac"])
        window_span = float(report["window_span"])
        print(
            f"  window={report['window_tmin']:.2f}--{report['window_tmax']:.2f}  "
            f"tau_ac={tau_ac:.2f}  window/tau_ac={window_span / tau_ac:.1f}  "
            f"rel_sem={report['rel_sem']:.1%}  mean flux={report['mean']:.4e}",
            flush=True,
        )
    print(
        f"  Wphi stationary={report['guard_stationary']}  "
        f"Wg stationary={report['Wg_guard_stationary']}",
        flush=True,
    )
    if report["reasons"]:
        print(f"  failed gates: {', '.join(report['reasons'])}", flush=True)
    print(f"  -> {'SATURATED' if report['saturated'] else 'NOT SATURATED'}", flush=True)

    if args.trace_out is not None:
        payload = {
            "time": absolute_times,
            "dt": steps_dt,
            "heat_flux": flux,
            "Wphi": wphi,
            "Wg": wg,
            "Nl": np.asarray(n_laguerre),
            "Nm": np.asarray(n_hermite),
            "elapsed_seconds": np.asarray(elapsed),
            "previous_t_end": np.asarray(previous_t_end),
            **replay_identity,
        }
        payload.update(
            _trace_spectral_payload(
                diagnostics.resolved,
                kx_full=spectral_grid.kx,
                ky_full=spectral_grid.ky,
            )
        )
        payload.update(_npz_source_provenance(source_provenance))
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.trace_out, **payload)  # type: ignore[arg-type]
        print(f"trace written: {args.trace_out}", flush=True)

    if args.state_out is not None and result.state is not None:
        args.state_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.state_out,
            state=np.asarray(result.state),
            saturated=report["saturated"],
            t_end=absolute_times[-1],
            # The step the trajectory was actually produced with. Without it the
            # consumer has to be told by hand, and a mismatched --dt silently
            # rescales its time axis: every t/tau_ac it reports would be wrong
            # while every number still looked plausible.
            adaptive_dt=float(np.nanmedian(steps_dt)),
            method=str(time_cfg.method),
            Nx=grid_shape["Nx"],
            Ny=grid_shape["Ny"],
            Nz=grid_shape["Nz"],
            Nl=n_laguerre,
            Nm=n_hermite,
            random_seed=int(runtime.init.random_seed),
            tau_ac=(
                float(report["tau_ac"])
                if report.get("tau_ac") is not None
                else float("nan")
            ),
            **{
                name: replay_identity[name]
                for name in _STATE_IDENTITY_FIELDS
                if name
                not in {"Nx", "Ny", "Nz", "Nl", "Nm", "random_seed"}
            },
            **_npz_source_provenance(source_provenance),
        )
        print(f"state written: {args.state_out}", flush=True)

    summary = {
        "kind": "nonlinear_saturated_state",
        "claim_level": "production_saturation_stop_policy_audit",
        "source_provenance": source_provenance,
        "case": args.toml.name,
        "initial_state": None
        if args.initial_state is None
        else str(args.initial_state),
        "previous_t_end": previous_t_end,
        "grid": grid_shape,
        "grid_override": grid_override,
        "geometry_override": geometry_override,
        "alpha": float(runtime.geometry.alpha),
        "npol": None if runtime.geometry.npol is None else float(runtime.geometry.npol),
        "random_seed": int(runtime.init.random_seed),
        "t_max": float(time_cfg.t_max),
        "fixed_dt": bool(time_cfg.fixed_dt),
        "timestep_policy": timestep_policy,
        "adaptive_dt": {
            "min": float(np.nanmin(steps_dt)),
            "median": float(np.nanmedian(steps_dt)),
            "max": float(np.nanmax(steps_dt)),
        },
        "wall_seconds": elapsed,
        "samples": int(times.size),
        "report": report,
    }
    summary.update(
        _summary_trace_payload(
            absolute_times, flux, wphi, wg, trace_path=args.trace_out
        )
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"written: {args.output}", flush=True)
    return 0 if report["saturated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
