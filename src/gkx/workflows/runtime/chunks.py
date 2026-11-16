"""Bounded-memory adaptive runtime chunks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields as dataclass_fields, replace
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from gkx.callbacks import _format_duration
from gkx.diagnostics import SimulationDiagnostics
from gkx.diagnostics.metadata import ResolvedDiagnostics
from gkx.workflows.runtime.diagnostic_arrays import (
    _time_sample_runtime_diagnostics,
    concat_runtime_diagnostics,
    slice_runtime_diagnostics,
    stride_runtime_diagnostics,
    validate_finite_runtime_diagnostics,
)
from gkx.terms.config import FieldState


def _flux_status(decision: dict[str, Any] | None) -> str:
    """Convergence wording for the saturation detector's latest verdict."""

    d = decision or {}
    if (
        d.get("mean") is None
        or d.get("sem") is None
        or d.get("status")
        in (
            "growing",
            "quiescent",
        )
    ):
        return "transient"
    word = "saturated" if d.get("saturated") else "saturating"
    return f"{word}: Q={d['mean']:.3g}+-{d['sem']:.2g}"


class ProgressReporter:
    """Status callback of a nonlinear run that prints one progress line per ``interval`` seconds.

    Pass it as ``status_callback``: setup messages are printed as they come, the
    run reports through :meth:`update`, and :meth:`finish` prints the summary
    and writes the figures. ``clock`` and ``emit`` can be replaced for tests.
    """

    def __init__(
        self,
        emit: Callable[[str], None] | None = None,
        *,
        interval: float = 2.0,
        window: float = 10.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.emit = emit or (lambda message: print(message, flush=True))
        self.interval, self.window, self.clock = interval, window, clock
        self.start = clock()
        self._last: float | None = None
        self._points: deque[tuple[float, float]] = deque()

    def __call__(self, message: str) -> None:
        self.emit(message)

    def update(
        self, t: float, t_max: float, q: Any, decision: dict[str, Any] | None = None
    ) -> None:
        """Record the simulation time reached; print when ``interval`` has passed."""

        now = self.clock()
        pts = self._points
        pts.append((now, t))
        while len(pts) > 2 and now - pts[1][0] >= self.window:
            pts.popleft()
        if self._last is not None and now - self._last < self.interval and t < t_max:
            return
        self._last = now
        (w0, t0), rate = pts[0], 0.0
        if t > t0:
            rate = (t - t0) / max(now - w0, 1e-12)
        eta = _format_duration((t_max - t) / rate) if rate > 0 else "--:--"
        print_q = ", ".join(f"{float(x):.3g}" for x in q)
        self.emit(
            f"{_format_duration(now - self.start)}  t={t:.6g}/{t_max:.6g}  "
            f"{100 * min(t / t_max, 1):3.0f}%  eta {eta}  Q={print_q}  "
            f"{_flux_status(decision)}"
        )

    def finish(self, result: Any, plot: Any = None, *, ky: Any = None) -> None:
        """Print the run summary; write the figures at ``plot`` (a path stem) if given."""

        sat = result.saturation or {}
        diag = result.diagnostics
        t_end = float(result.t[-1])
        steps = round(t_end / float(diag.dt_mean)) if diag is not None else 0
        sem = sat.get("sem")
        q = (
            "no saturation estimate"
            if sat.get("mean") is None
            else f"Q={sat['mean']:.4g}{'' if sem is None else f'+-{sem:.2g}'} ({sat['status']})"
        )
        self.emit(
            f"done in {_format_duration(self.clock() - self.start)}: "
            f"t={t_end:.6g}, {steps} steps, {q}"
        )
        if plot is not None:
            from gkx.artifacts.run_figures import plot_nonlinear_result

            for path in plot_nonlinear_result(result, plot, ky=ky):
                self.emit(f"saved {path}")


@dataclass(frozen=True)
class AdaptiveChunkResult:
    """Concatenated result from one adaptive runtime chunk loop."""

    diagnostics: SimulationDiagnostics
    state: Any
    fields: FieldState
    # Last stop_condition decision, when one was supplied. Carries the
    # measured saturation window statistics whether or not the run stopped
    # before t_max.
    stop_decision: dict[str, Any] | None = None


_TIME_PROGRESS_EPS = 1.0e-12


def _chunk_diagnostics_to_host(diag: SimulationDiagnostics) -> SimulationDiagnostics:
    """Move a completed chunk to host so device residency stays bounded."""

    def _host(value):
        return None if value is None else np.asarray(value)

    resolved = diag.resolved
    if resolved is not None:
        resolved = replace(
            resolved,
            **{
                field.name: _host(getattr(resolved, field.name))
                for field in dataclass_fields(ResolvedDiagnostics)
            },
        )
    converted = {
        field.name: _host(getattr(diag, field.name))
        for field in dataclass_fields(SimulationDiagnostics)
        if field.name != "resolved"
    }
    return replace(diag, **converted, resolved=resolved)


def _spill_chunk(
    diag: SimulationDiagnostics, spill_dir: Path, chunk_index: int
) -> Path:
    """Spill plain arrays so host memory also stays bounded."""

    payload: dict[str, np.ndarray] = {}
    for field in dataclass_fields(SimulationDiagnostics):
        if field.name == "resolved":
            continue
        value = getattr(diag, field.name)
        if value is not None:
            payload[field.name] = np.asarray(value)
    resolved = diag.resolved
    if resolved is not None:
        for field in dataclass_fields(ResolvedDiagnostics):
            value = getattr(resolved, field.name)
            if value is not None:
                payload[f"resolved.{field.name}"] = np.asarray(value)
    path = spill_dir / f"chunk_{chunk_index:06d}.npz"
    # numpy's stub types savez's second positional as allow_pickle, so the
    # keyword-array form it actually documents does not type-check.
    np.savez(path, **payload)  # type: ignore[arg-type]
    return path


def _load_spilled_chunk(path: Path) -> SimulationDiagnostics:
    """Read back one spilled chunk, restoring the resolved payload."""

    with np.load(path) as data:
        flat = {key: data[key] for key in data.files}
    resolved_fields = {
        name.split(".", 1)[1]: value
        for name, value in flat.items()
        if name.startswith("resolved.")
    }
    direct = {name: value for name, value in flat.items() if "." not in name}
    resolved = ResolvedDiagnostics(**resolved_fields) if resolved_fields else None
    return SimulationDiagnostics(**direct, resolved=resolved)


def _offset_chunk_diagnostics_time(
    diag: SimulationDiagnostics,
    *,
    offset: float,
) -> SimulationDiagnostics:
    """Return a chunk diagnostic payload shifted onto the accumulated time axis."""

    return replace(diag, t=np.asarray(diag.t) + float(offset))


def _chunk_end_time(
    diag: SimulationDiagnostics,
    *,
    label: str,
    chunk_index: int,
) -> float:
    """Return the last diagnostic time sample for one adaptive chunk."""

    t_arr = np.asarray(diag.t, dtype=float)
    if t_arr.size == 0:
        raise RuntimeError(
            f"adaptive {label} chunk {int(chunk_index)} produced no time samples"
        )
    return float(t_arr[-1])


def _next_elapsed_time(
    diag: SimulationDiagnostics,
    *,
    previous_elapsed: float,
    label: str,
    chunk_index: int,
) -> float:
    """Validate and return the accumulated end time for one adaptive chunk."""

    t_next = _chunk_end_time(diag, label=label, chunk_index=chunk_index)
    if t_next <= float(previous_elapsed) + _TIME_PROGRESS_EPS:
        raise RuntimeError(f"adaptive {label} runtime made no time-step progress")
    return t_next


def _effective_diagnostics_stride(diagnostics_stride: int) -> int:
    """Normalize runtime diagnostic stride with a floor at one."""

    return int(max(diagnostics_stride, 1))


def run_adaptive_runtime_chunk_loop(
    *,
    integrate_chunk: Callable[
        [bool, float], tuple[Any, SimulationDiagnostics, Any, FieldState | None]
    ],
    t_max: float,
    chunk_steps: int,
    label: str,
    show_progress: bool = False,
    status_callback: Callable[[str], None] | None = None,
    diagnostics_stride: int = 1,
    diagnostics_interval: float | None = None,
    time_offset: float = 0.0,
    progress_t_max: float | None = None,
    progress: Callable[..., None] | None = None,
    monitor: Callable[[np.ndarray, np.ndarray, bool], dict[str, Any]] | None = None,
    max_chunks: int = 100000,
    horizon_slack: float = 0.0,
    spill_dir: Path | None = None,
    stop_condition: Callable[[np.ndarray, np.ndarray, bool], dict[str, Any]]
    | None = None,
) -> AdaptiveChunkResult:
    """Run to a stop or ``t_max``; check unstrided traces between chunks."""

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    state_chunk = None
    t_elapsed = 0.0
    stride = _effective_diagnostics_stride(diagnostics_stride)
    # Preserve the global stride phase across chunks.
    samples_seen = 0
    diag_chunks: list[SimulationDiagnostics] = []
    # Stop checks retain the unstrided scalar traces.
    trace_chunks: list[tuple[np.ndarray, np.ndarray]] = []
    stop_decision: dict[str, Any] | None = None
    spill_paths: list[Path] = []
    if spill_dir is not None:
        spill_dir.mkdir(parents=True, exist_ok=True)
    fields_final: FieldState | None = None
    progress_target = (
        float(t_max) + time_offset if progress_t_max is None else progress_t_max
    )
    _status(
        f"integrating {label} from t={time_offset:.6g} up to t_max={progress_target:.6g}"
        " (the first progress line follows compilation)"
    )

    for chunk in range(max_chunks):
        remaining_time = max(float(t_max) - t_elapsed, 0.0)
        _t_chunk, diag_chunk, state_chunk, fields_final = integrate_chunk(
            False, remaining_time
        )
        chunk_index = chunk + 1
        diag_chunk = _chunk_diagnostics_to_host(diag_chunk)
        diag_chunk = _offset_chunk_diagnostics_time(diag_chunk, offset=t_elapsed)
        time_dtype = np.asarray(diag_chunk.t).dtype
        # horizon_slack: a fixed step budget can end short of t_max by the
        # rounding error summed over thousands of steps.
        time_tol = max(
            _TIME_PROGRESS_EPS,
            8.0 * np.finfo(time_dtype).eps * max(float(t_max), 1.0),
            horizon_slack,
        )
        terminal = np.flatnonzero(np.asarray(diag_chunk.t) >= float(t_max) - time_tol)
        if terminal.size:
            diag_chunk = slice_runtime_diagnostics(diag_chunk, int(terminal[0]) + 1)
        validate_finite_runtime_diagnostics(
            diag_chunk, label=f"adaptive {label} chunk {chunk_index}"
        )
        # t_elapsed must come from the unstrided chunk: striding can drop the
        # last sample, and a chunk whose reported end time moved backwards would
        # loop forever.
        t_next = _next_elapsed_time(
            diag_chunk,
            previous_elapsed=t_elapsed,
            label=label,
            chunk_index=chunk_index,
        )
        if t_next > float(t_max) + time_tol:
            raise RuntimeError(
                f"adaptive {label} chunk {chunk_index} crossed t_max; "
                "integrate_chunk must honor remaining_time"
            )
        previous_time = t_elapsed
        t_elapsed = t_next
        reached_horizon = t_elapsed >= float(t_max) - time_tol
        stop_now = False
        decision: dict[str, Any] | None = None
        check = stop_condition or monitor
        if check is not None:
            species = diag_chunk.heat_flux_species_t
            flux = diag_chunk.heat_flux_t if species is None else species
            trace_chunks.append(
                (np.asarray(diag_chunk.t), np.asarray(flux, dtype=float))
            )
            decision = check(
                np.concatenate([chunk[0] for chunk in trace_chunks]),
                np.concatenate([chunk[1] for chunk in trace_chunks]),
                reached_horizon,
            )
            if stop_condition is not None:
                stop_decision = decision
                stop_now = bool(decision.get("stop"))
        chunk_samples = int(np.asarray(diag_chunk.t).shape[0])
        by_species = diag_chunk.heat_flux_species_t
        q_last = np.atleast_1d(
            np.asarray(diag_chunk.heat_flux_t if by_species is None else by_species)[-1]
        )
        if diagnostics_interval is not None:
            diag_chunk = _time_sample_runtime_diagnostics(
                diag_chunk,
                diagnostics_interval,
                previous_time=previous_time + time_offset,
                time_offset=time_offset,
                keep_last=stop_now or reached_horizon,
            )
        elif stride > 1:
            diag_chunk = stride_runtime_diagnostics(
                diag_chunk,
                stride=stride,
                offset=(-samples_seen) % stride,
                keep_last=stop_now or reached_horizon,
            )
        samples_seen += chunk_samples
        if spill_dir is None:
            diag_chunks.append(diag_chunk)
        else:
            spill_paths.append(_spill_chunk(diag_chunk, spill_dir, chunk_index))
        if progress is not None:
            progress(t_elapsed + time_offset, progress_target, q_last, decision)
        if stop_now:
            _status(f"stopping {label} integration at t={t_elapsed + time_offset:.6g}")
            break
        if reached_horizon:
            break
    else:
        raise RuntimeError(
            f"adaptive {label} runtime exceeded chunk limit before reaching t_max"
        )

    if spill_dir is not None:
        diag_chunks = [_load_spilled_chunk(path) for path in spill_paths]
    diag = concat_runtime_diagnostics(diag_chunks)
    if fields_final is None:
        raise RuntimeError(f"adaptive {label} runtime did not produce final fields")
    return AdaptiveChunkResult(
        diagnostics=diag,
        state=state_chunk,
        fields=fields_final,
        stop_decision=stop_decision,
    )
