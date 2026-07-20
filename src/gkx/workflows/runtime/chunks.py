"""Adaptive runtime chunk execution helpers.

These helpers own the repeated adaptive runtime chunk loop used by the runtime
drivers. Keeping the loop outside ``runtime.py`` makes the execution layer
smaller without changing the accepted diagnostics truncation/stride behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Callable

import numpy as np

from gkx.diagnostics import SimulationDiagnostics
from gkx.workflows.runtime.diagnostic_arrays import (
    concat_runtime_diagnostics,
    stride_runtime_diagnostics,
    truncate_runtime_diagnostics,
    validate_finite_runtime_diagnostics,
)
from gkx.terms.config import FieldState


@dataclass(frozen=True)
class RuntimeProgressSnapshot:
    """Computed wall-clock progress fields for a chunked runtime update."""

    progress: float
    eta_seconds: float
    chunk_wall_seconds: float
    elapsed_seconds: float


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as ``MM:SS`` or ``H:MM:SS``."""

    seconds_i = max(int(round(seconds)), 0)
    minutes, secs = divmod(seconds_i, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_runtime_progress_message(
    *,
    label: str,
    chunk_index: int,
    t_elapsed: float,
    t_max: float,
    chunk_wall_seconds: float,
    elapsed_seconds: float,
) -> tuple[str, RuntimeProgressSnapshot]:
    """Return the standard adaptive-runtime progress line and policy snapshot."""

    progress = (
        min(max(float(t_elapsed) / float(t_max), 0.0), 1.0)
        if float(t_max) > 0.0
        else 1.0
    )
    eta = (
        float(elapsed_seconds) * (1.0 / progress - 1.0)
        if progress > 1.0e-12
        else float("inf")
    )
    eta_text = format_duration(eta) if np.isfinite(eta) else "--:--"
    snapshot = RuntimeProgressSnapshot(
        progress=float(progress),
        eta_seconds=float(eta),
        chunk_wall_seconds=max(float(chunk_wall_seconds), 0.0),
        elapsed_seconds=max(float(elapsed_seconds), 0.0),
    )
    message = (
        f"completed {label} chunk {int(chunk_index)}: "
        f"t={float(t_elapsed):.6g}/{float(t_max):.6g} "
        f"progress={100.0 * snapshot.progress:5.1f}% "
        f"chunk_wall={format_duration(snapshot.chunk_wall_seconds)} "
        f"elapsed={format_duration(snapshot.elapsed_seconds)} "
        f"eta={eta_text}"
    )
    return message, snapshot


@dataclass(frozen=True)
class AdaptiveChunkResult:
    """Concatenated result from one adaptive runtime chunk loop."""

    diagnostics: SimulationDiagnostics
    state: Any
    fields: FieldState


_TIME_PROGRESS_EPS = 1.0e-12


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
        [bool], tuple[Any, SimulationDiagnostics, Any, FieldState | None]
    ],
    t_max: float,
    chunk_steps: int,
    label: str,
    show_progress: bool = False,
    status_callback: Callable[[str], None] | None = None,
    diagnostics_stride: int = 1,
    max_chunks: int = 100000,
) -> AdaptiveChunkResult:
    """Run repeated diagnostic chunks until ``t_max`` is reached.

    ``integrate_chunk`` must return ``(t_chunk, diag_chunk, state, fields)`` in
    the same contract used by the runtime integrators.
    """

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    state_chunk = None
    t_elapsed = 0.0
    diag_chunks: list[SimulationDiagnostics] = []
    fields_final: FieldState | None = None
    wall_start = time.perf_counter()
    _status(
        f"starting adaptive {label} integration in chunks of {chunk_steps} steps up to t_max={float(t_max):.6g}"
    )

    for chunk in range(max_chunks):
        chunk_start = time.perf_counter()
        _t_chunk, diag_chunk, state_chunk, fields_final = integrate_chunk(show_progress)
        chunk_index = chunk + 1
        diag_chunk = _offset_chunk_diagnostics_time(diag_chunk, offset=t_elapsed)
        validate_finite_runtime_diagnostics(
            diag_chunk, label=f"adaptive {label} chunk {chunk_index}"
        )
        diag_chunks.append(diag_chunk)
        t_next = _next_elapsed_time(
            diag_chunk,
            previous_elapsed=t_elapsed,
            label=label,
            chunk_index=chunk_index,
        )
        t_elapsed = t_next
        chunk_wall = max(time.perf_counter() - chunk_start, 0.0)
        wall_elapsed = max(time.perf_counter() - wall_start, 0.0)
        message, _snapshot = build_runtime_progress_message(
            label=label,
            chunk_index=chunk_index,
            t_elapsed=t_elapsed,
            t_max=float(t_max),
            chunk_wall_seconds=chunk_wall,
            elapsed_seconds=wall_elapsed,
        )
        _status(message)
        if t_elapsed >= float(t_max):
            break
    else:
        raise RuntimeError(
            f"adaptive {label} runtime exceeded chunk limit before reaching t_max"
        )

    diag = concat_runtime_diagnostics(diag_chunks)
    diag = truncate_runtime_diagnostics(diag, t_max=float(t_max))
    stride = _effective_diagnostics_stride(diagnostics_stride)
    if stride > 1:
        diag = stride_runtime_diagnostics(diag, stride=stride)
    if fields_final is None:
        raise RuntimeError(f"adaptive {label} runtime did not produce final fields")
    return AdaptiveChunkResult(
        diagnostics=diag,
        state=state_chunk,
        fields=fields_final,
    )
