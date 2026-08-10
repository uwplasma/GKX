"""Adaptive runtime chunk execution helpers.

These helpers own the repeated adaptive runtime chunk loop used by the runtime
drivers. Keeping the loop outside ``runtime.py`` makes the execution layer
smaller without changing the accepted diagnostics truncation/stride behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, replace
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from gkx.diagnostics import SimulationDiagnostics
from gkx.diagnostics.metadata import ResolvedDiagnostics
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


def _chunk_diagnostics_to_host(diag: SimulationDiagnostics) -> SimulationDiagnostics:
    """Move one chunk's diagnostic arrays to host memory.

    The chunk list lives until the final concatenation, so leaving each
    chunk's arrays on the accelerator makes device memory grow linearly with
    ``t_max`` — long adaptive runs then fail inside a later chunk whose
    workspace no longer fits. ``concat_runtime_diagnostics`` converts to
    numpy anyway; doing it per chunk releases the device buffers as soon as
    the chunk completes and bounds device residency at one chunk.
    """

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
    """Write one chunk's diagnostics to disk and return the path.

    Striding per chunk already cuts the accumulated payload by the stride
    factor, but a long enough run still outgrows host RAM -- the chunk list and
    the concatenated result are both live at the end, so the peak is roughly
    twice the final series. Spilling holds one chunk at a time instead, which
    trades wall time for a peak that does not grow with ``t_max``.

    ``np.savez`` rather than pickle: the payload is plain arrays, the format is
    portable and inspectable, and a truncated file fails loudly on load.
    """

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
        [bool], tuple[Any, SimulationDiagnostics, Any, FieldState | None]
    ],
    t_max: float,
    chunk_steps: int,
    label: str,
    show_progress: bool = False,
    status_callback: Callable[[str], None] | None = None,
    diagnostics_stride: int = 1,
    max_chunks: int = 100000,
    spill_dir: Path | None = None,
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
    stride = _effective_diagnostics_stride(diagnostics_stride)
    # Samples seen before the current chunk, so the stride can keep the same
    # global indices it would have kept after concatenation.
    samples_seen = 0
    diag_chunks: list[SimulationDiagnostics] = []
    spill_paths: list[Path] = []
    if spill_dir is not None:
        spill_dir.mkdir(parents=True, exist_ok=True)
    fields_final: FieldState | None = None
    wall_start = time.perf_counter()
    _status(
        f"starting adaptive {label} integration in chunks of {chunk_steps} steps up to t_max={float(t_max):.6g}"
    )

    for chunk in range(max_chunks):
        chunk_start = time.perf_counter()
        _t_chunk, diag_chunk, state_chunk, fields_final = integrate_chunk(show_progress)
        chunk_index = chunk + 1
        diag_chunk = _chunk_diagnostics_to_host(diag_chunk)
        diag_chunk = _offset_chunk_diagnostics_time(diag_chunk, offset=t_elapsed)
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
        t_elapsed = t_next
        chunk_samples = int(np.asarray(diag_chunk.t).shape[0])
        if stride > 1:
            diag_chunk = stride_runtime_diagnostics(
                diag_chunk, stride=stride, offset=(-samples_seen) % stride
            )
        samples_seen += chunk_samples
        if spill_dir is None:
            diag_chunks.append(diag_chunk)
        else:
            spill_paths.append(_spill_chunk(diag_chunk, spill_dir, chunk_index))
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

    if spill_dir is not None:
        diag_chunks = [_load_spilled_chunk(path) for path in spill_paths]
    diag = concat_runtime_diagnostics(diag_chunks)
    # Already strided per chunk above; striding again would decimate twice.
    diag = truncate_runtime_diagnostics(diag, t_max=float(t_max))
    if fields_final is None:
        raise RuntimeError(f"adaptive {label} runtime did not produce final fields")
    return AdaptiveChunkResult(
        diagnostics=diag,
        state=state_chunk,
        fields=fields_final,
    )
