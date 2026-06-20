"""Runtime progress and ETA formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


__all__ = ["RuntimeProgressSnapshot", "build_runtime_progress_message", "format_duration"]
