"""Progress formatting helpers for explicit time integration."""

from __future__ import annotations

import math
import time

__all__ = ["_emit_time_progress", "_format_wall_time"]


def _format_wall_time(seconds: float) -> str:
    seconds_i = max(int(round(seconds)), 0)
    minutes, secs = divmod(seconds_i, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _emit_time_progress(
    *,
    step: int,
    total_steps: int,
    t: float,
    t_max: float,
    started_at: float,
    phi_max: float,
) -> None:
    elapsed = max(time.perf_counter() - started_at, 0.0)
    rate = step / elapsed if elapsed > 1.0e-12 else 0.0
    remaining = max(total_steps - step, 0)
    eta = remaining / rate if rate > 1.0e-12 else math.inf
    eta_text = "--:--" if not math.isfinite(eta) else _format_wall_time(eta)
    pct = 100.0 * step / max(total_steps, 1)
    print(
        "[spectrax-gk] "
        f"step={step}/{total_steps} progress={pct:5.1f}% "
        f"t={t:.6g}/{t_max:.6g} elapsed={_format_wall_time(elapsed)} "
        f"eta={eta_text} |phi|max={phi_max:.6e}",
        flush=True,
    )


