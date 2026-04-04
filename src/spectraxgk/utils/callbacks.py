"""Host-side progress callbacks used by JAX-integrated loops."""

from __future__ import annotations

import math
import time
from typing import Any

import jax
import jax.numpy as jnp


_PROGRESS_START: dict[str, float] = {}
_TARGET_PROGRESS_UPDATES = 50


def _format_duration(seconds: float) -> str:
    seconds_i = max(int(round(seconds)), 0)
    minutes, secs = divmod(seconds_i, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_update_stride(steps: int, *, target_updates: int = _TARGET_PROGRESS_UPDATES) -> int:
    """Return a bounded host-callback cadence for long integrations."""

    steps_i = max(int(steps), 1)
    target_i = max(int(target_updates), 1)
    return max((steps_i + target_i - 1) // target_i, 1)


def should_emit_progress(idx: Any, steps: Any, *, target_updates: int = _TARGET_PROGRESS_UPDATES) -> Any:
    """Return a JAX-friendly predicate for whether a progress update should fire."""

    idx_i = jnp.asarray(idx, dtype=jnp.int32)
    steps_i = jnp.maximum(jnp.asarray(steps, dtype=jnp.int32), 1)
    step_i = idx_i + 1
    target_i = jnp.asarray(max(int(target_updates), 1), dtype=jnp.int32)
    stride = jnp.maximum((steps_i + target_i - 1) // target_i, 1)
    return (step_i == 1) | (step_i == steps_i) | ((step_i % stride) == 0)


def _emit_progress(
    idx: Any,
    steps: Any,
    gamma: Any,
    omega: Any,
    wphi: Any,
    wg: Any,
    sim_time: Any = None,
    sim_total: Any = None,
    metric_labels: tuple[str, str] = ("Wphi", "Wg"),
) -> None:
    idx_i = int(idx)
    steps_i = max(int(steps), 1)
    gamma_f = float(gamma)
    omega_f = float(omega)
    wphi_f = float(wphi)
    wg_f = float(wg)
    pct = 100.0 * float(idx_i + 1) / float(steps_i)
    key = f"steps={steps_i}"
    now = time.perf_counter()
    if idx_i <= 0 or key not in _PROGRESS_START:
        _PROGRESS_START[key] = now
    elapsed = max(now - _PROGRESS_START[key], 0.0)
    rate = float(idx_i + 1) / elapsed if elapsed > 1.0e-12 else 0.0
    remaining_steps = max(steps_i - (idx_i + 1), 0)
    eta = remaining_steps / rate if rate > 1.0e-12 else float("inf")
    sim_text = ""
    if sim_time is not None:
        sim_time_f = float(sim_time)
        if sim_total is not None:
            sim_total_f = float(sim_total)
            if math.isfinite(sim_total_f) and sim_total_f > 0.0:
                sim_text = f" t={sim_time_f:.6g}/{sim_total_f:.6g}"
            else:
                sim_text = f" t={sim_time_f:.6g}"
        else:
            sim_text = f" t={sim_time_f:.6g}"
    print(
        f"[spectrax-gk] step={idx_i + 1}/{steps_i} "
        f"progress={pct:5.1f}%{sim_text} elapsed={_format_duration(elapsed)} "
        f"eta={_format_duration(eta) if eta != float('inf') else '--:--'} "
        f"gamma={gamma_f:.6g} omega={omega_f:.6g} "
        f"{metric_labels[0]}={wphi_f:.6g} {metric_labels[1]}={wg_f:.6g}"
    )
    if idx_i + 1 >= steps_i:
        _PROGRESS_START.pop(key, None)


def print_callback(
    state: Any,
    idx: Any,
    steps: Any,
    gamma: Any,
    omega: Any,
    wphi: Any,
    wg: Any,
    sim_time: Any = None,
    sim_total: Any = None,
    metric_labels: tuple[str, str] = ("Wphi", "Wg"),
) -> Any:
    """Emit a host-side progress update and return ``state`` unchanged."""

    jax.debug.callback(
        lambda *args: _emit_progress(*args, metric_labels=metric_labels),
        idx,
        steps,
        gamma,
        omega,
        wphi,
        wg,
        sim_time,
        sim_total,
    )
    return state
