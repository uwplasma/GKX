"""Benchmark scan batching and streaming-window helper policies."""

from __future__ import annotations

import numpy as np


__all__ = ["_is_array_like", "_iter_ky_batches", "_resolve_streaming_window"]


def _is_array_like(value) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _iter_ky_batches(
    ky_values: np.ndarray,
    *,
    ky_batch: int,
    fixed_batch_shape: bool,
):
    """Yield ky batches with optional edge padding for fixed-shape compilation."""

    n = int(len(ky_values))
    if ky_batch <= 1:
        for idx in range(n):
            ky = float(ky_values[idx])
            yield idx, np.asarray([ky], dtype=float), 1
        return
    for start in range(0, n, ky_batch):
        raw = np.asarray(ky_values[start : start + ky_batch], dtype=float)
        valid = int(raw.size)
        if valid == 0:
            continue
        if fixed_batch_shape and valid < ky_batch:
            pad = np.full((ky_batch - valid,), raw[-1], dtype=float)
            batch = np.concatenate([raw, pad], axis=0)
        else:
            batch = raw
        yield start, batch, valid


def _resolve_streaming_window(
    t_total: float,
    tmin: float | None,
    tmax: float | None,
    start_fraction: float,
    window_fraction: float,
    end_fraction: float,
) -> tuple[float, float]:
    if tmin is not None and tmax is not None:
        return float(tmin), float(tmax)
    t_start = float(start_fraction) * t_total
    t_end = float(end_fraction) * t_total
    t_end = min(t_end, t_start + float(window_fraction) * t_total)
    if t_end <= t_start:
        t_end = t_total
    return t_start, t_end
