"""Run-to-saturation stop policy for nonlinear transport runs.

The decision this module makes is when a run has produced a heat-flux mean
worth stopping on: spin-up removed, the correlation time resolved inside the
retained window, the correlated standard error below a relative threshold, and
both halves of the window agreeing with each other -- optionally with a guard
trace, the field energy, held to the same stationarity so a flat-looking flux
cannot end a run whose fields are still drifting.

It sits apart from ``transport_windows.py`` because the question is different.
That module scores a window after the fact against promotion gates; this one is
asked the same question repeatedly by a running solver, on a trace that grows
under it, and has to answer "not yet" with a reason the runtime can record.

Fixed-step traces retain the point-sample estimator shared with post-hoc
analysis. Adaptive runtime traces use physical interval means and their
estimated covariance, requiring both estimators to accept the same prefix.
Neither gate certifies stationarity or nominal sequential confidence coverage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence
import math

import numpy as np

from gkx.diagnostics.analysis import sokal_autocorrelation_time


@dataclass(frozen=True)
class SaturationStopConfig:
    """Stop policy for ``run_to = "saturation"`` nonlinear runs."""

    rel_sem: float = 0.05
    # Optional absolute floor on the averaging-window span. Every accepted
    # window must also span at least twenty integrated autocorrelation times.
    min_window: float | None = None
    min_samples: int = 256


_SATURATION_VALUE_FLOOR = 1.0e-12
# This absolute threshold only catches dead traces at its scale. The resolved
# correlation-time gate below supplies the scale-independent protection.
_SATURATION_SIGNAL_FACTOR = 1.0e3
_MIN_RETAINED_SAMPLES = 256
_MIN_IAT_SPANS = 20.0
_SATURATION_DECISION_FIELDS = (
    "window_tmin",
    "window_tmax",
    "window_span",
    "mean",
    "sem",
    "rel_sem",
    "tau_ac",
    "tau_ac_resolved",
    "min_samples",
    "min_iat_spans",
    "min_window",
    "first_half_mean",
    "second_half_mean",
    "halves_sem",
    "stationary",
    "guard_stationary",
    "Wg_guard_stationary",
)


_WindowStatistics = Callable[[np.ndarray, float], tuple[float, float, float, bool]]


def _sokal_window_mean_sem(
    values: np.ndarray, dt: float
) -> tuple[float, float, float, bool]:
    """Windowed mean and IAT-corrected SEM ``std / sqrt(n_eff)``.

    The naive SEM understates the error of a correlated mean by
    ``sqrt(n / n_eff)``. ``n_eff = n dt / (2 tau_ac)``, capped at ``n``, is the
    convention already validated for GKX transport windows in
    ``gkx.diagnostics.analysis._correlated_sample_stats``; the stop test has to
    use it so the SEM a run stops on is the same number the post-hoc window
    gates report for that window. The alternative ``n / (1 + 2 tau_ac / dt)``
    double-counts the zero-lag term and returns ``n/2`` for independent
    samples, inflating the SEM by 41% exactly where a run is closest to being
    allowed to stop.
    """

    tau, cut, rho = sokal_autocorrelation_time(values, dt)
    # A correlation time at or below one sample is the discretization floor,
    # not resolved physics. Positive lag-one noise made 190/400 flat traces
    # stop before this gate; their tau never exceeded 0.883 dt, while measured
    # deck traces span 8.5--81 dt.
    resolved = cut < rho.size and tau > dt
    n_eff = (
        min(float(values.size), values.size * dt / (2.0 * tau))
        if tau > 0.0 and dt > 0.0
        else float(values.size)
    )
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(max(n_eff, 1.0)))
    return mean, sem, tau, resolved


def _empty_saturation_decision(
    cfg: SaturationStopConfig, *, reason: str
) -> dict[str, Any]:
    decision: dict[str, Any] = {field: None for field in _SATURATION_DECISION_FIELDS}
    decision.update(
        kind="nonlinear_saturation_stop_decision",
        saturated=False,
        reasons=[reason],
        n_window=0,
        min_samples=max(int(cfg.min_samples), _MIN_RETAINED_SAMPLES),
        min_iat_spans=_MIN_IAT_SPANS,
        config=asdict(cfg),
    )
    return decision


# Two combined SEMs: a 1-sigma agreement gate would reject roughly a third of
# genuinely stationary windows and only prolong runs until they pass by luck.
_STATIONARITY_SEM_FACTOR = 2.0


def _halves_stationary(
    values: np.ndarray,
    dt: float,
    *,
    check_terminal: bool = True,
    statistics: _WindowStatistics | None = None,
) -> tuple[float, float, float, bool]:
    """Half-window means, combined SEM, and agreement including the late half."""

    half = values.size // 2
    stats = statistics or _sokal_window_mean_sem
    first_mean, first_sem, _, _ = stats(values[:half], dt)
    second_mean, second_sem, _, _ = stats(values[half:], dt)
    combined = float(math.hypot(first_sem, second_sem))
    stationary = abs(second_mean - first_mean) <= _STATIONARITY_SEM_FACTOR * combined
    if check_terminal and stationary:
        # Opposing startup decay and late growth can give equal full-window
        # halves. Also compare the terminal quarters on the same SEM scale.
        stationary = _halves_stationary(
            values[half:], dt, check_terminal=False, statistics=stats
        )[3]
    return first_mean, second_mean, combined, stationary


def _saturation_config(config: SaturationStopConfig | None) -> SaturationStopConfig:
    cfg = config or SaturationStopConfig()
    if float(cfg.rel_sem) <= 0.0:
        raise ValueError("rel_sem must be positive")
    if cfg.min_window is not None and (
        not math.isfinite(float(cfg.min_window)) or float(cfg.min_window) < 0.0
    ):
        raise ValueError("min_window must be finite and non-negative when supplied")
    return cfg


def saturation_stop_decision(
    time: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    guard: Sequence[float] | np.ndarray | None = None,
    free_energy_guard: Sequence[float] | np.ndarray | None = None,
    config: SaturationStopConfig | None = None,
) -> dict[str, Any]:
    """Decide whether a nonlinear trace has saturated well enough to stop.

    Spin-up removal: everything before the trace first reaches its own median
    is discarded. During the linear-growth phase the flux sits below its
    saturated level, so once any plateau exists the overall median lies on the
    plateau and the first crossing lands at the end of growth; the overshoot
    decay that follows is left to the stationarity gate rather than trimmed.

    Saturation requires all of: a resolved ``tau_ac`` (the autocorrelation
    crosses zero inside the window, and not at the first lag -- a trace that
    decorrelates within one diagnostic sample has shown no correlation time,
    only noise), at least ``min_samples`` retained samples, window span at
    least ``20 tau_ac`` and any larger configured ``min_window``, IAT-corrected
    relative SEM at most ``rel_sem``,
    first/second half-window means within twice their combined SEM, the same
    agreement between the third and fourth quarters, and --
    when ``guard`` (Wphi) or ``free_energy_guard`` (Wg) is given -- the same
    full-window and terminal-half stationarity on each guard. Guards have no
    relative-SEM gate: they protect against a flat-looking flux while either
    energy still drifts.
    The trace is assumed finite; the runtime validates each chunk before this.
    """

    return _saturation_decision(
        time,
        values,
        guard=guard,
        free_energy_guard=free_energy_guard,
        config=config,
        statistics=_sokal_window_mean_sem,
    )


def _saturation_decision(
    time: Any,
    values: Any,
    *,
    guard: Any,
    free_energy_guard: Any,
    config: SaturationStopConfig | None,
    statistics: _WindowStatistics,
) -> dict[str, Any]:
    cfg = _saturation_config(config)
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if t.size != y.size:
        raise ValueError("time and values must have the same length")
    g = None if guard is None else np.asarray(guard, dtype=float).reshape(-1)
    if g is not None and g.size != t.size:
        raise ValueError("guard must match the time axis")
    wg = (
        None
        if free_energy_guard is None
        else np.asarray(free_energy_guard, float).ravel()
    )
    if wg is not None and wg.size != t.size:
        raise ValueError("free_energy_guard must match the time axis")
    min_samples = max(int(cfg.min_samples), _MIN_RETAINED_SAMPLES)
    if t.size < min_samples:
        return _empty_saturation_decision(cfg, reason="trace_shorter_than_min_samples")
    start = int(np.argmax(y >= np.median(y)))
    wt = t[start:]
    wy = y[start:]
    if wy.size < min_samples:
        return _empty_saturation_decision(cfg, reason="post_spinup_window_too_short")
    intervals = np.diff(wt)
    if not np.all(np.isfinite(intervals)) or np.any(intervals <= 0.0):
        return _empty_saturation_decision(cfg, reason="degenerate_window_time_axis")
    dt = float(np.median(intervals))

    mean, sem, tau, tau_resolved = statistics(wy, dt)
    span = float(wt[-1] - wt[0])
    min_window = max(
        _MIN_IAT_SPANS * tau,
        0.0 if cfg.min_window is None else float(cfg.min_window),
    )
    rel_sem = float(sem / max(abs(mean), _SATURATION_VALUE_FLOOR))
    first_mean, second_mean, halves_sem, stationary = _halves_stationary(
        wy, dt, statistics=statistics
    )
    guard_stationary = (
        None
        if g is None
        else _halves_stationary(g[start:], dt, statistics=statistics)[3]
    )
    wg_stationary = (
        None
        if wg is None
        else _halves_stationary(wg[start:], dt, statistics=statistics)[3]
    )
    # A trace that never left zero has nothing to converge: the relative SEM
    # divides by a floor rather than a mean, so every gate below passes on a
    # dead signal and the run stops in its first chunk. A zonal-response case,
    # whose heat flux is identically zero by construction, hit exactly that.
    # Requiring a flux the floor cannot explain leaves such a run to t_max,
    # which is the only honest answer when there is no saturated mean to find.
    signal_present = abs(mean) > _SATURATION_SIGNAL_FACTOR * _SATURATION_VALUE_FLOOR
    gates = {
        "flux_indistinguishable_from_zero": bool(signal_present),
        "tau_ac_unresolved": bool(tau_resolved),
        "window_below_min_window": span >= min_window,
        "rel_sem_above_threshold": rel_sem <= float(cfg.rel_sem),
        "window_not_stationary": bool(stationary),
        "guard_not_stationary": guard_stationary is None or bool(guard_stationary),
        "Wg_guard_not_stationary": wg_stationary is None or bool(wg_stationary),
    }
    return {
        "kind": "nonlinear_saturation_stop_decision",
        "saturated": all(gates.values()),
        "reasons": [name for name, passed in gates.items() if not passed],
        "n_window": int(wy.size),
        "window_tmin": float(wt[0]),
        "window_tmax": float(wt[-1]),
        "window_span": span,
        "mean": mean,
        "sem": sem,
        "rel_sem": rel_sem,
        "tau_ac": float(tau),
        "tau_ac_resolved": bool(tau_resolved),
        "min_samples": min_samples,
        "min_iat_spans": _MIN_IAT_SPANS,
        "min_window": float(min_window),
        "first_half_mean": first_mean,
        "second_half_mean": second_mean,
        "halves_sem": halves_sem,
        "stationary": bool(stationary),
        "guard_stationary": guard_stationary,
        "Wg_guard_stationary": wg_stationary,
        "config": asdict(cfg),
    }


_MAX_PHYSICAL_BINS = 4096


def _batch_window_mean_sem(
    values: np.ndarray, dt: float
) -> tuple[float, float, float, bool]:
    """Estimated covariance of interval means, including residual batch correlation."""
    _, cut, rho = sokal_autocorrelation_time(values, dt)
    inflation = max(1.0, 1.0 + 2.0 * float(np.sum(rho[1:cut])))
    n_eff = len(values) / inflation
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1) / np.sqrt(n_eff)),
        0.5 * dt * inflation,
        cut < len(rho),
    )


def _physical_time_bins(
    time: Any, channels: Sequence[Any]
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], dict[str, Any]
]:
    """Integrate full observed support in O(N) time and bounded extra workspace."""
    t = np.asarray(time)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("need two clock nodes")
    if len(channels) != 3 or any(np.size(c) != len(t) for c in channels):
        raise ValueError("incomplete transport histories")
    origin = float(t[0])
    span = float(t[-1]) - origin
    n = min(_MAX_PHYSICAL_BINS, len(t) - 1)
    if not np.isfinite(span) or span <= 0:
        raise ValueError("invalid physical clock span")
    edges = np.linspace(0.0, span, n + 1)
    integrals = np.zeros((n + 1, 3))
    left_count = np.zeros(n + 1, dtype=int)
    right_count = np.zeros(n + 1, dtype=int)
    baseline = np.array([c[0] for c in channels], dtype=float)
    running = np.zeros(3)
    tol = 8 * np.finfo(float).eps * max(span, span / n)
    for start in range(0, len(t) - 1, _MAX_PHYSICAL_BINS):
        end = min(len(t), start + _MAX_PHYSICAL_BINS + 1)
        x = np.asarray(t[start:end], dtype=float) - origin
        y = (
            np.column_stack([np.asarray(c[start:end], dtype=float) for c in channels])
            - baseline
        )
        dx = np.diff(x)
        if not np.isfinite(x).all() or np.any(dx <= 0) or not np.isfinite(y).all():
            raise ValueError("invalid physical clock or channel")
        cumulative = np.vstack(
            (np.zeros(3), np.cumsum(0.5 * (y[1:] + y[:-1]) * dx[:, None], axis=0))
        )
        a = np.searchsorted(edges, x[0], side="left")
        b = np.searchsorted(edges, x[-1], side="right")
        e = edges[a:b]
        ix = np.clip(np.searchsorted(x, e, side="right") - 1, 0, len(x) - 2)
        offset = e - x[ix]
        integrals[a:b] = (
            running
            + cumulative[ix]
            + y[ix] * offset[:, None]
            + 0.5 * (np.diff(y, axis=0) / dx[:, None])[ix] * offset[:, None] ** 2
        )
        left_count[a:b] = start + np.searchsorted(x, e - tol, side="left")
        right_count[a:b] = start + np.searchsorted(x, e + tol, side="right")
        running += cumulative[-1]
    values = np.diff(integrals, axis=0) / np.diff(edges)[:, None] + baseline
    meta = dict(
        kind="physical_time_batch_covariance",
        origin=origin,
        cadence=float(span / n),
        max_bins=_MAX_PHYSICAL_BINS,
        n_raw_total=len(t),
        n_complete_total=n,
        partial_bins="none; full observed support",
        n_window_unit="complete_bins",
        clock_dtype=str(t.dtype),
        covariance_estimator="first_negative_discrete_sum",
        resolution_gate="native_same_prefix",
        correlation_floor="variance inflation at least one",
    )
    return (
        0.5 * (edges[:-1] + edges[1:]),
        values,
        edges,
        (left_count, right_count),
        meta,
    )


def _physical_saturation_stop_decision(
    time: Any, q: Any, w: Any, g: Any, *, config: SaturationStopConfig | None = None
) -> dict[str, Any]:
    """Admit physical-batch estimates only with native same-prefix acceptance.

    Report bin edges while preserving the center-span minimum-window gate.
    Covariance estimates and repeated stopping remain calibration dependent.
    """
    cfg = _saturation_config(config)
    for name, trace in (("values", q), ("guard", w), ("free_energy_guard", g)):
        if trace is None or np.size(trace) != np.size(time):
            raise ValueError(f"{name} must match the time axis")
    try:
        centers, bins, edges, counts, meta = _physical_time_bins(time, (q, w, g))
    except ValueError as exc:
        return _empty_saturation_decision(cfg, reason=str(exc))
    d = _saturation_decision(
        centers,
        bins[:, 0],
        guard=bins[:, 1],
        free_energy_guard=bins[:, 2],
        config=cfg,
        statistics=_batch_window_mean_sem,
    )
    d["analysis"] = meta
    meta.update(native_gate="not_evaluated", n_raw_window=0, n_eff=None)
    if d["mean"] is None:
        return d
    first = int(np.searchsorted(centers, d["window_tmin"]))
    raw = int(counts[1][-1] - counts[0][first])
    reasons = d["reasons"]
    meta.update(
        center_span=d["window_span"],
        center_min_window=d["min_window"],
        n_raw_window=raw,
        n_eff=min(
            float(d["n_window"]), d["n_window"] * meta["cadence"] / (2 * d["tau_ac"])
        ),
    )
    if raw < d["min_samples"]:
        reasons.append("raw_effort_below_minimum")
    if not reasons:
        native = saturation_stop_decision(
            time, q, guard=w, free_energy_guard=g, config=cfg
        )
        meta["native_gate"] = "passed" if native["saturated"] else "failed"
        if not native["saturated"]:
            reasons.append("sample_clock_crosscheck_failed")
    d.update(
        window_tmin=meta["origin"] + float(edges[first]),
        window_tmax=meta["origin"] + float(edges[-1]),
        window_span=float(edges[-1] - edges[first]),
        min_window=float(d["min_window"] + meta["cadence"]),
        saturated=not reasons,
    )
    return d


__all__ = [
    "SaturationStopConfig",
    "saturation_stop_decision",
    "sokal_autocorrelation_time",
]
