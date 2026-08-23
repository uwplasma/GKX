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

The Sokal estimator is shared with post-hoc analysis, so runtime and campaign
uncertainties use one definition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence
import math

import numpy as np

from gkx.diagnostics.analysis import sokal_autocorrelation_time


@dataclass(frozen=True)
class SaturationStopConfig:
    """Stop policy for ``run_to = "saturation"`` nonlinear runs."""

    rel_sem: float = 0.05
    # Minimum averaging-window span in time units. None derives it from the
    # trace itself as ten integrated autocorrelation times.
    min_window: float | None = None
    min_samples: int = 16


_SATURATION_VALUE_FLOOR = 1.0e-12
# How far above the floor a mean must sit before the relative SEM built on it
# means anything. Generous, because the cost of waiting is wall time and the
# cost of stopping early is a silently truncated run.
#
# This threshold is absolute, and a heat flux is not: its scale is set by the
# initial amplitude the deck seeds and by the diagnostic normalization it
# picks. So it covers a dead trace only when that trace happens to sit below
# it, and cannot be the whole protection against one. What a dead trace shows
# in every normalization is that it has no correlation time this sampling can
# resolve, which is the gate that actually carries the case; see the
# ``resolved`` definition in ``_sokal_window_mean_sem``.
_SATURATION_SIGNAL_FACTOR = 1.0e3
_SATURATION_DECISION_FIELDS = (
    "window_tmin",
    "window_tmax",
    "window_span",
    "mean",
    "sem",
    "rel_sem",
    "tau_ac",
    "tau_ac_resolved",
    "min_window",
    "first_half_mean",
    "second_half_mean",
    "halves_sem",
    "stationary",
    "guard_stationary",
    "Wg_guard_stationary",
)


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
    # Resolved means the trace showed this sampling a correlation time: the
    # autocorrelation came back inside the window, and the time it integrates
    # to is longer than the interval it was sampled at. A correlation time
    # shorter than one sample is not a measurement of anything -- it is the
    # discretization floor, and it is what uncorrelated noise returns.
    #
    # Both halves of that matter, and for the same reason: ``min_window`` is
    # derived as ``10 tau``, so a ``tau`` at the floor makes the window-length
    # requirement vacuous exactly on the traces carrying the least information.
    # Requiring only ``tau > 0`` is not enough, because the lag-one sample
    # autocorrelation of white noise is positive about half the time: measured
    # over 400 realizations of a flat trace, 190 produced a positive ``tau``
    # and saturated, at any amplitude -- scaling a trace cannot change its
    # autocorrelation. Across those same 400, ``tau`` never exceeded
    # ``0.883 dt``, while the shipped nonlinear decks measure ``tau`` between
    # ``8.5 dt`` and ``81 dt``. One sampling interval sits in that gap with an
    # order of magnitude of room on the physical side.
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
        config=asdict(cfg),
    )
    return decision


# Two combined SEMs: a 1-sigma agreement gate would reject roughly a third of
# genuinely stationary windows and only prolong runs until they pass by luck.
_STATIONARITY_SEM_FACTOR = 2.0


def _halves_stationary(
    values: np.ndarray, dt: float
) -> tuple[float, float, float, bool]:
    """First/second half-window means, their combined SEM, and agreement."""

    half = values.size // 2
    first_mean, first_sem, _, _ = _sokal_window_mean_sem(values[:half], dt)
    second_mean, second_sem, _, _ = _sokal_window_mean_sem(values[half:], dt)
    combined = float(math.hypot(first_sem, second_sem))
    stationary = abs(second_mean - first_mean) <= _STATIONARITY_SEM_FACTOR * combined
    return first_mean, second_mean, combined, stationary


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
    only noise), window span at least ``min_window``
    (default ``10 tau_ac``), IAT-corrected relative SEM at most ``rel_sem``,
    first/second half-window means within twice their combined SEM, and --
    when ``guard`` (Wphi) or ``free_energy_guard`` (Wg) is given -- the same
    half-window stationarity on each guard over the same window. Guards have no
    relative-SEM gate: they protect against a flat-looking flux while either
    energy still drifts.
    The trace is assumed finite; the runtime validates each chunk before this.
    """

    cfg = config or SaturationStopConfig()
    if float(cfg.rel_sem) <= 0.0:
        raise ValueError("rel_sem must be positive")
    if cfg.min_window is not None and float(cfg.min_window) < 0.0:
        raise ValueError("min_window must be non-negative when supplied")
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
    min_samples = max(int(cfg.min_samples), 8)
    if t.size < min_samples:
        return _empty_saturation_decision(cfg, reason="trace_shorter_than_min_samples")
    start = int(np.argmax(y >= np.median(y)))
    wt = t[start:]
    wy = y[start:]
    if wy.size < min_samples:
        return _empty_saturation_decision(cfg, reason="post_spinup_window_too_short")
    dt = float(np.median(np.diff(wt)))
    if not math.isfinite(dt) or dt <= 0.0:
        return _empty_saturation_decision(cfg, reason="degenerate_window_time_axis")

    mean, sem, tau, tau_resolved = _sokal_window_mean_sem(wy, dt)
    span = float(wt[-1] - wt[0])
    min_window = 10.0 * tau if cfg.min_window is None else float(cfg.min_window)
    rel_sem = float(sem / max(abs(mean), _SATURATION_VALUE_FLOOR))
    first_mean, second_mean, halves_sem, stationary = _halves_stationary(wy, dt)
    guard_stationary = None if g is None else _halves_stationary(g[start:], dt)[3]
    wg_stationary = None if wg is None else _halves_stationary(wg[start:], dt)[3]
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
        "min_window": float(min_window),
        "first_half_mean": first_mean,
        "second_half_mean": second_mean,
        "halves_sem": halves_sem,
        "stationary": bool(stationary),
        "guard_stationary": guard_stationary,
        "Wg_guard_stationary": wg_stationary,
        "config": asdict(cfg),
    }


__all__ = [
    "SaturationStopConfig",
    "saturation_stop_decision",
    "sokal_autocorrelation_time",
]
