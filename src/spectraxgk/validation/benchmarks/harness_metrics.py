"""Physics metric extractors for benchmark and validation traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectraxgk.diagnostics.analysis import extract_mode_time_series, fit_growth_rate
from spectraxgk.validation.benchmarks.harness_timeseries import (
    _tail_stats,
    _tail_window,
)
from spectraxgk.validation.gates import (
    BranchContinuationMetrics,
    LateTimeLinearMetrics,
    NonlinearHeatFluxConvergenceMetrics,
    NonlinearWindowMetrics,
    ObservedOrderMetrics,
)
from spectraxgk.validation.benchmarks.harness_zonal_metrics import (
    zonal_flow_response_metrics,
)


@dataclass(frozen=True)
class _HeatFluxWindow:
    t: np.ndarray
    q: np.ndarray
    tmin: float | None
    tmax: float | None


@dataclass(frozen=True)
class _HeatFluxConvergenceSummary:
    mean: float
    std: float
    cv: float
    rms: float
    terminal_mean: float
    mean_rel_delta: float
    trend: float


def late_time_linear_metrics(
    result: object,
    *,
    tail_fraction: float = 0.5,
    mode_method: str = "project",
) -> LateTimeLinearMetrics:
    """Return late-time growth/frequency metrics from a linear benchmark/runtime result."""

    t = getattr(result, "t", None)
    if t is None:
        gamma = float(getattr(result, "gamma"))
        omega = float(getattr(result, "omega"))
        return LateTimeLinearMetrics(
            gamma_fit=gamma,
            omega_fit=omega,
            gamma_tail_mean=gamma,
            omega_tail_mean=omega,
            gamma_tail_std=0.0,
            omega_tail_std=0.0,
            tmin=None,
            tmax=None,
            nsamples=1,
            signal_source="scalar",
        )

    t_arr = np.asarray(t, dtype=float)
    mask, tmin, tmax = _tail_window(t_arr, tail_fraction)

    gamma_series = getattr(result, "gamma_t", None)
    omega_series = getattr(result, "omega_t", None)
    signal_source = "scalar"
    gamma_fit = float(getattr(result, "gamma"))
    omega_fit = float(getattr(result, "omega"))

    signal = getattr(result, "signal", None)
    if signal is not None:
        signal_arr = np.asarray(signal, dtype=np.complex128)
        signal_source = "signal"
    elif hasattr(result, "phi_t") and hasattr(result, "selection"):
        signal_arr = np.asarray(
            extract_mode_time_series(
                np.asarray(getattr(result, "phi_t")),
                getattr(result, "selection"),
                method=mode_method,
            ),
            dtype=np.complex128,
        )
        signal_source = f"phi_t:{mode_method}"
    else:
        signal_arr = None

    if signal_arr is not None:
        finite = np.isfinite(signal_arr)
        signal_tail = signal_arr[mask & finite]
        t_tail = t_arr[mask & finite]
        if t_tail.size >= 2:
            gamma_fit, omega_fit = fit_growth_rate(t_tail, signal_tail)

    if gamma_series is not None:
        gamma_mean, gamma_std = _tail_stats(np.asarray(gamma_series), mask)
    else:
        gamma_mean, gamma_std = gamma_fit, 0.0
    if omega_series is not None:
        omega_mean, omega_std = _tail_stats(np.asarray(omega_series), mask)
    else:
        omega_mean, omega_std = omega_fit, 0.0

    nsamples = int(np.count_nonzero(mask))
    return LateTimeLinearMetrics(
        gamma_fit=float(gamma_fit),
        omega_fit=float(omega_fit),
        gamma_tail_mean=float(gamma_mean),
        omega_tail_mean=float(omega_mean),
        gamma_tail_std=float(gamma_std),
        omega_tail_std=float(omega_std),
        tmin=tmin,
        tmax=tmax,
        nsamples=nsamples,
        signal_source=signal_source,
    )


def windowed_nonlinear_metrics(
    result: object,
    *,
    start_fraction: float = 0.5,
) -> NonlinearWindowMetrics:
    """Return late-window transport and envelope metrics from a nonlinear runtime result."""

    diagnostics = getattr(result, "diagnostics", result)
    if diagnostics is None:
        raise ValueError("nonlinear diagnostics are required")
    if not 0.0 <= float(start_fraction) < 1.0:
        raise ValueError("start_fraction must be in [0, 1)")
    t = np.asarray(getattr(diagnostics, "t", None), dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("diagnostics.t must be a non-empty one-dimensional array")
    tail_fraction = max(np.finfo(float).eps, 1.0 - float(start_fraction))
    mask, tmin, tmax = _tail_window(t, tail_fraction)
    heat_flux = np.asarray(getattr(diagnostics, "heat_flux_t"), dtype=float)[mask]
    wphi = np.asarray(getattr(diagnostics, "Wphi_t"), dtype=float)[mask]
    wg = np.asarray(getattr(diagnostics, "Wg_t"), dtype=float)[mask]
    heat_flux = heat_flux[np.isfinite(heat_flux)]
    wphi = wphi[np.isfinite(wphi)]
    wg = wg[np.isfinite(wg)]
    if heat_flux.size == 0 or wphi.size == 0 or wg.size == 0:
        raise ValueError(
            "windowed diagnostics must contain finite heat/Wphi/Wg samples"
        )

    phi_mode = getattr(diagnostics, "phi_mode_t", None)
    envelope_mean: float | None = None
    envelope_std: float | None = None
    envelope_max: float | None = None
    if phi_mode is not None:
        envelope = np.abs(np.asarray(phi_mode)[mask])
        envelope = envelope[np.isfinite(envelope)]
        if envelope.size:
            envelope_mean = float(np.mean(envelope))
            envelope_std = float(np.std(envelope))
            envelope_max = float(np.max(envelope))

    return NonlinearWindowMetrics(
        tmin=float(tmin if tmin is not None else t[0]),
        tmax=float(tmax if tmax is not None else t[-1]),
        nsamples=int(np.count_nonzero(mask)),
        heat_flux_mean=float(np.mean(heat_flux)),
        heat_flux_std=float(np.std(heat_flux)),
        heat_flux_rms=float(np.sqrt(np.mean(np.square(heat_flux)))),
        wphi_mean=float(np.mean(wphi)),
        wphi_std=float(np.std(wphi)),
        wg_mean=float(np.mean(wg)),
        wg_std=float(np.std(wg)),
        phi_mode_envelope_mean=envelope_mean,
        phi_mode_envelope_std=envelope_std,
        phi_mode_envelope_max=envelope_max,
    )


def _validate_heat_flux_convergence_inputs(
    t: np.ndarray,
    heat_flux: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t_arr = np.asarray(t, dtype=float)
    q_arr = np.asarray(heat_flux, dtype=float)
    if t_arr.ndim != 1 or q_arr.ndim != 1 or t_arr.size != q_arr.size:
        raise ValueError(
            "t and heat_flux must be one-dimensional arrays of equal length"
        )
    if t_arr.size == 0:
        raise ValueError("t and heat_flux must be non-empty")

    finite = np.isfinite(t_arr) & np.isfinite(q_arr)
    t_arr = t_arr[finite]
    q_arr = q_arr[finite]
    if t_arr.size == 0:
        raise ValueError(
            "t and heat_flux must contain at least one finite paired sample"
        )
    if t_arr.size > 1 and np.any(np.diff(t_arr) <= 0.0):
        raise ValueError("t must be strictly increasing after finite-sample filtering")
    return t_arr, q_arr


def _validate_heat_flux_convergence_options(
    *,
    start_fraction: float,
    terminal_fraction: float,
    mean_floor: float,
) -> tuple[float, float, float]:
    start = float(start_fraction)
    terminal = float(terminal_fraction)
    floor = float(mean_floor)
    if not 0.0 <= start < 1.0:
        raise ValueError("start_fraction must be in [0, 1)")
    if not 0.0 < terminal <= 1.0:
        raise ValueError("terminal_fraction must be in (0, 1]")
    if floor < 0.0:
        raise ValueError("mean_floor must be non-negative")
    return start, terminal, floor


def _post_transient_heat_flux_window(
    t_arr: np.ndarray,
    q_arr: np.ndarray,
    *,
    start_fraction: float,
) -> _HeatFluxWindow:
    tail_fraction = max(np.finfo(float).eps, 1.0 - start_fraction)
    mask, tmin, tmax = _tail_window(t_arr, tail_fraction)
    t_win = t_arr[mask]
    q_win = q_arr[mask]
    if q_win.size == 0:
        raise ValueError("post-transient heat-flux window is empty")
    return _HeatFluxWindow(t=t_win, q=q_win, tmin=tmin, tmax=tmax)


def _terminal_heat_flux_window(
    window: _HeatFluxWindow,
    *,
    terminal_fraction: float,
) -> _HeatFluxWindow:
    terminal_start = max(
        0, int(np.floor((1.0 - terminal_fraction) * window.q.size))
    )
    t_terminal = window.t[terminal_start:]
    q_terminal = window.q[terminal_start:]
    if q_terminal.size == 0:
        raise ValueError("terminal heat-flux window is empty")
    return _HeatFluxWindow(
        t=t_terminal,
        q=q_terminal,
        tmin=float(t_terminal[0]),
        tmax=float(t_terminal[-1]),
    )


def _heat_flux_window_trend(
    window: _HeatFluxWindow,
    *,
    scale: float,
) -> float:
    if window.t.size < 2 or float(window.t[-1] - window.t[0]) <= 0.0:
        return 0.0
    slope, _offset = np.polyfit(window.t, window.q, 1)
    return (
        float(slope * (window.t[-1] - window.t[0]) / scale)
        if scale > 0.0
        else float("inf")
    )


def _summarize_heat_flux_convergence(
    window: _HeatFluxWindow,
    terminal: _HeatFluxWindow,
    *,
    mean_floor: float,
) -> _HeatFluxConvergenceSummary:
    mean = float(np.mean(window.q))
    std = float(np.std(window.q))
    rms = float(np.sqrt(np.mean(np.square(window.q))))
    terminal_mean = float(np.mean(terminal.q))
    scale = max(abs(mean), mean_floor)
    cv = float(std / scale) if scale > 0.0 else float("inf")
    mean_rel_delta = (
        float(abs(terminal_mean - mean) / scale) if scale > 0.0 else float("inf")
    )
    trend = _heat_flux_window_trend(window, scale=scale)
    return _HeatFluxConvergenceSummary(
        mean=mean,
        std=std,
        cv=cv,
        rms=rms,
        terminal_mean=terminal_mean,
        mean_rel_delta=mean_rel_delta,
        trend=trend,
    )


def nonlinear_heat_flux_convergence_metrics(
    t: np.ndarray,
    heat_flux: np.ndarray,
    *,
    start_fraction: float = 0.5,
    terminal_fraction: float = 0.5,
    mean_floor: float = 1.0e-30,
) -> NonlinearHeatFluxConvergenceMetrics:
    """Summarize whether a post-transient heat-flux average is stable.

    ``start_fraction`` discards startup samples. ``terminal_fraction`` compares
    the retained post-transient mean with the final subwindow of that retained
    region. The normalized trend is the least-squares slope multiplied by the
    post-transient time span and divided by the absolute post-transient mean.
    """

    t_arr, q_arr = _validate_heat_flux_convergence_inputs(t, heat_flux)
    start, terminal_fraction, mean_floor = _validate_heat_flux_convergence_options(
        start_fraction=start_fraction,
        terminal_fraction=terminal_fraction,
        mean_floor=mean_floor,
    )
    window = _post_transient_heat_flux_window(
        t_arr,
        q_arr,
        start_fraction=start,
    )
    terminal = _terminal_heat_flux_window(
        window,
        terminal_fraction=terminal_fraction,
    )
    summary = _summarize_heat_flux_convergence(
        window,
        terminal,
        mean_floor=mean_floor,
    )

    return NonlinearHeatFluxConvergenceMetrics(
        tmin=float(window.tmin if window.tmin is not None else window.t[0]),
        tmax=float(window.tmax if window.tmax is not None else window.t[-1]),
        nsamples=int(window.q.size),
        heat_flux_mean=summary.mean,
        heat_flux_std=summary.std,
        heat_flux_cv=summary.cv,
        heat_flux_rms=summary.rms,
        terminal_tmin=float(terminal.t[0]),
        terminal_tmax=float(terminal.t[-1]),
        terminal_nsamples=int(terminal.q.size),
        terminal_heat_flux_mean=summary.terminal_mean,
        mean_rel_delta=summary.mean_rel_delta,
        trend=summary.trend,
        abs_trend=float(abs(summary.trend)),
        start_fraction=start,
        terminal_fraction=terminal_fraction,
    )


def estimate_observed_order(
    step_sizes: np.ndarray, errors: np.ndarray
) -> ObservedOrderMetrics:
    """Estimate observed order from successive step-size refinements."""

    h = np.asarray(step_sizes, dtype=float)
    err = np.asarray(errors, dtype=float)
    if h.ndim != 1 or err.ndim != 1 or h.size != err.size or h.size < 2:
        raise ValueError(
            "step_sizes and errors must be one-dimensional arrays of equal length >= 2"
        )
    if np.any(~np.isfinite(h)) or np.any(~np.isfinite(err)):
        raise ValueError("step_sizes and errors must be finite")
    if np.any(h <= 0.0):
        raise ValueError("step_sizes must be positive")
    if np.any(err <= 0.0):
        raise ValueError("errors must be positive")

    orders: list[float] = []
    for i in range(h.size - 1):
        if np.isclose(h[i], h[i + 1]):
            raise ValueError("successive step sizes must differ")
        orders.append(float(np.log(err[i] / err[i + 1]) / np.log(h[i] / h[i + 1])))
    orders_arr = np.asarray(orders, dtype=float)
    return ObservedOrderMetrics(
        step_sizes=h,
        errors=err,
        orders=orders_arr,
        asymptotic_order=float(orders_arr[-1]),
    )


def branch_continuity_metrics(
    ky: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    *,
    successive_overlap: np.ndarray | None = None,
    floor_fraction: float = 1.0e-8,
) -> BranchContinuationMetrics:
    """Compute branch-continuity diagnostics for a linear scan.

    The relative jump normalization uses a local scale from adjacent values,
    with a floor tied to the largest value in the scan. This avoids false
    blow-ups near marginal points while still flagging branch jumps.
    """

    ky_arr = np.asarray(ky, dtype=float)
    gamma_arr = np.asarray(gamma, dtype=float)
    omega_arr = np.asarray(omega, dtype=float)
    if ky_arr.ndim != 1 or gamma_arr.ndim != 1 or omega_arr.ndim != 1:
        raise ValueError("ky, gamma, and omega must be one-dimensional arrays")
    if not (ky_arr.size == gamma_arr.size == omega_arr.size):
        raise ValueError("ky, gamma, and omega must have equal length")
    if ky_arr.size < 2:
        raise ValueError("branch continuity requires at least two ky samples")
    if (
        np.any(~np.isfinite(ky_arr))
        or np.any(~np.isfinite(gamma_arr))
        or np.any(~np.isfinite(omega_arr))
    ):
        raise ValueError("ky, gamma, and omega must be finite")
    floor = float(floor_fraction)
    if floor < 0.0:
        raise ValueError("floor_fraction must be non-negative")

    def _relative_jumps(values: np.ndarray) -> np.ndarray:
        jumps = np.abs(np.diff(values))
        global_floor = max(float(np.nanmax(np.abs(values))) * floor, 1.0e-30)
        local_scale = np.maximum(
            np.maximum(np.abs(values[:-1]), np.abs(values[1:])), global_floor
        )
        return jumps / local_scale

    overlap_min: float | None = None
    if successive_overlap is not None:
        overlap = np.asarray(successive_overlap, dtype=float)
        if overlap.ndim != 1 or overlap.size != ky_arr.size - 1:
            raise ValueError("successive_overlap must have length len(ky) - 1")
        if np.any(~np.isfinite(overlap)):
            raise ValueError("successive_overlap must be finite")
        overlap_min = float(np.min(overlap))

    gamma_jumps = _relative_jumps(gamma_arr)
    omega_jumps = _relative_jumps(omega_arr)
    return BranchContinuationMetrics(
        ky=ky_arr,
        gamma=gamma_arr,
        omega=omega_arr,
        rel_gamma_jumps=gamma_jumps,
        rel_omega_jumps=omega_jumps,
        max_rel_gamma_jump=float(np.max(gamma_jumps)),
        max_rel_omega_jump=float(np.max(omega_jumps)),
        min_successive_overlap=overlap_min,
    )


__all__ = [
    "branch_continuity_metrics",
    "estimate_observed_order",
    "late_time_linear_metrics",
    "nonlinear_heat_flux_convergence_metrics",
    "windowed_nonlinear_metrics",
    "zonal_flow_response_metrics",
]
