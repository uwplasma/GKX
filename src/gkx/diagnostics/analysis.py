"""Public facade for mode extraction and growth-rate diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from gkx.diagnostics.growth_windows import _tail_stats, _tail_window
from gkx.diagnostics.metadata import CFL_SCALE_LABELS
from gkx.diagnostics.growth_rates import (
    GrowthRateFitStats,
    _log_amp_phase,
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_uncertainty,
    fit_growth_rate_with_stats,
    instantaneous_growth_rate_from_phi,
    select_fit_window,
    select_fit_window_loglinear,
    select_fit_window_stationary,
    windowed_growth_rate_from_omega_series,
)
from gkx.diagnostics.modes import (
    ModeSelection,
    ModeSelectionBatch,
    density_moment,
    extract_eigenfunction,
    extract_mode,
    extract_mode_time_series,
    select_ky_index,
)


@dataclass(frozen=True)
class LateTimeLinearMetrics:
    """Late-time growth/frequency metrics for a linear run."""

    gamma_fit: float
    omega_fit: float
    gamma_tail_mean: float
    omega_tail_mean: float
    gamma_tail_std: float
    omega_tail_std: float
    tmin: float | None
    tmax: float | None
    nsamples: int
    signal_source: str


@dataclass(frozen=True)
class NonlinearWindowMetrics:
    """Windowed transport/envelope metrics for a nonlinear run."""

    tmin: float
    tmax: float
    nsamples: int
    heat_flux_mean: float
    heat_flux_std: float
    heat_flux_rms: float
    wphi_mean: float
    wphi_std: float
    wg_mean: float
    wg_std: float
    phi_mode_envelope_mean: float | None
    phi_mode_envelope_std: float | None
    phi_mode_envelope_max: float | None
    # Correlation statistics. heat_flux_stderr divides by n_eff, not nsamples,
    # because turbulence outputs are not independent draws. Defaults are
    # fail-closed so a hand-built object cannot pass a statistical gate by
    # omission; windowed_nonlinear_metrics always sets them.
    heat_flux_tau_ac: float = 0.0
    window_in_tau_ac: float = 0.0
    heat_flux_n_eff: float = 0.0
    heat_flux_stderr: float = float("inf")


@dataclass(frozen=True)
class NonlinearHeatFluxConvergenceMetrics:
    """Post-transient heat-flux averaging convergence summary."""

    tmin: float
    tmax: float
    nsamples: int
    heat_flux_mean: float
    heat_flux_std: float
    heat_flux_cv: float
    heat_flux_rms: float
    terminal_tmin: float
    terminal_tmax: float
    terminal_nsamples: int
    terminal_heat_flux_mean: float
    mean_rel_delta: float
    trend: float
    abs_trend: float
    start_fraction: float
    terminal_fraction: float
    # n_eff is 2.6 to 11.8 across the tracked traces where nsamples is 31 to 92.
    tau_ac: float = 0.0
    n_eff: float = 0.0


@dataclass(frozen=True)
class ObservedOrderMetrics:
    """Observed-order convergence summary from step sizes and errors."""

    step_sizes: np.ndarray
    errors: np.ndarray
    orders: np.ndarray
    asymptotic_order: float


@dataclass(frozen=True)
class BranchContinuationMetrics:
    """Continuity summary for a scanned linear branch."""

    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray
    rel_gamma_jumps: np.ndarray
    rel_omega_jumps: np.ndarray
    max_rel_gamma_jump: float
    max_rel_omega_jump: float
    min_successive_overlap: float | None


# Physics metric extractors for benchmark and validation traces.
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


def _scalar_late_time_linear_metrics(result: object) -> LateTimeLinearMetrics:
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


def _linear_signal_series(
    result: object,
    *,
    mode_method: str,
) -> tuple[np.ndarray | None, str]:
    signal = getattr(result, "signal", None)
    if signal is not None:
        return np.asarray(signal, dtype=np.complex128), "signal"
    if hasattr(result, "phi_t") and hasattr(result, "selection"):
        series = extract_mode_time_series(
            np.asarray(getattr(result, "phi_t")),
            getattr(result, "selection"),
            method=mode_method,
        )
        return np.asarray(series, dtype=np.complex128), f"phi_t:{mode_method}"
    return None, "scalar"


def _fit_tail_signal(
    t_arr: np.ndarray,
    mask: np.ndarray,
    signal_arr: np.ndarray | None,
    *,
    gamma_fallback: float,
    omega_fallback: float,
) -> tuple[float, float]:
    if signal_arr is None:
        return gamma_fallback, omega_fallback
    finite = np.isfinite(signal_arr)
    signal_tail = signal_arr[mask & finite]
    t_tail = t_arr[mask & finite]
    if t_tail.size < 2:
        return gamma_fallback, omega_fallback
    gamma_fit, omega_fit = fit_growth_rate(t_tail, signal_tail)
    return float(gamma_fit), float(omega_fit)


def _tail_series_or_fit(
    series: object | None,
    mask: np.ndarray,
    fit_value: float,
) -> tuple[float, float]:
    if series is None:
        return float(fit_value), 0.0
    mean, std = _tail_stats(np.asarray(series), mask)
    return float(mean), float(std)


def late_time_linear_metrics(
    result: object,
    *,
    tail_fraction: float = 0.5,
    mode_method: str = "project",
) -> LateTimeLinearMetrics:
    """Return late-time growth/frequency metrics from a linear benchmark/runtime result."""

    t = getattr(result, "t", None)
    if t is None:
        return _scalar_late_time_linear_metrics(result)

    t_arr = np.asarray(t, dtype=float)
    mask, tmin, tmax = _tail_window(t_arr, tail_fraction)

    gamma_fit = float(getattr(result, "gamma"))
    omega_fit = float(getattr(result, "omega"))
    signal_arr, signal_source = _linear_signal_series(result, mode_method=mode_method)
    gamma_fit, omega_fit = _fit_tail_signal(
        t_arr,
        mask,
        signal_arr,
        gamma_fallback=gamma_fit,
        omega_fallback=omega_fit,
    )
    gamma_mean, gamma_std = _tail_series_or_fit(
        getattr(result, "gamma_t", None), mask, gamma_fit
    )
    omega_mean, omega_std = _tail_series_or_fit(
        getattr(result, "omega_t", None), mask, omega_fit
    )

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


def sokal_autocorrelation_time(
    signal: np.ndarray, dt: float
) -> tuple[float, int, np.ndarray]:
    """Return the first-zero IAT, crossing index, and autocorrelation."""

    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    variance = float(x @ x)
    if x.size < 4 or variance <= 0.0 or dt <= 0.0:
        return 0.0, 1, np.zeros(1)
    size = int(2 ** np.ceil(np.log2(2 * x.size)))
    spectrum = np.fft.rfft(x, n=size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), n=size)[: x.size]
    rho = correlation / correlation[0]
    negative = np.nonzero(rho < 0.0)[0]
    cut = int(negative[0]) if negative.size else rho.size
    tau = float(np.trapezoid(rho[:cut], dx=dt)) if cut > 1 else 0.0
    return tau, cut, rho


def integrated_autocorrelation_time(signal: np.ndarray, dt: float) -> float:
    """Integrated autocorrelation time, truncated at the first zero crossing.

    Turbulence outputs are correlated, so ``std / sqrt(n)`` understates the
    uncertainty of their mean by ``sqrt(n / n_eff)`` -- 2.0x to 3.7x across the
    tracked traces. This converts an output count into an independent one.
    Truncation at the first zero crossing is a positive-window estimator;
    summing farther into a noisy tail adds variance with little signal.
    """

    return sokal_autocorrelation_time(signal, dt)[0]


def _correlated_sample_stats(
    t: np.ndarray, q: np.ndarray
) -> tuple[float, float, float]:
    """Return ``(tau_ac, n_eff, span)`` for a windowed series.

    ``n_eff = n dt / (2 tau)``, capped at ``n``. The estimator integrates the
    normalized autocorrelation from zero lag, so ``tau -> dt/2`` for an
    uncorrelated series; this form then returns ``n_eff = n``, as it must.
    Using ``n / (1 + 2 tau / dt)`` instead double-counts the zero-lag term and
    reports ``n/2`` for independent samples -- validated against the empirical
    scatter of independent realizations, that overestimated the standard error
    by 22% at zero correlation
    (``tools/artifacts/build_window_statistics_validation.py``).
    """

    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
    tau = integrated_autocorrelation_time(q, dt)
    n_eff = (
        min(float(q.size), q.size * dt / (2.0 * tau))
        if tau > 0.0 and dt > 0.0
        else float(q.size)
    )
    span = float(t[-1] - t[0]) if t.size > 1 else 0.0
    return tau, float(n_eff), span


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

    tau_ac, n_eff, span = _correlated_sample_stats(t[mask], heat_flux)
    heat_flux_std = float(np.std(heat_flux))

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
        heat_flux_tau_ac=tau_ac,
        window_in_tau_ac=(span / tau_ac if tau_ac > 0.0 else float("inf")),
        heat_flux_n_eff=float(n_eff),
        heat_flux_stderr=float(heat_flux_std / np.sqrt(max(n_eff, 1.0))),
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
    terminal_start = max(0, int(np.floor((1.0 - terminal_fraction) * window.q.size)))
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

    window_tau, window_n_eff, _ = _correlated_sample_stats(window.t, window.q)

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
        tau_ac=float(window_tau),
        n_eff=float(window_n_eff),
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
    "BranchContinuationMetrics",
    "CFLLimiterReport",
    "CFLScales",
    "CFL_TERM_NAMES",
    "CFL_TERM_UNRESOLVED",
    "LateTimeLinearMetrics",
    "NonlinearHeatFluxConvergenceMetrics",
    "NonlinearWindowMetrics",
    "ObservedOrderMetrics",
    "branch_continuity_metrics",
    "cfl_limiter_report",
    "cfl_limiting_term",
    "cfl_scales_from_array",
    "cfl_term_contributions",
    "estimate_observed_order",
    "late_time_linear_metrics",
    "nonlinear_heat_flux_convergence_metrics",
    "windowed_nonlinear_metrics",
    "ModeSelection",
    "ModeSelectionBatch",
    "_log_amp_phase",
    "density_moment",
    "extract_eigenfunction",
    "extract_mode",
    "extract_mode_time_series",
    "GrowthRateFitStats",
    "fit_growth_rate",
    "fit_growth_rate_auto",
    "fit_growth_rate_auto_with_stats",
    "fit_growth_rate_uncertainty",
    "fit_growth_rate_with_stats",
    "instantaneous_growth_rate_from_phi",
    "select_fit_window",
    "select_fit_window_loglinear",
    "select_fit_window_stationary",
    "select_ky_index",
    "windowed_growth_rate_from_omega_series",
]


def fit_growth_rate_auto_with_stats(
    t: np.ndarray,
    signal: np.ndarray,
    tmin: float | None = None,
    tmax: float | None = None,
    window_fraction: float = 0.3,
    min_points: int = 20,
    start_fraction: float = 0.0,
    growth_weight: float = 0.0,
    require_positive: bool = False,
    min_amp_fraction: float = 0.0,
    max_amp_fraction: float = 0.9,
    window_method: str = "stationary",
    max_fraction: float = 0.8,
    end_fraction: float = 0.9,
    num_windows: int = 8,
    phase_weight: float = 0.2,
    length_weight: float = 0.05,
    min_r2: float = 0.0,
    late_penalty: float = 0.1,
    min_slope: float | None = None,
    min_slope_frac: float = 0.0,
    slope_var_weight: float = 0.0,
) -> Tuple[float, float, float, float, float, float]:
    """Fit gamma/omega and report selected window plus R^2 scores.

    This wrapper intentionally calls the facade-level
    :func:`fit_growth_rate_with_stats` so tests and downstream users can
    monkeypatch the public analysis module without reaching into implementation
    modules.
    """

    gamma, omega, tmin_out, tmax_out = fit_growth_rate_auto(
        t,
        signal,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        max_amp_fraction=max_amp_fraction,
        window_method=window_method,
        max_fraction=max_fraction,
        end_fraction=end_fraction,
        num_windows=num_windows,
        phase_weight=phase_weight,
        length_weight=length_weight,
        min_r2=min_r2,
        late_penalty=late_penalty,
        min_slope=min_slope,
        min_slope_frac=min_slope_frac,
        slope_var_weight=slope_var_weight,
    )
    try:
        _gamma, _omega, r2_log, r2_phase = fit_growth_rate_with_stats(
            t, signal, tmin=tmin_out, tmax=tmax_out
        )
    except ValueError:
        r2_log = -np.inf
        r2_phase = -np.inf
    return gamma, omega, tmin_out, tmax_out, float(r2_log), float(r2_phase)


# Nonlinear CFL attribution. The adaptive nonlinear step is
# ``dt = clip(dt_cfl_numerator / omega_total, dt_min, dt_max)``, so a recorded
# dt trajectory plus the run-constant CFL scales invert back to the CFL
# frequency and split it by term -- turning "this surface is slow" into which
# term set the step and by how much it grew.

#: Terms that can set the nonlinear CFL frequency, in tie-break order.
CFL_TERM_NAMES: tuple[str, ...] = (
    "magnetic_drift_radial",
    "magnetic_drift_binormal",
    "parallel_streaming",
    "exb",
)


def cfl_term_contributions(
    *,
    magnetic_drift_radial: float,
    magnetic_drift_binormal: float,
    parallel_streaming: float,
    exb_radial: float,
    exb_binormal: float,
) -> dict[str, float]:
    """Split the nonlinear CFL frequency into additive per-term contributions.

    The integrator forms ``omega_total = max(drift_radial, exb_radial) +
    max(drift_binormal, exb_binormal) + parallel_streaming``. Since
    ``max(a, b) = a + max(0, b - a)``, ExB contributes exactly the excess it
    adds over the drift it displaces, so the returned values are additive and
    sum to ``omega_total`` -- a share is then a real fraction of the
    step-setting frequency, not a ranking of quantities never added together.
    """

    drift_x = float(magnetic_drift_radial)
    drift_y = float(magnetic_drift_binormal)
    return {
        "magnetic_drift_radial": drift_x,
        "magnetic_drift_binormal": drift_y,
        "parallel_streaming": float(parallel_streaming),
        "exb": max(0.0, float(exb_radial) - drift_x)
        + max(0.0, float(exb_binormal) - drift_y),
    }


def cfl_limiting_term(contributions: dict[str, float]) -> tuple[str, float]:
    """Return the largest CFL contribution and its share of the total.

    Ties resolve toward the earlier :data:`CFL_TERM_NAMES` entry, so a run
    whose ExB frequency merely matches a drift is not called ExB-limited.
    """

    total = sum(max(0.0, float(value)) for value in contributions.values())
    best = max(CFL_TERM_NAMES, key=lambda name: float(contributions.get(name, 0.0)))
    return best, (float(contributions[best]) / total if total > 0.0 else 0.0)


@dataclass(frozen=True)
class CFLScales:
    """Run-constant CFL scales recorded alongside a nonlinear dt trajectory."""

    omega_magnetic_drift_radial: float
    omega_magnetic_drift_binormal: float
    omega_parallel_streaming: float
    dt_cfl_numerator: float
    dt_min: float
    dt_max: float

    @property
    def omega_linear_floor(self) -> float:
        """CFL frequency the linear terms impose regardless of turbulence."""

        return (
            self.omega_magnetic_drift_radial
            + self.omega_magnetic_drift_binormal
            + self.omega_parallel_streaming
        )

    def contributions_for(self, omega_total: float) -> dict[str, float]:
        """Per-term contributions implied by an inverted total frequency."""

        return {
            "magnetic_drift_radial": self.omega_magnetic_drift_radial,
            "magnetic_drift_binormal": self.omega_magnetic_drift_binormal,
            "parallel_streaming": self.omega_parallel_streaming,
            "exb": max(0.0, float(omega_total) - self.omega_linear_floor),
        }


def cfl_scales_from_array(scales: object | None) -> CFLScales | None:
    """Decode a recorded ``cfl_scales`` vector, or ``None`` when unusable."""

    if scales is None:
        return None
    arr = np.asarray(scales, dtype=float).reshape(-1)
    if arr.size != len(CFL_SCALE_LABELS) or not np.all(np.isfinite(arr)) or arr[3] <= 0:
        return None
    return CFLScales(*(float(value) for value in arr))


#: Reported instead of a term name when no sample could be attributed.
CFL_TERM_UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class CFLLimiterReport:
    """How the CFL frequency split between terms over a nonlinear run."""

    omega_linear_floor: float
    omega_total_initial: float
    omega_total_final: float
    omega_total_max: float
    exb_share_initial: float
    exb_share_final: float
    exb_growth_ratio: float
    limiting_term_initial: str
    limiting_term_final: str
    limiting_term_final_share: float
    limiting_term_sample_fractions: dict[str, float]
    samples_total: int
    samples_attributed: int
    samples_at_dt_floor: int
    samples_at_dt_ceiling: int


def cfl_limiter_report(dt: np.ndarray, scales: CFLScales) -> CFLLimiterReport:
    """Attribute a recorded dt trajectory to the CFL terms that produced it.

    Samples clipped at ``dt_max`` are excluded from the attribution. There the
    solver wanted a larger step than the ceiling allows, so the inverted
    frequency is only an upper bound and naming a limiting term from it would
    invent one: a run that never leaves its ceiling is capped, not CFL-limited,
    and is reported as ``unresolved`` with the clip counts to say so. Samples
    pinned at ``dt_min`` are kept -- there the inverted frequency is a lower
    bound, so any ExB excess it shows is real and understated, not invented.
    """

    dt_arr = np.asarray(dt, dtype=float).reshape(-1)
    dt_arr = dt_arr[np.isfinite(dt_arr) & (dt_arr > 0.0)]
    if dt_arr.size == 0:
        raise ValueError("dt must contain at least one positive finite sample")
    tol = 1.0 + 1.0e-9
    at_ceiling = dt_arr >= scales.dt_max / tol
    usable = dt_arr[~at_ceiling]
    counts = {
        "samples_total": int(dt_arr.size),
        "samples_attributed": int(usable.size),
        "samples_at_dt_floor": int(np.count_nonzero(dt_arr <= scales.dt_min * tol)),
        "samples_at_dt_ceiling": int(np.count_nonzero(at_ceiling)),
    }
    if usable.size == 0:
        return CFLLimiterReport(
            omega_linear_floor=scales.omega_linear_floor,
            omega_total_initial=float("nan"),
            omega_total_final=float("nan"),
            omega_total_max=float("nan"),
            exb_share_initial=float("nan"),
            exb_share_final=float("nan"),
            exb_growth_ratio=float("nan"),
            limiting_term_initial=CFL_TERM_UNRESOLVED,
            limiting_term_final=CFL_TERM_UNRESOLVED,
            limiting_term_final_share=float("nan"),
            limiting_term_sample_fractions={name: 0.0 for name in CFL_TERM_NAMES},
            **counts,
        )
    omega = scales.dt_cfl_numerator / usable
    exb = np.maximum(0.0, omega - scales.omega_linear_floor)
    shares = np.where(omega > 0.0, exb / omega, 0.0)
    limiters = [
        cfl_limiting_term(scales.contributions_for(float(value)))[0] for value in omega
    ]
    return CFLLimiterReport(
        omega_linear_floor=scales.omega_linear_floor,
        omega_total_initial=float(omega[0]),
        omega_total_final=float(omega[-1]),
        omega_total_max=float(np.max(omega)),
        exb_share_initial=float(shares[0]),
        exb_share_final=float(shares[-1]),
        exb_growth_ratio=(
            float(np.max(exb)) / float(exb[0]) if exb[0] > 0.0 else float("inf")
        ),
        limiting_term_initial=limiters[0],
        limiting_term_final=limiters[-1],
        limiting_term_final_share=cfl_limiting_term(
            scales.contributions_for(float(omega[-1]))
        )[1],
        limiting_term_sample_fractions={
            name: limiters.count(name) / len(limiters) for name in CFL_TERM_NAMES
        },
        **counts,
    )
