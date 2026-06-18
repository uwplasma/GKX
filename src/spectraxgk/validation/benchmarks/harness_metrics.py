"""Physics metric extractors for benchmark and validation traces."""

from __future__ import annotations

import numpy as np

from spectraxgk.diagnostics.analysis import extract_mode_time_series, fit_growth_rate
from spectraxgk.validation.benchmarks.harness_timeseries import (
    _analytic_signal,
    _explicit_time_window,
    _leading_window,
    _tail_stats,
    _tail_window,
)
from spectraxgk.validation.gates import (
    BranchContinuationMetrics,
    LateTimeLinearMetrics,
    NonlinearHeatFluxConvergenceMetrics,
    NonlinearWindowMetrics,
    ObservedOrderMetrics,
    ZonalFlowResponseMetrics,
)


def zonal_flow_response_metrics(
    t: np.ndarray,
    response: np.ndarray,
    *,
    tail_fraction: float = 0.3,
    initial_fraction: float = 0.1,
    initial_policy: str = "window_abs_mean",
    initial_level_override: float | None = None,
    peak_fit_max_peaks: int | None = None,
    damping_fit_mode: str = "combined_envelope",
    frequency_fit_mode: str = "peak_spacing",
    fit_window_tmin: float | None = None,
    fit_window_tmax: float | None = None,
    hilbert_trim_fraction: float = 0.2,
) -> ZonalFlowResponseMetrics:
    """Estimate residual level and GAM envelope metrics from a zonal response.

    The input ``response`` should be a scalar zonal observable such as zonal
    potential or a normalized zonal-energy proxy on a uniform time trace.
    ``initial_policy="first_abs"`` follows Rosenbluth-Hinton/GAM convention by
    normalizing to the initial potential magnitude; ``"window_abs_mean"`` keeps
    the older robust behavior for generic noisy traces. ``initial_level_override``
    supports benchmarks whose published normalization is an external initial
    amplitude, for example a Gaussian potential maximum rather than the first
    line-averaged sample.
    """

    t_arr = np.asarray(t, dtype=float)
    resp = np.asarray(response, dtype=float)
    if t_arr.ndim != 1 or resp.ndim != 1 or t_arr.size != resp.size:
        raise ValueError(
            "t and response must be one-dimensional arrays of equal length"
        )
    if t_arr.size < 4:
        raise ValueError("zonal-flow response requires at least four samples")

    finite = np.isfinite(t_arr) & np.isfinite(resp)
    t_arr = t_arr[finite]
    resp = resp[finite]
    if t_arr.size < 4:
        raise ValueError("zonal-flow response requires at least four finite samples")

    policy = str(initial_policy).strip().lower().replace("-", "_")
    if policy not in {"window_abs_mean", "first_abs"}:
        raise ValueError(
            "initial_policy must be one of {'window_abs_mean', 'first_abs'}"
        )
    if peak_fit_max_peaks is not None and int(peak_fit_max_peaks) <= 0:
        raise ValueError("peak_fit_max_peaks must be > 0 when provided")
    damping_mode = str(damping_fit_mode).strip().lower().replace("-", "_")
    if damping_mode not in {"combined_envelope", "branchwise_extrema"}:
        raise ValueError(
            "damping_fit_mode must be one of {'combined_envelope', 'branchwise_extrema'}"
        )
    frequency_mode = str(frequency_fit_mode).strip().lower().replace("-", "_")
    if frequency_mode not in {"peak_spacing", "hilbert_phase"}:
        raise ValueError(
            "frequency_fit_mode must be one of {'peak_spacing', 'hilbert_phase'}"
        )
    if not 0.0 <= float(hilbert_trim_fraction) < 0.5:
        raise ValueError("hilbert_trim_fraction must be in [0, 0.5)")

    tail_mask, tail_tmin, tail_tmax = _tail_window(t_arr, float(tail_fraction))
    tail_vals = resp[tail_mask]
    if tail_vals.size == 0:
        raise ValueError("response windows must be non-empty")

    if initial_level_override is not None:
        initial_level = float(initial_level_override)
    elif policy == "first_abs":
        initial_level = float(abs(resp[0]))
    else:
        lead_mask, _lead_tmin, _lead_tmax = _leading_window(
            t_arr, float(initial_fraction)
        )
        initial_vals = resp[lead_mask]
        if initial_vals.size == 0:
            raise ValueError("response windows must be non-empty")
        initial_level = float(np.mean(np.abs(initial_vals)))
    if initial_level <= 0.0 or not np.isfinite(initial_level):
        raise ValueError("initial response level must be finite and positive")

    residual = float(np.mean(tail_vals))
    residual_std = float(np.std(tail_vals))
    response_norm = resp / initial_level
    residual_norm = residual / initial_level
    residual_std_norm = residual_std / initial_level
    response_rms = float(np.sqrt(np.mean(np.square(response_norm[tail_mask]))))

    detrended_norm = response_norm - residual_norm
    fit_mask, fit_tmin, fit_tmax = _explicit_time_window(
        t_arr,
        tmin=fit_window_tmin,
        tmax=fit_window_tmax,
    )

    max_peak_idx = np.asarray([], dtype=int)
    min_peak_idx = np.asarray([], dtype=int)
    peak_idx = np.asarray([], dtype=int)
    if detrended_norm.size >= 3:
        max_peak_idx = (
            np.flatnonzero(
                (detrended_norm[1:-1] > detrended_norm[:-2])
                & (detrended_norm[1:-1] >= detrended_norm[2:])
                & (detrended_norm[1:-1] > 1.0e-12)
            )
            + 1
        )
        min_peak_idx = (
            np.flatnonzero(
                (detrended_norm[1:-1] < detrended_norm[:-2])
                & (detrended_norm[1:-1] <= detrended_norm[2:])
                & (detrended_norm[1:-1] < -1.0e-12)
            )
            + 1
        )
        peak_idx = np.sort(np.concatenate([max_peak_idx, min_peak_idx]))
    peak_times = t_arr[peak_idx]
    peak_values = np.abs(detrended_norm[peak_idx])
    max_peak_times = t_arr[max_peak_idx]
    max_peak_values = response_norm[max_peak_idx]
    min_peak_times = t_arr[min_peak_idx]
    min_peak_values = response_norm[min_peak_idx]

    gam_frequency = float("nan")
    gam_damping = float("nan")
    peak_fit_count = 0
    if damping_mode == "combined_envelope":
        peak_fit_times = peak_times[fit_mask[peak_idx]]
        peak_fit_values = peak_values[fit_mask[peak_idx]]
        if peak_fit_max_peaks is not None and peak_fit_times.size:
            nfit = min(int(peak_fit_max_peaks), int(peak_fit_times.size))
            peak_fit_times = peak_fit_times[:nfit]
            peak_fit_values = peak_fit_values[:nfit]
        peak_fit_count = int(peak_fit_times.size)
        valid = np.isfinite(peak_fit_values) & (peak_fit_values > 0.0)
        if np.count_nonzero(valid) >= 2:
            slope, _offset = np.polyfit(
                peak_fit_times[valid], np.log(peak_fit_values[valid]), 1
            )
            gam_damping = float(-slope)
    else:
        branch_gammas: list[float] = []
        branch_counts: list[int] = []
        for branch_idx in (max_peak_idx, min_peak_idx):
            idx = branch_idx[fit_mask[branch_idx]]
            if peak_fit_max_peaks is not None and idx.size:
                idx = idx[: min(int(peak_fit_max_peaks), int(idx.size))]
            amp = np.abs(detrended_norm[idx])
            valid = np.isfinite(amp) & (amp > 0.0)
            if np.count_nonzero(valid) >= 2:
                slope, _offset = np.polyfit(t_arr[idx][valid], np.log(amp[valid]), 1)
                branch_gammas.append(float(-slope))
                branch_counts.append(int(np.count_nonzero(valid)))
        if branch_gammas:
            gam_damping = float(np.mean(branch_gammas))
            peak_fit_count = int(np.sum(branch_counts))

    fit_peak_times = peak_times[fit_mask[peak_idx]]
    if (
        peak_fit_max_peaks is not None
        and damping_mode == "combined_envelope"
        and fit_peak_times.size
    ):
        fit_peak_times = fit_peak_times[
            : min(int(peak_fit_max_peaks), int(fit_peak_times.size))
        ]

    if frequency_mode == "peak_spacing":
        freq_peak_times = (
            fit_peak_times
            if fit_peak_times.size >= 2
            else peak_times[fit_mask[peak_idx]]
        )
        if freq_peak_times.size >= 2:
            dt_peaks = np.diff(freq_peak_times)
            dt_peaks = dt_peaks[np.isfinite(dt_peaks) & (dt_peaks > 0.0)]
            if dt_peaks.size:
                gam_frequency = float(np.pi / np.mean(dt_peaks))
    else:
        fit_t = t_arr[fit_mask]
        fit_signal = detrended_norm[fit_mask]
        if fit_t.size >= 8:
            analytic = _analytic_signal(fit_signal)
            phase = np.unwrap(np.angle(analytic))
            omega = np.gradient(phase, fit_t)
            trim = int(np.floor(float(hilbert_trim_fraction) * fit_t.size))
            trim_mask = np.ones_like(fit_t, dtype=bool)
            if trim > 0:
                trim_mask[:trim] = False
                trim_mask[-trim:] = False
            amp = np.abs(analytic)
            valid = np.isfinite(omega) & np.isfinite(amp) & (amp > 1.0e-6) & trim_mask
            if np.count_nonzero(valid) >= 2:
                gam_frequency = float(np.mean(omega[valid]))

    return ZonalFlowResponseMetrics(
        initial_level=initial_level,
        initial_policy=policy,
        residual_level=residual_norm,
        residual_std=residual_std_norm,
        response_rms=response_rms,
        gam_frequency=gam_frequency,
        gam_damping_rate=gam_damping,
        damping_method=damping_mode,
        frequency_method=frequency_mode,
        peak_count=int(peak_times.size),
        peak_fit_count=int(peak_fit_count),
        tmin=float(tail_tmin if tail_tmin is not None else t_arr[0]),
        tmax=float(tail_tmax if tail_tmax is not None else t_arr[-1]),
        fit_tmin=float(fit_tmin),
        fit_tmax=float(fit_tmax),
        peak_times=np.asarray(peak_times, dtype=float),
        peak_envelope=np.asarray(peak_values, dtype=float),
        max_peak_times=np.asarray(max_peak_times, dtype=float),
        max_peak_values=np.asarray(max_peak_values, dtype=float),
        min_peak_times=np.asarray(min_peak_times, dtype=float),
        min_peak_values=np.asarray(min_peak_values, dtype=float),
    )


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

    t_arr = np.asarray(t, dtype=float)
    q_arr = np.asarray(heat_flux, dtype=float)
    if t_arr.ndim != 1 or q_arr.ndim != 1 or t_arr.size != q_arr.size:
        raise ValueError(
            "t and heat_flux must be one-dimensional arrays of equal length"
        )
    if t_arr.size == 0:
        raise ValueError("t and heat_flux must be non-empty")
    if not 0.0 <= float(start_fraction) < 1.0:
        raise ValueError("start_fraction must be in [0, 1)")
    if not 0.0 < float(terminal_fraction) <= 1.0:
        raise ValueError("terminal_fraction must be in (0, 1]")
    if float(mean_floor) < 0.0:
        raise ValueError("mean_floor must be non-negative")

    finite = np.isfinite(t_arr) & np.isfinite(q_arr)
    t_arr = t_arr[finite]
    q_arr = q_arr[finite]
    if t_arr.size == 0:
        raise ValueError(
            "t and heat_flux must contain at least one finite paired sample"
        )
    if t_arr.size > 1 and np.any(np.diff(t_arr) <= 0.0):
        raise ValueError("t must be strictly increasing after finite-sample filtering")

    tail_fraction = max(np.finfo(float).eps, 1.0 - float(start_fraction))
    mask, tmin, tmax = _tail_window(t_arr, tail_fraction)
    t_win = t_arr[mask]
    q_win = q_arr[mask]
    if q_win.size == 0:
        raise ValueError("post-transient heat-flux window is empty")

    terminal_start = max(
        0, int(np.floor((1.0 - float(terminal_fraction)) * q_win.size))
    )
    t_terminal = t_win[terminal_start:]
    q_terminal = q_win[terminal_start:]
    if q_terminal.size == 0:
        raise ValueError("terminal heat-flux window is empty")

    mean = float(np.mean(q_win))
    std = float(np.std(q_win))
    rms = float(np.sqrt(np.mean(np.square(q_win))))
    terminal_mean = float(np.mean(q_terminal))
    scale = max(abs(mean), float(mean_floor))
    cv = float(std / scale) if scale > 0.0 else float("inf")
    mean_rel_delta = (
        float(abs(terminal_mean - mean) / scale) if scale > 0.0 else float("inf")
    )

    trend = 0.0
    if t_win.size >= 2 and float(t_win[-1] - t_win[0]) > 0.0:
        slope, _offset = np.polyfit(t_win, q_win, 1)
        trend = (
            float(slope * (t_win[-1] - t_win[0]) / scale)
            if scale > 0.0
            else float("inf")
        )

    return NonlinearHeatFluxConvergenceMetrics(
        tmin=float(tmin if tmin is not None else t_win[0]),
        tmax=float(tmax if tmax is not None else t_win[-1]),
        nsamples=int(q_win.size),
        heat_flux_mean=mean,
        heat_flux_std=std,
        heat_flux_cv=cv,
        heat_flux_rms=rms,
        terminal_tmin=float(t_terminal[0]),
        terminal_tmax=float(t_terminal[-1]),
        terminal_nsamples=int(q_terminal.size),
        terminal_heat_flux_mean=terminal_mean,
        mean_rel_delta=mean_rel_delta,
        trend=trend,
        abs_trend=float(abs(trend)),
        start_fraction=float(start_fraction),
        terminal_fraction=float(terminal_fraction),
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
