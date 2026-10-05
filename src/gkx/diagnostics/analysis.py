"""Public facade for mode extraction and growth-rate diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gkx.diagnostics.metadata import CFL_SCALE_LABELS
from gkx.diagnostics.growth_rates import (
    GrowthRateFitStats,
    _log_amp_phase,
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_auto_with_stats,
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
    "cfl_limiter_report",
    "cfl_limiting_term",
    "cfl_scales_from_array",
    "cfl_term_contributions",
    "estimate_observed_order",
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
