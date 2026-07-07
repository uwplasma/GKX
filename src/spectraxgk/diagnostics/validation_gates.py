"""Validation gate metrics and report builders for diagnostics artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


@dataclass(frozen=True)
class ZonalFlowResponseMetrics:
    """Late-time residual and GAM-envelope metrics for zonal-flow responses."""

    initial_level: float
    initial_policy: str
    residual_level: float
    residual_std: float
    response_rms: float
    gam_frequency: float
    gam_damping_rate: float
    damping_method: str
    frequency_method: str
    peak_count: int
    peak_fit_count: int
    tmin: float
    tmax: float
    fit_tmin: float
    fit_tmax: float
    peak_times: np.ndarray
    peak_envelope: np.ndarray
    max_peak_times: np.ndarray
    max_peak_values: np.ndarray
    min_peak_times: np.ndarray
    min_peak_values: np.ndarray


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


@dataclass(frozen=True)
class ScalarGateResult:
    """Pass/fail result for one benchmark observable.

    The tolerance convention follows ``numpy.isclose``: a metric passes when
    ``abs_error <= atol + rtol * abs(reference)``. This keeps near-zero
    frequency and marginal-growth gates explicit through ``atol`` rather than
    hiding them behind unstable relative errors.
    """

    metric: str
    observed: float
    reference: float
    abs_error: float
    rel_error: float
    atol: float
    rtol: float
    passed: bool
    units: str
    notes: str


@dataclass(frozen=True)
class GateReport:
    """Collection of scalar gates for one validation artifact."""

    case: str
    source: str
    gates: tuple[ScalarGateResult, ...]
    passed: bool
    max_abs_error: float
    max_rel_error: float


@dataclass(frozen=True)
class EigenfunctionComparisonMetrics:
    """Phase-aligned eigenfunction comparison summary."""

    overlap: float
    relative_l2: float
    phase_shift: float


@dataclass(frozen=True)
class EigenfunctionReferenceBundle:
    """Frozen reference eigenfunction bundle for manuscript-grade overlays."""

    theta: np.ndarray
    mode: np.ndarray
    source: str
    case: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class DiagnosticTimeSeries:
    """Single benchmark-facing time series loaded from an ``out.nc`` artifact."""

    t: np.ndarray
    values: np.ndarray
    variable: str
    source_path: str





def evaluate_scalar_gate(
    metric: str,
    observed: float,
    reference: float,
    *,
    atol: float,
    rtol: float,
    units: str = "",
    notes: str = "",
) -> ScalarGateResult:
    """Evaluate one scalar benchmark gate.

    Use this helper for publication-facing metrics such as growth rates,
    frequencies, windowed heat fluxes, zonal residuals, and damping rates. The
    explicit ``atol``/``rtol`` pair forces each artifact to document whether its
    tolerance is absolute, relative, or both.
    """

    obs = float(observed)
    ref = float(reference)
    atol_f = float(atol)
    rtol_f = float(rtol)
    if atol_f < 0.0 or rtol_f < 0.0:
        raise ValueError("atol and rtol must be non-negative")
    abs_error = float(abs(obs - ref)) if np.isfinite(obs) and np.isfinite(ref) else float("inf")
    if np.isfinite(ref) and abs(ref) > 0.0:
        rel_error = float(abs_error / abs(ref))
    else:
        rel_error = 0.0 if abs_error == 0.0 else float("inf")
    tolerance = atol_f + rtol_f * abs(ref)
    passed = bool(np.isfinite(obs) and np.isfinite(ref) and abs_error <= tolerance)
    return ScalarGateResult(
        metric=str(metric),
        observed=obs,
        reference=ref,
        abs_error=abs_error,
        rel_error=rel_error,
        atol=atol_f,
        rtol=rtol_f,
        passed=passed,
        units=str(units),
        notes=str(notes),
    )


def _upper_limit_gate(
    metric: str,
    observed: float,
    limit: float,
    *,
    notes: str = "",
) -> ScalarGateResult:
    """Gate quantities that should stay below a documented upper limit."""

    return evaluate_scalar_gate(
        metric,
        observed,
        0.0,
        atol=float(limit),
        rtol=0.0,
        notes=notes,
    )


def gate_report(
    case: str,
    source: str,
    gates: list[ScalarGateResult] | tuple[ScalarGateResult, ...],
) -> GateReport:
    """Summarize a set of scalar gates for one artifact."""

    gate_tuple = tuple(gates)
    if not gate_tuple:
        raise ValueError("gate report requires at least one scalar gate")
    finite_abs = [gate.abs_error for gate in gate_tuple if np.isfinite(gate.abs_error)]
    finite_rel = [gate.rel_error for gate in gate_tuple if np.isfinite(gate.rel_error)]
    return GateReport(
        case=str(case),
        source=str(source),
        gates=gate_tuple,
        passed=all(gate.passed for gate in gate_tuple),
        max_abs_error=float(max(finite_abs)) if finite_abs else float("inf"),
        max_rel_error=float(max(finite_rel)) if finite_rel else float("inf"),
    )


def gate_report_to_dict(report: GateReport) -> dict[str, object]:
    """Return a strict JSON-serializable representation of a gate report."""

    def _finite_json_float(value: float) -> float | None:
        val = float(value)
        return val if np.isfinite(val) else None

    return {
        "case": report.case,
        "source": report.source,
        "passed": bool(report.passed),
        "max_abs_error": _finite_json_float(report.max_abs_error),
        "max_rel_error": _finite_json_float(report.max_rel_error),
        "gates": [
            {
                "metric": gate.metric,
                "observed": _finite_json_float(gate.observed),
                "reference": _finite_json_float(gate.reference),
                "abs_error": _finite_json_float(gate.abs_error),
                "rel_error": _finite_json_float(gate.rel_error),
                "atol": _finite_json_float(gate.atol),
                "rtol": _finite_json_float(gate.rtol),
                "passed": bool(gate.passed),
                "units": gate.units,
                "notes": gate.notes,
            }
            for gate in report.gates
        ],
    }


def linear_metrics_gate_report(
    observed: LateTimeLinearMetrics,
    reference: LateTimeLinearMetrics,
    *,
    case: str,
    source: str,
    gamma_atol: float = 0.0,
    gamma_rtol: float = 0.05,
    omega_atol: float = 0.0,
    omega_rtol: float = 0.05,
) -> GateReport:
    """Gate late-time linear growth and frequency metrics."""

    return gate_report(
        case,
        source,
        (
            evaluate_scalar_gate(
                "gamma_fit",
                observed.gamma_fit,
                reference.gamma_fit,
                atol=gamma_atol,
                rtol=gamma_rtol,
                units="v_t/R",
            ),
            evaluate_scalar_gate(
                "omega_fit",
                observed.omega_fit,
                reference.omega_fit,
                atol=omega_atol,
                rtol=omega_rtol,
                units="v_t/R",
            ),
        ),
    )


def nonlinear_window_gate_report(
    observed: NonlinearWindowMetrics,
    reference: NonlinearWindowMetrics,
    *,
    case: str,
    source: str,
    rtol: float = 0.1,
    atol: float = 0.0,
    include_envelope: bool = True,
) -> GateReport:
    """Gate windowed nonlinear transport and field-energy metrics."""

    metrics = ("heat_flux_mean", "heat_flux_rms", "wphi_mean", "wg_mean")
    gates: list[ScalarGateResult] = []
    for metric in metrics:
        gates.append(
            evaluate_scalar_gate(
                metric,
                getattr(observed, metric),
                getattr(reference, metric),
                atol=atol,
                rtol=rtol,
            )
        )
    if (
        include_envelope
        and observed.phi_mode_envelope_mean is not None
        and reference.phi_mode_envelope_mean is not None
    ):
        gates.append(
            evaluate_scalar_gate(
                "phi_mode_envelope_mean",
                observed.phi_mode_envelope_mean,
                reference.phi_mode_envelope_mean,
                atol=atol,
                rtol=rtol,
            )
        )
    return gate_report(case, source, gates)


def nonlinear_heat_flux_convergence_gate_report(
    metrics: NonlinearHeatFluxConvergenceMetrics,
    *,
    case: str,
    source: str,
    max_mean_rel_delta: float = 0.05,
    max_cv: float = 0.15,
    max_abs_trend: float = 0.10,
    min_samples: int = 8,
) -> GateReport:
    """Gate post-transient heat-flux averaging stability.

    This is an internal promotion gate for nonlinear transport claims: the
    post-transient average must agree with its terminal subwindow, have bounded
    coefficient of variation, show limited normalized drift across the window,
    and contain enough samples to be more than a reduced-window proxy.
    """

    mean_limit = float(max_mean_rel_delta)
    cv_limit = float(max_cv)
    trend_limit = float(max_abs_trend)
    sample_floor = int(min_samples)
    if mean_limit < 0.0 or cv_limit < 0.0 or trend_limit < 0.0:
        raise ValueError("heat-flux convergence thresholds must be non-negative")
    if sample_floor <= 0:
        raise ValueError("min_samples must be positive")

    gates = (
        _upper_limit_gate(
            "heat_flux_terminal_mean_rel_delta",
            metrics.mean_rel_delta,
            mean_limit,
            notes=f"Passes when terminal-window mean differs by <= {mean_limit:.6g}.",
        ),
        _upper_limit_gate(
            "heat_flux_window_cv",
            metrics.heat_flux_cv,
            cv_limit,
            notes=f"Passes when post-transient heat-flux CV <= {cv_limit:.6g}.",
        ),
        _upper_limit_gate(
            "heat_flux_window_abs_trend",
            metrics.abs_trend,
            trend_limit,
            notes=f"Passes when normalized drift across the window <= {trend_limit:.6g}.",
        ),
        _upper_limit_gate(
            "heat_flux_window_sample_deficit",
            max(0.0, float(sample_floor - int(metrics.nsamples))),
            0.0,
            notes=f"Passes when post-transient window has at least {sample_floor} samples.",
        ),
    )
    return gate_report(case, source, gates)


def zonal_response_gate_report(
    observed: ZonalFlowResponseMetrics,
    reference: ZonalFlowResponseMetrics,
    *,
    case: str,
    source: str,
    residual_atol: float,
    residual_rtol: float = 0.0,
    frequency_atol: float,
    frequency_rtol: float = 0.0,
    damping_atol: float,
    damping_rtol: float = 0.0,
) -> GateReport:
    """Gate Rosenbluth-Hinton/GAM-style response observables."""

    return gate_report(
        case,
        source,
        (
            evaluate_scalar_gate(
                "residual_level",
                observed.residual_level,
                reference.residual_level,
                atol=residual_atol,
                rtol=residual_rtol,
            ),
            evaluate_scalar_gate(
                "gam_frequency",
                observed.gam_frequency,
                reference.gam_frequency,
                atol=frequency_atol,
                rtol=frequency_rtol,
                units="v_t/R",
            ),
            evaluate_scalar_gate(
                "gam_damping_rate",
                observed.gam_damping_rate,
                reference.gam_damping_rate,
                atol=damping_atol,
                rtol=damping_rtol,
                units="v_t/R",
            ),
        ),
    )


def eigenfunction_gate_report(
    comparison: EigenfunctionComparisonMetrics,
    *,
    case: str,
    source: str,
    min_overlap: float = 0.95,
    max_relative_l2: float = 0.25,
) -> GateReport:
    """Gate a phase-aligned eigenfunction comparison.

    The ideal reference is overlap equal to one and relative L2 mismatch equal
    to zero. ``min_overlap`` and ``max_relative_l2`` make the acceptance policy
    explicit for manuscript overlays and branch-identity checks.
    """

    min_overlap_f = float(min_overlap)
    max_relative_l2_f = float(max_relative_l2)
    if not 0.0 <= min_overlap_f <= 1.0:
        raise ValueError("min_overlap must be in [0, 1]")
    if max_relative_l2_f < 0.0:
        raise ValueError("max_relative_l2 must be non-negative")
    return gate_report(
        case,
        source,
        (
            evaluate_scalar_gate(
                "eigenfunction_overlap",
                comparison.overlap,
                1.0,
                atol=1.0 - min_overlap_f,
                rtol=0.0,
                notes=f"Passes when overlap >= {min_overlap_f:.6g}.",
            ),
            _upper_limit_gate(
                "eigenfunction_relative_l2",
                comparison.relative_l2,
                max_relative_l2_f,
                notes=f"Passes when relative L2 <= {max_relative_l2_f:.6g}.",
            ),
        ),
    )


def observed_order_gate_report(
    metrics: ObservedOrderMetrics,
    *,
    case: str,
    source: str,
    min_asymptotic_order: float,
    min_pairwise_order: float | None = None,
    max_final_error: float | None = None,
    order_atol: float = 1.0e-12,
) -> GateReport:
    """Gate an observed-order convergence study.

    ``min_asymptotic_order`` encodes the expected method/order floor for the
    finest refinement pair. ``min_pairwise_order`` can additionally require the
    whole table to be monotone enough for publication use. ``max_final_error``
    can be used when both rate and absolute accuracy matter.
    """

    min_order = float(min_asymptotic_order)
    order_tol = float(order_atol)
    if min_order < 0.0 or order_tol < 0.0:
        raise ValueError("min_asymptotic_order and order_atol must be non-negative")
    gates = [
        _upper_limit_gate(
            "observed_order_deficit",
            max(0.0, min_order - float(metrics.asymptotic_order)),
            order_tol,
            notes=f"Passes when asymptotic observed order >= {min_order:.6g}.",
        )
    ]
    if min_pairwise_order is not None:
        min_pair_order = float(min_pairwise_order)
        if min_pair_order < 0.0:
            raise ValueError("min_pairwise_order must be non-negative")
        gates.append(
            _upper_limit_gate(
                "min_pairwise_order_deficit",
                max(0.0, min_pair_order - float(np.min(metrics.orders))),
                order_tol,
                notes=f"Passes when every pairwise observed order >= {min_pair_order:.6g}.",
            )
        )
    if max_final_error is not None:
        final_error_limit = float(max_final_error)
        if final_error_limit < 0.0:
            raise ValueError("max_final_error must be non-negative")
        gates.append(
            _upper_limit_gate(
                "final_error",
                float(metrics.errors[-1]),
                final_error_limit,
                notes=f"Passes when final-grid error <= {final_error_limit:.6g}.",
            )
        )
    return gate_report(case, source, gates)


def branch_continuity_gate_report(
    metrics: BranchContinuationMetrics,
    *,
    case: str,
    source: str,
    max_rel_gamma_jump: float,
    max_rel_omega_jump: float,
    min_successive_overlap: float | None = None,
) -> GateReport:
    """Gate branch-continuation diagnostics for branch-followed scans."""

    gamma_limit = float(max_rel_gamma_jump)
    omega_limit = float(max_rel_omega_jump)
    if gamma_limit < 0.0 or omega_limit < 0.0:
        raise ValueError("maximum relative jumps must be non-negative")
    gates = [
        _upper_limit_gate(
            "max_rel_gamma_jump",
            float(metrics.max_rel_gamma_jump),
            gamma_limit,
            notes=f"Passes when adjacent gamma jumps <= {gamma_limit:.6g}.",
        ),
        _upper_limit_gate(
            "max_rel_omega_jump",
            float(metrics.max_rel_omega_jump),
            omega_limit,
            notes=f"Passes when adjacent omega jumps <= {omega_limit:.6g}.",
        ),
    ]
    if min_successive_overlap is not None:
        min_overlap = float(min_successive_overlap)
        if not 0.0 <= min_overlap <= 1.0:
            raise ValueError("min_successive_overlap must be in [0, 1]")
        observed = float("nan") if metrics.min_successive_overlap is None else float(metrics.min_successive_overlap)
        gates.append(
            _upper_limit_gate(
                "successive_overlap_deficit",
                max(0.0, min_overlap - observed) if np.isfinite(observed) else float("nan"),
                0.0,
                notes=f"Passes when successive eigenfunction overlap >= {min_overlap:.6g}.",
            )
        )
    return gate_report(case, source, gates)



# Optional hooks preserve benchmark-harness monkeypatch seams without importing
# diagnostics.analysis during validation-gates import.
extract_mode_time_series = None
fit_growth_rate = None


def _metric_extract_mode_time_series(*args, **kwargs):
    extractor = extract_mode_time_series
    if extractor is None:
        from spectraxgk.diagnostics.modes import extract_mode_time_series as extractor
    return extractor(*args, **kwargs)


def _metric_fit_growth_rate(*args, **kwargs):
    fitter = fit_growth_rate
    if fitter is None:
        from spectraxgk.diagnostics.growth_fit import fit_growth_rate as fitter
    return fitter(*args, **kwargs)


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
        series = _metric_extract_mode_time_series(
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
    gamma_fit, omega_fit = _metric_fit_growth_rate(t_tail, signal_tail)
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


# Diagnostic artifact loading and time-window helpers used by validation gates.
def _decode_netcdf_values(var) -> np.ndarray:
    raw = np.asarray(var[:])
    dims = tuple(getattr(var, "dimensions", ()))
    if dims and dims[-1] == "ri":
        return np.asarray(raw[..., 0] + 1j * raw[..., 1], dtype=np.complex128)
    return raw


def _extract_diagnostic_values(
    values: np.ndarray,
    *,
    variable: str,
    kx_index: int | None,
) -> np.ndarray:
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        if kx_index is None:
            raise ValueError(
                f"diagnostics variable {variable!r} requires kx_index for 2D extraction"
            )
        return values[:, int(kx_index)]
    raise ValueError(
        f"diagnostics variable {variable!r} must reduce to a 1D time series"
    )


def _load_netcdf_time_axis(
    ds,
    *,
    src: Path,
    time_group: str,
    time_var: str,
) -> np.ndarray:
    if time_group in ds.groups and time_var in ds.groups[time_group].variables:
        return np.asarray(ds.groups[time_group].variables[time_var][:], dtype=float)
    if time_var in ds.variables:
        return np.asarray(ds.variables[time_var][:], dtype=float)
    raise ValueError(f"missing time variable {time_group}/{time_var} in {src}")


def _load_diagnostic_variable(
    ds,
    *,
    src: Path,
    diagnostics_group: str,
    variable: str,
    kx_index: int | None,
) -> np.ndarray:
    diag_group = ds.groups.get(diagnostics_group)
    if diag_group is None:
        raise ValueError(f"missing NetCDF group {diagnostics_group!r} in {src}")
    if variable not in diag_group.variables:
        raise ValueError(f"missing diagnostics variable {variable!r} in {src}")
    raw = _decode_netcdf_values(diag_group.variables[variable])
    return _extract_diagnostic_values(raw, variable=variable, kx_index=kx_index)


def _align_complex_phase(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    nz = finite & (np.abs(values) > 1.0e-30)
    if np.any(nz):
        first = values[np.flatnonzero(nz)[0]]
        return values * np.exp(-1j * np.angle(first))
    return values


def _select_complex_component(
    values: np.ndarray,
    *,
    component: str,
    align_phase: bool,
) -> np.ndarray:
    values_arr = _align_complex_phase(values) if align_phase else values
    component_key = str(component).lower()
    if component_key == "complex":
        return values_arr
    if component_key == "real":
        return np.real(values_arr)
    if component_key == "imag":
        return np.imag(values_arr)
    if component_key == "abs":
        return np.abs(values_arr)
    raise ValueError("component must be one of {'real', 'imag', 'abs', 'complex'}")


def _select_real_component(values: np.ndarray, *, component: str) -> np.ndarray:
    if component not in {"real", "abs"}:
        raise ValueError("real diagnostics only support component='real' or 'abs'")
    if component == "abs":
        return np.abs(values)
    return np.asarray(values, dtype=float)


def _select_series_component(
    values: np.ndarray,
    *,
    component: str,
    align_phase: bool,
) -> np.ndarray:
    values_arr = np.asarray(values)
    if np.iscomplexobj(values_arr):
        return _select_complex_component(
            values_arr,
            component=component,
            align_phase=align_phase,
        )
    return _select_real_component(values_arr, component=component)


def load_diagnostic_time_series(
    path: str | Path,
    *,
    variable: str,
    diagnostics_group: str = "Diagnostics",
    time_group: str = "Grids",
    time_var: str = "time",
    kx_index: int | None = None,
    component: str = "real",
    align_phase: bool = False,
) -> DiagnosticTimeSeries:
    """Load a 1D diagnostics time series from a grouped NetCDF output artifact."""

    src = Path(path)
    import netCDF4 as nc

    with nc.Dataset(src) as ds:
        values = _load_diagnostic_variable(
            ds,
            src=src,
            diagnostics_group=diagnostics_group,
            variable=variable,
            kx_index=kx_index,
        )
        t = _load_netcdf_time_axis(
            ds,
            src=src,
            time_group=time_group,
            time_var=time_var,
        )

    if t.ndim != 1 or t.size != values.size:
        raise ValueError(
            f"time axis for {variable!r} must be one-dimensional and match the diagnostics length"
        )
    selected = _select_series_component(
        values,
        component=component,
        align_phase=align_phase,
    )

    return DiagnosticTimeSeries(
        t=t,
        values=np.asarray(selected),
        variable=str(variable),
        source_path=str(src),
    )


def _tail_window(
    t: np.ndarray, tail_fraction: float
) -> tuple[np.ndarray, float | None, float | None]:
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional")
    if t.size == 0:
        raise ValueError("t must be non-empty")
    start = max(0, int(np.floor((1.0 - float(tail_fraction)) * t.size)))
    mask = np.zeros_like(t, dtype=bool)
    mask[start:] = True
    if not np.any(mask):
        mask[-1] = True
    tt = np.asarray(t[mask], dtype=float)
    return mask, float(tt[0]) if tt.size else None, float(tt[-1]) if tt.size else None


def late_time_window(
    t: np.ndarray, *, tail_fraction: float = 0.4
) -> tuple[float, float]:
    """Return the start/end of a late-time tail window.

    This is the windowing convention used for manuscript-facing eigenfunction
    extraction when the growth-rate fit window is not the same object as the
    late-time mode-shape window.
    """

    _mask, tmin, tmax = _tail_window(np.asarray(t, dtype=float), float(tail_fraction))
    if tmin is None or tmax is None:
        raise ValueError("late-time window requires a non-empty time axis")
    return float(tmin), float(tmax)


def infer_triple_dealiased_ny(nky_positive: int) -> int:
    """Infer the full ``Ny`` from the number of positive ``k_y`` points.

    Reference real-FFT outputs typically store only the non-negative
    ``k_y`` branch. For the linked-boundary spectral grid used here, the
    corresponding real-space ``Ny`` follows ``Ny = 3 * (nky - 1) + 1``.
    """

    nky = int(nky_positive)
    if nky < 2:
        raise ValueError("nky_positive must be >= 2")
    return 3 * (nky - 1) + 1


def _tail_stats(arr: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(arr, dtype=float)[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _leading_window(
    t: np.ndarray,
    lead_fraction: float,
) -> tuple[np.ndarray, float | None, float | None]:
    if not 0.0 < float(lead_fraction) <= 1.0:
        raise ValueError("lead_fraction must be in (0, 1]")
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional")
    if t.size == 0:
        raise ValueError("t must be non-empty")
    stop = max(1, int(np.ceil(float(lead_fraction) * t.size)))
    mask = np.zeros_like(t, dtype=bool)
    mask[:stop] = True
    tt = np.asarray(t[mask], dtype=float)
    return mask, float(tt[0]) if tt.size else None, float(tt[-1]) if tt.size else None


def _explicit_time_window(
    t: np.ndarray,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
) -> tuple[np.ndarray, float, float]:
    mask = np.ones_like(t, dtype=bool)
    if tmin is not None:
        mask &= t >= float(tmin)
    if tmax is not None:
        mask &= t <= float(tmax)
    if not np.any(mask):
        raise ValueError("explicit fit window is empty")
    tt = np.asarray(t[mask], dtype=float)
    return mask, float(tt[0]), float(tt[-1])


def _analytic_signal(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    spec = np.fft.fft(x)
    filt = np.zeros(x.size, dtype=float)
    if x.size % 2 == 0:
        filt[0] = 1.0
        filt[x.size // 2] = 1.0
        filt[1 : x.size // 2] = 2.0
    else:
        filt[0] = 1.0
        filt[1 : (x.size + 1) // 2] = 2.0
    return np.fft.ifft(spec * filt)



__all__ = [
    "branch_continuity_metrics",
    "estimate_observed_order",
    "late_time_linear_metrics",
    "nonlinear_heat_flux_convergence_metrics",
    "windowed_nonlinear_metrics",
    "_analytic_signal",
    "_explicit_time_window",
    "_leading_window",
    "_tail_stats",
    "_tail_window",
    "infer_triple_dealiased_ny",
    "late_time_window",
    "load_diagnostic_time_series",
    "BranchContinuationMetrics",
    "DiagnosticTimeSeries",
    "EigenfunctionComparisonMetrics",
    "EigenfunctionReferenceBundle",
    "GateReport",
    "LateTimeLinearMetrics",
    "NonlinearHeatFluxConvergenceMetrics",
    "NonlinearWindowMetrics",
    "ObservedOrderMetrics",
    "ScalarGateResult",
    "ZonalFlowResponseMetrics",
    "branch_continuity_gate_report",
    "eigenfunction_gate_report",
    "evaluate_scalar_gate",
    "gate_report",
    "gate_report_to_dict",
    "linear_metrics_gate_report",
    "nonlinear_heat_flux_convergence_gate_report",
    "nonlinear_window_gate_report",
    "observed_order_gate_report",
    "zonal_response_gate_report",
]
