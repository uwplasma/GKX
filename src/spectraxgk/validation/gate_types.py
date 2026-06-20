"""Typed metric containers used by validation gate reports."""

from __future__ import annotations

from dataclasses import dataclass

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



__all__ = [
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
]
