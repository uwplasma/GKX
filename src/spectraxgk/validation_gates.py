"""Validation gate primitives for benchmark and manuscript artifacts."""

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
            evaluate_scalar_gate(
                "eigenfunction_relative_l2",
                comparison.relative_l2,
                0.0,
                atol=max_relative_l2_f,
                rtol=0.0,
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
        evaluate_scalar_gate(
            "observed_order_deficit",
            max(0.0, min_order - float(metrics.asymptotic_order)),
            0.0,
            atol=order_tol,
            rtol=0.0,
            notes=f"Passes when asymptotic observed order >= {min_order:.6g}.",
        )
    ]
    if min_pairwise_order is not None:
        min_pair_order = float(min_pairwise_order)
        if min_pair_order < 0.0:
            raise ValueError("min_pairwise_order must be non-negative")
        gates.append(
            evaluate_scalar_gate(
                "min_pairwise_order_deficit",
                max(0.0, min_pair_order - float(np.min(metrics.orders))),
                0.0,
                atol=order_tol,
                rtol=0.0,
                notes=f"Passes when every pairwise observed order >= {min_pair_order:.6g}.",
            )
        )
    if max_final_error is not None:
        final_error_limit = float(max_final_error)
        if final_error_limit < 0.0:
            raise ValueError("max_final_error must be non-negative")
        gates.append(
            evaluate_scalar_gate(
                "final_error",
                float(metrics.errors[-1]),
                0.0,
                atol=final_error_limit,
                rtol=0.0,
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
        evaluate_scalar_gate(
            "max_rel_gamma_jump",
            float(metrics.max_rel_gamma_jump),
            0.0,
            atol=gamma_limit,
            rtol=0.0,
            notes=f"Passes when adjacent gamma jumps <= {gamma_limit:.6g}.",
        ),
        evaluate_scalar_gate(
            "max_rel_omega_jump",
            float(metrics.max_rel_omega_jump),
            0.0,
            atol=omega_limit,
            rtol=0.0,
            notes=f"Passes when adjacent omega jumps <= {omega_limit:.6g}.",
        ),
    ]
    if min_successive_overlap is not None:
        min_overlap = float(min_successive_overlap)
        if not 0.0 <= min_overlap <= 1.0:
            raise ValueError("min_successive_overlap must be in [0, 1]")
        observed = float("nan") if metrics.min_successive_overlap is None else float(metrics.min_successive_overlap)
        gates.append(
            evaluate_scalar_gate(
                "successive_overlap_deficit",
                max(0.0, min_overlap - observed) if np.isfinite(observed) else float("nan"),
                0.0,
                atol=0.0,
                rtol=0.0,
                notes=f"Passes when successive eigenfunction overlap >= {min_overlap:.6g}.",
            )
        )
    return gate_report(case, source, gates)


__all__ = [
    "BranchContinuationMetrics",
    "DiagnosticTimeSeries",
    "EigenfunctionComparisonMetrics",
    "EigenfunctionReferenceBundle",
    "GateReport",
    "LateTimeLinearMetrics",
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
    "nonlinear_window_gate_report",
    "observed_order_gate_report",
    "zonal_response_gate_report",
]
