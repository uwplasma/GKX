"""Validation gate metrics and report builders for diagnostics artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gkx.diagnostics.analysis import (
    BranchContinuationMetrics,
    LateTimeLinearMetrics,
    NonlinearHeatFluxConvergenceMetrics,
    NonlinearWindowMetrics,
    ObservedOrderMetrics,
)
from gkx.diagnostics.modes import EigenfunctionComparisonMetrics
import jax.numpy as jnp
from gkx.core_grid import SpectralGrid
from gkx.core_ky_layout import (
    KY_AXIS,
    _xp,
    symmetrize_self_conjugate_rows,
    to_full,
    to_half,
)
from gkx.operators.fluxes import turbulent_heating_species
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.dissipation import _species_collision_frequency
from gkx.operators.linear.params import LinearParams
from gkx.operators.moments import (
    _heat_flux_channel_contrib_species,
    _particle_flux_channel_contrib_species,
)
from gkx.operators.nonlinear.brackets import (
    _apply_mask_xy,
    _spectral_bracket,
    _spectral_bracket_multi_full,
    _spectral_bracket_multi_real_fft,
)
from scripts.checks._gates.transport_windows import (
    nonlinear_window_stats_promotion_ready,
)


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
    # Window the damping was actually fitted over. It is not always fit_tmin /
    # fit_tmax: period_rms_envelope states its window in GAM periods, so a
    # gated damping rate can be checked against the window it came from.
    damping_fit_tmin: float = float("nan")
    damping_fit_tmax: float = float("nan")


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
    abs_error = (
        float(abs(obs - ref)) if np.isfinite(obs) and np.isfinite(ref) else float("inf")
    )
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
    max_corrected_rel_stderr: float = 0.25,
) -> GateReport:
    """Gate post-transient heat-flux averaging stability.

    This is an internal promotion gate for nonlinear transport claims: the
    post-transient average must agree with its terminal subwindow, have bounded
    coefficient of variation, show limited normalized drift across the window,
    and contain enough samples to be more than a reduced-window proxy.

    The added gate is on ``sigma / (mean * sqrt(n_eff))``, not on ``n_eff``
    alone. Turbulence outputs are correlated, so a floor on ``nsamples`` passes
    windows whose mean rests on a handful of independent events. But a floor on
    ``n_eff`` alone is also wrong -- a very smooth trace is maximally
    autocorrelated and so has small ``n_eff`` while its mean is extremely well
    determined -- and would fail exactly the converged windows this gate exists
    to accept. The relative standard error combines both effects correctly.
    """

    mean_limit = float(max_mean_rel_delta)
    cv_limit = float(max_cv)
    trend_limit = float(max_abs_trend)
    sample_floor = int(min_samples)
    stderr_limit = float(max_corrected_rel_stderr)
    n_eff = float(getattr(metrics, "n_eff", 0.0))
    mean = abs(float(metrics.heat_flux_mean))
    corrected_rel_stderr = (
        float(metrics.heat_flux_std) / (mean * np.sqrt(n_eff))
        if n_eff > 0.0 and mean
        else float("inf")
    )
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
        _upper_limit_gate(
            "heat_flux_corrected_rel_stderr",
            corrected_rel_stderr,
            stderr_limit,
            notes=(
                f"Passes when sigma / (mean * sqrt(n_eff)) <= {stderr_limit:.6g}: "
                "a long window of correlated outputs must not read as precise."
            ),
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
        observed = (
            float("nan")
            if metrics.min_successive_overlap is None
            else float(metrics.min_successive_overlap)
        )
        gates.append(
            _upper_limit_gate(
                "successive_overlap_deficit",
                max(0.0, min_overlap - observed)
                if np.isfinite(observed)
                else float("nan"),
                0.0,
                notes=f"Passes when successive eigenfunction overlap >= {min_overlap:.6g}.",
            )
        )
    return gate_report(case, source, gates)


def _transport_stat(report: dict[str, Any], key: str) -> float | None:
    stats = report.get("statistics")
    if not isinstance(stats, dict):
        return None
    try:
        value = float(stats[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def matched_nonlinear_transport_report(
    baseline: dict[str, Any],
    treatment: dict[str, Any],
    *,
    case: str = "matched_nonlinear_transport",
    treatment_name: str = "treatment",
    min_relative_reduction: float = 0.0,
    min_uncertainty_z_score: float = 0.0,
    value_floor: float = 1.0e-12,
) -> dict[str, Any]:
    """Compare two independently converged post-transient transport windows."""

    if min_relative_reduction < 0.0 or min_uncertainty_z_score < 0.0:
        raise ValueError("matched transport thresholds must be non-negative")
    if value_floor <= 0.0:
        raise ValueError("value_floor must be positive")
    baseline_ready, baseline_failures = nonlinear_window_stats_promotion_ready(baseline)
    treatment_ready, treatment_failures = nonlinear_window_stats_promotion_ready(
        treatment
    )
    baseline_mean = _transport_stat(baseline, "late_mean")
    treatment_mean = _transport_stat(treatment, "late_mean")
    baseline_sem = _transport_stat(baseline, "sem")
    treatment_sem = _transport_stat(treatment, "sem")
    finite = all(
        value is not None
        for value in (baseline_mean, treatment_mean, baseline_sem, treatment_sem)
    )
    reduction = separation = None
    if finite:
        assert baseline_mean is not None and treatment_mean is not None
        assert baseline_sem is not None and treatment_sem is not None
        difference = baseline_mean - treatment_mean
        reduction = difference / max(abs(baseline_mean), value_floor)
        separation = difference / max(
            float(np.hypot(baseline_sem, treatment_sem)), value_floor
        )
    gate_specs = (
        ("baseline_window_passed", baseline_ready, baseline_failures),
        ("treatment_window_passed", treatment_ready, treatment_failures),
        ("finite_transport_statistics", finite, "means and SEMs must be finite"),
        (
            "relative_reduction",
            reduction is not None and reduction >= min_relative_reduction,
            f"value={reduction} gate={min_relative_reduction}",
        ),
        (
            "uncertainty_separation",
            separation is not None and separation >= min_uncertainty_z_score,
            f"value={separation} gate={min_uncertainty_z_score}",
        ),
    )
    gates = [
        {"metric": metric, "passed": bool(passed), "detail": str(detail)}
        for metric, passed, detail in gate_specs
    ]
    passed = all(gate["passed"] for gate in gates)
    return {
        "kind": "matched_nonlinear_transport_comparison",
        "claim_level": "matched_post_transient_transport_comparison",
        "case": case,
        "treatment": treatment_name,
        "passed": passed,
        "statistics": {
            "baseline_mean": baseline_mean,
            "treatment_mean": treatment_mean,
            "baseline_sem": baseline_sem,
            "treatment_sem": treatment_sem,
            "relative_reduction": reduction,
            "uncertainty_z_score": separation,
        },
        "gates": gates,
        "gate_report": {
            "case": case,
            "source": "matched_nonlinear_window_reports",
            "passed": passed,
            "max_abs_error": 0.0 if passed else 1.0,
            "max_rel_error": 0.0 if passed else 1.0,
            "gates": gates,
        },
        "baseline": baseline,
        "treatment_window": treatment,
        "windows_ready": baseline_ready and treatment_ready,
        "config": {
            "min_relative_reduction": min_relative_reduction,
            "min_uncertainty_z_score": min_uncertainty_z_score,
            "value_floor": value_floor,
        },
    }


__all__ = [
    "GateReport",
    "ScalarGateResult",
    "ZonalFlowResponseMetrics",
    "branch_continuity_gate_report",
    "eigenfunction_gate_report",
    "evaluate_scalar_gate",
    "gate_report",
    "gate_report_to_dict",
    "linear_metrics_gate_report",
    "matched_nonlinear_transport_report",
    "nonlinear_heat_flux_convergence_gate_report",
    "nonlinear_window_gate_report",
    "observed_order_gate_report",
    "zonal_response_gate_report",
]


# Reference kernels the test suite checks the solver against.


def drift_kinetic_dougherty_contribution(
    state: jnp.ndarray,
    *,
    nu: jnp.ndarray,
    weight: jnp.ndarray = jnp.asarray(1.0),
) -> jnp.ndarray:
    """Apply the linearized drift-kinetic Dougherty moment operator.

    This is Appendix C, equation (C6), of Frei, Hoffmann & Ricci (2022),
    mapped to GKX's ``(species, ell, m, ky, kx, z)`` ordering and
    Laguerre-sign convention. The density and parallel-flow moments and the
    combined thermal moment ``sqrt(2) G[0, 2] + 2 G[1, 0]`` are exact null
    directions. Five-dimensional single-species states are also accepted.

    The kernel is both an independently auditable reference and a usable
    long-wavelength collision operator. It is not the finite-Larmor-radius
    Sugama or Coulomb operator.
    """

    value = jnp.asarray(state)
    if value.ndim not in {5, 6}:
        raise ValueError("collision state must have five or six dimensions")
    expanded = value[None, ...] if value.ndim == 5 else value
    if expanded.shape[1] < 2 or expanded.shape[2] < 3:
        raise ValueError("drift-kinetic Dougherty requires Nl >= 2 and Nm >= 3")

    ns, nl, nm = map(int, expanded.shape[:3])
    real_dtype = jnp.real(expanded).dtype
    nu_s = _species_collision_frequency(nu, ns=ns, dtype=real_dtype)
    rate = nu_s[:, None, None, None, None, None]
    ell = jnp.arange(nl, dtype=real_dtype)[None, :, None, None, None, None]
    hermite = jnp.arange(nm, dtype=real_dtype)[None, None, :, None, None, None]
    contribution = -rate * (2.0 * ell + hermite) * expanded

    temperature = (
        jnp.sqrt(jnp.asarray(2.0, dtype=real_dtype)) * expanded[:, 0, 2]
        + 2.0 * expanded[:, 1, 0]
    ) / 3.0
    spatial_rate = nu_s[(slice(None),) + (None,) * (temperature.ndim - 1)]
    contribution = contribution.at[:, 0, 1].add(spatial_rate * expanded[:, 0, 1])
    contribution = contribution.at[:, 0, 2].add(
        spatial_rate * jnp.sqrt(jnp.asarray(2.0, dtype=real_dtype)) * temperature
    )
    contribution = contribution.at[:, 1, 0].add(2.0 * spatial_rate * temperature)
    result = jnp.asarray(weight, dtype=real_dtype) * contribution
    return result[0] if value.ndim == 5 else result


def heat_flux_channel_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ES, Apar, and Bpar heat-flux channels per species."""

    es_contrib, apar_contrib, bpar_contrib = _heat_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return (
        jnp.sum(es_contrib, axis=(1, 2, 3)),
        jnp.sum(apar_contrib, axis=(1, 2, 3)),
        jnp.sum(bpar_contrib, axis=(1, 2, 3)),
    )


def particle_flux_channel_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ES, Apar, and Bpar particle-flux channels per species."""

    es_contrib, apar_contrib, bpar_contrib = _particle_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return (
        jnp.sum(es_contrib, axis=(1, 2, 3)),
        jnp.sum(apar_contrib, axis=(1, 2, 3)),
        jnp.sum(bpar_contrib, axis=(1, 2, 3)),
    )


def turbulent_heating_total(
    G: jnp.ndarray,
    G_old: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    phi_old: jnp.ndarray,
    apar_old: jnp.ndarray,
    bpar_old: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    dt: jnp.ndarray | float,
    *,
    use_dealias: bool = True,
) -> jnp.ndarray:
    """Total turbulent-heating diagnostic."""

    return jnp.sum(
        turbulent_heating_species(
            G,
            G_old,
            phi,
            apar,
            bpar,
            phi_old,
            apar_old,
            bpar_old,
            cache,
            grid,
            params,
            vol_fac,
            dt,
            use_dealias=use_dealias,
        )
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


def exb_nonlinear_contribution(
    G: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    weight: jnp.ndarray,
    compressed_real_fft: bool = True,
    ny_full: int | None = None,
    radial_phase: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return the nonlinear E×B contribution using a pseudospectral bracket."""
    phi = _apply_mask_xy(phi, dealias_mask)
    bracket_hat = _spectral_bracket(
        G,
        phi,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=jnp.asarray(1.0),
        radial_phase=radial_phase,
        compressed_real_fft=compressed_real_fft,
        ny_full=ny_full,
    )
    real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
    return jnp.asarray(weight, dtype=real_dtype) * bracket_hat


def reality_residual(state: Any, *, ny_full: int | None = None) -> Any:
    """Return ``max|F - H(F)| / max|F|`` for a two-sided array.

    ``H`` rebuilds the array from its own ``ky >= 0`` rows, so the residual is
    zero exactly when the stored negative rows agree with the reality
    condition.  The self-conjugate rows are included: their internal constraint
    is part of ``H``.
    """

    xp = _xp(state)
    rows = int(state.shape[KY_AXIS])
    ny = rows if ny_full is None else int(ny_full)
    if rows != ny:
        raise ValueError(
            f"reality_residual needs the full ky axis; got {rows} rows for ny_full={ny}"
        )
    rebuilt = to_full(
        symmetrize_self_conjugate_rows(to_half(state, ny_full=ny), ny_full=ny),
        ny_full=ny,
    )
    scale = xp.max(xp.abs(state))
    denominator = xp.where(scale > 0, scale, xp.ones_like(scale))
    return xp.max(xp.abs(state - rebuilt)) / denominator


def single_precision_factorial(m: jnp.ndarray) -> jnp.ndarray:
    """Return the single-precision factorial approximation."""

    m_arr = jnp.asarray(m)
    dtype = m_arr.dtype
    exact = jnp.asarray([1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0], dtype=dtype)
    m_int = m_arr.astype(jnp.int32)
    m_clamped = jnp.clip(m_int, 0, exact.shape[0] - 1)
    m_safe = jnp.where(m_arr > 0, m_arr, jnp.asarray(1.0, dtype=dtype))
    stirling = (
        jnp.sqrt(2.0 * jnp.asarray(jnp.pi, dtype=dtype) * m_safe)
        * (m_safe**m_safe)
        * jnp.exp(-m_safe)
        * (1.0 + 1.0 / (12.0 * m_safe) + 1.0 / (288.0 * m_safe * m_safe))
    )
    return jnp.where(m_int <= 6, exact[m_clamped], stirling)


def _spectral_bracket_multi(
    G_hat: jnp.ndarray,
    chi_hat_stack: jnp.ndarray,
    *,
    compressed_real_fft: bool = True,
    **kwargs,
) -> jnp.ndarray:
    kernel = (
        _spectral_bracket_multi_real_fft
        if compressed_real_fft
        else _spectral_bracket_multi_full
    )
    return kernel(G_hat, chi_hat_stack, **kwargs)
