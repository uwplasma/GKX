"""Reduced differentiable QA/ITG contracts, model, and objective gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.objectives.autodiff_validation import (
    autodiff_finite_difference_report,
    covariance_diagnostics,
)
from spectraxgk.objectives.stellarator import (
    PARAMETER_NAMES,
    _validate_params,
    smooth_positive,
)


QA_LOW_TURBULENCE_DESIGN_NAMES = (
    "qa_constraints",
    "qa_plus_nonlinear_heat_flux",
)

QA_LOW_TURBULENCE_OBSERVABLE_NAMES = (
    "aspect",
    "mean_iota",
    "iota_floor_violation",
    "iota_operating_floor_violation",
    "qa_residual",
    "growth_rate",
    "kperp_eff2",
    "linear_heat_flux_weight",
    "quasilinear_heat_flux",
    "nonlinear_heat_flux_mean",
    "nonlinear_heat_flux_cv",
    "nonlinear_heat_flux_trend",
)


@dataclass(frozen=True)
class QALowTurbulenceConfig:
    """Configuration for the reduced QA low-turbulence comparison."""

    target_aspect: float = 6.0
    min_iota: float = 0.41
    iota_operating_floor: float = 0.70
    max_mode: int = 1
    aspect_weight: float = 8.0
    iota_floor_weight: float = 160.0
    iota_operating_weight: float = 70.0
    qa_weight: float = 8.0
    target_helical_amplitude: float = 0.16
    helical_shaping_weight: float = 24.0
    regularization: float = 2.0e-3
    nonlinear_weight: float = 8.0
    learning_rate: float = 0.030
    steps: int = 60
    nonlinear_dt: float = 0.20
    nonlinear_steps: int = 2000
    nonlinear_tail_fraction: float = 0.50
    long_window_min_time: float = 300.0
    long_window_max_cv: float = 0.03
    long_window_max_trend: float = 0.02
    long_window_max_half_mean_rel_change: float = 0.02
    fixed_density_gradient: float = 2.2
    fixed_temperature_gradient: float = 6.0
    scan_density_gradients: tuple[float, ...] = (
        0.6,
        1.0,
        1.4,
        1.8,
        2.2,
        2.8,
        3.4,
        4.0,
        4.8,
    )
    fd_step: float = 1.0e-4
    surface_ntheta: int = 72
    surface_nzeta: int = 72
    n_field_periods: int = 2


@dataclass(frozen=True)
class QALowTurbulenceResult:
    """JSON-ready result for one reduced QA optimization."""

    design_name: str
    includes_nonlinear_heat_flux: bool
    parameter_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    initial_params: tuple[float, ...]
    final_params: tuple[float, ...]
    initial_objective: float
    final_objective: float
    initial_observables: tuple[float, ...]
    final_observables: tuple[float, ...]
    history: tuple[dict[str, Any], ...]
    residual_gradient_gate: dict[str, Any]
    scalar_gradient_gate: dict[str, Any]
    observable_gradient_gate: dict[str, Any]
    covariance: dict[str, Any]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-friendly payload."""

        return {
            "design_name": self.design_name,
            "includes_nonlinear_heat_flux": self.includes_nonlinear_heat_flux,
            "parameter_names": list(self.parameter_names),
            "observable_names": list(self.observable_names),
            "initial_params": list(self.initial_params),
            "final_params": list(self.final_params),
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "initial_observables": list(self.initial_observables),
            "final_observables": list(self.final_observables),
            "history": list(self.history),
            "residual_gradient_gate": self.residual_gradient_gate,
            "scalar_gradient_gate": self.scalar_gradient_gate,
            "observable_gradient_gate": self.observable_gradient_gate,
            "covariance": self.covariance,
            "config": self.config,
            "claim_level": (
                "reduced_differentiable_qa_low_turbulence_comparison_"
                "not_full_vmec_nonlinear_transport_optimization"
            ),
        }


def default_qa_low_turbulence_initial_params() -> jnp.ndarray:
    """Return the shared off-optimum QA seed for the comparison."""

    return jnp.asarray([0.24, 0.34, 0.30, -0.22])


def _fd_gate_tolerances(fd_step: float) -> tuple[float, float, float]:
    if bool(jax.config.read("jax_enable_x64")):
        return float(fd_step), 5.0e-3, 7.0e-4
    return max(float(fd_step), 1.0e-3), 8.0e-2, 8.0e-3


def _qa_low_turbulence_core(
    params: jnp.ndarray | Sequence[float],
    config: QALowTurbulenceConfig | None = None,
) -> dict[str, jnp.ndarray]:
    """Return smooth reduced QA/ITG features for the aspect-6 comparison."""

    cfg = config or QALowTurbulenceConfig()
    p = _validate_params(params)
    minor_shift, elong_shift, ripple, shear_shift = p
    dtype = p.dtype
    target_aspect = jnp.asarray(cfg.target_aspect, dtype=dtype)
    min_iota = jnp.asarray(cfg.min_iota, dtype=dtype)
    operating_iota = jnp.asarray(cfg.iota_operating_floor, dtype=dtype)

    aspect = target_aspect * jnp.exp(
        -0.42 * minor_shift + 0.050 * elong_shift**2 + 0.035 * ripple**2
    )
    target_helical = jnp.asarray(cfg.target_helical_amplitude, dtype=dtype)
    helical_mismatch = ripple - target_helical
    mean_iota = (
        min_iota + 0.235 + 0.155 * shear_shift + 0.115 * ripple + 0.018 * elong_shift
    )
    floor_violation = smooth_positive(min_iota - mean_iota, beta=80.0)
    operating_floor_violation = smooth_positive(operating_iota - mean_iota, beta=45.0)
    qa_residual = jnp.sqrt(
        (0.040 * helical_mismatch) ** 2
        + (0.012 * elong_shift * helical_mismatch) ** 2
        + (0.010 * minor_shift * helical_mismatch) ** 2
        + (2.0e-4) ** 2
    )

    shaping_stabilizer = -0.085 * jnp.tanh(1.15 * elong_shift + 0.28 * shear_shift)
    shaping_cost = (
        0.020 * elong_shift**2
        + 0.030 * shear_shift**2
        + 0.018 * minor_shift**2
        + 0.006 * helical_mismatch**2
    )
    bad_curvature = (
        0.078
        + 0.16 * qa_residual
        + 0.040 * (aspect / target_aspect - 1.0) ** 2
        + shaping_cost
        + shaping_stabilizer
    )
    shear_metric = jnp.sqrt((shear_shift - 0.18) ** 2 + 4.0e-4)
    kperp_eff2 = (
        0.32
        + 0.17 / aspect
        + 0.30 * qa_residual
        + 0.045 * (elong_shift - 0.55) ** 2
        + 0.050 * shear_metric**2
    )
    raw_drive = 1.75 * bad_curvature + 0.070 * kperp_eff2 - 0.185
    growth_rate = 0.018 + smooth_positive(raw_drive, beta=22.0)
    flux_weight = (
        0.34
        + 1.80 * qa_residual
        + 0.085 * (elong_shift - 0.68) ** 2
        + 0.060 * (shear_shift - 0.24) ** 2
        + 0.025 * (aspect / target_aspect - 1.0) ** 2
    )
    quasilinear_heat_flux = (
        0.72 * flux_weight * growth_rate**2 / jnp.maximum(kperp_eff2, 1.0e-10)
    )
    return {
        "aspect": aspect,
        "mean_iota": mean_iota,
        "iota_floor_violation": floor_violation,
        "iota_operating_floor_violation": operating_floor_violation,
        "qa_residual": qa_residual,
        "helical_mismatch": helical_mismatch,
        "bad_curvature": bad_curvature,
        "kperp_eff2": kperp_eff2,
        "growth_rate": growth_rate,
        "linear_heat_flux_weight": flux_weight,
        "quasilinear_heat_flux": quasilinear_heat_flux,
        "shear_metric": shear_metric,
    }


def _qa_low_turbulence_gradient_drive(
    config: QALowTurbulenceConfig,
    dtype: jnp.dtype,
    *,
    density_gradient: float | None,
    temperature_gradient: float | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return gradient inputs and the smooth pressure-drive multiplier."""

    aln = jnp.asarray(config.fixed_density_gradient if density_gradient is None else density_gradient, dtype=dtype)
    alt = jnp.asarray(
        config.fixed_temperature_gradient if temperature_gradient is None else temperature_gradient,
        dtype=dtype,
    )
    eta_i = alt / jnp.maximum(aln, jnp.asarray(0.25, dtype=dtype))
    pressure_drive = 1.0 + 0.060 * (alt - 6.0) + 0.055 * (aln - 2.2) + 0.018 * (eta_i - 2.7)
    return aln, alt, eta_i, smooth_positive(pressure_drive, beta=10.0)


def _qa_low_turbulence_transport_shaping(p: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return smooth transport-enhancement and suppression factors."""

    minor_shift, elong_shift, _ripple, shear_shift = p
    transport_shaping = (
        jax.nn.sigmoid(8.0 * (elong_shift - 0.82))
        + 0.45 * jax.nn.sigmoid(8.0 * (minor_shift - 0.10))
        + 0.30 * jax.nn.sigmoid(8.0 * (shear_shift - 0.42))
    )
    shaping_suppression = 1.0 / (1.0 + 0.45 * transport_shaping)
    return transport_shaping, shaping_suppression


def _qa_low_turbulence_envelope_coefficients(
    p: jnp.ndarray,
    core: dict[str, jnp.ndarray],
    config: QALowTurbulenceConfig,
    *,
    density_gradient: float | None,
    temperature_gradient: float | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return growth, saturation, and heat-flux weight for the envelope model."""

    dtype = p.dtype
    aln, alt, eta_i, pressure_drive = _qa_low_turbulence_gradient_drive(
        config,
        dtype,
        density_gradient=density_gradient,
        temperature_gradient=temperature_gradient,
    )
    transport_shaping, shaping_suppression = _qa_low_turbulence_transport_shaping(p)
    growth = smooth_positive(core["growth_rate"] * pressure_drive * shaping_suppression, beta=18.0)
    saturation = (
        1.15
        + 2.45 * core["kperp_eff2"]
        + 0.40 * core["qa_residual"]
        + 0.055 * aln
        + 0.030 * alt
    )
    drive_weight = (
        core["linear_heat_flux_weight"]
        * (1.0 + 0.070 * aln + 0.040 * alt + 0.025 * smooth_positive(eta_i - 1.0, beta=6.0))
        / (1.0 + 0.30 * transport_shaping)
    )
    return growth, saturation, drive_weight


def _qa_low_turbulence_initial_energy(
    p: jnp.ndarray,
    growth: jnp.ndarray,
    saturation: jnp.ndarray,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Return a positive differentiable envelope seed."""

    equilibrium_energy = 2.0 * growth / jnp.maximum(saturation, jnp.asarray(1.0e-12, dtype=dtype))
    seed = jnp.asarray(1.0e-3, dtype=dtype) * (1.0 + 0.30 * p[2] ** 2 + 0.15 * p[1] ** 2)
    return jnp.maximum(seed, 0.35 * equilibrium_energy)


def _qa_low_turbulence_integrate_envelope(
    *,
    growth: jnp.ndarray,
    saturation: jnp.ndarray,
    drive_weight: jnp.ndarray,
    initial_energy: jnp.ndarray,
    dt: jnp.ndarray,
    steps: int,
    dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate the reduced logistic energy envelope with fixed-step RK2."""

    times = dt * jnp.arange(steps + 1, dtype=dtype)

    def rhs(energy: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * growth * energy - saturation * energy**2

    def step_fn(energy: jnp.ndarray, _idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        k1 = rhs(energy)
        predictor = jnp.maximum(energy + dt * k1, jnp.asarray(0.0, dtype=dtype))
        k2 = rhs(predictor)
        next_energy = jnp.maximum(energy + 0.5 * dt * (k1 + k2), jnp.asarray(0.0, dtype=dtype))
        return next_energy, next_energy

    _, tail = jax.lax.scan(step_fn, initial_energy, jnp.arange(steps, dtype=jnp.int32))
    energy = jnp.concatenate([jnp.asarray([initial_energy], dtype=dtype), tail])
    return times, drive_weight * energy


def qa_low_turbulence_heat_flux_trace(
    params: jnp.ndarray | Sequence[float],
    config: QALowTurbulenceConfig | None = None,
    *,
    density_gradient: float | None = None,
    temperature_gradient: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a reduced nonlinear ITG heat-flux envelope for one gradient point.

    ``density_gradient`` and ``temperature_gradient`` are normalized as
    ``a/L_n`` and ``a/L_T``. The envelope is a fixed-step differentiable RK2
    integration of ``dE/dt = 2 gamma E - alpha E^2`` and ``Q_i = W_i E``.
    """

    cfg = config or QALowTurbulenceConfig()
    p = _validate_params(params)
    core = _qa_low_turbulence_core(p, cfg)
    dtype = p.dtype
    growth, saturation, drive_weight = _qa_low_turbulence_envelope_coefficients(
        p,
        core,
        cfg,
        density_gradient=density_gradient,
        temperature_gradient=temperature_gradient,
    )
    dt = jnp.asarray(cfg.nonlinear_dt, dtype=dtype)
    steps = int(cfg.nonlinear_steps)
    return _qa_low_turbulence_integrate_envelope(
        growth=growth,
        saturation=saturation,
        drive_weight=drive_weight,
        initial_energy=_qa_low_turbulence_initial_energy(p, growth, saturation, dtype),
        dt=dt,
        steps=steps,
        dtype=dtype,
    )


def qa_low_turbulence_window_metrics(
    times: jnp.ndarray,
    heat_flux: jnp.ndarray,
    *,
    tail_fraction: float = 0.50,
    eps: float = 1.0e-12,
) -> dict[str, jnp.ndarray]:
    """Return differentiable late-window heat-flux statistics.

    The standard deviation uses ``sqrt(var + eps)`` so the Jacobian remains
    finite when a long reduced trace has fully saturated and the late-window
    variance is numerically zero.
    """

    t = jnp.asarray(times)
    q = jnp.asarray(heat_flux)
    if int(t.ndim) != 1 or int(q.ndim) != 1 or int(t.shape[0]) != int(q.shape[0]):
        raise ValueError(
            "times and heat_flux must be one-dimensional arrays with matching length"
        )
    n = int(q.shape[0])
    start = max(0, min(n - 2, int(round((1.0 - float(tail_fraction)) * n))))
    tw = t[start:]
    qw = q[start:]
    dtype = qw.dtype
    eps_arr = jnp.asarray(eps, dtype=dtype)
    mean = jnp.mean(qw)
    variance = jnp.mean((qw - mean) ** 2)
    std = jnp.sqrt(variance + eps_arr)
    centered_t = tw - jnp.mean(tw)
    denom = jnp.maximum(jnp.sum(centered_t**2), eps_arr)
    slope = jnp.sum(centered_t * (qw - mean)) / denom
    span = jnp.maximum(tw[-1] - tw[0], eps_arr)
    trend = jnp.abs(slope) * span / jnp.maximum(jnp.abs(mean), eps_arr)
    cv = std / jnp.maximum(jnp.abs(mean), eps_arr)
    return {
        "mean": mean,
        "std": std,
        "cv": cv,
        "trend": trend,
        "slope": slope,
        "start_index": jnp.asarray(start),
    }


def qa_low_turbulence_observables(
    params: jnp.ndarray | Sequence[float],
    config: QALowTurbulenceConfig | None = None,
    *,
    density_gradient: float | None = None,
    temperature_gradient: float | None = None,
) -> dict[str, jnp.ndarray]:
    """Return reduced QA constraints and ITG observables."""

    cfg = config or QALowTurbulenceConfig()
    core = _qa_low_turbulence_core(params, cfg)
    times, heat_flux = qa_low_turbulence_heat_flux_trace(
        params,
        cfg,
        density_gradient=density_gradient,
        temperature_gradient=temperature_gradient,
    )
    window = qa_low_turbulence_window_metrics(
        times,
        heat_flux,
        tail_fraction=cfg.nonlinear_tail_fraction,
    )
    return {
        "aspect": core["aspect"],
        "mean_iota": core["mean_iota"],
        "iota_floor_violation": core["iota_floor_violation"],
        "iota_operating_floor_violation": core["iota_operating_floor_violation"],
        "qa_residual": core["qa_residual"],
        "growth_rate": core["growth_rate"],
        "kperp_eff2": core["kperp_eff2"],
        "linear_heat_flux_weight": core["linear_heat_flux_weight"],
        "quasilinear_heat_flux": core["quasilinear_heat_flux"],
        "nonlinear_heat_flux_mean": window["mean"],
        "nonlinear_heat_flux_cv": window["cv"],
        "nonlinear_heat_flux_trend": window["trend"],
    }


def qa_low_turbulence_observable_vector(
    params: jnp.ndarray | Sequence[float],
    config: QALowTurbulenceConfig | None = None,
) -> jnp.ndarray:
    """Return QA low-turbulence observables in stable order."""

    obs = qa_low_turbulence_observables(params, config)
    return jnp.asarray([obs[name] for name in QA_LOW_TURBULENCE_OBSERVABLE_NAMES])


def qa_low_turbulence_residual_names(
    *,
    includes_nonlinear_heat_flux: bool,
) -> tuple[str, ...]:
    """Return the stable residual names for the comparison objective."""

    names = (
        "aspect_constraint",
        "minimum_iota_floor",
        "operating_iota_floor",
        "quasisymmetry_residual",
        "qa_helical_shaping_amplitude",
        *(f"regularization_{name}" for name in PARAMETER_NAMES),
    )
    if includes_nonlinear_heat_flux:
        return (*names, "reduced_nonlinear_heat_flux")
    return names


def qa_low_turbulence_residual_vector(
    params: jnp.ndarray | Sequence[float],
    config: QALowTurbulenceConfig | None = None,
    *,
    includes_nonlinear_heat_flux: bool,
) -> jnp.ndarray:
    """Return weighted residuals for the aspect-6 QA low-turbulence objective."""

    cfg = config or QALowTurbulenceConfig()
    p = _validate_params(params)
    obs = qa_low_turbulence_observables(p, cfg)
    dtype = p.dtype
    aspect_res = jnp.sqrt(jnp.asarray(cfg.aspect_weight, dtype=dtype)) * (
        (obs["aspect"] - cfg.target_aspect) / cfg.target_aspect
    )
    iota_res = (
        jnp.sqrt(jnp.asarray(cfg.iota_floor_weight, dtype=dtype))
        * obs["iota_floor_violation"]
    )
    operating_iota_res = (
        jnp.sqrt(jnp.asarray(cfg.iota_operating_weight, dtype=dtype))
        * obs["iota_operating_floor_violation"]
    )
    qa_res = jnp.sqrt(jnp.asarray(cfg.qa_weight, dtype=dtype)) * obs["qa_residual"]
    helical_res = (
        jnp.sqrt(jnp.asarray(cfg.helical_shaping_weight, dtype=dtype))
        * _qa_low_turbulence_core(p, cfg)["helical_mismatch"]
    )
    reg_res = jnp.sqrt(jnp.asarray(cfg.regularization, dtype=dtype)) * p
    parts = [
        jnp.asarray(
            [aspect_res, iota_res, operating_iota_res, qa_res, helical_res],
            dtype=dtype,
        ),
        reg_res,
    ]
    if includes_nonlinear_heat_flux:
        q_res = jnp.sqrt(
            jnp.maximum(
                jnp.asarray(cfg.nonlinear_weight, dtype=dtype)
                * obs["nonlinear_heat_flux_mean"],
                jnp.asarray(0.0, dtype=dtype),
            )
        )
        parts.append(jnp.asarray([q_res], dtype=dtype))
    return jnp.concatenate(parts)


def qa_low_turbulence_objective(
    params: jnp.ndarray | Sequence[float],
    config: QALowTurbulenceConfig | None = None,
    *,
    includes_nonlinear_heat_flux: bool,
) -> jnp.ndarray:
    """Return the scalar reduced QA comparison objective."""

    residual = qa_low_turbulence_residual_vector(
        params,
        config,
        includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
    )
    return jnp.dot(residual, residual)


def qa_low_turbulence_observable_sensitivity_report(
    params: jnp.ndarray | Sequence[float],
    config: QALowTurbulenceConfig | None = None,
    *,
    finite_difference_workers: int = 1,
) -> dict[str, Any]:
    """Compare the complete controls-to-observables Jacobian with finite differences."""

    cfg = config or QALowTurbulenceConfig()
    p = _validate_params(params)
    fd_step, rtol, atol = _fd_gate_tolerances(cfg.fd_step)
    report = autodiff_finite_difference_report(
        lambda x: qa_low_turbulence_observable_vector(x, cfg),
        p,
        step=fd_step,
        rtol=rtol,
        atol=atol,
        workers=finite_difference_workers,
    )
    report["observable_names"] = list(QA_LOW_TURBULENCE_OBSERVABLE_NAMES)
    report["parameter_names"] = list(PARAMETER_NAMES)
    report["kind"] = "qa_low_turbulence_observable_sensitivity_report"
    report["claim_level"] = (
        "full_reduced_controls_to_linear_quasilinear_nonlinear_observable_"
        "differentiability_gate"
    )
    return report


def _sensitivity_reports(
    params: jnp.ndarray,
    config: QALowTurbulenceConfig,
    *,
    includes_nonlinear_heat_flux: bool,
    finite_difference_workers: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    fd_step, rtol, atol = _fd_gate_tolerances(config.fd_step)
    scalar_gate = autodiff_finite_difference_report(
        lambda x: qa_low_turbulence_objective(
            x,
            config,
            includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        ),
        params,
        step=fd_step,
        rtol=rtol,
        atol=atol,
        workers=finite_difference_workers,
    )
    residual_gate = autodiff_finite_difference_report(
        lambda x: qa_low_turbulence_residual_vector(
            x,
            config,
            includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        ),
        params,
        step=fd_step,
        rtol=rtol,
        atol=atol,
        workers=finite_difference_workers,
    )
    jac = np.asarray(residual_gate["jacobian_ad"], dtype=float)
    residual = np.asarray(
        qa_low_turbulence_residual_vector(
            params,
            config,
            includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        ),
        dtype=float,
    )
    covariance = covariance_diagnostics(jac, residual, regularization=1.0e-8)
    covariance["residual_names"] = list(
        qa_low_turbulence_residual_names(
            includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        )
    )
    covariance["source"] = "qa_low_turbulence_weighted_residuals"
    observable_gate = qa_low_turbulence_observable_sensitivity_report(
        params,
        config,
        finite_difference_workers=finite_difference_workers,
    )
    return scalar_gate, residual_gate, observable_gate, covariance


__all__ = [
    "QA_LOW_TURBULENCE_DESIGN_NAMES",
    "QA_LOW_TURBULENCE_OBSERVABLE_NAMES",
    "QALowTurbulenceConfig",
    "QALowTurbulenceResult",
    "_fd_gate_tolerances",
    "_qa_low_turbulence_core",
    "_sensitivity_reports",
    "default_qa_low_turbulence_initial_params",
    "qa_low_turbulence_heat_flux_trace",
    "qa_low_turbulence_objective",
    "qa_low_turbulence_observable_sensitivity_report",
    "qa_low_turbulence_observable_vector",
    "qa_low_turbulence_observables",
    "qa_low_turbulence_residual_names",
    "qa_low_turbulence_residual_vector",
    "qa_low_turbulence_window_metrics",
]
