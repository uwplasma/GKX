"""Reduced QA low-turbulence stellarator optimization comparison tools.

This module adds a deliberately scoped, fully JAX-differentiable comparison
between two low-order quasi-axisymmetric (QA) stellarator designs:

* a control design constrained by quasisymmetry, aspect ratio, and an iota
  floor; and
* a transport-aware design with the same constraints plus a reduced nonlinear
  ITG heat-flux envelope in the objective.

The utilities are intended for optimization plumbing, sensitivity validation,
and manuscript figure generation. They are not a substitute for a production
VMEC/Boozer/full-nonlinear-GK optimization loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.autodiff_validation import (
    autodiff_finite_difference_report,
    covariance_diagnostics,
)
from spectraxgk.stellarator_optimization import (
    PARAMETER_NAMES,
    _validate_params,
    nonlinear_heat_flux_window_metrics,
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
    max_mode: int = 1
    aspect_weight: float = 0.85
    iota_floor_weight: float = 75.0
    qa_weight: float = 8.0
    regularization: float = 2.0e-3
    nonlinear_weight: float = 8.0
    learning_rate: float = 0.032
    steps: int = 40
    nonlinear_dt: float = 0.16
    nonlinear_steps: int = 540
    nonlinear_tail_fraction: float = 0.35
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
    surface_ntheta: int = 52
    surface_nzeta: int = 52
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
    return max(float(fd_step), 5.0e-3), 6.5e-2, 8.0e-3


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

    aspect = target_aspect * jnp.exp(
        -0.42 * minor_shift + 0.050 * elong_shift**2 + 0.035 * ripple**2
    )
    mean_iota = min_iota + 0.030 + 0.150 * shear_shift - 0.022 * ripple + 0.018 * elong_shift
    floor_violation = smooth_positive(min_iota - mean_iota, beta=80.0)
    qa_residual = jnp.sqrt(
        (0.165 * ripple) ** 2
        + (0.030 * elong_shift * ripple) ** 2
        + (0.014 * minor_shift * ripple) ** 2
        + (2.0e-4) ** 2
    )

    shaping_stabilizer = -0.085 * jnp.tanh(1.15 * elong_shift + 0.28 * shear_shift)
    shaping_cost = (
        0.020 * elong_shift**2
        + 0.030 * shear_shift**2
        + 0.018 * minor_shift**2
        + 0.012 * ripple**2
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
    quasilinear_heat_flux = 0.72 * flux_weight * growth_rate**2 / jnp.maximum(kperp_eff2, 1.0e-10)
    return {
        "aspect": aspect,
        "mean_iota": mean_iota,
        "iota_floor_violation": floor_violation,
        "qa_residual": qa_residual,
        "bad_curvature": bad_curvature,
        "kperp_eff2": kperp_eff2,
        "growth_rate": growth_rate,
        "linear_heat_flux_weight": flux_weight,
        "quasilinear_heat_flux": quasilinear_heat_flux,
        "shear_metric": shear_metric,
    }


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
    aln = jnp.asarray(
        cfg.fixed_density_gradient if density_gradient is None else density_gradient,
        dtype=dtype,
    )
    alt = jnp.asarray(
        cfg.fixed_temperature_gradient if temperature_gradient is None else temperature_gradient,
        dtype=dtype,
    )
    eta_i = alt / jnp.maximum(aln, jnp.asarray(0.25, dtype=dtype))
    pressure_drive = 1.0 + 0.060 * (alt - 6.0) + 0.055 * (aln - 2.2) + 0.018 * (eta_i - 2.7)
    pressure_drive = smooth_positive(pressure_drive, beta=10.0)
    growth = smooth_positive(core["growth_rate"] * pressure_drive, beta=18.0)
    saturation = (
        1.15
        + 2.45 * core["kperp_eff2"]
        + 0.40 * core["qa_residual"]
        + 0.055 * aln
        + 0.030 * alt
    )
    drive_weight = core["linear_heat_flux_weight"] * (
        1.0 + 0.070 * aln + 0.040 * alt + 0.025 * smooth_positive(eta_i - 1.0, beta=6.0)
    )
    dt = jnp.asarray(cfg.nonlinear_dt, dtype=dtype)
    steps = int(cfg.nonlinear_steps)
    times = dt * jnp.arange(steps + 1, dtype=dtype)
    equilibrium_energy = 2.0 * growth / jnp.maximum(saturation, jnp.asarray(1.0e-12, dtype=dtype))
    seed = jnp.asarray(1.0e-3, dtype=dtype) * (1.0 + 0.30 * p[2] ** 2 + 0.15 * p[1] ** 2)
    energy0 = jnp.maximum(seed, 0.35 * equilibrium_energy)

    def rhs(energy: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * growth * energy - saturation * energy**2

    def step_fn(energy: jnp.ndarray, _idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        k1 = rhs(energy)
        predictor = jnp.maximum(energy + dt * k1, jnp.asarray(0.0, dtype=dtype))
        k2 = rhs(predictor)
        next_energy = jnp.maximum(energy + 0.5 * dt * (k1 + k2), jnp.asarray(0.0, dtype=dtype))
        return next_energy, next_energy

    _, tail = jax.lax.scan(step_fn, energy0, jnp.arange(steps, dtype=jnp.int32))
    energy = jnp.concatenate([jnp.asarray([energy0], dtype=dtype), tail])
    return times, drive_weight * energy


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
    window = nonlinear_heat_flux_window_metrics(
        times,
        heat_flux,
        tail_fraction=cfg.nonlinear_tail_fraction,
    )
    return {
        "aspect": core["aspect"],
        "mean_iota": core["mean_iota"],
        "iota_floor_violation": core["iota_floor_violation"],
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
    *, includes_nonlinear_heat_flux: bool,
) -> tuple[str, ...]:
    """Return the stable residual names for the comparison objective."""

    names = (
        "aspect_constraint",
        "minimum_iota_floor",
        "quasisymmetry_residual",
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
    iota_res = jnp.sqrt(jnp.asarray(cfg.iota_floor_weight, dtype=dtype)) * obs[
        "iota_floor_violation"
    ]
    qa_res = jnp.sqrt(jnp.asarray(cfg.qa_weight, dtype=dtype)) * obs["qa_residual"]
    reg_res = jnp.sqrt(jnp.asarray(cfg.regularization, dtype=dtype)) * p
    parts = [jnp.asarray([aspect_res, iota_res, qa_res], dtype=dtype), reg_res]
    if includes_nonlinear_heat_flux:
        q_res = jnp.sqrt(
            jnp.maximum(
                jnp.asarray(cfg.nonlinear_weight, dtype=dtype) * obs["nonlinear_heat_flux_mean"],
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


def _history_row(
    step: int,
    params: jnp.ndarray,
    objective: jnp.ndarray,
    grad: jnp.ndarray,
    config: QALowTurbulenceConfig,
) -> dict[str, Any]:
    obs = np.asarray(qa_low_turbulence_observable_vector(params, config), dtype=float)
    return {
        "step": int(step),
        "objective": float(objective),
        "gradient_norm": float(jnp.linalg.norm(grad)),
        "params": [float(x) for x in np.asarray(params, dtype=float)],
        "observables": [float(x) for x in obs],
    }


def _sensitivity_reports(
    params: jnp.ndarray,
    config: QALowTurbulenceConfig,
    *,
    includes_nonlinear_heat_flux: bool,
    finite_difference_workers: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    return scalar_gate, residual_gate, covariance


def optimize_qa_low_turbulence(
    *,
    includes_nonlinear_heat_flux: bool,
    config: QALowTurbulenceConfig | None = None,
    initial_params: jnp.ndarray | Sequence[float] | None = None,
    finite_difference_workers: int = 1,
) -> QALowTurbulenceResult:
    """Optimize one reduced QA low-turbulence design with Adam."""

    cfg = config or QALowTurbulenceConfig()
    initial_p = default_qa_low_turbulence_initial_params() if initial_params is None else _validate_params(initial_params)
    p = jnp.asarray(initial_p)
    m = jnp.zeros_like(p)
    v = jnp.zeros_like(p)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    grad_fn = jax.value_and_grad(
        lambda x: qa_low_turbulence_objective(
            x,
            cfg,
            includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        )
    )
    history: list[dict[str, Any]] = []
    objective0, grad0 = grad_fn(p)
    history.append(_history_row(0, p, objective0, grad0, cfg))

    for step in range(1, int(cfg.steps) + 1):
        objective, grad = grad_fn(p)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        mhat = m / (1.0 - beta1**step)
        vhat = v / (1.0 - beta2**step)
        p = p - cfg.learning_rate * mhat / (jnp.sqrt(vhat) + eps)
        if step % 5 == 0 or step == int(cfg.steps):
            new_objective, new_grad = grad_fn(p)
            history.append(_history_row(step, p, new_objective, new_grad, cfg))

    final_objective, final_grad = grad_fn(p)
    if history[-1]["step"] != int(cfg.steps):
        history.append(_history_row(int(cfg.steps), p, final_objective, final_grad, cfg))
    scalar_gate, residual_gate, covariance = _sensitivity_reports(
        p,
        cfg,
        includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        finite_difference_workers=finite_difference_workers,
    )
    design_name = (
        "qa_plus_nonlinear_heat_flux"
        if includes_nonlinear_heat_flux
        else "qa_constraints"
    )
    return QALowTurbulenceResult(
        design_name=design_name,
        includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        parameter_names=tuple(PARAMETER_NAMES),
        observable_names=tuple(QA_LOW_TURBULENCE_OBSERVABLE_NAMES),
        initial_params=tuple(float(x) for x in np.asarray(initial_p, dtype=float)),
        final_params=tuple(float(x) for x in np.asarray(p, dtype=float)),
        initial_objective=float(objective0),
        final_objective=float(final_objective),
        initial_observables=tuple(
            float(x) for x in np.asarray(qa_low_turbulence_observable_vector(initial_p, cfg), dtype=float)
        ),
        final_observables=tuple(float(x) for x in np.asarray(qa_low_turbulence_observable_vector(p, cfg))),
        history=tuple(history),
        residual_gradient_gate=residual_gate,
        scalar_gradient_gate=scalar_gate,
        covariance=covariance,
        config=asdict(cfg),
    )


def reduced_boundary_surface(
    params: jnp.ndarray | Sequence[float],
    config: QALowTurbulenceConfig | None = None,
) -> dict[str, Any]:
    """Return a reduced max-mode-1 boundary surface for visualization."""

    cfg = config or QALowTurbulenceConfig()
    p = np.asarray(_validate_params(params), dtype=float)
    minor_shift, elong_shift, ripple, shear_shift = p
    aspect = float(_qa_low_turbulence_core(p, cfg)["aspect"])
    theta = np.linspace(0.0, 2.0 * np.pi, int(cfg.surface_ntheta), endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, int(cfg.surface_nzeta), endpoint=False)
    tt, zz = np.meshgrid(theta, zeta, indexing="ij")
    major_radius = 1.0
    minor_radius = major_radius / max(aspect, 1.0e-6)
    elongation = 1.0 + 0.26 * float(elong_shift)
    helical = float(ripple)
    nfp = int(cfg.n_field_periods)
    radius = major_radius + minor_radius * (
        np.cos(tt)
        + 0.055 * helical * np.cos(tt - nfp * zz)
        + 0.030 * float(shear_shift) * np.cos(2.0 * tt)
        + 0.020 * float(minor_shift) * np.cos(tt + nfp * zz)
    )
    height = minor_radius * (
        elongation * np.sin(tt)
        + 0.045 * helical * np.sin(tt - nfp * zz)
        + 0.018 * float(shear_shift) * np.sin(2.0 * tt)
    )
    x = radius * np.cos(zz)
    y = radius * np.sin(zz)
    return {
        "theta": theta.tolist(),
        "zeta": zeta.tolist(),
        "x": x.tolist(),
        "y": y.tolist(),
        "z": height.tolist(),
        "reduced_boundary_scope": "max-mode-1 visualization, not a solved VMEC equilibrium",
    }


def reduced_lcfs_bmag(
    params: jnp.ndarray | Sequence[float],
    config: QALowTurbulenceConfig | None = None,
) -> dict[str, Any]:
    """Return a reduced LCFS ``|B|`` map for QA visualization."""

    cfg = config or QALowTurbulenceConfig()
    p = np.asarray(_validate_params(params), dtype=float)
    core = _qa_low_turbulence_core(p, cfg)
    theta = np.linspace(0.0, 2.0 * np.pi, int(cfg.surface_ntheta), endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, int(cfg.surface_nzeta), endpoint=False)
    tt, zz = np.meshgrid(theta, zeta, indexing="ij")
    nfp = int(cfg.n_field_periods)
    ripple = float(p[2])
    elong = float(p[1])
    qa_amp = float(core["qa_residual"])
    bmag = (
        1.0
        + 0.055 * np.cos(tt)
        + 0.018 * elong * np.cos(2.0 * tt)
        + 0.105 * ripple * np.cos(tt - nfp * zz)
        + 0.030 * qa_amp * np.cos(2.0 * tt - nfp * zz)
    )
    return {
        "theta": theta.tolist(),
        "zeta": zeta.tolist(),
        "bmag": bmag.tolist(),
        "reduced_bmag_scope": "synthetic LCFS |B| map from reduced QA controls",
    }


def _window_mean_for_gradient(
    params: Sequence[float],
    config: QALowTurbulenceConfig,
    *,
    density_gradient: float,
    temperature_gradient: float,
) -> tuple[float, float, float, float]:
    times, heat_flux = qa_low_turbulence_heat_flux_trace(
        params,
        config,
        density_gradient=density_gradient,
        temperature_gradient=temperature_gradient,
    )
    window = nonlinear_heat_flux_window_metrics(
        times,
        heat_flux,
        tail_fraction=config.nonlinear_tail_fraction,
    )
    core = _qa_low_turbulence_core(params, config)
    return (
        float(window["mean"]),
        float(window["cv"]),
        float(window["trend"]),
        float(core["growth_rate"]),
    )


def _scan_density_gradient(
    params: Sequence[float],
    config: QALowTurbulenceConfig,
) -> dict[str, Any]:
    gradients = np.asarray(config.scan_density_gradients, dtype=float)
    means = []
    cvs = []
    trends = []
    gammas = []
    for aln in gradients:
        mean, cv, trend, gamma = _window_mean_for_gradient(
            params,
            config,
            density_gradient=float(aln),
            temperature_gradient=config.fixed_temperature_gradient,
        )
        means.append(mean)
        cvs.append(cv)
        trends.append(trend)
        gammas.append(gamma)
    slope = float(np.polyfit(gradients, np.asarray(means, dtype=float), deg=1)[0])
    return {
        "density_gradient_axis": gradients.tolist(),
        "fixed_temperature_gradient": float(config.fixed_temperature_gradient),
        "heat_flux_mean": means,
        "heat_flux_cv": cvs,
        "heat_flux_trend": trends,
        "growth_rate": gammas,
        "linear_slope_dQ_d_a_over_Ln": slope,
    }


def _fixed_trace_payload(params: Sequence[float], config: QALowTurbulenceConfig) -> dict[str, Any]:
    times, heat_flux = qa_low_turbulence_heat_flux_trace(
        params,
        config,
        density_gradient=config.fixed_density_gradient,
        temperature_gradient=config.fixed_temperature_gradient,
    )
    window = nonlinear_heat_flux_window_metrics(
        times,
        heat_flux,
        tail_fraction=config.nonlinear_tail_fraction,
    )
    window_payload = {key: float(value) for key, value in window.items()}
    window_payload["start_index"] = int(window["start_index"])
    return {
        "density_gradient": float(config.fixed_density_gradient),
        "temperature_gradient": float(config.fixed_temperature_gradient),
        "times": [float(x) for x in np.asarray(times, dtype=float)],
        "heat_flux": [float(x) for x in np.asarray(heat_flux, dtype=float)],
        "window": window_payload,
    }


def qa_low_turbulence_comparison_payload(
    config: QALowTurbulenceConfig | None = None,
    *,
    finite_difference_workers: int = 1,
) -> dict[str, Any]:
    """Build the full JSON-ready aspect-6 QA low-turbulence comparison."""

    cfg = config or QALowTurbulenceConfig()
    control = optimize_qa_low_turbulence(
        includes_nonlinear_heat_flux=False,
        config=cfg,
        finite_difference_workers=finite_difference_workers,
    )
    transport = optimize_qa_low_turbulence(
        includes_nonlinear_heat_flux=True,
        config=cfg,
        finite_difference_workers=finite_difference_workers,
    )
    results = [control.to_dict(), transport.to_dict()]
    obs_index = {name: i for i, name in enumerate(QA_LOW_TURBULENCE_OBSERVABLE_NAMES)}
    control_q = float(control.final_observables[obs_index["nonlinear_heat_flux_mean"]])
    transport_q = float(transport.final_observables[obs_index["nonlinear_heat_flux_mean"]])
    reduction = 1.0 - transport_q / max(control_q, 1.0e-14)
    design_payloads = []
    for result in results:
        params = result["final_params"]
        design_payloads.append(
            {
                "design_name": result["design_name"],
                "final_params": params,
                "final_observables": result["final_observables"],
                "density_gradient_scan": _scan_density_gradient(params, cfg),
                "fixed_gradient_trace": _fixed_trace_payload(params, cfg),
                "surface": reduced_boundary_surface(params, cfg),
                "lcfs_bmag": reduced_lcfs_bmag(params, cfg),
            }
        )
    all_gates_passed = all(
        bool(result["scalar_gradient_gate"]["passed"])
        and bool(result["residual_gradient_gate"]["passed"])
        for result in results
    )
    constraints_passed = all(
        abs(result["final_observables"][obs_index["aspect"]] - cfg.target_aspect) / cfg.target_aspect < 2.5e-2
        and result["final_observables"][obs_index["mean_iota"]] >= cfg.min_iota - 2.0e-3
        and result["final_observables"][obs_index["qa_residual"]] < 2.5e-2
        for result in results
    )
    transport_passed = bool(transport_q <= 0.95 * control_q)
    return {
        "kind": "qa_low_turbulence_comparison",
        "claim_level": (
            "reduced_differentiable_qa_low_turbulence_comparison_"
            "not_full_vmec_nonlinear_transport_optimization"
        ),
        "target_aspect": float(cfg.target_aspect),
        "minimum_iota": float(cfg.min_iota),
        "fixed_density_gradient": float(cfg.fixed_density_gradient),
        "fixed_temperature_gradient": float(cfg.fixed_temperature_gradient),
        "parameter_names": list(PARAMETER_NAMES),
        "observable_names": list(QA_LOW_TURBULENCE_OBSERVABLE_NAMES),
        "results": results,
        "designs": design_payloads,
        "comparison_metrics": {
            "control_design_heat_flux_mean": control_q,
            "transport_design_heat_flux_mean": transport_q,
            "relative_heat_flux_reduction_at_fixed_gradients": float(reduction),
            "constraints_passed": bool(constraints_passed),
            "transport_reduction_gate_passed": transport_passed,
            "ad_fd_gates_passed": bool(all_gates_passed),
            "passed": bool(all_gates_passed and constraints_passed and transport_passed),
        },
        "model_equations": {
            "objective": (
                "||r||^2 with aspect, iota-floor, QA, regularization, and optional "
                "sqrt(weight * late-window reduced nonlinear heat flux) residuals"
            ),
            "nonlinear_envelope": "dE/dt = 2 gamma E - alpha E^2; Q_i = W_i E; fixed-step RK2",
            "gradient_scan": "fixed a/L_T while scanning a/L_n and refitting late-window Q_i means",
        },
        "config": asdict(cfg),
        "scope_notes": [
            "The surface and |B| maps are reduced max-mode-1 visualizations, not solved VMEC equilibria.",
            "The nonlinear heat-flux objective is a differentiable envelope used for optimization plumbing.",
            "Production nonlinear claims still require long post-transient replicated SPECTRAX-GK windows.",
        ],
    }
