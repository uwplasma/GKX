"""Reduced QA stellarator ITG model, residuals, and sensitivity gates."""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.objectives.autodiff_validation import (
    autodiff_finite_difference_report,
    covariance_diagnostics,
)
from spectraxgk.objectives.stellarator_contracts import (
    OBSERVABLE_NAMES,
    PARAMETER_NAMES,
    StellaratorITGOptimizationConfig,
    StellaratorITGSampleSet,
    StellaratorObjectiveKind,
)
from spectraxgk.diagnostics.quasilinear_transport import quasilinear_feature_objective


def default_stellarator_initial_params() -> jnp.ndarray:
    """Return the shared off-optimum QA max-mode-1 starting point."""

    return jnp.asarray([0.28, 0.46, 0.42, -0.32])


def _validate_params(params: jnp.ndarray | Sequence[float]) -> jnp.ndarray:
    p = jnp.asarray(params)
    if p.ndim != 1 or int(p.shape[0]) != len(PARAMETER_NAMES):
        raise ValueError(f"params must be a length-{len(PARAMETER_NAMES)} vector")
    return p


def smooth_positive(x: jnp.ndarray | float, *, beta: float = 18.0) -> jnp.ndarray:
    """Smooth positive part used to keep objectives differentiable near marginality."""

    arr = jnp.asarray(x)
    beta_arr = jnp.asarray(beta, dtype=arr.dtype)
    return jax.nn.softplus(beta_arr * arr) / beta_arr


def _qa_gradient_drives(
    config: StellaratorITGOptimizationConfig,
    dtype: jnp.dtype,
    *,
    density_gradient: float | jnp.ndarray | None,
    temperature_gradient: float | jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return normalized density and temperature-gradient offsets."""

    aln_ref = jnp.asarray(config.reference_density_gradient, dtype=dtype)
    alti_ref = jnp.asarray(config.reference_temperature_gradient, dtype=dtype)
    aln = jnp.asarray(aln_ref if density_gradient is None else density_gradient, dtype=dtype)
    alti = jnp.asarray(alti_ref if temperature_gradient is None else temperature_gradient, dtype=dtype)
    density_drive = (aln - aln_ref) / jnp.maximum(aln_ref, jnp.asarray(1.0e-8, dtype=dtype))
    temperature_drive = (alti - alti_ref) / jnp.maximum(alti_ref, jnp.asarray(1.0e-8, dtype=dtype))
    return density_drive, temperature_drive


def _qa_geometry_features(
    minor_shift: jnp.ndarray,
    elong_shift: jnp.ndarray,
    ripple: jnp.ndarray,
    shear_shift: jnp.ndarray,
    aspect_target: jnp.ndarray,
    iota_target: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Return reduced QA shape, iota, residual, and curvature metrics."""

    aspect = aspect_target * jnp.exp(
        -0.48 * minor_shift + 0.060 * elong_shift**2 + 0.045 * ripple**2
    )
    mean_iota = iota_target + 0.19 * shear_shift - 0.030 * ripple + 0.018 * elong_shift
    qa_residual = jnp.sqrt(
        (0.18 * ripple) ** 2 + (0.035 * elong_shift * ripple) ** 2 + (2.0e-4) ** 2
    )
    shear_metric = jnp.sqrt(shear_shift**2 + 4.0e-4)
    bad_curvature = (
        0.055
        + 0.18 * qa_residual
        + 0.030 * (aspect / aspect_target - 1.0) ** 2
        + 0.035 * elong_shift**2
    )
    return {
        "aspect": aspect,
        "mean_iota": mean_iota,
        "qa_residual": qa_residual,
        "shear_metric": shear_metric,
        "bad_curvature": bad_curvature,
    }


def _qa_linear_itg_features(
    geometry: dict[str, jnp.ndarray],
    elong_shift: jnp.ndarray,
    ripple: jnp.ndarray,
    shear_shift: jnp.ndarray,
    iota_target: jnp.ndarray,
    density_drive: jnp.ndarray,
    temperature_drive: jnp.ndarray,
    dtype: jnp.dtype,
) -> dict[str, jnp.ndarray]:
    """Return reduced linear ITG frequency, growth, and flux-weight features."""

    aspect = geometry["aspect"]
    mean_iota = geometry["mean_iota"]
    qa_residual = geometry["qa_residual"]
    shear_metric = geometry["shear_metric"]
    kperp_eff2 = (
        0.34
        + 0.18 / aspect
        + 0.42 * qa_residual
        + 0.080 * shear_metric
        + 0.055 * elong_shift**2
    )
    raw_drive = (
        1.8 * geometry["bad_curvature"]
        + 0.25 * shear_metric
        + 0.08 * (mean_iota - iota_target) ** 2
        + 0.035 * density_drive
        + 0.10 * temperature_drive
        - 0.24
    )
    growth_rate = 0.025 + smooth_positive(raw_drive, beta=20.0)
    frequency = -0.42 * mean_iota + 0.090 * shear_shift - 0.045 * ripple
    linear_heat_flux_weight = (
        0.38
        + 2.4 * qa_residual
        + 0.18 * elong_shift**2
        + 0.10 * shear_metric
        + 0.06 * jnp.sqrt((mean_iota - iota_target) ** 2 + 1.0e-10)
    ) * jnp.maximum(
        jnp.asarray(0.15, dtype=dtype),
        1.0 + 0.12 * density_drive + 0.06 * temperature_drive,
    )
    return {
        "kperp_eff2": kperp_eff2,
        "growth_rate": growth_rate,
        "frequency": frequency,
        "linear_heat_flux_weight": linear_heat_flux_weight,
    }


def _qa_quasilinear_heat_flux(
    linear: dict[str, jnp.ndarray],
    config: StellaratorITGOptimizationConfig,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Return the shipped reduced mixing-length heat-flux proxy."""

    ql_features = jnp.asarray(
        [linear["growth_rate"], linear["kperp_eff2"], linear["linear_heat_flux_weight"]],
        dtype=dtype,
    )
    return quasilinear_feature_objective(
        ql_features,
        rule="mixing_length",
        csat=config.quasilinear_csat,
        gamma_floor=0.0,
    )


def _qa_core_features(
    params: jnp.ndarray | Sequence[float],
    config: StellaratorITGOptimizationConfig,
    *,
    density_gradient: float | jnp.ndarray | None = None,
    temperature_gradient: float | jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Return the linear/quasilinear QA-ITG features without nonlinear tracing."""

    p = _validate_params(params)
    minor_shift, elong_shift, ripple, shear_shift = p
    dtype = p.dtype
    aspect_target = jnp.asarray(config.target_aspect, dtype=dtype)
    iota_target = jnp.asarray(config.target_iota, dtype=dtype)
    density_drive, temperature_drive = _qa_gradient_drives(
        config,
        dtype=dtype,
        density_gradient=density_gradient,
        temperature_gradient=temperature_gradient,
    )
    geometry = _qa_geometry_features(
        minor_shift,
        elong_shift,
        ripple,
        shear_shift,
        aspect_target,
        iota_target,
    )
    linear = _qa_linear_itg_features(
        geometry,
        elong_shift,
        ripple,
        shear_shift,
        iota_target,
        density_drive,
        temperature_drive,
        dtype=dtype,
    )
    quasilinear_heat_flux = _qa_quasilinear_heat_flux(linear, config, dtype)
    return {
        "aspect": geometry["aspect"],
        "mean_iota": geometry["mean_iota"],
        "qa_residual": geometry["qa_residual"],
        "shear_metric": geometry["shear_metric"],
        "kperp_eff2": linear["kperp_eff2"],
        "growth_rate": linear["growth_rate"],
        "frequency": linear["frequency"],
        "linear_heat_flux_weight": linear["linear_heat_flux_weight"],
        "quasilinear_heat_flux": quasilinear_heat_flux,
    }


def qa_max_mode1_observables(
    params: jnp.ndarray | Sequence[float],
    config: StellaratorITGOptimizationConfig | None = None,
    *,
    density_gradient: float | jnp.ndarray | None = None,
    temperature_gradient: float | jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Map a QA max-mode-1 boundary/control vector to differentiable ITG observables.

    The four inputs represent the active low-order controls used by the example
    scripts. The map is calibrated as a smooth objective-reduction gate around a
    QA stellarator with aspect ratio 7 and mean rotational transform 0.41. It is
    not a replacement for the full VMEC/Boozer flux-tube geometry contract; its
    purpose is to validate gradient plumbing, UQ, optimizer behavior, and
    figure-generation before expensive production objectives are promoted.
    """

    cfg = config or StellaratorITGOptimizationConfig()
    p = _validate_params(params)
    core = _qa_core_features(
        p,
        cfg,
        density_gradient=density_gradient,
        temperature_gradient=temperature_gradient,
    )
    times, heat_flux = nonlinear_heat_flux_trace(
        p,
        cfg,
        density_gradient=density_gradient,
        temperature_gradient=temperature_gradient,
    )
    nl_summary = nonlinear_heat_flux_window_metrics(times, heat_flux, tail_fraction=cfg.nonlinear_tail_fraction)

    return {
        "aspect": core["aspect"],
        "mean_iota": core["mean_iota"],
        "qa_residual": core["qa_residual"],
        "kperp_eff2": core["kperp_eff2"],
        "growth_rate": core["growth_rate"],
        "frequency": core["frequency"],
        "linear_heat_flux_weight": core["linear_heat_flux_weight"],
        "quasilinear_heat_flux": core["quasilinear_heat_flux"],
        "nonlinear_heat_flux_mean": nl_summary["mean"],
        "nonlinear_heat_flux_cv": nl_summary["cv"],
        "nonlinear_heat_flux_trend": nl_summary["trend"],
    }


def qa_observable_vector(
    params: jnp.ndarray | Sequence[float],
    config: StellaratorITGOptimizationConfig | None = None,
) -> jnp.ndarray:
    """Return observables in the stable order defined by ``OBSERVABLE_NAMES``."""

    obs = qa_max_mode1_observables(params, config)
    return jnp.asarray([obs[name] for name in OBSERVABLE_NAMES])


def _sampled_qa_itg_fields(
    params: jnp.ndarray | Sequence[float],
    config: StellaratorITGOptimizationConfig,
    sample_set: StellaratorITGSampleSet,
) -> dict[str, jnp.ndarray]:
    """Return smooth reduced ITG fields over a surface/alpha/ky sample set."""

    p = _validate_params(params)
    dtype = p.dtype
    core = _qa_core_features(p, config)
    surfaces = jnp.asarray(sample_set.surfaces, dtype=dtype)[:, None, None]
    alphas = jnp.asarray(sample_set.alphas, dtype=dtype)[None, :, None]
    kys = jnp.asarray(sample_set.ky_values, dtype=dtype)[None, None, :]
    surface_delta = surfaces - jnp.asarray(0.64, dtype=dtype)
    ky_ratio = kys / jnp.asarray(0.30, dtype=dtype)
    alpha_cos = jnp.cos(alphas)
    alpha_sin = jnp.sin(alphas)
    qa_residual = core["qa_residual"]
    shear_metric = core["shear_metric"]

    kperp_eff2 = core["kperp_eff2"] * (
        0.58
        + 0.46 * ky_ratio**2
        + 0.10 * surface_delta**2
        + 0.025 * alpha_cos**2
        + 0.030 * qa_residual
    )
    drive_shift = (
        0.030 * surface_delta
        - 0.050 * (ky_ratio - 1.0) ** 2
        + 0.018 * qa_residual * alpha_cos
        + 0.010 * shear_metric * jnp.sin(alphas + 0.4 * surface_delta)
    )
    growth_rate = smooth_positive(core["growth_rate"] + drive_shift, beta=22.0)
    frequency = core["frequency"] * (1.0 + 0.08 * surface_delta) + 0.035 * (ky_ratio - 1.0) + 0.010 * alpha_sin
    linear_heat_flux_weight = core["linear_heat_flux_weight"] * (
        1.0
        + 0.11 * surface_delta**2
        + 0.065 * jnp.abs(alpha_sin) * (1.0 + qa_residual)
        + 0.055 * ky_ratio
    )
    ql_features = jnp.stack([growth_rate, kperp_eff2, linear_heat_flux_weight], axis=-1)
    quasilinear_heat_flux = quasilinear_feature_objective(
        ql_features,
        rule="mixing_length",
        csat=config.quasilinear_csat,
        gamma_floor=0.0,
    )
    nonlinear_window_proxy = quasilinear_heat_flux * (
        0.70
        + 0.18 / (1.0 + kperp_eff2)
        + 0.08 * jnp.tanh(8.0 * growth_rate)
        + 0.025 * jnp.abs(alpha_cos)
    )
    return {
        "growth": growth_rate,
        "growth_rate": growth_rate,
        "gamma": growth_rate,
        "frequency": frequency,
        "omega": frequency,
        "linear_heat_flux_weight": linear_heat_flux_weight,
        "kperp_eff2": kperp_eff2,
        "quasilinear_flux": quasilinear_heat_flux,
        "quasilinear_heat_flux": quasilinear_heat_flux,
        "mixing_length_heat_flux_proxy": quasilinear_heat_flux,
        "nonlinear_heat_flux": nonlinear_window_proxy,
        "nonlinear_window_heat_flux_mean": nonlinear_window_proxy,
    }


def nonlinear_heat_flux_trace(
    params: jnp.ndarray | Sequence[float],
    config: StellaratorITGOptimizationConfig | None = None,
    *,
    density_gradient: float | jnp.ndarray | None = None,
    temperature_gradient: float | jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a differentiable short-window ITG heat-flux envelope trace.

    The envelope evolves ``E`` with a fixed-step RK2 discretization,

    ``dE/dt = 2 gamma E - alpha E^2``, ``Q_env(t) = W_i E``.

    ``gamma`` and ``W_i`` come from the same differentiable QA/ITG feature map
    as the linear and quasilinear objectives. The output is therefore useful for
    nonlinear averaging, optimizer, and UQ gates while the full production
    nonlinear-GK geometry path is still being made traceable end-to-end.
    """

    cfg = config or StellaratorITGOptimizationConfig()
    p = _validate_params(params)
    dtype = p.dtype
    _, elong_shift, ripple, _ = p
    core = _qa_core_features(
        p,
        cfg,
        density_gradient=density_gradient,
        temperature_gradient=temperature_gradient,
    )
    growth_rate = core["growth_rate"]
    kperp_eff2 = core["kperp_eff2"]
    qa_residual = core["qa_residual"]
    shear_metric = core["shear_metric"]
    flux_weight = core["linear_heat_flux_weight"]
    saturation = 1.2 + 2.8 * kperp_eff2 + 0.45 * shear_metric + 1.4 * qa_residual
    drive_weight = flux_weight / (1.0 + 0.35 * kperp_eff2)
    dt = jnp.asarray(cfg.nonlinear_dt, dtype=dtype)
    steps = int(cfg.nonlinear_steps)
    times = dt * jnp.arange(steps + 1, dtype=dtype)
    equilibrium_energy = 2.0 * growth_rate / jnp.maximum(saturation, jnp.asarray(1.0e-12, dtype=dtype))
    seed_floor = jnp.asarray(8.0e-4, dtype=dtype) * (1.0 + 0.5 * ripple**2 + 0.2 * elong_shift**2)
    e0 = jnp.maximum(seed_floor, 0.40 * equilibrium_energy)

    def rhs(energy: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * growth_rate * energy - saturation * energy**2

    def step_fn(energy: jnp.ndarray, _idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        k1 = rhs(energy)
        predictor = jnp.maximum(energy + dt * k1, jnp.asarray(0.0, dtype=dtype))
        k2 = rhs(predictor)
        next_energy = jnp.maximum(energy + 0.5 * dt * (k1 + k2), jnp.asarray(0.0, dtype=dtype))
        return next_energy, next_energy

    _, energy_tail = jax.lax.scan(step_fn, e0, jnp.arange(steps, dtype=jnp.int32))
    energy = jnp.concatenate([jnp.asarray([e0], dtype=dtype), energy_tail])
    heat_flux = drive_weight * energy
    return times, heat_flux


def nonlinear_heat_flux_window_metrics(
    times: jnp.ndarray,
    heat_flux: jnp.ndarray,
    *,
    tail_fraction: float = 0.45,
    eps: float = 1.0e-14,
) -> dict[str, jnp.ndarray]:
    """Return mean, coefficient of variation, and trend on a late-time window."""

    t = jnp.asarray(times)
    q = jnp.asarray(heat_flux)
    if int(t.ndim) != 1 or int(q.ndim) != 1 or int(t.shape[0]) != int(q.shape[0]):
        raise ValueError("times and heat_flux must be one-dimensional arrays with matching length")
    n = int(q.shape[0])
    start = max(0, min(n - 2, int(round((1.0 - float(tail_fraction)) * n))))
    tw = t[start:]
    qw = q[start:]
    mean = jnp.mean(qw)
    centered_t = tw - jnp.mean(tw)
    denom = jnp.maximum(jnp.sum(centered_t**2), jnp.asarray(eps, dtype=qw.dtype))
    slope = jnp.sum(centered_t * (qw - mean)) / denom
    span = jnp.maximum(tw[-1] - tw[0], jnp.asarray(eps, dtype=qw.dtype))
    trend = jnp.abs(slope) * span / jnp.maximum(jnp.abs(mean), jnp.asarray(eps, dtype=qw.dtype))
    cv = jnp.std(qw) / jnp.maximum(jnp.abs(mean), jnp.asarray(eps, dtype=qw.dtype))
    return {
        "mean": mean,
        "std": jnp.std(qw),
        "cv": cv,
        "trend": trend,
        "slope": slope,
        "start_index": jnp.asarray(start),
    }


_RESIDUAL_CONDITION_NUMBER_LIMIT = 1.0e4


def _precision_gate_tolerances(fd_step: float) -> tuple[float, float, float]:
    """Return FD tolerances that are strict in x64 and stable in float32."""

    if bool(getattr(jax.config, "jax_enable_x64", False)):
        return float(fd_step), 5.0e-3, 6.0e-4
    return max(float(fd_step), 5.0e-3), 5.0e-2, 6.0e-3


def _residual_precision_gate_tolerances(
    fd_step: float,
) -> tuple[float, float, float]:
    """Return residual-Jacobian FD tolerances that remain local near zero."""

    if bool(getattr(jax.config, "jax_enable_x64", False)):
        return float(fd_step), 5.0e-3, 6.0e-4
    return min(float(fd_step), 1.0e-4), 5.0e-2, 6.0e-3


def _conditioning_gate_from_covariance(
    covariance: dict[str, Any],
    *,
    min_rank: int,
    condition_number_limit: float,
) -> dict[str, Any]:
    """Return a pass/fail gate for Gauss-Newton residual conditioning."""

    singular = np.asarray(covariance.get("jacobian_singular_values", ()), dtype=float)
    rank = int(covariance.get("sensitivity_map_rank", 0))
    condition_number = float(
        covariance.get("jacobian_condition_number", float("inf"))
    )
    finite_singular_values = bool(singular.size > 0 and np.all(np.isfinite(singular)))
    finite_condition = bool(np.isfinite(condition_number))
    smallest = float(singular[-1]) if finite_singular_values else 0.0
    limit = float(condition_number_limit)
    if int(min_rank) < 1:
        raise ValueError("min_rank must be >= 1")
    if limit <= 0.0:
        raise ValueError("condition_number_limit must be positive")
    passed = bool(
        finite_singular_values
        and finite_condition
        and rank >= int(min_rank)
        and condition_number <= limit
        and smallest > 0.0
    )
    return {
        "passed": passed,
        "finite_singular_values": finite_singular_values,
        "finite_condition_number": finite_condition,
        "sensitivity_map_rank": rank,
        "min_rank": int(min_rank),
        "rank_deficiency": int(max(int(min_rank) - rank, 0)),
        "jacobian_condition_number": condition_number,
        "condition_number_limit": limit,
        "smallest_singular_value": smallest,
    }


def stellarator_itg_objective(
    params: jnp.ndarray | Sequence[float],
    kind: StellaratorObjectiveKind,
    config: StellaratorITGOptimizationConfig | None = None,
) -> jnp.ndarray:
    """Return the scalar constrained QA + ITG objective for one optimization."""

    residual = stellarator_itg_objective_residual_vector(params, kind, config)
    return jnp.dot(residual, residual)


def stellarator_itg_objective_residual_names(
    kind: StellaratorObjectiveKind,
) -> tuple[str, ...]:
    """Return stable residual names for the weighted QA + ITG objective."""

    if kind not in ("growth", "quasilinear_flux", "nonlinear_heat_flux"):
        raise ValueError(f"unknown stellarator objective kind {kind!r}")
    return (
        "aspect_constraint",
        "iota_constraint",
        "qa_constraint",
        *(f"regularization_{name}" for name in PARAMETER_NAMES),
        f"{kind}_transport_objective",
    )


def stellarator_itg_objective_residual_vector(
    params: jnp.ndarray | Sequence[float],
    kind: StellaratorObjectiveKind,
    config: StellaratorITGOptimizationConfig | None = None,
) -> jnp.ndarray:
    """Return the weighted residual map used by the objective and covariance."""

    cfg = config or StellaratorITGOptimizationConfig()
    p = _validate_params(params)
    obs = _qa_core_features(p, cfg)
    dtype = p.dtype
    aspect_res = jnp.sqrt(jnp.asarray(cfg.aspect_weight, dtype=dtype)) * (
        (obs["aspect"] - cfg.target_aspect) / cfg.target_aspect
    )
    iota_res = jnp.sqrt(jnp.asarray(cfg.iota_weight, dtype=dtype)) * (
        obs["mean_iota"] - cfg.target_iota
    )
    qa_res = jnp.sqrt(jnp.asarray(cfg.qa_weight, dtype=dtype)) * obs["qa_residual"]
    reg_res = jnp.sqrt(jnp.asarray(cfg.regularization, dtype=dtype)) * p
    if kind == "growth":
        turbulence = obs["growth_rate"]
    elif kind == "quasilinear_flux":
        turbulence = obs["quasilinear_heat_flux"]
    elif kind == "nonlinear_heat_flux":
        times, heat_flux = nonlinear_heat_flux_trace(p, cfg)
        turbulence = nonlinear_heat_flux_window_metrics(
            times,
            heat_flux,
            tail_fraction=cfg.nonlinear_tail_fraction,
        )["mean"]
    else:
        raise ValueError(f"unknown stellarator objective kind {kind!r}")
    turbulence_res = jnp.sqrt(
        jnp.maximum(
            jnp.asarray(cfg.turbulence_weight, dtype=dtype) * turbulence,
            jnp.asarray(0.0, dtype=dtype),
        )
    )
    return jnp.concatenate(
        [
            jnp.asarray([aspect_res, iota_res, qa_res], dtype=dtype),
            reg_res,
            jnp.asarray([turbulence_res], dtype=dtype),
        ]
    )


def stellarator_itg_residual_sensitivity_report(
    params: jnp.ndarray | Sequence[float],
    kind: StellaratorObjectiveKind,
    config: StellaratorITGOptimizationConfig | None = None,
    *,
    step: float | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    min_rank: int = len(PARAMETER_NAMES),
    condition_number_limit: float = _RESIDUAL_CONDITION_NUMBER_LIMIT,
    covariance_regularization: float = 1.0e-8,
    finite_difference_workers: int = 1,
    finite_difference_executor: str = "thread",
) -> dict[str, Any]:
    """Check residual-Jacobian AD/FD parity and local conditioning."""

    cfg = config or StellaratorITGOptimizationConfig()
    p = _validate_params(params)
    default_step, default_rtol, default_atol = _residual_precision_gate_tolerances(
        cfg.fd_step
    )
    fd_step = default_step if step is None else float(step)
    fd_rtol = default_rtol if rtol is None else float(rtol)
    fd_atol = default_atol if atol is None else float(atol)

    def residual_fn(x: jnp.ndarray) -> jnp.ndarray:
        return stellarator_itg_objective_residual_vector(x, kind, cfg)

    residual_gate = autodiff_finite_difference_report(
        residual_fn,
        p,
        step=fd_step,
        rtol=fd_rtol,
        atol=fd_atol,
        workers=finite_difference_workers,
        parallel_executor=finite_difference_executor,
    )
    jac = np.asarray(residual_gate["jacobian_ad"], dtype=float)
    residual = np.asarray(residual_fn(p), dtype=float)
    covariance = covariance_diagnostics(
        jac, residual, regularization=covariance_regularization
    )
    covariance["source"] = "weighted_objective_residual"
    covariance["residual_names"] = list(
        stellarator_itg_objective_residual_names(kind)
    )
    conditioning_gate = _conditioning_gate_from_covariance(
        covariance,
        min_rank=min_rank,
        condition_number_limit=condition_number_limit,
    )
    covariance["conditioning_gate"] = conditioning_gate
    return {
        "kind": "stellarator_itg_residual_sensitivity_report",
        "objective_kind": kind,
        "passed": bool(residual_gate["passed"] and conditioning_gate["passed"]),
        "parameter_names": list(PARAMETER_NAMES),
        "residual_names": list(stellarator_itg_objective_residual_names(kind)),
        "finite_difference_gate": residual_gate,
        "conditioning_gate": conditioning_gate,
        "covariance": covariance,
    }
