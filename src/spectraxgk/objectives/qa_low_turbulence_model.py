"""Reduced differentiable QA/ITG model used by low-turbulence diagnostics."""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from spectraxgk.objectives.qa_low_turbulence_contracts import (
    QA_LOW_TURBULENCE_OBSERVABLE_NAMES,
    QALowTurbulenceConfig,
)
from spectraxgk.objectives.stellarator import _validate_params, smooth_positive


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
        cfg.fixed_temperature_gradient
        if temperature_gradient is None
        else temperature_gradient,
        dtype=dtype,
    )
    eta_i = alt / jnp.maximum(aln, jnp.asarray(0.25, dtype=dtype))
    pressure_drive = (
        1.0 + 0.060 * (alt - 6.0) + 0.055 * (aln - 2.2) + 0.018 * (eta_i - 2.7)
    )
    pressure_drive = smooth_positive(pressure_drive, beta=10.0)
    minor_shift, elong_shift, _ripple, shear_shift = p
    transport_shaping = (
        jax.nn.sigmoid(8.0 * (elong_shift - 0.82))
        + 0.45 * jax.nn.sigmoid(8.0 * (minor_shift - 0.10))
        + 0.30 * jax.nn.sigmoid(8.0 * (shear_shift - 0.42))
    )
    shaping_suppression = 1.0 / (1.0 + 0.45 * transport_shaping)
    growth = smooth_positive(
        core["growth_rate"] * pressure_drive * shaping_suppression, beta=18.0
    )
    saturation = (
        1.15
        + 2.45 * core["kperp_eff2"]
        + 0.40 * core["qa_residual"]
        + 0.055 * aln
        + 0.030 * alt
    )
    drive_weight = (
        core["linear_heat_flux_weight"]
        * (
            1.0
            + 0.070 * aln
            + 0.040 * alt
            + 0.025 * smooth_positive(eta_i - 1.0, beta=6.0)
        )
        / (1.0 + 0.30 * transport_shaping)
    )
    dt = jnp.asarray(cfg.nonlinear_dt, dtype=dtype)
    steps = int(cfg.nonlinear_steps)
    times = dt * jnp.arange(steps + 1, dtype=dtype)
    equilibrium_energy = (
        2.0 * growth / jnp.maximum(saturation, jnp.asarray(1.0e-12, dtype=dtype))
    )
    seed = jnp.asarray(1.0e-3, dtype=dtype) * (
        1.0 + 0.30 * p[2] ** 2 + 0.15 * p[1] ** 2
    )
    energy0 = jnp.maximum(seed, 0.35 * equilibrium_energy)

    def rhs(energy: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * growth * energy - saturation * energy**2

    def step_fn(
        energy: jnp.ndarray, _idx: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        k1 = rhs(energy)
        predictor = jnp.maximum(energy + dt * k1, jnp.asarray(0.0, dtype=dtype))
        k2 = rhs(predictor)
        next_energy = jnp.maximum(
            energy + 0.5 * dt * (k1 + k2), jnp.asarray(0.0, dtype=dtype)
        )
        return next_energy, next_energy

    _, tail = jax.lax.scan(step_fn, energy0, jnp.arange(steps, dtype=jnp.int32))
    energy = jnp.concatenate([jnp.asarray([energy0], dtype=dtype), tail])
    return times, drive_weight * energy


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


__all__ = [
    "_fd_gate_tolerances",
    "_qa_low_turbulence_core",
    "default_qa_low_turbulence_initial_params",
    "qa_low_turbulence_heat_flux_trace",
    "qa_low_turbulence_observable_vector",
    "qa_low_turbulence_observables",
    "qa_low_turbulence_window_metrics",
]
