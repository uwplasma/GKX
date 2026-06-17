"""Reduced nonlinear-window estimator metrics for differentiability gates."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _reduced_nonlinear_window_metrics_from_linear_observables(
    gamma: jnp.ndarray,
    kperp_eff2: jnp.ndarray,
    heat_weight: jnp.ndarray,
    *,
    dt: float = 0.18,
    steps: int = 96,
    tail_fraction: float = 0.30,
) -> jnp.ndarray:
    """Return smooth reduced nonlinear-window metrics from linear observables.

    This helper is intentionally a bounded estimator gate, not a replacement
    for converged nonlinear turbulence. It propagates a differentiable logistic
    heat-flux envelope whose drive is the isolated linear growth rate and whose
    amplitude is weighted by the electrostatic linear heat-flux response. The
    output is ``[window_mean, coefficient_of_variation, normalized_trend]`` over
    the late-time window.
    """

    steps_int = int(steps)
    if steps_int < 4:
        raise ValueError("steps must be at least 4 for a nonlinear-window metric")
    tail_fraction_float = float(tail_fraction)
    if not (0.0 < tail_fraction_float <= 1.0):
        raise ValueError("tail_fraction must be in (0, 1]")

    dtype = jnp.result_type(gamma, kperp_eff2, heat_weight)
    eps = jnp.asarray(1.0e-12, dtype=dtype)
    dt_arr = jnp.asarray(float(dt), dtype=dtype)
    beta = jnp.asarray(18.0, dtype=dtype)
    growth = jax.nn.softplus(beta * gamma) / beta
    kperp_pos = jnp.maximum(kperp_eff2, eps)
    heat_scale = jnp.sqrt(heat_weight * heat_weight + eps)
    saturation = jnp.asarray(1.0, dtype=dtype) + 2.5 * kperp_pos + 0.25 * heat_scale
    energy_sat = 2.0 * growth / jnp.maximum(saturation, eps)
    energy0 = jnp.maximum(jnp.asarray(1.0e-6, dtype=dtype), 0.08 * energy_sat)
    flux_scale = heat_scale / (1.0 + 0.35 * kperp_pos)

    def rk2_step(energy: jnp.ndarray, _unused: None) -> tuple[jnp.ndarray, jnp.ndarray]:
        rhs0 = 2.0 * growth * energy - saturation * energy * energy
        energy_mid = jnp.maximum(energy + 0.5 * dt_arr * rhs0, eps)
        rhs_mid = 2.0 * growth * energy_mid - saturation * energy_mid * energy_mid
        energy_next = jnp.maximum(energy + dt_arr * rhs_mid, eps)
        return energy_next, flux_scale * energy_next

    first_flux = flux_scale * energy0
    _energy_final, stepped_flux = jax.lax.scan(rk2_step, energy0, None, length=steps_int)
    flux_trace = jnp.concatenate([jnp.reshape(first_flux, (1,)), stepped_flux])
    n_samples = steps_int + 1
    start_index = max(0, min(n_samples - 2, int((1.0 - tail_fraction_float) * n_samples)))
    window = flux_trace[start_index:]
    mean = jnp.mean(window)
    centered_flux = window - mean
    std = jnp.sqrt(jnp.mean(centered_flux * centered_flux))
    cv = std / jnp.maximum(jnp.abs(mean), eps)
    times = dt_arr * jnp.arange(window.size, dtype=dtype)
    centered_time = times - jnp.mean(times)
    slope = jnp.sum(centered_time * centered_flux) / jnp.maximum(jnp.sum(centered_time * centered_time), eps)
    normalized_trend = slope * (dt_arr * jnp.asarray(window.size, dtype=dtype)) / jnp.maximum(jnp.abs(mean), eps)
    return jnp.asarray([mean, cv, normalized_trend], dtype=dtype)


__all__ = ["_reduced_nonlinear_window_metrics_from_linear_observables"]
