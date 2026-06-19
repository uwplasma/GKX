"""Explicit linear time integrators implemented in JAX."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics import SimulationDiagnostics
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.params import LinearParams, LinearTerms
from spectraxgk.solvers.time.explicit_diagnostics import (
    integrate_linear_explicit_diagnostics as _integrate_linear_explicit_diagnostics_impl,
)
from spectraxgk.terms.assembly import assemble_rhs_cached
from spectraxgk.utils.callbacks import progress_update_stride
from spectraxgk.solvers.time.explicit_cfl import (
    _cfl_wavenumber_arrays,
    _geometry_frequency_maxima,
    _gradient_ratio_max,
    _laguerre_velocity_max,
    _linear_frequency_bound,
    _non_twist_shift_frequency_max,
    _parallel_periods_from_grid,
)
from spectraxgk.solvers.time.explicit_progress import (
    _emit_time_progress,
    _format_wall_time,
)
from spectraxgk.solvers.time.explicit_steps import (
    _SSPX3_ADT,
    _SSPX3_W1,
    _SSPX3_W2,
    _SSPX3_W3,
    _apply_completed_step_state_mask,
    _completed_step_state_mask,
    _diagnostic_midplane_index,
    _growth_rate_mode_mask,
    _instantaneous_growth_rate_step,
    _linear_explicit_step as _linear_explicit_step_impl,
    _linear_term_config,
)

__all__ = [
    "ExplicitTimeConfig",
    "_SSPX3_ADT",
    "_SSPX3_W1",
    "_SSPX3_W2",
    "_SSPX3_W3",
    "_apply_completed_step_state_mask",
    "_cfl_wavenumber_arrays",
    "_completed_step_state_mask",
    "_diagnostic_midplane_index",
    "_emit_time_progress",
    "_format_wall_time",
    "_geometry_frequency_maxima",
    "_gradient_ratio_max",
    "_growth_rate_mode_mask",
    "_instantaneous_growth_rate_step",
    "_laguerre_velocity_max",
    "_linear_explicit_step",
    "_linear_frequency_bound",
    "_linear_term_config",
    "_non_twist_shift_frequency_max",
    "_parallel_periods_from_grid",
    "_rk3_heun_step",
    "_rk4_step",
    "integrate_linear_explicit",
    "integrate_linear_explicit_diagnostics",
]


@dataclass(frozen=True)
class ExplicitTimeConfig:
    """Explicit time integration configuration."""

    t_max: float
    dt: float
    method: str = "rk4"
    sample_stride: int = 1
    fixed_dt: bool = False
    use_dealias_mask: bool = False
    dt_min: float = 1.0e-7
    dt_max: float | None = None
    cfl: float = 0.9
    cfl_fac: float = 2.82


def _linear_explicit_step(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg,
    dt: float,
    *,
    method: str,
):
    """Explicit-module step seam used by tests and interactive diagnostics."""

    return _linear_explicit_step_impl(
        G,
        cache,
        params,
        term_cfg,
        dt,
        method=method,
        assemble_rhs_cached_fn=assemble_rhs_cached,
    )


def _rk4_step(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg,
    dt: float,
):
    """Single Explicit RK4 step through the public explicit facade."""

    return _linear_explicit_step(G, cache, params, term_cfg, dt, method="rk4")


def _rk3_heun_step(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg,
    dt: float,
):
    """Single Explicit RK3/Heun step through the public explicit facade."""

    return _linear_explicit_step(G, cache, params, term_cfg, dt, method="rk3")


def integrate_linear_explicit(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    cache: LinearCache,
    params: LinearParams,
    geom: FluxTubeGeometryLike,
    time_cfg: ExplicitTimeConfig,
    terms: LinearTerms | None = None,
    *,
    mode_method: str = "z_index",
    z_index: int | None = None,
    jit: bool = True,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Explicit time integrator with growth-rate diagnostics."""

    if mode_method not in {"z_index", "max"}:
        raise ValueError("mode_method must be 'z_index' or 'max'")

    method = time_cfg.method.strip().lower()
    if method not in {
        "euler",
        "rk2",
        "rk3",
        "rk3_classic",
        "rk3_heun",
        "rk4",
        "k10",
        "sspx3",
    }:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk3', 'rk3_classic', 'rk3_heun', 'rk4', 'k10', 'sspx3'}"
        )
    term_cfg = _linear_term_config(terms)
    t_max = float(time_cfg.t_max)
    dt = float(time_cfg.dt)
    dt_min = float(time_cfg.dt_min)
    # Explicit-time default behavior: when dt_max is unset, dt_max == dt.
    dt_max = float(time_cfg.dt_max) if time_cfg.dt_max is not None else dt
    sample_stride = int(max(time_cfg.sample_stride, 1))

    z_idx = _diagnostic_midplane_index(grid.z.size) if z_index is None else int(z_index)
    mask = _growth_rate_mode_mask(grid.ky, grid.kx, grid.dealias_mask)

    G = jnp.asarray(G0)
    t = 0.0
    step = 0

    # compute initial fields for growth-rate ratio
    _, fields0 = assemble_rhs_cached(G, cache, params, terms=term_cfg, dt=dt)
    phi_prev = fields0.phi

    omega_max = _linear_frequency_bound(grid, geom, params, G.shape[-5], G.shape[-4])
    wmax = float(np.sum(omega_max))
    if not time_cfg.fixed_dt and wmax > 0.0:
        dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
        dt = min(max(dt_guess, dt_min), dt_max)

    ts: list[float] = []
    phi_list: list[np.ndarray] = []
    gamma_list: list[np.ndarray] = []
    omega_list: list[np.ndarray] = []
    total_steps_est = max(int(math.ceil(max(t_max, 0.0) / max(dt, 1.0e-30))), 1)
    progress_stride = progress_update_stride(total_steps_est, target_updates=20)
    progress_started_at = time.perf_counter()
    if show_progress:
        print(
            "[spectrax-gk] linear initial-value integration started "
            f"(steps={total_steps_est}, dt={dt:.6g}, t_max={t_max:.6g}, "
            f"sample_stride={sample_stride})",
            flush=True,
        )

    def _step(G_state, cache_state, params_state, term_cfg_state, dt_state):
        return _linear_explicit_step(
            G_state, cache_state, params_state, term_cfg_state, dt_state, method=method
        )

    stepper = _step
    if jit:
        stepper = jax.jit(_step, donate_argnums=(0,))

    while t < t_max - 1.0e-12:
        if not time_cfg.fixed_dt and wmax > 0.0:
            dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
            dt = min(max(dt_guess, dt_min), dt_max)

        G, fields = stepper(G, cache, params, term_cfg, dt)
        phi = fields.phi
        step += 1
        t += dt

        sampled = False
        if step % sample_stride == 0 or t >= t_max:
            sampled = True
            gamma, omega = _instantaneous_growth_rate_step(
                phi,
                phi_prev,
                dt,
                z_index=z_idx,
                mask=mask,
                mode_method=mode_method,
            )
            ts.append(t)
            phi_list.append(np.asarray(phi))
            gamma_list.append(np.asarray(gamma))
            omega_list.append(np.asarray(omega))
            if show_progress and (
                step == 1 or step >= total_steps_est or (step % progress_stride) == 0
            ):
                _emit_time_progress(
                    step=step,
                    total_steps=total_steps_est,
                    t=float(t),
                    t_max=t_max,
                    started_at=progress_started_at,
                    phi_max=float(jnp.max(jnp.abs(phi))),
                )
        if (
            show_progress
            and not sampled
            and (step == 1 or step >= total_steps_est or (step % progress_stride) == 0)
        ):
            _emit_time_progress(
                step=step,
                total_steps=total_steps_est,
                t=float(t),
                t_max=t_max,
                started_at=progress_started_at,
                phi_max=float(jnp.max(jnp.abs(phi))),
            )
        phi_prev = phi

    if show_progress:
        print("[spectrax-gk] linear initial-value integration complete", flush=True)

    return (
        np.asarray(ts),
        np.asarray(phi_list),
        np.asarray(gamma_list),
        np.asarray(omega_list),
    )


def integrate_linear_explicit_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    cache: LinearCache,
    params: LinearParams,
    geom: FluxTubeGeometryLike,
    time_cfg: ExplicitTimeConfig,
    terms: LinearTerms | None = None,
    *,
    mode_method: str = "z_index",
    z_index: int | None = None,
    jit: bool = True,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SimulationDiagnostics]:
    """Compatibility facade for diagnostics-rich explicit linear integration."""

    return _integrate_linear_explicit_diagnostics_impl(
        G0,
        grid,
        cache,
        params,
        geom,
        time_cfg,
        terms,
        mode_method=mode_method,
        z_index=z_index,
        jit=jit,
        show_progress=show_progress,
        linear_explicit_step_fn=_linear_explicit_step,
    )
