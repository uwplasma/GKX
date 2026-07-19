"""Explicit linear time integrators implemented in JAX."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gkx.diagnostics import SimulationDiagnostics
from gkx.geometry import FluxTubeGeometryLike
from gkx.core.grid import SpectralGrid
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams, LinearTerms
from gkx.config import resolve_cfl_fac
from gkx.solvers.time.explicit_diagnostics import (
    integrate_linear_explicit_diagnostics as _integrate_linear_explicit_diagnostics_impl,
)
from gkx.terms.assembly import assemble_rhs_cached
from gkx.utils.callbacks import progress_update_stride
from gkx.solvers.time.explicit_cfl import (
    _cfl_wavenumber_arrays,
    _geometry_frequency_maxima,
    _gradient_ratio_max,
    _laguerre_velocity_max,
    _linear_frequency_bound,
    _non_twist_shift_frequency_max,
    _parallel_periods_from_grid,
)
from gkx.solvers.time.explicit_steps import (
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
    "integrate_linear_explicit_from_config",
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


@dataclass
class _LinearHistory:
    ts: list[float] = field(default_factory=list)
    phi: list[np.ndarray] = field(default_factory=list)
    gamma: list[np.ndarray] = field(default_factory=list)
    omega: list[np.ndarray] = field(default_factory=list)


def _format_wall_time(seconds: float) -> str:
    seconds_i = max(int(round(seconds)), 0)
    minutes, secs = divmod(seconds_i, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _emit_time_progress(
    *,
    step: int,
    total_steps: int,
    t: float,
    t_max: float,
    started_at: float,
    phi_max: float,
) -> None:
    elapsed = max(time.perf_counter() - started_at, 0.0)
    rate = step / elapsed if elapsed > 1.0e-12 else 0.0
    remaining = max(total_steps - step, 0)
    eta = remaining / rate if rate > 1.0e-12 else math.inf
    eta_text = "--:--" if not math.isfinite(eta) else _format_wall_time(eta)
    pct = 100.0 * step / max(total_steps, 1)
    print(
        "[gkx] "
        f"step={step}/{total_steps} progress={pct:5.1f}% "
        f"t={t:.6g}/{t_max:.6g} elapsed={_format_wall_time(elapsed)} "
        f"eta={eta_text} |phi|max={phi_max:.6e}",
        flush=True,
    )


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


def _resolve_explicit_method(method: str) -> str:
    method_key = method.strip().lower()
    if method_key not in {
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
    return method_key


def _validate_mode_method(mode_method: str) -> None:
    if mode_method not in {"z_index", "max"}:
        raise ValueError("mode_method must be 'z_index' or 'max'")


def _adaptive_linear_dt(
    time_cfg: ExplicitTimeConfig,
    *,
    dt: float,
    dt_min: float,
    dt_max: float,
    wmax: float,
) -> float:
    if time_cfg.fixed_dt or wmax <= 0.0:
        return dt
    dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
    return min(max(dt_guess, dt_min), dt_max)


def _linear_explicit_timing(
    time_cfg: ExplicitTimeConfig,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    state_shape: tuple[int, ...],
) -> tuple[float, float, float, float, int, float]:
    t_max = float(time_cfg.t_max)
    dt = float(time_cfg.dt)
    dt_min = float(time_cfg.dt_min)
    # Explicit-time default behavior: when dt_max is unset, dt_max == dt.
    dt_max = float(time_cfg.dt_max) if time_cfg.dt_max is not None else dt
    sample_stride = int(max(time_cfg.sample_stride, 1))
    omega_max = _linear_frequency_bound(
        grid, geom, params, state_shape[-5], state_shape[-4]
    )
    wmax = float(np.sum(omega_max))
    dt = _adaptive_linear_dt(
        time_cfg,
        dt=dt,
        dt_min=dt_min,
        dt_max=dt_max,
        wmax=wmax,
    )
    return t_max, dt, dt_min, dt_max, sample_stride, wmax


def _make_linear_stepper(method: str, *, jit_enabled: bool):
    def stepper(G_state, cache_state, params_state, term_cfg_state, dt_state):
        return _linear_explicit_step(
            G_state, cache_state, params_state, term_cfg_state, dt_state, method=method
        )

    if jit_enabled:
        return jax.jit(stepper, donate_argnums=(0,))
    return stepper


def _append_linear_sample(
    *,
    t: float,
    phi: jnp.ndarray,
    phi_prev: jnp.ndarray,
    dt: float,
    z_idx: int,
    mask: jnp.ndarray,
    mode_method: str,
    history: _LinearHistory,
) -> None:
    gamma, omega = _instantaneous_growth_rate_step(
        phi,
        phi_prev,
        dt,
        z_index=z_idx,
        mask=mask,
        mode_method=mode_method,
    )
    history.ts.append(t)
    history.phi.append(np.asarray(phi))
    history.gamma.append(np.asarray(gamma))
    history.omega.append(np.asarray(omega))


def _should_emit_linear_progress(
    *,
    step: int,
    total_steps_est: int,
    progress_stride: int,
) -> bool:
    return step == 1 or step >= total_steps_est or (step % progress_stride) == 0


def _emit_linear_progress_if_due(
    *,
    show_progress: bool,
    step: int,
    total_steps_est: int,
    progress_stride: int,
    t: float,
    t_max: float,
    started_at: float,
    phi: jnp.ndarray,
) -> None:
    if not show_progress:
        return
    if not _should_emit_linear_progress(
        step=step,
        total_steps_est=total_steps_est,
        progress_stride=progress_stride,
    ):
        return
    _emit_time_progress(
        step=step,
        total_steps=total_steps_est,
        t=float(t),
        t_max=t_max,
        started_at=started_at,
        phi_max=float(jnp.max(jnp.abs(phi))),
    )


def _emit_linear_start_if_requested(
    *,
    show_progress: bool,
    total_steps_est: int,
    dt: float,
    t_max: float,
    sample_stride: int,
) -> None:
    if show_progress:
        print(
            "[gkx] linear initial-value integration started "
            f"(steps={total_steps_est}, dt={dt:.6g}, t_max={t_max:.6g}, "
            f"sample_stride={sample_stride})",
            flush=True,
        )


def _append_linear_sample_if_due(
    *,
    sampled: bool,
    t: float,
    phi: jnp.ndarray,
    phi_prev: jnp.ndarray,
    dt: float,
    z_idx: int,
    mask: jnp.ndarray,
    mode_method: str,
    history: _LinearHistory,
) -> None:
    if sampled:
        _append_linear_sample(
            t=t,
            phi=phi,
            phi_prev=phi_prev,
            dt=dt,
            z_idx=z_idx,
            mask=mask,
            mode_method=mode_method,
            history=history,
        )


def _emit_linear_complete_if_requested(*, show_progress: bool) -> None:
    if show_progress:
        print("[gkx] linear initial-value integration complete", flush=True)


def _take_linear_explicit_step(
    *,
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: Any,
    time_cfg: ExplicitTimeConfig,
    stepper: Any,
    dt: float,
    dt_min: float,
    dt_max: float,
    wmax: float,
) -> tuple[jnp.ndarray, Any, float]:
    dt = _adaptive_linear_dt(
        time_cfg,
        dt=dt,
        dt_min=dt_min,
        dt_max=dt_max,
        wmax=wmax,
    )
    G, fields = stepper(G, cache, params, term_cfg, dt)
    return G, fields, dt


def _linear_history_arrays(
    history: _LinearHistory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(history.ts),
        np.asarray(history.phi),
        np.asarray(history.gamma),
        np.asarray(history.omega),
    )


def _linear_loop_progress_clock(t_max: float, dt: float) -> tuple[int, int, float]:
    total_steps_est = max(int(math.ceil(max(t_max, 0.0) / max(dt, 1.0e-30))), 1)
    return (
        total_steps_est,
        progress_update_stride(total_steps_est, target_updates=20),
        time.perf_counter(),
    )


def _run_linear_explicit_loop(
    *,
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: Any,
    time_cfg: ExplicitTimeConfig,
    method: str,
    mode_method: str,
    t_max: float,
    dt: float,
    dt_min: float,
    dt_max: float,
    wmax: float,
    sample_stride: int,
    z_idx: int,
    mask: jnp.ndarray,
    phi_prev: jnp.ndarray,
    jit_enabled: bool,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, step = 0.0, 0
    history = _LinearHistory()
    total_steps_est, progress_stride, progress_started_at = _linear_loop_progress_clock(t_max, dt)
    stepper = _make_linear_stepper(method, jit_enabled=jit_enabled)

    _emit_linear_start_if_requested(
        show_progress=show_progress,
        total_steps_est=total_steps_est,
        dt=dt,
        t_max=t_max,
        sample_stride=sample_stride,
    )

    while t < t_max - 1.0e-12:
        G, fields, dt = _take_linear_explicit_step(
            G=G,
            cache=cache,
            params=params,
            term_cfg=term_cfg,
            time_cfg=time_cfg,
            stepper=stepper,
            dt=dt,
            dt_min=dt_min,
            dt_max=dt_max,
            wmax=wmax,
        )
        phi = fields.phi
        step += 1
        t += dt

        sampled = step % sample_stride == 0 or t >= t_max
        _append_linear_sample_if_due(
            sampled=sampled,
            t=t,
            phi=phi,
            phi_prev=phi_prev,
            dt=dt,
            z_idx=z_idx,
            mask=mask,
            mode_method=mode_method,
            history=history,
        )
        _emit_linear_progress_if_due(
            show_progress=show_progress,
            step=step,
            total_steps_est=total_steps_est,
            progress_stride=progress_stride,
            t=t,
            t_max=t_max,
            started_at=progress_started_at,
            phi=phi,
        )
        phi_prev = phi

    _emit_linear_complete_if_requested(show_progress=show_progress)

    return _linear_history_arrays(history)


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

    _validate_mode_method(mode_method)
    method = _resolve_explicit_method(time_cfg.method)
    term_cfg = _linear_term_config(terms)
    z_idx = _diagnostic_midplane_index(grid.z.size) if z_index is None else int(z_index)
    mask = _growth_rate_mode_mask(grid.ky, grid.kx, grid.dealias_mask)
    G = jnp.asarray(G0)
    t_max, dt, dt_min, dt_max, sample_stride, wmax = _linear_explicit_timing(
        time_cfg, grid, geom, params, G.shape
    )
    _, fields0 = assemble_rhs_cached(G, cache, params, terms=term_cfg, dt=dt)
    return _run_linear_explicit_loop(
        G=G,
        cache=cache,
        params=params,
        term_cfg=term_cfg,
        time_cfg=time_cfg,
        method=method,
        mode_method=mode_method,
        t_max=t_max,
        dt=dt,
        dt_min=dt_min,
        dt_max=dt_max,
        wmax=wmax,
        sample_stride=sample_stride,
        z_idx=z_idx,
        mask=mask,
        phi_prev=fields0.phi,
        jit_enabled=jit,
        show_progress=show_progress,
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
    """Public facade for diagnostics-rich explicit linear integration."""

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


def integrate_linear_explicit_from_config(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    time_cfg: Any,
    *,
    Nl: int,
    Nm: int,
    terms: LinearTerms | None = None,
    z_index: int | None = None,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate with CFL control using the common ``TimeConfig`` contract."""

    explicit_cfg = ExplicitTimeConfig(
        dt=float(time_cfg.dt),
        t_max=float(time_cfg.t_max),
        sample_stride=max(int(time_cfg.sample_stride), 1),
        fixed_dt=bool(time_cfg.fixed_dt),
        use_dealias_mask=bool(getattr(time_cfg, "use_dealias_mask", False)),
        dt_min=float(time_cfg.dt_min),
        dt_max=None if time_cfg.dt_max is None else float(time_cfg.dt_max),
        cfl=float(time_cfg.cfl),
        cfl_fac=resolve_cfl_fac(str(time_cfg.method), time_cfg.cfl_fac),
    )
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    t, phi, _gamma, _omega, _diagnostics = integrate_linear_explicit_diagnostics(
        G0, grid, cache, params, geom, explicit_cfg, terms,
        mode_method="z_index", z_index=z_index, jit=True,
        show_progress=show_progress,
    )
    return t, phi
