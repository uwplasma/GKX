"""Explicit linear time integrators implemented in JAX."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Protocol

from gkx.diagnostics_contract import SimulationDiagnostics
from gkx.geometry import (
    FluxTubeGeometryLike,
    ensure_flux_tube_geometry_data,
)
from gkx.core_grid import SpectralGrid
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams, LinearTerms
from gkx.config import resolve_cfl_fac
from gkx.terms.assembly import assemble_rhs_cached
from gkx.callbacks import progress_update_stride
from gkx.solvers_time_explicit_cfl import (
    _cfl_wavenumber_arrays,
    _geometry_frequency_maxima,
    _gradient_ratio_max,
    _laguerre_velocity_max,
    _linear_frequency_bound,
    _non_twist_shift_frequency_max,
    _parallel_periods_from_grid,
)
from gkx.solvers_time_explicit_steps import (
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
from gkx.operators.fluxes import (
    heat_flux_total,
    particle_flux_total,
)
from gkx.operators.moments import (
    distribution_free_energy,
    electrostatic_field_energy,
    fieldline_quadrature_weights,
    magnetic_vector_potential_energy,
    total_energy,
)


@dataclass(frozen=True)
class ExplicitTimeConfig:
    """Explicit time integration configuration.

    ``dt`` is required here and defaulted in :class:`gkx.config.TimeConfig`,
    which is why this struct can default ``fixed_dt`` to ``False``: a caller
    who had to supply ``dt`` has chosen it, so using it as the CFL controller's
    initial guess cannot silently substitute a step nobody asked for. The
    ``rk4``/``fixed_dt=False`` pairing is the cheap one -- rk4's CFL prefactor
    of 2.82 against rk2's 1.0 more than pays for its four stages, measured at
    29.1% fewer right-hand-side evaluations over the same horizon. See
    :class:`gkx.config.TimeConfig` for the measurement and for why the deck
    surface defaults differently.
    """

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
    remaining_time: float,
) -> tuple[jnp.ndarray, Any, float]:
    dt = _adaptive_linear_dt(
        time_cfg,
        dt=dt,
        dt_min=dt_min,
        dt_max=dt_max,
        wmax=wmax,
    )
    # The endpoint takes precedence over the controller's minimum timestep.
    dt = min(dt, remaining_time)
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
    total_steps_est, progress_stride, progress_started_at = _linear_loop_progress_clock(
        t_max, dt
    )
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
            remaining_time=t_max - t,
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
    """Explicit time integrator; shorten the final step to end at ``t_max``."""

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


def _reject_unsupported_config_collision_operator(
    time_cfg: Any, path: str, remedy: str = "leave state_sharding unset"
) -> None:
    """Fail loudly when a solver path cannot honour the selected operator.

    Silently ignoring ``collision_operator`` would report Lenard-Bernstein
    results under an advanced-operator label, so the unsupported combinations
    raise instead.
    """

    name = str(time_cfg.collision_operator).strip().lower()
    if name in ("none", "lenard_bernstein"):
        return
    raise NotImplementedError(
        f"collision_operator={name!r} is not supported by the {path} path; "
        "it is currently available on the fixed-step cached integrator "
        f"({remedy})."
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

    if getattr(time_cfg, "collision_operator", None) is not None:
        # This integrator cannot carry a moment operator; refuse, don't run as LB.
        _reject_unsupported_config_collision_operator(
            time_cfg, "CFL-controlled explicit", remedy='set solver = "time"'
        )
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
        G0,
        grid,
        cache,
        params,
        geom,
        explicit_cfg,
        terms,
        mode_method="z_index",
        z_index=z_index,
        jit=True,
        show_progress=show_progress,
    )
    return t, phi


_ALLOWED_METHODS = {
    "euler",
    "rk2",
    "rk3",
    "rk3_classic",
    "rk3_heun",
    "rk4",
    "k10",
    "sspx3",
}


class ExplicitTimeConfigLike(Protocol):
    """Runtime fields required by explicit diagnostic integration."""

    @property
    def t_max(self) -> float: ...

    @property
    def dt(self) -> float: ...

    @property
    def method(self) -> str: ...

    @property
    def sample_stride(self) -> int: ...

    @property
    def fixed_dt(self) -> bool: ...

    @property
    def use_dealias_mask(self) -> bool: ...

    @property
    def dt_min(self) -> float: ...

    @property
    def dt_max(self) -> float | None: ...

    @property
    def cfl(self) -> float: ...

    @property
    def cfl_fac(self) -> float: ...


@dataclass(frozen=True)
class _ExplicitDiagnosticPolicy:
    method: str
    t_max: float
    dt: float
    dt_min: float
    dt_max: float
    sample_stride: int
    fixed_dt: bool
    cfl: float
    cfl_fac: float
    use_dealias: bool
    mode_method: str
    z_index: int
    mask: jnp.ndarray
    wmax: float

    def step_dt(self) -> float:
        if self.fixed_dt or self.wmax <= 0.0:
            return self.dt
        dt_guess = self.cfl_fac * self.cfl / self.wmax
        return min(max(dt_guess, self.dt_min), self.dt_max)


@dataclass
class _SampleBuffers:
    t: list[float] = field(default_factory=list)
    dt: list[float] = field(default_factory=list)
    phi: list[np.ndarray] = field(default_factory=list)
    gamma: list[np.ndarray] = field(default_factory=list)
    omega: list[np.ndarray] = field(default_factory=list)
    Wg: list[float] = field(default_factory=list)
    Wphi: list[float] = field(default_factory=list)
    Wapar: list[float] = field(default_factory=list)
    heat: list[float] = field(default_factory=list)
    particle: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class _DiagnosticSample:
    phi: jnp.ndarray
    gamma: jnp.ndarray
    omega: jnp.ndarray
    Wg: float
    Wphi: float
    Wapar: float
    heat: float
    particle: float


def _start_progress(show_progress: bool) -> bool:
    if show_progress:
        print("[gkx] explicit linear simulation started", flush=True)
    return show_progress


def _finish_progress(show_progress: bool) -> None:
    if show_progress:
        print("[gkx] explicit linear simulation complete", flush=True)


def _normalize_method(method: str) -> str:
    method_key = method.strip().lower()
    if method_key not in _ALLOWED_METHODS:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk3', 'rk3_classic', 'rk3_heun', 'rk4', 'k10', 'sspx3'}"
        )
    return method_key


def _diagnostic_policy(
    grid: SpectralGrid,
    geom_eff: Any,
    params: LinearParams,
    G: jnp.ndarray,
    time_cfg: ExplicitTimeConfigLike,
    *,
    mode_method: str,
    z_index: int | None,
) -> _ExplicitDiagnosticPolicy:
    _validate_mode_method(mode_method)
    dt = float(time_cfg.dt)
    return _ExplicitDiagnosticPolicy(
        method=_normalize_method(time_cfg.method),
        t_max=float(time_cfg.t_max),
        dt=dt,
        dt_min=float(time_cfg.dt_min),
        dt_max=float(time_cfg.dt_max) if time_cfg.dt_max is not None else dt,
        sample_stride=int(max(time_cfg.sample_stride, 1)),
        fixed_dt=bool(time_cfg.fixed_dt),
        cfl=float(time_cfg.cfl),
        cfl_fac=float(time_cfg.cfl_fac),
        use_dealias=bool(time_cfg.use_dealias_mask),
        mode_method=mode_method,
        z_index=(
            _diagnostic_midplane_index(grid.z.size) if z_index is None else int(z_index)
        ),
        mask=_growth_rate_mode_mask(grid.ky, grid.kx, grid.dealias_mask),
        wmax=float(
            np.sum(
                _linear_frequency_bound(
                    grid, geom_eff, params, G.shape[-5], G.shape[-4]
                )
            )
        ),
    )


def _make_stepper(
    step_fn: Callable[..., tuple[jnp.ndarray, Any]],
    policy: _ExplicitDiagnosticPolicy,
    *,
    jit: bool,
) -> Callable[..., tuple[jnp.ndarray, Any]]:
    def step(G_state, cache_state, params_state, term_cfg_state, dt_state):
        return step_fn(
            G_state,
            cache_state,
            params_state,
            term_cfg_state,
            dt_state,
            method=policy.method,
        )

    if jit:
        return jax.jit(step, donate_argnums=(0,))
    return step


def _diagnostic_sample(
    G: jnp.ndarray,
    fields: Any,
    phi_prev: jnp.ndarray,
    dt: float,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    policy: _ExplicitDiagnosticPolicy,
    vol_fac: jnp.ndarray,
    flux_fac: jnp.ndarray,
) -> _DiagnosticSample:
    phi = fields.phi
    apar = fields.apar if fields.apar is not None else jnp.zeros_like(phi)
    bpar = fields.bpar if fields.bpar is not None else jnp.zeros_like(phi)
    gamma, omega = _instantaneous_growth_rate_step(
        phi,
        phi_prev,
        dt,
        z_index=policy.z_index,
        mask=policy.mask,
        mode_method=policy.mode_method,
    )
    Wg = distribution_free_energy(
        G, grid, params, vol_fac, use_dealias=policy.use_dealias
    )
    Wphi = electrostatic_field_energy(
        phi, cache, params, vol_fac, use_dealias=policy.use_dealias
    )
    Wapar = magnetic_vector_potential_energy(
        apar, cache, vol_fac, use_dealias=policy.use_dealias
    )
    heat = heat_flux_total(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=policy.use_dealias,
    )
    particle = particle_flux_total(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=policy.use_dealias,
    )
    return _DiagnosticSample(
        phi=phi,
        gamma=gamma,
        omega=omega,
        Wg=float(Wg),
        Wphi=float(Wphi),
        Wapar=float(Wapar),
        heat=float(heat),
        particle=float(particle),
    )


def _append_sample(
    buffers: _SampleBuffers,
    *,
    t: float,
    dt: float,
    sample: _DiagnosticSample,
) -> None:
    buffers.t.append(t)
    buffers.dt.append(float(dt))
    buffers.phi.append(np.asarray(sample.phi))
    buffers.gamma.append(np.asarray(sample.gamma))
    buffers.omega.append(np.asarray(sample.omega))
    buffers.Wg.append(sample.Wg)
    buffers.Wphi.append(sample.Wphi)
    buffers.Wapar.append(sample.Wapar)
    buffers.heat.append(sample.heat)
    buffers.particle.append(sample.particle)


def _emit_sample_progress(
    show_progress: bool,
    *,
    step: int,
    t: float,
    t_max: float,
    sample: _DiagnosticSample,
) -> None:
    if show_progress:
        print(
            f"[gkx] progress={(t / t_max) * 100:.0f}% step={step} "
            f"t={float(t):.6g} Wg={sample.Wg:.4e} "
            f"Wphi={sample.Wphi:.4e} heat={sample.heat:.4e}",
            flush=True,
        )


def _build_diagnostics(buffers: _SampleBuffers) -> SimulationDiagnostics:
    return SimulationDiagnostics(
        t=np.asarray(buffers.t),
        dt_t=np.asarray(buffers.dt),
        dt_mean=np.asarray(np.mean(buffers.dt)) if buffers.dt else np.asarray(0.0),
        gamma_t=np.asarray(buffers.gamma),
        omega_t=np.asarray(buffers.omega),
        Wg_t=np.asarray(buffers.Wg),
        Wphi_t=np.asarray(buffers.Wphi),
        Wapar_t=np.asarray(buffers.Wapar),
        heat_flux_t=np.asarray(buffers.heat),
        particle_flux_t=np.asarray(buffers.particle),
        energy_t=np.asarray(
            total_energy(
                np.asarray(buffers.Wg),
                np.asarray(buffers.Wphi),
                np.asarray(buffers.Wapar),
            )
        ),
    )


def integrate_linear_explicit_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    cache: LinearCache,
    params: LinearParams,
    geom: FluxTubeGeometryLike,
    time_cfg: ExplicitTimeConfigLike,
    terms: LinearTerms | None = None,
    *,
    mode_method: str = "z_index",
    z_index: int | None = None,
    jit: bool = True,
    show_progress: bool = False,
    linear_explicit_step_fn: Callable[..., tuple[jnp.ndarray, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SimulationDiagnostics]:
    """Growth/energy/flux diagnostics, with the final step clipped to ``t_max``."""

    console = _start_progress(show_progress)
    term_cfg = _linear_term_config(terms if terms is not None else LinearTerms())
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    G = jnp.asarray(G0)
    policy = _diagnostic_policy(
        grid, geom_eff, params, G, time_cfg, mode_method=mode_method, z_index=z_index
    )
    _, fields0 = assemble_rhs_cached(G, cache, params, terms=term_cfg, dt=policy.dt)
    phi_prev = fields0.phi
    vol_fac, flux_fac = fieldline_quadrature_weights(geom_eff, grid)
    stepper = _make_stepper(
        _linear_explicit_step
        if linear_explicit_step_fn is None
        else linear_explicit_step_fn,
        policy,
        jit=jit,
    )

    buffers = _SampleBuffers()
    t = 0.0
    step = 0
    dt_current = policy.step_dt()
    next_progress_time = 0.0
    progress_interval = max(policy.t_max / 20.0, policy.dt_min)
    while t < policy.t_max - 1.0e-12:
        dt_current = min(policy.step_dt(), policy.t_max - t)
        G, fields = stepper(G, cache, params, term_cfg, dt_current)
        step += 1
        t += dt_current
        if step % policy.sample_stride == 0 or t >= policy.t_max:
            sample = _diagnostic_sample(
                G,
                fields,
                phi_prev,
                dt_current,
                cache,
                grid,
                params,
                policy,
                vol_fac,
                flux_fac,
            )
            _append_sample(buffers, t=t, dt=dt_current, sample=sample)
            if t >= next_progress_time or t >= policy.t_max:
                _emit_sample_progress(
                    console,
                    step=step,
                    t=t,
                    t_max=policy.t_max,
                    sample=sample,
                )
                next_progress_time = t + progress_interval
        phi_prev = fields.phi

    _finish_progress(console)
    diag = _build_diagnostics(buffers)
    return (
        np.asarray(buffers.t),
        np.asarray(buffers.phi),
        np.asarray(buffers.gamma),
        np.asarray(buffers.omega),
        diag,
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
    "ExplicitTimeConfigLike",
]
