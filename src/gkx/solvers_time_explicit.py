"""Explicit linear time integrators implemented in JAX."""

from __future__ import annotations

import math
import time
import jax
import jax.numpy as jnp
import numpy as np
import warnings
from dataclasses import dataclass, field
from numpy.polynomial.laguerre import laggauss
from typing import Any, Callable, Protocol
from gkx.callbacks import progress_update_stride
from gkx.config import method_error, resolve_cfl_fac
from gkx.core_grid import SpectralGrid
from gkx.core_ky_layout import HALF, half_ky_values, source_ky_layout, source_ny_full
from gkx.diagnostics_contract import SimulationDiagnostics
from gkx.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from gkx.operators.fluxes import heat_flux_total, particle_flux_total
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.params import (
    BUILTIN_COLLISION_OPERATORS,
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from gkx.operators.moments import (
    distribution_free_energy,
    electrostatic_field_energy,
    fieldline_quadrature_weights,
    magnetic_vector_potential_energy,
    total_energy,
)
from gkx.terms.assembly import assemble_rhs_cached
from gkx.terms.config import FieldState, TermConfig


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


def _resolve_explicit_method(method: str) -> str:
    return _normalize_method(method)


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
    _, fields0 = assemble_rhs_cached(G, cache, params, terms=term_cfg)
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
    if name in BUILTIN_COLLISION_OPERATORS:
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


_ALLOWED_METHODS = frozenset({"rk2", "rk3", "rk4", "sspx3"})


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
        raise method_error(method, _ALLOWED_METHODS)
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
    _, fields0 = assemble_rhs_cached(G, cache, params, terms=term_cfg)
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


# Ratio of dt to the explicit CFL bound above which a fixed-step nonlinear run
# is warned about. Not 1.0: the bound is deliberately conservative -- it takes
# maxima over the whole grid and carries the 0.9 cfl safety factor -- so decks
# sit above it and still integrate. Measured on the shipped decks: 1.33x runs
# clean over 500 steps, 1.60x (the W7-X zonal benchmark) goes non-finite at
# t=5.65 of 60, and 2.77x (the ETG nonlinear example) at t=0.021 of 0.5. 1.5
# sits in that gap, so it fires on both known failures and on neither known
# good deck. The linear path's equivalent warning uses 1.0 and predates this
# measurement; the two are not reconciled.
FIXED_DT_CFL_WARN_RATIO = 1.5


@dataclass(frozen=True)
class _CFLGridBounds:
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    kx_max: float
    ky_max: float
    kz_max: float
    kperp_min: float


@dataclass(frozen=True)
class _SpeciesFrequencyScales:
    tprim: np.ndarray
    fprim: np.ndarray
    temp: np.ndarray
    dens: np.ndarray
    charge: np.ndarray
    tzmax: float
    vtmax: float
    vtmin: float
    etamax: float
    vpar_max: float
    muB_max: float


@dataclass(frozen=True)
class _GeometryFrequencyScales:
    bmag_max: float
    cvdrift_max: float
    gbdrift_max: float
    cvdrift0_max: float
    gbdrift0_max: float
    gradpar: float
    m0_max: float


def _parallel_periods_from_grid(grid: SpectralGrid) -> float:
    """Return the parallel extent of the flux tube in 2*pi periods.

    Take the spacing off the host buffer rather than out of ``grid.z`` by
    ``jnp`` arithmetic. Inside a trace every ``jnp`` call stages out, so
    ``grid.z[1] - grid.z[0]`` is a tracer even when ``grid.z`` is the concrete
    axis built from the run's own config, and reading it back on the host then
    fails. Nothing here is differentiable -- the number of periods is grid
    topology -- so this refused every adaptive-dt run under ``jit`` for a
    quantity that was never traced.
    """

    if grid.z.size <= 1:
        return 1.0
    z_host = np.asarray(grid.z, dtype=float)
    dz = float(z_host[1] - z_host[0])
    return float(z_host[-1] - z_host[0] + dz) / (2.0 * np.pi)


def _cfl_wavenumber_arrays(
    grid: SpectralGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kx = np.asarray(grid.kx, dtype=float).reshape(-1)
    ky_full = np.asarray(grid.ky, dtype=float).reshape(-1)
    nz = int(grid.z.size)
    zp = _parallel_periods_from_grid(grid)
    kz = np.zeros(nz, dtype=float)

    # Preserve actual mode values on sliced ky grids. Reconstructing ky from
    # (Ny, y0) maps a single selected positive ky back to ky=0 and breaks the
    # one-mode CFL estimate used by linear benchmark runs.
    if ky_full.size == 0:
        ky = np.zeros(0, dtype=float)
    elif grid.ky_mode is not None or source_ky_layout(grid) == HALF:  # ky >= 0 rows
        ky = np.abs(ky_full)
    else:
        ky = half_ky_values(ky_full)

    for idx in range(nz):
        if idx < nz / 2 + 1:
            kz[idx] = float(idx) / zp
        else:
            kz[idx] = float(idx - nz) / zp
    return kx, ky, kz


def _laguerre_velocity_max(nl: int, *, node_max: bool = False) -> float:
    """``nl``-th Laguerre root; ``node_max``: last node of the padded B_par grid."""
    if nl <= 0:
        return 0.0
    nj = max(1, (3 * nl) // 2 - 1)
    roots, _weights = laggauss(nj)
    idx = roots.size - 1 if node_max else min(max(nl - 1, 0), roots.size - 1)
    return float(roots[idx])


def _gradient_ratio_max(tprim: np.ndarray, fprim: np.ndarray) -> float:
    if tprim.size == 0:
        return 0.0
    eta = np.zeros_like(tprim, dtype=float)
    mask = np.abs(fprim) > 0.0
    eta[mask] = tprim[mask] / fprim[mask]
    eta[~mask] = 1.0e6
    return float(np.max(eta))


_TRACED_CFL_INPUT_MESSAGE = (
    "the adaptive time step cannot read a traced {name}: the CFL bound is a "
    "host scalar that fixes dt_max before the scan is staged, so it needs the "
    "value and not a promise of one. Run with fixed_dt=True, where the step is "
    "the dt you pass and the trajectory is still exact and still "
    "differentiable, or keep {name} concrete. An analytic geometry sampled "
    "inside the trace arrives here traced even when nothing physical is: "
    "sample it once outside with ensure_flux_tube_geometry_data and pass the "
    "sampled geometry in."
)


def _host_cfl_value(value: Any, name: str) -> np.ndarray:
    """Return host samples of one CFL input, naming it if it is traced."""

    if isinstance(value, jax.core.Tracer):
        raise ValueError(_TRACED_CFL_INPUT_MESSAGE.format(name=name))
    return np.asarray(value, dtype=float)


def _geometry_frequency_maxima(
    geom: FluxTubeGeometryLike, theta: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    # Sample the geometry as a compile-time constant. ``theta`` is the host
    # axis and an analytic geometry evaluates it with ``jnp``, which stages out
    # inside a trace; the maxima below then could not be read back, so an
    # adaptive-dt run refused under ``jit`` for a geometry that was never
    # traced. Genuinely traced geometry stays traced here and is named below.
    with jax.ensure_compile_time_eval():
        theta_j = jnp.asarray(theta)
        cv_j, gb_j, cv0_j, gb0_j = geom.drift_coeffs(theta_j)
        bmag_j = geom.bmag(theta_j)
        gradpar_value = geom.gradpar()
    cv = _host_cfl_value(cv_j, "curvature drift")
    gb = _host_cfl_value(gb_j, "grad-B drift")
    cv0 = _host_cfl_value(cv0_j, "radial curvature drift")
    gb0 = _host_cfl_value(gb0_j, "radial grad-B drift")
    bmag = _host_cfl_value(bmag_j, "bmag")
    bmag_max = float(np.max(np.abs(bmag)))
    cvdrift_max = float(np.max(np.abs(cv)))
    gbdrift_max = float(np.max(np.abs(gb)))
    cvdrift0_max = float(np.max(np.abs(cv0)))
    gbdrift0_max = float(np.max(np.abs(gb0)))
    return (
        bmag_max,
        cvdrift_max,
        gbdrift_max,
        cvdrift0_max,
        gbdrift0_max,
        float(_host_cfl_value(gradpar_value, "gradpar")),
    )


def _non_twist_shift_frequency_max(
    geom: FluxTubeGeometryLike,
    grid: SpectralGrid,
    ky_max: float,
    vpar_max: float,
    muB_max: float,
) -> tuple[float, float, float]:
    theta = np.asarray(grid.z, dtype=float)
    # Same compile-time sampling as _geometry_frequency_maxima: the metric on a
    # host theta axis is a constant, not a stage-out.
    with jax.ensure_compile_time_eval():
        theta_j = jnp.asarray(theta)
        _gds2, gds21_j, gds22_j = geom.metric_coeffs(theta_j)
        _cv_j, _gb_j, cv0_j, gb0_j = geom.drift_coeffs(theta_j)
    gds21 = _host_cfl_value(gds21_j, "gds21")
    gds22 = _host_cfl_value(gds22_j, "gds22")
    shat = float(_host_cfl_value(geom.s_hat, "s_hat"))
    ftwist = shat * gds21 / gds22
    nz = theta.size
    if nz <= 1:
        return (
            0.0,
            float(np.max(np.abs(_host_cfl_value(cv0_j, "radial curvature drift")))),
            float(np.max(np.abs(_host_cfl_value(gb0_j, "radial grad-B drift")))),
        )
    delta = 0.01313
    x0 = float(grid.x0)
    kxfac = float(grid.kxfac)
    zp = _parallel_periods_from_grid(grid)
    mid = nz // 2
    mid_next = min(mid + 1, nz - 1)
    ref_term = (1.0 - delta) * ftwist[mid] + delta * ftwist[mid_next]

    cv0 = _host_cfl_value(cv0_j, "radial curvature drift")
    gb0 = _host_cfl_value(gb0_j, "radial grad-B drift")

    m0_max = 0.0
    cv0_max = float(np.max(np.abs(cv0)))
    gb0_max = float(np.max(np.abs(gb0)))
    m0_omega0 = 0.0
    for idz in range(nz):
        term1 = ftwist[idz] - 2.0 * np.pi * zp * kxfac * shat * np.floor(
            idz / (1.0 * nz)
        )
        term2 = ftwist[(idz + 1) % nz] - 2.0 * np.pi * zp * kxfac * shat * np.floor(
            (idz + 1) / (1.0 * nz)
        )
        m0 = -np.rint(x0 * ky_max * ((1.0 - delta) * term1 + delta * term2)) + np.rint(
            x0 * ky_max * ref_term
        )
        omega0 = float(m0) * (
            vpar_max * vpar_max * abs(cv0[idz]) + muB_max * abs(gb0[idz])
        )
        if omega0 > m0_omega0:
            m0_omega0 = omega0
            m0_max = abs(float(m0))
            cv0_max = abs(float(cv0[idz]))
            gb0_max = abs(float(gb0[idz]))
    return m0_max, cv0_max, gb0_max


def _grid_frequency_bounds(grid: SpectralGrid) -> _CFLGridBounds:
    kx, ky, kz = _cfl_wavenumber_arrays(grid)
    nz = kz.size
    nx = kx.size
    ny = source_ny_full(grid)  # the 2/3 index is into the two-sided axis
    kx_max = float(abs(kx[(nx - 1) // 3])) if nx > 1 else 0.0
    ky_max = float(abs(ky[(ny - 1) // 3])) if ky.size > 0 else 0.0
    kz_max = float(abs(kz[nz // 2])) if nz > 0 else 0.0
    kx_min = float(abs(kx[1])) if nx > 1 else np.inf
    ky_min = float(abs(ky[1])) if ky.size > 1 else np.inf
    finite_min = min(kx_min, ky_min)
    kperp_min = float(finite_min) if np.isfinite(finite_min) else 0.0
    return _CFLGridBounds(
        kx=kx,
        ky=ky,
        kz=kz,
        kx_max=kx_max,
        ky_max=ky_max,
        kz_max=kz_max,
        kperp_min=kperp_min,
    )


def _species_frequency_scales(
    params: LinearParams,
    *,
    nl: int,
    nm: int,
) -> _SpeciesFrequencyScales:
    tprim = np.atleast_1d(_host_cfl_value(params.tprim, "tprim"))
    fprim = np.atleast_1d(_host_cfl_value(params.fprim, "fprim"))
    tz = np.atleast_1d(_host_cfl_value(params.tz, "tz"))
    vth = np.atleast_1d(_host_cfl_value(params.vth, "vth"))
    temp = np.atleast_1d(_host_cfl_value(params.temp, "temp"))
    dens = np.atleast_1d(_host_cfl_value(params.density, "density"))
    charge = np.atleast_1d(_host_cfl_value(params.charge_sign, "charge_sign"))

    tzmax = float(np.max(np.abs(tz))) if tz.size else 0.0
    vtmax = float(np.max(np.abs(vth))) if vth.size else 0.0
    vtmin = float(np.min(np.abs(vth))) if vth.size else 1.0
    return _SpeciesFrequencyScales(
        tprim=tprim,
        fprim=fprim,
        temp=temp,
        dens=dens,
        charge=charge,
        tzmax=tzmax,
        vtmax=vtmax,
        vtmin=vtmin,
        etamax=_gradient_ratio_max(tprim, fprim),
        vpar_max=2.0 * float(np.sqrt(max(nm, 1))),
        muB_max=_laguerre_velocity_max(nl),
    )


def _effective_geometry_frequency_scales(
    geom: FluxTubeGeometryLike,
    grid: SpectralGrid,
    grid_bounds: _CFLGridBounds,
    species: _SpeciesFrequencyScales,
) -> _GeometryFrequencyScales:
    (
        bmag_max,
        cvdrift_max,
        gbdrift_max,
        cvdrift0_max,
        gbdrift0_max,
        gradpar,
    ) = _geometry_frequency_maxima(geom, np.asarray(grid.z, dtype=float))

    shat = float(_host_cfl_value(geom.s_hat, "s_hat"))
    non_twist = bool(getattr(grid, "non_twist", False))
    m0_max = 0.0
    if non_twist and abs(shat) > 0.0 and grid_bounds.ky.size > 0:
        m0_max, cvdrift0_max, gbdrift0_max = _non_twist_shift_frequency_max(
            geom,
            grid,
            float(grid_bounds.ky[-1]),
            vpar_max=species.vpar_max,
            muB_max=species.muB_max,
        )
    return _GeometryFrequencyScales(
        bmag_max=bmag_max,
        cvdrift_max=cvdrift_max,
        gbdrift_max=gbdrift_max,
        cvdrift0_max=cvdrift0_max,
        gbdrift0_max=gbdrift0_max,
        gradpar=gradpar,
        m0_max=m0_max,
    )


def _radial_drift_frequency(
    geom: FluxTubeGeometryLike,
    grid: SpectralGrid,
    grid_bounds: _CFLGridBounds,
    species: _SpeciesFrequencyScales,
    geometry: _GeometryFrequencyScales,
) -> float:
    drift_strength = species.vpar_max * species.vpar_max * abs(
        geometry.cvdrift0_max
    ) + species.muB_max * abs(geometry.gbdrift0_max)
    shat = float(_host_cfl_value(geom.s_hat, "s_hat"))
    if abs(shat) == 0.0:
        kx_effective = grid_bounds.kx_max
    elif bool(getattr(grid, "non_twist", False)):
        kx_effective = grid_bounds.kx_max + geometry.m0_max / float(grid.x0)
    else:
        kx_effective = grid_bounds.kx_max / abs(shat)
    return species.tzmax * kx_effective * drift_strength


def _binormal_drift_frequency(
    grid_bounds: _CFLGridBounds,
    species: _SpeciesFrequencyScales,
    geometry: _GeometryFrequencyScales,
    *,
    include_diamagnetic_drive: bool,
) -> float:
    omega = (
        species.tzmax
        * grid_bounds.ky_max
        * (
            species.vpar_max * species.vpar_max * geometry.cvdrift_max
            + species.muB_max * geometry.gbdrift_max
        )
    )
    if include_diamagnetic_drive and species.etamax < 1.0e5:
        omega += grid_bounds.ky_max * (
            1.0
            + species.etamax
            * (species.vpar_max * species.vpar_max / 2.0 + species.muB_max - 1.5)
        )
    return omega


def _electron_pressure_product(species: _SpeciesFrequencyScales) -> float:
    if species.charge.size:
        neg = species.charge < 0.0
        if np.any(neg):
            ne = float(species.dens[neg][0])
            Te = float(species.temp[neg][0])
        else:
            ne = float(species.dens[0])
            Te = float(species.temp[0])
    else:
        ne = 1.0
        Te = 1.0
    return ne * Te


def _parallel_streaming_frequency(
    params: LinearParams,
    grid_bounds: _CFLGridBounds,
    species: _SpeciesFrequencyScales,
    geometry: _GeometryFrequencyScales,
) -> float:
    beta = float(_host_cfl_value(params.beta, "beta"))
    nspec_in = int(max(species.charge.size, 1))
    nte = _electron_pressure_product(species)
    mime = (
        (species.vtmax * species.vtmax) / (species.vtmin * species.vtmin)
        if species.vtmin > 0.0
        else 0.0
    )
    kperprho2 = (
        grid_bounds.kperp_min
        * grid_bounds.kperp_min
        / (geometry.bmag_max * geometry.bmag_max)
        if geometry.bmag_max > 0.0
        else 0.0
    )
    if nspec_in > 1:
        denom = beta * nte / 2.0 * mime + kperprho2
        guard = 1.0 / np.sqrt(denom) if denom > 0.0 else 0.0
    else:
        guard = 0.0
    return (
        species.vtmax
        * grid_bounds.kz_max
        * abs(geometry.gradpar)
        * max(species.vpar_max, guard)
    )


def _linear_frequency_bound(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    nl: int,
    nm: int,
    *,
    include_diamagnetic_drive: bool = True,
    measure: bool = False,
) -> np.ndarray:
    grid_bounds = _grid_frequency_bounds(grid)
    species = _species_frequency_scales(params, nl=nl, nm=nm)
    geometry = _effective_geometry_frequency_scales(geom, grid, grid_bounds, species)
    omega_max = np.zeros(3, dtype=float)
    omega_max[0] = _radial_drift_frequency(geom, grid, grid_bounds, species, geometry)
    omega_max[1] = _binormal_drift_frequency(
        grid_bounds,
        species,
        geometry,
        include_diamagnetic_drive=include_diamagnetic_drive,
    )
    omega_max[2] = _parallel_streaming_frequency(params, grid_bounds, species, geometry)
    # The analytic streaming estimate misses the electromagnetic electron mode
    # of a kinetic-electron box and the hypercollision rate (see
    # ``measured_streaming_frequency``); the nonlinear path measures its own.
    vth = np.abs(np.atleast_1d(np.asarray(params.vth, dtype=float)))
    if measure and vth.size > 1 and float(vth.max()) > 4.0 * float(vth.min()):
        from gkx.operators.linear.cache_builder import build_linear_cache
        from gkx.operators.nonlinear.policies import measured_streaming_frequency

        cache = build_linear_cache(grid, geom, params, Nl=nl, Nm=nm)
        omega_max[2] = max(
            omega_max[2],
            measured_streaming_frequency(
                params, cache, nl=nl, nm=nm, hypercollisions=1.0
            ),
        )
    return omega_max


def warn_if_fixed_dt_exceeds_cfl(
    *,
    grid: Any,
    geom: Any,
    params: Any,
    n_laguerre: int,
    n_hermite: int,
    tcfg: Any,
    measure: bool = False,
) -> float | None:
    """Warn at startup when a fixed linear dt exceeds the explicit CFL estimate.

    The fixed-step linear paths advance the whole physical RHS explicitly (the
    imex methods only treat damping implicitly), so the explicit-time frequency
    bound applies to them too. An over-CFL dt overflows the trajectory, and the
    growth fit then fails loudly; this names the cause before the run starts.
    Returns the CFL-stable step, or None when it cannot be bounded. ``measure``
    adds the Arnoldi radius (one cache build) for kinetic-electron decks.

    Geometry is conformed to this grid and any failure is swallowed: an
    imported equilibrium carries a closed theta interval one point longer than
    the grid, which made this hint abort every ``gkx wout.nc --linear`` run. A
    diagnostic that can stop a working run is worse than no diagnostic.
    """

    from gkx.config import resolve_cfl_fac
    from gkx.geometry import ensure_flux_tube_geometry_data

    try:
        geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
        wmax = float(
            np.sum(
                _linear_frequency_bound(
                    grid, geom_eff, params, n_laguerre, n_hermite, measure=measure
                )
            )
        )
    except Exception:  # noqa: BLE001 - a startup hint must never stop a run
        return None
    if not np.isfinite(wmax) or wmax <= 0.0:
        return None
    cfl_fac = resolve_cfl_fac(str(tcfg.method), tcfg.cfl_fac)
    dt_stable = cfl_fac * float(tcfg.cfl) / wmax
    if float(tcfg.dt) > dt_stable:
        warnings.warn(
            f"requested dt={float(tcfg.dt):.4g} exceeds the estimated "
            f"CFL-stable step {dt_stable:.4g} (max linear frequency "
            f"{wmax:.4g}); the integration may overflow -- reduce dt",
            RuntimeWarning,
            stacklevel=2,
        )
    return dt_stable


LinearStageRhsFn = Callable[[jnp.ndarray], jnp.ndarray]


_SSPX3_ADT = float((1.0 / 6.0) ** (1.0 / 3.0))
_SSPX3_WGTFAC = float((9.0 - 2.0 * (6.0 ** (2.0 / 3.0))) ** 0.5)
_SSPX3_W1 = 0.5 * (_SSPX3_WGTFAC - 1.0)
_SSPX3_W2 = 0.5 * ((6.0 ** (2.0 / 3.0)) - 1.0 - _SSPX3_WGTFAC)
_SSPX3_W3 = (1.0 / _SSPX3_ADT) - 1.0 - _SSPX3_W2 * (_SSPX3_W1 + 1.0)
_IMEX2_GAMMA = 1.0 - 1.0 / 2.0**0.5
_IMEX2_DELTA = 1.0 - 1.0 / (2.0 * _IMEX2_GAMMA)


def _completed_step_state_mask(cache: LinearCache) -> jnp.ndarray:
    """Return the completed-step state-space mask applied after each completed step."""

    mask = _growth_rate_mode_mask(cache.ky, cache.kx, cache.dealias_mask)
    ky_zero = jnp.isclose(jnp.asarray(cache.ky), 0.0)
    kx_zero = jnp.isclose(jnp.asarray(cache.kx), 0.0)
    zonal00 = ky_zero[:, None] & kx_zero[None, :]
    return mask & ~zonal00


def _apply_completed_step_state_mask(
    state: jnp.ndarray, cache: LinearCache
) -> jnp.ndarray:
    """Apply the completed-step mask to a spectral state array."""

    mask = _completed_step_state_mask(cache).astype(state.real.dtype)[..., None]
    shape = (1,) * (state.ndim - mask.ndim) + mask.shape
    return state * jnp.reshape(mask, shape)


def _growth_rate_mode_mask(
    ky: jnp.ndarray,
    kx: jnp.ndarray,
    dealias_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Return the diagnostic mask used by explicit-time growth-rate extraction.

    Single selected nonzonal ``ky`` slices should remain diagnosable even when
    the originating full nonlinear mesh would mark that representative row as
    dealiased away.
    """

    mask = jnp.asarray(dealias_mask, dtype=bool)
    ky_arr = jnp.asarray(ky, dtype=float).reshape(-1)
    if mask.ndim == 2 and int(mask.shape[0]) == 1:
        promote = (~jnp.any(mask)) & jnp.any(jnp.abs(ky_arr) > 0.0)
        mask = jnp.where(promote, jnp.ones_like(mask, dtype=bool), mask)
    ky_zero = jnp.isclose(ky_arr, 0.0)
    kx_zero = jnp.isclose(jnp.asarray(kx, dtype=float).reshape(-1), 0.0)
    zonal00 = ky_zero[:, None] & kx_zero[None, :]
    return mask & ~zonal00


def _diagnostic_midplane_index(nz: int) -> int:
    if nz <= 1:
        return 0
    idx = nz // 2 + 1
    return min(idx, nz - 1)


def _instantaneous_growth_rate_step(
    phi_now: jnp.ndarray,
    phi_prev: jnp.ndarray,
    dt: float | jax.Array,
    *,
    z_index: int,
    mask: jnp.ndarray,
    mode_method: str = "z_index",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Instantaneous growth rates from phi ratios at the midplane."""

    if mode_method == "z_index":
        phi_now_z = phi_now[..., z_index]
        phi_prev_z = phi_prev[..., z_index]
    elif mode_method == "max":
        now_idx = jnp.argmax(jnp.abs(phi_now), axis=-1, keepdims=True)
        prev_idx = jnp.argmax(jnp.abs(phi_prev), axis=-1, keepdims=True)
        phi_now_z = jnp.take_along_axis(phi_now, now_idx, axis=-1)[..., 0]
        phi_prev_z = jnp.take_along_axis(phi_prev, prev_idx, axis=-1)[..., 0]
    else:
        raise ValueError("mode_method must be 'z_index' or 'max'")
    # Keep the diagnostic conservative: require non-zero real and imaginary
    # parts of phi at the current step only.
    valid = (jnp.abs(jnp.real(phi_now_z)) > 0.0) & (jnp.abs(jnp.imag(phi_now_z)) > 0.0)
    ratio = jnp.where(phi_prev_z != 0.0, phi_now_z / phi_prev_z, 0.0 + 0.0j)
    log_amp = jnp.log(jnp.abs(ratio))
    phase = jnp.angle(ratio)
    gamma = jnp.where(mask & valid, log_amp / dt, 0.0)
    omega = jnp.where(mask & valid, -phase / dt, 0.0)
    return gamma, omega


def _linear_term_config(terms: LinearTerms | None) -> TermConfig:
    return linear_terms_to_term_config(terms)


def _linear_stage_rhs(
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    assemble_rhs_cached_fn=assemble_rhs_cached,
) -> LinearStageRhsFn:
    """Return the linear RHS closure used by explicit staged methods."""

    def rhs(state: jnp.ndarray) -> jnp.ndarray:
        dG, _fields = assemble_rhs_cached_fn(state, cache, params, terms=term_cfg)
        return dG

    return rhs


def _linear_rk3_heun_state(
    G: jnp.ndarray,
    dt_val: jnp.ndarray,
    *,
    k1: jnp.ndarray,
    rhs: LinearStageRhsFn,
) -> jnp.ndarray:
    G1 = G + (dt_val / 3.0) * k1
    k2 = rhs(G1)
    G2 = G + (2.0 * dt_val / 3.0) * k2
    k3 = rhs(G2)
    G3 = G + 0.75 * dt_val * k3
    return G3 + 0.25 * dt_val * k1


def _linear_rk4_state(
    G: jnp.ndarray,
    dt_val: jnp.ndarray,
    *,
    k1: jnp.ndarray,
    rhs: LinearStageRhsFn,
) -> jnp.ndarray:
    k2 = rhs(G + 0.5 * dt_val * k1)
    k3 = rhs(G + 0.5 * dt_val * k2)
    k4 = rhs(G + dt_val * k3)
    return G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _linear_sspx3_state(
    G: jnp.ndarray,
    dt_val: jnp.ndarray,
    *,
    rhs: LinearStageRhsFn,
) -> jnp.ndarray:
    def euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
        return G_state + (_SSPX3_ADT * dt_val) * rhs(G_state)

    G1 = euler_step(G)
    G2_euler = euler_step(G1)
    G2 = (1.0 - _SSPX3_W1) * G + (_SSPX3_W1 - 1.0) * G1 + G2_euler
    G3 = euler_step(G2)
    return (
        (1.0 - _SSPX3_W2 - _SSPX3_W3) * G + _SSPX3_W3 * G1 + (_SSPX3_W2 - 1.0) * G2 + G3
    )


def _linear_explicit_stage_update(
    G: jnp.ndarray,
    dt_val: jnp.ndarray,
    *,
    method_key: str,
    rhs: LinearStageRhsFn,
) -> jnp.ndarray:
    """Advance one explicit method before post-step masking and field solve."""

    # These schemes build their own first stage; evaluating ``rhs(G)`` here would
    # add an unused full gyrokinetic RHS evaluation to every completed step.
    if method_key == "sspx3":
        return _linear_sspx3_state(G, dt_val, rhs=rhs)

    k1 = rhs(G)
    if method_key == "rk2":
        return G + dt_val * rhs(G + 0.5 * dt_val * k1)
    if method_key == "rk3":
        return _linear_rk3_heun_state(G, dt_val, k1=k1, rhs=rhs)
    if method_key == "rk4":
        return _linear_rk4_state(G, dt_val, k1=k1, rhs=rhs)
    raise method_error(method_key, {"rk2", "rk3", "rk4", "sspx3"})


def _linear_native_step(
    G: jnp.ndarray,
    damping: jnp.ndarray,
    dt_val: jnp.ndarray,
    *,
    method_key: str,
    rhs: LinearStageRhsFn,
) -> jnp.ndarray:
    """Advance one native explicit or diagonal-IMEX linear step."""

    if method_key == "imex":
        dG = rhs(G)
        dG_explicit = dG + damping * G
        return (G + dt_val * dG_explicit) / (1.0 + dt_val * damping)
    if method_key == "imex2":
        dG = rhs(G)
        dG_explicit = dG + damping * G
        G_stage = (G + _IMEX2_GAMMA * dt_val * dG_explicit) / (
            1.0 + _IMEX2_GAMMA * dt_val * damping
        )
        dG_stage_explicit = rhs(G_stage) + damping * G_stage
        return (
            G
            + dt_val
            * (_IMEX2_DELTA * dG_explicit + (1.0 - _IMEX2_DELTA) * dG_stage_explicit)
            - (1.0 - _IMEX2_GAMMA) * dt_val * damping * G_stage
        ) / (1.0 + _IMEX2_GAMMA * dt_val * damping)
    return _linear_explicit_stage_update(
        G,
        dt_val,
        method_key=method_key,
        rhs=rhs,
    )


def _linear_explicit_step_impl(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    dt: float,
    *,
    method: str,
    assemble_rhs_cached_fn=assemble_rhs_cached,
) -> tuple[jnp.ndarray, FieldState]:
    """Single explicit linear step matching explicit staged schemes."""

    dt_val = jnp.asarray(dt)
    method_key = method.strip().lower()
    rhs = _linear_stage_rhs(cache, params, term_cfg, assemble_rhs_cached_fn)
    G_next = _linear_explicit_stage_update(
        G,
        dt_val,
        method_key=method_key,
        rhs=rhs,
    )

    # Mask inactive modes only after the full explicit step, before the next field solve.
    G_next = _apply_completed_step_state_mask(jnp.asarray(G_next), cache)

    # fields at the end of step
    _, fields = assemble_rhs_cached_fn(G_next, cache, params, terms=term_cfg)
    return G_next, fields


__all__ = [
    "ExplicitTimeConfig",
    "ExplicitTimeConfigLike",
    "FIXED_DT_CFL_WARN_RATIO",
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
    "_linear_native_step",
    "_linear_term_config",
    "_non_twist_shift_frequency_max",
    "_parallel_periods_from_grid",
    "integrate_linear_explicit",
    "integrate_linear_explicit_diagnostics",
    "integrate_linear_explicit_from_config",
]
