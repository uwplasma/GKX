"""Diagnostics-rich explicit linear time integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from gkx.core.grid import SpectralGrid
from gkx.diagnostics import (
    SimulationDiagnostics,
    distribution_free_energy,
    electrostatic_field_energy,
    fieldline_quadrature_weights,
    heat_flux_total,
    magnetic_vector_potential_energy,
    particle_flux_total,
    total_energy,
)
from gkx.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.params import LinearParams, LinearTerms
from gkx.solvers.time.explicit_cfl import _linear_frequency_bound
from gkx.solvers.time.explicit_steps import (
    _diagnostic_midplane_index,
    _growth_rate_mode_mask,
    _instantaneous_growth_rate_step,
    _linear_explicit_step as _linear_explicit_step_impl,
    _linear_term_config,
)
from gkx.terms.assembly import assemble_rhs_cached

__all__ = ["ExplicitTimeConfigLike", "integrate_linear_explicit_diagnostics"]

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


def _linear_explicit_step(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg,
    dt: float,
    *,
    method: str,
):
    return _linear_explicit_step_impl(
        G,
        cache,
        params,
        term_cfg,
        dt,
        method=method,
        assemble_rhs_cached_fn=assemble_rhs_cached,
    )


def _start_progress(show_progress: bool) -> Any | None:
    if not show_progress:
        return None
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(
        Panel.fit(
            "[bold blue]GKX[/bold blue] | [bold green]Explicit Linear Simulation Started[/bold green]",
            border_style="blue",
        )
    )
    return console


def _finish_progress(console: Any | None) -> None:
    if console is None:
        return
    from rich.panel import Panel

    console.print(
        Panel.fit("[bold green]Simulation Complete![/bold green]", border_style="green")
    )


def _normalize_method(method: str) -> str:
    method_key = method.strip().lower()
    if method_key not in _ALLOWED_METHODS:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk3', 'rk3_classic', 'rk3_heun', 'rk4', 'k10', 'sspx3'}"
        )
    return method_key


def _validate_mode_method(mode_method: str) -> None:
    if mode_method not in {"z_index", "max"}:
        raise ValueError("mode_method must be 'z_index' or 'max'")


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
            np.sum(_linear_frequency_bound(grid, geom_eff, params, G.shape[-5], G.shape[-4]))
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
        G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=policy.use_dealias
    )
    particle = particle_flux_total(
        G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=policy.use_dealias
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
    console: Any | None,
    *,
    step: int,
    sample_stride: int,
    t: float,
    t_max: float,
    sample: _DiagnosticSample,
) -> None:
    if console is None:
        return
    from rich import box
    from rich.table import Table

    table = Table(
        box=box.HORIZONTALS,
        show_header=(step <= sample_stride),
        header_style="bold magenta",
    )
    if step <= sample_stride:
        table.add_column("Progress", justify="right", style="cyan")
        table.add_column("Step", justify="right", style="green")
        table.add_column("Time", justify="right", style="yellow")
        table.add_column("Wg", justify="right")
        table.add_column("Wphi", justify="right")
        table.add_column("Heat", justify="right")
    table.add_row(
        f"{(t / t_max) * 100:>3.0f}%",
        str(step),
        f"{float(t):.2f}",
        f"{sample.Wg:.4e}",
        f"{sample.Wphi:.4e}",
        f"{sample.heat:.4e}",
    )
    console.print(table)


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
    """Explicit time integrator with growth-rate plus energy/flux diagnostics."""

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
        _linear_explicit_step if linear_explicit_step_fn is None else linear_explicit_step_fn,
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
        dt_current = policy.step_dt()
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
                    sample_stride=policy.sample_stride,
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
