"""Diagnostics-rich explicit linear time integration."""

from __future__ import annotations

from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.core.grid import SpectralGrid
from spectraxgk.diagnostics import (
    SimulationDiagnostics,
    distribution_free_energy,
    electrostatic_field_energy,
    fieldline_quadrature_weights,
    heat_flux_total,
    magnetic_vector_potential_energy,
    particle_flux_total,
    total_energy,
)
from spectraxgk.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.params import LinearParams, LinearTerms
from spectraxgk.solvers.time.explicit_cfl import _linear_frequency_bound
from spectraxgk.solvers.time.explicit_steps import (
    _diagnostic_midplane_index,
    _growth_rate_mode_mask,
    _instantaneous_growth_rate_step,
    _linear_explicit_step as _linear_explicit_step_impl,
    _linear_term_config,
)
from spectraxgk.terms.assembly import assemble_rhs_cached

__all__ = ["ExplicitTimeConfigLike", "integrate_linear_explicit_diagnostics"]


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

    if show_progress:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        console.print(
            Panel.fit(
                "[bold blue]SPECTRAX-GK[/bold blue] | [bold green]Explicit Linear Simulation Started[/bold green]",
                border_style="blue",
            )
        )

    if mode_method not in {"z_index", "max"}:
        raise ValueError("mode_method must be 'z_index' or 'max'")

    if terms is None:
        terms = LinearTerms()
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
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)

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

    _, fields0 = assemble_rhs_cached(G, cache, params, terms=term_cfg, dt=dt)
    phi_prev = fields0.phi

    omega_max = _linear_frequency_bound(
        grid, geom_eff, params, G.shape[-5], G.shape[-4]
    )
    wmax = float(np.sum(omega_max))
    if not time_cfg.fixed_dt and wmax > 0.0:
        dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
        dt = min(max(dt_guess, dt_min), dt_max)

    ts: list[float] = []
    phi_list: list[np.ndarray] = []
    gamma_list: list[np.ndarray] = []
    omega_list: list[np.ndarray] = []
    dt_list: list[float] = []
    Wg_list: list[float] = []
    Wphi_list: list[float] = []
    Wapar_list: list[float] = []
    heat_list: list[float] = []
    pflux_list: list[float] = []

    vol_fac, flux_fac = fieldline_quadrature_weights(geom_eff, grid)
    use_dealias = bool(time_cfg.use_dealias_mask)
    step_fn = (
        _linear_explicit_step
        if linear_explicit_step_fn is None
        else linear_explicit_step_fn
    )

    def _step(G_state, cache_state, params_state, term_cfg_state, dt_state):
        return step_fn(
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
        step += 1
        t += dt

        if step % sample_stride == 0 or t >= t_max:
            phi = fields.phi
            apar = fields.apar if fields.apar is not None else jnp.zeros_like(phi)
            bpar = fields.bpar if fields.bpar is not None else jnp.zeros_like(phi)
            gamma, omega = _instantaneous_growth_rate_step(
                phi,
                phi_prev,
                dt,
                z_index=z_idx,
                mask=mask,
                mode_method=mode_method,
            )
            ts.append(t)
            dt_list.append(float(dt))
            phi_list.append(np.asarray(phi))
            gamma_list.append(np.asarray(gamma))
            omega_list.append(np.asarray(omega))

            Wg_val = distribution_free_energy(
                G, grid, params, vol_fac, use_dealias=use_dealias
            )
            Wphi_val = electrostatic_field_energy(
                phi,
                cache,
                params,
                vol_fac,
                use_dealias=use_dealias,
            )
            Wapar_val = magnetic_vector_potential_energy(
                apar, cache, vol_fac, use_dealias=use_dealias
            )
            heat_val = heat_flux_total(
                G,
                phi,
                apar,
                bpar,
                cache,
                grid,
                params,
                flux_fac,
                use_dealias=use_dealias,
            )
            pflux_val = particle_flux_total(
                G,
                phi,
                apar,
                bpar,
                cache,
                grid,
                params,
                flux_fac,
                use_dealias=use_dealias,
            )

            Wg_list.append(float(Wg_val))
            Wphi_list.append(float(Wphi_val))
            Wapar_list.append(float(Wapar_val))
            heat_list.append(float(heat_val))
            pflux_list.append(float(pflux_val))

            if show_progress:
                # This path runs in a Python loop, so rich output is sufficient.
                from rich.table import Table
                from rich import box

                pct = (t / t_max) * 100
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
                    f"{pct:>3.0f}%",
                    str(step),
                    f"{float(t):.2f}",
                    f"{float(Wg_val):.4e}",
                    f"{float(Wphi_val):.4e}",
                    f"{float(heat_val):.4e}",
                )
                console.print(table)

        phi_prev = fields.phi

    if show_progress:
        from rich.panel import Panel

        console.print(
            Panel.fit(
                "[bold green]Simulation Complete![/bold green]", border_style="green"
            )
        )

    diag = SimulationDiagnostics(
        t=np.asarray(ts),
        dt_t=np.asarray(dt_list),
        dt_mean=np.asarray(np.mean(dt_list)) if dt_list else np.asarray(0.0),
        gamma_t=np.asarray(gamma_list),
        omega_t=np.asarray(omega_list),
        Wg_t=np.asarray(Wg_list),
        Wphi_t=np.asarray(Wphi_list),
        Wapar_t=np.asarray(Wapar_list),
        heat_flux_t=np.asarray(heat_list),
        particle_flux_t=np.asarray(pflux_list),
        energy_t=np.asarray(
            total_energy(
                np.asarray(Wg_list), np.asarray(Wphi_list), np.asarray(Wapar_list)
            )
        ),
    )
    return (
        np.asarray(ts),
        np.asarray(phi_list),
        np.asarray(gamma_list),
        np.asarray(omega_list),
        diag,
    )
