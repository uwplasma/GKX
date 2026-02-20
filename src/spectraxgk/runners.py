"""Config-driven runners for time integration."""

from __future__ import annotations

from typing import Tuple

from spectraxgk.analysis import ModeSelection, ModeSelectionBatch
from spectraxgk.config import TimeConfig
from spectraxgk.diffrax_integrators import integrate_linear_diffrax, integrate_nonlinear_diffrax
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import LinearCache, LinearParams, LinearTerms, integrate_linear
from spectraxgk.nonlinear import integrate_nonlinear
from spectraxgk.terms.config import FieldState, TermConfig


def _steps_from_time(cfg: TimeConfig) -> int:
    if cfg.dt <= 0.0:
        raise ValueError("TimeConfig.dt must be > 0")
    steps = int(round(cfg.t_max / cfg.dt))
    if steps < 1:
        raise ValueError("TimeConfig.t_max must be >= dt")
    return steps


def integrate_linear_from_config(
    G0,
    grid: SpectralGrid,
    geom: SAlphaGeometry,
    params: LinearParams,
    time_cfg: TimeConfig,
    *,
    cache: LinearCache | None = None,
    terms: LinearTerms | None = None,
    save_mode: ModeSelection | ModeSelectionBatch | None = None,
    mode_method: str = "z_index",
    save_field: str = "phi",
    density_species_index: int | None = None,
) -> tuple:
    """Integrate the linear system using TimeConfig settings."""

    steps = _steps_from_time(time_cfg)
    if time_cfg.use_diffrax:
        return integrate_linear_diffrax(
            G0,
            grid,
            geom,
            params,
            dt=time_cfg.dt,
            steps=steps,
            method=time_cfg.diffrax_solver,
            cache=cache,
            terms=terms,
            adaptive=time_cfg.diffrax_adaptive,
            rtol=time_cfg.diffrax_rtol,
            atol=time_cfg.diffrax_atol,
            max_steps=time_cfg.diffrax_max_steps,
            progress_bar=time_cfg.progress_bar,
            checkpoint=time_cfg.checkpoint,
            sample_stride=time_cfg.sample_stride,
            return_state=time_cfg.save_state,
            save_mode=save_mode,
            mode_method=mode_method,
            save_field=save_field,
            density_species_index=density_species_index,
        )
    return integrate_linear(
        G0,
        grid,
        geom,
        params,
        dt=time_cfg.dt,
        steps=steps,
        method=time_cfg.method,
        cache=cache,
        implicit_restart=time_cfg.implicit_restart,
        implicit_preconditioner=time_cfg.implicit_preconditioner,
        implicit_solve_method=time_cfg.implicit_solve_method,
        checkpoint=time_cfg.checkpoint,
        sample_stride=time_cfg.sample_stride,
        terms=terms,
    )


def integrate_nonlinear_from_config(
    G0,
    grid: SpectralGrid,
    geom: SAlphaGeometry,
    params: LinearParams,
    time_cfg: TimeConfig,
    *,
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
) -> tuple:
    """Integrate the nonlinear system using TimeConfig settings."""

    steps = _steps_from_time(time_cfg)
    if time_cfg.use_diffrax:
        return integrate_nonlinear_diffrax(
            G0,
            grid,
            geom,
            params,
            dt=time_cfg.dt,
            steps=steps,
            method=time_cfg.diffrax_solver,
            cache=cache,
            terms=terms,
            adaptive=time_cfg.diffrax_adaptive,
            rtol=time_cfg.diffrax_rtol,
            atol=time_cfg.diffrax_atol,
            max_steps=time_cfg.diffrax_max_steps,
            progress_bar=time_cfg.progress_bar,
            checkpoint=time_cfg.checkpoint,
        )
    return integrate_nonlinear(
        G0,
        grid,
        geom,
        params,
        dt=time_cfg.dt,
        steps=steps,
        method=time_cfg.method,
        cache=cache,
        terms=terms,
        checkpoint=time_cfg.checkpoint,
    )
