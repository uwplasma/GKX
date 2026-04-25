"""Config-driven runners for time integration."""

from __future__ import annotations

from typing import cast

from spectraxgk.analysis import ModeSelection, ModeSelectionBatch
from spectraxgk.config import TimeConfig
from spectraxgk.diffrax_integrators import integrate_linear_diffrax, integrate_nonlinear_diffrax
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import LinearCache, LinearParams, LinearTerms, build_linear_cache, integrate_linear
from spectraxgk.nonlinear import integrate_nonlinear
from spectraxgk.sharding import resolve_state_sharding
from spectraxgk.sharded_integrators import integrate_nonlinear_sharded
from spectraxgk.terms.config import TermConfig


def _steps_from_time(cfg: TimeConfig) -> int:
    if cfg.dt <= 0.0:
        raise ValueError("TimeConfig.dt must be > 0")
    steps = int(round(cfg.t_max / cfg.dt))
    if steps < 1:
        raise ValueError("TimeConfig.t_max must be >= dt")
    return steps


def _validate_nonlinear_config_state_sharding(spec: str | None) -> None:
    """Keep config-level nonlinear sharding on release-gated state axes."""

    if spec is None:
        return
    key = str(spec).strip().lower()
    if key in {"", "none", "off", "false", "0"}:
        return
    if key not in {"auto", "ky", "kx"}:
        raise ValueError(
            "nonlinear TimeConfig.state_sharding currently supports only 'auto', 'ky', 'kx', or 'none'. "
            "Sharding along the z FFT axis is an exploratory domain-decomposition lane and is not a "
            "release-gated runtime path."
        )


def integrate_linear_from_config(
    G0,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    time_cfg: TimeConfig,
    *,
    cache: LinearCache | None = None,
    terms: LinearTerms | None = None,
    save_mode: ModeSelection | ModeSelectionBatch | None = None,
    mode_method: str = "z_index",
    save_field: str = "phi",
    density_species_index: int | None = None,
    show_progress: bool | None = None,
) -> tuple:
    """Integrate the linear system using TimeConfig settings."""

    steps = _steps_from_time(time_cfg)
    show_progress_use = bool(time_cfg.progress_bar if show_progress is None else show_progress)
    if time_cfg.use_diffrax:
        state_sharding = resolve_state_sharding(G0, time_cfg.state_sharding)
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
            show_progress=show_progress_use,
            progress_bar=show_progress_use,
            checkpoint=time_cfg.checkpoint,
            sample_stride=time_cfg.sample_stride,
            return_state=time_cfg.save_state,
            save_mode=save_mode,
            mode_method=mode_method,
            save_field=save_field,
            density_species_index=density_species_index,
            state_sharding=state_sharding,
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
        show_progress=show_progress_use,
    )


def integrate_nonlinear_from_config(
    G0,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    time_cfg: TimeConfig,
    *,
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    show_progress: bool | None = None,
) -> tuple:
    """Integrate the nonlinear system using TimeConfig settings."""

    steps = _steps_from_time(time_cfg)
    show_progress_use = bool(time_cfg.progress_bar if show_progress is None else show_progress)
    _validate_nonlinear_config_state_sharding(time_cfg.state_sharding)
    state_sharding = resolve_state_sharding(G0, time_cfg.state_sharding)
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
            show_progress=show_progress_use,
            progress_bar=show_progress_use,
            checkpoint=time_cfg.checkpoint,
            gx_real_fft=time_cfg.gx_real_fft,
            laguerre_mode=time_cfg.laguerre_nonlinear_mode,
            state_sharding=state_sharding,
        )
    if state_sharding is not None:
        if cache is None:
            if G0.ndim == 5:
                nl, nm = G0.shape[0], G0.shape[1]
            elif G0.ndim == 6:
                nl, nm = G0.shape[1], G0.shape[2]
            else:
                raise ValueError("G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
            cache = build_linear_cache(grid, geom, params, int(nl), int(nm))
        return cast(
            tuple,
            integrate_nonlinear_sharded(
                G0,
                cache,
                params,
                dt=time_cfg.dt,
                steps=steps,
                method=time_cfg.method,
                terms=terms,
                state_sharding=state_sharding,
                gx_real_fft=time_cfg.gx_real_fft,
                laguerre_mode=time_cfg.laguerre_nonlinear_mode,
                return_fields=True,
            ),
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
        gx_real_fft=time_cfg.gx_real_fft,
        laguerre_mode=time_cfg.laguerre_nonlinear_mode,
        show_progress=show_progress_use,
    )
