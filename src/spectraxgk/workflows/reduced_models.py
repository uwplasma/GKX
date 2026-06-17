"""Executable workflows for reduced gyrokinetic models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from spectraxgk.analysis import ModeSelection, select_ky_index
from spectraxgk.geometry import apply_geometry_grid_defaults
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.runtime_config import RuntimeConfig
from spectraxgk.runtime_policies import _midplane_index, _normalize_linear_solver_name
from spectraxgk.runtime_results import RuntimeLinearResult


@dataclass(frozen=True)
class CETGLinearRuntimeDeps:
    """Injected cETG workflow dependencies owned by the public runtime facade."""

    build_runtime_geometry: Callable[[RuntimeConfig], Any]
    validate_cetg_runtime_config: Callable[..., Any]
    build_initial_condition: Callable[..., Any]
    build_runtime_term_config: Callable[[RuntimeConfig], Any]
    build_cetg_model_params: Callable[..., Any]
    integrate_cetg_explicit_diagnostics_state: Callable[..., Any]
    fit_growth_rate_auto: Callable[..., tuple[float, float, float | None, float | None]]
    fit_growth_rate: Callable[..., tuple[float, float]]


def _resolved_fit_bounds(
    t_arr: np.ndarray,
    tmin_fit: float | None,
    tmax_fit: float | None,
) -> tuple[float | None, float | None]:
    if t_arr.size == 0:
        return None, None
    tmin_use = float(tmin_fit) if tmin_fit is not None else float(t_arr[0])
    tmax_use = float(tmax_fit) if tmax_fit is not None else float(t_arr[-1])
    return tmin_use, tmax_use


def run_cetg_linear_runtime(
    cfg: RuntimeConfig,
    *,
    deps: CETGLinearRuntimeDeps,
    ky_target: float,
    Nl: int,
    Nm: int,
    solver: str,
    method: str | None,
    dt: float | None,
    steps: int | None,
    sample_stride: int | None,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    return_state: bool,
    status_callback: Callable[[str], None] | None = None,
) -> RuntimeLinearResult:
    """Run the cETG reduced-model linear runtime path."""

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    geom = deps.build_runtime_geometry(cfg)
    deps.validate_cetg_runtime_config(cfg, geom, Nl=Nl, Nm=Nm)
    _status("building spectral grid")
    grid_cfg = apply_geometry_grid_defaults(geom, cfg.grid)
    grid_full = build_spectral_grid(grid_cfg)
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    _status(f"selected ky index {ky_index} at ky={float(grid.ky[0]):.4f}")
    _status("building initial condition")
    g0 = deps.build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        nspecies=1,
    )
    cetg_terms = deps.build_runtime_term_config(cfg)
    cetg_params = deps.build_cetg_model_params(cfg, geom, Nl=Nl, Nm=Nm)
    solver_key = _normalize_linear_solver_name(solver)
    if solver_key == "krylov":
        raise NotImplementedError(
            "solver='krylov' is not implemented for physics.reduced_model='cetg'"
        )
    if solver_key not in {"auto", "time", "explicit_time"}:
        raise ValueError(
            "solver must be one of {'auto', 'time', 'explicit_time', 'krylov'}"
        )
    dt_val = float(cfg.time.dt if dt is None else dt)
    if dt_val <= 0.0:
        raise ValueError("dt must be > 0")
    steps_val = (
        int(steps) if steps is not None else int(round(float(cfg.time.t_max) / dt_val))
    )
    if steps_val < 1:
        raise ValueError("steps must be >= 1")
    sample_stride_use = int(cfg.time.sample_stride if sample_stride is None else sample_stride)
    _status(f"running cETG time integration over {steps_val} steps")
    _t, diag, G_final, _fields = deps.integrate_cetg_explicit_diagnostics_state(
        g0,
        grid,
        cetg_params,
        cetg_terms,
        dt=dt_val,
        steps=steps_val,
        method=str(method or cfg.time.method),
        sample_stride=sample_stride_use,
        diagnostics_stride=1,
        compressed_real_fft=bool(cfg.time.compressed_real_fft),
        omega_ky_index=0,
        omega_kx_index=0,
        fixed_dt=bool(cfg.time.fixed_dt),
        dt_min=float(cfg.time.dt_min),
        dt_max=cfg.time.dt_max,
        cfl=float(cfg.time.cfl),
        cfl_fac=cfg.time.cfl_fac,
    )
    signal = np.asarray(
        diag.phi_mode_t if diag.phi_mode_t is not None else np.zeros_like(np.asarray(diag.t))
    )
    t_arr = np.asarray(diag.t, dtype=float)
    fit_window_tmin: float | None = None
    fit_window_tmax: float | None = None
    _status(f"integration complete; fitting growth rate from {t_arr.size} saved samples")
    if t_arr.size < 2:
        gamma = (
            float(np.asarray(diag.gamma_t)[-1]) if np.asarray(diag.gamma_t).size else 0.0
        )
        omega = (
            float(np.asarray(diag.omega_t)[-1]) if np.asarray(diag.omega_t).size else 0.0
        )
    elif auto_window:
        gamma, omega, fit_window_tmin, fit_window_tmax = deps.fit_growth_rate_auto(
            t_arr,
            signal,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
        )
    else:
        gamma, omega = deps.fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)
        fit_window_tmin, fit_window_tmax = _resolved_fit_bounds(t_arr, tmin, tmax)
    _status(f"fit complete: gamma={float(gamma):.6f} omega={float(omega):.6f}")
    return RuntimeLinearResult(
        ky=float(grid.ky[0]),
        gamma=float(gamma),
        omega=float(omega),
        selection=sel,
        t=t_arr,
        signal=np.asarray(signal),
        state=np.asarray(G_final) if return_state else None,
        fit_window_tmin=fit_window_tmin,
        fit_window_tmax=fit_window_tmax,
        fit_signal_used="phi",
    )
