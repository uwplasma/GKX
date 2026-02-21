"""High-level benchmark helpers for scans and eigenfunction extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from spectraxgk.analysis import extract_eigenfunction, extract_mode_time_series, fit_growth_rate_auto
from spectraxgk.benchmarks import LinearRunResult, LinearScanResult
from spectraxgk.grids import SpectralGrid, build_spectral_grid


@dataclass(frozen=True)
class ScanAndModeResult:
    scan: LinearScanResult
    eigenfunction: np.ndarray
    grid: SpectralGrid
    ky_selected: float
    tmin: float | None
    tmax: float | None


def normalize_eigenfunction(eigenfunction: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Normalize an eigenfunction by its value at theta=0 (nearest z=0)."""

    idx = int(np.argmin(np.abs(z)))
    scale = eigenfunction[idx]
    if scale == 0:
        return eigenfunction
    return eigenfunction / scale


def run_linear_scan(
    *,
    ky_values: np.ndarray,
    run_linear_fn,
    cfg,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    solver: str,
    krylov_cfg,
    window_kw: dict,
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    run_kwargs: dict | None = None,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], object] | None = None,
) -> LinearScanResult:
    """Run a linear scan over ky values."""

    gammas: list[float] = []
    omegas: list[float] = []
    ky_out: list[float] = []
    for i, ky in enumerate(ky_values):
        if resolution_policy is not None:
            Nl_i, Nm_i = resolution_policy(float(ky))
        else:
            Nl_i, Nm_i = int(Nl), int(Nm)
        dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        tmin_i = tmin[i] if isinstance(tmin, np.ndarray) else tmin
        tmax_i = tmax[i] if isinstance(tmax, np.ndarray) else tmax
        krylov_cfg_use = krylov_policy(float(ky)) if krylov_policy is not None else krylov_cfg
        result = run_linear_fn(
            ky_target=float(ky),
            cfg=cfg,
            Nl=int(Nl_i),
            Nm=int(Nm_i),
            dt=dt_i,
            steps=steps_i,
            method=method,
            solver=solver,
            krylov_cfg=krylov_cfg_use,
            auto_window=auto_window,
            tmin=tmin_i,
            tmax=tmax_i,
            **window_kw,
            **(run_kwargs or {}),
        )
        gammas.append(float(result.gamma))
        omegas.append(float(result.omega))
        ky_out.append(float(result.ky))

    return LinearScanResult(ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas))


def run_scan_and_mode(
    *,
    ky_values: np.ndarray,
    scan_fn,
    linear_fn,
    cfg,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    solver: str,
    mode_solver: str,
    krylov_cfg,
    window_kw: dict,
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    run_kwargs: dict | None = None,
    mode_kwargs: dict | None = None,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], object] | None = None,
    select_ky: Callable[[LinearScanResult], float] | None = None,
) -> ScanAndModeResult:
    """Run a scan and extract a representative eigenfunction."""

    if select_ky is None:
        select_ky = lambda scan: float(scan.ky[int(np.nanargmax(scan.gamma))])

    scan = run_linear_scan(
        ky_values=ky_values,
        run_linear_fn=linear_fn,
        cfg=cfg,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps,
        method=method,
        solver=solver,
        krylov_cfg=krylov_cfg,
        window_kw=window_kw,
        tmin=tmin,
        tmax=tmax,
        auto_window=auto_window,
        run_kwargs=run_kwargs,
        resolution_policy=resolution_policy,
        krylov_policy=krylov_policy,
    )
    ky_sel = float(select_ky(scan))
    if resolution_policy is not None:
        Nl_mode, Nm_mode = resolution_policy(ky_sel)
    else:
        Nl_mode, Nm_mode = int(Nl), int(Nm)
    idx = int(np.argmin(np.abs(scan.ky - ky_sel)))
    dt_mode = float(dt[idx]) if isinstance(dt, np.ndarray) else float(dt)
    steps_mode = int(steps[idx]) if isinstance(steps, np.ndarray) else int(steps)
    run: LinearRunResult = linear_fn(
        cfg=cfg,
        ky_target=ky_sel,
        Nl=int(Nl_mode),
        Nm=int(Nm_mode),
        dt=dt_mode,
        steps=steps_mode,
        method=method,
        solver=mode_solver,
        **window_kw,
        **(mode_kwargs or {}),
    )
    grid = build_spectral_grid(cfg.grid)
    if run.t.size < 2:
        tmin_fit = None
        tmax_fit = None
    else:
        signal = extract_mode_time_series(run.phi_t, run.selection, method="project")
        _g, _w, tmin_fit, tmax_fit = fit_growth_rate_auto(run.t, signal, **window_kw)
    eig = extract_eigenfunction(
        run.phi_t, run.t, run.selection, z=grid.z, method="snapshot", tmin=tmin_fit, tmax=tmax_fit
    )
    return ScanAndModeResult(
        scan=scan,
        eigenfunction=eig,
        grid=grid,
        ky_selected=ky_sel,
        tmin=tmin_fit,
        tmax=tmax_fit,
    )
