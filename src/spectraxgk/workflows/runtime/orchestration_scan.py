"""Runtime ky-scan orchestration and combined-batch scan policy."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Protocol

import numpy as np

from spectraxgk.diagnostics.modes import ModeSelection
from spectraxgk.workflows.runtime.config import RuntimeConfig
from spectraxgk.workflows.runtime.results import RuntimeLinearScanResult

class RuntimeScanBatchDeps(Protocol):
    """Dependency surface needed by the combined-ky scan batch helper."""

    build_runtime_geometry: Callable[[RuntimeConfig], Any]
    build_runtime_linear_params: Callable[..., Any]
    build_runtime_linear_terms: Callable[[RuntimeConfig], Any]
    build_initial_condition: Callable[..., Any]
    apply_geometry_grid_defaults: Callable[..., Any]
    build_spectral_grid: Callable[[Any], Any]
    select_ky_index: Callable[[Any, float], int]
    midplane_index: Callable[[Any], int]
    integrate_linear_diagnostics: Callable[..., Any]
    extract_mode_time_series: Callable[..., Any]
    fit_growth_rate_auto_with_stats: Callable[..., Any]
    fit_growth_rate_auto: Callable[..., Any]
    fit_growth_rate: Callable[..., Any]
    apply_diagnostic_normalization: Callable[..., tuple[float, float]]


class RuntimeScanDeps(Protocol):
    """Dependency surface for runtime ky-scan orchestration."""

    resolve_runtime_hl_dims: Callable[..., tuple[int, int]]
    normalize_linear_solver_name: Callable[[str], str]
    parallel_requests_combined_ky_scan: Callable[[RuntimeConfig], bool]
    run_runtime_scan_batch: Callable[..., RuntimeLinearScanResult]
    runtime_independent_parallel_plan: Callable[..., Any]
    independent_map: Callable[..., list[Any]]
    run_runtime_scan_ky_task: Callable[[dict[str, Any]], Any]


def run_runtime_scan_orchestration(
    cfg: RuntimeConfig,
    ky_values: Any,
    *,
    Nl: int | None,
    Nm: int | None,
    solver: str,
    method: str | None,
    dt: float | None,
    steps: int | None,
    sample_stride: int | None,
    batch_ky: bool,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    krylov_cfg: Any,
    mode_method: str,
    fit_signal: str,
    show_progress: bool,
    workers: int,
    parallel_executor: str,
    deps: RuntimeScanDeps,
) -> RuntimeLinearScanResult:
    """Coordinate serial, independent-worker, or combined-ky runtime scans."""

    ky_arr = np.asarray(ky_values, dtype=float)
    Nl_use, Nm_use = deps.resolve_runtime_hl_dims(cfg, Nl=Nl, Nm=Nm)
    solver_key = deps.normalize_linear_solver_name(solver)
    batch_ky = bool(batch_ky or deps.parallel_requests_combined_ky_scan(cfg))
    if batch_ky and solver_key == "krylov":
        raise ValueError("batch_ky is only supported for time integration")
    if batch_ky and bool(getattr(cfg.quasilinear, "enabled", False)):
        raise NotImplementedError(
            "quasilinear scan artifacts currently require serial ky evaluation"
        )
    if batch_ky:
        return deps.run_runtime_scan_batch(
            cfg,
            ky_arr,
            Nl=Nl_use,
            Nm=Nm_use,
            method=method,
            dt=dt,
            steps=steps,
            sample_stride=sample_stride,
            auto_window=auto_window,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
            mode_method=mode_method,
            fit_signal=fit_signal,
            show_progress=show_progress,
        )

    gamma = np.zeros_like(ky_arr)
    omega = np.zeros_like(ky_arr)
    ql_payloads: list[dict[str, Any]] = []
    tasks = [
        {
            "cfg": cfg,
            "ky": float(ky),
            "Nl": Nl_use,
            "Nm": Nm_use,
            "solver": solver,
            "method": method,
            "dt": dt,
            "steps": steps,
            "sample_stride": sample_stride,
            "auto_window": auto_window,
            "tmin": tmin,
            "tmax": tmax,
            "window_fraction": window_fraction,
            "min_points": min_points,
            "start_fraction": start_fraction,
            "growth_weight": growth_weight,
            "require_positive": require_positive,
            "min_amp_fraction": min_amp_fraction,
            "krylov_cfg": krylov_cfg,
            "mode_method": mode_method,
            "fit_signal": fit_signal,
            "show_progress": show_progress,
        }
        for ky in ky_arr
    ]
    parallel_plan = deps.runtime_independent_parallel_plan(
        cfg, problem_size=int(ky_arr.size), workers=workers, executor=parallel_executor
    )
    results = deps.independent_map(
        deps.run_runtime_scan_ky_task,
        tasks,
        workers=parallel_plan.requested_workers,
        executor=parallel_plan.executor,
    )
    for i, res in enumerate(results):
        gamma[i] = float(res.gamma)
        omega[i] = float(res.omega)
        if res.quasilinear is not None:
            ql_payloads.append(res.quasilinear)
    parallel_payload = parallel_plan.to_dict()
    parallel_payload.update(
        {
            "identity_contract": "independent ky workers must preserve serial ky ordering and values",
            "quasilinear_state_extraction": bool(ql_payloads),
        }
    )
    return RuntimeLinearScanResult(
        ky=ky_arr,
        gamma=gamma,
        omega=omega,
        quasilinear=tuple(ql_payloads) if ql_payloads else None,
        parallel=parallel_payload,
    )


def run_runtime_scan_batch(
    cfg: RuntimeConfig,
    ky_arr: np.ndarray,
    *,
    Nl: int,
    Nm: int,
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
    mode_method: str,
    fit_signal: str,
    show_progress: bool,
    deps: RuntimeScanBatchDeps,
) -> RuntimeLinearScanResult:
    """Batch a ky scan using one time integration over the full grid."""

    geom = deps.build_runtime_geometry(cfg)
    grid_cfg = deps.apply_geometry_grid_defaults(geom, cfg.grid)
    grid = deps.build_spectral_grid(grid_cfg)
    params = deps.build_runtime_linear_params(cfg, Nm=Nm, geom=geom)
    terms = deps.build_runtime_linear_terms(cfg)

    ky_indices = np.asarray(
        [deps.select_ky_index(np.asarray(grid.ky), ky) for ky in ky_arr], dtype=int
    )
    nspecies = max(len([s for s in cfg.species if s.kinetic]), 1)

    g0 = None
    for ky_idx in ky_indices:
        g0_local = deps.build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=int(ky_idx),
            kx_index=0,
            Nl=Nl,
            Nm=Nm,
            nspecies=nspecies,
        )
        g0 = g0_local if g0 is None else g0 + g0_local
    if g0 is None:
        raise ValueError("No ky values provided for batch scan")

    tcfg = cfg.time
    if method is not None:
        tcfg = replace(tcfg, method=str(method))
    if dt is not None:
        tcfg = replace(tcfg, dt=float(dt))
    if steps is not None:
        tcfg = replace(tcfg, t_max=float(steps) * float(tcfg.dt))
    if sample_stride is not None:
        tcfg = replace(tcfg, sample_stride=int(sample_stride))

    steps_val = int(round(tcfg.t_max / tcfg.dt))
    diag = deps.integrate_linear_diagnostics(
        g0,
        grid,
        geom,
        params,
        dt=tcfg.dt,
        steps=steps_val,
        method=tcfg.method,
        terms=terms,
        sample_stride=tcfg.sample_stride,
        species_index=0,
        record_hl_energy=False,
        show_progress=show_progress,
    )
    phi_t_np = np.asarray(diag[1])
    dens_t_np = np.asarray(diag[2])
    t_arr = (
        float(tcfg.dt)
        * float(tcfg.sample_stride)
        * (np.arange(phi_t_np.shape[0], dtype=float) + 1.0)
    )

    gamma = np.zeros_like(ky_arr, dtype=float)
    omega = np.zeros_like(ky_arr, dtype=float)
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")

    for i, ky_idx in enumerate(ky_indices):
        sel = ModeSelection(
            ky_index=int(ky_idx), kx_index=0, z_index=deps.midplane_index(grid)
        )
        if fit_key == "auto":
            phi_signal = deps.extract_mode_time_series(
                phi_t_np, sel, method=mode_method
            )
            gamma_phi, omega_phi, _, _, r2_phi, r2p_phi = (
                deps.fit_growth_rate_auto_with_stats(
                    t_arr,
                    phi_signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
            )
            dens_signal = deps.extract_mode_time_series(
                dens_t_np, sel, method=mode_method
            )
            gamma_den, omega_den, _, _, r2_den, r2p_den = (
                deps.fit_growth_rate_auto_with_stats(
                    t_arr,
                    dens_signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
            )
            score_phi = r2_phi + 0.2 * r2p_phi + growth_weight * gamma_phi
            score_den = r2_den + 0.2 * r2p_den + growth_weight * gamma_den
            g_val, o_val = (
                (gamma_phi, omega_phi)
                if score_phi >= score_den
                else (gamma_den, omega_den)
            )
        else:
            signal = deps.extract_mode_time_series(
                dens_t_np if fit_key == "density" else phi_t_np,
                sel,
                method=mode_method,
            )
            if auto_window:
                g_val, o_val, _tmin, _tmax = deps.fit_growth_rate_auto(
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
                g_val, o_val = deps.fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)

        g_val, o_val = deps.apply_diagnostic_normalization(
            g_val,
            o_val,
            rho_star=float(np.asarray(params.rho_star)),
            diagnostic_norm=cfg.normalization.diagnostic_norm,
        )
        gamma[i] = float(g_val)
        omega[i] = float(o_val)

    return RuntimeLinearScanResult(ky=ky_arr, gamma=gamma, omega=omega)


__all__ = ["RuntimeScanBatchDeps", "RuntimeScanDeps", "run_runtime_scan_batch", "run_runtime_scan_orchestration"]
