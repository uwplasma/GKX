"""Runtime ky-scan orchestration and combined-batch scan policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Protocol, Sequence, cast

import numpy as np

from spectraxgk.diagnostics.modes import ModeSelection
from spectraxgk.workflows.runtime.config import RuntimeConfig
from spectraxgk.workflows.runtime.results import (
    RuntimeLinearResult,
    RuntimeLinearScanResult,
    RuntimeParameterScanResult,
)

class RuntimeScanBatchDeps(Protocol):
    """Dependency surface needed by the combined-ky scan batch helper."""

    build_runtime_geometry: Callable[[RuntimeConfig], Any]
    build_runtime_linear_params: Callable[..., Any]
    build_runtime_linear_terms: Callable[[RuntimeConfig], Any]
    build_initial_condition: Callable[..., Any]
    apply_geometry_grid_defaults: Callable[..., Any]
    build_spectral_grid: Callable[[Any], Any]
    select_ky_grid: Callable[[Any, Any], Any]
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


@dataclass(frozen=True)
class _BatchScanSetup:
    geom: Any
    grid: Any
    params: Any
    terms: Any
    ky_indices: np.ndarray
    nspecies: int


@dataclass(frozen=True)
class _BatchDiagnostics:
    phi_t: np.ndarray
    density_t: np.ndarray
    time: np.ndarray


@dataclass(frozen=True)
class _RuntimeScanOptions:
    method: str | None
    dt: float | None
    steps: int | None
    sample_stride: int | None
    auto_window: bool
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    mode_method: str
    fit_signal: str
    show_progress: bool

    def task_fields(self) -> dict[str, Any]:
        return dict(vars(self))


def _runtime_scan_options(
    *,
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
) -> _RuntimeScanOptions:
    """Collect scan-window, time-step, and diagnostic options in one object."""

    return _RuntimeScanOptions(
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


def build_runtime_scan_orchestration_deps(facade: Any) -> RuntimeScanDeps:
    """Build ky-scan orchestration deps from the public runtime facade."""

    return cast(
        RuntimeScanDeps,
        SimpleNamespace(
            resolve_runtime_hl_dims=facade._resolve_runtime_hl_dims,
            normalize_linear_solver_name=facade._normalize_linear_solver_name,
            parallel_requests_combined_ky_scan=facade._parallel_requests_combined_ky_scan,
            run_runtime_scan_batch=facade._run_runtime_scan_batch,
            runtime_independent_parallel_plan=facade._runtime_independent_parallel_plan,
            independent_map=facade.independent_map,
            run_runtime_scan_ky_task=facade._run_runtime_scan_ky_task,
        ),
    )


def build_runtime_scan_batch_deps(facade: Any) -> RuntimeScanBatchDeps:
    """Build combined-ky scan deps from the public runtime facade."""

    return cast(
        RuntimeScanBatchDeps,
        SimpleNamespace(
            build_runtime_geometry=facade.build_runtime_geometry,
            build_runtime_linear_params=facade.build_runtime_linear_params,
            build_runtime_linear_terms=facade.build_runtime_linear_terms,
            build_initial_condition=facade._build_initial_condition,
            apply_geometry_grid_defaults=facade.apply_geometry_grid_defaults,
            build_spectral_grid=facade.build_spectral_grid,
            select_ky_grid=facade.select_ky_grid,
            select_ky_index=facade.select_ky_index,
            midplane_index=facade._midplane_index,
            integrate_linear_diagnostics=facade.integrate_linear_diagnostics,
            extract_mode_time_series=facade.extract_mode_time_series,
            fit_growth_rate_auto_with_stats=facade.fit_growth_rate_auto_with_stats,
            fit_growth_rate_auto=facade.fit_growth_rate_auto,
            fit_growth_rate=facade.fit_growth_rate,
            apply_diagnostic_normalization=facade.apply_diagnostic_normalization,
        ),
    )


def run_runtime_scan_ky_task(
    task: dict[str, Any],
    *,
    run_runtime_linear: Callable[..., Any],
) -> Any:
    """Run one independent ky point for ordered scan-worker execution."""

    return run_runtime_linear(
        task["cfg"],
        ky_target=float(task["ky"]),
        Nl=int(task["Nl"]),
        Nm=int(task["Nm"]),
        solver=str(task["solver"]),
        method=task["method"],
        dt=task["dt"],
        steps=task["steps"],
        sample_stride=task["sample_stride"],
        auto_window=bool(task["auto_window"]),
        tmin=task["tmin"],
        tmax=task["tmax"],
        window_fraction=float(task["window_fraction"]),
        min_points=int(task["min_points"]),
        start_fraction=float(task["start_fraction"]),
        growth_weight=float(task["growth_weight"]),
        require_positive=bool(task["require_positive"]),
        min_amp_fraction=float(task["min_amp_fraction"]),
        krylov_cfg=task["krylov_cfg"],
        mode_method=str(task["mode_method"]),
        fit_signal=str(task["fit_signal"]),
        show_progress=bool(task["show_progress"]),
    )


def _combined_ky_scan_requested(
    *,
    cfg: RuntimeConfig,
    batch_ky: bool,
    solver_key: str,
    deps: RuntimeScanDeps,
) -> bool:
    requested = bool(batch_ky or deps.parallel_requests_combined_ky_scan(cfg))
    if requested and solver_key == "krylov":
        raise ValueError("batch_ky is only supported for time integration")
    if requested and bool(getattr(cfg.quasilinear, "enabled", False)):
        raise NotImplementedError(
            "quasilinear scan artifacts currently require serial ky evaluation"
        )
    return requested


def _scan_worker_tasks(
    cfg: RuntimeConfig,
    ky_arr: np.ndarray,
    *,
    Nl: int,
    Nm: int,
    solver: str,
    krylov_cfg: Any,
    options: _RuntimeScanOptions,
) -> list[dict[str, Any]]:
    return [
        {
            "cfg": cfg,
            "ky": float(ky),
            "Nl": Nl,
            "Nm": Nm,
            "solver": solver,
            "krylov_cfg": krylov_cfg,
            **options.task_fields(),
        }
        for ky in ky_arr
    ]


def _scan_result_from_worker_results(
    ky_arr: np.ndarray,
    results: list[Any],
    parallel_plan: Any,
) -> RuntimeLinearScanResult:
    gamma = np.zeros_like(ky_arr)
    omega = np.zeros_like(ky_arr)
    ql_payloads: list[dict[str, Any]] = []
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


def _run_combined_ky_scan(
    cfg: RuntimeConfig,
    ky_arr: np.ndarray,
    *,
    Nl: int,
    Nm: int,
    options: _RuntimeScanOptions,
    deps: RuntimeScanDeps,
) -> RuntimeLinearScanResult:
    return deps.run_runtime_scan_batch(
        cfg,
        ky_arr,
        Nl=Nl,
        Nm=Nm,
        **options.task_fields(),
    )


def _run_independent_ky_scan(
    cfg: RuntimeConfig,
    ky_arr: np.ndarray,
    *,
    Nl: int,
    Nm: int,
    solver: str,
    krylov_cfg: Any,
    options: _RuntimeScanOptions,
    workers: int,
    parallel_executor: str,
    deps: RuntimeScanDeps,
) -> RuntimeLinearScanResult:
    tasks = _scan_worker_tasks(
        cfg,
        ky_arr,
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        krylov_cfg=krylov_cfg,
        options=options,
    )
    parallel_plan = deps.runtime_independent_parallel_plan(
        cfg, problem_size=int(ky_arr.size), workers=workers, executor=parallel_executor
    )
    results = deps.independent_map(
        deps.run_runtime_scan_ky_task,
        tasks,
        workers=parallel_plan.requested_workers,
        executor=parallel_plan.executor,
    )
    return _scan_result_from_worker_results(ky_arr, results, parallel_plan)


def _batch_scan_setup(
    cfg: RuntimeConfig,
    ky_arr: np.ndarray,
    *,
    Nl: int,
    Nm: int,
    deps: RuntimeScanBatchDeps,
) -> _BatchScanSetup:
    geom = deps.build_runtime_geometry(cfg)
    grid_cfg = deps.apply_geometry_grid_defaults(geom, cfg.grid)
    full_grid = deps.build_spectral_grid(grid_cfg)
    full_ky_indices = np.asarray(
        [deps.select_ky_index(np.asarray(full_grid.ky), ky) for ky in ky_arr],
        dtype=int,
    )
    # Linear scans retain every requested mode. The parent mask is only for
    # nonlinear convolution dealiasing and would otherwise suppress high ky.
    grid = deps.select_ky_grid(full_grid, full_ky_indices)
    params = deps.build_runtime_linear_params(cfg, Nm=Nm, geom=geom)
    terms = deps.build_runtime_linear_terms(cfg)
    ky_indices = np.arange(ky_arr.size, dtype=int)
    nspecies = max(len([s for s in cfg.species if s.kinetic]), 1)
    return _BatchScanSetup(
        geom=geom,
        grid=grid,
        params=params,
        terms=terms,
        ky_indices=ky_indices,
        nspecies=nspecies,
    )


def _combined_batch_initial_condition(
    cfg: RuntimeConfig,
    setup: _BatchScanSetup,
    *,
    Nl: int,
    Nm: int,
    deps: RuntimeScanBatchDeps,
) -> Any:
    g0 = None
    for ky_idx in setup.ky_indices:
        g0_local = deps.build_initial_condition(
            setup.grid,
            setup.geom,
            cfg,
            ky_index=int(ky_idx),
            kx_index=0,
            Nl=Nl,
            Nm=Nm,
            nspecies=setup.nspecies,
        )
        g0 = g0_local if g0 is None else g0 + g0_local
    if g0 is None:
        raise ValueError("No ky values provided for batch scan")
    return g0


def _batch_time_config(
    cfg: RuntimeConfig,
    *,
    options: _RuntimeScanOptions,
) -> Any:
    tcfg = cfg.time
    if options.method is not None:
        tcfg = replace(tcfg, method=str(options.method))
    if options.dt is not None:
        tcfg = replace(tcfg, dt=float(options.dt))
    if options.steps is not None:
        tcfg = replace(tcfg, t_max=float(options.steps) * float(tcfg.dt))
    if options.sample_stride is not None:
        tcfg = replace(tcfg, sample_stride=int(options.sample_stride))
    return tcfg


def _run_batch_diagnostics(
    setup: _BatchScanSetup,
    g0: Any,
    tcfg: Any,
    *,
    show_progress: bool,
    deps: RuntimeScanBatchDeps,
) -> _BatchDiagnostics:
    steps_val = int(round(tcfg.t_max / tcfg.dt))
    diag = deps.integrate_linear_diagnostics(
        g0,
        setup.grid,
        setup.geom,
        setup.params,
        dt=tcfg.dt,
        steps=steps_val,
        method=tcfg.method,
        terms=setup.terms,
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
    return _BatchDiagnostics(phi_t=phi_t_np, density_t=dens_t_np, time=t_arr)


def _fit_signal_key(fit_signal: str) -> str:
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    return fit_key


def _auto_fit_scan_point(
    diagnostics: _BatchDiagnostics,
    sel: ModeSelection,
    *,
    options: _RuntimeScanOptions,
    deps: RuntimeScanBatchDeps,
) -> tuple[float, float]:
    phi_signal = deps.extract_mode_time_series(
        diagnostics.phi_t, sel, method=options.mode_method
    )
    gamma_phi, omega_phi, _, _, r2_phi, r2p_phi = (
        deps.fit_growth_rate_auto_with_stats(
            diagnostics.time,
            phi_signal,
            window_fraction=options.window_fraction,
            min_points=options.min_points,
            start_fraction=options.start_fraction,
            growth_weight=options.growth_weight,
            require_positive=options.require_positive,
            min_amp_fraction=options.min_amp_fraction,
        )
    )
    dens_signal = deps.extract_mode_time_series(
        diagnostics.density_t, sel, method=options.mode_method
    )
    gamma_den, omega_den, _, _, r2_den, r2p_den = (
        deps.fit_growth_rate_auto_with_stats(
            diagnostics.time,
            dens_signal,
            window_fraction=options.window_fraction,
            min_points=options.min_points,
            start_fraction=options.start_fraction,
            growth_weight=options.growth_weight,
            require_positive=options.require_positive,
            min_amp_fraction=options.min_amp_fraction,
        )
    )
    score_phi = r2_phi + 0.2 * r2p_phi + options.growth_weight * gamma_phi
    score_den = r2_den + 0.2 * r2p_den + options.growth_weight * gamma_den
    return (gamma_phi, omega_phi) if score_phi >= score_den else (gamma_den, omega_den)


def _fit_batch_scan_point(
    diagnostics: _BatchDiagnostics,
    sel: ModeSelection,
    *,
    fit_key: str,
    options: _RuntimeScanOptions,
    deps: RuntimeScanBatchDeps,
) -> tuple[float, float]:
    if fit_key == "auto":
        return _auto_fit_scan_point(
            diagnostics,
            sel,
            options=options,
            deps=deps,
        )
    signal = deps.extract_mode_time_series(
        diagnostics.density_t if fit_key == "density" else diagnostics.phi_t,
        sel,
        method=options.mode_method,
    )
    if options.auto_window:
        g_val, o_val, _tmin, _tmax = deps.fit_growth_rate_auto(
            diagnostics.time,
            signal,
            window_fraction=options.window_fraction,
            min_points=options.min_points,
            start_fraction=options.start_fraction,
            growth_weight=options.growth_weight,
            require_positive=options.require_positive,
            min_amp_fraction=options.min_amp_fraction,
        )
        return g_val, o_val
    return deps.fit_growth_rate(
        diagnostics.time, signal, tmin=options.tmin, tmax=options.tmax
    )


def _fit_batch_scan_result(
    cfg: RuntimeConfig,
    ky_arr: np.ndarray,
    setup: _BatchScanSetup,
    diagnostics: _BatchDiagnostics,
    *,
    options: _RuntimeScanOptions,
    deps: RuntimeScanBatchDeps,
) -> RuntimeLinearScanResult:
    """Fit each requested ky from one combined-ky diagnostic time history."""

    gamma = np.zeros_like(ky_arr, dtype=float)
    omega = np.zeros_like(ky_arr, dtype=float)
    fit_key = _fit_signal_key(options.fit_signal)
    for i, ky_idx in enumerate(setup.ky_indices):
        sel = ModeSelection(
            ky_index=int(ky_idx), kx_index=0, z_index=deps.midplane_index(setup.grid)
        )
        g_val, o_val = _fit_batch_scan_point(
            diagnostics,
            sel,
            fit_key=fit_key,
            options=options,
            deps=deps,
        )
        gamma[i], omega[i] = deps.apply_diagnostic_normalization(
            g_val,
            o_val,
            rho_star=float(np.asarray(setup.params.rho_star)),
            diagnostic_norm=cfg.normalization.diagnostic_norm,
        )
    return RuntimeLinearScanResult(ky=ky_arr, gamma=gamma, omega=omega)


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
    options = _runtime_scan_options(
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
    if _combined_ky_scan_requested(
        cfg=cfg, batch_ky=batch_ky, solver_key=solver_key, deps=deps
    ):
        return _run_combined_ky_scan(
            cfg,
            ky_arr,
            Nl=Nl_use,
            Nm=Nm_use,
            options=options,
            deps=deps,
        )

    return _run_independent_ky_scan(
        cfg,
        ky_arr,
        Nl=Nl_use,
        Nm=Nm_use,
        solver=solver,
        krylov_cfg=krylov_cfg,
        options=options,
        workers=workers,
        parallel_executor=parallel_executor,
        deps=deps,
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

    if ky_arr.size == 0:
        raise ValueError("ky_values must not be empty")

    setup = _batch_scan_setup(cfg, ky_arr, Nl=Nl, Nm=Nm, deps=deps)
    g0 = _combined_batch_initial_condition(
        cfg, setup, Nl=Nl, Nm=Nm, deps=deps
    )
    options = _runtime_scan_options(
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
    tcfg = _batch_time_config(cfg, options=options)
    diagnostics = _run_batch_diagnostics(
        setup,
        g0,
        tcfg,
        show_progress=options.show_progress,
        deps=deps,
    )
    return _fit_batch_scan_result(
        cfg, ky_arr, setup, diagnostics, options=options, deps=deps
    )


def run_runtime_parameter_scan(
    cfg: RuntimeConfig,
    parameter_values: Sequence[float],
    *,
    parameter_name: str,
    update_config: Callable[[RuntimeConfig, float, int], RuntimeConfig],
    ky_target: float = 0.3,
    linear_options: Mapping[str, Any] | None = None,
    point_options: Callable[
        [float, int, RuntimeLinearResult | None], Mapping[str, Any]
    ] | None = None,
    candidate_options: Callable[
        [float, int, RuntimeLinearResult | None], Sequence[Mapping[str, Any]]
    ] | None = None,
    select_candidate: Callable[
        [float, int, tuple[RuntimeLinearResult, ...], RuntimeLinearResult | None], int
    ] | None = None,
    continuation: bool = False,
) -> RuntimeParameterScanResult:
    """Run a scalar scan and optionally continue a selected solution branch."""

    from spectraxgk.runtime import run_runtime_linear

    name = str(parameter_name).strip()
    if not name:
        raise ValueError("parameter_name must be nonempty")
    values = np.asarray(parameter_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("parameter_values must be a nonempty one-dimensional array")

    shared = dict(linear_options or {})
    runs: list[RuntimeLinearResult] = []
    previous: RuntimeLinearResult | None = None
    for index, value in enumerate(values):
        point_cfg = update_config(cfg, float(value), index)
        if not isinstance(point_cfg, RuntimeConfig):
            raise TypeError("update_config must return RuntimeConfig")
        options = dict(shared)
        if point_options is not None:
            options.update(point_options(float(value), index, previous))
        if continuation or candidate_options is not None:
            options["return_state"] = True
        if continuation and previous is not None:
            if previous.state is None:
                raise ValueError("continuation requires each point to return state")
            options["initial_state"] = previous.state

        overrides = (
            tuple(candidate_options(float(value), index, previous))
            if candidate_options is not None else ({},)
        )
        if not overrides:
            raise ValueError("candidate_options must return at least one candidate")
        candidates = tuple(
            run_runtime_linear(point_cfg, ky_target=ky_target, **(options | dict(item)))
            for item in overrides
        )
        selected = (
            int(select_candidate(float(value), index, candidates, previous))
            if select_candidate is not None else 0
        )
        if selected < 0 or selected >= len(candidates):
            raise IndexError("select_candidate returned an out-of-range index")
        previous = candidates[selected]
        runs.append(previous)

    return RuntimeParameterScanResult(
        parameter_name=name,
        values=values,
        gamma=np.asarray([result.gamma for result in runs], dtype=float),
        omega=np.asarray([result.omega for result in runs], dtype=float),
        runs=tuple(runs),
    )


__all__ = [
    "RuntimeScanBatchDeps",
    "RuntimeScanDeps",
    "build_runtime_scan_batch_deps",
    "build_runtime_scan_orchestration_deps",
    "run_runtime_scan_ky_task",
    "run_runtime_scan_batch",
    "run_runtime_scan_orchestration",
    "run_runtime_parameter_scan",
]
