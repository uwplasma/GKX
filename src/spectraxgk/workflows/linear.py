"""Executable linear runtime workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from spectraxgk.benchmarking.shared import LinearRunResult, LinearScanResult
from spectraxgk.core.grid import SpectralGrid, build_spectral_grid
from spectraxgk.diagnostics.analysis import fit_growth_rate_auto
from spectraxgk.diagnostics.modes import (
    ModeSelection,
    extract_eigenfunction,
    extract_mode_time_series,
)
from spectraxgk.workflows.runtime.config import RuntimeConfig
from spectraxgk.workflows.runtime.diagnostics import RuntimeQuasilinearFinalizationDeps
from spectraxgk.workflows.runtime.results import RuntimeLinearResult
from spectraxgk.solvers.time.explicit import integrate_linear_explicit_from_config

@dataclass(frozen=True)
class FullLinearRuntimeDeps:
    """Injected dependencies for the full-GK linear runtime workflow."""

    build_runtime_geometry: Callable[[RuntimeConfig], Any]
    apply_geometry_grid_defaults: Callable[..., Any]
    build_spectral_grid: Callable[..., Any]
    build_runtime_linear_params: Callable[..., Any]
    build_runtime_linear_terms: Callable[..., Any]
    select_ky_index: Callable[..., int]
    select_ky_grid: Callable[..., Any]
    midplane_index: Callable[..., int]
    build_initial_condition: Callable[..., Any]
    normalize_linear_solver_name: Callable[..., str]
    runtime_default_krylov_config: Callable[..., Any]
    build_linear_cache: Callable[..., Any]
    dominant_eigenpair: Callable[..., tuple[Any, Any]]
    apply_diagnostic_normalization: Callable[..., tuple[float, float]]
    integrate_linear_from_config: Callable[..., tuple[Any, Any]]
    integrate_linear_diagnostics: Callable[..., tuple[Any, ...]]
    fit_runtime_linear_diagnostics: Callable[..., Any]
    finalize_runtime_linear_quasilinear: Callable[..., RuntimeLinearResult]
    quasilinear_finalization_deps: RuntimeQuasilinearFinalizationDeps
    extract_mode_time_series: Callable[..., Any]
    fit_growth_rate_auto_with_stats: Callable[..., Any]
    fit_growth_rate_auto: Callable[..., Any]
    fit_growth_rate: Callable[..., Any]
    extract_eigenfunction: Callable[..., Any]


_StatusCallback = Callable[[str], None] | None


@dataclass(frozen=True)
class _LinearRuntimeContext:
    cfg: RuntimeConfig
    geom: Any
    grid: Any
    params: Any
    terms: Any
    selection: ModeSelection
    initial_state: Any
    solver_key: str
    fit_key: str
    ql_enabled: bool
    return_state_requested: bool
    return_state_effective: bool
    n_laguerre: int
    n_hermite: int


@dataclass(frozen=True)
class _LinearFitPolicy:
    mode_method: str
    auto_window: bool
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float


@dataclass(frozen=True)
class _LinearTimeSeriesRaw:
    """Raw linear time-integration arrays before NumPy normalization."""

    g_last: Any | None
    phi_t: Any
    density_t: Any | None
    t: Any | None = None


def _status(callback: _StatusCallback, message: str) -> None:
    if callback is not None:
        callback(message)


def _fit_signal_key(fit_signal: str) -> str:
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    return fit_key


def _prepare_linear_runtime_context(
    cfg: RuntimeConfig,
    *,
    deps: FullLinearRuntimeDeps,
    ky_target: float,
    n_laguerre: int,
    n_hermite: int,
    solver: str,
    fit_signal: str,
    return_state: bool,
    initial_state: Any | None,
    status_callback: _StatusCallback,
) -> _LinearRuntimeContext:
    ql_enabled = bool(getattr(cfg.quasilinear, "enabled", False))
    return_state_requested = bool(return_state)
    return_state_effective = return_state_requested or ql_enabled

    geom = deps.build_runtime_geometry(cfg)
    _status(status_callback, "building spectral grid")
    grid_cfg = deps.apply_geometry_grid_defaults(geom, cfg.grid)
    grid_full = deps.build_spectral_grid(grid_cfg)
    _status(status_callback, "building runtime linear parameters")
    params = deps.build_runtime_linear_params(cfg, Nm=n_hermite, geom=geom)
    terms = deps.build_runtime_linear_terms(cfg)

    ky_index = deps.select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = deps.select_ky_grid(grid_full, ky_index)
    selection = ModeSelection(
        ky_index=0,
        kx_index=0,
        z_index=deps.midplane_index(grid),
    )
    _status(
        status_callback,
        f"selected ky index {ky_index} at ky={float(grid.ky[selection.ky_index]):.4f}",
    )
    kinetic_species = max(len([s for s in cfg.species if s.kinetic]), 1)
    expected_shape = (
        kinetic_species, n_laguerre, n_hermite,
        int(grid.ky.size), int(grid.kx.size), int(grid.z.size),
    )
    if initial_state is None:
        _status(status_callback, "building initial condition")
        state = deps.build_initial_condition(
            grid, geom, cfg,
            ky_index=selection.ky_index,
            kx_index=selection.kx_index,
            Nl=n_laguerre,
            Nm=n_hermite,
            nspecies=kinetic_species,
        )
    else:
        state = jnp.asarray(initial_state)
        if tuple(state.shape) != expected_shape:
            raise ValueError(
                f"initial_state shape {tuple(state.shape)} does not match "
                f"runtime shape {expected_shape}"
            )

    return _LinearRuntimeContext(
        cfg=cfg,
        geom=geom,
        grid=grid,
        params=params,
        terms=terms,
        selection=selection,
        initial_state=state,
        solver_key=deps.normalize_linear_solver_name(solver),
        fit_key=_fit_signal_key(fit_signal),
        ql_enabled=ql_enabled,
        return_state_requested=return_state_requested,
        return_state_effective=return_state_effective,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
    )


def _valid_growth(gamma: float, omega: float, *, require_positive: bool) -> bool:
    if not np.isfinite(gamma) or not np.isfinite(omega):
        return False
    if require_positive and gamma <= 0.0:
        return False
    return True


def _build_linear_fit_policy(
    *,
    mode_method: str,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> _LinearFitPolicy:
    return _LinearFitPolicy(
        mode_method=mode_method,
        auto_window=auto_window,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
    )


def _finalize_linear_result(
    result: RuntimeLinearResult,
    *,
    ctx: _LinearRuntimeContext,
    deps: FullLinearRuntimeDeps,
    state_for_quasilinear: np.ndarray | None = None,
    status_callback: _StatusCallback,
) -> RuntimeLinearResult:
    return deps.finalize_runtime_linear_quasilinear(
        result,
        enabled=ctx.ql_enabled,
        cfg=ctx.cfg,
        grid=ctx.grid,
        geom=ctx.geom,
        params=ctx.params,
        terms=ctx.terms,
        Nl=ctx.n_laguerre,
        Nm=ctx.n_hermite,
        solver_name=ctx.solver_key,
        species_names=tuple(s.name for s in ctx.cfg.species if s.kinetic),
        return_state_requested=ctx.return_state_requested,
        state_for_quasilinear=state_for_quasilinear,
        deps=deps.quasilinear_finalization_deps,
        status_callback=lambda message: _status(status_callback, message),
    )


def _linear_result_from_eigenvector(
    ctx: _LinearRuntimeContext,
    *,
    gamma: float,
    omega: float,
    eigenvector: np.ndarray,
) -> RuntimeLinearResult:
    return RuntimeLinearResult(
        ky=float(ctx.grid.ky[ctx.selection.ky_index]),
        gamma=gamma,
        omega=omega,
        selection=ctx.selection,
        state=eigenvector if ctx.return_state_effective else None,
    )


def _run_krylov_linear(
    ctx: _LinearRuntimeContext,
    *,
    deps: FullLinearRuntimeDeps,
    krylov_cfg: Any | None,
    status_callback: _StatusCallback,
) -> tuple[float, float, np.ndarray]:
    _status(status_callback, "starting Krylov solve")
    kcfg = krylov_cfg or deps.runtime_default_krylov_config(ctx.cfg)
    _status(status_callback, "building linear cache")
    cache = deps.build_linear_cache(
        ctx.grid,
        ctx.geom,
        ctx.params,
        ctx.n_laguerre,
        ctx.n_hermite,
    )
    eig, vec = deps.dominant_eigenpair(
        ctx.initial_state,
        cache,
        ctx.params,
        terms=ctx.terms,
        krylov_dim=kcfg.krylov_dim,
        restarts=kcfg.restarts,
        omega_min_factor=kcfg.omega_min_factor,
        omega_target_factor=kcfg.omega_target_factor,
        omega_cap_factor=kcfg.omega_cap_factor,
        omega_sign=kcfg.omega_sign,
        method=kcfg.method,
        power_iters=kcfg.power_iters,
        power_dt=kcfg.power_dt,
        shift=kcfg.shift,
        shift_source=kcfg.shift_source,
        shift_tol=kcfg.shift_tol,
        shift_maxiter=kcfg.shift_maxiter,
        shift_restart=kcfg.shift_restart,
        shift_solve_method=kcfg.shift_solve_method,
        shift_preconditioner=kcfg.shift_preconditioner,
        shift_selection=kcfg.shift_selection,
        shift_outer_residual_tol=kcfg.shift_outer_residual_tol,
        mode_family=kcfg.mode_family,
        fallback_method=kcfg.fallback_method,
        fallback_real_floor=kcfg.fallback_real_floor,
        status_callback=lambda message: _status(status_callback, message),
    )
    gamma = float(jnp.real(eig))
    omega = float(-jnp.imag(eig))
    gamma, omega = deps.apply_diagnostic_normalization(
        gamma,
        omega,
        rho_star=float(np.asarray(ctx.params.rho_star)),
        diagnostic_norm=ctx.cfg.normalization.diagnostic_norm,
    )
    _status(status_callback, f"Krylov solve complete: gamma={gamma:.6f} omega={omega:.6f}")
    return gamma, omega, np.asarray(vec)


def _resolve_linear_time_config(
    ctx: _LinearRuntimeContext,
    *,
    method: str | None,
    dt: float | None,
    steps: int | None,
    sample_stride: int | None,
) -> Any:
    tcfg = ctx.cfg.time
    if method is not None:
        tcfg = replace(tcfg, method=str(method))
    if dt is not None:
        tcfg = replace(tcfg, dt=float(dt))
    if steps is not None:
        tcfg = replace(tcfg, t_max=float(steps) * float(tcfg.dt))
    if sample_stride is not None:
        tcfg = replace(tcfg, sample_stride=int(sample_stride))
    if ctx.return_state_effective and ctx.solver_key == "explicit_time":
        raise ValueError(
            "return_state/quasilinear diagnostics are not supported with solver='explicit_time'"
        )
    if ctx.return_state_effective:
        tcfg = replace(tcfg, save_state=True)
    return tcfg


def _validate_parallel_linear_time_path(ctx: _LinearRuntimeContext, tcfg: Any) -> None:
    need_density = ctx.fit_key in {"density", "auto"}
    parallel_strategy = (
        str(getattr(ctx.cfg.parallel, "strategy", "serial")).lower().replace("-", "_")
    )
    if parallel_strategy == "serial":
        return
    if tcfg.use_diffrax:
        raise NotImplementedError(
            "parallel linear RHS is currently supported only by the fixed-step cached integrator"
        )
    if need_density:
        raise NotImplementedError(
            "parallel linear RHS runtime path currently requires fit_signal='phi'"
        )


def _integrate_linear_diffrax_path(
    ctx: _LinearRuntimeContext,
    *,
    deps: FullLinearRuntimeDeps,
    tcfg: Any,
    mode_method: str,
    need_density: bool,
    show_progress: bool,
) -> _LinearTimeSeriesRaw:
    save_field = "phi+density" if need_density else "phi"
    g_last, saved = deps.integrate_linear_from_config(
        ctx.initial_state,
        ctx.grid,
        ctx.geom,
        ctx.params,
        tcfg,
        terms=ctx.terms,
        save_mode=None,
        mode_method=mode_method,
        save_field=save_field,
        density_species_index=0 if need_density else None,
        show_progress=show_progress,
        parallel=ctx.cfg.parallel,
    )
    phi_t, density_t = saved if need_density else (saved, None)
    return _LinearTimeSeriesRaw(g_last=g_last, phi_t=phi_t, density_t=density_t)


def _integrate_linear_density_path(
    ctx: _LinearRuntimeContext,
    *,
    deps: FullLinearRuntimeDeps,
    tcfg: Any,
    n_steps: int,
    show_progress: bool,
) -> _LinearTimeSeriesRaw:
    diag = deps.integrate_linear_diagnostics(
        ctx.initial_state,
        ctx.grid,
        ctx.geom,
        ctx.params,
        dt=tcfg.dt,
        steps=n_steps,
        method=tcfg.method,
        terms=ctx.terms,
        sample_stride=tcfg.sample_stride,
        species_index=0,
        record_hl_energy=False,
        show_progress=show_progress,
    )
    return _LinearTimeSeriesRaw(
        g_last=diag[0],
        phi_t=diag[1],
        density_t=diag[2] if len(diag) > 2 else None,
    )


def _integrate_linear_cached_phi_path(
    ctx: _LinearRuntimeContext,
    *,
    deps: FullLinearRuntimeDeps,
    tcfg: Any,
    mode_method: str,
    show_progress: bool,
) -> _LinearTimeSeriesRaw:
    g_last, phi_t = deps.integrate_linear_from_config(
        ctx.initial_state,
        ctx.grid,
        ctx.geom,
        ctx.params,
        tcfg,
        terms=ctx.terms,
        save_mode=ctx.selection,
        mode_method=mode_method,
        save_field="phi",
        show_progress=show_progress,
        parallel=ctx.cfg.parallel,
    )
    return _LinearTimeSeriesRaw(g_last=g_last, phi_t=phi_t, density_t=None)


def _linear_saved_sample_times(phi_t: np.ndarray, tcfg: Any) -> np.ndarray:
    return (
        float(tcfg.dt)
        * float(tcfg.sample_stride)
        * (np.arange(phi_t.shape[0], dtype=float) + 1.0)
    )


def _integrate_linear_time_series(
    ctx: _LinearRuntimeContext,
    *,
    deps: FullLinearRuntimeDeps,
    tcfg: Any,
    mode_method: str,
    show_progress: bool,
    status_callback: _StatusCallback,
) -> tuple[Any | None, np.ndarray, np.ndarray | None, np.ndarray]:
    need_density = ctx.fit_key in {"density", "auto"}
    n_steps = int(round(tcfg.t_max / tcfg.dt))
    if ctx.solver_key == "explicit_time":
        _status(status_callback, "running CFL-controlled explicit integrator")
        t_explicit, phi_t = integrate_linear_explicit_from_config(
            ctx.initial_state, ctx.grid, ctx.geom, ctx.params, tcfg,
            Nl=ctx.n_laguerre, Nm=ctx.n_hermite, terms=ctx.terms,
            z_index=ctx.selection.z_index, show_progress=show_progress,
        )
        raw = _LinearTimeSeriesRaw(None, phi_t, None, t_explicit)
    elif tcfg.use_diffrax:
        _status(
            status_callback,
            f"running diffrax integrator over {n_steps} steps with sample_stride={int(tcfg.sample_stride)}",
        )
        raw = _integrate_linear_diffrax_path(
            ctx,
            deps=deps,
            tcfg=tcfg,
            mode_method=mode_method,
            need_density=need_density,
            show_progress=show_progress,
        )
    elif need_density:
        _status(
            status_callback,
            f"running diagnostics integrator over {n_steps} steps with sample_stride={int(tcfg.sample_stride)}",
        )
        raw = _integrate_linear_density_path(
            ctx,
            deps=deps,
            tcfg=tcfg,
            n_steps=n_steps,
            show_progress=show_progress,
        )
    else:
        _status(
            status_callback,
            f"running cached linear integrator over {n_steps} steps with sample_stride={int(tcfg.sample_stride)}",
        )
        raw = _integrate_linear_cached_phi_path(
            ctx,
            deps=deps,
            tcfg=tcfg,
            mode_method=mode_method,
            show_progress=show_progress,
        )

    phi_t_np = np.asarray(raw.phi_t)
    density_np = None if raw.density_t is None else np.asarray(raw.density_t)
    times = _linear_saved_sample_times(phi_t_np, tcfg) if raw.t is None else raw.t
    return raw.g_last, phi_t_np, density_np, np.asarray(times, dtype=float)


def _fit_linear_time_series(
    ctx: _LinearRuntimeContext,
    *,
    deps: FullLinearRuntimeDeps,
    fit_policy: _LinearFitPolicy,
    t_arr: np.ndarray,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    g_last: Any | None,
    status_callback: _StatusCallback,
) -> RuntimeLinearResult:
    _status(
        status_callback,
        f"integration complete; fitting growth rate from {t_arr.size} saved samples",
    )
    fit_result = deps.fit_runtime_linear_diagnostics(
        t=t_arr,
        phi_t=phi_t,
        density_t=density_t,
        selection=ctx.selection,
        z=np.asarray(ctx.grid.z, dtype=float),
        fit_signal=ctx.fit_key,
        mode_method=fit_policy.mode_method,
        auto_window=fit_policy.auto_window,
        tmin=fit_policy.tmin,
        tmax=fit_policy.tmax,
        window_fraction=fit_policy.window_fraction,
        min_points=fit_policy.min_points,
        start_fraction=fit_policy.start_fraction,
        growth_weight=fit_policy.growth_weight,
        require_positive=fit_policy.require_positive,
        min_amp_fraction=fit_policy.min_amp_fraction,
        extract_mode_time_series_fn=deps.extract_mode_time_series,
        fit_growth_rate_auto_with_stats_fn=deps.fit_growth_rate_auto_with_stats,
        fit_growth_rate_auto_fn=deps.fit_growth_rate_auto,
        fit_growth_rate_fn=deps.fit_growth_rate,
        extract_eigenfunction_fn=deps.extract_eigenfunction,
    )
    if ctx.fit_key == "auto":
        _status(status_callback, f"automatic fit selected signal '{fit_result.fit_signal_used}'")
    gamma, omega = deps.apply_diagnostic_normalization(
        fit_result.gamma,
        fit_result.omega,
        rho_star=float(np.asarray(ctx.params.rho_star)),
        diagnostic_norm=ctx.cfg.normalization.diagnostic_norm,
    )
    _status(status_callback, f"fit complete: gamma={gamma:.6f} omega={omega:.6f}")
    return RuntimeLinearResult(
        ky=float(ctx.grid.ky[ctx.selection.ky_index]),
        gamma=float(gamma),
        omega=float(omega),
        selection=ctx.selection,
        t=t_arr,
        signal=fit_result.signal,
        field_history=phi_t,
        state=None if g_last is None or not ctx.return_state_effective else np.asarray(g_last),
        z=fit_result.z,
        eigenfunction=fit_result.eigenfunction,
        fit_window_tmin=fit_result.fit_window_tmin,
        fit_window_tmax=fit_result.fit_window_tmax,
        fit_signal_used=fit_result.fit_signal_used,
    )


def _run_time_linear(
    ctx: _LinearRuntimeContext,
    *,
    deps: FullLinearRuntimeDeps,
    method: str | None,
    dt: float | None,
    steps: int | None,
    sample_stride: int | None,
    fit_policy: _LinearFitPolicy,
    show_progress: bool,
    status_callback: _StatusCallback,
) -> RuntimeLinearResult:
    _status(status_callback, f"starting time integration path with fit_signal={ctx.fit_key}")
    tcfg = _resolve_linear_time_config(
        ctx,
        method=method,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
    )
    _validate_parallel_linear_time_path(ctx, tcfg)
    g_last, phi_t, density_t, t_arr = _integrate_linear_time_series(
        ctx,
        deps=deps,
        tcfg=tcfg,
        mode_method=fit_policy.mode_method,
        show_progress=show_progress,
        status_callback=status_callback,
    )
    return _fit_linear_time_series(
        ctx,
        deps=deps,
        fit_policy=fit_policy,
        t_arr=t_arr,
        phi_t=phi_t,
        density_t=density_t,
        g_last=g_last,
        status_callback=status_callback,
    )


def _run_krylov_linear_runtime(
    ctx: _LinearRuntimeContext,
    *,
    deps: FullLinearRuntimeDeps,
    krylov_cfg: Any | None,
    status_callback: _StatusCallback,
) -> RuntimeLinearResult:
    """Run the Krylov branch and finalize any requested quasilinear diagnostics."""
    gamma, omega, vec = _run_krylov_linear(
        ctx,
        deps=deps,
        krylov_cfg=krylov_cfg,
        status_callback=status_callback,
    )
    result = _linear_result_from_eigenvector(
        ctx,
        gamma=gamma,
        omega=omega,
        eigenvector=vec,
    )
    return _finalize_linear_result(
        result,
        ctx=ctx,
        deps=deps,
        state_for_quasilinear=vec,
        status_callback=status_callback,
    )


def _finalize_auto_linear_runtime(
    result: RuntimeLinearResult,
    ctx: _LinearRuntimeContext,
    *,
    deps: FullLinearRuntimeDeps,
    fit_policy: _LinearFitPolicy,
    krylov_cfg: Any | None,
    status_callback: _StatusCallback,
) -> RuntimeLinearResult:
    """Accept a valid time-path result or fall back to the Krylov branch."""

    if _valid_growth(
        result.gamma,
        result.omega,
        require_positive=fit_policy.require_positive,
    ):
        return _finalize_linear_result(
            result,
            ctx=ctx,
            deps=deps,
            status_callback=status_callback,
        )
    _status(status_callback, "time-path result rejected; falling back to Krylov solve")
    return _run_krylov_linear_runtime(
        ctx,
        deps=deps,
        krylov_cfg=krylov_cfg,
        status_callback=status_callback,
    )


def _run_linear_runtime_branch(
    ctx: _LinearRuntimeContext,
    *,
    deps: FullLinearRuntimeDeps,
    method: str | None,
    dt: float | None,
    steps: int | None,
    sample_stride: int | None,
    fit_policy: _LinearFitPolicy,
    krylov_cfg: Any | None,
    show_progress: bool,
    status_callback: _StatusCallback,
) -> RuntimeLinearResult:
    if ctx.solver_key == "krylov":
        return _run_krylov_linear_runtime(
            ctx,
            deps=deps,
            krylov_cfg=krylov_cfg,
            status_callback=status_callback,
        )

    result = _run_time_linear(
        ctx,
        deps=deps,
        method=method,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
        fit_policy=fit_policy,
        show_progress=show_progress,
        status_callback=status_callback,
    )
    if ctx.solver_key == "auto":
        return _finalize_auto_linear_runtime(
            result,
            ctx=ctx,
            deps=deps,
            fit_policy=fit_policy,
            krylov_cfg=krylov_cfg,
            status_callback=status_callback,
        )

    return _finalize_linear_result(
        result,
        ctx=ctx,
        deps=deps,
        status_callback=status_callback,
    )


def run_full_linear_runtime(
    cfg: RuntimeConfig,
    *,
    deps: FullLinearRuntimeDeps,
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
    krylov_cfg: Any | None,
    mode_method: str,
    fit_signal: str,
    return_state: bool,
    show_progress: bool,
    initial_state: Any | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> RuntimeLinearResult:
    """Run one full-GK linear point from a runtime config."""

    ctx = _prepare_linear_runtime_context(
        cfg,
        deps=deps,
        ky_target=ky_target,
        n_laguerre=Nl,
        n_hermite=Nm,
        solver=solver,
        fit_signal=fit_signal,
        return_state=return_state,
        initial_state=initial_state,
        status_callback=status_callback,
    )
    fit_policy = _build_linear_fit_policy(
        mode_method=mode_method,
        auto_window=auto_window,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
    )

    return _run_linear_runtime_branch(
        ctx,
        deps=deps,
        method=method,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
        fit_policy=fit_policy,
        krylov_cfg=krylov_cfg,
        show_progress=show_progress,
        status_callback=status_callback,
    )


@dataclass(frozen=True)
class ScanAndModeResult:
    """Linear scan plus a representative fitted eigenfunction."""

    scan: LinearScanResult
    eigenfunction: np.ndarray
    grid: SpectralGrid
    ky_selected: float
    tmin: float | None
    tmax: float | None


def _indexed_control(value: float | int | np.ndarray | None, index: int, cast):
    if isinstance(value, np.ndarray):
        return cast(value[index])
    return None if value is None else cast(value)


def run_linear_scan(
    *,
    ky_values: np.ndarray,
    run_linear_fn: Callable[..., LinearRunResult],
    cfg: Any,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    solver: str,
    krylov_cfg: Any,
    window_kw: dict[str, Any],
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    run_kwargs: dict[str, Any] | None = None,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], object] | None = None,
) -> LinearScanResult:
    """Run a deterministic pointwise linear scan over ``ky_values``."""

    rows: list[tuple[float, float, float]] = []
    for index, ky in enumerate(np.asarray(ky_values, dtype=float)):
        n_l, n_m = (
            resolution_policy(float(ky))
            if resolution_policy is not None
            else (Nl, Nm)
        )
        result = run_linear_fn(
            ky_target=float(ky),
            cfg=cfg,
            Nl=int(n_l),
            Nm=int(n_m),
            dt=_indexed_control(dt, index, float),
            steps=_indexed_control(steps, index, int),
            method=method,
            solver=solver,
            krylov_cfg=(
                krylov_policy(float(ky)) if krylov_policy is not None else krylov_cfg
            ),
            auto_window=auto_window,
            tmin=_indexed_control(tmin, index, float),
            tmax=_indexed_control(tmax, index, float),
            **window_kw,
            **(run_kwargs or {}),
        )
        rows.append((float(result.ky), float(result.gamma), float(result.omega)))
    if not rows:
        empty = np.asarray([], dtype=float)
        return LinearScanResult(ky=empty, gamma=empty.copy(), omega=empty.copy())
    values = np.asarray(rows, dtype=float)
    return LinearScanResult(ky=values[:, 0], gamma=values[:, 1], omega=values[:, 2])


def _representative_run(
    scan: LinearScanResult,
    ky_selected: float,
    *,
    linear_fn: Callable[..., LinearRunResult],
    cfg: Any,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    mode_solver: str,
    window_kw: dict[str, Any],
    mode_kwargs: dict[str, Any] | None,
    resolution_policy: Callable[[float], tuple[int, int]] | None,
) -> LinearRunResult:
    n_l, n_m = (
        resolution_policy(ky_selected)
        if resolution_policy is not None
        else (Nl, Nm)
    )
    index = int(np.argmin(np.abs(scan.ky - ky_selected)))
    return linear_fn(
        cfg=cfg,
        ky_target=ky_selected,
        Nl=int(n_l),
        Nm=int(n_m),
        dt=_indexed_control(dt, index, float),
        steps=_indexed_control(steps, index, int),
        method=method,
        solver=mode_solver,
        **window_kw,
        **(mode_kwargs or {}),
    )


def run_scan_and_mode(
    *,
    ky_values: np.ndarray,
    linear_fn: Callable[..., LinearRunResult],
    cfg: Any,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    solver: str,
    mode_solver: str,
    krylov_cfg: Any,
    window_kw: dict[str, Any],
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    run_kwargs: dict[str, Any] | None = None,
    mode_kwargs: dict[str, Any] | None = None,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], object] | None = None,
    select_ky: Callable[[LinearScanResult], float] | None = None,
) -> ScanAndModeResult:
    """Run a pointwise scan and extract the fastest or selected eigenmode."""

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
    ky_selected = (
        float(select_ky(scan))
        if select_ky is not None
        else float(scan.ky[int(np.nanargmax(scan.gamma))])
    )
    run = _representative_run(
        scan,
        ky_selected,
        linear_fn=linear_fn,
        cfg=cfg,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps,
        method=method,
        mode_solver=mode_solver,
        window_kw=window_kw,
        mode_kwargs=mode_kwargs,
        resolution_policy=resolution_policy,
    )
    grid = build_spectral_grid(cfg.grid)
    tmin_fit: float | None = None
    tmax_fit: float | None = None
    if run.t.size >= 2:
        signal = extract_mode_time_series(run.phi_t, run.selection, method="project")
        _gamma, _omega, tmin_fit, tmax_fit = fit_growth_rate_auto(
            run.t, signal, **window_kw
        )
    eigenfunction = extract_eigenfunction(
        run.phi_t,
        run.t,
        run.selection,
        z=np.asarray(grid.z),
        method="snapshot",
        tmin=tmin_fit,
        tmax=tmax_fit,
    )
    return ScanAndModeResult(
        scan=scan,
        eigenfunction=eigenfunction,
        grid=grid,
        ky_selected=ky_selected,
        tmin=tmin_fit,
        tmax=tmax_fit,
    )
