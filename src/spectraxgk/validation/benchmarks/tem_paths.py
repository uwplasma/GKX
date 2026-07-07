"""Solver path policies for the TEM benchmark family."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import ModeSelection, ModeSelectionBatch
from spectraxgk.validation.benchmarks.scan import _iter_ky_batches
from spectraxgk.diagnostics.growth_rates import _extract_mode_only_signal
from spectraxgk.validation.benchmarks.defaults import LinearRunResult, LinearScanResult
from spectraxgk.validation.benchmarks.scan import ScanFitWindowPolicy


@dataclass(frozen=True)
class TEMPathHooks:
    """Patchable numerical hooks supplied by the TEM public owner module."""

    linear_run_result: type[LinearRunResult]
    linear_scan_result: type[LinearScanResult]
    mode_selection: type[ModeSelection]
    mode_selection_batch: type[ModeSelectionBatch]
    select_ky_index: Callable[..., int]
    select_ky_grid: Callable[..., Any]
    build_initial_condition: Callable[..., Any]
    build_linear_cache: Callable[..., Any]
    dominant_eigenpair: Callable[..., tuple[Any, Any]]
    compute_fields_cached: Callable[..., Any]
    linear_terms_to_term_config: Callable[..., Any]
    integrate_linear: Callable[..., Any]
    integrate_linear_diagnostics: Callable[..., Any]
    integrate_linear_from_config: Callable[..., Any]
    integrate_linear_diffrax_streaming: Callable[..., Any]
    extract_mode_time_series: Callable[..., np.ndarray]
    fit_growth_rate: Callable[..., tuple[float, float]]
    fit_growth_rate_auto: Callable[..., tuple[float, float, float, float]]
    normalize_growth_rate: Callable[..., tuple[float, float]]
    resolve_streaming_window: Callable[..., tuple[float, float]]
    midplane_index: Callable[..., int]


@dataclass(frozen=True)
class _TEMBatchContext:
    """Fully prepared numerical context for one TEM scan batch."""

    batch_start: int
    ky_slice: np.ndarray
    valid_count: int
    grid: Any
    selection: ModeSelection | ModeSelectionBatch
    dt: float
    steps: int
    state: Any
    cache: Any


@dataclass(frozen=True)
class _TEMScanRuntimeOptions:
    """Options shared by every TEM scan batch."""

    n_laguerre: int
    n_hermite: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    time_cfg: Any | None
    solver_key: str
    krylov_cfg: Any | None
    krylov_default: Any
    fit_policy: ScanFitWindowPolicy
    mode_method: str
    mode_only: bool
    sample_stride: int | None
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    init_species_index: int
    diagnostic_norm: str
    use_batch: bool
    show_progress: bool
    hooks: TEMPathHooks


@dataclass
class _TEMScanAccumulator:
    """Mutable TEM scan rows collected across batches."""

    gammas: list[float]
    omegas: list[float]
    ky_out: list[float]

    def result(self, hooks: TEMPathHooks) -> LinearScanResult:
        """Pack accumulated rows with the public result type."""

        return hooks.linear_scan_result(
            ky=np.array(self.ky_out),
            gamma=np.array(self.gammas),
            omega=np.array(self.omegas),
        )


@dataclass(frozen=True)
class _TEMTimePathTiming:
    """Resolved sampling controls for one TEM initial-value solve."""

    dt: float
    steps: int
    stride: int
    time_cfg: Any | None


@dataclass(frozen=True)
class _TEMTimePathTrace:
    """Saved fields and sample times from one TEM initial-value solve."""

    t: np.ndarray
    phi_t: np.ndarray
    density_t: np.ndarray | None


@dataclass(frozen=True)
class _TEMTimePathFitPolicy:
    """Window-selection controls for fitting a TEM linear trace."""

    auto_window: bool
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float


def _tem_time_path_fit_policy_from_locals(
    values: dict[str, Any],
) -> _TEMTimePathFitPolicy:
    """Pack the single-run TEM fit policy from public path arguments."""

    return _TEMTimePathFitPolicy(
        **{field.name: values[field.name] for field in fields(_TEMTimePathFitPolicy)}
    )


def _tem_scan_fit_policy_from_locals(
    values: dict[str, Any],
    *,
    hooks: TEMPathHooks,
) -> ScanFitWindowPolicy:
    """Pack TEM scan growth-window policy with patchable fit hooks."""

    return ScanFitWindowPolicy(
        tmin=values["tmin"],
        tmax=values["tmax"],
        auto_window=values["auto_window"],
        window_fraction=values["window_fraction"],
        min_points=values["min_points"],
        start_fraction=values["start_fraction"],
        growth_weight=values["growth_weight"],
        require_positive=values["require_positive"],
        min_amp_fraction=values["min_amp_fraction"],
        fit_growth_rate_fn=hooks.fit_growth_rate,
        fit_growth_rate_auto_fn=hooks.fit_growth_rate_auto,
        normalize_growth_rate_fn=hooks.normalize_growth_rate,
    )


def _tem_scan_runtime_options_from_locals(
    values: dict[str, Any],
) -> _TEMScanRuntimeOptions:
    """Pack scan runtime options once after the fit policy is resolved."""

    return _TEMScanRuntimeOptions(
        **{field.name: values[field.name] for field in fields(_TEMScanRuntimeOptions)}
    )


_TEM_KRYLOV_FORWARD_KEYS = (
    "krylov_dim restarts omega_min_factor omega_target_factor omega_cap_factor omega_sign method "
    "power_iters power_dt shift shift_source shift_tol shift_maxiter shift_restart shift_solve_method "
    "shift_preconditioner shift_selection mode_family fallback_method fallback_real_floor"
).split()


def _krylov_eigenvalue(
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    cfg_use: Any,
    hooks: TEMPathHooks,
) -> tuple[Any, Any]:
    kwargs = {
        "terms": terms,
        **{name: getattr(cfg_use, name) for name in _TEM_KRYLOV_FORWARD_KEYS},
    }
    return hooks.dominant_eigenpair(G0_jax, cache, params, **kwargs)


def _prepare_tem_scan_batch(
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    grid_full: Any,
    geom: Any,
    params: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    init_species_index: int,
    use_batch: bool,
    hooks: TEMPathHooks,
) -> _TEMBatchContext:
    selection: ModeSelection | ModeSelectionBatch
    if use_batch:
        ky_indices = [
            hooks.select_ky_index(np.asarray(grid_full.ky), float(ky))
            for ky in ky_slice
        ]
        grid = hooks.select_ky_grid(grid_full, ky_indices)
        selection = hooks.mode_selection_batch(
            np.arange(len(ky_indices), dtype=int), 0, hooks.midplane_index(grid)
        )
        dt_i = float(dt)
        steps_i = int(steps)
    else:
        ky_indices = [
            hooks.select_ky_index(np.asarray(grid_full.ky), float(ky_slice[0]))
        ]
        grid = hooks.select_ky_grid(grid_full, ky_indices[0])
        selection = hooks.mode_selection(
            ky_index=0, kx_index=0, z_index=hooks.midplane_index(grid)
        )
        dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = (
            int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)
        )

    state_np = np.zeros(
        (2, n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=np.complex64,
    )
    state_single = hooks.build_initial_condition(
        grid,
        geom,
        ky_index=np.arange(len(ky_indices), dtype=int),
        kx_index=0,
        Nl=n_laguerre,
        Nm=n_hermite,
        init_cfg=init_cfg,
    )
    state_np[int(init_species_index)] = np.asarray(state_single, dtype=np.complex64)
    return _TEMBatchContext(
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        grid=grid,
        selection=selection,
        dt=dt_i,
        steps=steps_i,
        state=jnp.asarray(state_np),
        cache=hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite),
    )


def _tem_time_config_for_batch(
    time_cfg: Any | None,
    *,
    dt: float,
    steps: int,
    sample_stride: int | None,
) -> Any | None:
    if time_cfg is None:
        return None
    time_cfg_i = replace(time_cfg, dt=dt, t_max=dt * steps)
    if sample_stride is not None:
        time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)
    return time_cfg_i


def _append_tem_krylov_fit(
    *,
    context: _TEMBatchContext,
    params: Any,
    terms: Any,
    krylov_cfg: Any | None,
    krylov_default: Any,
    diagnostic_norm: str,
    hooks: TEMPathHooks,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> None:
    eig, _vec = _krylov_eigenvalue(
        context.state,
        context.cache,
        params,
        terms,
        krylov_cfg or krylov_default,
        hooks,
    )
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    gamma, omega = hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    gammas.append(gamma)
    omegas.append(omega)
    ky_out.append(float(context.ky_slice[0]))


def _append_tem_streaming_fit(
    *,
    context: _TEMBatchContext,
    geom: Any,
    params: Any,
    terms: Any,
    time_cfg_i: Any,
    fit_policy: ScanFitWindowPolicy,
    mode_method: str,
    streaming_amp_floor: float,
    show_progress: bool,
    hooks: TEMPathHooks,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> None:
    t_total = float(time_cfg_i.t_max)
    tmin_i, tmax_i = fit_policy.window_at(context.batch_start)
    tmin_i, tmax_i = hooks.resolve_streaming_window(
        t_total,
        tmin_i,
        tmax_i,
        fit_policy.start_fraction,
        fit_policy.window_fraction,
        1.0,
    )
    _, gamma_vals, omega_vals = hooks.integrate_linear_diffrax_streaming(
        context.state,
        context.grid,
        geom,
        params,
        dt=context.dt,
        steps=context.steps,
        method=time_cfg_i.diffrax_solver,
        cache=context.cache,
        terms=terms,
        adaptive=time_cfg_i.diffrax_adaptive,
        rtol=time_cfg_i.diffrax_rtol,
        atol=time_cfg_i.diffrax_atol,
        max_steps=time_cfg_i.diffrax_max_steps,
        progress_bar=time_cfg_i.progress_bar,
        checkpoint=time_cfg_i.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal="phi",
        show_progress=show_progress,
        mode_ky_indices=np.arange(context.valid_count, dtype=int),
        mode_kx_index=0,
        mode_z_index=hooks.midplane_index(context.grid),
        mode_method=mode_method,
        amp_floor=streaming_amp_floor,
        return_state=False,
    )
    gamma_arr = np.asarray(gamma_vals)
    omega_arr = np.asarray(omega_vals)
    for local_idx in range(context.valid_count):
        gammas.append(float(gamma_arr[local_idx]))
        omegas.append(float(omega_arr[local_idx]))
        ky_out.append(float(context.ky_slice[local_idx]))


def _append_tem_saved_fit(
    *,
    context: _TEMBatchContext,
    geom: Any,
    params: Any,
    terms: Any,
    method: str,
    time_cfg_i: Any | None,
    mode_method: str,
    mode_only: bool,
    sample_stride: int | None,
    fit_policy: ScanFitWindowPolicy,
    diagnostic_norm: str,
    hooks: TEMPathHooks,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> None:
    if time_cfg_i is not None:
        _, phi_t = hooks.integrate_linear_from_config(
            context.state,
            context.grid,
            geom,
            params,
            time_cfg_i,
            cache=context.cache,
            terms=terms,
            save_mode=context.selection if mode_only else None,
            mode_method=mode_method,
        )
        stride = time_cfg_i.sample_stride
    else:
        stride = 1 if sample_stride is None else int(sample_stride)
        _, phi_t = hooks.integrate_linear(
            context.state,
            context.grid,
            geom,
            params,
            dt=context.dt,
            steps=context.steps,
            method=method,
            cache=context.cache,
            terms=terms,
            sample_stride=stride,
        )

    phi_t_np = np.asarray(phi_t)
    for local_idx in range(context.valid_count):
        if mode_only and phi_t_np.ndim <= 2:
            signal = _extract_mode_only_signal(phi_t_np, local_idx=local_idx)
        else:
            sel_local = hooks.mode_selection(
                ky_index=local_idx,
                kx_index=0,
                z_index=hooks.midplane_index(context.grid),
            )
            signal = hooks.extract_mode_time_series(
                phi_t_np, sel_local, method=mode_method
            )
        gamma, omega = fit_policy.fit_signal(
            signal,
            idx=context.batch_start + local_idx,
            dt=context.dt,
            stride=stride,
            params=params,
            diagnostic_norm=diagnostic_norm,
        )
        gammas.append(gamma)
        omegas.append(omega)
        ky_out.append(float(context.ky_slice[local_idx]))


def _tem_scan_batches(
    ky_values: np.ndarray,
    options: _TEMScanRuntimeOptions,
) -> Any:
    """Select scalar or fixed-width ky batching for the TEM scan."""

    if options.use_batch:
        return _iter_ky_batches(
            ky_values,
            ky_batch=options.ky_batch,
            fixed_batch_shape=options.fixed_batch_shape,
        )
    return _iter_ky_batches(ky_values, ky_batch=1, fixed_batch_shape=False)


def _append_tem_scan_batch(
    *,
    context: _TEMBatchContext,
    geom: Any,
    params: Any,
    terms: Any,
    options: _TEMScanRuntimeOptions,
    acc: _TEMScanAccumulator,
) -> None:
    """Route one prepared TEM batch through its selected solver path."""

    hooks = options.hooks
    if options.solver_key == "krylov":
        _append_tem_krylov_fit(
            context=context,
            params=params,
            terms=terms,
            krylov_cfg=options.krylov_cfg,
            krylov_default=options.krylov_default,
            diagnostic_norm=options.diagnostic_norm,
            hooks=hooks,
            gammas=acc.gammas,
            omegas=acc.omegas,
            ky_out=acc.ky_out,
        )
        return

    time_cfg_i = _tem_time_config_for_batch(
        options.time_cfg,
        dt=context.dt,
        steps=context.steps,
        sample_stride=options.sample_stride,
    )
    if time_cfg_i is not None and time_cfg_i.use_diffrax and options.streaming_fit:
        _append_tem_streaming_fit(
            context=context,
            geom=geom,
            params=params,
            terms=terms,
            time_cfg_i=time_cfg_i,
            fit_policy=options.fit_policy,
            mode_method=options.mode_method,
            streaming_amp_floor=options.streaming_amp_floor,
            show_progress=options.show_progress,
            hooks=hooks,
            gammas=acc.gammas,
            omegas=acc.omegas,
            ky_out=acc.ky_out,
        )
        return

    _append_tem_saved_fit(
        context=context,
        geom=geom,
        params=params,
        terms=terms,
        method=options.method,
        time_cfg_i=time_cfg_i,
        mode_method=options.mode_method,
        mode_only=options.mode_only,
        sample_stride=options.sample_stride,
        fit_policy=options.fit_policy,
        diagnostic_norm=options.diagnostic_norm,
        hooks=hooks,
        gammas=acc.gammas,
        omegas=acc.omegas,
        ky_out=acc.ky_out,
    )


def _run_tem_scan_loop(
    *,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    terms: Any,
    init_cfg: Any,
    options: _TEMScanRuntimeOptions,
) -> _TEMScanAccumulator:
    """Prepare and execute all TEM scan batches."""

    acc = _TEMScanAccumulator(gammas=[], omegas=[], ky_out=[])
    for batch_start, ky_slice, valid_count in _tem_scan_batches(ky_values, options):
        context = _prepare_tem_scan_batch(
            batch_start=batch_start,
            ky_slice=ky_slice,
            valid_count=valid_count,
            grid_full=grid_full,
            geom=geom,
            params=params,
            init_cfg=init_cfg,
            n_laguerre=options.n_laguerre,
            n_hermite=options.n_hermite,
            dt=options.dt,
            steps=options.steps,
            init_species_index=options.init_species_index,
            use_batch=options.use_batch,
            hooks=options.hooks,
        )
        _append_tem_scan_batch(
            context=context,
            geom=geom,
            params=params,
            terms=terms,
            options=options,
            acc=acc,
        )
    return acc


def run_tem_krylov_linear_path(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    n_laguerre: int,
    n_hermite: int,
    sel: ModeSelection,
    krylov_cfg: Any | None,
    krylov_default: Any,
    diagnostic_norm: str,
    hooks: TEMPathHooks,
) -> LinearRunResult:
    """Run the single-ky TEM Krylov branch and package a linear result."""

    cfg_use = krylov_cfg or krylov_default
    cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)
    eig, vec = _krylov_eigenvalue(G0_jax, cache, params, terms, cfg_use, hooks)
    term_cfg = hooks.linear_terms_to_term_config(terms)
    phi = hooks.compute_fields_cached(vec, cache, params, terms=term_cfg).phi
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    gamma, omega = hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    return hooks.linear_run_result(
        t=np.array([0.0]),
        phi_t=np.asarray(phi)[None, ...],
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )


def _validate_tem_time_fit_signal(fit_signal: str) -> None:
    if fit_signal not in {"phi", "density"}:
        raise ValueError("fit_signal must be 'phi' or 'density'")


def _resolve_tem_time_path_timing(
    *,
    dt: float,
    steps: int,
    time_cfg: Any | None,
    sample_stride: int | None,
) -> _TEMTimePathTiming:
    time_cfg_use = time_cfg
    if time_cfg is not None:
        if sample_stride is not None:
            time_cfg_use = replace(time_cfg, sample_stride=sample_stride)
        assert time_cfg_use is not None
        return _TEMTimePathTiming(
            dt=float(time_cfg_use.dt),
            steps=int(round(time_cfg_use.t_max / time_cfg_use.dt)),
            stride=int(time_cfg_use.sample_stride),
            time_cfg=time_cfg_use,
        )
    return _TEMTimePathTiming(
        dt=float(dt),
        steps=int(steps),
        stride=1 if sample_stride is None else int(sample_stride),
        time_cfg=None,
    )


def _integrate_tem_time_path_trace(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    cache: Any,
    timing: _TEMTimePathTiming,
    method: str,
    fit_signal: str,
    density_species_index: int,
    show_progress: bool,
    hooks: TEMPathHooks,
) -> _TEMTimePathTrace:
    if fit_signal == "density":
        diag = hooks.integrate_linear_diagnostics(
            G0_jax,
            grid,
            geom,
            params,
            dt=timing.dt,
            steps=timing.steps,
            method=method,
            cache=cache,
            terms=terms,
            sample_stride=timing.stride,
            species_index=density_species_index,
        )
        _, phi_t, density_t, *_ = diag
    elif timing.time_cfg is not None:
        _, phi_t = hooks.integrate_linear_from_config(
            G0_jax,
            grid,
            geom,
            params,
            timing.time_cfg,
            cache=cache,
            terms=terms,
            show_progress=show_progress,
        )
        density_t = None
    else:
        _, phi_t = hooks.integrate_linear(
            G0_jax,
            grid,
            geom,
            params,
            dt=timing.dt,
            steps=timing.steps,
            method=method,
            cache=cache,
            terms=terms,
            sample_stride=timing.stride,
            show_progress=show_progress,
        )
        density_t = None

    phi_t_np = np.asarray(phi_t)
    return _TEMTimePathTrace(
        t=np.arange(phi_t_np.shape[0]) * timing.dt * timing.stride,
        phi_t=phi_t_np,
        density_t=None if density_t is None else np.asarray(density_t),
    )


def _tem_time_path_signal(
    trace: _TEMTimePathTrace,
    *,
    fit_signal: str,
    sel: ModeSelection,
    mode_method: str,
    hooks: TEMPathHooks,
) -> np.ndarray:
    if fit_signal == "density" and trace.density_t is not None:
        return hooks.extract_mode_time_series(trace.density_t, sel, method=mode_method)
    return hooks.extract_mode_time_series(trace.phi_t, sel, method=mode_method)


def _fit_tem_time_path_signal(
    *,
    t: np.ndarray,
    signal: np.ndarray,
    policy: _TEMTimePathFitPolicy,
    hooks: TEMPathHooks,
) -> tuple[float, float]:
    auto_fit_kwargs: dict[str, Any] = {
        "window_fraction": policy.window_fraction,
        "min_points": policy.min_points,
        "start_fraction": policy.start_fraction,
        "growth_weight": policy.growth_weight,
        "require_positive": policy.require_positive,
        "min_amp_fraction": policy.min_amp_fraction,
    }
    if policy.auto_window and policy.tmin is None and policy.tmax is None:
        gamma, omega, _tmin, _tmax = hooks.fit_growth_rate_auto(
            t, signal, **auto_fit_kwargs
        )
        return gamma, omega
    try:
        return hooks.fit_growth_rate(t, signal, tmin=policy.tmin, tmax=policy.tmax)
    except ValueError:
        gamma, omega, _tmin, _tmax = hooks.fit_growth_rate_auto(
            t, signal, **auto_fit_kwargs
        )
        return gamma, omega


def _tem_time_path_result(
    *,
    trace: _TEMTimePathTrace,
    gamma: float,
    omega: float,
    grid: Any,
    sel: ModeSelection,
    hooks: TEMPathHooks,
) -> LinearRunResult:
    return hooks.linear_run_result(
        t=trace.t,
        phi_t=trace.phi_t,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )


def run_tem_time_linear_path(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float,
    steps: int,
    method: str,
    time_cfg: Any | None,
    sample_stride: int | None,
    fit_signal: str,
    density_species_index: int,
    sel: ModeSelection,
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
    diagnostic_norm: str,
    show_progress: bool,
    hooks: TEMPathHooks,
) -> LinearRunResult:
    """Run the single-ky TEM time-integration branch and fit the selected signal."""

    _validate_tem_time_fit_signal(fit_signal)
    cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)
    timing = _resolve_tem_time_path_timing(
        dt=dt,
        steps=steps,
        time_cfg=time_cfg,
        sample_stride=sample_stride,
    )
    trace = _integrate_tem_time_path_trace(
        G0_jax=G0_jax,
        grid=grid,
        geom=geom,
        params=params,
        terms=terms,
        cache=cache,
        timing=timing,
        method=method,
        fit_signal=fit_signal,
        density_species_index=density_species_index,
        show_progress=show_progress,
        hooks=hooks,
    )
    signal = _tem_time_path_signal(
        trace,
        fit_signal=fit_signal,
        sel=sel,
        mode_method=mode_method,
        hooks=hooks,
    )
    fit_policy = _tem_time_path_fit_policy_from_locals(locals())
    gamma, omega = _fit_tem_time_path_signal(
        t=trace.t,
        signal=signal,
        policy=fit_policy,
        hooks=hooks,
    )
    gamma, omega = hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    return _tem_time_path_result(
        trace=trace,
        gamma=gamma,
        omega=omega,
        grid=grid,
        sel=sel,
        hooks=hooks,
    )


def run_tem_scan_batches(
    *,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    terms: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    time_cfg: Any | None,
    solver_key: str,
    krylov_cfg: Any | None,
    krylov_default: Any,
    tmin: float | None,
    tmax: float | None,
    auto_window: bool,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    mode_method: str,
    mode_only: bool,
    sample_stride: int | None,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    init_species_index: int,
    diagnostic_norm: str,
    use_batch: bool,
    hooks: TEMPathHooks,
    show_progress: bool,
) -> LinearScanResult:
    """Run TEM scan batches across Krylov, streaming, and saved-time branches."""

    fit_policy = _tem_scan_fit_policy_from_locals(locals(), hooks=hooks)
    options = _tem_scan_runtime_options_from_locals(locals())
    return _run_tem_scan_loop(
        ky_values=ky_values,
        grid_full=grid_full,
        geom=geom,
        params=params,
        terms=terms,
        init_cfg=init_cfg,
        options=options,
    ).result(hooks)
