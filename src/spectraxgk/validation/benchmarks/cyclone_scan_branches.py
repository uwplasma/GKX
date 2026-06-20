"""Solver branches for the Cyclone ky-scan benchmark."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from spectraxgk.diagnostics.analysis import ModeSelection, ModeSelectionBatch
from spectraxgk.validation.benchmarks.batching import _iter_ky_batches
from spectraxgk.validation.benchmarks.reference import CycloneScanResult
from spectraxgk.validation.benchmarks.scan import indexed_float_value
from spectraxgk.validation.benchmarks.cyclone_scan_explicit import (
    choose_reselected_frequency,
    explicit_reselection_target,
    explicit_time_config_for_scan_point,
    krylov_reselected_frequency,
    run_explicit_time_cyclone_scan,
)
from spectraxgk.validation.benchmarks.cyclone_scan_krylov import (
    run_krylov_cyclone_scan,
)
from spectraxgk.validation.benchmarks.cyclone_scan_seed import (
    reduced_seed_from_explicit_trace,
    seed_from_explicit_trace,
    seed_shift,
    use_explicit_seed,
)


@dataclass(frozen=True)
class CycloneScanHooks:
    """Patchable numerical hooks supplied by the Cyclone scan owner module."""

    cyclone_scan_result: type[CycloneScanResult]
    explicit_time_config: type
    mode_selection: type[ModeSelection]
    mode_selection_batch: type[ModeSelectionBatch]
    select_ky_index: Callable[..., int]
    select_ky_grid: Callable[..., Any]
    build_initial_condition: Callable[..., Any]
    build_linear_cache: Callable[..., Any]
    integrate_linear_explicit: Callable[..., Any]
    integrate_linear: Callable[..., Any]
    integrate_linear_diagnostics: Callable[..., Any]
    integrate_linear_from_config: Callable[..., Any]
    integrate_linear_diffrax_streaming: Callable[..., Any]
    instantaneous_growth_rate_from_phi: Callable[
        ..., tuple[float, float, Any, Any, Any]
    ]
    dominant_eigenpair: Callable[..., tuple[Any, Any]]
    extract_mode_time_series: Callable[..., np.ndarray]
    select_fit_signal_auto: Callable[..., tuple[np.ndarray, str, float, float]]
    run_cyclone_linear: Callable[..., Any]
    normalize_growth_rate: Callable[..., tuple[float, float]]
    resolve_streaming_window: Callable[..., tuple[float, float]]
    midplane_index: Callable[..., int]
    resolve_cfl_fac: Callable[..., float]


def _valid_time_branch_growth(
    gamma_val: float,
    omega_val: float,
    *,
    require_positive: bool,
) -> bool:
    if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
        return False
    if require_positive and gamma_val <= 0.0:
        return False
    return True


def _resolve_time_branch_growth(
    gamma: float,
    omega: float,
    *,
    ky_value: float,
    n_laguerre: int,
    n_hermite: int,
    dt: float,
    steps: int,
    method: str,
    params: Any,
    cfg: Any,
    time_cfg: Any | None,
    krylov_cfg: Any | None,
    diagnostic_norm: str,
    auto_solver: bool,
    require_positive: bool,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> tuple[float, float]:
    """Return fitted growth/frequency, using the Krylov fallback if required."""

    if not auto_solver or _valid_time_branch_growth(
        gamma, omega, require_positive=require_positive
    ):
        return gamma, omega

    result = hooks.run_cyclone_linear(
        ky_target=float(ky_value),
        Nl=n_laguerre,
        Nm=n_hermite,
        dt=dt,
        steps=steps,
        method=method,
        params=params,
        cfg=cfg,
        time_cfg=time_cfg,
        solver="krylov",
        krylov_cfg=krylov_cfg,
        diagnostic_norm=diagnostic_norm,
        fit_signal="phi",
        show_progress=show_progress,
    )
    return float(result.gamma), float(result.omega)


@dataclass(frozen=True)
class _CycloneTimeScanBatch:
    batch_start: int
    ky_slice: np.ndarray
    valid_count: int
    ky_local: np.ndarray
    grid: Any
    selection: ModeSelection | ModeSelectionBatch
    dt: float
    steps: int
    state: Any
    cache: Any


@dataclass(frozen=True)
class _CycloneHistoryFitOptions:
    n_laguerre: int
    n_hermite: int
    method: str
    params: Any
    cfg: Any
    time_cfg: Any | None
    krylov_cfg: Any | None
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    max_amp_fraction: float
    max_fraction: float
    end_fraction: float
    phase_weight: float
    length_weight: float
    min_r2: float
    late_penalty: float
    min_slope: float | None
    min_slope_frac: float
    slope_var_weight: float
    window_method: str
    mode_method: str
    mode_only: bool
    fit_key: str
    diagnostic_norm: str
    auto_solver: bool
    fit_policy: Any
    hooks: CycloneScanHooks
    show_progress: bool


@dataclass(frozen=True)
class _CycloneTimeRunOptions:
    ky_values: np.ndarray
    grid_full: Any
    geom: Any
    params: Any
    terms: Any
    cfg: Any
    time_cfg: Any | None
    init_cfg: Any
    n_laguerre: int
    n_hermite: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    sample_stride: int | None
    mode_method: str
    mode_only: bool
    fit_key: str
    need_density: bool
    diagnostic_norm: str
    use_jit: bool
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    use_batch: bool
    hooks: CycloneScanHooks
    show_progress: bool


@dataclass
class _CycloneScanOutput:
    gammas: list[float]
    omegas: list[float]
    ky: list[float]

    @classmethod
    def empty(cls) -> "_CycloneScanOutput":
        return cls(gammas=[], omegas=[], ky=[])


@dataclass(frozen=True)
class _CycloneTimeScanControls:
    run_options: _CycloneTimeRunOptions
    fit_options: _CycloneHistoryFitOptions


def _iter_cyclone_time_scan_batches(options: _CycloneTimeRunOptions):
    if options.use_batch:
        return _iter_ky_batches(
            options.ky_values,
            ky_batch=options.ky_batch,
            fixed_batch_shape=options.fixed_batch_shape,
        )
    return _iter_ky_batches(options.ky_values, ky_batch=1, fixed_batch_shape=False)


def _prepare_cyclone_time_batch(
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    use_batch: bool,
    grid_full: Any,
    geom: Any,
    params: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    hooks: CycloneScanHooks,
) -> _CycloneTimeScanBatch:
    if use_batch:
        ky_indices = [
            hooks.select_ky_index(np.asarray(grid_full.ky), float(ky))
            for ky in ky_slice
        ]
        grid = hooks.select_ky_grid(grid_full, ky_indices)
        ky_local = np.arange(len(ky_indices))
        selection: ModeSelection | ModeSelectionBatch = hooks.mode_selection_batch(
            ky_local.astype(int), 0, hooks.midplane_index(grid)
        )
        dt_i = float(dt)
        steps_i = int(steps)
    else:
        ky_indices = [
            hooks.select_ky_index(np.asarray(grid_full.ky), float(ky_slice[0]))
        ]
        grid = hooks.select_ky_grid(grid_full, ky_indices[0])
        ky_local = np.arange(len(ky_indices))
        selection = hooks.mode_selection(
            ky_index=0, kx_index=0, z_index=hooks.midplane_index(grid)
        )
        dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)

    state = hooks.build_initial_condition(
        grid,
        geom,
        ky_index=ky_local,
        kx_index=0,
        Nl=n_laguerre,
        Nm=n_hermite,
        init_cfg=init_cfg,
    )
    cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)
    return _CycloneTimeScanBatch(
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        ky_local=ky_local,
        grid=grid,
        selection=selection,
        dt=dt_i,
        steps=steps_i,
        state=state,
        cache=cache,
    )


def _cyclone_time_config_for_batch(
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


def _append_cyclone_streaming_results(
    batch: _CycloneTimeScanBatch,
    *,
    geom: Any,
    params: Any,
    terms: Any,
    time_cfg: Any,
    tmin: float | None,
    tmax: float | None,
    start_fraction: float,
    window_fraction: float,
    mode_method: str,
    streaming_amp_floor: float,
    diagnostic_norm: str,
    hooks: CycloneScanHooks,
    show_progress: bool,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> None:
    t_total = float(time_cfg.t_max)
    tmin_i, tmax_i = hooks.resolve_streaming_window(
        t_total,
        indexed_float_value(tmin, batch.batch_start),
        indexed_float_value(tmax, batch.batch_start),
        start_fraction,
        window_fraction,
        1.0,
    )
    _, gamma_vals, omega_vals = hooks.integrate_linear_diffrax_streaming(
        batch.state,
        batch.grid,
        geom,
        params,
        dt=batch.dt,
        steps=batch.steps,
        method=time_cfg.diffrax_solver,
        cache=batch.cache,
        terms=terms,
        adaptive=False,
        rtol=time_cfg.diffrax_rtol,
        atol=time_cfg.diffrax_atol,
        max_steps=time_cfg.diffrax_max_steps,
        progress_bar=time_cfg.progress_bar,
        checkpoint=time_cfg.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal="phi",
        show_progress=show_progress,
        mode_ky_indices=batch.ky_local[: batch.valid_count],
        mode_kx_index=0,
        mode_z_index=hooks.midplane_index(batch.grid),
        mode_method=mode_method,
        amp_floor=streaming_amp_floor,
        return_state=False,
    )
    gamma_arr = np.asarray(gamma_vals)
    omega_arr = np.asarray(omega_vals)
    for local_idx in range(batch.valid_count):
        gamma_i, omega_i = hooks.normalize_growth_rate(
            float(gamma_arr[local_idx]),
            float(omega_arr[local_idx]),
            params,
            diagnostic_norm,
        )
        gammas.append(gamma_i)
        omegas.append(omega_i)
        ky_out.append(float(batch.ky_slice[local_idx]))


def _integrate_cyclone_time_history(
    batch: _CycloneTimeScanBatch,
    *,
    geom: Any,
    params: Any,
    terms: Any,
    time_cfg: Any | None,
    method: str,
    mode_method: str,
    mode_only: bool,
    sample_stride: int | None,
    fit_key: str,
    need_density: bool,
    use_jit: bool,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    if time_cfg is not None:
        save_field = (
            "phi+density"
            if fit_key == "auto"
            else ("density" if fit_key == "density" else "phi")
        )
        save_mode = None if fit_key == "auto" else (batch.selection if mode_only else None)
        _, saved = hooks.integrate_linear_from_config(
            batch.state,
            batch.grid,
            geom,
            params,
            time_cfg,
            cache=batch.cache,
            terms=terms,
            save_mode=save_mode,
            mode_method=mode_method,
            save_field=save_field,
            density_species_index=0 if need_density else None,
        )
        if fit_key == "auto":
            phi_t, density_t = saved
            return np.asarray(phi_t), np.asarray(density_t), time_cfg.sample_stride
        return np.asarray(saved), None, time_cfg.sample_stride

    stride = 1 if sample_stride is None else int(sample_stride)
    if use_jit and not need_density:
        _, phi_out_time = hooks.integrate_linear(
            batch.state,
            batch.grid,
            geom,
            params,
            dt=batch.dt,
            steps=batch.steps,
            method=method,
            cache=batch.cache,
            terms=terms,
            sample_stride=stride,
            show_progress=show_progress,
        )
        return np.asarray(phi_out_time), None, stride

    diag = hooks.integrate_linear_diagnostics(
        batch.state,
        batch.grid,
        geom,
        params,
        dt=batch.dt,
        steps=batch.steps,
        method=method,
        cache=batch.cache,
        terms=terms,
        sample_stride=stride,
        species_index=None,
        record_hl_energy=False,
    )
    density_t = np.asarray(diag[2]) if len(diag) > 2 else None
    return np.asarray(diag[1]), density_t, stride


def _resolve_and_append_cyclone_fit(
    *,
    gamma: float,
    omega: float,
    ky_val: float,
    batch: _CycloneTimeScanBatch,
    options: _CycloneHistoryFitOptions,
    output: _CycloneScanOutput,
) -> None:
    gamma, omega = _resolve_time_branch_growth(
        gamma,
        omega,
        ky_value=float(ky_val),
        n_laguerre=options.n_laguerre,
        n_hermite=options.n_hermite,
        dt=batch.dt,
        steps=batch.steps,
        method=options.method,
        params=options.params,
        cfg=options.cfg,
        time_cfg=options.time_cfg,
        krylov_cfg=options.krylov_cfg,
        diagnostic_norm=options.diagnostic_norm,
        auto_solver=options.auto_solver,
        require_positive=options.require_positive,
        hooks=options.hooks,
        show_progress=options.show_progress,
    )
    output.gammas.append(gamma)
    output.omegas.append(omega)
    output.ky.append(float(ky_val))


def _auto_history_fit_for_local_mode(
    *,
    t: np.ndarray,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    batch: _CycloneTimeScanBatch,
    local_idx: int,
    options: _CycloneHistoryFitOptions,
) -> tuple[float, float]:
    sel_local = options.hooks.mode_selection(
        ky_index=local_idx,
        kx_index=0,
        z_index=options.hooks.midplane_index(batch.grid),
    )
    _signal, _name, gamma, omega = options.hooks.select_fit_signal_auto(
        t,
        phi_t,
        density_t,
        sel_local,
        mode_method=options.mode_method,
        tmin=indexed_float_value(options.tmin, batch.batch_start + local_idx),
        tmax=indexed_float_value(options.tmax, batch.batch_start + local_idx),
        window_fraction=options.window_fraction,
        min_points=options.min_points,
        start_fraction=options.start_fraction,
        growth_weight=options.growth_weight,
        require_positive=options.require_positive,
        min_amp_fraction=options.min_amp_fraction,
        max_amp_fraction=options.max_amp_fraction,
        window_method=options.window_method,
        max_fraction=options.max_fraction,
        end_fraction=options.end_fraction,
        num_windows=8,
        phase_weight=options.phase_weight,
        length_weight=options.length_weight,
        min_r2=options.min_r2,
        late_penalty=options.late_penalty,
        min_slope=options.min_slope,
        min_slope_frac=options.min_slope_frac,
        slope_var_weight=options.slope_var_weight,
    )
    return options.hooks.normalize_growth_rate(
        gamma, omega, options.params, options.diagnostic_norm
    )


def _history_signal_for_local_mode(
    *,
    phi_t: np.ndarray,
    signal_t: np.ndarray | None,
    batch: _CycloneTimeScanBatch,
    local_idx: int,
    options: _CycloneHistoryFitOptions,
) -> np.ndarray:
    if signal_t is not None:
        return signal_t[:, local_idx] if signal_t.ndim > 1 else signal_t
    sel_local = options.hooks.mode_selection(
        ky_index=local_idx,
        kx_index=0,
        z_index=options.hooks.midplane_index(batch.grid),
    )
    return options.hooks.extract_mode_time_series(
        phi_t, sel_local, method=options.mode_method
    )


def _history_fit_for_local_mode(
    *,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    signal_t: np.ndarray | None,
    t: np.ndarray,
    batch: _CycloneTimeScanBatch,
    local_idx: int,
    stride: int,
    options: _CycloneHistoryFitOptions,
) -> tuple[float, float]:
    if signal_t is None and options.fit_key == "auto":
        return _auto_history_fit_for_local_mode(
            t=t,
            phi_t=phi_t,
            density_t=density_t,
            batch=batch,
            local_idx=local_idx,
            options=options,
        )
    signal = _history_signal_for_local_mode(
        phi_t=phi_t,
        signal_t=signal_t,
        batch=batch,
        local_idx=local_idx,
        options=options,
    )
    return options.fit_policy.fit_signal(
        signal,
        idx=batch.batch_start + local_idx,
        dt=batch.dt,
        stride=stride,
        params=options.params,
        diagnostic_norm=options.diagnostic_norm,
    )


def _append_cyclone_history_fit_results(
    batch: _CycloneTimeScanBatch,
    *,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    stride: int,
    options: _CycloneHistoryFitOptions,
    output: _CycloneScanOutput,
) -> None:
    signal_t = phi_t if options.mode_only and phi_t.ndim == 2 else None
    t = np.arange(phi_t.shape[0]) * batch.dt * stride
    for local_idx in range(batch.valid_count):
        ky_val = batch.ky_slice[local_idx]
        gamma, omega = _history_fit_for_local_mode(
            phi_t=phi_t,
            density_t=density_t,
            signal_t=signal_t,
            t=t,
            batch=batch,
            local_idx=local_idx,
            stride=stride,
            options=options,
        )
        _resolve_and_append_cyclone_fit(
            gamma=gamma,
            omega=omega,
            ky_val=float(ky_val),
            batch=batch,
            options=options,
            output=output,
        )


def _append_cyclone_time_batch_results(
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    run_options: _CycloneTimeRunOptions,
    fit_options: _CycloneHistoryFitOptions,
    output: _CycloneScanOutput,
) -> None:
    batch = _prepare_cyclone_time_batch(
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        use_batch=run_options.use_batch,
        grid_full=run_options.grid_full,
        geom=run_options.geom,
        params=run_options.params,
        init_cfg=run_options.init_cfg,
        n_laguerre=run_options.n_laguerre,
        n_hermite=run_options.n_hermite,
        dt=run_options.dt,
        steps=run_options.steps,
        hooks=run_options.hooks,
    )
    time_cfg_i = _cyclone_time_config_for_batch(
        run_options.time_cfg,
        dt=batch.dt,
        steps=batch.steps,
        sample_stride=run_options.sample_stride,
    )

    if (
        time_cfg_i is not None
        and time_cfg_i.use_diffrax
        and run_options.streaming_fit
    ):
        _append_cyclone_streaming_results(
            batch,
            geom=run_options.geom,
            params=run_options.params,
            terms=run_options.terms,
            time_cfg=time_cfg_i,
            tmin=fit_options.tmin,
            tmax=fit_options.tmax,
            start_fraction=fit_options.start_fraction,
            window_fraction=fit_options.window_fraction,
            mode_method=run_options.mode_method,
            streaming_amp_floor=run_options.streaming_amp_floor,
            diagnostic_norm=run_options.diagnostic_norm,
            hooks=run_options.hooks,
            show_progress=run_options.show_progress,
            gammas=output.gammas,
            omegas=output.omegas,
            ky_out=output.ky,
        )
        return

    phi_t_np, density_np, stride = _integrate_cyclone_time_history(
        batch,
        geom=run_options.geom,
        params=run_options.params,
        terms=run_options.terms,
        time_cfg=time_cfg_i,
        method=run_options.method,
        mode_method=run_options.mode_method,
        mode_only=run_options.mode_only,
        sample_stride=run_options.sample_stride,
        fit_key=run_options.fit_key,
        need_density=run_options.need_density,
        use_jit=run_options.use_jit,
        hooks=run_options.hooks,
        show_progress=run_options.show_progress,
    )
    _append_cyclone_history_fit_results(
        batch,
        phi_t=phi_t_np,
        density_t=density_np,
        stride=stride,
        options=fit_options,
        output=output,
    )


def _build_cyclone_time_run_options(
    *,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    terms: Any,
    cfg: Any,
    time_cfg: Any | None,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    sample_stride: int | None,
    mode_method: str,
    mode_only: bool,
    fit_key: str,
    need_density: bool,
    diagnostic_norm: str,
    use_jit: bool,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    use_batch: bool,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> _CycloneTimeRunOptions:
    """Pack Cyclone time-scan controls that are shared by every ky batch."""

    return _CycloneTimeRunOptions(
        ky_values=ky_values,
        grid_full=grid_full,
        geom=geom,
        params=params,
        terms=terms,
        cfg=cfg,
        time_cfg=time_cfg,
        init_cfg=init_cfg,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        mode_method=mode_method,
        mode_only=mode_only,
        fit_key=fit_key,
        need_density=need_density,
        diagnostic_norm=diagnostic_norm,
        use_jit=use_jit,
        ky_batch=ky_batch,
        fixed_batch_shape=fixed_batch_shape,
        streaming_fit=streaming_fit,
        streaming_amp_floor=streaming_amp_floor,
        use_batch=use_batch,
        hooks=hooks,
        show_progress=show_progress,
    )


def _build_cyclone_history_fit_options(
    *,
    n_laguerre: int,
    n_hermite: int,
    method: str,
    params: Any,
    cfg: Any,
    time_cfg: Any | None,
    krylov_cfg: Any | None,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    max_amp_fraction: float,
    max_fraction: float,
    end_fraction: float,
    phase_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    min_slope: float | None,
    min_slope_frac: float,
    slope_var_weight: float,
    window_method: str,
    mode_method: str,
    mode_only: bool,
    fit_key: str,
    diagnostic_norm: str,
    auto_solver: bool,
    fit_policy: Any,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> _CycloneHistoryFitOptions:
    """Pack growth/frequency fitting controls for Cyclone time histories."""

    return _CycloneHistoryFitOptions(
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        method=method,
        params=params,
        cfg=cfg,
        time_cfg=time_cfg,
        krylov_cfg=krylov_cfg,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        max_amp_fraction=max_amp_fraction,
        max_fraction=max_fraction,
        end_fraction=end_fraction,
        phase_weight=phase_weight,
        length_weight=length_weight,
        min_r2=min_r2,
        late_penalty=late_penalty,
        min_slope=min_slope,
        min_slope_frac=min_slope_frac,
        slope_var_weight=slope_var_weight,
        window_method=window_method,
        mode_method=mode_method,
        mode_only=mode_only,
        fit_key=fit_key,
        diagnostic_norm=diagnostic_norm,
        auto_solver=auto_solver,
        fit_policy=fit_policy,
        hooks=hooks,
        show_progress=show_progress,
    )


def _prepare_cyclone_time_scan_controls(
    *,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    terms: Any,
    cfg: Any,
    time_cfg: Any | None,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    krylov_cfg: Any | None,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    max_amp_fraction: float,
    max_fraction: float,
    end_fraction: float,
    phase_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    min_slope: float | None,
    min_slope_frac: float,
    slope_var_weight: float,
    window_method: str,
    mode_method: str,
    mode_only: bool,
    sample_stride: int | None,
    fit_key: str,
    need_density: bool,
    diagnostic_norm: str,
    use_jit: bool,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    auto_solver: bool,
    use_batch: bool,
    fit_policy: Any,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> _CycloneTimeScanControls:
    """Build all shared Cyclone time-scan controls before batch execution."""

    run_options = _build_cyclone_time_run_options(
        ky_values=ky_values,
        grid_full=grid_full,
        geom=geom,
        params=params,
        terms=terms,
        cfg=cfg,
        time_cfg=time_cfg,
        init_cfg=init_cfg,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        mode_method=mode_method,
        mode_only=mode_only,
        fit_key=fit_key,
        need_density=need_density,
        diagnostic_norm=diagnostic_norm,
        use_jit=use_jit,
        ky_batch=ky_batch,
        fixed_batch_shape=fixed_batch_shape,
        streaming_fit=streaming_fit,
        streaming_amp_floor=streaming_amp_floor,
        use_batch=use_batch,
        hooks=hooks,
        show_progress=show_progress,
    )
    fit_options = _build_cyclone_history_fit_options(
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        method=method,
        params=params,
        cfg=cfg,
        time_cfg=time_cfg,
        krylov_cfg=krylov_cfg,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        max_amp_fraction=max_amp_fraction,
        max_fraction=max_fraction,
        end_fraction=end_fraction,
        phase_weight=phase_weight,
        length_weight=length_weight,
        min_r2=min_r2,
        late_penalty=late_penalty,
        min_slope=min_slope,
        min_slope_frac=min_slope_frac,
        slope_var_weight=slope_var_weight,
        window_method=window_method,
        mode_method=mode_method,
        mode_only=mode_only,
        fit_key=fit_key,
        diagnostic_norm=diagnostic_norm,
        auto_solver=auto_solver,
        fit_policy=fit_policy,
        hooks=hooks,
        show_progress=show_progress,
    )
    return _CycloneTimeScanControls(
        run_options=run_options,
        fit_options=fit_options,
    )


def _run_cyclone_time_scan_batches(
    *,
    run_options: _CycloneTimeRunOptions,
    fit_options: _CycloneHistoryFitOptions,
) -> _CycloneScanOutput:
    """Execute all Cyclone time-scan batches and collect fitted rows."""

    output = _CycloneScanOutput.empty()
    for batch_start, ky_slice, valid_count in _iter_cyclone_time_scan_batches(
        run_options
    ):
        _append_cyclone_time_batch_results(
            batch_start=batch_start,
            ky_slice=ky_slice,
            valid_count=valid_count,
            run_options=run_options,
            fit_options=fit_options,
            output=output,
        )
    return output


def _cyclone_time_scan_result(
    output: _CycloneScanOutput,
    *,
    hooks: CycloneScanHooks,
) -> CycloneScanResult:
    """Pack fitted Cyclone scan rows into the public result type."""

    return hooks.cyclone_scan_result(
        ky=np.array(output.ky),
        gamma=np.array(output.gammas),
        omega=np.array(output.omegas),
    )


def run_time_cyclone_scan(
    *,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    terms: Any,
    cfg: Any,
    time_cfg: Any | None,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    krylov_cfg: Any | None,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    max_amp_fraction: float,
    max_fraction: float,
    end_fraction: float,
    phase_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    min_slope: float | None,
    min_slope_frac: float,
    slope_var_weight: float,
    window_method: str,
    mode_method: str,
    mode_only: bool,
    sample_stride: int | None,
    fit_key: str,
    need_density: bool,
    diagnostic_norm: str,
    use_jit: bool,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    auto_solver: bool,
    use_batch: bool,
    fit_policy: Any,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> CycloneScanResult:
    """Run the standard Cyclone scan time-integration branches."""

    controls = _prepare_cyclone_time_scan_controls(
        ky_values=ky_values,
        grid_full=grid_full,
        geom=geom,
        params=params,
        terms=terms,
        cfg=cfg,
        time_cfg=time_cfg,
        init_cfg=init_cfg,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        dt=dt,
        steps=steps,
        method=method,
        krylov_cfg=krylov_cfg,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        max_amp_fraction=max_amp_fraction,
        max_fraction=max_fraction,
        end_fraction=end_fraction,
        phase_weight=phase_weight,
        length_weight=length_weight,
        min_r2=min_r2,
        late_penalty=late_penalty,
        min_slope=min_slope,
        min_slope_frac=min_slope_frac,
        slope_var_weight=slope_var_weight,
        window_method=window_method,
        sample_stride=sample_stride,
        mode_method=mode_method,
        mode_only=mode_only,
        fit_key=fit_key,
        need_density=need_density,
        diagnostic_norm=diagnostic_norm,
        use_jit=use_jit,
        ky_batch=ky_batch,
        fixed_batch_shape=fixed_batch_shape,
        streaming_fit=streaming_fit,
        streaming_amp_floor=streaming_amp_floor,
        auto_solver=auto_solver,
        use_batch=use_batch,
        fit_policy=fit_policy,
        hooks=hooks,
        show_progress=show_progress,
    )
    output = _run_cyclone_time_scan_batches(
        run_options=controls.run_options,
        fit_options=controls.fit_options,
    )
    return _cyclone_time_scan_result(output, hooks=hooks)


__all__ = [
    "CycloneScanHooks",
    "choose_reselected_frequency",
    "explicit_reselection_target",
    "explicit_time_config_for_scan_point",
    "krylov_reselected_frequency",
    "reduced_seed_from_explicit_trace",
    "seed_from_explicit_trace",
    "seed_shift",
    "use_explicit_seed",
    "_resolve_time_branch_growth",
    "_valid_time_branch_growth",
    "run_explicit_time_cyclone_scan",
    "run_krylov_cyclone_scan",
    "run_time_cyclone_scan",
]
