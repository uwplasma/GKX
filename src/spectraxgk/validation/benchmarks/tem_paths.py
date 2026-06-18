"""Solver path policies for the TEM benchmark family."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import ModeSelection, ModeSelectionBatch
from spectraxgk.validation.benchmarks.batching import _iter_ky_batches
from spectraxgk.validation.benchmarks.fit_signals import _extract_mode_only_signal
from spectraxgk.validation.benchmarks.reference import LinearRunResult, LinearScanResult
from spectraxgk.validation.benchmarks.scan import ScanFitWindowPolicy, indexed_float_value


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


def _krylov_eigenvalue(
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    cfg_use: Any,
    hooks: TEMPathHooks,
) -> tuple[Any, Any]:
    return hooks.dominant_eigenpair(
        G0_jax,
        cache,
        params,
        terms=terms,
        krylov_dim=cfg_use.krylov_dim,
        restarts=cfg_use.restarts,
        omega_min_factor=cfg_use.omega_min_factor,
        omega_target_factor=cfg_use.omega_target_factor,
        omega_cap_factor=cfg_use.omega_cap_factor,
        omega_sign=cfg_use.omega_sign,
        method=cfg_use.method,
        power_iters=cfg_use.power_iters,
        power_dt=cfg_use.power_dt,
        shift=cfg_use.shift,
        shift_source=cfg_use.shift_source,
        shift_tol=cfg_use.shift_tol,
        shift_maxiter=cfg_use.shift_maxiter,
        shift_restart=cfg_use.shift_restart,
        shift_solve_method=cfg_use.shift_solve_method,
        shift_preconditioner=cfg_use.shift_preconditioner,
        shift_selection=cfg_use.shift_selection,
        mode_family=cfg_use.mode_family,
        fallback_method=cfg_use.fallback_method,
        fallback_real_floor=cfg_use.fallback_real_floor,
    )


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

    if fit_signal not in {"phi", "density"}:
        raise ValueError("fit_signal must be 'phi' or 'density'")
    cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)
    if time_cfg is not None:
        time_cfg_use = time_cfg
        if sample_stride is not None:
            time_cfg_use = replace(time_cfg, sample_stride=sample_stride)
        dt = float(time_cfg_use.dt)
        steps = int(round(time_cfg_use.t_max / time_cfg_use.dt))
        if fit_signal == "density":
            diag = hooks.integrate_linear_diagnostics(
                G0_jax,
                grid,
                geom,
                params,
                dt=dt,
                steps=steps,
                method=method,
                cache=cache,
                terms=terms,
                sample_stride=time_cfg_use.sample_stride,
                species_index=density_species_index,
            )
            if len(diag) == 4:
                _, phi_t, density_t, _ = diag
            else:
                _, phi_t, density_t = diag
        else:
            _, phi_t = hooks.integrate_linear_from_config(
                G0_jax,
                grid,
                geom,
                params,
                time_cfg_use,
                cache=cache,
                terms=terms,
                show_progress=show_progress,
            )
            density_t = None
        stride = time_cfg_use.sample_stride
    else:
        stride = 1 if sample_stride is None else int(sample_stride)
        if fit_signal == "density":
            diag = hooks.integrate_linear_diagnostics(
                G0_jax,
                grid,
                geom,
                params,
                dt=dt,
                steps=steps,
                method=method,
                cache=cache,
                terms=terms,
                sample_stride=stride,
                species_index=density_species_index,
            )
            if len(diag) == 4:
                _, phi_t, density_t, _ = diag
            else:
                _, phi_t, density_t = diag
        else:
            _, phi_t = hooks.integrate_linear(
                G0_jax,
                grid,
                geom,
                params,
                dt=dt,
                steps=steps,
                method=method,
                cache=cache,
                terms=terms,
                sample_stride=stride,
                show_progress=show_progress,
            )
            density_t = None

    phi_t_np = np.asarray(phi_t)
    t = np.arange(phi_t_np.shape[0]) * dt * stride
    if fit_signal == "density" and density_t is not None:
        signal = hooks.extract_mode_time_series(
            np.asarray(density_t), sel, method=mode_method
        )
    else:
        signal = hooks.extract_mode_time_series(phi_t_np, sel, method=mode_method)
    if auto_window and tmin is None and tmax is None:
        gamma, omega, _tmin, _tmax = hooks.fit_growth_rate_auto(
            t,
            signal,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
        )
    else:
        try:
            gamma, omega = hooks.fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)
        except ValueError:
            gamma, omega, _tmin, _tmax = hooks.fit_growth_rate_auto(
                t,
                signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
    gamma, omega = hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    return hooks.linear_run_result(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
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

    fit_policy = ScanFitWindowPolicy(
        tmin=tmin,
        tmax=tmax,
        auto_window=auto_window,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        fit_growth_rate_fn=hooks.fit_growth_rate,
        fit_growth_rate_auto_fn=hooks.fit_growth_rate_auto,
        normalize_growth_rate_fn=hooks.normalize_growth_rate,
    )
    if use_batch:
        ky_iter = _iter_ky_batches(
            ky_values,
            ky_batch=ky_batch,
            fixed_batch_shape=fixed_batch_shape,
        )
    else:
        ky_iter = _iter_ky_batches(ky_values, ky_batch=1, fixed_batch_shape=False)

    gammas: list[float] = []
    omegas: list[float] = []
    ky_out: list[float] = []
    for batch_start, ky_slice, valid_count in ky_iter:
        sel: ModeSelection | ModeSelectionBatch
        if use_batch:
            ky_indices = [
                hooks.select_ky_index(np.asarray(grid_full.ky), float(ky))
                for ky in ky_slice
            ]
            grid = hooks.select_ky_grid(grid_full, ky_indices)
            sel = hooks.mode_selection_batch(
                np.arange(len(ky_indices), dtype=int), 0, hooks.midplane_index(grid)
            )
            dt_i = float(dt)
            steps_i = int(steps)
        else:
            ky_indices = [
                hooks.select_ky_index(np.asarray(grid_full.ky), float(ky_slice[0]))
            ]
            grid = hooks.select_ky_grid(grid_full, ky_indices[0])
            sel = hooks.mode_selection(
                ky_index=0, kx_index=0, z_index=hooks.midplane_index(grid)
            )
            dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
            steps_i = (
                int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)
            )

        G0 = np.zeros(
            (2, n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size),
            dtype=np.complex64,
        )
        G0_single = hooks.build_initial_condition(
            grid,
            geom,
            ky_index=np.arange(len(ky_indices), dtype=int),
            kx_index=0,
            Nl=n_laguerre,
            Nm=n_hermite,
            init_cfg=init_cfg,
        )
        G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)

        cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)
        G0_jax = jnp.asarray(G0)
        if solver_key == "krylov":
            eig, _vec = _krylov_eigenvalue(
                G0_jax,
                cache,
                params,
                terms,
                krylov_cfg or krylov_default,
                hooks,
            )
            gamma = float(np.real(eig))
            omega = float(-np.imag(eig))
            gamma, omega = hooks.normalize_growth_rate(
                gamma, omega, params, diagnostic_norm
            )
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_slice[0]))
            continue

        time_cfg_i = None
        if time_cfg is not None:
            time_cfg_i = replace(time_cfg, dt=dt_i, t_max=dt_i * steps_i)
            if sample_stride is not None:
                time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)

        if time_cfg_i is not None and time_cfg_i.use_diffrax and streaming_fit:
            t_total = float(time_cfg_i.t_max)
            tmin_i, tmax_i = hooks.resolve_streaming_window(
                t_total,
                indexed_float_value(tmin, batch_start),
                indexed_float_value(tmax, batch_start),
                start_fraction,
                window_fraction,
                1.0,
            )
            _, gamma_vals, omega_vals = hooks.integrate_linear_diffrax_streaming(
                G0_jax,
                grid,
                geom,
                params,
                dt=dt_i,
                steps=steps_i,
                method=time_cfg_i.diffrax_solver,
                cache=cache,
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
                mode_ky_indices=np.arange(valid_count, dtype=int),
                mode_kx_index=0,
                mode_z_index=hooks.midplane_index(grid),
                mode_method=mode_method,
                amp_floor=streaming_amp_floor,
                return_state=False,
            )
            gamma_arr = np.asarray(gamma_vals)
            omega_arr = np.asarray(omega_vals)
            for local_idx in range(valid_count):
                gammas.append(float(gamma_arr[local_idx]))
                omegas.append(float(omega_arr[local_idx]))
                ky_out.append(float(ky_slice[local_idx]))
            continue

        if time_cfg_i is not None:
            _, phi_t = hooks.integrate_linear_from_config(
                G0_jax,
                grid,
                geom,
                params,
                time_cfg_i,
                cache=cache,
                terms=terms,
                save_mode=sel if mode_only else None,
                mode_method=mode_method,
            )
            stride = time_cfg_i.sample_stride
        else:
            stride = 1 if sample_stride is None else int(sample_stride)
            _, phi_t = hooks.integrate_linear(
                G0_jax,
                grid,
                geom,
                params,
                dt=dt_i,
                steps=steps_i,
                method=method,
                cache=cache,
                terms=terms,
                sample_stride=stride,
            )

        phi_t_np = np.asarray(phi_t)
        for local_idx in range(valid_count):
            if mode_only and phi_t_np.ndim <= 2:
                signal = _extract_mode_only_signal(phi_t_np, local_idx=local_idx)
            else:
                sel_local = hooks.mode_selection(
                    ky_index=local_idx, kx_index=0, z_index=hooks.midplane_index(grid)
                )
                signal = hooks.extract_mode_time_series(
                    phi_t_np, sel_local, method=mode_method
                )
            gamma, omega = fit_policy.fit_signal(
                signal,
                idx=batch_start + local_idx,
                dt=dt_i,
                stride=stride,
                params=params,
                diagnostic_norm=diagnostic_norm,
            )
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_slice[local_idx]))
    return hooks.linear_scan_result(
        ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas)
    )
