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
        sel_scan: ModeSelection | ModeSelectionBatch
        if use_batch:
            ky_indices = [
                hooks.select_ky_index(np.asarray(grid_full.ky), float(ky))
                for ky in ky_slice
            ]
            grid = hooks.select_ky_grid(grid_full, ky_indices)
            ky_local = np.arange(len(ky_indices))
            sel_scan = hooks.mode_selection_batch(
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
            sel_scan = hooks.mode_selection(
                ky_index=0, kx_index=0, z_index=hooks.midplane_index(grid)
            )
            dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
            steps_i = (
                int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)
            )

        G0_jax = hooks.build_initial_condition(
            grid,
            geom,
            ky_index=ky_local,
            kx_index=0,
            Nl=n_laguerre,
            Nm=n_hermite,
            init_cfg=init_cfg,
        )
        cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)

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
                adaptive=False,
                rtol=time_cfg_i.diffrax_rtol,
                atol=time_cfg_i.diffrax_atol,
                max_steps=time_cfg_i.diffrax_max_steps,
                progress_bar=time_cfg_i.progress_bar,
                checkpoint=time_cfg_i.checkpoint,
                tmin=tmin_i,
                tmax=tmax_i,
                fit_signal="phi",
                show_progress=show_progress,
                mode_ky_indices=ky_local[:valid_count],
                mode_kx_index=0,
                mode_z_index=hooks.midplane_index(grid),
                mode_method=mode_method,
                amp_floor=streaming_amp_floor,
                return_state=False,
            )
            gamma_arr = np.asarray(gamma_vals)
            omega_arr = np.asarray(omega_vals)
            for local_idx in range(valid_count):
                gamma_i, omega_i = hooks.normalize_growth_rate(
                    float(gamma_arr[local_idx]),
                    float(omega_arr[local_idx]),
                    params,
                    diagnostic_norm,
                )
                gammas.append(gamma_i)
                omegas.append(omega_i)
                ky_out.append(float(ky_slice[local_idx]))
            continue

        if time_cfg_i is not None:
            save_field = (
                "phi+density"
                if fit_key == "auto"
                else ("density" if fit_key == "density" else "phi")
            )
            save_mode = None if fit_key == "auto" else (sel_scan if mode_only else None)
            _, saved = hooks.integrate_linear_from_config(
                G0_jax,
                grid,
                geom,
                params,
                time_cfg_i,
                cache=cache,
                terms=terms,
                save_mode=save_mode,
                mode_method=mode_method,
                save_field=save_field,
                density_species_index=0 if need_density else None,
            )
            if fit_key == "auto":
                phi_t, density_t = saved
                phi_t = np.asarray(phi_t)
                density_t = np.asarray(density_t)
            else:
                phi_t = np.asarray(saved)
                density_t = None
            stride = time_cfg_i.sample_stride
        else:
            stride = 1 if sample_stride is None else int(sample_stride)
            if use_jit and not need_density:
                _, phi_out_time = hooks.integrate_linear(
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
                    show_progress=show_progress,
                )
                phi_t = phi_out_time
                density_t = None
            else:
                diag = hooks.integrate_linear_diagnostics(
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
                    species_index=None,
                    record_hl_energy=False,
                )
                phi_t = np.asarray(diag[1])
                density_t = np.asarray(diag[2]) if len(diag) > 2 else None

        phi_t_np = np.asarray(phi_t)
        signal_t = phi_t_np if mode_only and phi_t_np.ndim == 2 else None
        density_np = None if density_t is None else np.asarray(density_t)
        t = np.arange(phi_t_np.shape[0]) * dt_i * stride
        for local_idx in range(valid_count):
            ky_val = ky_slice[local_idx]
            if signal_t is None:
                sel_local = hooks.mode_selection(
                    ky_index=local_idx, kx_index=0, z_index=hooks.midplane_index(grid)
                )
                if fit_key == "auto":
                    _signal, _name, gamma, omega = hooks.select_fit_signal_auto(
                        t,
                        phi_t_np,
                        density_np,
                        sel_local,
                        mode_method=mode_method,
                        tmin=indexed_float_value(tmin, batch_start + local_idx),
                        tmax=indexed_float_value(tmax, batch_start + local_idx),
                        window_fraction=window_fraction,
                        min_points=min_points,
                        start_fraction=start_fraction,
                        growth_weight=growth_weight,
                        require_positive=require_positive,
                        min_amp_fraction=min_amp_fraction,
                        max_amp_fraction=max_amp_fraction,
                        window_method=window_method,
                        max_fraction=max_fraction,
                        end_fraction=end_fraction,
                        num_windows=8,
                        phase_weight=phase_weight,
                        length_weight=length_weight,
                        min_r2=min_r2,
                        late_penalty=late_penalty,
                        min_slope=min_slope,
                        min_slope_frac=min_slope_frac,
                        slope_var_weight=slope_var_weight,
                    )
                    gamma, omega = hooks.normalize_growth_rate(
                        gamma, omega, params, diagnostic_norm
                    )
                    gamma, omega = _resolve_time_branch_growth(
                        gamma,
                        omega,
                        ky_value=float(ky_val),
                        n_laguerre=n_laguerre,
                        n_hermite=n_hermite,
                        dt=dt_i,
                        steps=steps_i,
                        method=method,
                        params=params,
                        cfg=cfg,
                        time_cfg=time_cfg,
                        krylov_cfg=krylov_cfg,
                        diagnostic_norm=diagnostic_norm,
                        auto_solver=auto_solver,
                        require_positive=require_positive,
                        hooks=hooks,
                        show_progress=show_progress,
                    )
                    gammas.append(gamma)
                    omegas.append(omega)
                    ky_out.append(float(ky_val))
                    continue
                signal = hooks.extract_mode_time_series(
                    phi_t_np, sel_local, method=mode_method
                )
            else:
                signal = signal_t[:, local_idx] if signal_t.ndim > 1 else signal_t
            gamma, omega = fit_policy.fit_signal(
                signal,
                idx=batch_start + local_idx,
                dt=dt_i,
                stride=stride,
                params=params,
                diagnostic_norm=diagnostic_norm,
            )
            gamma, omega = _resolve_time_branch_growth(
                gamma,
                omega,
                ky_value=float(ky_val),
                n_laguerre=n_laguerre,
                n_hermite=n_hermite,
                dt=dt_i,
                steps=steps_i,
                method=method,
                params=params,
                cfg=cfg,
                time_cfg=time_cfg,
                krylov_cfg=krylov_cfg,
                diagnostic_norm=diagnostic_norm,
                auto_solver=auto_solver,
                require_positive=require_positive,
                hooks=hooks,
                show_progress=show_progress,
            )
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_val))
    return hooks.cyclone_scan_result(
        ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas)
    )


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
