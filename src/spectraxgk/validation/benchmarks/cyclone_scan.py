"""Cyclone linear benchmark ky-scan runner."""

# ruff: noqa: F401

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    ModeSelectionBatch,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.validation.benchmarks.defaults import (
    CYCLONE_KRYLOV_DEFAULT,
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
)
from spectraxgk.validation.benchmarks.batching import (
    _iter_ky_batches,
    _resolve_streaming_window,
)
from spectraxgk.validation.benchmarks.fit_signals import (
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.initialization import _build_initial_condition
from spectraxgk.validation.benchmarks.reference import CycloneRunResult, CycloneScanResult
from spectraxgk.validation.benchmarks.solver_policy import _midplane_index
from spectraxgk.validation.benchmarks.species import (
    REFERENCE_DAMP_ENDS_AMP,
    REFERENCE_DAMP_ENDS_WIDTHFRAC,
    _apply_reference_hypercollisions,
)
from spectraxgk.validation.benchmarks.scan import (
    ScanFitWindowPolicy,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    normalize_fit_signal,
    normalize_solver_key,
    resolve_scan_mode_method,
    should_use_ky_batch,
)
from spectraxgk.config import (
    CycloneBaseCase,
    InitializationConfig,
    TimeConfig,
    resolve_cfl_fac,
)
from spectraxgk.solvers.time.diffrax import integrate_linear_diffrax_streaming
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.solvers.time.explicit import (
    ExplicitTimeConfig,
    integrate_linear_explicit,
)
from spectraxgk.linear import integrate_linear, integrate_linear_diagnostics
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from spectraxgk.solvers.linear.krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.solvers.time.runners import integrate_linear_from_config
from spectraxgk.terms.assembly import compute_fields_cached


from spectraxgk.validation.benchmarks.cyclone_linear import run_cyclone_linear
from spectraxgk.validation.benchmarks.cyclone_scan_branches import (
    CycloneScanHooks,
    run_explicit_time_cyclone_scan,
    run_krylov_cyclone_scan,
)


def _scan_hooks() -> CycloneScanHooks:
    return CycloneScanHooks(
        cyclone_scan_result=CycloneScanResult,
        explicit_time_config=ExplicitTimeConfig,
        mode_selection=ModeSelection,
        select_ky_index=select_ky_index,
        select_ky_grid=select_ky_grid,
        build_initial_condition=_build_initial_condition,
        build_linear_cache=build_linear_cache,
        integrate_linear_explicit=integrate_linear_explicit,
        instantaneous_growth_rate_from_phi=instantaneous_growth_rate_from_phi,
        dominant_eigenpair=dominant_eigenpair,
        normalize_growth_rate=_normalize_growth_rate,
        midplane_index=_midplane_index,
        resolve_cfl_fac=resolve_cfl_fac,
    )

def run_cyclone_scan(
    ky_values: np.ndarray,
    Nl: int = 6,
    Nm: int = 12,
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
    method: str = "imex2",
    params: LinearParams | None = None,
    cfg: CycloneBaseCase | None = None,
    time_cfg: TimeConfig | None = None,
    solver: str = "auto",
    krylov_cfg: KrylovConfig | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    auto_window: bool = True,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 1.0,
    require_positive: bool = True,
    min_amp_fraction: float = 0.0,
    max_fraction: float = 0.8,
    end_fraction: float = 1.0,
    max_amp_fraction: float = 1.0,
    phase_weight: float = 0.2,
    length_weight: float = 0.05,
    min_r2: float = 0.0,
    late_penalty: float = 0.0,
    min_slope: float | None = None,
    min_slope_frac: float = 0.0,
    slope_var_weight: float = 0.0,
    window_method: str = "loglinear",
    mode_method: str = "project",
    mode_only: bool = True,
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    fit_signal: str = "auto",
    diagnostic_norm: str = "none",
    use_jit: bool = True,
    ky_batch: int = 4,
    fixed_batch_shape: bool = True,
    streaming_fit: bool = True,
    streaming_amp_floor: float = 1.0e-30,
    mode_follow: bool = True,
    reference_aligned: bool | None = None,
    gx_reference: bool | None = None,
    show_progress: bool = False,
) -> CycloneScanResult:
    """Run the linear Cyclone benchmark for a list of ky values.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    cfg = cfg or CycloneBaseCase()
    init_cfg = getattr(cfg, "init", None) or InitializationConfig()
    grid_full = build_spectral_grid(cfg.grid)
    if gx_reference is not None:
        reference_aligned = gx_reference
    reference_aligned_use = (
        bool(cfg.reference_aligned)
        if reference_aligned is None
        else bool(reference_aligned)
    )
    geom_cfg = cfg.geometry
    if reference_aligned_use:
        geom_cfg = replace(geom_cfg, drift_scale=1.0)
        if diagnostic_norm == "none":
            diagnostic_norm = "rho_star"
        if mode_method not in {"z_index", "max"}:
            mode_method = "z_index"
    geom = SAlphaGeometry.from_config(geom_cfg)
    if params is None:
        params = LinearParams(
            R_over_Ln=cfg.model.R_over_Ln,
            R_over_LTi=cfg.model.R_over_LTi,
            R_over_LTe=cfg.model.R_over_LTe,
            omega_d_scale=CYCLONE_OMEGA_D_SCALE,
            omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
            rho_star=CYCLONE_RHO_STAR,
            kpar_scale=float(geom.gradpar()),
            nu=cfg.model.nu_i,
            damp_ends_amp=REFERENCE_DAMP_ENDS_AMP,
            damp_ends_widthfrac=REFERENCE_DAMP_ENDS_WIDTHFRAC,
        )
        params = _apply_reference_hypercollisions(params, nhermite=Nm)
    if terms is None:
        if getattr(cfg.model, "adiabatic_ions", False):
            terms = LinearTerms(bpar=0.0)
        else:
            terms = LinearTerms()
    solver_key = normalize_solver_key(solver)
    fit_key = normalize_fit_signal(fit_signal)
    auto_solver = solver_key == "auto"
    if auto_solver:
        solver_key = "explicit_time" if reference_aligned_use else "time"
    streaming_fit, mode_only = apply_auto_fit_scan_policy(
        fit_key, streaming_fit=streaming_fit, mode_only=mode_only
    )
    need_density = fit_key in {"density", "auto"}
    gammas = []
    omegas = []
    ky_out = []

    mode_method = resolve_scan_mode_method(mode_method, mode_only=mode_only)
    use_batch = should_use_ky_batch(
        ky_batch=ky_batch,
        solver_key=solver_key,
        dt=dt,
        steps=steps,
        tmin=tmin,
        tmax=tmax,
    )
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
        max_fraction=max_fraction,
        end_fraction=end_fraction,
        max_amp_fraction=max_amp_fraction,
        phase_weight=phase_weight,
        length_weight=length_weight,
        min_r2=min_r2,
        late_penalty=late_penalty,
        min_slope=min_slope,
        min_slope_frac=min_slope_frac,
        slope_var_weight=slope_var_weight,
        window_method=window_method,
        fit_growth_rate_fn=fit_growth_rate,
        fit_growth_rate_auto_fn=fit_growth_rate_auto,
        normalize_growth_rate_fn=_normalize_growth_rate,
    )

    def _fit_signal(
        signal: np.ndarray, idx: int, dt_i: float, stride: int
    ) -> tuple[float, float]:
        return fit_policy.fit_signal(
            signal,
            idx=idx,
            dt=dt_i,
            stride=stride,
            params=params,
            diagnostic_norm=diagnostic_norm,
        )

    ky_values_arr = np.asarray(ky_values, dtype=float)
    phi_t: jnp.ndarray | np.ndarray
    density_t: jnp.ndarray | np.ndarray | None

    if solver_key == "krylov":
        return run_krylov_cyclone_scan(
            ky_values=ky_values_arr,
            grid_full=grid_full,
            geom=geom,
            params=params,
            terms=terms,
            init_cfg=init_cfg,
            n_laguerre=Nl,
            n_hermite=Nm,
            mode_follow=mode_follow,
            krylov_cfg=krylov_cfg,
            krylov_default=CYCLONE_KRYLOV_DEFAULT,
            diagnostic_norm=diagnostic_norm,
            show_progress=show_progress,
            hooks=_scan_hooks(),
        )

    if solver_key == "explicit_time":
        return run_explicit_time_cyclone_scan(
            ky_values=ky_values_arr,
            grid_full=grid_full,
            geom=geom,
            params=params,
            terms=terms,
            cfg=cfg,
            time_cfg=time_cfg,
            init_cfg=init_cfg,
            n_laguerre=Nl,
            n_hermite=Nm,
            dt=dt,
            steps=steps,
            krylov_cfg=krylov_cfg,
            krylov_default=CYCLONE_KRYLOV_DEFAULT,
            reference_aligned=reference_aligned_use,
            diagnostic_norm=diagnostic_norm,
            show_progress=show_progress,
            hooks=_scan_hooks(),
        )
    if use_batch:
        ky_iter = _iter_ky_batches(
            ky_values_arr,
            ky_batch=ky_batch,
            fixed_batch_shape=fixed_batch_shape,
        )
    else:
        ky_iter = _iter_ky_batches(ky_values_arr, ky_batch=1, fixed_batch_shape=False)
    ky_slice: np.ndarray
    ky_indices: list[int]
    sel_scan: ModeSelection | ModeSelectionBatch

    for batch_start, ky_slice, valid_count in ky_iter:
        if use_batch:
            ky_indices = [
                select_ky_index(np.asarray(grid_full.ky), float(ky)) for ky in ky_slice
            ]
            grid = select_ky_grid(grid_full, ky_indices)
            sel_indices = np.arange(len(ky_indices), dtype=int)
            sel_scan = ModeSelectionBatch(sel_indices, 0, _midplane_index(grid))
            dt_i = float(dt)
            steps_i = int(steps)
        else:
            ky_indices = [select_ky_index(np.asarray(grid_full.ky), float(ky_slice[0]))]
            grid = select_ky_grid(grid_full, ky_indices[0])
            sel_scan = ModeSelection(
                ky_index=0, kx_index=0, z_index=_midplane_index(grid)
            )
            dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
            steps_i = (
                int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)
            )

        ky_local = np.arange(len(ky_indices))
        G0_jax = _build_initial_condition(
            grid,
            geom,
            ky_index=ky_local,
            kx_index=0,
            Nl=Nl,
            Nm=Nm,
            init_cfg=init_cfg,
        )
        cache = build_linear_cache(grid, geom, params, Nl, Nm)

        time_cfg_i = None
        if time_cfg is not None:
            time_cfg_i = replace(time_cfg, dt=dt_i, t_max=dt_i * steps_i)
            if sample_stride is not None:
                time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)

        params_use = params
        if time_cfg_i is not None and time_cfg_i.use_diffrax and streaming_fit:
            t_total = float(time_cfg_i.t_max)
            tmin_i, tmax_i = _resolve_streaming_window(
                t_total,
                indexed_float_value(tmin, batch_start),
                indexed_float_value(tmax, batch_start),
                start_fraction,
                window_fraction,
                1.0,
            )
            _, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
                G0_jax,
                grid,
                geom,
                params_use,
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
                mode_z_index=_midplane_index(grid),
                mode_method=mode_method,
                amp_floor=streaming_amp_floor,
                return_state=False,
            )
            gamma_arr = np.asarray(gamma_vals)
            omega_arr = np.asarray(omega_vals)
            for local_idx in range(valid_count):
                ky_val = ky_slice[local_idx]
                gamma_i, omega_i = _normalize_growth_rate(
                    float(gamma_arr[local_idx]),
                    float(omega_arr[local_idx]),
                    params_use,
                    diagnostic_norm,
                )
                gammas.append(gamma_i)
                omegas.append(omega_i)
                ky_out.append(float(ky_val))
            continue

        if time_cfg_i is not None:
            save_field = (
                "phi+density"
                if fit_key == "auto"
                else ("density" if fit_key == "density" else "phi")
            )
            save_mode = None if fit_key == "auto" else (sel_scan if mode_only else None)
            _, saved = integrate_linear_from_config(
                G0_jax,
                grid,
                geom,
                params_use,
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
                _, phi_out_time = integrate_linear(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
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
                _diag = integrate_linear_diagnostics(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    dt=dt_i,
                    steps=steps_i,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=stride,
                    species_index=None,
                    record_hl_energy=False,
                )
                phi_t = np.asarray(_diag[1])
                density_t = np.asarray(_diag[2]) if len(_diag) > 2 else None

        phi_t_np = np.asarray(phi_t)
        signal_t = None
        if mode_only and phi_t_np.ndim == 2:
            signal_t = phi_t_np

        density_np = None if density_t is None else np.asarray(density_t)
        t = np.arange(phi_t_np.shape[0]) * dt_i * stride

        def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
            if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
                return False
            if require_positive and gamma_val <= 0.0:
                return False
            return True

        for local_idx in range(valid_count):
            ky_val = ky_slice[local_idx]
            if signal_t is None:
                sel_local = ModeSelection(
                    ky_index=local_idx, kx_index=0, z_index=_midplane_index(grid)
                )
                if fit_key == "auto":
                    signal, _name, gamma, omega = _select_fit_signal_auto(
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
                    gamma, omega = _normalize_growth_rate(
                        gamma, omega, params_use, diagnostic_norm
                    )
                    if auto_solver and not _is_valid_growth(gamma, omega):
                        res = run_cyclone_linear(
                            ky_target=float(ky_val),
                            Nl=Nl,
                            Nm=Nm,
                            dt=dt_i,
                            steps=steps_i,
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
                        gamma = float(res.gamma)
                        omega = float(res.omega)
                    gammas.append(gamma)
                    omegas.append(omega)
                    ky_out.append(float(ky_val))
                    continue
                signal = extract_mode_time_series(
                    phi_t_np, sel_local, method=mode_method
                )
            else:
                signal = signal_t[:, local_idx] if signal_t.ndim > 1 else signal_t
            gamma, omega = _fit_signal(signal, batch_start + local_idx, dt_i, stride)
            if auto_solver and not _is_valid_growth(gamma, omega):
                res = run_cyclone_linear(
                    ky_target=float(ky_val),
                    Nl=Nl,
                    Nm=Nm,
                    dt=dt_i,
                    steps=steps_i,
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
                gamma = float(res.gamma)
                omega = float(res.omega)
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_val))
    return CycloneScanResult(
        ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas)
    )
