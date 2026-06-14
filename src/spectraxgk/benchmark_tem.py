"""TEM benchmark runners behind the public :mod:`spectraxgk.benchmarks` facade."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from spectraxgk.analysis import (
    ModeSelection,
    ModeSelectionBatch,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    select_ky_index,
)
from spectraxgk.benchmark_defaults import (
    TEM_KRYLOV_DEFAULT,
    TEM_OMEGA_D_SCALE,
    TEM_OMEGA_STAR_SCALE,
    TEM_RHO_STAR,
)
from spectraxgk.benchmark_helpers import (
    LinearRunResult,
    LinearScanResult,
    _build_initial_condition,
    _extract_mode_only_signal,
    _iter_ky_batches,
    _midplane_index,
    _normalize_growth_rate,
    _resolve_streaming_window,
    _two_species_params,
)
from spectraxgk.benchmark_scan import (
    ScanFitWindowPolicy,
    indexed_float_value,
    normalize_solver_key,
    resolve_scan_mode_method,
    should_use_ky_batch,
)
from spectraxgk.config import TEMBaseCase, TimeConfig
from spectraxgk.diffrax_integrators import integrate_linear_diffrax_streaming
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import (
    LinearParams,
    LinearTerms,
    build_linear_cache,
    integrate_linear,
    integrate_linear_diagnostics,
    linear_terms_to_term_config,
)
from spectraxgk.linear_krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.runners import integrate_linear_from_config
from spectraxgk.terms.assembly import compute_fields_cached


def run_tem_linear(
    ky_target: float = 0.3,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: TEMBaseCase | None = None,
    time_cfg: TimeConfig | None = None,
    solver: str = "krylov",
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
    mode_method: str = "project",
    fit_signal: str = "phi",
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    init_species_index: int = 1,
    density_species_index: int = 1,
    diagnostic_norm: str = "none",
    show_progress: bool = False,
) -> LinearRunResult:
    """Run the TEM benchmark and extract growth rate."""

    cfg = cfg or TEMBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=TEM_OMEGA_D_SCALE,
            omega_star_scale=TEM_OMEGA_STAR_SCALE,
            rho_star=TEM_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=Nm,
        )
    if terms is None:
        terms = LinearTerms(bpar=0.0)

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))

    ns = 2
    if init_species_index < 0 or init_species_index >= ns:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= ns:
        raise ValueError("density_species_index out of range for kinetic species")
    G0 = np.zeros(
        (ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
    )
    G0_single = _build_initial_condition(
        grid,
        geom,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=cfg.init,
    )
    G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)

    G0_jax = jnp.asarray(G0)
    if solver.lower() == "krylov":
        krylov_cfg = krylov_cfg or TEM_KRYLOV_DEFAULT
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        eig, vec = dominant_eigenpair(
            G0_jax,
            cache,
            params,
            terms=terms,
            krylov_dim=krylov_cfg.krylov_dim,
            restarts=krylov_cfg.restarts,
            omega_min_factor=krylov_cfg.omega_min_factor,
            omega_target_factor=krylov_cfg.omega_target_factor,
            omega_cap_factor=krylov_cfg.omega_cap_factor,
            omega_sign=krylov_cfg.omega_sign,
            method=krylov_cfg.method,
            power_iters=krylov_cfg.power_iters,
            power_dt=krylov_cfg.power_dt,
            shift=krylov_cfg.shift,
            shift_source=krylov_cfg.shift_source,
            shift_tol=krylov_cfg.shift_tol,
            shift_maxiter=krylov_cfg.shift_maxiter,
            shift_restart=krylov_cfg.shift_restart,
            shift_solve_method=krylov_cfg.shift_solve_method,
            shift_preconditioner=krylov_cfg.shift_preconditioner,
            shift_selection=krylov_cfg.shift_selection,
            mode_family=krylov_cfg.mode_family,
            fallback_method=krylov_cfg.fallback_method,
            fallback_real_floor=krylov_cfg.fallback_real_floor,
        )
        term_cfg = linear_terms_to_term_config(terms)
        phi = compute_fields_cached(vec, cache, params, terms=term_cfg).phi
        phi_t_np = np.asarray(phi)[None, ...]
        t = np.array([0.0])
        gamma = float(np.real(eig))
        omega = float(-np.imag(eig))
        gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    else:
        if fit_signal not in {"phi", "density"}:
            raise ValueError("fit_signal must be 'phi' or 'density'")
        if time_cfg is not None:
            time_cfg_use = time_cfg
            if sample_stride is not None:
                time_cfg_use = replace(time_cfg, sample_stride=sample_stride)
            dt = float(time_cfg_use.dt)
            steps = int(round(time_cfg_use.t_max / time_cfg_use.dt))
            cache = build_linear_cache(grid, geom, params, Nl, Nm)
            if fit_signal == "density":
                _diag = integrate_linear_diagnostics(
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
                if len(_diag) == 4:
                    _, phi_t, density_t, _ = _diag
                else:
                    _, phi_t, density_t = _diag
            else:
                _, phi_t = integrate_linear_from_config(
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
            cache = build_linear_cache(grid, geom, params, Nl, Nm)
            if fit_signal == "density":
                _diag = integrate_linear_diagnostics(
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
                if len(_diag) == 4:
                    _, phi_t, density_t, _ = _diag
                else:
                    _, phi_t, density_t = _diag
            else:
                _, phi_t = integrate_linear(
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
            density_t_np = np.asarray(density_t)
            signal = extract_mode_time_series(density_t_np, sel, method=mode_method)
        else:
            signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
        if auto_window and tmin is None and tmax is None:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
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
                gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)
            except ValueError:
                gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                    t,
                    signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
        gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)

    return LinearRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )


def run_tem_scan(
    ky_values: np.ndarray,
    Nl: int = 6,
    Nm: int = 12,
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
    method: str = "imex2",
    params: LinearParams | None = None,
    cfg: TEMBaseCase | None = None,
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
    mode_method: str = "project",
    mode_only: bool = True,
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    ky_batch: int = 4,
    fixed_batch_shape: bool = True,
    streaming_fit: bool = True,
    streaming_amp_floor: float = 1.0e-30,
    init_species_index: int = 1,
    density_species_index: int = 1,
    diagnostic_norm: str = "none",
    show_progress: bool = False,
) -> LinearScanResult:
    """Run the TEM benchmark for a list of ky values.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    cfg = cfg or TEMBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=TEM_OMEGA_D_SCALE,
            omega_star_scale=TEM_OMEGA_STAR_SCALE,
            rho_star=TEM_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=Nm,
        )
    if terms is None:
        terms = LinearTerms(bpar=0.0)
    solver_key = normalize_solver_key(solver)
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
    sel: ModeSelection | ModeSelectionBatch

    if init_species_index < 0 or init_species_index >= 2:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= 2:
        raise ValueError("density_species_index out of range for kinetic species")

    for batch_start, ky_slice, valid_count in ky_iter:
        if use_batch:
            ky_indices = [
                select_ky_index(np.asarray(grid_full.ky), float(ky)) for ky in ky_slice
            ]
            grid = select_ky_grid(grid_full, ky_indices)
            sel_indices = np.arange(len(ky_indices), dtype=int)
            sel = ModeSelectionBatch(sel_indices, 0, _midplane_index(grid))
            dt_i = float(dt)
            steps_i = int(steps)
        else:
            ky_indices = [select_ky_index(np.asarray(grid_full.ky), float(ky_slice[0]))]
            grid = select_ky_grid(grid_full, ky_indices[0])
            sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
            dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
            steps_i = (
                int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)
            )

        ns = 2
        G0 = np.zeros(
            (ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        )
        G0_single = _build_initial_condition(
            grid,
            geom,
            ky_index=np.arange(len(ky_indices), dtype=int),
            kx_index=0,
            Nl=Nl,
            Nm=Nm,
            init_cfg=cfg.init,
        )
        G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)

        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        G0_jax = jnp.asarray(G0)
        if solver_key == "krylov":
            cfg_use = krylov_cfg or TEM_KRYLOV_DEFAULT
            eig, _vec = dominant_eigenpair(
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
            gamma = float(np.real(eig))
            omega = float(-np.imag(eig))
            gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_slice[0]))
            continue

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
                mode_z_index=_midplane_index(grid),
                mode_method=mode_method,
                amp_floor=streaming_amp_floor,
                return_state=False,
            )
            gamma_arr = np.asarray(gamma_vals)
            omega_arr = np.asarray(omega_vals)
            for local_idx in range(valid_count):
                ky_val = ky_slice[local_idx]
                gammas.append(float(gamma_arr[local_idx]))
                omegas.append(float(omega_arr[local_idx]))
                ky_out.append(float(ky_val))
            continue

        if time_cfg_i is not None:
            _, phi_t = integrate_linear_from_config(
                G0_jax,
                grid,
                geom,
                params_use,
                time_cfg_i,
                cache=cache,
                terms=terms,
                save_mode=sel if mode_only else None,
                mode_method=mode_method,
            )
            stride = time_cfg_i.sample_stride
        else:
            stride = 1 if sample_stride is None else int(sample_stride)
            _, phi_t = integrate_linear(
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
            )

        phi_t_np = np.asarray(phi_t)
        for local_idx in range(valid_count):
            ky_val = ky_slice[local_idx]
            if mode_only and phi_t_np.ndim <= 2:
                signal = _extract_mode_only_signal(phi_t_np, local_idx=local_idx)
            else:
                sel_local = ModeSelection(
                    ky_index=local_idx, kx_index=0, z_index=_midplane_index(grid)
                )
                signal = extract_mode_time_series(
                    phi_t_np, sel_local, method=mode_method
                )
            gamma, omega = _fit_signal(signal, batch_start + local_idx, dt_i, stride)
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_val))
    return LinearScanResult(
        ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas)
    )
