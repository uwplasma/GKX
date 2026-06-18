"""ETG linear ky-scan benchmark runner."""

# ruff: noqa: F401

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    ModeSelectionBatch,
    fit_growth_rate,
    fit_growth_rate_auto,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.validation.benchmarks.defaults import (
    ETG_KRYLOV_DEFAULT,
    ETG_OMEGA_D_SCALE,
    ETG_OMEGA_STAR_SCALE,
    ETG_RHO_STAR,
)
from spectraxgk.validation.benchmarks.batching import (
    _iter_ky_batches,
    _resolve_streaming_window,
)
from spectraxgk.validation.benchmarks.fit_signals import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.initialization import _build_initial_condition
from spectraxgk.validation.benchmarks.reference import LinearRunResult, LinearScanResult
from spectraxgk.validation.benchmarks.solver_policy import _midplane_index
from spectraxgk.validation.benchmarks.species import (
    _electron_only_params,
    _two_species_params,
)
from spectraxgk.validation.benchmarks.scan import (
    ScanFitWindowPolicy,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    normalize_fit_signal,
    normalize_solver_key,
    resolve_scan_mode_method,
    scan_window_valid,
    should_use_ky_batch,
)
from spectraxgk.config import ETGBaseCase, TimeConfig
from spectraxgk.solvers.time.diffrax import (
    integrate_linear_diffrax,
    integrate_linear_diffrax_streaming,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
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


from spectraxgk.validation.benchmarks.etg_linear import run_etg_linear
from spectraxgk.validation.benchmarks import etg_scan_paths as _paths

def run_etg_scan(
    ky_values: np.ndarray,
    Nl: int = 6,
    Nm: int = 12,
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
    method: str = "imex2",
    params: LinearParams | None = None,
    cfg: ETGBaseCase | None = None,
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
    ky_batch: int = 4,
    fixed_batch_shape: bool = True,
    streaming_fit: bool = True,
    streaming_amp_floor: float = 1.0e-30,
    gx_growth: bool = False,
    gx_navg_fraction: float = 0.5,
    diagnostic_norm: str = "none",
    show_progress: bool = False,
) -> LinearScanResult:
    """Run an ETG linear benchmark for a list of ky values.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    cfg = cfg or ETGBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        if getattr(cfg.model, "adiabatic_ions", False):
            params = _electron_only_params(
                cfg.model,
                kpar_scale=float(geom.gradpar()),
                omega_d_scale=ETG_OMEGA_D_SCALE,
                omega_star_scale=ETG_OMEGA_STAR_SCALE,
                rho_star=ETG_RHO_STAR,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
                nhermite=Nm,
            )
        else:
            params = _two_species_params(
                cfg.model,
                kpar_scale=float(geom.gradpar()),
                omega_d_scale=ETG_OMEGA_D_SCALE,
                omega_star_scale=ETG_OMEGA_STAR_SCALE,
                rho_star=ETG_RHO_STAR,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
                nhermite=Nm,
            )
    if terms is None:
        # Keep the ETG scan helper on the same electrostatic benchmark contract
        # as the single-ky ETG wrapper and the tracked ETG figure builders.
        terms = LinearTerms(apar=0.0, bpar=0.0, hypercollisions=1.0)
    solver_key = normalize_solver_key(solver)
    fit_key = normalize_fit_signal(fit_signal)
    auto_solver = solver_key == "auto"
    if auto_solver:
        solver_key = "time"
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

    ky_values_arr = np.asarray(ky_values, dtype=float)
    if use_batch:
        ky_iter = _iter_ky_batches(
            ky_values_arr,
            ky_batch=ky_batch,
            fixed_batch_shape=fixed_batch_shape,
        )
    else:
        ky_iter = _iter_ky_batches(ky_values_arr, ky_batch=1, fixed_batch_shape=False)
    prev_vec: jnp.ndarray | None = None
    prev_eig: complex | None = None
    ky_slice: np.ndarray
    ky_indices: list[int]
    sel: ModeSelection | ModeSelectionBatch
    _paths.sync_path_hooks(globals())

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

        charge = np.atleast_1d(np.asarray(params.charge_sign))
        ns = int(charge.size)
        electron_index = int(np.argmin(charge))
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
        G0[electron_index] = np.asarray(G0_single, dtype=np.complex64)

        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        G0_jax = jnp.asarray(G0)
        if solver_key == "krylov":
            gamma, omega, prev_vec, prev_eig = _paths.run_etg_krylov_batch(
                G0_jax=G0_jax,
                cache=cache,
                params=params,
                terms=terms,
                krylov_cfg=krylov_cfg,
                prev_vec=prev_vec,
                prev_eig=prev_eig,
                diagnostic_norm=diagnostic_norm,
            )
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_slice[0]))
            continue

        time_result = _paths.run_etg_time_batch(
            G0_jax=G0_jax,
            grid=grid,
            geom=geom,
            params=params,
            cache=cache,
            terms=terms,
            time_cfg=time_cfg,
            dt_i=dt_i,
            steps_i=steps_i,
            method=method,
            sample_stride=sample_stride,
            fit_key=fit_key,
            need_density=need_density,
            streaming_fit=streaming_fit,
            streaming_amp_floor=streaming_amp_floor,
            mode_method=mode_method,
            mode_only=mode_only,
            sel=sel,
            batch_start=batch_start,
            valid_count=valid_count,
            ky_slice=ky_slice,
            tmin=tmin,
            tmax=tmax,
            start_fraction=start_fraction,
            window_fraction=window_fraction,
            electron_index=electron_index,
            diagnostic_norm=diagnostic_norm,
            show_progress=show_progress,
            gammas=gammas,
            omegas=omegas,
            ky_out=ky_out,
        )
        if time_result.handled:
            continue

        _paths.append_etg_time_fit_results(
            result=time_result,
            ky_slice=ky_slice,
            valid_count=valid_count,
            batch_start=batch_start,
            fit_key=fit_key,
            fit_policy=fit_policy,
            params=params,
            diagnostic_norm=diagnostic_norm,
            mode_method=mode_method,
            mode_only=mode_only,
            mode_z_index=_midplane_index(grid),
            gx_growth=gx_growth,
            gx_navg_fraction=gx_navg_fraction,
            auto_solver=auto_solver,
            require_positive=require_positive,
            cfg=cfg,
            Nl=Nl,
            Nm=Nm,
            dt_i=dt_i,
            steps_i=steps_i,
            method=method,
            krylov_cfg=krylov_cfg,
            show_progress=show_progress,
            gammas=gammas,
            omegas=omegas,
            ky_out=ky_out,
        )
    return LinearScanResult(
        ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas)
    )
