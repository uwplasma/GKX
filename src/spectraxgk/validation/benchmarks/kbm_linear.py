"""KBM single-ky linear benchmark runner."""

# ruff: noqa: F401

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    windowed_growth_rate_from_omega_series,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.validation.benchmarks.defaults import (
    KBM_KRYLOV_DEFAULT,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
)
from spectraxgk.validation.benchmarks.batching import _resolve_streaming_window
from spectraxgk.validation.benchmarks.fit_signals import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.initialization import _build_initial_condition
from spectraxgk.validation.benchmarks.reference import LinearRunResult, LinearScanResult
from spectraxgk.validation.benchmarks.solver_policy import (
    _kbm_use_multi_target_krylov,
    _midplane_index,
    select_kbm_solver_auto,
)
from spectraxgk.validation.benchmarks.species import (
    _linked_boundary_end_damping,
    _two_species_params,
)
from spectraxgk.validation.benchmarks.scan import (
    ScanFitWindowPolicy,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    indexed_scan_value,
    normalize_fit_signal,
    normalize_solver_key,
    scan_window_valid,
)
from spectraxgk.config import KBMBaseCase, TimeConfig, resolve_cfl_fac
from spectraxgk.solvers.time.diffrax import (
    integrate_linear_diffrax_streaming,
)
from spectraxgk.geometry import (
    SAlphaGeometry,
    apply_geometry_grid_defaults,
    build_flux_tube_geometry,
)
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.solvers.time.explicit import (
    ExplicitTimeConfig,
    integrate_linear_explicit_diagnostics,
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
from spectraxgk.validation.benchmarks import kbm_linear_paths as _paths


def run_kbm_linear(
    ky_target: float = 0.3,
    *,
    beta_value: float | None = None,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "imex2",
    params: LinearParams | None = None,
    cfg: KBMBaseCase | None = None,
    time_cfg: TimeConfig | None = None,
    solver: str = "auto",
    krylov_cfg: KrylovConfig | None = None,
    kbm_target_factors: Sequence[float] | None = (0.7, 1.5),
    kbm_beta_transition: float | None = None,
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
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    fit_signal: str = "auto",
    streaming_fit: bool = False,
    init_species_index: int = 1,
    density_species_index: int = 1,
    diagnostic_norm: str = "none",
    fapar_override: float | None = None,
    apar_beta_scale: float | None = None,
    ampere_g0_scale: float | None = None,
    bpar_beta_scale: float | None = None,
    reference_aligned: bool | None = True,
    gx_reference: bool | None = None,
    show_progress: bool = False,
) -> LinearRunResult:
    """Run a single linear KBM point and return the stored field history."""

    cfg_in = cfg or KBMBaseCase()
    beta_use = float(cfg_in.model.beta) if beta_value is None else float(beta_value)
    cfg_use = replace(cfg_in, model=replace(cfg_in.model, beta=beta_use))
    geom = build_flux_tube_geometry(cfg_use.geometry)
    grid_full = build_spectral_grid(apply_geometry_grid_defaults(geom, cfg_use.grid))
    if terms is None:
        terms = LinearTerms(bpar=0.0)
    if gx_reference is not None:
        reference_aligned = gx_reference
    reference_aligned_use = bool(
        True if reference_aligned is None else reference_aligned
    )
    if reference_aligned_use and diagnostic_norm == "none":
        diagnostic_norm = "rho_star"
    damp_ends_amp, damp_ends_widthfrac = _linked_boundary_end_damping(
        reference_aligned_use
    )

    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")

    if init_species_index < 0 or init_species_index >= 2:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= 2:
        raise ValueError("density_species_index out of range for kinetic species")

    if params is None:
        params = _two_species_params(
            cfg_use.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KBM_OMEGA_D_SCALE,
            omega_star_scale=KBM_OMEGA_STAR_SCALE,
            rho_star=KBM_RHO_STAR,
            beta_override=beta_use,
            fapar_override=fapar_override,
            apar_beta_scale=apar_beta_scale,
            ampere_g0_scale=ampere_g0_scale,
            bpar_beta_scale=bpar_beta_scale,
            damp_ends_amp=damp_ends_amp,
            damp_ends_widthfrac=damp_ends_widthfrac,
            nhermite=Nm,
        )

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    G0 = np.zeros(
        (2, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
    )
    G0_single = _build_initial_condition(
        grid,
        geom,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=cfg_use.init,
    )
    G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)
    G0_jax = jnp.asarray(G0)

    solver_key = select_kbm_solver_auto(
        solver,
        ky_target=float(ky_target),
        reference_aligned=reference_aligned_use,
    )
    _paths.sync_path_hooks(globals())

    def _fit_with_window(signal: np.ndarray, t_arr: np.ndarray) -> tuple[float, float]:
        use_auto = auto_window and tmin is None and tmax is None
        if not use_auto and not scan_window_valid(t_arr, tmin, tmax):
            use_auto = True
        if use_auto:
            gamma_val, omega_val, _tmin, _tmax = fit_growth_rate_auto(
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
            try:
                gamma_val, omega_val = fit_growth_rate(
                    t_arr, signal, tmin=tmin, tmax=tmax
                )
            except ValueError:
                gamma_val, omega_val, _tmin, _tmax = fit_growth_rate_auto(
                    t_arr,
                    signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
        return gamma_val, omega_val

    if solver_key == "explicit_time":
        return _paths.run_kbm_explicit_time_path(
            G0_jax=G0_jax,
            grid=grid,
            cache=cache,
            params=params,
            geom=geom,
            terms=terms,
            sel=sel,
            ky_target=ky_target,
            dt=dt,
            steps=steps,
            time_cfg=time_cfg,
            sample_stride=sample_stride,
            mode_method=mode_method,
            diagnostic_norm=diagnostic_norm,
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

    if solver_key == "krylov":
        return _paths.run_kbm_krylov_path(
            G0_jax=G0_jax,
            cache=cache,
            params=params,
            terms=terms,
            sel=sel,
            ky_target=ky_target,
            beta_use=beta_use,
            cfg_use=cfg_use,
            krylov_cfg=krylov_cfg,
            kbm_target_factors=kbm_target_factors,
            kbm_beta_transition=kbm_beta_transition,
            diagnostic_norm=diagnostic_norm,
        )

    stride = 1 if sample_stride is None else int(sample_stride)
    time_cfg_use = time_cfg
    if time_cfg_use is not None:
        time_cfg_use = replace(time_cfg_use, dt=dt, t_max=dt * steps)
        if sample_stride is not None:
            time_cfg_use = replace(time_cfg_use, sample_stride=stride)
    params_use = params
    if time_cfg_use is not None:
        stride = int(time_cfg_use.sample_stride)
        if time_cfg_use.use_diffrax:
            save_field = "phi+density" if fit_key in {"density", "auto"} else "phi"
            _, phi_out = integrate_linear_from_config(
                G0_jax,
                grid,
                geom,
                params_use,
                time_cfg_use,
                cache=cache,
                terms=terms,
                save_mode=sel if fit_key == "phi" else None,
                mode_method=mode_method,
                save_field=save_field,
                density_species_index=density_species_index
                if fit_key in {"density", "auto"}
                else None,
            )
            if fit_key in {"density", "auto"}:
                phi_t_np, density_np = (np.asarray(phi_out[0]), np.asarray(phi_out[1]))
            else:
                phi_t_np = np.asarray(phi_out)
                density_np = None
        else:
            if fit_key in {"density", "auto"}:
                diag_out = integrate_linear_diagnostics(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    dt=dt,
                    steps=steps,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=stride,
                    species_index=density_species_index,
                )
                phi_t_np = np.asarray(diag_out[1])
                density_np = None if len(diag_out) <= 2 else np.asarray(diag_out[2])
            else:
                _, phi_out_time = integrate_linear(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    dt=dt,
                    steps=steps,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=stride,
                    show_progress=show_progress,
                )
                phi_t_np = np.asarray(phi_out_time)
                density_np = None
    else:
        if fit_key in {"density", "auto"}:
            diag_out = integrate_linear_diagnostics(
                G0_jax,
                grid,
                geom,
                params_use,
                dt=dt,
                steps=steps,
                method=method,
                cache=cache,
                terms=terms,
                sample_stride=stride,
                species_index=density_species_index,
            )
            phi_t_np = np.asarray(diag_out[1])
            density_np = None if len(diag_out) <= 2 else np.asarray(diag_out[2])
        else:
            _, phi_out_time = integrate_linear(
                G0_jax,
                grid,
                geom,
                params_use,
                dt=dt,
                steps=steps,
                method=method,
                cache=cache,
                terms=terms,
                sample_stride=stride,
            )
            phi_t_np = np.asarray(phi_out_time)
            density_np = None

    if fit_key == "density" and density_np is None:
        density_np = phi_t_np
    if fit_key == "auto":
        signal, _name, gamma, omega = _select_fit_signal_auto(
            np.arange(phi_t_np.shape[0]) * dt * stride,
            phi_t_np,
            density_np,
            sel,
            mode_method=mode_method,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
            max_amp_fraction=0.9,
            window_method="loglinear",
            max_fraction=0.8,
            end_fraction=0.9,
            num_windows=8,
            phase_weight=0.2,
            length_weight=0.05,
            min_r2=0.0,
            late_penalty=0.1,
            min_slope=None,
            min_slope_frac=0.0,
            slope_var_weight=0.0,
        )
        _ = signal
    else:
        signal = _select_fit_signal(
            phi_t_np,
            density_np,
            sel,
            fit_signal=fit_key,
            mode_method=mode_method,
        )
        t_out = np.arange(signal.shape[0]) * dt * stride
        gamma, omega = _fit_with_window(signal, t_out)
    gamma, omega = _normalize_growth_rate(gamma, omega, params_use, diagnostic_norm)
    return LinearRunResult(
        t=np.arange(phi_t_np.shape[0]) * dt * stride,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(ky_target),
        selection=sel,
    )

