"""KBM fixed-ky beta-scan benchmark runner."""

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
from spectraxgk.validation.benchmarks.kbm_beta_solver_paths import (
    KBMBetaExplicitHooks,
    KBMBetaKrylovHooks,
    fit_kbm_beta_explicit_time_sample,
    solve_kbm_beta_krylov_sample,
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


def run_kbm_beta_scan(
    betas: np.ndarray,
    ky_target: float = 0.3,
    Nl: int = 6,
    Nm: int = 12,
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
    method: str = "imex2",
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
    mode_only: bool = True,
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    fit_signal: str = "auto",
    ky_batch: int = 4,
    fixed_batch_shape: bool = True,
    streaming_fit: bool = True,
    streaming_amp_floor: float = 1.0e-30,
    init_species_index: int = 1,
    density_species_index: int = 1,
    diagnostic_norm: str = "none",
    fapar_override: float | None = None,
    apar_beta_scale: float | None = None,
    ampere_g0_scale: float | None = None,
    bpar_beta_scale: float | None = None,
    reference_aligned: bool | None = True,
) -> LinearScanResult:
    """Run a KBM beta scan at fixed ky.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    cfg = cfg or KBMBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if terms is None:
        terms = LinearTerms(bpar=0.0)
    reference_aligned_use = bool(
        True if reference_aligned is None else reference_aligned
    )
    if reference_aligned_use and diagnostic_norm == "none":
        diagnostic_norm = "rho_star"
    damp_ends_amp, damp_ends_widthfrac = _linked_boundary_end_damping(
        reference_aligned_use
    )

    solver_key = normalize_solver_key(solver)
    fit_key = normalize_fit_signal(fit_signal)
    streaming_fit, mode_only = apply_auto_fit_scan_policy(
        fit_key, streaming_fit=streaming_fit, mode_only=mode_only
    )

    krylov_cfg_use = krylov_cfg or KBM_KRYLOV_DEFAULT
    use_continuation = bool(getattr(krylov_cfg_use, "continuation", False))
    prev_vec = None
    prev_eig = None

    gammas = []
    omegas = []
    beta_out = []
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))

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
    explicit_hooks = KBMBetaExplicitHooks(
        integrate_linear_explicit_diagnostics=integrate_linear_explicit_diagnostics,
        instantaneous_growth_rate_from_phi=instantaneous_growth_rate_from_phi,
        windowed_growth_rate_from_omega_series=windowed_growth_rate_from_omega_series,
        extract_mode_time_series=extract_mode_time_series,
        fit_growth_rate_auto=fit_growth_rate_auto,
        normalize_growth_rate=_normalize_growth_rate,
        resolve_cfl_fac=resolve_cfl_fac,
    )
    krylov_hooks = KBMBetaKrylovHooks(
        dominant_eigenpair=dominant_eigenpair,
        use_multi_target_krylov=_kbm_use_multi_target_krylov,
        normalize_growth_rate=_normalize_growth_rate,
    )

    if init_species_index < 0 or init_species_index >= 2:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= 2:
        raise ValueError("density_species_index out of range for kinetic species")

    def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
        if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
            return False
        if require_positive and gamma_val <= 0.0:
            return False
        return True

    for i, beta in enumerate(betas):
        dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KBM_OMEGA_D_SCALE,
            omega_star_scale=KBM_OMEGA_STAR_SCALE,
            rho_star=KBM_RHO_STAR,
            beta_override=float(beta),
            fapar_override=fapar_override,
            apar_beta_scale=apar_beta_scale,
            ampere_g0_scale=ampere_g0_scale,
            bpar_beta_scale=bpar_beta_scale,
            damp_ends_amp=damp_ends_amp,
            damp_ends_widthfrac=damp_ends_widthfrac,
            nhermite=Nm,
        )
        cache = build_linear_cache(grid, geom, params, Nl, Nm)

        ns = 2
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
        solver_use = select_kbm_solver_auto(
            solver_key,
            ky_target=ky_target,
            reference_aligned=reference_aligned_use,
        )

        if solver_use == "explicit_time":
            gamma, omega = fit_kbm_beta_explicit_time_sample(
                G0_jax=G0_jax,
                grid=grid,
                cache=cache,
                params=params,
                geom=geom,
                terms=terms,
                dt_i=dt_i,
                steps_i=steps_i,
                time_cfg=time_cfg,
                sample_stride=sample_stride,
                mode_method=mode_method,
                sel=sel,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
                diagnostic_norm=diagnostic_norm,
                hooks=explicit_hooks,
            )
        elif solver_use == "krylov":
            krylov_result = solve_kbm_beta_krylov_sample(
                beta=float(beta),
                cfg=cfg,
                G0_jax=G0_jax,
                cache=cache,
                params=params,
                terms=terms,
                solver_key=solver_key,
                krylov_cfg_use=krylov_cfg_use,
                use_continuation=use_continuation,
                prev_vec=prev_vec,
                prev_eig=prev_eig,
                kbm_target_factors=kbm_target_factors,
                kbm_beta_transition=kbm_beta_transition,
                diagnostic_norm=diagnostic_norm,
                is_valid_growth=_is_valid_growth,
                hooks=krylov_hooks,
            )
            gamma = krylov_result.gamma
            omega = krylov_result.omega
            if krylov_result.fallback_to_time:
                solver_use = "time"
            else:
                prev_vec = krylov_result.prev_vec
                prev_eig = krylov_result.prev_eig

        if solver_use not in {"krylov", "explicit_time"}:
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
                    indexed_float_value(tmin, i),
                    indexed_float_value(tmax, i),
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
                    fit_signal=fit_key,
                    mode_ky_indices=[0],
                    mode_kx_index=0,
                    mode_z_index=_midplane_index(grid),
                    mode_method=mode_method,
                    amp_floor=streaming_amp_floor,
                    density_species_index=density_species_index
                    if fit_key == "density"
                    else None,
                    return_state=False,
                )
                gamma = float(np.asarray(gamma_vals)[0])
                omega = float(np.asarray(omega_vals)[0])
                gamma, omega = _normalize_growth_rate(
                    gamma, omega, params_use, diagnostic_norm
                )
            else:
                if time_cfg_i is not None:
                    stride = time_cfg_i.sample_stride
                    if time_cfg_i.use_diffrax:
                        save_mode_method = (
                            mode_method
                            if mode_method in {"z_index", "max"}
                            else "z_index"
                        )
                        _, phi_t = integrate_linear_from_config(
                            G0_jax,
                            grid,
                            geom,
                            params_use,
                            time_cfg_i,
                            cache=cache,
                            terms=terms,
                            save_mode=sel if mode_only else None,
                            mode_method=save_mode_method,
                            save_field="phi+density"
                            if fit_key == "auto"
                            else ("density" if fit_key == "density" else "phi"),
                            density_species_index=density_species_index
                            if fit_key in {"density", "auto"}
                            else None,
                        )
                        if fit_key == "auto":
                            phi_t, density_t = phi_t
                        else:
                            density_t = None
                    else:
                        if fit_key in {"density", "auto"}:
                            diag_out = integrate_linear_diagnostics(
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
                                species_index=density_species_index,
                            )
                            phi_t = diag_out[1]
                            density_t = diag_out[2] if len(diag_out) > 2 else None
                        else:
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
                            density_t = None
                else:
                    stride = 1 if sample_stride is None else int(sample_stride)
                    if fit_key in {"density", "auto"}:
                        diag_out = integrate_linear_diagnostics(
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
                            species_index=density_species_index,
                        )
                        phi_t = diag_out[1]
                        density_t = diag_out[2] if len(diag_out) > 2 else None
                    else:
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
                        density_t = None

                phi_t_np = np.asarray(phi_t)
                density_np = None if density_t is None else np.asarray(density_t)
                if fit_key == "density" and density_np is None:
                    density_np = phi_t_np
                if fit_key == "auto":
                    signal, _name, gamma, omega = _select_fit_signal_auto(
                        np.arange(phi_t_np.shape[0]) * dt_i * stride,
                        phi_t_np,
                        density_np,
                        sel,
                        mode_method=mode_method,
                        tmin=indexed_float_value(tmin, i),
                        tmax=indexed_float_value(tmax, i),
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
                    gamma, omega = _normalize_growth_rate(
                        gamma, omega, params_use, diagnostic_norm
                    )
                    gammas.append(gamma)
                    omegas.append(omega)
                    beta_out.append(float(beta))
                    continue

                if (
                    mode_only
                    and fit_key == "density"
                    and density_np is not None
                    and density_np.ndim <= 3
                ):
                    signal = _extract_mode_only_signal(
                        density_np,
                        local_idx=0,
                        species_index=density_species_index,
                    )
                elif mode_only and phi_t_np.ndim <= 2:
                    signal = _extract_mode_only_signal(phi_t_np, local_idx=0)
                else:
                    signal = _select_fit_signal(
                        phi_t_np,
                        density_np,
                        sel,
                        fit_signal=fit_key,
                        mode_method=mode_method,
                    )
                gamma, omega = fit_policy.fit_signal(
                    signal,
                    idx=i,
                    dt=dt_i,
                    stride=stride,
                    params=params_use,
                    diagnostic_norm=diagnostic_norm,
                )

        gammas.append(gamma)
        omegas.append(omega)
        beta_out.append(float(beta))

    return LinearScanResult(
        ky=np.array(beta_out), gamma=np.array(gammas), omega=np.array(omegas)
    )
