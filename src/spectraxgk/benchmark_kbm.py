"""KBM benchmark runners behind the public :mod:`spectraxgk.benchmarks` facade."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from spectraxgk.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    windowed_growth_rate_from_omega_series,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.benchmark_defaults import (
    KBM_KRYLOV_DEFAULT,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
)
from spectraxgk.benchmark_helpers import (
    LinearRunResult,
    LinearScanResult,
    _build_initial_condition,
    _extract_mode_only_signal,
    _linked_boundary_end_damping,
    _kbm_use_multi_target_krylov,
    _midplane_index,
    _normalize_growth_rate,
    _resolve_streaming_window,
    _select_fit_signal,
    _select_fit_signal_auto,
    _two_species_params,
    select_kbm_solver_auto,
)
from spectraxgk.benchmark_scan import (
    ScanFitWindowPolicy,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    indexed_scan_value,
    normalize_fit_signal,
    normalize_solver_key,
    scan_window_valid,
)
from spectraxgk.config import KBMBaseCase, TimeConfig, resolve_cfl_fac
from spectraxgk.diffrax_integrators import (
    integrate_linear_diffrax_streaming,
)
from spectraxgk.geometry import (
    SAlphaGeometry,
    apply_geometry_grid_defaults,
    build_flux_tube_geometry,
)
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.explicit_time_integrators import (
    ExplicitTimeConfig,
    integrate_linear_explicit_diagnostics,
)
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
    gx_reference: bool | None = None,
) -> LinearScanResult:
    """Run a KBM beta scan at fixed ky.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    cfg = cfg or KBMBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
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
            explicit_mode_method = (
                mode_method if mode_method in {"z_index", "max"} else "z_index"
            )
            explicit_time_cfg = ExplicitTimeConfig(
                dt=dt_i,
                t_max=dt_i * steps_i,
                sample_stride=max(int(sample_stride or 1), 1),
                fixed_dt=bool(time_cfg.fixed_dt) if time_cfg is not None else False,
                use_dealias_mask=bool(getattr(time_cfg, "use_dealias_mask", False))
                if time_cfg is not None
                else False,
                dt_min=float(time_cfg.dt_min) if time_cfg is not None else 1.0e-7,
                dt_max=float(time_cfg.dt_max)
                if (time_cfg is not None and time_cfg.dt_max is not None)
                else None,
                cfl=float(time_cfg.cfl) if time_cfg is not None else 0.9,
                cfl_fac=(
                    resolve_cfl_fac(str(time_cfg.method), time_cfg.cfl_fac)
                    if time_cfg is not None
                    else float(ExplicitTimeConfig.cfl_fac)
                ),
            )
            t_arr, _phi_t, gamma_t, omega_t, _gx_diag = (
                integrate_linear_explicit_diagnostics(
                    G0_jax,
                    grid,
                    cache,
                    params,
                    geom,
                    explicit_time_cfg,
                    terms=terms,
                    mode_method=explicit_mode_method,
                    z_index=sel.z_index,
                    jit=True,
                )
            )
            if t_arr.size > 1:
                phi_np = np.asarray(_phi_t)
                t_np = np.asarray(t_arr, dtype=float)
                if mode_method in {"z_index", "max"}:
                    try:
                        gamma, omega, _g_t, _o_t, _t_mid = (
                            instantaneous_growth_rate_from_phi(
                                phi_np,
                                t_np,
                                sel,
                                navg_fraction=0.5,
                                mode_method=mode_method,
                            )
                        )
                    except ValueError:
                        try:
                            gamma, omega, _g_t, _o_t = (
                                windowed_growth_rate_from_omega_series(
                                    np.asarray(gamma_t),
                                    np.asarray(omega_t),
                                    sel,
                                    navg_fraction=0.5,
                                )
                            )
                        except ValueError:
                            signal = extract_mode_time_series(
                                phi_np, sel, method=mode_method
                            )
                            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                                t_np,
                                signal,
                                window_method="fixed",
                                window_fraction=window_fraction,
                                min_points=min_points,
                                start_fraction=start_fraction,
                                growth_weight=growth_weight,
                                require_positive=require_positive,
                                min_amp_fraction=min_amp_fraction,
                            )
                else:
                    signal = extract_mode_time_series(phi_np, sel, method=mode_method)
                    gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                        t_np,
                        signal,
                        window_method="fixed",
                        window_fraction=window_fraction,
                        min_points=min_points,
                        start_fraction=start_fraction,
                        growth_weight=growth_weight,
                        require_positive=require_positive,
                        min_amp_fraction=min_amp_fraction,
                    )
            else:
                gamma = float("nan")
                omega = float("nan")
            gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
        elif solver_use == "krylov":
            shift_val = krylov_cfg_use.shift
            shift_selection = krylov_cfg_use.shift_selection
            if use_continuation and prev_eig is not None:
                shift_val = complex(np.asarray(prev_eig))

            targets: Sequence[float] | None = (
                kbm_target_factors if kbm_target_factors else None
            )
            use_multi_target = _kbm_use_multi_target_krylov(
                krylov_cfg_use,
                targets,
                shift=shift_val,
            )
            if use_multi_target:
                assert targets is not None
                beta_transition = (
                    float(cfg.model.beta)
                    if kbm_beta_transition is None
                    else float(kbm_beta_transition)
                )
                eig_candidates = []
                vec_candidates = []
                for target in targets:
                    eig_i, vec_i = dominant_eigenpair(
                        G0_jax,
                        cache,
                        params,
                        terms=terms,
                        v_ref=None,
                        select_overlap=False,
                        krylov_dim=krylov_cfg_use.krylov_dim,
                        restarts=krylov_cfg_use.restarts,
                        omega_min_factor=krylov_cfg_use.omega_min_factor,
                        omega_target_factor=float(target),
                        omega_cap_factor=krylov_cfg_use.omega_cap_factor,
                        omega_sign=krylov_cfg_use.omega_sign,
                        method=krylov_cfg_use.method,
                        power_iters=krylov_cfg_use.power_iters,
                        power_dt=krylov_cfg_use.power_dt,
                        shift=None,
                        shift_source="target",
                        shift_tol=krylov_cfg_use.shift_tol,
                        shift_maxiter=krylov_cfg_use.shift_maxiter,
                        shift_restart=krylov_cfg_use.shift_restart,
                        shift_solve_method=krylov_cfg_use.shift_solve_method,
                        shift_preconditioner=krylov_cfg_use.shift_preconditioner,
                        shift_selection="targeted",
                        mode_family=krylov_cfg_use.mode_family,
                        fallback_method=krylov_cfg_use.fallback_method,
                        fallback_real_floor=krylov_cfg_use.fallback_real_floor,
                    )
                    eig_candidates.append(eig_i)
                    vec_candidates.append(vec_i)
                if len(eig_candidates) >= 2 and np.isfinite(beta_transition):
                    pick_high = float(beta) >= beta_transition
                    idx = 1 if pick_high else 0
                    eig = eig_candidates[idx]
                    _vec = vec_candidates[idx]
                else:
                    eig_arr = np.asarray(
                        [complex(np.asarray(e)) for e in eig_candidates]
                    )
                    growth = np.real(eig_arr)
                    if np.all(~np.isfinite(growth)):
                        eig = eig_candidates[0]
                        _vec = vec_candidates[0]
                    else:
                        idx = int(
                            np.nanargmax(np.where(np.isfinite(growth), growth, -np.inf))
                        )
                        eig = eig_candidates[idx]
                        _vec = vec_candidates[idx]
            else:
                eig, _vec = dominant_eigenpair(
                    G0_jax,
                    cache,
                    params,
                    terms=terms,
                    v_ref=prev_vec,
                    select_overlap=use_continuation,
                    krylov_dim=krylov_cfg_use.krylov_dim,
                    restarts=krylov_cfg_use.restarts,
                    omega_min_factor=krylov_cfg_use.omega_min_factor,
                    omega_target_factor=krylov_cfg_use.omega_target_factor,
                    omega_cap_factor=krylov_cfg_use.omega_cap_factor,
                    omega_sign=krylov_cfg_use.omega_sign,
                    method=krylov_cfg_use.method,
                    power_iters=krylov_cfg_use.power_iters,
                    power_dt=krylov_cfg_use.power_dt,
                    shift=shift_val,
                    shift_source=krylov_cfg_use.shift_source,
                    shift_tol=krylov_cfg_use.shift_tol,
                    shift_maxiter=krylov_cfg_use.shift_maxiter,
                    shift_restart=krylov_cfg_use.shift_restart,
                    shift_solve_method=krylov_cfg_use.shift_solve_method,
                    shift_preconditioner=krylov_cfg_use.shift_preconditioner,
                    shift_selection=shift_selection,
                    mode_family=krylov_cfg_use.mode_family,
                    fallback_method=krylov_cfg_use.fallback_method,
                    fallback_real_floor=krylov_cfg_use.fallback_real_floor,
                )
            gamma = float(np.real(eig))
            omega = float(-np.imag(eig))
            if krylov_cfg_use.omega_sign != 0:
                omega = float(np.sign(krylov_cfg_use.omega_sign)) * abs(omega)
            gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
            if solver_key == "auto" and not _is_valid_growth(gamma, omega):
                solver_use = "time"
            elif use_continuation:
                prev_vec = _vec
                prev_eig = eig

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
    krylov_cfg_use = krylov_cfg or KBM_KRYLOV_DEFAULT

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
        explicit_mode_method = (
            mode_method if mode_method in {"z_index", "max"} else "z_index"
        )
        explicit_time_cfg = ExplicitTimeConfig(
            dt=dt,
            t_max=dt * steps,
            sample_stride=max(int(sample_stride or 1), 1),
            fixed_dt=bool(time_cfg.fixed_dt) if time_cfg is not None else False,
            use_dealias_mask=bool(getattr(time_cfg, "use_dealias_mask", False))
            if time_cfg is not None
            else False,
            dt_min=float(time_cfg.dt_min) if time_cfg is not None else 1.0e-7,
            dt_max=float(time_cfg.dt_max)
            if (time_cfg is not None and time_cfg.dt_max is not None)
            else None,
            cfl=float(time_cfg.cfl) if time_cfg is not None else 0.9,
            cfl_fac=(
                resolve_cfl_fac(str(time_cfg.method), time_cfg.cfl_fac)
                if time_cfg is not None
                else float(ExplicitTimeConfig.cfl_fac)
            ),
        )
        t_arr, phi_t, gamma_t, omega_t, _gx_diag = (
            integrate_linear_explicit_diagnostics(
                G0_jax,
                grid,
                cache,
                params,
                geom,
                explicit_time_cfg,
                terms=terms,
                mode_method=explicit_mode_method,
                z_index=sel.z_index,
                jit=True,
            )
        )
        t_out = np.asarray(t_arr, dtype=float)
        phi_t_np = np.asarray(phi_t)
        if t_out.size > 1:
            if mode_method in {"z_index", "max"}:
                try:
                    gamma, omega, _g_t, _o_t, _t_mid = (
                        instantaneous_growth_rate_from_phi(
                            phi_t_np,
                            t_out,
                            sel,
                            navg_fraction=0.5,
                            mode_method=mode_method,
                        )
                    )
                except ValueError:
                    try:
                        gamma, omega, _g_t, _o_t = (
                            windowed_growth_rate_from_omega_series(
                                np.asarray(gamma_t),
                                np.asarray(omega_t),
                                sel,
                                navg_fraction=0.5,
                            )
                        )
                    except ValueError:
                        signal = extract_mode_time_series(
                            phi_t_np, sel, method=mode_method
                        )
                        gamma, omega = _fit_with_window(signal, t_out)
            else:
                signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
                if auto_window and tmin is None and tmax is None:
                    gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                        t_out,
                        signal,
                        window_method="fixed",
                        window_fraction=window_fraction,
                        min_points=min_points,
                        start_fraction=start_fraction,
                        growth_weight=growth_weight,
                        require_positive=require_positive,
                        min_amp_fraction=min_amp_fraction,
                    )
                else:
                    gamma, omega = _fit_with_window(signal, t_out)
        else:
            gamma = float("nan")
            omega = float("nan")
        gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
        return LinearRunResult(
            t=t_out,
            phi_t=phi_t_np,
            gamma=gamma,
            omega=omega,
            ky=float(ky_target),
            selection=sel,
            gamma_t=np.asarray(gamma_t),
            omega_t=np.asarray(omega_t),
        )

    if solver_key == "krylov":
        shift_val = krylov_cfg_use.shift
        targets: Sequence[float] | None = (
            kbm_target_factors if kbm_target_factors else None
        )
        use_multi_target = _kbm_use_multi_target_krylov(
            krylov_cfg_use,
            targets,
            shift=shift_val,
        )
        if use_multi_target:
            assert targets is not None
            beta_transition = (
                float(cfg_use.model.beta)
                if kbm_beta_transition is None
                else float(kbm_beta_transition)
            )
            eig_candidates = []
            vec_candidates = []
            for target in targets:
                eig_i, vec_i = dominant_eigenpair(
                    G0_jax,
                    cache,
                    params,
                    terms=terms,
                    v_ref=None,
                    select_overlap=False,
                    krylov_dim=krylov_cfg_use.krylov_dim,
                    restarts=krylov_cfg_use.restarts,
                    omega_min_factor=krylov_cfg_use.omega_min_factor,
                    omega_target_factor=float(target),
                    omega_cap_factor=krylov_cfg_use.omega_cap_factor,
                    omega_sign=krylov_cfg_use.omega_sign,
                    method=krylov_cfg_use.method,
                    power_iters=krylov_cfg_use.power_iters,
                    power_dt=krylov_cfg_use.power_dt,
                    shift=None,
                    shift_source="target",
                    shift_tol=krylov_cfg_use.shift_tol,
                    shift_maxiter=krylov_cfg_use.shift_maxiter,
                    shift_restart=krylov_cfg_use.shift_restart,
                    shift_solve_method=krylov_cfg_use.shift_solve_method,
                    shift_preconditioner=krylov_cfg_use.shift_preconditioner,
                    shift_selection="targeted",
                    mode_family=krylov_cfg_use.mode_family,
                    fallback_method=krylov_cfg_use.fallback_method,
                    fallback_real_floor=krylov_cfg_use.fallback_real_floor,
                )
                eig_candidates.append(eig_i)
                vec_candidates.append(vec_i)
            if len(eig_candidates) >= 2 and np.isfinite(beta_transition):
                idx = 1 if beta_use >= beta_transition else 0
            else:
                eig_arr = np.asarray([complex(np.asarray(e)) for e in eig_candidates])
                growth = np.real(eig_arr)
                idx = (
                    0
                    if np.all(~np.isfinite(growth))
                    else int(
                        np.nanargmax(np.where(np.isfinite(growth), growth, -np.inf))
                    )
                )
            eig = eig_candidates[idx]
            vec = vec_candidates[idx]
        else:
            eig, vec = dominant_eigenpair(
                G0_jax,
                cache,
                params,
                terms=terms,
                v_ref=None,
                select_overlap=False,
                krylov_dim=krylov_cfg_use.krylov_dim,
                restarts=krylov_cfg_use.restarts,
                omega_min_factor=krylov_cfg_use.omega_min_factor,
                omega_target_factor=krylov_cfg_use.omega_target_factor,
                omega_cap_factor=krylov_cfg_use.omega_cap_factor,
                omega_sign=krylov_cfg_use.omega_sign,
                method=krylov_cfg_use.method,
                power_iters=krylov_cfg_use.power_iters,
                power_dt=krylov_cfg_use.power_dt,
                shift=shift_val,
                shift_source=krylov_cfg_use.shift_source,
                shift_tol=krylov_cfg_use.shift_tol,
                shift_maxiter=krylov_cfg_use.shift_maxiter,
                shift_restart=krylov_cfg_use.shift_restart,
                shift_solve_method=krylov_cfg_use.shift_solve_method,
                shift_preconditioner=krylov_cfg_use.shift_preconditioner,
                shift_selection=krylov_cfg_use.shift_selection,
                mode_family=krylov_cfg_use.mode_family,
                fallback_method=krylov_cfg_use.fallback_method,
                fallback_real_floor=krylov_cfg_use.fallback_real_floor,
            )
        gamma = float(np.real(eig))
        omega = float(-np.imag(eig))
        if krylov_cfg_use.omega_sign != 0:
            omega = float(np.sign(krylov_cfg_use.omega_sign)) * abs(omega)
        gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
        term_cfg = linear_terms_to_term_config(terms)
        phi = compute_fields_cached(vec, cache, params, terms=term_cfg).phi
        return LinearRunResult(
            t=np.array([0.0], dtype=float),
            phi_t=np.asarray(phi)[None, ...],
            gamma=gamma,
            omega=omega,
            ky=float(ky_target),
            selection=sel,
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


def run_kbm_scan(
    ky_values: np.ndarray,
    *,
    beta_value: float | None = None,
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
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
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
    gx_reference: bool | None = None,
) -> LinearScanResult:
    """Run a KBM ky scan at fixed beta.

    This is a thin wrapper over :func:`run_kbm_beta_scan` used for
    reference-comparison workflows where the external benchmark is a ky scan
    at fixed beta.
    """

    cfg_in = cfg or KBMBaseCase()
    if beta_value is None:
        beta_use = float(cfg_in.model.beta)
    else:
        beta_use = float(beta_value)
    cfg_use = replace(cfg_in, model=replace(cfg_in.model, beta=beta_use))

    ky_vals = np.asarray(ky_values, dtype=float)
    gamma_out: list[float] = []
    omega_out: list[float] = []
    ky_out: list[float] = []

    for i, ky_val in enumerate(ky_vals):
        dt_i = indexed_scan_value(dt, i)
        steps_i = indexed_scan_value(steps, i)
        if dt_i is None:
            dt_i = dt
        if steps_i is None:
            steps_i = steps
        out = run_kbm_beta_scan(
            betas=np.asarray([beta_use], dtype=float),
            ky_target=float(ky_val),
            Nl=Nl,
            Nm=Nm,
            dt=float(dt_i),
            steps=int(steps_i),
            method=method,
            cfg=cfg_use,
            time_cfg=time_cfg,
            solver=solver,
            krylov_cfg=krylov_cfg,
            kbm_target_factors=kbm_target_factors,
            kbm_beta_transition=kbm_beta_transition,
            tmin=indexed_scan_value(tmin, i),
            tmax=indexed_scan_value(tmax, i),
            auto_window=auto_window,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
            mode_method=mode_method,
            mode_only=mode_only,
            terms=terms,
            sample_stride=sample_stride,
            fit_signal=fit_signal,
            ky_batch=ky_batch,
            fixed_batch_shape=fixed_batch_shape,
            streaming_fit=streaming_fit,
            streaming_amp_floor=streaming_amp_floor,
            init_species_index=init_species_index,
            density_species_index=density_species_index,
            diagnostic_norm=diagnostic_norm,
            fapar_override=fapar_override,
            apar_beta_scale=apar_beta_scale,
            ampere_g0_scale=ampere_g0_scale,
            bpar_beta_scale=bpar_beta_scale,
            reference_aligned=reference_aligned,
            gx_reference=gx_reference,
        )
        ky_out.append(float(ky_val))
        gamma_out.append(float(out.gamma[0]))
        omega_out.append(float(out.omega[0]))

    return LinearScanResult(
        ky=np.asarray(ky_out, dtype=float),
        gamma=np.asarray(gamma_out, dtype=float),
        omega=np.asarray(omega_out, dtype=float),
    )


__all__ = [
    "run_kbm_beta_scan",
    "run_kbm_linear",
    "run_kbm_scan",
]
