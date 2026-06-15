"""Cyclone benchmark runners behind the public :mod:`spectraxgk.benchmarks` facade."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import jax.numpy as jnp
import numpy as np

from spectraxgk.analysis import (
    ModeSelection,
    ModeSelectionBatch,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.benchmark_defaults import (
    CYCLONE_KRYLOV_DEFAULT,
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
)
from spectraxgk.benchmark_helpers import (
    REFERENCE_DAMP_ENDS_AMP,
    REFERENCE_DAMP_ENDS_WIDTHFRAC,
    CycloneRunResult,
    CycloneScanResult,
    _apply_reference_hypercollisions,
    _build_initial_condition,
    _iter_ky_batches,
    _midplane_index,
    _normalize_growth_rate,
    _resolve_streaming_window,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.benchmark_scan import (
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
from spectraxgk.diffrax_integrators import integrate_linear_diffrax_streaming
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.explicit_time_integrators import (
    ExplicitTimeConfig,
    integrate_linear_explicit,
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


def run_cyclone_linear(
    ky_target: float = 0.3,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
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
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    fit_signal: str = "auto",
    init_cfg: InitializationConfig | None = None,
    diagnostic_norm: str = "none",
    use_jit: bool = True,
    reference_aligned: bool | None = None,
    gx_reference: bool | None = None,
    show_progress: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> CycloneRunResult:
    """Run the linear Cyclone benchmark and extract growth rate."""

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    cfg = cfg or CycloneBaseCase()
    init_cfg = init_cfg or getattr(cfg, "init", None) or InitializationConfig()
    _status("building spectral grid")
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
            diagnostic_norm = "gx"
        if mode_method not in {"z_index", "max"}:
            mode_method = "z_index"
    _status("building s-alpha geometry")
    geom = SAlphaGeometry.from_config(geom_cfg)
    if params is None:
        _status("building Cyclone linear parameters")
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
    solver_key = solver.strip().lower()
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    need_density = fit_key in {"density", "auto"}

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    _status(f"selected ky index {ky_index} at ky={float(grid.ky[sel.ky_index]):.4f}")
    _status("building initial condition")
    G0_base = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            ky_index=sel.ky_index,
            kx_index=sel.kx_index,
            Nl=Nl,
            Nm=Nm,
            init_cfg=init_cfg,
        )
    )

    def _fresh_G0() -> jnp.ndarray:
        return jnp.asarray(G0_base)

    _status("building linear cache")
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
        if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
            return False
        if require_positive and gamma_val <= 0.0:
            return False
        return True

    def _run_krylov() -> tuple[float, float, np.ndarray, np.ndarray]:
        _status("starting Krylov solve")
        kcfg = krylov_cfg or CYCLONE_KRYLOV_DEFAULT
        # reference-aligned explicit time seed to stabilize branch selection.  If the caller
        # supplied an explicit shift, respect it directly and avoid the seed
        # march; this keeps explicit-shift scans bounded and deterministic.
        gamma_seed = 0.0
        omega_seed = 0.0
        seed_ok = False
        omega_ok = False
        if kcfg.shift is None:
            try:
                _status("estimating frequency seed with short explicit time march")
                t_seed = min(150.0, float(kcfg.power_dt) * 15000.0)
                time_cfg = ExplicitTimeConfig(
                    dt=float(kcfg.power_dt),
                    t_max=t_seed,
                    sample_stride=1,
                    fixed_dt=True,
                )
                G0_seed = _fresh_G0()
                t_short, phi_t, _g_t, _o_t = integrate_linear_explicit(
                    G0_seed,
                    grid,
                    cache,
                    params,
                    geom,
                    time_cfg,
                    terms=terms,
                    mode_method="z_index",
                    show_progress=show_progress,
                )
                sel = ModeSelection(
                    ky_index=0, kx_index=0, z_index=_midplane_index(grid)
                )
                gamma_seed, omega_seed, _g, _o, _t_mid = (
                    instantaneous_growth_rate_from_phi(
                        phi_t,
                        t_short,
                        sel,
                        navg_fraction=0.5,
                        mode_method="z_index",
                    )
                )
                omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
                seed_ok = omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
            except Exception:
                seed_ok = False
                omega_ok = False

            if not seed_ok:
                try:
                    _status(
                        "primary seed failed; retrying reduced Hermite-Laguerre seed"
                    )
                    Nl_seed = min(Nl, 16)
                    Nm_seed = min(Nm, 12)
                    cache_seed = build_linear_cache(
                        grid, geom, params, Nl_seed, Nm_seed
                    )
                    G0_seed = _build_initial_condition(
                        grid,
                        geom,
                        ky_index=sel.ky_index,
                        kx_index=sel.kx_index,
                        Nl=Nl_seed,
                        Nm=Nm_seed,
                        init_cfg=init_cfg,
                    )
                    t_seed = min(150.0, float(kcfg.power_dt) * 15000.0)
                    time_cfg = ExplicitTimeConfig(
                        dt=float(kcfg.power_dt),
                        t_max=t_seed,
                        sample_stride=1,
                        fixed_dt=True,
                    )
                    G0_seed = jnp.asarray(np.asarray(G0_seed))
                    t_short, phi_t, _g_t, _o_t = integrate_linear_explicit(
                        G0_seed,
                        grid,
                        cache_seed,
                        params,
                        geom,
                        time_cfg,
                        terms=terms,
                        mode_method="z_index",
                        show_progress=show_progress,
                    )
                    sel_seed = ModeSelection(
                        ky_index=0, kx_index=0, z_index=_midplane_index(grid)
                    )
                    gamma_seed, omega_seed, _g, _o, _t_mid = (
                        instantaneous_growth_rate_from_phi(
                            phi_t,
                            t_short,
                            sel_seed,
                            navg_fraction=0.5,
                            mode_method="z_index",
                        )
                    )
                    omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
                    seed_ok = omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
                except Exception:
                    seed_ok = False
                    omega_ok = False

        shift = None
        if omega_ok:
            shift = complex(float(gamma_seed) if seed_ok else 0.0, float(-omega_seed))
        G0_krylov = _fresh_G0()
        _status("running dominant eigenpair solve")
        eig, vec = dominant_eigenpair(
            G0_krylov,
            cache,
            params,
            terms=terms,
            krylov_dim=kcfg.krylov_dim,
            restarts=kcfg.restarts,
            omega_min_factor=kcfg.omega_min_factor,
            omega_target_factor=kcfg.omega_target_factor,
            omega_cap_factor=kcfg.omega_cap_factor,
            omega_sign=kcfg.omega_sign,
            method=kcfg.method,
            power_iters=kcfg.power_iters,
            power_dt=kcfg.power_dt,
            shift=shift if shift is not None else kcfg.shift,
            shift_source=kcfg.shift_source,
            shift_tol=kcfg.shift_tol,
            shift_maxiter=kcfg.shift_maxiter,
            shift_restart=kcfg.shift_restart,
            shift_solve_method=kcfg.shift_solve_method,
            shift_preconditioner=kcfg.shift_preconditioner,
            shift_selection=kcfg.shift_selection,
            mode_family=kcfg.mode_family,
            fallback_method=kcfg.fallback_method,
            fallback_real_floor=kcfg.fallback_real_floor,
            status_callback=_status,
        )
        term_cfg = linear_terms_to_term_config(terms)
        phi = compute_fields_cached(vec, cache, params, terms=term_cfg).phi
        phi_t_out = np.asarray(phi)[None, ...]
        t_out = np.array([0.0])
        gamma_out = float(np.real(eig))
        omega_out = float(-np.imag(eig))
        if seed_ok:
            seed_strong = (gamma_seed > 0.0) and (abs(omega_seed) > 1.0e-6)
            if seed_strong:
                omega_tol = 0.15 * max(abs(omega_seed), 1.0e-6)
                gamma_tol = 0.15 * max(abs(gamma_seed), 1.0e-6)
                use_seed = (
                    not np.isfinite(gamma_out)
                    or not np.isfinite(omega_out)
                    or (gamma_seed > 0.0 and gamma_out < 0.0)
                    or abs(omega_out - omega_seed) > omega_tol
                    or abs(gamma_out - gamma_seed) > gamma_tol
                )
                if use_seed:
                    gamma_out = float(gamma_seed)
                    omega_out = float(omega_seed)
        if kcfg.omega_sign != 0:
            omega_out = float(np.sign(kcfg.omega_sign)) * abs(omega_out)
        gamma_out, omega_out = _normalize_growth_rate(
            gamma_out, omega_out, params, diagnostic_norm
        )
        _status(f"Krylov solve complete: gamma={gamma_out:.6f} omega={omega_out:.6f}")
        return gamma_out, omega_out, phi_t_out, t_out

    def _run_time() -> tuple[float, float, np.ndarray, np.ndarray]:
        _status(f"starting time integration path with fit_signal={fit_key}")
        method_key = method.lower()
        phi_t: jnp.ndarray | np.ndarray
        density_t: jnp.ndarray | np.ndarray | None
        time_cfg_use = None
        if time_cfg is not None:
            time_cfg_use = replace(time_cfg, dt=float(dt), t_max=float(dt) * int(steps))
            if sample_stride is not None:
                time_cfg_use = replace(time_cfg_use, sample_stride=sample_stride)
        elif cfg.time.use_diffrax and not (
            method_key.startswith("imex") or method_key.startswith("implicit")
        ):
            time_cfg_use = replace(cfg.time, dt=float(dt), t_max=float(dt) * int(steps))
            if sample_stride is not None:
                time_cfg_use = replace(time_cfg_use, sample_stride=sample_stride)

        if reference_aligned_use:
            # reference-aligned integrator applies damping with per-time scaling internally.
            params_use = params
            _status("running reference-aligned explicit integrator")
            t_max_val = (
                float(dt) * int(steps)
                if time_cfg_use is None
                else float(time_cfg_use.t_max)
            )
            stride = (
                int(sample_stride)
                if sample_stride is not None
                else (1 if time_cfg_use is None else int(time_cfg_use.sample_stride))
            )
            explicit_time_cfg = ExplicitTimeConfig(
                dt=float(dt),
                t_max=t_max_val,
                sample_stride=stride,
                fixed_dt=True,
            )
            t, phi_t, _g_t, _o_t = integrate_linear_explicit(
                _fresh_G0(),
                grid,
                cache,
                params_use,
                geom,
                explicit_time_cfg,
                terms=terms,
                mode_method="z_index",
                show_progress=show_progress,
            )
            sel_local = ModeSelection(
                ky_index=0, kx_index=0, z_index=_midplane_index(grid)
            )
            gamma, omega, _g, _o, _t_mid = instantaneous_growth_rate_from_phi(
                phi_t, t, sel_local, navg_fraction=0.5, mode_method="z_index"
            )
            gamma, omega = _normalize_growth_rate(
                gamma, omega, params_use, diagnostic_norm
            )
            return gamma, omega, np.asarray(phi_t), np.asarray(t)

        params_use = params
        if time_cfg_use is not None:
            _status(
                f"running runtime-configured integrator over {int(steps)} steps with sample_stride={int(time_cfg_use.sample_stride)}"
            )
            if need_density:
                _status(
                    "saving phi and density diagnostics for automatic fit selection"
                )
                _, saved = integrate_linear_from_config(
                    _fresh_G0(),
                    grid,
                    geom,
                    params_use,
                    time_cfg_use,
                    terms=terms,
                    save_field="phi+density",
                    density_species_index=0,
                    show_progress=show_progress,
                )
                phi_t, density_t = saved
            else:
                _, phi_t = integrate_linear_from_config(
                    _fresh_G0(),
                    grid,
                    geom,
                    params_use,
                    time_cfg_use,
                    terms=terms,
                    show_progress=show_progress,
                )
                density_t = None
            stride = time_cfg_use.sample_stride
        else:
            stride = 1 if sample_stride is None else int(sample_stride)
            if need_density or not use_jit:
                _status(
                    f"running explicit diagnostics integrator over {int(steps)} steps with sample_stride={stride}"
                )
                _diag = integrate_linear_diagnostics(
                    _fresh_G0(),
                    grid,
                    geom,
                    params_use,
                    dt=dt,
                    steps=steps,
                    method=method,
                    terms=terms,
                    sample_stride=stride,
                    species_index=0,
                    record_hl_energy=False,
                    show_progress=show_progress,
                )
                phi_t = _diag[1]
                density_t = _diag[2] if len(_diag) > 2 else None
            else:
                _status(
                    f"running cached linear integrator over {int(steps)} steps with sample_stride={stride}"
                )
                _, phi_out_time = integrate_linear(
                    _fresh_G0(),
                    grid,
                    geom,
                    params_use,
                    dt=dt,
                    steps=steps,
                    method=method,
                    terms=terms,
                    sample_stride=stride,
                    show_progress=show_progress,
                )
                phi_t = phi_out_time
                density_t = None

        phi_t_np = np.asarray(phi_t)
        t_arr = np.arange(phi_t_np.shape[0]) * dt * stride
        density_np = None if density_t is None else np.asarray(density_t)
        _status(
            f"integration complete; fitting growth rate from {phi_t_np.shape[0]} saved samples"
        )
        if fit_key == "auto":
            signal, _name, gamma_out, omega_out = _select_fit_signal_auto(
                t_arr,
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
            _status(f"automatic fit selected signal '{_name}'")
            if not np.isfinite(gamma_out) or not np.isfinite(omega_out):
                gamma_out, omega_out = 0.0, 0.0
        else:
            signal = _select_fit_signal(
                phi_t_np,
                density_np,
                sel,
                fit_signal=fit_key,
                mode_method=mode_method,
            )
            if auto_window and tmin is None and tmax is None:
                gamma_out, omega_out, _tmin, _tmax = fit_growth_rate_auto(
                    t_arr,
                    signal,
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
                )
            else:
                gamma_out, omega_out = fit_growth_rate(
                    t_arr, signal, tmin=tmin, tmax=tmax
                )
        gamma_out, omega_out = _normalize_growth_rate(
            gamma_out, omega_out, params_use, diagnostic_norm
        )
        _status(
            f"time integration fit complete: gamma={gamma_out:.6f} omega={omega_out:.6f}"
        )
        return float(gamma_out), float(omega_out), phi_t_np, t_arr

    if solver_key == "krylov":
        gamma, omega, phi_t_np, t = _run_krylov()
    elif solver_key == "auto":
        try:
            gamma, omega, phi_t_np, t = _run_time()
        except ValueError as exc:
            _status(f"time-path failed ({exc}); falling back to Krylov solve")
            gamma, omega, phi_t_np, t = _run_krylov()
        if not _is_valid_growth(gamma, omega):
            _status("time-path result rejected; falling back to Krylov solve")
            gamma, omega, phi_t_np, t = _run_krylov()
    else:
        gamma, omega, phi_t_np, t = _run_time()

    _status(f"completed Cyclone linear run at ky={float(grid.ky[sel.ky_index]):.4f}")

    return CycloneRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
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
            diagnostic_norm = "gx"
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
        if ky_values_arr.size == 0:
            return CycloneScanResult(
                ky=ky_values_arr, gamma=np.array([]), omega=np.array([])
            )
        order = (
            np.argsort(ky_values_arr) if mode_follow else np.arange(ky_values_arr.size)
        )
        gamma_out = np.zeros_like(ky_values_arr, dtype=float)
        omega_out = np.zeros_like(ky_values_arr, dtype=float)
        v_ref: jnp.ndarray | None = None
        prev_eig: complex | None = None
        cfg_use = krylov_cfg or CYCLONE_KRYLOV_DEFAULT
        for idx in order:
            ky_val = float(ky_values_arr[idx])
            ky_index = select_ky_index(np.asarray(grid_full.ky), ky_val)
            grid = select_ky_grid(grid_full, ky_index)
            G0_jax = _build_initial_condition(
                grid,
                geom,
                ky_index=0,
                kx_index=0,
                Nl=Nl,
                Nm=Nm,
                init_cfg=init_cfg,
            )
            cache = build_linear_cache(grid, geom, params, Nl, Nm)
            # Use a short reference-aligned explicit time integration to seed the branch.
            gamma_seed = 0.0
            omega_seed = 0.0
            seed_ok = False
            omega_ok = False
            if prev_eig is None:
                try:
                    t_seed = min(150.0, float(cfg_use.power_dt) * 15000.0)
                    explicit_time_cfg = ExplicitTimeConfig(
                        dt=float(cfg_use.power_dt),
                        t_max=t_seed,
                        sample_stride=1,
                        fixed_dt=True,
                    )
                    G0_seed = jnp.array(G0_jax)
                    t_short, phi_seed, _g_t, _o_t = integrate_linear_explicit(
                        G0_seed,
                        grid,
                        cache,
                        params,
                        geom,
                        explicit_time_cfg,
                        terms=terms,
                        mode_method="z_index",
                        show_progress=show_progress,
                    )

                    sel = ModeSelection(
                        ky_index=0, kx_index=0, z_index=_midplane_index(grid)
                    )
                    gamma_seed, omega_seed, _g, _o, _t_mid = (
                        instantaneous_growth_rate_from_phi(
                            phi_seed,
                            t_short,
                            sel,
                            navg_fraction=0.5,
                            mode_method="z_index",
                        )
                    )
                    omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
                    seed_ok = omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
                except Exception:
                    seed_ok = False
                    omega_ok = False
            if not seed_ok:
                try:
                    Nl_seed = min(Nl, 16)
                    Nm_seed = min(Nm, 12)
                    cache_seed = build_linear_cache(
                        grid, geom, params, Nl_seed, Nm_seed
                    )
                    G0_seed = _build_initial_condition(
                        grid,
                        geom,
                        ky_index=0,
                        kx_index=0,
                        Nl=Nl_seed,
                        Nm=Nm_seed,
                        init_cfg=init_cfg,
                    )
                    t_seed = min(150.0, float(cfg_use.power_dt) * 15000.0)
                    explicit_time_cfg = ExplicitTimeConfig(
                        dt=float(cfg_use.power_dt),
                        t_max=t_seed,
                        sample_stride=1,
                        fixed_dt=True,
                    )
                    t_short, phi_seed, _g_t, _o_t = integrate_linear_explicit(
                        G0_seed,
                        grid,
                        cache_seed,
                        params,
                        geom,
                        explicit_time_cfg,
                        terms=terms,
                        mode_method="z_index",
                        show_progress=show_progress,
                    )

                    sel_seed = ModeSelection(
                        ky_index=0, kx_index=0, z_index=_midplane_index(grid)
                    )
                    gamma_seed, omega_seed, _g, _o, _t_mid = (
                        instantaneous_growth_rate_from_phi(
                            phi_seed,
                            t_short,
                            sel_seed,
                            navg_fraction=0.5,
                            mode_method="z_index",
                        )
                    )
                    omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
                    seed_ok = omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
                except Exception:
                    seed_ok = False
                    omega_ok = False

            shift: complex | None
            if prev_eig is not None and np.isfinite(prev_eig):
                shift = prev_eig
            elif omega_ok:
                shift = complex(
                    float(gamma_seed) if seed_ok else 0.0, float(-omega_seed)
                )
            else:
                shift = None
            eig, vec = dominant_eigenpair(
                G0_jax,
                cache,
                params,
                terms=terms,
                v_ref=v_ref,
                select_overlap=v_ref is not None,
                krylov_dim=cfg_use.krylov_dim,
                restarts=cfg_use.restarts,
                omega_min_factor=cfg_use.omega_min_factor,
                omega_target_factor=cfg_use.omega_target_factor,
                omega_cap_factor=cfg_use.omega_cap_factor,
                omega_sign=cfg_use.omega_sign,
                method=cfg_use.method,
                power_iters=cfg_use.power_iters,
                power_dt=cfg_use.power_dt,
                shift=shift if shift is not None else cfg_use.shift,
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
            # If Krylov lands on the wrong branch, fall back to reference-aligned explicit seed.
            use_seed = False
            if seed_ok:
                seed_strong = (gamma_seed > 0.0) and (abs(omega_seed) > 1.0e-6)
                if seed_strong:
                    omega_tol = 0.15 * max(abs(omega_seed), 1.0e-6)
                    gamma_tol = 0.15 * max(abs(gamma_seed), 1.0e-6)
                    use_seed = (
                        not np.isfinite(gamma)
                        or not np.isfinite(omega)
                        or (gamma_seed > 0.0 and gamma < 0.0)
                        or abs(omega - omega_seed) > omega_tol
                        or abs(gamma - gamma_seed) > gamma_tol
                    )
            if use_seed and seed_ok:
                gamma = float(gamma_seed)
                omega = float(omega_seed)
            else:
                v_ref = vec
            prev_eig = complex(float(gamma), float(-omega))
            gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
            gamma_out[idx] = gamma
            omega_out[idx] = omega
        return CycloneScanResult(ky=ky_values_arr, gamma=gamma_out, omega=omega_out)

    if solver_key == "explicit_time":
        if ky_values_arr.size == 0:
            return CycloneScanResult(
                ky=ky_values_arr, gamma=np.array([]), omega=np.array([])
            )
        gamma_out = np.zeros_like(ky_values_arr, dtype=float)
        omega_out = np.zeros_like(ky_values_arr, dtype=float)
        prev_omega: float | None = None
        prev_prev_omega: float | None = None
        kcfg = krylov_cfg or CYCLONE_KRYLOV_DEFAULT
        time_base = time_cfg or cfg.time
        for idx, ky_val in enumerate(ky_values_arr):
            ky_index = select_ky_index(np.asarray(grid_full.ky), float(ky_val))
            grid = select_ky_grid(grid_full, ky_index)
            G0_jax = _build_initial_condition(
                grid,
                geom,
                ky_index=0,
                kx_index=0,
                Nl=Nl,
                Nm=Nm,
                init_cfg=init_cfg,
            )
            cache = build_linear_cache(grid, geom, params, Nl, Nm)
            dt_i = float(dt[idx]) if isinstance(dt, np.ndarray) else float(dt)
            steps_i = int(steps[idx]) if isinstance(steps, np.ndarray) else int(steps)
            t_max_val = dt_i * float(steps_i)
            if reference_aligned_use and time_cfg is None:
                fixed_dt_i = True
                dt_min_i = dt_i
                dt_max_i: float | None = dt_i
                cfl_i = 1.0
                cfl_fac_i = 1.0
            else:
                fixed_dt_i = bool(time_base.fixed_dt)
                dt_min_i = float(time_base.dt_min)
                dt_max_i = None if time_base.dt_max is None else float(time_base.dt_max)
                cfl_i = float(time_base.cfl)
                cfl_fac_i = resolve_cfl_fac(str(time_base.method), time_base.cfl_fac)
            explicit_time_cfg = ExplicitTimeConfig(
                dt=dt_i,
                t_max=t_max_val,
                sample_stride=1,
                fixed_dt=fixed_dt_i,
                dt_min=dt_min_i,
                dt_max=dt_max_i,
                cfl=cfl_i,
                cfl_fac=cfl_fac_i,
            )
            G0_time = jnp.array(G0_jax)
            t, phi_gx, _g_t, _o_t = integrate_linear_explicit(
                G0_time,
                grid,
                cache,
                params,
                geom,
                explicit_time_cfg,
                terms=terms,
                mode_method="z_index",
                show_progress=show_progress,
            )
            sel_local = ModeSelection(
                ky_index=0, kx_index=0, z_index=_midplane_index(grid)
            )
            explicit_growth_ok = True
            try:
                gamma, omega, _g, _o, _t_mid = instantaneous_growth_rate_from_phi(
                    phi_gx, t, sel_local, navg_fraction=0.5, mode_method="z_index"
                )
                gamma, omega = _normalize_growth_rate(
                    gamma, omega, params, diagnostic_norm
                )
            except ValueError:
                explicit_growth_ok = False
                gamma = float("nan")
                omega = float("nan")
            if reference_aligned_use and prev_omega is None and omega < 0.0:
                omega = abs(omega)
            need_reselect = (
                (reference_aligned_use and explicit_growth_ok)
                and prev_omega is not None
                and prev_omega > 0.0
                and (omega <= 0.0 or ((idx >= 2) and (omega < 0.85 * prev_omega)))
            )
            if need_reselect or not explicit_growth_ok:
                target_omega: float | None = (
                    prev_omega
                    if (explicit_growth_ok and prev_omega is not None)
                    else None
                )
                if (
                    target_omega is not None
                    and prev_prev_omega is not None
                    and prev_omega is not None
                    and prev_omega > prev_prev_omega
                ):
                    target_omega = prev_omega + (prev_omega - prev_prev_omega)
                G0_krylov = jnp.array(G0_jax)
                eig, _vec = dominant_eigenpair(
                    G0_krylov,
                    cache,
                    params,
                    terms=terms,
                    krylov_dim=kcfg.krylov_dim,
                    restarts=kcfg.restarts,
                    omega_min_factor=kcfg.omega_min_factor,
                    omega_target_factor=kcfg.omega_target_factor,
                    omega_cap_factor=kcfg.omega_cap_factor,
                    omega_sign=kcfg.omega_sign,
                    method=kcfg.method,
                    power_iters=kcfg.power_iters,
                    power_dt=kcfg.power_dt,
                    shift=kcfg.shift,
                    shift_source=kcfg.shift_source,
                    shift_tol=kcfg.shift_tol,
                    shift_maxiter=kcfg.shift_maxiter,
                    shift_restart=kcfg.shift_restart,
                    shift_solve_method=kcfg.shift_solve_method,
                    shift_preconditioner=kcfg.shift_preconditioner,
                    shift_selection=kcfg.shift_selection,
                    mode_family=kcfg.mode_family,
                    fallback_method=kcfg.fallback_method,
                    fallback_real_floor=kcfg.fallback_real_floor,
                )
                gamma_k = float(np.real(eig))
                omega_k = float(abs(-np.imag(eig)))
                gamma_k, omega_k = _normalize_growth_rate(
                    gamma_k, omega_k, params, diagnostic_norm
                )
                if not explicit_growth_ok:
                    gamma, omega = gamma_k, omega_k
                else:
                    assert target_omega is not None
                    candidates: list[tuple[float, float]] = [
                        (float(gamma), float(abs(omega)))
                    ]
                    gamma_base = abs(float(gamma))
                    gamma_delta_limit = max(3.0 * gamma_base, gamma_base + 0.05, 1.0e-3)
                    if (
                        np.isfinite(gamma_k)
                        and np.isfinite(omega_k)
                        and gamma_k > 0.0
                        and abs(gamma_k - float(gamma)) <= gamma_delta_limit
                    ):
                        candidates.append((gamma_k, omega_k))

                    def _score(candidate: tuple[float, float]) -> float:
                        g_val, o_val = candidate
                        penalty = 0.0 if g_val > 0.0 else 1.0e3
                        return penalty + abs(o_val - target_omega)

                    gamma, omega = min(candidates, key=_score)
            gamma_out[idx] = gamma
            omega_out[idx] = omega
            prev_prev_omega = prev_omega
            prev_omega = float(omega)
        return CycloneScanResult(ky=ky_values_arr, gamma=gamma_out, omega=omega_out)
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
