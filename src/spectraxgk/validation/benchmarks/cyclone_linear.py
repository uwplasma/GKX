"""Cyclone linear benchmark single-mode runner."""

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
            diagnostic_norm = "rho_star"
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


