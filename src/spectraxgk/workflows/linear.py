"""Executable linear runtime workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from spectraxgk.analysis import ModeSelection
from spectraxgk.runtime_config import RuntimeConfig
from spectraxgk.runtime_diagnostics import RuntimeQuasilinearFinalizationDeps
from spectraxgk.runtime_results import RuntimeLinearResult


@dataclass(frozen=True)
class FullLinearRuntimeDeps:
    """Injected dependencies for the full-GK linear runtime workflow."""

    build_runtime_geometry: Callable[[RuntimeConfig], Any]
    apply_geometry_grid_defaults: Callable[..., Any]
    build_spectral_grid: Callable[..., Any]
    build_runtime_linear_params: Callable[..., Any]
    build_runtime_linear_terms: Callable[..., Any]
    select_ky_index: Callable[..., int]
    select_ky_grid: Callable[..., Any]
    midplane_index: Callable[..., int]
    build_initial_condition: Callable[..., Any]
    normalize_linear_solver_name: Callable[..., str]
    runtime_default_krylov_config: Callable[..., Any]
    build_linear_cache: Callable[..., Any]
    dominant_eigenpair: Callable[..., tuple[Any, Any]]
    apply_diagnostic_normalization: Callable[..., tuple[float, float]]
    integrate_linear_from_config: Callable[..., tuple[Any, Any]]
    integrate_linear_diagnostics: Callable[..., tuple[Any, ...]]
    fit_runtime_linear_diagnostics: Callable[..., Any]
    finalize_runtime_linear_quasilinear: Callable[..., RuntimeLinearResult]
    quasilinear_finalization_deps: RuntimeQuasilinearFinalizationDeps
    extract_mode_time_series: Callable[..., Any]
    fit_growth_rate_auto_with_stats: Callable[..., Any]
    fit_growth_rate_auto: Callable[..., Any]
    fit_growth_rate: Callable[..., Any]
    extract_eigenfunction: Callable[..., Any]


def run_full_linear_runtime(
    cfg: RuntimeConfig,
    *,
    deps: FullLinearRuntimeDeps,
    ky_target: float,
    Nl: int,
    Nm: int,
    solver: str,
    method: str | None,
    dt: float | None,
    steps: int | None,
    sample_stride: int | None,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    krylov_cfg: Any | None,
    mode_method: str,
    fit_signal: str,
    return_state: bool,
    show_progress: bool,
    status_callback: Callable[[str], None] | None = None,
) -> RuntimeLinearResult:
    """Run one full-GK linear point from a runtime config."""

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    ql_enabled = bool(getattr(cfg.quasilinear, "enabled", False))
    return_state_requested = bool(return_state)
    return_state_eff = return_state_requested or ql_enabled

    geom = deps.build_runtime_geometry(cfg)
    _status("building spectral grid")
    grid_cfg = deps.apply_geometry_grid_defaults(geom, cfg.grid)
    grid_full = deps.build_spectral_grid(grid_cfg)
    _status("building runtime linear parameters")
    params = deps.build_runtime_linear_params(cfg, Nm=Nm, geom=geom)
    terms = deps.build_runtime_linear_terms(cfg)

    ky_index = deps.select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = deps.select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=deps.midplane_index(grid))
    _status(f"selected ky index {ky_index} at ky={float(grid.ky[sel.ky_index]):.4f}")
    _status("building initial condition")
    g0 = deps.build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        nspecies=max(len([s for s in cfg.species if s.kinetic]), 1),
    )

    solver_key = deps.normalize_linear_solver_name(solver)
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")

    def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
        if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
            return False
        if require_positive and gamma_val <= 0.0:
            return False
        return True

    def _finalize_linear_result(
        result: RuntimeLinearResult,
        *,
        state_for_quasilinear: np.ndarray | None = None,
    ) -> RuntimeLinearResult:
        return deps.finalize_runtime_linear_quasilinear(
            result,
            enabled=ql_enabled,
            cfg=cfg,
            grid=grid,
            geom=geom,
            params=params,
            terms=terms,
            Nl=Nl,
            Nm=Nm,
            solver_name=deps.normalize_linear_solver_name(solver),
            species_names=tuple(s.name for s in cfg.species if s.kinetic),
            return_state_requested=return_state_requested,
            state_for_quasilinear=state_for_quasilinear,
            deps=deps.quasilinear_finalization_deps,
            status_callback=_status,
        )

    def _run_krylov() -> tuple[float, float, np.ndarray]:
        _status("starting Krylov solve")
        kcfg = krylov_cfg or deps.runtime_default_krylov_config(cfg)
        _status("building linear cache")
        cache = deps.build_linear_cache(grid, geom, params, Nl, Nm)
        eig, vec = deps.dominant_eigenpair(
            g0,
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
            status_callback=_status,
        )
        gamma = float(jnp.real(eig))
        omega = float(-jnp.imag(eig))
        gamma, omega = deps.apply_diagnostic_normalization(
            gamma,
            omega,
            rho_star=float(np.asarray(params.rho_star)),
            diagnostic_norm=cfg.normalization.diagnostic_norm,
        )
        _status(f"Krylov solve complete: gamma={gamma:.6f} omega={omega:.6f}")
        return gamma, omega, np.asarray(vec)

    def _run_time() -> RuntimeLinearResult:
        _status(f"starting time integration path with fit_signal={fit_key}")
        tcfg = cfg.time
        if method is not None:
            tcfg = replace(tcfg, method=str(method))
        if dt is not None:
            tcfg = replace(tcfg, dt=float(dt))
        if steps is not None:
            tcfg = replace(tcfg, t_max=float(steps) * float(tcfg.dt))
        if sample_stride is not None:
            tcfg = replace(tcfg, sample_stride=int(sample_stride))
        if return_state_eff and solver_key == "explicit_time":
            raise ValueError(
                "return_state/quasilinear diagnostics are not supported with solver='explicit_time'"
            )
        if return_state_eff:
            tcfg = replace(tcfg, save_state=True)

        need_density = fit_key in {"density", "auto"}
        parallel_strategy = (
            str(getattr(cfg.parallel, "strategy", "serial")).lower().replace("-", "_")
        )
        if parallel_strategy != "serial":
            if tcfg.use_diffrax:
                raise NotImplementedError(
                    "parallel linear RHS is currently supported only by the fixed-step cached integrator"
                )
            if need_density:
                raise NotImplementedError(
                    "parallel linear RHS runtime path currently requires fit_signal='phi'"
                )
        g_last = None
        if tcfg.use_diffrax:
            _status(
                f"running diffrax integrator over {int(round(tcfg.t_max / tcfg.dt))} steps with sample_stride={int(tcfg.sample_stride)}"
            )
            save_field = "phi+density" if need_density else "phi"
            save_mode = None
            g_last, saved = deps.integrate_linear_from_config(
                g0,
                grid,
                geom,
                params,
                tcfg,
                terms=terms,
                save_mode=save_mode,
                mode_method=mode_method,
                save_field=save_field,
                density_species_index=0 if need_density else None,
                show_progress=show_progress,
                parallel=cfg.parallel,
            )
            if need_density:
                phi_t, density_t = saved
            else:
                phi_t, density_t = saved, None
        else:
            if need_density:
                _status(
                    f"running diagnostics integrator over {int(round(tcfg.t_max / tcfg.dt))} steps with sample_stride={int(tcfg.sample_stride)}"
                )
                _diag = deps.integrate_linear_diagnostics(
                    g0,
                    grid,
                    geom,
                    params,
                    dt=tcfg.dt,
                    steps=int(round(tcfg.t_max / tcfg.dt)),
                    method=tcfg.method,
                    terms=terms,
                    sample_stride=tcfg.sample_stride,
                    species_index=0,
                    record_hl_energy=False,
                    show_progress=show_progress,
                )
                g_last = _diag[0]
                phi_t = _diag[1]
                density_t = _diag[2] if len(_diag) > 2 else None
            else:
                _status(
                    f"running cached linear integrator over {int(round(tcfg.t_max / tcfg.dt))} steps with sample_stride={int(tcfg.sample_stride)}"
                )
                g_last, phi_t = deps.integrate_linear_from_config(
                    g0,
                    grid,
                    geom,
                    params,
                    tcfg,
                    terms=terms,
                    save_mode=sel,
                    mode_method=mode_method,
                    save_field="phi",
                    show_progress=show_progress,
                    parallel=cfg.parallel,
                )
                density_t = None

        phi_t_np = np.asarray(phi_t)
        t_arr = (
            float(tcfg.dt)
            * float(tcfg.sample_stride)
            * (np.arange(phi_t_np.shape[0], dtype=float) + 1.0)
        )
        density_np = None if density_t is None else np.asarray(density_t)
        _status(f"integration complete; fitting growth rate from {t_arr.size} saved samples")

        fit_result = deps.fit_runtime_linear_diagnostics(
            t=t_arr,
            phi_t=phi_t_np,
            density_t=density_np,
            selection=sel,
            z=np.asarray(grid.z, dtype=float),
            fit_signal=fit_key,
            mode_method=mode_method,
            auto_window=auto_window,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
            extract_mode_time_series_fn=deps.extract_mode_time_series,
            fit_growth_rate_auto_with_stats_fn=deps.fit_growth_rate_auto_with_stats,
            fit_growth_rate_auto_fn=deps.fit_growth_rate_auto,
            fit_growth_rate_fn=deps.fit_growth_rate,
            extract_eigenfunction_fn=deps.extract_eigenfunction,
        )
        if fit_key == "auto":
            _status(f"automatic fit selected signal '{fit_result.fit_signal_used}'")
        gamma, omega = deps.apply_diagnostic_normalization(
            fit_result.gamma,
            fit_result.omega,
            rho_star=float(np.asarray(params.rho_star)),
            diagnostic_norm=cfg.normalization.diagnostic_norm,
        )
        _status(f"fit complete: gamma={gamma:.6f} omega={omega:.6f}")
        return RuntimeLinearResult(
            ky=float(grid.ky[sel.ky_index]),
            gamma=float(gamma),
            omega=float(omega),
            selection=sel,
            t=t_arr,
            signal=fit_result.signal,
            state=None if g_last is None or not return_state_eff else np.asarray(g_last),
            z=fit_result.z if fit_result.eigenfunction is not None else None,
            eigenfunction=fit_result.eigenfunction,
            fit_window_tmin=fit_result.fit_window_tmin,
            fit_window_tmax=fit_result.fit_window_tmax,
            fit_signal_used=fit_result.fit_signal_used,
        )

    if solver_key == "krylov":
        gamma, omega, vec = _run_krylov()
        result = RuntimeLinearResult(
            ky=float(grid.ky[sel.ky_index]),
            gamma=gamma,
            omega=omega,
            selection=sel,
            state=vec if return_state_eff else None,
        )
        return _finalize_linear_result(result, state_for_quasilinear=vec)
    if solver_key == "auto":
        result = _run_time()
        if not _is_valid_growth(result.gamma, result.omega):
            _status("time-path result rejected; falling back to Krylov solve")
            gamma, omega, vec = _run_krylov()
            result = RuntimeLinearResult(
                ky=float(grid.ky[sel.ky_index]),
                gamma=gamma,
                omega=omega,
                selection=sel,
                state=vec if return_state_eff else None,
            )
            return _finalize_linear_result(result, state_for_quasilinear=vec)
        return _finalize_linear_result(result)

    return _finalize_linear_result(_run_time())
