"""KBM single-ky linear benchmark runner."""

# ruff: noqa: F401

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Sequence

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
from spectraxgk.validation.benchmarks.scan import _resolve_streaming_window
from spectraxgk.diagnostics.growth_rates import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.defaults import _build_initial_condition
from spectraxgk.validation.benchmarks.defaults import LinearRunResult, LinearScanResult
from spectraxgk.validation.benchmarks.defaults import (
    _kbm_use_multi_target_krylov,
    _midplane_index,
    select_kbm_solver_auto,
)
from spectraxgk.validation.benchmarks.defaults import (
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


@dataclass(frozen=True)
class _KBMLinearSetup:
    cfg: KBMBaseCase
    beta: float
    geom: Any
    grid_full: Any
    params: LinearParams
    terms: LinearTerms
    diagnostic_norm: str
    reference_aligned: bool
    fit_key: str


@dataclass(frozen=True)
class _KBMLinearState:
    grid: Any
    selection: ModeSelection
    cache: Any
    state: Any


@dataclass(frozen=True)
class _KBMLinearRunOptions:
    ky_target: float
    Nl: int
    Nm: int
    dt: float
    steps: int
    method: str
    time_cfg: TimeConfig | None
    krylov_cfg: KrylovConfig | None
    kbm_target_factors: Sequence[float] | None
    kbm_beta_transition: float | None
    auto_window: bool
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    mode_method: str
    sample_stride: int | None
    density_species_index: int
    show_progress: bool


@dataclass(frozen=True)
class _KBMLinearRequest:
    ky_target: float
    beta_value: float | None
    Nl: int
    Nm: int
    dt: float
    steps: int
    method: str
    params: LinearParams | None
    cfg: KBMBaseCase | None
    time_cfg: TimeConfig | None
    solver: str
    krylov_cfg: KrylovConfig | None
    kbm_target_factors: Sequence[float] | None
    kbm_beta_transition: float | None
    tmin: float | None
    tmax: float | None
    auto_window: bool
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    mode_method: str
    terms: LinearTerms | None
    sample_stride: int | None
    fit_signal: str
    streaming_fit: bool
    init_species_index: int
    density_species_index: int
    diagnostic_norm: str
    fapar_override: float | None
    apar_beta_scale: float | None
    ampere_g0_scale: float | None
    bpar_beta_scale: float | None
    reference_aligned: bool | None
    show_progress: bool


def _kbm_linear_request_from_locals(values: dict[str, Any]) -> _KBMLinearRequest:
    """Pack public ``run_kbm_linear`` arguments once for internal routing."""

    return _KBMLinearRequest(
        **{field.name: values[field.name] for field in fields(_KBMLinearRequest)}
    )


def _resolve_kbm_fit_signal(fit_signal: str) -> str:
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    return fit_key


def _validate_kbm_species_indices(
    *, init_species_index: int, density_species_index: int, nspecies: int = 2
) -> None:
    if init_species_index < 0 or init_species_index >= nspecies:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= nspecies:
        raise ValueError("density_species_index out of range for kinetic species")


def _resolve_kbm_linear_setup(
    *,
    cfg: KBMBaseCase | None,
    beta_value: float | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    diagnostic_norm: str,
    fit_signal: str,
    reference_aligned: bool | None,
    Nm: int,
    fapar_override: float | None,
    apar_beta_scale: float | None,
    ampere_g0_scale: float | None,
    bpar_beta_scale: float | None,
) -> _KBMLinearSetup:
    cfg_in = cfg or KBMBaseCase()
    beta_use = float(cfg_in.model.beta) if beta_value is None else float(beta_value)
    cfg_use = replace(cfg_in, model=replace(cfg_in.model, beta=beta_use))
    geom = build_flux_tube_geometry(cfg_use.geometry)
    grid_full = build_spectral_grid(apply_geometry_grid_defaults(geom, cfg_use.grid))
    terms_use = terms if terms is not None else LinearTerms(bpar=0.0)
    reference_aligned_use = bool(
        True if reference_aligned is None else reference_aligned
    )
    diagnostic_norm_use = diagnostic_norm
    if reference_aligned_use and diagnostic_norm_use == "none":
        diagnostic_norm_use = "rho_star"
    damp_ends_amp, damp_ends_widthfrac = _linked_boundary_end_damping(
        reference_aligned_use
    )
    params_use = params
    if params_use is None:
        params_use = _two_species_params(
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
    return _KBMLinearSetup(
        cfg=cfg_use,
        beta=beta_use,
        geom=geom,
        grid_full=grid_full,
        params=params_use,
        terms=terms_use,
        diagnostic_norm=diagnostic_norm_use,
        reference_aligned=reference_aligned_use,
        fit_key=_resolve_kbm_fit_signal(fit_signal),
    )


def _prepare_kbm_linear_state(
    setup: _KBMLinearSetup,
    *,
    ky_target: float,
    Nl: int,
    Nm: int,
    init_species_index: int,
) -> _KBMLinearState:
    ky_index = select_ky_index(np.asarray(setup.grid_full.ky), ky_target)
    grid = select_ky_grid(setup.grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    cache = build_linear_cache(grid, setup.geom, setup.params, Nl, Nm)
    G0 = np.zeros(
        (2, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
    )
    G0_single = _build_initial_condition(
        grid,
        setup.geom,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=setup.cfg.init,
    )
    G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)
    return _KBMLinearState(
        grid=grid,
        selection=sel,
        cache=cache,
        state=jnp.asarray(G0),
    )


def _fit_kbm_signal_with_window(
    signal: np.ndarray,
    t_arr: np.ndarray,
    *,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> tuple[float, float]:
    use_auto = auto_window and tmin is None and tmax is None
    if not use_auto and not scan_window_valid(t_arr, tmin, tmax):
        use_auto = True
    auto_fit_kwargs: dict[str, Any] = {
        "window_fraction": window_fraction,
        "min_points": min_points,
        "start_fraction": start_fraction,
        "growth_weight": growth_weight,
        "require_positive": require_positive,
        "min_amp_fraction": min_amp_fraction,
    }
    if use_auto:
        gamma_val, omega_val, _tmin, _tmax = fit_growth_rate_auto(
            t_arr, signal, **auto_fit_kwargs
        )
    else:
        try:
            gamma_val, omega_val = fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)
        except ValueError:
            gamma_val, omega_val, _tmin, _tmax = fit_growth_rate_auto(
                t_arr, signal, **auto_fit_kwargs
            )
    return gamma_val, omega_val


def _resolve_kbm_time_config(
    time_cfg: TimeConfig | None,
    *,
    dt: float,
    steps: int,
    stride: int,
    sample_stride: int | None,
) -> TimeConfig | None:
    if time_cfg is None:
        return None
    time_cfg_use = replace(time_cfg, dt=dt, t_max=dt * steps)
    if sample_stride is not None:
        time_cfg_use = replace(time_cfg_use, sample_stride=stride)
    return time_cfg_use


def _integrate_kbm_configured_history(
    state: _KBMLinearState,
    setup: _KBMLinearSetup,
    *,
    time_cfg: TimeConfig,
    dt: float,
    steps: int,
    method: str,
    density_species_index: int,
    mode_method: str,
    show_progress: bool,
    stride: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    stride = int(time_cfg.sample_stride)
    if time_cfg.use_diffrax:
        save_field = "phi+density" if setup.fit_key in {"density", "auto"} else "phi"
        _, phi_out = integrate_linear_from_config(
            state.state,
            state.grid,
            setup.geom,
            setup.params,
            time_cfg,
            cache=state.cache,
            terms=setup.terms,
            save_mode=state.selection if setup.fit_key == "phi" else None,
            mode_method=mode_method,
            save_field=save_field,
            density_species_index=density_species_index
            if setup.fit_key in {"density", "auto"}
            else None,
        )
        if setup.fit_key in {"density", "auto"}:
            return np.asarray(phi_out[0]), np.asarray(phi_out[1]), stride
        return np.asarray(phi_out), None, stride

    if setup.fit_key in {"density", "auto"}:
        diag_out = integrate_linear_diagnostics(
            state.state,
            state.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=method,
            cache=state.cache,
            terms=setup.terms,
            sample_stride=stride,
            species_index=density_species_index,
        )
        density_np = None if len(diag_out) <= 2 else np.asarray(diag_out[2])
        return np.asarray(diag_out[1]), density_np, stride

    _, phi_out_time = integrate_linear(
        state.state,
        state.grid,
        setup.geom,
        setup.params,
        dt=dt,
        steps=steps,
        method=method,
        cache=state.cache,
        terms=setup.terms,
        sample_stride=stride,
        show_progress=show_progress,
    )
    return np.asarray(phi_out_time), None, stride


def _integrate_kbm_fixed_history(
    state: _KBMLinearState,
    setup: _KBMLinearSetup,
    *,
    dt: float,
    steps: int,
    method: str,
    density_species_index: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    if setup.fit_key in {"density", "auto"}:
        diag_out = integrate_linear_diagnostics(
            state.state,
            state.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=method,
            cache=state.cache,
            terms=setup.terms,
            sample_stride=stride,
            species_index=density_species_index,
        )
        density_np = None if len(diag_out) <= 2 else np.asarray(diag_out[2])
        return np.asarray(diag_out[1]), density_np, stride

    _, phi_out_time = integrate_linear(
        state.state,
        state.grid,
        setup.geom,
        setup.params,
        dt=dt,
        steps=steps,
        method=method,
        cache=state.cache,
        terms=setup.terms,
        sample_stride=stride,
    )
    return np.asarray(phi_out_time), None, stride


def _integrate_kbm_saved_history(
    state: _KBMLinearState,
    setup: _KBMLinearSetup,
    *,
    time_cfg: TimeConfig | None,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    density_species_index: int,
    mode_method: str,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    stride = 1 if sample_stride is None else int(sample_stride)
    time_cfg_use = _resolve_kbm_time_config(
        time_cfg,
        dt=dt,
        steps=steps,
        stride=stride,
        sample_stride=sample_stride,
    )
    if time_cfg_use is not None:
        return _integrate_kbm_configured_history(
            state,
            setup,
            time_cfg=time_cfg_use,
            dt=dt,
            steps=steps,
            method=method,
            density_species_index=density_species_index,
            mode_method=mode_method,
            show_progress=show_progress,
            stride=stride,
        )
    return _integrate_kbm_fixed_history(
        state,
        setup,
        dt=dt,
        steps=steps,
        method=method,
        density_species_index=density_species_index,
        stride=stride,
    )


def _fit_kbm_auto_history(
    state: _KBMLinearState,
    setup: _KBMLinearSetup,
    *,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    dt: float,
    stride: int,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> tuple[float, float]:
    _signal, _name, gamma, omega = _select_fit_signal_auto(
        np.arange(phi_t.shape[0]) * dt * stride,
        phi_t,
        density_t,
        state.selection,
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
    return gamma, omega


def _fit_kbm_saved_history(
    state: _KBMLinearState,
    setup: _KBMLinearSetup,
    *,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    dt: float,
    stride: int,
    mode_method: str,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> tuple[float, float]:
    density_np = (
        phi_t if setup.fit_key == "density" and density_t is None else density_t
    )
    if setup.fit_key == "auto":
        gamma, omega = _fit_kbm_auto_history(
            state,
            setup,
            phi_t=phi_t,
            density_t=density_np,
            dt=dt,
            stride=stride,
            mode_method=mode_method,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
        )
    else:
        signal = _select_fit_signal(
            phi_t,
            density_np,
            state.selection,
            fit_signal=setup.fit_key,
            mode_method=mode_method,
        )
        t_out = np.arange(signal.shape[0]) * dt * stride
        gamma, omega = _fit_kbm_signal_with_window(
            signal,
            t_out,
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
    return _normalize_growth_rate(
        gamma, omega, setup.params, setup.diagnostic_norm
    )


def _run_kbm_explicit_solver_path(
    setup: _KBMLinearSetup,
    state: _KBMLinearState,
    options: _KBMLinearRunOptions,
) -> LinearRunResult:
    return _paths.run_kbm_explicit_time_path(
        G0_jax=state.state,
        grid=state.grid,
        cache=state.cache,
        params=setup.params,
        geom=setup.geom,
        terms=setup.terms,
        sel=state.selection,
        ky_target=options.ky_target,
        dt=options.dt,
        steps=options.steps,
        time_cfg=options.time_cfg,
        sample_stride=options.sample_stride,
        mode_method=options.mode_method,
        diagnostic_norm=setup.diagnostic_norm,
        auto_window=options.auto_window,
        tmin=options.tmin,
        tmax=options.tmax,
        window_fraction=options.window_fraction,
        min_points=options.min_points,
        start_fraction=options.start_fraction,
        growth_weight=options.growth_weight,
        require_positive=options.require_positive,
        min_amp_fraction=options.min_amp_fraction,
    )


def _run_kbm_krylov_solver_path(
    setup: _KBMLinearSetup,
    state: _KBMLinearState,
    options: _KBMLinearRunOptions,
) -> LinearRunResult:
    return _paths.run_kbm_krylov_path(
        G0_jax=state.state,
        cache=state.cache,
        params=setup.params,
        terms=setup.terms,
        sel=state.selection,
        ky_target=options.ky_target,
        beta_use=setup.beta,
        cfg_use=setup.cfg,
        krylov_cfg=options.krylov_cfg,
        kbm_target_factors=options.kbm_target_factors,
        kbm_beta_transition=options.kbm_beta_transition,
        diagnostic_norm=setup.diagnostic_norm,
    )


def _run_kbm_saved_time_solver_path(
    setup: _KBMLinearSetup,
    state: _KBMLinearState,
    options: _KBMLinearRunOptions,
) -> LinearRunResult:
    phi_t_np, density_np, stride = _integrate_kbm_saved_history(
        state,
        setup,
        time_cfg=options.time_cfg,
        dt=options.dt,
        steps=options.steps,
        method=options.method,
        sample_stride=options.sample_stride,
        density_species_index=options.density_species_index,
        mode_method=options.mode_method,
        show_progress=options.show_progress,
    )
    gamma, omega = _fit_kbm_saved_history(
        state,
        setup,
        phi_t=phi_t_np,
        density_t=density_np,
        dt=options.dt,
        stride=stride,
        mode_method=options.mode_method,
        auto_window=options.auto_window,
        tmin=options.tmin,
        tmax=options.tmax,
        window_fraction=options.window_fraction,
        min_points=options.min_points,
        start_fraction=options.start_fraction,
        growth_weight=options.growth_weight,
        require_positive=options.require_positive,
        min_amp_fraction=options.min_amp_fraction,
    )
    return LinearRunResult(
        t=np.arange(phi_t_np.shape[0]) * options.dt * stride,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(options.ky_target),
        selection=state.selection,
    )


def _kbm_linear_setup_from_request(request: _KBMLinearRequest) -> _KBMLinearSetup:
    return _resolve_kbm_linear_setup(
        cfg=request.cfg,
        beta_value=request.beta_value,
        params=request.params,
        terms=request.terms,
        diagnostic_norm=request.diagnostic_norm,
        fit_signal=request.fit_signal,
        reference_aligned=request.reference_aligned,
        Nm=request.Nm,
        fapar_override=request.fapar_override,
        apar_beta_scale=request.apar_beta_scale,
        ampere_g0_scale=request.ampere_g0_scale,
        bpar_beta_scale=request.bpar_beta_scale,
    )


def _kbm_linear_options_from_request(
    request: _KBMLinearRequest,
) -> _KBMLinearRunOptions:
    return _KBMLinearRunOptions(
        ky_target=float(request.ky_target),
        Nl=request.Nl,
        Nm=request.Nm,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        time_cfg=request.time_cfg,
        krylov_cfg=request.krylov_cfg,
        kbm_target_factors=request.kbm_target_factors,
        kbm_beta_transition=request.kbm_beta_transition,
        auto_window=request.auto_window,
        tmin=request.tmin,
        tmax=request.tmax,
        window_fraction=request.window_fraction,
        min_points=request.min_points,
        start_fraction=request.start_fraction,
        growth_weight=request.growth_weight,
        require_positive=request.require_positive,
        min_amp_fraction=request.min_amp_fraction,
        mode_method=request.mode_method,
        sample_stride=request.sample_stride,
        density_species_index=request.density_species_index,
        show_progress=request.show_progress,
    )


def _run_kbm_linear_request(request: _KBMLinearRequest) -> LinearRunResult:
    setup = _kbm_linear_setup_from_request(request)
    _validate_kbm_species_indices(
        init_species_index=request.init_species_index,
        density_species_index=request.density_species_index,
    )
    state = _prepare_kbm_linear_state(
        setup,
        ky_target=request.ky_target,
        Nl=request.Nl,
        Nm=request.Nm,
        init_species_index=request.init_species_index,
    )
    options = _kbm_linear_options_from_request(request)
    solver_key = select_kbm_solver_auto(
        request.solver,
        ky_target=float(request.ky_target),
        reference_aligned=setup.reference_aligned,
    )
    _paths.sync_path_hooks(globals())
    if solver_key == "explicit_time":
        return _run_kbm_explicit_solver_path(setup, state, options)
    if solver_key == "krylov":
        return _run_kbm_krylov_solver_path(setup, state, options)
    return _run_kbm_saved_time_solver_path(setup, state, options)


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
    show_progress: bool = False,
) -> LinearRunResult:
    """Run a single linear KBM point and return the stored field history."""

    return _run_kbm_linear_request(_kbm_linear_request_from_locals(locals()))
