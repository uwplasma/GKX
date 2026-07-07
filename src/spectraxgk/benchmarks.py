"""Benchmark utilities for documented SPECTRAX-GK comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from spectraxgk.artifacts.restart import write_netcdf_restart_state
from spectraxgk.core.grid import SpectralGrid, build_spectral_grid, select_ky_grid
from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    select_ky_index,
)
import spectraxgk.diagnostics.validation_gates as _gate_metrics
import spectraxgk.diagnostics.zonal_validation as _zonal_validation
from spectraxgk.diagnostics.modes import (
    compare_eigenfunctions,
    load_eigenfunction_reference_bundle,
    normalize_eigenfunction,
    phase_align_eigenfunction,
    save_eigenfunction_reference_bundle,
)
from spectraxgk.diagnostics.validation_gates import (
    _analytic_signal,
    _explicit_time_window,
    _leading_window,
    BranchContinuationMetrics,
    DiagnosticTimeSeries,
    EigenfunctionComparisonMetrics,
    EigenfunctionReferenceBundle,
    GateReport,
    LateTimeLinearMetrics,
    NonlinearHeatFluxConvergenceMetrics,
    NonlinearWindowMetrics,
    ObservedOrderMetrics,
    ScalarGateResult,
    ZonalFlowResponseMetrics,
    branch_continuity_gate_report,
    eigenfunction_gate_report,
    evaluate_scalar_gate,
    gate_report,
    gate_report_to_dict,
    infer_triple_dealiased_ny,
    late_time_window,
    linear_metrics_gate_report,
    load_diagnostic_time_series,
    nonlinear_heat_flux_convergence_gate_report,
    nonlinear_window_gate_report,
    observed_order_gate_report,
    zonal_response_gate_report,
)
from spectraxgk.geometry import (
    SAlphaGeometry,
    apply_geometry_grid_defaults,
    build_flux_tube_geometry,
)
from spectraxgk.linear import integrate_linear, integrate_linear_diagnostics
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from spectraxgk.runtime import run_runtime_linear, run_runtime_nonlinear
from spectraxgk.solvers.linear.krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.solvers.time.explicit import ExplicitTimeConfig
from spectraxgk.solvers.time.runners import integrate_linear_from_config
from spectraxgk.terms.assembly import compute_fields_cached
from spectraxgk.workflows.runtime.config import RuntimeConfig, RuntimeExpertConfig

from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    KBMBaseCase,
    KineticElectronBaseCase,
    TEMBaseCase,
    TimeConfig,
)
from spectraxgk.validation.benchmarks.defaults import (
    CYCLONE_KRYLOV_DEFAULT,
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    ETG_KRYLOV_DEFAULT,
    ETG_OMEGA_D_SCALE,
    ETG_OMEGA_STAR_SCALE,
    ETG_RHO_STAR,
    KBM_KRYLOV_DEFAULT,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
    KINETIC_KRYLOV_DEFAULT,
    KINETIC_KRYLOV_REFERENCE_ALIGNED,
    KINETIC_OMEGA_D_SCALE,
    KINETIC_OMEGA_STAR_SCALE,
    KINETIC_RHO_STAR,
    TEM_KRYLOV_DEFAULT,
    TEM_OMEGA_D_SCALE,
    TEM_OMEGA_STAR_SCALE,
    TEM_RHO_STAR,
)
from spectraxgk.validation.benchmarks.defaults import (
    _is_array_like,
    _iter_ky_batches,
    _resolve_streaming_window,
    indexed_scan_value,
    scan_window_valid,
)
from spectraxgk.diagnostics.growth_rates import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _score_fit_signal_auto,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.defaults import (
    _build_gaussian_profile,
    _build_initial_condition,
    _kinetic_reference_init_cfg,
)
from spectraxgk.validation.benchmarks.defaults import (
    CycloneComparison,
    CycloneReference,
    CycloneRunResult,
    CycloneScanResult,
    LinearRunResult,
    LinearScanResult,
    _load_reference_with_header,
    compare_cyclone_to_reference,
    load_cyclone_reference,
    load_cyclone_reference_kinetic,
    load_etg_reference,
    load_kbm_reference,
    load_tem_reference,
)
from spectraxgk.validation.benchmarks.defaults import (
    KBM_EXPLICIT_SOLVER_LOCK,
    KBM_EXPLICIT_SOLVER_LOCK_TOL,
    _kbm_use_multi_target_krylov,
    _midplane_index,
    select_kbm_solver_auto,
)
from spectraxgk.validation.benchmarks.defaults import (
    REFERENCE_DAMP_ENDS_AMP,
    REFERENCE_DAMP_ENDS_WIDTHFRAC,
    REFERENCE_NU_HYPER_L,
    REFERENCE_NU_HYPER_M,
    REFERENCE_P_HYPER_L,
    REFERENCE_P_HYPER_M,
    _apply_reference_hypercollisions,
    _electron_only_params,
    _linked_boundary_end_damping,
    _reference_hypercollision_power,
    _two_species_params,
)

from spectraxgk.validation.benchmarks.cyclone_linear import run_cyclone_linear
from spectraxgk.validation.benchmarks.cyclone_scan import run_cyclone_scan

from spectraxgk.validation.benchmarks.kbm_beta import run_kbm_beta_scan
from spectraxgk.validation.benchmarks.kbm_linear import run_kbm_linear

from spectraxgk.validation.benchmarks.kinetic_scan import run_kinetic_scan

from spectraxgk.validation.benchmarks.etg_linear import run_etg_linear
from spectraxgk.validation.benchmarks.etg_scan import run_etg_scan

from spectraxgk.validation.benchmarks.tem import (
    run_tem_linear,
    run_tem_scan,
)


@dataclass(frozen=True)
class _KineticLinearSetup:
    cfg: KineticElectronBaseCase
    grid_full: Any
    geom: Any
    params: LinearParams
    terms: LinearTerms
    init_cfg: Any
    diagnostic_norm: str
    reference_aligned: bool


@dataclass(frozen=True)
class _KineticLinearState:
    grid: Any
    selection: ModeSelection
    state: jnp.ndarray


@dataclass(frozen=True)
class _KineticHistory:
    t: np.ndarray
    phi_t: np.ndarray
    density_t: np.ndarray | None


@dataclass(frozen=True)
class _KineticFitOptions:
    fit_signal: str
    mode_method: str
    auto_window: bool
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float


@dataclass(frozen=True)
class _KineticTimePathOptions:
    time_cfg: TimeConfig | None
    dt: float
    steps: int
    method: str
    sample_stride: int | None
    density_species_index: int
    show_progress: bool
    n_laguerre: int
    n_hermite: int
    fit: _KineticFitOptions


def _resolve_kinetic_linear_setup(
    *,
    cfg: KineticElectronBaseCase | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    diagnostic_norm: str,
    reference_aligned: bool | None,
    Nm: int,
) -> _KineticLinearSetup:
    """Resolve kinetic benchmark setup shared by Krylov and time paths."""

    cfg_use = cfg or KineticElectronBaseCase()
    grid_full = build_spectral_grid(cfg_use.grid)
    geom = SAlphaGeometry.from_config(cfg_use.geometry)
    reference_aligned_use = bool(
        True if reference_aligned is None else reference_aligned
    )
    diagnostic_norm_use = diagnostic_norm
    if reference_aligned_use and diagnostic_norm_use == "none":
        diagnostic_norm_use = "rho_star"
    init_cfg_use = _kinetic_reference_init_cfg(
        cfg_use.init, reference_aligned=reference_aligned_use
    )
    damp_ends_amp, damp_ends_widthfrac = _linked_boundary_end_damping(
        reference_aligned_use
    )
    params_use = params
    if params_use is None:
        params_use = _two_species_params(
            cfg_use.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KINETIC_OMEGA_D_SCALE,
            omega_star_scale=KINETIC_OMEGA_STAR_SCALE,
            rho_star=KINETIC_RHO_STAR,
            damp_ends_amp=damp_ends_amp,
            damp_ends_widthfrac=damp_ends_widthfrac,
            nhermite=Nm,
        )
        if reference_aligned_use:
            params_use = _apply_reference_hypercollisions(params_use, nhermite=Nm)
    terms_use = terms if terms is not None else LinearTerms(bpar=0.0)
    return _KineticLinearSetup(
        cfg=cfg_use,
        grid_full=grid_full,
        geom=geom,
        params=params_use,
        terms=terms_use,
        init_cfg=init_cfg_use,
        diagnostic_norm=diagnostic_norm_use,
        reference_aligned=reference_aligned_use,
    )


def _validate_kinetic_species_indices(
    *, init_species_index: int, density_species_index: int, nspecies: int = 2
) -> None:
    """Validate the kinetic two-species index contract."""

    if init_species_index < 0 or init_species_index >= nspecies:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= nspecies:
        raise ValueError("density_species_index out of range for kinetic species")


def _build_kinetic_linear_state(
    setup: _KineticLinearSetup,
    *,
    ky_target: float,
    Nl: int,
    Nm: int,
    init_species_index: int,
    density_species_index: int,
) -> _KineticLinearState:
    """Select the ky grid and build the kinetic initial perturbation."""

    ky_index = select_ky_index(np.asarray(setup.grid_full.ky), ky_target)
    grid = select_ky_grid(setup.grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    nspecies = 2
    _validate_kinetic_species_indices(
        init_species_index=init_species_index,
        density_species_index=density_species_index,
        nspecies=nspecies,
    )
    G0 = np.zeros(
        (nspecies, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=np.complex64,
    )
    G0_single = _build_initial_condition(
        grid,
        setup.geom,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=setup.init_cfg,
    )
    G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)
    return _KineticLinearState(grid=grid, selection=sel, state=jnp.asarray(G0))


def _prepare_kinetic_linear_setup_and_state(
    *,
    cfg: KineticElectronBaseCase | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    diagnostic_norm: str,
    reference_aligned: bool | None,
    ky_target: float,
    n_laguerre: int,
    n_hermite: int,
    init_species_index: int,
    density_species_index: int,
) -> tuple[_KineticLinearSetup, _KineticLinearState]:
    """Resolve the kinetic benchmark setup and selected-ky initial state."""

    setup = _resolve_kinetic_linear_setup(
        cfg=cfg,
        params=params,
        terms=terms,
        diagnostic_norm=diagnostic_norm,
        reference_aligned=reference_aligned,
        Nm=n_hermite,
    )
    state = _build_kinetic_linear_state(
        setup,
        ky_target=ky_target,
        Nl=n_laguerre,
        Nm=n_hermite,
        init_species_index=init_species_index,
        density_species_index=density_species_index,
    )
    return setup, state


def _kinetic_krylov_config(
    setup: _KineticLinearSetup,
    krylov_cfg: KrylovConfig | None,
) -> KrylovConfig:
    """Return the kinetic Krylov policy, including reference-aligned defaults."""

    if krylov_cfg is not None:
        return krylov_cfg
    if setup.reference_aligned:
        return KINETIC_KRYLOV_REFERENCE_ALIGNED
    return KINETIC_KRYLOV_DEFAULT


def _run_kinetic_krylov_path(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    *,
    Nl: int,
    Nm: int,
    krylov_cfg: KrylovConfig | None,
) -> LinearRunResult:
    """Solve one kinetic benchmark point with the Krylov eigenpath."""

    cfg_use = _kinetic_krylov_config(setup, krylov_cfg)
    cache = build_linear_cache(state.grid, setup.geom, setup.params, Nl, Nm)
    eig, vec = dominant_eigenpair(
        state.state,
        cache,
        setup.params,
        terms=setup.terms,
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
    term_cfg = linear_terms_to_term_config(setup.terms)
    phi = compute_fields_cached(vec, cache, setup.params, terms=term_cfg).phi
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    gamma, omega = _normalize_growth_rate(
        gamma, omega, setup.params, setup.diagnostic_norm
    )
    return _pack_kinetic_result(
        state,
        t=np.array([0.0]),
        phi_t=np.asarray(phi)[None, ...],
        gamma=gamma,
        omega=omega,
    )


def _resolve_time_config(
    time_cfg: TimeConfig | None,
    *,
    sample_stride: int | None,
) -> TimeConfig | None:
    """Apply user sample-stride override without changing time-config semantics."""

    if time_cfg is None or sample_stride is None:
        return time_cfg
    return replace(time_cfg, sample_stride=sample_stride)


def _integrate_configured_kinetic_history(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    *,
    time_cfg: TimeConfig,
    method_key: str,
    fit_signal: str,
    density_species_index: int,
    Nl: int,
    Nm: int,
) -> tuple[Any, Any | None, float, int]:
    """Integrate kinetic time history with an explicit runtime TimeConfig."""

    dt = float(time_cfg.dt)
    steps = int(round(time_cfg.t_max / time_cfg.dt))
    cache = build_linear_cache(state.grid, setup.geom, setup.params, Nl, Nm)
    if time_cfg.use_diffrax and not (
        method_key.startswith("imex") or method_key.startswith("implicit")
    ):
        save_field = "density" if fit_signal == "density" else "phi"
        _, phi_t = integrate_linear_from_config(
            state.state,
            state.grid,
            setup.geom,
            setup.params,
            time_cfg,
            cache=cache,
            terms=setup.terms,
            save_field=save_field,
            density_species_index=density_species_index
            if fit_signal == "density"
            else None,
        )
        density_t = phi_t if fit_signal == "density" else None
        return phi_t, density_t, dt, time_cfg.sample_stride

    if fit_signal == "density":
        diag = integrate_linear_diagnostics(
            state.state,
            state.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=time_cfg.method,
            cache=cache,
            terms=setup.terms,
            sample_stride=time_cfg.sample_stride,
            species_index=density_species_index,
        )
        phi_t = diag[1]
        density_t = diag[2] if len(diag) > 2 else None
        return phi_t, density_t, dt, time_cfg.sample_stride

    _, phi_t = integrate_linear_from_config(
        state.state,
        state.grid,
        setup.geom,
        setup.params,
        time_cfg,
        cache=cache,
        terms=setup.terms,
        density_species_index=density_species_index if fit_signal == "density" else None,
    )
    return phi_t, None, dt, time_cfg.sample_stride


def _integrate_unconfigured_kinetic_history(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    *,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    fit_signal: str,
    density_species_index: int,
    show_progress: bool,
) -> tuple[Any, Any | None, float, int]:
    """Integrate kinetic time history without a runtime TimeConfig."""

    stride = 1 if sample_stride is None else int(sample_stride)
    if fit_signal == "density":
        diag = integrate_linear_diagnostics(
            state.state,
            state.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=method,
            terms=setup.terms,
            sample_stride=stride,
            species_index=density_species_index,
        )
        phi_t = diag[1]
        density_t = diag[2] if len(diag) > 2 else None
        return phi_t, density_t, float(dt), stride

    _, phi_t = integrate_linear(
        state.state,
        state.grid,
        setup.geom,
        setup.params,
        dt=dt,
        steps=steps,
        method=method,
        terms=setup.terms,
        sample_stride=stride,
        show_progress=show_progress,
    )
    return phi_t, None, float(dt), stride


def _integrate_kinetic_history(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    *,
    time_cfg: TimeConfig | None,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    fit_signal: str,
    density_species_index: int,
    show_progress: bool,
    Nl: int,
    Nm: int,
) -> _KineticHistory:
    """Integrate a kinetic time history and preserve saved-observable semantics."""

    time_cfg_use = _resolve_time_config(time_cfg, sample_stride=sample_stride)
    if time_cfg_use is not None:
        phi_t, density_t, dt_eff, stride = _integrate_configured_kinetic_history(
            setup,
            state,
            time_cfg=time_cfg_use,
            method_key=method.lower(),
            fit_signal=fit_signal,
            density_species_index=density_species_index,
            Nl=Nl,
            Nm=Nm,
        )
    else:
        phi_t, density_t, dt_eff, stride = _integrate_unconfigured_kinetic_history(
            setup,
            state,
            dt=dt,
            steps=steps,
            method=method,
            sample_stride=sample_stride,
            fit_signal=fit_signal,
            density_species_index=density_species_index,
            show_progress=show_progress,
        )
    phi_t_np = np.asarray(phi_t)
    t = np.arange(phi_t_np.shape[0]) * dt_eff * stride
    density_np = None if density_t is None else np.asarray(density_t)
    return _KineticHistory(t=t, phi_t=phi_t_np, density_t=density_np)


def _fit_kinetic_history(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    history: _KineticHistory,
    *,
    options: _KineticFitOptions,
) -> tuple[float, float]:
    """Fit growth/frequency from a saved kinetic time history."""

    signal = _select_fit_signal(
        history.phi_t,
        history.density_t,
        state.selection,
        fit_signal=options.fit_signal,
        mode_method=options.mode_method,
    )
    use_auto = options.auto_window and options.tmin is None and options.tmax is None
    if not use_auto and not scan_window_valid(history.t, options.tmin, options.tmax):
        use_auto = True
    auto_fit_kwargs: dict[str, Any] = {
        "window_fraction": options.window_fraction,
        "min_points": options.min_points,
        "start_fraction": options.start_fraction,
        "growth_weight": options.growth_weight,
        "require_positive": options.require_positive,
        "min_amp_fraction": options.min_amp_fraction,
    }
    if use_auto:
        gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
            history.t, signal, **auto_fit_kwargs
        )
    else:
        try:
            gamma, omega = fit_growth_rate(
                history.t, signal, tmin=options.tmin, tmax=options.tmax
            )
        except ValueError:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                history.t, signal, **auto_fit_kwargs
            )
    return _normalize_growth_rate(
        gamma, omega, setup.params, setup.diagnostic_norm
    )


def _run_kinetic_time_path(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    *,
    options: _KineticTimePathOptions,
) -> LinearRunResult:
    """Run and fit the saved-time kinetic benchmark path."""

    history = _integrate_kinetic_history(
        setup,
        state,
        time_cfg=options.time_cfg,
        dt=options.dt,
        steps=options.steps,
        method=options.method,
        sample_stride=options.sample_stride,
        fit_signal=options.fit.fit_signal,
        density_species_index=options.density_species_index,
        show_progress=options.show_progress,
        Nl=options.n_laguerre,
        Nm=options.n_hermite,
    )
    gamma, omega = _fit_kinetic_history(
        setup,
        state,
        history,
        options=options.fit,
    )
    return _pack_kinetic_result(
        state, t=history.t, phi_t=history.phi_t, gamma=gamma, omega=omega
    )


def _pack_kinetic_result(
    state: _KineticLinearState,
    *,
    t: np.ndarray,
    phi_t: np.ndarray,
    gamma: float,
    omega: float,
) -> LinearRunResult:
    """Pack the public kinetic linear benchmark result."""

    return LinearRunResult(
        t=t,
        phi_t=phi_t,
        gamma=gamma,
        omega=omega,
        ky=float(state.grid.ky[state.selection.ky_index]),
        selection=state.selection,
    )


def _kinetic_time_path_options(
    *,
    time_cfg: TimeConfig | None,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    density_species_index: int,
    show_progress: bool,
    n_laguerre: int,
    n_hermite: int,
    fit_signal: str,
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
) -> _KineticTimePathOptions:
    """Pack public time-path keyword controls into the internal request object."""

    return _KineticTimePathOptions(
        time_cfg,
        dt,
        steps,
        method,
        sample_stride,
        density_species_index,
        show_progress,
        n_laguerre,
        n_hermite,
        _KineticFitOptions(
            fit_signal,
            mode_method,
            auto_window,
            tmin,
            tmax,
            window_fraction,
            min_points,
            start_fraction,
            growth_weight,
            require_positive,
            min_amp_fraction,
        ),
    )


def run_kinetic_linear(
    ky_target: float = 0.3,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: KineticElectronBaseCase | None = None,
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
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    fit_signal: str = "density",
    init_species_index: int = 1,
    density_species_index: int = 1,
    diagnostic_norm: str = "none",
    reference_aligned: bool | None = True,
    show_progress: bool = False,
) -> LinearRunResult:
    """Run a kinetic-electron ITG/TEM benchmark and extract growth rate."""

    setup, state = _prepare_kinetic_linear_setup_and_state(
        cfg=cfg,
        params=params,
        terms=terms,
        diagnostic_norm=diagnostic_norm,
        reference_aligned=reference_aligned,
        ky_target=ky_target,
        n_laguerre=Nl,
        n_hermite=Nm,
        init_species_index=init_species_index,
        density_species_index=density_species_index,
    )
    if solver.lower() == "krylov":
        return _run_kinetic_krylov_path(
            setup,
            state,
            Nl=Nl,
            Nm=Nm,
            krylov_cfg=krylov_cfg,
        )
    return _run_kinetic_time_path(
        setup,
        state,
        options=_kinetic_time_path_options(
            time_cfg=time_cfg,
            dt=dt,
            steps=steps,
            method=method,
            sample_stride=sample_stride,
            density_species_index=density_species_index,
            show_progress=show_progress,
            n_laguerre=Nl,
            n_hermite=Nm,
            fit_signal=fit_signal,
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
        ),
    )

@dataclass(frozen=True)
class _KBMScanCase:
    cfg: KBMBaseCase
    beta: float
    reference_aligned: bool


@dataclass(frozen=True)
class _KBMScanOptions:
    n_laguerre: int
    n_hermite: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    time_cfg: TimeConfig | None
    solver: str
    krylov_cfg: KrylovConfig | None
    kbm_target_factors: Sequence[float] | None
    kbm_beta_transition: float | None
    tmin: float | np.ndarray | None
    tmax: float | np.ndarray | None
    auto_window: bool
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    mode_method: str
    mode_only: bool
    terms: LinearTerms | None
    sample_stride: int | None
    fit_signal: str
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    init_species_index: int
    density_species_index: int
    diagnostic_norm: str
    fapar_override: float | None
    apar_beta_scale: float | None
    ampere_g0_scale: float | None
    bpar_beta_scale: float | None


@dataclass(frozen=True)
class _KBMScanRequest:
    ky_values: np.ndarray
    beta_value: float | None
    Nl: int
    Nm: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    cfg: KBMBaseCase | None
    time_cfg: TimeConfig | None
    solver: str
    krylov_cfg: KrylovConfig | None
    kbm_target_factors: Sequence[float] | None
    kbm_beta_transition: float | None
    tmin: float | np.ndarray | None
    tmax: float | np.ndarray | None
    auto_window: bool
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    mode_method: str
    mode_only: bool
    terms: LinearTerms | None
    sample_stride: int | None
    fit_signal: str
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    init_species_index: int
    density_species_index: int
    diagnostic_norm: str
    fapar_override: float | None
    apar_beta_scale: float | None
    ampere_g0_scale: float | None
    bpar_beta_scale: float | None
    reference_aligned: bool | None


@dataclass
class _KBMScanOutput:
    ky: list[float]
    gamma: list[float]
    omega: list[float]

    @classmethod
    def empty(cls) -> "_KBMScanOutput":
        return cls(ky=[], gamma=[], omega=[])

    def append(self, *, ky: float, gamma: float, omega: float) -> None:
        self.ky.append(float(ky))
        self.gamma.append(float(gamma))
        self.omega.append(float(omega))

    def result(self) -> LinearScanResult:
        return LinearScanResult(
            ky=np.asarray(self.ky, dtype=float),
            gamma=np.asarray(self.gamma, dtype=float),
            omega=np.asarray(self.omega, dtype=float),
        )


def _kbm_scan_request_from_locals(values: dict[str, Any]) -> _KBMScanRequest:
    """Pack public ``run_kbm_scan`` arguments once for internal routing."""

    return _KBMScanRequest(
        **{field.name: values[field.name] for field in fields(_KBMScanRequest)}
    )


def _resolve_kbm_scan_case(
    *,
    cfg: KBMBaseCase | None,
    beta_value: float | None,
    reference_aligned: bool | None,
) -> _KBMScanCase:
    cfg_in = cfg or KBMBaseCase()
    beta_use = float(cfg_in.model.beta) if beta_value is None else float(beta_value)
    return _KBMScanCase(
        cfg=replace(cfg_in, model=replace(cfg_in.model, beta=beta_use)),
        beta=beta_use,
        reference_aligned=bool(True if reference_aligned is None else reference_aligned),
    )


def _build_kbm_scan_options(
    *,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    time_cfg: TimeConfig | None,
    solver: str,
    krylov_cfg: KrylovConfig | None,
    kbm_target_factors: Sequence[float] | None,
    kbm_beta_transition: float | None,
    tmin: float | np.ndarray | None,
    tmax: float | np.ndarray | None,
    auto_window: bool,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    mode_method: str,
    mode_only: bool,
    terms: LinearTerms | None,
    sample_stride: int | None,
    fit_signal: str,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    init_species_index: int,
    density_species_index: int,
    diagnostic_norm: str,
    fapar_override: float | None,
    apar_beta_scale: float | None,
    ampere_g0_scale: float | None,
    bpar_beta_scale: float | None,
) -> _KBMScanOptions:
    return _KBMScanOptions(
        n_laguerre=Nl,
        n_hermite=Nm,
        dt=dt,
        steps=steps,
        method=method,
        time_cfg=time_cfg,
        solver=solver,
        krylov_cfg=krylov_cfg,
        kbm_target_factors=kbm_target_factors,
        kbm_beta_transition=kbm_beta_transition,
        tmin=tmin,
        tmax=tmax,
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
    )


def _build_kbm_scan_options_from_request(request: _KBMScanRequest) -> _KBMScanOptions:
    return _build_kbm_scan_options(
        Nl=request.Nl,
        Nm=request.Nm,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        time_cfg=request.time_cfg,
        solver=request.solver,
        krylov_cfg=request.krylov_cfg,
        kbm_target_factors=request.kbm_target_factors,
        kbm_beta_transition=request.kbm_beta_transition,
        tmin=request.tmin,
        tmax=request.tmax,
        auto_window=request.auto_window,
        window_fraction=request.window_fraction,
        min_points=request.min_points,
        start_fraction=request.start_fraction,
        growth_weight=request.growth_weight,
        require_positive=request.require_positive,
        min_amp_fraction=request.min_amp_fraction,
        mode_method=request.mode_method,
        mode_only=request.mode_only,
        terms=request.terms,
        sample_stride=request.sample_stride,
        fit_signal=request.fit_signal,
        ky_batch=request.ky_batch,
        fixed_batch_shape=request.fixed_batch_shape,
        streaming_fit=request.streaming_fit,
        streaming_amp_floor=request.streaming_amp_floor,
        init_species_index=request.init_species_index,
        density_species_index=request.density_species_index,
        diagnostic_norm=request.diagnostic_norm,
        fapar_override=request.fapar_override,
        apar_beta_scale=request.apar_beta_scale,
        ampere_g0_scale=request.ampere_g0_scale,
        bpar_beta_scale=request.bpar_beta_scale,
    )


def _indexed_kbm_scan_time_value(value: float | np.ndarray, index: int):
    indexed = indexed_scan_value(value, index)
    return value if indexed is None else indexed


def _run_kbm_scan_point(
    *,
    ky_value: float,
    index: int,
    case: _KBMScanCase,
    options: _KBMScanOptions,
) -> tuple[float, float]:
    dt_i = _indexed_kbm_scan_time_value(options.dt, index)
    steps_i = _indexed_kbm_scan_time_value(options.steps, index)
    out = run_kbm_beta_scan(
        betas=np.asarray([case.beta], dtype=float),
        ky_target=float(ky_value),
        Nl=options.n_laguerre,
        Nm=options.n_hermite,
        dt=float(dt_i),
        steps=int(steps_i),
        method=options.method,
        cfg=case.cfg,
        time_cfg=options.time_cfg,
        solver=options.solver,
        krylov_cfg=options.krylov_cfg,
        kbm_target_factors=options.kbm_target_factors,
        kbm_beta_transition=options.kbm_beta_transition,
        tmin=indexed_scan_value(options.tmin, index),
        tmax=indexed_scan_value(options.tmax, index),
        auto_window=options.auto_window,
        window_fraction=options.window_fraction,
        min_points=options.min_points,
        start_fraction=options.start_fraction,
        growth_weight=options.growth_weight,
        require_positive=options.require_positive,
        min_amp_fraction=options.min_amp_fraction,
        mode_method=options.mode_method,
        mode_only=options.mode_only,
        terms=options.terms,
        sample_stride=options.sample_stride,
        fit_signal=options.fit_signal,
        ky_batch=options.ky_batch,
        fixed_batch_shape=options.fixed_batch_shape,
        streaming_fit=options.streaming_fit,
        streaming_amp_floor=options.streaming_amp_floor,
        init_species_index=options.init_species_index,
        density_species_index=options.density_species_index,
        diagnostic_norm=options.diagnostic_norm,
        fapar_override=options.fapar_override,
        apar_beta_scale=options.apar_beta_scale,
        ampere_g0_scale=options.ampere_g0_scale,
        bpar_beta_scale=options.bpar_beta_scale,
        reference_aligned=case.reference_aligned,
    )
    return float(out.gamma[0]), float(out.omega[0])


def _run_kbm_scan_loop(
    ky_values: np.ndarray,
    *,
    case: _KBMScanCase,
    options: _KBMScanOptions,
) -> _KBMScanOutput:
    output = _KBMScanOutput.empty()
    for index, ky_value in enumerate(np.asarray(ky_values, dtype=float)):
        gamma, omega = _run_kbm_scan_point(
            ky_value=float(ky_value),
            index=index,
            case=case,
            options=options,
        )
        output.append(ky=float(ky_value), gamma=gamma, omega=omega)
    return output


def _run_kbm_scan_request(request: _KBMScanRequest) -> LinearScanResult:
    case = _resolve_kbm_scan_case(
        cfg=request.cfg,
        beta_value=request.beta_value,
        reference_aligned=request.reference_aligned,
    )
    options = _build_kbm_scan_options_from_request(request)
    return _run_kbm_scan_loop(request.ky_values, case=case, options=options).result()


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
) -> LinearScanResult:
    """Run a KBM ky scan at fixed beta.

    This is a thin wrapper over :func:`run_kbm_beta_scan` used for
    reference-comparison workflows where the external benchmark is a ky scan
    at fixed beta.
    """

    return _run_kbm_scan_request(_kbm_scan_request_from_locals(locals()))



@dataclass(frozen=True)
class SecondaryModeResult:
    """Per-mode nonlinear secondary-instability diagnostic summary."""

    ky: float
    kx: float
    gamma: float
    omega: float


def _leading_finite_prefix(
    t: ArrayLike,
    signal: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the leading finite segment of a complex diagnostic mode."""

    t_arr = np.asarray(t, dtype=float)
    sig_arr = np.asarray(signal, dtype=np.complex128)
    finite = np.isfinite(sig_arr)
    if not np.any(finite):
        return t_arr[:0], sig_arr[:0]
    first_bad = np.where(~finite)[0]
    stop = int(first_bad[0]) if first_bad.size else int(sig_arr.size)
    return t_arr[:stop], sig_arr[:stop]


def _tail_mean_pair(
    gamma_t: ArrayLike,
    omega_t: ArrayLike,
    *,
    tail_fraction: float | None,
) -> tuple[float, float] | None:
    """Return a late-time average from finite gamma/omega diagnostic series."""

    gamma_arr = np.asarray(gamma_t, dtype=float)
    omega_arr = np.asarray(omega_t, dtype=float)
    finite = np.isfinite(gamma_arr) & np.isfinite(omega_arr)
    if not np.any(finite):
        return None
    gamma_finite = gamma_arr[finite]
    omega_finite = omega_arr[finite]
    if tail_fraction is None:
        return float(gamma_finite[-1]), float(omega_finite[-1])
    istart = int(len(gamma_finite) * (1.0 - float(tail_fraction)))
    istart = max(0, min(istart, len(gamma_finite) - 1))
    return float(np.mean(gamma_finite[istart:])), float(np.mean(omega_finite[istart:]))


def write_restart_state(path: str | Path, state: np.ndarray) -> Path:
    """Write a complex restart state in the runtime NetCDF layout."""

    return write_netcdf_restart_state(path, state)


def _embed_linear_seed_on_full_grid(
    cfg: RuntimeConfig,
    state: np.ndarray,
    *,
    ky_target: float,
) -> np.ndarray:
    """Embed a selected-ky linear seed into the full nonlinear runtime grid."""

    geom = build_flux_tube_geometry(cfg.geometry)
    grid_cfg = apply_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    full_shape = (
        state.shape[0],
        state.shape[1],
        state.shape[2],
        grid.ky.size,
        grid.kx.size,
        grid.z.size,
    )
    if tuple(state.shape) == full_shape:
        return np.asarray(state, dtype=np.complex64)
    if state.ndim != 6 or state.shape[3] != 1:
        raise ValueError(
            f"expected selected-ky linear state with shape (..., 1, Nx, Nz), got {state.shape}"
        )
    ky = np.asarray(grid.ky, dtype=float)
    ky_idx = int(np.argmin(np.abs(ky - float(ky_target))))
    full_state = np.zeros(full_shape, dtype=np.complex64)
    full_state[..., ky_idx : ky_idx + 1, :, :] = np.asarray(state, dtype=np.complex64)
    return full_state


def run_secondary_seed(
    cfg: RuntimeConfig,
    *,
    restart_path: str | Path,
    ky_target: float,
    Nl: int,
    Nm: int,
    dt: float = 1.0,
    steps: int = 2,
    method: str = "sspx3",
    solver: str = "time",
) -> Path:
    """Run the linear secondary seed stage and write its final restart state."""

    result = run_runtime_linear(
        cfg,
        ky_target=ky_target,
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        method=method,
        dt=dt,
        steps=steps,
        return_state=True,
    )
    if result.state is None:
        raise RuntimeError("Secondary seed run did not return a final state.")
    state_full = _embed_linear_seed_on_full_grid(cfg, result.state, ky_target=ky_target)
    return write_restart_state(restart_path, state_full)


def build_secondary_stage2_config(
    cfg: RuntimeConfig,
    *,
    restart_file: str | Path,
    restart_scale: float = 500.0,
    init_amp: float = 1.0e-5,
    dt: float = 0.01,
    t_max: float = 100.0,
    method: str = "sspx3",
    iky_fixed: int = 1,
    ikx_fixed: int = 0,
) -> RuntimeConfig:
    """Return a nonlinear stage-2 config for the secondary slab benchmark."""

    time_cfg = replace(
        cfg.time,
        t_max=float(t_max),
        dt=float(dt),
        method=str(method),
        use_diffrax=False,
        fixed_dt=True,
    )
    init_cfg = replace(
        cfg.init,
        init_amp=float(init_amp),
        init_single=False,
        init_file=str(restart_file),
        init_file_scale=float(restart_scale),
        init_file_mode="add",
    )
    physics_cfg = replace(cfg.physics, linear=False, nonlinear=True)
    terms_cfg = replace(cfg.terms, nonlinear=1.0)
    expert_cfg = RuntimeExpertConfig(
        fixed_mode=True, iky_fixed=int(iky_fixed), ikx_fixed=int(ikx_fixed)
    )
    return replace(
        cfg,
        time=time_cfg,
        init=init_cfg,
        physics=physics_cfg,
        terms=terms_cfg,
        expert=expert_cfg,
    )


def run_secondary_modes(
    cfg: RuntimeConfig,
    *,
    modes: Sequence[tuple[float, float]],
    Nl: int,
    Nm: int,
    steps: int | None = None,
    sample_stride: int = 100,
    fit_fraction: float | None = 0.5,
) -> list[SecondaryModeResult]:
    """Run one nonlinear secondary stage per requested diagnostic mode."""

    rows: list[SecondaryModeResult] = []
    for ky_target, kx_target in modes:
        result = run_runtime_nonlinear(
            cfg,
            ky_target=float(ky_target),
            kx_target=float(kx_target),
            Nl=Nl,
            Nm=Nm,
            steps=steps,
            sample_stride=sample_stride,
        )
        if result.diagnostics is None:
            raise RuntimeError("Secondary nonlinear run did not produce diagnostics.")
        tail_mean = _tail_mean_pair(
            result.diagnostics.gamma_t,
            result.diagnostics.omega_t,
            tail_fraction=fit_fraction,
        )
        gamma = float(tail_mean[0]) if tail_mean is not None else 0.0
        omega = float(tail_mean[1]) if tail_mean is not None else 0.0
        phi_mode_t = result.diagnostics.phi_mode_t
        if fit_fraction is not None and phi_mode_t is not None:
            t, signal = _leading_finite_prefix(result.diagnostics.t, phi_mode_t)
            if t.size >= 2 and np.max(np.abs(signal)) > 0.0:
                t_span = float(t[-1] - t[0]) if t.size > 1 else 0.0
                tmin = (
                    float(t[0] + (1.0 - float(fit_fraction)) * t_span)
                    if t_span > 0.0
                    else None
                )
                try:
                    gamma_fit, omega_fit = fit_growth_rate(t, signal, tmin=tmin)
                    gamma = float(gamma_fit)
                    if tail_mean is None:
                        omega = float(omega_fit)
                except ValueError:
                    gamma = float(tail_mean[0]) if tail_mean is not None else 0.0
                    omega = float(tail_mean[1]) if tail_mean is not None else 0.0
        rows.append(
            SecondaryModeResult(
                ky=float(ky_target),
                kx=float(kx_target),
                gamma=float(gamma),
                omega=float(omega),
            )
        )
    return rows



def _sync_metric_hooks() -> None:
    _gate_metrics.extract_mode_time_series = extract_mode_time_series
    _gate_metrics.fit_growth_rate = fit_growth_rate


def zonal_flow_response_metrics(*args: Any, **kwargs: Any) -> ZonalFlowResponseMetrics:
    """Estimate residual level and GAM envelope metrics from a zonal response."""

    return _zonal_validation.zonal_flow_response_metrics(*args, **kwargs)


def late_time_linear_metrics(*args: Any, **kwargs: Any) -> LateTimeLinearMetrics:
    """Return late-time growth/frequency metrics from a linear result."""

    _sync_metric_hooks()
    return _gate_metrics.late_time_linear_metrics(*args, **kwargs)


def windowed_nonlinear_metrics(*args: Any, **kwargs: Any) -> NonlinearWindowMetrics:
    """Return late-window transport metrics from nonlinear diagnostics."""

    return _gate_metrics.windowed_nonlinear_metrics(*args, **kwargs)


def nonlinear_heat_flux_convergence_metrics(
    *args: Any, **kwargs: Any
) -> NonlinearHeatFluxConvergenceMetrics:
    """Summarize post-transient heat-flux average stability."""

    return _gate_metrics.nonlinear_heat_flux_convergence_metrics(*args, **kwargs)


def estimate_observed_order(*args: Any, **kwargs: Any) -> ObservedOrderMetrics:
    """Estimate observed order from step-size refinements."""

    return _gate_metrics.estimate_observed_order(*args, **kwargs)


def branch_continuity_metrics(*args: Any, **kwargs: Any) -> BranchContinuationMetrics:
    """Compute branch-continuity diagnostics for a linear scan."""

    return _gate_metrics.branch_continuity_metrics(*args, **kwargs)


@dataclass(frozen=True)
class ScanAndModeResult:
    scan: LinearScanResult
    eigenfunction: np.ndarray
    grid: SpectralGrid
    ky_selected: float
    tmin: float | None
    tmax: float | None


def run_linear_scan(
    *,
    ky_values: np.ndarray,
    run_linear_fn,
    cfg,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    solver: str,
    krylov_cfg,
    window_kw: dict,
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    run_kwargs: dict | None = None,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], object] | None = None,
) -> LinearScanResult:
    """Run a linear scan over ky values."""

    gammas: list[float] = []
    omegas: list[float] = []
    ky_out: list[float] = []
    for i, ky in enumerate(ky_values):
        if resolution_policy is not None:
            Nl_i, Nm_i = resolution_policy(float(ky))
        else:
            Nl_i, Nm_i = int(Nl), int(Nm)
        dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        tmin_i = tmin[i] if isinstance(tmin, np.ndarray) else tmin
        tmax_i = tmax[i] if isinstance(tmax, np.ndarray) else tmax
        krylov_cfg_use = (
            krylov_policy(float(ky)) if krylov_policy is not None else krylov_cfg
        )
        result = run_linear_fn(
            ky_target=float(ky),
            cfg=cfg,
            Nl=int(Nl_i),
            Nm=int(Nm_i),
            dt=dt_i,
            steps=steps_i,
            method=method,
            solver=solver,
            krylov_cfg=krylov_cfg_use,
            auto_window=auto_window,
            tmin=tmin_i,
            tmax=tmax_i,
            **window_kw,
            **(run_kwargs or {}),
        )
        gammas.append(float(result.gamma))
        omegas.append(float(result.omega))
        ky_out.append(float(result.ky))

    return LinearScanResult(
        ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas)
    )


def _select_representative_ky(
    scan: LinearScanResult,
    select_ky: Callable[[LinearScanResult], float] | None,
) -> float:
    if select_ky is not None:
        return float(select_ky(scan))
    return float(scan.ky[int(np.nanargmax(scan.gamma))])


def _resolution_for_ky(
    ky: float,
    *,
    Nl: int,
    Nm: int,
    resolution_policy: Callable[[float], tuple[int, int]] | None,
) -> tuple[int, int]:
    if resolution_policy is not None:
        n_l, n_m = resolution_policy(float(ky))
        return int(n_l), int(n_m)
    return int(Nl), int(Nm)


def _mode_control_value(
    value: float | int | np.ndarray,
    idx: int,
    cast,
):
    if isinstance(value, np.ndarray):
        return cast(value[idx])
    return cast(value)


def _run_representative_mode(
    *,
    scan: LinearScanResult,
    ky_selected: float,
    linear_fn,
    cfg,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    mode_solver: str,
    window_kw: dict,
    mode_kwargs: dict | None,
    resolution_policy: Callable[[float], tuple[int, int]] | None,
) -> LinearRunResult:
    n_l, n_m = _resolution_for_ky(
        ky_selected, Nl=Nl, Nm=Nm, resolution_policy=resolution_policy
    )
    idx = int(np.argmin(np.abs(scan.ky - ky_selected)))
    return linear_fn(
        cfg=cfg,
        ky_target=ky_selected,
        Nl=n_l,
        Nm=n_m,
        dt=_mode_control_value(dt, idx, float),
        steps=_mode_control_value(steps, idx, int),
        method=method,
        solver=mode_solver,
        **window_kw,
        **(mode_kwargs or {}),
    )


def _fit_representative_mode_window(
    run: LinearRunResult,
    window_kw: dict,
) -> tuple[float | None, float | None]:
    if run.t.size < 2:
        return None, None
    signal = extract_mode_time_series(run.phi_t, run.selection, method="project")
    _g, _w, tmin_fit, tmax_fit = fit_growth_rate_auto(run.t, signal, **window_kw)
    return tmin_fit, tmax_fit


def _extract_representative_eigenfunction(
    run: LinearRunResult,
    grid: SpectralGrid,
    *,
    tmin_fit: float | None,
    tmax_fit: float | None,
) -> np.ndarray:
    return extract_eigenfunction(
        run.phi_t,
        run.t,
        run.selection,
        z=np.asarray(grid.z),
        method="snapshot",
        tmin=tmin_fit,
        tmax=tmax_fit,
    )


def run_scan_and_mode(
    *,
    ky_values: np.ndarray,
    scan_fn,
    linear_fn,
    cfg,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    solver: str,
    mode_solver: str,
    krylov_cfg,
    window_kw: dict,
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    run_kwargs: dict | None = None,
    mode_kwargs: dict | None = None,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], object] | None = None,
    select_ky: Callable[[LinearScanResult], float] | None = None,
) -> ScanAndModeResult:
    """Run a scan and extract a representative eigenfunction."""

    scan = run_linear_scan(
        ky_values=ky_values,
        run_linear_fn=linear_fn,
        cfg=cfg,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps,
        method=method,
        solver=solver,
        krylov_cfg=krylov_cfg,
        window_kw=window_kw,
        tmin=tmin,
        tmax=tmax,
        auto_window=auto_window,
        run_kwargs=run_kwargs,
        resolution_policy=resolution_policy,
        krylov_policy=krylov_policy,
    )
    ky_sel = _select_representative_ky(scan, select_ky)
    run = _run_representative_mode(
        scan=scan,
        ky_selected=ky_sel,
        linear_fn=linear_fn,
        cfg=cfg,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps,
        method=method,
        mode_solver=mode_solver,
        window_kw=window_kw,
        mode_kwargs=mode_kwargs,
        resolution_policy=resolution_policy,
    )
    grid = build_spectral_grid(cfg.grid)
    tmin_fit, tmax_fit = _fit_representative_mode_window(run, window_kw)
    eig = _extract_representative_eigenfunction(
        run, grid, tmin_fit=tmin_fit, tmax_fit=tmax_fit
    )
    return ScanAndModeResult(
        scan=scan,
        eigenfunction=eig,
        grid=grid,
        ky_selected=ky_sel,
        tmin=tmin_fit,
        tmax=tmax_fit,
    )


__all__ = [
    "CYCLONE_KRYLOV_DEFAULT",
    "CYCLONE_OMEGA_D_SCALE",
    "CYCLONE_OMEGA_STAR_SCALE",
    "CYCLONE_RHO_STAR",
    "ETG_KRYLOV_DEFAULT",
    "ETG_OMEGA_D_SCALE",
    "ETG_OMEGA_STAR_SCALE",
    "ETG_RHO_STAR",
    "KBM_KRYLOV_DEFAULT",
    "KBM_EXPLICIT_SOLVER_LOCK",
    "KBM_EXPLICIT_SOLVER_LOCK_TOL",
    "KBM_OMEGA_D_SCALE",
    "KBM_OMEGA_STAR_SCALE",
    "KBM_RHO_STAR",
    "KINETIC_KRYLOV_DEFAULT",
    "KINETIC_KRYLOV_REFERENCE_ALIGNED",
    "KINETIC_OMEGA_D_SCALE",
    "KINETIC_OMEGA_STAR_SCALE",
    "KINETIC_RHO_STAR",
    "REFERENCE_DAMP_ENDS_AMP",
    "REFERENCE_DAMP_ENDS_WIDTHFRAC",
    "REFERENCE_NU_HYPER_L",
    "REFERENCE_NU_HYPER_M",
    "REFERENCE_P_HYPER_L",
    "REFERENCE_P_HYPER_M",
    "TEM_KRYLOV_DEFAULT",
    "TEM_OMEGA_D_SCALE",
    "TEM_OMEGA_STAR_SCALE",
    "TEM_RHO_STAR",
    "CycloneBaseCase",
    "CycloneComparison",
    "CycloneReference",
    "CycloneRunResult",
    "CycloneScanResult",
    "ETGBaseCase",
    "KBMBaseCase",
    "KineticElectronBaseCase",
    "KrylovConfig",
    "LinearRunResult",
    "LinearScanResult",
    "ModeSelection",
    "SecondaryModeResult",
    "TEMBaseCase",
    "_apply_reference_hypercollisions",
    "_build_gaussian_profile",
    "_build_initial_condition",
    "_electron_only_params",
    "_extract_mode_only_signal",
    "_linked_boundary_end_damping",
    "_reference_hypercollision_power",
    "_is_array_like",
    "_iter_ky_batches",
    "_kbm_use_multi_target_krylov",
    "_kinetic_reference_init_cfg",
    "_load_reference_with_header",
    "_midplane_index",
    "_normalize_growth_rate",
    "_resolve_streaming_window",
    "_score_fit_signal_auto",
    "_select_fit_signal",
    "_select_fit_signal_auto",
    "_two_species_params",
    "compare_cyclone_to_reference",
    "ExplicitTimeConfig",
    "load_cyclone_reference",
    "load_cyclone_reference_kinetic",
    "load_etg_reference",
    "load_kbm_reference",
    "load_tem_reference",
    "run_cyclone_linear",
    "run_cyclone_scan",
    "run_etg_linear",
    "run_etg_scan",
    "run_kbm_beta_scan",
    "run_kbm_linear",
    "run_kbm_scan",
    "run_kinetic_linear",
    "run_kinetic_scan",
    "run_secondary_modes",
    "run_secondary_seed",
    "run_tem_linear",
    "run_tem_scan",
    "select_kbm_solver_auto",
    "build_secondary_stage2_config",
    "write_restart_state",
    "_analytic_signal",
    "_explicit_time_window",
    "_leading_window",
    "BranchContinuationMetrics",
    "DiagnosticTimeSeries",
    "EigenfunctionComparisonMetrics",
    "EigenfunctionReferenceBundle",
    "GateReport",
    "LateTimeLinearMetrics",
    "NonlinearHeatFluxConvergenceMetrics",
    "NonlinearWindowMetrics",
    "ObservedOrderMetrics",
    "ScalarGateResult",
    "ScanAndModeResult",
    "ZonalFlowResponseMetrics",
    "branch_continuity_gate_report",
    "branch_continuity_metrics",
    "compare_eigenfunctions",
    "eigenfunction_gate_report",
    "estimate_observed_order",
    "evaluate_scalar_gate",
    "gate_report",
    "gate_report_to_dict",
    "infer_triple_dealiased_ny",
    "late_time_linear_metrics",
    "late_time_window",
    "linear_metrics_gate_report",
    "load_diagnostic_time_series",
    "load_eigenfunction_reference_bundle",
    "nonlinear_heat_flux_convergence_gate_report",
    "nonlinear_heat_flux_convergence_metrics",
    "nonlinear_window_gate_report",
    "normalize_eigenfunction",
    "observed_order_gate_report",
    "phase_align_eigenfunction",
    "run_linear_scan",
    "run_scan_and_mode",
    "save_eigenfunction_reference_bundle",
    "windowed_nonlinear_metrics",
    "zonal_flow_response_metrics",
    "zonal_response_gate_report",
]
