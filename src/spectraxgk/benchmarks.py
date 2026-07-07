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
    ModeSelectionBatch,
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    instantaneous_growth_rate_from_phi,
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
from spectraxgk.solvers.time.diffrax import (
    integrate_linear_diffrax,
    integrate_linear_diffrax_streaming,
)
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
    ScanFitWindowPolicy,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    indexed_scan_value,
    normalize_fit_signal,
    normalize_solver_key,
    resolve_scan_mode_method,
    scan_window_valid,
    should_use_ky_batch,
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



_ETG_KRYLOV_FORWARD_KEYS = (
    "krylov_dim restarts omega_min_factor omega_target_factor omega_cap_factor omega_sign method "
    "power_iters power_dt shift shift_source shift_tol shift_maxiter shift_restart shift_solve_method "
    "shift_preconditioner shift_selection mode_family fallback_method fallback_real_floor"
).split()


# TEM scan and single-mode path policies live with the TEM benchmark owner.
@dataclass(frozen=True)
class TEMPathHooks:
    """Patchable numerical hooks supplied by the TEM public owner module."""

    linear_run_result: type[LinearRunResult]
    linear_scan_result: type[LinearScanResult]
    mode_selection: type[ModeSelection]
    mode_selection_batch: type[ModeSelectionBatch]
    select_ky_index: Callable[..., int]
    select_ky_grid: Callable[..., Any]
    build_initial_condition: Callable[..., Any]
    build_linear_cache: Callable[..., Any]
    dominant_eigenpair: Callable[..., tuple[Any, Any]]
    compute_fields_cached: Callable[..., Any]
    linear_terms_to_term_config: Callable[..., Any]
    integrate_linear: Callable[..., Any]
    integrate_linear_diagnostics: Callable[..., Any]
    integrate_linear_from_config: Callable[..., Any]
    integrate_linear_diffrax_streaming: Callable[..., Any]
    extract_mode_time_series: Callable[..., np.ndarray]
    fit_growth_rate: Callable[..., tuple[float, float]]
    fit_growth_rate_auto: Callable[..., tuple[float, float, float, float]]
    normalize_growth_rate: Callable[..., tuple[float, float]]
    resolve_streaming_window: Callable[..., tuple[float, float]]
    midplane_index: Callable[..., int]


@dataclass(frozen=True)
class _TEMBatchContext:
    """Fully prepared numerical context for one TEM scan batch."""

    batch_start: int
    ky_slice: np.ndarray
    valid_count: int
    grid: Any
    selection: ModeSelection | ModeSelectionBatch
    dt: float
    steps: int
    state: Any
    cache: Any


@dataclass(frozen=True)
class _TEMScanRuntimeOptions:
    """Options shared by every TEM scan batch."""

    n_laguerre: int
    n_hermite: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    time_cfg: Any | None
    solver_key: str
    krylov_cfg: Any | None
    krylov_default: Any
    fit_policy: ScanFitWindowPolicy
    mode_method: str
    mode_only: bool
    sample_stride: int | None
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    init_species_index: int
    diagnostic_norm: str
    use_batch: bool
    show_progress: bool
    hooks: TEMPathHooks


@dataclass
class _TEMScanAccumulator:
    """Mutable TEM scan rows collected across batches."""

    gammas: list[float]
    omegas: list[float]
    ky_out: list[float]

    def result(self, hooks: TEMPathHooks) -> LinearScanResult:
        """Pack accumulated rows with the public result type."""

        return hooks.linear_scan_result(
            ky=np.array(self.ky_out),
            gamma=np.array(self.gammas),
            omega=np.array(self.omegas),
        )


@dataclass(frozen=True)
class _TEMTimePathTiming:
    """Resolved sampling controls for one TEM initial-value solve."""

    dt: float
    steps: int
    stride: int
    time_cfg: Any | None


@dataclass(frozen=True)
class _TEMTimePathTrace:
    """Saved fields and sample times from one TEM initial-value solve."""

    t: np.ndarray
    phi_t: np.ndarray
    density_t: np.ndarray | None


@dataclass(frozen=True)
class _TEMTimePathFitPolicy:
    """Window-selection controls for fitting a TEM linear trace."""

    auto_window: bool
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float


def _tem_time_path_fit_policy_from_locals(
    values: dict[str, Any],
) -> _TEMTimePathFitPolicy:
    """Pack the single-run TEM fit policy from public path arguments."""

    return _TEMTimePathFitPolicy(
        **{field.name: values[field.name] for field in fields(_TEMTimePathFitPolicy)}
    )


def _tem_scan_fit_policy_from_locals(
    values: dict[str, Any],
    *,
    hooks: TEMPathHooks,
) -> ScanFitWindowPolicy:
    """Pack TEM scan growth-window policy with patchable fit hooks."""

    return ScanFitWindowPolicy(
        tmin=values["tmin"],
        tmax=values["tmax"],
        auto_window=values["auto_window"],
        window_fraction=values["window_fraction"],
        min_points=values["min_points"],
        start_fraction=values["start_fraction"],
        growth_weight=values["growth_weight"],
        require_positive=values["require_positive"],
        min_amp_fraction=values["min_amp_fraction"],
        fit_growth_rate_fn=hooks.fit_growth_rate,
        fit_growth_rate_auto_fn=hooks.fit_growth_rate_auto,
        normalize_growth_rate_fn=hooks.normalize_growth_rate,
    )


def _tem_scan_runtime_options_from_locals(
    values: dict[str, Any],
) -> _TEMScanRuntimeOptions:
    """Pack scan runtime options once after the fit policy is resolved."""

    return _TEMScanRuntimeOptions(
        **{field.name: values[field.name] for field in fields(_TEMScanRuntimeOptions)}
    )


_TEM_KRYLOV_FORWARD_KEYS = (
    "krylov_dim restarts omega_min_factor omega_target_factor omega_cap_factor omega_sign method "
    "power_iters power_dt shift shift_source shift_tol shift_maxiter shift_restart shift_solve_method "
    "shift_preconditioner shift_selection mode_family fallback_method fallback_real_floor"
).split()


def _krylov_eigenvalue(
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    cfg_use: Any,
    hooks: TEMPathHooks,
) -> tuple[Any, Any]:
    kwargs = {
        "terms": terms,
        **{name: getattr(cfg_use, name) for name in _TEM_KRYLOV_FORWARD_KEYS},
    }
    return hooks.dominant_eigenpair(G0_jax, cache, params, **kwargs)


def _prepare_tem_scan_batch(
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    grid_full: Any,
    geom: Any,
    params: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    init_species_index: int,
    use_batch: bool,
    hooks: TEMPathHooks,
) -> _TEMBatchContext:
    selection: ModeSelection | ModeSelectionBatch
    if use_batch:
        ky_indices = [
            hooks.select_ky_index(np.asarray(grid_full.ky), float(ky))
            for ky in ky_slice
        ]
        grid = hooks.select_ky_grid(grid_full, ky_indices)
        selection = hooks.mode_selection_batch(
            np.arange(len(ky_indices), dtype=int), 0, hooks.midplane_index(grid)
        )
        dt_i = float(dt)
        steps_i = int(steps)
    else:
        ky_indices = [
            hooks.select_ky_index(np.asarray(grid_full.ky), float(ky_slice[0]))
        ]
        grid = hooks.select_ky_grid(grid_full, ky_indices[0])
        selection = hooks.mode_selection(
            ky_index=0, kx_index=0, z_index=hooks.midplane_index(grid)
        )
        dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = (
            int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)
        )

    state_np = np.zeros(
        (2, n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=np.complex64,
    )
    state_single = hooks.build_initial_condition(
        grid,
        geom,
        ky_index=np.arange(len(ky_indices), dtype=int),
        kx_index=0,
        Nl=n_laguerre,
        Nm=n_hermite,
        init_cfg=init_cfg,
    )
    state_np[int(init_species_index)] = np.asarray(state_single, dtype=np.complex64)
    return _TEMBatchContext(
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        grid=grid,
        selection=selection,
        dt=dt_i,
        steps=steps_i,
        state=jnp.asarray(state_np),
        cache=hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite),
    )


def _tem_time_config_for_batch(
    time_cfg: Any | None,
    *,
    dt: float,
    steps: int,
    sample_stride: int | None,
) -> Any | None:
    if time_cfg is None:
        return None
    time_cfg_i = replace(time_cfg, dt=dt, t_max=dt * steps)
    if sample_stride is not None:
        time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)
    return time_cfg_i


def _append_tem_krylov_fit(
    *,
    context: _TEMBatchContext,
    params: Any,
    terms: Any,
    krylov_cfg: Any | None,
    krylov_default: Any,
    diagnostic_norm: str,
    hooks: TEMPathHooks,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> None:
    eig, _vec = _krylov_eigenvalue(
        context.state,
        context.cache,
        params,
        terms,
        krylov_cfg or krylov_default,
        hooks,
    )
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    gamma, omega = hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    gammas.append(gamma)
    omegas.append(omega)
    ky_out.append(float(context.ky_slice[0]))


def _append_tem_streaming_fit(
    *,
    context: _TEMBatchContext,
    geom: Any,
    params: Any,
    terms: Any,
    time_cfg_i: Any,
    fit_policy: ScanFitWindowPolicy,
    mode_method: str,
    streaming_amp_floor: float,
    show_progress: bool,
    hooks: TEMPathHooks,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> None:
    t_total = float(time_cfg_i.t_max)
    tmin_i, tmax_i = fit_policy.window_at(context.batch_start)
    tmin_i, tmax_i = hooks.resolve_streaming_window(
        t_total,
        tmin_i,
        tmax_i,
        fit_policy.start_fraction,
        fit_policy.window_fraction,
        1.0,
    )
    _, gamma_vals, omega_vals = hooks.integrate_linear_diffrax_streaming(
        context.state,
        context.grid,
        geom,
        params,
        dt=context.dt,
        steps=context.steps,
        method=time_cfg_i.diffrax_solver,
        cache=context.cache,
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
        mode_ky_indices=np.arange(context.valid_count, dtype=int),
        mode_kx_index=0,
        mode_z_index=hooks.midplane_index(context.grid),
        mode_method=mode_method,
        amp_floor=streaming_amp_floor,
        return_state=False,
    )
    gamma_arr = np.asarray(gamma_vals)
    omega_arr = np.asarray(omega_vals)
    for local_idx in range(context.valid_count):
        gammas.append(float(gamma_arr[local_idx]))
        omegas.append(float(omega_arr[local_idx]))
        ky_out.append(float(context.ky_slice[local_idx]))


def _append_tem_saved_fit(
    *,
    context: _TEMBatchContext,
    geom: Any,
    params: Any,
    terms: Any,
    method: str,
    time_cfg_i: Any | None,
    mode_method: str,
    mode_only: bool,
    sample_stride: int | None,
    fit_policy: ScanFitWindowPolicy,
    diagnostic_norm: str,
    hooks: TEMPathHooks,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> None:
    if time_cfg_i is not None:
        _, phi_t = hooks.integrate_linear_from_config(
            context.state,
            context.grid,
            geom,
            params,
            time_cfg_i,
            cache=context.cache,
            terms=terms,
            save_mode=context.selection if mode_only else None,
            mode_method=mode_method,
        )
        stride = time_cfg_i.sample_stride
    else:
        stride = 1 if sample_stride is None else int(sample_stride)
        _, phi_t = hooks.integrate_linear(
            context.state,
            context.grid,
            geom,
            params,
            dt=context.dt,
            steps=context.steps,
            method=method,
            cache=context.cache,
            terms=terms,
            sample_stride=stride,
        )

    phi_t_np = np.asarray(phi_t)
    for local_idx in range(context.valid_count):
        if mode_only and phi_t_np.ndim <= 2:
            signal = _extract_mode_only_signal(phi_t_np, local_idx=local_idx)
        else:
            sel_local = hooks.mode_selection(
                ky_index=local_idx,
                kx_index=0,
                z_index=hooks.midplane_index(context.grid),
            )
            signal = hooks.extract_mode_time_series(
                phi_t_np, sel_local, method=mode_method
            )
        gamma, omega = fit_policy.fit_signal(
            signal,
            idx=context.batch_start + local_idx,
            dt=context.dt,
            stride=stride,
            params=params,
            diagnostic_norm=diagnostic_norm,
        )
        gammas.append(gamma)
        omegas.append(omega)
        ky_out.append(float(context.ky_slice[local_idx]))


def _tem_scan_batches(
    ky_values: np.ndarray,
    options: _TEMScanRuntimeOptions,
) -> Any:
    """Select scalar or fixed-width ky batching for the TEM scan."""

    if options.use_batch:
        return _iter_ky_batches(
            ky_values,
            ky_batch=options.ky_batch,
            fixed_batch_shape=options.fixed_batch_shape,
        )
    return _iter_ky_batches(ky_values, ky_batch=1, fixed_batch_shape=False)


def _append_tem_scan_batch(
    *,
    context: _TEMBatchContext,
    geom: Any,
    params: Any,
    terms: Any,
    options: _TEMScanRuntimeOptions,
    acc: _TEMScanAccumulator,
) -> None:
    """Route one prepared TEM batch through its selected solver path."""

    hooks = options.hooks
    if options.solver_key == "krylov":
        _append_tem_krylov_fit(
            context=context,
            params=params,
            terms=terms,
            krylov_cfg=options.krylov_cfg,
            krylov_default=options.krylov_default,
            diagnostic_norm=options.diagnostic_norm,
            hooks=hooks,
            gammas=acc.gammas,
            omegas=acc.omegas,
            ky_out=acc.ky_out,
        )
        return

    time_cfg_i = _tem_time_config_for_batch(
        options.time_cfg,
        dt=context.dt,
        steps=context.steps,
        sample_stride=options.sample_stride,
    )
    if time_cfg_i is not None and time_cfg_i.use_diffrax and options.streaming_fit:
        _append_tem_streaming_fit(
            context=context,
            geom=geom,
            params=params,
            terms=terms,
            time_cfg_i=time_cfg_i,
            fit_policy=options.fit_policy,
            mode_method=options.mode_method,
            streaming_amp_floor=options.streaming_amp_floor,
            show_progress=options.show_progress,
            hooks=hooks,
            gammas=acc.gammas,
            omegas=acc.omegas,
            ky_out=acc.ky_out,
        )
        return

    _append_tem_saved_fit(
        context=context,
        geom=geom,
        params=params,
        terms=terms,
        method=options.method,
        time_cfg_i=time_cfg_i,
        mode_method=options.mode_method,
        mode_only=options.mode_only,
        sample_stride=options.sample_stride,
        fit_policy=options.fit_policy,
        diagnostic_norm=options.diagnostic_norm,
        hooks=hooks,
        gammas=acc.gammas,
        omegas=acc.omegas,
        ky_out=acc.ky_out,
    )


def _run_tem_scan_loop(
    *,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    terms: Any,
    init_cfg: Any,
    options: _TEMScanRuntimeOptions,
) -> _TEMScanAccumulator:
    """Prepare and execute all TEM scan batches."""

    acc = _TEMScanAccumulator(gammas=[], omegas=[], ky_out=[])
    for batch_start, ky_slice, valid_count in _tem_scan_batches(ky_values, options):
        context = _prepare_tem_scan_batch(
            batch_start=batch_start,
            ky_slice=ky_slice,
            valid_count=valid_count,
            grid_full=grid_full,
            geom=geom,
            params=params,
            init_cfg=init_cfg,
            n_laguerre=options.n_laguerre,
            n_hermite=options.n_hermite,
            dt=options.dt,
            steps=options.steps,
            init_species_index=options.init_species_index,
            use_batch=options.use_batch,
            hooks=options.hooks,
        )
        _append_tem_scan_batch(
            context=context,
            geom=geom,
            params=params,
            terms=terms,
            options=options,
            acc=acc,
        )
    return acc


def run_tem_krylov_linear_path(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    n_laguerre: int,
    n_hermite: int,
    sel: ModeSelection,
    krylov_cfg: Any | None,
    krylov_default: Any,
    diagnostic_norm: str,
    hooks: TEMPathHooks,
) -> LinearRunResult:
    """Run the single-ky TEM Krylov branch and package a linear result."""

    cfg_use = krylov_cfg or krylov_default
    cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)
    eig, vec = _krylov_eigenvalue(G0_jax, cache, params, terms, cfg_use, hooks)
    term_cfg = hooks.linear_terms_to_term_config(terms)
    phi = hooks.compute_fields_cached(vec, cache, params, terms=term_cfg).phi
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    gamma, omega = hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    return hooks.linear_run_result(
        t=np.array([0.0]),
        phi_t=np.asarray(phi)[None, ...],
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )


def _validate_tem_time_fit_signal(fit_signal: str) -> None:
    if fit_signal not in {"phi", "density"}:
        raise ValueError("fit_signal must be 'phi' or 'density'")


def _resolve_tem_time_path_timing(
    *,
    dt: float,
    steps: int,
    time_cfg: Any | None,
    sample_stride: int | None,
) -> _TEMTimePathTiming:
    time_cfg_use = time_cfg
    if time_cfg is not None:
        if sample_stride is not None:
            time_cfg_use = replace(time_cfg, sample_stride=sample_stride)
        assert time_cfg_use is not None
        return _TEMTimePathTiming(
            dt=float(time_cfg_use.dt),
            steps=int(round(time_cfg_use.t_max / time_cfg_use.dt)),
            stride=int(time_cfg_use.sample_stride),
            time_cfg=time_cfg_use,
        )
    return _TEMTimePathTiming(
        dt=float(dt),
        steps=int(steps),
        stride=1 if sample_stride is None else int(sample_stride),
        time_cfg=None,
    )


def _integrate_tem_time_path_trace(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    cache: Any,
    timing: _TEMTimePathTiming,
    method: str,
    fit_signal: str,
    density_species_index: int,
    show_progress: bool,
    hooks: TEMPathHooks,
) -> _TEMTimePathTrace:
    if fit_signal == "density":
        diag = hooks.integrate_linear_diagnostics(
            G0_jax,
            grid,
            geom,
            params,
            dt=timing.dt,
            steps=timing.steps,
            method=method,
            cache=cache,
            terms=terms,
            sample_stride=timing.stride,
            species_index=density_species_index,
        )
        _, phi_t, density_t, *_ = diag
    elif timing.time_cfg is not None:
        _, phi_t = hooks.integrate_linear_from_config(
            G0_jax,
            grid,
            geom,
            params,
            timing.time_cfg,
            cache=cache,
            terms=terms,
            show_progress=show_progress,
        )
        density_t = None
    else:
        _, phi_t = hooks.integrate_linear(
            G0_jax,
            grid,
            geom,
            params,
            dt=timing.dt,
            steps=timing.steps,
            method=method,
            cache=cache,
            terms=terms,
            sample_stride=timing.stride,
            show_progress=show_progress,
        )
        density_t = None

    phi_t_np = np.asarray(phi_t)
    return _TEMTimePathTrace(
        t=np.arange(phi_t_np.shape[0]) * timing.dt * timing.stride,
        phi_t=phi_t_np,
        density_t=None if density_t is None else np.asarray(density_t),
    )


def _tem_time_path_signal(
    trace: _TEMTimePathTrace,
    *,
    fit_signal: str,
    sel: ModeSelection,
    mode_method: str,
    hooks: TEMPathHooks,
) -> np.ndarray:
    if fit_signal == "density" and trace.density_t is not None:
        return hooks.extract_mode_time_series(trace.density_t, sel, method=mode_method)
    return hooks.extract_mode_time_series(trace.phi_t, sel, method=mode_method)


def _fit_tem_time_path_signal(
    *,
    t: np.ndarray,
    signal: np.ndarray,
    policy: _TEMTimePathFitPolicy,
    hooks: TEMPathHooks,
) -> tuple[float, float]:
    auto_fit_kwargs: dict[str, Any] = {
        "window_fraction": policy.window_fraction,
        "min_points": policy.min_points,
        "start_fraction": policy.start_fraction,
        "growth_weight": policy.growth_weight,
        "require_positive": policy.require_positive,
        "min_amp_fraction": policy.min_amp_fraction,
    }
    if policy.auto_window and policy.tmin is None and policy.tmax is None:
        gamma, omega, _tmin, _tmax = hooks.fit_growth_rate_auto(
            t, signal, **auto_fit_kwargs
        )
        return gamma, omega
    try:
        return hooks.fit_growth_rate(t, signal, tmin=policy.tmin, tmax=policy.tmax)
    except ValueError:
        gamma, omega, _tmin, _tmax = hooks.fit_growth_rate_auto(
            t, signal, **auto_fit_kwargs
        )
        return gamma, omega


def _tem_time_path_result(
    *,
    trace: _TEMTimePathTrace,
    gamma: float,
    omega: float,
    grid: Any,
    sel: ModeSelection,
    hooks: TEMPathHooks,
) -> LinearRunResult:
    return hooks.linear_run_result(
        t=trace.t,
        phi_t=trace.phi_t,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )


def run_tem_time_linear_path(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float,
    steps: int,
    method: str,
    time_cfg: Any | None,
    sample_stride: int | None,
    fit_signal: str,
    density_species_index: int,
    sel: ModeSelection,
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
    diagnostic_norm: str,
    show_progress: bool,
    hooks: TEMPathHooks,
) -> LinearRunResult:
    """Run the single-ky TEM time-integration branch and fit the selected signal."""

    _validate_tem_time_fit_signal(fit_signal)
    cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)
    timing = _resolve_tem_time_path_timing(
        dt=dt,
        steps=steps,
        time_cfg=time_cfg,
        sample_stride=sample_stride,
    )
    trace = _integrate_tem_time_path_trace(
        G0_jax=G0_jax,
        grid=grid,
        geom=geom,
        params=params,
        terms=terms,
        cache=cache,
        timing=timing,
        method=method,
        fit_signal=fit_signal,
        density_species_index=density_species_index,
        show_progress=show_progress,
        hooks=hooks,
    )
    signal = _tem_time_path_signal(
        trace,
        fit_signal=fit_signal,
        sel=sel,
        mode_method=mode_method,
        hooks=hooks,
    )
    fit_policy = _tem_time_path_fit_policy_from_locals(locals())
    gamma, omega = _fit_tem_time_path_signal(
        t=trace.t,
        signal=signal,
        policy=fit_policy,
        hooks=hooks,
    )
    gamma, omega = hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    return _tem_time_path_result(
        trace=trace,
        gamma=gamma,
        omega=omega,
        grid=grid,
        sel=sel,
        hooks=hooks,
    )


def run_tem_scan_batches(
    *,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    terms: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    time_cfg: Any | None,
    solver_key: str,
    krylov_cfg: Any | None,
    krylov_default: Any,
    tmin: float | None,
    tmax: float | None,
    auto_window: bool,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    mode_method: str,
    mode_only: bool,
    sample_stride: int | None,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    init_species_index: int,
    diagnostic_norm: str,
    use_batch: bool,
    hooks: TEMPathHooks,
    show_progress: bool,
) -> LinearScanResult:
    """Run TEM scan batches across Krylov, streaming, and saved-time branches."""

    fit_policy = _tem_scan_fit_policy_from_locals(locals(), hooks=hooks)
    options = _tem_scan_runtime_options_from_locals(locals())
    return _run_tem_scan_loop(
        ky_values=ky_values,
        grid_full=grid_full,
        geom=geom,
        params=params,
        terms=terms,
        init_cfg=init_cfg,
        options=options,
    ).result(hooks)


def _tem_hooks() -> TEMPathHooks:
    return TEMPathHooks(
        linear_run_result=LinearRunResult,
        linear_scan_result=LinearScanResult,
        mode_selection=ModeSelection,
        mode_selection_batch=ModeSelectionBatch,
        select_ky_index=select_ky_index,
        select_ky_grid=select_ky_grid,
        build_initial_condition=_build_initial_condition,
        build_linear_cache=build_linear_cache,
        dominant_eigenpair=dominant_eigenpair,
        compute_fields_cached=compute_fields_cached,
        linear_terms_to_term_config=linear_terms_to_term_config,
        integrate_linear=integrate_linear,
        integrate_linear_diagnostics=integrate_linear_diagnostics,
        integrate_linear_from_config=integrate_linear_from_config,
        integrate_linear_diffrax_streaming=integrate_linear_diffrax_streaming,
        extract_mode_time_series=extract_mode_time_series,
        fit_growth_rate=fit_growth_rate,
        fit_growth_rate_auto=fit_growth_rate_auto,
        normalize_growth_rate=_normalize_growth_rate,
        resolve_streaming_window=_resolve_streaming_window,
        midplane_index=_midplane_index,
    )


def _tem_params_and_terms(
    cfg: TEMBaseCase,
    geom: SAlphaGeometry,
    params: LinearParams | None,
    terms: LinearTerms | None,
    n_hermite: int,
) -> tuple[LinearParams, LinearTerms]:
    if params is None:
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=TEM_OMEGA_D_SCALE,
            omega_star_scale=TEM_OMEGA_STAR_SCALE,
            rho_star=TEM_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=n_hermite,
        )
    if terms is None:
        terms = LinearTerms(bpar=0.0)
    return params, terms


@dataclass(frozen=True)
class _TEMLinearRequest:
    ky_target: float
    Nl: int
    Nm: int
    dt: float
    steps: int
    method: str
    params: LinearParams | None
    cfg: TEMBaseCase | None
    time_cfg: TimeConfig | None
    solver: str
    krylov_cfg: KrylovConfig | None
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
    fit_signal: str
    terms: LinearTerms | None
    sample_stride: int | None
    init_species_index: int
    density_species_index: int
    diagnostic_norm: str
    show_progress: bool


@dataclass(frozen=True)
class _TEMScanRequest:
    ky_values: np.ndarray
    Nl: int
    Nm: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    params: LinearParams | None
    cfg: TEMBaseCase | None
    time_cfg: TimeConfig | None
    solver: str
    krylov_cfg: KrylovConfig | None
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
    mode_only: bool
    terms: LinearTerms | None
    sample_stride: int | None
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    init_species_index: int
    density_species_index: int
    diagnostic_norm: str
    show_progress: bool


@dataclass(frozen=True)
class _TEMLinearSetup:
    cfg: TEMBaseCase
    grid_full: Any
    geom: SAlphaGeometry
    params: LinearParams
    terms: LinearTerms
    hooks: TEMPathHooks


@dataclass(frozen=True)
class _TEMLinearState:
    grid: Any
    selection: ModeSelection
    state: jnp.ndarray


def _tem_linear_request_from_locals(values: dict[str, Any]) -> _TEMLinearRequest:
    """Pack public ``run_tem_linear`` arguments once for internal routing."""

    return _TEMLinearRequest(
        **{field.name: values[field.name] for field in fields(_TEMLinearRequest)}
    )


def _tem_scan_request_from_locals(values: dict[str, Any]) -> _TEMScanRequest:
    """Pack public ``run_tem_scan`` arguments once for internal routing."""

    return _TEMScanRequest(
        **{field.name: values[field.name] for field in fields(_TEMScanRequest)}
    )


def _validate_tem_species_indices(
    *,
    init_species_index: int,
    density_species_index: int,
) -> None:
    ns = 2
    if init_species_index < 0 or init_species_index >= ns:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= ns:
        raise ValueError("density_species_index out of range for kinetic species")


def _resolve_tem_linear_setup(request: _TEMLinearRequest) -> _TEMLinearSetup:
    cfg = request.cfg or TEMBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params, terms = _tem_params_and_terms(
        cfg,
        geom,
        request.params,
        request.terms,
        request.Nm,
    )
    return _TEMLinearSetup(
        cfg=cfg,
        grid_full=grid_full,
        geom=geom,
        params=params,
        terms=terms,
        hooks=_tem_hooks(),
    )


def _resolve_tem_scan_setup(request: _TEMScanRequest) -> _TEMLinearSetup:
    cfg = request.cfg or TEMBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params, terms = _tem_params_and_terms(
        cfg,
        geom,
        request.params,
        request.terms,
        request.Nm,
    )
    return _TEMLinearSetup(
        cfg=cfg,
        grid_full=grid_full,
        geom=geom,
        params=params,
        terms=terms,
        hooks=_tem_hooks(),
    )


def _prepare_tem_linear_state(
    setup: _TEMLinearSetup,
    request: _TEMLinearRequest,
) -> _TEMLinearState:
    _validate_tem_species_indices(
        init_species_index=request.init_species_index,
        density_species_index=request.density_species_index,
    )
    ky_index = select_ky_index(np.asarray(setup.grid_full.ky), request.ky_target)
    grid = select_ky_grid(setup.grid_full, ky_index)
    selection = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    state_np = np.zeros(
        (2, request.Nl, request.Nm, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=np.complex64,
    )
    state_single = _build_initial_condition(
        grid,
        setup.geom,
        ky_index=selection.ky_index,
        kx_index=selection.kx_index,
        Nl=request.Nl,
        Nm=request.Nm,
        init_cfg=setup.cfg.init,
    )
    state_np[int(request.init_species_index)] = np.asarray(
        state_single,
        dtype=np.complex64,
    )
    return _TEMLinearState(
        grid=grid,
        selection=selection,
        state=jnp.asarray(state_np),
    )


def _run_tem_scan_request(request: _TEMScanRequest) -> LinearScanResult:
    setup = _resolve_tem_scan_setup(request)
    solver_key = normalize_solver_key(request.solver)
    mode_method = resolve_scan_mode_method(
        request.mode_method, mode_only=request.mode_only
    )
    use_batch = should_use_ky_batch(
        ky_batch=request.ky_batch,
        solver_key=solver_key,
        dt=request.dt,
        steps=request.steps,
        tmin=request.tmin,
        tmax=request.tmax,
    )
    _validate_tem_species_indices(
        init_species_index=request.init_species_index,
        density_species_index=request.density_species_index,
    )
    return run_tem_scan_batches(
        ky_values=np.asarray(request.ky_values, dtype=float),
        grid_full=setup.grid_full,
        geom=setup.geom,
        params=setup.params,
        terms=setup.terms,
        init_cfg=setup.cfg.init,
        n_laguerre=request.Nl,
        n_hermite=request.Nm,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        time_cfg=request.time_cfg,
        solver_key=solver_key,
        krylov_cfg=request.krylov_cfg,
        krylov_default=TEM_KRYLOV_DEFAULT,
        tmin=request.tmin,
        tmax=request.tmax,
        auto_window=request.auto_window,
        window_fraction=request.window_fraction,
        min_points=request.min_points,
        start_fraction=request.start_fraction,
        growth_weight=request.growth_weight,
        require_positive=request.require_positive,
        min_amp_fraction=request.min_amp_fraction,
        mode_method=mode_method,
        mode_only=request.mode_only,
        sample_stride=request.sample_stride,
        ky_batch=request.ky_batch,
        fixed_batch_shape=request.fixed_batch_shape,
        streaming_fit=request.streaming_fit,
        streaming_amp_floor=request.streaming_amp_floor,
        init_species_index=request.init_species_index,
        diagnostic_norm=request.diagnostic_norm,
        use_batch=use_batch,
        hooks=setup.hooks,
        show_progress=request.show_progress,
    )


def _run_tem_linear_request(request: _TEMLinearRequest) -> LinearRunResult:
    setup = _resolve_tem_linear_setup(request)
    state = _prepare_tem_linear_state(setup, request)
    if request.solver.lower() == "krylov":
        return run_tem_krylov_linear_path(
            G0_jax=state.state,
            grid=state.grid,
            geom=setup.geom,
            params=setup.params,
            terms=setup.terms,
            n_laguerre=request.Nl,
            n_hermite=request.Nm,
            sel=state.selection,
            krylov_cfg=request.krylov_cfg,
            krylov_default=TEM_KRYLOV_DEFAULT,
            diagnostic_norm=request.diagnostic_norm,
            hooks=setup.hooks,
        )
    return run_tem_time_linear_path(
        G0_jax=state.state,
        grid=state.grid,
        geom=setup.geom,
        params=setup.params,
        terms=setup.terms,
        n_laguerre=request.Nl,
        n_hermite=request.Nm,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        time_cfg=request.time_cfg,
        sample_stride=request.sample_stride,
        fit_signal=request.fit_signal,
        density_species_index=request.density_species_index,
        sel=state.selection,
        mode_method=request.mode_method,
        auto_window=request.auto_window,
        tmin=request.tmin,
        tmax=request.tmax,
        window_fraction=request.window_fraction,
        min_points=request.min_points,
        start_fraction=request.start_fraction,
        growth_weight=request.growth_weight,
        require_positive=request.require_positive,
        min_amp_fraction=request.min_amp_fraction,
        diagnostic_norm=request.diagnostic_norm,
        show_progress=request.show_progress,
        hooks=setup.hooks,
    )


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

    return _run_tem_linear_request(_tem_linear_request_from_locals(locals()))


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
    """Run the TEM benchmark for a list of ky values."""

    return _run_tem_scan_request(_tem_scan_request_from_locals(locals()))

@dataclass(frozen=True)
class _ETGLinearSetup:
    """Solver-ready single-ky ETG state shared by Krylov and time paths."""

    cfg: ETGBaseCase
    grid: Any
    geom: Any
    params: Any
    terms: LinearTerms
    selection: ModeSelection
    electron_index: int
    initial_state: Any


@dataclass(frozen=True)
class _ETGTimePathOptions:
    """Private fit and streaming policy for ETG saved-time integrations."""

    fit_key: str
    streaming_fit: bool
    streaming_amp_floor: float
    reference_growth_window: bool
    reference_navg_fraction: float
    mode_method: str
    tmin: float | None
    tmax: float | None
    auto_window: bool
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    diagnostic_norm: str


@dataclass(frozen=True)
class _ETGLinearRequest:
    """Raw public ETG single-ky inputs before solver policies are resolved."""

    ky_target: float
    Nl: int
    Nm: int
    dt: float
    steps: int
    method: str
    params: LinearParams | None
    cfg: ETGBaseCase | None
    time_cfg: TimeConfig | None
    solver: str
    krylov_cfg: KrylovConfig | None
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
    streaming_amp_floor: float
    reference_growth_window: bool
    reference_navg_fraction: float
    diagnostic_norm: str
    show_progress: bool


def _etg_linear_request_from_locals(values: dict[str, Any]) -> _ETGLinearRequest:
    """Build an ETG request from ``run_etg_linear`` locals."""

    names = {field.name for field in fields(_ETGLinearRequest)}
    return _ETGLinearRequest(**{name: values[name] for name in names})


def _default_etg_params(cfg: ETGBaseCase, geom: Any, Nm: int) -> LinearParams:
    """Build ETG benchmark species parameters using the tracked normalization."""

    if getattr(cfg.model, "adiabatic_ions", False):
        return _electron_only_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=ETG_OMEGA_D_SCALE,
            omega_star_scale=ETG_OMEGA_STAR_SCALE,
            rho_star=ETG_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=Nm,
        )
    return _two_species_params(
        cfg.model,
        kpar_scale=float(geom.gradpar()),
        omega_d_scale=ETG_OMEGA_D_SCALE,
        omega_star_scale=ETG_OMEGA_STAR_SCALE,
        rho_star=ETG_RHO_STAR,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
        nhermite=Nm,
    )


def _default_etg_terms() -> LinearTerms:
    """Return the electrostatic ETG benchmark term contract."""

    return LinearTerms(apar=0.0, bpar=0.0, hypercollisions=1.0)


def _build_etg_linear_setup(
    *,
    cfg: ETGBaseCase | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    ky_target: float,
    Nl: int,
    Nm: int,
) -> _ETGLinearSetup:
    """Create the selected-grid initial state for one ETG benchmark point."""

    cfg_use = cfg or ETGBaseCase()
    grid_full = build_spectral_grid(cfg_use.grid)
    geom = SAlphaGeometry.from_config(cfg_use.geometry)
    params_use = params if params is not None else _default_etg_params(cfg_use, geom, Nm)
    terms_use = terms if terms is not None else _default_etg_terms()

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))

    charge = np.atleast_1d(np.asarray(params_use.charge_sign))
    ns = int(charge.size)
    electron_index = int(np.argmin(charge))
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
        init_cfg=cfg_use.init,
    )
    G0[electron_index] = np.asarray(G0_single, dtype=np.complex64)
    return _ETGLinearSetup(
        cfg=cfg_use,
        grid=grid,
        geom=geom,
        params=params_use,
        terms=terms_use,
        selection=sel,
        electron_index=electron_index,
        initial_state=jnp.asarray(G0),
    )


def _etg_linear_result(
    setup: _ETGLinearSetup,
    *,
    t: np.ndarray,
    phi_t_np: np.ndarray,
    gamma: float,
    omega: float,
) -> LinearRunResult:
    """Pack a single-ky ETG run result with the selected physical ky."""

    return LinearRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(setup.grid.ky[setup.selection.ky_index]),
        selection=setup.selection,
    )


def _valid_etg_growth(
    gamma_val: float, omega_val: float, *, require_positive: bool
) -> bool:
    """Return whether a Krylov ETG result is acceptable for auto solver mode."""

    if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
        return False
    if require_positive and gamma_val <= 0.0:
        return False
    return True


def _run_etg_krylov_path(
    setup: _ETGLinearSetup,
    *,
    Nl: int,
    Nm: int,
    krylov_cfg: KrylovConfig | None,
    diagnostic_norm: str,
) -> LinearRunResult:
    """Solve one ETG point with the Krylov eigenpath."""

    cfg_use = krylov_cfg or ETG_KRYLOV_DEFAULT
    cache = build_linear_cache(setup.grid, setup.geom, setup.params, Nl, Nm)
    krylov_kwargs = {
        "terms": setup.terms,
        **{name: getattr(cfg_use, name) for name in _ETG_KRYLOV_FORWARD_KEYS},
    }
    eig, vec = dominant_eigenpair(
        setup.initial_state, cache, setup.params, **krylov_kwargs
    )
    term_cfg = linear_terms_to_term_config(setup.terms)
    phi = compute_fields_cached(vec, cache, setup.params, terms=term_cfg).phi
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    if cfg_use.omega_sign != 0:
        omega = float(np.sign(cfg_use.omega_sign)) * abs(omega)
    gamma, omega = _normalize_growth_rate(
        gamma, omega, setup.params, diagnostic_norm
    )
    return _etg_linear_result(
        setup,
        t=np.array([0.0]),
        phi_t_np=np.asarray(phi)[None, ...],
        gamma=gamma,
        omega=omega,
    )


def _resolve_etg_time_config(
    cfg: ETGBaseCase,
    time_cfg: TimeConfig | None,
    *,
    streaming_fit: bool,
    dt: float,
    steps: int,
    sample_stride: int | None,
) -> tuple[TimeConfig | None, float, int]:
    """Resolve explicit ETG time configuration without changing fit semantics."""

    time_cfg_use = time_cfg
    if time_cfg_use is None and streaming_fit and cfg.time.use_diffrax:
        max_steps = max(int(cfg.time.diffrax_max_steps), int(steps))
        time_cfg_use = replace(
            cfg.time,
            dt=dt,
            t_max=dt * steps,
            diffrax_max_steps=max_steps,
        )
        if sample_stride is not None:
            time_cfg_use = replace(time_cfg_use, sample_stride=sample_stride)
    if time_cfg_use is not None:
        if sample_stride is not None:
            time_cfg_use = replace(time_cfg_use, sample_stride=sample_stride)
        if time_cfg is not None:
            dt = float(time_cfg_use.dt)
            steps = int(round(time_cfg_use.t_max / time_cfg_use.dt))
    return time_cfg_use, dt, steps


def _etg_auto_fit_options(
    *,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> dict[str, Any]:
    """Pack the shared automatic-window policy for ETG trace fits."""

    return {
        "window_fraction": window_fraction,
        "min_points": min_points,
        "start_fraction": start_fraction,
        "growth_weight": growth_weight,
        "require_positive": require_positive,
        "min_amp_fraction": min_amp_fraction,
    }


def _fit_etg_reference_growth(
    setup: _ETGLinearSetup,
    *,
    phi_t_np: np.ndarray,
    t: np.ndarray,
    reference_navg_fraction: float,
    mode_method: str,
    diagnostic_norm: str,
) -> tuple[float, float]:
    """Fit ETG ``phi`` with the legacy instantaneous-growth reference window."""

    gamma, omega, _gamma_t, _omega_t, _t_mid = instantaneous_growth_rate_from_phi(
        phi_t_np,
        t,
        setup.selection,
        navg_fraction=reference_navg_fraction,
        mode_method=mode_method,
    )
    return _normalize_growth_rate(gamma, omega, setup.params, diagnostic_norm)


def _fit_etg_auto_signal(
    setup: _ETGLinearSetup,
    *,
    phi_t_np: np.ndarray,
    density_np: np.ndarray | None,
    t: np.ndarray,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    diagnostic_norm: str,
) -> tuple[float, float]:
    """Select the most stable ETG signal and fit its automatic growth window."""

    gamma, omega = _select_fit_signal_auto(
        t,
        phi_t_np,
        density_np,
        setup.selection,
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
    )[2:]
    return _normalize_growth_rate(gamma, omega, setup.params, diagnostic_norm)


def _fit_etg_selected_signal(
    setup: _ETGLinearSetup,
    *,
    phi_t_np: np.ndarray,
    density_np: np.ndarray | None,
    t: np.ndarray,
    fit_key: str,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    auto_window: bool,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    diagnostic_norm: str,
) -> tuple[float, float]:
    """Fit a caller-selected ETG signal with manual-window fallback."""

    signal = _select_fit_signal(
        phi_t_np,
        density_np,
        setup.selection,
        fit_signal=fit_key,
        mode_method=mode_method,
    )
    auto_fit_kwargs = _etg_auto_fit_options(
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
    )
    use_auto = auto_window and tmin is None and tmax is None
    if not use_auto and not scan_window_valid(t, tmin, tmax):
        use_auto = True
    if use_auto:
        gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
            t, signal, **auto_fit_kwargs
        )
    else:
        try:
            gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)
        except ValueError:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                t, signal, **auto_fit_kwargs
            )
    return _normalize_growth_rate(gamma, omega, setup.params, diagnostic_norm)


def _fit_etg_time_trace(
    setup: _ETGLinearSetup,
    *,
    phi_t: Any,
    density_t: Any,
    dt: float,
    stride: int,
    options: _ETGTimePathOptions,
) -> LinearRunResult:
    """Fit ETG growth and frequency from a saved time trace."""

    phi_t_np = np.asarray(phi_t)
    t = np.arange(phi_t_np.shape[0]) * dt * stride
    density_np = None if density_t is None else np.asarray(density_t)
    if options.reference_growth_window and options.fit_key == "phi":
        gamma, omega = _fit_etg_reference_growth(
            setup,
            phi_t_np=phi_t_np,
            t=t,
            reference_navg_fraction=options.reference_navg_fraction,
            mode_method=options.mode_method,
            diagnostic_norm=options.diagnostic_norm,
        )
    elif options.fit_key == "auto":
        gamma, omega = _fit_etg_auto_signal(
            setup,
            phi_t_np=phi_t_np,
            density_np=density_np,
            t=t,
            mode_method=options.mode_method,
            tmin=options.tmin,
            tmax=options.tmax,
            window_fraction=options.window_fraction,
            min_points=options.min_points,
            start_fraction=options.start_fraction,
            growth_weight=options.growth_weight,
            require_positive=options.require_positive,
            min_amp_fraction=options.min_amp_fraction,
            diagnostic_norm=options.diagnostic_norm,
        )
    else:
        gamma, omega = _fit_etg_selected_signal(
            setup,
            phi_t_np=phi_t_np,
            density_np=density_np,
            t=t,
            fit_key=options.fit_key,
            mode_method=options.mode_method,
            tmin=options.tmin,
            tmax=options.tmax,
            auto_window=options.auto_window,
            window_fraction=options.window_fraction,
            min_points=options.min_points,
            start_fraction=options.start_fraction,
            growth_weight=options.growth_weight,
            require_positive=options.require_positive,
            min_amp_fraction=options.min_amp_fraction,
            diagnostic_norm=options.diagnostic_norm,
        )
    return _etg_linear_result(setup, t=t, phi_t_np=phi_t_np, gamma=gamma, omega=omega)


def _run_etg_streaming_density_fit(
    setup: _ETGLinearSetup,
    *,
    time_cfg: TimeConfig,
    cache: Any,
    dt: float,
    steps: int,
    options: _ETGTimePathOptions,
    show_progress: bool,
) -> LinearRunResult:
    """Run the memory-light Diffrax density streaming fit path."""

    t_total = float(dt * steps)
    tmin_i, tmax_i = _resolve_streaming_window(
        t_total,
        options.tmin,
        options.tmax,
        options.start_fraction,
        options.window_fraction,
        1.0,
    )
    G_last, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
        setup.initial_state,
        setup.grid,
        setup.geom,
        setup.params,
        dt=dt,
        steps=steps,
        method=time_cfg.diffrax_solver,
        cache=cache,
        terms=setup.terms,
        adaptive=False,
        rtol=time_cfg.diffrax_rtol,
        atol=time_cfg.diffrax_atol,
        max_steps=time_cfg.diffrax_max_steps,
        show_progress=show_progress,
        progress_bar=time_cfg.progress_bar,
        checkpoint=time_cfg.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal="density",
        mode_ky_indices=np.array([0], dtype=int),
        mode_kx_index=0,
        mode_z_index=_midplane_index(setup.grid),
        mode_method=options.mode_method,
        amp_floor=options.streaming_amp_floor,
        density_species_index=setup.electron_index,
        return_state=True,
    )
    gamma = float(np.asarray(gamma_vals)[0])
    omega = float(np.asarray(omega_vals)[0])
    gamma, omega = _normalize_growth_rate(
        gamma, omega, setup.params, options.diagnostic_norm
    )
    if G_last is not None and G_last.ndim == 7:
        G_last = G_last[0]
    if G_last is None:
        raise ValueError("Expected final state from streaming fit; got None.")
    term_cfg = linear_terms_to_term_config(setup.terms)
    phi_last = compute_fields_cached(
        G_last, cache, setup.params, terms=term_cfg
    ).phi
    return _etg_linear_result(
        setup,
        t=np.array([tmax_i]),
        phi_t_np=np.asarray(jnp.asarray(phi_last)[None, ...]),
        gamma=gamma,
        omega=omega,
    )


def _integrate_etg_configured_history(
    setup: _ETGLinearSetup,
    *,
    time_cfg: TimeConfig,
    cache: Any,
    dt: float,
    steps: int,
    fit_key: str,
    show_progress: bool,
) -> tuple[Any, Any | None, int]:
    """Integrate ETG saved history using an explicit or synthesized TimeConfig."""

    if fit_key in {"density", "auto"}:
        if time_cfg.use_diffrax:
            _, saved = integrate_linear_diffrax(
                setup.initial_state,
                setup.grid,
                setup.geom,
                setup.params,
                dt=dt,
                steps=steps,
                method=time_cfg.diffrax_solver,
                cache=cache,
                terms=setup.terms,
                adaptive=time_cfg.diffrax_adaptive,
                rtol=time_cfg.diffrax_rtol,
                atol=time_cfg.diffrax_atol,
                max_steps=time_cfg.diffrax_max_steps,
                show_progress=show_progress,
                progress_bar=time_cfg.progress_bar,
                checkpoint=time_cfg.checkpoint,
                sample_stride=time_cfg.sample_stride,
                return_state=time_cfg.save_state,
                save_field="phi+density",
                density_species_index=setup.electron_index,
            )
            phi_t, density_t = saved
            return phi_t, density_t, time_cfg.sample_stride
        diag = integrate_linear_diagnostics(
            setup.initial_state,
            setup.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=time_cfg.method,
            cache=cache,
            terms=setup.terms,
            sample_stride=time_cfg.sample_stride,
            species_index=setup.electron_index,
        )
        phi_t = diag[1]
        density_t = diag[2] if len(diag) > 2 else None
        return phi_t, density_t, time_cfg.sample_stride

    _, phi_t = integrate_linear_from_config(
        setup.initial_state,
        setup.grid,
        setup.geom,
        setup.params,
        time_cfg,
        cache=cache,
        terms=setup.terms,
        show_progress=show_progress,
    )
    return phi_t, None, time_cfg.sample_stride


def _integrate_etg_unconfigured_history(
    setup: _ETGLinearSetup,
    *,
    dt: float,
    steps: int,
    method: str,
    fit_key: str,
    sample_stride: int | None,
    show_progress: bool,
) -> tuple[Any, Any | None, int]:
    """Integrate ETG saved history without a TimeConfig object."""

    stride = 1 if sample_stride is None else int(sample_stride)
    if fit_key in {"density", "auto"}:
        diag = integrate_linear_diagnostics(
            setup.initial_state,
            setup.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=method,
            terms=setup.terms,
            sample_stride=stride,
            species_index=setup.electron_index,
        )
        phi_t = diag[1]
        density_t = diag[2] if len(diag) > 2 else None
        return phi_t, density_t, stride

    _, phi_t = integrate_linear(
        setup.initial_state,
        setup.grid,
        setup.geom,
        setup.params,
        dt=dt,
        steps=steps,
        method=method,
        terms=setup.terms,
        sample_stride=stride,
        show_progress=show_progress,
    )
    return phi_t, None, stride


def _run_etg_time_path(
    setup: _ETGLinearSetup,
    *,
    Nl: int,
    Nm: int,
    time_cfg: TimeConfig | None,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    options: _ETGTimePathOptions,
    show_progress: bool,
) -> LinearRunResult:
    """Run ETG saved-time or streaming time paths and fit the trace."""

    time_cfg_use, dt, steps = _resolve_etg_time_config(
        setup.cfg,
        time_cfg,
        streaming_fit=options.streaming_fit,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
    )
    if time_cfg_use is not None:
        cache = build_linear_cache(
            setup.grid,
            setup.geom,
            setup.params,
            Nl,
            Nm,
        )
        if (
            options.fit_key in {"density", "auto"}
            and options.streaming_fit
            and time_cfg_use.use_diffrax
        ):
            return _run_etg_streaming_density_fit(
                setup,
                time_cfg=time_cfg_use,
                cache=cache,
                dt=dt,
                steps=steps,
                options=options,
                show_progress=show_progress,
            )
        phi_t, density_t, stride = _integrate_etg_configured_history(
            setup,
            time_cfg=time_cfg_use,
            cache=cache,
            dt=dt,
            steps=steps,
            fit_key=options.fit_key,
            show_progress=show_progress,
        )
    else:
        phi_t, density_t, stride = _integrate_etg_unconfigured_history(
            setup,
            dt=dt,
            steps=steps,
            method=method,
            fit_key=options.fit_key,
            sample_stride=sample_stride,
            show_progress=show_progress,
        )

    return _fit_etg_time_trace(
        setup,
        phi_t=phi_t,
        density_t=density_t,
        dt=dt,
        stride=stride,
        options=options,
    )


def _run_etg_linear_request(request: _ETGLinearRequest) -> LinearRunResult:
    """Resolve ETG solver policies and execute one single-ky linear point."""

    setup = _build_etg_linear_setup(
        cfg=request.cfg,
        params=request.params,
        terms=request.terms,
        ky_target=request.ky_target,
        Nl=request.Nl,
        Nm=request.Nm,
    )
    solver_key = request.solver.strip().lower()
    fit_key = request.fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    streaming_fit = request.streaming_fit
    if fit_key == "auto" and streaming_fit:
        streaming_fit = False
    time_options = _ETGTimePathOptions(
        fit_key=fit_key,
        streaming_fit=streaming_fit,
        streaming_amp_floor=request.streaming_amp_floor,
        reference_growth_window=request.reference_growth_window,
        reference_navg_fraction=request.reference_navg_fraction,
        mode_method=request.mode_method,
        tmin=request.tmin,
        tmax=request.tmax,
        auto_window=request.auto_window,
        window_fraction=request.window_fraction,
        min_points=request.min_points,
        start_fraction=request.start_fraction,
        growth_weight=request.growth_weight,
        require_positive=request.require_positive,
        min_amp_fraction=request.min_amp_fraction,
        diagnostic_norm=request.diagnostic_norm,
    )
    auto_solver = solver_key == "auto"
    if auto_solver:
        solver_key = "krylov"

    if solver_key == "krylov":
        krylov_result = _run_etg_krylov_path(
            setup,
            Nl=request.Nl,
            Nm=request.Nm,
            krylov_cfg=request.krylov_cfg,
            diagnostic_norm=request.diagnostic_norm,
        )
        if auto_solver and not _valid_etg_growth(
            krylov_result.gamma,
            krylov_result.omega,
            require_positive=request.require_positive,
        ):
            solver_key = "time"
        else:
            return krylov_result

    if solver_key != "krylov":
        return _run_etg_time_path(
            setup,
            Nl=request.Nl,
            Nm=request.Nm,
            time_cfg=request.time_cfg,
            dt=request.dt,
            steps=request.steps,
            method=request.method,
            sample_stride=request.sample_stride,
            options=time_options,
            show_progress=request.show_progress,
        )

    raise ValueError(f"Unsupported ETG linear solver '{request.solver}'.")


def run_etg_linear(
    ky_target: float = 3.0,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
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
    mode_method: str = "project",
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    fit_signal: str = "auto",
    streaming_fit: bool = False,
    streaming_amp_floor: float = 1.0e-30,
    reference_growth_window: bool = False,
    reference_navg_fraction: float = 0.5,
    diagnostic_norm: str = "none",
    show_progress: bool = False,
) -> LinearRunResult:
    """Run an ETG linear benchmark and extract growth rate."""

    return _run_etg_linear_request(_etg_linear_request_from_locals(locals()))

@dataclass(frozen=True)
class _ETGScanSetup:
    cfg: ETGBaseCase
    grid_full: Any
    geom: Any
    params: LinearParams
    terms: LinearTerms
    solver_key: str
    auto_solver: bool
    fit_key: str
    need_density: bool
    streaming_fit: bool
    mode_method: str
    mode_only: bool
    use_batch: bool
    fit_policy: ScanFitWindowPolicy


@dataclass(frozen=True)
class _ETGScanBatch:
    batch_start: int
    ky_slice: np.ndarray
    valid_count: int
    ky_indices: list[int]
    grid: Any
    sel: ModeSelection | ModeSelectionBatch
    dt_i: float
    steps_i: int
    electron_index: int
    G0_jax: jnp.ndarray
    cache: Any


@dataclass(frozen=True)
class _ETGScanRuntimeOptions:
    """Runtime options that are constant across ETG scan batches."""

    time_cfg: TimeConfig | None
    method: str
    sample_stride: int | None
    streaming_amp_floor: float
    tmin: float | None
    tmax: float | None
    start_fraction: float
    window_fraction: float
    reference_growth_window: bool
    reference_navg_fraction: float
    require_positive: bool
    Nl: int
    Nm: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    ky_batch: int
    fixed_batch_shape: bool
    krylov_cfg: KrylovConfig | None
    diagnostic_norm: str
    show_progress: bool


@dataclass
class _ETGScanAccumulator:
    """Mutable scan output and continuation state for ETG batches."""

    gammas: list[float]
    omegas: list[float]
    ky_out: list[float]
    prev_vec: jnp.ndarray | None = None
    prev_eig: complex | None = None

    def result(self) -> LinearScanResult:
        """Pack accumulated scan rows into the public result object."""

        return LinearScanResult(
            ky=np.array(self.ky_out),
            gamma=np.array(self.gammas),
            omega=np.array(self.omegas),
        )


@dataclass(frozen=True)
class _ETGScanRequest:
    ky_values: np.ndarray
    Nl: int
    Nm: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    params: LinearParams | None
    cfg: ETGBaseCase | None
    time_cfg: TimeConfig | None
    solver: str
    krylov_cfg: KrylovConfig | None
    tmin: float | None
    tmax: float | None
    auto_window: bool
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    max_fraction: float
    end_fraction: float
    max_amp_fraction: float
    phase_weight: float
    length_weight: float
    min_r2: float
    late_penalty: float
    min_slope: float | None
    min_slope_frac: float
    slope_var_weight: float
    window_method: str
    mode_method: str
    mode_only: bool
    terms: LinearTerms | None
    sample_stride: int | None
    fit_signal: str
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    reference_growth_window: bool
    reference_navg_fraction: float
    diagnostic_norm: str
    show_progress: bool


def _etg_scan_request_from_locals(values: dict[str, Any]) -> _ETGScanRequest:
    """Pack public ``run_etg_scan`` arguments once for internal routing."""

    return _ETGScanRequest(
        **{field.name: values[field.name] for field in fields(_ETGScanRequest)}
    )


def _default_etg_scan_params(
    cfg: ETGBaseCase,
    geom: Any,
    Nm: int,
    params: LinearParams | None,
) -> LinearParams:
    """Return ETG scan species parameters using the tracked normalization."""

    if params is not None:
        return params
    species_builder = (
        _electron_only_params
        if getattr(cfg.model, "adiabatic_ions", False)
        else _two_species_params
    )
    return species_builder(
        cfg.model,
        kpar_scale=float(geom.gradpar()),
        omega_d_scale=ETG_OMEGA_D_SCALE,
        omega_star_scale=ETG_OMEGA_STAR_SCALE,
        rho_star=ETG_RHO_STAR,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
        nhermite=Nm,
    )


def _default_etg_scan_terms(terms: LinearTerms | None) -> LinearTerms:
    """Return the electrostatic ETG benchmark term contract."""

    if terms is not None:
        return terms
    # Keep the ETG scan helper on the same electrostatic benchmark contract as
    # the single-ky ETG wrapper and the tracked ETG figure builders.
    return LinearTerms(apar=0.0, bpar=0.0, hypercollisions=1.0)


def _build_etg_scan_fit_policy(
    *,
    tmin: float | None,
    tmax: float | None,
    auto_window: bool,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    max_fraction: float,
    end_fraction: float,
    max_amp_fraction: float,
    phase_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    min_slope: float | None,
    min_slope_frac: float,
    slope_var_weight: float,
    window_method: str,
) -> ScanFitWindowPolicy:
    """Build the ETG scan fit-window policy without changing fit formulas."""

    return ScanFitWindowPolicy(
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


def _prepare_etg_scan_setup(request: _ETGScanRequest) -> _ETGScanSetup:
    """Prepare ETG scan geometry, species, solver, and fit policies."""

    cfg_use = request.cfg or ETGBaseCase()
    grid_full = build_spectral_grid(cfg_use.grid)
    geom = SAlphaGeometry.from_config(cfg_use.geometry)
    params_use = _default_etg_scan_params(cfg_use, geom, request.Nm, request.params)
    terms_use = _default_etg_scan_terms(request.terms)
    solver_key = normalize_solver_key(request.solver)
    fit_key = normalize_fit_signal(request.fit_signal)
    auto_solver = solver_key == "auto"
    if auto_solver:
        solver_key = "time"
    streaming_fit, mode_only = apply_auto_fit_scan_policy(
        fit_key,
        streaming_fit=request.streaming_fit,
        mode_only=request.mode_only,
    )
    mode_method = resolve_scan_mode_method(request.mode_method, mode_only=mode_only)
    fit_policy = _build_etg_scan_fit_policy(
        tmin=request.tmin,
        tmax=request.tmax,
        auto_window=request.auto_window,
        window_fraction=request.window_fraction,
        min_points=request.min_points,
        start_fraction=request.start_fraction,
        growth_weight=request.growth_weight,
        require_positive=request.require_positive,
        min_amp_fraction=request.min_amp_fraction,
        max_fraction=request.max_fraction,
        end_fraction=request.end_fraction,
        max_amp_fraction=request.max_amp_fraction,
        phase_weight=request.phase_weight,
        length_weight=request.length_weight,
        min_r2=request.min_r2,
        late_penalty=request.late_penalty,
        min_slope=request.min_slope,
        min_slope_frac=request.min_slope_frac,
        slope_var_weight=request.slope_var_weight,
        window_method=request.window_method,
    )
    return _ETGScanSetup(
        cfg=cfg_use,
        grid_full=grid_full,
        geom=geom,
        params=params_use,
        terms=terms_use,
        solver_key=solver_key,
        auto_solver=auto_solver,
        fit_key=fit_key,
        need_density=fit_key in {"density", "auto"},
        streaming_fit=streaming_fit,
        mode_method=mode_method,
        mode_only=mode_only,
        use_batch=should_use_ky_batch(
            ky_batch=request.ky_batch,
            solver_key=solver_key,
            dt=request.dt,
            steps=request.steps,
            tmin=request.tmin,
            tmax=request.tmax,
        ),
        fit_policy=fit_policy,
    )


# ETG scan Krylov and time-batch policies live with the scan owner.
@dataclass(frozen=True)
class ETGTimeBatchResult:
    """Time-path data for one ETG scan batch after optional streaming handling."""

    handled: bool
    phi_t: np.ndarray | None = None
    density_t: np.ndarray | None = None
    t: np.ndarray | None = None
    stride: int = 1


@dataclass(frozen=True)
class _ETGTimeFitContext:
    ky_slice: np.ndarray
    valid_count: int
    batch_start: int
    fit_key: str
    fit_policy: Any
    params: Any
    diagnostic_norm: str
    mode_method: str
    mode_only: bool
    mode_z_index: int
    reference_growth_window: bool
    reference_navg_fraction: float
    auto_solver: bool
    require_positive: bool
    cfg: Any
    Nl: int
    Nm: int
    dt_i: float
    steps_i: int
    method: str
    krylov_cfg: Any
    show_progress: bool
    gammas: list[float]
    omegas: list[float]
    ky_out: list[float]


@dataclass(frozen=True)
class _ETGTimeBatchContext:
    G0_jax: jnp.ndarray
    grid: Any
    geom: Any
    params: Any
    cache: Any
    terms: Any
    time_cfg: Any
    dt_i: float
    steps_i: int
    method: str
    sample_stride: int | None
    fit_key: str
    need_density: bool
    streaming_fit: bool
    streaming_amp_floor: float
    mode_method: str
    mode_only: bool
    sel: Any
    batch_start: int
    valid_count: int
    ky_slice: np.ndarray
    tmin: float | None
    tmax: float | None
    start_fraction: float
    window_fraction: float
    electron_index: int
    diagnostic_norm: str
    show_progress: bool
    gammas: list[float]
    omegas: list[float]
    ky_out: list[float]


def _etg_time_batch_context_from_locals(values: dict[str, Any]) -> _ETGTimeBatchContext:
    """Pack ``run_etg_time_batch`` arguments for internal routing."""

    return _ETGTimeBatchContext(
        **{field.name: values[field.name] for field in fields(_ETGTimeBatchContext)}
    )


def run_etg_krylov_batch(
    *,
    G0_jax: jnp.ndarray,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg: Any,
    prev_vec: jnp.ndarray | None,
    prev_eig: complex | None,
    diagnostic_norm: str,
) -> tuple[float, float, jnp.ndarray | None, complex | None]:
    """Run one ETG Krylov scan point with continuation-aware branch selection."""

    cfg_use = krylov_cfg or ETG_KRYLOV_DEFAULT
    use_cont = bool(cfg_use.continuation)
    v0_use = G0_jax
    v_ref = None
    shift_override = cfg_use.shift
    shift_selection_use = cfg_use.shift_selection
    if use_cont and prev_vec is not None and prev_vec.shape == G0_jax.shape:
        v0_use = prev_vec
        v_ref = prev_vec
        if cfg_use.method.strip().lower() == "shift_invert" and prev_eig is not None:
            if shift_override is None:
                shift_override = prev_eig
                shift_selection_use = "shift"
    select_overlap = (
        use_cont
        and v_ref is not None
        and (cfg_use.continuation_selection.strip().lower() == "overlap")
    )
    krylov_kwargs = {
        "terms": terms,
        "v_ref": v_ref,
        "select_overlap": select_overlap,
        **{name: getattr(cfg_use, name) for name in _ETG_KRYLOV_FORWARD_KEYS},
        "shift": shift_override,
        "shift_selection": shift_selection_use,
    }
    eig, vec = dominant_eigenpair(v0_use, cache, params, **krylov_kwargs)
    if use_cont:
        eig_host = complex(np.asarray(eig))
        if np.isfinite(eig_host.real) and np.isfinite(eig_host.imag):
            prev_vec = vec
            prev_eig = eig_host
        else:
            prev_vec = None
            prev_eig = None
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    if cfg_use.omega_sign != 0:
        omega = float(np.sign(cfg_use.omega_sign)) * abs(omega)
    gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    return gamma, omega, prev_vec, prev_eig


def _etg_time_config_for_batch(context: _ETGTimeBatchContext) -> Any | None:
    if context.time_cfg is None:
        return None
    time_cfg_i = replace(
        context.time_cfg,
        dt=context.dt_i,
        t_max=context.dt_i * context.steps_i,
    )
    if context.sample_stride is not None:
        time_cfg_i = replace(time_cfg_i, sample_stride=context.sample_stride)
    return time_cfg_i


def _append_etg_streaming_time_results(
    context: _ETGTimeBatchContext,
    *,
    time_cfg_i: Any,
) -> None:
    t_total = float(time_cfg_i.t_max)
    tmin_i, tmax_i = _resolve_streaming_window(
        t_total,
        indexed_float_value(context.tmin, context.batch_start),
        indexed_float_value(context.tmax, context.batch_start),
        context.start_fraction,
        context.window_fraction,
        1.0,
    )
    _, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
        context.G0_jax,
        context.grid,
        context.geom,
        context.params,
        dt=context.dt_i,
        steps=context.steps_i,
        method=time_cfg_i.diffrax_solver,
        cache=context.cache,
        terms=context.terms,
        adaptive=time_cfg_i.diffrax_adaptive,
        rtol=time_cfg_i.diffrax_rtol,
        atol=time_cfg_i.diffrax_atol,
        max_steps=time_cfg_i.diffrax_max_steps,
        progress_bar=time_cfg_i.progress_bar,
        checkpoint=time_cfg_i.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal=context.fit_key,
        mode_ky_indices=np.arange(context.valid_count, dtype=int),
        mode_kx_index=0,
        mode_z_index=_midplane_index(context.grid),
        mode_method=context.mode_method,
        amp_floor=context.streaming_amp_floor,
        density_species_index=context.electron_index
        if context.fit_key == "density"
        else None,
        return_state=False,
    )
    gamma_arr = np.asarray(gamma_vals)
    omega_arr = np.asarray(omega_vals)
    for local_idx in range(context.valid_count):
        gamma_i, omega_i = _normalize_growth_rate(
            float(gamma_arr[local_idx]),
            float(omega_arr[local_idx]),
            context.params,
            context.diagnostic_norm,
        )
        context.gammas.append(gamma_i)
        context.omegas.append(omega_i)
        context.ky_out.append(float(context.ky_slice[local_idx]))


def _configured_etg_time_history(
    context: _ETGTimeBatchContext,
    *,
    time_cfg_i: Any,
) -> tuple[Any, Any | None, int]:
    save_field = (
        "phi+density"
        if context.fit_key == "auto"
        else ("density" if context.fit_key == "density" else "phi")
    )
    save_mode = None
    if context.fit_key != "auto" and context.mode_only and context.fit_key == "phi":
        save_mode = context.sel
    _, saved = integrate_linear_from_config(
        context.G0_jax,
        context.grid,
        context.geom,
        context.params,
        time_cfg_i,
        cache=context.cache,
        terms=context.terms,
        save_mode=save_mode,
        mode_method=context.mode_method,
        save_field=save_field,
        density_species_index=context.electron_index
        if context.need_density
        else None,
        show_progress=context.show_progress,
    )
    if context.fit_key == "auto":
        phi_t, density_t = saved
    else:
        phi_t = saved
        density_t = None
    return phi_t, density_t, int(time_cfg_i.sample_stride)


def _unconfigured_etg_time_history(
    context: _ETGTimeBatchContext,
) -> tuple[Any, Any | None, int]:
    stride = 1 if context.sample_stride is None else int(context.sample_stride)
    if context.need_density:
        diag = integrate_linear_diagnostics(
            context.G0_jax,
            context.grid,
            context.geom,
            context.params,
            dt=context.dt_i,
            steps=context.steps_i,
            method=context.method,
            cache=context.cache,
            terms=context.terms,
            sample_stride=stride,
            species_index=1,
            show_progress=context.show_progress,
        )
        return diag[1], diag[2] if len(diag) > 2 else None, stride
    _, phi_out_time = integrate_linear(
        context.G0_jax,
        context.grid,
        context.geom,
        context.params,
        dt=context.dt_i,
        steps=context.steps_i,
        method=context.method,
        cache=context.cache,
        terms=context.terms,
        sample_stride=stride,
        show_progress=context.show_progress,
    )
    return phi_out_time, None, stride


def _pack_etg_time_history_result(
    *,
    phi_t: Any,
    density_t: Any | None,
    dt_i: float,
    stride: int,
    fit_key: str,
) -> ETGTimeBatchResult:
    phi_t_np = np.asarray(phi_t)
    density_np = None if density_t is None else np.asarray(density_t)
    if fit_key == "density" and density_np is None:
        density_np = phi_t_np
    t = np.arange(phi_t_np.shape[0]) * dt_i * stride
    return ETGTimeBatchResult(
        handled=False,
        phi_t=phi_t_np,
        density_t=density_np,
        t=t,
        stride=stride,
    )


def run_etg_time_batch(
    *,
    G0_jax: jnp.ndarray,
    grid: Any,
    geom: Any,
    params: Any,
    cache: Any,
    terms: Any,
    time_cfg: Any,
    dt_i: float,
    steps_i: int,
    method: str,
    sample_stride: int | None,
    fit_key: str,
    need_density: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    mode_method: str,
    mode_only: bool,
    sel: Any,
    batch_start: int,
    valid_count: int,
    ky_slice: np.ndarray,
    tmin: float | None,
    tmax: float | None,
    start_fraction: float,
    window_fraction: float,
    electron_index: int,
    diagnostic_norm: str,
    show_progress: bool,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> ETGTimeBatchResult:
    """Integrate one ETG time-path batch and append streaming-fit results if used."""

    context = _etg_time_batch_context_from_locals(locals())
    time_cfg_i = _etg_time_config_for_batch(context)
    if time_cfg_i is not None and time_cfg_i.use_diffrax and streaming_fit:
        _append_etg_streaming_time_results(context, time_cfg_i=time_cfg_i)
        return ETGTimeBatchResult(handled=True)

    if time_cfg_i is not None:
        phi_t, density_t, stride = _configured_etg_time_history(
            context, time_cfg_i=time_cfg_i
        )
    else:
        phi_t, density_t, stride = _unconfigured_etg_time_history(context)
    return _pack_etg_time_history_result(
        phi_t=phi_t,
        density_t=density_t,
        dt_i=dt_i,
        stride=stride,
        fit_key=fit_key,
    )


def _etg_local_selection(local_idx: int, context: _ETGTimeFitContext) -> ModeSelection:
    return ModeSelection(
        ky_index=local_idx,
        kx_index=0,
        z_index=context.mode_z_index,
    )


def _auto_etg_time_fit(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> tuple[float, float]:
    if result.phi_t is None or result.t is None:
        raise ValueError("ETG time-batch result has no saved signal to fit")
    _signal, _name, gamma, omega = _select_fit_signal_auto(
        result.t,
        result.phi_t,
        result.density_t,
        _etg_local_selection(local_idx, context),
        mode_method=context.mode_method,
        tmin=indexed_float_value(context.fit_policy.tmin, context.batch_start + local_idx),
        tmax=indexed_float_value(context.fit_policy.tmax, context.batch_start + local_idx),
        window_fraction=context.fit_policy.window_fraction,
        min_points=context.fit_policy.min_points,
        start_fraction=context.fit_policy.start_fraction,
        growth_weight=context.fit_policy.growth_weight,
        require_positive=context.fit_policy.require_positive,
        min_amp_fraction=context.fit_policy.min_amp_fraction,
        max_amp_fraction=context.fit_policy.max_amp_fraction,
        window_method=context.fit_policy.window_method,
        max_fraction=context.fit_policy.max_fraction,
        end_fraction=context.fit_policy.end_fraction,
        num_windows=8,
        phase_weight=context.fit_policy.phase_weight,
        length_weight=context.fit_policy.length_weight,
        min_r2=context.fit_policy.min_r2,
        late_penalty=context.fit_policy.late_penalty,
        min_slope=context.fit_policy.min_slope,
        min_slope_frac=context.fit_policy.min_slope_frac,
        slope_var_weight=context.fit_policy.slope_var_weight,
    )
    return _normalize_growth_rate(
        gamma, omega, context.params, context.diagnostic_norm
    )


def _direct_etg_time_signal(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> np.ndarray:
    if result.phi_t is None:
        raise ValueError("ETG time-batch result has no saved signal to fit")
    if context.mode_only and context.fit_key == "phi" and result.phi_t.ndim <= 2:
        return _extract_mode_only_signal(result.phi_t, local_idx=local_idx)
    return _select_fit_signal(
        result.phi_t,
        result.density_t,
        _etg_local_selection(local_idx, context),
        fit_signal=context.fit_key,
        mode_method=context.mode_method,
    )


def _reference_window_etg_time_fit(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> tuple[float, float]:
    if result.phi_t is None or result.t is None:
        raise ValueError("ETG time-batch result has no saved signal to fit")
    gamma, omega, _gamma_t, _omega_t, _t_mid = instantaneous_growth_rate_from_phi(
        result.phi_t,
        result.t,
        _etg_local_selection(local_idx, context),
        navg_fraction=context.reference_navg_fraction,
        mode_method=context.mode_method,
    )
    return _normalize_growth_rate(
        gamma, omega, context.params, context.diagnostic_norm
    )


def _direct_etg_time_fit(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> tuple[float, float]:
    if context.reference_growth_window and context.fit_key == "phi":
        return _reference_window_etg_time_fit(
            result, local_idx=local_idx, context=context
        )
    signal = _direct_etg_time_signal(result, local_idx=local_idx, context=context)
    return context.fit_policy.fit_signal(
        signal,
        idx=context.batch_start + local_idx,
        dt=context.dt_i,
        stride=result.stride,
        params=context.params,
        diagnostic_norm=context.diagnostic_norm,
    )


def _resolve_etg_time_fit(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> tuple[float, float]:
    if context.fit_key == "auto":
        return _auto_etg_time_fit(result, local_idx=local_idx, context=context)
    return _direct_etg_time_fit(result, local_idx=local_idx, context=context)


def _fallback_etg_krylov_fit(
    *,
    ky_val: float,
    context: _ETGTimeFitContext,
) -> tuple[float, float]:
    res = run_etg_linear(
        ky_target=float(ky_val),
        cfg=context.cfg,
        Nl=context.Nl,
        Nm=context.Nm,
        dt=context.dt_i,
        steps=context.steps_i,
        method=context.method,
        params=context.params,
        solver="krylov",
        krylov_cfg=context.krylov_cfg,
        diagnostic_norm=context.diagnostic_norm,
        fit_signal="phi",
        show_progress=context.show_progress,
    )
    return float(res.gamma), float(res.omega)


def _append_resolved_etg_time_fit(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> None:
    ky_val = float(context.ky_slice[local_idx])
    gamma, omega = _resolve_etg_time_fit(
        result, local_idx=local_idx, context=context
    )
    if context.auto_solver and not _valid_etg_growth(
        gamma, omega, require_positive=context.require_positive
    ):
        gamma, omega = _fallback_etg_krylov_fit(ky_val=ky_val, context=context)
    context.gammas.append(float(gamma))
    context.omegas.append(float(omega))
    context.ky_out.append(ky_val)


def append_etg_time_fit_results(
    *,
    result: ETGTimeBatchResult,
    ky_slice: np.ndarray,
    valid_count: int,
    batch_start: int,
    fit_key: str,
    fit_policy: Any,
    params: Any,
    diagnostic_norm: str,
    mode_method: str,
    mode_only: bool,
    mode_z_index: int,
    reference_growth_window: bool,
    reference_navg_fraction: float,
    auto_solver: bool,
    require_positive: bool,
    cfg: Any,
    Nl: int,
    Nm: int,
    dt_i: float,
    steps_i: int,
    method: str,
    krylov_cfg: Any,
    show_progress: bool,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> None:
    """Fit and append ETG growth/frequency values from a saved time batch."""

    if result.phi_t is None or result.t is None:
        raise ValueError("ETG time-batch result has no saved signal to fit")
    context = _ETGTimeFitContext(
        ky_slice=ky_slice,
        valid_count=valid_count,
        batch_start=batch_start,
        fit_key=fit_key,
        fit_policy=fit_policy,
        params=params,
        diagnostic_norm=diagnostic_norm,
        mode_method=mode_method,
        mode_only=mode_only,
        mode_z_index=mode_z_index,
        reference_growth_window=reference_growth_window,
        reference_navg_fraction=reference_navg_fraction,
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
    for local_idx in range(valid_count):
        _append_resolved_etg_time_fit(result, local_idx=local_idx, context=context)


def _etg_scan_ky_batches(
    ky_values: np.ndarray,
    *,
    use_batch: bool,
    ky_batch: int,
    fixed_batch_shape: bool,
):
    """Yield ETG scan ky batches with the requested fixed-shape policy."""

    ky_values_arr = np.asarray(ky_values, dtype=float)
    if use_batch:
        return _iter_ky_batches(
            ky_values_arr,
            ky_batch=ky_batch,
            fixed_batch_shape=fixed_batch_shape,
        )
    return _iter_ky_batches(ky_values_arr, ky_batch=1, fixed_batch_shape=False)


def _build_etg_scan_batch(
    setup: _ETGScanSetup,
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
) -> _ETGScanBatch:
    """Build the grid, initial condition, and cache for one ETG scan batch."""

    sel: ModeSelection | ModeSelectionBatch
    if setup.use_batch:
        ky_indices = [
            select_ky_index(np.asarray(setup.grid_full.ky), float(ky))
            for ky in ky_slice
        ]
        grid = select_ky_grid(setup.grid_full, ky_indices)
        sel = ModeSelectionBatch(
            np.arange(len(ky_indices), dtype=int),
            0,
            _midplane_index(grid),
        )
        dt_i = float(dt)
        steps_i = int(steps)
    else:
        ky_indices = [select_ky_index(np.asarray(setup.grid_full.ky), float(ky_slice[0]))]
        grid = select_ky_grid(setup.grid_full, ky_indices[0])
        sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
        dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)

    charge = np.atleast_1d(np.asarray(setup.params.charge_sign))
    electron_index = int(np.argmin(charge))
    G0 = np.zeros(
        (
            int(charge.size),
            Nl,
            Nm,
            grid.ky.size,
            grid.kx.size,
            grid.z.size,
        ),
        dtype=np.complex64,
    )
    G0_single = _build_initial_condition(
        grid,
        setup.geom,
        ky_index=np.arange(len(ky_indices), dtype=int),
        kx_index=0,
        Nl=Nl,
        Nm=Nm,
        init_cfg=setup.cfg.init,
    )
    G0[electron_index] = np.asarray(G0_single, dtype=np.complex64)
    return _ETGScanBatch(
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        ky_indices=ky_indices,
        grid=grid,
        sel=sel,
        dt_i=dt_i,
        steps_i=steps_i,
        electron_index=electron_index,
        G0_jax=jnp.asarray(G0),
        cache=build_linear_cache(grid, setup.geom, setup.params, Nl, Nm),
    )


def _run_etg_scan_krylov_batch(
    setup: _ETGScanSetup,
    batch: _ETGScanBatch,
    *,
    options: _ETGScanRuntimeOptions,
    acc: _ETGScanAccumulator,
) -> tuple[jnp.ndarray | None, complex | None]:
    """Run one Krylov ETG scan batch and update continuation state."""

    gamma, omega, prev_vec, prev_eig = run_etg_krylov_batch(
        G0_jax=batch.G0_jax,
        cache=batch.cache,
        params=setup.params,
        terms=setup.terms,
        krylov_cfg=options.krylov_cfg,
        prev_vec=acc.prev_vec,
        prev_eig=acc.prev_eig,
        diagnostic_norm=options.diagnostic_norm,
    )
    acc.gammas.append(gamma)
    acc.omegas.append(omega)
    acc.ky_out.append(float(batch.ky_slice[0]))
    return prev_vec, prev_eig


def _append_etg_scan_time_fit_results(
    setup: _ETGScanSetup,
    batch: _ETGScanBatch,
    *,
    options: _ETGScanRuntimeOptions,
    acc: _ETGScanAccumulator,
    time_result: Any,
) -> None:
    """Append fitted ETG time-batch results using the shared path helper."""
    append_etg_time_fit_results(
        result=time_result,
        ky_slice=batch.ky_slice,
        valid_count=batch.valid_count,
        batch_start=batch.batch_start,
        fit_key=setup.fit_key,
        fit_policy=setup.fit_policy,
        params=setup.params,
        diagnostic_norm=options.diagnostic_norm,
        mode_method=setup.mode_method,
        mode_only=setup.mode_only,
        mode_z_index=_midplane_index(batch.grid),
        reference_growth_window=options.reference_growth_window,
        reference_navg_fraction=options.reference_navg_fraction,
        auto_solver=setup.auto_solver,
        require_positive=options.require_positive,
        cfg=setup.cfg,
        Nl=options.Nl,
        Nm=options.Nm,
        dt_i=batch.dt_i,
        steps_i=batch.steps_i,
        method=options.method,
        krylov_cfg=options.krylov_cfg,
        show_progress=options.show_progress,
        gammas=acc.gammas,
        omegas=acc.omegas,
        ky_out=acc.ky_out,
    )


def _run_etg_scan_time_batch(
    setup: _ETGScanSetup,
    batch: _ETGScanBatch,
    *,
    options: _ETGScanRuntimeOptions,
    acc: _ETGScanAccumulator,
) -> tuple[jnp.ndarray | None, complex | None]:
    """Run one time-integrated ETG scan batch and append fit results."""

    time_result = run_etg_time_batch(
        G0_jax=batch.G0_jax,
        grid=batch.grid,
        geom=setup.geom,
        params=setup.params,
        cache=batch.cache,
        terms=setup.terms,
        time_cfg=options.time_cfg,
        dt_i=batch.dt_i,
        steps_i=batch.steps_i,
        method=options.method,
        sample_stride=options.sample_stride,
        fit_key=setup.fit_key,
        need_density=setup.need_density,
        streaming_fit=setup.streaming_fit,
        streaming_amp_floor=options.streaming_amp_floor,
        mode_method=setup.mode_method,
        mode_only=setup.mode_only,
        sel=batch.sel,
        batch_start=batch.batch_start,
        valid_count=batch.valid_count,
        ky_slice=batch.ky_slice,
        tmin=options.tmin,
        tmax=options.tmax,
        start_fraction=options.start_fraction,
        window_fraction=options.window_fraction,
        electron_index=batch.electron_index,
        diagnostic_norm=options.diagnostic_norm,
        show_progress=options.show_progress,
        gammas=acc.gammas,
        omegas=acc.omegas,
        ky_out=acc.ky_out,
    )
    if not time_result.handled:
        _append_etg_scan_time_fit_results(
            setup,
            batch,
            options=options,
            acc=acc,
            time_result=time_result,
        )
    return acc.prev_vec, acc.prev_eig


def _run_etg_scan_batch(
    setup: _ETGScanSetup,
    batch: _ETGScanBatch,
    *,
    options: _ETGScanRuntimeOptions,
    acc: _ETGScanAccumulator,
) -> tuple[jnp.ndarray | None, complex | None]:
    """Run one ETG scan batch and append growth/frequency outputs."""

    if setup.solver_key == "krylov":
        return _run_etg_scan_krylov_batch(setup, batch, options=options, acc=acc)
    return _run_etg_scan_time_batch(setup, batch, options=options, acc=acc)


def _run_etg_scan_loop(
    setup: _ETGScanSetup,
    ky_values: np.ndarray,
    options: _ETGScanRuntimeOptions,
) -> _ETGScanAccumulator:
    """Run all ETG scan batches and preserve Krylov continuation state."""

    acc = _ETGScanAccumulator(gammas=[], omegas=[], ky_out=[])
    ky_iter = _etg_scan_ky_batches(
        ky_values,
        use_batch=setup.use_batch,
        ky_batch=options.ky_batch,
        fixed_batch_shape=options.fixed_batch_shape,
    )
    for batch_start, ky_slice, valid_count in ky_iter:
        batch = _build_etg_scan_batch(
            setup,
            batch_start=batch_start,
            ky_slice=ky_slice,
            valid_count=valid_count,
            Nl=options.Nl,
            Nm=options.Nm,
            dt=options.dt,
            steps=options.steps,
        )
        acc.prev_vec, acc.prev_eig = _run_etg_scan_batch(
            setup,
            batch,
            options=options,
            acc=acc,
        )
    return acc


def _etg_scan_setup_from_request(request: _ETGScanRequest) -> _ETGScanSetup:
    return _prepare_etg_scan_setup(request)


def _etg_scan_runtime_options_from_request(
    request: _ETGScanRequest,
) -> _ETGScanRuntimeOptions:
    return _ETGScanRuntimeOptions(
        time_cfg=request.time_cfg,
        method=request.method,
        sample_stride=request.sample_stride,
        streaming_amp_floor=request.streaming_amp_floor,
        tmin=request.tmin,
        tmax=request.tmax,
        start_fraction=request.start_fraction,
        window_fraction=request.window_fraction,
        reference_growth_window=request.reference_growth_window,
        reference_navg_fraction=request.reference_navg_fraction,
        require_positive=request.require_positive,
        Nl=request.Nl,
        Nm=request.Nm,
        dt=request.dt,
        steps=request.steps,
        ky_batch=request.ky_batch,
        fixed_batch_shape=request.fixed_batch_shape,
        krylov_cfg=request.krylov_cfg,
        diagnostic_norm=request.diagnostic_norm,
        show_progress=request.show_progress,
    )


def _run_etg_scan_request(request: _ETGScanRequest) -> LinearScanResult:
    setup = _etg_scan_setup_from_request(request)
    options = _etg_scan_runtime_options_from_request(request)
    return _run_etg_scan_loop(setup, request.ky_values, options).result()


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
    reference_growth_window: bool = False,
    reference_navg_fraction: float = 0.5,
    diagnostic_norm: str = "none",
    show_progress: bool = False,
) -> LinearScanResult:
    """Run an ETG linear benchmark for a list of ky values.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    return _run_etg_scan_request(_etg_scan_request_from_locals(locals()))

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
class _KineticScanSetup:
    grid_full: Any
    geom: Any
    params: LinearParams
    terms: LinearTerms
    init_cfg: Any
    diagnostic_norm: str
    reference_aligned: bool


@dataclass(frozen=True)
class _KineticScanBatch:
    batch_start: int
    ky_slice: np.ndarray
    valid_count: int
    grid: Any
    selection: ModeSelection | ModeSelectionBatch
    dt: float
    steps: int
    state: Any
    cache: Any


@dataclass(frozen=True)
class _KineticScanRunOptions:
    ky_values: np.ndarray
    time_cfg: TimeConfig | None
    solver_key: str
    krylov_cfg: KrylovConfig | None
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    sample_stride: int | None
    mode_method: str
    mode_only: bool
    fit_key: str
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    init_species_index: int
    density_species_index: int
    use_batch: bool
    show_progress: bool


@dataclass(frozen=True)
class _KineticScanFitOptions:
    tmin: float | None
    tmax: float | None
    start_fraction: float
    window_fraction: float
    fit_policy: ScanFitWindowPolicy


@dataclass
class _KineticScanOutput:
    gammas: list[float]
    omegas: list[float]
    ky: list[float]

    @classmethod
    def empty(cls) -> "_KineticScanOutput":
        return cls(gammas=[], omegas=[], ky=[])


@dataclass(frozen=True)
class _KineticScanControls:
    setup: _KineticScanSetup
    run_options: _KineticScanRunOptions
    fit_options: _KineticScanFitOptions


@dataclass(frozen=True)
class _KineticScanControlRequest:
    """Raw public scan inputs before setup, run, and fit policies are resolved."""

    ky_values: np.ndarray
    Nm: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    params: LinearParams | None
    cfg: KineticElectronBaseCase | None
    time_cfg: TimeConfig | None
    solver: str
    krylov_cfg: KrylovConfig | None
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
    reference_aligned: bool | None
    show_progress: bool


def _kinetic_scan_control_request_from_locals(
    values: dict[str, Any],
) -> _KineticScanControlRequest:
    """Build a request from ``run_kinetic_scan`` locals without forwarding Nl."""

    names = {field.name for field in fields(_KineticScanControlRequest)}
    return _KineticScanControlRequest(**{name: values[name] for name in names})


def _resolve_kinetic_scan_setup(
    *,
    cfg: KineticElectronBaseCase | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    diagnostic_norm: str,
    reference_aligned: bool | None,
    Nm: int,
) -> _KineticScanSetup:
    cfg_use = cfg or KineticElectronBaseCase()
    grid_full = build_spectral_grid(cfg_use.grid)
    geom = SAlphaGeometry.from_config(cfg_use.geometry)
    reference_aligned_use = bool(
        True if reference_aligned is None else reference_aligned
    )
    diagnostic_norm_use = diagnostic_norm
    if reference_aligned_use and diagnostic_norm_use == "none":
        diagnostic_norm_use = "rho_star"
    init_cfg = _kinetic_reference_init_cfg(
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
    return _KineticScanSetup(
        grid_full=grid_full,
        geom=geom,
        params=params_use,
        terms=terms_use,
        init_cfg=init_cfg,
        diagnostic_norm=diagnostic_norm_use,
        reference_aligned=reference_aligned_use,
    )


def _iter_kinetic_scan_batches(options: _KineticScanRunOptions):
    if options.use_batch:
        return _iter_ky_batches(
            options.ky_values,
            ky_batch=options.ky_batch,
            fixed_batch_shape=options.fixed_batch_shape,
        )
    return _iter_ky_batches(options.ky_values, ky_batch=1, fixed_batch_shape=False)


def _prepare_kinetic_scan_batch(
    setup: _KineticScanSetup,
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    use_batch: bool,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    Nl: int,
    Nm: int,
    init_species_index: int,
) -> _KineticScanBatch:
    if use_batch:
        ky_indices = [
            select_ky_index(np.asarray(setup.grid_full.ky), float(ky))
            for ky in ky_slice
        ]
        grid = select_ky_grid(setup.grid_full, ky_indices)
        sel_indices = np.arange(len(ky_indices), dtype=int)
        selection: ModeSelection | ModeSelectionBatch = ModeSelectionBatch(
            sel_indices, 0, _midplane_index(grid)
        )
        dt_i = float(dt)
        steps_i = int(steps)
    else:
        ky_indices = [
            select_ky_index(np.asarray(setup.grid_full.ky), float(ky_slice[0]))
        ]
        grid = select_ky_grid(setup.grid_full, ky_indices[0])
        selection = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
        dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)

    nspecies = 2
    G0 = np.zeros(
        (nspecies, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=np.complex64,
    )
    G0_single = _build_initial_condition(
        grid,
        setup.geom,
        ky_index=np.arange(len(ky_indices), dtype=int),
        kx_index=0,
        Nl=Nl,
        Nm=Nm,
        init_cfg=setup.init_cfg,
    )
    G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)
    state = jnp.asarray(G0)
    cache = build_linear_cache(grid, setup.geom, setup.params, Nl, Nm)
    return _KineticScanBatch(
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        grid=grid,
        selection=selection,
        dt=dt_i,
        steps=steps_i,
        state=state,
        cache=cache,
    )


def _kinetic_scan_time_config(
    time_cfg: TimeConfig | None,
    *,
    dt: float,
    steps: int,
    sample_stride: int | None,
) -> TimeConfig | None:
    if time_cfg is None:
        return None
    time_cfg_i = replace(time_cfg, dt=dt, t_max=dt * steps)
    if sample_stride is not None:
        time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)
    return time_cfg_i


def _run_kinetic_scan_krylov(
    batch: _KineticScanBatch,
    setup: _KineticScanSetup,
    krylov_cfg: KrylovConfig | None,
) -> tuple[float, float]:
    cfg_use = krylov_cfg or (
        KINETIC_KRYLOV_REFERENCE_ALIGNED
        if setup.reference_aligned
        else KINETIC_KRYLOV_DEFAULT
    )
    eig, _vec = dominant_eigenpair(
        batch.state,
        batch.cache,
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
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    return _normalize_growth_rate(
        gamma, omega, setup.params, setup.diagnostic_norm
    )


def _append_kinetic_streaming_results(
    batch: _KineticScanBatch,
    setup: _KineticScanSetup,
    *,
    time_cfg: TimeConfig,
    fit_key: str,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    start_fraction: float,
    window_fraction: float,
    streaming_amp_floor: float,
    density_species_index: int,
    output: _KineticScanOutput,
) -> None:
    t_total = float(time_cfg.t_max)
    tmin_i, tmax_i = _resolve_streaming_window(
        t_total,
        indexed_float_value(tmin, batch.batch_start),
        indexed_float_value(tmax, batch.batch_start),
        start_fraction,
        window_fraction,
        1.0,
    )
    _, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
        batch.state,
        batch.grid,
        setup.geom,
        setup.params,
        dt=batch.dt,
        steps=batch.steps,
        method=time_cfg.diffrax_solver,
        cache=batch.cache,
        terms=setup.terms,
        adaptive=time_cfg.diffrax_adaptive,
        rtol=time_cfg.diffrax_rtol,
        atol=time_cfg.diffrax_atol,
        max_steps=time_cfg.diffrax_max_steps,
        progress_bar=time_cfg.progress_bar,
        checkpoint=time_cfg.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal=fit_key,
        mode_ky_indices=np.arange(batch.valid_count, dtype=int),
        mode_kx_index=0,
        mode_z_index=_midplane_index(batch.grid),
        mode_method=mode_method,
        amp_floor=streaming_amp_floor,
        density_species_index=density_species_index if fit_key == "density" else None,
        return_state=False,
    )
    gamma_arr = np.asarray(gamma_vals)
    omega_arr = np.asarray(omega_vals)
    for local_idx in range(batch.valid_count):
        gamma_i, omega_i = _normalize_growth_rate(
            float(gamma_arr[local_idx]),
            float(omega_arr[local_idx]),
            setup.params,
            setup.diagnostic_norm,
        )
        output.gammas.append(gamma_i)
        output.omegas.append(omega_i)
        output.ky.append(float(batch.ky_slice[local_idx]))


def _integrate_kinetic_scan_history(
    batch: _KineticScanBatch,
    setup: _KineticScanSetup,
    *,
    time_cfg: TimeConfig | None,
    method: str,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    sample_stride: int | None,
    density_species_index: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    if time_cfg is not None:
        save_mode_method = mode_method if mode_method in {"z_index", "max"} else "z_index"
        _, phi_t = integrate_linear_from_config(
            batch.state,
            batch.grid,
            setup.geom,
            setup.params,
            time_cfg,
            cache=batch.cache,
            terms=setup.terms,
            save_mode=batch.selection if (mode_only and fit_key == "phi") else None,
            mode_method=save_mode_method,
            save_field="density" if fit_key == "density" else "phi",
            density_species_index=density_species_index
            if fit_key == "density"
            else None,
        )
        return np.asarray(phi_t), None, time_cfg.sample_stride

    stride = 1 if sample_stride is None else int(sample_stride)
    if fit_key == "density":
        _diag = integrate_linear_diagnostics(
            batch.state,
            batch.grid,
            setup.geom,
            setup.params,
            dt=batch.dt,
            steps=batch.steps,
            method=method,
            cache=batch.cache,
            terms=setup.terms,
            sample_stride=stride,
            species_index=density_species_index,
        )
        phi_t = _diag[1]
        density_t = _diag[2] if len(_diag) > 2 else None
    else:
        _, phi_t = integrate_linear(
            batch.state,
            batch.grid,
            setup.geom,
            setup.params,
            dt=batch.dt,
            steps=batch.steps,
            method=method,
            cache=batch.cache,
            terms=setup.terms,
            sample_stride=stride,
            show_progress=show_progress,
        )
        density_t = None
    return (
        np.asarray(phi_t),
        None if density_t is None else np.asarray(density_t),
        stride,
    )


def _append_kinetic_sampled_results(
    batch: _KineticScanBatch,
    setup: _KineticScanSetup,
    *,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    fit_policy: ScanFitWindowPolicy,
    density_species_index: int,
    stride: int,
    output: _KineticScanOutput,
) -> None:
    density_np = phi_t if fit_key == "density" and density_t is None else density_t
    for local_idx in range(batch.valid_count):
        if mode_only and fit_key == "phi" and phi_t.ndim <= 2:
            signal = _extract_mode_only_signal(phi_t, local_idx=local_idx)
        elif (
            mode_only
            and fit_key == "density"
            and density_np is not None
            and density_np.ndim <= 3
        ):
            signal = _extract_mode_only_signal(
                density_np,
                local_idx=local_idx,
                species_index=density_species_index,
            )
        else:
            sel_local = ModeSelection(
                ky_index=local_idx,
                kx_index=0,
                z_index=_midplane_index(batch.grid),
            )
            signal = _select_fit_signal(
                phi_t,
                density_np,
                sel_local,
                fit_signal=fit_key,
                mode_method=mode_method,
            )
        gamma, omega = fit_policy.fit_signal(
            signal,
            idx=batch.batch_start + local_idx,
            dt=batch.dt,
            stride=stride,
            params=setup.params,
            diagnostic_norm=setup.diagnostic_norm,
        )
        output.gammas.append(gamma)
        output.omegas.append(omega)
        output.ky.append(float(batch.ky_slice[local_idx]))


def _append_kinetic_krylov_result(
    batch: _KineticScanBatch,
    setup: _KineticScanSetup,
    *,
    krylov_cfg: KrylovConfig | None,
    output: _KineticScanOutput,
) -> None:
    gamma, omega = _run_kinetic_scan_krylov(batch, setup, krylov_cfg)
    output.gammas.append(gamma)
    output.omegas.append(omega)
    output.ky.append(float(batch.ky_slice[0]))


def _append_kinetic_time_batch_results(
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    setup: _KineticScanSetup,
    run_options: _KineticScanRunOptions,
    fit_options: _KineticScanFitOptions,
    Nl: int,
    Nm: int,
    output: _KineticScanOutput,
) -> None:
    batch = _prepare_kinetic_scan_batch(
        setup,
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        use_batch=run_options.use_batch,
        dt=run_options.dt,
        steps=run_options.steps,
        Nl=Nl,
        Nm=Nm,
        init_species_index=run_options.init_species_index,
    )
    if run_options.solver_key == "krylov":
        _append_kinetic_krylov_result(
            batch, setup, krylov_cfg=run_options.krylov_cfg, output=output
        )
        return

    time_cfg_i = _kinetic_scan_time_config(
        run_options.time_cfg,
        dt=batch.dt,
        steps=batch.steps,
        sample_stride=run_options.sample_stride,
    )
    if time_cfg_i is not None and time_cfg_i.use_diffrax and run_options.streaming_fit:
        _append_kinetic_streaming_results(
            batch,
            setup,
            time_cfg=time_cfg_i,
            fit_key=run_options.fit_key,
            mode_method=run_options.mode_method,
            tmin=fit_options.tmin,
            tmax=fit_options.tmax,
            start_fraction=fit_options.start_fraction,
            window_fraction=fit_options.window_fraction,
            streaming_amp_floor=run_options.streaming_amp_floor,
            density_species_index=run_options.density_species_index,
            output=output,
        )
        return

    phi_t, density_t, stride = _integrate_kinetic_scan_history(
        batch,
        setup,
        time_cfg=time_cfg_i,
        method=run_options.method,
        fit_key=run_options.fit_key,
        mode_only=run_options.mode_only,
        mode_method=run_options.mode_method,
        sample_stride=run_options.sample_stride,
        density_species_index=run_options.density_species_index,
        show_progress=run_options.show_progress,
    )
    _append_kinetic_sampled_results(
        batch,
        setup,
        phi_t=phi_t,
        density_t=density_t,
        fit_key=run_options.fit_key,
        mode_only=run_options.mode_only,
        mode_method=run_options.mode_method,
        fit_policy=fit_options.fit_policy,
        density_species_index=run_options.density_species_index,
        stride=stride,
        output=output,
    )


def _build_kinetic_scan_fit_policy(
    *,
    tmin: float | None,
    tmax: float | None,
    auto_window: bool,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> ScanFitWindowPolicy:
    """Pack kinetic-scan growth-window options once for all batches."""

    return ScanFitWindowPolicy(
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


def _build_kinetic_scan_run_options(
    *,
    ky_values: np.ndarray,
    time_cfg: TimeConfig | None,
    solver_key: str,
    krylov_cfg: KrylovConfig | None,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    sample_stride: int | None,
    mode_method: str,
    mode_only: bool,
    fit_key: str,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    init_species_index: int,
    density_species_index: int,
    use_batch: bool,
    show_progress: bool,
) -> _KineticScanRunOptions:
    """Pack kinetic scan runtime controls for batch execution."""

    return _KineticScanRunOptions(
        ky_values=ky_values,
        time_cfg=time_cfg,
        solver_key=solver_key,
        krylov_cfg=krylov_cfg,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        mode_method=mode_method,
        mode_only=mode_only,
        fit_key=fit_key,
        ky_batch=ky_batch,
        fixed_batch_shape=fixed_batch_shape,
        streaming_fit=streaming_fit,
        streaming_amp_floor=streaming_amp_floor,
        init_species_index=init_species_index,
        density_species_index=density_species_index,
        use_batch=use_batch,
        show_progress=show_progress,
    )


def _run_kinetic_scan_batches(
    *,
    setup: _KineticScanSetup,
    run_options: _KineticScanRunOptions,
    fit_options: _KineticScanFitOptions,
    n_laguerre: int,
    n_hermite: int,
) -> _KineticScanOutput:
    """Execute all kinetic ky batches and collect scan rows."""

    output = _KineticScanOutput.empty()
    for batch_start, ky_slice, valid_count in _iter_kinetic_scan_batches(run_options):
        _append_kinetic_time_batch_results(
            batch_start=batch_start,
            ky_slice=ky_slice,
            valid_count=valid_count,
            setup=setup,
            run_options=run_options,
            fit_options=fit_options,
            Nl=n_laguerre,
            Nm=n_hermite,
            output=output,
        )
    return output


def _kinetic_scan_result(output: _KineticScanOutput) -> LinearScanResult:
    """Pack kinetic scan rows into the public scan result."""

    return LinearScanResult(
        ky=np.array(output.ky),
        gamma=np.array(output.gammas),
        omega=np.array(output.omegas),
    )


def _prepare_kinetic_scan_controls(
    request: _KineticScanControlRequest,
) -> _KineticScanControls:
    """Resolve setup, execution, and fitting controls for one kinetic scan."""

    setup = _resolve_kinetic_scan_setup(
        cfg=request.cfg,
        params=request.params,
        terms=request.terms,
        diagnostic_norm=request.diagnostic_norm,
        reference_aligned=request.reference_aligned,
        Nm=request.Nm,
    )
    solver_key = normalize_solver_key(request.solver)
    fit_key = normalize_fit_signal(request.fit_signal)
    resolved_mode_method = resolve_scan_mode_method(
        request.mode_method,
        mode_only=request.mode_only,
    )
    use_batch = should_use_ky_batch(
        ky_batch=request.ky_batch,
        solver_key=solver_key,
        dt=request.dt,
        steps=request.steps,
        tmin=request.tmin,
        tmax=request.tmax,
    )
    fit_policy = _build_kinetic_scan_fit_policy(
        tmin=request.tmin,
        tmax=request.tmax,
        auto_window=request.auto_window,
        window_fraction=request.window_fraction,
        min_points=request.min_points,
        start_fraction=request.start_fraction,
        growth_weight=request.growth_weight,
        require_positive=request.require_positive,
        min_amp_fraction=request.min_amp_fraction,
    )

    ky_values_arr = np.asarray(request.ky_values, dtype=float)
    _validate_kinetic_species_indices(
        init_species_index=request.init_species_index,
        density_species_index=request.density_species_index,
    )
    run_options = _build_kinetic_scan_run_options(
        ky_values=ky_values_arr,
        time_cfg=request.time_cfg,
        solver_key=solver_key,
        krylov_cfg=request.krylov_cfg,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        sample_stride=request.sample_stride,
        mode_method=resolved_mode_method,
        mode_only=request.mode_only,
        fit_key=fit_key,
        ky_batch=request.ky_batch,
        fixed_batch_shape=request.fixed_batch_shape,
        streaming_fit=request.streaming_fit,
        streaming_amp_floor=request.streaming_amp_floor,
        init_species_index=request.init_species_index,
        density_species_index=request.density_species_index,
        use_batch=use_batch,
        show_progress=request.show_progress,
    )
    fit_options = _KineticScanFitOptions(
        tmin=request.tmin,
        tmax=request.tmax,
        start_fraction=request.start_fraction,
        window_fraction=request.window_fraction,
        fit_policy=fit_policy,
    )
    return _KineticScanControls(
        setup=setup,
        run_options=run_options,
        fit_options=fit_options,
    )


def run_kinetic_scan(
    ky_values: np.ndarray,
    Nl: int = 6,
    Nm: int = 12,
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
    method: str = "imex2",
    params: LinearParams | None = None,
    cfg: KineticElectronBaseCase | None = None,
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
    fit_signal: str = "density",
    ky_batch: int = 4,
    fixed_batch_shape: bool = True,
    streaming_fit: bool = True,
    streaming_amp_floor: float = 1.0e-30,
    init_species_index: int = 1,
    density_species_index: int = 1,
    diagnostic_norm: str = "none",
    reference_aligned: bool | None = True,
    show_progress: bool = False,
) -> LinearScanResult:
    """Run a kinetic-electron ITG/TEM benchmark for a list of ky values.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    controls = _prepare_kinetic_scan_controls(
        _kinetic_scan_control_request_from_locals(locals())
    )
    output = _run_kinetic_scan_batches(
        setup=controls.setup,
        run_options=controls.run_options,
        fit_options=controls.fit_options,
        n_laguerre=Nl,
        n_hermite=Nm,
    )
    return _kinetic_scan_result(output)

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
