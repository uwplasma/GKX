"""Benchmark utilities for documented SPECTRAX-GK comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from jax.typing import ArrayLike

from spectraxgk.artifacts.restart import write_netcdf_restart_state
from spectraxgk.core.grid import SpectralGrid, build_spectral_grid
from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
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
from spectraxgk.geometry import apply_geometry_grid_defaults, build_flux_tube_geometry
from spectraxgk.runtime import run_runtime_linear, run_runtime_nonlinear
from spectraxgk.solvers.linear.krylov import KrylovConfig
from spectraxgk.solvers.time.explicit import ExplicitTimeConfig
from spectraxgk.workflows.runtime.config import RuntimeConfig, RuntimeExpertConfig

from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    KBMBaseCase,
    KineticElectronBaseCase,
    TEMBaseCase,
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
from spectraxgk.validation.benchmarks.kbm_scan import run_kbm_scan

from spectraxgk.validation.benchmarks.kinetic_linear import run_kinetic_linear
from spectraxgk.validation.benchmarks.kinetic_scan import run_kinetic_scan

from spectraxgk.validation.benchmarks.etg_linear import run_etg_linear
from spectraxgk.validation.benchmarks.etg_scan import run_etg_scan

from spectraxgk.validation.benchmarks.tem import (
    run_tem_linear,
    run_tem_scan,
)


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
