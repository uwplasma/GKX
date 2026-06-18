"""Unified runtime-configured linear driver (case-agnostic core path)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from spectraxgk.terms.reduced.cetg_integrator import (
    integrate_cetg_explicit_diagnostics_state,
)
from spectraxgk.terms.reduced.cetg_model import (
    build_cetg_model_params,
    validate_cetg_runtime_config,
)
from spectraxgk.diagnostics.growth_rates import (
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_auto_with_stats,
)
from spectraxgk.diagnostics.modes import (
    extract_eigenfunction,
    extract_mode_time_series,
    select_ky_index,
)
from spectraxgk.geometry import apply_geometry_grid_defaults, FluxTubeGeometryLike
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.linear import integrate_linear_diagnostics
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from spectraxgk.nonlinear import integrate_nonlinear_explicit_diagnostics_state
from spectraxgk.solvers.linear.krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.diagnostics.normalization import apply_diagnostic_normalization
from spectraxgk.parallel import independent_map
from spectraxgk.diagnostics.quasilinear_transport import compute_quasilinear_from_linear_state
from spectraxgk.workflows.runtime.config import RuntimeConfig
from spectraxgk.workflows.runtime import startup as runtime_startup
from spectraxgk.workflows.runtime.execution import (
    RuntimeLinearDispatchDeps,
    RuntimeNonlinearDispatchDeps,
    build_runtime_linear_dispatch_deps,
    build_runtime_nonlinear_dispatch_deps,
    run_runtime_linear_impl,
    run_runtime_nonlinear_impl,
)
from spectraxgk.workflows.runtime.diagnostic_arrays import (
    concat_runtime_diagnostics,
    slice_runtime_diagnostics,
    stride_runtime_diagnostics,
    truncate_runtime_diagnostics,
)
from spectraxgk.workflows.runtime.diagnostics import (
    finalize_runtime_linear_quasilinear,
    fit_runtime_linear_diagnostics,
)
from spectraxgk.workflows.runtime.chunks import run_adaptive_runtime_chunk_loop
from spectraxgk.workflows.runtime.results import (
    RuntimeLinearResult,
    RuntimeLinearScanResult,
    RuntimeNonlinearResult,
    build_runtime_nonlinear_result,
)
from spectraxgk.workflows.runtime.orchestration import (
    run_runtime_scan_batch as _run_runtime_scan_batch_impl,
    run_runtime_scan_orchestration as _run_runtime_scan_orchestration_impl,
)
from spectraxgk.workflows.runtime.policies import (
    RuntimeIndependentParallelPlan,
    build_runtime_nonlinear_diagnostics_kwargs,
    _infer_runtime_nonlinear_steps,
    _midplane_index,
    _normalize_linear_solver_name,
    _parallel_requests_combined_ky_scan,
    _runtime_external_phi,
    _runtime_independent_parallel_plan,
    _select_nonlinear_mode_indices,
    _zero_kx_index,
)
from spectraxgk.workflows.runtime.startup import (
    _build_gaussian_profile,
    _build_initial_condition,
    _enforce_full_ky_hermitian,
    _expand_ky,
    _default_hermite_hypercollision_exponent,
    _require_full_gk_runtime_model,
    _resolve_runtime_hl_dims,
    _reshape_netcdf_state,
    _runtime_default_krylov_config,
    _runtime_model_key,
    _species_to_linear,
)
from spectraxgk.solvers.time.runners import (
    integrate_linear_from_config,
    integrate_nonlinear_from_config,
)
from spectraxgk.workflows.cases import (
    RUNTIME_CASE_FIT_KEYS as _WORKFLOW_RUNTIME_CASE_FIT_KEYS,
    RuntimeCaseDeps,
    run_linear_case as _run_linear_case_impl,
    run_nonlinear_case as _run_nonlinear_case_impl,
)
from spectraxgk.workflows.linear import run_full_linear_runtime
from spectraxgk.workflows.nonlinear import run_full_nonlinear_runtime
from spectraxgk.workflows.reduced_models import (
    run_cetg_linear_runtime,
    run_cetg_nonlinear_runtime,
)
from spectraxgk.terms.config import TermConfig
from spectraxgk.geometry.miller_eik import generate_runtime_miller_eik
from spectraxgk.geometry.vmec_eik import generate_runtime_vmec_eik

_RUNTIME_CASE_FIT_KEYS = _WORKFLOW_RUNTIME_CASE_FIT_KEYS

__all__ = [
    "RuntimeIndependentParallelPlan", "RuntimeLinearResult",
    "RuntimeLinearScanResult", "RuntimeNonlinearResult",
    "_build_gaussian_profile", "_build_initial_condition",
    "_concat_runtime_diagnostics", "_enforce_full_ky_hermitian", "_expand_ky",
    "_centered_glibc_random_pairs", "_default_hermite_hypercollision_exponent",
    "_dealiased_initial_mode_pairs", "_periodic_zp_from_grid",
    "_infer_runtime_nonlinear_steps", "_load_initial_state_from_file",
    "_midplane_index", "_normalize_linear_solver_name",
    "_require_full_gk_runtime_model", "_resolve_runtime_hl_dims",
    "_reshape_netcdf_state", "_run_runtime_scan_batch",
    "_runtime_default_krylov_config", "_runtime_external_phi",
    "_runtime_independent_parallel_plan", "_runtime_model_key",
    "_select_nonlinear_mode_indices", "_slice_runtime_diagnostics",
    "_species_to_linear", "_stride_runtime_diagnostics",
    "_truncate_runtime_diagnostics", "_zero_kx_index",
    "build_runtime_geometry", "build_runtime_linear_params",
    "build_runtime_linear_terms", "build_runtime_term_config", "run_linear_case",
    "run_nonlinear_case", "run_runtime_linear", "run_runtime_nonlinear",
    "run_runtime_scan",
]


def _run_runtime_scan_ky_task(task: dict[str, Any]) -> RuntimeLinearResult:
    """Run one independent ky point for ordered scan-worker execution."""

    return run_runtime_linear(
        task["cfg"],
        ky_target=float(task["ky"]),
        Nl=int(task["Nl"]),
        Nm=int(task["Nm"]),
        solver=str(task["solver"]),
        method=task["method"],
        dt=task["dt"],
        steps=task["steps"],
        sample_stride=task["sample_stride"],
        auto_window=bool(task["auto_window"]),
        tmin=task["tmin"],
        tmax=task["tmax"],
        window_fraction=float(task["window_fraction"]),
        min_points=int(task["min_points"]),
        start_fraction=float(task["start_fraction"]),
        growth_weight=float(task["growth_weight"]),
        require_positive=bool(task["require_positive"]),
        min_amp_fraction=float(task["min_amp_fraction"]),
        krylov_cfg=task["krylov_cfg"],
        mode_method=str(task["mode_method"]),
        fit_signal=str(task["fit_signal"]),
        show_progress=bool(task["show_progress"]),
    )


build_flux_tube_geometry = runtime_startup.build_flux_tube_geometry
load_netcdf_restart_state = runtime_startup.load_netcdf_restart_state
_centered_glibc_random_pairs = runtime_startup._centered_glibc_random_pairs
_dealiased_initial_mode_pairs = runtime_startup._dealiased_initial_mode_pairs
_periodic_zp_from_grid = runtime_startup._periodic_zp_from_grid

_PATCHABLE_RUNTIME_DISPATCH_GLOBALS = (
    apply_diagnostic_normalization,
    apply_geometry_grid_defaults,
    build_cetg_model_params,
    build_linear_cache,
    build_runtime_nonlinear_diagnostics_kwargs,
    build_runtime_nonlinear_result,
    build_spectral_grid,
    compute_quasilinear_from_linear_state,
    dominant_eigenpair,
    extract_eigenfunction,
    extract_mode_time_series,
    finalize_runtime_linear_quasilinear,
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_auto_with_stats,
    fit_runtime_linear_diagnostics,
    integrate_cetg_explicit_diagnostics_state,
    integrate_linear_diagnostics,
    integrate_linear_from_config,
    integrate_nonlinear_explicit_diagnostics_state,
    integrate_nonlinear_from_config,
    linear_terms_to_term_config,
    run_adaptive_runtime_chunk_loop,
    run_cetg_linear_runtime,
    run_cetg_nonlinear_runtime,
    run_full_linear_runtime,
    run_full_nonlinear_runtime,
    select_ky_grid,
    select_ky_index,
    validate_cetg_runtime_config,
)


def build_runtime_geometry(cfg: RuntimeConfig) -> FluxTubeGeometryLike:
    """Resolve runtime geometry while preserving the runtime module patch surface."""

    model = cfg.geometry.model.strip().lower()
    if model == "vmec":
        eik_path = generate_runtime_vmec_eik(cfg)
        geom_cfg = replace(cfg.geometry, model="vmec-eik", geometry_file=str(eik_path))
        return build_flux_tube_geometry(geom_cfg)
    if model == "miller":
        eik_path = generate_runtime_miller_eik(cfg)
        geom_cfg = replace(
            cfg.geometry, model="imported-eik", geometry_file=str(eik_path)
        )
        return build_flux_tube_geometry(geom_cfg)
    return build_flux_tube_geometry(cfg.geometry)


def build_runtime_linear_params(
    cfg: RuntimeConfig,
    *,
    Nm: int | None = None,
    geom: FluxTubeGeometryLike | None = None,
) -> LinearParams:
    """Build runtime linear parameters using the runtime module geometry surface."""

    if geom is None:
        geom = build_runtime_geometry(cfg)
    return runtime_startup.build_runtime_linear_params(cfg, Nm=Nm, geom=geom)


def build_runtime_linear_terms(cfg: RuntimeConfig) -> LinearTerms:
    """Build runtime linear term toggles."""

    return runtime_startup.build_runtime_linear_terms(cfg)


def build_runtime_term_config(cfg: RuntimeConfig) -> TermConfig:
    """Build runtime nonlinear-ready term config."""

    return runtime_startup.build_runtime_term_config(cfg)


def _load_initial_state_from_file(
    path: Path,
    *,
    nspecies: int,
    Nl: int,
    Nm: int,
    ny: int,
    nx: int,
    nz: int,
) -> np.ndarray:
    """Load an initial state while preserving the runtime module patch surface."""

    if path.suffix.lower() == ".nc":
        return load_netcdf_restart_state(
            path,
            nspecies=nspecies,
            Nl=Nl,
            Nm=Nm,
            ny=ny,
            nx=nx,
            nz=nz,
        )
    return runtime_startup._load_initial_state_from_file(
        path,
        nspecies=nspecies,
        Nl=Nl,
        Nm=Nm,
        ny=ny,
        nx=nx,
        nz=nz,
    )


_slice_runtime_diagnostics = slice_runtime_diagnostics
_truncate_runtime_diagnostics = truncate_runtime_diagnostics
_stride_runtime_diagnostics = stride_runtime_diagnostics
_concat_runtime_diagnostics = concat_runtime_diagnostics


def _runtime_linear_dispatch_deps() -> RuntimeLinearDispatchDeps:
    """Build linear runtime dispatch dependencies from patchable module globals."""

    return build_runtime_linear_dispatch_deps(sys.modules[__name__])


def run_runtime_linear(
    cfg: RuntimeConfig,
    *,
    ky_target: float = 0.3,
    Nl: int | None = None,
    Nm: int | None = None,
    solver: str = "auto",
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
    auto_window: bool = True,
    tmin: float | None = None,
    tmax: float | None = None,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 0.2,
    require_positive: bool = True,
    min_amp_fraction: float = 0.0,
    krylov_cfg: KrylovConfig | None = None,
    mode_method: str = "project",
    fit_signal: str = "auto",
    return_state: bool = False,
    show_progress: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> RuntimeLinearResult:
    """Run one linear point from a case-agnostic runtime config."""

    return run_runtime_linear_impl(
        cfg,
        ky_target=ky_target,
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        method=method,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
        auto_window=auto_window,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        krylov_cfg=krylov_cfg,
        mode_method=mode_method,
        fit_signal=fit_signal,
        return_state=return_state,
        show_progress=show_progress,
        status_callback=status_callback,
        deps=_runtime_linear_dispatch_deps(),
    )


def run_runtime_scan(
    cfg: RuntimeConfig,
    ky_values: Sequence[float],
    *,
    Nl: int | None = None,
    Nm: int | None = None,
    solver: str = "auto",
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
    batch_ky: bool = False,
    auto_window: bool = True,
    tmin: float | None = None,
    tmax: float | None = None,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 0.2,
    require_positive: bool = True,
    min_amp_fraction: float = 0.0,
    krylov_cfg: KrylovConfig | None = None,
    mode_method: str = "project",
    fit_signal: str = "auto",
    show_progress: bool = False,
    workers: int = 1,
    parallel_executor: str = "thread",
) -> RuntimeLinearScanResult:
    """Run a ky scan using the unified runtime config path.

    The public facade keeps runtime monkeypatch seams intact while scan
    coordination lives in ``workflows/runtime/orchestration.py``.
    """

    return _run_runtime_scan_orchestration_impl(
        cfg,
        ky_values,
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        method=method,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
        batch_ky=batch_ky,
        auto_window=auto_window,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        krylov_cfg=krylov_cfg,
        mode_method=mode_method,
        fit_signal=fit_signal,
        show_progress=show_progress,
        workers=workers,
        parallel_executor=parallel_executor,
        deps=_runtime_scan_orchestration_deps(),
    )


def _runtime_scan_orchestration_deps() -> SimpleNamespace:
    """Build ky-scan orchestration dependencies from patchable facade globals."""

    return SimpleNamespace(
        resolve_runtime_hl_dims=_resolve_runtime_hl_dims,
        normalize_linear_solver_name=_normalize_linear_solver_name,
        parallel_requests_combined_ky_scan=_parallel_requests_combined_ky_scan,
        run_runtime_scan_batch=_run_runtime_scan_batch,
        runtime_independent_parallel_plan=_runtime_independent_parallel_plan,
        independent_map=independent_map,
        run_runtime_scan_ky_task=_run_runtime_scan_ky_task,
    )


def _run_runtime_scan_batch(
    cfg: RuntimeConfig,
    ky_arr: np.ndarray,
    *,
    Nl: int,
    Nm: int,
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
    mode_method: str,
    fit_signal: str,
    show_progress: bool,
) -> RuntimeLinearScanResult:
    """Facade wrapper for the extracted combined-ky scan batch helper."""

    return _run_runtime_scan_batch_impl(
        cfg,
        ky_arr,
        Nl=Nl,
        Nm=Nm,
        method=method,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
        auto_window=auto_window,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        mode_method=mode_method,
        fit_signal=fit_signal,
        show_progress=show_progress,
        deps=_runtime_scan_batch_deps(),
    )


def _runtime_scan_batch_deps() -> SimpleNamespace:
    """Build combined-ky scan dependencies from patchable facade globals."""

    return SimpleNamespace(
        build_runtime_geometry=build_runtime_geometry,
        build_runtime_linear_params=build_runtime_linear_params,
        build_runtime_linear_terms=build_runtime_linear_terms,
        build_initial_condition=_build_initial_condition,
        apply_geometry_grid_defaults=apply_geometry_grid_defaults,
        build_spectral_grid=build_spectral_grid,
        select_ky_index=select_ky_index,
        midplane_index=_midplane_index,
        integrate_linear_diagnostics=integrate_linear_diagnostics,
        extract_mode_time_series=extract_mode_time_series,
        fit_growth_rate_auto_with_stats=fit_growth_rate_auto_with_stats,
        fit_growth_rate_auto=fit_growth_rate_auto,
        fit_growth_rate=fit_growth_rate,
        apply_diagnostic_normalization=apply_diagnostic_normalization,
    )


def _runtime_nonlinear_dispatch_deps() -> RuntimeNonlinearDispatchDeps:
    """Build nonlinear runtime dispatch dependencies from patchable module globals."""

    return build_runtime_nonlinear_dispatch_deps(sys.modules[__name__])


def run_runtime_nonlinear(
    cfg: RuntimeConfig,
    *,
    ky_target: float = 0.3,
    kx_target: float | None = None,
    Nl: int | None = None,
    Nm: int | None = None,
    dt: float | None = None,
    steps: int | None = None,
    method: str | None = None,
    sample_stride: int | None = None,
    diagnostics_stride: int | None = None,
    laguerre_mode: str | None = None,
    diagnostics: bool | None = None,
    resolved_diagnostics: bool = True,
    return_state: bool = False,
    show_progress: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> RuntimeNonlinearResult:
    """Run a nonlinear point using the unified runtime config path."""

    return run_runtime_nonlinear_impl(
        cfg,
        ky_target=ky_target,
        kx_target=kx_target,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        laguerre_mode=laguerre_mode,
        diagnostics=diagnostics,
        resolved_diagnostics=resolved_diagnostics,
        return_state=return_state,
        show_progress=show_progress,
        status_callback=status_callback,
        deps=_runtime_nonlinear_dispatch_deps(),
    )


def _runtime_case_deps() -> RuntimeCaseDeps:
    """Build case-workflow dependencies from this module's patchable globals."""

    from spectraxgk.workflows.runtime.toml import load_runtime_from_toml
    from spectraxgk.workflows.runtime.artifacts import (
        run_runtime_nonlinear_with_artifacts,
        write_runtime_linear_artifacts,
    )

    return RuntimeCaseDeps(
        load_runtime_from_toml=load_runtime_from_toml,
        run_runtime_linear=run_runtime_linear,
        run_runtime_nonlinear=run_runtime_nonlinear,
        write_runtime_linear_artifacts=write_runtime_linear_artifacts,
        run_runtime_nonlinear_with_artifacts=run_runtime_nonlinear_with_artifacts,
    )


def run_linear_case(
    config_path: str | Path,
    *,
    ky: float | None = None,
    Nl: int | None = None,
    Nm: int | None = None,
    solver: str | None = None,
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
    show_progress: bool = True,
) -> int:
    """Run a linear case from a runtime TOML with optional overrides."""

    return _run_linear_case_impl(
        config_path,
        ky=ky,
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        method=method,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
        show_progress=show_progress,
        deps=_runtime_case_deps(),
    )


def run_nonlinear_case(
    config_path: str | Path,
    *,
    ky: float | None = None,
    Nl: int | None = None,
    Nm: int | None = None,
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
    diagnostics_stride: int | None = None,
    show_progress: bool = True,
) -> int:
    """Run a nonlinear case from a runtime TOML with optional overrides."""

    return _run_nonlinear_case_impl(
        config_path,
        ky=ky,
        Nl=Nl,
        Nm=Nm,
        method=method,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        show_progress=show_progress,
        deps=_runtime_case_deps(),
    )
