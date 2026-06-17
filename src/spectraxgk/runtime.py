"""Unified runtime-configured linear driver (case-agnostic core path)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from spectraxgk.cetg import (
    build_cetg_model_params,
    integrate_cetg_explicit_diagnostics_state,
    validate_cetg_runtime_config,
)
from spectraxgk.analysis import (
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_auto_with_stats,
    select_ky_index,
)
from spectraxgk.geometry import apply_geometry_grid_defaults, FluxTubeGeometryLike
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import integrate_linear_diagnostics
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from spectraxgk.nonlinear import integrate_nonlinear_explicit_diagnostics_state
from spectraxgk.solvers.linear.krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.normalization import apply_diagnostic_normalization
from spectraxgk.parallel import independent_map
from spectraxgk.quasilinear import compute_quasilinear_from_linear_state
from spectraxgk.runtime_config import RuntimeConfig
from spectraxgk import runtime_startup
from spectraxgk.runtime_diagnostics import (
    concat_runtime_diagnostics,
    finalize_runtime_linear_quasilinear,
    RuntimeQuasilinearFinalizationDeps,
    slice_runtime_diagnostics,
    stride_runtime_diagnostics,
    truncate_runtime_diagnostics,
    fit_runtime_linear_diagnostics,
)
from spectraxgk.runtime_chunks import run_adaptive_runtime_chunk_loop
from spectraxgk.runtime_results import (
    RuntimeLinearResult,
    RuntimeLinearScanResult,
    RuntimeNonlinearResult,
    build_runtime_nonlinear_result,
)
from spectraxgk.runtime_orchestration import (
    run_runtime_scan_batch as _run_runtime_scan_batch_impl,
    run_runtime_scan_orchestration as _run_runtime_scan_orchestration_impl,
)
from spectraxgk.runtime_policies import (
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
from spectraxgk.runtime_startup import (
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
from spectraxgk.runners import (
    integrate_linear_from_config,
    integrate_nonlinear_from_config,
)
from spectraxgk.workflows.cases import (
    RUNTIME_CASE_FIT_KEYS as _WORKFLOW_RUNTIME_CASE_FIT_KEYS,
    RuntimeCaseDeps,
    run_linear_case as _run_linear_case_impl,
    run_nonlinear_case as _run_nonlinear_case_impl,
)
from spectraxgk.workflows.linear import FullLinearRuntimeDeps, run_full_linear_runtime
from spectraxgk.workflows.reduced_models import (
    CETGLinearRuntimeDeps,
    CETGNonlinearRuntimeDeps,
    run_cetg_linear_runtime,
    run_cetg_nonlinear_runtime,
)
from spectraxgk.terms.config import TermConfig
from spectraxgk.miller_eik import generate_runtime_miller_eik
from spectraxgk.vmec_eik import generate_runtime_vmec_eik

_RUNTIME_CASE_FIT_KEYS = _WORKFLOW_RUNTIME_CASE_FIT_KEYS

__all__ = [
    "RuntimeIndependentParallelPlan",
    "RuntimeLinearResult",
    "RuntimeLinearScanResult",
    "RuntimeNonlinearResult",
    "_build_gaussian_profile",
    "_build_initial_condition",
    "_concat_runtime_diagnostics",
    "_enforce_full_ky_hermitian",
    "_expand_ky",
    "_centered_glibc_random_pairs",
    "_default_hermite_hypercollision_exponent",
    "_dealiased_initial_mode_pairs",
    "_periodic_zp_from_grid",
    "_infer_runtime_nonlinear_steps",
    "_load_initial_state_from_file",
    "_midplane_index",
    "_normalize_linear_solver_name",
    "_require_full_gk_runtime_model",
    "_resolve_runtime_hl_dims",
    "_reshape_netcdf_state",
    "_run_runtime_scan_batch",
    "_runtime_default_krylov_config",
    "_runtime_external_phi",
    "_runtime_independent_parallel_plan",
    "_runtime_model_key",
    "_select_nonlinear_mode_indices",
    "_slice_runtime_diagnostics",
    "_species_to_linear",
    "_stride_runtime_diagnostics",
    "_truncate_runtime_diagnostics",
    "_zero_kx_index",
    "build_runtime_geometry",
    "build_runtime_linear_params",
    "build_runtime_linear_terms",
    "build_runtime_term_config",
    "run_linear_case",
    "run_nonlinear_case",
    "run_runtime_linear",
    "run_runtime_nonlinear",
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

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    ql_enabled = bool(getattr(cfg.quasilinear, "enabled", False))

    Nl_use, Nm_use = _resolve_runtime_hl_dims(cfg, Nl=Nl, Nm=Nm)
    _status("building runtime geometry")
    if _runtime_model_key(cfg) == "cetg":
        if ql_enabled:
            raise NotImplementedError(
                "quasilinear diagnostics are not yet validated for reduced_model='cetg'"
            )
        return run_cetg_linear_runtime(
            cfg,
            deps=CETGLinearRuntimeDeps(
                build_runtime_geometry=build_runtime_geometry,
                validate_cetg_runtime_config=validate_cetg_runtime_config,
                build_initial_condition=_build_initial_condition,
                build_runtime_term_config=build_runtime_term_config,
                build_cetg_model_params=build_cetg_model_params,
                integrate_cetg_explicit_diagnostics_state=integrate_cetg_explicit_diagnostics_state,
                fit_growth_rate_auto=fit_growth_rate_auto,
                fit_growth_rate=fit_growth_rate,
            ),
            ky_target=ky_target,
            Nl=Nl_use,
            Nm=Nm_use,
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
            return_state=return_state,
            status_callback=status_callback,
        )

    return run_full_linear_runtime(
        cfg,
        deps=FullLinearRuntimeDeps(
            build_runtime_geometry=build_runtime_geometry,
            apply_geometry_grid_defaults=apply_geometry_grid_defaults,
            build_spectral_grid=build_spectral_grid,
            build_runtime_linear_params=build_runtime_linear_params,
            build_runtime_linear_terms=build_runtime_linear_terms,
            select_ky_index=select_ky_index,
            select_ky_grid=select_ky_grid,
            midplane_index=_midplane_index,
            build_initial_condition=_build_initial_condition,
            normalize_linear_solver_name=_normalize_linear_solver_name,
            runtime_default_krylov_config=_runtime_default_krylov_config,
            build_linear_cache=build_linear_cache,
            dominant_eigenpair=dominant_eigenpair,
            apply_diagnostic_normalization=apply_diagnostic_normalization,
            integrate_linear_from_config=integrate_linear_from_config,
            integrate_linear_diagnostics=integrate_linear_diagnostics,
            fit_runtime_linear_diagnostics=fit_runtime_linear_diagnostics,
            finalize_runtime_linear_quasilinear=finalize_runtime_linear_quasilinear,
            quasilinear_finalization_deps=RuntimeQuasilinearFinalizationDeps(
                build_linear_cache=build_linear_cache,
                compute_quasilinear_from_linear_state=compute_quasilinear_from_linear_state,
                linear_terms_to_term_config=linear_terms_to_term_config,
            ),
            extract_mode_time_series=extract_mode_time_series,
            fit_growth_rate_auto_with_stats=fit_growth_rate_auto_with_stats,
            fit_growth_rate_auto=fit_growth_rate_auto,
            fit_growth_rate=fit_growth_rate,
            extract_eigenfunction=extract_eigenfunction,
        ),
        ky_target=ky_target,
        Nl=Nl_use,
        Nm=Nm_use,
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
    coordination lives in ``runtime_orchestration.py``.
    """

    deps = SimpleNamespace(
        resolve_runtime_hl_dims=_resolve_runtime_hl_dims,
        normalize_linear_solver_name=_normalize_linear_solver_name,
        parallel_requests_combined_ky_scan=_parallel_requests_combined_ky_scan,
        run_runtime_scan_batch=_run_runtime_scan_batch,
        runtime_independent_parallel_plan=_runtime_independent_parallel_plan,
        independent_map=independent_map,
        run_runtime_scan_ky_task=_run_runtime_scan_ky_task,
    )
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
        deps=deps,
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
    """Compatibility wrapper for the extracted combined-ky scan batch helper."""

    deps = SimpleNamespace(
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
        deps=deps,
    )


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

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    Nl_use, Nm_use = _resolve_runtime_hl_dims(cfg, Nl=Nl, Nm=Nm)
    _status("building runtime geometry")
    if _runtime_model_key(cfg) == "cetg":
        return run_cetg_nonlinear_runtime(
            cfg,
            deps=CETGNonlinearRuntimeDeps(
                build_runtime_geometry=build_runtime_geometry,
                validate_cetg_runtime_config=validate_cetg_runtime_config,
                select_nonlinear_mode_indices=_select_nonlinear_mode_indices,
                build_initial_condition=_build_initial_condition,
                build_cetg_model_params=build_cetg_model_params,
                build_runtime_term_config=build_runtime_term_config,
                integrate_cetg_explicit_diagnostics_state=integrate_cetg_explicit_diagnostics_state,
                run_adaptive_runtime_chunk_loop=run_adaptive_runtime_chunk_loop,
                build_runtime_nonlinear_result=build_runtime_nonlinear_result,
            ),
            ky_target=ky_target,
            kx_target=kx_target,
            Nl=Nl_use,
            Nm=Nm_use,
            dt=dt,
            steps=steps,
            method=method,
            sample_stride=sample_stride,
            diagnostics_stride=diagnostics_stride,
            diagnostics=diagnostics,
            return_state=return_state,
            show_progress=show_progress,
            status_callback=status_callback,
        )

    geom = build_runtime_geometry(cfg)
    _status("building spectral grid")
    grid_cfg = apply_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    _status("building runtime nonlinear parameters")
    params = build_runtime_linear_params(cfg, Nm=Nm_use, geom=geom)
    term_cfg = build_runtime_term_config(cfg)

    ky_index, kx_index = _select_nonlinear_mode_indices(
        grid,
        ky_target=ky_target,
        kx_target=kx_target,
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
    )
    _status(
        f"selected nonlinear mode ky={float(np.asarray(grid.ky[ky_index])):.6g} kx={float(np.asarray(grid.kx[kx_index])):.6g}"
    )
    _status("building initial condition")
    G0 = _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=Nl_use,
        Nm=Nm_use,
        nspecies=len(_species_to_linear(cfg.species)),
    )

    dt_val = float(cfg.time.dt if dt is None else dt)
    if dt_val <= 0.0:
        raise ValueError("dt must be > 0")
    adaptive_chunked = steps is None and not bool(cfg.time.fixed_dt)
    steps_val = _infer_runtime_nonlinear_steps(cfg, dt=dt_val, steps=steps)

    fixed_mode_on = bool(cfg.expert.fixed_mode)
    fixed_ky_index = cfg.expert.iky_fixed
    fixed_kx_index = cfg.expert.ikx_fixed
    external_phi = _runtime_external_phi(cfg)
    source_on = external_phi is not None
    fixed_ky_index_use: int | None = None
    fixed_kx_index_use: int | None = None
    if fixed_mode_on:
        if fixed_ky_index is None or fixed_kx_index is None:
            raise ValueError(
                "expert.iky_fixed and expert.ikx_fixed must be set when expert.fixed_mode=true"
            )
        fixed_ky_index_use = int(fixed_ky_index)
        fixed_kx_index_use = int(fixed_kx_index)

    diagnostics_on = cfg.time.diagnostics if diagnostics is None else bool(diagnostics)
    _status(
        f"nonlinear diagnostics={'on' if diagnostics_on else 'off'} fixed_mode={'on' if fixed_mode_on else 'off'} source={cfg.expert.source}"
    )
    if diagnostics_on or fixed_mode_on or return_state or adaptive_chunked or source_on:
        sample_stride_use = (
            cfg.time.sample_stride if sample_stride is None else int(sample_stride)
        )
        diag_stride = (
            cfg.time.diagnostics_stride
            if diagnostics_stride is None
            else int(diagnostics_stride)
        )
        laguerre_mode_use = (
            cfg.time.laguerre_nonlinear_mode
            if laguerre_mode is None
            else str(laguerre_mode)
        )
        _status(
            f"sample_stride={int(sample_stride_use)} diagnostics_stride={int(diag_stride)} laguerre_mode={laguerre_mode_use}"
        )
        if adaptive_chunked:
            chunk_steps = min(steps_val, 1024)
            G_chunk = G0

            def _run_nonlinear_chunk(chunk_show_progress: bool):
                nonlocal G_chunk
                kwargs = build_runtime_nonlinear_diagnostics_kwargs(
                    cfg,
                    dt=dt_val,
                    steps=chunk_steps,
                    method=method,
                    term_config=term_cfg,
                    sample_stride=1,
                    diagnostics_stride=1,
                    laguerre_mode=laguerre_mode_use,
                    ky_index=int(ky_index),
                    kx_index=int(kx_index),
                    fixed_dt=False,
                    fixed_mode_ky_index=fixed_ky_index_use,
                    fixed_mode_kx_index=fixed_kx_index_use,
                    external_phi=external_phi,
                    resolved_diagnostics=resolved_diagnostics,
                    show_progress=chunk_show_progress,
                )
                t_chunk, diag_chunk, G_next, fields_next = (
                    integrate_nonlinear_explicit_diagnostics_state(
                        G_chunk,
                        grid,
                        geom,
                        params,
                        **kwargs,
                    )
                )
                G_chunk = G_next
                return t_chunk, diag_chunk, G_next, fields_next

            chunk_result = run_adaptive_runtime_chunk_loop(
                integrate_chunk=_run_nonlinear_chunk,
                t_max=float(cfg.time.t_max),
                chunk_steps=chunk_steps,
                label="nonlinear",
                show_progress=show_progress,
                status_callback=_status,
                diagnostics_stride=max(int(sample_stride_use), int(diag_stride), 1),
            )
            diag = chunk_result.diagnostics
            t = jnp.asarray(diag.t)
            G_final = chunk_result.state
            fields_final = chunk_result.fields
        else:
            _status(
                f"running nonlinear diagnostics integrator over {steps_val} steps with dt={dt_val:.6g}"
            )
            diagnostics_call_kwargs = build_runtime_nonlinear_diagnostics_kwargs(
                cfg,
                dt=dt_val,
                steps=steps_val,
                method=method,
                term_config=term_cfg,
                sample_stride=int(sample_stride_use),
                diagnostics_stride=int(diag_stride),
                laguerre_mode=laguerre_mode_use,
                ky_index=int(ky_index),
                kx_index=int(kx_index),
                fixed_dt=bool(cfg.time.fixed_dt),
                fixed_mode_ky_index=fixed_ky_index_use,
                fixed_mode_kx_index=fixed_kx_index_use,
                external_phi=external_phi,
                resolved_diagnostics=resolved_diagnostics,
                show_progress=show_progress,
            )
            t, diag, G_final, fields_final = (
                integrate_nonlinear_explicit_diagnostics_state(
                    G0,
                    grid,
                    geom,
                    params,
                    **diagnostics_call_kwargs,
                )
            )
        if diagnostics_on:
            _status(
                f"completed nonlinear run with {int(np.asarray(t).size)} saved samples"
            )
            state_out = np.asarray(G_final) if return_state else None
            return build_runtime_nonlinear_result(
                t=np.asarray(t),
                diagnostics=diag,
                fields=fields_final,
                state=state_out,
                ky_selected=float(np.asarray(grid.ky[ky_index])),
                kx_selected=float(np.asarray(grid.kx[kx_index])),
                summarize_fields=False,
            )
        if fields_final is None:
            raise RuntimeError(
                "adaptive nonlinear runtime did not produce final fields"
            )
        _status("diagnostics disabled; returning final nonlinear field summary")
        return build_runtime_nonlinear_result(
            t=np.asarray([]),
            diagnostics=None,
            fields=fields_final,
            state=np.asarray(G_final) if return_state else None,
            ky_selected=float(np.asarray(grid.ky[ky_index])),
            kx_selected=float(np.asarray(grid.kx[kx_index])),
            summarize_fields=True,
        )

    # Diagnostics disabled: use the config-driven integrator for final state.
    _status(
        f"diagnostics disabled; running final-state nonlinear integrator over {steps_val} steps with dt={dt_val:.6g}"
    )
    t_cfg = replace(cfg.time, dt=dt_val, t_max=dt_val * steps_val)
    if show_progress:
        G_final, fields = integrate_nonlinear_from_config(
            G0,
            grid,
            geom,
            params,
            t_cfg,
            terms=term_cfg,
            show_progress=True,
        )
    else:
        G_final, fields = integrate_nonlinear_from_config(
            G0,
            grid,
            geom,
            params,
            t_cfg,
            terms=term_cfg,
        )
    _status("completed nonlinear final-state integration")
    return build_runtime_nonlinear_result(
        t=np.asarray([]),
        diagnostics=None,
        fields=fields,
        state=np.asarray(G_final) if return_state else None,
        ky_selected=float(np.asarray(grid.ky[ky_index])),
        kx_selected=float(np.asarray(grid.kx[kx_index])),
        summarize_fields=True,
    )


def _runtime_case_deps() -> RuntimeCaseDeps:
    """Build case-workflow dependencies from this module's patchable globals."""

    from spectraxgk.io import load_runtime_from_toml
    from spectraxgk.runtime_artifacts import (
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
