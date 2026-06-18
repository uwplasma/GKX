"""Runtime execution dispatch for linear and nonlinear configured runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from spectraxgk.workflows.runtime.config import RuntimeConfig
from spectraxgk.workflows.runtime.results import RuntimeLinearResult, RuntimeNonlinearResult


@dataclass(frozen=True)
class RuntimeLinearDispatchDeps:
    """Patchable dependencies for one configured linear runtime run."""

    resolve_runtime_hl_dims: Callable[..., tuple[int, int]]
    runtime_model_key: Callable[[RuntimeConfig], str]
    run_cetg_linear_runtime: Callable[..., RuntimeLinearResult]
    cetg_deps: Any
    run_full_linear_runtime: Callable[..., RuntimeLinearResult]
    full_deps: Any


def build_runtime_linear_dispatch_deps(scope: Any) -> RuntimeLinearDispatchDeps:
    """Build linear dispatch dependencies from a patchable runtime facade scope."""

    from spectraxgk.workflows.linear import FullLinearRuntimeDeps
    from spectraxgk.workflows.reduced_models import CETGLinearRuntimeDeps
    from spectraxgk.workflows.runtime.diagnostics import (
        RuntimeQuasilinearFinalizationDeps,
    )

    return RuntimeLinearDispatchDeps(
        resolve_runtime_hl_dims=scope._resolve_runtime_hl_dims,
        runtime_model_key=scope._runtime_model_key,
        run_cetg_linear_runtime=scope.run_cetg_linear_runtime,
        cetg_deps=CETGLinearRuntimeDeps(
            build_runtime_geometry=scope.build_runtime_geometry,
            validate_cetg_runtime_config=scope.validate_cetg_runtime_config,
            build_initial_condition=scope._build_initial_condition,
            build_runtime_term_config=scope.build_runtime_term_config,
            build_cetg_model_params=scope.build_cetg_model_params,
            integrate_cetg_explicit_diagnostics_state=scope.integrate_cetg_explicit_diagnostics_state,
            fit_growth_rate_auto=scope.fit_growth_rate_auto,
            fit_growth_rate=scope.fit_growth_rate,
        ),
        run_full_linear_runtime=scope.run_full_linear_runtime,
        full_deps=FullLinearRuntimeDeps(
            build_runtime_geometry=scope.build_runtime_geometry,
            apply_geometry_grid_defaults=scope.apply_geometry_grid_defaults,
            build_spectral_grid=scope.build_spectral_grid,
            build_runtime_linear_params=scope.build_runtime_linear_params,
            build_runtime_linear_terms=scope.build_runtime_linear_terms,
            select_ky_index=scope.select_ky_index,
            select_ky_grid=scope.select_ky_grid,
            midplane_index=scope._midplane_index,
            build_initial_condition=scope._build_initial_condition,
            normalize_linear_solver_name=scope._normalize_linear_solver_name,
            runtime_default_krylov_config=scope._runtime_default_krylov_config,
            build_linear_cache=scope.build_linear_cache,
            dominant_eigenpair=scope.dominant_eigenpair,
            apply_diagnostic_normalization=scope.apply_diagnostic_normalization,
            integrate_linear_from_config=scope.integrate_linear_from_config,
            integrate_linear_diagnostics=scope.integrate_linear_diagnostics,
            fit_runtime_linear_diagnostics=scope.fit_runtime_linear_diagnostics,
            finalize_runtime_linear_quasilinear=scope.finalize_runtime_linear_quasilinear,
            quasilinear_finalization_deps=RuntimeQuasilinearFinalizationDeps(
                build_linear_cache=scope.build_linear_cache,
                compute_quasilinear_from_linear_state=scope.compute_quasilinear_from_linear_state,
                linear_terms_to_term_config=scope.linear_terms_to_term_config,
            ),
            extract_mode_time_series=scope.extract_mode_time_series,
            fit_growth_rate_auto_with_stats=scope.fit_growth_rate_auto_with_stats,
            fit_growth_rate_auto=scope.fit_growth_rate_auto,
            fit_growth_rate=scope.fit_growth_rate,
            extract_eigenfunction=scope.extract_eigenfunction,
        ),
    )


def run_runtime_linear_impl(
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
    krylov_cfg: Any = None,
    mode_method: str = "project",
    fit_signal: str = "auto",
    return_state: bool = False,
    show_progress: bool = False,
    status_callback: Callable[[str], None] | None = None,
    deps: RuntimeLinearDispatchDeps,
) -> RuntimeLinearResult:
    """Run one linear point from a case-agnostic runtime config."""

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    ql_enabled = bool(getattr(cfg.quasilinear, "enabled", False))

    Nl_use, Nm_use = deps.resolve_runtime_hl_dims(cfg, Nl=Nl, Nm=Nm)
    _status("building runtime geometry")
    if deps.runtime_model_key(cfg) == "cetg":
        if ql_enabled:
            raise NotImplementedError(
                "quasilinear diagnostics are not yet validated for reduced_model='cetg'"
            )
        return deps.run_cetg_linear_runtime(
            cfg,
            deps=deps.cetg_deps,
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

    return deps.run_full_linear_runtime(
        cfg,
        deps=deps.full_deps,
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


@dataclass(frozen=True)
class RuntimeNonlinearDispatchDeps:
    """Patchable dependencies for one configured nonlinear runtime run."""

    resolve_runtime_hl_dims: Callable[..., tuple[int, int]]
    runtime_model_key: Callable[[RuntimeConfig], str]
    run_cetg_nonlinear_runtime: Callable[..., RuntimeNonlinearResult]
    cetg_deps: Any
    run_full_nonlinear_runtime: Callable[..., RuntimeNonlinearResult]
    full_deps: Any


def build_runtime_nonlinear_dispatch_deps(scope: Any) -> RuntimeNonlinearDispatchDeps:
    """Build nonlinear dispatch dependencies from a patchable runtime facade scope."""

    from spectraxgk.workflows.nonlinear import FullNonlinearRuntimeDeps
    from spectraxgk.workflows.reduced_models import CETGNonlinearRuntimeDeps

    return RuntimeNonlinearDispatchDeps(
        resolve_runtime_hl_dims=scope._resolve_runtime_hl_dims,
        runtime_model_key=scope._runtime_model_key,
        run_cetg_nonlinear_runtime=scope.run_cetg_nonlinear_runtime,
        cetg_deps=CETGNonlinearRuntimeDeps(
            build_runtime_geometry=scope.build_runtime_geometry,
            validate_cetg_runtime_config=scope.validate_cetg_runtime_config,
            select_nonlinear_mode_indices=scope._select_nonlinear_mode_indices,
            build_initial_condition=scope._build_initial_condition,
            build_cetg_model_params=scope.build_cetg_model_params,
            build_runtime_term_config=scope.build_runtime_term_config,
            integrate_cetg_explicit_diagnostics_state=scope.integrate_cetg_explicit_diagnostics_state,
            run_adaptive_runtime_chunk_loop=scope.run_adaptive_runtime_chunk_loop,
            build_runtime_nonlinear_result=scope.build_runtime_nonlinear_result,
        ),
        run_full_nonlinear_runtime=scope.run_full_nonlinear_runtime,
        full_deps=FullNonlinearRuntimeDeps(
            build_runtime_geometry=scope.build_runtime_geometry,
            apply_geometry_grid_defaults=scope.apply_geometry_grid_defaults,
            build_spectral_grid=scope.build_spectral_grid,
            build_runtime_linear_params=scope.build_runtime_linear_params,
            build_runtime_term_config=scope.build_runtime_term_config,
            select_nonlinear_mode_indices=scope._select_nonlinear_mode_indices,
            build_initial_condition=scope._build_initial_condition,
            species_to_linear=scope._species_to_linear,
            infer_runtime_nonlinear_steps=scope._infer_runtime_nonlinear_steps,
            runtime_external_phi=scope._runtime_external_phi,
            build_runtime_nonlinear_diagnostics_kwargs=scope.build_runtime_nonlinear_diagnostics_kwargs,
            integrate_nonlinear_explicit_diagnostics_state=scope.integrate_nonlinear_explicit_diagnostics_state,
            run_adaptive_runtime_chunk_loop=scope.run_adaptive_runtime_chunk_loop,
            build_runtime_nonlinear_result=scope.build_runtime_nonlinear_result,
            integrate_nonlinear_from_config=scope.integrate_nonlinear_from_config,
        ),
    )


def run_runtime_nonlinear_impl(
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
    deps: RuntimeNonlinearDispatchDeps,
) -> RuntimeNonlinearResult:
    """Run one nonlinear point from a case-agnostic runtime config."""

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    Nl_use, Nm_use = deps.resolve_runtime_hl_dims(cfg, Nl=Nl, Nm=Nm)
    _status("building runtime geometry")
    if deps.runtime_model_key(cfg) == "cetg":
        return deps.run_cetg_nonlinear_runtime(
            cfg,
            deps=deps.cetg_deps,
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

    return deps.run_full_nonlinear_runtime(
        cfg,
        deps=deps.full_deps,
        ky_target=ky_target,
        kx_target=kx_target,
        Nl=Nl_use,
        Nm=Nm_use,
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
    )
