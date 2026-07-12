"""Unified runtime-configured linear driver (case-agnostic core path)."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence
from pathlib import Path
import sys

import numpy as np

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
from spectraxgk.operators.linear.cache_builder import build_linear_cache
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
    RuntimeParameterScanResult,
    build_runtime_nonlinear_result,
)
from spectraxgk.workflows.runtime.orchestration_scan import (
    build_runtime_scan_batch_deps,
    build_runtime_scan_orchestration_deps,
    run_runtime_scan_ky_task as _run_runtime_scan_ky_task_impl,
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
    default_runtime_case_deps as _default_runtime_case_deps,
    run_linear_case as _run_linear_case_impl,
    run_nonlinear_case as _run_nonlinear_case_impl,
)
from spectraxgk.workflows.linear import run_full_linear_runtime
from spectraxgk.workflows.nonlinear import run_full_nonlinear_runtime
from spectraxgk.terms.config import TermConfig
from spectraxgk.geometry.miller_eik import generate_runtime_miller_eik
from spectraxgk.geometry.vmec_eik import generate_runtime_vmec_eik

_RUNTIME_CASE_FIT_KEYS = _WORKFLOW_RUNTIME_CASE_FIT_KEYS

# These symbols are intentionally imported into the runtime facade because the
# dispatch/workflow dependency builders read them from ``sys.modules[__name__]``.
_PATCHABLE_RUNTIME_GLOBALS = (
    apply_diagnostic_normalization,
    apply_geometry_grid_defaults,
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
    independent_map,
    integrate_linear_diagnostics,
    integrate_linear_from_config,
    integrate_nonlinear_explicit_diagnostics_state,
    integrate_nonlinear_from_config,
    linear_terms_to_term_config,
    run_adaptive_runtime_chunk_loop,
    run_full_linear_runtime,
    run_full_nonlinear_runtime,
    select_ky_grid,
    select_ky_index,
    _parallel_requests_combined_ky_scan,
)

_RUNTIME_LINEAR_TIME_FIT_OPTION_KEYS = (
    "method",
    "dt",
    "steps",
    "sample_stride",
    "auto_window",
    "tmin",
    "tmax",
    "window_fraction",
    "min_points",
    "start_fraction",
    "growth_weight",
    "require_positive",
    "min_amp_fraction",
    "mode_method",
    "fit_signal",
)

__all__ = [
    "RuntimeIndependentParallelPlan", "RuntimeLinearResult",
    "RuntimeLinearScanResult", "RuntimeNonlinearResult",
    "RuntimeParameterScanResult",
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

    return _run_runtime_scan_ky_task_impl(task, run_runtime_linear=run_runtime_linear)


build_flux_tube_geometry = runtime_startup.build_flux_tube_geometry
load_netcdf_restart_state = runtime_startup.load_netcdf_restart_state
_centered_glibc_random_pairs = runtime_startup._centered_glibc_random_pairs
_dealiased_initial_mode_pairs = runtime_startup._dealiased_initial_mode_pairs
_periodic_zp_from_grid = runtime_startup._periodic_zp_from_grid


def _runtime_geometry_config_for_builder(cfg: RuntimeConfig) -> Any:
    """Resolve the geometry config that should be passed to the flux-tube builder."""

    return runtime_startup.runtime_geometry_config_for_builder(
        cfg,
        vmec_eik_builder=generate_runtime_vmec_eik,
        miller_eik_builder=generate_runtime_miller_eik,
    )


def build_runtime_geometry(cfg: RuntimeConfig) -> FluxTubeGeometryLike:
    """Resolve runtime geometry while preserving the runtime module patch surface."""

    return build_flux_tube_geometry(_runtime_geometry_config_for_builder(cfg))


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

    shape_kwargs = {
        "nspecies": nspecies,
        "Nl": Nl,
        "Nm": Nm,
        "ny": ny,
        "nx": nx,
        "nz": nz,
    }
    if path.suffix.lower() == ".nc":
        return load_netcdf_restart_state(path, **shape_kwargs)
    return runtime_startup._load_initial_state_from_file(path, **shape_kwargs)


_slice_runtime_diagnostics = slice_runtime_diagnostics
_truncate_runtime_diagnostics = truncate_runtime_diagnostics
_stride_runtime_diagnostics = stride_runtime_diagnostics
_concat_runtime_diagnostics = concat_runtime_diagnostics


def _runtime_facade_module() -> Any:
    """Return the patchable runtime facade module used by dependency builders."""

    return sys.modules[__name__]


def _runtime_linear_dispatch_deps() -> RuntimeLinearDispatchDeps:
    """Build linear runtime dispatch dependencies from patchable module globals."""

    return build_runtime_linear_dispatch_deps(_runtime_facade_module())


def _runtime_linear_time_fit_options(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return shared runtime linear time-integration and fit options."""

    return {name: values[name] for name in _RUNTIME_LINEAR_TIME_FIT_OPTION_KEYS}


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
    initial_state: Any | None = None,
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
        **_runtime_linear_time_fit_options(locals()),
        krylov_cfg=krylov_cfg,
        return_state=return_state,
        initial_state=initial_state,
        show_progress=show_progress,
        status_callback=status_callback,
        deps=_runtime_linear_dispatch_deps(),
    )


def run_runtime_parameter_scan(
    cfg: RuntimeConfig,
    parameter_values: Sequence[float],
    *,
    parameter_name: str,
    update_config: Callable[[RuntimeConfig, float, int], RuntimeConfig],
    ky_target: float = 0.3,
    linear_options: Mapping[str, Any] | None = None,
    point_options: Callable[
        [float, int, RuntimeLinearResult | None], Mapping[str, Any]
    ] | None = None,
    continuation: bool = False,
) -> RuntimeParameterScanResult:
    """Run an ordered scan over one scalar runtime-configuration parameter.

    When continuation is enabled, each result state initializes the next point.
    Case-specific parameter transforms and solver policy remain in callbacks.
    """

    name = str(parameter_name).strip()
    if not name:
        raise ValueError("parameter_name must be nonempty")
    values = np.asarray(parameter_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("parameter_values must be a nonempty one-dimensional array")

    shared = dict(linear_options or {})
    runs: list[RuntimeLinearResult] = []
    previous: RuntimeLinearResult | None = None
    for index, value in enumerate(values):
        point_cfg = update_config(cfg, float(value), index)
        if not isinstance(point_cfg, RuntimeConfig):
            raise TypeError("update_config must return RuntimeConfig")
        options = dict(shared)
        if point_options is not None:
            options.update(point_options(float(value), index, previous))
        if continuation:
            options["return_state"] = True
            if previous is not None:
                if previous.state is None:
                    raise ValueError("continuation requires each point to return state")
                options["initial_state"] = previous.state
        result = run_runtime_linear(point_cfg, ky_target=ky_target, **options)
        runs.append(result)
        previous = result

    return RuntimeParameterScanResult(
        parameter_name=name,
        values=values,
        gamma=np.asarray([result.gamma for result in runs], dtype=float),
        omega=np.asarray([result.omega for result in runs], dtype=float),
        runs=tuple(runs),
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
    coordination lives in ``workflows/runtime/orchestration_scan.py``.
    """

    return _run_runtime_scan_orchestration_impl(
        cfg,
        ky_values,
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        batch_ky=batch_ky,
        **_runtime_linear_time_fit_options(locals()),
        krylov_cfg=krylov_cfg,
        show_progress=show_progress,
        workers=workers,
        parallel_executor=parallel_executor,
        deps=_runtime_scan_orchestration_deps(),
    )


def _runtime_scan_orchestration_deps() -> Any:
    """Build ky-scan orchestration dependencies from patchable facade globals."""

    return build_runtime_scan_orchestration_deps(_runtime_facade_module())


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
        **_runtime_linear_time_fit_options(locals()),
        show_progress=show_progress,
        deps=_runtime_scan_batch_deps(),
    )


def _runtime_scan_batch_deps() -> Any:
    """Build combined-ky scan dependencies from patchable facade globals."""

    return build_runtime_scan_batch_deps(_runtime_facade_module())


def _runtime_nonlinear_dispatch_deps() -> RuntimeNonlinearDispatchDeps:
    """Build nonlinear runtime dispatch dependencies from patchable module globals."""

    return build_runtime_nonlinear_dispatch_deps(_runtime_facade_module())


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
        deps=_default_runtime_case_deps(),
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
        deps=_default_runtime_case_deps(),
    )
