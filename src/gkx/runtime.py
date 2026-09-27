"""Unified runtime-configured linear driver (case-agnostic core path)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast
import sys
import numpy as np
from dataclasses import dataclass
from dataclasses import asdict

from gkx.diagnostics.growth_rates import (
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_auto_with_stats,
    fit_growth_rate_with_stats,
)
from gkx.diagnostics.modes import (
    extract_eigenfunction,
    extract_mode_time_series,
    select_ky_index,
)
from gkx.geometry import apply_geometry_grid_defaults, FluxTubeGeometryLike
from gkx.core_grid import (
    build_spectral_grid,
    select_ky_grid,
    SpectralGrid,
)
from gkx.core_ky_layout import KyLayout
from gkx.solvers_linear_integrators import integrate_linear_diagnostics
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from gkx.solvers_nonlinear_diagnostic_integration import (
    integrate_nonlinear_explicit_diagnostics_state,
    prepare_nonlinear_explicit_diagnostics,
)
from gkx.solvers_linear_krylov import KrylovConfig, dominant_eigenpair
from gkx.diagnostics.normalization import apply_diagnostic_normalization
from gkx.parallel import independent_map
from gkx.diagnostics.quasilinear_transport import compute_quasilinear_from_linear_state
from gkx.config import (
    Case,
    RuntimeConfig,
    resolve_cfl_fac,
)
from gkx.workflows.runtime import startup as runtime_startup
from gkx.workflows.runtime.diagnostics import (
    _RuntimeLinearFitOptions,
    finalize_runtime_linear_quasilinear,
    fit_runtime_linear_diagnostics,
)
from gkx.workflows.runtime.results import (
    RuntimeLinearResult,
    RuntimeNonlinearResult,
    RuntimeLinearScanResult,
    build_runtime_nonlinear_result,
)
from gkx.workflows.runtime.diagnostic_arrays import (
    concat_runtime_diagnostics,
    slice_runtime_diagnostics,
    stride_runtime_diagnostics,
)
from gkx.workflows.runtime.chunks import run_adaptive_runtime_chunk_loop
from gkx.workflows.runtime.orchestration_scan import (
    build_runtime_scan_batch_deps,
    build_runtime_scan_orchestration_deps,
    run_runtime_scan_ky_task as _run_runtime_scan_ky_task_impl,
    run_runtime_scan_batch as _run_runtime_scan_batch_impl,
    run_runtime_scan_orchestration as _run_runtime_scan_orchestration_impl,
)
from gkx.workflows.runtime.startup import (
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
from gkx.solvers_time_runners import (
    integrate_linear_from_config,
    integrate_nonlinear_from_config,
)
from gkx.workflows.runtime.commands import (
    RUNTIME_CASE_FIT_KEYS as _WORKFLOW_RUNTIME_CASE_FIT_KEYS,
)
from gkx.workflows.linear import run_full_linear_runtime
from gkx.workflows.nonlinear import run_full_nonlinear_runtime
from gkx.terms.config import TermConfig
from gkx.geometry.miller_eik import generate_runtime_miller_eik
from gkx.geometry.vmec_eik import generate_runtime_vmec_eik


@dataclass(frozen=True)
class RuntimeIndependentParallelPlan:
    """Resolved independent-worker policy for runtime scan workloads."""

    requested_workers: int
    effective_workers: int
    executor: str
    strategy: str
    axis: str
    source: str
    problem_size: int

    @property
    def enabled(self) -> bool:
        """Whether the resolved plan uses more than one independent worker."""

        return self.effective_workers > 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly policy payload for runtime artifacts."""

        payload = asdict(self)
        payload["enabled"] = self.enabled
        return payload


def build_runtime_nonlinear_diagnostics_kwargs(
    cfg: RuntimeConfig,
    *,
    dt: float,
    steps: int,
    method: str | None,
    term_config: Any,
    sample_stride: int,
    diagnostics_stride: int,
    laguerre_mode: str,
    ky_index: int,
    kx_index: int,
    fixed_dt: bool,
    fixed_mode_ky_index: int | None,
    fixed_mode_kx_index: int | None,
    external_phi: float | None,
    resolved_diagnostics: bool,
    show_progress: bool,
) -> dict[str, Any]:
    """Build keyword arguments for nonlinear diagnostic integration.

    Runtime drivers use the same physics kwargs for fixed-window and adaptive
    chunked diagnostics. Keeping them in one policy helper prevents drift
    between the two branches while leaving the actual integrator call in the
    public runtime facade.
    """

    method_use = str(method or cfg.time.method)
    kwargs: dict[str, Any] = dict(
        dt=float(dt),
        steps=int(steps),
        method=method_use,
        terms=term_config,
        sample_stride=int(sample_stride),
        diagnostics_stride=int(diagnostics_stride),
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
        laguerre_mode=str(laguerre_mode),
        omega_ky_index=int(ky_index),
        omega_kx_index=int(kx_index),
        flux_scale=float(cfg.normalization.flux_scale),
        wphi_scale=float(cfg.normalization.wphi_scale),
        fixed_dt=bool(fixed_dt),
        dt_min=float(cfg.time.dt_min),
        dt_max=cfg.time.dt_max,
        cfl=float(cfg.time.cfl),
        cfl_fac=resolve_cfl_fac(method_use, cfg.time.cfl_fac),
        collision_split=bool(cfg.time.collision_split),
        collision_scheme=str(cfg.time.collision_scheme),
        implicit_restart=int(cfg.time.implicit_restart),
        implicit_preconditioner=cfg.time.implicit_preconditioner,
        fixed_mode_ky_index=fixed_mode_ky_index,
        fixed_mode_kx_index=fixed_mode_kx_index,
        external_phi=external_phi,
    )
    if not resolved_diagnostics:
        kwargs["resolved_diagnostics"] = False
    if show_progress:
        kwargs["show_progress"] = True
    return kwargs


def _normalize_linear_solver_name(solver: str) -> str:
    solver_key = solver.strip().lower().replace("-", "_")
    if solver_key == "explicit_time":
        return "explicit_time"
    return solver_key


def _parallel_requests_combined_ky_scan(cfg: RuntimeConfig) -> bool:
    """Return whether runtime parallel config requests the combined-ky scan path."""

    parallel = getattr(cfg, "parallel", None)
    if parallel is None:
        return False
    return (
        str(getattr(parallel, "strategy", "serial")).lower() == "combined_ky"
        and str(getattr(parallel, "axis", "ky")).lower() == "ky"
    )


def _normalize_independent_executor(backend: str, fallback: str) -> str:
    backend_key = str(backend).strip().lower().replace("-", "_")
    fallback_key = str(fallback).strip().lower().replace("-", "_")
    aliases = {
        "thread": "thread",
        "threads": "thread",
        "process": "process",
        "processes": "process",
    }
    if backend_key in {"", "auto"}:
        try:
            return aliases[fallback_key]
        except KeyError as exc:
            raise ValueError("parallel_executor must be 'thread' or 'process'") from exc
    try:
        return aliases[backend_key]
    except KeyError as exc:
        raise ValueError(
            "runtime [parallel] backend for independent scans must be "
            "'auto', 'thread', or 'process'"
        ) from exc


def _runtime_independent_parallel_plan(
    cfg: RuntimeConfig,
    *,
    problem_size: int,
    workers: int,
    executor: str,
) -> RuntimeIndependentParallelPlan:
    """Resolve independent ``k_y`` worker policy from arguments and config."""

    size = int(problem_size)
    if size < 0:
        raise ValueError("problem_size must be non-negative")
    requested = int(workers)
    if requested < 1:
        raise ValueError("workers must be >= 1")
    executor_key = _normalize_independent_executor("auto", executor)
    source = "arguments"
    strategy = "serial"
    axis = "ky"

    parallel = getattr(cfg, "parallel", None)
    if parallel is not None:
        strategy = str(getattr(parallel, "strategy", "serial")).strip().lower()
        axis = str(getattr(parallel, "axis", "ky")).strip().lower()
        if requested == 1 and strategy == "batch":
            if axis != "ky":
                raise ValueError(
                    "runtime [parallel] strategy='batch' is supported only for axis='ky'"
                )
            configured_workers = getattr(parallel, "num_devices", None)
            if configured_workers is None:
                configured_workers = getattr(parallel, "batch_size", None)
            requested = max(int(configured_workers or 1), 1)
            executor_key = _normalize_independent_executor(
                str(getattr(parallel, "backend", "auto")), executor
            )
            source = "runtime_config"

    effective = 0 if size == 0 else min(requested, size)
    return RuntimeIndependentParallelPlan(
        requested_workers=requested,
        effective_workers=effective,
        executor=executor_key,
        strategy=strategy,
        axis=axis,
        source=source,
        problem_size=size,
    )


def _midplane_index(grid: SpectralGrid) -> int:
    if grid.z.size <= 1:
        return 0
    return min(int(grid.z.size // 2 + 1), int(grid.z.size) - 1)


def _zero_kx_index(grid: SpectralGrid) -> int:
    kx = np.asarray(grid.kx, dtype=float)
    return int(np.argmin(np.abs(kx)))


def _nearest_index_from_candidates(
    values: np.ndarray,
    target: float,
    candidates: np.ndarray,
) -> int:
    """Return the candidate index nearest to ``target`` in physical coordinates."""

    values_arr = np.asarray(values, dtype=float)
    candidate_arr = np.asarray(candidates, dtype=int)
    if values_arr.size == 0:
        raise ValueError("values must be non-empty")
    if candidate_arr.size == 0:
        raise ValueError("candidate indices must be non-empty")
    return int(
        candidate_arr[int(np.argmin(np.abs(values_arr[candidate_arr] - float(target))))]
    )


def _validate_dealias_mask_shape(
    mask: Any,
    *,
    ky_size: int,
    kx_size: int,
) -> np.ndarray:
    """Return a boolean dealias mask after validating it matches ky/kx axes."""

    mask_arr = np.asarray(mask, dtype=bool)
    expected = (int(ky_size), int(kx_size))
    if mask_arr.shape != expected:
        raise ValueError(
            "dealias_mask shape must match (ky, kx) grid sizes; "
            f"got {mask_arr.shape}, expected {expected}"
        )
    return mask_arr


def _active_ky_indices(mask: np.ndarray, ky_size: int) -> np.ndarray:
    """Return ky rows with at least one retained kx, falling back to all ky."""

    candidates = np.where(np.any(mask, axis=1))[0]
    if candidates.size == 0:
        return np.arange(int(ky_size), dtype=int)
    return candidates


def _active_kx_indices(mask: np.ndarray, ky_index: int, kx_size: int) -> np.ndarray:
    """Return retained kx entries for ``ky_index``, falling back to all kx."""

    candidates = np.where(mask[int(ky_index)])[0]
    if candidates.size == 0:
        return np.arange(int(kx_size), dtype=int)
    return candidates


def _select_nonlinear_mode_indices(
    grid: SpectralGrid,
    *,
    ky_target: float,
    kx_target: float | None,
    use_dealias_mask: bool,
) -> tuple[int, int]:
    ky = np.asarray(grid.ky, dtype=float)
    kx = np.asarray(grid.kx, dtype=float)
    kx_pick_target = 0.0 if kx_target is None else float(kx_target)
    if not use_dealias_mask:
        ky_pick = select_ky_index(ky, ky_target)
        kx_pick = _nearest_index_from_candidates(
            kx, kx_pick_target, np.arange(kx.size, dtype=int)
        )
        return ky_pick, kx_pick

    mask = _validate_dealias_mask_shape(
        grid.dealias_mask,
        ky_size=ky.size,
        kx_size=kx.size,
    )
    ky_pick = _nearest_index_from_candidates(
        ky, ky_target, _active_ky_indices(mask, ky.size)
    )
    kx_pick = _nearest_index_from_candidates(
        kx, kx_pick_target, _active_kx_indices(mask, ky_pick, kx.size)
    )
    return int(ky_pick), int(kx_pick)


def _infer_runtime_nonlinear_steps(
    cfg: RuntimeConfig,
    *,
    dt: float,
    steps: int | None,
) -> int:
    """Infer nonlinear explicit step counts with the same dt ceiling as the integrator."""

    if steps is not None:
        steps_val = int(steps)
    elif bool(cfg.time.fixed_dt):
        steps_val = int(
            np.round(float(cfg.time.t_max) / max(float(cfg.time.dt), 1.0e-12))
        )
    else:
        # Keep runtime inference aligned with adaptive stepping: when
        # dt_max is unset, the nonlinear integrator clamps at dt itself.
        dt_cap = float(cfg.time.dt_max) if cfg.time.dt_max is not None else float(dt)
        steps_val = int(np.ceil(float(cfg.time.t_max) / max(dt_cap, 1.0e-12)))
    if steps_val < 1:
        raise ValueError("steps must be >= 1")
    return steps_val


def _runtime_external_phi(cfg: RuntimeConfig) -> float | None:
    """Return a runtime external-phi source if requested."""

    source = str(cfg.expert.source).strip().lower()
    if source in {"", "default"}:
        return None
    if source != "phiext_full":
        raise ValueError(
            f"unsupported expert.source={cfg.expert.source!r}; expected 'default' or 'phiext_full'"
        )
    return float(cfg.expert.phi_ext)


_RUNTIME_CASE_FIT_KEYS = _WORKFLOW_RUNTIME_CASE_FIT_KEYS

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
    fit_growth_rate_with_stats,
    fit_runtime_linear_diagnostics,
    independent_map,
    integrate_linear_diagnostics,
    integrate_linear_from_config,
    integrate_nonlinear_explicit_diagnostics_state,
    prepare_nonlinear_explicit_diagnostics,
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
    "window_method",
    "mode_method",
    "fit_signal",
)


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
    ky_layout: str = "full",
) -> np.ndarray:
    """Load an initial state while preserving the runtime module patch surface.

    ``ny`` is the two-sided axis length and ``ky_layout`` the axis the caller
    wants the state on (:mod:`gkx.core_ky_layout`).
    """

    layout = cast(KyLayout, str(ky_layout).strip().lower())
    if path.suffix.lower() == ".nc":
        return load_netcdf_restart_state(
            path,
            nspecies=nspecies,
            Nl=Nl,
            Nm=Nm,
            ny=ny,
            nx=nx,
            nz=nz,
            ky_layout=layout,
        )
    return runtime_startup._load_initial_state_from_file(
        path,
        nspecies=nspecies,
        Nl=Nl,
        Nm=Nm,
        ny=ny,
        nx=nx,
        nz=nz,
        ky_layout=layout,
    )


_slice_runtime_diagnostics = slice_runtime_diagnostics
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


def solve(case: Case, **options: Any) -> RuntimeLinearResult | RuntimeNonlinearResult:
    """Solve one case through its existing linear or nonlinear runtime owner."""
    if case.physics.nonlinear:
        return run_runtime_nonlinear(case, **options)
    if case.physics.linear:
        return run_runtime_linear(case, **options)
    raise ValueError("case must enable linear or nonlinear physics")


def prepare(case: Case, **options: Any) -> Any:
    """Prepare a reusable compiled nonlinear simulation; see its class docstring."""
    if not case.physics.nonlinear:
        raise ValueError("prepare currently requires nonlinear physics")
    if options.pop("diagnostics", True) is not True:
        raise ValueError("prepare requires diagnostics=True")
    return run_runtime_nonlinear_impl(
        case,
        diagnostics=True,
        prepare_only=True,
        deps=_runtime_nonlinear_dispatch_deps(),
        **options,
    )


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
    window_method: str = "stationary",
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
    window_method: str = "stationary",
    krylov_cfg: KrylovConfig | None = None,
    mode_method: str = "project",
    fit_signal: str = "auto",
    show_progress: bool = False,
    workers: int = 1,
    parallel_executor: str = "thread",
    warm_start: bool | None = None,
) -> RuntimeLinearScanResult:
    """Run a ky scan; ``warm_start`` overrides the case output policy."""

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
        warm_start=warm_start,
        deps=_runtime_scan_orchestration_deps(),
    )


def _runtime_scan_orchestration_deps() -> Any:
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
    window_method: str = "stationary",
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
    return build_runtime_scan_batch_deps(_runtime_facade_module())


def _runtime_nonlinear_dispatch_deps() -> RuntimeNonlinearDispatchDeps:
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


# ---- merged from workflows/runtime/execution.py ----
# That module had exactly one consumer -- this one -- which imported most
# of its names. A boundary that wide is a split, not an interface.


@dataclass(frozen=True)
class RuntimeLinearDispatchDeps:
    """Patchable dependencies for one configured linear runtime run."""

    resolve_runtime_hl_dims: Callable[..., tuple[int, int]]
    run_full_linear_runtime: Callable[..., RuntimeLinearResult]
    full_deps: Any


@dataclass(frozen=True)
class _RuntimeLinearRequest(_RuntimeLinearFitOptions):
    cfg: RuntimeConfig
    ky_target: float
    Nl: int | None
    Nm: int | None
    solver: str
    method: str | None
    dt: float | None
    steps: int | None
    sample_stride: int | None
    krylov_cfg: Any
    fit_signal: str
    return_state: bool
    initial_state: Any | None
    show_progress: bool
    status_callback: Callable[[str], None] | None
    deps: RuntimeLinearDispatchDeps


def build_runtime_linear_dispatch_deps(scope: Any) -> RuntimeLinearDispatchDeps:
    """Build linear dispatch dependencies from a patchable runtime facade scope."""

    from gkx.workflows.linear import FullLinearRuntimeDeps
    from gkx.workflows.runtime.diagnostics import (
        RuntimeQuasilinearFinalizationDeps,
    )

    return RuntimeLinearDispatchDeps(
        resolve_runtime_hl_dims=scope._resolve_runtime_hl_dims,
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
            fit_growth_rate_with_stats=scope.fit_growth_rate_with_stats,
            extract_eigenfunction=scope.extract_eigenfunction,
        ),
    )


def _runtime_linear_status(request: _RuntimeLinearRequest, message: str) -> None:
    if request.status_callback is not None:
        request.status_callback(message)


def _run_full_linear_request(
    request: _RuntimeLinearRequest,
    *,
    Nl_use: int,
    Nm_use: int,
) -> RuntimeLinearResult:
    return request.deps.run_full_linear_runtime(
        request.cfg,
        deps=request.deps.full_deps,
        ky_target=request.ky_target,
        Nl=Nl_use,
        Nm=Nm_use,
        solver=request.solver,
        method=request.method,
        dt=request.dt,
        steps=request.steps,
        sample_stride=request.sample_stride,
        **request.fit_fields(),
        krylov_cfg=request.krylov_cfg,
        fit_signal=request.fit_signal,
        return_state=request.return_state,
        initial_state=request.initial_state,
        show_progress=request.show_progress,
        status_callback=request.status_callback,
    )


def _dispatch_runtime_linear_request(
    request: _RuntimeLinearRequest,
) -> RuntimeLinearResult:
    Nl_use, Nm_use = request.deps.resolve_runtime_hl_dims(
        request.cfg, Nl=request.Nl, Nm=request.Nm
    )
    _runtime_linear_status(request, "building runtime geometry")
    return _run_full_linear_request(request, Nl_use=Nl_use, Nm_use=Nm_use)


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
    window_method: str = "stationary",
    krylov_cfg: Any = None,
    mode_method: str = "project",
    fit_signal: str = "auto",
    return_state: bool = False,
    initial_state: Any | None = None,
    show_progress: bool = False,
    status_callback: Callable[[str], None] | None = None,
    deps: RuntimeLinearDispatchDeps,
) -> RuntimeLinearResult:
    """Run one linear point from a case-agnostic runtime config."""

    return _dispatch_runtime_linear_request(
        _RuntimeLinearRequest(
            cfg=cfg,
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
            window_method=window_method,
            krylov_cfg=krylov_cfg,
            mode_method=mode_method,
            fit_signal=fit_signal,
            return_state=return_state,
            initial_state=initial_state,
            show_progress=show_progress,
            status_callback=status_callback,
            deps=deps,
        )
    )


@dataclass(frozen=True)
class RuntimeNonlinearDispatchDeps:
    """Patchable dependencies for one configured nonlinear runtime run."""

    resolve_runtime_hl_dims: Callable[..., tuple[int, int]]
    run_full_nonlinear_runtime: Callable[..., RuntimeNonlinearResult]
    full_deps: Any


def build_runtime_nonlinear_dispatch_deps(scope: Any) -> RuntimeNonlinearDispatchDeps:
    """Build nonlinear dispatch dependencies from a patchable runtime facade scope."""

    from gkx.workflows.nonlinear import FullNonlinearRuntimeDeps

    return RuntimeNonlinearDispatchDeps(
        resolve_runtime_hl_dims=scope._resolve_runtime_hl_dims,
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
            prepare_nonlinear_explicit_diagnostics=scope.prepare_nonlinear_explicit_diagnostics,
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
    prepare_only: bool = False,
    deps: RuntimeNonlinearDispatchDeps,
) -> Any:
    """Run one nonlinear point from a case-agnostic runtime config."""

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    Nl_use, Nm_use = deps.resolve_runtime_hl_dims(cfg, Nl=Nl, Nm=Nm)
    _status("building runtime geometry")
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
        prepare_only=prepare_only,
    )


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
    "_zero_kx_index",
    "build_runtime_geometry",
    "build_runtime_linear_params",
    "build_runtime_linear_terms",
    "build_runtime_term_config",
    "run_runtime_linear",
    "run_runtime_nonlinear",
    "run_runtime_scan",
    "solve",
    "prepare",
    "build_runtime_nonlinear_diagnostics_kwargs",
    "_parallel_requests_combined_ky_scan",
]
