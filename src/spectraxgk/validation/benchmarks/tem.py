"""TEM benchmark runners behind the public :mod:`spectraxgk.benchmarks` facade."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    ModeSelectionBatch,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    select_ky_index,
)
from spectraxgk.validation.benchmarks import tem_paths as _paths
from spectraxgk.validation.benchmarks.batching import _resolve_streaming_window
from spectraxgk.validation.benchmarks.defaults import (
    TEM_KRYLOV_DEFAULT,
    TEM_OMEGA_D_SCALE,
    TEM_OMEGA_STAR_SCALE,
    TEM_RHO_STAR,
)
from spectraxgk.validation.benchmarks.fit_signals import _normalize_growth_rate
from spectraxgk.validation.benchmarks.initialization import _build_initial_condition
from spectraxgk.validation.benchmarks.reference import LinearRunResult, LinearScanResult
from spectraxgk.validation.benchmarks.solver_policy import _midplane_index
from spectraxgk.validation.benchmarks.species import _two_species_params
from spectraxgk.validation.benchmarks.scan import (
    normalize_solver_key,
    resolve_scan_mode_method,
    should_use_ky_batch,
)
from spectraxgk.config import TEMBaseCase, TimeConfig
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.linear import integrate_linear, integrate_linear_diagnostics
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from spectraxgk.solvers.linear.krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.solvers.time.diffrax import integrate_linear_diffrax_streaming
from spectraxgk.solvers.time.runners import integrate_linear_from_config
from spectraxgk.terms.assembly import compute_fields_cached


def _tem_hooks() -> _paths.TEMPathHooks:
    return _paths.TEMPathHooks(
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
class _TEMLinearSetup:
    cfg: TEMBaseCase
    grid_full: Any
    geom: SAlphaGeometry
    params: LinearParams
    terms: LinearTerms
    hooks: _paths.TEMPathHooks


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


def _run_tem_linear_request(request: _TEMLinearRequest) -> LinearRunResult:
    setup = _resolve_tem_linear_setup(request)
    state = _prepare_tem_linear_state(setup, request)
    if request.solver.lower() == "krylov":
        return _paths.run_tem_krylov_linear_path(
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
    return _paths.run_tem_time_linear_path(
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

    cfg = cfg or TEMBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params, terms = _tem_params_and_terms(cfg, geom, params, terms, Nm)
    solver_key = normalize_solver_key(solver)
    mode_method = resolve_scan_mode_method(mode_method, mode_only=mode_only)
    use_batch = should_use_ky_batch(
        ky_batch=ky_batch,
        solver_key=solver_key,
        dt=dt,
        steps=steps,
        tmin=tmin,
        tmax=tmax,
    )
    _validate_tem_species_indices(
        init_species_index=init_species_index,
        density_species_index=density_species_index,
    )
    return _paths.run_tem_scan_batches(
        ky_values=np.asarray(ky_values, dtype=float),
        grid_full=grid_full,
        geom=geom,
        params=params,
        terms=terms,
        init_cfg=cfg.init,
        n_laguerre=Nl,
        n_hermite=Nm,
        dt=dt,
        steps=steps,
        method=method,
        time_cfg=time_cfg,
        solver_key=solver_key,
        krylov_cfg=krylov_cfg,
        krylov_default=TEM_KRYLOV_DEFAULT,
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
        sample_stride=sample_stride,
        ky_batch=ky_batch,
        fixed_batch_shape=fixed_batch_shape,
        streaming_fit=streaming_fit,
        streaming_amp_floor=streaming_amp_floor,
        init_species_index=init_species_index,
        diagnostic_norm=diagnostic_norm,
        use_batch=use_batch,
        hooks=_tem_hooks(),
        show_progress=show_progress,
    )
