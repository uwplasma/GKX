"""ETG linear ky-scan benchmark runner."""

# ruff: noqa: F401

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    ModeSelectionBatch,
    fit_growth_rate,
    fit_growth_rate_auto,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.validation.benchmarks.defaults import (
    ETG_KRYLOV_DEFAULT,
    ETG_OMEGA_D_SCALE,
    ETG_OMEGA_STAR_SCALE,
    ETG_RHO_STAR,
)
from spectraxgk.validation.benchmarks.batching import (
    _iter_ky_batches,
    _resolve_streaming_window,
)
from spectraxgk.validation.benchmarks.fit_signals import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.initialization import _build_initial_condition
from spectraxgk.validation.benchmarks.reference import LinearRunResult, LinearScanResult
from spectraxgk.validation.benchmarks.solver_policy import _midplane_index
from spectraxgk.validation.benchmarks.species import (
    _electron_only_params,
    _two_species_params,
)
from spectraxgk.validation.benchmarks.scan import (
    ScanFitWindowPolicy,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    normalize_fit_signal,
    normalize_solver_key,
    resolve_scan_mode_method,
    scan_window_valid,
    should_use_ky_batch,
)
from spectraxgk.config import ETGBaseCase, TimeConfig
from spectraxgk.solvers.time.diffrax import (
    integrate_linear_diffrax,
    integrate_linear_diffrax_streaming,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
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


from spectraxgk.validation.benchmarks.etg_linear import run_etg_linear
from spectraxgk.validation.benchmarks import etg_scan_paths as _paths


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


def _prepare_etg_scan_setup(
    *,
    cfg: ETGBaseCase | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    Nm: int,
    solver: str,
    fit_signal: str,
    streaming_fit: bool,
    mode_only: bool,
    mode_method: str,
    ky_batch: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
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
) -> _ETGScanSetup:
    """Prepare ETG scan geometry, species, solver, and fit policies."""

    cfg_use = cfg or ETGBaseCase()
    grid_full = build_spectral_grid(cfg_use.grid)
    geom = SAlphaGeometry.from_config(cfg_use.geometry)
    params_use = _default_etg_scan_params(cfg_use, geom, Nm, params)
    terms_use = _default_etg_scan_terms(terms)
    solver_key = normalize_solver_key(solver)
    fit_key = normalize_fit_signal(fit_signal)
    auto_solver = solver_key == "auto"
    if auto_solver:
        solver_key = "time"
    streaming_fit, mode_only = apply_auto_fit_scan_policy(
        fit_key, streaming_fit=streaming_fit, mode_only=mode_only
    )
    mode_method = resolve_scan_mode_method(mode_method, mode_only=mode_only)
    fit_policy = _build_etg_scan_fit_policy(
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
            ky_batch=ky_batch,
            solver_key=solver_key,
            dt=dt,
            steps=steps,
            tmin=tmin,
            tmax=tmax,
        ),
        fit_policy=fit_policy,
    )


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


def _run_etg_scan_batch(
    setup: _ETGScanSetup,
    batch: _ETGScanBatch,
    *,
    options: _ETGScanRuntimeOptions,
    acc: _ETGScanAccumulator,
) -> tuple[jnp.ndarray | None, complex | None]:
    """Run one ETG scan batch and append growth/frequency outputs."""

    if setup.solver_key == "krylov":
        gamma, omega, prev_vec, prev_eig = _paths.run_etg_krylov_batch(
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

    time_result = _paths.run_etg_time_batch(
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
    if time_result.handled:
        return acc.prev_vec, acc.prev_eig

    _paths.append_etg_time_fit_results(
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
    return acc.prev_vec, acc.prev_eig


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
    _paths.sync_path_hooks(globals())
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
    return _prepare_etg_scan_setup(
        cfg=request.cfg,
        params=request.params,
        terms=request.terms,
        Nm=request.Nm,
        solver=request.solver,
        fit_signal=request.fit_signal,
        streaming_fit=request.streaming_fit,
        mode_only=request.mode_only,
        mode_method=request.mode_method,
        ky_batch=request.ky_batch,
        dt=request.dt,
        steps=request.steps,
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
