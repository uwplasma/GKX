"""Cyclone linear benchmark single-mode runner."""

# ruff: noqa: F401

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    ModeSelectionBatch,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.validation.benchmarks.defaults import (
    CYCLONE_KRYLOV_DEFAULT,
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
)
from spectraxgk.validation.benchmarks.scan import (
    _iter_ky_batches,
    _resolve_streaming_window,
)
from spectraxgk.diagnostics.growth_rates import (
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.defaults import _build_initial_condition
from spectraxgk.validation.benchmarks.defaults import CycloneRunResult, CycloneScanResult
from spectraxgk.validation.benchmarks.defaults import _midplane_index
from spectraxgk.validation.benchmarks.defaults import (
    REFERENCE_DAMP_ENDS_AMP,
    REFERENCE_DAMP_ENDS_WIDTHFRAC,
    _apply_reference_hypercollisions,
)
from spectraxgk.validation.benchmarks.scan import (
    ScanFitWindowPolicy,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    normalize_fit_signal,
    normalize_solver_key,
    resolve_scan_mode_method,
    should_use_ky_batch,
)
from spectraxgk.config import (
    CycloneBaseCase,
    InitializationConfig,
    TimeConfig,
    resolve_cfl_fac,
)
from spectraxgk.solvers.time.diffrax import integrate_linear_diffrax_streaming
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.solvers.time.explicit import (
    ExplicitTimeConfig,
    integrate_linear_explicit,
)
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
from spectraxgk.validation.benchmarks import cyclone_linear_paths as _paths


@dataclass(frozen=True)
class _CycloneLinearSetup:
    cfg: CycloneBaseCase
    init_cfg: InitializationConfig
    grid: Any
    geom: SAlphaGeometry
    params: LinearParams
    terms: LinearTerms
    selection: ModeSelection
    cache: Any
    G0_base: np.ndarray
    reference_aligned: bool
    diagnostic_norm: str
    mode_method: str
    fit_key: str
    need_density: bool


@dataclass(frozen=True)
class _CycloneLinearFitOptions:
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


@dataclass(frozen=True)
class _CycloneLinearRequest:
    ky_target: float
    Nl: int
    Nm: int
    dt: float
    steps: int
    method: str
    params: LinearParams | None
    cfg: CycloneBaseCase | None
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
    terms: LinearTerms | None
    sample_stride: int | None
    fit_signal: str
    init_cfg: InitializationConfig | None
    diagnostic_norm: str
    use_jit: bool
    reference_aligned: bool | None
    show_progress: bool
    status_callback: Callable[[str], None] | None


def _cyclone_linear_request_from_locals(values: dict[str, Any]) -> _CycloneLinearRequest:
    """Pack public ``run_cyclone_linear`` arguments once for internal routing."""

    return _CycloneLinearRequest(
        **{field.name: values[field.name] for field in fields(_CycloneLinearRequest)}
    )


def _default_cyclone_params(cfg: CycloneBaseCase, geom: SAlphaGeometry, Nm: int) -> LinearParams:
    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        R_over_LTe=cfg.model.R_over_LTe,
        omega_d_scale=CYCLONE_OMEGA_D_SCALE,
        omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
        rho_star=CYCLONE_RHO_STAR,
        kpar_scale=float(geom.gradpar()),
        nu=cfg.model.nu_i,
        damp_ends_amp=REFERENCE_DAMP_ENDS_AMP,
        damp_ends_widthfrac=REFERENCE_DAMP_ENDS_WIDTHFRAC,
    )
    return _apply_reference_hypercollisions(params, nhermite=Nm)


def _default_cyclone_terms(cfg: CycloneBaseCase) -> LinearTerms:
    return LinearTerms(bpar=0.0) if getattr(cfg.model, "adiabatic_ions", False) else LinearTerms()


def _reference_aligned_geometry_and_options(
    cfg: CycloneBaseCase,
    reference_aligned: bool | None,
    diagnostic_norm: str,
    mode_method: str,
) -> tuple[Any, bool, str, str]:
    reference_aligned_use = bool(cfg.reference_aligned) if reference_aligned is None else bool(reference_aligned)
    geom_cfg = cfg.geometry
    if reference_aligned_use:
        geom_cfg = replace(geom_cfg, drift_scale=1.0)
        diagnostic_norm = "rho_star" if diagnostic_norm == "none" else diagnostic_norm
        mode_method = "z_index" if mode_method not in {"z_index", "max"} else mode_method
    return geom_cfg, reference_aligned_use, diagnostic_norm, mode_method


def _resolve_fit_signal(fit_signal: str) -> tuple[str, bool]:
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    return fit_key, fit_key in {"density", "auto"}


def _build_cyclone_linear_setup(
    *,
    ky_target: float,
    Nl: int,
    Nm: int,
    cfg: CycloneBaseCase,
    init_cfg: InitializationConfig,
    params: LinearParams | None,
    terms: LinearTerms | None,
    fit_signal: str,
    diagnostic_norm: str,
    mode_method: str,
    reference_aligned: bool | None,
    status: Callable[[str], None],
) -> _CycloneLinearSetup:
    status("building spectral grid")
    grid_full = build_spectral_grid(cfg.grid)
    geom_cfg, reference_aligned_use, diagnostic_norm, mode_method = _reference_aligned_geometry_and_options(
        cfg,
        reference_aligned,
        diagnostic_norm,
        mode_method,
    )
    status("building s-alpha geometry")
    geom = SAlphaGeometry.from_config(geom_cfg)
    if params is None:
        status("building Cyclone linear parameters")
        params = _default_cyclone_params(cfg, geom, Nm)
    terms = _default_cyclone_terms(cfg) if terms is None else terms
    fit_key, need_density = _resolve_fit_signal(fit_signal)
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    status(f"selected ky index {ky_index} at ky={float(grid.ky[sel.ky_index]):.4f}")
    status("building initial condition")
    G0_base = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            ky_index=sel.ky_index,
            kx_index=sel.kx_index,
            Nl=Nl,
            Nm=Nm,
            init_cfg=init_cfg,
        )
    )
    status("building linear cache")
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    return _CycloneLinearSetup(
        cfg=cfg,
        init_cfg=init_cfg,
        grid=grid,
        geom=geom,
        params=params,
        terms=terms,
        selection=sel,
        cache=cache,
        G0_base=G0_base,
        reference_aligned=reference_aligned_use,
        diagnostic_norm=diagnostic_norm,
        mode_method=mode_method,
        fit_key=fit_key,
        need_density=need_density,
    )


def _fresh_cyclone_initial_state(setup: _CycloneLinearSetup) -> jnp.ndarray:
    return jnp.asarray(setup.G0_base)


def _valid_cyclone_growth(gamma_val: float, omega_val: float, *, require_positive: bool) -> bool:
    if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
        return False
    return not (require_positive and gamma_val <= 0.0)


def _run_cyclone_linear_krylov_path(
    *,
    setup: _CycloneLinearSetup,
    Nl: int,
    Nm: int,
    krylov_cfg: KrylovConfig | None,
    show_progress: bool,
    status: Callable[[str], None],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    return _paths.run_cyclone_krylov_path(
        grid=setup.grid,
        cache=setup.cache,
        params=setup.params,
        geom=setup.geom,
        terms=setup.terms,
        Nl=Nl,
        Nm=Nm,
        init_cfg=setup.init_cfg,
        krylov_cfg=krylov_cfg,
        diagnostic_norm=setup.diagnostic_norm,
        show_progress=show_progress,
        status=status,
        fresh_G0=lambda: _fresh_cyclone_initial_state(setup),
    )


def _run_cyclone_linear_time_path(
    *,
    setup: _CycloneLinearSetup,
    fit: _CycloneLinearFitOptions,
    time_cfg: TimeConfig | None,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    use_jit: bool,
    show_progress: bool,
    status: Callable[[str], None],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    return _paths.run_cyclone_time_path(
        grid=setup.grid,
        cache=setup.cache,
        params=setup.params,
        geom=setup.geom,
        terms=setup.terms,
        cfg=setup.cfg,
        time_cfg=time_cfg,
        sel=setup.selection,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        fit_key=setup.fit_key,
        need_density=setup.need_density,
        reference_aligned=setup.reference_aligned,
        use_jit=use_jit,
        diagnostic_norm=setup.diagnostic_norm,
        show_progress=show_progress,
        status=status,
        fresh_G0=lambda: _fresh_cyclone_initial_state(setup),
        mode_method=setup.mode_method,
        tmin=fit.tmin,
        tmax=fit.tmax,
        auto_window=fit.auto_window,
        window_fraction=fit.window_fraction,
        min_points=fit.min_points,
        start_fraction=fit.start_fraction,
        growth_weight=fit.growth_weight,
        require_positive=fit.require_positive,
        min_amp_fraction=fit.min_amp_fraction,
        max_fraction=fit.max_fraction,
        end_fraction=fit.end_fraction,
        max_amp_fraction=fit.max_amp_fraction,
        phase_weight=fit.phase_weight,
        length_weight=fit.length_weight,
        min_r2=fit.min_r2,
        late_penalty=fit.late_penalty,
        min_slope=fit.min_slope,
        min_slope_frac=fit.min_slope_frac,
        slope_var_weight=fit.slope_var_weight,
        window_method=fit.window_method,
    )


def _dispatch_cyclone_linear_solver(
    *,
    solver_key: str,
    setup: _CycloneLinearSetup,
    fit: _CycloneLinearFitOptions,
    Nl: int,
    Nm: int,
    krylov_cfg: KrylovConfig | None,
    time_cfg: TimeConfig | None,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    use_jit: bool,
    show_progress: bool,
    status: Callable[[str], None],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    def run_krylov() -> tuple[float, float, np.ndarray, np.ndarray]:
        return _run_cyclone_linear_krylov_path(
            setup=setup,
            Nl=Nl,
            Nm=Nm,
            krylov_cfg=krylov_cfg,
            show_progress=show_progress,
            status=status,
        )

    def run_time() -> tuple[float, float, np.ndarray, np.ndarray]:
        return _run_cyclone_linear_time_path(
            setup=setup,
            fit=fit,
            time_cfg=time_cfg,
            dt=dt,
            steps=steps,
            method=method,
            sample_stride=sample_stride,
            use_jit=use_jit,
            show_progress=show_progress,
            status=status,
        )

    if solver_key == "krylov":
        return run_krylov()
    if solver_key != "auto":
        return run_time()
    try:
        gamma, omega, phi_t_np, t = run_time()
    except ValueError as exc:
        status(f"time-path failed ({exc}); falling back to Krylov solve")
        return run_krylov()
    if not _valid_cyclone_growth(gamma, omega, require_positive=fit.require_positive):
        status("time-path result rejected; falling back to Krylov solve")
        return run_krylov()
    return gamma, omega, phi_t_np, t


def _pack_cyclone_linear_result(
    *,
    setup: _CycloneLinearSetup,
    gamma: float,
    omega: float,
    phi_t_np: np.ndarray,
    t: np.ndarray,
) -> CycloneRunResult:
    return CycloneRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(setup.grid.ky[setup.selection.ky_index]),
        selection=setup.selection,
    )


def _cyclone_linear_fit_options_from_request(
    request: _CycloneLinearRequest,
) -> _CycloneLinearFitOptions:
    return _CycloneLinearFitOptions(
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


def _emit_cyclone_linear_status(
    callback: Callable[[str], None] | None,
    message: str,
) -> None:
    if callback is not None:
        callback(message)


def _run_cyclone_linear_request(request: _CycloneLinearRequest) -> CycloneRunResult:
    cfg = request.cfg or CycloneBaseCase()
    init_cfg = request.init_cfg or getattr(cfg, "init", None) or InitializationConfig()

    def status(message: str) -> None:
        _emit_cyclone_linear_status(request.status_callback, message)

    setup = _build_cyclone_linear_setup(
        ky_target=request.ky_target,
        Nl=request.Nl,
        Nm=request.Nm,
        cfg=cfg,
        init_cfg=init_cfg,
        params=request.params,
        terms=request.terms,
        fit_signal=request.fit_signal,
        diagnostic_norm=request.diagnostic_norm,
        mode_method=request.mode_method,
        reference_aligned=request.reference_aligned,
        status=status,
    )
    _paths.sync_path_hooks(globals())
    gamma, omega, phi_t_np, t = _dispatch_cyclone_linear_solver(
        solver_key=request.solver.strip().lower(),
        setup=setup,
        fit=_cyclone_linear_fit_options_from_request(request),
        Nl=request.Nl,
        Nm=request.Nm,
        krylov_cfg=request.krylov_cfg,
        time_cfg=request.time_cfg,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        sample_stride=request.sample_stride,
        use_jit=request.use_jit,
        show_progress=request.show_progress,
        status=status,
    )
    status(f"completed Cyclone linear run at ky={float(setup.grid.ky[setup.selection.ky_index]):.4f}")
    return _pack_cyclone_linear_result(
        setup=setup,
        gamma=gamma,
        omega=omega,
        phi_t_np=phi_t_np,
        t=t,
    )


def run_cyclone_linear(
    ky_target: float = 0.3,
    Nl: int = 6, Nm: int = 12,
    dt: float = 0.01, steps: int = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: CycloneBaseCase | None = None,
    time_cfg: TimeConfig | None = None,
    solver: str = "auto",
    krylov_cfg: KrylovConfig | None = None,
    tmin: float | None = None, tmax: float | None = None,
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
    phase_weight: float = 0.2, length_weight: float = 0.05,
    min_r2: float = 0.0, late_penalty: float = 0.0,
    min_slope: float | None = None,
    min_slope_frac: float = 0.0,
    slope_var_weight: float = 0.0,
    window_method: str = "loglinear",
    mode_method: str = "project",
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    fit_signal: str = "auto",
    init_cfg: InitializationConfig | None = None,
    diagnostic_norm: str = "none",
    use_jit: bool = True,
    reference_aligned: bool | None = None,
    show_progress: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> CycloneRunResult:
    """Run the linear Cyclone benchmark and extract growth rate."""

    return _run_cyclone_linear_request(_cyclone_linear_request_from_locals(locals()))
