"""Cyclone linear benchmark ky-scan runner."""

# ruff: noqa: F401

from __future__ import annotations

from dataclasses import dataclass, replace
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
from spectraxgk.validation.benchmarks.batching import (
    _iter_ky_batches,
    _resolve_streaming_window,
)
from spectraxgk.validation.benchmarks.fit_signals import (
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.initialization import _build_initial_condition
from spectraxgk.validation.benchmarks.reference import CycloneRunResult, CycloneScanResult
from spectraxgk.validation.benchmarks.solver_policy import _midplane_index
from spectraxgk.validation.benchmarks.species import (
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


from spectraxgk.validation.benchmarks.cyclone_linear import run_cyclone_linear
from spectraxgk.validation.benchmarks.cyclone_scan_branches import (
    CycloneScanHooks,
    run_explicit_time_cyclone_scan,
    run_krylov_cyclone_scan,
    run_time_cyclone_scan,
)


@dataclass(frozen=True)
class _CycloneScanSetup:
    """Shared Cyclone ky-scan setup resolved before solver dispatch."""

    cfg: CycloneBaseCase
    init_cfg: InitializationConfig
    grid_full: Any
    geom: Any
    params: LinearParams
    terms: LinearTerms
    reference_aligned: bool
    diagnostic_norm: str
    solver_key: str
    fit_key: str
    auto_solver: bool
    mode_method: str
    mode_only: bool
    streaming_fit: bool
    need_density: bool
    use_batch: bool
    fit_policy: ScanFitWindowPolicy


def _scan_hooks() -> CycloneScanHooks:
    return CycloneScanHooks(
        cyclone_scan_result=CycloneScanResult,
        explicit_time_config=ExplicitTimeConfig,
        mode_selection=ModeSelection,
        mode_selection_batch=ModeSelectionBatch,
        select_ky_index=select_ky_index,
        select_ky_grid=select_ky_grid,
        build_initial_condition=_build_initial_condition,
        build_linear_cache=build_linear_cache,
        integrate_linear_explicit=integrate_linear_explicit,
        integrate_linear=integrate_linear,
        integrate_linear_diagnostics=integrate_linear_diagnostics,
        integrate_linear_from_config=integrate_linear_from_config,
        integrate_linear_diffrax_streaming=integrate_linear_diffrax_streaming,
        instantaneous_growth_rate_from_phi=instantaneous_growth_rate_from_phi,
        dominant_eigenpair=dominant_eigenpair,
        extract_mode_time_series=extract_mode_time_series,
        select_fit_signal_auto=_select_fit_signal_auto,
        run_cyclone_linear=run_cyclone_linear,
        normalize_growth_rate=_normalize_growth_rate,
        resolve_streaming_window=_resolve_streaming_window,
        midplane_index=_midplane_index,
        resolve_cfl_fac=resolve_cfl_fac,
    )


def _default_cyclone_scan_params(
    cfg: CycloneBaseCase,
    geom: Any,
    *,
    n_hermite: int,
) -> LinearParams:
    """Build the reference-aligned Cyclone ion species parameters."""

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
    return _apply_reference_hypercollisions(params, nhermite=n_hermite)


def _default_cyclone_scan_terms(cfg: CycloneBaseCase) -> LinearTerms:
    """Return the Cyclone scan default field-term policy."""

    if getattr(cfg.model, "adiabatic_ions", False):
        return LinearTerms(bpar=0.0)
    return LinearTerms()


def _build_cyclone_scan_fit_policy(
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
    """Pack scan growth-window options once for all Cyclone solver branches."""

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


def _build_cyclone_scan_setup(
    *,
    cfg: CycloneBaseCase | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    Nm: int,
    solver: str,
    fit_signal: str,
    diagnostic_norm: str,
    mode_method: str,
    mode_only: bool,
    streaming_fit: bool,
    ky_batch: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    tmin: float | None,
    tmax: float | None,
    reference_aligned: bool | None,
    fit_policy: ScanFitWindowPolicy,
) -> _CycloneScanSetup:
    """Resolve Cyclone scan geometry, species, term, and branch policies."""

    cfg_use = cfg or CycloneBaseCase()
    init_cfg = getattr(cfg_use, "init", None) or InitializationConfig()
    grid_full = build_spectral_grid(cfg_use.grid)
    reference_aligned_use = (
        bool(cfg_use.reference_aligned)
        if reference_aligned is None
        else bool(reference_aligned)
    )
    geom_cfg = cfg_use.geometry
    diagnostic_norm_use = diagnostic_norm
    mode_method_use = mode_method
    if reference_aligned_use:
        geom_cfg = replace(geom_cfg, drift_scale=1.0)
        if diagnostic_norm_use == "none":
            diagnostic_norm_use = "rho_star"
        if mode_method_use not in {"z_index", "max"}:
            mode_method_use = "z_index"
    geom = SAlphaGeometry.from_config(geom_cfg)
    params_use = (
        _default_cyclone_scan_params(cfg_use, geom, n_hermite=Nm)
        if params is None
        else params
    )
    terms_use = _default_cyclone_scan_terms(cfg_use) if terms is None else terms
    solver_key = normalize_solver_key(solver)
    fit_key = normalize_fit_signal(fit_signal)
    auto_solver = solver_key == "auto"
    if auto_solver:
        solver_key = "explicit_time" if reference_aligned_use else "time"
    streaming_fit_use, mode_only_use = apply_auto_fit_scan_policy(
        fit_key, streaming_fit=streaming_fit, mode_only=mode_only
    )
    mode_method_use = resolve_scan_mode_method(
        mode_method_use, mode_only=mode_only_use
    )
    use_batch = should_use_ky_batch(
        ky_batch=ky_batch,
        solver_key=solver_key,
        dt=dt,
        steps=steps,
        tmin=tmin,
        tmax=tmax,
    )
    return _CycloneScanSetup(
        cfg=cfg_use,
        init_cfg=init_cfg,
        grid_full=grid_full,
        geom=geom,
        params=params_use,
        terms=terms_use,
        reference_aligned=reference_aligned_use,
        diagnostic_norm=diagnostic_norm_use,
        solver_key=solver_key,
        fit_key=fit_key,
        auto_solver=auto_solver,
        mode_method=mode_method_use,
        mode_only=mode_only_use,
        streaming_fit=streaming_fit_use,
        need_density=fit_key in {"density", "auto"},
        use_batch=use_batch,
        fit_policy=fit_policy,
    )


def _run_cyclone_scan_krylov_branch(
    *,
    setup: _CycloneScanSetup,
    ky_values: np.ndarray,
    Nl: int,
    Nm: int,
    mode_follow: bool,
    krylov_cfg: KrylovConfig | None,
    show_progress: bool,
) -> CycloneScanResult:
    """Dispatch the Cyclone ky scan to the Krylov branch-following path."""

    return run_krylov_cyclone_scan(
        ky_values=ky_values,
        grid_full=setup.grid_full,
        geom=setup.geom,
        params=setup.params,
        terms=setup.terms,
        init_cfg=setup.init_cfg,
        n_laguerre=Nl,
        n_hermite=Nm,
        mode_follow=mode_follow,
        krylov_cfg=krylov_cfg,
        krylov_default=CYCLONE_KRYLOV_DEFAULT,
        diagnostic_norm=setup.diagnostic_norm,
        show_progress=show_progress,
        hooks=_scan_hooks(),
    )


def _run_cyclone_scan_explicit_branch(
    *,
    setup: _CycloneScanSetup,
    ky_values: np.ndarray,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    time_cfg: TimeConfig | None,
    krylov_cfg: KrylovConfig | None,
    show_progress: bool,
) -> CycloneScanResult:
    """Dispatch the Cyclone ky scan to the reference-aligned explicit path."""

    return run_explicit_time_cyclone_scan(
        ky_values=ky_values,
        grid_full=setup.grid_full,
        geom=setup.geom,
        params=setup.params,
        terms=setup.terms,
        cfg=setup.cfg,
        time_cfg=time_cfg,
        init_cfg=setup.init_cfg,
        n_laguerre=Nl,
        n_hermite=Nm,
        dt=dt,
        steps=steps,
        krylov_cfg=krylov_cfg,
        krylov_default=CYCLONE_KRYLOV_DEFAULT,
        reference_aligned=setup.reference_aligned,
        diagnostic_norm=setup.diagnostic_norm,
        show_progress=show_progress,
        hooks=_scan_hooks(),
    )


def _run_cyclone_scan_time_branch(
    *,
    setup: _CycloneScanSetup,
    ky_values: np.ndarray,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    time_cfg: TimeConfig | None,
    krylov_cfg: KrylovConfig | None,
    tmin: float | None,
    tmax: float | None,
    sample_stride: int | None,
    use_jit: bool,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_amp_floor: float,
    show_progress: bool,
) -> CycloneScanResult:
    """Dispatch the Cyclone ky scan to saved-time or streaming time paths."""

    return run_time_cyclone_scan(
        ky_values=ky_values,
        grid_full=setup.grid_full,
        geom=setup.geom,
        params=setup.params,
        terms=setup.terms,
        cfg=setup.cfg,
        time_cfg=time_cfg,
        init_cfg=setup.init_cfg,
        n_laguerre=Nl,
        n_hermite=Nm,
        dt=dt,
        steps=steps,
        method=method,
        krylov_cfg=krylov_cfg,
        tmin=tmin,
        tmax=tmax,
        window_fraction=setup.fit_policy.window_fraction,
        min_points=setup.fit_policy.min_points,
        start_fraction=setup.fit_policy.start_fraction,
        growth_weight=setup.fit_policy.growth_weight,
        require_positive=setup.fit_policy.require_positive,
        min_amp_fraction=setup.fit_policy.min_amp_fraction,
        max_amp_fraction=setup.fit_policy.max_amp_fraction,
        max_fraction=setup.fit_policy.max_fraction,
        end_fraction=setup.fit_policy.end_fraction,
        phase_weight=setup.fit_policy.phase_weight,
        length_weight=setup.fit_policy.length_weight,
        min_r2=setup.fit_policy.min_r2,
        late_penalty=setup.fit_policy.late_penalty,
        min_slope=setup.fit_policy.min_slope,
        min_slope_frac=setup.fit_policy.min_slope_frac,
        slope_var_weight=setup.fit_policy.slope_var_weight,
        window_method=setup.fit_policy.window_method,
        mode_method=setup.mode_method,
        mode_only=setup.mode_only,
        sample_stride=sample_stride,
        fit_key=setup.fit_key,
        need_density=setup.need_density,
        diagnostic_norm=setup.diagnostic_norm,
        use_jit=use_jit,
        ky_batch=ky_batch,
        fixed_batch_shape=fixed_batch_shape,
        streaming_fit=setup.streaming_fit,
        streaming_amp_floor=streaming_amp_floor,
        auto_solver=setup.auto_solver,
        use_batch=setup.use_batch,
        fit_policy=setup.fit_policy,
        hooks=_scan_hooks(),
        show_progress=show_progress,
    )


def run_cyclone_scan(
    ky_values: np.ndarray,
    Nl: int = 6,
    Nm: int = 12,
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
    method: str = "imex2",
    params: LinearParams | None = None,
    cfg: CycloneBaseCase | None = None,
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
    diagnostic_norm: str = "none",
    use_jit: bool = True,
    ky_batch: int = 4,
    fixed_batch_shape: bool = True,
    streaming_fit: bool = True,
    streaming_amp_floor: float = 1.0e-30,
    mode_follow: bool = True,
    reference_aligned: bool | None = None,
    show_progress: bool = False,
) -> CycloneScanResult:
    """Run the linear Cyclone benchmark for a list of ky values.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    fit_policy = _build_cyclone_scan_fit_policy(
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
    setup = _build_cyclone_scan_setup(
        cfg=cfg,
        params=params,
        terms=terms,
        Nm=Nm,
        solver=solver,
        fit_signal=fit_signal,
        diagnostic_norm=diagnostic_norm,
        mode_method=mode_method,
        mode_only=mode_only,
        streaming_fit=streaming_fit,
        ky_batch=ky_batch,
        dt=dt,
        steps=steps,
        tmin=tmin,
        tmax=tmax,
        reference_aligned=reference_aligned,
        fit_policy=fit_policy,
    )

    ky_values_arr = np.asarray(ky_values, dtype=float)

    if setup.solver_key == "krylov":
        return _run_cyclone_scan_krylov_branch(
            setup=setup,
            ky_values=ky_values_arr,
            Nl=Nl,
            Nm=Nm,
            mode_follow=mode_follow,
            krylov_cfg=krylov_cfg,
            show_progress=show_progress,
        )

    if setup.solver_key == "explicit_time":
        return _run_cyclone_scan_explicit_branch(
            setup=setup,
            ky_values=ky_values_arr,
            Nl=Nl,
            Nm=Nm,
            dt=dt,
            steps=steps,
            time_cfg=time_cfg,
            krylov_cfg=krylov_cfg,
            show_progress=show_progress,
        )
    return _run_cyclone_scan_time_branch(
        setup=setup,
        ky_values=ky_values_arr,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps,
        method=method,
        time_cfg=time_cfg,
        krylov_cfg=krylov_cfg,
        tmin=tmin,
        tmax=tmax,
        sample_stride=sample_stride,
        use_jit=use_jit,
        ky_batch=ky_batch,
        fixed_batch_shape=fixed_batch_shape,
        streaming_amp_floor=streaming_amp_floor,
        show_progress=show_progress,
    )
