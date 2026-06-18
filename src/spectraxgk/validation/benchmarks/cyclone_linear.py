"""Cyclone linear benchmark single-mode runner."""

# ruff: noqa: F401

from __future__ import annotations

from dataclasses import replace
from typing import Callable

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
from spectraxgk.validation.benchmarks import cyclone_linear_paths as _paths


def run_cyclone_linear(
    ky_target: float = 0.3,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
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
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    fit_signal: str = "auto",
    init_cfg: InitializationConfig | None = None,
    diagnostic_norm: str = "none",
    use_jit: bool = True,
    reference_aligned: bool | None = None,
    gx_reference: bool | None = None,
    show_progress: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> CycloneRunResult:
    """Run the linear Cyclone benchmark and extract growth rate."""

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    cfg = cfg or CycloneBaseCase()
    init_cfg = init_cfg or getattr(cfg, "init", None) or InitializationConfig()
    _status("building spectral grid")
    grid_full = build_spectral_grid(cfg.grid)
    if gx_reference is not None:
        reference_aligned = gx_reference
    reference_aligned_use = (
        bool(cfg.reference_aligned)
        if reference_aligned is None
        else bool(reference_aligned)
    )
    geom_cfg = cfg.geometry
    if reference_aligned_use:
        geom_cfg = replace(geom_cfg, drift_scale=1.0)
        if diagnostic_norm == "none":
            diagnostic_norm = "rho_star"
        if mode_method not in {"z_index", "max"}:
            mode_method = "z_index"
    _status("building s-alpha geometry")
    geom = SAlphaGeometry.from_config(geom_cfg)
    if params is None:
        _status("building Cyclone linear parameters")
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
        params = _apply_reference_hypercollisions(params, nhermite=Nm)
    if terms is None:
        if getattr(cfg.model, "adiabatic_ions", False):
            terms = LinearTerms(bpar=0.0)
        else:
            terms = LinearTerms()
    solver_key = solver.strip().lower()
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    need_density = fit_key in {"density", "auto"}

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    _status(f"selected ky index {ky_index} at ky={float(grid.ky[sel.ky_index]):.4f}")
    _status("building initial condition")
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

    def _fresh_G0() -> jnp.ndarray:
        return jnp.asarray(G0_base)

    _status("building linear cache")
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
        if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
            return False
        if require_positive and gamma_val <= 0.0:
            return False
        return True

    _paths.sync_path_hooks(globals())

    def _run_krylov() -> tuple[float, float, np.ndarray, np.ndarray]:
        return _paths.run_cyclone_krylov_path(
            grid=grid,
            cache=cache,
            params=params,
            geom=geom,
            terms=terms,
            Nl=Nl,
            Nm=Nm,
            init_cfg=init_cfg,
            krylov_cfg=krylov_cfg,
            diagnostic_norm=diagnostic_norm,
            show_progress=show_progress,
            status=_status,
            fresh_G0=_fresh_G0,
        )

    def _run_time() -> tuple[float, float, np.ndarray, np.ndarray]:
        return _paths.run_cyclone_time_path(
            grid=grid,
            cache=cache,
            params=params,
            geom=geom,
            terms=terms,
            cfg=cfg,
            time_cfg=time_cfg,
            sel=sel,
            dt=dt,
            steps=steps,
            method=method,
            sample_stride=sample_stride,
            fit_key=fit_key,
            need_density=need_density,
            reference_aligned=reference_aligned_use,
            use_jit=use_jit,
            diagnostic_norm=diagnostic_norm,
            show_progress=show_progress,
            status=_status,
            fresh_G0=_fresh_G0,
            mode_method=mode_method,
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

    if solver_key == "krylov":
        gamma, omega, phi_t_np, t = _run_krylov()
    elif solver_key == "auto":
        try:
            gamma, omega, phi_t_np, t = _run_time()
        except ValueError as exc:
            _status(f"time-path failed ({exc}); falling back to Krylov solve")
            gamma, omega, phi_t_np, t = _run_krylov()
        if not _is_valid_growth(gamma, omega):
            _status("time-path result rejected; falling back to Krylov solve")
            gamma, omega, phi_t_np, t = _run_krylov()
    else:
        gamma, omega, phi_t_np, t = _run_time()

    _status(f"completed Cyclone linear run at ky={float(grid.ky[sel.ky_index]):.4f}")

    return CycloneRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )
