"""KBM fixed-beta ky-scan benchmark runner."""

# ruff: noqa: F401

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    windowed_growth_rate_from_omega_series,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.validation.benchmarks.defaults import (
    KBM_KRYLOV_DEFAULT,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
)
from spectraxgk.validation.benchmarks.batching import _resolve_streaming_window
from spectraxgk.validation.benchmarks.fit_signals import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.initialization import _build_initial_condition
from spectraxgk.validation.benchmarks.reference import LinearRunResult, LinearScanResult
from spectraxgk.validation.benchmarks.solver_policy import (
    _kbm_use_multi_target_krylov,
    _midplane_index,
    select_kbm_solver_auto,
)
from spectraxgk.validation.benchmarks.species import (
    _linked_boundary_end_damping,
    _two_species_params,
)
from spectraxgk.validation.benchmarks.scan import (
    ScanFitWindowPolicy,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    indexed_scan_value,
    normalize_fit_signal,
    normalize_solver_key,
    scan_window_valid,
)
from spectraxgk.config import KBMBaseCase, TimeConfig, resolve_cfl_fac
from spectraxgk.solvers.time.diffrax import (
    integrate_linear_diffrax_streaming,
)
from spectraxgk.geometry import (
    SAlphaGeometry,
    apply_geometry_grid_defaults,
    build_flux_tube_geometry,
)
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.solvers.time.explicit import (
    ExplicitTimeConfig,
    integrate_linear_explicit_diagnostics,
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


from spectraxgk.validation.benchmarks.kbm_beta import run_kbm_beta_scan

def run_kbm_scan(
    ky_values: np.ndarray,
    *,
    beta_value: float | None = None,
    Nl: int = 6,
    Nm: int = 12,
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
    method: str = "imex2",
    cfg: KBMBaseCase | None = None,
    time_cfg: TimeConfig | None = None,
    solver: str = "auto",
    krylov_cfg: KrylovConfig | None = None,
    kbm_target_factors: Sequence[float] | None = (0.7, 1.5),
    kbm_beta_transition: float | None = None,
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
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
    fit_signal: str = "auto",
    ky_batch: int = 4,
    fixed_batch_shape: bool = True,
    streaming_fit: bool = True,
    streaming_amp_floor: float = 1.0e-30,
    init_species_index: int = 1,
    density_species_index: int = 1,
    diagnostic_norm: str = "none",
    fapar_override: float | None = None,
    apar_beta_scale: float | None = None,
    ampere_g0_scale: float | None = None,
    bpar_beta_scale: float | None = None,
    reference_aligned: bool | None = True,
    gx_reference: bool | None = None,
) -> LinearScanResult:
    """Run a KBM ky scan at fixed beta.

    This is a thin wrapper over :func:`run_kbm_beta_scan` used for
    reference-comparison workflows where the external benchmark is a ky scan
    at fixed beta.
    """

    cfg_in = cfg or KBMBaseCase()
    if beta_value is None:
        beta_use = float(cfg_in.model.beta)
    else:
        beta_use = float(beta_value)
    cfg_use = replace(cfg_in, model=replace(cfg_in.model, beta=beta_use))

    ky_vals = np.asarray(ky_values, dtype=float)
    gamma_out: list[float] = []
    omega_out: list[float] = []
    ky_out: list[float] = []

    for i, ky_val in enumerate(ky_vals):
        dt_i = indexed_scan_value(dt, i)
        steps_i = indexed_scan_value(steps, i)
        if dt_i is None:
            dt_i = dt
        if steps_i is None:
            steps_i = steps
        out = run_kbm_beta_scan(
            betas=np.asarray([beta_use], dtype=float),
            ky_target=float(ky_val),
            Nl=Nl,
            Nm=Nm,
            dt=float(dt_i),
            steps=int(steps_i),
            method=method,
            cfg=cfg_use,
            time_cfg=time_cfg,
            solver=solver,
            krylov_cfg=krylov_cfg,
            kbm_target_factors=kbm_target_factors,
            kbm_beta_transition=kbm_beta_transition,
            tmin=indexed_scan_value(tmin, i),
            tmax=indexed_scan_value(tmax, i),
            auto_window=auto_window,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
            mode_method=mode_method,
            mode_only=mode_only,
            terms=terms,
            sample_stride=sample_stride,
            fit_signal=fit_signal,
            ky_batch=ky_batch,
            fixed_batch_shape=fixed_batch_shape,
            streaming_fit=streaming_fit,
            streaming_amp_floor=streaming_amp_floor,
            init_species_index=init_species_index,
            density_species_index=density_species_index,
            diagnostic_norm=diagnostic_norm,
            fapar_override=fapar_override,
            apar_beta_scale=apar_beta_scale,
            ampere_g0_scale=ampere_g0_scale,
            bpar_beta_scale=bpar_beta_scale,
            reference_aligned=reference_aligned,
            gx_reference=gx_reference,
        )
        ky_out.append(float(ky_val))
        gamma_out.append(float(out.gamma[0]))
        omega_out.append(float(out.omega[0]))

    return LinearScanResult(
        ky=np.asarray(ky_out, dtype=float),
        gamma=np.asarray(gamma_out, dtype=float),
        omega=np.asarray(omega_out, dtype=float),
    )


__all__ = ["run_kbm_scan"]
