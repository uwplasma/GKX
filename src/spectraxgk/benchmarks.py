"""Benchmark utilities for documented SPECTRAX-GK comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Sequence

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from spectraxgk.artifacts.restart import write_netcdf_restart_state
from spectraxgk.core.grid import SpectralGrid, build_spectral_grid, select_ky_grid
from spectraxgk.core.species import Species, build_linear_params
from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    ModeSelectionBatch,
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
    windowed_growth_rate_from_omega_series,
)
import spectraxgk.diagnostics.validation_gates as _gate_metrics
import spectraxgk.diagnostics.zonal_validation as _zonal_validation
from spectraxgk.diagnostics.modes import (
    compare_eigenfunctions,
    load_eigenfunction_reference_bundle,
    normalize_eigenfunction,
    phase_align_eigenfunction,
    save_eigenfunction_reference_bundle,
)
from spectraxgk.diagnostics.validation_gates import (
    _analytic_signal,
    _explicit_time_window,
    _leading_window,
    BranchContinuationMetrics,
    DiagnosticTimeSeries,
    EigenfunctionComparisonMetrics,
    EigenfunctionReferenceBundle,
    GateReport,
    LateTimeLinearMetrics,
    NonlinearHeatFluxConvergenceMetrics,
    NonlinearWindowMetrics,
    ObservedOrderMetrics,
    ScalarGateResult,
    ZonalFlowResponseMetrics,
    branch_continuity_gate_report,
    eigenfunction_gate_report,
    evaluate_scalar_gate,
    gate_report,
    gate_report_to_dict,
    infer_triple_dealiased_ny,
    late_time_window,
    linear_metrics_gate_report,
    load_diagnostic_time_series,
    nonlinear_heat_flux_convergence_gate_report,
    nonlinear_window_gate_report,
    observed_order_gate_report,
    zonal_response_gate_report,
)
from spectraxgk.geometry import (
    FluxTubeGeometryLike,
    SAlphaGeometry,
    apply_geometry_grid_defaults,
    build_flux_tube_geometry,
)
from spectraxgk.linear import integrate_linear, integrate_linear_diagnostics
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.diagnostics.normalization import (
    CYCLONE_NORMALIZATION,
    ETG_NORMALIZATION,
    KBM_NORMALIZATION,
    KINETIC_NORMALIZATION,
    TEM_NORMALIZATION,
)
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from spectraxgk.runtime import run_runtime_linear, run_runtime_nonlinear
from spectraxgk.solvers.linear.krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.solvers.time.diffrax import (
    integrate_linear_diffrax,
    integrate_linear_diffrax_streaming,
)
from spectraxgk.solvers.time.explicit import (
    ExplicitTimeConfig,
    integrate_linear_explicit,
    integrate_linear_explicit_diagnostics,
)
from spectraxgk.solvers.time.runners import integrate_linear_from_config
from spectraxgk.terms.assembly import compute_fields_cached
from spectraxgk.workflows.runtime.config import RuntimeConfig, RuntimeExpertConfig

from spectraxgk.config import (
    CycloneBaseCase,
    InitializationConfig,
    ETGBaseCase,
    KBMBaseCase,
    KineticElectronBaseCase,
    TEMBaseCase,
    TimeConfig,
    resolve_cfl_fac,
)
from spectraxgk.diagnostics.growth_rates import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _score_fit_signal_auto,
    _select_fit_signal,
    _select_fit_signal_auto,
)


# Benchmark policy defaults and Cyclone benchmark implementations live with the
# public benchmark facade after retiring the installable validation package.
VALID_FIT_SIGNALS = frozenset({"phi", "density", "auto"})

def _is_array_like(value: Any) -> bool:
    """Return whether a scan option is an indexed per-ky value."""

    return isinstance(value, (list, tuple, np.ndarray))

def _iter_ky_batches(
    ky_values: np.ndarray,
    *,
    ky_batch: int,
    fixed_batch_shape: bool,
):
    """Yield ky batches with optional edge padding for fixed-shape compilation."""

    n = int(len(ky_values))
    if ky_batch <= 1:
        for idx in range(n):
            ky = float(ky_values[idx])
            yield idx, np.asarray([ky], dtype=float), 1
        return
    for start in range(0, n, ky_batch):
        raw = np.asarray(ky_values[start : start + ky_batch], dtype=float)
        valid = int(raw.size)
        if valid == 0:
            continue
        if fixed_batch_shape and valid < ky_batch:
            pad = np.full((ky_batch - valid,), raw[-1], dtype=float)
            batch = np.concatenate([raw, pad], axis=0)
        else:
            batch = raw
        yield start, batch, valid

def _resolve_streaming_window(
    t_total: float,
    tmin: float | None,
    tmax: float | None,
    start_fraction: float,
    window_fraction: float,
    end_fraction: float,
) -> tuple[float, float]:
    """Resolve the sampled time window used for streaming linear fits."""

    if tmin is not None and tmax is not None:
        return float(tmin), float(tmax)
    t_start = float(start_fraction) * t_total
    t_end = float(end_fraction) * t_total
    t_end = min(t_end, t_start + float(window_fraction) * t_total)
    if t_end <= t_start:
        t_end = t_total
    return t_start, t_end

def normalize_solver_key(solver: str) -> str:
    """Normalize a benchmark solver selector to canonical SPECTRAX-GK keys."""

    return solver.strip().lower().replace("-", "_")

def normalize_fit_signal(fit_signal: str) -> str:
    """Normalize and validate benchmark fit-signal selectors."""

    fit_key = fit_signal.strip().lower()
    if fit_key not in VALID_FIT_SIGNALS:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    return fit_key

def apply_auto_fit_scan_policy(
    fit_key: str, *, streaming_fit: bool, mode_only: bool
) -> tuple[bool, bool]:
    """Disable streaming and mode-only saves when auto signal selection needs both fields."""

    if fit_key == "auto":
        return False, False
    return streaming_fit, mode_only

def resolve_scan_mode_method(mode_method: str, *, mode_only: bool) -> str:
    """Use direct mode extraction when a runner saved only a mode time series."""

    if mode_only and mode_method not in {"z_index", "max"}:
        return "z_index"
    return mode_method

def indexed_float_value(value: Any, idx: int) -> float | None:
    """Return a scalar or indexed scan value as ``float`` for window policies."""

    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[idx])
    return float(value)

def indexed_scan_value(value: Any, idx: int) -> Any:
    """Return a scalar or indexed scan value while preserving non-float types."""

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value[idx].item()
    if isinstance(value, (list, tuple)):
        return value[idx]
    return value

def scan_window_valid(
    t: np.ndarray, tmin: float | None, tmax: float | None, *, min_points: int = 2
) -> bool:
    """Return whether an explicit fit window contains enough sampled points."""

    if tmin is None or tmax is None:
        return False
    mask = (t >= tmin) & (t <= tmax)
    return int(np.count_nonzero(mask)) >= int(min_points)

def should_use_ky_batch(
    *,
    ky_batch: int,
    solver_key: str,
    dt: Any,
    steps: Any,
    tmin: Any,
    tmax: Any,
) -> bool:
    """Return whether a ky scan can use a fixed-shape batch path."""

    if ky_batch < 1:
        raise ValueError("ky_batch must be >= 1")
    return (
        ky_batch > 1
        and solver_key != "krylov"
        and not _is_array_like(dt)
        and not _is_array_like(steps)
        and not _is_array_like(tmin)
        and not _is_array_like(tmax)
    )

@dataclass(frozen=True)
class ScanFitWindowPolicy:
    """Window-selection and normalization policy shared by benchmark scans."""

    tmin: Any = None
    tmax: Any = None
    auto_window: bool = True
    window_fraction: float = 0.3
    min_points: int = 20
    start_fraction: float = 0.0
    growth_weight: float = 0.0
    require_positive: bool = False
    min_amp_fraction: float = 0.0
    max_fraction: float = 0.8
    end_fraction: float = 0.9
    max_amp_fraction: float = 0.9
    phase_weight: float = 0.2
    length_weight: float = 0.05
    min_r2: float = 0.0
    late_penalty: float = 0.1
    min_slope: float | None = None
    min_slope_frac: float = 0.0
    slope_var_weight: float = 0.0
    window_method: str = "loglinear"
    fit_growth_rate_fn: Callable[..., tuple[float, float]] = fit_growth_rate
    fit_growth_rate_auto_fn: Callable[..., tuple[float, float, float, float]] = (
        fit_growth_rate_auto
    )
    normalize_growth_rate_fn: Callable[
        [float, float, LinearParams, str], tuple[float, float]
    ] = _normalize_growth_rate

    def window_at(self, idx: int) -> tuple[float | None, float | None]:
        return indexed_float_value(self.tmin, idx), indexed_float_value(self.tmax, idx)

    def use_auto_window(self, t: np.ndarray, idx: int) -> tuple[bool, float | None, float | None]:
        tmin_i, tmax_i = self.window_at(idx)
        use_auto = self.auto_window and tmin_i is None and tmax_i is None
        if not use_auto and not scan_window_valid(t, tmin_i, tmax_i):
            use_auto = True
        return use_auto, tmin_i, tmax_i

    def auto_kwargs(self) -> dict[str, Any]:
        return {
            "window_fraction": self.window_fraction,
            "min_points": self.min_points,
            "start_fraction": self.start_fraction,
            "growth_weight": self.growth_weight,
            "require_positive": self.require_positive,
            "min_amp_fraction": self.min_amp_fraction,
            "max_fraction": self.max_fraction,
            "end_fraction": self.end_fraction,
            "max_amp_fraction": self.max_amp_fraction,
            "phase_weight": self.phase_weight,
            "length_weight": self.length_weight,
            "min_r2": self.min_r2,
            "late_penalty": self.late_penalty,
            "min_slope": self.min_slope,
            "min_slope_frac": self.min_slope_frac,
            "slope_var_weight": self.slope_var_weight,
            "window_method": self.window_method,
        }

    def fit_signal(
        self,
        signal: np.ndarray,
        *,
        idx: int,
        dt: float,
        stride: int,
        params: LinearParams,
        diagnostic_norm: str,
    ) -> tuple[float, float]:
        """Fit one scan signal and apply the configured diagnostic normalization."""

        t = np.arange(signal.shape[0]) * float(dt) * int(stride)
        use_auto, tmin_i, tmax_i = self.use_auto_window(t, idx)
        if use_auto:
            gamma, omega, _tmin, _tmax = self.fit_growth_rate_auto_fn(
                t,
                signal,
                **self.auto_kwargs(),
            )
        else:
            try:
                gamma, omega = self.fit_growth_rate_fn(
                    t, signal, tmin=tmin_i, tmax=tmax_i
                )
            except ValueError:
                gamma, omega, _tmin, _tmax = self.fit_growth_rate_auto_fn(
                    t,
                    signal,
                    **self.auto_kwargs(),
                )
        return self.normalize_growth_rate_fn(gamma, omega, params, diagnostic_norm)

CYCLONE_OMEGA_D_SCALE = CYCLONE_NORMALIZATION.omega_d_scale

CYCLONE_OMEGA_STAR_SCALE = CYCLONE_NORMALIZATION.omega_star_scale

CYCLONE_RHO_STAR = CYCLONE_NORMALIZATION.rho_star

ETG_OMEGA_D_SCALE = ETG_NORMALIZATION.omega_d_scale

ETG_OMEGA_STAR_SCALE = ETG_NORMALIZATION.omega_star_scale

ETG_RHO_STAR = ETG_NORMALIZATION.rho_star

KINETIC_OMEGA_D_SCALE = KINETIC_NORMALIZATION.omega_d_scale

KINETIC_OMEGA_STAR_SCALE = KINETIC_NORMALIZATION.omega_star_scale

KINETIC_RHO_STAR = KINETIC_NORMALIZATION.rho_star

TEM_OMEGA_D_SCALE = TEM_NORMALIZATION.omega_d_scale

TEM_OMEGA_STAR_SCALE = TEM_NORMALIZATION.omega_star_scale

TEM_RHO_STAR = TEM_NORMALIZATION.rho_star

KBM_OMEGA_D_SCALE = KBM_NORMALIZATION.omega_d_scale

KBM_OMEGA_STAR_SCALE = KBM_NORMALIZATION.omega_star_scale

KBM_RHO_STAR = KBM_NORMALIZATION.rho_star

REFERENCE_NU_HYPER_L = 0.0

REFERENCE_NU_HYPER_M = 1.0

REFERENCE_P_HYPER_L = 6.0

REFERENCE_P_HYPER_M = 20.0

REFERENCE_DAMP_ENDS_AMP = 0.1

REFERENCE_DAMP_ENDS_WIDTHFRAC = 1.0 / 8.0

def _reference_hypercollision_power(nhermite: int | None) -> float:
    if nhermite is None:
        return REFERENCE_P_HYPER_M
    return float(min(REFERENCE_P_HYPER_M, max(int(nhermite) // 2, 1)))

def _apply_reference_hypercollisions(
    params: LinearParams, *, nhermite: int | None = None
) -> LinearParams:
    return replace(
        params,
        nu_hyper=0.0,
        nu_hyper_l=REFERENCE_NU_HYPER_L,
        nu_hyper_m=REFERENCE_NU_HYPER_M,
        p_hyper_l=REFERENCE_P_HYPER_L,
        p_hyper_m=_reference_hypercollision_power(nhermite),
        hypercollisions_const=0.0,
        hypercollisions_kz=1.0,
    )

def _linked_boundary_end_damping(reference_aligned: bool) -> tuple[float, float]:
    if reference_aligned:
        return REFERENCE_DAMP_ENDS_AMP, REFERENCE_DAMP_ENDS_WIDTHFRAC
    return 0.0, 0.0

def _two_species_params(
    model,
    *,
    kpar_scale: float,
    omega_d_scale: float,
    omega_star_scale: float,
    rho_star: float,
    beta_override: float | None = None,
    fapar_override: float | None = None,
    apar_beta_scale: float | None = None,
    ampere_g0_scale: float | None = None,
    bpar_beta_scale: float | None = None,
    damp_ends_amp: float | None = None,
    damp_ends_widthfrac: float | None = None,
    nhermite: int | None = None,
) -> LinearParams:
    """Build ``LinearParams`` for a two-species kinetic model."""

    mass_ratio = float(model.mass_ratio)
    if mass_ratio <= 0.0:
        raise ValueError("mass_ratio must be > 0")
    Te_over_Ti = float(model.Te_over_Ti)
    if Te_over_Ti <= 0.0:
        raise ValueError("Te_over_Ti must be > 0")
    ion_fprim_raw = getattr(model, "R_over_Lni", None)
    ele_fprim_raw = getattr(model, "R_over_Lne", None)
    ion_fprim = (
        float(model.R_over_Ln) if ion_fprim_raw is None else float(ion_fprim_raw)
    )
    ele_fprim = (
        float(model.R_over_Ln) if ele_fprim_raw is None else float(ele_fprim_raw)
    )

    nu_i = float(getattr(model, "nu_i", 0.0))
    nu_e = float(getattr(model, "nu_e", 0.0))
    beta = float(getattr(model, "beta", 1.0e-5))
    if beta_override is not None:
        beta = float(beta_override)

    ion = Species(
        charge=1.0,
        mass=1.0,
        density=1.0,
        temperature=1.0,
        tprim=float(getattr(model, "R_over_LTi", model.R_over_LTe)),
        fprim=ion_fprim,
        nu=nu_i,
    )
    electron = Species(
        charge=-1.0,
        mass=1.0 / mass_ratio,
        density=1.0,
        temperature=Te_over_Ti,
        tprim=float(model.R_over_LTe),
        fprim=ele_fprim,
        nu=nu_e,
    )
    params = build_linear_params(
        [ion, electron],
        tau_e=0.0,
        kpar_scale=kpar_scale,
        omega_d_scale=omega_d_scale,
        omega_star_scale=omega_star_scale,
        rho_star=rho_star,
        beta=beta,
        fapar=1.0 if beta > 0.0 else 0.0,
        apar_beta_scale=0.5 if apar_beta_scale is None else float(apar_beta_scale),
        ampere_g0_scale=0.5 if ampere_g0_scale is None else float(ampere_g0_scale),
        bpar_beta_scale=0.5 if bpar_beta_scale is None else float(bpar_beta_scale),
    )
    params = _apply_reference_hypercollisions(params, nhermite=nhermite)
    if fapar_override is not None:
        params = replace(params, fapar=float(fapar_override))
    if damp_ends_amp is not None:
        params = replace(params, damp_ends_amp=float(damp_ends_amp))
    if damp_ends_widthfrac is not None:
        params = replace(params, damp_ends_widthfrac=float(damp_ends_widthfrac))
    return params

def _electron_only_params(
    model,
    *,
    kpar_scale: float,
    omega_d_scale: float,
    omega_star_scale: float,
    rho_star: float,
    beta_override: float | None = None,
    fapar_override: float | None = None,
    apar_beta_scale: float | None = None,
    ampere_g0_scale: float | None = None,
    bpar_beta_scale: float | None = None,
    damp_ends_amp: float | None = None,
    damp_ends_widthfrac: float | None = None,
    nhermite: int | None = None,
) -> LinearParams:
    """Build ``LinearParams`` for kinetic electrons with Boltzmann ions."""

    mass_ratio = float(model.mass_ratio)
    if mass_ratio <= 0.0:
        raise ValueError("mass_ratio must be > 0")
    Te_over_Ti = float(model.Te_over_Ti)
    if Te_over_Ti <= 0.0:
        raise ValueError("Te_over_Ti must be > 0")

    nu_e = float(getattr(model, "nu_e", 0.0))
    beta = float(getattr(model, "beta", 1.0e-5))
    if beta_override is not None:
        beta = float(beta_override)

    electron = Species(
        charge=-1.0,
        mass=1.0 / mass_ratio,
        density=1.0,
        temperature=Te_over_Ti,
        tprim=float(model.R_over_LTe),
        fprim=float(model.R_over_Ln),
        nu=nu_e,
    )
    params = build_linear_params(
        [electron],
        tau_e=Te_over_Ti,
        kpar_scale=kpar_scale,
        omega_d_scale=omega_d_scale,
        omega_star_scale=omega_star_scale,
        rho_star=rho_star,
        beta=beta,
        fapar=1.0 if beta > 0.0 else 0.0,
        apar_beta_scale=0.5 if apar_beta_scale is None else float(apar_beta_scale),
        ampere_g0_scale=0.5 if ampere_g0_scale is None else float(ampere_g0_scale),
        bpar_beta_scale=0.5 if bpar_beta_scale is None else float(bpar_beta_scale),
    )
    params = _apply_reference_hypercollisions(params, nhermite=nhermite)
    if fapar_override is not None:
        params = replace(params, fapar=float(fapar_override))
    if damp_ends_amp is not None:
        params = replace(params, damp_ends_amp=float(damp_ends_amp))
    if damp_ends_widthfrac is not None:
        params = replace(params, damp_ends_widthfrac=float(damp_ends_widthfrac))
    return params

KBM_EXPLICIT_SOLVER_LOCK: tuple[tuple[float, str], ...] = (
    (0.10, "explicit_time"),
    (0.30, "explicit_time"),
    (0.40, "explicit_time"),
)

KBM_EXPLICIT_SOLVER_LOCK_TOL = 0.03

def _midplane_index(grid: SpectralGrid) -> int:
    """Return reference midplane index for growth-rate diagnostics."""

    if grid.z.size <= 1:
        return 0
    idx = int(grid.z.size // 2 + 1)
    return min(idx, int(grid.z.size) - 1)

def select_kbm_solver_auto(
    solver: str,
    *,
    ky_target: float,
    reference_aligned: bool | None = None,
) -> str:
    """Return deterministic KBM solver choice for auto mode."""

    solver_key = solver.strip().lower()
    if solver_key != "auto":
        return solver_key
    if not bool(True if reference_aligned is None else reference_aligned):
        return "time"
    ky_abs = abs(float(ky_target))
    for ky_ref, solver_ref in KBM_EXPLICIT_SOLVER_LOCK:
        if abs(ky_abs - ky_ref) <= KBM_EXPLICIT_SOLVER_LOCK_TOL:
            return solver_ref
    return "explicit_time"

def _kbm_use_multi_target_krylov(
    kcfg: KrylovConfig,
    targets: Sequence[float] | None,
    *,
    shift: complex | None,
) -> bool:
    """Return whether KBM benchmark helpers should sweep target factors."""

    if targets is None:
        return False
    if kcfg.mode_family.strip().lower() != "kbm":
        return False
    if kcfg.method.strip().lower() != "shift_invert":
        return False
    if shift is not None:
        return False
    if kcfg.shift_selection.strip().lower() == "shift":
        return False
    return True

CYCLONE_KRYLOV_DEFAULT = KrylovConfig(
    method="shift_invert",
    krylov_dim=16,
    restarts=1,
    omega_target_factor=0.3,
    power_iters=60,
    power_dt=0.001,
    shift_maxiter=30,
    shift_restart=10,
    shift_tol=1.0e-3,
    shift_preconditioner="hermite-line",
    omega_sign=1,
    mode_family="cyclone",
    fallback_method="propagator",
)

KINETIC_KRYLOV_DEFAULT = KrylovConfig(
    method="shift_invert",
    krylov_dim=16,
    restarts=1,
    omega_min_factor=0.05,
    omega_cap_factor=0.8,
    omega_target_factor=0.3,
    omega_sign=1,
    power_iters=60,
    power_dt=0.001,
    shift_source="target",
    shift_maxiter=40,
    shift_restart=12,
    shift_tol=5.0e-4,
    shift_preconditioner="hermite-line",
    mode_family="cyclone",
    fallback_method="propagator",
)

KINETIC_KRYLOV_REFERENCE_ALIGNED = replace(
    KINETIC_KRYLOV_DEFAULT, shift_source="history"
)

ETG_KRYLOV_DEFAULT = KrylovConfig(
    method="propagator",
    krylov_dim=16,
    restarts=1,
    omega_min_factor=0.0,
    omega_target_factor=0.3,
    omega_cap_factor=0.6,
    omega_sign=-1,
    power_iters=80,
    power_dt=0.002,
    shift_maxiter=40,
    shift_restart=12,
    shift_tol=2.0e-3,
    mode_family="etg",
    fallback_method="arnoldi",
    continuation=True,
    continuation_selection="overlap",
)

KBM_KRYLOV_DEFAULT = KrylovConfig(
    method="shift_invert",
    krylov_dim=16,
    restarts=1,
    omega_min_factor=0.0,
    omega_cap_factor=2.0,
    omega_target_factor=1.5,
    omega_sign=-1,
    power_iters=60,
    power_dt=0.005,
    shift_source="target",
    shift_maxiter=40,
    shift_restart=12,
    shift_tol=5.0e-4,
    shift_preconditioner="hermite-line",
    shift_selection="targeted",
    mode_family="kbm",
    fallback_method="propagator",
    continuation=False,
)

TEM_KRYLOV_DEFAULT = KrylovConfig(
    method="shift_invert",
    krylov_dim=16,
    restarts=1,
    omega_min_factor=0.05,
    omega_cap_factor=0.6,
    omega_target_factor=0.25,
    omega_sign=-1,
    power_iters=60,
    power_dt=0.005,
    shift_source="target",
    shift_maxiter=40,
    shift_restart=12,
    shift_tol=5.0e-4,
    shift_preconditioner="hermite-line",
    mode_family="tem",
    fallback_method="propagator",
)

@dataclass(frozen=True)
class CycloneReference:
    ky: np.ndarray
    omega: np.ndarray
    gamma: np.ndarray

@dataclass(frozen=True)
class CycloneRunResult:
    t: np.ndarray
    phi_t: np.ndarray
    gamma: float
    omega: float
    ky: float
    selection: ModeSelection

@dataclass(frozen=True)
class CycloneScanResult:
    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray

@dataclass(frozen=True)
class CycloneComparison:
    ky: float
    gamma: float
    omega: float
    gamma_ref: float
    omega_ref: float
    rel_gamma: float
    rel_omega: float

@dataclass(frozen=True)
class LinearRunResult:
    t: np.ndarray
    phi_t: np.ndarray
    gamma: float
    omega: float
    ky: float
    selection: ModeSelection
    gamma_t: np.ndarray | None = None
    omega_t: np.ndarray | None = None

@dataclass(frozen=True)
class LinearScanResult:
    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray

def _load_csv_reference(filename: str) -> CycloneReference:
    data_path = resources.files("spectraxgk").joinpath("data", filename)
    arr = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)

def load_cyclone_reference() -> CycloneReference:
    """Load Cyclone base case reference data (adiabatic electrons)."""

    return _load_csv_reference("cyclone_reference_adiabatic.csv")

def _load_reference_with_header(filename: str) -> CycloneReference:
    """Load reference CSVs with columns ky,gamma,omega."""

    data_path = resources.files("spectraxgk").joinpath("data", filename)
    arr = np.genfromtxt(str(data_path), delimiter=",", names=True, dtype=float)
    ky = np.atleast_1d(np.asarray(arr["ky"], dtype=float))
    gamma = np.atleast_1d(np.asarray(arr["gamma"], dtype=float))
    omega = np.atleast_1d(np.asarray(arr["omega"], dtype=float))
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)

def load_cyclone_reference_kinetic() -> CycloneReference:
    """Load Cyclone base case reference data (kinetic electrons)."""

    return _load_csv_reference("cyclone_reference_kinetic.csv")

def load_kbm_reference() -> CycloneReference:
    """Load KBM reference data (finite beta, kinetic electrons)."""

    return _load_csv_reference("kbm_reference.csv")

def load_etg_reference() -> CycloneReference:
    """Load ETG reference data for the tracked two-species ETG lane."""

    return _load_csv_reference("etg_reference.csv")

def load_tem_reference() -> CycloneReference:
    """Load the provisional TEM reference digitized from the literature.

    This lane remains an extended stress case while the literature case
    definition is being reconstructed.
    """

    return _load_csv_reference("tem_reference.csv")

def compare_cyclone_to_reference(
    result: CycloneRunResult, reference: CycloneReference
) -> CycloneComparison:
    """Compare a Cyclone run result against the reference data set."""

    idx = int(np.argmin(np.abs(reference.ky - result.ky)))
    gamma_ref = float(reference.gamma[idx])
    omega_ref = float(reference.omega[idx])
    rel_gamma = (result.gamma - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
    rel_omega = (result.omega - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
    return CycloneComparison(
        ky=float(reference.ky[idx]),
        gamma=result.gamma,
        omega=result.omega,
        gamma_ref=gamma_ref,
        omega_ref=omega_ref,
        rel_gamma=rel_gamma,
        rel_omega=rel_omega,
    )

def _build_gaussian_profile(
    z: np.ndarray,
    *,
    kx: float,
    ky: float,
    s_hat: float,
    init_cfg: InitializationConfig,
) -> np.ndarray:
    if ky == 0.0:
        return np.zeros_like(z)
    theta0 = kx / (s_hat * ky)
    envelope = (
        init_cfg.gaussian_envelope_constant
        + init_cfg.gaussian_envelope_sine * np.sin(z - theta0)
    )
    width = init_cfg.gaussian_width
    if width <= 0.0:
        raise ValueError("gaussian_width must be > 0")
    return envelope * np.exp(-(((z - theta0) / width) ** 2))

def _build_initial_condition(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    *,
    ky_index: int | Sequence[int] | np.ndarray,
    kx_index: int,
    Nl: int,
    Nm: int,
    init_cfg: InitializationConfig,
) -> jnp.ndarray:
    init_field = init_cfg.init_field.lower()
    field_map = {
        "density": (0, 0),
        "upar": (0, 1),
        "tpar": (0, 2),
        "tperp": (1, 0),
        "qpar": (0, 3),
        "qperp": (1, 1),
    }
    # Moment-normalized initializer amplitudes for init_field="all".
    all_scales = {
        "density": 1.0,
        "upar": 1.0,
        "tpar": 1.0 / np.sqrt(2.0),
        "tperp": 1.0,
        "qpar": 1.0 / np.sqrt(6.0),
        "qperp": 1.0,
    }
    if init_field != "all" and init_field not in field_map:
        raise ValueError(
            "init_field must be one of {'density','upar','tpar','tperp','qpar','qperp','all'}"
        )

    G0 = np.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    amp = float(init_cfg.init_amp)
    ky_idx = np.atleast_1d(np.asarray(ky_index, dtype=int))
    for ky_i in ky_idx:
        if init_cfg.gaussian_init:
            profile = _build_gaussian_profile(
                np.asarray(grid.z),
                kx=float(grid.kx[kx_index]),
                ky=float(grid.ky[ky_i]),
                s_hat=geom.s_hat,
                init_cfg=init_cfg,
            )
            init_vals = amp * profile * (1.0 + 1.0j)
        else:
            init_vals = amp * (1.0 + 1.0j) * np.ones_like(grid.z)
        if grid.ky[ky_i] != 0.0:
            if init_field == "all":
                for field_name, (l_idx, m_idx) in field_map.items():
                    if l_idx < Nl and m_idx < Nm:
                        scale = all_scales.get(field_name, 1.0)
                        G0[l_idx, m_idx, ky_i, kx_index, :] = init_vals * scale
            else:
                l_idx, m_idx = field_map[init_field]
                if l_idx >= Nl or m_idx >= Nm:
                    raise ValueError("init_field moment exceeds (Nl, Nm) resolution")
                G0[l_idx, m_idx, ky_i, kx_index, :] = init_vals
    return jnp.asarray(G0)

def _kinetic_reference_init_cfg(
    init_cfg: InitializationConfig,
    *,
    reference_aligned: bool | None = None,
) -> InitializationConfig:
    """Use the reference-aligned kinetic benchmark seed when requested.

    Reference-aligned kinetic runs seed a constant electron-density moment.
    Explicit user overrides are preserved by replacing only the exact current
    kinetic default initialization.
    """

    if not bool(True if reference_aligned is None else reference_aligned):
        return init_cfg
    kinetic_default_init = KineticElectronBaseCase().init
    if init_cfg != kinetic_default_init:
        return init_cfg
    return InitializationConfig(
        init_field="density",
        init_amp=1.0e-3,
        init_single=True,
        random_seed=kinetic_default_init.random_seed,
        gaussian_init=False,
        gaussian_width=kinetic_default_init.gaussian_width,
        gaussian_envelope_constant=kinetic_default_init.gaussian_envelope_constant,
        gaussian_envelope_sine=kinetic_default_init.gaussian_envelope_sine,
        kpar_init=kinetic_default_init.kpar_init,
        init_file=kinetic_default_init.init_file,
        init_file_scale=kinetic_default_init.init_file_scale,
        init_file_mode=kinetic_default_init.init_file_mode,
        init_electrons_only=kinetic_default_init.init_electrons_only,
    )

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

@dataclass(frozen=True)
class _CycloneTimeFitOptions:
    """Private growth-window policy for Cyclone saved-time fits."""

    fit_key: str
    mode_method: str
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
class _CycloneKrylovSeed:
    """Frequency seed extracted before the Cyclone Krylov solve."""

    gamma: float = 0.0
    omega: float = 0.0
    seed_ok: bool = False
    omega_ok: bool = False

@dataclass(frozen=True)
class _CycloneTimeTrace:
    """Saved field history and sampling stride for a Cyclone time run."""

    phi_t: Any
    density_t: Any | None
    stride: int

@dataclass(frozen=True)
class _CycloneTimePathControls:
    """Resolved runtime and fit controls for one Cyclone time path."""

    time_cfg: Any | None
    fit_options: _CycloneTimeFitOptions

@dataclass(frozen=True)
class _CycloneTimePathRequest:
    """Public Cyclone time-path inputs packed for private solver routing."""

    grid: Any
    cache: Any
    params: Any
    geom: Any
    terms: Any
    cfg: Any
    time_cfg: Any
    sel: Any
    dt: float
    steps: int
    method: str
    sample_stride: int | None
    fit_key: str
    need_density: bool
    reference_aligned: bool
    use_jit: bool
    diagnostic_norm: str
    show_progress: bool
    status: Callable[[str], None]
    fresh_G0: Callable[[], jnp.ndarray]
    mode_method: str
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

def _cyclone_time_path_request_from_locals(values: dict[str, Any]) -> _CycloneTimePathRequest:
    return _CycloneTimePathRequest(
        **{field.name: values[field.name] for field in fields(_CycloneTimePathRequest)}
    )

def _fit_cyclone_explicit_seed(
    *,
    state: Any,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    kcfg: Any,
    selection: ModeSelection,
    show_progress: bool,
) -> _CycloneKrylovSeed:
    """Estimate a Cyclone growth/frequency seed with a short explicit march."""

    t_seed = min(150.0, float(kcfg.power_dt) * 15000.0)
    time_cfg = ExplicitTimeConfig(
        dt=float(kcfg.power_dt),
        t_max=t_seed,
        sample_stride=1,
        fixed_dt=True,
    )
    t_short, phi_t, _g_t, _o_t = integrate_linear_explicit(
        state,
        grid,
        cache,
        params,
        geom,
        time_cfg,
        terms=terms,
        mode_method="z_index",
        show_progress=show_progress,
    )
    gamma_seed, omega_seed, _g, _o, _t_mid = instantaneous_growth_rate_from_phi(
        phi_t,
        t_short,
        selection,
        navg_fraction=0.5,
        mode_method="z_index",
    )
    omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
    seed_ok = omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
    return _CycloneKrylovSeed(
        gamma=float(gamma_seed),
        omega=float(omega_seed),
        seed_ok=bool(seed_ok),
        omega_ok=bool(omega_ok),
    )

def _estimate_cyclone_primary_seed(
    *,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    kcfg: Any,
    selection: ModeSelection,
    show_progress: bool,
    fresh_G0: Callable[[], jnp.ndarray],
) -> _CycloneKrylovSeed:
    """Try the full-resolution explicit seed and preserve silent fallback."""

    try:
        return _fit_cyclone_explicit_seed(
            state=fresh_G0(),
            grid=grid,
            cache=cache,
            params=params,
            geom=geom,
            terms=terms,
            kcfg=kcfg,
            selection=selection,
            show_progress=show_progress,
        )
    except Exception:
        return _CycloneKrylovSeed()

def _estimate_cyclone_reduced_seed(
    *,
    grid: Any,
    params: Any,
    geom: Any,
    terms: Any,
    Nl: int,
    Nm: int,
    init_cfg: Any,
    kcfg: Any,
    show_progress: bool,
) -> _CycloneKrylovSeed:
    """Try the reduced Hermite-Laguerre explicit seed and preserve fallback."""

    try:
        Nl_seed = min(Nl, 16)
        Nm_seed = min(Nm, 12)
        cache_seed = build_linear_cache(grid, geom, params, Nl_seed, Nm_seed)
        G0_seed = _build_initial_condition(
            grid,
            geom,
            ky_index=0,
            kx_index=0,
            Nl=Nl_seed,
            Nm=Nm_seed,
            init_cfg=init_cfg,
        )
        selection = ModeSelection(
            ky_index=0,
            kx_index=0,
            z_index=_midplane_index(grid),
        )
        return _fit_cyclone_explicit_seed(
            state=jnp.asarray(np.asarray(G0_seed)),
            grid=grid,
            cache=cache_seed,
            params=params,
            geom=geom,
            terms=terms,
            kcfg=kcfg,
            selection=selection,
            show_progress=show_progress,
        )
    except Exception:
        return _CycloneKrylovSeed()

def _estimate_cyclone_krylov_seed(
    *,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    Nl: int,
    Nm: int,
    init_cfg: Any,
    kcfg: Any,
    selection: ModeSelection,
    show_progress: bool,
    status: Callable[[str], None],
    fresh_G0: Callable[[], jnp.ndarray],
) -> _CycloneKrylovSeed:
    """Run the Cyclone primary then reduced seed ladder when no shift is given."""

    if kcfg.shift is not None:
        return _CycloneKrylovSeed()
    status("estimating frequency seed with short explicit time march")
    seed = _estimate_cyclone_primary_seed(
        grid=grid,
        cache=cache,
        params=params,
        geom=geom,
        terms=terms,
        kcfg=kcfg,
        selection=selection,
        show_progress=show_progress,
        fresh_G0=fresh_G0,
    )
    if seed.seed_ok:
        return seed
    status("primary seed failed; retrying reduced Hermite-Laguerre seed")
    return _estimate_cyclone_reduced_seed(
        grid=grid,
        params=params,
        geom=geom,
        terms=terms,
        Nl=Nl,
        Nm=Nm,
        init_cfg=init_cfg,
        kcfg=kcfg,
        show_progress=show_progress,
    )

def _cyclone_krylov_shift(seed: _CycloneKrylovSeed) -> complex | None:
    """Convert a valid frequency seed into the shifted-eigenvalue target."""

    if not seed.omega_ok:
        return None
    return complex(float(seed.gamma) if seed.seed_ok else 0.0, float(-seed.omega))

def _solve_cyclone_dominant_pair(
    *,
    grid: Any,
    cache: Any,
    params: Any,
    terms: Any,
    kcfg: Any,
    shift: complex | None,
    status: Callable[[str], None],
    fresh_G0: Callable[[], jnp.ndarray],
) -> tuple[Any, Any]:
    """Call the Cyclone dominant-eigenpair solver with one option policy."""

    status("running dominant eigenpair solve")
    return dominant_eigenpair(
        fresh_G0(),
        cache,
        params,
        terms=terms,
        krylov_dim=kcfg.krylov_dim,
        restarts=kcfg.restarts,
        omega_min_factor=kcfg.omega_min_factor,
        omega_target_factor=kcfg.omega_target_factor,
        omega_cap_factor=kcfg.omega_cap_factor,
        omega_sign=kcfg.omega_sign,
        method=kcfg.method,
        power_iters=kcfg.power_iters,
        power_dt=kcfg.power_dt,
        shift=shift if shift is not None else kcfg.shift,
        shift_source=kcfg.shift_source,
        shift_tol=kcfg.shift_tol,
        shift_maxiter=kcfg.shift_maxiter,
        shift_restart=kcfg.shift_restart,
        shift_solve_method=kcfg.shift_solve_method,
        shift_preconditioner=kcfg.shift_preconditioner,
        shift_selection=kcfg.shift_selection,
        mode_family=kcfg.mode_family,
        fallback_method=kcfg.fallback_method,
        fallback_real_floor=kcfg.fallback_real_floor,
        status_callback=status,
    )

def _apply_cyclone_seed_branch_guard(
    *,
    gamma: float,
    omega: float,
    seed: _CycloneKrylovSeed,
) -> tuple[float, float]:
    """Prefer a strong explicit seed when Krylov lands on an inconsistent branch."""

    if not seed.seed_ok:
        return gamma, omega
    seed_strong = (seed.gamma > 0.0) and (abs(seed.omega) > 1.0e-6)
    if not seed_strong:
        return gamma, omega
    omega_tol = 0.15 * max(abs(seed.omega), 1.0e-6)
    gamma_tol = 0.15 * max(abs(seed.gamma), 1.0e-6)
    use_seed = (
        not np.isfinite(gamma)
        or not np.isfinite(omega)
        or (seed.gamma > 0.0 and gamma < 0.0)
        or abs(omega - seed.omega) > omega_tol
        or abs(gamma - seed.gamma) > gamma_tol
    )
    if not use_seed:
        return gamma, omega
    return float(seed.gamma), float(seed.omega)

def _pack_cyclone_krylov_result(
    *,
    eig: Any,
    vec: Any,
    cache: Any,
    params: Any,
    terms: Any,
    kcfg: Any,
    seed: _CycloneKrylovSeed,
    diagnostic_norm: str,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute fields, guard branch selection, normalize, and pack Krylov output."""

    term_cfg = linear_terms_to_term_config(terms)
    phi = compute_fields_cached(vec, cache, params, terms=term_cfg).phi
    phi_t_out = np.asarray(phi)[None, ...]
    t_out = np.array([0.0])
    gamma_out = float(np.real(eig))
    omega_out = float(-np.imag(eig))
    gamma_out, omega_out = _apply_cyclone_seed_branch_guard(
        gamma=gamma_out,
        omega=omega_out,
        seed=seed,
    )
    if kcfg.omega_sign != 0:
        omega_out = float(np.sign(kcfg.omega_sign)) * abs(omega_out)
    gamma_out, omega_out = _normalize_growth_rate(
        gamma_out,
        omega_out,
        params,
        diagnostic_norm,
    )
    return gamma_out, omega_out, phi_t_out, t_out

def run_cyclone_krylov_path(
    *,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    Nl: int,
    Nm: int,
    init_cfg: Any,
    krylov_cfg: Any,
    diagnostic_norm: str,
    show_progress: bool,
    status: Callable[[str], None],
    fresh_G0: Callable[[], jnp.ndarray],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run the Cyclone Krylov branch with the explicit seed policy."""

    status("starting Krylov solve")
    kcfg = krylov_cfg or CYCLONE_KRYLOV_DEFAULT
    selection = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    seed = _estimate_cyclone_krylov_seed(
        grid=grid,
        cache=cache,
        params=params,
        geom=geom,
        terms=terms,
        Nl=Nl,
        Nm=Nm,
        init_cfg=init_cfg,
        kcfg=kcfg,
        selection=selection,
        show_progress=show_progress,
        status=status,
        fresh_G0=fresh_G0,
    )
    eig, vec = _solve_cyclone_dominant_pair(
        grid=grid,
        cache=cache,
        params=params,
        terms=terms,
        kcfg=kcfg,
        shift=_cyclone_krylov_shift(seed),
        status=status,
        fresh_G0=fresh_G0,
    )
    gamma_out, omega_out, phi_t_out, t_out = _pack_cyclone_krylov_result(
        eig=eig,
        vec=vec,
        cache=cache,
        params=params,
        terms=terms,
        kcfg=kcfg,
        seed=seed,
        diagnostic_norm=diagnostic_norm,
    )
    status(f"Krylov solve complete: gamma={gamma_out:.6f} omega={omega_out:.6f}")
    return gamma_out, omega_out, phi_t_out, t_out

def _resolve_cyclone_time_config(
    *,
    cfg: Any,
    time_cfg: Any,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
) -> Any | None:
    """Resolve a runtime time config using the existing Cyclone branch policy."""

    method_key = method.lower()
    time_cfg_use = None
    if time_cfg is not None:
        time_cfg_use = replace(time_cfg, dt=float(dt), t_max=float(dt) * int(steps))
    elif cfg.time.use_diffrax and not (
        method_key.startswith("imex") or method_key.startswith("implicit")
    ):
        time_cfg_use = replace(cfg.time, dt=float(dt), t_max=float(dt) * int(steps))
    if time_cfg_use is not None and sample_stride is not None:
        time_cfg_use = replace(time_cfg_use, sample_stride=sample_stride)
    return time_cfg_use

def _run_cyclone_reference_aligned_time(
    *,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    time_cfg_use: Any,
    dt: float,
    steps: int,
    sample_stride: int | None,
    diagnostic_norm: str,
    show_progress: bool,
    fresh_G0: Callable[[], jnp.ndarray],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run the reference-aligned explicit Cyclone time path."""

    t_max_val = float(dt) * int(steps) if time_cfg_use is None else float(time_cfg_use.t_max)
    stride = (
        int(sample_stride)
        if sample_stride is not None
        else (1 if time_cfg_use is None else int(time_cfg_use.sample_stride))
    )
    explicit_time_cfg = ExplicitTimeConfig(
        dt=float(dt),
        t_max=t_max_val,
        sample_stride=stride,
        fixed_dt=True,
    )
    t, phi_ref, _g_t, _o_t = integrate_linear_explicit(
        fresh_G0(),
        grid,
        cache,
        params,
        geom,
        explicit_time_cfg,
        terms=terms,
        mode_method="z_index",
        show_progress=show_progress,
    )
    sel_local = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    gamma, omega, _g, _o, _t_mid = instantaneous_growth_rate_from_phi(
        phi_ref,
        t,
        sel_local,
        navg_fraction=0.5,
        mode_method="z_index",
    )
    gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    return gamma, omega, np.asarray(phi_ref), np.asarray(t)

def _integrate_cyclone_configured_time(
    *,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    time_cfg_use: Any,
    need_density: bool,
    show_progress: bool,
    fresh_G0: Callable[[], jnp.ndarray],
) -> _CycloneTimeTrace:
    """Integrate Cyclone with an explicit or synthesized runtime config."""

    if need_density:
        _, saved = integrate_linear_from_config(
            fresh_G0(),
            grid,
            geom,
            params,
            time_cfg_use,
            terms=terms,
            save_field="phi+density",
            density_species_index=0,
            show_progress=show_progress,
        )
        phi_t, density_t = saved
        return _CycloneTimeTrace(phi_t, density_t, int(time_cfg_use.sample_stride))
    _, phi_t = integrate_linear_from_config(
        fresh_G0(),
        grid,
        geom,
        params,
        time_cfg_use,
        terms=terms,
        show_progress=show_progress,
    )
    return _CycloneTimeTrace(phi_t, None, int(time_cfg_use.sample_stride))

def _integrate_cyclone_unconfigured_time(
    *,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    need_density: bool,
    use_jit: bool,
    show_progress: bool,
    fresh_G0: Callable[[], jnp.ndarray],
) -> _CycloneTimeTrace:
    """Integrate Cyclone with the fixed-step path selected by diagnostics needs."""

    stride = 1 if sample_stride is None else int(sample_stride)
    if need_density or not use_jit:
        diag = integrate_linear_diagnostics(
            fresh_G0(),
            grid,
            geom,
            params,
            dt=dt,
            steps=steps,
            method=method,
            terms=terms,
            sample_stride=stride,
            species_index=0,
            record_hl_energy=False,
            show_progress=show_progress,
        )
        phi_t = diag[1]
        density_t = diag[2] if len(diag) > 2 else None
        return _CycloneTimeTrace(phi_t, density_t, stride)
    _, phi_out_time = integrate_linear(
        fresh_G0(),
        grid,
        geom,
        params,
        dt=dt,
        steps=steps,
        method=method,
        terms=terms,
        sample_stride=stride,
        show_progress=show_progress,
    )
    return _CycloneTimeTrace(phi_out_time, None, stride)

def _integrate_cyclone_time_trace(
    *,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    time_cfg_use: Any | None,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    need_density: bool,
    use_jit: bool,
    show_progress: bool,
    status: Callable[[str], None],
    fresh_G0: Callable[[], jnp.ndarray],
) -> _CycloneTimeTrace:
    """Route Cyclone time integration to the configured or fixed-step backend."""

    if time_cfg_use is not None:
        status(
            f"running runtime-configured integrator over {int(steps)} steps with sample_stride={int(time_cfg_use.sample_stride)}"
        )
        if need_density:
            status("saving phi and density diagnostics for automatic fit selection")
        return _integrate_cyclone_configured_time(
            grid=grid,
            geom=geom,
            params=params,
            terms=terms,
            time_cfg_use=time_cfg_use,
            need_density=need_density,
            show_progress=show_progress,
            fresh_G0=fresh_G0,
        )

    stride = 1 if sample_stride is None else int(sample_stride)
    status(
        f"running {'explicit diagnostics' if need_density or not use_jit else 'cached linear'} integrator over {int(steps)} steps with sample_stride={stride}"
    )
    return _integrate_cyclone_unconfigured_time(
        grid=grid,
        geom=geom,
        params=params,
        terms=terms,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        need_density=need_density,
        use_jit=use_jit,
        show_progress=show_progress,
        fresh_G0=fresh_G0,
    )

def _cyclone_auto_fit_kwargs(options: _CycloneTimeFitOptions) -> dict[str, Any]:
    """Pack automatic-window options shared by Cyclone time-fit branches."""

    return {
        "tmin": options.tmin,
        "tmax": options.tmax,
        "window_fraction": options.window_fraction,
        "min_points": options.min_points,
        "start_fraction": options.start_fraction,
        "growth_weight": options.growth_weight,
        "require_positive": options.require_positive,
        "min_amp_fraction": options.min_amp_fraction,
        "max_amp_fraction": options.max_amp_fraction,
        "window_method": options.window_method,
        "max_fraction": options.max_fraction,
        "end_fraction": options.end_fraction,
        "num_windows": 8,
        "phase_weight": options.phase_weight,
        "length_weight": options.length_weight,
        "min_r2": options.min_r2,
        "late_penalty": options.late_penalty,
        "min_slope": options.min_slope,
        "min_slope_frac": options.min_slope_frac,
        "slope_var_weight": options.slope_var_weight,
    }

def _build_cyclone_time_fit_options(
    *,
    fit_key: str,
    mode_method: str,
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
) -> _CycloneTimeFitOptions:
    """Collect user/runtime fit knobs into the immutable fit policy object."""

    return _CycloneTimeFitOptions(
        fit_key=fit_key,
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

def _fit_cyclone_time_trace(
    *,
    phi_t: Any,
    density_t: Any | None,
    dt: float,
    stride: int,
    sel: Any,
    params: Any,
    diagnostic_norm: str,
    options: _CycloneTimeFitOptions,
    status: Callable[[str], None],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit Cyclone growth/frequency from the saved time trace."""

    phi_t_np = np.asarray(phi_t)
    t_arr = np.arange(phi_t_np.shape[0]) * dt * stride
    density_np = None if density_t is None else np.asarray(density_t)
    status(
        f"integration complete; fitting growth rate from {phi_t_np.shape[0]} saved samples"
    )
    auto_fit_kwargs = _cyclone_auto_fit_kwargs(options)
    if options.fit_key == "auto":
        _signal, name, gamma_out, omega_out = _select_fit_signal_auto(
            t_arr,
            phi_t_np,
            density_np,
            sel,
            mode_method=options.mode_method,
            **auto_fit_kwargs,
        )
        status(f"automatic fit selected signal '{name}'")
        if not np.isfinite(gamma_out) or not np.isfinite(omega_out):
            gamma_out, omega_out = 0.0, 0.0
    else:
        signal = _select_fit_signal(
            phi_t_np,
            density_np,
            sel,
            fit_signal=options.fit_key,
            mode_method=options.mode_method,
        )
        if options.auto_window and options.tmin is None and options.tmax is None:
            gamma_out, omega_out, _tmin, _tmax = fit_growth_rate_auto(
                t_arr,
                signal,
                **auto_fit_kwargs,
            )
        else:
            gamma_out, omega_out = fit_growth_rate(
                t_arr, signal, tmin=options.tmin, tmax=options.tmax
            )
    gamma_out, omega_out = _normalize_growth_rate(
        gamma_out,
        omega_out,
        params,
        diagnostic_norm,
    )
    return float(gamma_out), float(omega_out), phi_t_np, t_arr

def _prepare_cyclone_time_path_controls(
    *,
    cfg: Any,
    time_cfg: Any,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    fit_key: str,
    mode_method: str,
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
) -> _CycloneTimePathControls:
    """Resolve time-config overrides and fit-window policy for Cyclone."""

    return _CycloneTimePathControls(
        time_cfg=_resolve_cyclone_time_config(
            cfg=cfg,
            time_cfg=time_cfg,
            dt=dt,
            steps=steps,
            method=method,
            sample_stride=sample_stride,
        ),
        fit_options=_build_cyclone_time_fit_options(
            fit_key=fit_key,
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
        ),
    )

def _run_cyclone_saved_time_path(
    *,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    sel: Any,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    need_density: bool,
    use_jit: bool,
    diagnostic_norm: str,
    show_progress: bool,
    status: Callable[[str], None],
    fresh_G0: Callable[[], jnp.ndarray],
    controls: _CycloneTimePathControls,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run non-reference Cyclone time integration and fit the saved trace."""

    trace = _integrate_cyclone_time_trace(
        grid=grid,
        geom=geom,
        params=params,
        terms=terms,
        time_cfg_use=controls.time_cfg,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        need_density=need_density,
        use_jit=use_jit,
        show_progress=show_progress,
        status=status,
        fresh_G0=fresh_G0,
    )
    gamma_out, omega_out, phi_t_np, t_arr = _fit_cyclone_time_trace(
        phi_t=trace.phi_t,
        density_t=trace.density_t,
        dt=dt,
        stride=trace.stride,
        sel=sel,
        params=params,
        diagnostic_norm=diagnostic_norm,
        options=controls.fit_options,
        status=status,
    )
    status(f"time integration fit complete: gamma={gamma_out:.6f} omega={omega_out:.6f}")
    return gamma_out, omega_out, phi_t_np, t_arr

def _cyclone_time_path_controls_from_request(
    request: _CycloneTimePathRequest,
) -> _CycloneTimePathControls:
    return _prepare_cyclone_time_path_controls(
        cfg=request.cfg,
        time_cfg=request.time_cfg,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        sample_stride=request.sample_stride,
        fit_key=request.fit_key,
        mode_method=request.mode_method,
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

def _run_cyclone_time_path_request(
    request: _CycloneTimePathRequest,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    request.status(f"starting time integration path with fit_signal={request.fit_key}")
    controls = _cyclone_time_path_controls_from_request(request)
    if request.reference_aligned:
        request.status("running reference-aligned explicit integrator")
        return _run_cyclone_reference_aligned_time(
            grid=request.grid,
            cache=request.cache,
            params=request.params,
            geom=request.geom,
            terms=request.terms,
            time_cfg_use=controls.time_cfg,
            dt=request.dt,
            steps=request.steps,
            sample_stride=request.sample_stride,
            diagnostic_norm=request.diagnostic_norm,
            show_progress=request.show_progress,
            fresh_G0=request.fresh_G0,
        )

    return _run_cyclone_saved_time_path(
        grid=request.grid,
        geom=request.geom,
        params=request.params,
        terms=request.terms,
        sel=request.sel,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        sample_stride=request.sample_stride,
        need_density=request.need_density,
        use_jit=request.use_jit,
        diagnostic_norm=request.diagnostic_norm,
        show_progress=request.show_progress,
        status=request.status,
        fresh_G0=request.fresh_G0,
        controls=controls,
    )

def run_cyclone_time_path(
    *,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    cfg: Any,
    time_cfg: Any,
    sel: Any,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    fit_key: str,
    need_density: bool,
    reference_aligned: bool,
    use_jit: bool,
    diagnostic_norm: str,
    show_progress: bool,
    status: Callable[[str], None],
    fresh_G0: Callable[[], jnp.ndarray],
    mode_method: str,
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
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run the Cyclone time-integration branch and fit late-time growth."""

    return _run_cyclone_time_path_request(_cyclone_time_path_request_from_locals(locals()))

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
    return run_cyclone_krylov_path(
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
    return run_cyclone_time_path(
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

@dataclass(frozen=True)
class CycloneScanHooks:
    """Patchable numerical hooks supplied by the Cyclone scan owner module."""

    cyclone_scan_result: type[CycloneScanResult]
    explicit_time_config: type
    mode_selection: type[ModeSelection]
    mode_selection_batch: type[ModeSelectionBatch]
    select_ky_index: Callable[..., int]
    select_ky_grid: Callable[..., Any]
    build_initial_condition: Callable[..., Any]
    build_linear_cache: Callable[..., Any]
    integrate_linear_explicit: Callable[..., Any]
    integrate_linear: Callable[..., Any]
    integrate_linear_diagnostics: Callable[..., Any]
    integrate_linear_from_config: Callable[..., Any]
    integrate_linear_diffrax_streaming: Callable[..., Any]
    instantaneous_growth_rate_from_phi: Callable[
        ..., tuple[float, float, Any, Any, Any]
    ]
    dominant_eigenpair: Callable[..., tuple[Any, Any]]
    extract_mode_time_series: Callable[..., np.ndarray]
    select_fit_signal_auto: Callable[..., tuple[np.ndarray, str, float, float]]
    run_cyclone_linear: Callable[..., Any]
    normalize_growth_rate: Callable[..., tuple[float, float]]
    resolve_streaming_window: Callable[..., tuple[float, float]]
    midplane_index: Callable[..., int]
    resolve_cfl_fac: Callable[..., float]

def seed_from_explicit_trace(
    G0: Any,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    cfg_use: Any,
    hooks: Any,
    *,
    show_progress: bool,
) -> tuple[bool, bool, float, float]:
    """Estimate the starting branch from a short explicit trace."""

    gamma_seed = 0.0
    omega_seed = 0.0
    try:
        t_seed = min(150.0, float(cfg_use.power_dt) * 15000.0)
        explicit_time_cfg = hooks.explicit_time_config(
            dt=float(cfg_use.power_dt),
            t_max=t_seed,
            sample_stride=1,
            fixed_dt=True,
        )
        t_short, phi_seed, _g_t, _o_t = hooks.integrate_linear_explicit(
            jnp.array(G0),
            grid,
            cache,
            params,
            geom,
            explicit_time_cfg,
            terms=terms,
            mode_method="z_index",
            show_progress=show_progress,
        )
        sel = hooks.mode_selection(
            ky_index=0,
            kx_index=0,
            z_index=hooks.midplane_index(grid),
        )
        gamma_seed, omega_seed, _g, _o, _t_mid = (
            hooks.instantaneous_growth_rate_from_phi(
                phi_seed,
                t_short,
                sel,
                navg_fraction=0.5,
                mode_method="z_index",
            )
        )
        omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
        seed_ok = omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
    except Exception:
        seed_ok = False
        omega_ok = False
    return seed_ok, omega_ok, float(gamma_seed), float(omega_seed)

def reduced_seed_from_explicit_trace(
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    cfg_use: Any,
    hooks: Any,
    *,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    show_progress: bool,
) -> tuple[bool, bool, float, float]:
    """Build a smaller velocity-space seed trace when the full seed fails."""

    n_laguerre_seed = min(n_laguerre, 16)
    n_hermite_seed = min(n_hermite, 12)
    try:
        cache_seed = hooks.build_linear_cache(
            grid,
            geom,
            params,
            n_laguerre_seed,
            n_hermite_seed,
        )
        G0_seed = hooks.build_initial_condition(
            grid,
            geom,
            ky_index=0,
            kx_index=0,
            Nl=n_laguerre_seed,
            Nm=n_hermite_seed,
            init_cfg=init_cfg,
        )
    except Exception:
        return False, False, 0.0, 0.0
    return seed_from_explicit_trace(
        G0_seed,
        grid,
        cache_seed,
        params,
        geom,
        terms,
        cfg_use,
        hooks,
        show_progress=show_progress,
    )

def seed_shift(
    prev_eig: complex | None,
    *,
    omega_ok: bool,
    seed_ok: bool,
    gamma_seed: float,
    omega_seed: float,
) -> complex | None:
    """Choose the Krylov shift implied by continuation or a trace seed."""

    if prev_eig is not None and np.isfinite(prev_eig):
        return prev_eig
    if omega_ok:
        return complex(float(gamma_seed) if seed_ok else 0.0, float(-omega_seed))
    return None

def use_explicit_seed(
    gamma: float,
    omega: float,
    *,
    seed_ok: bool,
    gamma_seed: float,
    omega_seed: float,
) -> bool:
    """Return whether the explicit trace should override the eigen-solver result."""

    if not seed_ok:
        return False
    seed_strong = (gamma_seed > 0.0) and (abs(omega_seed) > 1.0e-6)
    if not seed_strong:
        return False
    omega_tol = 0.15 * max(abs(omega_seed), 1.0e-6)
    gamma_tol = 0.15 * max(abs(gamma_seed), 1.0e-6)
    return (
        not np.isfinite(gamma)
        or not np.isfinite(omega)
        or (gamma_seed > 0.0 and gamma < 0.0)
        or abs(omega - omega_seed) > omega_tol
        or abs(gamma - gamma_seed) > gamma_tol
    )

def explicit_time_config_for_scan_point(
    *,
    dt_i: float,
    steps_i: int,
    reference_aligned: bool,
    time_cfg: Any | None,
    time_base: Any,
    hooks: Any,
) -> Any:
    """Build the explicit-time configuration for one ky scan point."""

    t_max_val = dt_i * float(steps_i)
    if reference_aligned and time_cfg is None:
        fixed_dt_i = True
        dt_min_i = dt_i
        dt_max_i: float | None = dt_i
        cfl_i = 1.0
        cfl_fac_i = 1.0
    else:
        fixed_dt_i = bool(time_base.fixed_dt)
        dt_min_i = float(time_base.dt_min)
        dt_max_i = None if time_base.dt_max is None else float(time_base.dt_max)
        cfl_i = float(time_base.cfl)
        cfl_fac_i = hooks.resolve_cfl_fac(str(time_base.method), time_base.cfl_fac)
    return hooks.explicit_time_config(
        dt=dt_i,
        t_max=t_max_val,
        sample_stride=1,
        fixed_dt=fixed_dt_i,
        dt_min=dt_min_i,
        dt_max=dt_max_i,
        cfl=cfl_i,
        cfl_fac=cfl_fac_i,
    )

def explicit_reselection_target(
    *,
    explicit_growth_ok: bool,
    prev_omega: float | None,
    prev_prev_omega: float | None,
) -> float | None:
    """Predict the next branch frequency used when explicit fits jump branch."""

    target_omega = (
        prev_omega if (explicit_growth_ok and prev_omega is not None) else None
    )
    if (
        target_omega is not None
        and prev_prev_omega is not None
        and prev_omega is not None
        and prev_omega > prev_prev_omega
    ):
        target_omega = prev_omega + (prev_omega - prev_prev_omega)
    return target_omega

def krylov_reselected_frequency(
    G0: Any,
    cache: Any,
    params: Any,
    terms: Any,
    kcfg: Any,
    hooks: Any,
    diagnostic_norm: str,
) -> tuple[float, float]:
    """Re-evaluate a scan point with Krylov mode selection after a bad fit."""

    eig, _vec = hooks.dominant_eigenpair(
        jnp.array(G0),
        cache,
        params,
        terms=terms,
        krylov_dim=kcfg.krylov_dim,
        restarts=kcfg.restarts,
        omega_min_factor=kcfg.omega_min_factor,
        omega_target_factor=kcfg.omega_target_factor,
        omega_cap_factor=kcfg.omega_cap_factor,
        omega_sign=kcfg.omega_sign,
        method=kcfg.method,
        power_iters=kcfg.power_iters,
        power_dt=kcfg.power_dt,
        shift=kcfg.shift,
        shift_source=kcfg.shift_source,
        shift_tol=kcfg.shift_tol,
        shift_maxiter=kcfg.shift_maxiter,
        shift_restart=kcfg.shift_restart,
        shift_solve_method=kcfg.shift_solve_method,
        shift_preconditioner=kcfg.shift_preconditioner,
        shift_selection=kcfg.shift_selection,
        mode_family=kcfg.mode_family,
        fallback_method=kcfg.fallback_method,
        fallback_real_floor=kcfg.fallback_real_floor,
    )
    gamma_k = float(np.real(eig))
    omega_k = float(abs(-np.imag(eig)))
    return hooks.normalize_growth_rate(gamma_k, omega_k, params, diagnostic_norm)

def choose_reselected_frequency(
    *,
    gamma: float,
    omega: float,
    gamma_k: float,
    omega_k: float,
    target_omega: float,
) -> tuple[float, float]:
    """Pick the frequency closest to the continuation target without growth jumps."""

    candidates: list[tuple[float, float]] = [(float(gamma), float(abs(omega)))]
    gamma_base = abs(float(gamma))
    gamma_delta_limit = max(3.0 * gamma_base, gamma_base + 0.05, 1.0e-3)
    if (
        np.isfinite(gamma_k)
        and np.isfinite(omega_k)
        and gamma_k > 0.0
        and abs(gamma_k - float(gamma)) <= gamma_delta_limit
    ):
        candidates.append((gamma_k, omega_k))

    def _score(candidate: tuple[float, float]) -> float:
        g_val, o_val = candidate
        penalty = 0.0 if g_val > 0.0 else 1.0e3
        return penalty + abs(o_val - target_omega)

    return min(candidates, key=_score)

@dataclass(frozen=True)
class _CycloneExplicitPoint:
    """Prepared state for one ky point in the explicit Cyclone scan."""

    index: int
    ky_value: float
    grid: Any
    state: Any
    cache: Any
    dt: float
    steps: int

@dataclass(frozen=True)
class _CycloneExplicitFit:
    """Raw explicit-time fit before optional branch reselection."""

    gamma: float
    omega: float
    growth_ok: bool

@dataclass
class _CycloneExplicitContinuation:
    """Previous-frequency state used by explicit branch reselection."""

    prev_omega: float | None = None
    prev_prev_omega: float | None = None

    def update(self, omega: float) -> None:
        self.prev_prev_omega = self.prev_omega
        self.prev_omega = float(omega)

def _prepare_explicit_scan_point(
    *,
    index: int,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    hooks: Any,
) -> _CycloneExplicitPoint:
    """Build grid, initial condition, cache, and time step for one ky point."""

    ky_val = float(ky_values[index])
    ky_index = hooks.select_ky_index(np.asarray(grid_full.ky), ky_val)
    grid = hooks.select_ky_grid(grid_full, ky_index)
    state = hooks.build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=n_laguerre,
        Nm=n_hermite,
        init_cfg=init_cfg,
    )
    cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)
    dt_i = float(dt[index]) if isinstance(dt, np.ndarray) else float(dt)
    steps_i = int(steps[index]) if isinstance(steps, np.ndarray) else int(steps)
    return _CycloneExplicitPoint(
        index=index,
        ky_value=ky_val,
        grid=grid,
        state=state,
        cache=cache,
        dt=dt_i,
        steps=steps_i,
    )

def _fit_explicit_scan_point(
    point: _CycloneExplicitPoint,
    *,
    geom: Any,
    params: Any,
    terms: Any,
    explicit_time_cfg: Any,
    diagnostic_norm: str,
    hooks: Any,
    show_progress: bool,
) -> _CycloneExplicitFit:
    """Integrate one explicit-time trace and fit its instantaneous growth."""

    t, phi_t, _g_t, _o_t = hooks.integrate_linear_explicit(
        jnp.array(point.state),
        point.grid,
        point.cache,
        params,
        geom,
        explicit_time_cfg,
        terms=terms,
        mode_method="z_index",
        show_progress=show_progress,
    )
    sel_local = hooks.mode_selection(
        ky_index=0,
        kx_index=0,
        z_index=hooks.midplane_index(point.grid),
    )
    try:
        gamma, omega, _g, _o, _t_mid = hooks.instantaneous_growth_rate_from_phi(
            phi_t,
            t,
            sel_local,
            navg_fraction=0.5,
            mode_method="z_index",
        )
        gamma, omega = hooks.normalize_growth_rate(
            gamma,
            omega,
            params,
            diagnostic_norm,
        )
        return _CycloneExplicitFit(float(gamma), float(omega), True)
    except ValueError:
        return _CycloneExplicitFit(float("nan"), float("nan"), False)

def _explicit_scan_needs_reselection(
    *,
    reference_aligned: bool,
    fit: _CycloneExplicitFit,
    continuation: _CycloneExplicitContinuation,
    index: int,
) -> bool:
    """Return whether explicit branch history suggests a Krylov reselection."""

    return (
        (reference_aligned and fit.growth_ok)
        and continuation.prev_omega is not None
        and continuation.prev_omega > 0.0
        and (
            fit.omega <= 0.0
            or ((index >= 2) and (fit.omega < 0.85 * continuation.prev_omega))
        )
    )

def _reselected_explicit_scan_fit(
    point: _CycloneExplicitPoint,
    fit: _CycloneExplicitFit,
    *,
    params: Any,
    terms: Any,
    kcfg: Any,
    diagnostic_norm: str,
    continuation: _CycloneExplicitContinuation,
    hooks: Any,
) -> _CycloneExplicitFit:
    """Apply Krylov reselection when explicit-time branch tracking fails."""

    target_omega = explicit_reselection_target(
        explicit_growth_ok=fit.growth_ok,
        prev_omega=continuation.prev_omega,
        prev_prev_omega=continuation.prev_prev_omega,
    )
    gamma_k, omega_k = krylov_reselected_frequency(
        point.state,
        point.cache,
        params,
        terms,
        kcfg,
        hooks,
        diagnostic_norm,
    )
    if not fit.growth_ok:
        return _CycloneExplicitFit(gamma_k, omega_k, True)
    assert target_omega is not None
    gamma, omega = choose_reselected_frequency(
        gamma=fit.gamma,
        omega=fit.omega,
        gamma_k=gamma_k,
        omega_k=omega_k,
        target_omega=target_omega,
    )
    return _CycloneExplicitFit(gamma, omega, True)

def _final_explicit_scan_fit(
    point: _CycloneExplicitPoint,
    fit: _CycloneExplicitFit,
    *,
    params: Any,
    terms: Any,
    kcfg: Any,
    diagnostic_norm: str,
    reference_aligned: bool,
    continuation: _CycloneExplicitContinuation,
    hooks: Any,
) -> _CycloneExplicitFit:
    """Resolve sign convention and optional branch reselection for one point."""

    if reference_aligned and continuation.prev_omega is None and fit.omega < 0.0:
        fit = _CycloneExplicitFit(fit.gamma, abs(fit.omega), fit.growth_ok)
    if _explicit_scan_needs_reselection(
        reference_aligned=reference_aligned,
        fit=fit,
        continuation=continuation,
        index=point.index,
    ) or not fit.growth_ok:
        return _reselected_explicit_scan_fit(
            point,
            fit,
            params=params,
            terms=terms,
            kcfg=kcfg,
            diagnostic_norm=diagnostic_norm,
            continuation=continuation,
            hooks=hooks,
        )
    return fit

def _append_explicit_scan_point(
    *,
    point: _CycloneExplicitPoint,
    geom: Any,
    params: Any,
    terms: Any,
    time_cfg: Any | None,
    time_base: Any,
    kcfg: Any,
    reference_aligned: bool,
    diagnostic_norm: str,
    continuation: _CycloneExplicitContinuation,
    gamma_out: np.ndarray,
    omega_out: np.ndarray,
    hooks: Any,
    show_progress: bool,
) -> None:
    """Evaluate one explicit scan point and update continuation/output arrays."""

    explicit_time_cfg = explicit_time_config_for_scan_point(
        dt_i=point.dt,
        steps_i=point.steps,
        reference_aligned=reference_aligned,
        time_cfg=time_cfg,
        time_base=time_base,
        hooks=hooks,
    )
    fit = _fit_explicit_scan_point(
        point,
        geom=geom,
        params=params,
        terms=terms,
        explicit_time_cfg=explicit_time_cfg,
        diagnostic_norm=diagnostic_norm,
        hooks=hooks,
        show_progress=show_progress,
    )
    fit = _final_explicit_scan_fit(
        point,
        fit,
        params=params,
        terms=terms,
        kcfg=kcfg,
        diagnostic_norm=diagnostic_norm,
        reference_aligned=reference_aligned,
        continuation=continuation,
        hooks=hooks,
    )
    gamma_out[point.index] = fit.gamma
    omega_out[point.index] = fit.omega
    continuation.update(fit.omega)

def run_explicit_time_cyclone_scan(
    *,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    terms: Any,
    cfg: Any,
    time_cfg: Any | None,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    krylov_cfg: Any | None,
    krylov_default: Any,
    reference_aligned: bool,
    diagnostic_norm: str,
    show_progress: bool,
    hooks: Any,
) -> Any:
    """Run the reference-aligned explicit-time branch for a Cyclone ky scan."""

    if ky_values.size == 0:
        return hooks.cyclone_scan_result(
            ky=ky_values,
            gamma=np.array([]),
            omega=np.array([]),
        )

    gamma_out = np.zeros_like(ky_values, dtype=float)
    omega_out = np.zeros_like(ky_values, dtype=float)
    continuation = _CycloneExplicitContinuation()
    kcfg = krylov_cfg or krylov_default
    time_base = time_cfg or cfg.time
    for idx, _ky_val in enumerate(ky_values):
        point = _prepare_explicit_scan_point(
            index=idx,
            ky_values=ky_values,
            grid_full=grid_full,
            geom=geom,
            params=params,
            init_cfg=init_cfg,
            n_laguerre=n_laguerre,
            n_hermite=n_hermite,
            dt=dt,
            steps=steps,
            hooks=hooks,
        )
        _append_explicit_scan_point(
            point=point,
            geom=geom,
            params=params,
            terms=terms,
            time_cfg=time_cfg,
            time_base=time_base,
            kcfg=kcfg,
            reference_aligned=reference_aligned,
            diagnostic_norm=diagnostic_norm,
            continuation=continuation,
            gamma_out=gamma_out,
            omega_out=omega_out,
            hooks=hooks,
            show_progress=show_progress,
        )
    return hooks.cyclone_scan_result(ky=ky_values, gamma=gamma_out, omega=omega_out)

def _empty_scan_result(ky_values: np.ndarray, hooks: Any) -> CycloneScanResult:
    return hooks.cyclone_scan_result(
        ky=ky_values,
        gamma=np.array([]),
        omega=np.array([]),
    )

@dataclass(frozen=True)
class _CycloneKrylovPoint:
    """Prepared state for one ky point in the Cyclone Krylov scan."""

    index: int
    ky_value: float
    grid: Any
    state: Any
    cache: Any

@dataclass(frozen=True)
class _CycloneScanKrylovSeed:
    """Explicit-trace seed information used to stabilize branch following."""

    seed_ok: bool
    omega_ok: bool
    gamma: float
    omega: float

@dataclass
class _CycloneKrylovContinuation:
    """Mutable branch-following state carried between ky points."""

    v_ref: Any = None
    prev_eig: complex | None = None

def _prepare_krylov_scan_point(
    *,
    index: int,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    hooks: Any,
) -> _CycloneKrylovPoint:
    """Build grid, initial condition, and cache for one Cyclone scan point."""

    ky_val = float(ky_values[index])
    ky_index = hooks.select_ky_index(np.asarray(grid_full.ky), ky_val)
    grid = hooks.select_ky_grid(grid_full, ky_index)
    state = hooks.build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=n_laguerre,
        Nm=n_hermite,
        init_cfg=init_cfg,
    )
    cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)
    return _CycloneKrylovPoint(
        index=index,
        ky_value=ky_val,
        grid=grid,
        state=state,
        cache=cache,
    )

def _explicit_seed_for_krylov_point(
    point: _CycloneKrylovPoint,
    *,
    geom: Any,
    params: Any,
    terms: Any,
    cfg_use: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    prev_eig: complex | None,
    hooks: Any,
    show_progress: bool,
) -> _CycloneScanKrylovSeed:
    """Resolve the full or reduced explicit seed used by the Krylov branch."""

    seed_ok = False
    omega_ok = False
    gamma_seed = 0.0
    omega_seed = 0.0
    if prev_eig is None:
        seed_ok, omega_ok, gamma_seed, omega_seed = seed_from_explicit_trace(
            point.state,
            point.grid,
            point.cache,
            params,
            geom,
            terms,
            cfg_use,
            hooks,
            show_progress=show_progress,
        )
    if not seed_ok:
        seed_ok, omega_ok, gamma_seed, omega_seed = (
            reduced_seed_from_explicit_trace(
                point.grid,
                geom,
                params,
                terms,
                cfg_use,
                hooks,
                init_cfg=init_cfg,
                n_laguerre=n_laguerre,
                n_hermite=n_hermite,
                show_progress=show_progress,
            )
        )
    return _CycloneScanKrylovSeed(
        seed_ok=bool(seed_ok),
        omega_ok=bool(omega_ok),
        gamma=float(gamma_seed),
        omega=float(omega_seed),
    )

def _dominant_krylov_scan_pair(
    point: _CycloneKrylovPoint,
    *,
    params: Any,
    terms: Any,
    cfg_use: Any,
    continuation: _CycloneKrylovContinuation,
    seed: _CycloneScanKrylovSeed,
    hooks: Any,
) -> tuple[Any, Any]:
    """Run the dominant-eigenpair solve for one prepared Cyclone scan point."""

    shift = seed_shift(
        continuation.prev_eig,
        omega_ok=seed.omega_ok,
        seed_ok=seed.seed_ok,
        gamma_seed=seed.gamma,
        omega_seed=seed.omega,
    )
    return hooks.dominant_eigenpair(
        point.state,
        point.cache,
        params,
        terms=terms,
        v_ref=continuation.v_ref,
        select_overlap=continuation.v_ref is not None,
        krylov_dim=cfg_use.krylov_dim,
        restarts=cfg_use.restarts,
        omega_min_factor=cfg_use.omega_min_factor,
        omega_target_factor=cfg_use.omega_target_factor,
        omega_cap_factor=cfg_use.omega_cap_factor,
        omega_sign=cfg_use.omega_sign,
        method=cfg_use.method,
        power_iters=cfg_use.power_iters,
        power_dt=cfg_use.power_dt,
        shift=shift if shift is not None else cfg_use.shift,
        shift_source=cfg_use.shift_source,
        shift_tol=cfg_use.shift_tol,
        shift_maxiter=cfg_use.shift_maxiter,
        shift_restart=cfg_use.shift_restart,
        shift_solve_method=cfg_use.shift_solve_method,
        shift_preconditioner=cfg_use.shift_preconditioner,
        shift_selection=cfg_use.shift_selection,
        mode_family=cfg_use.mode_family,
        fallback_method=cfg_use.fallback_method,
        fallback_real_floor=cfg_use.fallback_real_floor,
    )

def _raw_krylov_scan_rates(
    *,
    eig: Any,
    vec: Any,
    seed: _CycloneScanKrylovSeed,
    continuation: _CycloneKrylovContinuation,
) -> tuple[float, float]:
    """Return raw rates and update continuation before normalization."""

    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    if use_explicit_seed(
        gamma,
        omega,
        seed_ok=seed.seed_ok,
        gamma_seed=seed.gamma,
        omega_seed=seed.omega,
    ):
        gamma = float(seed.gamma)
        omega = float(seed.omega)
    else:
        continuation.v_ref = vec
    continuation.prev_eig = complex(float(gamma), float(-omega))
    return gamma, omega

def _append_krylov_scan_point(
    *,
    point: _CycloneKrylovPoint,
    geom: Any,
    params: Any,
    terms: Any,
    cfg_use: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    diagnostic_norm: str,
    continuation: _CycloneKrylovContinuation,
    gamma_out: np.ndarray,
    omega_out: np.ndarray,
    hooks: Any,
    show_progress: bool,
) -> None:
    """Evaluate one ky point and write normalized scan outputs."""

    seed = _explicit_seed_for_krylov_point(
        point,
        geom=geom,
        params=params,
        terms=terms,
        cfg_use=cfg_use,
        init_cfg=init_cfg,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        prev_eig=continuation.prev_eig,
        hooks=hooks,
        show_progress=show_progress,
    )
    eig, vec = _dominant_krylov_scan_pair(
        point,
        params=params,
        terms=terms,
        cfg_use=cfg_use,
        continuation=continuation,
        seed=seed,
        hooks=hooks,
    )
    gamma, omega = _raw_krylov_scan_rates(
        eig=eig,
        vec=vec,
        seed=seed,
        continuation=continuation,
    )
    gamma, omega = hooks.normalize_growth_rate(
        gamma,
        omega,
        params,
        diagnostic_norm,
    )
    gamma_out[point.index] = gamma
    omega_out[point.index] = omega

def run_krylov_cyclone_scan(
    *,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    terms: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    mode_follow: bool,
    krylov_cfg: Any | None,
    krylov_default: Any,
    diagnostic_norm: str,
    show_progress: bool,
    hooks: Any,
) -> CycloneScanResult:
    """Run the Krylov branch-following path for a Cyclone ky scan."""

    if ky_values.size == 0:
        return _empty_scan_result(ky_values, hooks)

    order = np.argsort(ky_values) if mode_follow else np.arange(ky_values.size)
    gamma_out = np.zeros_like(ky_values, dtype=float)
    omega_out = np.zeros_like(ky_values, dtype=float)
    continuation = _CycloneKrylovContinuation()
    cfg_use = krylov_cfg or krylov_default
    for idx in order:
        point = _prepare_krylov_scan_point(
            index=int(idx),
            ky_values=ky_values,
            grid_full=grid_full,
            geom=geom,
            params=params,
            init_cfg=init_cfg,
            n_laguerre=n_laguerre,
            n_hermite=n_hermite,
            hooks=hooks,
        )
        _append_krylov_scan_point(
            point=point,
            geom=geom,
            params=params,
            terms=terms,
            cfg_use=cfg_use,
            init_cfg=init_cfg,
            n_laguerre=n_laguerre,
            n_hermite=n_hermite,
            diagnostic_norm=diagnostic_norm,
            continuation=continuation,
            gamma_out=gamma_out,
            omega_out=omega_out,
            hooks=hooks,
            show_progress=show_progress,
        )
    return hooks.cyclone_scan_result(ky=ky_values, gamma=gamma_out, omega=omega_out)

def _valid_time_branch_growth(
    gamma_val: float,
    omega_val: float,
    *,
    require_positive: bool,
) -> bool:
    if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
        return False
    if require_positive and gamma_val <= 0.0:
        return False
    return True

def _resolve_time_branch_growth(
    gamma: float,
    omega: float,
    *,
    ky_value: float,
    n_laguerre: int,
    n_hermite: int,
    dt: float,
    steps: int,
    method: str,
    params: Any,
    cfg: Any,
    time_cfg: Any | None,
    krylov_cfg: Any | None,
    diagnostic_norm: str,
    auto_solver: bool,
    require_positive: bool,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> tuple[float, float]:
    """Return fitted growth/frequency, using the Krylov fallback if required."""

    if not auto_solver or _valid_time_branch_growth(
        gamma, omega, require_positive=require_positive
    ):
        return gamma, omega

    result = hooks.run_cyclone_linear(
        ky_target=float(ky_value),
        Nl=n_laguerre,
        Nm=n_hermite,
        dt=dt,
        steps=steps,
        method=method,
        params=params,
        cfg=cfg,
        time_cfg=time_cfg,
        solver="krylov",
        krylov_cfg=krylov_cfg,
        diagnostic_norm=diagnostic_norm,
        fit_signal="phi",
        show_progress=show_progress,
    )
    return float(result.gamma), float(result.omega)

@dataclass(frozen=True)
class _CycloneTimeScanBatch:
    batch_start: int
    ky_slice: np.ndarray
    valid_count: int
    ky_local: np.ndarray
    grid: Any
    selection: ModeSelection | ModeSelectionBatch
    dt: float
    steps: int
    state: Any
    cache: Any

@dataclass(frozen=True)
class _CycloneHistoryFitOptions:
    n_laguerre: int
    n_hermite: int
    method: str
    params: Any
    cfg: Any
    time_cfg: Any | None
    krylov_cfg: Any | None
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    max_amp_fraction: float
    max_fraction: float
    end_fraction: float
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
    fit_key: str
    diagnostic_norm: str
    auto_solver: bool
    fit_policy: Any
    hooks: CycloneScanHooks
    show_progress: bool

@dataclass(frozen=True)
class _CycloneTimeRunOptions:
    ky_values: np.ndarray
    grid_full: Any
    geom: Any
    params: Any
    terms: Any
    cfg: Any
    time_cfg: Any | None
    init_cfg: Any
    n_laguerre: int
    n_hermite: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    sample_stride: int | None
    mode_method: str
    mode_only: bool
    fit_key: str
    need_density: bool
    diagnostic_norm: str
    use_jit: bool
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    use_batch: bool
    hooks: CycloneScanHooks
    show_progress: bool

@dataclass
class _CycloneScanOutput:
    gammas: list[float]
    omegas: list[float]
    ky: list[float]

    @classmethod
    def empty(cls) -> "_CycloneScanOutput":
        return cls(gammas=[], omegas=[], ky=[])

@dataclass(frozen=True)
class _CycloneTimeScanControls:
    run_options: _CycloneTimeRunOptions
    fit_options: _CycloneHistoryFitOptions

@dataclass(frozen=True)
class _CycloneTimeScanInputs:
    ky_values: np.ndarray
    grid_full: Any
    geom: Any
    params: Any
    terms: Any
    cfg: Any
    time_cfg: Any | None
    init_cfg: Any
    n_laguerre: int
    n_hermite: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    krylov_cfg: Any | None
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    max_amp_fraction: float
    max_fraction: float
    end_fraction: float
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
    sample_stride: int | None
    fit_key: str
    need_density: bool
    diagnostic_norm: str
    use_jit: bool
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    auto_solver: bool
    use_batch: bool
    fit_policy: Any
    hooks: CycloneScanHooks
    show_progress: bool

def _cyclone_time_scan_inputs_from_locals(values: dict[str, Any]) -> _CycloneTimeScanInputs:
    """Pack public ``run_time_cyclone_scan`` arguments once for internal routing."""

    return _CycloneTimeScanInputs(
        **{field.name: values[field.name] for field in fields(_CycloneTimeScanInputs)}
    )

def _iter_cyclone_time_scan_batches(options: _CycloneTimeRunOptions):
    if options.use_batch:
        return _iter_ky_batches(
            options.ky_values,
            ky_batch=options.ky_batch,
            fixed_batch_shape=options.fixed_batch_shape,
        )
    return _iter_ky_batches(options.ky_values, ky_batch=1, fixed_batch_shape=False)

def _prepare_cyclone_time_batch(
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    use_batch: bool,
    grid_full: Any,
    geom: Any,
    params: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    hooks: CycloneScanHooks,
) -> _CycloneTimeScanBatch:
    if use_batch:
        ky_indices = [
            hooks.select_ky_index(np.asarray(grid_full.ky), float(ky))
            for ky in ky_slice
        ]
        grid = hooks.select_ky_grid(grid_full, ky_indices)
        ky_local = np.arange(len(ky_indices))
        selection: ModeSelection | ModeSelectionBatch = hooks.mode_selection_batch(
            ky_local.astype(int), 0, hooks.midplane_index(grid)
        )
        dt_i = float(dt)
        steps_i = int(steps)
    else:
        ky_indices = [
            hooks.select_ky_index(np.asarray(grid_full.ky), float(ky_slice[0]))
        ]
        grid = hooks.select_ky_grid(grid_full, ky_indices[0])
        ky_local = np.arange(len(ky_indices))
        selection = hooks.mode_selection(
            ky_index=0, kx_index=0, z_index=hooks.midplane_index(grid)
        )
        dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)

    state = hooks.build_initial_condition(
        grid,
        geom,
        ky_index=ky_local,
        kx_index=0,
        Nl=n_laguerre,
        Nm=n_hermite,
        init_cfg=init_cfg,
    )
    cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)
    return _CycloneTimeScanBatch(
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        ky_local=ky_local,
        grid=grid,
        selection=selection,
        dt=dt_i,
        steps=steps_i,
        state=state,
        cache=cache,
    )

def _cyclone_time_config_for_batch(
    time_cfg: Any | None,
    *,
    dt: float,
    steps: int,
    sample_stride: int | None,
) -> Any | None:
    if time_cfg is None:
        return None
    time_cfg_i = replace(time_cfg, dt=dt, t_max=dt * steps)
    if sample_stride is not None:
        time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)
    return time_cfg_i

def _append_cyclone_streaming_results(
    batch: _CycloneTimeScanBatch,
    *,
    geom: Any,
    params: Any,
    terms: Any,
    time_cfg: Any,
    tmin: float | None,
    tmax: float | None,
    start_fraction: float,
    window_fraction: float,
    mode_method: str,
    streaming_amp_floor: float,
    diagnostic_norm: str,
    hooks: CycloneScanHooks,
    show_progress: bool,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> None:
    t_total = float(time_cfg.t_max)
    tmin_i, tmax_i = hooks.resolve_streaming_window(
        t_total,
        indexed_float_value(tmin, batch.batch_start),
        indexed_float_value(tmax, batch.batch_start),
        start_fraction,
        window_fraction,
        1.0,
    )
    _, gamma_vals, omega_vals = hooks.integrate_linear_diffrax_streaming(
        batch.state,
        batch.grid,
        geom,
        params,
        dt=batch.dt,
        steps=batch.steps,
        method=time_cfg.diffrax_solver,
        cache=batch.cache,
        terms=terms,
        adaptive=False,
        rtol=time_cfg.diffrax_rtol,
        atol=time_cfg.diffrax_atol,
        max_steps=time_cfg.diffrax_max_steps,
        progress_bar=time_cfg.progress_bar,
        checkpoint=time_cfg.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal="phi",
        show_progress=show_progress,
        mode_ky_indices=batch.ky_local[: batch.valid_count],
        mode_kx_index=0,
        mode_z_index=hooks.midplane_index(batch.grid),
        mode_method=mode_method,
        amp_floor=streaming_amp_floor,
        return_state=False,
    )
    gamma_arr = np.asarray(gamma_vals)
    omega_arr = np.asarray(omega_vals)
    for local_idx in range(batch.valid_count):
        gamma_i, omega_i = hooks.normalize_growth_rate(
            float(gamma_arr[local_idx]),
            float(omega_arr[local_idx]),
            params,
            diagnostic_norm,
        )
        gammas.append(gamma_i)
        omegas.append(omega_i)
        ky_out.append(float(batch.ky_slice[local_idx]))

def _integrate_cyclone_time_history(
    batch: _CycloneTimeScanBatch,
    *,
    geom: Any,
    params: Any,
    terms: Any,
    time_cfg: Any | None,
    method: str,
    mode_method: str,
    mode_only: bool,
    sample_stride: int | None,
    fit_key: str,
    need_density: bool,
    use_jit: bool,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    if time_cfg is not None:
        save_field = (
            "phi+density"
            if fit_key == "auto"
            else ("density" if fit_key == "density" else "phi")
        )
        save_mode = None if fit_key == "auto" else (batch.selection if mode_only else None)
        _, saved = hooks.integrate_linear_from_config(
            batch.state,
            batch.grid,
            geom,
            params,
            time_cfg,
            cache=batch.cache,
            terms=terms,
            save_mode=save_mode,
            mode_method=mode_method,
            save_field=save_field,
            density_species_index=0 if need_density else None,
        )
        if fit_key == "auto":
            phi_t, density_t = saved
            return np.asarray(phi_t), np.asarray(density_t), time_cfg.sample_stride
        return np.asarray(saved), None, time_cfg.sample_stride

    stride = 1 if sample_stride is None else int(sample_stride)
    if use_jit and not need_density:
        _, phi_out_time = hooks.integrate_linear(
            batch.state,
            batch.grid,
            geom,
            params,
            dt=batch.dt,
            steps=batch.steps,
            method=method,
            cache=batch.cache,
            terms=terms,
            sample_stride=stride,
            show_progress=show_progress,
        )
        return np.asarray(phi_out_time), None, stride

    diag = hooks.integrate_linear_diagnostics(
        batch.state,
        batch.grid,
        geom,
        params,
        dt=batch.dt,
        steps=batch.steps,
        method=method,
        cache=batch.cache,
        terms=terms,
        sample_stride=stride,
        species_index=None,
        record_hl_energy=False,
    )
    density_t = np.asarray(diag[2]) if len(diag) > 2 else None
    return np.asarray(diag[1]), density_t, stride

def _resolve_and_append_cyclone_fit(
    *,
    gamma: float,
    omega: float,
    ky_val: float,
    batch: _CycloneTimeScanBatch,
    options: _CycloneHistoryFitOptions,
    output: _CycloneScanOutput,
) -> None:
    gamma, omega = _resolve_time_branch_growth(
        gamma,
        omega,
        ky_value=float(ky_val),
        n_laguerre=options.n_laguerre,
        n_hermite=options.n_hermite,
        dt=batch.dt,
        steps=batch.steps,
        method=options.method,
        params=options.params,
        cfg=options.cfg,
        time_cfg=options.time_cfg,
        krylov_cfg=options.krylov_cfg,
        diagnostic_norm=options.diagnostic_norm,
        auto_solver=options.auto_solver,
        require_positive=options.require_positive,
        hooks=options.hooks,
        show_progress=options.show_progress,
    )
    output.gammas.append(gamma)
    output.omegas.append(omega)
    output.ky.append(float(ky_val))

def _auto_history_fit_for_local_mode(
    *,
    t: np.ndarray,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    batch: _CycloneTimeScanBatch,
    local_idx: int,
    options: _CycloneHistoryFitOptions,
) -> tuple[float, float]:
    sel_local = options.hooks.mode_selection(
        ky_index=local_idx,
        kx_index=0,
        z_index=options.hooks.midplane_index(batch.grid),
    )
    _signal, _name, gamma, omega = options.hooks.select_fit_signal_auto(
        t,
        phi_t,
        density_t,
        sel_local,
        mode_method=options.mode_method,
        tmin=indexed_float_value(options.tmin, batch.batch_start + local_idx),
        tmax=indexed_float_value(options.tmax, batch.batch_start + local_idx),
        window_fraction=options.window_fraction,
        min_points=options.min_points,
        start_fraction=options.start_fraction,
        growth_weight=options.growth_weight,
        require_positive=options.require_positive,
        min_amp_fraction=options.min_amp_fraction,
        max_amp_fraction=options.max_amp_fraction,
        window_method=options.window_method,
        max_fraction=options.max_fraction,
        end_fraction=options.end_fraction,
        num_windows=8,
        phase_weight=options.phase_weight,
        length_weight=options.length_weight,
        min_r2=options.min_r2,
        late_penalty=options.late_penalty,
        min_slope=options.min_slope,
        min_slope_frac=options.min_slope_frac,
        slope_var_weight=options.slope_var_weight,
    )
    return options.hooks.normalize_growth_rate(
        gamma, omega, options.params, options.diagnostic_norm
    )

def _history_signal_for_local_mode(
    *,
    phi_t: np.ndarray,
    signal_t: np.ndarray | None,
    batch: _CycloneTimeScanBatch,
    local_idx: int,
    options: _CycloneHistoryFitOptions,
) -> np.ndarray:
    if signal_t is not None:
        return signal_t[:, local_idx] if signal_t.ndim > 1 else signal_t
    sel_local = options.hooks.mode_selection(
        ky_index=local_idx,
        kx_index=0,
        z_index=options.hooks.midplane_index(batch.grid),
    )
    return options.hooks.extract_mode_time_series(
        phi_t, sel_local, method=options.mode_method
    )

def _history_fit_for_local_mode(
    *,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    signal_t: np.ndarray | None,
    t: np.ndarray,
    batch: _CycloneTimeScanBatch,
    local_idx: int,
    stride: int,
    options: _CycloneHistoryFitOptions,
) -> tuple[float, float]:
    if signal_t is None and options.fit_key == "auto":
        return _auto_history_fit_for_local_mode(
            t=t,
            phi_t=phi_t,
            density_t=density_t,
            batch=batch,
            local_idx=local_idx,
            options=options,
        )
    signal = _history_signal_for_local_mode(
        phi_t=phi_t,
        signal_t=signal_t,
        batch=batch,
        local_idx=local_idx,
        options=options,
    )
    return options.fit_policy.fit_signal(
        signal,
        idx=batch.batch_start + local_idx,
        dt=batch.dt,
        stride=stride,
        params=options.params,
        diagnostic_norm=options.diagnostic_norm,
    )

def _append_cyclone_history_fit_results(
    batch: _CycloneTimeScanBatch,
    *,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    stride: int,
    options: _CycloneHistoryFitOptions,
    output: _CycloneScanOutput,
) -> None:
    signal_t = phi_t if options.mode_only and phi_t.ndim == 2 else None
    t = np.arange(phi_t.shape[0]) * batch.dt * stride
    for local_idx in range(batch.valid_count):
        ky_val = batch.ky_slice[local_idx]
        gamma, omega = _history_fit_for_local_mode(
            phi_t=phi_t,
            density_t=density_t,
            signal_t=signal_t,
            t=t,
            batch=batch,
            local_idx=local_idx,
            stride=stride,
            options=options,
        )
        _resolve_and_append_cyclone_fit(
            gamma=gamma,
            omega=omega,
            ky_val=float(ky_val),
            batch=batch,
            options=options,
            output=output,
        )

def _append_cyclone_streaming_batch_if_requested(
    batch: _CycloneTimeScanBatch,
    *,
    time_cfg: Any | None,
    run_options: _CycloneTimeRunOptions,
    fit_options: _CycloneHistoryFitOptions,
    output: _CycloneScanOutput,
) -> bool:
    """Append Diffrax streaming fits when the batch selected that path."""

    if time_cfg is None or not time_cfg.use_diffrax or not run_options.streaming_fit:
        return False
    _append_cyclone_streaming_results(
        batch,
        geom=run_options.geom,
        params=run_options.params,
        terms=run_options.terms,
        time_cfg=time_cfg,
        tmin=fit_options.tmin,
        tmax=fit_options.tmax,
        start_fraction=fit_options.start_fraction,
        window_fraction=fit_options.window_fraction,
        mode_method=run_options.mode_method,
        streaming_amp_floor=run_options.streaming_amp_floor,
        diagnostic_norm=run_options.diagnostic_norm,
        hooks=run_options.hooks,
        show_progress=run_options.show_progress,
        gammas=output.gammas,
        omegas=output.omegas,
        ky_out=output.ky,
    )
    return True

def _append_cyclone_integrated_history_results(
    batch: _CycloneTimeScanBatch,
    *,
    time_cfg: Any | None,
    run_options: _CycloneTimeRunOptions,
    fit_options: _CycloneHistoryFitOptions,
    output: _CycloneScanOutput,
) -> None:
    """Integrate one Cyclone time batch and append fitted local modes."""

    phi_t_np, density_np, stride = _integrate_cyclone_time_history(
        batch,
        geom=run_options.geom,
        params=run_options.params,
        terms=run_options.terms,
        time_cfg=time_cfg,
        method=run_options.method,
        mode_method=run_options.mode_method,
        mode_only=run_options.mode_only,
        sample_stride=run_options.sample_stride,
        fit_key=run_options.fit_key,
        need_density=run_options.need_density,
        use_jit=run_options.use_jit,
        hooks=run_options.hooks,
        show_progress=run_options.show_progress,
    )
    _append_cyclone_history_fit_results(
        batch,
        phi_t=phi_t_np,
        density_t=density_np,
        stride=stride,
        options=fit_options,
        output=output,
    )

def _append_cyclone_time_batch_results(
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    run_options: _CycloneTimeRunOptions,
    fit_options: _CycloneHistoryFitOptions,
    output: _CycloneScanOutput,
) -> None:
    batch = _prepare_cyclone_time_batch(
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        use_batch=run_options.use_batch,
        grid_full=run_options.grid_full,
        geom=run_options.geom,
        params=run_options.params,
        init_cfg=run_options.init_cfg,
        n_laguerre=run_options.n_laguerre,
        n_hermite=run_options.n_hermite,
        dt=run_options.dt,
        steps=run_options.steps,
        hooks=run_options.hooks,
    )
    time_cfg_i = _cyclone_time_config_for_batch(
        run_options.time_cfg,
        dt=batch.dt,
        steps=batch.steps,
        sample_stride=run_options.sample_stride,
    )
    if _append_cyclone_streaming_batch_if_requested(
        batch,
        time_cfg=time_cfg_i,
        run_options=run_options,
        fit_options=fit_options,
        output=output,
    ):
        return

    _append_cyclone_integrated_history_results(
        batch,
        time_cfg=time_cfg_i,
        run_options=run_options,
        fit_options=fit_options,
        output=output,
    )

def _build_cyclone_time_run_options(
    *,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    terms: Any,
    cfg: Any,
    time_cfg: Any | None,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    sample_stride: int | None,
    mode_method: str,
    mode_only: bool,
    fit_key: str,
    need_density: bool,
    diagnostic_norm: str,
    use_jit: bool,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    use_batch: bool,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> _CycloneTimeRunOptions:
    """Pack Cyclone time-scan controls that are shared by every ky batch."""

    return _CycloneTimeRunOptions(
        ky_values=ky_values,
        grid_full=grid_full,
        geom=geom,
        params=params,
        terms=terms,
        cfg=cfg,
        time_cfg=time_cfg,
        init_cfg=init_cfg,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        mode_method=mode_method,
        mode_only=mode_only,
        fit_key=fit_key,
        need_density=need_density,
        diagnostic_norm=diagnostic_norm,
        use_jit=use_jit,
        ky_batch=ky_batch,
        fixed_batch_shape=fixed_batch_shape,
        streaming_fit=streaming_fit,
        streaming_amp_floor=streaming_amp_floor,
        use_batch=use_batch,
        hooks=hooks,
        show_progress=show_progress,
    )

def _build_cyclone_history_fit_options(
    *,
    n_laguerre: int,
    n_hermite: int,
    method: str,
    params: Any,
    cfg: Any,
    time_cfg: Any | None,
    krylov_cfg: Any | None,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    max_amp_fraction: float,
    max_fraction: float,
    end_fraction: float,
    phase_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    min_slope: float | None,
    min_slope_frac: float,
    slope_var_weight: float,
    window_method: str,
    mode_method: str,
    mode_only: bool,
    fit_key: str,
    diagnostic_norm: str,
    auto_solver: bool,
    fit_policy: Any,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> _CycloneHistoryFitOptions:
    """Pack growth/frequency fitting controls for Cyclone time histories."""

    return _CycloneHistoryFitOptions(
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        method=method,
        params=params,
        cfg=cfg,
        time_cfg=time_cfg,
        krylov_cfg=krylov_cfg,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        max_amp_fraction=max_amp_fraction,
        max_fraction=max_fraction,
        end_fraction=end_fraction,
        phase_weight=phase_weight,
        length_weight=length_weight,
        min_r2=min_r2,
        late_penalty=late_penalty,
        min_slope=min_slope,
        min_slope_frac=min_slope_frac,
        slope_var_weight=slope_var_weight,
        window_method=window_method,
        mode_method=mode_method,
        mode_only=mode_only,
        fit_key=fit_key,
        diagnostic_norm=diagnostic_norm,
        auto_solver=auto_solver,
        fit_policy=fit_policy,
        hooks=hooks,
        show_progress=show_progress,
    )

def _prepare_cyclone_time_scan_controls(
    inputs: _CycloneTimeScanInputs,
) -> _CycloneTimeScanControls:
    """Build all shared Cyclone time-scan controls before batch execution."""

    run_options = _build_cyclone_time_run_options(
        ky_values=inputs.ky_values,
        grid_full=inputs.grid_full,
        geom=inputs.geom,
        params=inputs.params,
        terms=inputs.terms,
        cfg=inputs.cfg,
        time_cfg=inputs.time_cfg,
        init_cfg=inputs.init_cfg,
        n_laguerre=inputs.n_laguerre,
        n_hermite=inputs.n_hermite,
        dt=inputs.dt,
        steps=inputs.steps,
        method=inputs.method,
        sample_stride=inputs.sample_stride,
        mode_method=inputs.mode_method,
        mode_only=inputs.mode_only,
        fit_key=inputs.fit_key,
        need_density=inputs.need_density,
        diagnostic_norm=inputs.diagnostic_norm,
        use_jit=inputs.use_jit,
        ky_batch=inputs.ky_batch,
        fixed_batch_shape=inputs.fixed_batch_shape,
        streaming_fit=inputs.streaming_fit,
        streaming_amp_floor=inputs.streaming_amp_floor,
        use_batch=inputs.use_batch,
        hooks=inputs.hooks,
        show_progress=inputs.show_progress,
    )
    fit_options = _build_cyclone_history_fit_options(
        n_laguerre=inputs.n_laguerre,
        n_hermite=inputs.n_hermite,
        method=inputs.method,
        params=inputs.params,
        cfg=inputs.cfg,
        time_cfg=inputs.time_cfg,
        krylov_cfg=inputs.krylov_cfg,
        tmin=inputs.tmin,
        tmax=inputs.tmax,
        window_fraction=inputs.window_fraction,
        min_points=inputs.min_points,
        start_fraction=inputs.start_fraction,
        growth_weight=inputs.growth_weight,
        require_positive=inputs.require_positive,
        min_amp_fraction=inputs.min_amp_fraction,
        max_amp_fraction=inputs.max_amp_fraction,
        max_fraction=inputs.max_fraction,
        end_fraction=inputs.end_fraction,
        phase_weight=inputs.phase_weight,
        length_weight=inputs.length_weight,
        min_r2=inputs.min_r2,
        late_penalty=inputs.late_penalty,
        min_slope=inputs.min_slope,
        min_slope_frac=inputs.min_slope_frac,
        slope_var_weight=inputs.slope_var_weight,
        window_method=inputs.window_method,
        mode_method=inputs.mode_method,
        mode_only=inputs.mode_only,
        fit_key=inputs.fit_key,
        diagnostic_norm=inputs.diagnostic_norm,
        auto_solver=inputs.auto_solver,
        fit_policy=inputs.fit_policy,
        hooks=inputs.hooks,
        show_progress=inputs.show_progress,
    )
    return _CycloneTimeScanControls(
        run_options=run_options,
        fit_options=fit_options,
    )

def _run_cyclone_time_scan_batches(
    *,
    run_options: _CycloneTimeRunOptions,
    fit_options: _CycloneHistoryFitOptions,
) -> _CycloneScanOutput:
    """Execute all Cyclone time-scan batches and collect fitted rows."""

    output = _CycloneScanOutput.empty()
    for batch_start, ky_slice, valid_count in _iter_cyclone_time_scan_batches(
        run_options
    ):
        _append_cyclone_time_batch_results(
            batch_start=batch_start,
            ky_slice=ky_slice,
            valid_count=valid_count,
            run_options=run_options,
            fit_options=fit_options,
            output=output,
        )
    return output

def _cyclone_time_scan_result(
    output: _CycloneScanOutput,
    *,
    hooks: CycloneScanHooks,
) -> CycloneScanResult:
    """Pack fitted Cyclone scan rows into the public result type."""

    return hooks.cyclone_scan_result(
        ky=np.array(output.ky),
        gamma=np.array(output.gammas),
        omega=np.array(output.omegas),
    )

def run_time_cyclone_scan(
    *,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    terms: Any,
    cfg: Any,
    time_cfg: Any | None,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    krylov_cfg: Any | None,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    max_amp_fraction: float,
    max_fraction: float,
    end_fraction: float,
    phase_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    min_slope: float | None,
    min_slope_frac: float,
    slope_var_weight: float,
    window_method: str,
    mode_method: str,
    mode_only: bool,
    sample_stride: int | None,
    fit_key: str,
    need_density: bool,
    diagnostic_norm: str,
    use_jit: bool,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    auto_solver: bool,
    use_batch: bool,
    fit_policy: Any,
    hooks: CycloneScanHooks,
    show_progress: bool,
) -> CycloneScanResult:
    """Run the standard Cyclone scan time-integration branches."""

    inputs = _cyclone_time_scan_inputs_from_locals(locals())
    controls = _prepare_cyclone_time_scan_controls(inputs)
    output = _run_cyclone_time_scan_batches(
        run_options=controls.run_options,
        fit_options=controls.fit_options,
    )
    return _cyclone_time_scan_result(output, hooks=hooks)

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

@dataclass(frozen=True)
class _CycloneReferencePolicy:
    """Geometry and diagnostic defaults after reference-alignment resolution."""

    reference_aligned: bool
    geometry_config: Any
    diagnostic_norm: str
    mode_method: str

@dataclass(frozen=True)
class _CycloneScanSolverPolicy:
    """Solver, fit-signal, and batching policy for a Cyclone scan."""

    solver_key: str
    fit_key: str
    auto_solver: bool
    mode_method: str
    mode_only: bool
    streaming_fit: bool
    need_density: bool
    use_batch: bool

@dataclass(frozen=True)
class _CycloneScanExecutionOptions:
    """Resolved numerical controls passed to the Cyclone scan dispatcher."""

    ky_values: np.ndarray
    Nl: int
    Nm: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    time_cfg: TimeConfig | None
    krylov_cfg: KrylovConfig | None
    tmin: float | None
    tmax: float | None
    sample_stride: int | None
    use_jit: bool
    ky_batch: int
    fixed_batch_shape: bool
    streaming_amp_floor: float
    mode_follow: bool
    show_progress: bool

@dataclass(frozen=True)
class _CycloneScanRequest:
    ky_values: np.ndarray
    Nl: int
    Nm: int
    dt: float | np.ndarray
    steps: int | np.ndarray
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
    mode_only: bool
    terms: LinearTerms | None
    sample_stride: int | None
    fit_signal: str
    diagnostic_norm: str
    use_jit: bool
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    mode_follow: bool
    reference_aligned: bool | None
    show_progress: bool

def _cyclone_scan_request_from_locals(values: dict[str, Any]) -> _CycloneScanRequest:
    """Pack public ``run_cyclone_scan`` arguments once for internal routing."""

    return _CycloneScanRequest(
        **{field.name: values[field.name] for field in fields(_CycloneScanRequest)}
    )

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

def _resolve_cyclone_reference_policy(
    cfg: CycloneBaseCase,
    *,
    reference_aligned: bool | None,
    diagnostic_norm: str,
    mode_method: str,
) -> _CycloneReferencePolicy:
    reference_aligned_use = (
        bool(cfg.reference_aligned)
        if reference_aligned is None
        else bool(reference_aligned)
    )
    geom_cfg = cfg.geometry
    diagnostic_norm_use = diagnostic_norm
    mode_method_use = mode_method
    if reference_aligned_use:
        geom_cfg = replace(geom_cfg, drift_scale=1.0)
        if diagnostic_norm_use == "none":
            diagnostic_norm_use = "rho_star"
        if mode_method_use not in {"z_index", "max"}:
            mode_method_use = "z_index"
    return _CycloneReferencePolicy(
        reference_aligned=reference_aligned_use,
        geometry_config=geom_cfg,
        diagnostic_norm=diagnostic_norm_use,
        mode_method=mode_method_use,
    )

def _resolve_cyclone_scan_solver_policy(
    *,
    solver: str,
    fit_signal: str,
    reference_aligned: bool,
    mode_method: str,
    mode_only: bool,
    streaming_fit: bool,
    ky_batch: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    tmin: float | None,
    tmax: float | None,
) -> _CycloneScanSolverPolicy:
    solver_key = normalize_solver_key(solver)
    fit_key = normalize_fit_signal(fit_signal)
    auto_solver = solver_key == "auto"
    if auto_solver:
        solver_key = "explicit_time" if reference_aligned else "time"
    streaming_fit_use, mode_only_use = apply_auto_fit_scan_policy(
        fit_key,
        streaming_fit=streaming_fit,
        mode_only=mode_only,
    )
    mode_method_use = resolve_scan_mode_method(
        mode_method,
        mode_only=mode_only_use,
    )
    return _CycloneScanSolverPolicy(
        solver_key=solver_key,
        fit_key=fit_key,
        auto_solver=auto_solver,
        mode_method=mode_method_use,
        mode_only=mode_only_use,
        streaming_fit=streaming_fit_use,
        need_density=fit_key in {"density", "auto"},
        use_batch=should_use_ky_batch(
            ky_batch=ky_batch,
            solver_key=solver_key,
            dt=dt,
            steps=steps,
            tmin=tmin,
            tmax=tmax,
        ),
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
    reference_policy = _resolve_cyclone_reference_policy(
        cfg_use,
        reference_aligned=reference_aligned,
        diagnostic_norm=diagnostic_norm,
        mode_method=mode_method,
    )
    geom = SAlphaGeometry.from_config(reference_policy.geometry_config)
    params_use = (
        _default_cyclone_scan_params(cfg_use, geom, n_hermite=Nm)
        if params is None
        else params
    )
    terms_use = _default_cyclone_scan_terms(cfg_use) if terms is None else terms
    solver_policy = _resolve_cyclone_scan_solver_policy(
        solver=solver,
        fit_signal=fit_signal,
        reference_aligned=reference_policy.reference_aligned,
        mode_method=reference_policy.mode_method,
        mode_only=mode_only,
        streaming_fit=streaming_fit,
        ky_batch=ky_batch,
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
        reference_aligned=reference_policy.reference_aligned,
        diagnostic_norm=reference_policy.diagnostic_norm,
        solver_key=solver_policy.solver_key,
        fit_key=solver_policy.fit_key,
        auto_solver=solver_policy.auto_solver,
        mode_method=solver_policy.mode_method,
        mode_only=solver_policy.mode_only,
        streaming_fit=solver_policy.streaming_fit,
        need_density=solver_policy.need_density,
        use_batch=solver_policy.use_batch,
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

def _dispatch_cyclone_scan(
    setup: _CycloneScanSetup,
    options: _CycloneScanExecutionOptions,
) -> CycloneScanResult:
    """Dispatch a resolved Cyclone scan to the selected numerical path."""

    if setup.solver_key == "krylov":
        return _run_cyclone_scan_krylov_branch(
            setup=setup,
            ky_values=options.ky_values,
            Nl=options.Nl,
            Nm=options.Nm,
            mode_follow=options.mode_follow,
            krylov_cfg=options.krylov_cfg,
            show_progress=options.show_progress,
        )

    if setup.solver_key == "explicit_time":
        return _run_cyclone_scan_explicit_branch(
            setup=setup,
            ky_values=options.ky_values,
            Nl=options.Nl,
            Nm=options.Nm,
            dt=options.dt,
            steps=options.steps,
            time_cfg=options.time_cfg,
            krylov_cfg=options.krylov_cfg,
            show_progress=options.show_progress,
        )
    return _run_cyclone_scan_time_branch(
        setup=setup,
        ky_values=options.ky_values,
        Nl=options.Nl,
        Nm=options.Nm,
        dt=options.dt,
        steps=options.steps,
        method=options.method,
        time_cfg=options.time_cfg,
        krylov_cfg=options.krylov_cfg,
        tmin=options.tmin,
        tmax=options.tmax,
        sample_stride=options.sample_stride,
        use_jit=options.use_jit,
        ky_batch=options.ky_batch,
        fixed_batch_shape=options.fixed_batch_shape,
        streaming_amp_floor=options.streaming_amp_floor,
        show_progress=options.show_progress,
    )

def _cyclone_scan_fit_policy_from_request(
    request: _CycloneScanRequest,
) -> ScanFitWindowPolicy:
    return _build_cyclone_scan_fit_policy(
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

def _cyclone_scan_setup_from_request(
    request: _CycloneScanRequest,
    fit_policy: ScanFitWindowPolicy,
) -> _CycloneScanSetup:
    return _build_cyclone_scan_setup(
        cfg=request.cfg,
        params=request.params,
        terms=request.terms,
        Nm=request.Nm,
        solver=request.solver,
        fit_signal=request.fit_signal,
        diagnostic_norm=request.diagnostic_norm,
        mode_method=request.mode_method,
        mode_only=request.mode_only,
        streaming_fit=request.streaming_fit,
        ky_batch=request.ky_batch,
        dt=request.dt,
        steps=request.steps,
        tmin=request.tmin,
        tmax=request.tmax,
        reference_aligned=request.reference_aligned,
        fit_policy=fit_policy,
    )

def _cyclone_scan_execution_options_from_request(
    request: _CycloneScanRequest,
) -> _CycloneScanExecutionOptions:
    return _CycloneScanExecutionOptions(
        ky_values=np.asarray(request.ky_values, dtype=float),
        Nl=request.Nl,
        Nm=request.Nm,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        time_cfg=request.time_cfg,
        krylov_cfg=request.krylov_cfg,
        tmin=request.tmin,
        tmax=request.tmax,
        sample_stride=request.sample_stride,
        use_jit=request.use_jit,
        ky_batch=request.ky_batch,
        fixed_batch_shape=request.fixed_batch_shape,
        streaming_amp_floor=request.streaming_amp_floor,
        mode_follow=request.mode_follow,
        show_progress=request.show_progress,
    )

def _run_cyclone_scan_request(request: _CycloneScanRequest) -> CycloneScanResult:
    fit_policy = _cyclone_scan_fit_policy_from_request(request)
    setup = _cyclone_scan_setup_from_request(request, fit_policy)
    options = _cyclone_scan_execution_options_from_request(request)
    return _dispatch_cyclone_scan(setup, options)

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

    return _run_cyclone_scan_request(_cyclone_scan_request_from_locals(locals()))


# KBM single-mode and beta-scan benchmark implementations live with the public
# benchmark facade while the validation package is being retired.
@dataclass(frozen=True)
class _KBMLinearSetup:
    cfg: KBMBaseCase
    beta: float
    geom: Any
    grid_full: Any
    params: LinearParams
    terms: LinearTerms
    diagnostic_norm: str
    reference_aligned: bool
    fit_key: str


@dataclass(frozen=True)
class _KBMLinearState:
    grid: Any
    selection: ModeSelection
    cache: Any
    state: Any


@dataclass(frozen=True)
class _KBMLinearRunOptions:
    ky_target: float
    Nl: int
    Nm: int
    dt: float
    steps: int
    method: str
    time_cfg: TimeConfig | None
    krylov_cfg: KrylovConfig | None
    kbm_target_factors: Sequence[float] | None
    kbm_beta_transition: float | None
    auto_window: bool
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    mode_method: str
    sample_stride: int | None
    density_species_index: int
    show_progress: bool


@dataclass(frozen=True)
class _KBMLinearRequest:
    ky_target: float
    beta_value: float | None
    Nl: int
    Nm: int
    dt: float
    steps: int
    method: str
    params: LinearParams | None
    cfg: KBMBaseCase | None
    time_cfg: TimeConfig | None
    solver: str
    krylov_cfg: KrylovConfig | None
    kbm_target_factors: Sequence[float] | None
    kbm_beta_transition: float | None
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
    terms: LinearTerms | None
    sample_stride: int | None
    fit_signal: str
    streaming_fit: bool
    init_species_index: int
    density_species_index: int
    diagnostic_norm: str
    fapar_override: float | None
    apar_beta_scale: float | None
    ampere_g0_scale: float | None
    bpar_beta_scale: float | None
    reference_aligned: bool | None
    show_progress: bool


def _kbm_linear_request_from_locals(values: dict[str, Any]) -> _KBMLinearRequest:
    """Pack public ``run_kbm_linear`` arguments once for internal routing."""

    return _KBMLinearRequest(
        **{field.name: values[field.name] for field in fields(_KBMLinearRequest)}
    )


def _resolve_kbm_fit_signal(fit_signal: str) -> str:
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    return fit_key


def _validate_kbm_species_indices(
    *, init_species_index: int, density_species_index: int, nspecies: int = 2
) -> None:
    if init_species_index < 0 or init_species_index >= nspecies:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= nspecies:
        raise ValueError("density_species_index out of range for kinetic species")


def _resolve_kbm_linear_setup(
    *,
    cfg: KBMBaseCase | None,
    beta_value: float | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    diagnostic_norm: str,
    fit_signal: str,
    reference_aligned: bool | None,
    Nm: int,
    fapar_override: float | None,
    apar_beta_scale: float | None,
    ampere_g0_scale: float | None,
    bpar_beta_scale: float | None,
) -> _KBMLinearSetup:
    cfg_in = cfg or KBMBaseCase()
    beta_use = float(cfg_in.model.beta) if beta_value is None else float(beta_value)
    cfg_use = replace(cfg_in, model=replace(cfg_in.model, beta=beta_use))
    geom = build_flux_tube_geometry(cfg_use.geometry)
    grid_full = build_spectral_grid(apply_geometry_grid_defaults(geom, cfg_use.grid))
    terms_use = terms if terms is not None else LinearTerms(bpar=0.0)
    reference_aligned_use = bool(
        True if reference_aligned is None else reference_aligned
    )
    diagnostic_norm_use = diagnostic_norm
    if reference_aligned_use and diagnostic_norm_use == "none":
        diagnostic_norm_use = "rho_star"
    damp_ends_amp, damp_ends_widthfrac = _linked_boundary_end_damping(
        reference_aligned_use
    )
    params_use = params
    if params_use is None:
        params_use = _two_species_params(
            cfg_use.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KBM_OMEGA_D_SCALE,
            omega_star_scale=KBM_OMEGA_STAR_SCALE,
            rho_star=KBM_RHO_STAR,
            beta_override=beta_use,
            fapar_override=fapar_override,
            apar_beta_scale=apar_beta_scale,
            ampere_g0_scale=ampere_g0_scale,
            bpar_beta_scale=bpar_beta_scale,
            damp_ends_amp=damp_ends_amp,
            damp_ends_widthfrac=damp_ends_widthfrac,
            nhermite=Nm,
        )
    return _KBMLinearSetup(
        cfg=cfg_use,
        beta=beta_use,
        geom=geom,
        grid_full=grid_full,
        params=params_use,
        terms=terms_use,
        diagnostic_norm=diagnostic_norm_use,
        reference_aligned=reference_aligned_use,
        fit_key=_resolve_kbm_fit_signal(fit_signal),
    )


def _prepare_kbm_linear_state(
    setup: _KBMLinearSetup,
    *,
    ky_target: float,
    Nl: int,
    Nm: int,
    init_species_index: int,
) -> _KBMLinearState:
    ky_index = select_ky_index(np.asarray(setup.grid_full.ky), ky_target)
    grid = select_ky_grid(setup.grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    cache = build_linear_cache(grid, setup.geom, setup.params, Nl, Nm)
    G0 = np.zeros(
        (2, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
    )
    G0_single = _build_initial_condition(
        grid,
        setup.geom,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=setup.cfg.init,
    )
    G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)
    return _KBMLinearState(
        grid=grid,
        selection=sel,
        cache=cache,
        state=jnp.asarray(G0),
    )


def _fit_kbm_signal_with_window(
    signal: np.ndarray,
    t_arr: np.ndarray,
    *,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> tuple[float, float]:
    use_auto = auto_window and tmin is None and tmax is None
    if not use_auto and not scan_window_valid(t_arr, tmin, tmax):
        use_auto = True
    auto_fit_kwargs: dict[str, Any] = {
        "window_fraction": window_fraction,
        "min_points": min_points,
        "start_fraction": start_fraction,
        "growth_weight": growth_weight,
        "require_positive": require_positive,
        "min_amp_fraction": min_amp_fraction,
    }
    if use_auto:
        gamma_val, omega_val, _tmin, _tmax = fit_growth_rate_auto(
            t_arr, signal, **auto_fit_kwargs
        )
    else:
        try:
            gamma_val, omega_val = fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)
        except ValueError:
            gamma_val, omega_val, _tmin, _tmax = fit_growth_rate_auto(
                t_arr, signal, **auto_fit_kwargs
            )
    return gamma_val, omega_val


def _resolve_kbm_time_config(
    time_cfg: TimeConfig | None,
    *,
    dt: float,
    steps: int,
    stride: int,
    sample_stride: int | None,
) -> TimeConfig | None:
    if time_cfg is None:
        return None
    time_cfg_use = replace(time_cfg, dt=dt, t_max=dt * steps)
    if sample_stride is not None:
        time_cfg_use = replace(time_cfg_use, sample_stride=stride)
    return time_cfg_use


def _integrate_kbm_configured_history(
    state: _KBMLinearState,
    setup: _KBMLinearSetup,
    *,
    time_cfg: TimeConfig,
    dt: float,
    steps: int,
    method: str,
    density_species_index: int,
    mode_method: str,
    show_progress: bool,
    stride: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    stride = int(time_cfg.sample_stride)
    if time_cfg.use_diffrax:
        save_field = "phi+density" if setup.fit_key in {"density", "auto"} else "phi"
        _, phi_out = integrate_linear_from_config(
            state.state,
            state.grid,
            setup.geom,
            setup.params,
            time_cfg,
            cache=state.cache,
            terms=setup.terms,
            save_mode=state.selection if setup.fit_key == "phi" else None,
            mode_method=mode_method,
            save_field=save_field,
            density_species_index=density_species_index
            if setup.fit_key in {"density", "auto"}
            else None,
        )
        if setup.fit_key in {"density", "auto"}:
            return np.asarray(phi_out[0]), np.asarray(phi_out[1]), stride
        return np.asarray(phi_out), None, stride

    if setup.fit_key in {"density", "auto"}:
        diag_out = integrate_linear_diagnostics(
            state.state,
            state.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=method,
            cache=state.cache,
            terms=setup.terms,
            sample_stride=stride,
            species_index=density_species_index,
        )
        density_np = None if len(diag_out) <= 2 else np.asarray(diag_out[2])
        return np.asarray(diag_out[1]), density_np, stride

    _, phi_out_time = integrate_linear(
        state.state,
        state.grid,
        setup.geom,
        setup.params,
        dt=dt,
        steps=steps,
        method=method,
        cache=state.cache,
        terms=setup.terms,
        sample_stride=stride,
        show_progress=show_progress,
    )
    return np.asarray(phi_out_time), None, stride


def _integrate_kbm_fixed_history(
    state: _KBMLinearState,
    setup: _KBMLinearSetup,
    *,
    dt: float,
    steps: int,
    method: str,
    density_species_index: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    if setup.fit_key in {"density", "auto"}:
        diag_out = integrate_linear_diagnostics(
            state.state,
            state.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=method,
            cache=state.cache,
            terms=setup.terms,
            sample_stride=stride,
            species_index=density_species_index,
        )
        density_np = None if len(diag_out) <= 2 else np.asarray(diag_out[2])
        return np.asarray(diag_out[1]), density_np, stride

    _, phi_out_time = integrate_linear(
        state.state,
        state.grid,
        setup.geom,
        setup.params,
        dt=dt,
        steps=steps,
        method=method,
        cache=state.cache,
        terms=setup.terms,
        sample_stride=stride,
    )
    return np.asarray(phi_out_time), None, stride


def _integrate_kbm_saved_history(
    state: _KBMLinearState,
    setup: _KBMLinearSetup,
    *,
    time_cfg: TimeConfig | None,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    density_species_index: int,
    mode_method: str,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    stride = 1 if sample_stride is None else int(sample_stride)
    time_cfg_use = _resolve_kbm_time_config(
        time_cfg,
        dt=dt,
        steps=steps,
        stride=stride,
        sample_stride=sample_stride,
    )
    if time_cfg_use is not None:
        return _integrate_kbm_configured_history(
            state,
            setup,
            time_cfg=time_cfg_use,
            dt=dt,
            steps=steps,
            method=method,
            density_species_index=density_species_index,
            mode_method=mode_method,
            show_progress=show_progress,
            stride=stride,
        )
    return _integrate_kbm_fixed_history(
        state,
        setup,
        dt=dt,
        steps=steps,
        method=method,
        density_species_index=density_species_index,
        stride=stride,
    )


def _fit_kbm_auto_history(
    state: _KBMLinearState,
    setup: _KBMLinearSetup,
    *,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    dt: float,
    stride: int,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> tuple[float, float]:
    _signal, _name, gamma, omega = _select_fit_signal_auto(
        np.arange(phi_t.shape[0]) * dt * stride,
        phi_t,
        density_t,
        state.selection,
        mode_method=mode_method,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        max_amp_fraction=0.9,
        window_method="loglinear",
        max_fraction=0.8,
        end_fraction=0.9,
        num_windows=8,
        phase_weight=0.2,
        length_weight=0.05,
        min_r2=0.0,
        late_penalty=0.1,
        min_slope=None,
        min_slope_frac=0.0,
        slope_var_weight=0.0,
    )
    return gamma, omega


def _fit_kbm_saved_history(
    state: _KBMLinearState,
    setup: _KBMLinearSetup,
    *,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    dt: float,
    stride: int,
    mode_method: str,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> tuple[float, float]:
    density_np = (
        phi_t if setup.fit_key == "density" and density_t is None else density_t
    )
    if setup.fit_key == "auto":
        gamma, omega = _fit_kbm_auto_history(
            state,
            setup,
            phi_t=phi_t,
            density_t=density_np,
            dt=dt,
            stride=stride,
            mode_method=mode_method,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
        )
    else:
        signal = _select_fit_signal(
            phi_t,
            density_np,
            state.selection,
            fit_signal=setup.fit_key,
            mode_method=mode_method,
        )
        t_out = np.arange(signal.shape[0]) * dt * stride
        gamma, omega = _fit_kbm_signal_with_window(
            signal,
            t_out,
            auto_window=auto_window,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
        )
    return _normalize_growth_rate(
        gamma, omega, setup.params, setup.diagnostic_norm
    )


# KBM explicit-time and Krylov path policies live with the runner.
def fit_kbm_window(
    signal: np.ndarray,
    t_arr: np.ndarray,
    *,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> tuple[float, float]:
    """Fit a KBM time trace with the runner's automatic-window fallback policy."""

    use_auto = auto_window and tmin is None and tmax is None
    if not use_auto and not scan_window_valid(t_arr, tmin, tmax):
        use_auto = True
    auto_fit_kwargs: dict[str, Any] = {
        "window_fraction": window_fraction,
        "min_points": min_points,
        "start_fraction": start_fraction,
        "growth_weight": growth_weight,
        "require_positive": require_positive,
        "min_amp_fraction": min_amp_fraction,
    }
    if use_auto:
        gamma_val, omega_val, _tmin, _tmax = fit_growth_rate_auto(
            t_arr, signal, **auto_fit_kwargs
        )
    else:
        try:
            gamma_val, omega_val = fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)
        except ValueError:
            gamma_val, omega_val, _tmin, _tmax = fit_growth_rate_auto(
                t_arr, signal, **auto_fit_kwargs
            )
    return gamma_val, omega_val


def _build_kbm_explicit_time_config(
    *,
    dt: float,
    steps: int,
    time_cfg: Any,
    sample_stride: int | None,
) -> ExplicitTimeConfig:
    """Build the explicit-time policy used by reference-aligned KBM runs."""

    return ExplicitTimeConfig(
        dt=dt,
        t_max=dt * steps,
        sample_stride=max(int(sample_stride or 1), 1),
        fixed_dt=bool(time_cfg.fixed_dt) if time_cfg is not None else False,
        use_dealias_mask=bool(getattr(time_cfg, "use_dealias_mask", False))
        if time_cfg is not None
        else False,
        dt_min=float(time_cfg.dt_min) if time_cfg is not None else 1.0e-7,
        dt_max=float(time_cfg.dt_max)
        if (time_cfg is not None and time_cfg.dt_max is not None)
        else None,
        cfl=float(time_cfg.cfl) if time_cfg is not None else 0.9,
        cfl_fac=(
            resolve_cfl_fac(str(time_cfg.method), time_cfg.cfl_fac)
            if time_cfg is not None
            else float(ExplicitTimeConfig.cfl_fac)
        ),
    )


def _fit_kbm_projected_rates(
    *,
    phi_t_np: np.ndarray,
    gamma_t: Any,
    omega_t: Any,
    t_out: np.ndarray,
    sel: Any,
    mode_method: str,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> tuple[float, float]:
    """Fit KBM rates from instantaneous diagnostics with robust fallbacks."""

    try:
        gamma, omega, _g_t, _o_t, _t_mid = instantaneous_growth_rate_from_phi(
            phi_t_np,
            t_out,
            sel,
            navg_fraction=0.5,
            mode_method=mode_method,
        )
        return gamma, omega
    except ValueError:
        try:
            gamma, omega, _g_t, _o_t = windowed_growth_rate_from_omega_series(
                np.asarray(gamma_t),
                np.asarray(omega_t),
                sel,
                navg_fraction=0.5,
            )
            return gamma, omega
        except ValueError:
            signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
            return fit_kbm_window(
                signal,
                t_out,
                auto_window=auto_window,
                tmin=tmin,
                tmax=tmax,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )


def _fit_kbm_trace_rates(
    *,
    phi_t_np: np.ndarray,
    t_out: np.ndarray,
    sel: Any,
    mode_method: str,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> tuple[float, float]:
    """Fit KBM rates directly from the selected complex field trace."""

    signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
    if auto_window and tmin is None and tmax is None:
        gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
            t_out,
            signal,
            window_method="fixed",
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
        )
        return gamma, omega
    return fit_kbm_window(
        signal,
        t_out,
        auto_window=auto_window,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
    )


def _fit_kbm_explicit_rates(
    *,
    phi_t_np: np.ndarray,
    gamma_t: Any,
    omega_t: Any,
    t_out: np.ndarray,
    sel: Any,
    mode_method: str,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> tuple[float, float]:
    """Extract the explicit-time KBM growth rate and frequency."""

    if t_out.size <= 1:
        return float("nan"), float("nan")
    if mode_method in {"z_index", "max"}:
        return _fit_kbm_projected_rates(
            phi_t_np=phi_t_np,
            gamma_t=gamma_t,
            omega_t=omega_t,
            t_out=t_out,
            sel=sel,
            mode_method=mode_method,
            auto_window=auto_window,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
        )
    return _fit_kbm_trace_rates(
        phi_t_np=phi_t_np,
        t_out=t_out,
        sel=sel,
        mode_method=mode_method,
        auto_window=auto_window,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
    )


def _dominant_kbm_krylov_eigenpair(
    *,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg: Any,
    omega_target_factor: float,
    shift: Any,
    shift_source: str,
    shift_selection: str,
) -> tuple[Any, Any]:
    """Call the KBM Krylov eigensolver with an explicit target policy."""

    return dominant_eigenpair(
        G0_jax,
        cache,
        params,
        terms=terms,
        v_ref=None,
        select_overlap=False,
        krylov_dim=krylov_cfg.krylov_dim,
        restarts=krylov_cfg.restarts,
        omega_min_factor=krylov_cfg.omega_min_factor,
        omega_target_factor=omega_target_factor,
        omega_cap_factor=krylov_cfg.omega_cap_factor,
        omega_sign=krylov_cfg.omega_sign,
        method=krylov_cfg.method,
        power_iters=krylov_cfg.power_iters,
        power_dt=krylov_cfg.power_dt,
        shift=shift,
        shift_source=shift_source,
        shift_tol=krylov_cfg.shift_tol,
        shift_maxiter=krylov_cfg.shift_maxiter,
        shift_restart=krylov_cfg.shift_restart,
        shift_solve_method=krylov_cfg.shift_solve_method,
        shift_preconditioner=krylov_cfg.shift_preconditioner,
        shift_selection=shift_selection,
        mode_family=krylov_cfg.mode_family,
        fallback_method=krylov_cfg.fallback_method,
        fallback_real_floor=krylov_cfg.fallback_real_floor,
    )


def _collect_kbm_target_candidates(
    *,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg: Any,
    targets: Sequence[float],
) -> tuple[list[Any], list[Any]]:
    """Evaluate the KBM Krylov solve at each target frequency factor."""

    eig_candidates = []
    vec_candidates = []
    for target in targets:
        eig_i, vec_i = _dominant_kbm_krylov_eigenpair(
            G0_jax=G0_jax,
            cache=cache,
            params=params,
            terms=terms,
            krylov_cfg=krylov_cfg,
            omega_target_factor=float(target),
            shift=None,
            shift_source="target",
            shift_selection="targeted",
        )
        eig_candidates.append(eig_i)
        vec_candidates.append(vec_i)
    return eig_candidates, vec_candidates


def _select_kbm_target_candidate(
    *,
    eig_candidates: Sequence[Any],
    beta_use: float,
    beta_transition: float,
) -> int:
    """Choose the KBM branch candidate from beta-continuity or max growth."""

    if len(eig_candidates) >= 2 and np.isfinite(beta_transition):
        return 1 if beta_use >= beta_transition else 0
    eig_arr = np.asarray([complex(np.asarray(e)) for e in eig_candidates])
    growth = np.real(eig_arr)
    finite_growth = np.where(np.isfinite(growth), growth, -np.inf)
    return 0 if np.all(~np.isfinite(growth)) else int(np.nanargmax(finite_growth))


def _run_kbm_multi_target_krylov(
    *,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    beta_use: float,
    cfg_use: Any,
    krylov_cfg: Any,
    targets: Sequence[float],
    kbm_beta_transition: float | None,
) -> tuple[Any, Any]:
    """Run the KBM multi-target branch-continuity Krylov policy."""

    beta_transition = (
        float(cfg_use.model.beta)
        if kbm_beta_transition is None
        else float(kbm_beta_transition)
    )
    eig_candidates, vec_candidates = _collect_kbm_target_candidates(
        G0_jax=G0_jax,
        cache=cache,
        params=params,
        terms=terms,
        krylov_cfg=krylov_cfg,
        targets=targets,
    )
    idx = _select_kbm_target_candidate(
        eig_candidates=eig_candidates,
        beta_use=beta_use,
        beta_transition=beta_transition,
    )
    return eig_candidates[idx], vec_candidates[idx]


def run_kbm_explicit_time_path(
    *,
    G0_jax: Any,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    sel: Any,
    ky_target: float,
    dt: float,
    steps: int,
    time_cfg: Any,
    sample_stride: int | None,
    mode_method: str,
    diagnostic_norm: str,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> LinearRunResult:
    """Run KBM's reference-aligned explicit diagnostics branch."""

    explicit_mode_method = mode_method if mode_method in {"z_index", "max"} else "z_index"
    explicit_time_cfg = _build_kbm_explicit_time_config(
        dt=dt,
        steps=steps,
        time_cfg=time_cfg,
        sample_stride=sample_stride,
    )
    t_arr, phi_t, gamma_t, omega_t, _diag = integrate_linear_explicit_diagnostics(
        G0_jax,
        grid,
        cache,
        params,
        geom,
        explicit_time_cfg,
        terms=terms,
        mode_method=explicit_mode_method,
        z_index=sel.z_index,
        jit=True,
    )
    t_out = np.asarray(t_arr, dtype=float)
    phi_t_np = np.asarray(phi_t)
    gamma, omega = _fit_kbm_explicit_rates(
        phi_t_np=phi_t_np,
        gamma_t=gamma_t,
        omega_t=omega_t,
        t_out=t_out,
        sel=sel,
        mode_method=mode_method,
        auto_window=auto_window,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
    )
    gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    return LinearRunResult(
        t=t_out,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(ky_target),
        selection=sel,
        gamma_t=np.asarray(gamma_t),
        omega_t=np.asarray(omega_t),
    )


def _run_kbm_single_target_krylov(
    *,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg: Any,
    shift_val: Any,
) -> tuple[Any, Any]:
    """Run the KBM Krylov policy with the configured single target."""

    return _dominant_kbm_krylov_eigenpair(
        G0_jax=G0_jax,
        cache=cache,
        params=params,
        terms=terms,
        krylov_cfg=krylov_cfg,
        omega_target_factor=krylov_cfg.omega_target_factor,
        shift=shift_val,
        shift_source=krylov_cfg.shift_source,
        shift_selection=krylov_cfg.shift_selection,
    )


def _kbm_krylov_result(
    *,
    eig: Any,
    vec: Any,
    cache: Any,
    params: Any,
    terms: Any,
    sel: Any,
    ky_target: float,
    diagnostic_norm: str,
    krylov_cfg: Any,
) -> LinearRunResult:
    """Package a KBM Krylov eigenpair as a linear benchmark result."""

    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    if krylov_cfg.omega_sign != 0:
        omega = float(np.sign(krylov_cfg.omega_sign)) * abs(omega)
    gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    term_cfg = linear_terms_to_term_config(terms)
    phi = compute_fields_cached(vec, cache, params, terms=term_cfg).phi
    return LinearRunResult(
        t=np.array([0.0], dtype=float),
        phi_t=np.asarray(phi)[None, ...],
        gamma=gamma,
        omega=omega,
        ky=float(ky_target),
        selection=sel,
    )


def run_kbm_krylov_path(
    *,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    sel: Any,
    ky_target: float,
    beta_use: float,
    cfg_use: Any,
    krylov_cfg: Any,
    kbm_target_factors: Sequence[float] | None,
    kbm_beta_transition: float | None,
    diagnostic_norm: str,
) -> LinearRunResult:
    """Run KBM's single-target or beta-transition multi-target Krylov branch."""

    krylov_cfg_use = krylov_cfg or KBM_KRYLOV_DEFAULT
    shift_val = krylov_cfg_use.shift
    targets: Sequence[float] | None = kbm_target_factors if kbm_target_factors else None
    use_multi_target = _kbm_use_multi_target_krylov(
        krylov_cfg_use,
        targets,
        shift=shift_val,
    )
    if use_multi_target:
        assert targets is not None
        eig, vec = _run_kbm_multi_target_krylov(
            G0_jax=G0_jax,
            cache=cache,
            params=params,
            terms=terms,
            beta_use=beta_use,
            cfg_use=cfg_use,
            krylov_cfg=krylov_cfg_use,
            targets=targets,
            kbm_beta_transition=kbm_beta_transition,
        )
    else:
        eig, vec = _run_kbm_single_target_krylov(
            G0_jax=G0_jax,
            cache=cache,
            params=params,
            terms=terms,
            krylov_cfg=krylov_cfg_use,
            shift_val=shift_val,
        )
    return _kbm_krylov_result(
        eig=eig,
        vec=vec,
        cache=cache,
        params=params,
        terms=terms,
        sel=sel,
        ky_target=ky_target,
        diagnostic_norm=diagnostic_norm,
        krylov_cfg=krylov_cfg_use,
    )


def _run_kbm_explicit_solver_path(
    setup: _KBMLinearSetup,
    state: _KBMLinearState,
    options: _KBMLinearRunOptions,
) -> LinearRunResult:
    return run_kbm_explicit_time_path(
        G0_jax=state.state,
        grid=state.grid,
        cache=state.cache,
        params=setup.params,
        geom=setup.geom,
        terms=setup.terms,
        sel=state.selection,
        ky_target=options.ky_target,
        dt=options.dt,
        steps=options.steps,
        time_cfg=options.time_cfg,
        sample_stride=options.sample_stride,
        mode_method=options.mode_method,
        diagnostic_norm=setup.diagnostic_norm,
        auto_window=options.auto_window,
        tmin=options.tmin,
        tmax=options.tmax,
        window_fraction=options.window_fraction,
        min_points=options.min_points,
        start_fraction=options.start_fraction,
        growth_weight=options.growth_weight,
        require_positive=options.require_positive,
        min_amp_fraction=options.min_amp_fraction,
    )


def _run_kbm_krylov_solver_path(
    setup: _KBMLinearSetup,
    state: _KBMLinearState,
    options: _KBMLinearRunOptions,
) -> LinearRunResult:
    return run_kbm_krylov_path(
        G0_jax=state.state,
        cache=state.cache,
        params=setup.params,
        terms=setup.terms,
        sel=state.selection,
        ky_target=options.ky_target,
        beta_use=setup.beta,
        cfg_use=setup.cfg,
        krylov_cfg=options.krylov_cfg,
        kbm_target_factors=options.kbm_target_factors,
        kbm_beta_transition=options.kbm_beta_transition,
        diagnostic_norm=setup.diagnostic_norm,
    )


def _run_kbm_saved_time_solver_path(
    setup: _KBMLinearSetup,
    state: _KBMLinearState,
    options: _KBMLinearRunOptions,
) -> LinearRunResult:
    phi_t_np, density_np, stride = _integrate_kbm_saved_history(
        state,
        setup,
        time_cfg=options.time_cfg,
        dt=options.dt,
        steps=options.steps,
        method=options.method,
        sample_stride=options.sample_stride,
        density_species_index=options.density_species_index,
        mode_method=options.mode_method,
        show_progress=options.show_progress,
    )
    gamma, omega = _fit_kbm_saved_history(
        state,
        setup,
        phi_t=phi_t_np,
        density_t=density_np,
        dt=options.dt,
        stride=stride,
        mode_method=options.mode_method,
        auto_window=options.auto_window,
        tmin=options.tmin,
        tmax=options.tmax,
        window_fraction=options.window_fraction,
        min_points=options.min_points,
        start_fraction=options.start_fraction,
        growth_weight=options.growth_weight,
        require_positive=options.require_positive,
        min_amp_fraction=options.min_amp_fraction,
    )
    return LinearRunResult(
        t=np.arange(phi_t_np.shape[0]) * options.dt * stride,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(options.ky_target),
        selection=state.selection,
    )


def _kbm_linear_setup_from_request(request: _KBMLinearRequest) -> _KBMLinearSetup:
    return _resolve_kbm_linear_setup(
        cfg=request.cfg,
        beta_value=request.beta_value,
        params=request.params,
        terms=request.terms,
        diagnostic_norm=request.diagnostic_norm,
        fit_signal=request.fit_signal,
        reference_aligned=request.reference_aligned,
        Nm=request.Nm,
        fapar_override=request.fapar_override,
        apar_beta_scale=request.apar_beta_scale,
        ampere_g0_scale=request.ampere_g0_scale,
        bpar_beta_scale=request.bpar_beta_scale,
    )


def _kbm_linear_options_from_request(
    request: _KBMLinearRequest,
) -> _KBMLinearRunOptions:
    return _KBMLinearRunOptions(
        ky_target=float(request.ky_target),
        Nl=request.Nl,
        Nm=request.Nm,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        time_cfg=request.time_cfg,
        krylov_cfg=request.krylov_cfg,
        kbm_target_factors=request.kbm_target_factors,
        kbm_beta_transition=request.kbm_beta_transition,
        auto_window=request.auto_window,
        tmin=request.tmin,
        tmax=request.tmax,
        window_fraction=request.window_fraction,
        min_points=request.min_points,
        start_fraction=request.start_fraction,
        growth_weight=request.growth_weight,
        require_positive=request.require_positive,
        min_amp_fraction=request.min_amp_fraction,
        mode_method=request.mode_method,
        sample_stride=request.sample_stride,
        density_species_index=request.density_species_index,
        show_progress=request.show_progress,
    )


def _run_kbm_linear_request(request: _KBMLinearRequest) -> LinearRunResult:
    setup = _kbm_linear_setup_from_request(request)
    _validate_kbm_species_indices(
        init_species_index=request.init_species_index,
        density_species_index=request.density_species_index,
    )
    state = _prepare_kbm_linear_state(
        setup,
        ky_target=request.ky_target,
        Nl=request.Nl,
        Nm=request.Nm,
        init_species_index=request.init_species_index,
    )
    options = _kbm_linear_options_from_request(request)
    solver_key = select_kbm_solver_auto(
        request.solver,
        ky_target=float(request.ky_target),
        reference_aligned=setup.reference_aligned,
    )
    if solver_key == "explicit_time":
        return _run_kbm_explicit_solver_path(setup, state, options)
    if solver_key == "krylov":
        return _run_kbm_krylov_solver_path(setup, state, options)
    return _run_kbm_saved_time_solver_path(setup, state, options)


def run_kbm_linear(
    ky_target: float = 0.3,
    *,
    beta_value: float | None = None,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "imex2",
    params: LinearParams | None = None,
    cfg: KBMBaseCase | None = None,
    time_cfg: TimeConfig | None = None,
    solver: str = "auto",
    krylov_cfg: KrylovConfig | None = None,
    kbm_target_factors: Sequence[float] | None = (0.7, 1.5),
    kbm_beta_transition: float | None = None,
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
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    fit_signal: str = "auto",
    streaming_fit: bool = False,
    init_species_index: int = 1,
    density_species_index: int = 1,
    diagnostic_norm: str = "none",
    fapar_override: float | None = None,
    apar_beta_scale: float | None = None,
    ampere_g0_scale: float | None = None,
    bpar_beta_scale: float | None = None,
    reference_aligned: bool | None = True,
    show_progress: bool = False,
) -> LinearRunResult:
    """Run a single linear KBM point and return the stored field history."""

    return _run_kbm_linear_request(_kbm_linear_request_from_locals(locals()))

@dataclass(frozen=True)
class _KBMBetaScanSetup:
    """Shared fixed-ky KBM beta-scan state and patchable solver policies."""

    cfg: KBMBaseCase
    grid: Any
    geom: Any
    selection: ModeSelection
    terms: LinearTerms
    reference_aligned: bool
    damp_ends_amp: float
    damp_ends_widthfrac: float
    solver_key: str
    fit_key: str
    streaming_fit: bool
    mode_only: bool
    diagnostic_norm: str
    krylov_cfg: KrylovConfig
    use_continuation: bool
    fit_policy: ScanFitWindowPolicy
    explicit_hooks: KBMBetaExplicitHooks
    krylov_hooks: KBMBetaKrylovHooks
    time_hooks: KBMBetaTimeHooks


@dataclass(frozen=True)
class _KBMBetaSample:
    """One beta point after species parameters, cache, and initial state exist."""

    beta: float
    index: int
    dt: float
    steps: int
    params: LinearParams
    cache: Any
    initial_state: Any
    solver_use: str


@dataclass(frozen=True)
class _KBMBetaContinuation:
    """Krylov continuation state carried between beta points."""

    prev_vec: Any = None
    prev_eig: Any = None


@dataclass(frozen=True)
class _KBMBetaScanOptions:
    ky_target: float
    n_laguerre: int
    n_hermite: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    time_cfg: TimeConfig | None
    kbm_target_factors: Sequence[float] | None
    kbm_beta_transition: float | None
    tmin: float | None
    tmax: float | None
    require_positive: bool
    mode_method: str
    sample_stride: int | None
    streaming_amp_floor: float
    init_species_index: int
    density_species_index: int
    fapar_override: float | None
    apar_beta_scale: float | None
    ampere_g0_scale: float | None
    bpar_beta_scale: float | None


@dataclass
class _KBMBetaScanOutput:
    beta: list[float]
    gamma: list[float]
    omega: list[float]

    @classmethod
    def empty(cls) -> "_KBMBetaScanOutput":
        return cls(beta=[], gamma=[], omega=[])

    def append(self, *, beta: float, gamma: float, omega: float) -> None:
        self.beta.append(float(beta))
        self.gamma.append(float(gamma))
        self.omega.append(float(omega))

    def result(self) -> LinearScanResult:
        return LinearScanResult(
            ky=np.array(self.beta),
            gamma=np.array(self.gamma),
            omega=np.array(self.omega),
        )


@dataclass(frozen=True)
class _KBMBetaScanRequest:
    """Raw public fixed-ky beta-scan inputs before policies are resolved."""

    betas: np.ndarray
    ky_target: float
    Nl: int
    Nm: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    cfg: KBMBaseCase | None
    time_cfg: TimeConfig | None
    solver: str
    krylov_cfg: KrylovConfig | None
    kbm_target_factors: Sequence[float] | None
    kbm_beta_transition: float | None
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
    mode_only: bool
    terms: LinearTerms | None
    sample_stride: int | None
    fit_signal: str
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    init_species_index: int
    density_species_index: int
    diagnostic_norm: str
    fapar_override: float | None
    apar_beta_scale: float | None
    ampere_g0_scale: float | None
    bpar_beta_scale: float | None
    reference_aligned: bool | None


def _kbm_beta_scan_request_from_locals(values: dict[str, Any]) -> _KBMBetaScanRequest:
    """Build a beta-scan request from ``run_kbm_beta_scan`` locals."""

    names = {field.name for field in fields(_KBMBetaScanRequest)}
    return _KBMBetaScanRequest(**{name: values[name] for name in names})


# Fixed-ky beta-scan path contracts and sample solvers live with the scan owner.
@dataclass(frozen=True)
class KBMBetaExplicitHooks:
    """Patchable numerical hooks used by the explicit-time KBM beta path."""

    integrate_linear_explicit_diagnostics: Callable[..., Any]
    instantaneous_growth_rate_from_phi: Callable[..., Any]
    windowed_growth_rate_from_omega_series: Callable[..., Any]
    extract_mode_time_series: Callable[..., Any]
    fit_growth_rate_auto: Callable[..., Any]
    normalize_growth_rate: Callable[..., tuple[float, float]]
    resolve_cfl_fac: Callable[..., float]


@dataclass(frozen=True)
class KBMBetaKrylovHooks:
    """Patchable numerical hooks used by the Krylov KBM beta path."""

    dominant_eigenpair: Callable[..., Any]
    use_multi_target_krylov: Callable[..., bool]
    normalize_growth_rate: Callable[..., tuple[float, float]]


@dataclass(frozen=True)
class KBMBetaTimeHooks:
    """Patchable numerical hooks used by the saved-time KBM beta path."""

    integrate_linear_diffrax_streaming: Callable[..., Any]
    integrate_linear_from_config: Callable[..., Any]
    integrate_linear_diagnostics: Callable[..., Any]
    integrate_linear: Callable[..., Any]
    resolve_streaming_window: Callable[..., tuple[float, float]]
    midplane_index: Callable[..., int]
    select_fit_signal_auto: Callable[..., tuple[Any, str, float, float]]
    extract_mode_only_signal: Callable[..., Any]
    select_fit_signal: Callable[..., Any]
    normalize_growth_rate: Callable[..., tuple[float, float]]


@dataclass(frozen=True)
class KBMBetaTimeSampleRequest:
    """Inputs for one saved-time or streaming KBM beta sample."""

    G0_jax: Any
    grid: Any
    geom: Any
    cache: Any
    params: Any
    terms: Any
    dt_i: float
    steps_i: int
    method: str
    time_cfg: Any
    sample_stride: int | None
    fit_key: str
    streaming_fit: bool
    streaming_amp_floor: float
    mode_only: bool
    mode_method: str
    sel: Any
    tmin: Any
    tmax: Any
    sample_index: int
    diagnostic_norm: str
    density_species_index: int
    fit_policy: Any


@dataclass(frozen=True)
class KBMBetaKrylovResult:
    """Krylov growth result plus continuation state for the next beta point."""

    gamma: float
    omega: float
    prev_vec: Any
    prev_eig: Any
    fallback_to_time: bool = False


_KBM_KRYLOV_FORWARD_KEYS = (
    "krylov_dim restarts omega_min_factor omega_target_factor omega_cap_factor omega_sign method "
    "power_iters power_dt shift shift_source shift_tol shift_maxiter shift_restart shift_solve_method "
    "shift_preconditioner shift_selection mode_family fallback_method fallback_real_floor"
).split()


def _dominant_kbm_eigenpair(
    hooks: KBMBetaKrylovHooks,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg_use: Any,
    **overrides: Any,
) -> tuple[Any, Any]:
    """Call the KBM Krylov solver with one shared target/shift policy."""

    kwargs = {
        "terms": terms,
        "v_ref": None,
        "select_overlap": False,
        **{name: getattr(krylov_cfg_use, name) for name in _KBM_KRYLOV_FORWARD_KEYS},
        **overrides,
    }
    return hooks.dominant_eigenpair(G0_jax, cache, params, **kwargs)


def fit_kbm_beta_explicit_time_sample(
    *,
    G0_jax: Any,
    grid: Any,
    cache: Any,
    params: Any,
    geom: Any,
    terms: Any,
    dt_i: float,
    steps_i: int,
    time_cfg: Any,
    sample_stride: int | None,
    mode_method: str,
    sel: Any,
    diagnostic_norm: str,
    fit_policy: Any,
    hooks: KBMBetaExplicitHooks,
) -> tuple[float, float]:
    """Run and fit one explicit-time KBM beta sample."""

    explicit_mode_method = (
        mode_method if mode_method in {"z_index", "max"} else "z_index"
    )
    explicit_time_cfg = ExplicitTimeConfig(
        dt=dt_i,
        t_max=dt_i * steps_i,
        sample_stride=max(int(sample_stride or 1), 1),
        fixed_dt=bool(time_cfg.fixed_dt) if time_cfg is not None else False,
        use_dealias_mask=bool(getattr(time_cfg, "use_dealias_mask", False))
        if time_cfg is not None
        else False,
        dt_min=float(time_cfg.dt_min) if time_cfg is not None else 1.0e-7,
        dt_max=float(time_cfg.dt_max)
        if (time_cfg is not None and time_cfg.dt_max is not None)
        else None,
        cfl=float(time_cfg.cfl) if time_cfg is not None else 0.9,
        cfl_fac=(
            hooks.resolve_cfl_fac(str(time_cfg.method), time_cfg.cfl_fac)
            if time_cfg is not None
            else float(ExplicitTimeConfig.cfl_fac)
        ),
    )
    t_arr, phi_t, gamma_t, omega_t, _diagnostics = (
        hooks.integrate_linear_explicit_diagnostics(
            G0_jax,
            grid,
            cache,
            params,
            geom,
            explicit_time_cfg,
            terms=terms,
            mode_method=explicit_mode_method,
            z_index=sel.z_index,
            jit=True,
        )
    )
    if t_arr.size > 1:
        gamma, omega = _fit_explicit_growth_history(
            phi_t=phi_t,
            t_arr=t_arr,
            gamma_t=gamma_t,
            omega_t=omega_t,
            mode_method=mode_method,
            sel=sel,
            fit_policy=fit_policy,
            hooks=hooks,
        )
    else:
        gamma = float("nan")
        omega = float("nan")
    return hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)


def _fit_explicit_growth_history(
    *,
    phi_t: Any,
    t_arr: Any,
    gamma_t: Any,
    omega_t: Any,
    mode_method: str,
    sel: Any,
    fit_policy: Any,
    hooks: KBMBetaExplicitHooks,
) -> tuple[float, float]:
    """Fit the explicit-time trace using the KBM fallback ladder."""

    phi_np = np.asarray(phi_t)
    t_np = np.asarray(t_arr, dtype=float)
    if mode_method in {"z_index", "max"}:
        try:
            gamma, omega, _gamma_t, _omega_t, _t_mid = (
                hooks.instantaneous_growth_rate_from_phi(
                    phi_np,
                    t_np,
                    sel,
                    navg_fraction=0.5,
                    mode_method=mode_method,
                )
            )
            return gamma, omega
        except ValueError:
            try:
                gamma, omega, _gamma_t, _omega_t = (
                    hooks.windowed_growth_rate_from_omega_series(
                        np.asarray(gamma_t),
                        np.asarray(omega_t),
                        sel,
                        navg_fraction=0.5,
                    )
                )
                return gamma, omega
            except ValueError:
                pass

    signal = hooks.extract_mode_time_series(phi_np, sel, method=mode_method)
    gamma, omega, _tmin, _tmax = hooks.fit_growth_rate_auto(
        t_np,
        signal,
        **{**fit_policy.auto_kwargs(), "window_method": "fixed"},
    )
    return gamma, omega


def fit_kbm_beta_time_sample(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    cache: Any,
    params: Any,
    terms: Any,
    dt_i: float,
    steps_i: int,
    method: str,
    time_cfg: Any,
    sample_stride: int | None,
    fit_key: str,
    streaming_fit: bool,
    streaming_amp_floor: float,
    mode_only: bool,
    mode_method: str,
    sel: Any,
    tmin: Any,
    tmax: Any,
    sample_index: int,
    diagnostic_norm: str,
    density_species_index: int,
    fit_policy: Any,
    hooks: KBMBetaTimeHooks,
) -> tuple[float, float]:
    """Run and fit one saved-time or streaming KBM beta sample."""

    return _fit_kbm_beta_time_sample_request(
        KBMBetaTimeSampleRequest(
            G0_jax=G0_jax,
            grid=grid,
            geom=geom,
            cache=cache,
            params=params,
            terms=terms,
            dt_i=dt_i,
            steps_i=steps_i,
            method=method,
            time_cfg=time_cfg,
            sample_stride=sample_stride,
            fit_key=fit_key,
            streaming_fit=streaming_fit,
            streaming_amp_floor=streaming_amp_floor,
            mode_only=mode_only,
            mode_method=mode_method,
            sel=sel,
            tmin=tmin,
            tmax=tmax,
            sample_index=sample_index,
            diagnostic_norm=diagnostic_norm,
            density_species_index=density_species_index,
            fit_policy=fit_policy,
        ),
        hooks=hooks,
    )


def _fit_kbm_beta_time_sample_request(
    request: KBMBetaTimeSampleRequest,
    *,
    hooks: KBMBetaTimeHooks,
) -> tuple[float, float]:
    time_cfg_i = _sample_time_config(
        request.time_cfg,
        request.dt_i,
        request.steps_i,
        request.sample_stride,
    )
    if time_cfg_i is not None and time_cfg_i.use_diffrax and request.streaming_fit:
        return _fit_streaming_time_sample(
            G0_jax=request.G0_jax,
            grid=request.grid,
            geom=request.geom,
            cache=request.cache,
            params=request.params,
            terms=request.terms,
            dt_i=request.dt_i,
            steps_i=request.steps_i,
            time_cfg_i=time_cfg_i,
            fit_key=request.fit_key,
            streaming_amp_floor=request.streaming_amp_floor,
            mode_method=request.mode_method,
            tmin=request.tmin,
            tmax=request.tmax,
            sample_index=request.sample_index,
            fit_policy=request.fit_policy,
            diagnostic_norm=request.diagnostic_norm,
            density_species_index=request.density_species_index,
            hooks=hooks,
        )

    phi_t, density_t, stride = _integrate_time_sample_series(
        G0_jax=request.G0_jax,
        grid=request.grid,
        geom=request.geom,
        cache=request.cache,
        params=request.params,
        terms=request.terms,
        dt_i=request.dt_i,
        steps_i=request.steps_i,
        method=request.method,
        time_cfg_i=time_cfg_i,
        sample_stride=request.sample_stride,
        fit_key=request.fit_key,
        mode_only=request.mode_only,
        mode_method=request.mode_method,
        sel=request.sel,
        density_species_index=request.density_species_index,
        hooks=hooks,
    )
    return _fit_saved_time_sample(
        phi_t=phi_t,
        density_t=density_t,
        dt_i=request.dt_i,
        stride=stride,
        fit_key=request.fit_key,
        mode_only=request.mode_only,
        mode_method=request.mode_method,
        sel=request.sel,
        tmin=request.tmin,
        tmax=request.tmax,
        sample_index=request.sample_index,
        diagnostic_norm=request.diagnostic_norm,
        density_species_index=request.density_species_index,
        params=request.params,
        fit_policy=request.fit_policy,
        hooks=hooks,
    )


def _sample_time_config(
    time_cfg: Any,
    dt_i: float,
    steps_i: int,
    sample_stride: int | None,
) -> Any | None:
    if time_cfg is None:
        return None
    cfg_i = replace(time_cfg, dt=dt_i, t_max=dt_i * steps_i)
    if sample_stride is not None:
        cfg_i = replace(cfg_i, sample_stride=sample_stride)
    return cfg_i


def _fit_streaming_time_sample(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    cache: Any,
    params: Any,
    terms: Any,
    dt_i: float,
    steps_i: int,
    time_cfg_i: Any,
    fit_key: str,
    streaming_amp_floor: float,
    mode_method: str,
    tmin: Any,
    tmax: Any,
    sample_index: int,
    fit_policy: Any,
    diagnostic_norm: str,
    density_species_index: int,
    hooks: KBMBetaTimeHooks,
) -> tuple[float, float]:
    tmin_i, tmax_i = hooks.resolve_streaming_window(
        float(time_cfg_i.t_max),
        _indexed_float_value(tmin, sample_index),
        _indexed_float_value(tmax, sample_index),
        fit_policy.start_fraction,
        fit_policy.window_fraction,
        1.0,
    )
    _, gamma_vals, omega_vals = hooks.integrate_linear_diffrax_streaming(
        G0_jax,
        grid,
        geom,
        params,
        dt=dt_i,
        steps=steps_i,
        method=time_cfg_i.diffrax_solver,
        cache=cache,
        terms=terms,
        adaptive=time_cfg_i.diffrax_adaptive,
        rtol=time_cfg_i.diffrax_rtol,
        atol=time_cfg_i.diffrax_atol,
        max_steps=time_cfg_i.diffrax_max_steps,
        progress_bar=time_cfg_i.progress_bar,
        checkpoint=time_cfg_i.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal=fit_key,
        mode_ky_indices=[0],
        mode_kx_index=0,
        mode_z_index=hooks.midplane_index(grid),
        mode_method=mode_method,
        amp_floor=streaming_amp_floor,
        density_species_index=density_species_index if fit_key == "density" else None,
        return_state=False,
    )
    gamma = float(np.asarray(gamma_vals)[0])
    omega = float(np.asarray(omega_vals)[0])
    return hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)


def _integrate_time_sample_series(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    cache: Any,
    params: Any,
    terms: Any,
    dt_i: float,
    steps_i: int,
    method: str,
    time_cfg_i: Any | None,
    sample_stride: int | None,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    sel: Any,
    density_species_index: int,
    hooks: KBMBetaTimeHooks,
) -> tuple[Any, Any | None, int]:
    if time_cfg_i is not None and time_cfg_i.use_diffrax:
        stride = int(time_cfg_i.sample_stride)
        phi_t, density_t = _integrate_config_time_series(
            G0_jax=G0_jax,
            grid=grid,
            geom=geom,
            cache=cache,
            params=params,
            terms=terms,
            time_cfg_i=time_cfg_i,
            fit_key=fit_key,
            mode_only=mode_only,
            mode_method=mode_method,
            sel=sel,
            density_species_index=density_species_index,
            hooks=hooks,
        )
        return phi_t, density_t, stride

    stride = (
        int(time_cfg_i.sample_stride)
        if time_cfg_i is not None
        else 1 if sample_stride is None else int(sample_stride)
    )
    phi_t, density_t = _integrate_saved_time_series(
        G0_jax=G0_jax,
        grid=grid,
        geom=geom,
        params=params,
        cache=cache,
        terms=terms,
        dt_i=dt_i,
        steps_i=steps_i,
        method=method,
        stride=stride,
        fit_key=fit_key,
        density_species_index=density_species_index,
        hooks=hooks,
    )
    return phi_t, density_t, stride


def _integrate_config_time_series(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    cache: Any,
    params: Any,
    terms: Any,
    time_cfg_i: Any,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    sel: Any,
    density_species_index: int,
    hooks: KBMBetaTimeHooks,
) -> tuple[Any, Any | None]:
    save_mode_method = mode_method if mode_method in {"z_index", "max"} else "z_index"
    _, phi_t = hooks.integrate_linear_from_config(
        G0_jax,
        grid,
        geom,
        params,
        time_cfg_i,
        cache=cache,
        terms=terms,
        save_mode=sel if mode_only else None,
        mode_method=save_mode_method,
        save_field="phi+density"
        if fit_key == "auto"
        else ("density" if fit_key == "density" else "phi"),
        density_species_index=density_species_index
        if fit_key in {"density", "auto"}
        else None,
    )
    if fit_key == "auto":
        phi_values, density_values = phi_t
        return phi_values, density_values
    return phi_t, None


def _fit_saved_time_sample(
    *,
    phi_t: Any,
    density_t: Any | None,
    dt_i: float,
    stride: int,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    sel: Any,
    tmin: Any,
    tmax: Any,
    sample_index: int,
    diagnostic_norm: str,
    density_species_index: int,
    params: Any,
    fit_policy: Any,
    hooks: KBMBetaTimeHooks,
) -> tuple[float, float]:
    phi_t_np = np.asarray(phi_t)
    density_np = None if density_t is None else np.asarray(density_t)
    if fit_key == "density" and density_np is None:
        density_np = phi_t_np
    if fit_key == "auto":
        _signal, _name, gamma, omega = hooks.select_fit_signal_auto(
            np.arange(phi_t_np.shape[0]) * dt_i * stride,
            phi_t_np,
            density_np,
            sel,
            mode_method=mode_method,
            tmin=_indexed_float_value(tmin, sample_index),
            tmax=_indexed_float_value(tmax, sample_index),
            num_windows=8,
            **fit_policy.auto_kwargs(),
        )
        return hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)

    signal = _select_time_fit_signal(
        phi_t_np=phi_t_np,
        density_np=density_np,
        fit_key=fit_key,
        mode_only=mode_only,
        mode_method=mode_method,
        sel=sel,
        density_species_index=density_species_index,
        hooks=hooks,
    )
    return fit_policy.fit_signal(
        signal,
        idx=sample_index,
        dt=dt_i,
        stride=stride,
        params=params,
        diagnostic_norm=diagnostic_norm,
    )


def _select_time_fit_signal(
    *,
    phi_t_np: Any,
    density_np: Any | None,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    sel: Any,
    density_species_index: int,
    hooks: KBMBetaTimeHooks,
) -> Any:
    if (
        mode_only
        and fit_key == "density"
        and density_np is not None
        and density_np.ndim <= 3
    ):
        return hooks.extract_mode_only_signal(
            density_np,
            local_idx=0,
            species_index=density_species_index,
        )
    if mode_only and phi_t_np.ndim <= 2:
        return hooks.extract_mode_only_signal(phi_t_np, local_idx=0)
    return hooks.select_fit_signal(
        phi_t_np,
        density_np,
        sel,
        fit_signal=fit_key,
        mode_method=mode_method,
    )


def _integrate_saved_time_series(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    params: Any,
    cache: Any,
    terms: Any,
    dt_i: float,
    steps_i: int,
    method: str,
    stride: int,
    fit_key: str,
    density_species_index: int,
    hooks: KBMBetaTimeHooks,
) -> tuple[Any, Any | None]:
    if fit_key in {"density", "auto"}:
        diag_out = hooks.integrate_linear_diagnostics(
            G0_jax,
            grid,
            geom,
            params,
            dt=dt_i,
            steps=steps_i,
            method=method,
            cache=cache,
            terms=terms,
            sample_stride=stride,
            species_index=density_species_index,
        )
        return diag_out[1], diag_out[2] if len(diag_out) > 2 else None

    _, phi_t = hooks.integrate_linear(
        G0_jax,
        grid,
        geom,
        params,
        dt=dt_i,
        steps=steps_i,
        method=method,
        cache=cache,
        terms=terms,
        sample_stride=stride,
    )
    return phi_t, None


def _select_kbm_beta_eigen_candidate(
    *,
    beta: float,
    beta_transition: float,
    eig_candidates: Sequence[Any],
    vec_candidates: Sequence[Any],
) -> tuple[Any, Any]:
    """Select the KBM branch from targeted Krylov candidates."""

    if len(eig_candidates) >= 2 and np.isfinite(beta_transition):
        idx = 1 if float(beta) >= float(beta_transition) else 0
    else:
        growth = np.real(np.asarray([complex(np.asarray(e)) for e in eig_candidates]))
        idx = (
            0
            if np.all(~np.isfinite(growth))
            else int(np.nanargmax(np.where(np.isfinite(growth), growth, -np.inf)))
        )
    return eig_candidates[idx], vec_candidates[idx]


def _solve_multi_target_kbm_eigenpair(
    *,
    beta: float,
    cfg: Any,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg_use: Any,
    targets: Sequence[float],
    kbm_beta_transition: float | None,
    hooks: KBMBetaKrylovHooks,
) -> tuple[Any, Any]:
    """Solve all targeted KBM candidates and select the physical beta branch."""

    beta_transition = (
        float(cfg.model.beta)
        if kbm_beta_transition is None
        else float(kbm_beta_transition)
    )
    target_results = [
        _dominant_kbm_eigenpair(
            hooks,
            G0_jax,
            cache,
            params,
            terms,
            krylov_cfg_use,
            omega_target_factor=float(target),
            shift=None,
            shift_source="target",
            shift_selection="targeted",
        )
        for target in targets
    ]
    eig_candidates, vec_candidates = zip(*target_results, strict=True)
    return _select_kbm_beta_eigen_candidate(
        beta=beta,
        beta_transition=beta_transition,
        eig_candidates=eig_candidates,
        vec_candidates=vec_candidates,
    )


def _resolve_kbm_krylov_shift(
    krylov_cfg_use: Any, *, use_continuation: bool, prev_eig: Any
) -> tuple[Any, Any]:
    shift_val = krylov_cfg_use.shift
    if use_continuation and prev_eig is not None:
        shift_val = complex(np.asarray(prev_eig))
    return shift_val, krylov_cfg_use.shift_selection


def _solve_kbm_beta_selected_eigenpair(
    *,
    beta: float,
    cfg: Any,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg_use: Any,
    use_continuation: bool,
    prev_vec: Any,
    targets: Sequence[float] | None,
    kbm_beta_transition: float | None,
    shift_val: Any,
    shift_selection: Any,
    hooks: KBMBetaKrylovHooks,
) -> tuple[Any, Any]:
    use_multi_target = hooks.use_multi_target_krylov(
        krylov_cfg_use,
        targets,
        shift=shift_val,
    )
    if use_multi_target:
        assert targets is not None
        return _solve_multi_target_kbm_eigenpair(
            beta=beta,
            cfg=cfg,
            G0_jax=G0_jax,
            cache=cache,
            params=params,
            terms=terms,
            krylov_cfg_use=krylov_cfg_use,
            targets=targets,
            kbm_beta_transition=kbm_beta_transition,
            hooks=hooks,
        )
    return _dominant_kbm_eigenpair(
        hooks,
        G0_jax,
        cache,
        params,
        terms,
        krylov_cfg_use,
        v_ref=prev_vec,
        select_overlap=use_continuation,
        shift=shift_val,
        shift_selection=shift_selection,
    )


def _normalized_kbm_beta_krylov_growth(
    eig: Any,
    *,
    krylov_cfg_use: Any,
    params: Any,
    diagnostic_norm: str,
    hooks: KBMBetaKrylovHooks,
) -> tuple[float, float]:
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    if krylov_cfg_use.omega_sign != 0:
        omega = float(np.sign(krylov_cfg_use.omega_sign)) * abs(omega)
    return hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)


def solve_kbm_beta_krylov_sample(
    *,
    beta: float,
    cfg: Any,
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    solver_key: str,
    krylov_cfg_use: Any,
    use_continuation: bool,
    prev_vec: Any,
    prev_eig: Any,
    kbm_target_factors: Sequence[float] | None,
    kbm_beta_transition: float | None,
    diagnostic_norm: str,
    is_valid_growth: Callable[[float, float], bool],
    hooks: KBMBetaKrylovHooks,
) -> KBMBetaKrylovResult:
    """Run one Krylov KBM beta sample and decide whether time fallback is needed."""

    shift_val, shift_selection = _resolve_kbm_krylov_shift(
        krylov_cfg_use,
        use_continuation=use_continuation,
        prev_eig=prev_eig,
    )
    targets: Sequence[float] | None = kbm_target_factors if kbm_target_factors else None
    eig, vec = _solve_kbm_beta_selected_eigenpair(
        beta=beta,
        cfg=cfg,
        G0_jax=G0_jax,
        cache=cache,
        params=params,
        terms=terms,
        krylov_cfg_use=krylov_cfg_use,
        use_continuation=use_continuation,
        prev_vec=prev_vec,
        targets=targets,
        kbm_beta_transition=kbm_beta_transition,
        shift_val=shift_val,
        shift_selection=shift_selection,
        hooks=hooks,
    )
    gamma, omega = _normalized_kbm_beta_krylov_growth(
        eig,
        krylov_cfg_use=krylov_cfg_use,
        params=params,
        diagnostic_norm=diagnostic_norm,
        hooks=hooks,
    )
    if solver_key == "auto" and not is_valid_growth(gamma, omega):
        return KBMBetaKrylovResult(
            gamma=gamma,
            omega=omega,
            prev_vec=prev_vec,
            prev_eig=prev_eig,
            fallback_to_time=True,
        )
    if use_continuation:
        prev_vec = vec
        prev_eig = eig
    return KBMBetaKrylovResult(
        gamma=gamma,
        omega=omega,
        prev_vec=prev_vec,
        prev_eig=prev_eig,
        fallback_to_time=False,
    )


def _indexed_float_value(value: Any, idx: int) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return float(arr[int(idx)])


def _valid_kbm_growth(
    gamma_val: float, omega_val: float, *, require_positive: bool
) -> bool:
    """Return whether a Krylov result is acceptable before time fallback."""

    if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
        return False
    if require_positive and gamma_val <= 0.0:
        return False
    return True




def _build_kbm_beta_hooks() -> tuple[
    KBMBetaExplicitHooks, KBMBetaKrylovHooks, KBMBetaTimeHooks
]:
    """Build patchable hook bundles for the KBM beta solver paths."""

    explicit_hooks = KBMBetaExplicitHooks(
        integrate_linear_explicit_diagnostics=integrate_linear_explicit_diagnostics,
        instantaneous_growth_rate_from_phi=instantaneous_growth_rate_from_phi,
        windowed_growth_rate_from_omega_series=windowed_growth_rate_from_omega_series,
        extract_mode_time_series=extract_mode_time_series,
        fit_growth_rate_auto=fit_growth_rate_auto,
        normalize_growth_rate=_normalize_growth_rate,
        resolve_cfl_fac=resolve_cfl_fac,
    )
    krylov_hooks = KBMBetaKrylovHooks(
        dominant_eigenpair=dominant_eigenpair,
        use_multi_target_krylov=_kbm_use_multi_target_krylov,
        normalize_growth_rate=_normalize_growth_rate,
    )
    time_hooks = KBMBetaTimeHooks(
        integrate_linear_diffrax_streaming=integrate_linear_diffrax_streaming,
        integrate_linear_from_config=integrate_linear_from_config,
        integrate_linear_diagnostics=integrate_linear_diagnostics,
        integrate_linear=integrate_linear,
        resolve_streaming_window=_resolve_streaming_window,
        midplane_index=_midplane_index,
        select_fit_signal_auto=_select_fit_signal_auto,
        extract_mode_only_signal=_extract_mode_only_signal,
        select_fit_signal=_select_fit_signal,
        normalize_growth_rate=_normalize_growth_rate,
    )
    return explicit_hooks, krylov_hooks, time_hooks


def _build_kbm_beta_fit_policy(
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
) -> ScanFitWindowPolicy:
    """Pack the shared beta-scan growth-window policy."""

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
        fit_growth_rate_fn=fit_growth_rate,
        fit_growth_rate_auto_fn=fit_growth_rate_auto,
        normalize_growth_rate_fn=_normalize_growth_rate,
    )


def _build_kbm_beta_scan_setup(
    *,
    cfg: KBMBaseCase | None,
    ky_target: float,
    terms: LinearTerms | None,
    reference_aligned: bool | None,
    diagnostic_norm: str,
    solver: str,
    fit_signal: str,
    streaming_fit: bool,
    mode_only: bool,
    krylov_cfg: KrylovConfig | None,
    fit_policy: ScanFitWindowPolicy,
) -> _KBMBetaScanSetup:
    """Build shared grid, geometry, policy, and hook state for a beta scan."""

    cfg_use = cfg or KBMBaseCase()
    grid_full = build_spectral_grid(cfg_use.grid)
    geom = SAlphaGeometry.from_config(cfg_use.geometry)
    terms_use = terms if terms is not None else LinearTerms(bpar=0.0)
    reference_aligned_use = bool(
        True if reference_aligned is None else reference_aligned
    )
    diagnostic_norm_use = diagnostic_norm
    if reference_aligned_use and diagnostic_norm_use == "none":
        diagnostic_norm_use = "rho_star"
    damp_ends_amp, damp_ends_widthfrac = _linked_boundary_end_damping(
        reference_aligned_use
    )
    fit_key = normalize_fit_signal(fit_signal)
    streaming_fit_use, mode_only_use = apply_auto_fit_scan_policy(
        fit_key, streaming_fit=streaming_fit, mode_only=mode_only
    )
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    selection = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    explicit_hooks, krylov_hooks, time_hooks = _build_kbm_beta_hooks()
    krylov_cfg_use = krylov_cfg or KBM_KRYLOV_DEFAULT
    return _KBMBetaScanSetup(
        cfg=cfg_use,
        grid=grid,
        geom=geom,
        selection=selection,
        terms=terms_use,
        reference_aligned=reference_aligned_use,
        damp_ends_amp=damp_ends_amp,
        damp_ends_widthfrac=damp_ends_widthfrac,
        solver_key=normalize_solver_key(solver),
        fit_key=fit_key,
        streaming_fit=streaming_fit_use,
        mode_only=mode_only_use,
        diagnostic_norm=diagnostic_norm_use,
        krylov_cfg=krylov_cfg_use,
        use_continuation=bool(getattr(krylov_cfg_use, "continuation", False)),
        fit_policy=fit_policy,
        explicit_hooks=explicit_hooks,
        krylov_hooks=krylov_hooks,
        time_hooks=time_hooks,
    )


def _build_kbm_beta_initial_state(
    setup: _KBMBetaScanSetup,
    *,
    Nl: int,
    Nm: int,
    init_species_index: int,
) -> Any:
    """Build the selected-species KBM beta-scan initial condition."""

    state = np.zeros(
        (2, Nl, Nm, setup.grid.ky.size, setup.grid.kx.size, setup.grid.z.size),
        dtype=np.complex64,
    )
    single_species_state = _build_initial_condition(
        setup.grid,
        setup.geom,
        ky_index=setup.selection.ky_index,
        kx_index=setup.selection.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=setup.cfg.init,
    )
    state[int(init_species_index)] = np.asarray(
        single_species_state, dtype=np.complex64
    )
    return jnp.asarray(state)


def _build_kbm_beta_sample(
    setup: _KBMBetaScanSetup,
    *,
    beta: float,
    sample_index: int,
    ky_target: float,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    init_species_index: int,
    fapar_override: float | None,
    apar_beta_scale: float | None,
    ampere_g0_scale: float | None,
    bpar_beta_scale: float | None,
) -> _KBMBetaSample:
    """Create all per-beta inputs needed by the selected solver path."""

    dt_i = float(dt[sample_index]) if isinstance(dt, np.ndarray) else float(dt)
    steps_i = (
        int(steps[sample_index]) if isinstance(steps, np.ndarray) else int(steps)
    )
    params = _two_species_params(
        setup.cfg.model,
        kpar_scale=float(setup.geom.gradpar()),
        omega_d_scale=KBM_OMEGA_D_SCALE,
        omega_star_scale=KBM_OMEGA_STAR_SCALE,
        rho_star=KBM_RHO_STAR,
        beta_override=float(beta),
        fapar_override=fapar_override,
        apar_beta_scale=apar_beta_scale,
        ampere_g0_scale=ampere_g0_scale,
        bpar_beta_scale=bpar_beta_scale,
        damp_ends_amp=setup.damp_ends_amp,
        damp_ends_widthfrac=setup.damp_ends_widthfrac,
        nhermite=Nm,
    )
    return _KBMBetaSample(
        beta=float(beta),
        index=sample_index,
        dt=dt_i,
        steps=steps_i,
        params=params,
        cache=build_linear_cache(setup.grid, setup.geom, params, Nl, Nm),
        initial_state=_build_kbm_beta_initial_state(
            setup, Nl=Nl, Nm=Nm, init_species_index=init_species_index
        ),
        solver_use=select_kbm_solver_auto(
            setup.solver_key,
            ky_target=ky_target,
            reference_aligned=setup.reference_aligned,
        ),
    )


def _fit_kbm_beta_explicit_sample(
    setup: _KBMBetaScanSetup,
    sample: _KBMBetaSample,
    *,
    time_cfg: TimeConfig | None,
    sample_stride: int | None,
    mode_method: str,
) -> tuple[float, float]:
    """Fit one beta sample with the explicit-time diagnostic path."""

    return fit_kbm_beta_explicit_time_sample(
        G0_jax=sample.initial_state,
        grid=setup.grid,
        cache=sample.cache,
        params=sample.params,
        geom=setup.geom,
        terms=setup.terms,
        dt_i=sample.dt,
        steps_i=sample.steps,
        time_cfg=time_cfg,
        sample_stride=sample_stride,
        mode_method=mode_method,
        sel=setup.selection,
        diagnostic_norm=setup.diagnostic_norm,
        fit_policy=setup.fit_policy,
        hooks=setup.explicit_hooks,
    )


def _fit_kbm_beta_krylov_sample(
    setup: _KBMBetaScanSetup,
    sample: _KBMBetaSample,
    *,
    continuation: _KBMBetaContinuation,
    kbm_target_factors: Sequence[float] | None,
    kbm_beta_transition: float | None,
    require_positive: bool,
) -> tuple[float, float, _KBMBetaContinuation, bool]:
    """Fit one beta sample with Krylov and return continuation/fallback state."""

    krylov_result = solve_kbm_beta_krylov_sample(
        beta=sample.beta,
        cfg=setup.cfg,
        G0_jax=sample.initial_state,
        cache=sample.cache,
        params=sample.params,
        terms=setup.terms,
        solver_key=setup.solver_key,
        krylov_cfg_use=setup.krylov_cfg,
        use_continuation=setup.use_continuation,
        prev_vec=continuation.prev_vec,
        prev_eig=continuation.prev_eig,
        kbm_target_factors=kbm_target_factors,
        kbm_beta_transition=kbm_beta_transition,
        diagnostic_norm=setup.diagnostic_norm,
        is_valid_growth=lambda gamma, omega: _valid_kbm_growth(
            gamma, omega, require_positive=require_positive
        ),
        hooks=setup.krylov_hooks,
    )
    if krylov_result.fallback_to_time:
        return (
            krylov_result.gamma,
            krylov_result.omega,
            continuation,
            True,
        )
    return (
        krylov_result.gamma,
        krylov_result.omega,
        _KBMBetaContinuation(
            prev_vec=krylov_result.prev_vec,
            prev_eig=krylov_result.prev_eig,
        ),
        False,
    )


def _fit_kbm_beta_saved_time_sample(
    setup: _KBMBetaScanSetup,
    sample: _KBMBetaSample,
    *,
    method: str,
    time_cfg: TimeConfig | None,
    sample_stride: int | None,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    streaming_amp_floor: float,
    density_species_index: int,
) -> tuple[float, float]:
    """Fit one beta sample with the saved-time or streaming time path."""

    return fit_kbm_beta_time_sample(
        G0_jax=sample.initial_state,
        grid=setup.grid,
        geom=setup.geom,
        cache=sample.cache,
        params=sample.params,
        terms=setup.terms,
        dt_i=sample.dt,
        steps_i=sample.steps,
        method=method,
        time_cfg=time_cfg,
        sample_stride=sample_stride,
        fit_key=setup.fit_key,
        streaming_fit=setup.streaming_fit,
        streaming_amp_floor=streaming_amp_floor,
        mode_only=setup.mode_only,
        mode_method=mode_method,
        sel=setup.selection,
        tmin=tmin,
        tmax=tmax,
        sample_index=sample.index,
        diagnostic_norm=setup.diagnostic_norm,
        density_species_index=density_species_index,
        fit_policy=setup.fit_policy,
        hooks=setup.time_hooks,
    )


def _fit_kbm_beta_sample(
    setup: _KBMBetaScanSetup,
    sample: _KBMBetaSample,
    *,
    method: str,
    time_cfg: TimeConfig | None,
    sample_stride: int | None,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    streaming_amp_floor: float,
    density_species_index: int,
    continuation: _KBMBetaContinuation,
    kbm_target_factors: Sequence[float] | None,
    kbm_beta_transition: float | None,
    require_positive: bool,
) -> tuple[float, float, _KBMBetaContinuation]:
    """Dispatch one beta sample through the selected solver policy."""

    solver_use = sample.solver_use
    if solver_use == "explicit_time":
        gamma, omega = _fit_kbm_beta_explicit_sample(
            setup,
            sample,
            time_cfg=time_cfg,
            sample_stride=sample_stride,
            mode_method=mode_method,
        )
        return gamma, omega, continuation

    if solver_use == "krylov":
        gamma, omega, continuation, fallback_to_time = _fit_kbm_beta_krylov_sample(
            setup,
            sample,
            continuation=continuation,
            kbm_target_factors=kbm_target_factors,
            kbm_beta_transition=kbm_beta_transition,
            require_positive=require_positive,
        )
        if not fallback_to_time:
            return gamma, omega, continuation

    gamma, omega = _fit_kbm_beta_saved_time_sample(
        setup,
        sample,
        method=method,
        time_cfg=time_cfg,
        sample_stride=sample_stride,
        mode_method=mode_method,
        tmin=tmin,
        tmax=tmax,
        streaming_amp_floor=streaming_amp_floor,
        density_species_index=density_species_index,
    )
    return gamma, omega, continuation


def _build_kbm_beta_scan_options(
    *,
    ky_target: float,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    time_cfg: TimeConfig | None,
    kbm_target_factors: Sequence[float] | None,
    kbm_beta_transition: float | None,
    tmin: float | None,
    tmax: float | None,
    require_positive: bool,
    mode_method: str,
    sample_stride: int | None,
    streaming_amp_floor: float,
    init_species_index: int,
    density_species_index: int,
    fapar_override: float | None,
    apar_beta_scale: float | None,
    ampere_g0_scale: float | None,
    bpar_beta_scale: float | None,
) -> _KBMBetaScanOptions:
    return _KBMBetaScanOptions(
        ky_target=float(ky_target),
        n_laguerre=int(Nl),
        n_hermite=int(Nm),
        dt=dt,
        steps=steps,
        method=method,
        time_cfg=time_cfg,
        kbm_target_factors=kbm_target_factors,
        kbm_beta_transition=kbm_beta_transition,
        tmin=tmin,
        tmax=tmax,
        require_positive=bool(require_positive),
        mode_method=mode_method,
        sample_stride=sample_stride,
        streaming_amp_floor=float(streaming_amp_floor),
        init_species_index=int(init_species_index),
        density_species_index=int(density_species_index),
        fapar_override=fapar_override,
        apar_beta_scale=apar_beta_scale,
        ampere_g0_scale=ampere_g0_scale,
        bpar_beta_scale=bpar_beta_scale,
    )


def _run_kbm_beta_scan_point(
    *,
    setup: _KBMBetaScanSetup,
    beta: float,
    index: int,
    options: _KBMBetaScanOptions,
    continuation: _KBMBetaContinuation,
) -> tuple[float, float, _KBMBetaContinuation]:
    sample = _build_kbm_beta_sample(
        setup,
        beta=float(beta),
        sample_index=index,
        ky_target=options.ky_target,
        Nl=options.n_laguerre,
        Nm=options.n_hermite,
        dt=options.dt,
        steps=options.steps,
        init_species_index=options.init_species_index,
        fapar_override=options.fapar_override,
        apar_beta_scale=options.apar_beta_scale,
        ampere_g0_scale=options.ampere_g0_scale,
        bpar_beta_scale=options.bpar_beta_scale,
    )
    return _fit_kbm_beta_sample(
        setup,
        sample,
        method=options.method,
        time_cfg=options.time_cfg,
        sample_stride=options.sample_stride,
        mode_method=options.mode_method,
        tmin=options.tmin,
        tmax=options.tmax,
        streaming_amp_floor=options.streaming_amp_floor,
        density_species_index=options.density_species_index,
        continuation=continuation,
        kbm_target_factors=options.kbm_target_factors,
        kbm_beta_transition=options.kbm_beta_transition,
        require_positive=options.require_positive,
    )


def _run_kbm_beta_scan_loop(
    betas: np.ndarray,
    *,
    setup: _KBMBetaScanSetup,
    options: _KBMBetaScanOptions,
) -> _KBMBetaScanOutput:
    output = _KBMBetaScanOutput.empty()
    continuation = _KBMBetaContinuation()
    for index, beta in enumerate(betas):
        gamma, omega, continuation = _run_kbm_beta_scan_point(
            setup=setup,
            beta=float(beta),
            index=index,
            options=options,
            continuation=continuation,
        )
        output.append(beta=float(beta), gamma=gamma, omega=omega)
    return output


def _run_kbm_beta_scan_request(request: _KBMBetaScanRequest) -> LinearScanResult:
    """Resolve KBM beta-scan policies and execute the fixed-ky beta sweep."""

    _validate_kbm_species_indices(
        init_species_index=request.init_species_index,
        density_species_index=request.density_species_index,
    )
    fit_policy = _build_kbm_beta_fit_policy(
        tmin=request.tmin,
        tmax=request.tmax,
        auto_window=request.auto_window,
        window_fraction=request.window_fraction,
        min_points=request.min_points,
        start_fraction=request.start_fraction,
        growth_weight=request.growth_weight,
        require_positive=request.require_positive,
        min_amp_fraction=request.min_amp_fraction,
    )
    setup = _build_kbm_beta_scan_setup(
        cfg=request.cfg,
        ky_target=request.ky_target,
        terms=request.terms,
        reference_aligned=request.reference_aligned,
        diagnostic_norm=request.diagnostic_norm,
        solver=request.solver,
        fit_signal=request.fit_signal,
        streaming_fit=request.streaming_fit,
        mode_only=request.mode_only,
        krylov_cfg=request.krylov_cfg,
        fit_policy=fit_policy,
    )
    options = _build_kbm_beta_scan_options(
        ky_target=request.ky_target,
        Nl=request.Nl,
        Nm=request.Nm,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        time_cfg=request.time_cfg,
        kbm_target_factors=request.kbm_target_factors,
        kbm_beta_transition=request.kbm_beta_transition,
        tmin=request.tmin,
        tmax=request.tmax,
        require_positive=request.require_positive,
        mode_method=request.mode_method,
        sample_stride=request.sample_stride,
        streaming_amp_floor=request.streaming_amp_floor,
        init_species_index=request.init_species_index,
        density_species_index=request.density_species_index,
        fapar_override=request.fapar_override,
        apar_beta_scale=request.apar_beta_scale,
        ampere_g0_scale=request.ampere_g0_scale,
        bpar_beta_scale=request.bpar_beta_scale,
    )
    return _run_kbm_beta_scan_loop(
        np.asarray(request.betas, dtype=float),
        setup=setup,
        options=options,
    ).result()


def run_kbm_beta_scan(
    betas: np.ndarray,
    ky_target: float = 0.3,
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
) -> LinearScanResult:
    """Run a KBM beta scan at fixed ky.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    return _run_kbm_beta_scan_request(_kbm_beta_scan_request_from_locals(locals()))


_ETG_KRYLOV_FORWARD_KEYS = (
    "krylov_dim restarts omega_min_factor omega_target_factor omega_cap_factor omega_sign method "
    "power_iters power_dt shift shift_source shift_tol shift_maxiter shift_restart shift_solve_method "
    "shift_preconditioner shift_selection mode_family fallback_method fallback_real_floor"
).split()


# TEM scan and single-mode path policies live with the TEM benchmark owner.
@dataclass(frozen=True)
class TEMPathHooks:
    """Patchable numerical hooks supplied by the TEM public owner module."""

    linear_run_result: type[LinearRunResult]
    linear_scan_result: type[LinearScanResult]
    mode_selection: type[ModeSelection]
    mode_selection_batch: type[ModeSelectionBatch]
    select_ky_index: Callable[..., int]
    select_ky_grid: Callable[..., Any]
    build_initial_condition: Callable[..., Any]
    build_linear_cache: Callable[..., Any]
    dominant_eigenpair: Callable[..., tuple[Any, Any]]
    compute_fields_cached: Callable[..., Any]
    linear_terms_to_term_config: Callable[..., Any]
    integrate_linear: Callable[..., Any]
    integrate_linear_diagnostics: Callable[..., Any]
    integrate_linear_from_config: Callable[..., Any]
    integrate_linear_diffrax_streaming: Callable[..., Any]
    extract_mode_time_series: Callable[..., np.ndarray]
    fit_growth_rate: Callable[..., tuple[float, float]]
    fit_growth_rate_auto: Callable[..., tuple[float, float, float, float]]
    normalize_growth_rate: Callable[..., tuple[float, float]]
    resolve_streaming_window: Callable[..., tuple[float, float]]
    midplane_index: Callable[..., int]


@dataclass(frozen=True)
class _TEMBatchContext:
    """Fully prepared numerical context for one TEM scan batch."""

    batch_start: int
    ky_slice: np.ndarray
    valid_count: int
    grid: Any
    selection: ModeSelection | ModeSelectionBatch
    dt: float
    steps: int
    state: Any
    cache: Any


@dataclass(frozen=True)
class _TEMScanRuntimeOptions:
    """Options shared by every TEM scan batch."""

    n_laguerre: int
    n_hermite: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    time_cfg: Any | None
    solver_key: str
    krylov_cfg: Any | None
    krylov_default: Any
    fit_policy: ScanFitWindowPolicy
    mode_method: str
    mode_only: bool
    sample_stride: int | None
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    init_species_index: int
    diagnostic_norm: str
    use_batch: bool
    show_progress: bool
    hooks: TEMPathHooks


@dataclass
class _TEMScanAccumulator:
    """Mutable TEM scan rows collected across batches."""

    gammas: list[float]
    omegas: list[float]
    ky_out: list[float]

    def result(self, hooks: TEMPathHooks) -> LinearScanResult:
        """Pack accumulated rows with the public result type."""

        return hooks.linear_scan_result(
            ky=np.array(self.ky_out),
            gamma=np.array(self.gammas),
            omega=np.array(self.omegas),
        )


@dataclass(frozen=True)
class _TEMTimePathTiming:
    """Resolved sampling controls for one TEM initial-value solve."""

    dt: float
    steps: int
    stride: int
    time_cfg: Any | None


@dataclass(frozen=True)
class _TEMTimePathTrace:
    """Saved fields and sample times from one TEM initial-value solve."""

    t: np.ndarray
    phi_t: np.ndarray
    density_t: np.ndarray | None


@dataclass(frozen=True)
class _TEMTimePathFitPolicy:
    """Window-selection controls for fitting a TEM linear trace."""

    auto_window: bool
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float


def _tem_time_path_fit_policy_from_locals(
    values: dict[str, Any],
) -> _TEMTimePathFitPolicy:
    """Pack the single-run TEM fit policy from public path arguments."""

    return _TEMTimePathFitPolicy(
        **{field.name: values[field.name] for field in fields(_TEMTimePathFitPolicy)}
    )


def _tem_scan_fit_policy_from_locals(
    values: dict[str, Any],
    *,
    hooks: TEMPathHooks,
) -> ScanFitWindowPolicy:
    """Pack TEM scan growth-window policy with patchable fit hooks."""

    return ScanFitWindowPolicy(
        tmin=values["tmin"],
        tmax=values["tmax"],
        auto_window=values["auto_window"],
        window_fraction=values["window_fraction"],
        min_points=values["min_points"],
        start_fraction=values["start_fraction"],
        growth_weight=values["growth_weight"],
        require_positive=values["require_positive"],
        min_amp_fraction=values["min_amp_fraction"],
        fit_growth_rate_fn=hooks.fit_growth_rate,
        fit_growth_rate_auto_fn=hooks.fit_growth_rate_auto,
        normalize_growth_rate_fn=hooks.normalize_growth_rate,
    )


def _tem_scan_runtime_options_from_locals(
    values: dict[str, Any],
) -> _TEMScanRuntimeOptions:
    """Pack scan runtime options once after the fit policy is resolved."""

    return _TEMScanRuntimeOptions(
        **{field.name: values[field.name] for field in fields(_TEMScanRuntimeOptions)}
    )


_TEM_KRYLOV_FORWARD_KEYS = (
    "krylov_dim restarts omega_min_factor omega_target_factor omega_cap_factor omega_sign method "
    "power_iters power_dt shift shift_source shift_tol shift_maxiter shift_restart shift_solve_method "
    "shift_preconditioner shift_selection mode_family fallback_method fallback_real_floor"
).split()


def _krylov_eigenvalue(
    G0_jax: Any,
    cache: Any,
    params: Any,
    terms: Any,
    cfg_use: Any,
    hooks: TEMPathHooks,
) -> tuple[Any, Any]:
    kwargs = {
        "terms": terms,
        **{name: getattr(cfg_use, name) for name in _TEM_KRYLOV_FORWARD_KEYS},
    }
    return hooks.dominant_eigenpair(G0_jax, cache, params, **kwargs)


def _prepare_tem_scan_batch(
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    grid_full: Any,
    geom: Any,
    params: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    init_species_index: int,
    use_batch: bool,
    hooks: TEMPathHooks,
) -> _TEMBatchContext:
    selection: ModeSelection | ModeSelectionBatch
    if use_batch:
        ky_indices = [
            hooks.select_ky_index(np.asarray(grid_full.ky), float(ky))
            for ky in ky_slice
        ]
        grid = hooks.select_ky_grid(grid_full, ky_indices)
        selection = hooks.mode_selection_batch(
            np.arange(len(ky_indices), dtype=int), 0, hooks.midplane_index(grid)
        )
        dt_i = float(dt)
        steps_i = int(steps)
    else:
        ky_indices = [
            hooks.select_ky_index(np.asarray(grid_full.ky), float(ky_slice[0]))
        ]
        grid = hooks.select_ky_grid(grid_full, ky_indices[0])
        selection = hooks.mode_selection(
            ky_index=0, kx_index=0, z_index=hooks.midplane_index(grid)
        )
        dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = (
            int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)
        )

    state_np = np.zeros(
        (2, n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=np.complex64,
    )
    state_single = hooks.build_initial_condition(
        grid,
        geom,
        ky_index=np.arange(len(ky_indices), dtype=int),
        kx_index=0,
        Nl=n_laguerre,
        Nm=n_hermite,
        init_cfg=init_cfg,
    )
    state_np[int(init_species_index)] = np.asarray(state_single, dtype=np.complex64)
    return _TEMBatchContext(
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        grid=grid,
        selection=selection,
        dt=dt_i,
        steps=steps_i,
        state=jnp.asarray(state_np),
        cache=hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite),
    )


def _tem_time_config_for_batch(
    time_cfg: Any | None,
    *,
    dt: float,
    steps: int,
    sample_stride: int | None,
) -> Any | None:
    if time_cfg is None:
        return None
    time_cfg_i = replace(time_cfg, dt=dt, t_max=dt * steps)
    if sample_stride is not None:
        time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)
    return time_cfg_i


def _append_tem_krylov_fit(
    *,
    context: _TEMBatchContext,
    params: Any,
    terms: Any,
    krylov_cfg: Any | None,
    krylov_default: Any,
    diagnostic_norm: str,
    hooks: TEMPathHooks,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> None:
    eig, _vec = _krylov_eigenvalue(
        context.state,
        context.cache,
        params,
        terms,
        krylov_cfg or krylov_default,
        hooks,
    )
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    gamma, omega = hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    gammas.append(gamma)
    omegas.append(omega)
    ky_out.append(float(context.ky_slice[0]))


def _append_tem_streaming_fit(
    *,
    context: _TEMBatchContext,
    geom: Any,
    params: Any,
    terms: Any,
    time_cfg_i: Any,
    fit_policy: ScanFitWindowPolicy,
    mode_method: str,
    streaming_amp_floor: float,
    show_progress: bool,
    hooks: TEMPathHooks,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> None:
    t_total = float(time_cfg_i.t_max)
    tmin_i, tmax_i = fit_policy.window_at(context.batch_start)
    tmin_i, tmax_i = hooks.resolve_streaming_window(
        t_total,
        tmin_i,
        tmax_i,
        fit_policy.start_fraction,
        fit_policy.window_fraction,
        1.0,
    )
    _, gamma_vals, omega_vals = hooks.integrate_linear_diffrax_streaming(
        context.state,
        context.grid,
        geom,
        params,
        dt=context.dt,
        steps=context.steps,
        method=time_cfg_i.diffrax_solver,
        cache=context.cache,
        terms=terms,
        adaptive=time_cfg_i.diffrax_adaptive,
        rtol=time_cfg_i.diffrax_rtol,
        atol=time_cfg_i.diffrax_atol,
        max_steps=time_cfg_i.diffrax_max_steps,
        progress_bar=time_cfg_i.progress_bar,
        checkpoint=time_cfg_i.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal="phi",
        show_progress=show_progress,
        mode_ky_indices=np.arange(context.valid_count, dtype=int),
        mode_kx_index=0,
        mode_z_index=hooks.midplane_index(context.grid),
        mode_method=mode_method,
        amp_floor=streaming_amp_floor,
        return_state=False,
    )
    gamma_arr = np.asarray(gamma_vals)
    omega_arr = np.asarray(omega_vals)
    for local_idx in range(context.valid_count):
        gammas.append(float(gamma_arr[local_idx]))
        omegas.append(float(omega_arr[local_idx]))
        ky_out.append(float(context.ky_slice[local_idx]))


def _append_tem_saved_fit(
    *,
    context: _TEMBatchContext,
    geom: Any,
    params: Any,
    terms: Any,
    method: str,
    time_cfg_i: Any | None,
    mode_method: str,
    mode_only: bool,
    sample_stride: int | None,
    fit_policy: ScanFitWindowPolicy,
    diagnostic_norm: str,
    hooks: TEMPathHooks,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> None:
    if time_cfg_i is not None:
        _, phi_t = hooks.integrate_linear_from_config(
            context.state,
            context.grid,
            geom,
            params,
            time_cfg_i,
            cache=context.cache,
            terms=terms,
            save_mode=context.selection if mode_only else None,
            mode_method=mode_method,
        )
        stride = time_cfg_i.sample_stride
    else:
        stride = 1 if sample_stride is None else int(sample_stride)
        _, phi_t = hooks.integrate_linear(
            context.state,
            context.grid,
            geom,
            params,
            dt=context.dt,
            steps=context.steps,
            method=method,
            cache=context.cache,
            terms=terms,
            sample_stride=stride,
        )

    phi_t_np = np.asarray(phi_t)
    for local_idx in range(context.valid_count):
        if mode_only and phi_t_np.ndim <= 2:
            signal = _extract_mode_only_signal(phi_t_np, local_idx=local_idx)
        else:
            sel_local = hooks.mode_selection(
                ky_index=local_idx,
                kx_index=0,
                z_index=hooks.midplane_index(context.grid),
            )
            signal = hooks.extract_mode_time_series(
                phi_t_np, sel_local, method=mode_method
            )
        gamma, omega = fit_policy.fit_signal(
            signal,
            idx=context.batch_start + local_idx,
            dt=context.dt,
            stride=stride,
            params=params,
            diagnostic_norm=diagnostic_norm,
        )
        gammas.append(gamma)
        omegas.append(omega)
        ky_out.append(float(context.ky_slice[local_idx]))


def _tem_scan_batches(
    ky_values: np.ndarray,
    options: _TEMScanRuntimeOptions,
) -> Any:
    """Select scalar or fixed-width ky batching for the TEM scan."""

    if options.use_batch:
        return _iter_ky_batches(
            ky_values,
            ky_batch=options.ky_batch,
            fixed_batch_shape=options.fixed_batch_shape,
        )
    return _iter_ky_batches(ky_values, ky_batch=1, fixed_batch_shape=False)


def _append_tem_scan_batch(
    *,
    context: _TEMBatchContext,
    geom: Any,
    params: Any,
    terms: Any,
    options: _TEMScanRuntimeOptions,
    acc: _TEMScanAccumulator,
) -> None:
    """Route one prepared TEM batch through its selected solver path."""

    hooks = options.hooks
    if options.solver_key == "krylov":
        _append_tem_krylov_fit(
            context=context,
            params=params,
            terms=terms,
            krylov_cfg=options.krylov_cfg,
            krylov_default=options.krylov_default,
            diagnostic_norm=options.diagnostic_norm,
            hooks=hooks,
            gammas=acc.gammas,
            omegas=acc.omegas,
            ky_out=acc.ky_out,
        )
        return

    time_cfg_i = _tem_time_config_for_batch(
        options.time_cfg,
        dt=context.dt,
        steps=context.steps,
        sample_stride=options.sample_stride,
    )
    if time_cfg_i is not None and time_cfg_i.use_diffrax and options.streaming_fit:
        _append_tem_streaming_fit(
            context=context,
            geom=geom,
            params=params,
            terms=terms,
            time_cfg_i=time_cfg_i,
            fit_policy=options.fit_policy,
            mode_method=options.mode_method,
            streaming_amp_floor=options.streaming_amp_floor,
            show_progress=options.show_progress,
            hooks=hooks,
            gammas=acc.gammas,
            omegas=acc.omegas,
            ky_out=acc.ky_out,
        )
        return

    _append_tem_saved_fit(
        context=context,
        geom=geom,
        params=params,
        terms=terms,
        method=options.method,
        time_cfg_i=time_cfg_i,
        mode_method=options.mode_method,
        mode_only=options.mode_only,
        sample_stride=options.sample_stride,
        fit_policy=options.fit_policy,
        diagnostic_norm=options.diagnostic_norm,
        hooks=hooks,
        gammas=acc.gammas,
        omegas=acc.omegas,
        ky_out=acc.ky_out,
    )


def _run_tem_scan_loop(
    *,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    terms: Any,
    init_cfg: Any,
    options: _TEMScanRuntimeOptions,
) -> _TEMScanAccumulator:
    """Prepare and execute all TEM scan batches."""

    acc = _TEMScanAccumulator(gammas=[], omegas=[], ky_out=[])
    for batch_start, ky_slice, valid_count in _tem_scan_batches(ky_values, options):
        context = _prepare_tem_scan_batch(
            batch_start=batch_start,
            ky_slice=ky_slice,
            valid_count=valid_count,
            grid_full=grid_full,
            geom=geom,
            params=params,
            init_cfg=init_cfg,
            n_laguerre=options.n_laguerre,
            n_hermite=options.n_hermite,
            dt=options.dt,
            steps=options.steps,
            init_species_index=options.init_species_index,
            use_batch=options.use_batch,
            hooks=options.hooks,
        )
        _append_tem_scan_batch(
            context=context,
            geom=geom,
            params=params,
            terms=terms,
            options=options,
            acc=acc,
        )
    return acc


def run_tem_krylov_linear_path(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    n_laguerre: int,
    n_hermite: int,
    sel: ModeSelection,
    krylov_cfg: Any | None,
    krylov_default: Any,
    diagnostic_norm: str,
    hooks: TEMPathHooks,
) -> LinearRunResult:
    """Run the single-ky TEM Krylov branch and package a linear result."""

    cfg_use = krylov_cfg or krylov_default
    cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)
    eig, vec = _krylov_eigenvalue(G0_jax, cache, params, terms, cfg_use, hooks)
    term_cfg = hooks.linear_terms_to_term_config(terms)
    phi = hooks.compute_fields_cached(vec, cache, params, terms=term_cfg).phi
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    gamma, omega = hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    return hooks.linear_run_result(
        t=np.array([0.0]),
        phi_t=np.asarray(phi)[None, ...],
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )


def _validate_tem_time_fit_signal(fit_signal: str) -> None:
    if fit_signal not in {"phi", "density"}:
        raise ValueError("fit_signal must be 'phi' or 'density'")


def _resolve_tem_time_path_timing(
    *,
    dt: float,
    steps: int,
    time_cfg: Any | None,
    sample_stride: int | None,
) -> _TEMTimePathTiming:
    time_cfg_use = time_cfg
    if time_cfg is not None:
        if sample_stride is not None:
            time_cfg_use = replace(time_cfg, sample_stride=sample_stride)
        assert time_cfg_use is not None
        return _TEMTimePathTiming(
            dt=float(time_cfg_use.dt),
            steps=int(round(time_cfg_use.t_max / time_cfg_use.dt)),
            stride=int(time_cfg_use.sample_stride),
            time_cfg=time_cfg_use,
        )
    return _TEMTimePathTiming(
        dt=float(dt),
        steps=int(steps),
        stride=1 if sample_stride is None else int(sample_stride),
        time_cfg=None,
    )


def _integrate_tem_time_path_trace(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    cache: Any,
    timing: _TEMTimePathTiming,
    method: str,
    fit_signal: str,
    density_species_index: int,
    show_progress: bool,
    hooks: TEMPathHooks,
) -> _TEMTimePathTrace:
    if fit_signal == "density":
        diag = hooks.integrate_linear_diagnostics(
            G0_jax,
            grid,
            geom,
            params,
            dt=timing.dt,
            steps=timing.steps,
            method=method,
            cache=cache,
            terms=terms,
            sample_stride=timing.stride,
            species_index=density_species_index,
        )
        _, phi_t, density_t, *_ = diag
    elif timing.time_cfg is not None:
        _, phi_t = hooks.integrate_linear_from_config(
            G0_jax,
            grid,
            geom,
            params,
            timing.time_cfg,
            cache=cache,
            terms=terms,
            show_progress=show_progress,
        )
        density_t = None
    else:
        _, phi_t = hooks.integrate_linear(
            G0_jax,
            grid,
            geom,
            params,
            dt=timing.dt,
            steps=timing.steps,
            method=method,
            cache=cache,
            terms=terms,
            sample_stride=timing.stride,
            show_progress=show_progress,
        )
        density_t = None

    phi_t_np = np.asarray(phi_t)
    return _TEMTimePathTrace(
        t=np.arange(phi_t_np.shape[0]) * timing.dt * timing.stride,
        phi_t=phi_t_np,
        density_t=None if density_t is None else np.asarray(density_t),
    )


def _tem_time_path_signal(
    trace: _TEMTimePathTrace,
    *,
    fit_signal: str,
    sel: ModeSelection,
    mode_method: str,
    hooks: TEMPathHooks,
) -> np.ndarray:
    if fit_signal == "density" and trace.density_t is not None:
        return hooks.extract_mode_time_series(trace.density_t, sel, method=mode_method)
    return hooks.extract_mode_time_series(trace.phi_t, sel, method=mode_method)


def _fit_tem_time_path_signal(
    *,
    t: np.ndarray,
    signal: np.ndarray,
    policy: _TEMTimePathFitPolicy,
    hooks: TEMPathHooks,
) -> tuple[float, float]:
    auto_fit_kwargs: dict[str, Any] = {
        "window_fraction": policy.window_fraction,
        "min_points": policy.min_points,
        "start_fraction": policy.start_fraction,
        "growth_weight": policy.growth_weight,
        "require_positive": policy.require_positive,
        "min_amp_fraction": policy.min_amp_fraction,
    }
    if policy.auto_window and policy.tmin is None and policy.tmax is None:
        gamma, omega, _tmin, _tmax = hooks.fit_growth_rate_auto(
            t, signal, **auto_fit_kwargs
        )
        return gamma, omega
    try:
        return hooks.fit_growth_rate(t, signal, tmin=policy.tmin, tmax=policy.tmax)
    except ValueError:
        gamma, omega, _tmin, _tmax = hooks.fit_growth_rate_auto(
            t, signal, **auto_fit_kwargs
        )
        return gamma, omega


def _tem_time_path_result(
    *,
    trace: _TEMTimePathTrace,
    gamma: float,
    omega: float,
    grid: Any,
    sel: ModeSelection,
    hooks: TEMPathHooks,
) -> LinearRunResult:
    return hooks.linear_run_result(
        t=trace.t,
        phi_t=trace.phi_t,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )


def run_tem_time_linear_path(
    *,
    G0_jax: Any,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float,
    steps: int,
    method: str,
    time_cfg: Any | None,
    sample_stride: int | None,
    fit_signal: str,
    density_species_index: int,
    sel: ModeSelection,
    mode_method: str,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    diagnostic_norm: str,
    show_progress: bool,
    hooks: TEMPathHooks,
) -> LinearRunResult:
    """Run the single-ky TEM time-integration branch and fit the selected signal."""

    _validate_tem_time_fit_signal(fit_signal)
    cache = hooks.build_linear_cache(grid, geom, params, n_laguerre, n_hermite)
    timing = _resolve_tem_time_path_timing(
        dt=dt,
        steps=steps,
        time_cfg=time_cfg,
        sample_stride=sample_stride,
    )
    trace = _integrate_tem_time_path_trace(
        G0_jax=G0_jax,
        grid=grid,
        geom=geom,
        params=params,
        terms=terms,
        cache=cache,
        timing=timing,
        method=method,
        fit_signal=fit_signal,
        density_species_index=density_species_index,
        show_progress=show_progress,
        hooks=hooks,
    )
    signal = _tem_time_path_signal(
        trace,
        fit_signal=fit_signal,
        sel=sel,
        mode_method=mode_method,
        hooks=hooks,
    )
    fit_policy = _tem_time_path_fit_policy_from_locals(locals())
    gamma, omega = _fit_tem_time_path_signal(
        t=trace.t,
        signal=signal,
        policy=fit_policy,
        hooks=hooks,
    )
    gamma, omega = hooks.normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    return _tem_time_path_result(
        trace=trace,
        gamma=gamma,
        omega=omega,
        grid=grid,
        sel=sel,
        hooks=hooks,
    )


def run_tem_scan_batches(
    *,
    ky_values: np.ndarray,
    grid_full: Any,
    geom: Any,
    params: Any,
    terms: Any,
    init_cfg: Any,
    n_laguerre: int,
    n_hermite: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    time_cfg: Any | None,
    solver_key: str,
    krylov_cfg: Any | None,
    krylov_default: Any,
    tmin: float | None,
    tmax: float | None,
    auto_window: bool,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    mode_method: str,
    mode_only: bool,
    sample_stride: int | None,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    init_species_index: int,
    diagnostic_norm: str,
    use_batch: bool,
    hooks: TEMPathHooks,
    show_progress: bool,
) -> LinearScanResult:
    """Run TEM scan batches across Krylov, streaming, and saved-time branches."""

    fit_policy = _tem_scan_fit_policy_from_locals(locals(), hooks=hooks)
    options = _tem_scan_runtime_options_from_locals(locals())
    return _run_tem_scan_loop(
        ky_values=ky_values,
        grid_full=grid_full,
        geom=geom,
        params=params,
        terms=terms,
        init_cfg=init_cfg,
        options=options,
    ).result(hooks)


def _tem_hooks() -> TEMPathHooks:
    return TEMPathHooks(
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
class _TEMScanRequest:
    ky_values: np.ndarray
    Nl: int
    Nm: int
    dt: float | np.ndarray
    steps: int | np.ndarray
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
    mode_only: bool
    terms: LinearTerms | None
    sample_stride: int | None
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
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
    hooks: TEMPathHooks


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


def _tem_scan_request_from_locals(values: dict[str, Any]) -> _TEMScanRequest:
    """Pack public ``run_tem_scan`` arguments once for internal routing."""

    return _TEMScanRequest(
        **{field.name: values[field.name] for field in fields(_TEMScanRequest)}
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


def _resolve_tem_scan_setup(request: _TEMScanRequest) -> _TEMLinearSetup:
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


def _run_tem_scan_request(request: _TEMScanRequest) -> LinearScanResult:
    setup = _resolve_tem_scan_setup(request)
    solver_key = normalize_solver_key(request.solver)
    mode_method = resolve_scan_mode_method(
        request.mode_method, mode_only=request.mode_only
    )
    use_batch = should_use_ky_batch(
        ky_batch=request.ky_batch,
        solver_key=solver_key,
        dt=request.dt,
        steps=request.steps,
        tmin=request.tmin,
        tmax=request.tmax,
    )
    _validate_tem_species_indices(
        init_species_index=request.init_species_index,
        density_species_index=request.density_species_index,
    )
    return run_tem_scan_batches(
        ky_values=np.asarray(request.ky_values, dtype=float),
        grid_full=setup.grid_full,
        geom=setup.geom,
        params=setup.params,
        terms=setup.terms,
        init_cfg=setup.cfg.init,
        n_laguerre=request.Nl,
        n_hermite=request.Nm,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        time_cfg=request.time_cfg,
        solver_key=solver_key,
        krylov_cfg=request.krylov_cfg,
        krylov_default=TEM_KRYLOV_DEFAULT,
        tmin=request.tmin,
        tmax=request.tmax,
        auto_window=request.auto_window,
        window_fraction=request.window_fraction,
        min_points=request.min_points,
        start_fraction=request.start_fraction,
        growth_weight=request.growth_weight,
        require_positive=request.require_positive,
        min_amp_fraction=request.min_amp_fraction,
        mode_method=mode_method,
        mode_only=request.mode_only,
        sample_stride=request.sample_stride,
        ky_batch=request.ky_batch,
        fixed_batch_shape=request.fixed_batch_shape,
        streaming_fit=request.streaming_fit,
        streaming_amp_floor=request.streaming_amp_floor,
        init_species_index=request.init_species_index,
        diagnostic_norm=request.diagnostic_norm,
        use_batch=use_batch,
        hooks=setup.hooks,
        show_progress=request.show_progress,
    )


def _run_tem_linear_request(request: _TEMLinearRequest) -> LinearRunResult:
    setup = _resolve_tem_linear_setup(request)
    state = _prepare_tem_linear_state(setup, request)
    if request.solver.lower() == "krylov":
        return run_tem_krylov_linear_path(
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
    return run_tem_time_linear_path(
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

    return _run_tem_scan_request(_tem_scan_request_from_locals(locals()))

@dataclass(frozen=True)
class _ETGLinearSetup:
    """Solver-ready single-ky ETG state shared by Krylov and time paths."""

    cfg: ETGBaseCase
    grid: Any
    geom: Any
    params: Any
    terms: LinearTerms
    selection: ModeSelection
    electron_index: int
    initial_state: Any


@dataclass(frozen=True)
class _ETGTimePathOptions:
    """Private fit and streaming policy for ETG saved-time integrations."""

    fit_key: str
    streaming_fit: bool
    streaming_amp_floor: float
    reference_growth_window: bool
    reference_navg_fraction: float
    mode_method: str
    tmin: float | None
    tmax: float | None
    auto_window: bool
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    diagnostic_norm: str


@dataclass(frozen=True)
class _ETGLinearRequest:
    """Raw public ETG single-ky inputs before solver policies are resolved."""

    ky_target: float
    Nl: int
    Nm: int
    dt: float
    steps: int
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
    mode_method: str
    terms: LinearTerms | None
    sample_stride: int | None
    fit_signal: str
    streaming_fit: bool
    streaming_amp_floor: float
    reference_growth_window: bool
    reference_navg_fraction: float
    diagnostic_norm: str
    show_progress: bool


def _etg_linear_request_from_locals(values: dict[str, Any]) -> _ETGLinearRequest:
    """Build an ETG request from ``run_etg_linear`` locals."""

    names = {field.name for field in fields(_ETGLinearRequest)}
    return _ETGLinearRequest(**{name: values[name] for name in names})


def _default_etg_params(cfg: ETGBaseCase, geom: Any, Nm: int) -> LinearParams:
    """Build ETG benchmark species parameters using the tracked normalization."""

    if getattr(cfg.model, "adiabatic_ions", False):
        return _electron_only_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=ETG_OMEGA_D_SCALE,
            omega_star_scale=ETG_OMEGA_STAR_SCALE,
            rho_star=ETG_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=Nm,
        )
    return _two_species_params(
        cfg.model,
        kpar_scale=float(geom.gradpar()),
        omega_d_scale=ETG_OMEGA_D_SCALE,
        omega_star_scale=ETG_OMEGA_STAR_SCALE,
        rho_star=ETG_RHO_STAR,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
        nhermite=Nm,
    )


def _default_etg_terms() -> LinearTerms:
    """Return the electrostatic ETG benchmark term contract."""

    return LinearTerms(apar=0.0, bpar=0.0, hypercollisions=1.0)


def _build_etg_linear_setup(
    *,
    cfg: ETGBaseCase | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    ky_target: float,
    Nl: int,
    Nm: int,
) -> _ETGLinearSetup:
    """Create the selected-grid initial state for one ETG benchmark point."""

    cfg_use = cfg or ETGBaseCase()
    grid_full = build_spectral_grid(cfg_use.grid)
    geom = SAlphaGeometry.from_config(cfg_use.geometry)
    params_use = params if params is not None else _default_etg_params(cfg_use, geom, Nm)
    terms_use = terms if terms is not None else _default_etg_terms()

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))

    charge = np.atleast_1d(np.asarray(params_use.charge_sign))
    ns = int(charge.size)
    electron_index = int(np.argmin(charge))
    G0 = np.zeros(
        (ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
    )
    G0_single = _build_initial_condition(
        grid,
        geom,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=cfg_use.init,
    )
    G0[electron_index] = np.asarray(G0_single, dtype=np.complex64)
    return _ETGLinearSetup(
        cfg=cfg_use,
        grid=grid,
        geom=geom,
        params=params_use,
        terms=terms_use,
        selection=sel,
        electron_index=electron_index,
        initial_state=jnp.asarray(G0),
    )


def _etg_linear_result(
    setup: _ETGLinearSetup,
    *,
    t: np.ndarray,
    phi_t_np: np.ndarray,
    gamma: float,
    omega: float,
) -> LinearRunResult:
    """Pack a single-ky ETG run result with the selected physical ky."""

    return LinearRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(setup.grid.ky[setup.selection.ky_index]),
        selection=setup.selection,
    )


def _valid_etg_growth(
    gamma_val: float, omega_val: float, *, require_positive: bool
) -> bool:
    """Return whether a Krylov ETG result is acceptable for auto solver mode."""

    if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
        return False
    if require_positive and gamma_val <= 0.0:
        return False
    return True


def _run_etg_krylov_path(
    setup: _ETGLinearSetup,
    *,
    Nl: int,
    Nm: int,
    krylov_cfg: KrylovConfig | None,
    diagnostic_norm: str,
) -> LinearRunResult:
    """Solve one ETG point with the Krylov eigenpath."""

    cfg_use = krylov_cfg or ETG_KRYLOV_DEFAULT
    cache = build_linear_cache(setup.grid, setup.geom, setup.params, Nl, Nm)
    krylov_kwargs = {
        "terms": setup.terms,
        **{name: getattr(cfg_use, name) for name in _ETG_KRYLOV_FORWARD_KEYS},
    }
    eig, vec = dominant_eigenpair(
        setup.initial_state, cache, setup.params, **krylov_kwargs
    )
    term_cfg = linear_terms_to_term_config(setup.terms)
    phi = compute_fields_cached(vec, cache, setup.params, terms=term_cfg).phi
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    if cfg_use.omega_sign != 0:
        omega = float(np.sign(cfg_use.omega_sign)) * abs(omega)
    gamma, omega = _normalize_growth_rate(
        gamma, omega, setup.params, diagnostic_norm
    )
    return _etg_linear_result(
        setup,
        t=np.array([0.0]),
        phi_t_np=np.asarray(phi)[None, ...],
        gamma=gamma,
        omega=omega,
    )


def _resolve_etg_time_config(
    cfg: ETGBaseCase,
    time_cfg: TimeConfig | None,
    *,
    streaming_fit: bool,
    dt: float,
    steps: int,
    sample_stride: int | None,
) -> tuple[TimeConfig | None, float, int]:
    """Resolve explicit ETG time configuration without changing fit semantics."""

    time_cfg_use = time_cfg
    if time_cfg_use is None and streaming_fit and cfg.time.use_diffrax:
        max_steps = max(int(cfg.time.diffrax_max_steps), int(steps))
        time_cfg_use = replace(
            cfg.time,
            dt=dt,
            t_max=dt * steps,
            diffrax_max_steps=max_steps,
        )
        if sample_stride is not None:
            time_cfg_use = replace(time_cfg_use, sample_stride=sample_stride)
    if time_cfg_use is not None:
        if sample_stride is not None:
            time_cfg_use = replace(time_cfg_use, sample_stride=sample_stride)
        if time_cfg is not None:
            dt = float(time_cfg_use.dt)
            steps = int(round(time_cfg_use.t_max / time_cfg_use.dt))
    return time_cfg_use, dt, steps


def _etg_auto_fit_options(
    *,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> dict[str, Any]:
    """Pack the shared automatic-window policy for ETG trace fits."""

    return {
        "window_fraction": window_fraction,
        "min_points": min_points,
        "start_fraction": start_fraction,
        "growth_weight": growth_weight,
        "require_positive": require_positive,
        "min_amp_fraction": min_amp_fraction,
    }


def _fit_etg_reference_growth(
    setup: _ETGLinearSetup,
    *,
    phi_t_np: np.ndarray,
    t: np.ndarray,
    reference_navg_fraction: float,
    mode_method: str,
    diagnostic_norm: str,
) -> tuple[float, float]:
    """Fit ETG ``phi`` with the legacy instantaneous-growth reference window."""

    gamma, omega, _gamma_t, _omega_t, _t_mid = instantaneous_growth_rate_from_phi(
        phi_t_np,
        t,
        setup.selection,
        navg_fraction=reference_navg_fraction,
        mode_method=mode_method,
    )
    return _normalize_growth_rate(gamma, omega, setup.params, diagnostic_norm)


def _fit_etg_auto_signal(
    setup: _ETGLinearSetup,
    *,
    phi_t_np: np.ndarray,
    density_np: np.ndarray | None,
    t: np.ndarray,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    diagnostic_norm: str,
) -> tuple[float, float]:
    """Select the most stable ETG signal and fit its automatic growth window."""

    gamma, omega = _select_fit_signal_auto(
        t,
        phi_t_np,
        density_np,
        setup.selection,
        mode_method=mode_method,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        max_amp_fraction=0.9,
        window_method="loglinear",
        max_fraction=0.8,
        end_fraction=0.9,
        num_windows=8,
        phase_weight=0.2,
        length_weight=0.05,
        min_r2=0.0,
        late_penalty=0.1,
        min_slope=None,
        min_slope_frac=0.0,
        slope_var_weight=0.0,
    )[2:]
    return _normalize_growth_rate(gamma, omega, setup.params, diagnostic_norm)


def _fit_etg_selected_signal(
    setup: _ETGLinearSetup,
    *,
    phi_t_np: np.ndarray,
    density_np: np.ndarray | None,
    t: np.ndarray,
    fit_key: str,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    auto_window: bool,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    diagnostic_norm: str,
) -> tuple[float, float]:
    """Fit a caller-selected ETG signal with manual-window fallback."""

    signal = _select_fit_signal(
        phi_t_np,
        density_np,
        setup.selection,
        fit_signal=fit_key,
        mode_method=mode_method,
    )
    auto_fit_kwargs = _etg_auto_fit_options(
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
    )
    use_auto = auto_window and tmin is None and tmax is None
    if not use_auto and not scan_window_valid(t, tmin, tmax):
        use_auto = True
    if use_auto:
        gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
            t, signal, **auto_fit_kwargs
        )
    else:
        try:
            gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)
        except ValueError:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                t, signal, **auto_fit_kwargs
            )
    return _normalize_growth_rate(gamma, omega, setup.params, diagnostic_norm)


def _fit_etg_time_trace(
    setup: _ETGLinearSetup,
    *,
    phi_t: Any,
    density_t: Any,
    dt: float,
    stride: int,
    options: _ETGTimePathOptions,
) -> LinearRunResult:
    """Fit ETG growth and frequency from a saved time trace."""

    phi_t_np = np.asarray(phi_t)
    t = np.arange(phi_t_np.shape[0]) * dt * stride
    density_np = None if density_t is None else np.asarray(density_t)
    if options.reference_growth_window and options.fit_key == "phi":
        gamma, omega = _fit_etg_reference_growth(
            setup,
            phi_t_np=phi_t_np,
            t=t,
            reference_navg_fraction=options.reference_navg_fraction,
            mode_method=options.mode_method,
            diagnostic_norm=options.diagnostic_norm,
        )
    elif options.fit_key == "auto":
        gamma, omega = _fit_etg_auto_signal(
            setup,
            phi_t_np=phi_t_np,
            density_np=density_np,
            t=t,
            mode_method=options.mode_method,
            tmin=options.tmin,
            tmax=options.tmax,
            window_fraction=options.window_fraction,
            min_points=options.min_points,
            start_fraction=options.start_fraction,
            growth_weight=options.growth_weight,
            require_positive=options.require_positive,
            min_amp_fraction=options.min_amp_fraction,
            diagnostic_norm=options.diagnostic_norm,
        )
    else:
        gamma, omega = _fit_etg_selected_signal(
            setup,
            phi_t_np=phi_t_np,
            density_np=density_np,
            t=t,
            fit_key=options.fit_key,
            mode_method=options.mode_method,
            tmin=options.tmin,
            tmax=options.tmax,
            auto_window=options.auto_window,
            window_fraction=options.window_fraction,
            min_points=options.min_points,
            start_fraction=options.start_fraction,
            growth_weight=options.growth_weight,
            require_positive=options.require_positive,
            min_amp_fraction=options.min_amp_fraction,
            diagnostic_norm=options.diagnostic_norm,
        )
    return _etg_linear_result(setup, t=t, phi_t_np=phi_t_np, gamma=gamma, omega=omega)


def _run_etg_streaming_density_fit(
    setup: _ETGLinearSetup,
    *,
    time_cfg: TimeConfig,
    cache: Any,
    dt: float,
    steps: int,
    options: _ETGTimePathOptions,
    show_progress: bool,
) -> LinearRunResult:
    """Run the memory-light Diffrax density streaming fit path."""

    t_total = float(dt * steps)
    tmin_i, tmax_i = _resolve_streaming_window(
        t_total,
        options.tmin,
        options.tmax,
        options.start_fraction,
        options.window_fraction,
        1.0,
    )
    G_last, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
        setup.initial_state,
        setup.grid,
        setup.geom,
        setup.params,
        dt=dt,
        steps=steps,
        method=time_cfg.diffrax_solver,
        cache=cache,
        terms=setup.terms,
        adaptive=False,
        rtol=time_cfg.diffrax_rtol,
        atol=time_cfg.diffrax_atol,
        max_steps=time_cfg.diffrax_max_steps,
        show_progress=show_progress,
        progress_bar=time_cfg.progress_bar,
        checkpoint=time_cfg.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal="density",
        mode_ky_indices=np.array([0], dtype=int),
        mode_kx_index=0,
        mode_z_index=_midplane_index(setup.grid),
        mode_method=options.mode_method,
        amp_floor=options.streaming_amp_floor,
        density_species_index=setup.electron_index,
        return_state=True,
    )
    gamma = float(np.asarray(gamma_vals)[0])
    omega = float(np.asarray(omega_vals)[0])
    gamma, omega = _normalize_growth_rate(
        gamma, omega, setup.params, options.diagnostic_norm
    )
    if G_last is not None and G_last.ndim == 7:
        G_last = G_last[0]
    if G_last is None:
        raise ValueError("Expected final state from streaming fit; got None.")
    term_cfg = linear_terms_to_term_config(setup.terms)
    phi_last = compute_fields_cached(
        G_last, cache, setup.params, terms=term_cfg
    ).phi
    return _etg_linear_result(
        setup,
        t=np.array([tmax_i]),
        phi_t_np=np.asarray(jnp.asarray(phi_last)[None, ...]),
        gamma=gamma,
        omega=omega,
    )


def _integrate_etg_configured_history(
    setup: _ETGLinearSetup,
    *,
    time_cfg: TimeConfig,
    cache: Any,
    dt: float,
    steps: int,
    fit_key: str,
    show_progress: bool,
) -> tuple[Any, Any | None, int]:
    """Integrate ETG saved history using an explicit or synthesized TimeConfig."""

    if fit_key in {"density", "auto"}:
        if time_cfg.use_diffrax:
            _, saved = integrate_linear_diffrax(
                setup.initial_state,
                setup.grid,
                setup.geom,
                setup.params,
                dt=dt,
                steps=steps,
                method=time_cfg.diffrax_solver,
                cache=cache,
                terms=setup.terms,
                adaptive=time_cfg.diffrax_adaptive,
                rtol=time_cfg.diffrax_rtol,
                atol=time_cfg.diffrax_atol,
                max_steps=time_cfg.diffrax_max_steps,
                show_progress=show_progress,
                progress_bar=time_cfg.progress_bar,
                checkpoint=time_cfg.checkpoint,
                sample_stride=time_cfg.sample_stride,
                return_state=time_cfg.save_state,
                save_field="phi+density",
                density_species_index=setup.electron_index,
            )
            phi_t, density_t = saved
            return phi_t, density_t, time_cfg.sample_stride
        diag = integrate_linear_diagnostics(
            setup.initial_state,
            setup.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=time_cfg.method,
            cache=cache,
            terms=setup.terms,
            sample_stride=time_cfg.sample_stride,
            species_index=setup.electron_index,
        )
        phi_t = diag[1]
        density_t = diag[2] if len(diag) > 2 else None
        return phi_t, density_t, time_cfg.sample_stride

    _, phi_t = integrate_linear_from_config(
        setup.initial_state,
        setup.grid,
        setup.geom,
        setup.params,
        time_cfg,
        cache=cache,
        terms=setup.terms,
        show_progress=show_progress,
    )
    return phi_t, None, time_cfg.sample_stride


def _integrate_etg_unconfigured_history(
    setup: _ETGLinearSetup,
    *,
    dt: float,
    steps: int,
    method: str,
    fit_key: str,
    sample_stride: int | None,
    show_progress: bool,
) -> tuple[Any, Any | None, int]:
    """Integrate ETG saved history without a TimeConfig object."""

    stride = 1 if sample_stride is None else int(sample_stride)
    if fit_key in {"density", "auto"}:
        diag = integrate_linear_diagnostics(
            setup.initial_state,
            setup.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=method,
            terms=setup.terms,
            sample_stride=stride,
            species_index=setup.electron_index,
        )
        phi_t = diag[1]
        density_t = diag[2] if len(diag) > 2 else None
        return phi_t, density_t, stride

    _, phi_t = integrate_linear(
        setup.initial_state,
        setup.grid,
        setup.geom,
        setup.params,
        dt=dt,
        steps=steps,
        method=method,
        terms=setup.terms,
        sample_stride=stride,
        show_progress=show_progress,
    )
    return phi_t, None, stride


def _run_etg_time_path(
    setup: _ETGLinearSetup,
    *,
    Nl: int,
    Nm: int,
    time_cfg: TimeConfig | None,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    options: _ETGTimePathOptions,
    show_progress: bool,
) -> LinearRunResult:
    """Run ETG saved-time or streaming time paths and fit the trace."""

    time_cfg_use, dt, steps = _resolve_etg_time_config(
        setup.cfg,
        time_cfg,
        streaming_fit=options.streaming_fit,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
    )
    if time_cfg_use is not None:
        cache = build_linear_cache(
            setup.grid,
            setup.geom,
            setup.params,
            Nl,
            Nm,
        )
        if (
            options.fit_key in {"density", "auto"}
            and options.streaming_fit
            and time_cfg_use.use_diffrax
        ):
            return _run_etg_streaming_density_fit(
                setup,
                time_cfg=time_cfg_use,
                cache=cache,
                dt=dt,
                steps=steps,
                options=options,
                show_progress=show_progress,
            )
        phi_t, density_t, stride = _integrate_etg_configured_history(
            setup,
            time_cfg=time_cfg_use,
            cache=cache,
            dt=dt,
            steps=steps,
            fit_key=options.fit_key,
            show_progress=show_progress,
        )
    else:
        phi_t, density_t, stride = _integrate_etg_unconfigured_history(
            setup,
            dt=dt,
            steps=steps,
            method=method,
            fit_key=options.fit_key,
            sample_stride=sample_stride,
            show_progress=show_progress,
        )

    return _fit_etg_time_trace(
        setup,
        phi_t=phi_t,
        density_t=density_t,
        dt=dt,
        stride=stride,
        options=options,
    )


def _run_etg_linear_request(request: _ETGLinearRequest) -> LinearRunResult:
    """Resolve ETG solver policies and execute one single-ky linear point."""

    setup = _build_etg_linear_setup(
        cfg=request.cfg,
        params=request.params,
        terms=request.terms,
        ky_target=request.ky_target,
        Nl=request.Nl,
        Nm=request.Nm,
    )
    solver_key = request.solver.strip().lower()
    fit_key = request.fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    streaming_fit = request.streaming_fit
    if fit_key == "auto" and streaming_fit:
        streaming_fit = False
    time_options = _ETGTimePathOptions(
        fit_key=fit_key,
        streaming_fit=streaming_fit,
        streaming_amp_floor=request.streaming_amp_floor,
        reference_growth_window=request.reference_growth_window,
        reference_navg_fraction=request.reference_navg_fraction,
        mode_method=request.mode_method,
        tmin=request.tmin,
        tmax=request.tmax,
        auto_window=request.auto_window,
        window_fraction=request.window_fraction,
        min_points=request.min_points,
        start_fraction=request.start_fraction,
        growth_weight=request.growth_weight,
        require_positive=request.require_positive,
        min_amp_fraction=request.min_amp_fraction,
        diagnostic_norm=request.diagnostic_norm,
    )
    auto_solver = solver_key == "auto"
    if auto_solver:
        solver_key = "krylov"

    if solver_key == "krylov":
        krylov_result = _run_etg_krylov_path(
            setup,
            Nl=request.Nl,
            Nm=request.Nm,
            krylov_cfg=request.krylov_cfg,
            diagnostic_norm=request.diagnostic_norm,
        )
        if auto_solver and not _valid_etg_growth(
            krylov_result.gamma,
            krylov_result.omega,
            require_positive=request.require_positive,
        ):
            solver_key = "time"
        else:
            return krylov_result

    if solver_key != "krylov":
        return _run_etg_time_path(
            setup,
            Nl=request.Nl,
            Nm=request.Nm,
            time_cfg=request.time_cfg,
            dt=request.dt,
            steps=request.steps,
            method=request.method,
            sample_stride=request.sample_stride,
            options=time_options,
            show_progress=request.show_progress,
        )

    raise ValueError(f"Unsupported ETG linear solver '{request.solver}'.")


def run_etg_linear(
    ky_target: float = 3.0,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
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
    mode_method: str = "project",
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    fit_signal: str = "auto",
    streaming_fit: bool = False,
    streaming_amp_floor: float = 1.0e-30,
    reference_growth_window: bool = False,
    reference_navg_fraction: float = 0.5,
    diagnostic_norm: str = "none",
    show_progress: bool = False,
) -> LinearRunResult:
    """Run an ETG linear benchmark and extract growth rate."""

    return _run_etg_linear_request(_etg_linear_request_from_locals(locals()))

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


def _prepare_etg_scan_setup(request: _ETGScanRequest) -> _ETGScanSetup:
    """Prepare ETG scan geometry, species, solver, and fit policies."""

    cfg_use = request.cfg or ETGBaseCase()
    grid_full = build_spectral_grid(cfg_use.grid)
    geom = SAlphaGeometry.from_config(cfg_use.geometry)
    params_use = _default_etg_scan_params(cfg_use, geom, request.Nm, request.params)
    terms_use = _default_etg_scan_terms(request.terms)
    solver_key = normalize_solver_key(request.solver)
    fit_key = normalize_fit_signal(request.fit_signal)
    auto_solver = solver_key == "auto"
    if auto_solver:
        solver_key = "time"
    streaming_fit, mode_only = apply_auto_fit_scan_policy(
        fit_key,
        streaming_fit=request.streaming_fit,
        mode_only=request.mode_only,
    )
    mode_method = resolve_scan_mode_method(request.mode_method, mode_only=mode_only)
    fit_policy = _build_etg_scan_fit_policy(
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
            ky_batch=request.ky_batch,
            solver_key=solver_key,
            dt=request.dt,
            steps=request.steps,
            tmin=request.tmin,
            tmax=request.tmax,
        ),
        fit_policy=fit_policy,
    )


# ETG scan Krylov and time-batch policies live with the scan owner.
@dataclass(frozen=True)
class ETGTimeBatchResult:
    """Time-path data for one ETG scan batch after optional streaming handling."""

    handled: bool
    phi_t: np.ndarray | None = None
    density_t: np.ndarray | None = None
    t: np.ndarray | None = None
    stride: int = 1


@dataclass(frozen=True)
class _ETGTimeFitContext:
    ky_slice: np.ndarray
    valid_count: int
    batch_start: int
    fit_key: str
    fit_policy: Any
    params: Any
    diagnostic_norm: str
    mode_method: str
    mode_only: bool
    mode_z_index: int
    reference_growth_window: bool
    reference_navg_fraction: float
    auto_solver: bool
    require_positive: bool
    cfg: Any
    Nl: int
    Nm: int
    dt_i: float
    steps_i: int
    method: str
    krylov_cfg: Any
    show_progress: bool
    gammas: list[float]
    omegas: list[float]
    ky_out: list[float]


@dataclass(frozen=True)
class _ETGTimeBatchContext:
    G0_jax: jnp.ndarray
    grid: Any
    geom: Any
    params: Any
    cache: Any
    terms: Any
    time_cfg: Any
    dt_i: float
    steps_i: int
    method: str
    sample_stride: int | None
    fit_key: str
    need_density: bool
    streaming_fit: bool
    streaming_amp_floor: float
    mode_method: str
    mode_only: bool
    sel: Any
    batch_start: int
    valid_count: int
    ky_slice: np.ndarray
    tmin: float | None
    tmax: float | None
    start_fraction: float
    window_fraction: float
    electron_index: int
    diagnostic_norm: str
    show_progress: bool
    gammas: list[float]
    omegas: list[float]
    ky_out: list[float]


def _etg_time_batch_context_from_locals(values: dict[str, Any]) -> _ETGTimeBatchContext:
    """Pack ``run_etg_time_batch`` arguments for internal routing."""

    return _ETGTimeBatchContext(
        **{field.name: values[field.name] for field in fields(_ETGTimeBatchContext)}
    )


def run_etg_krylov_batch(
    *,
    G0_jax: jnp.ndarray,
    cache: Any,
    params: Any,
    terms: Any,
    krylov_cfg: Any,
    prev_vec: jnp.ndarray | None,
    prev_eig: complex | None,
    diagnostic_norm: str,
) -> tuple[float, float, jnp.ndarray | None, complex | None]:
    """Run one ETG Krylov scan point with continuation-aware branch selection."""

    cfg_use = krylov_cfg or ETG_KRYLOV_DEFAULT
    use_cont = bool(cfg_use.continuation)
    v0_use = G0_jax
    v_ref = None
    shift_override = cfg_use.shift
    shift_selection_use = cfg_use.shift_selection
    if use_cont and prev_vec is not None and prev_vec.shape == G0_jax.shape:
        v0_use = prev_vec
        v_ref = prev_vec
        if cfg_use.method.strip().lower() == "shift_invert" and prev_eig is not None:
            if shift_override is None:
                shift_override = prev_eig
                shift_selection_use = "shift"
    select_overlap = (
        use_cont
        and v_ref is not None
        and (cfg_use.continuation_selection.strip().lower() == "overlap")
    )
    krylov_kwargs = {
        "terms": terms,
        "v_ref": v_ref,
        "select_overlap": select_overlap,
        **{name: getattr(cfg_use, name) for name in _ETG_KRYLOV_FORWARD_KEYS},
        "shift": shift_override,
        "shift_selection": shift_selection_use,
    }
    eig, vec = dominant_eigenpair(v0_use, cache, params, **krylov_kwargs)
    if use_cont:
        eig_host = complex(np.asarray(eig))
        if np.isfinite(eig_host.real) and np.isfinite(eig_host.imag):
            prev_vec = vec
            prev_eig = eig_host
        else:
            prev_vec = None
            prev_eig = None
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    if cfg_use.omega_sign != 0:
        omega = float(np.sign(cfg_use.omega_sign)) * abs(omega)
    gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    return gamma, omega, prev_vec, prev_eig


def _etg_time_config_for_batch(context: _ETGTimeBatchContext) -> Any | None:
    if context.time_cfg is None:
        return None
    time_cfg_i = replace(
        context.time_cfg,
        dt=context.dt_i,
        t_max=context.dt_i * context.steps_i,
    )
    if context.sample_stride is not None:
        time_cfg_i = replace(time_cfg_i, sample_stride=context.sample_stride)
    return time_cfg_i


def _append_etg_streaming_time_results(
    context: _ETGTimeBatchContext,
    *,
    time_cfg_i: Any,
) -> None:
    t_total = float(time_cfg_i.t_max)
    tmin_i, tmax_i = _resolve_streaming_window(
        t_total,
        indexed_float_value(context.tmin, context.batch_start),
        indexed_float_value(context.tmax, context.batch_start),
        context.start_fraction,
        context.window_fraction,
        1.0,
    )
    _, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
        context.G0_jax,
        context.grid,
        context.geom,
        context.params,
        dt=context.dt_i,
        steps=context.steps_i,
        method=time_cfg_i.diffrax_solver,
        cache=context.cache,
        terms=context.terms,
        adaptive=time_cfg_i.diffrax_adaptive,
        rtol=time_cfg_i.diffrax_rtol,
        atol=time_cfg_i.diffrax_atol,
        max_steps=time_cfg_i.diffrax_max_steps,
        progress_bar=time_cfg_i.progress_bar,
        checkpoint=time_cfg_i.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal=context.fit_key,
        mode_ky_indices=np.arange(context.valid_count, dtype=int),
        mode_kx_index=0,
        mode_z_index=_midplane_index(context.grid),
        mode_method=context.mode_method,
        amp_floor=context.streaming_amp_floor,
        density_species_index=context.electron_index
        if context.fit_key == "density"
        else None,
        return_state=False,
    )
    gamma_arr = np.asarray(gamma_vals)
    omega_arr = np.asarray(omega_vals)
    for local_idx in range(context.valid_count):
        gamma_i, omega_i = _normalize_growth_rate(
            float(gamma_arr[local_idx]),
            float(omega_arr[local_idx]),
            context.params,
            context.diagnostic_norm,
        )
        context.gammas.append(gamma_i)
        context.omegas.append(omega_i)
        context.ky_out.append(float(context.ky_slice[local_idx]))


def _configured_etg_time_history(
    context: _ETGTimeBatchContext,
    *,
    time_cfg_i: Any,
) -> tuple[Any, Any | None, int]:
    save_field = (
        "phi+density"
        if context.fit_key == "auto"
        else ("density" if context.fit_key == "density" else "phi")
    )
    save_mode = None
    if context.fit_key != "auto" and context.mode_only and context.fit_key == "phi":
        save_mode = context.sel
    _, saved = integrate_linear_from_config(
        context.G0_jax,
        context.grid,
        context.geom,
        context.params,
        time_cfg_i,
        cache=context.cache,
        terms=context.terms,
        save_mode=save_mode,
        mode_method=context.mode_method,
        save_field=save_field,
        density_species_index=context.electron_index
        if context.need_density
        else None,
        show_progress=context.show_progress,
    )
    if context.fit_key == "auto":
        phi_t, density_t = saved
    else:
        phi_t = saved
        density_t = None
    return phi_t, density_t, int(time_cfg_i.sample_stride)


def _unconfigured_etg_time_history(
    context: _ETGTimeBatchContext,
) -> tuple[Any, Any | None, int]:
    stride = 1 if context.sample_stride is None else int(context.sample_stride)
    if context.need_density:
        diag = integrate_linear_diagnostics(
            context.G0_jax,
            context.grid,
            context.geom,
            context.params,
            dt=context.dt_i,
            steps=context.steps_i,
            method=context.method,
            cache=context.cache,
            terms=context.terms,
            sample_stride=stride,
            species_index=1,
            show_progress=context.show_progress,
        )
        return diag[1], diag[2] if len(diag) > 2 else None, stride
    _, phi_out_time = integrate_linear(
        context.G0_jax,
        context.grid,
        context.geom,
        context.params,
        dt=context.dt_i,
        steps=context.steps_i,
        method=context.method,
        cache=context.cache,
        terms=context.terms,
        sample_stride=stride,
        show_progress=context.show_progress,
    )
    return phi_out_time, None, stride


def _pack_etg_time_history_result(
    *,
    phi_t: Any,
    density_t: Any | None,
    dt_i: float,
    stride: int,
    fit_key: str,
) -> ETGTimeBatchResult:
    phi_t_np = np.asarray(phi_t)
    density_np = None if density_t is None else np.asarray(density_t)
    if fit_key == "density" and density_np is None:
        density_np = phi_t_np
    t = np.arange(phi_t_np.shape[0]) * dt_i * stride
    return ETGTimeBatchResult(
        handled=False,
        phi_t=phi_t_np,
        density_t=density_np,
        t=t,
        stride=stride,
    )


def run_etg_time_batch(
    *,
    G0_jax: jnp.ndarray,
    grid: Any,
    geom: Any,
    params: Any,
    cache: Any,
    terms: Any,
    time_cfg: Any,
    dt_i: float,
    steps_i: int,
    method: str,
    sample_stride: int | None,
    fit_key: str,
    need_density: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    mode_method: str,
    mode_only: bool,
    sel: Any,
    batch_start: int,
    valid_count: int,
    ky_slice: np.ndarray,
    tmin: float | None,
    tmax: float | None,
    start_fraction: float,
    window_fraction: float,
    electron_index: int,
    diagnostic_norm: str,
    show_progress: bool,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> ETGTimeBatchResult:
    """Integrate one ETG time-path batch and append streaming-fit results if used."""

    context = _etg_time_batch_context_from_locals(locals())
    time_cfg_i = _etg_time_config_for_batch(context)
    if time_cfg_i is not None and time_cfg_i.use_diffrax and streaming_fit:
        _append_etg_streaming_time_results(context, time_cfg_i=time_cfg_i)
        return ETGTimeBatchResult(handled=True)

    if time_cfg_i is not None:
        phi_t, density_t, stride = _configured_etg_time_history(
            context, time_cfg_i=time_cfg_i
        )
    else:
        phi_t, density_t, stride = _unconfigured_etg_time_history(context)
    return _pack_etg_time_history_result(
        phi_t=phi_t,
        density_t=density_t,
        dt_i=dt_i,
        stride=stride,
        fit_key=fit_key,
    )


def _etg_local_selection(local_idx: int, context: _ETGTimeFitContext) -> ModeSelection:
    return ModeSelection(
        ky_index=local_idx,
        kx_index=0,
        z_index=context.mode_z_index,
    )


def _auto_etg_time_fit(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> tuple[float, float]:
    if result.phi_t is None or result.t is None:
        raise ValueError("ETG time-batch result has no saved signal to fit")
    _signal, _name, gamma, omega = _select_fit_signal_auto(
        result.t,
        result.phi_t,
        result.density_t,
        _etg_local_selection(local_idx, context),
        mode_method=context.mode_method,
        tmin=indexed_float_value(context.fit_policy.tmin, context.batch_start + local_idx),
        tmax=indexed_float_value(context.fit_policy.tmax, context.batch_start + local_idx),
        window_fraction=context.fit_policy.window_fraction,
        min_points=context.fit_policy.min_points,
        start_fraction=context.fit_policy.start_fraction,
        growth_weight=context.fit_policy.growth_weight,
        require_positive=context.fit_policy.require_positive,
        min_amp_fraction=context.fit_policy.min_amp_fraction,
        max_amp_fraction=context.fit_policy.max_amp_fraction,
        window_method=context.fit_policy.window_method,
        max_fraction=context.fit_policy.max_fraction,
        end_fraction=context.fit_policy.end_fraction,
        num_windows=8,
        phase_weight=context.fit_policy.phase_weight,
        length_weight=context.fit_policy.length_weight,
        min_r2=context.fit_policy.min_r2,
        late_penalty=context.fit_policy.late_penalty,
        min_slope=context.fit_policy.min_slope,
        min_slope_frac=context.fit_policy.min_slope_frac,
        slope_var_weight=context.fit_policy.slope_var_weight,
    )
    return _normalize_growth_rate(
        gamma, omega, context.params, context.diagnostic_norm
    )


def _direct_etg_time_signal(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> np.ndarray:
    if result.phi_t is None:
        raise ValueError("ETG time-batch result has no saved signal to fit")
    if context.mode_only and context.fit_key == "phi" and result.phi_t.ndim <= 2:
        return _extract_mode_only_signal(result.phi_t, local_idx=local_idx)
    return _select_fit_signal(
        result.phi_t,
        result.density_t,
        _etg_local_selection(local_idx, context),
        fit_signal=context.fit_key,
        mode_method=context.mode_method,
    )


def _reference_window_etg_time_fit(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> tuple[float, float]:
    if result.phi_t is None or result.t is None:
        raise ValueError("ETG time-batch result has no saved signal to fit")
    gamma, omega, _gamma_t, _omega_t, _t_mid = instantaneous_growth_rate_from_phi(
        result.phi_t,
        result.t,
        _etg_local_selection(local_idx, context),
        navg_fraction=context.reference_navg_fraction,
        mode_method=context.mode_method,
    )
    return _normalize_growth_rate(
        gamma, omega, context.params, context.diagnostic_norm
    )


def _direct_etg_time_fit(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> tuple[float, float]:
    if context.reference_growth_window and context.fit_key == "phi":
        return _reference_window_etg_time_fit(
            result, local_idx=local_idx, context=context
        )
    signal = _direct_etg_time_signal(result, local_idx=local_idx, context=context)
    return context.fit_policy.fit_signal(
        signal,
        idx=context.batch_start + local_idx,
        dt=context.dt_i,
        stride=result.stride,
        params=context.params,
        diagnostic_norm=context.diagnostic_norm,
    )


def _resolve_etg_time_fit(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> tuple[float, float]:
    if context.fit_key == "auto":
        return _auto_etg_time_fit(result, local_idx=local_idx, context=context)
    return _direct_etg_time_fit(result, local_idx=local_idx, context=context)


def _fallback_etg_krylov_fit(
    *,
    ky_val: float,
    context: _ETGTimeFitContext,
) -> tuple[float, float]:
    res = run_etg_linear(
        ky_target=float(ky_val),
        cfg=context.cfg,
        Nl=context.Nl,
        Nm=context.Nm,
        dt=context.dt_i,
        steps=context.steps_i,
        method=context.method,
        params=context.params,
        solver="krylov",
        krylov_cfg=context.krylov_cfg,
        diagnostic_norm=context.diagnostic_norm,
        fit_signal="phi",
        show_progress=context.show_progress,
    )
    return float(res.gamma), float(res.omega)


def _append_resolved_etg_time_fit(
    result: ETGTimeBatchResult,
    *,
    local_idx: int,
    context: _ETGTimeFitContext,
) -> None:
    ky_val = float(context.ky_slice[local_idx])
    gamma, omega = _resolve_etg_time_fit(
        result, local_idx=local_idx, context=context
    )
    if context.auto_solver and not _valid_etg_growth(
        gamma, omega, require_positive=context.require_positive
    ):
        gamma, omega = _fallback_etg_krylov_fit(ky_val=ky_val, context=context)
    context.gammas.append(float(gamma))
    context.omegas.append(float(omega))
    context.ky_out.append(ky_val)


def append_etg_time_fit_results(
    *,
    result: ETGTimeBatchResult,
    ky_slice: np.ndarray,
    valid_count: int,
    batch_start: int,
    fit_key: str,
    fit_policy: Any,
    params: Any,
    diagnostic_norm: str,
    mode_method: str,
    mode_only: bool,
    mode_z_index: int,
    reference_growth_window: bool,
    reference_navg_fraction: float,
    auto_solver: bool,
    require_positive: bool,
    cfg: Any,
    Nl: int,
    Nm: int,
    dt_i: float,
    steps_i: int,
    method: str,
    krylov_cfg: Any,
    show_progress: bool,
    gammas: list[float],
    omegas: list[float],
    ky_out: list[float],
) -> None:
    """Fit and append ETG growth/frequency values from a saved time batch."""

    if result.phi_t is None or result.t is None:
        raise ValueError("ETG time-batch result has no saved signal to fit")
    context = _ETGTimeFitContext(
        ky_slice=ky_slice,
        valid_count=valid_count,
        batch_start=batch_start,
        fit_key=fit_key,
        fit_policy=fit_policy,
        params=params,
        diagnostic_norm=diagnostic_norm,
        mode_method=mode_method,
        mode_only=mode_only,
        mode_z_index=mode_z_index,
        reference_growth_window=reference_growth_window,
        reference_navg_fraction=reference_navg_fraction,
        auto_solver=auto_solver,
        require_positive=require_positive,
        cfg=cfg,
        Nl=Nl,
        Nm=Nm,
        dt_i=dt_i,
        steps_i=steps_i,
        method=method,
        krylov_cfg=krylov_cfg,
        show_progress=show_progress,
        gammas=gammas,
        omegas=omegas,
        ky_out=ky_out,
    )
    for local_idx in range(valid_count):
        _append_resolved_etg_time_fit(result, local_idx=local_idx, context=context)


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


def _run_etg_scan_krylov_batch(
    setup: _ETGScanSetup,
    batch: _ETGScanBatch,
    *,
    options: _ETGScanRuntimeOptions,
    acc: _ETGScanAccumulator,
) -> tuple[jnp.ndarray | None, complex | None]:
    """Run one Krylov ETG scan batch and update continuation state."""

    gamma, omega, prev_vec, prev_eig = run_etg_krylov_batch(
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


def _append_etg_scan_time_fit_results(
    setup: _ETGScanSetup,
    batch: _ETGScanBatch,
    *,
    options: _ETGScanRuntimeOptions,
    acc: _ETGScanAccumulator,
    time_result: Any,
) -> None:
    """Append fitted ETG time-batch results using the shared path helper."""
    append_etg_time_fit_results(
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


def _run_etg_scan_time_batch(
    setup: _ETGScanSetup,
    batch: _ETGScanBatch,
    *,
    options: _ETGScanRuntimeOptions,
    acc: _ETGScanAccumulator,
) -> tuple[jnp.ndarray | None, complex | None]:
    """Run one time-integrated ETG scan batch and append fit results."""

    time_result = run_etg_time_batch(
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
    if not time_result.handled:
        _append_etg_scan_time_fit_results(
            setup,
            batch,
            options=options,
            acc=acc,
            time_result=time_result,
        )
    return acc.prev_vec, acc.prev_eig


def _run_etg_scan_batch(
    setup: _ETGScanSetup,
    batch: _ETGScanBatch,
    *,
    options: _ETGScanRuntimeOptions,
    acc: _ETGScanAccumulator,
) -> tuple[jnp.ndarray | None, complex | None]:
    """Run one ETG scan batch and append growth/frequency outputs."""

    if setup.solver_key == "krylov":
        return _run_etg_scan_krylov_batch(setup, batch, options=options, acc=acc)
    return _run_etg_scan_time_batch(setup, batch, options=options, acc=acc)


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
    return _prepare_etg_scan_setup(request)


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

@dataclass(frozen=True)
class _KineticLinearSetup:
    cfg: KineticElectronBaseCase
    grid_full: Any
    geom: Any
    params: LinearParams
    terms: LinearTerms
    init_cfg: Any
    diagnostic_norm: str
    reference_aligned: bool


@dataclass(frozen=True)
class _KineticLinearState:
    grid: Any
    selection: ModeSelection
    state: jnp.ndarray


@dataclass(frozen=True)
class _KineticHistory:
    t: np.ndarray
    phi_t: np.ndarray
    density_t: np.ndarray | None


@dataclass(frozen=True)
class _KineticFitOptions:
    fit_signal: str
    mode_method: str
    auto_window: bool
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float


@dataclass(frozen=True)
class _KineticTimePathOptions:
    time_cfg: TimeConfig | None
    dt: float
    steps: int
    method: str
    sample_stride: int | None
    density_species_index: int
    show_progress: bool
    n_laguerre: int
    n_hermite: int
    fit: _KineticFitOptions


def _resolve_kinetic_linear_setup(
    *,
    cfg: KineticElectronBaseCase | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    diagnostic_norm: str,
    reference_aligned: bool | None,
    Nm: int,
) -> _KineticLinearSetup:
    """Resolve kinetic benchmark setup shared by Krylov and time paths."""

    cfg_use = cfg or KineticElectronBaseCase()
    grid_full = build_spectral_grid(cfg_use.grid)
    geom = SAlphaGeometry.from_config(cfg_use.geometry)
    reference_aligned_use = bool(
        True if reference_aligned is None else reference_aligned
    )
    diagnostic_norm_use = diagnostic_norm
    if reference_aligned_use and diagnostic_norm_use == "none":
        diagnostic_norm_use = "rho_star"
    init_cfg_use = _kinetic_reference_init_cfg(
        cfg_use.init, reference_aligned=reference_aligned_use
    )
    damp_ends_amp, damp_ends_widthfrac = _linked_boundary_end_damping(
        reference_aligned_use
    )
    params_use = params
    if params_use is None:
        params_use = _two_species_params(
            cfg_use.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KINETIC_OMEGA_D_SCALE,
            omega_star_scale=KINETIC_OMEGA_STAR_SCALE,
            rho_star=KINETIC_RHO_STAR,
            damp_ends_amp=damp_ends_amp,
            damp_ends_widthfrac=damp_ends_widthfrac,
            nhermite=Nm,
        )
        if reference_aligned_use:
            params_use = _apply_reference_hypercollisions(params_use, nhermite=Nm)
    terms_use = terms if terms is not None else LinearTerms(bpar=0.0)
    return _KineticLinearSetup(
        cfg=cfg_use,
        grid_full=grid_full,
        geom=geom,
        params=params_use,
        terms=terms_use,
        init_cfg=init_cfg_use,
        diagnostic_norm=diagnostic_norm_use,
        reference_aligned=reference_aligned_use,
    )


def _validate_kinetic_species_indices(
    *, init_species_index: int, density_species_index: int, nspecies: int = 2
) -> None:
    """Validate the kinetic two-species index contract."""

    if init_species_index < 0 or init_species_index >= nspecies:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= nspecies:
        raise ValueError("density_species_index out of range for kinetic species")


def _build_kinetic_linear_state(
    setup: _KineticLinearSetup,
    *,
    ky_target: float,
    Nl: int,
    Nm: int,
    init_species_index: int,
    density_species_index: int,
) -> _KineticLinearState:
    """Select the ky grid and build the kinetic initial perturbation."""

    ky_index = select_ky_index(np.asarray(setup.grid_full.ky), ky_target)
    grid = select_ky_grid(setup.grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    nspecies = 2
    _validate_kinetic_species_indices(
        init_species_index=init_species_index,
        density_species_index=density_species_index,
        nspecies=nspecies,
    )
    G0 = np.zeros(
        (nspecies, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=np.complex64,
    )
    G0_single = _build_initial_condition(
        grid,
        setup.geom,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=setup.init_cfg,
    )
    G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)
    return _KineticLinearState(grid=grid, selection=sel, state=jnp.asarray(G0))


def _prepare_kinetic_linear_setup_and_state(
    *,
    cfg: KineticElectronBaseCase | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    diagnostic_norm: str,
    reference_aligned: bool | None,
    ky_target: float,
    n_laguerre: int,
    n_hermite: int,
    init_species_index: int,
    density_species_index: int,
) -> tuple[_KineticLinearSetup, _KineticLinearState]:
    """Resolve the kinetic benchmark setup and selected-ky initial state."""

    setup = _resolve_kinetic_linear_setup(
        cfg=cfg,
        params=params,
        terms=terms,
        diagnostic_norm=diagnostic_norm,
        reference_aligned=reference_aligned,
        Nm=n_hermite,
    )
    state = _build_kinetic_linear_state(
        setup,
        ky_target=ky_target,
        Nl=n_laguerre,
        Nm=n_hermite,
        init_species_index=init_species_index,
        density_species_index=density_species_index,
    )
    return setup, state


def _kinetic_krylov_config(
    setup: _KineticLinearSetup,
    krylov_cfg: KrylovConfig | None,
) -> KrylovConfig:
    """Return the kinetic Krylov policy, including reference-aligned defaults."""

    if krylov_cfg is not None:
        return krylov_cfg
    if setup.reference_aligned:
        return KINETIC_KRYLOV_REFERENCE_ALIGNED
    return KINETIC_KRYLOV_DEFAULT


def _run_kinetic_krylov_path(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    *,
    Nl: int,
    Nm: int,
    krylov_cfg: KrylovConfig | None,
) -> LinearRunResult:
    """Solve one kinetic benchmark point with the Krylov eigenpath."""

    cfg_use = _kinetic_krylov_config(setup, krylov_cfg)
    cache = build_linear_cache(state.grid, setup.geom, setup.params, Nl, Nm)
    eig, vec = dominant_eigenpair(
        state.state,
        cache,
        setup.params,
        terms=setup.terms,
        krylov_dim=cfg_use.krylov_dim,
        restarts=cfg_use.restarts,
        omega_min_factor=cfg_use.omega_min_factor,
        omega_target_factor=cfg_use.omega_target_factor,
        omega_cap_factor=cfg_use.omega_cap_factor,
        omega_sign=cfg_use.omega_sign,
        method=cfg_use.method,
        power_iters=cfg_use.power_iters,
        power_dt=cfg_use.power_dt,
        shift=cfg_use.shift,
        shift_source=cfg_use.shift_source,
        shift_tol=cfg_use.shift_tol,
        shift_maxiter=cfg_use.shift_maxiter,
        shift_restart=cfg_use.shift_restart,
        shift_solve_method=cfg_use.shift_solve_method,
        shift_preconditioner=cfg_use.shift_preconditioner,
        shift_selection=cfg_use.shift_selection,
        mode_family=cfg_use.mode_family,
        fallback_method=cfg_use.fallback_method,
        fallback_real_floor=cfg_use.fallback_real_floor,
    )
    term_cfg = linear_terms_to_term_config(setup.terms)
    phi = compute_fields_cached(vec, cache, setup.params, terms=term_cfg).phi
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    gamma, omega = _normalize_growth_rate(
        gamma, omega, setup.params, setup.diagnostic_norm
    )
    return _pack_kinetic_result(
        state,
        t=np.array([0.0]),
        phi_t=np.asarray(phi)[None, ...],
        gamma=gamma,
        omega=omega,
    )


def _resolve_time_config(
    time_cfg: TimeConfig | None,
    *,
    sample_stride: int | None,
) -> TimeConfig | None:
    """Apply user sample-stride override without changing time-config semantics."""

    if time_cfg is None or sample_stride is None:
        return time_cfg
    return replace(time_cfg, sample_stride=sample_stride)


def _integrate_configured_kinetic_history(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    *,
    time_cfg: TimeConfig,
    method_key: str,
    fit_signal: str,
    density_species_index: int,
    Nl: int,
    Nm: int,
) -> tuple[Any, Any | None, float, int]:
    """Integrate kinetic time history with an explicit runtime TimeConfig."""

    dt = float(time_cfg.dt)
    steps = int(round(time_cfg.t_max / time_cfg.dt))
    cache = build_linear_cache(state.grid, setup.geom, setup.params, Nl, Nm)
    if time_cfg.use_diffrax and not (
        method_key.startswith("imex") or method_key.startswith("implicit")
    ):
        save_field = "density" if fit_signal == "density" else "phi"
        _, phi_t = integrate_linear_from_config(
            state.state,
            state.grid,
            setup.geom,
            setup.params,
            time_cfg,
            cache=cache,
            terms=setup.terms,
            save_field=save_field,
            density_species_index=density_species_index
            if fit_signal == "density"
            else None,
        )
        density_t = phi_t if fit_signal == "density" else None
        return phi_t, density_t, dt, time_cfg.sample_stride

    if fit_signal == "density":
        diag = integrate_linear_diagnostics(
            state.state,
            state.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=time_cfg.method,
            cache=cache,
            terms=setup.terms,
            sample_stride=time_cfg.sample_stride,
            species_index=density_species_index,
        )
        phi_t = diag[1]
        density_t = diag[2] if len(diag) > 2 else None
        return phi_t, density_t, dt, time_cfg.sample_stride

    _, phi_t = integrate_linear_from_config(
        state.state,
        state.grid,
        setup.geom,
        setup.params,
        time_cfg,
        cache=cache,
        terms=setup.terms,
        density_species_index=density_species_index if fit_signal == "density" else None,
    )
    return phi_t, None, dt, time_cfg.sample_stride


def _integrate_unconfigured_kinetic_history(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    *,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    fit_signal: str,
    density_species_index: int,
    show_progress: bool,
) -> tuple[Any, Any | None, float, int]:
    """Integrate kinetic time history without a runtime TimeConfig."""

    stride = 1 if sample_stride is None else int(sample_stride)
    if fit_signal == "density":
        diag = integrate_linear_diagnostics(
            state.state,
            state.grid,
            setup.geom,
            setup.params,
            dt=dt,
            steps=steps,
            method=method,
            terms=setup.terms,
            sample_stride=stride,
            species_index=density_species_index,
        )
        phi_t = diag[1]
        density_t = diag[2] if len(diag) > 2 else None
        return phi_t, density_t, float(dt), stride

    _, phi_t = integrate_linear(
        state.state,
        state.grid,
        setup.geom,
        setup.params,
        dt=dt,
        steps=steps,
        method=method,
        terms=setup.terms,
        sample_stride=stride,
        show_progress=show_progress,
    )
    return phi_t, None, float(dt), stride


def _integrate_kinetic_history(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    *,
    time_cfg: TimeConfig | None,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    fit_signal: str,
    density_species_index: int,
    show_progress: bool,
    Nl: int,
    Nm: int,
) -> _KineticHistory:
    """Integrate a kinetic time history and preserve saved-observable semantics."""

    time_cfg_use = _resolve_time_config(time_cfg, sample_stride=sample_stride)
    if time_cfg_use is not None:
        phi_t, density_t, dt_eff, stride = _integrate_configured_kinetic_history(
            setup,
            state,
            time_cfg=time_cfg_use,
            method_key=method.lower(),
            fit_signal=fit_signal,
            density_species_index=density_species_index,
            Nl=Nl,
            Nm=Nm,
        )
    else:
        phi_t, density_t, dt_eff, stride = _integrate_unconfigured_kinetic_history(
            setup,
            state,
            dt=dt,
            steps=steps,
            method=method,
            sample_stride=sample_stride,
            fit_signal=fit_signal,
            density_species_index=density_species_index,
            show_progress=show_progress,
        )
    phi_t_np = np.asarray(phi_t)
    t = np.arange(phi_t_np.shape[0]) * dt_eff * stride
    density_np = None if density_t is None else np.asarray(density_t)
    return _KineticHistory(t=t, phi_t=phi_t_np, density_t=density_np)


def _fit_kinetic_history(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    history: _KineticHistory,
    *,
    options: _KineticFitOptions,
) -> tuple[float, float]:
    """Fit growth/frequency from a saved kinetic time history."""

    signal = _select_fit_signal(
        history.phi_t,
        history.density_t,
        state.selection,
        fit_signal=options.fit_signal,
        mode_method=options.mode_method,
    )
    use_auto = options.auto_window and options.tmin is None and options.tmax is None
    if not use_auto and not scan_window_valid(history.t, options.tmin, options.tmax):
        use_auto = True
    auto_fit_kwargs: dict[str, Any] = {
        "window_fraction": options.window_fraction,
        "min_points": options.min_points,
        "start_fraction": options.start_fraction,
        "growth_weight": options.growth_weight,
        "require_positive": options.require_positive,
        "min_amp_fraction": options.min_amp_fraction,
    }
    if use_auto:
        gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
            history.t, signal, **auto_fit_kwargs
        )
    else:
        try:
            gamma, omega = fit_growth_rate(
                history.t, signal, tmin=options.tmin, tmax=options.tmax
            )
        except ValueError:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                history.t, signal, **auto_fit_kwargs
            )
    return _normalize_growth_rate(
        gamma, omega, setup.params, setup.diagnostic_norm
    )


def _run_kinetic_time_path(
    setup: _KineticLinearSetup,
    state: _KineticLinearState,
    *,
    options: _KineticTimePathOptions,
) -> LinearRunResult:
    """Run and fit the saved-time kinetic benchmark path."""

    history = _integrate_kinetic_history(
        setup,
        state,
        time_cfg=options.time_cfg,
        dt=options.dt,
        steps=options.steps,
        method=options.method,
        sample_stride=options.sample_stride,
        fit_signal=options.fit.fit_signal,
        density_species_index=options.density_species_index,
        show_progress=options.show_progress,
        Nl=options.n_laguerre,
        Nm=options.n_hermite,
    )
    gamma, omega = _fit_kinetic_history(
        setup,
        state,
        history,
        options=options.fit,
    )
    return _pack_kinetic_result(
        state, t=history.t, phi_t=history.phi_t, gamma=gamma, omega=omega
    )


def _pack_kinetic_result(
    state: _KineticLinearState,
    *,
    t: np.ndarray,
    phi_t: np.ndarray,
    gamma: float,
    omega: float,
) -> LinearRunResult:
    """Pack the public kinetic linear benchmark result."""

    return LinearRunResult(
        t=t,
        phi_t=phi_t,
        gamma=gamma,
        omega=omega,
        ky=float(state.grid.ky[state.selection.ky_index]),
        selection=state.selection,
    )


def _kinetic_time_path_options(
    *,
    time_cfg: TimeConfig | None,
    dt: float,
    steps: int,
    method: str,
    sample_stride: int | None,
    density_species_index: int,
    show_progress: bool,
    n_laguerre: int,
    n_hermite: int,
    fit_signal: str,
    mode_method: str,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
) -> _KineticTimePathOptions:
    """Pack public time-path keyword controls into the internal request object."""

    return _KineticTimePathOptions(
        time_cfg,
        dt,
        steps,
        method,
        sample_stride,
        density_species_index,
        show_progress,
        n_laguerre,
        n_hermite,
        _KineticFitOptions(
            fit_signal,
            mode_method,
            auto_window,
            tmin,
            tmax,
            window_fraction,
            min_points,
            start_fraction,
            growth_weight,
            require_positive,
            min_amp_fraction,
        ),
    )


def run_kinetic_linear(
    ky_target: float = 0.3,
    Nl: int = 6,
    Nm: int = 12,
    dt: float = 0.01,
    steps: int = 800,
    method: str = "rk4",
    params: LinearParams | None = None,
    cfg: KineticElectronBaseCase | None = None,
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
    terms: LinearTerms | None = None,
    sample_stride: int | None = None,
    fit_signal: str = "density",
    init_species_index: int = 1,
    density_species_index: int = 1,
    diagnostic_norm: str = "none",
    reference_aligned: bool | None = True,
    show_progress: bool = False,
) -> LinearRunResult:
    """Run a kinetic-electron ITG/TEM benchmark and extract growth rate."""

    setup, state = _prepare_kinetic_linear_setup_and_state(
        cfg=cfg,
        params=params,
        terms=terms,
        diagnostic_norm=diagnostic_norm,
        reference_aligned=reference_aligned,
        ky_target=ky_target,
        n_laguerre=Nl,
        n_hermite=Nm,
        init_species_index=init_species_index,
        density_species_index=density_species_index,
    )
    if solver.lower() == "krylov":
        return _run_kinetic_krylov_path(
            setup,
            state,
            Nl=Nl,
            Nm=Nm,
            krylov_cfg=krylov_cfg,
        )
    return _run_kinetic_time_path(
        setup,
        state,
        options=_kinetic_time_path_options(
            time_cfg=time_cfg,
            dt=dt,
            steps=steps,
            method=method,
            sample_stride=sample_stride,
            density_species_index=density_species_index,
            show_progress=show_progress,
            n_laguerre=Nl,
            n_hermite=Nm,
            fit_signal=fit_signal,
            mode_method=mode_method,
            auto_window=auto_window,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
        ),
    )

@dataclass(frozen=True)
class _KineticScanSetup:
    grid_full: Any
    geom: Any
    params: LinearParams
    terms: LinearTerms
    init_cfg: Any
    diagnostic_norm: str
    reference_aligned: bool


@dataclass(frozen=True)
class _KineticScanBatch:
    batch_start: int
    ky_slice: np.ndarray
    valid_count: int
    grid: Any
    selection: ModeSelection | ModeSelectionBatch
    dt: float
    steps: int
    state: Any
    cache: Any


@dataclass(frozen=True)
class _KineticScanRunOptions:
    ky_values: np.ndarray
    time_cfg: TimeConfig | None
    solver_key: str
    krylov_cfg: KrylovConfig | None
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    sample_stride: int | None
    mode_method: str
    mode_only: bool
    fit_key: str
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    init_species_index: int
    density_species_index: int
    use_batch: bool
    show_progress: bool


@dataclass(frozen=True)
class _KineticScanFitOptions:
    tmin: float | None
    tmax: float | None
    start_fraction: float
    window_fraction: float
    fit_policy: ScanFitWindowPolicy


@dataclass
class _KineticScanOutput:
    gammas: list[float]
    omegas: list[float]
    ky: list[float]

    @classmethod
    def empty(cls) -> "_KineticScanOutput":
        return cls(gammas=[], omegas=[], ky=[])


@dataclass(frozen=True)
class _KineticScanControls:
    setup: _KineticScanSetup
    run_options: _KineticScanRunOptions
    fit_options: _KineticScanFitOptions


@dataclass(frozen=True)
class _KineticScanControlRequest:
    """Raw public scan inputs before setup, run, and fit policies are resolved."""

    ky_values: np.ndarray
    Nm: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    params: LinearParams | None
    cfg: KineticElectronBaseCase | None
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
    mode_only: bool
    terms: LinearTerms | None
    sample_stride: int | None
    fit_signal: str
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    init_species_index: int
    density_species_index: int
    diagnostic_norm: str
    reference_aligned: bool | None
    show_progress: bool


def _kinetic_scan_control_request_from_locals(
    values: dict[str, Any],
) -> _KineticScanControlRequest:
    """Build a request from ``run_kinetic_scan`` locals without forwarding Nl."""

    names = {field.name for field in fields(_KineticScanControlRequest)}
    return _KineticScanControlRequest(**{name: values[name] for name in names})


def _resolve_kinetic_scan_setup(
    *,
    cfg: KineticElectronBaseCase | None,
    params: LinearParams | None,
    terms: LinearTerms | None,
    diagnostic_norm: str,
    reference_aligned: bool | None,
    Nm: int,
) -> _KineticScanSetup:
    cfg_use = cfg or KineticElectronBaseCase()
    grid_full = build_spectral_grid(cfg_use.grid)
    geom = SAlphaGeometry.from_config(cfg_use.geometry)
    reference_aligned_use = bool(
        True if reference_aligned is None else reference_aligned
    )
    diagnostic_norm_use = diagnostic_norm
    if reference_aligned_use and diagnostic_norm_use == "none":
        diagnostic_norm_use = "rho_star"
    init_cfg = _kinetic_reference_init_cfg(
        cfg_use.init, reference_aligned=reference_aligned_use
    )
    damp_ends_amp, damp_ends_widthfrac = _linked_boundary_end_damping(
        reference_aligned_use
    )
    params_use = params
    if params_use is None:
        params_use = _two_species_params(
            cfg_use.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KINETIC_OMEGA_D_SCALE,
            omega_star_scale=KINETIC_OMEGA_STAR_SCALE,
            rho_star=KINETIC_RHO_STAR,
            damp_ends_amp=damp_ends_amp,
            damp_ends_widthfrac=damp_ends_widthfrac,
            nhermite=Nm,
        )
        if reference_aligned_use:
            params_use = _apply_reference_hypercollisions(params_use, nhermite=Nm)
    terms_use = terms if terms is not None else LinearTerms(bpar=0.0)
    return _KineticScanSetup(
        grid_full=grid_full,
        geom=geom,
        params=params_use,
        terms=terms_use,
        init_cfg=init_cfg,
        diagnostic_norm=diagnostic_norm_use,
        reference_aligned=reference_aligned_use,
    )


def _iter_kinetic_scan_batches(options: _KineticScanRunOptions):
    if options.use_batch:
        return _iter_ky_batches(
            options.ky_values,
            ky_batch=options.ky_batch,
            fixed_batch_shape=options.fixed_batch_shape,
        )
    return _iter_ky_batches(options.ky_values, ky_batch=1, fixed_batch_shape=False)


def _prepare_kinetic_scan_batch(
    setup: _KineticScanSetup,
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    use_batch: bool,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    Nl: int,
    Nm: int,
    init_species_index: int,
) -> _KineticScanBatch:
    if use_batch:
        ky_indices = [
            select_ky_index(np.asarray(setup.grid_full.ky), float(ky))
            for ky in ky_slice
        ]
        grid = select_ky_grid(setup.grid_full, ky_indices)
        sel_indices = np.arange(len(ky_indices), dtype=int)
        selection: ModeSelection | ModeSelectionBatch = ModeSelectionBatch(
            sel_indices, 0, _midplane_index(grid)
        )
        dt_i = float(dt)
        steps_i = int(steps)
    else:
        ky_indices = [
            select_ky_index(np.asarray(setup.grid_full.ky), float(ky_slice[0]))
        ]
        grid = select_ky_grid(setup.grid_full, ky_indices[0])
        selection = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
        dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)

    nspecies = 2
    G0 = np.zeros(
        (nspecies, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=np.complex64,
    )
    G0_single = _build_initial_condition(
        grid,
        setup.geom,
        ky_index=np.arange(len(ky_indices), dtype=int),
        kx_index=0,
        Nl=Nl,
        Nm=Nm,
        init_cfg=setup.init_cfg,
    )
    G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)
    state = jnp.asarray(G0)
    cache = build_linear_cache(grid, setup.geom, setup.params, Nl, Nm)
    return _KineticScanBatch(
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        grid=grid,
        selection=selection,
        dt=dt_i,
        steps=steps_i,
        state=state,
        cache=cache,
    )


def _kinetic_scan_time_config(
    time_cfg: TimeConfig | None,
    *,
    dt: float,
    steps: int,
    sample_stride: int | None,
) -> TimeConfig | None:
    if time_cfg is None:
        return None
    time_cfg_i = replace(time_cfg, dt=dt, t_max=dt * steps)
    if sample_stride is not None:
        time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)
    return time_cfg_i


def _run_kinetic_scan_krylov(
    batch: _KineticScanBatch,
    setup: _KineticScanSetup,
    krylov_cfg: KrylovConfig | None,
) -> tuple[float, float]:
    cfg_use = krylov_cfg or (
        KINETIC_KRYLOV_REFERENCE_ALIGNED
        if setup.reference_aligned
        else KINETIC_KRYLOV_DEFAULT
    )
    eig, _vec = dominant_eigenpair(
        batch.state,
        batch.cache,
        setup.params,
        terms=setup.terms,
        krylov_dim=cfg_use.krylov_dim,
        restarts=cfg_use.restarts,
        omega_min_factor=cfg_use.omega_min_factor,
        omega_target_factor=cfg_use.omega_target_factor,
        omega_cap_factor=cfg_use.omega_cap_factor,
        omega_sign=cfg_use.omega_sign,
        method=cfg_use.method,
        power_iters=cfg_use.power_iters,
        power_dt=cfg_use.power_dt,
        shift=cfg_use.shift,
        shift_source=cfg_use.shift_source,
        shift_tol=cfg_use.shift_tol,
        shift_maxiter=cfg_use.shift_maxiter,
        shift_restart=cfg_use.shift_restart,
        shift_solve_method=cfg_use.shift_solve_method,
        shift_preconditioner=cfg_use.shift_preconditioner,
        shift_selection=cfg_use.shift_selection,
        mode_family=cfg_use.mode_family,
        fallback_method=cfg_use.fallback_method,
        fallback_real_floor=cfg_use.fallback_real_floor,
    )
    gamma = float(np.real(eig))
    omega = float(-np.imag(eig))
    return _normalize_growth_rate(
        gamma, omega, setup.params, setup.diagnostic_norm
    )


def _append_kinetic_streaming_results(
    batch: _KineticScanBatch,
    setup: _KineticScanSetup,
    *,
    time_cfg: TimeConfig,
    fit_key: str,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    start_fraction: float,
    window_fraction: float,
    streaming_amp_floor: float,
    density_species_index: int,
    output: _KineticScanOutput,
) -> None:
    t_total = float(time_cfg.t_max)
    tmin_i, tmax_i = _resolve_streaming_window(
        t_total,
        indexed_float_value(tmin, batch.batch_start),
        indexed_float_value(tmax, batch.batch_start),
        start_fraction,
        window_fraction,
        1.0,
    )
    _, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
        batch.state,
        batch.grid,
        setup.geom,
        setup.params,
        dt=batch.dt,
        steps=batch.steps,
        method=time_cfg.diffrax_solver,
        cache=batch.cache,
        terms=setup.terms,
        adaptive=time_cfg.diffrax_adaptive,
        rtol=time_cfg.diffrax_rtol,
        atol=time_cfg.diffrax_atol,
        max_steps=time_cfg.diffrax_max_steps,
        progress_bar=time_cfg.progress_bar,
        checkpoint=time_cfg.checkpoint,
        tmin=tmin_i,
        tmax=tmax_i,
        fit_signal=fit_key,
        mode_ky_indices=np.arange(batch.valid_count, dtype=int),
        mode_kx_index=0,
        mode_z_index=_midplane_index(batch.grid),
        mode_method=mode_method,
        amp_floor=streaming_amp_floor,
        density_species_index=density_species_index if fit_key == "density" else None,
        return_state=False,
    )
    gamma_arr = np.asarray(gamma_vals)
    omega_arr = np.asarray(omega_vals)
    for local_idx in range(batch.valid_count):
        gamma_i, omega_i = _normalize_growth_rate(
            float(gamma_arr[local_idx]),
            float(omega_arr[local_idx]),
            setup.params,
            setup.diagnostic_norm,
        )
        output.gammas.append(gamma_i)
        output.omegas.append(omega_i)
        output.ky.append(float(batch.ky_slice[local_idx]))


def _integrate_kinetic_scan_history(
    batch: _KineticScanBatch,
    setup: _KineticScanSetup,
    *,
    time_cfg: TimeConfig | None,
    method: str,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    sample_stride: int | None,
    density_species_index: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    if time_cfg is not None:
        save_mode_method = mode_method if mode_method in {"z_index", "max"} else "z_index"
        _, phi_t = integrate_linear_from_config(
            batch.state,
            batch.grid,
            setup.geom,
            setup.params,
            time_cfg,
            cache=batch.cache,
            terms=setup.terms,
            save_mode=batch.selection if (mode_only and fit_key == "phi") else None,
            mode_method=save_mode_method,
            save_field="density" if fit_key == "density" else "phi",
            density_species_index=density_species_index
            if fit_key == "density"
            else None,
        )
        return np.asarray(phi_t), None, time_cfg.sample_stride

    stride = 1 if sample_stride is None else int(sample_stride)
    if fit_key == "density":
        _diag = integrate_linear_diagnostics(
            batch.state,
            batch.grid,
            setup.geom,
            setup.params,
            dt=batch.dt,
            steps=batch.steps,
            method=method,
            cache=batch.cache,
            terms=setup.terms,
            sample_stride=stride,
            species_index=density_species_index,
        )
        phi_t = _diag[1]
        density_t = _diag[2] if len(_diag) > 2 else None
    else:
        _, phi_t = integrate_linear(
            batch.state,
            batch.grid,
            setup.geom,
            setup.params,
            dt=batch.dt,
            steps=batch.steps,
            method=method,
            cache=batch.cache,
            terms=setup.terms,
            sample_stride=stride,
            show_progress=show_progress,
        )
        density_t = None
    return (
        np.asarray(phi_t),
        None if density_t is None else np.asarray(density_t),
        stride,
    )


def _append_kinetic_sampled_results(
    batch: _KineticScanBatch,
    setup: _KineticScanSetup,
    *,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    fit_key: str,
    mode_only: bool,
    mode_method: str,
    fit_policy: ScanFitWindowPolicy,
    density_species_index: int,
    stride: int,
    output: _KineticScanOutput,
) -> None:
    density_np = phi_t if fit_key == "density" and density_t is None else density_t
    for local_idx in range(batch.valid_count):
        if mode_only and fit_key == "phi" and phi_t.ndim <= 2:
            signal = _extract_mode_only_signal(phi_t, local_idx=local_idx)
        elif (
            mode_only
            and fit_key == "density"
            and density_np is not None
            and density_np.ndim <= 3
        ):
            signal = _extract_mode_only_signal(
                density_np,
                local_idx=local_idx,
                species_index=density_species_index,
            )
        else:
            sel_local = ModeSelection(
                ky_index=local_idx,
                kx_index=0,
                z_index=_midplane_index(batch.grid),
            )
            signal = _select_fit_signal(
                phi_t,
                density_np,
                sel_local,
                fit_signal=fit_key,
                mode_method=mode_method,
            )
        gamma, omega = fit_policy.fit_signal(
            signal,
            idx=batch.batch_start + local_idx,
            dt=batch.dt,
            stride=stride,
            params=setup.params,
            diagnostic_norm=setup.diagnostic_norm,
        )
        output.gammas.append(gamma)
        output.omegas.append(omega)
        output.ky.append(float(batch.ky_slice[local_idx]))


def _append_kinetic_krylov_result(
    batch: _KineticScanBatch,
    setup: _KineticScanSetup,
    *,
    krylov_cfg: KrylovConfig | None,
    output: _KineticScanOutput,
) -> None:
    gamma, omega = _run_kinetic_scan_krylov(batch, setup, krylov_cfg)
    output.gammas.append(gamma)
    output.omegas.append(omega)
    output.ky.append(float(batch.ky_slice[0]))


def _append_kinetic_time_batch_results(
    *,
    batch_start: int,
    ky_slice: np.ndarray,
    valid_count: int,
    setup: _KineticScanSetup,
    run_options: _KineticScanRunOptions,
    fit_options: _KineticScanFitOptions,
    Nl: int,
    Nm: int,
    output: _KineticScanOutput,
) -> None:
    batch = _prepare_kinetic_scan_batch(
        setup,
        batch_start=batch_start,
        ky_slice=ky_slice,
        valid_count=valid_count,
        use_batch=run_options.use_batch,
        dt=run_options.dt,
        steps=run_options.steps,
        Nl=Nl,
        Nm=Nm,
        init_species_index=run_options.init_species_index,
    )
    if run_options.solver_key == "krylov":
        _append_kinetic_krylov_result(
            batch, setup, krylov_cfg=run_options.krylov_cfg, output=output
        )
        return

    time_cfg_i = _kinetic_scan_time_config(
        run_options.time_cfg,
        dt=batch.dt,
        steps=batch.steps,
        sample_stride=run_options.sample_stride,
    )
    if time_cfg_i is not None and time_cfg_i.use_diffrax and run_options.streaming_fit:
        _append_kinetic_streaming_results(
            batch,
            setup,
            time_cfg=time_cfg_i,
            fit_key=run_options.fit_key,
            mode_method=run_options.mode_method,
            tmin=fit_options.tmin,
            tmax=fit_options.tmax,
            start_fraction=fit_options.start_fraction,
            window_fraction=fit_options.window_fraction,
            streaming_amp_floor=run_options.streaming_amp_floor,
            density_species_index=run_options.density_species_index,
            output=output,
        )
        return

    phi_t, density_t, stride = _integrate_kinetic_scan_history(
        batch,
        setup,
        time_cfg=time_cfg_i,
        method=run_options.method,
        fit_key=run_options.fit_key,
        mode_only=run_options.mode_only,
        mode_method=run_options.mode_method,
        sample_stride=run_options.sample_stride,
        density_species_index=run_options.density_species_index,
        show_progress=run_options.show_progress,
    )
    _append_kinetic_sampled_results(
        batch,
        setup,
        phi_t=phi_t,
        density_t=density_t,
        fit_key=run_options.fit_key,
        mode_only=run_options.mode_only,
        mode_method=run_options.mode_method,
        fit_policy=fit_options.fit_policy,
        density_species_index=run_options.density_species_index,
        stride=stride,
        output=output,
    )


def _build_kinetic_scan_fit_policy(
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
) -> ScanFitWindowPolicy:
    """Pack kinetic-scan growth-window options once for all batches."""

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
        fit_growth_rate_fn=fit_growth_rate,
        fit_growth_rate_auto_fn=fit_growth_rate_auto,
        normalize_growth_rate_fn=_normalize_growth_rate,
    )


def _build_kinetic_scan_run_options(
    *,
    ky_values: np.ndarray,
    time_cfg: TimeConfig | None,
    solver_key: str,
    krylov_cfg: KrylovConfig | None,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    sample_stride: int | None,
    mode_method: str,
    mode_only: bool,
    fit_key: str,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    init_species_index: int,
    density_species_index: int,
    use_batch: bool,
    show_progress: bool,
) -> _KineticScanRunOptions:
    """Pack kinetic scan runtime controls for batch execution."""

    return _KineticScanRunOptions(
        ky_values=ky_values,
        time_cfg=time_cfg,
        solver_key=solver_key,
        krylov_cfg=krylov_cfg,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        mode_method=mode_method,
        mode_only=mode_only,
        fit_key=fit_key,
        ky_batch=ky_batch,
        fixed_batch_shape=fixed_batch_shape,
        streaming_fit=streaming_fit,
        streaming_amp_floor=streaming_amp_floor,
        init_species_index=init_species_index,
        density_species_index=density_species_index,
        use_batch=use_batch,
        show_progress=show_progress,
    )


def _run_kinetic_scan_batches(
    *,
    setup: _KineticScanSetup,
    run_options: _KineticScanRunOptions,
    fit_options: _KineticScanFitOptions,
    n_laguerre: int,
    n_hermite: int,
) -> _KineticScanOutput:
    """Execute all kinetic ky batches and collect scan rows."""

    output = _KineticScanOutput.empty()
    for batch_start, ky_slice, valid_count in _iter_kinetic_scan_batches(run_options):
        _append_kinetic_time_batch_results(
            batch_start=batch_start,
            ky_slice=ky_slice,
            valid_count=valid_count,
            setup=setup,
            run_options=run_options,
            fit_options=fit_options,
            Nl=n_laguerre,
            Nm=n_hermite,
            output=output,
        )
    return output


def _kinetic_scan_result(output: _KineticScanOutput) -> LinearScanResult:
    """Pack kinetic scan rows into the public scan result."""

    return LinearScanResult(
        ky=np.array(output.ky),
        gamma=np.array(output.gammas),
        omega=np.array(output.omegas),
    )


def _prepare_kinetic_scan_controls(
    request: _KineticScanControlRequest,
) -> _KineticScanControls:
    """Resolve setup, execution, and fitting controls for one kinetic scan."""

    setup = _resolve_kinetic_scan_setup(
        cfg=request.cfg,
        params=request.params,
        terms=request.terms,
        diagnostic_norm=request.diagnostic_norm,
        reference_aligned=request.reference_aligned,
        Nm=request.Nm,
    )
    solver_key = normalize_solver_key(request.solver)
    fit_key = normalize_fit_signal(request.fit_signal)
    resolved_mode_method = resolve_scan_mode_method(
        request.mode_method,
        mode_only=request.mode_only,
    )
    use_batch = should_use_ky_batch(
        ky_batch=request.ky_batch,
        solver_key=solver_key,
        dt=request.dt,
        steps=request.steps,
        tmin=request.tmin,
        tmax=request.tmax,
    )
    fit_policy = _build_kinetic_scan_fit_policy(
        tmin=request.tmin,
        tmax=request.tmax,
        auto_window=request.auto_window,
        window_fraction=request.window_fraction,
        min_points=request.min_points,
        start_fraction=request.start_fraction,
        growth_weight=request.growth_weight,
        require_positive=request.require_positive,
        min_amp_fraction=request.min_amp_fraction,
    )

    ky_values_arr = np.asarray(request.ky_values, dtype=float)
    _validate_kinetic_species_indices(
        init_species_index=request.init_species_index,
        density_species_index=request.density_species_index,
    )
    run_options = _build_kinetic_scan_run_options(
        ky_values=ky_values_arr,
        time_cfg=request.time_cfg,
        solver_key=solver_key,
        krylov_cfg=request.krylov_cfg,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        sample_stride=request.sample_stride,
        mode_method=resolved_mode_method,
        mode_only=request.mode_only,
        fit_key=fit_key,
        ky_batch=request.ky_batch,
        fixed_batch_shape=request.fixed_batch_shape,
        streaming_fit=request.streaming_fit,
        streaming_amp_floor=request.streaming_amp_floor,
        init_species_index=request.init_species_index,
        density_species_index=request.density_species_index,
        use_batch=use_batch,
        show_progress=request.show_progress,
    )
    fit_options = _KineticScanFitOptions(
        tmin=request.tmin,
        tmax=request.tmax,
        start_fraction=request.start_fraction,
        window_fraction=request.window_fraction,
        fit_policy=fit_policy,
    )
    return _KineticScanControls(
        setup=setup,
        run_options=run_options,
        fit_options=fit_options,
    )


def run_kinetic_scan(
    ky_values: np.ndarray,
    Nl: int = 6,
    Nm: int = 12,
    dt: float | np.ndarray = 0.01,
    steps: int | np.ndarray = 800,
    method: str = "imex2",
    params: LinearParams | None = None,
    cfg: KineticElectronBaseCase | None = None,
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
    fit_signal: str = "density",
    ky_batch: int = 4,
    fixed_batch_shape: bool = True,
    streaming_fit: bool = True,
    streaming_amp_floor: float = 1.0e-30,
    init_species_index: int = 1,
    density_species_index: int = 1,
    diagnostic_norm: str = "none",
    reference_aligned: bool | None = True,
    show_progress: bool = False,
) -> LinearScanResult:
    """Run a kinetic-electron ITG/TEM benchmark for a list of ky values.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    controls = _prepare_kinetic_scan_controls(
        _kinetic_scan_control_request_from_locals(locals())
    )
    output = _run_kinetic_scan_batches(
        setup=controls.setup,
        run_options=controls.run_options,
        fit_options=controls.fit_options,
        n_laguerre=Nl,
        n_hermite=Nm,
    )
    return _kinetic_scan_result(output)

@dataclass(frozen=True)
class _KBMScanCase:
    cfg: KBMBaseCase
    beta: float
    reference_aligned: bool


@dataclass(frozen=True)
class _KBMScanOptions:
    n_laguerre: int
    n_hermite: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    time_cfg: TimeConfig | None
    solver: str
    krylov_cfg: KrylovConfig | None
    kbm_target_factors: Sequence[float] | None
    kbm_beta_transition: float | None
    tmin: float | np.ndarray | None
    tmax: float | np.ndarray | None
    auto_window: bool
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    mode_method: str
    mode_only: bool
    terms: LinearTerms | None
    sample_stride: int | None
    fit_signal: str
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    init_species_index: int
    density_species_index: int
    diagnostic_norm: str
    fapar_override: float | None
    apar_beta_scale: float | None
    ampere_g0_scale: float | None
    bpar_beta_scale: float | None


@dataclass(frozen=True)
class _KBMScanRequest:
    ky_values: np.ndarray
    beta_value: float | None
    Nl: int
    Nm: int
    dt: float | np.ndarray
    steps: int | np.ndarray
    method: str
    cfg: KBMBaseCase | None
    time_cfg: TimeConfig | None
    solver: str
    krylov_cfg: KrylovConfig | None
    kbm_target_factors: Sequence[float] | None
    kbm_beta_transition: float | None
    tmin: float | np.ndarray | None
    tmax: float | np.ndarray | None
    auto_window: bool
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    mode_method: str
    mode_only: bool
    terms: LinearTerms | None
    sample_stride: int | None
    fit_signal: str
    ky_batch: int
    fixed_batch_shape: bool
    streaming_fit: bool
    streaming_amp_floor: float
    init_species_index: int
    density_species_index: int
    diagnostic_norm: str
    fapar_override: float | None
    apar_beta_scale: float | None
    ampere_g0_scale: float | None
    bpar_beta_scale: float | None
    reference_aligned: bool | None


@dataclass
class _KBMScanOutput:
    ky: list[float]
    gamma: list[float]
    omega: list[float]

    @classmethod
    def empty(cls) -> "_KBMScanOutput":
        return cls(ky=[], gamma=[], omega=[])

    def append(self, *, ky: float, gamma: float, omega: float) -> None:
        self.ky.append(float(ky))
        self.gamma.append(float(gamma))
        self.omega.append(float(omega))

    def result(self) -> LinearScanResult:
        return LinearScanResult(
            ky=np.asarray(self.ky, dtype=float),
            gamma=np.asarray(self.gamma, dtype=float),
            omega=np.asarray(self.omega, dtype=float),
        )


def _kbm_scan_request_from_locals(values: dict[str, Any]) -> _KBMScanRequest:
    """Pack public ``run_kbm_scan`` arguments once for internal routing."""

    return _KBMScanRequest(
        **{field.name: values[field.name] for field in fields(_KBMScanRequest)}
    )


def _resolve_kbm_scan_case(
    *,
    cfg: KBMBaseCase | None,
    beta_value: float | None,
    reference_aligned: bool | None,
) -> _KBMScanCase:
    cfg_in = cfg or KBMBaseCase()
    beta_use = float(cfg_in.model.beta) if beta_value is None else float(beta_value)
    return _KBMScanCase(
        cfg=replace(cfg_in, model=replace(cfg_in.model, beta=beta_use)),
        beta=beta_use,
        reference_aligned=bool(True if reference_aligned is None else reference_aligned),
    )


def _build_kbm_scan_options(
    *,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    time_cfg: TimeConfig | None,
    solver: str,
    krylov_cfg: KrylovConfig | None,
    kbm_target_factors: Sequence[float] | None,
    kbm_beta_transition: float | None,
    tmin: float | np.ndarray | None,
    tmax: float | np.ndarray | None,
    auto_window: bool,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    mode_method: str,
    mode_only: bool,
    terms: LinearTerms | None,
    sample_stride: int | None,
    fit_signal: str,
    ky_batch: int,
    fixed_batch_shape: bool,
    streaming_fit: bool,
    streaming_amp_floor: float,
    init_species_index: int,
    density_species_index: int,
    diagnostic_norm: str,
    fapar_override: float | None,
    apar_beta_scale: float | None,
    ampere_g0_scale: float | None,
    bpar_beta_scale: float | None,
) -> _KBMScanOptions:
    return _KBMScanOptions(
        n_laguerre=Nl,
        n_hermite=Nm,
        dt=dt,
        steps=steps,
        method=method,
        time_cfg=time_cfg,
        solver=solver,
        krylov_cfg=krylov_cfg,
        kbm_target_factors=kbm_target_factors,
        kbm_beta_transition=kbm_beta_transition,
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
    )


def _build_kbm_scan_options_from_request(request: _KBMScanRequest) -> _KBMScanOptions:
    return _build_kbm_scan_options(
        Nl=request.Nl,
        Nm=request.Nm,
        dt=request.dt,
        steps=request.steps,
        method=request.method,
        time_cfg=request.time_cfg,
        solver=request.solver,
        krylov_cfg=request.krylov_cfg,
        kbm_target_factors=request.kbm_target_factors,
        kbm_beta_transition=request.kbm_beta_transition,
        tmin=request.tmin,
        tmax=request.tmax,
        auto_window=request.auto_window,
        window_fraction=request.window_fraction,
        min_points=request.min_points,
        start_fraction=request.start_fraction,
        growth_weight=request.growth_weight,
        require_positive=request.require_positive,
        min_amp_fraction=request.min_amp_fraction,
        mode_method=request.mode_method,
        mode_only=request.mode_only,
        terms=request.terms,
        sample_stride=request.sample_stride,
        fit_signal=request.fit_signal,
        ky_batch=request.ky_batch,
        fixed_batch_shape=request.fixed_batch_shape,
        streaming_fit=request.streaming_fit,
        streaming_amp_floor=request.streaming_amp_floor,
        init_species_index=request.init_species_index,
        density_species_index=request.density_species_index,
        diagnostic_norm=request.diagnostic_norm,
        fapar_override=request.fapar_override,
        apar_beta_scale=request.apar_beta_scale,
        ampere_g0_scale=request.ampere_g0_scale,
        bpar_beta_scale=request.bpar_beta_scale,
    )


def _indexed_kbm_scan_time_value(value: float | np.ndarray, index: int):
    indexed = indexed_scan_value(value, index)
    return value if indexed is None else indexed


def _run_kbm_scan_point(
    *,
    ky_value: float,
    index: int,
    case: _KBMScanCase,
    options: _KBMScanOptions,
) -> tuple[float, float]:
    dt_i = _indexed_kbm_scan_time_value(options.dt, index)
    steps_i = _indexed_kbm_scan_time_value(options.steps, index)
    out = run_kbm_beta_scan(
        betas=np.asarray([case.beta], dtype=float),
        ky_target=float(ky_value),
        Nl=options.n_laguerre,
        Nm=options.n_hermite,
        dt=float(dt_i),
        steps=int(steps_i),
        method=options.method,
        cfg=case.cfg,
        time_cfg=options.time_cfg,
        solver=options.solver,
        krylov_cfg=options.krylov_cfg,
        kbm_target_factors=options.kbm_target_factors,
        kbm_beta_transition=options.kbm_beta_transition,
        tmin=indexed_scan_value(options.tmin, index),
        tmax=indexed_scan_value(options.tmax, index),
        auto_window=options.auto_window,
        window_fraction=options.window_fraction,
        min_points=options.min_points,
        start_fraction=options.start_fraction,
        growth_weight=options.growth_weight,
        require_positive=options.require_positive,
        min_amp_fraction=options.min_amp_fraction,
        mode_method=options.mode_method,
        mode_only=options.mode_only,
        terms=options.terms,
        sample_stride=options.sample_stride,
        fit_signal=options.fit_signal,
        ky_batch=options.ky_batch,
        fixed_batch_shape=options.fixed_batch_shape,
        streaming_fit=options.streaming_fit,
        streaming_amp_floor=options.streaming_amp_floor,
        init_species_index=options.init_species_index,
        density_species_index=options.density_species_index,
        diagnostic_norm=options.diagnostic_norm,
        fapar_override=options.fapar_override,
        apar_beta_scale=options.apar_beta_scale,
        ampere_g0_scale=options.ampere_g0_scale,
        bpar_beta_scale=options.bpar_beta_scale,
        reference_aligned=case.reference_aligned,
    )
    return float(out.gamma[0]), float(out.omega[0])


def _run_kbm_scan_loop(
    ky_values: np.ndarray,
    *,
    case: _KBMScanCase,
    options: _KBMScanOptions,
) -> _KBMScanOutput:
    output = _KBMScanOutput.empty()
    for index, ky_value in enumerate(np.asarray(ky_values, dtype=float)):
        gamma, omega = _run_kbm_scan_point(
            ky_value=float(ky_value),
            index=index,
            case=case,
            options=options,
        )
        output.append(ky=float(ky_value), gamma=gamma, omega=omega)
    return output


def _run_kbm_scan_request(request: _KBMScanRequest) -> LinearScanResult:
    case = _resolve_kbm_scan_case(
        cfg=request.cfg,
        beta_value=request.beta_value,
        reference_aligned=request.reference_aligned,
    )
    options = _build_kbm_scan_options_from_request(request)
    return _run_kbm_scan_loop(request.ky_values, case=case, options=options).result()


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
) -> LinearScanResult:
    """Run a KBM ky scan at fixed beta.

    This is a thin wrapper over :func:`run_kbm_beta_scan` used for
    reference-comparison workflows where the external benchmark is a ky scan
    at fixed beta.
    """

    return _run_kbm_scan_request(_kbm_scan_request_from_locals(locals()))



@dataclass(frozen=True)
class SecondaryModeResult:
    """Per-mode nonlinear secondary-instability diagnostic summary."""

    ky: float
    kx: float
    gamma: float
    omega: float


def _leading_finite_prefix(
    t: ArrayLike,
    signal: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the leading finite segment of a complex diagnostic mode."""

    t_arr = np.asarray(t, dtype=float)
    sig_arr = np.asarray(signal, dtype=np.complex128)
    finite = np.isfinite(sig_arr)
    if not np.any(finite):
        return t_arr[:0], sig_arr[:0]
    first_bad = np.where(~finite)[0]
    stop = int(first_bad[0]) if first_bad.size else int(sig_arr.size)
    return t_arr[:stop], sig_arr[:stop]


def _tail_mean_pair(
    gamma_t: ArrayLike,
    omega_t: ArrayLike,
    *,
    tail_fraction: float | None,
) -> tuple[float, float] | None:
    """Return a late-time average from finite gamma/omega diagnostic series."""

    gamma_arr = np.asarray(gamma_t, dtype=float)
    omega_arr = np.asarray(omega_t, dtype=float)
    finite = np.isfinite(gamma_arr) & np.isfinite(omega_arr)
    if not np.any(finite):
        return None
    gamma_finite = gamma_arr[finite]
    omega_finite = omega_arr[finite]
    if tail_fraction is None:
        return float(gamma_finite[-1]), float(omega_finite[-1])
    istart = int(len(gamma_finite) * (1.0 - float(tail_fraction)))
    istart = max(0, min(istart, len(gamma_finite) - 1))
    return float(np.mean(gamma_finite[istart:])), float(np.mean(omega_finite[istart:]))


def write_restart_state(path: str | Path, state: np.ndarray) -> Path:
    """Write a complex restart state in the runtime NetCDF layout."""

    return write_netcdf_restart_state(path, state)


def _embed_linear_seed_on_full_grid(
    cfg: RuntimeConfig,
    state: np.ndarray,
    *,
    ky_target: float,
) -> np.ndarray:
    """Embed a selected-ky linear seed into the full nonlinear runtime grid."""

    geom = build_flux_tube_geometry(cfg.geometry)
    grid_cfg = apply_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    full_shape = (
        state.shape[0],
        state.shape[1],
        state.shape[2],
        grid.ky.size,
        grid.kx.size,
        grid.z.size,
    )
    if tuple(state.shape) == full_shape:
        return np.asarray(state, dtype=np.complex64)
    if state.ndim != 6 or state.shape[3] != 1:
        raise ValueError(
            f"expected selected-ky linear state with shape (..., 1, Nx, Nz), got {state.shape}"
        )
    ky = np.asarray(grid.ky, dtype=float)
    ky_idx = int(np.argmin(np.abs(ky - float(ky_target))))
    full_state = np.zeros(full_shape, dtype=np.complex64)
    full_state[..., ky_idx : ky_idx + 1, :, :] = np.asarray(state, dtype=np.complex64)
    return full_state


def run_secondary_seed(
    cfg: RuntimeConfig,
    *,
    restart_path: str | Path,
    ky_target: float,
    Nl: int,
    Nm: int,
    dt: float = 1.0,
    steps: int = 2,
    method: str = "sspx3",
    solver: str = "time",
) -> Path:
    """Run the linear secondary seed stage and write its final restart state."""

    result = run_runtime_linear(
        cfg,
        ky_target=ky_target,
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        method=method,
        dt=dt,
        steps=steps,
        return_state=True,
    )
    if result.state is None:
        raise RuntimeError("Secondary seed run did not return a final state.")
    state_full = _embed_linear_seed_on_full_grid(cfg, result.state, ky_target=ky_target)
    return write_restart_state(restart_path, state_full)


def build_secondary_stage2_config(
    cfg: RuntimeConfig,
    *,
    restart_file: str | Path,
    restart_scale: float = 500.0,
    init_amp: float = 1.0e-5,
    dt: float = 0.01,
    t_max: float = 100.0,
    method: str = "sspx3",
    iky_fixed: int = 1,
    ikx_fixed: int = 0,
) -> RuntimeConfig:
    """Return a nonlinear stage-2 config for the secondary slab benchmark."""

    time_cfg = replace(
        cfg.time,
        t_max=float(t_max),
        dt=float(dt),
        method=str(method),
        use_diffrax=False,
        fixed_dt=True,
    )
    init_cfg = replace(
        cfg.init,
        init_amp=float(init_amp),
        init_single=False,
        init_file=str(restart_file),
        init_file_scale=float(restart_scale),
        init_file_mode="add",
    )
    physics_cfg = replace(cfg.physics, linear=False, nonlinear=True)
    terms_cfg = replace(cfg.terms, nonlinear=1.0)
    expert_cfg = RuntimeExpertConfig(
        fixed_mode=True, iky_fixed=int(iky_fixed), ikx_fixed=int(ikx_fixed)
    )
    return replace(
        cfg,
        time=time_cfg,
        init=init_cfg,
        physics=physics_cfg,
        terms=terms_cfg,
        expert=expert_cfg,
    )


def run_secondary_modes(
    cfg: RuntimeConfig,
    *,
    modes: Sequence[tuple[float, float]],
    Nl: int,
    Nm: int,
    steps: int | None = None,
    sample_stride: int = 100,
    fit_fraction: float | None = 0.5,
) -> list[SecondaryModeResult]:
    """Run one nonlinear secondary stage per requested diagnostic mode."""

    rows: list[SecondaryModeResult] = []
    for ky_target, kx_target in modes:
        result = run_runtime_nonlinear(
            cfg,
            ky_target=float(ky_target),
            kx_target=float(kx_target),
            Nl=Nl,
            Nm=Nm,
            steps=steps,
            sample_stride=sample_stride,
        )
        if result.diagnostics is None:
            raise RuntimeError("Secondary nonlinear run did not produce diagnostics.")
        tail_mean = _tail_mean_pair(
            result.diagnostics.gamma_t,
            result.diagnostics.omega_t,
            tail_fraction=fit_fraction,
        )
        gamma = float(tail_mean[0]) if tail_mean is not None else 0.0
        omega = float(tail_mean[1]) if tail_mean is not None else 0.0
        phi_mode_t = result.diagnostics.phi_mode_t
        if fit_fraction is not None and phi_mode_t is not None:
            t, signal = _leading_finite_prefix(result.diagnostics.t, phi_mode_t)
            if t.size >= 2 and np.max(np.abs(signal)) > 0.0:
                t_span = float(t[-1] - t[0]) if t.size > 1 else 0.0
                tmin = (
                    float(t[0] + (1.0 - float(fit_fraction)) * t_span)
                    if t_span > 0.0
                    else None
                )
                try:
                    gamma_fit, omega_fit = fit_growth_rate(t, signal, tmin=tmin)
                    gamma = float(gamma_fit)
                    if tail_mean is None:
                        omega = float(omega_fit)
                except ValueError:
                    gamma = float(tail_mean[0]) if tail_mean is not None else 0.0
                    omega = float(tail_mean[1]) if tail_mean is not None else 0.0
        rows.append(
            SecondaryModeResult(
                ky=float(ky_target),
                kx=float(kx_target),
                gamma=float(gamma),
                omega=float(omega),
            )
        )
    return rows



def _sync_metric_hooks() -> None:
    _gate_metrics.extract_mode_time_series = extract_mode_time_series
    _gate_metrics.fit_growth_rate = fit_growth_rate


def zonal_flow_response_metrics(*args: Any, **kwargs: Any) -> ZonalFlowResponseMetrics:
    """Estimate residual level and GAM envelope metrics from a zonal response."""

    return _zonal_validation.zonal_flow_response_metrics(*args, **kwargs)


def late_time_linear_metrics(*args: Any, **kwargs: Any) -> LateTimeLinearMetrics:
    """Return late-time growth/frequency metrics from a linear result."""

    _sync_metric_hooks()
    return _gate_metrics.late_time_linear_metrics(*args, **kwargs)


def windowed_nonlinear_metrics(*args: Any, **kwargs: Any) -> NonlinearWindowMetrics:
    """Return late-window transport metrics from nonlinear diagnostics."""

    return _gate_metrics.windowed_nonlinear_metrics(*args, **kwargs)


def nonlinear_heat_flux_convergence_metrics(
    *args: Any, **kwargs: Any
) -> NonlinearHeatFluxConvergenceMetrics:
    """Summarize post-transient heat-flux average stability."""

    return _gate_metrics.nonlinear_heat_flux_convergence_metrics(*args, **kwargs)


def estimate_observed_order(*args: Any, **kwargs: Any) -> ObservedOrderMetrics:
    """Estimate observed order from step-size refinements."""

    return _gate_metrics.estimate_observed_order(*args, **kwargs)


def branch_continuity_metrics(*args: Any, **kwargs: Any) -> BranchContinuationMetrics:
    """Compute branch-continuity diagnostics for a linear scan."""

    return _gate_metrics.branch_continuity_metrics(*args, **kwargs)


@dataclass(frozen=True)
class ScanAndModeResult:
    scan: LinearScanResult
    eigenfunction: np.ndarray
    grid: SpectralGrid
    ky_selected: float
    tmin: float | None
    tmax: float | None


def run_linear_scan(
    *,
    ky_values: np.ndarray,
    run_linear_fn,
    cfg,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    solver: str,
    krylov_cfg,
    window_kw: dict,
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    run_kwargs: dict | None = None,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], object] | None = None,
) -> LinearScanResult:
    """Run a linear scan over ky values."""

    gammas: list[float] = []
    omegas: list[float] = []
    ky_out: list[float] = []
    for i, ky in enumerate(ky_values):
        if resolution_policy is not None:
            Nl_i, Nm_i = resolution_policy(float(ky))
        else:
            Nl_i, Nm_i = int(Nl), int(Nm)
        dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        tmin_i = tmin[i] if isinstance(tmin, np.ndarray) else tmin
        tmax_i = tmax[i] if isinstance(tmax, np.ndarray) else tmax
        krylov_cfg_use = (
            krylov_policy(float(ky)) if krylov_policy is not None else krylov_cfg
        )
        result = run_linear_fn(
            ky_target=float(ky),
            cfg=cfg,
            Nl=int(Nl_i),
            Nm=int(Nm_i),
            dt=dt_i,
            steps=steps_i,
            method=method,
            solver=solver,
            krylov_cfg=krylov_cfg_use,
            auto_window=auto_window,
            tmin=tmin_i,
            tmax=tmax_i,
            **window_kw,
            **(run_kwargs or {}),
        )
        gammas.append(float(result.gamma))
        omegas.append(float(result.omega))
        ky_out.append(float(result.ky))

    return LinearScanResult(
        ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas)
    )


def _select_representative_ky(
    scan: LinearScanResult,
    select_ky: Callable[[LinearScanResult], float] | None,
) -> float:
    if select_ky is not None:
        return float(select_ky(scan))
    return float(scan.ky[int(np.nanargmax(scan.gamma))])


def _resolution_for_ky(
    ky: float,
    *,
    Nl: int,
    Nm: int,
    resolution_policy: Callable[[float], tuple[int, int]] | None,
) -> tuple[int, int]:
    if resolution_policy is not None:
        n_l, n_m = resolution_policy(float(ky))
        return int(n_l), int(n_m)
    return int(Nl), int(Nm)


def _mode_control_value(
    value: float | int | np.ndarray,
    idx: int,
    cast,
):
    if isinstance(value, np.ndarray):
        return cast(value[idx])
    return cast(value)


def _run_representative_mode(
    *,
    scan: LinearScanResult,
    ky_selected: float,
    linear_fn,
    cfg,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    mode_solver: str,
    window_kw: dict,
    mode_kwargs: dict | None,
    resolution_policy: Callable[[float], tuple[int, int]] | None,
) -> LinearRunResult:
    n_l, n_m = _resolution_for_ky(
        ky_selected, Nl=Nl, Nm=Nm, resolution_policy=resolution_policy
    )
    idx = int(np.argmin(np.abs(scan.ky - ky_selected)))
    return linear_fn(
        cfg=cfg,
        ky_target=ky_selected,
        Nl=n_l,
        Nm=n_m,
        dt=_mode_control_value(dt, idx, float),
        steps=_mode_control_value(steps, idx, int),
        method=method,
        solver=mode_solver,
        **window_kw,
        **(mode_kwargs or {}),
    )


def _fit_representative_mode_window(
    run: LinearRunResult,
    window_kw: dict,
) -> tuple[float | None, float | None]:
    if run.t.size < 2:
        return None, None
    signal = extract_mode_time_series(run.phi_t, run.selection, method="project")
    _g, _w, tmin_fit, tmax_fit = fit_growth_rate_auto(run.t, signal, **window_kw)
    return tmin_fit, tmax_fit


def _extract_representative_eigenfunction(
    run: LinearRunResult,
    grid: SpectralGrid,
    *,
    tmin_fit: float | None,
    tmax_fit: float | None,
) -> np.ndarray:
    return extract_eigenfunction(
        run.phi_t,
        run.t,
        run.selection,
        z=np.asarray(grid.z),
        method="snapshot",
        tmin=tmin_fit,
        tmax=tmax_fit,
    )


def run_scan_and_mode(
    *,
    ky_values: np.ndarray,
    scan_fn,
    linear_fn,
    cfg,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    solver: str,
    mode_solver: str,
    krylov_cfg,
    window_kw: dict,
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    run_kwargs: dict | None = None,
    mode_kwargs: dict | None = None,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], object] | None = None,
    select_ky: Callable[[LinearScanResult], float] | None = None,
) -> ScanAndModeResult:
    """Run a scan and extract a representative eigenfunction."""

    scan = run_linear_scan(
        ky_values=ky_values,
        run_linear_fn=linear_fn,
        cfg=cfg,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps,
        method=method,
        solver=solver,
        krylov_cfg=krylov_cfg,
        window_kw=window_kw,
        tmin=tmin,
        tmax=tmax,
        auto_window=auto_window,
        run_kwargs=run_kwargs,
        resolution_policy=resolution_policy,
        krylov_policy=krylov_policy,
    )
    ky_sel = _select_representative_ky(scan, select_ky)
    run = _run_representative_mode(
        scan=scan,
        ky_selected=ky_sel,
        linear_fn=linear_fn,
        cfg=cfg,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps,
        method=method,
        mode_solver=mode_solver,
        window_kw=window_kw,
        mode_kwargs=mode_kwargs,
        resolution_policy=resolution_policy,
    )
    grid = build_spectral_grid(cfg.grid)
    tmin_fit, tmax_fit = _fit_representative_mode_window(run, window_kw)
    eig = _extract_representative_eigenfunction(
        run, grid, tmin_fit=tmin_fit, tmax_fit=tmax_fit
    )
    return ScanAndModeResult(
        scan=scan,
        eigenfunction=eig,
        grid=grid,
        ky_selected=ky_sel,
        tmin=tmin_fit,
        tmax=tmax_fit,
    )


__all__ = [
    "CYCLONE_KRYLOV_DEFAULT",
    "CYCLONE_OMEGA_D_SCALE",
    "CYCLONE_OMEGA_STAR_SCALE",
    "CYCLONE_RHO_STAR",
    "ETG_KRYLOV_DEFAULT",
    "ETG_OMEGA_D_SCALE",
    "ETG_OMEGA_STAR_SCALE",
    "ETG_RHO_STAR",
    "KBM_KRYLOV_DEFAULT",
    "KBM_EXPLICIT_SOLVER_LOCK",
    "KBM_EXPLICIT_SOLVER_LOCK_TOL",
    "KBM_OMEGA_D_SCALE",
    "KBM_OMEGA_STAR_SCALE",
    "KBM_RHO_STAR",
    "KINETIC_KRYLOV_DEFAULT",
    "KINETIC_KRYLOV_REFERENCE_ALIGNED",
    "KINETIC_OMEGA_D_SCALE",
    "KINETIC_OMEGA_STAR_SCALE",
    "KINETIC_RHO_STAR",
    "REFERENCE_DAMP_ENDS_AMP",
    "REFERENCE_DAMP_ENDS_WIDTHFRAC",
    "REFERENCE_NU_HYPER_L",
    "REFERENCE_NU_HYPER_M",
    "REFERENCE_P_HYPER_L",
    "REFERENCE_P_HYPER_M",
    "TEM_KRYLOV_DEFAULT",
    "TEM_OMEGA_D_SCALE",
    "TEM_OMEGA_STAR_SCALE",
    "TEM_RHO_STAR",
    "CycloneBaseCase",
    "CycloneComparison",
    "CycloneReference",
    "CycloneRunResult",
    "CycloneScanResult",
    "ETGBaseCase",
    "KBMBaseCase",
    "KineticElectronBaseCase",
    "KrylovConfig",
    "LinearRunResult",
    "LinearScanResult",
    "ModeSelection",
    "SecondaryModeResult",
    "TEMBaseCase",
    "_apply_reference_hypercollisions",
    "_build_gaussian_profile",
    "_build_initial_condition",
    "_electron_only_params",
    "_extract_mode_only_signal",
    "_linked_boundary_end_damping",
    "_reference_hypercollision_power",
    "_is_array_like",
    "_iter_ky_batches",
    "_kbm_use_multi_target_krylov",
    "_kinetic_reference_init_cfg",
    "_load_reference_with_header",
    "_midplane_index",
    "_normalize_growth_rate",
    "_resolve_streaming_window",
    "_score_fit_signal_auto",
    "_select_fit_signal",
    "_select_fit_signal_auto",
    "_two_species_params",
    "compare_cyclone_to_reference",
    "ExplicitTimeConfig",
    "load_cyclone_reference",
    "load_cyclone_reference_kinetic",
    "load_etg_reference",
    "load_kbm_reference",
    "load_tem_reference",
    "run_cyclone_linear",
    "run_cyclone_scan",
    "run_etg_linear",
    "run_etg_scan",
    "run_kbm_beta_scan",
    "run_kbm_linear",
    "run_kbm_scan",
    "run_kinetic_linear",
    "run_kinetic_scan",
    "run_secondary_modes",
    "run_secondary_seed",
    "run_tem_linear",
    "run_tem_scan",
    "select_kbm_solver_auto",
    "build_secondary_stage2_config",
    "write_restart_state",
    "_analytic_signal",
    "_explicit_time_window",
    "_leading_window",
    "BranchContinuationMetrics",
    "DiagnosticTimeSeries",
    "EigenfunctionComparisonMetrics",
    "EigenfunctionReferenceBundle",
    "GateReport",
    "LateTimeLinearMetrics",
    "NonlinearHeatFluxConvergenceMetrics",
    "NonlinearWindowMetrics",
    "ObservedOrderMetrics",
    "ScalarGateResult",
    "ScanAndModeResult",
    "ZonalFlowResponseMetrics",
    "branch_continuity_gate_report",
    "branch_continuity_metrics",
    "compare_eigenfunctions",
    "eigenfunction_gate_report",
    "estimate_observed_order",
    "evaluate_scalar_gate",
    "gate_report",
    "gate_report_to_dict",
    "infer_triple_dealiased_ny",
    "late_time_linear_metrics",
    "late_time_window",
    "linear_metrics_gate_report",
    "load_diagnostic_time_series",
    "load_eigenfunction_reference_bundle",
    "nonlinear_heat_flux_convergence_gate_report",
    "nonlinear_heat_flux_convergence_metrics",
    "nonlinear_window_gate_report",
    "normalize_eigenfunction",
    "observed_order_gate_report",
    "phase_align_eigenfunction",
    "run_linear_scan",
    "run_scan_and_mode",
    "save_eigenfunction_reference_bundle",
    "windowed_nonlinear_metrics",
    "zonal_flow_response_metrics",
    "zonal_response_gate_report",
]
