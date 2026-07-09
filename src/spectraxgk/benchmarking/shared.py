"""Shared benchmark reference data, normalization, and scan policies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from importlib import resources
from typing import Any, Sequence, TypeVar, cast

import jax.numpy as jnp
import numpy as np

from spectraxgk.config import InitializationConfig, KineticElectronBaseCase
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.core.species import Species, build_linear_params
from spectraxgk.diagnostics.analysis import fit_growth_rate, fit_growth_rate_auto
from spectraxgk.diagnostics.growth_rates import _normalize_growth_rate
from spectraxgk.diagnostics.modes import ModeSelection
from spectraxgk.diagnostics.normalization import (
    CYCLONE_NORMALIZATION,
    ETG_NORMALIZATION,
    KBM_NORMALIZATION,
    KINETIC_NORMALIZATION,
    TEM_NORMALIZATION,
)
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.solvers.linear.krylov import KrylovConfig

VALID_FIT_SIGNALS = frozenset({"phi", "density", "auto"})
_Record = TypeVar("_Record")


def _pack_dataclass_fields(
    record_type: type[_Record], values: Mapping[str, Any]
) -> _Record:
    """Build an internal request record from a public function's locals."""

    constructor = cast(Any, record_type)
    return constructor(
        **{
            field.name: values[field.name]
            for field in fields(cast(Any, record_type))
        }
    )

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


__all__ = [
    'resources',
    'VALID_FIT_SIGNALS',
    '_is_array_like',
    '_pack_dataclass_fields',
    '_iter_ky_batches',
    '_resolve_streaming_window',
    'normalize_solver_key',
    'normalize_fit_signal',
    'apply_auto_fit_scan_policy',
    'resolve_scan_mode_method',
    'indexed_float_value',
    'indexed_scan_value',
    'scan_window_valid',
    'should_use_ky_batch',
    'ScanFitWindowPolicy',
    'CYCLONE_OMEGA_D_SCALE',
    'CYCLONE_OMEGA_STAR_SCALE',
    'CYCLONE_RHO_STAR',
    'ETG_OMEGA_D_SCALE',
    'ETG_OMEGA_STAR_SCALE',
    'ETG_RHO_STAR',
    'KINETIC_OMEGA_D_SCALE',
    'KINETIC_OMEGA_STAR_SCALE',
    'KINETIC_RHO_STAR',
    'TEM_OMEGA_D_SCALE',
    'TEM_OMEGA_STAR_SCALE',
    'TEM_RHO_STAR',
    'KBM_OMEGA_D_SCALE',
    'KBM_OMEGA_STAR_SCALE',
    'KBM_RHO_STAR',
    'REFERENCE_NU_HYPER_L',
    'REFERENCE_NU_HYPER_M',
    'REFERENCE_P_HYPER_L',
    'REFERENCE_P_HYPER_M',
    'REFERENCE_DAMP_ENDS_AMP',
    'REFERENCE_DAMP_ENDS_WIDTHFRAC',
    '_reference_hypercollision_power',
    '_apply_reference_hypercollisions',
    '_linked_boundary_end_damping',
    '_two_species_params',
    '_electron_only_params',
    'KBM_EXPLICIT_SOLVER_LOCK',
    'KBM_EXPLICIT_SOLVER_LOCK_TOL',
    '_midplane_index',
    'select_kbm_solver_auto',
    '_kbm_use_multi_target_krylov',
    'CYCLONE_KRYLOV_DEFAULT',
    'KINETIC_KRYLOV_DEFAULT',
    'KINETIC_KRYLOV_REFERENCE_ALIGNED',
    'ETG_KRYLOV_DEFAULT',
    'KBM_KRYLOV_DEFAULT',
    'TEM_KRYLOV_DEFAULT',
    'CycloneReference',
    'CycloneRunResult',
    'CycloneScanResult',
    'CycloneComparison',
    'LinearRunResult',
    'LinearScanResult',
    '_load_csv_reference',
    'load_cyclone_reference',
    '_load_reference_with_header',
    'load_cyclone_reference_kinetic',
    'load_kbm_reference',
    'load_etg_reference',
    'load_tem_reference',
    'compare_cyclone_to_reference',
    '_build_gaussian_profile',
    '_build_initial_condition',
    '_kinetic_reference_init_cfg',
]
