"""Benchmark utilities for linear Cyclone base case comparisons."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence
import warnings
import numpy as np
from importlib import resources

import jax.numpy as jnp

from spectraxgk.analysis import (
    ModeSelection,
    ModeSelectionBatch,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_auto_with_stats,
    gx_growth_rate_from_omega_series,
    gx_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    InitializationConfig,
    KineticElectronBaseCase,
    KBMBaseCase,
    TEMBaseCase,
    TimeConfig,
    resolve_cfl_fac,
)
from spectraxgk.geometry import (
    FluxTubeGeometryLike,
    SAlphaGeometry,
    apply_gx_geometry_grid_defaults,
    build_flux_tube_geometry,
)
from spectraxgk.grids import SpectralGrid, build_spectral_grid, select_ky_grid
from spectraxgk.diffrax_integrators import (
    integrate_linear_diffrax,
    integrate_linear_diffrax_streaming,
)
from spectraxgk.gx_integrators import (
    GXTimeConfig,
    integrate_linear_gx,
    integrate_linear_gx_diagnostics,
)
from spectraxgk.linear import (
    LinearParams,
    LinearTerms,
    build_linear_cache,
    integrate_linear,
    integrate_linear_diagnostics,
    linear_terms_to_term_config,
)
from spectraxgk.linear_krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.normalization import (
    KBM_NORMALIZATION,
    KINETIC_NORMALIZATION,
    TEM_NORMALIZATION,
    CYCLONE_NORMALIZATION,
    ETG_NORMALIZATION,
    apply_diagnostic_normalization,
)
from spectraxgk.runners import integrate_linear_from_config
from spectraxgk.species import Species, build_linear_params
from spectraxgk.terms.assembly import compute_fields_cached


CYCLONE_OMEGA_D_SCALE = CYCLONE_NORMALIZATION.omega_d_scale
CYCLONE_OMEGA_STAR_SCALE = CYCLONE_NORMALIZATION.omega_star_scale
CYCLONE_RHO_STAR = CYCLONE_NORMALIZATION.rho_star

ETG_OMEGA_D_SCALE = ETG_NORMALIZATION.omega_d_scale
ETG_OMEGA_STAR_SCALE = ETG_NORMALIZATION.omega_star_scale
ETG_RHO_STAR = ETG_NORMALIZATION.rho_star

Kinetic_OMEGA_D_SCALE = KINETIC_NORMALIZATION.omega_d_scale
Kinetic_OMEGA_STAR_SCALE = KINETIC_NORMALIZATION.omega_star_scale
Kinetic_RHO_STAR = KINETIC_NORMALIZATION.rho_star

TEM_OMEGA_D_SCALE = TEM_NORMALIZATION.omega_d_scale
TEM_OMEGA_STAR_SCALE = TEM_NORMALIZATION.omega_star_scale
TEM_RHO_STAR = TEM_NORMALIZATION.rho_star

KBM_OMEGA_D_SCALE = KBM_NORMALIZATION.omega_d_scale
KBM_OMEGA_STAR_SCALE = KBM_NORMALIZATION.omega_star_scale
KBM_RHO_STAR = KBM_NORMALIZATION.rho_star

GX_NU_HYPER_L = 0.0
GX_NU_HYPER_M = 1.0
GX_P_HYPER_L = 6.0
GX_P_HYPER_M = 20.0
GX_DAMP_ENDS_AMP = 0.1
GX_DAMP_ENDS_WIDTHFRAC = 1.0 / 8.0

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
    omega_sign=-1,
    power_iters=60,
    power_dt=0.001,
    shift_source="target",
    shift_maxiter=40,
    shift_restart=12,
    shift_tol=5.0e-4,
    shift_preconditioner="hermite-line",
    mode_family="tem",
    fallback_method="propagator",
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

KBM_GX_SOLVER_LOCK: tuple[tuple[float, str], ...] = (
    (0.10, "gx_time"),
    (0.30, "gx_time"),
    (0.40, "gx_time"),
)
KBM_GX_SOLVER_LOCK_TOL = 0.03

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


def _gx_p_hyper_m(nhermite: int | None) -> float:
    if nhermite is None:
        return GX_P_HYPER_M
    return float(min(GX_P_HYPER_M, max(int(nhermite) // 2, 1)))


def _apply_gx_hypercollisions(params: LinearParams, *, nhermite: int | None = None) -> LinearParams:
    return replace(
        params,
        nu_hyper=0.0,
        nu_hyper_l=GX_NU_HYPER_L,
        nu_hyper_m=GX_NU_HYPER_M,
        p_hyper_l=GX_P_HYPER_L,
        p_hyper_m=_gx_p_hyper_m(nhermite),
        hypercollisions_const=0.0,
        hypercollisions_kz=1.0,
    )


def _gx_linked_end_damping(gx_reference: bool) -> tuple[float, float]:
    if gx_reference:
        return GX_DAMP_ENDS_AMP, GX_DAMP_ENDS_WIDTHFRAC
    return 0.0, 0.0


def _midplane_index(grid: SpectralGrid) -> int:
    """Return GX-style midplane index for growth-rate diagnostics."""

    if grid.z.size <= 1:
        return 0
    idx = int(grid.z.size // 2 + 1)
    return min(idx, int(grid.z.size) - 1)


def select_kbm_solver_auto(solver: str, *, ky_target: float, gx_reference: bool) -> str:
    """Return deterministic KBM solver choice for auto mode."""

    solver_key = solver.strip().lower()
    if solver_key != "auto":
        return solver_key
    if not gx_reference:
        return "time"
    ky_abs = abs(float(ky_target))
    for ky_ref, solver_ref in KBM_GX_SOLVER_LOCK:
        if abs(ky_abs - ky_ref) <= KBM_GX_SOLVER_LOCK_TOL:
            return solver_ref
    return "gx_time"


def _select_fit_signal(
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    sel: ModeSelection,
    *,
    fit_signal: str,
    mode_method: str,
    fallback: bool = True,
) -> np.ndarray:
    def _extract(arr: np.ndarray) -> np.ndarray:
        return extract_mode_time_series(arr, sel, method=mode_method)

    def _is_valid(arr: np.ndarray) -> bool:
        finite = np.isfinite(arr)
        return int(np.count_nonzero(finite)) >= 2

    if fit_signal == "phi":
        signal = _extract(phi_t)
        if fallback and not _is_valid(signal) and density_t is not None:
            alt = _extract(density_t)
            if _is_valid(alt):
                return alt
        if not _is_valid(signal):
            warnings.warn(
                "Fit signal has insufficient finite samples; falling back to zeros.",
                RuntimeWarning,
            )
            return np.zeros(phi_t.shape[0], dtype=np.complex128)
        return signal
    if fit_signal == "density":
        if density_t is None:
            raise ValueError("density_t must be provided when fit_signal='density'")
        signal = _extract(density_t)
        if fallback and not _is_valid(signal):
            alt = _extract(phi_t)
            if _is_valid(alt):
                return alt
        if not _is_valid(signal):
            warnings.warn(
                "Fit signal has insufficient finite samples; falling back to zeros.",
                RuntimeWarning,
            )
            return np.zeros(phi_t.shape[0], dtype=np.complex128)
        return signal
    raise ValueError("fit_signal must be 'phi' or 'density'")


def _score_fit_signal_auto(
    t: np.ndarray,
    signal: np.ndarray,
    *,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    max_amp_fraction: float,
    window_method: str,
    max_fraction: float,
    end_fraction: float,
    num_windows: int,
    phase_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    min_slope: float | None,
    min_slope_frac: float,
    slope_var_weight: float,
) -> tuple[float, float, float]:
    """Score a candidate fit signal using auto-window stats."""

    try:
        gamma, omega, _tmin, _tmax, r2_log, r2_phase = fit_growth_rate_auto_with_stats(
            t,
            signal,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
            max_amp_fraction=max_amp_fraction,
            window_method=window_method,
            max_fraction=max_fraction,
            end_fraction=end_fraction,
                        num_windows=8,
            phase_weight=phase_weight,
            length_weight=length_weight,
            min_r2=min_r2,
            late_penalty=late_penalty,
            min_slope=min_slope,
            min_slope_frac=min_slope_frac,
            slope_var_weight=slope_var_weight,
        )
    except ValueError:
        return 0.0, 0.0, -np.inf

    if not np.isfinite(gamma) or not np.isfinite(omega):
        return gamma, omega, -np.inf
    if require_positive and gamma <= 0.0:
        return gamma, omega, -np.inf
    if r2_log < min_r2:
        return gamma, omega, -np.inf
    score = float(r2_log + phase_weight * r2_phase + growth_weight * gamma)
    return gamma, omega, score


def _select_fit_signal_auto(
    t: np.ndarray,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    sel: ModeSelection,
    *,
    mode_method: str,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    max_amp_fraction: float,
    window_method: str,
    max_fraction: float,
    end_fraction: float,
    num_windows: int,
    phase_weight: float,
    length_weight: float,
    min_r2: float,
    late_penalty: float,
    min_slope: float | None,
    min_slope_frac: float,
    slope_var_weight: float,
) -> tuple[np.ndarray, str, float, float]:
    """Choose between phi/density signals based on fit quality."""

    phi_signal = extract_mode_time_series(phi_t, sel, method=mode_method)
    gamma_phi, omega_phi, score_phi = _score_fit_signal_auto(
        t,
        phi_signal,
        tmin=tmin,
        tmax=tmax,
        window_fraction=window_fraction,
        min_points=min_points,
        start_fraction=start_fraction,
        growth_weight=growth_weight,
        require_positive=require_positive,
        min_amp_fraction=min_amp_fraction,
        max_amp_fraction=max_amp_fraction,
        window_method=window_method,
        max_fraction=max_fraction,
        end_fraction=end_fraction,
                        num_windows=8,
        phase_weight=phase_weight,
        length_weight=length_weight,
        min_r2=min_r2,
        late_penalty=late_penalty,
        min_slope=min_slope,
        min_slope_frac=min_slope_frac,
        slope_var_weight=slope_var_weight,
    )
    best_signal = phi_signal
    best_name = "phi"
    best_gamma = gamma_phi
    best_omega = omega_phi
    best_score = score_phi

    if density_t is not None:
        density_signal = extract_mode_time_series(density_t, sel, method=mode_method)
        gamma_den, omega_den, score_den = _score_fit_signal_auto(
            t,
            density_signal,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
            max_amp_fraction=max_amp_fraction,
            window_method=window_method,
            max_fraction=max_fraction,
            end_fraction=end_fraction,
            num_windows=num_windows,
            phase_weight=phase_weight,
            length_weight=length_weight,
            min_r2=min_r2,
            late_penalty=late_penalty,
            min_slope=min_slope,
            min_slope_frac=min_slope_frac,
            slope_var_weight=slope_var_weight,
        )
        if score_den > best_score:
            best_signal = density_signal
            best_name = "density"
            best_gamma = gamma_den
            best_omega = omega_den
            best_score = score_den

    return best_signal, best_name, float(best_gamma), float(best_omega)


def _extract_mode_only_signal(
    source: np.ndarray,
    *,
    local_idx: int,
    species_index: int | None = None,
) -> np.ndarray:
    """Extract a 1D time trace from reduced mode-only outputs."""

    arr = np.asarray(source)
    if arr.ndim == 0:
        return np.asarray([arr], dtype=np.complex128)
    if arr.ndim == 1:
        return arr

    # Some save modes return (t, species, ky). Select requested species first.
    if species_index is not None and arr.ndim >= 3 and arr.shape[1] > 0:
        idx = min(max(int(species_index), 0), arr.shape[1] - 1)
        arr = arr[:, idx, ...]

    if arr.ndim == 2:
        idx = min(max(int(local_idx), 0), arr.shape[1] - 1)
        return arr[:, idx]

    # Final fallback: flatten non-time axes and select one column.
    arr2 = arr.reshape(arr.shape[0], -1)
    idx = min(max(int(local_idx), 0), arr2.shape[1] - 1)
    return arr2[:, idx]


def _is_array_like(value) -> bool:
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
    if tmin is not None and tmax is not None:
        return float(tmin), float(tmax)
    t_start = float(start_fraction) * t_total
    t_end = float(end_fraction) * t_total
    t_end = min(t_end, t_start + float(window_fraction) * t_total)
    if t_end <= t_start:
        t_end = t_total
    return t_start, t_end


def _normalize_growth_rate(
    gamma: float,
    omega: float,
    params: LinearParams,
    diagnostic_norm: str,
) -> tuple[float, float]:
    return apply_diagnostic_normalization(
        gamma,
        omega,
        rho_star=float(np.asarray(params.rho_star)),
        diagnostic_norm=diagnostic_norm,
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
    envelope = init_cfg.gaussian_envelope_constant + init_cfg.gaussian_envelope_sine * np.sin(z - theta0)
    width = init_cfg.gaussian_width
    if width <= 0.0:
        raise ValueError("gaussian_width must be > 0")
    return envelope * np.exp(-((z - theta0) / width) ** 2)


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
    # GX scales some moments when init_field="all" (see moments.cu).
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


def load_cyclone_reference() -> CycloneReference:
    """Load Cyclone base case reference data (adiabatic electrons)."""

    data_path = resources.files("spectraxgk").joinpath("data", "cyclone_reference_adiabatic.csv")
    arr = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


def _load_reference_with_header(filename: str) -> CycloneReference:
    """Load reference CSVs with columns ky,gamma,omega."""

    data_path = resources.files("spectraxgk").joinpath("data", filename)
    arr = np.genfromtxt(str(data_path), delimiter=",", names=True, dtype=float)
    ky = np.asarray(arr["ky"], dtype=float)
    gamma = np.asarray(arr["gamma"], dtype=float)
    omega = np.asarray(arr["omega"], dtype=float)
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


def load_cyclone_reference_kinetic() -> CycloneReference:
    """Load Cyclone base case reference data (kinetic electrons)."""

    data_path = resources.files("spectraxgk").joinpath("data", "cyclone_reference_kinetic.csv")
    arr = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


def load_kbm_reference() -> CycloneReference:
    """Load KBM reference data (finite beta, kinetic electrons)."""

    data_path = resources.files("spectraxgk").joinpath("data", "kbm_reference.csv")
    arr = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


def load_etg_reference() -> CycloneReference:
    """Load GX-backed ETG reference data for the tracked two-species ETG lane."""

    data_path = resources.files("spectraxgk").joinpath("data", "etg_reference.csv")
    arr = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


def load_tem_reference() -> CycloneReference:
    """Load TEM reference data digitized from the literature."""

    data_path = resources.files("spectraxgk").joinpath("data", "tem_reference.csv")
    arr = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


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
    """Build LinearParams for a two-species kinetic model (ions + electrons)."""

    mass_ratio = float(model.mass_ratio)
    if mass_ratio <= 0.0:
        raise ValueError("mass_ratio must be > 0")
    Te_over_Ti = float(model.Te_over_Ti)
    if Te_over_Ti <= 0.0:
        raise ValueError("Te_over_Ti must be > 0")
    ion_fprim_raw = getattr(model, "R_over_Lni", None)
    ele_fprim_raw = getattr(model, "R_over_Lne", None)
    ion_fprim = float(model.R_over_Ln) if ion_fprim_raw is None else float(ion_fprim_raw)
    ele_fprim = float(model.R_over_Ln) if ele_fprim_raw is None else float(ele_fprim_raw)

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
    params = _apply_gx_hypercollisions(params, nhermite=nhermite)
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
    """Build LinearParams for a single kinetic electron species + Boltzmann ions."""

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
    params = _apply_gx_hypercollisions(params, nhermite=nhermite)
    if fapar_override is not None:
        params = replace(params, fapar=float(fapar_override))
    if damp_ends_amp is not None:
        params = replace(params, damp_ends_amp=float(damp_ends_amp))
    if damp_ends_widthfrac is not None:
        params = replace(params, damp_ends_widthfrac=float(damp_ends_widthfrac))
    return params


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
    gx_reference: bool | None = None,
) -> CycloneRunResult:
    """Run the linear Cyclone benchmark and extract growth rate."""

    cfg = cfg or CycloneBaseCase()
    init_cfg = init_cfg or getattr(cfg, "init", None) or InitializationConfig()
    grid_full = build_spectral_grid(cfg.grid)
    gx_reference_use = bool(cfg.gx_reference) if gx_reference is None else bool(gx_reference)
    geom_cfg = cfg.geometry
    if gx_reference_use:
        geom_cfg = replace(geom_cfg, drift_scale=1.0)
        if diagnostic_norm == "none":
            diagnostic_norm = "gx"
        if mode_method not in {"z_index", "max"}:
            mode_method = "z_index"
    geom = SAlphaGeometry.from_config(geom_cfg)
    if params is None:
        params = LinearParams(
            R_over_Ln=cfg.model.R_over_Ln,
            R_over_LTi=cfg.model.R_over_LTi,
            R_over_LTe=cfg.model.R_over_LTe,
            omega_d_scale=CYCLONE_OMEGA_D_SCALE,
            omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
            rho_star=CYCLONE_RHO_STAR,
            kpar_scale=float(geom.gradpar()),
            nu=cfg.model.nu_i,
            damp_ends_amp=GX_DAMP_ENDS_AMP,
            damp_ends_widthfrac=GX_DAMP_ENDS_WIDTHFRAC,
        )
        params = _apply_gx_hypercollisions(params, nhermite=Nm)
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
    G0_jax = _build_initial_condition(
        grid,
        geom,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=init_cfg,
    )
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
        if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
            return False
        if require_positive and gamma_val <= 0.0:
            return False
        return True

    def _run_krylov() -> tuple[float, float, np.ndarray, np.ndarray]:
        kcfg = krylov_cfg or CYCLONE_KRYLOV_DEFAULT
        # GX-style time seed to stabilize the branch selection.
        gamma_seed = 0.0
        omega_seed = 0.0
        seed_ok = False
        omega_ok = False
        try:
            t_seed = min(150.0, float(kcfg.power_dt) * 15000.0)
            time_cfg = GXTimeConfig(
                dt=float(kcfg.power_dt),
                t_max=t_seed,
                sample_stride=1,
                fixed_dt=True,
            )
            G0_seed = jnp.asarray(np.asarray(G0_jax))
            t_short, phi_t, _g_t, _o_t = integrate_linear_gx(
                G0_seed,
                grid,
                cache,
                params,
                geom,
                time_cfg,
                terms=terms,
                mode_method="z_index",
            )
            sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
            gamma_seed, omega_seed, _g, _o, _t_mid = gx_growth_rate_from_phi(
                phi_t,
                t_short,
                sel,
                navg_fraction=0.5,
                mode_method="z_index",
            )
            omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
            seed_ok = (
                omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
            )
        except Exception:
            seed_ok = False
            omega_ok = False

        if not seed_ok:
            try:
                Nl_seed = min(Nl, 16)
                Nm_seed = min(Nm, 12)
                cache_seed = build_linear_cache(grid, geom, params, Nl_seed, Nm_seed)
                G0_seed = _build_initial_condition(
                    grid,
                    geom,
                    ky_index=sel.ky_index,
                    kx_index=sel.kx_index,
                    Nl=Nl_seed,
                    Nm=Nm_seed,
                    init_cfg=init_cfg,
                )
                t_seed = min(150.0, float(kcfg.power_dt) * 15000.0)
                time_cfg = GXTimeConfig(
                    dt=float(kcfg.power_dt),
                    t_max=t_seed,
                    sample_stride=1,
                    fixed_dt=True,
                )
                G0_seed = jnp.asarray(np.asarray(G0_seed))
                t_short, phi_t, _g_t, _o_t = integrate_linear_gx(
                    G0_seed,
                    grid,
                    cache_seed,
                    params,
                    geom,
                    time_cfg,
                    terms=terms,
                    mode_method="z_index",
                )
                sel_seed = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
                gamma_seed, omega_seed, _g, _o, _t_mid = gx_growth_rate_from_phi(
                    phi_t,
                    t_short,
                    sel_seed,
                    navg_fraction=0.5,
                    mode_method="z_index",
                )
                omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
                seed_ok = (
                    omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
                )
            except Exception:
                seed_ok = False
                omega_ok = False

        shift = None
        if omega_ok:
            shift = complex(float(gamma_seed) if seed_ok else 0.0, float(-omega_seed))
        G0_krylov = jnp.asarray(np.asarray(G0_jax))
        eig, vec = dominant_eigenpair(
            G0_krylov,
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
        )
        term_cfg = linear_terms_to_term_config(terms)
        phi = compute_fields_cached(vec, cache, params, terms=term_cfg).phi
        phi_t_out = np.asarray(phi)[None, ...]
        t_out = np.array([0.0])
        gamma_out = float(np.real(eig))
        omega_out = float(-np.imag(eig))
        if seed_ok:
            seed_strong = (gamma_seed > 0.0) and (abs(omega_seed) > 1.0e-6)
            if seed_strong:
                omega_tol = 0.15 * max(abs(omega_seed), 1.0e-6)
                gamma_tol = 0.15 * max(abs(gamma_seed), 1.0e-6)
                use_seed = (
                    not np.isfinite(gamma_out)
                    or not np.isfinite(omega_out)
                    or (gamma_seed > 0.0 and gamma_out < 0.0)
                    or abs(omega_out - omega_seed) > omega_tol
                    or abs(gamma_out - gamma_seed) > gamma_tol
                )
                if use_seed:
                    gamma_out = float(gamma_seed)
                    omega_out = float(omega_seed)
        if kcfg.omega_sign != 0:
            omega_out = float(np.sign(kcfg.omega_sign)) * abs(omega_out)
        gamma_out, omega_out = _normalize_growth_rate(
            gamma_out, omega_out, params, diagnostic_norm
        )
        return gamma_out, omega_out, phi_t_out, t_out

    def _run_time() -> tuple[float, float, np.ndarray, np.ndarray]:
        method_key = method.lower()
        phi_t: jnp.ndarray | np.ndarray
        density_t: jnp.ndarray | np.ndarray | None
        time_cfg_use = None
        if time_cfg is not None:
            time_cfg_use = replace(time_cfg, dt=float(dt), t_max=float(dt) * int(steps))
            if sample_stride is not None:
                time_cfg_use = replace(time_cfg_use, sample_stride=sample_stride)
        elif cfg.time.use_diffrax and not (
            method_key.startswith("imex") or method_key.startswith("implicit")
        ):
            time_cfg_use = replace(cfg.time, dt=float(dt), t_max=float(dt) * int(steps))
            if sample_stride is not None:
                time_cfg_use = replace(time_cfg_use, sample_stride=sample_stride)

        if gx_reference_use:
            # GX integrator applies damping with per-time scaling internally.
            params_use = params
            t_max_val = float(dt) * int(steps) if time_cfg_use is None else float(time_cfg_use.t_max)
            stride = 1 if time_cfg_use is None else int(time_cfg_use.sample_stride)
            gx_time_cfg = GXTimeConfig(
                dt=float(dt),
                t_max=t_max_val,
                sample_stride=stride,
                fixed_dt=True,
            )
            t, phi_t, _g_t, _o_t = integrate_linear_gx(
                G0_jax,
                grid,
                cache,
                params_use,
                geom,
                gx_time_cfg,
                terms=terms,
                mode_method="z_index",
            )
            sel_local = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
            gamma, omega, _g, _o, _t_mid = gx_growth_rate_from_phi(
                phi_t, t, sel_local, navg_fraction=0.5, mode_method="z_index"
            )
            gamma, omega = _normalize_growth_rate(gamma, omega, params_use, diagnostic_norm)
            return gamma, omega, np.asarray(phi_t), np.asarray(t)

        params_use = params
        if time_cfg_use is not None:
            if need_density:
                _, saved = integrate_linear_from_config(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    time_cfg_use,
                    terms=terms,
                    save_field="phi+density",
                    density_species_index=0,
                )
                phi_t, density_t = saved
            else:
                _, phi_t = integrate_linear_from_config(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    time_cfg_use,
                    terms=terms,
                )
                density_t = None
            stride = time_cfg_use.sample_stride
        else:
            stride = 1 if sample_stride is None else int(sample_stride)
            if need_density or not use_jit:
                _diag = integrate_linear_diagnostics(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    dt=dt,
                    steps=steps,
                    method=method,
                    terms=terms,
                    sample_stride=stride,
                    species_index=0,
                    record_hl_energy=False,
                )
                phi_t = _diag[1]
                density_t = _diag[2] if len(_diag) > 2 else None
            else:
                _, phi_out_time = integrate_linear(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    dt=dt,
                    steps=steps,
                    method=method,
                    terms=terms,
                    sample_stride=stride,
                )
                density_t = None

        phi_t_np = np.asarray(phi_t)
        t_arr = np.arange(phi_t_np.shape[0]) * dt * stride
        density_np = None if density_t is None else np.asarray(density_t)
        if fit_key == "auto":
            signal, _name, gamma_out, omega_out = _select_fit_signal_auto(
                t_arr,
                phi_t_np,
                density_np,
                sel,
                mode_method=mode_method,
                tmin=tmin,
                tmax=tmax,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
                max_amp_fraction=max_amp_fraction,
                window_method=window_method,
                max_fraction=max_fraction,
                end_fraction=end_fraction,
                num_windows=8,
                phase_weight=phase_weight,
                length_weight=length_weight,
                min_r2=min_r2,
                late_penalty=late_penalty,
                min_slope=min_slope,
                min_slope_frac=min_slope_frac,
                slope_var_weight=slope_var_weight,
            )
            if not np.isfinite(gamma_out) or not np.isfinite(omega_out):
                gamma_out, omega_out = 0.0, 0.0
        else:
            signal = _select_fit_signal(
                phi_t_np,
                density_np,
                sel,
                fit_signal=fit_key,
                mode_method=mode_method,
            )
            if auto_window and tmin is None and tmax is None:
                gamma_out, omega_out, _tmin, _tmax = fit_growth_rate_auto(
                    t_arr,
                    signal,
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
            else:
                gamma_out, omega_out = fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)
        gamma_out, omega_out = _normalize_growth_rate(
            gamma_out, omega_out, params_use, diagnostic_norm
        )
        return float(gamma_out), float(omega_out), phi_t_np, t_arr

    if solver_key == "krylov":
        gamma, omega, phi_t_np, t = _run_krylov()
    elif solver_key == "auto":
        gamma, omega, phi_t_np, t = _run_time()
        if not _is_valid_growth(gamma, omega):
            gamma, omega, phi_t_np, t = _run_krylov()
    else:
        gamma, omega, phi_t_np, t = _run_time()

    return CycloneRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
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
    gx_reference: bool | None = None,
) -> CycloneScanResult:
    """Run the linear Cyclone benchmark for a list of ky values.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    cfg = cfg or CycloneBaseCase()
    init_cfg = getattr(cfg, "init", None) or InitializationConfig()
    grid_full = build_spectral_grid(cfg.grid)
    gx_reference_use = bool(cfg.gx_reference) if gx_reference is None else bool(gx_reference)
    geom_cfg = cfg.geometry
    if gx_reference_use:
        geom_cfg = replace(geom_cfg, drift_scale=1.0)
        if diagnostic_norm == "none":
            diagnostic_norm = "gx"
        if mode_method not in {"z_index", "max"}:
            mode_method = "z_index"
    geom = SAlphaGeometry.from_config(geom_cfg)
    if params is None:
        params = LinearParams(
            R_over_Ln=cfg.model.R_over_Ln,
            R_over_LTi=cfg.model.R_over_LTi,
            R_over_LTe=cfg.model.R_over_LTe,
            omega_d_scale=CYCLONE_OMEGA_D_SCALE,
            omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
            rho_star=CYCLONE_RHO_STAR,
            kpar_scale=float(geom.gradpar()),
            nu=cfg.model.nu_i,
            damp_ends_amp=GX_DAMP_ENDS_AMP,
            damp_ends_widthfrac=GX_DAMP_ENDS_WIDTHFRAC,
        )
        params = _apply_gx_hypercollisions(params, nhermite=Nm)
    if terms is None:
        if getattr(cfg.model, "adiabatic_ions", False):
            terms = LinearTerms(bpar=0.0)
        else:
            terms = LinearTerms()
    solver_key = solver.strip().lower()
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    auto_solver = solver_key == "auto"
    if auto_solver:
        solver_key = "gx_time" if gx_reference_use else "time"
    if fit_key == "auto":
        streaming_fit = False
        mode_only = False
    need_density = fit_key in {"density", "auto"}
    gammas = []
    omegas = []
    ky_out = []
    def _window_value(val, idx):
        if val is None:
            return None
        if isinstance(val, (list, tuple, np.ndarray)):
            return float(val[idx])
        return float(val)

    def _window_valid(t_arr, tmin_val, tmax_val):
        if tmin_val is None or tmax_val is None:
            return False
        mask = (t_arr >= tmin_val) & (t_arr <= tmax_val)
        return int(np.count_nonzero(mask)) >= 2

    if mode_only and mode_method not in {"z_index", "max"}:
        mode_method = "z_index"

    if ky_batch < 1:
        raise ValueError("ky_batch must be >= 1")
    use_batch = (
        ky_batch > 1
        and solver_key != "krylov"
        and not _is_array_like(dt)
        and not _is_array_like(steps)
        and not _is_array_like(tmin)
        and not _is_array_like(tmax)
    )

    def _fit_signal(signal: np.ndarray, idx: int, dt_i: float, stride: int) -> tuple[float, float]:
        t = np.arange(signal.shape[0]) * dt_i * stride
        tmin_i = _window_value(tmin, idx)
        tmax_i = _window_value(tmax, idx)
        use_auto = auto_window and tmin_i is None and tmax_i is None
        if not use_auto and not _window_valid(t, tmin_i, tmax_i):
            use_auto = True
        if use_auto:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                t,
                signal,
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
        else:
            try:
                gamma, omega = fit_growth_rate(t, signal, tmin=tmin_i, tmax=tmax_i)
            except ValueError:
                gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                    t,
                    signal,
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
        return _normalize_growth_rate(gamma, omega, params, diagnostic_norm)

    ky_values_arr = np.asarray(ky_values, dtype=float)
    phi_t: jnp.ndarray | np.ndarray
    density_t: jnp.ndarray | np.ndarray | None

    if solver_key == "krylov":
        if ky_values_arr.size == 0:
            return CycloneScanResult(ky=ky_values_arr, gamma=np.array([]), omega=np.array([]))
        order = np.argsort(ky_values_arr) if mode_follow else np.arange(ky_values_arr.size)
        gamma_out = np.zeros_like(ky_values_arr, dtype=float)
        omega_out = np.zeros_like(ky_values_arr, dtype=float)
        v_ref: jnp.ndarray | None = None
        prev_eig: complex | None = None
        cfg_use = krylov_cfg or CYCLONE_KRYLOV_DEFAULT
        for idx in order:
            ky_val = float(ky_values_arr[idx])
            ky_index = select_ky_index(np.asarray(grid_full.ky), ky_val)
            grid = select_ky_grid(grid_full, ky_index)
            G0_jax = _build_initial_condition(
                grid,
                geom,
                ky_index=0,
                kx_index=0,
                Nl=Nl,
                Nm=Nm,
                init_cfg=init_cfg,
            )
            cache = build_linear_cache(grid, geom, params, Nl, Nm)
            # Use a short GX-style time integration to seed the branch.
            gamma_seed = 0.0
            omega_seed = 0.0
            seed_ok = False
            omega_ok = False
            if prev_eig is None:
                try:
                    t_seed = min(150.0, float(cfg_use.power_dt) * 15000.0)
                    gx_time_cfg = GXTimeConfig(
                        dt=float(cfg_use.power_dt), t_max=t_seed, sample_stride=1, fixed_dt=True
                    )
                    G0_seed = jnp.array(G0_jax)
                    t_short, phi_seed, _g_t, _o_t = integrate_linear_gx(
                        G0_seed,
                        grid,
                        cache,
                        params,
                        geom,
                        gx_time_cfg,
                        terms=terms,
                        mode_method="z_index",
                    )
                    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
                    gamma_seed, omega_seed, _g, _o, _t_mid = gx_growth_rate_from_phi(
                        phi_seed,
                        t_short,
                        sel,
                        navg_fraction=0.5,
                        mode_method="z_index",
                    )
                    omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
                    seed_ok = (
                        omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
                    )
                except Exception:
                    seed_ok = False
                    omega_ok = False
            if not seed_ok:
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
                    t_seed = min(150.0, float(cfg_use.power_dt) * 15000.0)
                    gx_time_cfg = GXTimeConfig(
                        dt=float(cfg_use.power_dt), t_max=t_seed, sample_stride=1, fixed_dt=True
                    )
                    t_short, phi_seed, _g_t, _o_t = integrate_linear_gx(
                        G0_seed,
                        grid,
                        cache_seed,
                        params,
                        geom,
                        gx_time_cfg,
                        terms=terms,
                        mode_method="z_index",
                    )
                    sel_seed = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
                    gamma_seed, omega_seed, _g, _o, _t_mid = gx_growth_rate_from_phi(
                        phi_seed,
                        t_short,
                        sel_seed,
                        navg_fraction=0.5,
                        mode_method="z_index",
                    )
                    omega_ok = np.isfinite(omega_seed) and abs(omega_seed) > 1.0e-8
                    seed_ok = (
                        omega_ok and np.isfinite(gamma_seed) and gamma_seed > 0.0
                    )
                except Exception:
                    seed_ok = False
                    omega_ok = False

            shift: complex | None
            if prev_eig is not None and np.isfinite(prev_eig):
                shift = prev_eig
            elif omega_ok:
                shift = complex(float(gamma_seed) if seed_ok else 0.0, float(-omega_seed))
            else:
                shift = None
            eig, vec = dominant_eigenpair(
                G0_jax,
                cache,
                params,
                terms=terms,
                v_ref=v_ref,
                select_overlap=v_ref is not None,
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
            gamma = float(np.real(eig))
            omega = float(-np.imag(eig))
            # If Krylov lands on the wrong branch, fall back to GX-style seed.
            use_seed = False
            if seed_ok:
                seed_strong = (gamma_seed > 0.0) and (abs(omega_seed) > 1.0e-6)
                if seed_strong:
                    omega_tol = 0.15 * max(abs(omega_seed), 1.0e-6)
                    gamma_tol = 0.15 * max(abs(gamma_seed), 1.0e-6)
                    use_seed = (
                        not np.isfinite(gamma)
                        or not np.isfinite(omega)
                        or (gamma_seed > 0.0 and gamma < 0.0)
                        or abs(omega - omega_seed) > omega_tol
                        or abs(gamma - gamma_seed) > gamma_tol
                    )
            if use_seed and seed_ok:
                gamma = float(gamma_seed)
                omega = float(omega_seed)
            else:
                v_ref = vec
            prev_eig = complex(float(gamma), float(-omega))
            gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
            gamma_out[idx] = gamma
            omega_out[idx] = omega
        return CycloneScanResult(ky=ky_values_arr, gamma=gamma_out, omega=omega_out)

    if solver_key == "gx_time":
        if ky_values_arr.size == 0:
            return CycloneScanResult(ky=ky_values_arr, gamma=np.array([]), omega=np.array([]))
        gamma_out = np.zeros_like(ky_values_arr, dtype=float)
        omega_out = np.zeros_like(ky_values_arr, dtype=float)
        prev_omega: float | None = None
        prev_prev_omega: float | None = None
        kcfg = krylov_cfg or CYCLONE_KRYLOV_DEFAULT
        time_base = time_cfg or cfg.time
        for idx, ky_val in enumerate(ky_values_arr):
            ky_index = select_ky_index(np.asarray(grid_full.ky), float(ky_val))
            grid = select_ky_grid(grid_full, ky_index)
            G0_jax = _build_initial_condition(
                grid,
                geom,
                ky_index=0,
                kx_index=0,
                Nl=Nl,
                Nm=Nm,
                init_cfg=init_cfg,
            )
            cache = build_linear_cache(grid, geom, params, Nl, Nm)
            dt_i = float(dt[idx]) if isinstance(dt, np.ndarray) else float(dt)
            steps_i = int(steps[idx]) if isinstance(steps, np.ndarray) else int(steps)
            t_max_val = dt_i * float(steps_i)
            if gx_reference_use and time_cfg is None:
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
                cfl_fac_i = resolve_cfl_fac(str(time_base.method), time_base.cfl_fac)
            gx_time_cfg = GXTimeConfig(
                dt=dt_i,
                t_max=t_max_val,
                sample_stride=1,
                fixed_dt=fixed_dt_i,
                dt_min=dt_min_i,
                dt_max=dt_max_i,
                cfl=cfl_i,
                cfl_fac=cfl_fac_i,
            )
            G0_time = jnp.array(G0_jax)
            t, phi_gx, _g_t, _o_t = integrate_linear_gx(
                G0_time,
                grid,
                cache,
                params,
                geom,
                gx_time_cfg,
                terms=terms,
                mode_method="z_index",
            )
            sel_local = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
            gx_growth_ok = True
            try:
                gamma, omega, _g, _o, _t_mid = gx_growth_rate_from_phi(
                    phi_gx, t, sel_local, navg_fraction=0.5, mode_method="z_index"
                )
                gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
            except ValueError:
                gx_growth_ok = False
                gamma = float("nan")
                omega = float("nan")
            if gx_reference_use and prev_omega is None and omega < 0.0:
                omega = abs(omega)
            need_reselect = (
                (gx_reference_use and gx_growth_ok)
                and prev_omega is not None
                and prev_omega > 0.0
                and (
                    omega <= 0.0
                    or ((idx >= 2) and (omega < 0.85 * prev_omega))
                )
            )
            if need_reselect or not gx_growth_ok:
                target_omega: float | None = prev_omega if (gx_growth_ok and prev_omega is not None) else None
                if (
                    target_omega is not None
                    and prev_prev_omega is not None
                    and prev_omega is not None
                    and prev_omega > prev_prev_omega
                ):
                    target_omega = prev_omega + (prev_omega - prev_prev_omega)
                G0_krylov = jnp.array(G0_jax)
                eig, _vec = dominant_eigenpair(
                    G0_krylov,
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
                gamma_k, omega_k = _normalize_growth_rate(gamma_k, omega_k, params, diagnostic_norm)
                if not gx_growth_ok:
                    gamma, omega = gamma_k, omega_k
                else:
                    assert target_omega is not None
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

                    gamma, omega = min(candidates, key=_score)
            gamma_out[idx] = gamma
            omega_out[idx] = omega
            prev_prev_omega = prev_omega
            prev_omega = float(omega)
        return CycloneScanResult(ky=ky_values_arr, gamma=gamma_out, omega=omega_out)
    if use_batch:
        ky_iter = _iter_ky_batches(
            ky_values_arr,
            ky_batch=ky_batch,
            fixed_batch_shape=fixed_batch_shape,
        )
    else:
        ky_iter = _iter_ky_batches(ky_values_arr, ky_batch=1, fixed_batch_shape=False)
    prev_vec: jnp.ndarray | None = None
    prev_eig_scan: complex | None = None
    ky_slice: np.ndarray
    ky_indices: list[int]
    sel_scan: ModeSelection | ModeSelectionBatch

    for batch_start, ky_slice, valid_count in ky_iter:
        if use_batch:
            ky_indices = [select_ky_index(np.asarray(grid_full.ky), float(ky)) for ky in ky_slice]
            grid = select_ky_grid(grid_full, ky_indices)
            sel_indices = np.arange(len(ky_indices), dtype=int)
            sel_scan = ModeSelectionBatch(sel_indices, 0, _midplane_index(grid))
            dt_i = float(dt)
            steps_i = int(steps)
        else:
            ky_indices = [select_ky_index(np.asarray(grid_full.ky), float(ky_slice[0]))]
            grid = select_ky_grid(grid_full, ky_indices[0])
            sel_scan = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
            dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
            steps_i = int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)

        ky_local = np.arange(len(ky_indices))
        G0_jax = _build_initial_condition(
            grid,
            geom,
            ky_index=ky_local,
            kx_index=0,
            Nl=Nl,
            Nm=Nm,
            init_cfg=init_cfg,
        )
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        if solver_key == "krylov":
            for local_idx in range(valid_count):
                ky_val = ky_slice[local_idx]
                cfg_use = krylov_cfg or CYCLONE_KRYLOV_DEFAULT
                eig, _vec = dominant_eigenpair(
                    G0_jax,
                    cache,
                    params,
                    terms=terms,
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
                gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
                gammas.append(gamma)
                omegas.append(omega)
                ky_out.append(float(ky_val))
            continue

        method_key = method.lower()
        time_cfg_i = None
        if time_cfg is not None:
            time_cfg_i = replace(time_cfg, dt=dt_i, t_max=dt_i * steps_i)
            if sample_stride is not None:
                time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)

        params_use = params
        if time_cfg_i is not None and time_cfg_i.use_diffrax and streaming_fit:
            t_total = float(time_cfg_i.t_max)
            tmin_i, tmax_i = _resolve_streaming_window(
                t_total, _window_value(tmin, batch_start), _window_value(tmax, batch_start), start_fraction, window_fraction, 1.0
            )
            _, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
                G0_jax,
                grid,
                geom,
                params_use,
                dt=dt_i,
                steps=steps_i,
                method=time_cfg_i.diffrax_solver,
                cache=cache,
                terms=terms,
                adaptive=False,
                rtol=time_cfg_i.diffrax_rtol,
                atol=time_cfg_i.diffrax_atol,
                max_steps=time_cfg_i.diffrax_max_steps,
                progress_bar=time_cfg_i.progress_bar,
                checkpoint=time_cfg_i.checkpoint,
                tmin=tmin_i,
                tmax=tmax_i,
                fit_signal="phi",
                mode_ky_indices=ky_local[:valid_count],
                mode_kx_index=0,
                mode_z_index=_midplane_index(grid),
                mode_method=mode_method,
                amp_floor=streaming_amp_floor,
                return_state=False,
            )
            gamma_arr = np.asarray(gamma_vals)
            omega_arr = np.asarray(omega_vals)
            for local_idx in range(valid_count):
                ky_val = ky_slice[local_idx]
                gamma_i, omega_i = _normalize_growth_rate(
                    float(gamma_arr[local_idx]),
                    float(omega_arr[local_idx]),
                    params_use,
                    diagnostic_norm,
                )
                gammas.append(gamma_i)
                omegas.append(omega_i)
                ky_out.append(float(ky_val))
            continue

        if time_cfg_i is not None:
            save_field = "phi+density" if fit_key == "auto" else ("density" if fit_key == "density" else "phi")
            save_mode = None if fit_key == "auto" else (sel_scan if mode_only else None)
            _, saved = integrate_linear_from_config(
                G0_jax,
                grid,
                geom,
                params_use,
                time_cfg_i,
                cache=cache,
                terms=terms,
                save_mode=save_mode,
                mode_method=mode_method,
                save_field=save_field,
                density_species_index=0 if need_density else None,
            )
            if fit_key == "auto":
                phi_t, density_t = saved
                phi_t = np.asarray(phi_t)
                density_t = np.asarray(density_t)
            else:
                phi_t = np.asarray(saved)
                density_t = None
            stride = time_cfg_i.sample_stride
        else:
            stride = 1 if sample_stride is None else int(sample_stride)
            if use_jit and not need_density:
                _, phi_out_time = integrate_linear(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    dt=dt_i,
                    steps=steps_i,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=stride,
                )
                phi_t = np.asarray(phi_t)
                density_t = None
            else:
                _diag = integrate_linear_diagnostics(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    dt=dt_i,
                    steps=steps_i,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=stride,
                    species_index=None,
                    record_hl_energy=False,
                )
                phi_t = np.asarray(_diag[1])
                density_t = np.asarray(_diag[2]) if len(_diag) > 2 else None

        phi_t_np = np.asarray(phi_t)
        signal_t = None
        if mode_only and phi_t_np.ndim == 2:
            signal_t = phi_t_np

        density_np = None if density_t is None else np.asarray(density_t)
        t = np.arange(phi_t_np.shape[0]) * dt_i * stride

        def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
            if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
                return False
            if require_positive and gamma_val <= 0.0:
                return False
            return True

        for local_idx in range(valid_count):
            ky_val = ky_slice[local_idx]
            if signal_t is None:
                sel_local = ModeSelection(ky_index=local_idx, kx_index=0, z_index=_midplane_index(grid))
                if fit_key == "auto":
                    signal, _name, gamma, omega = _select_fit_signal_auto(
                        t,
                        phi_t_np,
                        density_np,
                        sel_local,
                        mode_method=mode_method,
                        tmin=_window_value(tmin, batch_start + local_idx),
                        tmax=_window_value(tmax, batch_start + local_idx),
                        window_fraction=window_fraction,
                        min_points=min_points,
                        start_fraction=start_fraction,
                        growth_weight=growth_weight,
                        require_positive=require_positive,
                        min_amp_fraction=min_amp_fraction,
                        max_amp_fraction=max_amp_fraction,
                        window_method=window_method,
                        max_fraction=max_fraction,
                        end_fraction=end_fraction,
                        num_windows=8,
                        phase_weight=phase_weight,
                        length_weight=length_weight,
                        min_r2=min_r2,
                        late_penalty=late_penalty,
                        min_slope=min_slope,
                        min_slope_frac=min_slope_frac,
                        slope_var_weight=slope_var_weight,
                    )
                    gamma, omega = _normalize_growth_rate(gamma, omega, params_use, diagnostic_norm)
                    if auto_solver and not _is_valid_growth(gamma, omega):
                        res = run_cyclone_linear(
                            ky_target=float(ky_val),
                            Nl=Nl,
                            Nm=Nm,
                            dt=dt_i,
                            steps=steps_i,
                            method=method,
                            params=params,
                            cfg=cfg,
                            time_cfg=time_cfg,
                            solver="krylov",
                            krylov_cfg=krylov_cfg,
                            diagnostic_norm=diagnostic_norm,
                            fit_signal="phi",
                        )
                        gamma = float(res.gamma)
                        omega = float(res.omega)
                    gammas.append(gamma)
                    omegas.append(omega)
                    ky_out.append(float(ky_val))
                    continue
                signal = extract_mode_time_series(phi_t_np, sel_local, method=mode_method)
            else:
                signal = signal_t[:, local_idx] if signal_t.ndim > 1 else signal_t
            gamma, omega = _fit_signal(signal, batch_start + local_idx, dt_i, stride)
            if auto_solver and not _is_valid_growth(gamma, omega):
                res = run_cyclone_linear(
                    ky_target=float(ky_val),
                    Nl=Nl,
                    Nm=Nm,
                    dt=dt_i,
                    steps=steps_i,
                    method=method,
                    params=params,
                    cfg=cfg,
                    time_cfg=time_cfg,
                    solver="krylov",
                    krylov_cfg=krylov_cfg,
                    diagnostic_norm=diagnostic_norm,
                    fit_signal="phi",
                )
                gamma = float(res.gamma)
                omega = float(res.omega)
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_val))
    return CycloneScanResult(ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas))


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
    gx_growth: bool = False,
    gx_navg_fraction: float = 0.5,
    diagnostic_norm: str = "none",
) -> LinearRunResult:
    """Run an ETG linear benchmark and extract growth rate."""

    cfg = cfg or ETGBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        if getattr(cfg.model, "adiabatic_ions", False):
            params = _electron_only_params(
                cfg.model,
                kpar_scale=float(geom.gradpar()),
                omega_d_scale=ETG_OMEGA_D_SCALE,
                omega_star_scale=ETG_OMEGA_STAR_SCALE,
                rho_star=ETG_RHO_STAR,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
                nhermite=Nm,
            )
        else:
            params = _two_species_params(
                cfg.model,
                kpar_scale=float(geom.gradpar()),
                omega_d_scale=ETG_OMEGA_D_SCALE,
                omega_star_scale=ETG_OMEGA_STAR_SCALE,
                rho_star=ETG_RHO_STAR,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
                nhermite=Nm,
            )
    if terms is None:
        # The ETG benchmark contract is electrostatic for both the adiabatic-ion
        # and two-species variants. Keep the default ETG wrappers aligned with
        # the tracked ETG asset-generation tools.
        terms = LinearTerms(apar=0.0, bpar=0.0, hypercollisions=1.0)

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))

    charge = np.atleast_1d(np.asarray(params.charge_sign))
    ns = int(charge.size)
    electron_index = int(np.argmin(charge))
    G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G0_single = _build_initial_condition(
        grid,
        geom,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=cfg.init,
    )
    G0[electron_index] = np.asarray(G0_single, dtype=np.complex64)

    G0_jax = jnp.asarray(G0)
    solver_key = solver.strip().lower()
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    if fit_key == "auto" and streaming_fit:
        streaming_fit = False
    auto_solver = solver_key == "auto"
    if auto_solver:
        solver_key = "krylov"

    def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
        if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
            return False
        if require_positive and gamma_val <= 0.0:
            return False
        return True

    if solver_key == "krylov":
        krylov_cfg = krylov_cfg or ETG_KRYLOV_DEFAULT
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        eig, vec = dominant_eigenpair(
            G0_jax,
            cache,
            params,
            terms=terms,
            krylov_dim=krylov_cfg.krylov_dim,
            restarts=krylov_cfg.restarts,
            omega_min_factor=krylov_cfg.omega_min_factor,
            omega_target_factor=krylov_cfg.omega_target_factor,
            omega_cap_factor=krylov_cfg.omega_cap_factor,
            omega_sign=krylov_cfg.omega_sign,
            method=krylov_cfg.method,
            power_iters=krylov_cfg.power_iters,
            power_dt=krylov_cfg.power_dt,
            shift=krylov_cfg.shift,
            shift_source=krylov_cfg.shift_source,
            shift_tol=krylov_cfg.shift_tol,
            shift_maxiter=krylov_cfg.shift_maxiter,
            shift_restart=krylov_cfg.shift_restart,
            shift_solve_method=krylov_cfg.shift_solve_method,
            shift_preconditioner=krylov_cfg.shift_preconditioner,
            shift_selection=krylov_cfg.shift_selection,
            mode_family=krylov_cfg.mode_family,
            fallback_method=krylov_cfg.fallback_method,
            fallback_real_floor=krylov_cfg.fallback_real_floor,
        )
        term_cfg = linear_terms_to_term_config(terms)
        phi = compute_fields_cached(vec, cache, params, terms=term_cfg).phi
        phi_t_np = np.asarray(phi)[None, ...]
        t = np.array([0.0])
        gamma = float(np.real(eig))
        omega = float(-np.imag(eig))
        if krylov_cfg.omega_sign != 0:
            omega = float(np.sign(krylov_cfg.omega_sign)) * abs(omega)
        gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
        if auto_solver and not _is_valid_growth(gamma, omega):
            solver_key = "time"

    if solver_key != "krylov":
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
            cache = build_linear_cache(grid, geom, params, Nl, Nm)
            if fit_key in {"density", "auto"}:
                if streaming_fit and time_cfg_use.use_diffrax:
                    t_total = float(dt * steps)
                    tmin_i, tmax_i = _resolve_streaming_window(
                        t_total, tmin, tmax, start_fraction, window_fraction, 1.0
                    )
                    G_last, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
                        G0_jax,
                        grid,
                        geom,
                        params,
                        dt=dt,
                        steps=steps,
                        method=time_cfg_use.diffrax_solver,
                        cache=cache,
                        terms=terms,
                        adaptive=False,
                        rtol=time_cfg_use.diffrax_rtol,
                        atol=time_cfg_use.diffrax_atol,
                        max_steps=time_cfg_use.diffrax_max_steps,
                        progress_bar=time_cfg_use.progress_bar,
                        checkpoint=time_cfg_use.checkpoint,
                        tmin=tmin_i,
                        tmax=tmax_i,
                        fit_signal="density",
                        mode_ky_indices=np.array([0], dtype=int),
                        mode_kx_index=0,
                        mode_z_index=_midplane_index(grid),
                        mode_method=mode_method,
                        amp_floor=streaming_amp_floor,
                        density_species_index=electron_index,
                        return_state=True,
                    )
                    gamma = float(np.asarray(gamma_vals)[0])
                    omega = float(np.asarray(omega_vals)[0])
                    gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
                    if G_last is not None and G_last.ndim == 7:
                        G_last = G_last[0]
                    term_cfg = linear_terms_to_term_config(terms)
                    if G_last is None:
                        raise ValueError("Expected final state from streaming fit; got None.")
                    phi_last = compute_fields_cached(G_last, cache, params, terms=term_cfg).phi
                    phi_t = jnp.asarray(phi_last)[None, ...]
                    density_t = None
                    stride = time_cfg_use.sample_stride
                    phi_t_np = np.asarray(phi_t)
                    t = np.array([tmax_i])
                    return LinearRunResult(
                        t=t,
                        phi_t=phi_t_np,
                        gamma=gamma,
                        omega=omega,
                        ky=float(grid.ky[sel.ky_index]),
                        selection=sel,
                    )
                if time_cfg_use.use_diffrax:
                    _, saved = integrate_linear_diffrax(
                        G0_jax,
                        grid,
                        geom,
                        params,
                        dt=dt,
                        steps=steps,
                        method=time_cfg_use.diffrax_solver,
                        cache=cache,
                        terms=terms,
                        adaptive=time_cfg_use.diffrax_adaptive,
                        rtol=time_cfg_use.diffrax_rtol,
                        atol=time_cfg_use.diffrax_atol,
                        max_steps=time_cfg_use.diffrax_max_steps,
                        progress_bar=time_cfg_use.progress_bar,
                        checkpoint=time_cfg_use.checkpoint,
                        sample_stride=time_cfg_use.sample_stride,
                        return_state=time_cfg_use.save_state,
                        save_field="phi+density",
                        density_species_index=electron_index,
                    )
                    phi_t, density_t = saved
                else:
                    _diag = integrate_linear_diagnostics(
                        G0_jax,
                        grid,
                        geom,
                        params,
                        dt=dt,
                        steps=steps,
                        method=time_cfg_use.method,
                        cache=cache,
                        terms=terms,
                        sample_stride=time_cfg_use.sample_stride,
                        species_index=electron_index,
                    )
                    phi_t = _diag[1]
                    density_t = _diag[2] if len(_diag) > 2 else None
            else:
                _, phi_t = integrate_linear_from_config(
                    G0_jax,
                    grid,
                    geom,
                    params,
                    time_cfg_use,
                    cache=cache,
                    terms=terms,
                )
                density_t = None
            stride = time_cfg_use.sample_stride
        else:
            stride = 1 if sample_stride is None else int(sample_stride)
            if fit_key in {"density", "auto"}:
                _diag = integrate_linear_diagnostics(
                    G0_jax,
                    grid,
                    geom,
                    params,
                    dt=dt,
                    steps=steps,
                    method=method,
                    terms=terms,
                    sample_stride=stride,
                    species_index=electron_index,
                )
                phi_t = _diag[1]
                density_t = _diag[2] if len(_diag) > 2 else None
            else:
                _, phi_t = integrate_linear(
                    G0_jax,
                    grid,
                    geom,
                    params,
                    dt=dt,
                    steps=steps,
                    method=method,
                    terms=terms,
                    sample_stride=stride,
                )
                density_t = None

        phi_t_np = np.asarray(phi_t)
        t = np.arange(phi_t_np.shape[0]) * dt * stride
        density_np = None if density_t is None else np.asarray(density_t)
        if gx_growth and fit_key == "phi":
            gamma, omega, _gamma_t, _omega_t, _t_mid = gx_growth_rate_from_phi(
                phi_t_np,
                t,
                sel,
                navg_fraction=gx_navg_fraction,
                mode_method=mode_method,
            )
            gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
            return LinearRunResult(
                t=t,
                phi_t=phi_t_np,
                gamma=gamma,
                omega=omega,
                ky=float(grid.ky[sel.ky_index]),
                selection=sel,
            )
        if fit_key == "auto":
            signal, _name, gamma, omega = _select_fit_signal_auto(
                t,
                phi_t_np,
                density_np,
                sel,
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
            gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
            return LinearRunResult(
                t=t,
                phi_t=phi_t_np,
                gamma=gamma,
                omega=omega,
                ky=float(grid.ky[sel.ky_index]),
                selection=sel,
            )

        signal = _select_fit_signal(
            phi_t_np,
            density_np,
            sel,
            fit_signal=fit_key,
            mode_method=mode_method,
        )
        def _window_valid(t_arr: np.ndarray, tmin_val: float | None, tmax_val: float | None) -> bool:
            if tmin_val is None or tmax_val is None:
                return False
            mask = (t_arr >= tmin_val) & (t_arr <= tmax_val)
            return int(np.count_nonzero(mask)) >= 2

        use_auto = auto_window and tmin is None and tmax is None
        if not use_auto and not _window_valid(t, tmin, tmax):
            use_auto = True
        if use_auto:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                t,
                signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
        else:
            try:
                gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)
            except ValueError:
                gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                    t,
                    signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
        gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)

    return LinearRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )


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
    gx_growth: bool = False,
    gx_navg_fraction: float = 0.5,
    diagnostic_norm: str = "none",
) -> LinearScanResult:
    """Run an ETG linear benchmark for a list of ky values.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    cfg = cfg or ETGBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        if getattr(cfg.model, "adiabatic_ions", False):
            params = _electron_only_params(
                cfg.model,
                kpar_scale=float(geom.gradpar()),
                omega_d_scale=ETG_OMEGA_D_SCALE,
                omega_star_scale=ETG_OMEGA_STAR_SCALE,
                rho_star=ETG_RHO_STAR,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
                nhermite=Nm,
            )
        else:
            params = _two_species_params(
                cfg.model,
                kpar_scale=float(geom.gradpar()),
                omega_d_scale=ETG_OMEGA_D_SCALE,
                omega_star_scale=ETG_OMEGA_STAR_SCALE,
                rho_star=ETG_RHO_STAR,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
                nhermite=Nm,
            )
    if terms is None:
        # Keep the ETG scan helper on the same electrostatic benchmark contract
        # as the single-ky ETG wrapper and the tracked ETG figure builders.
        terms = LinearTerms(apar=0.0, bpar=0.0, hypercollisions=1.0)
    solver_key = solver.strip().lower()
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    auto_solver = solver_key == "auto"
    if auto_solver:
        solver_key = "time"
    if fit_key == "auto":
        streaming_fit = False
        mode_only = False
    need_density = fit_key in {"density", "auto"}
    gammas = []
    omegas = []
    ky_out = []
    def _window_value(val, idx):
        if val is None:
            return None
        if isinstance(val, (list, tuple, np.ndarray)):
            return float(val[idx])
        return float(val)

    def _window_valid(t_arr, tmin_val, tmax_val):
        if tmin_val is None or tmax_val is None:
            return False
        mask = (t_arr >= tmin_val) & (t_arr <= tmax_val)
        return int(np.count_nonzero(mask)) >= 2

    if mode_only and mode_method not in {"z_index", "max"}:
        mode_method = "z_index"

    if ky_batch < 1:
        raise ValueError("ky_batch must be >= 1")
    use_batch = (
        ky_batch > 1
        and solver_key != "krylov"
        and not _is_array_like(dt)
        and not _is_array_like(steps)
        and not _is_array_like(tmin)
        and not _is_array_like(tmax)
    )

    def _fit_signal(signal: np.ndarray, idx: int, dt_i: float, stride: int) -> tuple[float, float]:
        t = np.arange(signal.shape[0]) * dt_i * stride
        tmin_i = _window_value(tmin, idx)
        tmax_i = _window_value(tmax, idx)
        use_auto = auto_window and tmin_i is None and tmax_i is None
        if not use_auto and not _window_valid(t, tmin_i, tmax_i):
            use_auto = True
        if use_auto:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                t,
                signal,
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
        else:
            try:
                gamma, omega = fit_growth_rate(t, signal, tmin=tmin_i, tmax=tmax_i)
            except ValueError:
                gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                    t,
                    signal,
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
        return _normalize_growth_rate(gamma, omega, params, diagnostic_norm)

    ky_values_arr = np.asarray(ky_values, dtype=float)
    if use_batch:
        ky_iter = _iter_ky_batches(
            ky_values_arr,
            ky_batch=ky_batch,
            fixed_batch_shape=fixed_batch_shape,
        )
    else:
        ky_iter = _iter_ky_batches(ky_values_arr, ky_batch=1, fixed_batch_shape=False)
    prev_vec: jnp.ndarray | None = None
    prev_eig: complex | None = None
    ky_slice: np.ndarray
    ky_indices: list[int]
    sel: ModeSelection | ModeSelectionBatch

    for batch_start, ky_slice, valid_count in ky_iter:
        if use_batch:
            ky_indices = [select_ky_index(np.asarray(grid_full.ky), float(ky)) for ky in ky_slice]
            grid = select_ky_grid(grid_full, ky_indices)
            sel_indices = np.arange(len(ky_indices), dtype=int)
            sel = ModeSelectionBatch(sel_indices, 0, _midplane_index(grid))
            dt_i = float(dt)
            steps_i = int(steps)
        else:
            ky_indices = [select_ky_index(np.asarray(grid_full.ky), float(ky_slice[0]))]
            grid = select_ky_grid(grid_full, ky_indices[0])
            sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
            dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
            steps_i = int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)

        charge = np.atleast_1d(np.asarray(params.charge_sign))
        ns = int(charge.size)
        electron_index = int(np.argmin(charge))
        G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
        G0_single = _build_initial_condition(
            grid,
            geom,
            ky_index=np.arange(len(ky_indices), dtype=int),
            kx_index=0,
            Nl=Nl,
            Nm=Nm,
            init_cfg=cfg.init,
        )
        G0[electron_index] = np.asarray(G0_single, dtype=np.complex64)

        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        G0_jax = jnp.asarray(G0)
        if solver_key == "krylov":
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
                        # When continuation carries an explicit previous eigenvalue
                        # as the shift, select the closest shifted branch first and
                        # let overlap tracking keep the mode family coherent.
                        shift_selection_use = "shift"
            select_overlap = use_cont and v_ref is not None and (
                cfg_use.continuation_selection.strip().lower() == "overlap"
            )
            eig, vec = dominant_eigenpair(
                v0_use,
                cache,
                params,
                terms=terms,
                v_ref=v_ref,
                select_overlap=select_overlap,
                krylov_dim=cfg_use.krylov_dim,
                restarts=cfg_use.restarts,
                omega_min_factor=cfg_use.omega_min_factor,
                omega_target_factor=cfg_use.omega_target_factor,
                omega_cap_factor=cfg_use.omega_cap_factor,
                omega_sign=cfg_use.omega_sign,
                method=cfg_use.method,
                power_iters=cfg_use.power_iters,
                power_dt=cfg_use.power_dt,
                shift=shift_override,
                shift_source=cfg_use.shift_source,
                shift_tol=cfg_use.shift_tol,
                shift_maxiter=cfg_use.shift_maxiter,
                shift_restart=cfg_use.shift_restart,
                shift_solve_method=cfg_use.shift_solve_method,
                shift_preconditioner=cfg_use.shift_preconditioner,
                shift_selection=shift_selection_use,
                mode_family=cfg_use.mode_family,
                fallback_method=cfg_use.fallback_method,
                fallback_real_floor=cfg_use.fallback_real_floor,
            )
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
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_slice[0]))
            continue

        method_key = method.lower()
        time_cfg_i = None
        if time_cfg is not None:
            time_cfg_i = replace(time_cfg, dt=dt_i, t_max=dt_i * steps_i)
            if sample_stride is not None:
                time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)

        params_use = params
        if time_cfg_i is not None and time_cfg_i.use_diffrax and streaming_fit:
            t_total = float(time_cfg_i.t_max)
            tmin_i, tmax_i = _resolve_streaming_window(
                t_total, _window_value(tmin, batch_start), _window_value(tmax, batch_start), start_fraction, window_fraction, 1.0
            )
            _, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
                G0_jax,
                grid,
                geom,
                params_use,
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
                mode_ky_indices=np.arange(valid_count, dtype=int),
                mode_kx_index=0,
                mode_z_index=_midplane_index(grid),
                mode_method=mode_method,
                amp_floor=streaming_amp_floor,
                density_species_index=electron_index if fit_key == "density" else None,
                return_state=False,
            )
            gamma_arr = np.asarray(gamma_vals)
            omega_arr = np.asarray(omega_vals)
            for local_idx in range(valid_count):
                ky_val = ky_slice[local_idx]
                gamma_i, omega_i = _normalize_growth_rate(
                    float(gamma_arr[local_idx]),
                    float(omega_arr[local_idx]),
                    params_use,
                    diagnostic_norm,
                )
                gammas.append(gamma_i)
                omegas.append(omega_i)
                ky_out.append(float(ky_val))
            continue

        if time_cfg_i is not None:
            save_field = "phi+density" if fit_key == "auto" else ("density" if fit_key == "density" else "phi")
            save_mode = None if fit_key == "auto" else (sel if (mode_only and fit_key == "phi") else None)
            _, saved = integrate_linear_from_config(
                G0_jax,
                grid,
                geom,
                params_use,
                time_cfg_i,
                cache=cache,
                terms=terms,
                save_mode=save_mode,
                mode_method=mode_method,
                save_field=save_field,
                density_species_index=electron_index if need_density else None,
            )
            if fit_key == "auto":
                phi_t, density_t = saved
            else:
                phi_t = saved
                density_t = None
            stride = time_cfg_i.sample_stride
        else:
            stride = 1 if sample_stride is None else int(sample_stride)
            if need_density:
                _diag = integrate_linear_diagnostics(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    dt=dt_i,
                    steps=steps_i,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=stride,
                    species_index=1,
                )
                phi_t = _diag[1]
                density_t = _diag[2] if len(_diag) > 2 else None
            else:
                _, phi_out_time = integrate_linear(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    dt=dt_i,
                    steps=steps_i,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=stride,
                )
                density_t = None

        phi_t_np = np.asarray(phi_t)
        density_np = None if density_t is None else np.asarray(density_t)
        if fit_key == "density" and density_np is None:
            density_np = phi_t_np
        t = np.arange(phi_t_np.shape[0]) * dt_i * stride

        def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
            if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
                return False
            if require_positive and gamma_val <= 0.0:
                return False
            return True

        for local_idx in range(valid_count):
            ky_val = ky_slice[local_idx]
            if fit_key == "auto":
                sel_local = ModeSelection(ky_index=local_idx, kx_index=0, z_index=_midplane_index(grid))
                _signal, _name, gamma, omega = _select_fit_signal_auto(
                    t,
                    phi_t_np,
                    density_np,
                    sel_local,
                    mode_method=mode_method,
                    tmin=_window_value(tmin, batch_start + local_idx),
                    tmax=_window_value(tmax, batch_start + local_idx),
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                    max_amp_fraction=max_amp_fraction,
                    window_method=window_method,
                    max_fraction=max_fraction,
                    end_fraction=end_fraction,
                    num_windows=8,
                    phase_weight=phase_weight,
                    length_weight=length_weight,
                    min_r2=min_r2,
                    late_penalty=late_penalty,
                    min_slope=min_slope,
                    min_slope_frac=min_slope_frac,
                    slope_var_weight=slope_var_weight,
                )
                gamma, omega = _normalize_growth_rate(gamma, omega, params_use, diagnostic_norm)
                if auto_solver and not _is_valid_growth(gamma, omega):
                    res = run_etg_linear(
                        ky_target=float(ky_val),
                        cfg=cfg,
                        Nl=Nl,
                        Nm=Nm,
                        dt=dt_i,
                        steps=steps_i,
                        method=method,
                        params=params,
                        solver="krylov",
                        krylov_cfg=krylov_cfg,
                        diagnostic_norm=diagnostic_norm,
                        fit_signal="phi",
                    )
                    gamma = float(res.gamma)
                    omega = float(res.omega)
                gammas.append(gamma)
                omegas.append(omega)
                ky_out.append(float(ky_val))
                continue

            if mode_only and fit_key == "phi" and phi_t_np.ndim <= 2:
                signal = _extract_mode_only_signal(phi_t_np, local_idx=local_idx)
            else:
                sel_local = ModeSelection(ky_index=local_idx, kx_index=0, z_index=_midplane_index(grid))
                signal = _select_fit_signal(
                    phi_t_np,
                    density_np,
                    sel_local,
                    fit_signal=fit_key,
                    mode_method=mode_method,
                )
            if gx_growth and fit_key == "phi":
                sel_local = ModeSelection(ky_index=local_idx, kx_index=0, z_index=_midplane_index(grid))
                gamma, omega, _gamma_t, _omega_t, _t_mid = gx_growth_rate_from_phi(
                    phi_t_np,
                    t,
                    sel_local,
                    navg_fraction=gx_navg_fraction,
                    mode_method=mode_method,
                )
                gamma, omega = _normalize_growth_rate(gamma, omega, params_use, diagnostic_norm)
            else:
                gamma, omega = _fit_signal(signal, batch_start + local_idx, dt_i, stride)
            if auto_solver and not _is_valid_growth(gamma, omega):
                res = run_etg_linear(
                    ky_target=float(ky_val),
                    cfg=cfg,
                    Nl=Nl,
                    Nm=Nm,
                    dt=dt_i,
                    steps=steps_i,
                    method=method,
                    params=params,
                    solver="krylov",
                    krylov_cfg=krylov_cfg,
                    diagnostic_norm=diagnostic_norm,
                    fit_signal="phi",
                )
                gamma = float(res.gamma)
                omega = float(res.omega)
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_val))
    return LinearScanResult(ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas))


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
) -> LinearRunResult:
    """Run a kinetic-electron ITG/TEM benchmark and extract growth rate."""

    cfg = cfg or KineticElectronBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=Kinetic_OMEGA_D_SCALE,
            omega_star_scale=Kinetic_OMEGA_STAR_SCALE,
            rho_star=Kinetic_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=Nm,
        )
    if terms is None:
        terms = LinearTerms(bpar=-1.0)

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))

    ns = 2
    if init_species_index < 0 or init_species_index >= ns:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= ns:
        raise ValueError("density_species_index out of range for kinetic species")
    G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G0_single = _build_initial_condition(
        grid,
        geom,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=cfg.init,
    )
    G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)

    G0_jax = jnp.asarray(G0)
    if solver.lower() == "krylov":
        krylov_cfg = krylov_cfg or KINETIC_KRYLOV_DEFAULT
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        eig, vec = dominant_eigenpair(
            G0_jax,
            cache,
            params,
            terms=terms,
            krylov_dim=krylov_cfg.krylov_dim,
            restarts=krylov_cfg.restarts,
            omega_min_factor=krylov_cfg.omega_min_factor,
            omega_target_factor=krylov_cfg.omega_target_factor,
            omega_cap_factor=krylov_cfg.omega_cap_factor,
            omega_sign=krylov_cfg.omega_sign,
            method=krylov_cfg.method,
            power_iters=krylov_cfg.power_iters,
            power_dt=krylov_cfg.power_dt,
            shift=krylov_cfg.shift,
            shift_source=krylov_cfg.shift_source,
            shift_tol=krylov_cfg.shift_tol,
            shift_maxiter=krylov_cfg.shift_maxiter,
            shift_restart=krylov_cfg.shift_restart,
            shift_solve_method=krylov_cfg.shift_solve_method,
            shift_preconditioner=krylov_cfg.shift_preconditioner,
            shift_selection=krylov_cfg.shift_selection,
            mode_family=krylov_cfg.mode_family,
            fallback_method=krylov_cfg.fallback_method,
            fallback_real_floor=krylov_cfg.fallback_real_floor,
        )
        term_cfg = linear_terms_to_term_config(terms)
        phi = compute_fields_cached(vec, cache, params, terms=term_cfg).phi
        phi_t_np = np.asarray(phi)[None, ...]
        t = np.array([0.0])
        gamma = float(np.real(eig))
        omega = float(-np.imag(eig))
        gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    else:
        method_key = method.lower()
        if time_cfg is not None:
            time_cfg_use = time_cfg
            if sample_stride is not None:
                time_cfg_use = replace(time_cfg, sample_stride=sample_stride)
            dt = float(time_cfg_use.dt)
            steps = int(round(time_cfg_use.t_max / time_cfg_use.dt))
            cache = build_linear_cache(grid, geom, params, Nl, Nm)
            if time_cfg_use.use_diffrax and not (
                method_key.startswith("imex") or method_key.startswith("implicit")
            ):
                save_field = "density" if fit_signal == "density" else "phi"
                _, phi_t = integrate_linear_from_config(
                    G0_jax,
                    grid,
                    geom,
                    params,
                    time_cfg_use,
                    cache=cache,
                    terms=terms,
                    save_field=save_field,
                    density_species_index=density_species_index if fit_signal == "density" else None,
                )
                density_t = phi_t if fit_signal == "density" else None
            else:
                if fit_signal == "density":
                    _diag = integrate_linear_diagnostics(
                        G0_jax,
                        grid,
                        geom,
                        params,
                        dt=dt,
                        steps=steps,
                        method=time_cfg_use.method,
                        cache=cache,
                        terms=terms,
                        sample_stride=time_cfg_use.sample_stride,
                        species_index=density_species_index,
                    )
                    phi_t = _diag[1]
                    density_t = _diag[2] if len(_diag) > 2 else None
                else:
                    _, phi_t = integrate_linear_from_config(
                        G0_jax,
                        grid,
                        geom,
                        params,
                        time_cfg_use,
                        cache=cache,
                        terms=terms,
                        density_species_index=density_species_index if fit_signal == "density" else None,
                    )
                    density_t = None
            stride = time_cfg_use.sample_stride
        else:
            stride = 1 if sample_stride is None else int(sample_stride)
            if fit_signal == "density":
                _diag = integrate_linear_diagnostics(
                    G0_jax,
                    grid,
                    geom,
                    params,
                    dt=dt,
                    steps=steps,
                    method=method,
                    terms=terms,
                    sample_stride=stride,
                    species_index=density_species_index,
                )
                phi_t = _diag[1]
                density_t = _diag[2] if len(_diag) > 2 else None
            else:
                _, phi_t = integrate_linear(
                    G0_jax,
                    grid,
                    geom,
                    params,
                    dt=dt,
                    steps=steps,
                    method=method,
                    terms=terms,
                    sample_stride=stride,
                )
                density_t = None

        phi_t_np = np.asarray(phi_t)
        t = np.arange(phi_t_np.shape[0]) * dt * stride
        density_np = None if density_t is None else np.asarray(density_t)
        signal = _select_fit_signal(
            phi_t_np,
            density_np,
            sel,
            fit_signal=fit_signal,
            mode_method=mode_method,
        )
        def _window_valid(t_arr: np.ndarray, tmin_val: float | None, tmax_val: float | None) -> bool:
            if tmin_val is None or tmax_val is None:
                return False
            mask = (t_arr >= tmin_val) & (t_arr <= tmax_val)
            return int(np.count_nonzero(mask)) >= 2

        use_auto = auto_window and tmin is None and tmax is None
        if not use_auto and not _window_valid(t, tmin, tmax):
            use_auto = True
        if use_auto:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                t,
                signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
        else:
            try:
                gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)
            except ValueError:
                gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                    t,
                    signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )

        gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    return LinearRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
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
) -> LinearScanResult:
    """Run a kinetic-electron ITG/TEM benchmark for a list of ky values.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    cfg = cfg or KineticElectronBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=Kinetic_OMEGA_D_SCALE,
            omega_star_scale=Kinetic_OMEGA_STAR_SCALE,
            rho_star=Kinetic_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=Nm,
        )
    if terms is None:
        terms = LinearTerms()
    gammas = []
    omegas = []
    ky_out = []
    def _window_value(val, idx):
        if val is None:
            return None
        if isinstance(val, (list, tuple, np.ndarray)):
            return float(val[idx])
        return float(val)

    def _window_valid(t_arr, tmin_val, tmax_val):
        if tmin_val is None or tmax_val is None:
            return False
        mask = (t_arr >= tmin_val) & (t_arr <= tmax_val)
        return int(np.count_nonzero(mask)) >= 2

    if mode_only and mode_method not in {"z_index", "max"}:
        mode_method = "z_index"

    if ky_batch < 1:
        raise ValueError("ky_batch must be >= 1")
    use_batch = (
        ky_batch > 1
        and solver.lower() != "krylov"
        and not _is_array_like(dt)
        and not _is_array_like(steps)
        and not _is_array_like(tmin)
        and not _is_array_like(tmax)
    )

    def _fit_signal(signal: np.ndarray, idx: int, dt_i: float, stride: int) -> tuple[float, float]:
        t = np.arange(signal.shape[0]) * dt_i * stride
        tmin_i = _window_value(tmin, idx)
        tmax_i = _window_value(tmax, idx)
        use_auto = auto_window and tmin_i is None and tmax_i is None
        if not use_auto and not _window_valid(t, tmin_i, tmax_i):
            use_auto = True
        if use_auto:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                t,
                signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
        else:
            try:
                gamma, omega = fit_growth_rate(t, signal, tmin=tmin_i, tmax=tmax_i)
            except ValueError:
                gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                    t,
                    signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
        return _normalize_growth_rate(gamma, omega, params, diagnostic_norm)

    ky_values_arr = np.asarray(ky_values, dtype=float)
    if use_batch:
        ky_iter = _iter_ky_batches(
            ky_values_arr,
            ky_batch=ky_batch,
            fixed_batch_shape=fixed_batch_shape,
        )
    else:
        ky_iter = _iter_ky_batches(ky_values_arr, ky_batch=1, fixed_batch_shape=False)
    ky_slice: np.ndarray
    ky_indices: list[int]
    sel: ModeSelection | ModeSelectionBatch
    if init_species_index < 0 or init_species_index >= 2:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= 2:
        raise ValueError("density_species_index out of range for kinetic species")

    for batch_start, ky_slice, valid_count in ky_iter:
        if use_batch:
            ky_indices = [select_ky_index(np.asarray(grid_full.ky), float(ky)) for ky in ky_slice]
            grid = select_ky_grid(grid_full, ky_indices)
            sel_indices = np.arange(len(ky_indices), dtype=int)
            sel = ModeSelectionBatch(sel_indices, 0, _midplane_index(grid))
            dt_i = float(dt)
            steps_i = int(steps)
        else:
            ky_indices = [select_ky_index(np.asarray(grid_full.ky), float(ky_slice[0]))]
            grid = select_ky_grid(grid_full, ky_indices[0])
            sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
            dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
            steps_i = int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)

        ns = 2
        G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
        G0_single = _build_initial_condition(
            grid,
            geom,
            ky_index=np.arange(len(ky_indices), dtype=int),
            kx_index=0,
            Nl=Nl,
            Nm=Nm,
            init_cfg=cfg.init,
        )
        G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)

        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        G0_jax = jnp.asarray(G0)
        if solver.lower() == "krylov":
            cfg_use = krylov_cfg or KINETIC_KRYLOV_DEFAULT
            eig, _vec = dominant_eigenpair(
                G0_jax,
                cache,
                params,
                terms=terms,
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
            gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_slice[0]))
            continue

        method_key = method.lower()
        time_cfg_i = None
        if time_cfg is not None:
            time_cfg_i = replace(time_cfg, dt=dt_i, t_max=dt_i * steps_i)
            if sample_stride is not None:
                time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)

        params_use = params
        if time_cfg_i is not None and time_cfg_i.use_diffrax and streaming_fit:
            t_total = float(time_cfg_i.t_max)
            tmin_i, tmax_i = _resolve_streaming_window(
                t_total, _window_value(tmin, batch_start), _window_value(tmax, batch_start), start_fraction, window_fraction, 1.0
            )
            _, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
                G0_jax,
                grid,
                geom,
                params_use,
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
                fit_signal=fit_signal,
                mode_ky_indices=np.arange(valid_count, dtype=int),
                mode_kx_index=0,
                mode_z_index=_midplane_index(grid),
                mode_method=mode_method,
                amp_floor=streaming_amp_floor,
                density_species_index=density_species_index if fit_signal == "density" else None,
                return_state=False,
            )
            gamma_arr = np.asarray(gamma_vals)
            omega_arr = np.asarray(omega_vals)
            for local_idx in range(valid_count):
                ky_val = ky_slice[local_idx]
                gamma_i, omega_i = _normalize_growth_rate(
                    float(gamma_arr[local_idx]),
                    float(omega_arr[local_idx]),
                    params_use,
                    diagnostic_norm,
                )
                gammas.append(gamma_i)
                omegas.append(omega_i)
                ky_out.append(float(ky_val))
            continue

        if time_cfg_i is not None:
            save_mode_method = mode_method if mode_method in {"z_index", "max"} else "z_index"
            _, phi_t = integrate_linear_from_config(
                G0_jax,
                grid,
                geom,
                params_use,
                time_cfg_i,
                cache=cache,
                terms=terms,
                save_mode=sel if (mode_only and fit_signal == "phi") else None,
                mode_method=mode_method,
                save_field="density" if fit_signal == "density" else "phi",
                density_species_index=density_species_index if fit_signal == "density" else None,
            )
            stride = time_cfg_i.sample_stride
            density_t = None
        else:
            stride = 1 if sample_stride is None else int(sample_stride)
            if fit_signal == "density":
                _diag = integrate_linear_diagnostics(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    dt=dt_i,
                    steps=steps_i,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=stride,
                    species_index=density_species_index,
                )
                phi_t = _diag[1]
                density_t = _diag[2] if len(_diag) > 2 else None
            else:
                _, phi_t = integrate_linear(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    dt=dt_i,
                    steps=steps_i,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=stride,
                )
                density_t = None

        phi_t_np = np.asarray(phi_t)
        density_np = None if density_t is None else np.asarray(density_t)
        if fit_signal == "density" and density_np is None:
            density_np = phi_t_np
        for local_idx in range(valid_count):
            ky_val = ky_slice[local_idx]
            if mode_only and fit_signal == "phi" and phi_t_np.ndim <= 2:
                signal = _extract_mode_only_signal(phi_t_np, local_idx=local_idx)
            elif mode_only and fit_signal == "density" and density_np is not None and density_np.ndim <= 3:
                signal = _extract_mode_only_signal(
                    density_np,
                    local_idx=local_idx,
                    species_index=density_species_index,
                )
            else:
                sel_local = ModeSelection(ky_index=local_idx, kx_index=0, z_index=_midplane_index(grid))
                signal = _select_fit_signal(
                    phi_t_np,
                    density_np,
                    sel_local,
                    fit_signal=fit_signal,
                    mode_method=mode_method,
                )
            gamma, omega = _fit_signal(signal, batch_start + local_idx, dt_i, stride)
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_val))
    return LinearScanResult(ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas))


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
) -> LinearRunResult:
    """Run the TEM benchmark and extract growth rate."""

    cfg = cfg or TEMBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=TEM_OMEGA_D_SCALE,
            omega_star_scale=TEM_OMEGA_STAR_SCALE,
            rho_star=TEM_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=Nm,
        )
    if terms is None:
        terms = LinearTerms()

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))

    ns = 2
    if init_species_index < 0 or init_species_index >= ns:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= ns:
        raise ValueError("density_species_index out of range for kinetic species")
    G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G0_single = _build_initial_condition(
        grid,
        geom,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=cfg.init,
    )
    G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)

    G0_jax = jnp.asarray(G0)
    if solver.lower() == "krylov":
        krylov_cfg = krylov_cfg or TEM_KRYLOV_DEFAULT
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        eig, vec = dominant_eigenpair(
            G0_jax,
            cache,
            params,
            terms=terms,
            krylov_dim=krylov_cfg.krylov_dim,
            restarts=krylov_cfg.restarts,
            omega_min_factor=krylov_cfg.omega_min_factor,
            omega_target_factor=krylov_cfg.omega_target_factor,
            omega_cap_factor=krylov_cfg.omega_cap_factor,
            omega_sign=krylov_cfg.omega_sign,
            method=krylov_cfg.method,
            power_iters=krylov_cfg.power_iters,
            power_dt=krylov_cfg.power_dt,
            shift=krylov_cfg.shift,
            shift_source=krylov_cfg.shift_source,
            shift_tol=krylov_cfg.shift_tol,
            shift_maxiter=krylov_cfg.shift_maxiter,
            shift_restart=krylov_cfg.shift_restart,
            shift_solve_method=krylov_cfg.shift_solve_method,
            shift_preconditioner=krylov_cfg.shift_preconditioner,
            shift_selection=krylov_cfg.shift_selection,
            mode_family=krylov_cfg.mode_family,
            fallback_method=krylov_cfg.fallback_method,
            fallback_real_floor=krylov_cfg.fallback_real_floor,
        )
        term_cfg = linear_terms_to_term_config(terms)
        phi = compute_fields_cached(vec, cache, params, terms=term_cfg).phi
        phi_t_np = np.asarray(phi)[None, ...]
        t = np.array([0.0])
        gamma = float(np.real(eig))
        omega = float(-np.imag(eig))
        gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
    else:
        if fit_signal not in {"phi", "density"}:
            raise ValueError("fit_signal must be 'phi' or 'density'")
        if time_cfg is not None:
            time_cfg_use = time_cfg
            if sample_stride is not None:
                time_cfg_use = replace(time_cfg, sample_stride=sample_stride)
            dt = float(time_cfg_use.dt)
            steps = int(round(time_cfg_use.t_max / time_cfg_use.dt))
            cache = build_linear_cache(grid, geom, params, Nl, Nm)
            if fit_signal == "density":
                _diag = integrate_linear_diagnostics(
                    G0_jax,
                    grid,
                    geom,
                    params,
                    dt=dt,
                    steps=steps,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=time_cfg_use.sample_stride,
                    species_index=density_species_index,
                )
                if len(_diag) == 4:
                    _, phi_t, density_t, _ = _diag
                else:
                    _, phi_t, density_t = _diag
            else:
                _, phi_t = integrate_linear_from_config(
                    G0_jax,
                    grid,
                    geom,
                    params,
                    time_cfg_use,
                    cache=cache,
                    terms=terms,
                )
                density_t = None
            stride = time_cfg_use.sample_stride
        else:
            stride = 1 if sample_stride is None else int(sample_stride)
            cache = build_linear_cache(grid, geom, params, Nl, Nm)
            if fit_signal == "density":
                _diag = integrate_linear_diagnostics(
                    G0_jax,
                    grid,
                    geom,
                    params,
                    dt=dt,
                    steps=steps,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=stride,
                    species_index=density_species_index,
                )
                if len(_diag) == 4:
                    _, phi_t, density_t, _ = _diag
                else:
                    _, phi_t, density_t = _diag
            else:
                _, phi_t = integrate_linear(
                    G0_jax,
                    grid,
                    geom,
                    params,
                    dt=dt,
                    steps=steps,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=stride,
                )
                density_t = None

        phi_t_np = np.asarray(phi_t)
        t = np.arange(phi_t_np.shape[0]) * dt * stride
        if fit_signal == "density" and density_t is not None:
            density_t_np = np.asarray(density_t)
            signal = extract_mode_time_series(density_t_np, sel, method=mode_method)
        else:
            signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
        if auto_window and tmin is None and tmax is None:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                t,
                signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
        else:
            try:
                gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)
            except ValueError:
                gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                    t,
                    signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
        gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)

    return LinearRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )


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
) -> LinearScanResult:
    """Run the TEM benchmark for a list of ky values.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    cfg = cfg or TEMBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if params is None:
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=TEM_OMEGA_D_SCALE,
            omega_star_scale=TEM_OMEGA_STAR_SCALE,
            rho_star=TEM_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=Nm,
        )
    if terms is None:
        terms = LinearTerms(bpar=0.0)
    gammas = []
    omegas = []
    ky_out = []
    def _window_value(val, idx):
        if val is None:
            return None
        if isinstance(val, (list, tuple, np.ndarray)):
            return float(val[idx])
        return float(val)

    def _window_valid(t_arr, tmin_val, tmax_val):
        if tmin_val is None or tmax_val is None:
            return False
        mask = (t_arr >= tmin_val) & (t_arr <= tmax_val)
        return int(np.count_nonzero(mask)) >= 2

    if mode_only and mode_method not in {"z_index", "max"}:
        mode_method = "z_index"

    if ky_batch < 1:
        raise ValueError("ky_batch must be >= 1")
    use_batch = (
        ky_batch > 1
        and solver.lower() != "krylov"
        and not _is_array_like(dt)
        and not _is_array_like(steps)
        and not _is_array_like(tmin)
        and not _is_array_like(tmax)
    )

    def _fit_signal(signal: np.ndarray, idx: int, dt_i: float, stride: int) -> tuple[float, float]:
        t = np.arange(signal.shape[0]) * dt_i * stride
        tmin_i = _window_value(tmin, idx)
        tmax_i = _window_value(tmax, idx)
        use_auto = auto_window and tmin_i is None and tmax_i is None
        if not use_auto and not _window_valid(t, tmin_i, tmax_i):
            use_auto = True
        if use_auto:
            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                t,
                signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
        else:
            try:
                gamma, omega = fit_growth_rate(t, signal, tmin=tmin_i, tmax=tmax_i)
            except ValueError:
                gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                    t,
                    signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
        return _normalize_growth_rate(gamma, omega, params, diagnostic_norm)

    ky_values_arr = np.asarray(ky_values, dtype=float)
    if use_batch:
        ky_iter = _iter_ky_batches(
            ky_values_arr,
            ky_batch=ky_batch,
            fixed_batch_shape=fixed_batch_shape,
        )
    else:
        ky_iter = _iter_ky_batches(ky_values_arr, ky_batch=1, fixed_batch_shape=False)
    ky_slice: np.ndarray
    ky_indices: list[int]
    sel: ModeSelection | ModeSelectionBatch

    if init_species_index < 0 or init_species_index >= 2:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= 2:
        raise ValueError("density_species_index out of range for kinetic species")

    for batch_start, ky_slice, valid_count in ky_iter:
        if use_batch:
            ky_indices = [select_ky_index(np.asarray(grid_full.ky), float(ky)) for ky in ky_slice]
            grid = select_ky_grid(grid_full, ky_indices)
            sel_indices = np.arange(len(ky_indices), dtype=int)
            sel = ModeSelectionBatch(sel_indices, 0, _midplane_index(grid))
            dt_i = float(dt)
            steps_i = int(steps)
        else:
            ky_indices = [select_ky_index(np.asarray(grid_full.ky), float(ky_slice[0]))]
            grid = select_ky_grid(grid_full, ky_indices[0])
            sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
            dt_i = float(dt[batch_start]) if isinstance(dt, np.ndarray) else float(dt)
            steps_i = int(steps[batch_start]) if isinstance(steps, np.ndarray) else int(steps)

        ns = 2
        G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
        G0_single = _build_initial_condition(
            grid,
            geom,
            ky_index=np.arange(len(ky_indices), dtype=int),
            kx_index=0,
            Nl=Nl,
            Nm=Nm,
            init_cfg=cfg.init,
        )
        G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)

        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        G0_jax = jnp.asarray(G0)
        if solver.lower() == "krylov":
            cfg_use = krylov_cfg or TEM_KRYLOV_DEFAULT
            eig, _vec = dominant_eigenpair(
                G0_jax,
                cache,
                params,
                terms=terms,
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
            gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_slice[0]))
            continue

        method_key = method.lower()
        time_cfg_i = None
        if time_cfg is not None:
            time_cfg_i = replace(time_cfg, dt=dt_i, t_max=dt_i * steps_i)
            if sample_stride is not None:
                time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)

        params_use = params
        if time_cfg_i is not None and time_cfg_i.use_diffrax and streaming_fit:
            t_total = float(time_cfg_i.t_max)
            tmin_i, tmax_i = _resolve_streaming_window(
                t_total, _window_value(tmin, batch_start), _window_value(tmax, batch_start), start_fraction, window_fraction, 1.0
            )
            _, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
                G0_jax,
                grid,
                geom,
                params_use,
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
                fit_signal="phi",
                mode_ky_indices=np.arange(valid_count, dtype=int),
                mode_kx_index=0,
                mode_z_index=_midplane_index(grid),
                mode_method=mode_method,
                amp_floor=streaming_amp_floor,
                return_state=False,
            )
            gamma_arr = np.asarray(gamma_vals)
            omega_arr = np.asarray(omega_vals)
            for local_idx in range(valid_count):
                ky_val = ky_slice[local_idx]
                gammas.append(float(gamma_arr[local_idx]))
                omegas.append(float(omega_arr[local_idx]))
                ky_out.append(float(ky_val))
            continue

        if time_cfg_i is not None:
            _, phi_t = integrate_linear_from_config(
                G0_jax,
                grid,
                geom,
                params_use,
                time_cfg_i,
                cache=cache,
                terms=terms,
                save_mode=sel if mode_only else None,
                mode_method=mode_method,
            )
            stride = time_cfg_i.sample_stride
        else:
            stride = 1 if sample_stride is None else int(sample_stride)
            _, phi_t = integrate_linear(
                G0_jax,
                grid,
                geom,
                params_use,
                dt=dt_i,
                steps=steps_i,
                method=method,
                cache=cache,
                terms=terms,
                sample_stride=stride,
            )

        phi_t_np = np.asarray(phi_t)
        for local_idx in range(valid_count):
            ky_val = ky_slice[local_idx]
            if mode_only and phi_t_np.ndim <= 2:
                signal = _extract_mode_only_signal(phi_t_np, local_idx=local_idx)
            else:
                sel_local = ModeSelection(ky_index=local_idx, kx_index=0, z_index=_midplane_index(grid))
                signal = extract_mode_time_series(phi_t_np, sel_local, method=mode_method)
            gamma, omega = _fit_signal(signal, batch_start + local_idx, dt_i, stride)
            gammas.append(gamma)
            omegas.append(omega)
            ky_out.append(float(ky_val))
    return LinearScanResult(ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas))


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
    gx_reference: bool | None = True,
) -> LinearScanResult:
    """Run a KBM beta scan at fixed ky.

    If ``time_cfg`` is provided, its ``dt`` and ``t_max`` override ``dt``/``steps``.
    """

    cfg = cfg or KBMBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if terms is None:
        terms = LinearTerms(bpar=0.0)
    gx_reference_use = bool(gx_reference)
    if gx_reference_use and diagnostic_norm == "none":
        diagnostic_norm = "gx"
    damp_ends_amp, damp_ends_widthfrac = _gx_linked_end_damping(gx_reference_use)

    solver_key = solver.strip().lower()
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    if fit_key == "auto":
        streaming_fit = False
        mode_only = False

    krylov_cfg_use = krylov_cfg or KBM_KRYLOV_DEFAULT
    use_continuation = bool(getattr(krylov_cfg_use, "continuation", False))
    prev_vec = None
    prev_eig = None

    gammas = []
    omegas = []
    beta_out = []
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))

    def _window_value(val, idx):
        if val is None:
            return None
        if isinstance(val, (list, tuple, np.ndarray)):
            return float(val[idx])
        return float(val)

    def _window_valid(t_arr, tmin_val, tmax_val):
        if tmin_val is None or tmax_val is None:
            return False
        mask = (t_arr >= tmin_val) & (t_arr <= tmax_val)
        return int(np.count_nonzero(mask)) >= 2

    if init_species_index < 0 or init_species_index >= 2:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= 2:
        raise ValueError("density_species_index out of range for kinetic species")

    def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
        if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
            return False
        if require_positive and gamma_val <= 0.0:
            return False
        return True

    for i, beta in enumerate(betas):
        dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KBM_OMEGA_D_SCALE,
            omega_star_scale=KBM_OMEGA_STAR_SCALE,
            rho_star=KBM_RHO_STAR,
            beta_override=float(beta),
            fapar_override=fapar_override,
            apar_beta_scale=apar_beta_scale,
            ampere_g0_scale=ampere_g0_scale,
            bpar_beta_scale=bpar_beta_scale,
            damp_ends_amp=damp_ends_amp,
            damp_ends_widthfrac=damp_ends_widthfrac,
            nhermite=Nm,
        )
        cache = build_linear_cache(grid, geom, params, Nl, Nm)

        ns = 2
        G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
        G0_single = _build_initial_condition(
            grid,
            geom,
            ky_index=sel.ky_index,
            kx_index=sel.kx_index,
            Nl=Nl,
            Nm=Nm,
            init_cfg=cfg.init,
        )
        G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)

        G0_jax = jnp.asarray(G0)
        solver_use = select_kbm_solver_auto(solver_key, ky_target=ky_target, gx_reference=gx_reference_use)

        if solver_use == "gx_time":
            gx_mode_method = mode_method if mode_method in {"z_index", "max"} else "z_index"
            gx_time_cfg = GXTimeConfig(
                dt=dt_i,
                t_max=dt_i * steps_i,
                sample_stride=max(int(sample_stride or 1), 1),
                fixed_dt=bool(time_cfg.fixed_dt) if time_cfg is not None else False,
                use_dealias_mask=bool(getattr(time_cfg, "use_dealias_mask", False))
                if time_cfg is not None
                else False,
                dt_min=float(time_cfg.dt_min) if time_cfg is not None else 1.0e-7,
                dt_max=float(time_cfg.dt_max) if (time_cfg is not None and time_cfg.dt_max is not None) else None,
                cfl=float(time_cfg.cfl) if time_cfg is not None else 0.9,
                cfl_fac=(
                    resolve_cfl_fac(str(time_cfg.method), time_cfg.cfl_fac)
                    if time_cfg is not None
                    else float(GXTimeConfig.cfl_fac)
                ),
            )
            t_arr, _phi_t, gamma_t, omega_t, _gx_diag = integrate_linear_gx_diagnostics(
                G0_jax,
                grid,
                cache,
                params,
                geom,
                gx_time_cfg,
                terms=terms,
                mode_method=gx_mode_method,
                z_index=sel.z_index,
                jit=True,
            )
            if t_arr.size > 1:
                phi_np = np.asarray(_phi_t)
                t_np = np.asarray(t_arr, dtype=float)
                if mode_method in {"z_index", "max"}:
                    try:
                        gamma, omega, _g_t, _o_t, _t_mid = gx_growth_rate_from_phi(
                            phi_np,
                            t_np,
                            sel,
                            navg_fraction=0.5,
                            mode_method=mode_method,
                        )
                    except ValueError:
                        try:
                            gamma, omega, _g_t, _o_t = gx_growth_rate_from_omega_series(
                                np.asarray(gamma_t),
                                np.asarray(omega_t),
                                sel,
                                navg_fraction=0.5,
                            )
                        except ValueError:
                            signal = extract_mode_time_series(phi_np, sel, method=mode_method)
                            gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                                t_np,
                                signal,
                                window_method="fixed",
                                window_fraction=window_fraction,
                                min_points=min_points,
                                start_fraction=start_fraction,
                                growth_weight=growth_weight,
                                require_positive=require_positive,
                                min_amp_fraction=min_amp_fraction,
                            )
                else:
                    signal = extract_mode_time_series(phi_np, sel, method=mode_method)
                    gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                        t_np,
                        signal,
                        window_method="fixed",
                        window_fraction=window_fraction,
                        min_points=min_points,
                        start_fraction=start_fraction,
                        growth_weight=growth_weight,
                        require_positive=require_positive,
                        min_amp_fraction=min_amp_fraction,
                    )
            else:
                gamma = float("nan")
                omega = float("nan")
            gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
            gamma_out = gamma
            omega_out = omega
        elif solver_use == "krylov":
            shift_val = krylov_cfg_use.shift
            shift_selection = krylov_cfg_use.shift_selection
            if use_continuation and prev_eig is not None:
                shift_val = complex(np.asarray(prev_eig))

            targets: Sequence[float] | None = kbm_target_factors if kbm_target_factors else None
            use_multi_target = _kbm_use_multi_target_krylov(
                krylov_cfg_use,
                targets,
                shift=shift_val,
            )
            if use_multi_target:
                assert targets is not None
                beta_transition = (
                    float(cfg.model.beta)
                    if kbm_beta_transition is None
                    else float(kbm_beta_transition)
                )
                eig_candidates = []
                vec_candidates = []
                for target in targets:
                    eig_i, vec_i = dominant_eigenpair(
                        G0_jax,
                        cache,
                        params,
                        terms=terms,
                        v_ref=None,
                        select_overlap=False,
                        krylov_dim=krylov_cfg_use.krylov_dim,
                        restarts=krylov_cfg_use.restarts,
                        omega_min_factor=krylov_cfg_use.omega_min_factor,
                        omega_target_factor=float(target),
                        omega_cap_factor=krylov_cfg_use.omega_cap_factor,
                        omega_sign=krylov_cfg_use.omega_sign,
                        method=krylov_cfg_use.method,
                        power_iters=krylov_cfg_use.power_iters,
                        power_dt=krylov_cfg_use.power_dt,
                        shift=None,
                        shift_source="target",
                        shift_tol=krylov_cfg_use.shift_tol,
                        shift_maxiter=krylov_cfg_use.shift_maxiter,
                        shift_restart=krylov_cfg_use.shift_restart,
                        shift_solve_method=krylov_cfg_use.shift_solve_method,
                        shift_preconditioner=krylov_cfg_use.shift_preconditioner,
                        shift_selection="targeted",
                        mode_family=krylov_cfg_use.mode_family,
                        fallback_method=krylov_cfg_use.fallback_method,
                        fallback_real_floor=krylov_cfg_use.fallback_real_floor,
                    )
                    eig_candidates.append(eig_i)
                    vec_candidates.append(vec_i)
                if len(eig_candidates) >= 2 and np.isfinite(beta_transition):
                    pick_high = float(beta) >= beta_transition
                    idx = 1 if pick_high else 0
                    eig = eig_candidates[idx]
                    _vec = vec_candidates[idx]
                else:
                    eig_arr = np.asarray([complex(np.asarray(e)) for e in eig_candidates])
                    growth = np.real(eig_arr)
                    if np.all(~np.isfinite(growth)):
                        eig = eig_candidates[0]
                        _vec = vec_candidates[0]
                    else:
                        idx = int(np.nanargmax(np.where(np.isfinite(growth), growth, -np.inf)))
                        eig = eig_candidates[idx]
                        _vec = vec_candidates[idx]
            else:
                eig, _vec = dominant_eigenpair(
                    G0_jax,
                    cache,
                    params,
                    terms=terms,
                    v_ref=prev_vec,
                    select_overlap=use_continuation,
                    krylov_dim=krylov_cfg_use.krylov_dim,
                    restarts=krylov_cfg_use.restarts,
                    omega_min_factor=krylov_cfg_use.omega_min_factor,
                    omega_target_factor=krylov_cfg_use.omega_target_factor,
                    omega_cap_factor=krylov_cfg_use.omega_cap_factor,
                    omega_sign=krylov_cfg_use.omega_sign,
                    method=krylov_cfg_use.method,
                    power_iters=krylov_cfg_use.power_iters,
                    power_dt=krylov_cfg_use.power_dt,
                    shift=shift_val,
                    shift_source=krylov_cfg_use.shift_source,
                    shift_tol=krylov_cfg_use.shift_tol,
                    shift_maxiter=krylov_cfg_use.shift_maxiter,
                    shift_restart=krylov_cfg_use.shift_restart,
                    shift_solve_method=krylov_cfg_use.shift_solve_method,
                    shift_preconditioner=krylov_cfg_use.shift_preconditioner,
                    shift_selection=shift_selection,
                    mode_family=krylov_cfg_use.mode_family,
                    fallback_method=krylov_cfg_use.fallback_method,
                    fallback_real_floor=krylov_cfg_use.fallback_real_floor,
                )
            gamma = float(np.real(eig))
            omega = float(-np.imag(eig))
            if krylov_cfg_use.omega_sign != 0:
                omega = float(np.sign(krylov_cfg_use.omega_sign)) * abs(omega)
            gamma, omega = _normalize_growth_rate(gamma, omega, params, diagnostic_norm)
            if solver_key == "auto" and not _is_valid_growth(gamma, omega):
                solver_use = "time"
            elif use_continuation:
                prev_vec = _vec
                prev_eig = eig

        if solver_use not in {"krylov", "gx_time"}:
            method_key = method.lower()
            time_cfg_i = None
            if time_cfg is not None:
                time_cfg_i = replace(time_cfg, dt=dt_i, t_max=dt_i * steps_i)
                if sample_stride is not None:
                    time_cfg_i = replace(time_cfg_i, sample_stride=sample_stride)

            params_use = params
            if time_cfg_i is not None and time_cfg_i.use_diffrax and streaming_fit:
                t_total = float(time_cfg_i.t_max)
                tmin_i, tmax_i = _resolve_streaming_window(
                    t_total, _window_value(tmin, i), _window_value(tmax, i), start_fraction, window_fraction, 1.0
                )
                _, gamma_vals, omega_vals = integrate_linear_diffrax_streaming(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
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
                    mode_z_index=_midplane_index(grid),
                    mode_method=mode_method,
                    amp_floor=streaming_amp_floor,
                    density_species_index=density_species_index if fit_key == "density" else None,
                    return_state=False,
                )
                gamma = float(np.asarray(gamma_vals)[0])
                omega = float(np.asarray(omega_vals)[0])
                gamma, omega = _normalize_growth_rate(gamma, omega, params_use, diagnostic_norm)
            else:
                if time_cfg_i is not None:
                    stride = time_cfg_i.sample_stride
                    if time_cfg_i.use_diffrax:
                        save_mode_method = mode_method if mode_method in {"z_index", "max"} else "z_index"
                        _, phi_t = integrate_linear_from_config(
                            G0_jax,
                            grid,
                            geom,
                            params_use,
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
                            phi_t, density_t = phi_t
                        else:
                            density_t = None
                    else:
                        if fit_key in {"density", "auto"}:
                            diag_out = integrate_linear_diagnostics(
                                G0_jax,
                                grid,
                                geom,
                                params_use,
                                dt=dt_i,
                                steps=steps_i,
                                method=method,
                                cache=cache,
                                terms=terms,
                                sample_stride=stride,
                                species_index=density_species_index,
                            )
                            phi_t = diag_out[1]
                            density_t = diag_out[2] if len(diag_out) > 2 else None
                        else:
                            _, phi_t = integrate_linear(
                                G0_jax,
                                grid,
                                geom,
                                params_use,
                                dt=dt_i,
                                steps=steps_i,
                                method=method,
                                cache=cache,
                                terms=terms,
                                sample_stride=stride,
                            )
                            density_t = None
                else:
                    stride = 1 if sample_stride is None else int(sample_stride)
                    if fit_key in {"density", "auto"}:
                        diag_out = integrate_linear_diagnostics(
                            G0_jax,
                            grid,
                            geom,
                            params_use,
                            dt=dt_i,
                            steps=steps_i,
                            method=method,
                            cache=cache,
                            terms=terms,
                            sample_stride=stride,
                            species_index=density_species_index,
                        )
                        phi_t = diag_out[1]
                        density_t = diag_out[2] if len(diag_out) > 2 else None
                    else:
                        _, phi_t = integrate_linear(
                            G0_jax,
                            grid,
                            geom,
                            params_use,
                            dt=dt_i,
                            steps=steps_i,
                            method=method,
                            cache=cache,
                            terms=terms,
                            sample_stride=stride,
                        )
                        density_t = None

                phi_t_np = np.asarray(phi_t)
                density_np = None if density_t is None else np.asarray(density_t)
                if fit_key == "density" and density_np is None:
                    density_np = phi_t_np
                if fit_key == "auto":
                    signal, _name, gamma, omega = _select_fit_signal_auto(
                        np.arange(phi_t_np.shape[0]) * dt_i * stride,
                        phi_t_np,
                        density_np,
                        sel,
                        mode_method=mode_method,
                        tmin=_window_value(tmin, i),
                        tmax=_window_value(tmax, i),
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
                    gamma, omega = _normalize_growth_rate(gamma, omega, params_use, diagnostic_norm)
                    gammas.append(gamma)
                    omegas.append(omega)
                    beta_out.append(float(beta))
                    continue

                if mode_only and fit_key == "density" and density_np is not None and density_np.ndim <= 3:
                    signal = _extract_mode_only_signal(
                        density_np,
                        local_idx=0,
                        species_index=density_species_index,
                    )
                elif mode_only and phi_t_np.ndim <= 2:
                    signal = _extract_mode_only_signal(phi_t_np, local_idx=0)
                else:
                    signal = _select_fit_signal(
                        phi_t_np,
                        density_np,
                        sel,
                        fit_signal=fit_key,
                        mode_method=mode_method,
                    )
                t = np.arange(signal.shape[0]) * dt_i * stride
                tmin_i = _window_value(tmin, i)
                tmax_i = _window_value(tmax, i)
                use_auto = auto_window and tmin_i is None and tmax_i is None
                if not use_auto and not _window_valid(t, tmin_i, tmax_i):
                    use_auto = True
                if use_auto:
                    gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                        t,
                        signal,
                        window_fraction=window_fraction,
                        min_points=min_points,
                        start_fraction=start_fraction,
                        growth_weight=growth_weight,
                        require_positive=require_positive,
                        min_amp_fraction=min_amp_fraction,
                    )
                else:
                    try:
                        gamma, omega = fit_growth_rate(t, signal, tmin=tmin_i, tmax=tmax_i)
                    except ValueError:
                        gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                            t,
                            signal,
                            window_fraction=window_fraction,
                            min_points=min_points,
                            start_fraction=start_fraction,
                            growth_weight=growth_weight,
                            require_positive=require_positive,
                            min_amp_fraction=min_amp_fraction,
                        )
                gamma, omega = _normalize_growth_rate(gamma, omega, params_use, diagnostic_norm)

        gammas.append(gamma)
        omegas.append(omega)
        beta_out.append(float(beta))

    return LinearScanResult(ky=np.array(beta_out), gamma=np.array(gammas), omega=np.array(omegas))


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
    gx_reference: bool | None = True,
) -> LinearRunResult:
    """Run a single linear KBM point and return the stored field history."""

    cfg_in = cfg or KBMBaseCase()
    beta_use = float(cfg_in.model.beta) if beta_value is None else float(beta_value)
    cfg_use = replace(cfg_in, model=replace(cfg_in.model, beta=beta_use))
    geom = build_flux_tube_geometry(cfg_use.geometry)
    grid_full = build_spectral_grid(apply_gx_geometry_grid_defaults(geom, cfg_use.grid))
    if terms is None:
        terms = LinearTerms(bpar=0.0)
    gx_reference_use = bool(gx_reference)
    if gx_reference_use and diagnostic_norm == "none":
        diagnostic_norm = "gx"
    damp_ends_amp, damp_ends_widthfrac = _gx_linked_end_damping(gx_reference_use)

    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    if fit_key == "auto":
        streaming_fit = False

    if init_species_index < 0 or init_species_index >= 2:
        raise ValueError("init_species_index out of range for kinetic species")
    if density_species_index < 0 or density_species_index >= 2:
        raise ValueError("density_species_index out of range for kinetic species")

    if params is None:
        params = _two_species_params(
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

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    G0 = np.zeros((2, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G0_single = _build_initial_condition(
        grid,
        geom,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl,
        Nm=Nm,
        init_cfg=cfg_use.init,
    )
    G0[int(init_species_index)] = np.asarray(G0_single, dtype=np.complex64)
    G0_jax = jnp.asarray(G0)

    solver_key = select_kbm_solver_auto(
        solver,
        ky_target=float(ky_target),
        gx_reference=gx_reference_use,
    )
    krylov_cfg_use = krylov_cfg or KBM_KRYLOV_DEFAULT

    def _window_valid(t_arr: np.ndarray, tmin_val: float | None, tmax_val: float | None) -> bool:
        if tmin_val is None or tmax_val is None:
            return False
        mask = (t_arr >= tmin_val) & (t_arr <= tmax_val)
        return int(np.count_nonzero(mask)) >= 2

    def _fit_with_window(signal: np.ndarray, t_arr: np.ndarray) -> tuple[float, float]:
        use_auto = auto_window and tmin is None and tmax is None
        if not use_auto and not _window_valid(t_arr, tmin, tmax):
            use_auto = True
        if use_auto:
            gamma_val, omega_val, _tmin, _tmax = fit_growth_rate_auto(
                t_arr,
                signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
        else:
            try:
                gamma_val, omega_val = fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)
            except ValueError:
                gamma_val, omega_val, _tmin, _tmax = fit_growth_rate_auto(
                    t_arr,
                    signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
        return gamma_val, omega_val

    if solver_key == "gx_time":
        gx_mode_method = mode_method if mode_method in {"z_index", "max"} else "z_index"
        gx_time_cfg = GXTimeConfig(
            dt=dt,
            t_max=dt * steps,
            sample_stride=max(int(sample_stride or 1), 1),
            fixed_dt=bool(time_cfg.fixed_dt) if time_cfg is not None else False,
            use_dealias_mask=bool(getattr(time_cfg, "use_dealias_mask", False))
            if time_cfg is not None
            else False,
            dt_min=float(time_cfg.dt_min) if time_cfg is not None else 1.0e-7,
            dt_max=float(time_cfg.dt_max) if (time_cfg is not None and time_cfg.dt_max is not None) else None,
            cfl=float(time_cfg.cfl) if time_cfg is not None else 0.9,
            cfl_fac=(
                resolve_cfl_fac(str(time_cfg.method), time_cfg.cfl_fac)
                if time_cfg is not None
                else float(GXTimeConfig.cfl_fac)
            ),
        )
        t_arr, phi_t, gamma_t, omega_t, _gx_diag = integrate_linear_gx_diagnostics(
            G0_jax,
            grid,
            cache,
            params,
            geom,
            gx_time_cfg,
            terms=terms,
            mode_method=gx_mode_method,
            z_index=sel.z_index,
            jit=True,
        )
        t_out = np.asarray(t_arr, dtype=float)
        phi_t_np = np.asarray(phi_t)
        if t_out.size > 1:
            if mode_method in {"z_index", "max"}:
                try:
                    gamma, omega, _g_t, _o_t, _t_mid = gx_growth_rate_from_phi(
                        phi_t_np,
                        t_out,
                        sel,
                        navg_fraction=0.5,
                        mode_method=mode_method,
                    )
                except ValueError:
                    try:
                        gamma, omega, _g_t, _o_t = gx_growth_rate_from_omega_series(
                            np.asarray(gamma_t),
                            np.asarray(omega_t),
                            sel,
                            navg_fraction=0.5,
                        )
                    except ValueError:
                        signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
                        gamma, omega = _fit_with_window(signal, t_out)
            else:
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
                else:
                    gamma, omega = _fit_with_window(signal, t_out)
        else:
            gamma = float("nan")
            omega = float("nan")
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

    if solver_key == "krylov":
        shift_val = krylov_cfg_use.shift
        targets: Sequence[float] | None = kbm_target_factors if kbm_target_factors else None
        use_multi_target = _kbm_use_multi_target_krylov(
            krylov_cfg_use,
            targets,
            shift=shift_val,
        )
        if use_multi_target:
            assert targets is not None
            beta_transition = (
                float(cfg_use.model.beta)
                if kbm_beta_transition is None
                else float(kbm_beta_transition)
            )
            eig_candidates = []
            vec_candidates = []
            for target in targets:
                eig_i, vec_i = dominant_eigenpair(
                    G0_jax,
                    cache,
                    params,
                    terms=terms,
                    v_ref=None,
                    select_overlap=False,
                    krylov_dim=krylov_cfg_use.krylov_dim,
                    restarts=krylov_cfg_use.restarts,
                    omega_min_factor=krylov_cfg_use.omega_min_factor,
                    omega_target_factor=float(target),
                    omega_cap_factor=krylov_cfg_use.omega_cap_factor,
                    omega_sign=krylov_cfg_use.omega_sign,
                    method=krylov_cfg_use.method,
                    power_iters=krylov_cfg_use.power_iters,
                    power_dt=krylov_cfg_use.power_dt,
                    shift=None,
                    shift_source="target",
                    shift_tol=krylov_cfg_use.shift_tol,
                    shift_maxiter=krylov_cfg_use.shift_maxiter,
                    shift_restart=krylov_cfg_use.shift_restart,
                    shift_solve_method=krylov_cfg_use.shift_solve_method,
                    shift_preconditioner=krylov_cfg_use.shift_preconditioner,
                    shift_selection="targeted",
                    mode_family=krylov_cfg_use.mode_family,
                    fallback_method=krylov_cfg_use.fallback_method,
                    fallback_real_floor=krylov_cfg_use.fallback_real_floor,
                )
                eig_candidates.append(eig_i)
                vec_candidates.append(vec_i)
            if len(eig_candidates) >= 2 and np.isfinite(beta_transition):
                idx = 1 if beta_use >= beta_transition else 0
            else:
                eig_arr = np.asarray([complex(np.asarray(e)) for e in eig_candidates])
                growth = np.real(eig_arr)
                idx = 0 if np.all(~np.isfinite(growth)) else int(
                    np.nanargmax(np.where(np.isfinite(growth), growth, -np.inf))
                )
            eig = eig_candidates[idx]
            vec = vec_candidates[idx]
        else:
            eig, vec = dominant_eigenpair(
                G0_jax,
                cache,
                params,
                terms=terms,
                v_ref=None,
                select_overlap=False,
                krylov_dim=krylov_cfg_use.krylov_dim,
                restarts=krylov_cfg_use.restarts,
                omega_min_factor=krylov_cfg_use.omega_min_factor,
                omega_target_factor=krylov_cfg_use.omega_target_factor,
                omega_cap_factor=krylov_cfg_use.omega_cap_factor,
                omega_sign=krylov_cfg_use.omega_sign,
                method=krylov_cfg_use.method,
                power_iters=krylov_cfg_use.power_iters,
                power_dt=krylov_cfg_use.power_dt,
                shift=shift_val,
                shift_source=krylov_cfg_use.shift_source,
                shift_tol=krylov_cfg_use.shift_tol,
                shift_maxiter=krylov_cfg_use.shift_maxiter,
                shift_restart=krylov_cfg_use.shift_restart,
                shift_solve_method=krylov_cfg_use.shift_solve_method,
                shift_preconditioner=krylov_cfg_use.shift_preconditioner,
                shift_selection=krylov_cfg_use.shift_selection,
                mode_family=krylov_cfg_use.mode_family,
                fallback_method=krylov_cfg_use.fallback_method,
                fallback_real_floor=krylov_cfg_use.fallback_real_floor,
            )
        gamma = float(np.real(eig))
        omega = float(-np.imag(eig))
        if krylov_cfg_use.omega_sign != 0:
            omega = float(np.sign(krylov_cfg_use.omega_sign)) * abs(omega)
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

    stride = 1 if sample_stride is None else int(sample_stride)
    time_cfg_use = time_cfg
    if time_cfg_use is not None:
        time_cfg_use = replace(time_cfg_use, dt=dt, t_max=dt * steps)
        if sample_stride is not None:
            time_cfg_use = replace(time_cfg_use, sample_stride=stride)
    params_use = params
    if time_cfg_use is not None:
        stride = int(time_cfg_use.sample_stride)
        if time_cfg_use.use_diffrax:
            save_field = "phi+density" if fit_key in {"density", "auto"} else "phi"
            _, phi_out = integrate_linear_from_config(
                G0_jax,
                grid,
                geom,
                params_use,
                time_cfg_use,
                cache=cache,
                terms=terms,
                save_mode=sel if fit_key == "phi" else None,
                mode_method=mode_method,
                save_field=save_field,
                density_species_index=density_species_index if fit_key in {"density", "auto"} else None,
            )
            if fit_key in {"density", "auto"}:
                phi_t_np, density_np = (np.asarray(phi_out[0]), np.asarray(phi_out[1]))
            else:
                phi_t_np = np.asarray(phi_out)
                density_np = None
        else:
            if fit_key in {"density", "auto"}:
                diag_out = integrate_linear_diagnostics(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    dt=dt,
                    steps=steps,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=stride,
                    species_index=density_species_index,
                )
                phi_t_np = np.asarray(diag_out[1])
                density_np = None if len(diag_out) <= 2 else np.asarray(diag_out[2])
            else:
                _, phi_out_time = integrate_linear(
                    G0_jax,
                    grid,
                    geom,
                    params_use,
                    dt=dt,
                    steps=steps,
                    method=method,
                    cache=cache,
                    terms=terms,
                    sample_stride=stride,
                )
                phi_t_np = np.asarray(phi_out_time)
                density_np = None
    else:
        if fit_key in {"density", "auto"}:
            diag_out = integrate_linear_diagnostics(
                G0_jax,
                grid,
                geom,
                params_use,
                dt=dt,
                steps=steps,
                method=method,
                cache=cache,
                terms=terms,
                sample_stride=stride,
                species_index=density_species_index,
            )
            phi_t_np = np.asarray(diag_out[1])
            density_np = None if len(diag_out) <= 2 else np.asarray(diag_out[2])
        else:
            _, phi_out_time = integrate_linear(
                G0_jax,
                grid,
                geom,
                params_use,
                dt=dt,
                steps=steps,
                method=method,
                cache=cache,
                terms=terms,
                sample_stride=stride,
            )
            phi_t_np = np.asarray(phi_out_time)
            density_np = None

    if fit_key == "density" and density_np is None:
        density_np = phi_t_np
    if fit_key == "auto":
        signal, _name, gamma, omega = _select_fit_signal_auto(
            np.arange(phi_t_np.shape[0]) * dt * stride,
            phi_t_np,
            density_np,
            sel,
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
        _ = signal
    else:
        signal = _select_fit_signal(
            phi_t_np,
            density_np,
            sel,
            fit_signal=fit_key,
            mode_method=mode_method,
        )
        t_out = np.arange(signal.shape[0]) * dt * stride
        gamma, omega = _fit_with_window(signal, t_out)
    gamma, omega = _normalize_growth_rate(gamma, omega, params_use, diagnostic_norm)
    return LinearRunResult(
        t=np.arange(phi_t_np.shape[0]) * dt * stride,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(ky_target),
        selection=sel,
    )


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
    gx_reference: bool | None = True,
) -> LinearScanResult:
    """Run a KBM ky scan at fixed beta.

    This is a thin wrapper over :func:`run_kbm_beta_scan` used for
    GX-reference workflows where the GX benchmark is a ky scan at fixed beta.
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

    def _pick(value, idx):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value[idx].item()
        if isinstance(value, (list, tuple)):
            return value[idx]
        return value

    for i, ky_val in enumerate(ky_vals):
        dt_i = _pick(dt, i)
        steps_i = _pick(steps, i)
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
            tmin=_pick(tmin, i),
            tmax=_pick(tmax, i),
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
