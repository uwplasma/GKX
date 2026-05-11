"""Shared helpers for benchmark entry points.

This module keeps pure policies, reference-data loaders, small result containers,
and initializer builders out of the large benchmark runner module while preserving
``spectraxgk.benchmarks`` as the compatibility import surface.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import resources
from typing import Sequence
import warnings

import jax.numpy as jnp
import numpy as np

from spectraxgk.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate_auto_with_stats,
)
from spectraxgk.config import InitializationConfig, KineticElectronBaseCase
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import LinearParams
from spectraxgk.linear_krylov import KrylovConfig
from spectraxgk.normalization import apply_diagnostic_normalization
from spectraxgk.species import Species, build_linear_params


REFERENCE_NU_HYPER_L = 0.0
REFERENCE_NU_HYPER_M = 1.0
REFERENCE_P_HYPER_L = 6.0
REFERENCE_P_HYPER_M = 20.0
REFERENCE_DAMP_ENDS_AMP = 0.1
REFERENCE_DAMP_ENDS_WIDTHFRAC = 1.0 / 8.0

KBM_GX_SOLVER_LOCK: tuple[tuple[float, str], ...] = (
    (0.10, "gx_time"),
    (0.30, "gx_time"),
    (0.40, "gx_time"),
)
KBM_GX_SOLVER_LOCK_TOL = 0.03


def _gx_p_hyper_m(nhermite: int | None) -> float:
    if nhermite is None:
        return REFERENCE_P_HYPER_M
    return float(min(REFERENCE_P_HYPER_M, max(int(nhermite) // 2, 1)))


def _apply_gx_hypercollisions(
    params: LinearParams, *, nhermite: int | None = None
) -> LinearParams:
    return replace(
        params,
        nu_hyper=0.0,
        nu_hyper_l=REFERENCE_NU_HYPER_L,
        nu_hyper_m=REFERENCE_NU_HYPER_M,
        p_hyper_l=REFERENCE_P_HYPER_L,
        p_hyper_m=_gx_p_hyper_m(nhermite),
        hypercollisions_const=0.0,
        hypercollisions_kz=1.0,
    )


def _gx_linked_end_damping(gx_reference: bool) -> tuple[float, float]:
    if gx_reference:
        return REFERENCE_DAMP_ENDS_AMP, REFERENCE_DAMP_ENDS_WIDTHFRAC
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
            num_windows=num_windows,
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
        num_windows=num_windows,
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


def _kinetic_reference_init_cfg(
    init_cfg: InitializationConfig, *, gx_reference: bool
) -> InitializationConfig:
    """Restore the historical kinetic benchmark seed on the GX-reference path.

    Older kinetic parity runs seeded a constant electron-density moment rather than
    the newer tiny Gaussian default. Preserve explicit user overrides by only
    replacing the exact current kinetic default init.
    """

    if not gx_reference:
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

    data_path = resources.files("spectraxgk").joinpath(
        "data", "cyclone_reference_adiabatic.csv"
    )
    arr = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


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

    data_path = resources.files("spectraxgk").joinpath(
        "data", "cyclone_reference_kinetic.csv"
    )
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
    """Load the provisional TEM reference digitized from the literature.

    This lane is not backed by a GX reference dump. It remains an extended
    stress case while the literature case definition is being reconstructed.
    """

    data_path = resources.files("spectraxgk").joinpath("data", "tem_reference.csv")
    arr = np.loadtxt(str(data_path), delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


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

