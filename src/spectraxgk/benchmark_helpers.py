"""Shared helpers for benchmark entry points.

This module keeps pure policies, reference-data loaders, small result containers,
and initializer builders out of the large benchmark runner module while preserving
``spectraxgk.benchmarks`` as the compatibility import surface.
"""

from __future__ import annotations

from typing import Sequence
import warnings

import numpy as np

from spectraxgk.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate_auto_with_stats,
)
from spectraxgk.benchmark_initialization import (
    _build_gaussian_profile,
    _build_initial_condition,
    _kinetic_reference_init_cfg,
)
from spectraxgk.benchmark_reference import (
    CycloneComparison,
    CycloneReference,
    CycloneRunResult,
    CycloneScanResult,
    LinearRunResult,
    LinearScanResult,
    _load_reference_with_header,
    compare_cyclone_to_reference,
    load_cyclone_reference,
    load_cyclone_reference_kinetic,
    load_etg_reference,
    load_kbm_reference,
    load_tem_reference,
)
from spectraxgk.benchmark_species import (
    REFERENCE_DAMP_ENDS_AMP,
    REFERENCE_DAMP_ENDS_WIDTHFRAC,
    REFERENCE_NU_HYPER_L,
    REFERENCE_NU_HYPER_M,
    REFERENCE_P_HYPER_L,
    REFERENCE_P_HYPER_M,
    _apply_gx_hypercollisions,
    _electron_only_params,
    _gx_linked_end_damping,
    _gx_p_hyper_m,
    _two_species_params,
)
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import LinearParams
from spectraxgk.linear_krylov import KrylovConfig
from spectraxgk.normalization import apply_diagnostic_normalization


__all__ = [
    "REFERENCE_NU_HYPER_L",
    "REFERENCE_NU_HYPER_M",
    "REFERENCE_P_HYPER_L",
    "REFERENCE_P_HYPER_M",
    "REFERENCE_DAMP_ENDS_AMP",
    "REFERENCE_DAMP_ENDS_WIDTHFRAC",
    "KBM_GX_SOLVER_LOCK",
    "KBM_GX_SOLVER_LOCK_TOL",
    "CycloneComparison",
    "CycloneReference",
    "CycloneRunResult",
    "CycloneScanResult",
    "LinearRunResult",
    "LinearScanResult",
    "_apply_gx_hypercollisions",
    "_build_gaussian_profile",
    "_build_initial_condition",
    "_electron_only_params",
    "_extract_mode_only_signal",
    "_gx_linked_end_damping",
    "_gx_p_hyper_m",
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
    "load_cyclone_reference",
    "load_cyclone_reference_kinetic",
    "load_etg_reference",
    "load_kbm_reference",
    "load_tem_reference",
    "select_kbm_solver_auto",
]


KBM_GX_SOLVER_LOCK: tuple[tuple[float, str], ...] = (
    (0.10, "gx_time"),
    (0.30, "gx_time"),
    (0.40, "gx_time"),
)
KBM_GX_SOLVER_LOCK_TOL = 0.03


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
