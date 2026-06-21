"""Benchmark fit-signal, mode-extraction, and diagnostic-normalization policies."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate_auto_with_stats,
)
from spectraxgk.diagnostics.normalization import apply_diagnostic_normalization
from spectraxgk.operators.linear.params import LinearParams


__all__ = [
    "_extract_mode_only_signal",
    "_normalize_growth_rate",
    "_score_fit_signal_auto",
    "_select_fit_signal",
    "_select_fit_signal_auto",
]


@dataclass(frozen=True)
class _AutoFitSignalOptions:
    tmin: float | None
    tmax: float | None
    window_fraction: float
    min_points: int
    start_fraction: float
    growth_weight: float
    require_positive: bool
    min_amp_fraction: float
    max_amp_fraction: float
    window_method: str
    max_fraction: float
    end_fraction: float
    num_windows: int
    phase_weight: float
    length_weight: float
    min_r2: float
    late_penalty: float
    min_slope: float | None
    min_slope_frac: float
    slope_var_weight: float


@dataclass(frozen=True)
class _FitSignalCandidate:
    signal: np.ndarray
    name: str
    gamma: float
    omega: float
    score: float


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


def _score_fit_signal_candidate(
    t: np.ndarray,
    source: np.ndarray,
    sel: ModeSelection,
    *,
    name: str,
    mode_method: str,
    options: _AutoFitSignalOptions,
) -> _FitSignalCandidate:
    signal = extract_mode_time_series(source, sel, method=mode_method)
    gamma, omega, score = _score_fit_signal_auto(
        t,
        signal,
        tmin=options.tmin,
        tmax=options.tmax,
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
        num_windows=options.num_windows,
        phase_weight=options.phase_weight,
        length_weight=options.length_weight,
        min_r2=options.min_r2,
        late_penalty=options.late_penalty,
        min_slope=options.min_slope,
        min_slope_frac=options.min_slope_frac,
        slope_var_weight=options.slope_var_weight,
    )
    return _FitSignalCandidate(
        signal=signal,
        name=name,
        gamma=float(gamma),
        omega=float(omega),
        score=float(score),
    )


def _best_fit_signal_candidate(
    current: _FitSignalCandidate,
    candidate: _FitSignalCandidate,
) -> _FitSignalCandidate:
    if candidate.score > current.score:
        return candidate
    return current


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

    options = _AutoFitSignalOptions(
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
    best = _score_fit_signal_candidate(
        t,
        phi_t,
        sel,
        name="phi",
        mode_method=mode_method,
        options=options,
    )
    if density_t is not None:
        best = _best_fit_signal_candidate(
            best,
            _score_fit_signal_candidate(
                t,
                density_t,
                sel,
                name="density",
                mode_method=mode_method,
                options=options,
            ),
        )

    return best.signal, best.name, best.gamma, best.omega


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
