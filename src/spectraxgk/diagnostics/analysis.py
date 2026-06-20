"""Public facade for mode extraction and growth-rate diagnostics."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from spectraxgk.diagnostics.growth_rates import (
    _log_amp_phase,
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_with_stats,
    instantaneous_growth_rate_from_phi,
    select_fit_window,
    select_fit_window_loglinear,
    windowed_growth_rate_from_omega_series,
)
from spectraxgk.diagnostics.modes import (
    ModeSelection,
    ModeSelectionBatch,
    density_moment,
    extract_eigenfunction,
    extract_mode,
    extract_mode_time_series,
    select_ky_index,
)

__all__ = [
    "ModeSelection",
    "ModeSelectionBatch",
    "_log_amp_phase",
    "density_moment",
    "extract_eigenfunction",
    "extract_mode",
    "extract_mode_time_series",
    "fit_growth_rate",
    "fit_growth_rate_auto",
    "fit_growth_rate_auto_with_stats",
    "fit_growth_rate_with_stats",
    "instantaneous_growth_rate_from_phi",
    "select_fit_window",
    "select_fit_window_loglinear",
    "select_ky_index",
    "windowed_growth_rate_from_omega_series",
]


def fit_growth_rate_auto_with_stats(
    t: np.ndarray,
    signal: np.ndarray,
    tmin: float | None = None,
    tmax: float | None = None,
    window_fraction: float = 0.3,
    min_points: int = 20,
    start_fraction: float = 0.0,
    growth_weight: float = 0.0,
    require_positive: bool = False,
    min_amp_fraction: float = 0.0,
    max_amp_fraction: float = 0.9,
    window_method: str = "loglinear",
    max_fraction: float = 0.8,
    end_fraction: float = 0.9,
    num_windows: int = 8,
    phase_weight: float = 0.2,
    length_weight: float = 0.05,
    min_r2: float = 0.0,
    late_penalty: float = 0.1,
    min_slope: float | None = None,
    min_slope_frac: float = 0.0,
    slope_var_weight: float = 0.0,
) -> Tuple[float, float, float, float, float, float]:
    """Fit gamma/omega and report selected window plus R^2 scores.

    This wrapper intentionally calls the facade-level
    :func:`fit_growth_rate_with_stats` so tests and downstream users can
    monkeypatch the public analysis module without reaching into implementation
    modules.
    """

    gamma, omega, tmin_out, tmax_out = fit_growth_rate_auto(
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
    try:
        _gamma, _omega, r2_log, r2_phase = fit_growth_rate_with_stats(
            t, signal, tmin=tmin_out, tmax=tmax_out
        )
    except ValueError:
        r2_log = -np.inf
        r2_phase = -np.inf
    return gamma, omega, tmin_out, tmax_out, float(r2_log), float(r2_phase)
