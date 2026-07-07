"""Shared scan policies for benchmark runners.

The public benchmark API stays in :mod:`spectraxgk.benchmarks`; this module
keeps small, deterministic scan decisions separate from solver orchestration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from spectraxgk.diagnostics.analysis import fit_growth_rate, fit_growth_rate_auto
from spectraxgk.validation.benchmarks.batching import _is_array_like
from spectraxgk.diagnostics.growth_rates import _normalize_growth_rate
from spectraxgk.operators.linear.params import LinearParams


VALID_FIT_SIGNALS = frozenset({"phi", "density", "auto"})


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
