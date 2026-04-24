"""Helpers for zonal-response validation artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def kx_token(kx: float) -> str:
    """Return the canonical three-digit token for ``kx rho_i`` values."""

    return f"{int(round(1000.0 * float(kx))):03d}"


def w7x_trace_path(trace_dir: Path, kx: float) -> Path:
    """Return the per-``kx`` W7-X test-4 trace path in a generator output directory."""

    return trace_dir / f"w7x_test4_kx{kx_token(kx)}.csv"


def normalize_trace(
    t: np.ndarray,
    y: np.ndarray,
    *,
    initial_level: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort, finite-filter, and normalize a scalar zonal-response trace."""

    order = np.argsort(t)
    t_sorted = np.asarray(t, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]
    finite = np.isfinite(t_sorted) & np.isfinite(y_sorted)
    t_sorted = t_sorted[finite]
    y_sorted = y_sorted[finite]
    if t_sorted.size == 0:
        raise ValueError("trace is empty after finite filtering")
    if initial_level is None:
        nz = np.flatnonzero(np.abs(y_sorted) > 1.0e-30)
        scale = float(abs(y_sorted[nz[0]])) if nz.size else 1.0
    else:
        scale = float(initial_level)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("trace normalization level must be finite and positive")
    return t_sorted, y_sorted / scale


def load_w7x_trace_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a W7-X trace CSV with either ``t`` or ``t_reference`` as the time column."""

    trace = pd.read_csv(path)
    time_col = "t_reference" if "t_reference" in trace.columns else "t"
    if "phi_zonal_real" not in trace.columns or time_col not in trace.columns:
        raise ValueError(f"{path} must contain phi_zonal_real and either t or t_reference columns")
    return np.asarray(trace[time_col], dtype=float), np.asarray(trace["phi_zonal_real"], dtype=float)


def load_w7x_combined_trace_csv(
    path: Path,
    kx: float,
    *,
    normalized: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one ``kx`` trace from a combined W7-X zonal trace CSV."""

    trace = pd.read_csv(path)
    required = {"kx_target", "t_reference"}
    missing = required.difference(trace.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    value_col = "response_normalized" if normalized else "phi_zonal_real"
    if value_col not in trace.columns:
        raise ValueError(f"{path} missing column: {value_col}")
    subset = trace[np.isclose(trace["kx_target"], float(kx))]
    if subset.empty:
        raise ValueError(f"{path} has no trace for kx={kx}")
    subset = subset.sort_values("t_reference")
    return np.asarray(subset["t_reference"], dtype=float), np.asarray(subset[value_col], dtype=float)


def reference_residual_table(path: Path) -> pd.DataFrame:
    """Build a per-``kx`` residual table from digitized stella/GENE inset data."""

    table = pd.read_csv(path)
    required = {"kx_rhoi", "code", "residual_median"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    rows: list[dict[str, float]] = []
    for kx, group in table.groupby("kx_rhoi"):
        medians = np.asarray(group["residual_median"], dtype=float)
        if medians.size < 1:
            continue
        center = float(np.mean(medians))
        spread = float(np.max(np.abs(medians - center))) if medians.size > 1 else 0.0
        rows.append(
            {
                "kx": float(kx),
                "reference_residual": center,
                "reference_code_spread": spread,
                "reference_min": float(np.min(medians)),
                "reference_max": float(np.max(medians)),
            }
        )
    return pd.DataFrame(rows).sort_values("kx").reset_index(drop=True)


def reference_time_limits(trace_table: pd.DataFrame) -> pd.DataFrame:
    """Return digitized reference time limits for each W7-X zonal ``kx`` value."""

    required = {"kx_rhoi", "t_vti_over_a"}
    missing = required.difference(trace_table.columns)
    if missing:
        raise ValueError(f"reference trace table missing columns: {sorted(missing)}")
    rows = []
    for kx, group in trace_table.groupby("kx_rhoi"):
        t = np.asarray(group["t_vti_over_a"], dtype=float)
        rows.append({"kx": float(kx), "reference_tmax": float(np.nanmax(t)), "reference_tmin": float(np.nanmin(t))})
    return pd.DataFrame(rows)


def reference_mean_trace(trace_table: pd.DataFrame, kx: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean digitized stella/GENE trace for one W7-X zonal ``kx``."""

    ref_subset = trace_table[np.isclose(trace_table["kx_rhoi"], float(kx))]
    if ref_subset.empty:
        raise ValueError(f"missing reference trace for kx={kx}")
    ref_pivot = ref_subset.pivot_table(index="t_vti_over_a", columns="code", values="response", aggfunc="mean")
    ref_pivot = ref_pivot.sort_index()
    return np.asarray(ref_pivot.index, dtype=float), np.asarray(ref_pivot.mean(axis=1), dtype=float)


def tail_trace_metrics(
    *,
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    tail_fraction: float,
) -> dict[str, float | None]:
    """Compare observed and reference traces over the late reference window."""

    ref_tmax = float(np.nanmax(t_ref))
    tail_start = ref_tmax - float(tail_fraction) * (ref_tmax - float(np.nanmin(t_ref)))
    mask = (np.asarray(t_obs, dtype=float) >= tail_start) & (np.asarray(t_obs, dtype=float) <= ref_tmax)
    if not np.any(mask):
        return {
            "tail_std": None,
            "reference_tail_std": None,
            "tail_mean_abs_error": None,
            "tail_max_abs_error": None,
        }
    ref_interp = np.interp(np.asarray(t_obs, dtype=float)[mask], np.asarray(t_ref, dtype=float), np.asarray(y_ref, dtype=float))
    obs_tail = np.asarray(y_obs, dtype=float)[mask]
    diff = obs_tail - ref_interp
    ref_tail = np.asarray(y_ref, dtype=float)[np.asarray(t_ref, dtype=float) >= tail_start]
    return {
        "tail_std": float(np.std(obs_tail)),
        "reference_tail_std": float(np.std(ref_tail)),
        "tail_mean_abs_error": float(np.mean(np.abs(diff))),
        "tail_max_abs_error": float(np.max(np.abs(diff))),
    }
