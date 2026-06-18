"""Runtime linear-fit and quasilinear diagnostic helpers.

The finite-value and diagnostic-array composition helpers are re-exported here
for compatibility, but their implementation lives in
``workflows.runtime.diagnostic_arrays`` so this module can focus on fitted
linear observables and quasilinear finalization.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from spectraxgk.diagnostics.analysis import (
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_auto_with_stats,
)
from spectraxgk.workflows.runtime.diagnostic_arrays import (
    concat_runtime_diagnostics,
    slice_runtime_diagnostics,
    stride_runtime_diagnostics,
    truncate_runtime_diagnostics,
    validate_finite_runtime_diagnostics,
)
from spectraxgk.workflows.runtime.results import RuntimeLinearResult

__all__ = [
    "RuntimeLinearFitResult",
    "RuntimeQuasilinearFinalizationDeps",
    "concat_runtime_diagnostics",
    "finalize_runtime_linear_quasilinear",
    "fit_runtime_linear_diagnostics",
    "slice_runtime_diagnostics",
    "stride_runtime_diagnostics",
    "truncate_runtime_diagnostics",
    "validate_finite_runtime_diagnostics",
]


@dataclass(frozen=True)
class RuntimeLinearFitResult:
    """Linear runtime fit payload before diagnostic normalization."""

    gamma: float
    omega: float
    signal: np.ndarray
    z: np.ndarray
    eigenfunction: np.ndarray | None
    fit_window_tmin: float | None
    fit_window_tmax: float | None
    fit_signal_used: str


@dataclass(frozen=True)
class RuntimeQuasilinearFinalizationDeps:
    """Injected dependencies for runtime quasilinear post-processing."""

    build_linear_cache: Any
    compute_quasilinear_from_linear_state: Any
    linear_terms_to_term_config: Any


def finalize_runtime_linear_quasilinear(
    result: RuntimeLinearResult,
    *,
    enabled: bool,
    cfg: Any,
    grid: Any,
    geom: Any,
    params: Any,
    terms: Any,
    Nl: int,
    Nm: int,
    solver_name: str,
    species_names: tuple[str, ...],
    return_state_requested: bool,
    state_for_quasilinear: np.ndarray | None = None,
    deps: RuntimeQuasilinearFinalizationDeps,
    status_callback: Any | None = None,
) -> RuntimeLinearResult:
    """Attach optional quasilinear diagnostics to a linear runtime result."""

    ql_payload = None
    state_for_ql = state_for_quasilinear if state_for_quasilinear is not None else result.state
    if enabled:
        if state_for_ql is None:
            raise RuntimeError("quasilinear diagnostics require a final linear state")
        ql_cfg = cfg.quasilinear
        if status_callback is not None:
            status_callback("computing quasilinear transport weights")
        cache = deps.build_linear_cache(grid, geom, params, Nl, Nm)
        ql_payload = deps.compute_quasilinear_from_linear_state(
            state_for_ql,
            cache=cache,
            grid=grid,
            geom=geom,
            params=params,
            ky=float(result.ky),
            gamma=float(result.gamma),
            omega=float(result.omega),
            terms=deps.linear_terms_to_term_config(terms),
            mode=str(ql_cfg.mode),
            saturation_rule=str(ql_cfg.saturation_rule),
            amplitude_normalization=str(ql_cfg.amplitude_normalization),
            kperp_average=str(ql_cfg.kperp_average),
            csat=float(ql_cfg.csat),
            gamma_floor=float(ql_cfg.gamma_floor),
            include_stable_modes=bool(ql_cfg.include_stable_modes),
            channels=ql_cfg.channels,
            species_names=species_names,
            flux_scale=float(cfg.normalization.flux_scale),
            metadata={
                "runtime_config_enabled": True,
                "solver": solver_name,
                "delta_ky": ql_cfg.delta_ky,
                "species_selection": ql_cfg.species,
                "write_spectrum": bool(ql_cfg.write_spectrum),
            },
        ).to_dict()
        if status_callback is not None:
            status_callback("quasilinear transport weights complete")
    return replace(
        result,
        state=result.state if return_state_requested else None,
        quasilinear=ql_payload,
    )


def _resolved_fit_bounds(
    t_arr: np.ndarray,
    tmin_fit: float | None,
    tmax_fit: float | None,
) -> tuple[float | None, float | None]:
    if t_arr.size == 0:
        return None, None
    tmin_use = float(tmin_fit) if tmin_fit is not None else float(t_arr[0])
    tmax_use = float(tmax_fit) if tmax_fit is not None else float(t_arr[-1])
    return tmin_use, tmax_use


def fit_runtime_linear_diagnostics(
    *,
    t: np.ndarray,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    selection: Any,
    z: np.ndarray,
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
    extract_mode_time_series_fn: Any = extract_mode_time_series,
    fit_growth_rate_auto_with_stats_fn: Any = fit_growth_rate_auto_with_stats,
    fit_growth_rate_auto_fn: Any = fit_growth_rate_auto,
    fit_growth_rate_fn: Any = fit_growth_rate,
    extract_eigenfunction_fn: Any = extract_eigenfunction,
) -> RuntimeLinearFitResult:
    """Fit linear growth/frequency and extract the eigenfunction diagnostic."""

    fit_key = str(fit_signal).strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")

    t_arr = np.asarray(t, dtype=float)
    phi_arr = np.asarray(phi_t)
    density_arr = None if density_t is None else np.asarray(density_t)
    z_arr = np.asarray(z, dtype=float)

    fit_window_tmin: float | None = None
    fit_window_tmax: float | None = None
    if fit_key == "auto":
        phi_signal = extract_mode_time_series_fn(phi_arr, selection, method=mode_method)
        gamma_phi, omega_phi, phi_tmin, phi_tmax, r2_phi, r2p_phi = (
            fit_growth_rate_auto_with_stats_fn(
                t_arr,
                phi_signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
        )
        gamma, omega = gamma_phi, omega_phi
        signal_out = np.asarray(phi_signal)
        fit_window_tmin, fit_window_tmax = phi_tmin, phi_tmax
        fit_signal_used = "phi"
        best_score = r2_phi + 0.2 * r2p_phi + growth_weight * gamma_phi
        if density_arr is not None:
            dens_signal = extract_mode_time_series_fn(
                density_arr, selection, method=mode_method
            )
            gamma_den, omega_den, den_tmin, den_tmax, r2_den, r2p_den = (
                fit_growth_rate_auto_with_stats_fn(
                    t_arr,
                    dens_signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
            )
            score_den = r2_den + 0.2 * r2p_den + growth_weight * gamma_den
            if score_den > best_score:
                gamma, omega = gamma_den, omega_den
                signal_out = np.asarray(dens_signal)
                fit_window_tmin, fit_window_tmax = den_tmin, den_tmax
                fit_signal_used = "density"
    else:
        signal = extract_mode_time_series_fn(
            density_arr
            if fit_key == "density" and density_arr is not None
            else phi_arr,
            selection,
            method=mode_method,
        )
        signal_out = np.asarray(signal)
        fit_signal_used = (
            "density" if fit_key == "density" and density_arr is not None else "phi"
        )
        if auto_window:
            gamma, omega, fit_window_tmin, fit_window_tmax = fit_growth_rate_auto_fn(
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
            gamma, omega = fit_growth_rate_fn(t_arr, signal, tmin=tmin, tmax=tmax)
            fit_window_tmin, fit_window_tmax = _resolved_fit_bounds(t_arr, tmin, tmax)

    try:
        eigenfunction = np.asarray(
            extract_eigenfunction_fn(
                phi_arr,
                t_arr,
                selection,
                z=z_arr,
                method="svd",
                tmin=fit_window_tmin,
                tmax=fit_window_tmax,
            )
        )
    except Exception:
        eigenfunction = None

    return RuntimeLinearFitResult(
        gamma=float(gamma),
        omega=float(omega),
        signal=signal_out,
        z=z_arr,
        eigenfunction=eigenfunction,
        fit_window_tmin=fit_window_tmin,
        fit_window_tmax=fit_window_tmax,
        fit_signal_used=fit_signal_used,
    )
