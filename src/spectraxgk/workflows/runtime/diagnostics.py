"""Runtime linear-fit and quasilinear diagnostic helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from spectraxgk.diagnostics.growth_rates import (
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_auto_with_stats,
)
from spectraxgk.diagnostics.modes import (
    extract_eigenfunction,
    extract_mode_time_series,
)
from spectraxgk.workflows.runtime.results import RuntimeLinearResult

__all__ = [
    "RuntimeLinearFitResult",
    "RuntimeQuasilinearFinalizationDeps",
    "finalize_runtime_linear_quasilinear",
    "fit_runtime_linear_diagnostics",
    "refit_runtime_linear_trajectory",
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


@dataclass(frozen=True)
class _RuntimeLinearFitInputs:
    """Validated arrays and mode-selection policy for a linear fit."""

    fit_key: str
    t: np.ndarray
    phi: np.ndarray
    density: np.ndarray | None
    z: np.ndarray


@dataclass(frozen=True)
class _RuntimeLinearFitCandidate:
    """Candidate growth/frequency fit for one diagnostic channel."""

    signal_name: str
    signal: np.ndarray
    gamma: float
    omega: float
    fit_window_tmin: float | None
    fit_window_tmax: float | None
    score: float


@dataclass(frozen=True)
class _RuntimeLinearFitOptions:
    """Windowing and mode policy for runtime linear fits."""

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
class _RuntimeLinearDiagnosticDeps:
    """Injected numerical routines used by runtime linear diagnostics."""

    extract_mode_time_series: Any
    fit_growth_rate_auto_with_stats: Any
    fit_growth_rate_auto: Any
    fit_growth_rate: Any
    extract_eigenfunction: Any


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


def _prepare_runtime_linear_fit_inputs(
    *,
    t: np.ndarray,
    phi_t: np.ndarray,
    density_t: np.ndarray | None,
    z: np.ndarray,
    fit_signal: str,
) -> _RuntimeLinearFitInputs:
    """Normalize fit arrays and validate the requested diagnostic channel."""

    fit_key = str(fit_signal).strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")
    inputs = _RuntimeLinearFitInputs(
        fit_key=fit_key,
        t=np.asarray(t, dtype=float),
        phi=np.asarray(phi_t),
        density=None if density_t is None else np.asarray(density_t),
        z=np.asarray(z, dtype=float),
    )
    for name, values in (
        ("time", inputs.t),
        ("field", inputs.phi),
        ("density", inputs.density),
    ):
        if values is not None and not np.all(np.isfinite(values)):
            raise FloatingPointError(
                f"linear integration produced a non-finite {name} history; "
                "reduce the timestep or select a stable integration policy"
            )
    return inputs


def _fit_auto_candidate(
    *,
    name: str,
    data: np.ndarray,
    inputs: _RuntimeLinearFitInputs,
    selection: Any,
    options: _RuntimeLinearFitOptions,
    deps: _RuntimeLinearDiagnosticDeps,
) -> _RuntimeLinearFitCandidate:
    """Fit and score one channel for automatic runtime fit-signal selection."""

    signal = np.asarray(
        deps.extract_mode_time_series(data, selection, method=options.mode_method)
    )
    gamma, omega, tmin, tmax, r2, r2_phase = deps.fit_growth_rate_auto_with_stats(
        inputs.t,
        signal,
        window_fraction=options.window_fraction,
        min_points=options.min_points,
        start_fraction=options.start_fraction,
        growth_weight=options.growth_weight,
        require_positive=options.require_positive,
        min_amp_fraction=options.min_amp_fraction,
    )
    score = float(r2) + 0.2 * float(r2_phase) + options.growth_weight * float(gamma)
    return _RuntimeLinearFitCandidate(
        signal_name=name,
        signal=signal,
        gamma=float(gamma),
        omega=float(omega),
        fit_window_tmin=tmin,
        fit_window_tmax=tmax,
        score=score,
    )


def _choose_auto_runtime_linear_fit(
    inputs: _RuntimeLinearFitInputs,
    *,
    selection: Any,
    options: _RuntimeLinearFitOptions,
    deps: _RuntimeLinearDiagnosticDeps,
) -> _RuntimeLinearFitCandidate:
    """Choose between phi and density using the runtime automatic fit score."""

    candidates = [
        _fit_auto_candidate(
            name="phi",
            data=inputs.phi,
            inputs=inputs,
            selection=selection,
            options=options,
            deps=deps,
        )
    ]
    if inputs.density is not None:
        candidates.append(
            _fit_auto_candidate(
                name="density",
                data=inputs.density,
                inputs=inputs,
                selection=selection,
                options=options,
                deps=deps,
            )
        )
    return max(candidates, key=lambda candidate: candidate.score)


def _fit_requested_runtime_linear_signal(
    inputs: _RuntimeLinearFitInputs,
    *,
    selection: Any,
    options: _RuntimeLinearFitOptions,
    deps: _RuntimeLinearDiagnosticDeps,
) -> _RuntimeLinearFitCandidate:
    """Fit the explicitly requested phi or density runtime signal."""

    use_density = inputs.fit_key == "density" and inputs.density is not None
    signal_name = "density" if use_density else "phi"
    source = inputs.density if use_density else inputs.phi
    signal = np.asarray(
        deps.extract_mode_time_series(source, selection, method=options.mode_method)
    )
    if options.auto_window:
        gamma, omega, fit_tmin, fit_tmax = deps.fit_growth_rate_auto(
            inputs.t,
            signal,
            window_fraction=options.window_fraction,
            min_points=options.min_points,
            start_fraction=options.start_fraction,
            growth_weight=options.growth_weight,
            require_positive=options.require_positive,
            min_amp_fraction=options.min_amp_fraction,
        )
    else:
        gamma, omega = deps.fit_growth_rate(
            inputs.t,
            signal,
            tmin=options.tmin,
            tmax=options.tmax,
        )
        fit_tmin, fit_tmax = _resolved_fit_bounds(
            inputs.t,
            options.tmin,
            options.tmax,
        )
    return _RuntimeLinearFitCandidate(
        signal_name=signal_name,
        signal=signal,
        gamma=float(gamma),
        omega=float(omega),
        fit_window_tmin=fit_tmin,
        fit_window_tmax=fit_tmax,
        score=float("nan"),
    )


def _extract_runtime_linear_eigenfunction(
    inputs: _RuntimeLinearFitInputs,
    *,
    selection: Any,
    fit_window_tmin: float | None,
    fit_window_tmax: float | None,
    deps: _RuntimeLinearDiagnosticDeps,
) -> np.ndarray | None:
    """Extract a phi eigenfunction, returning None when the SVD path is ill-conditioned."""

    try:
        return np.asarray(
            deps.extract_eigenfunction(
                inputs.phi,
                inputs.t,
                selection,
                z=inputs.z,
                method="svd",
                tmin=fit_window_tmin,
                tmax=fit_window_tmax,
            )
        )
    except Exception:
        return None


def _select_runtime_linear_fit(
    inputs: _RuntimeLinearFitInputs,
    *,
    selection: Any,
    options: _RuntimeLinearFitOptions,
    deps: _RuntimeLinearDiagnosticDeps,
) -> _RuntimeLinearFitCandidate:
    """Select and fit the runtime diagnostic signal."""

    if inputs.fit_key == "auto":
        return _choose_auto_runtime_linear_fit(
            inputs,
            selection=selection,
            options=options,
            deps=deps,
        )
    return _fit_requested_runtime_linear_signal(
        inputs,
        selection=selection,
        options=options,
        deps=deps,
    )


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

    inputs = _prepare_runtime_linear_fit_inputs(
        t=t,
        phi_t=phi_t,
        density_t=density_t,
        z=z,
        fit_signal=fit_signal,
    )
    options = _RuntimeLinearFitOptions(
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
    deps = _RuntimeLinearDiagnosticDeps(
        extract_mode_time_series=extract_mode_time_series_fn,
        fit_growth_rate_auto_with_stats=fit_growth_rate_auto_with_stats_fn,
        fit_growth_rate_auto=fit_growth_rate_auto_fn,
        fit_growth_rate=fit_growth_rate_fn,
        extract_eigenfunction=extract_eigenfunction_fn,
    )
    fit = _select_runtime_linear_fit(
        inputs,
        selection=selection,
        options=options,
        deps=deps,
    )
    eigenfunction = _extract_runtime_linear_eigenfunction(
        inputs,
        selection=selection,
        fit_window_tmin=fit.fit_window_tmin,
        fit_window_tmax=fit.fit_window_tmax,
        deps=deps,
    )

    return RuntimeLinearFitResult(
        gamma=fit.gamma,
        omega=fit.omega,
        signal=fit.signal,
        z=inputs.z,
        eigenfunction=eigenfunction,
        fit_window_tmin=fit.fit_window_tmin,
        fit_window_tmax=fit.fit_window_tmax,
        fit_signal_used=fit.signal_name,
    )


def refit_runtime_linear_trajectory(
    result: RuntimeLinearResult,
    *,
    mode_method: str = "project",
    auto_window: bool = True,
    tmin: float | None = None,
    tmax: float | None = None,
    window_fraction: float = 0.3,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 1.0,
    require_positive: bool = True,
    min_amp_fraction: float = 0.0,
) -> RuntimeLinearResult:
    """Refit one stored trajectory without repeating its integration."""

    if result.t is None or result.field_history is None or result.z is None:
        raise ValueError("result must contain t, field_history, and z")
    fit = fit_runtime_linear_diagnostics(
        t=result.t,
        phi_t=result.field_history,
        density_t=None,
        selection=result.selection,
        z=result.z,
        fit_signal="phi",
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
    return replace(
        result,
        gamma=float(fit.gamma),
        omega=float(fit.omega),
        signal=fit.signal,
        eigenfunction=fit.eigenfunction,
        fit_window_tmin=fit.fit_window_tmin,
        fit_window_tmax=fit.fit_window_tmax,
        fit_signal_used=fit.fit_signal_used,
    )
