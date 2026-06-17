"""Runtime diagnostic chunk helpers.

This module owns the runtime diagnostic slicing, truncation, striding, and
concatenation helpers used by the runtime drivers. Keeping these utilities out
of ``runtime.py`` makes the execution/control-flow layer smaller while
preserving the existing public runtime behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, replace
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_auto_with_stats,
)
from spectraxgk.diagnostics import (
    ResolvedDiagnostics,
    SimulationDiagnostics,
    total_energy,
)
from spectraxgk.workflows.runtime.results import RuntimeLinearResult


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


def _first_nonfinite_sample(
    value: np.ndarray | jnp.ndarray, *, nsamples: int
) -> int | None:
    arr = np.asarray(value)
    if arr.size == 0 or np.isfinite(arr).all():
        return None
    if arr.ndim >= 1 and arr.shape[0] == nsamples:
        finite_by_sample = np.isfinite(arr).reshape(arr.shape[0], -1).all(axis=1)
        bad = np.flatnonzero(~finite_by_sample)
        if bad.size:
            return int(bad[0])
    return 0


def validate_finite_runtime_diagnostics(
    diag: SimulationDiagnostics, *, label: str = "runtime"
) -> None:
    """Raise if a runtime diagnostic chunk contains NaN or infinite values.

    Long validation runs can otherwise continue for thousands of fixed steps
    after the first unstable sample. This host-side guard keeps the expensive
    artifact path fail-fast and reports the first offending diagnostic channel.
    """

    t_arr = np.asarray(diag.t, dtype=float)
    nsamples = int(t_arr.size)
    fields_to_check = [
        "t",
        "dt_t",
        "gamma_t",
        "omega_t",
        "Wg_t",
        "Wphi_t",
        "Wapar_t",
        "heat_flux_t",
        "particle_flux_t",
        "energy_t",
        "heat_flux_species_t",
        "particle_flux_species_t",
        "turbulent_heating_t",
        "turbulent_heating_species_t",
        "phi_mode_t",
    ]
    for name in fields_to_check:
        value = getattr(diag, name, None)
        if value is None:
            continue
        sample = _first_nonfinite_sample(value, nsamples=nsamples)
        if sample is None:
            continue
        t_text = ""
        if t_arr.size and sample < t_arr.size and np.isfinite(t_arr[sample]):
            t_text = f" at t={float(t_arr[sample]):.6g}"
        raise RuntimeError(
            f"{label} produced non-finite diagnostics in {name} at sample {sample}{t_text}"
        )

    if diag.resolved is None:
        return
    for field in dataclass_fields(ResolvedDiagnostics):
        value = getattr(diag.resolved, field.name)
        if value is None:
            continue
        sample = _first_nonfinite_sample(value, nsamples=nsamples)
        if sample is None:
            continue
        t_text = ""
        if t_arr.size and sample < t_arr.size and np.isfinite(t_arr[sample]):
            t_text = f" at t={float(t_arr[sample]):.6g}"
        raise RuntimeError(
            f"{label} produced non-finite diagnostics in resolved.{field.name} at sample {sample}{t_text}"
        )


def slice_runtime_diagnostics(
    diag: SimulationDiagnostics, stop: int
) -> SimulationDiagnostics:
    """Return the first ``stop`` diagnostic samples."""

    if stop < 0:
        raise ValueError("stop must be >= 0")

    def _slice_optional(arr: np.ndarray | jnp.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        return np.asarray(arr)[:stop, ...]

    def _slice_resolved(
        resolved: ResolvedDiagnostics | None,
    ) -> ResolvedDiagnostics | None:
        if resolved is None:
            return None
        payload: dict[str, np.ndarray | None] = {}
        for field in dataclass_fields(ResolvedDiagnostics):
            value = getattr(resolved, field.name)
            payload[field.name] = (
                None if value is None else np.asarray(value)[:stop, ...]
            )
        return ResolvedDiagnostics(**payload)

    dt_t = np.asarray(diag.dt_t)[:stop]
    Wg_t = np.asarray(diag.Wg_t)[:stop]
    Wphi_t = np.asarray(diag.Wphi_t)[:stop]
    Wapar_t = np.asarray(diag.Wapar_t)[:stop]
    if dt_t.size == 0:
        dt_mean = np.asarray(0.0, dtype=float)
    else:
        dt_mean = np.asarray(np.mean(dt_t), dtype=float)
    return SimulationDiagnostics(
        t=np.asarray(diag.t)[:stop],
        dt_t=dt_t,
        dt_mean=dt_mean,
        gamma_t=np.asarray(diag.gamma_t)[:stop],
        omega_t=np.asarray(diag.omega_t)[:stop],
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=np.asarray(diag.heat_flux_t)[:stop],
        particle_flux_t=np.asarray(diag.particle_flux_t)[:stop],
        energy_t=np.asarray(
            total_energy(jnp.asarray(Wg_t), jnp.asarray(Wphi_t), jnp.asarray(Wapar_t))
        ),
        heat_flux_species_t=_slice_optional(diag.heat_flux_species_t),
        particle_flux_species_t=_slice_optional(diag.particle_flux_species_t),
        turbulent_heating_t=_slice_optional(diag.turbulent_heating_t),
        turbulent_heating_species_t=_slice_optional(diag.turbulent_heating_species_t),
        phi_mode_t=_slice_optional(diag.phi_mode_t),
        resolved=_slice_resolved(diag.resolved),
    )


def truncate_runtime_diagnostics(
    diag: SimulationDiagnostics, *, t_max: float
) -> SimulationDiagnostics:
    """Keep samples through the first entry that reaches ``t_max``."""

    t_arr = np.asarray(diag.t, dtype=float)
    if t_arr.size == 0:
        return diag
    stop = int(np.searchsorted(t_arr, float(t_max), side="left")) + 1
    stop = min(max(stop, 1), int(t_arr.size))
    return slice_runtime_diagnostics(diag, stop)


def stride_runtime_diagnostics(
    diag: SimulationDiagnostics, *, stride: int
) -> SimulationDiagnostics:
    """Apply the runtime output stride after concatenating chunk diagnostics."""

    stride_use = int(max(stride, 1))
    if stride_use == 1:
        return diag

    def _stride_optional(arr: np.ndarray | jnp.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        return np.asarray(arr)[::stride_use, ...]

    def _stride_resolved(
        resolved: ResolvedDiagnostics | None,
    ) -> ResolvedDiagnostics | None:
        if resolved is None:
            return None
        payload: dict[str, np.ndarray | None] = {}
        for field in dataclass_fields(ResolvedDiagnostics):
            value = getattr(resolved, field.name)
            payload[field.name] = (
                None if value is None else np.asarray(value)[::stride_use, ...]
            )
        return ResolvedDiagnostics(**payload)

    dt_t = np.asarray(diag.dt_t)[::stride_use]
    Wg_t = np.asarray(diag.Wg_t)[::stride_use]
    Wphi_t = np.asarray(diag.Wphi_t)[::stride_use]
    Wapar_t = np.asarray(diag.Wapar_t)[::stride_use]
    if dt_t.size == 0:
        dt_mean = np.asarray(0.0, dtype=float)
    else:
        dt_mean = np.asarray(np.mean(dt_t), dtype=float)
    return SimulationDiagnostics(
        t=np.asarray(diag.t)[::stride_use],
        dt_t=dt_t,
        dt_mean=dt_mean,
        gamma_t=np.asarray(diag.gamma_t)[::stride_use],
        omega_t=np.asarray(diag.omega_t)[::stride_use],
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=np.asarray(diag.heat_flux_t)[::stride_use],
        particle_flux_t=np.asarray(diag.particle_flux_t)[::stride_use],
        energy_t=np.asarray(
            total_energy(jnp.asarray(Wg_t), jnp.asarray(Wphi_t), jnp.asarray(Wapar_t))
        ),
        heat_flux_species_t=_stride_optional(diag.heat_flux_species_t),
        particle_flux_species_t=_stride_optional(diag.particle_flux_species_t),
        turbulent_heating_t=_stride_optional(diag.turbulent_heating_t),
        turbulent_heating_species_t=_stride_optional(diag.turbulent_heating_species_t),
        phi_mode_t=_stride_optional(diag.phi_mode_t),
        resolved=_stride_resolved(diag.resolved),
    )


def concat_runtime_diagnostics(
    diags: Sequence[SimulationDiagnostics],
) -> SimulationDiagnostics:
    """Concatenate one or more diagnostic chunks."""

    if not diags:
        raise ValueError("at least one diagnostic chunk is required")

    def _concat(name: str) -> np.ndarray:
        return np.concatenate(
            [np.asarray(getattr(diag, name)) for diag in diags], axis=0
        )

    def _concat_optional(name: str) -> np.ndarray | None:
        values = [getattr(diag, name) for diag in diags]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(
                f"inconsistent optional diagnostic {name}: every concatenated chunk must either provide it or omit it"
            )
        return np.concatenate(
            [np.asarray(value) for value in values if value is not None], axis=0
        )

    def _concat_resolved() -> ResolvedDiagnostics | None:
        values = [diag.resolved for diag in diags]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(
                "inconsistent resolved diagnostics: every concatenated chunk must either provide resolved data or omit it"
            )
        payload: dict[str, np.ndarray | None] = {}
        for field in dataclass_fields(ResolvedDiagnostics):
            series = [
                None if value is None else getattr(value, field.name)
                for value in values
            ]
            if all(item is None for item in series):
                payload[field.name] = None
            elif any(item is None for item in series):
                raise ValueError(
                    f"inconsistent resolved diagnostic {field.name}: every concatenated chunk must either provide it or omit it"
                )
            else:
                payload[field.name] = np.concatenate(
                    [np.asarray(item) for item in series if item is not None],
                    axis=0,
                )
        return ResolvedDiagnostics(**payload)

    dt_t = _concat("dt_t")
    Wg_t = _concat("Wg_t")
    Wphi_t = _concat("Wphi_t")
    Wapar_t = _concat("Wapar_t")
    dt_mean = np.asarray(np.mean(dt_t), dtype=float)
    return SimulationDiagnostics(
        t=_concat("t"),
        dt_t=dt_t,
        dt_mean=dt_mean,
        gamma_t=_concat("gamma_t"),
        omega_t=_concat("omega_t"),
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=_concat("heat_flux_t"),
        particle_flux_t=_concat("particle_flux_t"),
        energy_t=np.asarray(
            total_energy(jnp.asarray(Wg_t), jnp.asarray(Wphi_t), jnp.asarray(Wapar_t))
        ),
        heat_flux_species_t=_concat_optional("heat_flux_species_t"),
        particle_flux_species_t=_concat_optional("particle_flux_species_t"),
        turbulent_heating_t=_concat_optional("turbulent_heating_t"),
        turbulent_heating_species_t=_concat_optional("turbulent_heating_species_t"),
        phi_mode_t=_concat_optional("phi_mode_t"),
        resolved=_concat_resolved(),
    )
