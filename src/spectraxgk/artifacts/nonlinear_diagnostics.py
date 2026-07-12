"""Nonlinear runtime diagnostic loading and restart-path helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from spectraxgk.diagnostics import (
    ResolvedDiagnostics,
    SimulationDiagnostics,
    total_energy,
)
from spectraxgk.artifacts.spectral_layout import (
    _condense_kx,
    _condense_ky,
    _condense_kykx,
    _require_netcdf4,
)
from spectraxgk.artifacts.io import _netcdf_bundle_base


def _resolved_species_time(arr: Any | None, *, fallback: np.ndarray) -> np.ndarray:
    if arr is None:
        return np.asarray(fallback, dtype=np.float32)
    return np.sum(np.asarray(arr, dtype=np.float32), axis=-1)


def _read_optional_var(group: Any, name: str) -> np.ndarray | None:
    if name not in group.variables:
        return None
    var = group.variables[name]
    arr = np.asarray(var[:])
    dims = tuple(getattr(var, "dimensions", ()))
    if dims and dims[-1] == "ri":
        return np.asarray(arr[..., 0] + 1j * arr[..., 1])
    return arr


def _resolve_restart_path(out: str | Path, cfg: Any, *, for_write: bool) -> Path:
    configured = (
        cfg.output.restart_to_file if for_write else cfg.output.restart_from_file
    )
    if configured is not None:
        return Path(configured)
    base = _netcdf_bundle_base(Path(out))
    return Path(f"{base}.restart.nc")


def _condense_resolved_for_output(
    resolved: ResolvedDiagnostics | None,
) -> ResolvedDiagnostics | None:
    if resolved is None:
        return None
    payload: dict[str, np.ndarray | None] = {}
    for field in ResolvedDiagnostics.__dataclass_fields__.values():
        value = getattr(resolved, field.name)
        if value is None:
            payload[field.name] = None
        elif field.name.endswith(("_kxt", "_kxst")):
            payload[field.name] = _condense_kx(np.asarray(value))
        elif field.name.endswith(("_kyt", "_kyst")):
            payload[field.name] = _condense_ky(np.asarray(value))
        elif field.name.endswith(("_kxkyt", "_kxkyst")):
            payload[field.name] = _condense_kykx(np.asarray(value))
        else:
            payload[field.name] = np.asarray(value)
    return ResolvedDiagnostics(**payload)


def _condense_diagnostics_for_netcdf_output(
    diag: SimulationDiagnostics,
) -> SimulationDiagnostics:
    # Nonlinear NetCDF output artifacts do not persist the monitored complex mode trace.
    # Drop it when appending from an existing artifact so restart concatenation
    # preserves the exact on-disk schema instead of mixing persisted and transient
    # diagnostics.
    return replace(
        diag, phi_mode_t=None, resolved=_condense_resolved_for_output(diag.resolved)
    )


def load_nonlinear_netcdf_diagnostics(path: str | Path) -> SimulationDiagnostics:
    Dataset = _require_netcdf4()
    with Dataset(Path(path), "r") as root:
        grids = root.groups["Grids"]
        diag_group = root.groups["Diagnostics"]
        time_vals = np.asarray(grids.variables["time"][:], dtype=np.float64)
        wg_st = np.asarray(diag_group.variables["Wg_st"][:], dtype=np.float32)
        wphi_st = np.asarray(diag_group.variables["Wphi_st"][:], dtype=np.float32)
        wapar_st = np.asarray(diag_group.variables["Wapar_st"][:], dtype=np.float32)
        heat_st = np.asarray(diag_group.variables["HeatFlux_st"][:], dtype=np.float32)
        pflux_st = np.asarray(
            diag_group.variables["ParticleFlux_st"][:], dtype=np.float32
        )
        turb_heat_st = _read_optional_var(diag_group, "TurbulentHeating_st")
        resolved_payload = {
            field.name: _read_optional_var(diag_group, field.name)
            for field in ResolvedDiagnostics.__dataclass_fields__.values()
        }
    if turb_heat_st is None:
        turb_heat_st = np.zeros_like(heat_st)
    dt_t = (
        np.diff(np.concatenate(([0.0], time_vals)))
        if time_vals.size
        else np.asarray([], dtype=np.float64)
    )
    dt_mean = np.asarray(
        np.mean(dt_t[dt_t > 0.0]) if np.any(dt_t > 0.0) else 0.0, dtype=np.float64
    )
    Wg_t = np.sum(wg_st, axis=1)
    Wphi_t = np.sum(wphi_st, axis=1)
    Wapar_t = np.sum(wapar_st, axis=1)
    heat_t = np.sum(heat_st, axis=1)
    pflux_t = np.sum(pflux_st, axis=1)
    turb_heat_t = np.sum(np.asarray(turb_heat_st, dtype=np.float32), axis=1)
    return SimulationDiagnostics(
        t=time_vals,
        dt_t=dt_t,
        dt_mean=dt_mean,
        gamma_t=np.zeros_like(time_vals, dtype=np.float32),
        omega_t=np.zeros_like(time_vals, dtype=np.float32),
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=heat_t,
        particle_flux_t=pflux_t,
        energy_t=np.asarray(total_energy(Wg_t, Wphi_t, Wapar_t), dtype=np.float32),
        heat_flux_species_t=heat_st,
        particle_flux_species_t=pflux_st,
        turbulent_heating_t=turb_heat_t,
        turbulent_heating_species_t=np.asarray(turb_heat_st, dtype=np.float32),
        phi_mode_t=None,
        resolved=ResolvedDiagnostics(**resolved_payload),
    )




@dataclass(frozen=True)
class DiagnosticTimeSeries:
    """Single benchmark-facing time series loaded from an ``out.nc`` artifact."""

    t: np.ndarray
    values: np.ndarray
    variable: str
    source_path: str


def _decode_netcdf_values(var) -> np.ndarray:
    raw = np.asarray(var[:])
    dims = tuple(getattr(var, "dimensions", ()))
    if dims and dims[-1] == "ri":
        return np.asarray(raw[..., 0] + 1j * raw[..., 1], dtype=np.complex128)
    return raw


def _extract_diagnostic_values(
    values: np.ndarray,
    *,
    variable: str,
    kx_index: int | None,
) -> np.ndarray:
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        if kx_index is None:
            raise ValueError(
                f"diagnostics variable {variable!r} requires kx_index for 2D extraction"
            )
        return values[:, int(kx_index)]
    raise ValueError(
        f"diagnostics variable {variable!r} must reduce to a 1D time series"
    )


def _load_netcdf_time_axis(
    ds,
    *,
    src: Path,
    time_group: str,
    time_var: str,
) -> np.ndarray:
    if time_group in ds.groups and time_var in ds.groups[time_group].variables:
        return np.asarray(ds.groups[time_group].variables[time_var][:], dtype=float)
    if time_var in ds.variables:
        return np.asarray(ds.variables[time_var][:], dtype=float)
    raise ValueError(f"missing time variable {time_group}/{time_var} in {src}")


def _load_diagnostic_variable(
    ds,
    *,
    src: Path,
    diagnostics_group: str,
    variable: str,
    kx_index: int | None,
) -> np.ndarray:
    diag_group = ds.groups.get(diagnostics_group)
    if diag_group is None:
        raise ValueError(f"missing NetCDF group {diagnostics_group!r} in {src}")
    if variable not in diag_group.variables:
        raise ValueError(f"missing diagnostics variable {variable!r} in {src}")
    raw = _decode_netcdf_values(diag_group.variables[variable])
    return _extract_diagnostic_values(raw, variable=variable, kx_index=kx_index)


def _align_complex_phase(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    nz = finite & (np.abs(values) > 1.0e-30)
    if np.any(nz):
        first = values[np.flatnonzero(nz)[0]]
        return values * np.exp(-1j * np.angle(first))
    return values


def _select_complex_component(
    values: np.ndarray,
    *,
    component: str,
    align_phase: bool,
) -> np.ndarray:
    values_arr = _align_complex_phase(values) if align_phase else values
    component_key = str(component).lower()
    if component_key == "complex":
        return values_arr
    if component_key == "real":
        return np.real(values_arr)
    if component_key == "imag":
        return np.imag(values_arr)
    if component_key == "abs":
        return np.abs(values_arr)
    raise ValueError("component must be one of {'real', 'imag', 'abs', 'complex'}")


def _select_real_component(values: np.ndarray, *, component: str) -> np.ndarray:
    if component not in {"real", "abs"}:
        raise ValueError("real diagnostics only support component='real' or 'abs'")
    if component == "abs":
        return np.abs(values)
    return np.asarray(values, dtype=float)


def _select_series_component(
    values: np.ndarray,
    *,
    component: str,
    align_phase: bool,
) -> np.ndarray:
    values_arr = np.asarray(values)
    if np.iscomplexobj(values_arr):
        return _select_complex_component(
            values_arr,
            component=component,
            align_phase=align_phase,
        )
    return _select_real_component(values_arr, component=component)


def load_diagnostic_time_series(
    path: str | Path,
    *,
    variable: str,
    diagnostics_group: str = "Diagnostics",
    time_group: str = "Grids",
    time_var: str = "time",
    kx_index: int | None = None,
    component: str = "real",
    align_phase: bool = False,
) -> DiagnosticTimeSeries:
    """Load a 1D diagnostics time series from a grouped NetCDF output artifact."""

    src = Path(path)
    import netCDF4 as nc

    with nc.Dataset(src) as ds:
        values = _load_diagnostic_variable(
            ds,
            src=src,
            diagnostics_group=diagnostics_group,
            variable=variable,
            kx_index=kx_index,
        )
        t = _load_netcdf_time_axis(
            ds,
            src=src,
            time_group=time_group,
            time_var=time_var,
        )

    if t.ndim != 1 or t.size != values.size:
        raise ValueError(
            f"time axis for {variable!r} must be one-dimensional and match the diagnostics length"
        )
    selected = _select_series_component(
        values,
        component=component,
        align_phase=align_phase,
    )

    return DiagnosticTimeSeries(
        t=t,
        values=np.asarray(selected),
        variable=str(variable),
        source_path=str(src),
    )


__all__ = [
    "DiagnosticTimeSeries",
    "load_diagnostic_time_series",
    "_condense_diagnostics_for_netcdf_output",
    "_condense_resolved_for_output",
    "_read_optional_var",
    "_resolved_species_time",
    "_resolve_restart_path",
    "load_nonlinear_netcdf_diagnostics",
]
