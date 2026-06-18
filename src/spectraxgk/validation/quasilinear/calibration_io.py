"""Nonlinear-window artifact ingestion for quasilinear calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from spectraxgk.validation.quasilinear.calibration_core import QuasilinearCalibrationPoint
from spectraxgk.validation.quasilinear.calibration_spectrum import integrated_quasilinear_flux_from_spectrum
from spectraxgk.validation.quasilinear.window_config import (
    NonlinearWindowConvergenceConfig,
)
from spectraxgk.validation.quasilinear.window_promotion import (
    nonlinear_window_stats_promotion_ready,
)
from spectraxgk.validation.quasilinear.window_statistics import (
    nonlinear_window_convergence_report,
)

ROOT = Path(__file__).resolve().parents[2]

_NETCDF_HEAT_FLUX_COLUMNS = {
    "heat_flux": "Diagnostics/HeatFlux_st",
    "heat_flux_es": "Diagnostics/HeatFluxES_st",
    "heat_flux_apar": "Diagnostics/HeatFluxApar_st",
    "heat_flux_bpar": "Diagnostics/HeatFluxBpar_st",
}

def _resolve_summary_artifact(summary_path: Path, source: object) -> Path:
    diag_path = Path(str(source))
    if diag_path.is_absolute():
        return diag_path
    candidates = (
        (ROOT / diag_path).resolve(),
        (Path.cwd() / diag_path).resolve(),
        (summary_path.parent / diag_path).resolve(),
        (summary_path.parent.parent / diag_path).resolve(),
    )
    return next(
        (candidate for candidate in candidates if candidate.exists()), candidates[0]
    )


def _window_from_summary(summary: dict[str, Any], t: np.ndarray) -> tuple[float, float]:
    raw_tmin = summary.get("tmin")
    raw_tmax = summary.get("tmax")
    tmin = float(np.nanmin(t) if raw_tmin is None else raw_tmin)
    tmax = float(np.nanmax(t) if raw_tmax is None else raw_tmax)
    return tmin, tmax


def _read_csv_window_heat_flux(
    path: Path,
    summary: dict[str, Any],
    *,
    heat_flux_column: str,
) -> dict[str, Any]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    names = set(data.dtype.names or ())
    if "t" not in names:
        raise ValueError(f"{path} is missing a 't' column")
    if heat_flux_column not in names:
        raise ValueError(f"{path} is missing heat-flux column '{heat_flux_column}'")
    t = np.asarray(data["t"], dtype=float)
    heat = np.asarray(data[heat_flux_column], dtype=float)
    tmin, tmax = _window_from_summary(summary, t)
    mask = (t >= tmin) & (t <= tmax) & np.isfinite(heat)
    if not np.any(mask):
        raise ValueError(f"no finite heat-flux samples in [{tmin}, {tmax}] from {path}")
    return {
        "mean": float(np.mean(heat[mask])),
        "std": float(np.std(heat[mask])),
        "tmin": tmin,
        "tmax": tmax,
        "n_samples": int(np.count_nonzero(mask)),
        "variable": heat_flux_column,
    }


def _csv_heat_flux_trace(
    path: Path, *, heat_flux_column: str
) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    names = set(data.dtype.names or ())
    if "t" not in names:
        raise ValueError(f"{path} is missing a 't' column")
    if heat_flux_column not in names:
        raise ValueError(f"{path} is missing heat-flux column '{heat_flux_column}'")
    return np.asarray(data["t"], dtype=float), np.asarray(
        data[heat_flux_column], dtype=float
    )


def _netcdf_variable(root: Any, path: str) -> Any:
    current = root
    parts = [part for part in path.strip("/").split("/") if part]
    if not parts:
        raise ValueError("NetCDF variable path must not be empty")
    for group in parts[:-1]:
        if group not in current.groups:
            raise KeyError(f"NetCDF group '{group}' not found in '{path}'")
        current = current.groups[group]
    name = parts[-1]
    if name not in current.variables:
        raise KeyError(f"NetCDF variable '{name}' not found in '{path}'")
    return current.variables[name]


def _netcdf_heat_flux_variable(heat_flux_column: str) -> str:
    key = str(heat_flux_column).strip()
    if "/" in key:
        return key
    if key.startswith("Diagnostics"):
        return key
    if key in _NETCDF_HEAT_FLUX_COLUMNS:
        return _NETCDF_HEAT_FLUX_COLUMNS[key]
    if key.endswith("_st"):
        return f"Diagnostics/{key}"
    raise ValueError(
        f"unknown NetCDF heat-flux column '{heat_flux_column}'. "
        "Use one of heat_flux, heat_flux_es, heat_flux_apar, heat_flux_bpar, "
        "or an explicit NetCDF path such as Diagnostics/HeatFlux_st."
    )


def _read_netcdf_window_heat_flux(
    path: Path,
    summary: dict[str, Any],
    *,
    heat_flux_column: str,
    species_index: int | None,
) -> dict[str, Any]:
    try:
        import netCDF4
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only without optional dependency.
        raise RuntimeError(
            "netCDF4 is required to read nonlinear NetCDF calibration windows"
        ) from exc

    variable_path = _netcdf_heat_flux_variable(heat_flux_column)
    with netCDF4.Dataset(path) as root:
        t = np.asarray(_netcdf_variable(root, "Grids/time")[:], dtype=float)
        values = np.asarray(_netcdf_variable(root, variable_path)[:], dtype=float)
    if values.shape[0] != t.size:
        raise ValueError(
            f"{variable_path} first dimension does not match Grids/time in {path}"
        )
    if values.ndim == 1:
        heat = values
    elif values.ndim == 2:
        if species_index is None:
            heat = np.sum(values, axis=1)
        else:
            if species_index < 0 or species_index >= values.shape[1]:
                raise ValueError(
                    f"species_index {species_index} is out of bounds for {values.shape[1]} species"
                )
            heat = values[:, int(species_index)]
    else:
        raise ValueError(
            f"{variable_path} must have shape (time,) or (time, species), got {values.shape}"
        )
    tmin, tmax = _window_from_summary(summary, t)
    mask = (t >= tmin) & (t <= tmax) & np.isfinite(heat)
    if not np.any(mask):
        raise ValueError(f"no finite heat-flux samples in [{tmin}, {tmax}] from {path}")
    return {
        "mean": float(np.mean(heat[mask])),
        "std": float(np.std(heat[mask])),
        "tmin": tmin,
        "tmax": tmax,
        "n_samples": int(np.count_nonzero(mask)),
        "variable": variable_path,
    }


def _netcdf_heat_flux_trace(
    path: Path,
    *,
    heat_flux_column: str,
    species_index: int | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        import netCDF4
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only without optional dependency.
        raise RuntimeError(
            "netCDF4 is required to read nonlinear NetCDF calibration windows"
        ) from exc

    variable_path = _netcdf_heat_flux_variable(heat_flux_column)
    with netCDF4.Dataset(path) as root:
        t = np.asarray(_netcdf_variable(root, "Grids/time")[:], dtype=float)
        values = np.asarray(_netcdf_variable(root, variable_path)[:], dtype=float)
    if values.shape[0] != t.size:
        raise ValueError(
            f"{variable_path} first dimension does not match Grids/time in {path}"
        )
    if values.ndim == 1:
        heat = values
    elif values.ndim == 2:
        if species_index is None:
            heat = np.sum(values, axis=1)
        else:
            if species_index < 0 or species_index >= values.shape[1]:
                raise ValueError(
                    f"species_index {species_index} is out of bounds for {values.shape[1]} species"
                )
            heat = values[:, int(species_index)]
    else:
        raise ValueError(
            f"{variable_path} must have shape (time,) or (time, species), got {values.shape}"
        )
    return t, heat, variable_path


def _window_convergence_config(
    window: dict[str, Any],
    config: NonlinearWindowConvergenceConfig | None,
) -> NonlinearWindowConvergenceConfig:
    if config is not None:
        return config
    return NonlinearWindowConvergenceConfig(
        tmin=window.get("tmin"),
        tmax=window.get("tmax"),
        transient_fraction=0.0,
    )


def calibration_point_from_nonlinear_window_summary(
    summary_json: str | Path,
    *,
    predicted_heat_flux: float,
    split: str,
    saturation_rule: str,
    diagnostics_source: str = "spectrax",
    heat_flux_column: str = "heat_flux",
    case: str | None = None,
    geometry: str = "unspecified",
    electron_model: str = "unspecified",
    quasilinear_artifact: str | None = None,
    species_index: int | None = None,
    window_convergence_config: NonlinearWindowConvergenceConfig | None = None,
    notes: str | None = None,
) -> QuasilinearCalibrationPoint:
    """Create a calibration point from a nonlinear window-summary JSON.

    The helper reads the window bounds from a tracked nonlinear gate summary and
    computes the mean/std of a heat-flux column from the selected diagnostics
    CSV or runtime NetCDF. For NetCDF inputs, ``heat_flux_column='heat_flux'``
    maps to ``Diagnostics/HeatFlux_st`` and species are summed by default.
    """

    summary_path = Path(summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("kind") == "nonlinear_window_ensemble_report":
        if not bool(summary.get("passed", False)):
            raise ValueError("nonlinear ensemble summary did not pass")
        ready, failures = nonlinear_window_stats_promotion_ready(summary)
        if not ready:
            raise ValueError(
                "nonlinear ensemble summary is not promotion-ready: "
                + "; ".join(failures)
            )
        statistics = summary.get("statistics", {})
        observed_mean = float(statistics["ensemble_mean"])
        observed_sem = float(statistics["combined_sem"])
        note_items = [
            notes,
            "nonlinear_source=replicated_ensemble_gate",
            f"nonlinear_ensemble_reports={statistics.get('n_reports')}",
            f"nonlinear_ensemble_combined_sem_rel={statistics.get('combined_sem_rel')}",
        ]
        return QuasilinearCalibrationPoint(
            case=str(case or summary.get("case", summary_path.stem)),
            split=str(split),
            predicted_heat_flux=float(predicted_heat_flux),
            observed_heat_flux=observed_mean,
            observed_heat_flux_std=observed_sem,
            nonlinear_window_stats=summary,
            saturation_rule=str(saturation_rule),
            geometry=str(geometry),
            electron_model=str(electron_model),
            quasilinear_artifact=quasilinear_artifact,
            nonlinear_artifact=str(summary_path),
            notes="; ".join(str(item) for item in note_items if item),
        )
    source = summary.get(diagnostics_source)
    if source is None:
        raise ValueError(
            f"summary does not contain diagnostics source '{diagnostics_source}'"
        )
    diag_path = _resolve_summary_artifact(summary_path, source)
    suffixes = [suffix.lower() for suffix in diag_path.suffixes]
    if diag_path.suffix.lower() == ".csv":
        window = _read_csv_window_heat_flux(
            diag_path, summary, heat_flux_column=heat_flux_column
        )
        trace_t, trace_heat = _csv_heat_flux_trace(
            diag_path, heat_flux_column=heat_flux_column
        )
        convergence_variable = heat_flux_column
    elif suffixes[-2:] == [".out", ".nc"] or diag_path.suffix.lower() == ".nc":
        window = _read_netcdf_window_heat_flux(
            diag_path,
            summary,
            heat_flux_column=heat_flux_column,
            species_index=species_index,
        )
        trace_t, trace_heat, convergence_variable = _netcdf_heat_flux_trace(
            diag_path,
            heat_flux_column=heat_flux_column,
            species_index=species_index,
        )
    else:
        raise NotImplementedError(
            "nonlinear calibration ingestion supports diagnostics CSV and NetCDF files"
        )
    convergence = nonlinear_window_convergence_report(
        trace_t,
        trace_heat,
        case=str(case or summary.get("case", summary_path.stem)),
        observable=str(convergence_variable),
        source_artifact=str(diag_path),
        summary_artifact=str(summary_path),
        config=_window_convergence_config(window, window_convergence_config),
    )
    note_items = [
        notes,
        None
        if diag_path.suffix.lower() == ".csv"
        else f"nonlinear_variable={window['variable']}",
        f"nonlinear_window=[{window['tmin']:.12g},{window['tmax']:.12g}]",
        f"nonlinear_window_samples={window['n_samples']}",
    ]
    return QuasilinearCalibrationPoint(
        case=str(case or summary.get("case", summary_path.stem)),
        split=str(split),
        predicted_heat_flux=float(predicted_heat_flux),
        observed_heat_flux=float(window["mean"]),
        observed_heat_flux_std=float(window["std"]),
        nonlinear_window_stats=convergence,
        saturation_rule=str(saturation_rule),
        geometry=str(geometry),
        electron_model=str(electron_model),
        quasilinear_artifact=quasilinear_artifact,
        nonlinear_artifact=str(diag_path),
        notes="; ".join(str(item) for item in note_items if item),
    )


def calibration_point_from_spectrum_and_nonlinear_window(
    spectrum_csv: str | Path,
    summary_json: str | Path,
    *,
    split: str,
    saturation_rule: str,
    spectrum_column: str = "saturated_heat_flux_total",
    spectrum_method: str = "sum",
    delta_ky: float | None = None,
    diagnostics_source: str = "spectrax",
    heat_flux_column: str = "heat_flux",
    case: str | None = None,
    geometry: str = "unspecified",
    electron_model: str = "unspecified",
    species_index: int | None = None,
    window_convergence_config: NonlinearWindowConvergenceConfig | None = None,
    notes: str | None = None,
) -> QuasilinearCalibrationPoint:
    """Create a calibration point from a quasilinear spectrum and nonlinear window."""

    estimate = integrated_quasilinear_flux_from_spectrum(
        spectrum_csv,
        column=spectrum_column,
        method=spectrum_method,
        delta_ky=delta_ky,
    )
    point = calibration_point_from_nonlinear_window_summary(
        summary_json,
        predicted_heat_flux=float(estimate["estimate"]),
        split=split,
        saturation_rule=saturation_rule,
        diagnostics_source=diagnostics_source,
        heat_flux_column=heat_flux_column,
        case=case,
        geometry=geometry,
        electron_model=electron_model,
        quasilinear_artifact=str(spectrum_csv),
        species_index=species_index,
        window_convergence_config=window_convergence_config,
        notes=notes,
    )
    ratio = None
    if point.predicted_heat_flux != 0.0 and np.isfinite(point.predicted_heat_flux):
        ratio = point.observed_heat_flux / point.predicted_heat_flux
    merged_notes = [
        item
        for item in (
            point.notes,
            f"ql_spectrum_method={estimate['method']}",
            f"ql_spectrum_column={estimate['column']}",
            None if ratio is None else f"observed_to_predicted={ratio:.6g}",
        )
        if item
    ]
    return QuasilinearCalibrationPoint(
        **{
            **point.to_dict(),
            "notes": "; ".join(merged_notes),
        }
    )


def write_quasilinear_calibration_report(
    path: str | Path, report: dict[str, Any]
) -> Path:
    """Write a quasilinear calibration report to JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out



__all__ = [
    "calibration_point_from_nonlinear_window_summary",
    "calibration_point_from_spectrum_and_nonlinear_window",
    "write_quasilinear_calibration_report",
]
