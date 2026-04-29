"""Calibration artifact helpers for quasilinear transport models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np


_NETCDF_HEAT_FLUX_COLUMNS = {
    "heat_flux": "Diagnostics/HeatFlux_st",
    "heat_flux_es": "Diagnostics/HeatFluxES_st",
    "heat_flux_apar": "Diagnostics/HeatFluxApar_st",
    "heat_flux_bpar": "Diagnostics/HeatFluxBpar_st",
}


@dataclass(frozen=True)
class QuasilinearCalibrationPoint:
    """One quasilinear-vs-nonlinear transport comparison point."""

    case: str
    split: str
    predicted_heat_flux: float
    observed_heat_flux: float
    saturation_rule: str
    raw_predicted_heat_flux: float | None = None
    calibration_scale: float | None = None
    geometry: str = "unspecified"
    electron_model: str = "unspecified"
    ky: float | None = None
    observed_heat_flux_std: float | None = None
    quasilinear_artifact: str | None = None
    nonlinear_artifact: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _point_from_mapping(item: QuasilinearCalibrationPoint | dict[str, Any]) -> QuasilinearCalibrationPoint:
    if isinstance(item, QuasilinearCalibrationPoint):
        return item
    return QuasilinearCalibrationPoint(**dict(item))


def _split_metrics(points: list[QuasilinearCalibrationPoint], *, observed_floor: float) -> dict[str, Any]:
    if not points:
        return {
            "n": 0,
            "rmse": None,
            "mean_abs_relative_error": None,
            "max_abs_relative_error": None,
        }
    pred = np.asarray([p.predicted_heat_flux for p in points], dtype=float)
    obs = np.asarray([p.observed_heat_flux for p in points], dtype=float)
    residual = pred - obs
    denom = np.maximum(np.abs(obs), float(observed_floor))
    rel = np.abs(residual) / denom
    return {
        "n": int(pred.size),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_abs_relative_error": float(np.mean(rel)),
        "max_abs_relative_error": float(np.max(rel)),
    }


def fit_train_heat_flux_scale(
    points: Iterable[QuasilinearCalibrationPoint | dict[str, Any]],
    *,
    train_split: str = "train",
    prediction_floor: float = 1.0e-300,
) -> dict[str, Any]:
    """Fit one multiplicative heat-flux scale from training points.

    The fit is a through-origin least-squares estimate,
    ``scale = sum(q_i Q_i) / sum(q_i^2)``, where ``q_i`` is the raw
    quasilinear heat-flux estimate and ``Q_i`` is the nonlinear window mean.
    This is the minimal calibration constant used by simple mixing-length
    models; any held-out failure after this fit is therefore a model failure,
    not a missing constant factor.
    """

    pts = [_point_from_mapping(item) for item in points]
    train = [
        p
        for p in pts
        if p.split == train_split
        and np.isfinite(p.predicted_heat_flux)
        and np.isfinite(p.observed_heat_flux)
        and abs(p.predicted_heat_flux) > prediction_floor
    ]
    if not train:
        raise ValueError(f"no finite nonzero '{train_split}' points available for scale fit")
    pred = np.asarray([p.predicted_heat_flux for p in train], dtype=float)
    obs = np.asarray([p.observed_heat_flux for p in train], dtype=float)
    denom = float(np.dot(pred, pred))
    if not np.isfinite(denom) or denom <= prediction_floor:
        raise ValueError("training predictions are too small to fit a stable scale")
    scale = float(np.dot(pred, obs) / denom)
    if scale < 0.0:
        raise ValueError("fitted heat-flux scale is negative; do not treat this as a saturation constant")
    scaled_residual = scale * pred - obs
    return {
        "scale": scale,
        "train_split": str(train_split),
        "n_train": int(pred.size),
        "fit_kind": "through_origin_least_squares",
        "prediction_floor": float(prediction_floor),
        "train_rmse": float(np.sqrt(np.mean(scaled_residual**2))),
    }


def apply_heat_flux_scale(
    points: Iterable[QuasilinearCalibrationPoint | dict[str, Any]],
    *,
    scale: float,
    note_label: str = "heat_flux_scale",
) -> list[QuasilinearCalibrationPoint]:
    """Return calibration points with heat-flux predictions multiplied by ``scale``."""

    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("scale must be finite and non-negative")
    scaled = []
    for item in points:
        point = _point_from_mapping(item)
        notes = [point.notes, f"{note_label}={scale:.12g}"]
        scaled.append(
            QuasilinearCalibrationPoint(
                **{
                    **point.to_dict(),
                    "predicted_heat_flux": float(scale * point.predicted_heat_flux),
                    "raw_predicted_heat_flux": float(point.predicted_heat_flux),
                    "calibration_scale": float(scale),
                    "notes": "; ".join(str(note) for note in notes if note),
                }
            )
        )
    return scaled


def quasilinear_calibration_report(
    points: Iterable[QuasilinearCalibrationPoint | dict[str, Any]],
    *,
    saturation_rule: str,
    version: str = "0.1",
    holdout_mean_rel_gate: float = 0.35,
    observed_floor: float = 1.0e-12,
    fit_train_scale: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-friendly calibration/holdout report.

    A report is considered a calibrated absolute-flux claim only when it has at
    least one training point, at least one holdout point, and the holdout mean
    relative error passes the supplied gate.
    """

    pts = [_point_from_mapping(item) for item in points]
    if not pts:
        raise ValueError("at least one calibration point is required")
    if observed_floor <= 0.0:
        raise ValueError("observed_floor must be positive")
    if holdout_mean_rel_gate <= 0.0:
        raise ValueError("holdout_mean_rel_gate must be positive")
    allowed_splits = {"train", "holdout", "audit"}
    bad_splits = sorted({p.split for p in pts if p.split not in allowed_splits})
    if bad_splits:
        raise ValueError(f"unsupported calibration split(s): {bad_splits}")

    scale_fit = None
    if fit_train_scale:
        scale_fit = fit_train_heat_flux_scale(pts)
        pts = apply_heat_flux_scale(
            pts,
            scale=float(scale_fit["scale"]),
            note_label="train_fitted_heat_flux_scale",
        )

    by_split_points = {split: [p for p in pts if p.split == split] for split in allowed_splits}
    by_split = {
        split: _split_metrics(split_points, observed_floor=observed_floor)
        for split, split_points in sorted(by_split_points.items())
    }
    all_metrics = _split_metrics(pts, observed_floor=observed_floor)
    holdout = by_split["holdout"]
    has_train = by_split["train"]["n"] > 0
    has_holdout = holdout["n"] > 0
    holdout_error = holdout["mean_abs_relative_error"]
    passed = bool(has_train and has_holdout and holdout_error is not None and holdout_error <= holdout_mean_rel_gate)
    claim_level = "calibrated_absolute_flux" if passed else "calibration_dataset"
    if not has_holdout:
        claim_level = "training_or_audit_only"

    return {
        "kind": "quasilinear_calibration_report",
        "version": str(version),
        "saturation_rule": str(saturation_rule),
        "claim_level": claim_level,
        "passed": passed,
        "holdout_mean_rel_gate": float(holdout_mean_rel_gate),
        "observed_floor": float(observed_floor),
        "metrics": all_metrics,
        "by_split": by_split,
        "points": [p.to_dict() for p in pts],
        "metadata": {
            **dict(metadata or {}),
            **({} if scale_fit is None else {"heat_flux_scale_fit": scale_fit}),
        },
    }


def integrated_quasilinear_flux_from_spectrum(
    spectrum_csv: str | Path,
    *,
    column: str = "saturated_heat_flux_total",
    ky_column: str = "ky",
    method: str = "sum",
    delta_ky: float | None = None,
) -> dict[str, Any]:
    """Integrate one quasilinear spectrum column into a scalar flux estimate.

    ``method="sum"`` preserves the discrete spectral-sum convention used by
    most runtime diagnostics. ``method="trapezoid"`` is available for smooth
    scan studies where the CSV is treated as a sampled function of ``ky``.
    """

    path = Path(spectrum_csv)
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    names = set(data.dtype.names or ())
    if column not in names:
        raise ValueError(f"{path} is missing quasilinear column '{column}'")
    values = np.asarray(data[column], dtype=float)
    finite = np.isfinite(values)
    if ky_column in names:
        ky = np.asarray(data[ky_column], dtype=float)
        finite &= np.isfinite(ky)
    else:
        ky = np.arange(values.size, dtype=float)
    if not np.any(finite):
        raise ValueError(f"{path} contains no finite samples in column '{column}'")
    values = values[finite]
    ky = ky[finite]

    method_key = method.strip().lower()
    if method_key == "sum":
        estimate = float(np.sum(values))
        if delta_ky is not None:
            estimate *= float(delta_ky)
    elif method_key == "mean":
        estimate = float(np.mean(values))
    elif method_key == "trapezoid":
        if values.size < 2:
            raise ValueError("trapezoid integration requires at least two finite spectrum samples")
        order = np.argsort(ky)
        estimate = float(np.trapezoid(values[order], ky[order]))
    else:
        raise ValueError("method must be one of {'sum', 'mean', 'trapezoid'}")

    return {
        "estimate": estimate,
        "method": method_key,
        "column": str(column),
        "ky_column": str(ky_column),
        "delta_ky": None if delta_ky is None else float(delta_ky),
        "n_samples": int(values.size),
        "ky_min": float(np.min(ky)),
        "ky_max": float(np.max(ky)),
        "artifact": str(path),
    }


def _resolve_summary_artifact(summary_path: Path, source: object) -> Path:
    diag_path = Path(str(source))
    if diag_path.is_absolute():
        return diag_path
    candidates = (
        (summary_path.parent / diag_path).resolve(),
        (summary_path.parent.parent / diag_path).resolve(),
        (Path.cwd() / diag_path).resolve(),
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


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
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
        raise RuntimeError("netCDF4 is required to read nonlinear NetCDF calibration windows") from exc

    variable_path = _netcdf_heat_flux_variable(heat_flux_column)
    with netCDF4.Dataset(path) as root:
        t = np.asarray(_netcdf_variable(root, "Grids/time")[:], dtype=float)
        values = np.asarray(_netcdf_variable(root, variable_path)[:], dtype=float)
    if values.shape[0] != t.size:
        raise ValueError(f"{variable_path} first dimension does not match Grids/time in {path}")
    if values.ndim == 1:
        heat = values
    elif values.ndim == 2:
        if species_index is None:
            heat = np.sum(values, axis=1)
        else:
            if species_index < 0 or species_index >= values.shape[1]:
                raise ValueError(f"species_index {species_index} is out of bounds for {values.shape[1]} species")
            heat = values[:, int(species_index)]
    else:
        raise ValueError(f"{variable_path} must have shape (time,) or (time, species), got {values.shape}")
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
    source = summary.get(diagnostics_source)
    if source is None:
        raise ValueError(f"summary does not contain diagnostics source '{diagnostics_source}'")
    diag_path = _resolve_summary_artifact(summary_path, source)
    suffixes = [suffix.lower() for suffix in diag_path.suffixes]
    if diag_path.suffix.lower() == ".csv":
        window = _read_csv_window_heat_flux(diag_path, summary, heat_flux_column=heat_flux_column)
    elif suffixes[-2:] == [".out", ".nc"] or diag_path.suffix.lower() == ".nc":
        window = _read_netcdf_window_heat_flux(
            diag_path,
            summary,
            heat_flux_column=heat_flux_column,
            species_index=species_index,
        )
    else:
        raise NotImplementedError("nonlinear calibration ingestion supports diagnostics CSV and NetCDF files")
    note_items = [
        notes,
        None if diag_path.suffix.lower() == ".csv" else f"nonlinear_variable={window['variable']}",
        f"nonlinear_window=[{window['tmin']:.12g},{window['tmax']:.12g}]",
        f"nonlinear_window_samples={window['n_samples']}",
    ]
    return QuasilinearCalibrationPoint(
        case=str(case or summary.get("case", summary_path.stem)),
        split=str(split),
        predicted_heat_flux=float(predicted_heat_flux),
        observed_heat_flux=float(window["mean"]),
        observed_heat_flux_std=float(window["std"]),
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


def write_quasilinear_calibration_report(path: str | Path, report: dict[str, Any]) -> Path:
    """Write a quasilinear calibration report to JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


__all__ = [
    "QuasilinearCalibrationPoint",
    "apply_heat_flux_scale",
    "calibration_point_from_nonlinear_window_summary",
    "calibration_point_from_spectrum_and_nonlinear_window",
    "fit_train_heat_flux_scale",
    "integrated_quasilinear_flux_from_spectrum",
    "quasilinear_calibration_report",
    "write_quasilinear_calibration_report",
]
