"""Calibration artifact helpers for quasilinear transport models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np


@dataclass(frozen=True)
class QuasilinearCalibrationPoint:
    """One quasilinear-vs-nonlinear transport comparison point."""

    case: str
    split: str
    predicted_heat_flux: float
    observed_heat_flux: float
    saturation_rule: str
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


def quasilinear_calibration_report(
    points: Iterable[QuasilinearCalibrationPoint | dict[str, Any]],
    *,
    saturation_rule: str,
    version: str = "0.1",
    holdout_mean_rel_gate: float = 0.35,
    observed_floor: float = 1.0e-12,
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
        "metadata": dict(metadata or {}),
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
    notes: str | None = None,
) -> QuasilinearCalibrationPoint:
    """Create a calibration point from a nonlinear window-summary JSON.

    The helper reads the window bounds from a tracked nonlinear gate summary and
    computes the mean/std of a heat-flux column from the selected diagnostics CSV.
    It is intentionally conservative: NetCDF-only summaries are rejected until
    a dedicated reader supplies the same windowed observable contract.
    """

    summary_path = Path(summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source = summary.get(diagnostics_source)
    if source is None:
        raise ValueError(f"summary does not contain diagnostics source '{diagnostics_source}'")
    diag_path = Path(str(source))
    if not diag_path.is_absolute():
        candidates = (
            (summary_path.parent / diag_path).resolve(),
            (summary_path.parent.parent / diag_path).resolve(),
            (Path.cwd() / diag_path).resolve(),
        )
        diag_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if diag_path.suffix.lower() != ".csv":
        raise NotImplementedError("nonlinear calibration ingestion currently supports diagnostics CSV files")
    data = np.genfromtxt(diag_path, delimiter=",", names=True)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    names = set(data.dtype.names or ())
    if "t" not in names:
        raise ValueError(f"{diag_path} is missing a 't' column")
    if heat_flux_column not in names:
        raise ValueError(f"{diag_path} is missing heat-flux column '{heat_flux_column}'")
    t = np.asarray(data["t"], dtype=float)
    heat = np.asarray(data[heat_flux_column], dtype=float)
    tmin = float(summary.get("tmin", np.min(t)))
    tmax = float(summary.get("tmax", np.max(t)))
    mask = (t >= tmin) & (t <= tmax) & np.isfinite(heat)
    if not np.any(mask):
        raise ValueError(f"no finite heat-flux samples in [{tmin}, {tmax}] from {diag_path}")
    return QuasilinearCalibrationPoint(
        case=str(case or summary.get("case", summary_path.stem)),
        split=str(split),
        predicted_heat_flux=float(predicted_heat_flux),
        observed_heat_flux=float(np.mean(heat[mask])),
        observed_heat_flux_std=float(np.std(heat[mask])),
        saturation_rule=str(saturation_rule),
        geometry=str(geometry),
        electron_model=str(electron_model),
        quasilinear_artifact=quasilinear_artifact,
        nonlinear_artifact=str(diag_path),
        notes=notes,
    )


def write_quasilinear_calibration_report(path: str | Path, report: dict[str, Any]) -> Path:
    """Write a quasilinear calibration report to JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


__all__ = [
    "QuasilinearCalibrationPoint",
    "calibration_point_from_nonlinear_window_summary",
    "quasilinear_calibration_report",
    "write_quasilinear_calibration_report",
]
