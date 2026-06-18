"""Core quasilinear calibration point, scaling, and report helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from spectraxgk.validation.quasilinear.window import nonlinear_window_stats_promotion_ready

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
    nonlinear_window_stats: dict[str, Any] | None = None
    quasilinear_artifact: str | None = None
    nonlinear_artifact: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _point_from_mapping(
    item: QuasilinearCalibrationPoint | dict[str, Any],
) -> QuasilinearCalibrationPoint:
    if isinstance(item, QuasilinearCalibrationPoint):
        return item
    return QuasilinearCalibrationPoint(**dict(item))


def _split_metrics(
    points: list[QuasilinearCalibrationPoint], *, observed_floor: float
) -> dict[str, Any]:
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


def _holdout_window_convergence_summary(
    points: list[QuasilinearCalibrationPoint],
) -> dict[str, Any]:
    holdouts = [point for point in points if point.split == "holdout"]
    failures: list[str] = []
    passed_cases: list[str] = []
    for point in holdouts:
        ready, point_failures = nonlinear_window_stats_promotion_ready(
            point.nonlinear_window_stats
        )
        if ready:
            passed_cases.append(point.case)
        else:
            failures.extend(f"{point.case}: {failure}" for failure in point_failures)
    return {
        "passed": bool(holdouts) and not failures,
        "n_holdout": len(holdouts),
        "n_passed": len(passed_cases),
        "passed_cases": passed_cases,
        "failures": failures,
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

    floor = float(prediction_floor)
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("prediction_floor must be finite and non-negative")
    pts = [_point_from_mapping(item) for item in points]
    train = [
        p
        for p in pts
        if p.split == train_split
        and np.isfinite(p.predicted_heat_flux)
        and np.isfinite(p.observed_heat_flux)
        and abs(p.predicted_heat_flux) > floor
    ]
    if not train:
        raise ValueError(
            f"no finite nonzero '{train_split}' points available for scale fit"
        )
    pred = np.asarray([p.predicted_heat_flux for p in train], dtype=float)
    obs = np.asarray([p.observed_heat_flux for p in train], dtype=float)
    denom = float(np.dot(pred, pred))
    if not np.isfinite(denom) or denom <= floor:
        raise ValueError("training predictions are too small to fit a stable scale")
    scale = float(np.dot(pred, obs) / denom)
    if scale < 0.0:
        raise ValueError(
            "fitted heat-flux scale is negative; do not treat this as a saturation constant"
        )
    scaled_residual = scale * pred - obs
    return {
        "scale": scale,
        "train_split": str(train_split),
        "n_train": int(pred.size),
        "fit_kind": "through_origin_least_squares",
        "prediction_floor": floor,
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
    nonfinite_cases = [
        p.case
        for p in pts
        if not np.isfinite(p.predicted_heat_flux)
        or not np.isfinite(p.observed_heat_flux)
        or (
            p.observed_heat_flux_std is not None
            and not np.isfinite(p.observed_heat_flux_std)
        )
    ]
    if nonfinite_cases:
        raise ValueError(
            f"calibration points contain non-finite values: {nonfinite_cases}"
        )
    negative_std_cases = [
        p.case
        for p in pts
        if p.observed_heat_flux_std is not None and p.observed_heat_flux_std < 0.0
    ]
    if negative_std_cases:
        raise ValueError(
            f"calibration points contain negative observed_heat_flux_std: {negative_std_cases}"
        )
    point_rules = {p.saturation_rule for p in pts}
    if point_rules != {str(saturation_rule)}:
        raise ValueError(
            "all calibration points must use the report saturation_rule "
            f"{saturation_rule!r}; found {sorted(point_rules)}"
        )

    scale_fit = None
    if fit_train_scale:
        scale_fit = fit_train_heat_flux_scale(pts)
        pts = apply_heat_flux_scale(
            pts,
            scale=float(scale_fit["scale"]),
            note_label="train_fitted_heat_flux_scale",
        )

    by_split_points = {
        split: [p for p in pts if p.split == split] for split in allowed_splits
    }
    by_split = {
        split: _split_metrics(split_points, observed_floor=observed_floor)
        for split, split_points in sorted(by_split_points.items())
    }
    all_metrics = _split_metrics(pts, observed_floor=observed_floor)
    holdout = by_split["holdout"]
    has_train = by_split["train"]["n"] > 0
    has_holdout = holdout["n"] > 0
    holdout_error = holdout["mean_abs_relative_error"]
    holdout_window_convergence = _holdout_window_convergence_summary(pts)
    passed = bool(
        has_train
        and has_holdout
        and holdout_error is not None
        and holdout_error <= holdout_mean_rel_gate
        and holdout_window_convergence["passed"]
    )
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
            "holdout_window_convergence": holdout_window_convergence,
            **({} if scale_fit is None else {"heat_flux_scale_fit": scale_fit}),
        },
    }



__all__ = [
    "QuasilinearCalibrationPoint",
    "apply_heat_flux_scale",
    "fit_train_heat_flux_scale",
    "quasilinear_calibration_report",
]
