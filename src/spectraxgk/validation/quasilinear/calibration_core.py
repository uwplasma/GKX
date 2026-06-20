"""Core quasilinear calibration point, scaling, and report helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from spectraxgk.validation.quasilinear.window_promotion import (
    nonlinear_window_stats_promotion_ready,
)

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


@dataclass(frozen=True)
class _CalibrationReportControls:
    saturation_rule: str
    version: str
    holdout_mean_rel_gate: float
    observed_floor: float


def _point_from_mapping(
    item: QuasilinearCalibrationPoint | dict[str, Any],
) -> QuasilinearCalibrationPoint:
    if isinstance(item, QuasilinearCalibrationPoint):
        return item
    return QuasilinearCalibrationPoint(**dict(item))


def _normalized_calibration_points(
    points: Iterable[QuasilinearCalibrationPoint | dict[str, Any]],
) -> list[QuasilinearCalibrationPoint]:
    pts = [_point_from_mapping(item) for item in points]
    if not pts:
        raise ValueError("at least one calibration point is required")
    return pts


def _calibration_report_controls(
    *,
    saturation_rule: str,
    version: str,
    holdout_mean_rel_gate: float,
    observed_floor: float,
) -> _CalibrationReportControls:
    if observed_floor <= 0.0:
        raise ValueError("observed_floor must be positive")
    if holdout_mean_rel_gate <= 0.0:
        raise ValueError("holdout_mean_rel_gate must be positive")
    return _CalibrationReportControls(
        saturation_rule=str(saturation_rule),
        version=str(version),
        holdout_mean_rel_gate=float(holdout_mean_rel_gate),
        observed_floor=float(observed_floor),
    )


def _validate_calibration_points(
    points: list[QuasilinearCalibrationPoint],
    *,
    saturation_rule: str,
) -> None:
    allowed_splits = {"train", "holdout", "audit"}
    bad_splits = sorted({p.split for p in points if p.split not in allowed_splits})
    if bad_splits:
        raise ValueError(f"unsupported calibration split(s): {bad_splits}")
    nonfinite_cases = [
        p.case
        for p in points
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
        for p in points
        if p.observed_heat_flux_std is not None and p.observed_heat_flux_std < 0.0
    ]
    if negative_std_cases:
        raise ValueError(
            f"calibration points contain negative observed_heat_flux_std: {negative_std_cases}"
        )
    point_rules = {p.saturation_rule for p in points}
    if point_rules != {str(saturation_rule)}:
        raise ValueError(
            "all calibration points must use the report saturation_rule "
            f"{saturation_rule!r}; found {sorted(point_rules)}"
        )


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


def _maybe_apply_train_scale(
    points: list[QuasilinearCalibrationPoint],
    *,
    fit_train_scale_enabled: bool,
) -> tuple[list[QuasilinearCalibrationPoint], dict[str, Any] | None]:
    if not fit_train_scale_enabled:
        return points, None
    scale_fit = fit_train_heat_flux_scale(points)
    scaled_points = apply_heat_flux_scale(
        points,
        scale=float(scale_fit["scale"]),
        note_label="train_fitted_heat_flux_scale",
    )
    return scaled_points, scale_fit


def _calibration_split_metrics(
    points: list[QuasilinearCalibrationPoint],
    *,
    observed_floor: float,
) -> tuple[dict[str, list[QuasilinearCalibrationPoint]], dict[str, Any], dict[str, Any]]:
    allowed_splits = {"train", "holdout", "audit"}
    by_split_points = {
        split: [p for p in points if p.split == split] for split in allowed_splits
    }
    by_split = {
        split: _split_metrics(split_points, observed_floor=observed_floor)
        for split, split_points in sorted(by_split_points.items())
    }
    all_metrics = _split_metrics(points, observed_floor=observed_floor)
    return by_split_points, by_split, all_metrics


def _calibration_report_claim(
    *,
    by_split: dict[str, Any],
    holdout_window_convergence: dict[str, Any],
    holdout_mean_rel_gate: float,
) -> tuple[bool, str]:
    holdout = by_split["holdout"]
    has_train = by_split["train"]["n"] > 0
    has_holdout = holdout["n"] > 0
    holdout_error = holdout["mean_abs_relative_error"]
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
    return passed, claim_level


def _calibration_report_payload(
    *,
    controls: _CalibrationReportControls,
    points: list[QuasilinearCalibrationPoint],
    metrics: dict[str, Any],
    by_split: dict[str, Any],
    passed: bool,
    claim_level: str,
    holdout_window_convergence: dict[str, Any],
    scale_fit: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "kind": "quasilinear_calibration_report",
        "version": controls.version,
        "saturation_rule": controls.saturation_rule,
        "claim_level": claim_level,
        "passed": passed,
        "holdout_mean_rel_gate": controls.holdout_mean_rel_gate,
        "observed_floor": controls.observed_floor,
        "metrics": metrics,
        "by_split": by_split,
        "points": [p.to_dict() for p in points],
        "metadata": {
            **dict(metadata or {}),
            "holdout_window_convergence": holdout_window_convergence,
            **({} if scale_fit is None else {"heat_flux_scale_fit": scale_fit}),
        },
    }


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

    controls = _calibration_report_controls(
        saturation_rule=saturation_rule,
        version=version,
        holdout_mean_rel_gate=holdout_mean_rel_gate,
        observed_floor=observed_floor,
    )
    pts = _normalized_calibration_points(points)
    _validate_calibration_points(pts, saturation_rule=controls.saturation_rule)
    pts, scale_fit = _maybe_apply_train_scale(
        pts,
        fit_train_scale_enabled=fit_train_scale,
    )
    _, by_split, all_metrics = _calibration_split_metrics(
        pts,
        observed_floor=controls.observed_floor,
    )
    holdout_window_convergence = _holdout_window_convergence_summary(pts)
    passed, claim_level = _calibration_report_claim(
        by_split=by_split,
        holdout_window_convergence=holdout_window_convergence,
        holdout_mean_rel_gate=controls.holdout_mean_rel_gate,
    )
    return _calibration_report_payload(
        controls=controls,
        points=pts,
        metrics=all_metrics,
        by_split=by_split,
        passed=passed,
        claim_level=claim_level,
        holdout_window_convergence=holdout_window_convergence,
        scale_fit=scale_fit,
        metadata=metadata,
    )



__all__ = [
    "QuasilinearCalibrationPoint",
    "apply_heat_flux_scale",
    "fit_train_heat_flux_scale",
    "quasilinear_calibration_report",
]
