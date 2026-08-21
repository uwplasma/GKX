#!/usr/bin/env python3
"""Plot quasilinear-vs-nonlinear calibration report points."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import textwrap
from typing import Any
import sys

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gkx.artifacts.plotting import set_plot_style  # noqa: E402


SPLIT_COLORS = {
    "train": "#0f4c81",
    "holdout": "#2a9d8f",
    "audit": "#c44e52",
}

CASE_LABELS = {
    "cyclone_long_window": "Cyclone train",
    "cyclone_miller_long_window": "Cyclone Miller",
    "hsx_nonlinear_window": "HSX",
    "w7x_nonlinear_window": "W7-X",
    "dshape_external_vmec_t250_window": "D-shaped VMEC",
    "itermodel_external_vmec_t350_window": "ITERModel VMEC",
    "updown_asym_external_vmec_t450_window": "Up-Down Asym VMEC",
    "circular_external_vmec_t450_window": "Circular VMEC",
    "cth_like_external_vmec_t700_high_grid_window": "CTH-like VMEC",
    "shaped_tokamak_pressure_external_vmec_t650_high_grid_window": "Shaped-pressure VMEC",
    "qp_diag_nfp2_m4_final_t250": "QP VMEC",
}


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def load_calibration_report(path: str | Path) -> dict[str, Any]:
    """Load and validate a quasilinear calibration report."""

    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("kind") != "quasilinear_calibration_report":
        raise ValueError(f"{path} is not a quasilinear calibration report")
    points = report.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("calibration report must contain a non-empty points list")
    required = {"case", "split", "predicted_heat_flux", "observed_heat_flux"}
    for point in points:
        missing = required - set(point)
        if missing:
            raise ValueError(f"calibration point is missing keys: {sorted(missing)}")
    return report


def _point_arrays(
    points: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(
        [float(p["predicted_heat_flux"]) for p in points], dtype=float
    )
    observed = np.asarray([float(p["observed_heat_flux"]) for p in points], dtype=float)
    std = np.asarray(
        [
            np.nan
            if p.get("observed_heat_flux_std") is None
            else float(p["observed_heat_flux_std"])
            for p in points
        ],
        dtype=float,
    )
    return predicted, observed, std


def _case_label(case: object) -> str:
    raw = str(case)
    label = CASE_LABELS.get(raw, raw.replace("_", " "))
    return textwrap.fill(label, width=22)


def calibration_figure(
    report: dict[str, Any],
    *,
    title: str = "Quasilinear calibration audit",
) -> plt.Figure:
    """Create a publication-facing calibration/audit figure."""

    points = list(report["points"])
    predicted, observed, std = _point_arrays(points)
    finite = np.isfinite(predicted) & np.isfinite(observed)
    if not np.any(finite):
        raise ValueError("no finite calibration points to plot")

    set_plot_style()
    fig_height = max(5.2, 0.62 * len(points) + 2.4)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, fig_height), constrained_layout=True)
    ax = axes[0]
    residual_ax = axes[1]

    positive_values = np.concatenate(
        [predicted[finite & (predicted > 0.0)], observed[finite & (observed > 0.0)]]
    )
    log_floor = None
    if positive_values.size:
        log_floor = float(np.min(positive_values)) * 0.35
        lo = log_floor * 0.7
        hi = float(np.max(positive_values)) * 1.4
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(
            [lo, hi],
            [lo, hi],
            color="#333333",
            linestyle="--",
            linewidth=1.6,
            label="1:1",
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    else:
        lo = float(min(np.min(predicted[finite]), np.min(observed[finite]), 0.0))
        hi = float(max(np.max(predicted[finite]), np.max(observed[finite]), 1.0))
        ax.plot(
            [lo, hi],
            [lo, hi],
            color="#333333",
            linestyle="--",
            linewidth=1.6,
            label="1:1",
        )

    labels = []
    rel_errors = []
    colors = []
    clipped_nonpositive = False
    for point, pred, obs, err in zip(points, predicted, observed, std, strict=True):
        if not np.isfinite(pred) or not np.isfinite(obs):
            continue
        split = str(point["split"])
        color = SPLIT_COLORS.get(split, "#6c757d")
        plot_pred = pred
        plot_obs = obs
        marker_face = color
        marker_edge = "white"
        if log_floor is not None and (pred <= 0.0 or obs <= 0.0):
            plot_pred = pred if pred > 0.0 else log_floor
            plot_obs = obs if obs > 0.0 else log_floor
            marker_face = "none"
            marker_edge = color
            clipped_nonpositive = True
        yerr = None
        if np.isfinite(err):
            if log_floor is not None:
                lower = max(min(err, plot_obs - log_floor), 0.0)
                yerr = np.asarray([[lower], [err]], dtype=float)
            else:
                yerr = err
        ax.errorbar(
            plot_pred,
            plot_obs,
            yerr=yerr,
            marker="o",
            markersize=8,
            color=color,
            markerfacecolor=marker_face,
            markeredgecolor=marker_edge,
            markeredgewidth=0.8,
            capsize=3.0,
            linestyle="None",
            label=split,
        )
        labels.append(_case_label(point["case"]))
        denom = max(abs(obs), float(report.get("observed_floor", 1.0e-12)))
        rel_errors.append(abs(pred - obs) / denom)
        colors.append(color)

    ax.set_title("Absolute flux comparison")
    ax.set_xlabel("quasilinear estimate")
    ax.set_ylabel("nonlinear window mean")
    if clipped_nonpositive and log_floor is not None:
        ax.text(
            0.03,
            0.03,
            f"open marker: non-positive estimate\nplotted at floor {log_floor:.2e}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "white",
                "ec": "0.75",
                "alpha": 0.92,
            },
        )
    handles, legend_labels = ax.get_legend_handles_labels()
    legend = dict(zip(legend_labels, handles, strict=False))
    ax.legend(legend.values(), legend.keys())
    ax.grid(True, which="both", alpha=0.22)

    y = np.arange(len(labels))
    rel_arr = np.asarray(rel_errors, dtype=float)
    positive_rel = rel_arr[np.isfinite(rel_arr) & (rel_arr > 0.0)]
    gate = float(report.get("holdout_mean_rel_gate", 0.35))
    use_log_error_axis = bool(
        positive_rel.size
        and (
            float(np.max(positive_rel)) / max(float(np.min(positive_rel)), 1.0e-300)
            > 50.0
            or float(np.max(positive_rel)) > 10.0
        )
    )
    if use_log_error_axis:
        error_floor = min(float(np.min(positive_rel)), max(gate, 1.0e-12)) * 0.35
        plot_rel = np.where(
            (rel_arr > 0.0) & np.isfinite(rel_arr), rel_arr, error_floor
        )
        for yi, value, raw_value, color in zip(
            y, plot_rel, rel_arr, colors, strict=True
        ):
            residual_ax.hlines(
                yi, error_floor, value, color=color, linewidth=3.0, alpha=0.9
            )
            residual_ax.plot(
                value,
                yi,
                marker="o",
                markersize=7.0,
                color=color,
                markerfacecolor="none" if raw_value <= 0.0 else color,
                markeredgewidth=1.2,
            )
        residual_ax.set_xscale("log")
        residual_ax.set_xlim(
            error_floor * 0.7, max(float(np.max(plot_rel)), gate) * 1.5
        )
        residual_ax.text(
            0.98,
            0.96,
            f"log error axis; zero errors plotted at {error_floor:.2e}",
            transform=residual_ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "white",
                "ec": "0.75",
                "alpha": 0.92,
            },
        )
    else:
        residual_ax.barh(y, rel_errors, color=colors, edgecolor="white", linewidth=0.8)
    residual_ax.axvline(
        gate, color="#c2410c", linestyle="--", linewidth=1.6, label="holdout gate"
    )
    residual_ax.set_yticks(y, labels)
    residual_ax.invert_yaxis()
    residual_ax.set_xlabel("absolute relative error")
    residual_ax.set_title(f"Claim level: {report.get('claim_level', 'unknown')}")
    residual_ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    residual_ax.grid(True, axis="x", alpha=0.22)

    fig.suptitle(title, y=1.03, fontsize=14, fontweight="bold")
    return fig


def write_calibration_figure(
    report_path: str | Path, *, out: str | Path, title: str
) -> dict[str, str]:
    """Write PNG/PDF/JSON artifacts for a calibration report."""

    report = load_calibration_report(report_path)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = calibration_figure(report, title=title)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    points = report["points"]
    predicted, observed, _std = _point_arrays(points)
    denom = np.maximum(np.abs(observed), float(report.get("observed_floor", 1.0e-12)))
    rel = np.abs(predicted - observed) / denom
    meta_path = out_path.with_suffix(".json")
    meta = {
        "kind": "quasilinear_calibration_figure",
        "source": str(report_path),
        "png": str(out_path),
        "pdf": str(pdf_path),
        "n_points": int(len(points)),
        "claim_level": report.get("claim_level"),
        "passed": bool(report.get("passed", False)),
        "max_abs_relative_error": float(np.nanmax(rel)),
        "mean_abs_relative_error": float(np.nanmean(rel)),
    }
    meta_path.write_text(
        json.dumps(_json_clean(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(meta_path)}


def _load_points(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "points" in data:
        data = data["points"]
    if not isinstance(data, list):
        raise TypeError(
            "input JSON must be a list of calibration points or an object with a 'points' list"
        )
    return [dict(item) for item in data]



def build_report_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a quasilinear calibration/holdout report from JSON points."
    )
    parser.add_argument("--points", help="JSON list of calibration points")
    parser.add_argument(
        "--spectrum", help="Quasilinear spectrum CSV for one generated point"
    )
    parser.add_argument(
        "--nonlinear-summary",
        help="Nonlinear window summary JSON for one generated point",
    )
    parser.add_argument(
        "--split", default="audit", help="train, holdout, or audit for generated point"
    )
    parser.add_argument(
        "--case", default=None, help="Optional case label for generated point"
    )
    parser.add_argument(
        "--geometry", default="unspecified", help="Geometry label for generated point"
    )
    parser.add_argument(
        "--electron-model",
        default="unspecified",
        help="Electron model label for generated point",
    )
    parser.add_argument("--spectrum-column", default="saturated_heat_flux_total")
    parser.add_argument(
        "--spectrum-method", default="sum", help="sum, mean, or trapezoid"
    )
    parser.add_argument(
        "--species-index",
        type=int,
        default=None,
        help="Species index for nonlinear NetCDF heat-flux variables; omit to sum all species.",
    )
    parser.add_argument("--out", required=True, help="Output calibration report JSON")
    parser.add_argument("--saturation-rule", default="mixing_length")
    parser.add_argument("--holdout-mean-rel-gate", type=float, default=0.35)
    parser.add_argument("--observed-floor", type=float, default=1.0e-12)
    parser.add_argument(
        "--fit-train-scale",
        action="store_true",
        help="Fit one multiplicative heat-flux scale from train points before scoring holdout points.",
    )
    parser.add_argument("--version", default="0.1")
    return parser


def _run_report(argv: list[str] | None = None) -> int:
    from gkx.diagnostics.quasilinear_calibration import (
        calibration_point_from_spectrum_and_nonlinear_window,
        quasilinear_calibration_report,
        write_quasilinear_calibration_report,
    )

    args = build_report_parser().parse_args(argv)
    points = []
    if args.points is not None:
        points.extend(_load_points(Path(args.points)))
    if args.spectrum is not None or args.nonlinear_summary is not None:
        if args.spectrum is None or args.nonlinear_summary is None:
            raise ValueError(
                "--spectrum and --nonlinear-summary must be supplied together"
            )
        points.append(
            calibration_point_from_spectrum_and_nonlinear_window(
                args.spectrum,
                args.nonlinear_summary,
                split=args.split,
                saturation_rule=args.saturation_rule,
                spectrum_column=args.spectrum_column,
                spectrum_method=args.spectrum_method,
                case=args.case,
                geometry=args.geometry,
                electron_model=args.electron_model,
                species_index=args.species_index,
                notes="generated by plot_quasilinear_calibration.py report",
            ).to_dict()
        )
    if not points:
        raise ValueError("provide --points or --spectrum plus --nonlinear-summary")
    report = quasilinear_calibration_report(
        points,
        saturation_rule=args.saturation_rule,
        version=args.version,
        holdout_mean_rel_gate=args.holdout_mean_rel_gate,
        observed_floor=args.observed_floor,
        fit_train_scale=args.fit_train_scale,
        metadata={
            "source_points": None if args.points is None else str(Path(args.points)),
            "source_spectrum": args.spectrum,
            "source_nonlinear_summary": args.nonlinear_summary,
        },
    )
    out = write_quasilinear_calibration_report(args.out, report)
    print(f"saved {out}")
    print(f"claim_level={report['claim_level']} passed={report['passed']}")
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", required=True, help="Quasilinear calibration report JSON"
    )
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Quasilinear calibration audit")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if argv_list and argv_list[0] == "report":
        return _run_report(argv_list[1:])
    if any(flag in argv_list for flag in ("--points", "--spectrum", "--nonlinear-summary")):
        return _run_report(argv_list)
    args = build_parser().parse_args(argv_list)
    paths = write_calibration_figure(args.report, out=args.out, title=args.title)
    print(f"saved {paths['png']}")
    print(f"saved {paths['pdf']}")
    print(f"saved {paths['json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
