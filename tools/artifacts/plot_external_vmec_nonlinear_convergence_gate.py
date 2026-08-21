#!/usr/bin/env python3
"""Build an external-VMEC nonlinear grid-convergence gate from pilot reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gkx.artifacts.plotting import set_plot_style  # noqa: E402
from gkx.diagnostics.validation_gates import (
    evaluate_scalar_gate,
    gate_report,
    gate_report_to_dict,
)  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    ROOT / "docs" / "_static" / "external_vmec_cth_like_grid_convergence_gate.png"
)
DEFAULT_TIME_HORIZON_OUT = (
    ROOT / "docs" / "_static" / "external_vmec_time_horizon_gate.png"
)
DEFAULT_COMMON_START_FRACTION = 0.5
DEFAULT_MAX_RELATIVE_SLOPE_PER_TIME = 2.0e-3
DEFAULT_MAX_COEFFICIENT_OF_VARIATION = 0.20
DEFAULT_MAX_PAIRWISE_RELATIVE_DIFFERENCE = 0.15
DEFAULT_MIN_WINDOW_SAMPLES = 5
DEFAULT_TIME_HORIZON_MAX_RELATIVE_CHANGE = 0.15


@dataclass(frozen=True)
class WindowStats:
    """Windowed heat-flux and field-energy statistics."""

    tmin: float
    tmax: float
    n_samples: int
    heat_flux_mean: float
    heat_flux_std: float
    heat_flux_slope: float
    heat_flux_relative_slope_per_time: float
    heat_flux_coefficient_of_variation: float
    wphi_mean: float
    wphi_std: float


@dataclass(frozen=True)
class PilotRun:
    """One nonlinear feasibility run loaded from a pilot JSON/CSV pair."""

    label: str
    json_path: Path
    csv_path: Path
    t: np.ndarray
    heat_flux: np.ndarray
    wphi: np.ndarray
    least_window: WindowStats
    source_label: str


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


def _repo_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _resolve_artifact_path(raw: str, *, json_path: Path) -> Path:
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([ROOT / path, json_path.parent / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not resolve artifact path {raw!r} from {json_path}")


def _as_finite_1d(values: list[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _read_trace_csv(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError(f"{path} contains no trace rows")
    out: dict[str, list[float]] = {"t": [], "heat_flux": [], "wphi": []}
    for row in rows:
        for key in out:
            if key not in row:
                raise ValueError(f"{path} is missing required column {key!r}")
            out[key].append(float(row[key]))
    return {key: _as_finite_1d(values, name=key) for key, values in out.items()}


def _window_stats(
    t: np.ndarray, heat_flux: np.ndarray, wphi: np.ndarray, *, tmin: float, tmax: float
) -> WindowStats:
    lo = float(tmin)
    hi = float(tmax)
    if hi < lo:
        raise ValueError("window tmax must be greater than or equal to tmin")
    mask = (t >= lo - 1.0e-12) & (t <= hi + 1.0e-12)
    if int(np.count_nonzero(mask)) < 3:
        raise ValueError(f"window [{lo}, {hi}] has fewer than three samples")
    tt = t[mask]
    heat = heat_flux[mask]
    wphi_win = wphi[mask]
    slope = float(np.polyfit(tt, heat, 1)[0])
    mean = float(np.mean(heat))
    std = float(np.std(heat))
    scale = max(abs(mean), 1.0e-300)
    return WindowStats(
        tmin=float(tt[0]),
        tmax=float(tt[-1]),
        n_samples=int(tt.size),
        heat_flux_mean=mean,
        heat_flux_std=std,
        heat_flux_slope=slope,
        heat_flux_relative_slope_per_time=float(slope / scale),
        heat_flux_coefficient_of_variation=float(std / scale),
        wphi_mean=float(np.mean(wphi_win)),
        wphi_std=float(np.std(wphi_win)),
    )


def _fallback_label(payload: dict[str, Any], path: Path) -> str:
    label = str(payload.get("label", "")).strip()
    if "Nx=Ny=" in label:
        _, rest = label.split("Nx=Ny=", maxsplit=1)
        return "Nx=Ny=" + rest.split(", t=", maxsplit=1)[0]
    return label or path.stem


def load_pilot_run(path: str | Path, *, label: str | None = None) -> PilotRun:
    """Load one nonlinear feasibility pilot JSON and companion trace CSV."""

    json_path = Path(path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{json_path} does not contain a JSON object")
    csv_raw = str(payload.get("csv", ""))
    if not csv_raw:
        raise ValueError(f"{json_path} does not declare a companion csv")
    csv_path = _resolve_artifact_path(csv_raw, json_path=json_path)
    trace = _read_trace_csv(csv_path)
    t = trace["t"]
    heat_flux = trace["heat_flux"]
    wphi = trace["wphi"]
    if not (t.size == heat_flux.size == wphi.size):
        raise ValueError(f"{csv_path} trace columns have inconsistent lengths")
    least = payload.get("least_trending_window")
    if isinstance(least, dict):
        least_stats = _window_stats(
            t,
            heat_flux,
            wphi,
            tmin=float(least["tmin"]),
            tmax=float(least["tmax"]),
        )
    else:
        tmin = float(t[0] + DEFAULT_COMMON_START_FRACTION * (t[-1] - t[0]))
        least_stats = _window_stats(t, heat_flux, wphi, tmin=tmin, tmax=float(t[-1]))
    return PilotRun(
        label=str(label) if label is not None else _fallback_label(payload, json_path),
        json_path=json_path,
        csv_path=csv_path,
        t=t,
        heat_flux=heat_flux,
        wphi=wphi,
        least_window=least_stats,
        source_label=str(payload.get("label", json_path.stem)),
    )


def _max_symmetric_relative_difference(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    max_diff = 0.0
    for i, left in enumerate(values):
        for right in values[i + 1 :]:
            denom = max(abs(left) + abs(right), 1.0e-300)
            max_diff = max(max_diff, float(2.0 * abs(left - right) / denom))
    return max_diff


def build_convergence_payload(
    runs: list[PilotRun],
    *,
    case: str = "CTH-like external VMEC nonlinear grid convergence",
    common_start_fraction: float = DEFAULT_COMMON_START_FRACTION,
    max_relative_slope_per_time: float = DEFAULT_MAX_RELATIVE_SLOPE_PER_TIME,
    max_coefficient_of_variation: float = DEFAULT_MAX_COEFFICIENT_OF_VARIATION,
    max_pairwise_relative_difference: float = DEFAULT_MAX_PAIRWISE_RELATIVE_DIFFERENCE,
    min_window_samples: int = DEFAULT_MIN_WINDOW_SAMPLES,
) -> dict[str, Any]:
    """Return a strict JSON-ready convergence-gate payload."""

    if len(runs) < 2:
        raise ValueError("at least two pilot runs are required for a convergence gate")
    if not 0.0 <= common_start_fraction < 1.0:
        raise ValueError("common_start_fraction must satisfy 0 <= value < 1")

    overlap_tmin = max(float(run.t[0]) for run in runs)
    overlap_tmax = min(float(run.t[-1]) for run in runs)
    common_tmin = overlap_tmin + common_start_fraction * (overlap_tmax - overlap_tmin)
    common_stats = [
        _window_stats(
            run.t, run.heat_flux, run.wphi, tmin=common_tmin, tmax=overlap_tmax
        )
        for run in runs
    ]
    least_stats = [run.least_window for run in runs]

    common_grid_diff = _max_symmetric_relative_difference(
        [item.heat_flux_mean for item in common_stats]
    )
    least_grid_diff = _max_symmetric_relative_difference(
        [item.heat_flux_mean for item in least_stats]
    )
    common_trend_max = max(
        abs(item.heat_flux_relative_slope_per_time) for item in common_stats
    )
    least_trend_max = max(
        abs(item.heat_flux_relative_slope_per_time) for item in least_stats
    )
    common_cv_max = max(
        abs(item.heat_flux_coefficient_of_variation) for item in common_stats
    )
    least_cv_max = max(
        abs(item.heat_flux_coefficient_of_variation) for item in least_stats
    )
    common_min_samples = min(item.n_samples for item in common_stats)
    least_min_samples = min(item.n_samples for item in least_stats)

    gates = [
        evaluate_scalar_gate(
            "common_window_max_relative_slope_per_time",
            common_trend_max,
            0.0,
            atol=max_relative_slope_per_time,
            rtol=0.0,
            units="1/time",
            notes="Same physical late-time window across all grids.",
        ),
        evaluate_scalar_gate(
            "least_window_max_relative_slope_per_time",
            least_trend_max,
            0.0,
            atol=max_relative_slope_per_time,
            rtol=0.0,
            units="1/time",
            notes="Best late-time window selected independently for each run.",
        ),
        evaluate_scalar_gate(
            "common_window_max_heat_flux_cv",
            common_cv_max,
            0.0,
            atol=max_coefficient_of_variation,
            rtol=0.0,
            notes="Standard deviation divided by absolute mean on the common window.",
        ),
        evaluate_scalar_gate(
            "least_window_max_heat_flux_cv",
            least_cv_max,
            0.0,
            atol=max_coefficient_of_variation,
            rtol=0.0,
            notes="Standard deviation divided by absolute mean on each least-trending window.",
        ),
        evaluate_scalar_gate(
            "common_window_min_samples_deficit",
            max(0.0, float(min_window_samples - common_min_samples)),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="samples",
            notes="The common window must contain enough diagnostic samples on every grid.",
        ),
        evaluate_scalar_gate(
            "least_window_min_samples_deficit",
            max(0.0, float(min_window_samples - least_min_samples)),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="samples",
            notes="Each independently selected late window must contain enough diagnostic samples.",
        ),
        evaluate_scalar_gate(
            "common_window_pairwise_heat_flux_symmetric_relative_difference",
            common_grid_diff,
            0.0,
            atol=max_pairwise_relative_difference,
            rtol=0.0,
            notes="Grid-refined heat-flux means on the same physical window.",
        ),
        evaluate_scalar_gate(
            "least_window_pairwise_heat_flux_symmetric_relative_difference",
            least_grid_diff,
            0.0,
            atol=max_pairwise_relative_difference,
            rtol=0.0,
            notes="Grid-refined heat-flux means on least-trending windows.",
        ),
    ]
    report = gate_report(case, "external_vmec_nonlinear_pilot_reports", gates)

    def stats_to_dict(stats: WindowStats) -> dict[str, float | int]:
        return {
            "tmin": stats.tmin,
            "tmax": stats.tmax,
            "n_samples": stats.n_samples,
            "heat_flux_mean": stats.heat_flux_mean,
            "heat_flux_std": stats.heat_flux_std,
            "heat_flux_slope": stats.heat_flux_slope,
            "heat_flux_relative_slope_per_time": stats.heat_flux_relative_slope_per_time,
            "heat_flux_coefficient_of_variation": stats.heat_flux_coefficient_of_variation,
            "wphi_mean": stats.wphi_mean,
            "wphi_std": stats.wphi_std,
        }

    passed = bool(report.passed)
    payload = {
        "kind": "external_vmec_nonlinear_grid_convergence_gate",
        "case": case,
        "passed": passed,
        "gate_index_include": False,
        "claim_level": (
            "passed_grid_convergence_candidate_for_transport_holdout"
            if passed
            else "negative_grid_convergence_result_not_transport_validation"
        ),
        "literature_policy": {
            "summary": (
                "Nonlinear gyrokinetic heat-flux claims require finite late-time traces, "
                "statistically interpretable saturated windows, and resolution/convergence checks "
                "before a run is admitted as a calibration holdout."
            ),
            "anchors": [
                {
                    "name": "Dimits et al. 2000 Cyclone benchmark and nonlinear saturated heat-flux comparisons",
                    "url": "https://doi.org/10.1063/1.873896",
                },
                {
                    "name": "GX/JPP nonlinear CBC and W7-X time-trace plus velocity-space convergence practice",
                    "url": "https://doi.org/10.1017/S0022377822000617",
                },
                {
                    "name": "Sanchez et al. 2021 stellarator domain sensitivity and flux-tube convergence",
                    "url": "https://doi.org/10.1088/1741-4326/ac2a87",
                },
                {
                    "name": "Papadopoulos et al. 2023 W7-X nonlinear heat-flux time-series analysis",
                    "url": "https://doi.org/10.3390/e25060942",
                },
            ],
        },
        "thresholds": {
            "common_start_fraction": common_start_fraction,
            "max_relative_slope_per_time": max_relative_slope_per_time,
            "max_coefficient_of_variation": max_coefficient_of_variation,
            "max_pairwise_relative_difference": max_pairwise_relative_difference,
            "min_window_samples": min_window_samples,
        },
        "common_window": {
            "requested_start_fraction": common_start_fraction,
            "tmin": common_tmin,
            "tmax": overlap_tmax,
            "max_pairwise_heat_flux_symmetric_relative_difference": common_grid_diff,
        },
        "least_windows": {
            "max_pairwise_heat_flux_symmetric_relative_difference": least_grid_diff,
        },
        "runs": [
            {
                "label": run.label,
                "source_label": run.source_label,
                "json": _repo_relative_path(run.json_path),
                "csv": _repo_relative_path(run.csv_path),
                "n_trace_samples": int(run.t.size),
                "tmin": float(run.t[0]),
                "tmax": float(run.t[-1]),
                "least_trending_window": stats_to_dict(least),
                "common_window": stats_to_dict(common),
            }
            for run, least, common in zip(runs, least_stats, common_stats, strict=True)
        ],
        "gate_report": gate_report_to_dict(report),
        "promotion_gate": {
            "passed": passed,
            "reason": (
                f"{case} passed external-VMEC nonlinear grid-convergence gate"
                if passed
                else f"{case} is finite but not grid/window converged enough for quasilinear calibration"
            ),
        },
    }
    return _json_clean(payload)


def write_summary_csv(path: Path, payload: dict[str, Any]) -> None:
    """Write the run/window rows used by the convergence gate."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "window",
        "tmin",
        "tmax",
        "n_samples",
        "heat_flux_mean",
        "heat_flux_std",
        "heat_flux_coefficient_of_variation",
        "heat_flux_slope",
        "heat_flux_relative_slope_per_time",
        "wphi_mean",
        "wphi_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run in payload["runs"]:
            for window in ("least_trending_window", "common_window"):
                row = {"run": run["label"], "window": window}
                row.update(run[window])
                writer.writerow(row)


def write_convergence_panel(
    runs: list[PilotRun], payload: dict[str, Any], *, out: str | Path = DEFAULT_OUT
) -> dict[str, str]:
    """Write PNG/PDF/JSON/CSV artifacts for the convergence gate."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#0f4c81", "#c44e52", "#2a9d8f", "#b45309", "#7c3aed"]

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    ax_trace, ax_bar, ax_text = axes

    common = payload["common_window"]
    for idx, run in enumerate(runs):
        color = colors[idx % len(colors)]
        ax_trace.plot(
            run.t,
            run.heat_flux,
            marker="o",
            markersize=3.0,
            linewidth=2.0,
            color=color,
            label=run.label,
        )
        ax_trace.axvspan(
            run.least_window.tmin, run.least_window.tmax, color=color, alpha=0.07
        )
    ax_trace.axvspan(
        float(common["tmin"]),
        float(common["tmax"]),
        color="#111827",
        alpha=0.08,
        label="common gate window",
    )
    ax_trace.set_xlabel("time")
    ax_trace.set_ylabel("heat flux")
    ax_trace.set_title("Nonlinear traces")
    ax_trace.grid(True, alpha=0.25)
    ax_trace.legend(frameon=False, fontsize=8)

    x = np.arange(len(runs))
    common_means = [
        float(run["common_window"]["heat_flux_mean"]) for run in payload["runs"]
    ]
    common_stds = [
        float(run["common_window"]["heat_flux_std"]) for run in payload["runs"]
    ]
    least_means = [
        float(run["least_trending_window"]["heat_flux_mean"]) for run in payload["runs"]
    ]
    ax_bar.bar(
        x - 0.18,
        common_means,
        width=0.36,
        yerr=common_stds,
        color="#4b5563",
        alpha=0.85,
        label="common window",
    )
    ax_bar.scatter(
        x + 0.18,
        least_means,
        s=70,
        color="#f59e0b",
        edgecolor="black",
        linewidth=0.5,
        label="least-trending",
    )
    ax_bar.set_xticks(x, [run.label for run in runs], rotation=18, ha="right")
    ax_bar.set_ylabel("window mean heat flux")
    ax_bar.set_title("Grid-refined window means")
    ax_bar.grid(True, axis="y", alpha=0.25)
    ax_bar.legend(frameon=False, fontsize=8)

    gate = payload["gate_report"]
    failed = [item for item in gate["gates"] if not bool(item["passed"])]
    passed = bool(payload["promotion_gate"]["passed"])
    metric_labels = {
        "common_window_max_relative_slope_per_time": "common trend",
        "least_window_max_relative_slope_per_time": "least-window trend",
        "common_window_max_heat_flux_cv": "common coefficient of variation",
        "least_window_max_heat_flux_cv": "least-window coefficient of variation",
        "common_window_min_samples_deficit": "common sample deficit",
        "least_window_min_samples_deficit": "least-window sample deficit",
        "common_window_pairwise_heat_flux_symmetric_relative_difference": "common grid difference",
        "least_window_pairwise_heat_flux_symmetric_relative_difference": "least-window grid difference",
    }
    lines = [
        "Gate status: " + ("PASS" if passed else "FAIL"),
        f"common rel. grid diff: {float(common['max_pairwise_heat_flux_symmetric_relative_difference']):.3f}",
        f"least-window rel. grid diff: {float(payload['least_windows']['max_pairwise_heat_flux_symmetric_relative_difference']):.3f}",
        f"allowed grid diff: {float(payload['thresholds']['max_pairwise_relative_difference']):.3f}",
        f"common start fraction: {float(payload['thresholds']['common_start_fraction']):.2f}",
        "",
        "Failed metrics:",
    ]
    lines.extend(
        f"- {metric_labels.get(str(item['metric']), str(item['metric']))}: "
        f"{float(item['observed']):.3g} > {float(item['atol']):.3g}"
        for item in failed[:6]
    )
    if not failed:
        lines.append("- none")
    lines.extend(["", "Interpretation:"])
    case_text = str(payload.get("case", "")).lower()
    if passed and "audit" in case_text:
        lines.extend(
            [
                "same-family audit passes;",
                "reproducibility evidence,",
                "not an independent",
                "calibration holdout.",
            ]
        )
    elif passed:
        lines.extend(
            [
                "finite external-VMEC pilot",
                "with passing late-window and",
                "grid agreement; candidate",
                "transport holdout pending",
                "calibration admission.",
            ]
        )
    else:
        lines.extend(
            [
                "finite external-VMEC pilot,",
                "but not a transport holdout until",
                "late-window and grid agreement pass.",
            ]
        )
    ax_text.axis("off")
    ax_text.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10.5,
        family="monospace",
    )
    fig.suptitle(str(payload["case"]), fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    payload = dict(payload)
    payload.update(
        {
            "png": _repo_relative_path(out_path),
            "pdf": _repo_relative_path(pdf_path),
            "csv": _repo_relative_path(csv_path),
        }
    )
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    write_summary_csv(csv_path, payload)
    return {
        "png": str(out_path),
        "pdf": str(pdf_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def write_convergence_gate(
    pilot_jsons: list[str | Path],
    *,
    out: str | Path = DEFAULT_OUT,
    labels: list[str] | None = None,
    case: str = "CTH-like external VMEC nonlinear grid convergence",
    common_start_fraction: float = DEFAULT_COMMON_START_FRACTION,
    max_relative_slope_per_time: float = DEFAULT_MAX_RELATIVE_SLOPE_PER_TIME,
    max_coefficient_of_variation: float = DEFAULT_MAX_COEFFICIENT_OF_VARIATION,
    max_pairwise_relative_difference: float = DEFAULT_MAX_PAIRWISE_RELATIVE_DIFFERENCE,
    min_window_samples: int = DEFAULT_MIN_WINDOW_SAMPLES,
) -> dict[str, str]:
    """Build and write a convergence gate from pilot JSON reports."""

    if labels is not None and len(labels) != len(pilot_jsons):
        raise ValueError("labels must have the same length as pilot_jsons")
    runs = [
        load_pilot_run(path, label=None if labels is None else labels[idx])
        for idx, path in enumerate(pilot_jsons)
    ]
    payload = build_convergence_payload(
        runs,
        case=case,
        common_start_fraction=common_start_fraction,
        max_relative_slope_per_time=max_relative_slope_per_time,
        max_coefficient_of_variation=max_coefficient_of_variation,
        max_pairwise_relative_difference=max_pairwise_relative_difference,
        min_window_samples=min_window_samples,
    )
    return write_convergence_panel(runs, payload, out=out)


def _parse_time_horizon_entry(raw: str) -> tuple[float, Path]:
    if ":" not in raw:
        raise ValueError("entries must have format HORIZON:PATH")
    horizon_raw, path_raw = raw.split(":", maxsplit=1)
    horizon = float(horizon_raw)
    if horizon <= 0.0:
        raise ValueError("horizon must be positive")
    path = Path(path_raw)
    if not path.is_absolute():
        path = ROOT / path
    return horizon, path


def _load_grid_gate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "external_vmec_nonlinear_grid_convergence_gate":
        raise ValueError(f"{path} is not an external-VMEC nonlinear grid gate")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) < 2:
        raise ValueError(f"{path} must contain at least two run rows")
    return payload


def build_time_horizon_payload(
    entries: list[tuple[float, Path]],
    *,
    case: str = "External-VMEC high-grid time-horizon stability",
    max_relative_change: float = DEFAULT_TIME_HORIZON_MAX_RELATIVE_CHANGE,
) -> dict[str, Any]:
    """Return a JSON-ready time-horizon stability report from grid gates."""

    if len(entries) < 2:
        raise ValueError("at least two horizon entries are required")
    if max_relative_change <= 0.0:
        raise ValueError("max_relative_change must be positive")

    rows = []
    for horizon, path in sorted(entries, key=lambda item: item[0]):
        gate = _load_grid_gate(path)
        common_means = [
            float(run["common_window"]["heat_flux_mean"]) for run in gate["runs"]
        ]
        least_means = [
            float(run["least_trending_window"]["heat_flux_mean"])
            for run in gate["runs"]
        ]
        rows.append(
            {
                "horizon": float(horizon),
                "gate_json": _repo_relative_path(path),
                "grid_gate_passed": bool(gate.get("passed")),
                "grid_common_relative_difference": float(
                    gate["common_window"][
                        "max_pairwise_heat_flux_symmetric_relative_difference"
                    ]
                ),
                "grid_least_relative_difference": float(
                    gate["least_windows"][
                        "max_pairwise_heat_flux_symmetric_relative_difference"
                    ]
                ),
                "common_heat_flux_mean": float(np.mean(common_means)),
                "common_heat_flux_half_spread": float(
                    0.5 * (max(common_means) - min(common_means))
                ),
                "least_heat_flux_mean": float(np.mean(least_means)),
                "least_heat_flux_half_spread": float(
                    0.5 * (max(least_means) - min(least_means))
                ),
                "grid_labels": [str(run["label"]) for run in gate["runs"]],
                "grid_common_means": common_means,
                "grid_least_means": least_means,
            }
        )

    common_change = _max_symmetric_relative_difference(
        [float(row["common_heat_flux_mean"]) for row in rows]
    )
    least_change = _max_symmetric_relative_difference(
        [float(row["least_heat_flux_mean"]) for row in rows]
    )
    failed_grid_gates = sum(0 if row["grid_gate_passed"] else 1 for row in rows)
    gates = [
        evaluate_scalar_gate(
            "failed_grid_gate_count",
            float(failed_grid_gates),
            0.0,
            atol=0.0,
            rtol=0.0,
            units="gates",
            notes="Every input high-grid gate must pass before checking horizon stability.",
        ),
        evaluate_scalar_gate(
            "common_window_time_horizon_relative_change",
            common_change,
            0.0,
            atol=max_relative_change,
            rtol=0.0,
            notes="Pairwise symmetric relative change of high-grid averaged common-window means.",
        ),
        evaluate_scalar_gate(
            "least_window_time_horizon_relative_change",
            least_change,
            0.0,
            atol=max_relative_change,
            rtol=0.0,
            notes="Pairwise symmetric relative change of high-grid averaged least-trending means.",
        ),
    ]
    report = gate_report(case, "external_vmec_high_grid_horizon_gates", gates)
    passed = bool(report.passed)
    return _json_clean(
        {
            "kind": "external_vmec_time_horizon_gate",
            "case": case,
            "passed": passed,
            "gate_index_include": False,
            "claim_level": (
                "passed_high_grid_time_horizon_candidate_not_replicated_holdout"
                if passed
                else "negative_time_horizon_result_not_transport_validation"
            ),
            "thresholds": {"max_relative_change": float(max_relative_change)},
            "common_window_time_horizon_relative_change": common_change,
            "least_window_time_horizon_relative_change": least_change,
            "rows": rows,
            "gate_report": gate_report_to_dict(report),
            "promotion_gate": {
                "passed": False,
                "reason": (
                    "time-horizon stability is necessary but not sufficient; "
                    "replicate/seed/timestep evidence is still required before holdout admission"
                    if passed
                    else "time-horizon stability gate failed"
                ),
            },
        }
    )


def write_time_horizon_summary_csv(path: Path, payload: dict[str, Any]) -> None:
    """Write time-horizon rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "horizon",
        "grid_gate_passed",
        "grid_common_relative_difference",
        "grid_least_relative_difference",
        "common_heat_flux_mean",
        "common_heat_flux_half_spread",
        "least_heat_flux_mean",
        "least_heat_flux_half_spread",
        "gate_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow({key: row[key] for key in fields})


def write_time_horizon_panel(
    entries: list[tuple[float, Path]],
    *,
    out: str | Path = DEFAULT_TIME_HORIZON_OUT,
    case: str = "External-VMEC high-grid time-horizon stability",
    max_relative_change: float = DEFAULT_TIME_HORIZON_MAX_RELATIVE_CHANGE,
) -> dict[str, str]:
    """Write PNG/PDF/JSON/CSV artifacts for the time-horizon gate."""

    payload = build_time_horizon_payload(
        entries, case=case, max_relative_change=max_relative_change
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    horizons = np.asarray(
        [float(row["horizon"]) for row in payload["rows"]], dtype=float
    )
    common = np.asarray(
        [float(row["common_heat_flux_mean"]) for row in payload["rows"]], dtype=float
    )
    common_err = np.asarray(
        [float(row["common_heat_flux_half_spread"]) for row in payload["rows"]],
        dtype=float,
    )
    least = np.asarray(
        [float(row["least_heat_flux_mean"]) for row in payload["rows"]], dtype=float
    )
    least_err = np.asarray(
        [float(row["least_heat_flux_half_spread"]) for row in payload["rows"]],
        dtype=float,
    )

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    ax, ax_text = axes
    ax.errorbar(
        horizons,
        common,
        yerr=common_err,
        marker="o",
        linewidth=2.2,
        capsize=4,
        label="common window",
    )
    ax.errorbar(
        horizons,
        least,
        yerr=least_err,
        marker="s",
        linewidth=2.2,
        capsize=4,
        label="least-trending window",
    )
    ax.set_xlabel("final time")
    ax.set_ylabel("high-grid averaged heat flux")
    ax.set_title("High-grid horizon stability")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    failed = [
        item for item in payload["gate_report"]["gates"] if not bool(item["passed"])
    ]
    lines = [
        "Gate status: " + ("PASS" if payload["passed"] else "FAIL"),
        f"common horizon diff: {float(payload['common_window_time_horizon_relative_change']):.3f}",
        f"least horizon diff: {float(payload['least_window_time_horizon_relative_change']):.3f}",
        f"allowed diff: {float(payload['thresholds']['max_relative_change']):.3f}",
        "",
        "Failed metrics:",
    ]
    lines.extend(
        f"- {item['metric']}: {float(item['observed']):.3g} > {float(item['atol']):.3g}"
        for item in failed
    )
    if not failed:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Interpretation:",
            "necessary horizon check only;",
            "replicate evidence is still",
            "required before calibration use.",
        ]
    )
    ax_text.axis("off")
    ax_text.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10.5,
        family="monospace",
    )
    fig.suptitle(case, fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    payload = dict(payload)
    payload.update(
        {
            "png": _repo_relative_path(out_path),
            "pdf": _repo_relative_path(pdf_path),
            "csv": _repo_relative_path(csv_path),
        }
    )
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    write_time_horizon_summary_csv(csv_path, payload)
    return {
        "png": str(out_path),
        "pdf": str(pdf_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def build_convergence_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-json",
        action="append",
        required=True,
        help="Input nonlinear feasibility pilot JSON.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional label matching each --pilot-json.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output PNG path.")
    parser.add_argument(
        "--case", default="CTH-like external VMEC nonlinear grid convergence"
    )
    parser.add_argument(
        "--common-start-fraction", type=float, default=DEFAULT_COMMON_START_FRACTION
    )
    parser.add_argument(
        "--max-relative-slope-per-time",
        type=float,
        default=DEFAULT_MAX_RELATIVE_SLOPE_PER_TIME,
    )
    parser.add_argument(
        "--max-coefficient-of-variation",
        type=float,
        default=DEFAULT_MAX_COEFFICIENT_OF_VARIATION,
    )
    parser.add_argument(
        "--max-pairwise-relative-difference",
        type=float,
        default=DEFAULT_MAX_PAIRWISE_RELATIVE_DIFFERENCE,
    )
    parser.add_argument(
        "--min-window-samples", type=int, default=DEFAULT_MIN_WINDOW_SAMPLES
    )
    return parser


def build_time_horizon_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an external-VMEC time-horizon gate.")
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Horizon/gate entry encoded as HORIZON:PATH",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_TIME_HORIZON_OUT), help="Output PNG path."
    )
    parser.add_argument(
        "--case", default="External-VMEC high-grid time-horizon stability"
    )
    parser.add_argument(
        "--max-relative-change",
        type=float,
        default=DEFAULT_TIME_HORIZON_MAX_RELATIVE_CHANGE,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    if raw_argv[:1] == ["time-horizon"]:
        args = build_time_horizon_parser().parse_args(raw_argv[1:])
        paths = write_time_horizon_panel(
            [_parse_time_horizon_entry(raw) for raw in args.entry],
            out=args.out,
            case=args.case,
            max_relative_change=args.max_relative_change,
        )
        print(json.dumps(paths, indent=2, sort_keys=True))
        return 0

    args = build_convergence_parser().parse_args(raw_argv)
    paths = write_convergence_gate(
        args.pilot_json,
        out=args.out,
        labels=args.label,
        case=args.case,
        common_start_fraction=args.common_start_fraction,
        max_relative_slope_per_time=args.max_relative_slope_per_time,
        max_coefficient_of_variation=args.max_coefficient_of_variation,
        max_pairwise_relative_difference=args.max_pairwise_relative_difference,
        min_window_samples=args.min_window_samples,
    )
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
