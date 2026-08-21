#!/usr/bin/env python3
"""Build nonlinear feasibility or frozen-window validation panels."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gkx.artifacts.plotting import set_plot_style  # noqa: E402
from gkx.workflows.runtime.artifacts import load_nonlinear_netcdf_diagnostics  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEASIBILITY_OUT = (
    ROOT / "docs" / "_static" / "external_vmec_nonlinear_feasibility.png"
)
DEFAULT_FRACTIONS = (0.5, 0.6, 0.7, 0.8)
DEFAULT_WINDOW_GLOB = str(ROOT / "docs" / "_static" / "nonlinear_*_gate_summary.json")
DEFAULT_WINDOW_OUT = ROOT / "docs" / "_static" / "nonlinear_window_statistics.png"
DEFAULT_METRICS = ("Phi2", "Wg", "Wphi", "Wapar", "HeatFlux", "ParticleFlux")
DEFAULT_RELEASE_GATE = 0.10
CASE_LABELS = {
    "cyclone_nonlinear_long_window": "Cyclone",
    "cyclone_miller_nonlinear_window": "Cyclone Miller",
    "hsx_nonlinear_window": "HSX",
    "kbm_nonlinear_window": "KBM",
    "w7x_nonlinear_window": "W7-X",
}
CASE_ORDER = {
    "cyclone_nonlinear_long_window": 0,
    "cyclone_miller_nonlinear_window": 1,
    "kbm_nonlinear_window": 2,
    "w7x_nonlinear_window": 3,
    "hsx_nonlinear_window": 4,
}
METRIC_COLORS = {
    "Phi2": "#7b2cbf",
    "Wg": "#0f4c81",
    "Wphi": "#2a9d8f",
    "Wapar": "#6c757d",
    "HeatFlux": "#c44e52",
    "ParticleFlux": "#f4a261",
}
CASE_MEAN_REL_GATES = {
    # Cases near 0.10 retain the broad release envelope until paper-level retuning.
    "cyclone_nonlinear_long_window": 0.10,
    "cyclone_miller_nonlinear_window": 0.095,
    "kbm_nonlinear_window": 0.02,
    "w7x_nonlinear_window": 0.10,
    "hsx_nonlinear_window": 0.05,
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


def _as_1d(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def window_summaries(
    t: Any,
    heat_flux: Any,
    wphi: Any,
    *,
    start_fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
) -> list[dict[str, float | int]]:
    """Return late-window statistics for a nonlinear feasibility trace."""

    t_arr = _as_1d(t, name="t")
    heat_arr = _as_1d(heat_flux, name="heat_flux")
    wphi_arr = _as_1d(wphi, name="wphi")
    if not (t_arr.size == heat_arr.size == wphi_arr.size):
        raise ValueError("t, heat_flux, and wphi must have the same length")
    if t_arr.size < 3:
        raise ValueError("at least three samples are required")
    out: list[dict[str, float | int]] = []
    for fraction in start_fractions:
        if not 0.0 <= float(fraction) < 1.0:
            raise ValueError("start fractions must satisfy 0 <= fraction < 1")
        start = min(int(t_arr.size * float(fraction)), t_arr.size - 2)
        tt = t_arr[start:]
        heat = heat_arr[start:]
        wphi_win = wphi_arr[start:]
        slope = float(np.polyfit(tt, heat, 1)[0]) if tt.size >= 2 else float("nan")
        heat_mean = float(np.mean(heat))
        out.append(
            {
                "start_fraction": float(fraction),
                "start_index": int(start),
                "tmin": float(tt[0]),
                "tmax": float(tt[-1]),
                "n_samples": int(tt.size),
                "heat_flux_mean": heat_mean,
                "heat_flux_std": float(np.std(heat)),
                "heat_flux_last": float(heat_arr[-1]),
                "heat_flux_slope": slope,
                "heat_flux_relative_slope_per_time": float(
                    slope / max(abs(heat_mean), 1.0e-300)
                ),
                "wphi_mean": float(np.mean(wphi_win)),
                "wphi_std": float(np.std(wphi_win)),
                "wphi_last": float(wphi_arr[-1]),
            }
        )
    return out


def load_nonlinear_trace(path: str | Path) -> dict[str, np.ndarray]:
    """Load scalar traces from a nonlinear diagnostic NetCDF output."""

    diagnostics = load_nonlinear_netcdf_diagnostics(path)
    return {
        "t": _as_1d(diagnostics.t, name="t"),
        "heat_flux": _as_1d(diagnostics.heat_flux_t, name="heat_flux"),
        "wphi": _as_1d(diagnostics.Wphi_t, name="wphi"),
        "wg": _as_1d(diagnostics.Wg_t, name="wg"),
    }


def write_trace_csv(path: str | Path, trace: dict[str, np.ndarray]) -> None:
    """Write scalar nonlinear feasibility traces to CSV."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["t", "heat_flux", "wphi", "wg"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(keys)
        for values in zip(
            *(np.asarray(trace[key], dtype=float) for key in keys), strict=True
        ):
            writer.writerow([f"{float(value):.16e}" for value in values])


def write_pilot_panel(
    trace: dict[str, np.ndarray],
    *,
    out: str | Path = DEFAULT_FEASIBILITY_OUT,
    source: str | Path | None = None,
    title: str = "Nonlinear Feasibility Pilot",
    label: str = "external VMEC",
    claim_level: str = "finite_nonlinear_feasibility_not_transport_validation",
    start_fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
) -> dict[str, str]:
    """Write a nonlinear feasibility PNG/PDF/JSON/CSV artifact set."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = _as_1d(trace["t"], name="t")
    heat = _as_1d(trace["heat_flux"], name="heat_flux")
    wphi = _as_1d(trace["wphi"], name="wphi")
    wg = _as_1d(trace.get("wg", np.zeros_like(t)), name="wg")
    summaries = window_summaries(t, heat, wphi, start_fractions=start_fractions)
    chosen = min(
        summaries,
        key=lambda item: abs(float(item["heat_flux_relative_slope_per_time"])),
    )

    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), constrained_layout=True)
    ax_heat, ax_wphi, ax_zoom, ax_text = axes.ravel()

    heat_plot = np.maximum(np.abs(heat), 1.0e-300)
    wphi_plot = np.maximum(np.abs(wphi), 1.0e-300)
    ax_heat.semilogy(
        t, heat_plot, marker="o", markersize=3.2, linewidth=2.0, color="#0f4c81"
    )
    ax_heat.set_xlabel("time")
    ax_heat.set_ylabel("|heat flux|")
    ax_heat.set_title("Transport trace")
    ax_heat.grid(True, alpha=0.25)

    ax_wphi.semilogy(
        t, wphi_plot, marker="s", markersize=3.2, linewidth=2.0, color="#c44e52"
    )
    ax_wphi.set_xlabel("time")
    ax_wphi.set_ylabel(r"$W_\phi$")
    ax_wphi.set_title("Field-energy trace")
    ax_wphi.grid(True, alpha=0.25)

    colors = ["#2a9d8f", "#b45309", "#7c3aed", "#6b7280"]
    for summary, color in zip(summaries, colors, strict=False):
        ax_heat.axvspan(
            float(summary["tmin"]), float(summary["tmax"]), color=color, alpha=0.08
        )
        ax_wphi.axvspan(
            float(summary["tmin"]), float(summary["tmax"]), color=color, alpha=0.08
        )
    start_idx = int(chosen["start_index"])
    late_t = t[start_idx:]
    late_heat = heat[start_idx:]
    ax_zoom.plot(
        late_t,
        late_heat,
        marker="o",
        markersize=3.5,
        linewidth=2.0,
        color="#0f4c81",
        label="heat flux",
    )
    ax_zoom.axhline(
        float(chosen["heat_flux_mean"]),
        color="#c44e52",
        linewidth=1.8,
        label="window mean",
    )
    ax_zoom.fill_between(
        late_t,
        float(chosen["heat_flux_mean"]) - float(chosen["heat_flux_std"]),
        float(chosen["heat_flux_mean"]) + float(chosen["heat_flux_std"]),
        color="#c44e52",
        alpha=0.12,
        label=r"$\pm 1\sigma$",
    )
    ax_zoom.set_xlabel("time")
    ax_zoom.set_ylabel("heat flux")
    ax_zoom.set_title(f"Least-trending window starts at t={float(chosen['tmin']):.2f}")
    ax_zoom.grid(True, alpha=0.25)
    ax_zoom.legend(frameon=False)

    ax_text.axis("off")
    lines = [
        label,
        f"claim: {claim_level}",
        f"samples: {t.size}, t=[{t[0]:.3g}, {t[-1]:.3g}]",
        f"final heat flux: {heat[-1]:.4g}",
        f"final W_phi: {wphi[-1]:.4g}",
        f"least-trending window: [{float(chosen['tmin']):.3g}, {float(chosen['tmax']):.3g}]",
        f"window heat mean: {float(chosen['heat_flux_mean']):.4g}",
        f"window heat std: {float(chosen['heat_flux_std']):.4g}",
        f"relative slope/time: {float(chosen['heat_flux_relative_slope_per_time']):.3g}",
        "",
        "Interpretation:",
        "finite long feasibility run; not promoted unless",
        "a saturated-window gate is defined and passed.",
    ]
    ax_text.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = out_path.with_suffix(".traces.csv")
    write_trace_csv(csv_path, {"t": t, "heat_flux": heat, "wphi": wphi, "wg": wg})
    json_path = out_path.with_suffix(".json")
    payload = {
        "kind": "nonlinear_feasibility_pilot",
        "claim_level": claim_level,
        "label": label,
        "source": None if source is None else str(source),
        "png": str(out_path),
        "pdf": str(pdf_path),
        "csv": str(csv_path),
        "n_samples": int(t.size),
        "tmin": float(t[0]),
        "tmax": float(t[-1]),
        "heat_flux_last": float(heat[-1]),
        "wphi_last": float(wphi[-1]),
        "wg_last": float(wg[-1]),
        "window_summaries": summaries,
        "least_trending_window": chosen,
        "promotion_gate": {
            "passed": False,
            "reason": "feasibility panel only; no external nonlinear transport acceptance gate is defined",
        },
    }
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "png": str(out_path),
        "pdf": str(pdf_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def _repo_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _repo_relative_pattern(pattern: str) -> str:
    repo = str(ROOT.resolve())
    raw = str(pattern)
    return raw[len(repo) + 1 :] if raw.startswith(repo + "/") else raw


def _case_label(case: str) -> str:
    return CASE_LABELS.get(
        case, case.replace("_nonlinear_window", "").replace("_", " ").title()
    )


def _case_mean_rel_gate(case: str, fallback: float | None) -> float:
    """Return the release mean-relative gate for one nonlinear window case."""

    if case in CASE_MEAN_REL_GATES:
        return float(CASE_MEAN_REL_GATES[case])
    if fallback is not None and np.isfinite(float(fallback)):
        return float(fallback)
    return float(DEFAULT_RELEASE_GATE)


def load_window_rows(
    paths: list[Path], metrics: tuple[str, ...] = DEFAULT_METRICS
) -> list[dict[str, object]]:
    """Load per-case/per-diagnostic windowed mismatch rows from gate summaries."""

    allowed = set(metrics)
    rows: list[dict[str, object]] = []
    for path in sorted(set(paths)):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} does not contain a JSON object")
        if data.get("gate_index_include") is False:
            continue
        case = str(data.get("case", path.stem))
        summary = data.get("summary", [])
        if not isinstance(summary, list):
            raise ValueError(f"{path} has a non-list summary")
        for item in summary:
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric", ""))
            if metric not in allowed:
                continue
            rows.append(
                {
                    "case": case,
                    "case_label": _case_label(case),
                    "metric": metric,
                    "mean_rel_abs": float(item.get("mean_rel_abs", np.nan)),
                    "max_rel_abs": float(item.get("max_rel_abs", np.nan)),
                    "final_rel": float(item.get("final_rel", np.nan)),
                    "gate_mean_rel": float(data.get("gate_mean_rel", np.nan)),
                    "case_gate_mean_rel": _case_mean_rel_gate(
                        case, float(data.get("gate_mean_rel", np.nan))
                    ),
                    "gate_passed": bool(data.get("gate_passed", False)),
                    "source": str(data.get("source", "")),
                    "artifact": _repo_relative_path(path),
                }
            )
    return sorted(
        rows,
        key=lambda row: (CASE_ORDER.get(str(row["case"]), 999), str(row["metric"])),
    )


def _metric_offsets(metrics: list[str]) -> dict[str, float]:
    if len(metrics) == 1:
        return {metrics[0]: 0.0}
    offsets = np.linspace(-0.26, 0.26, len(metrics))
    return {
        metric: float(offset) for metric, offset in zip(metrics, offsets, strict=True)
    }


def window_statistics_figure(
    rows: list[dict[str, object]],
    *,
    gate_threshold: float = DEFAULT_RELEASE_GATE,
    title: str = "Windowed nonlinear diagnostic agreement",
) -> plt.Figure:
    """Create the two-panel windowed nonlinear statistics summary figure."""

    if not rows:
        raise ValueError("no nonlinear window rows to plot")
    set_plot_style()
    cases = sorted(
        {str(row["case"]) for row in rows}, key=lambda case: CASE_ORDER.get(case, 999)
    )
    labels = [_case_label(case) for case in cases]
    metrics = [
        metric
        for metric in DEFAULT_METRICS
        if any(str(row["metric"]) == metric for row in rows)
    ]
    offsets = _metric_offsets(metrics)
    y_base = {case: idx for idx, case in enumerate(cases)}

    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, max(4.3, 0.62 * len(cases) + 1.6)), constrained_layout=True
    )
    panels = (
        (axes[0], "mean_rel_abs", "Mean relative mismatch", gate_threshold),
        (axes[1], "max_rel_abs", "Maximum relative mismatch", None),
    )
    for ax, key, xlabel, threshold in panels:
        for metric in metrics:
            xs = []
            ys = []
            for row in rows:
                if str(row["metric"]) != metric:
                    continue
                value = float(row[key])
                if not np.isfinite(value):
                    continue
                xs.append(value)
                ys.append(y_base[str(row["case"])] + offsets[metric])
            if xs:
                ax.scatter(
                    xs,
                    ys,
                    s=54,
                    color=METRIC_COLORS.get(metric, "#333333"),
                    edgecolor="white",
                    linewidth=0.7,
                    label=metric,
                    zorder=3,
                )
        if threshold is not None:
            labeled_gate = False
            for case in cases:
                case_threshold = _case_mean_rel_gate(case, threshold)
                ax.plot(
                    [case_threshold, case_threshold],
                    [y_base[case] - 0.34, y_base[case] + 0.34],
                    color="#c2410c",
                    linestyle="--",
                    linewidth=1.9,
                    label="case gate" if not labeled_gate else None,
                    zorder=2,
                )
                labeled_gate = True
        ax.set_yticks(range(len(cases)), labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", alpha=0.25)
        ax.grid(False, axis="y")
        ax.set_xlim(left=-0.005)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_title("Release-window gate statistic")
    axes[1].set_title("Transient/envelope sensitivity")
    handles, labels_ = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels_, handles, strict=False))
    axes[1].legend(
        by_label.values(),
        by_label.keys(),
        loc="lower right",
        frameon=True,
        framealpha=0.92,
    )
    fig.suptitle(title, y=1.03, fontsize=14, fontweight="bold")
    return fig


def write_rows_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write the plotted rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "case_label",
        "metric",
        "mean_rel_abs",
        "max_rel_abs",
        "final_rel",
        "gate_mean_rel",
        "case_gate_mean_rel",
        "gate_passed",
        "source",
        "artifact",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(
    rows: list[dict[str, object]],
    path: Path,
    *,
    gate_threshold: float,
    patterns: list[str],
) -> None:
    """Write machine-readable metadata for the nonlinear window panel."""

    cases = sorted(
        {str(row["case"]) for row in rows}, key=lambda case: CASE_ORDER.get(case, 999)
    )
    metrics = sorted({str(row["metric"]) for row in rows})
    max_mean_by_case = {
        case: max(
            float(row["mean_rel_abs"])
            for row in rows
            if str(row["case"]) == case and np.isfinite(float(row["mean_rel_abs"]))
        )
        for case in cases
    }
    case_gate_thresholds = {
        case: _case_mean_rel_gate(case, gate_threshold) for case in cases
    }
    case_gate_passed = {
        case: bool(max_mean_by_case[case] <= case_gate_thresholds[case])
        for case in cases
    }
    payload = {
        "n_cases": len(cases),
        "n_rows": len(rows),
        "cases": cases,
        "metrics": metrics,
        "gate_threshold": float(gate_threshold),
        "all_cases_pass_gate": all(
            value <= float(gate_threshold) for value in max_mean_by_case.values()
        ),
        "case_gate_thresholds": case_gate_thresholds,
        "case_gate_passed": case_gate_passed,
        "all_cases_pass_case_gates": all(case_gate_passed.values()),
        "max_mean_rel_abs_by_case": max_mean_by_case,
        "patterns": [_repo_relative_pattern(pattern) for pattern in patterns],
        "rows": rows,
        "notes": (
            "This artifact summarizes frozen nonlinear release-window gate JSONs. "
            "It excludes exploratory summaries with gate_index_include=false and does not rerun simulations. "
            "Case-specific mean-relative gates tighten KBM/HSX and the closed Cyclone-Miller window while "
            "leaving Cyclone and W7-X at the broad release envelope pending paper-level retuning."
        ),
    }
    path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _parse_fractions(raw: str) -> tuple[float, ...]:
    return tuple(float(item) for item in raw.split(",") if item.strip())


def build_feasibility_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a finite nonlinear feasibility panel without promoting transport gates."
    )
    parser.add_argument(
        "--input", required=True, help="Nonlinear diagnostic *.out.nc file."
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_FEASIBILITY_OUT), help="Output PNG path."
    )
    parser.add_argument("--title", default="Nonlinear Feasibility Pilot")
    parser.add_argument("--label", default="external VMEC")
    parser.add_argument(
        "--claim-level", default="finite_nonlinear_feasibility_not_transport_validation"
    )
    parser.add_argument("--fractions", default="0.5,0.6,0.7,0.8")
    return parser


def build_window_statistics_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the frozen release-window nonlinear statistics panel."
    )
    parser.add_argument(
        "--glob",
        action="append",
        dest="patterns",
        default=None,
        help="Input gate-summary JSON glob.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_WINDOW_OUT, help="Output PNG path."
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=DEFAULT_RELEASE_GATE,
        help="Fallback mean-relative release gate threshold.",
    )
    parser.add_argument(
        "--title",
        default="Windowed nonlinear diagnostic agreement",
        help="Figure title.",
    )
    return parser


def run_feasibility(argv: list[str]) -> int:
    args = build_feasibility_parser().parse_args(argv)
    trace = load_nonlinear_trace(args.input)
    paths = write_pilot_panel(
        trace,
        out=args.out,
        source=args.input,
        title=args.title,
        label=args.label,
        claim_level=args.claim_level,
        start_fractions=_parse_fractions(args.fractions),
    )
    for kind in ("png", "pdf", "json", "csv"):
        print(f"saved {paths[kind]}")
    return 0


def run_window_statistics(argv: list[str]) -> int:
    args = build_window_statistics_parser().parse_args(argv)
    patterns = args.patterns or [DEFAULT_WINDOW_GLOB]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(item) for item in glob.glob(pattern, recursive=True))
    rows = load_window_rows(paths)
    fig = window_statistics_figure(
        rows, gate_threshold=float(args.gate_threshold), title=str(args.title)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    write_rows_csv(rows, args.out.with_suffix(".csv"))
    write_summary_json(
        rows,
        args.out.with_suffix(".json"),
        gate_threshold=float(args.gate_threshold),
        patterns=patterns,
    )
    for path in (
        args.out,
        args.out.with_suffix(".pdf"),
        args.out.with_suffix(".csv"),
        args.out.with_suffix(".json"),
    ):
        print(f"Wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("feasibility", "window-statistics"))
    args, remainder = parser.parse_known_args(argv)
    if args.mode == "feasibility":
        return run_feasibility(remainder)
    return run_window_statistics(remainder)


if __name__ == "__main__":
    raise SystemExit(main())
