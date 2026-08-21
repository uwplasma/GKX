#!/usr/bin/env python3
"""Build digitized and comparison artifacts for the W7-X zonal reference."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from gkx.artifacts.plotting import set_plot_style
from gkx.diagnostics.validation_gates import (
    evaluate_scalar_gate,
    gate_report,
    gate_report_to_dict,
)
from gkx.diagnostics.zonal_validation import (
    kx_token,
    load_w7x_combined_trace_csv,
    load_w7x_trace_csv,
    normalize_trace,
    reference_mean_trace,
    reference_residual_table,
    reference_time_limits,
    tail_trace_metrics,
    w7x_trace_path,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_IMAGE_SHAPE = (1152, 2160)
KX_VALUES = (0.05, 0.07, 0.10, 0.30)


@dataclass(frozen=True)
class PanelCalibration:
    label: str
    kx_rhoi: float
    main_box: tuple[int, int, int, int]
    inset_box: tuple[int, int, int, int]
    t_range: tuple[float, float]
    inset_t_range: tuple[float, float]
    y_range: tuple[float, float] = (-0.5, 1.0)
    inset_y_range: tuple[float, float] = (-0.1, 0.2)


PANEL_CALIBRATIONS = (
    PanelCalibration(
        "a",
        0.05,
        (216, 972, 57, 460),
        (648, 941, 74, 207),
        (0.0, 3500.0),
        (3300.0, 3500.0),
    ),
    PanelCalibration(
        "b",
        0.07,
        (1188, 1944, 57, 460),
        (1620, 1913, 74, 207),
        (0.0, 2000.0),
        (1800.0, 2000.0),
    ),
    PanelCalibration(
        "c",
        0.10,
        (216, 972, 610, 1013),
        (648, 941, 627, 760),
        (0.0, 2000.0),
        (1800.0, 2000.0),
    ),
    PanelCalibration(
        "d",
        0.30,
        (1188, 1944, 610, 1013),
        (1620, 1913, 627, 760),
        (0.0, 2000.0),
        (1800.0, 2000.0),
    ),
)

EXCLUSION_BOXES = (
    (560, 945, 70, 260),
    (1530, 1918, 70, 260),
    (560, 945, 625, 812),
    (1530, 1918, 625, 812),
    (705, 960, 335, 450),
    (1678, 1935, 335, 450),
    (705, 960, 888, 1005),
    (1678, 1935, 888, 1005),
)

REFERENCE_METADATA = {
    "paper": "Gonzalez-Jerez et al., Journal of Plasma Physics 88, 905880310 (2022)",
    "arxiv": "2107.06060",
    "figure": "Figure 11 / source figs/ZF.pdf",
    "observable": "line-averaged electrostatic potential normalized to its t=0 value",
    "test": "W7-X high-mirror bean-tube test 4 zonal-flow relaxation",
}


def _build_digitize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        type=Path,
        required=True,
        help="Path to the arXiv-source figs/ZF.pdf or a 2160x1152 PNG rendered from it at 2x.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized.csv",
        help="Output CSV containing the digitized main traces.",
    )
    parser.add_argument(
        "--out-residual-csv",
        type=Path,
        default=ROOT
        / "docs"
        / "_static"
        / "w7x_zonal_reference_digitized_residuals.csv",
        help="Output CSV containing the digitized inset residual levels.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized.json",
        help="Output JSON metadata path.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized.png",
        help="Output QA plot path.",
    )
    parser.add_argument("--samples-per-trace", type=int, default=1001)
    return parser


def _load_reference_image(path: Path) -> np.ndarray:
    """Load the source figure as RGB pixels using the calibration render size."""

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - exercised only without PyMuPDF
            raise RuntimeError(
                "PyMuPDF/fitz is required to render the source PDF figure"
            ) from exc
        doc = fitz.open(path)
        if len(doc) != 1:
            raise ValueError(f"{path} should contain one page")
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    if arr.shape[:2] != EXPECTED_IMAGE_SHAPE:
        raise ValueError(
            f"{path} renders to {arr.shape[:2]}, expected {EXPECTED_IMAGE_SHAPE}; "
            "use the arXiv source figs/ZF.pdf or a 2x render of that file"
        )
    return arr


def _curve_mask(rgb: np.ndarray, code: str) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.int16)
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    if code == "stella":
        return (
            (red > 170)
            & (green < 110)
            & (blue < 110)
            & ((red - green) > 70)
            & ((red - blue) > 70)
        )
    if code == "GENE":
        return (
            (blue > 140)
            & (red < 130)
            & (green < 150)
            & ((blue - red) > 50)
            & ((blue - green) > 30)
        )
    raise ValueError("code must be 'stella' or 'GENE'")


def _apply_exclusions(
    mask: np.ndarray,
    *,
    box: tuple[int, int, int, int],
    exclusions: Iterable[tuple[int, int, int, int]],
) -> np.ndarray:
    x0, _x1, y0, _y1 = box
    yy, xx = np.indices(mask.shape)
    global_x = xx + int(x0)
    global_y = yy + int(y0)
    out = np.array(mask, dtype=bool, copy=True)
    for ex0, ex1, ey0, ey1 in exclusions:
        out &= ~(
            (global_x >= ex0)
            & (global_x <= ex1)
            & (global_y >= ey0)
            & (global_y <= ey1)
        )
    return out


def _pixel_to_data(
    x_pixels: np.ndarray,
    y_pixels: np.ndarray,
    *,
    box: tuple[int, int, int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1 = box
    x = float(x_range[0]) + (np.asarray(x_pixels, dtype=float) - float(x0)) / float(
        x1 - x0
    ) * float(x_range[1] - x_range[0])
    y = float(y_range[1]) - (np.asarray(y_pixels, dtype=float) - float(y0)) / float(
        y1 - y0
    ) * float(y_range[1] - y_range[0])
    return x, y


def _digitize_box(
    image: np.ndarray,
    *,
    box: tuple[int, int, int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    code: str,
    exclusions: Iterable[tuple[int, int, int, int]] = (),
) -> tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1 = box
    sub = image[y0 : y1 + 1, x0 : x1 + 1]
    mask = _curve_mask(sub, code)
    if exclusions:
        mask = _apply_exclusions(mask, box=box, exclusions=exclusions)
    x_pixels: list[int] = []
    y_pixels: list[float] = []
    for local_x in range(mask.shape[1]):
        local_y = np.flatnonzero(mask[:, local_x])
        if local_y.size:
            x_pixels.append(local_x + x0)
            y_pixels.append(float(np.median(local_y + y0)))
    if not x_pixels:
        raise ValueError(f"no {code} pixels found in calibrated box {box}")
    x_data, y_data = _pixel_to_data(
        np.asarray(x_pixels, dtype=float),
        np.asarray(y_pixels, dtype=float),
        box=box,
        x_range=x_range,
        y_range=y_range,
    )
    order = np.argsort(x_data)
    return x_data[order], y_data[order]


def digitize_reference(
    image: np.ndarray, *, samples_per_trace: int = 1001
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return digitized main traces and inset residual summaries."""

    trace_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    for panel in PANEL_CALIBRATIONS:
        for code in ("stella", "GENE"):
            t_raw, y_raw = _digitize_box(
                image,
                box=panel.main_box,
                x_range=panel.t_range,
                y_range=panel.y_range,
                code=code,
                exclusions=EXCLUSION_BOXES,
            )
            t_grid = np.linspace(
                float(panel.t_range[0]), float(panel.t_range[1]), int(samples_per_trace)
            )
            y_grid = np.interp(t_grid, t_raw, y_raw)
            for t_value, response in zip(t_grid, y_grid, strict=True):
                trace_rows.append(
                    {
                        "panel": panel.label,
                        "kx_rhoi": panel.kx_rhoi,
                        "code": code,
                        "t_vti_over_a": float(t_value),
                        "response": float(response),
                    }
                )

            _t_inset, y_inset = _digitize_box(
                image,
                box=panel.inset_box,
                x_range=panel.inset_t_range,
                y_range=panel.inset_y_range,
                code=code,
            )
            residual_rows.append(
                {
                    "panel": panel.label,
                    "kx_rhoi": panel.kx_rhoi,
                    "code": code,
                    "residual_mean": float(np.mean(y_inset)),
                    "residual_median": float(np.median(y_inset)),
                    "residual_min": float(np.min(y_inset)),
                    "residual_max": float(np.max(y_inset)),
                    "n_pixels": int(y_inset.size),
                }
            )
    return pd.DataFrame(trace_rows), pd.DataFrame(residual_rows)


def _write_digitize_plot(
    trace_df: pd.DataFrame, residual_df: pd.DataFrame, out_png: Path
) -> None:
    set_plot_style()
    colors = {"stella": "#d62728", "GENE": "#1f4fff"}
    styles = {"stella": "-", "GENE": "--"}
    fig, axes = plt.subplots(
        2, 2, figsize=(11.0, 6.4), sharey=True, constrained_layout=True
    )
    for axis, panel in zip(axes.flat, PANEL_CALIBRATIONS, strict=True):
        subset = trace_df[trace_df["panel"] == panel.label]
        residual_subset = residual_df[residual_df["panel"] == panel.label]
        for code in ("stella", "GENE"):
            rows = subset[subset["code"] == code]
            axis.plot(
                rows["t_vti_over_a"],
                rows["response"],
                color=colors[code],
                linestyle=styles[code],
                linewidth=2.0,
                label=code,
            )
            residual = residual_subset[residual_subset["code"] == code][
                "residual_median"
            ].iloc[0]
            axis.axhline(
                float(residual),
                color=colors[code],
                linestyle=styles[code],
                linewidth=1.2,
                alpha=0.55,
            )
        axis.set_title(rf"$k_x \rho_i = {panel.kx_rhoi:.2f}$")
        axis.set_xlabel(r"$t v_{th,i}/a$")
        axis.set_ylabel(
            r"$\langle \mathrm{Re}(\hat{\phi}_k)\rangle_z / \langle \mathrm{Re}(\hat{\phi}_k)\rangle_z(t=0)$"
        )
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Digitized W7-X test-4 zonal-flow reference (Gonzalez-Jerez et al. 2022)"
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _repo_relative(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _main_digitize(argv: list[str]) -> int:
    args = _build_digitize_parser().parse_args(argv)
    image = _load_reference_image(args.figure)
    trace_df, residual_df = digitize_reference(
        image, samples_per_trace=int(args.samples_per_trace)
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(args.out_csv, index=False)
    residual_df.to_csv(args.out_residual_csv, index=False)
    _write_digitize_plot(trace_df, residual_df, args.out_png)

    payload = {
        **REFERENCE_METADATA,
        "source_figure": str(args.figure),
        "digitization_method": (
            "Color-threshold digitization of the arXiv source Figure 11 PDF rendered at 2x. "
            "Main traces exclude inset and legend boxes; residual levels are read from the insets."
        ),
        "image_shape": list(image.shape[:2]),
        "trace_csv": _repo_relative(args.out_csv),
        "residual_csv": _repo_relative(args.out_residual_csv),
        "qa_plot": _repo_relative(args.out_png),
        "calibrations": [asdict(item) for item in PANEL_CALIBRATIONS],
        "residual_summary": residual_df.to_dict(orient="records"),
        "validation_status": "reference",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_residual_csv}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_png}")
    return 0


def _build_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gkx-summary",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_response_panel.csv",
        help="GKX zonal summary CSV written by the response-panel mode.",
    )
    parser.add_argument(
        "--gkx-trace-dir",
        type=Path,
        default=None,
        help="Optional directory containing per-kx w7x_test4_kxNNN.csv trace files.",
    )
    parser.add_argument(
        "--gkx-traces",
        type=Path,
        default=None,
        help=(
            "Optional combined trace CSV written next to the W7-X response panel. "
            "This is mutually exclusive with --gkx-trace-dir."
        ),
    )
    parser.add_argument(
        "--reference-traces",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized.csv",
        help="Digitized stella/GENE main trace CSV.",
    )
    parser.add_argument(
        "--reference-residuals",
        type=Path,
        default=ROOT
        / "docs"
        / "_static"
        / "w7x_zonal_reference_digitized_residuals.csv",
        help="Digitized stella/GENE inset residual CSV.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_compare.csv",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_compare.json",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_compare.png",
    )
    parser.add_argument("--residual-atol", type=float, default=0.02)
    parser.add_argument("--residual-rtol", type=float, default=0.10)
    parser.add_argument("--coverage-fraction", type=float, default=0.98)
    parser.add_argument("--tail-fraction", type=float, default=0.10)
    parser.add_argument("--envelope-atol", type=float, default=0.03)
    parser.add_argument(
        "--trace-normalization",
        choices=("summary_initial_level", "first_nonzero"),
        default="summary_initial_level",
        help=(
            "Normalization for optional trace-shape gates. summary_initial_level "
            "keeps the envelope metric consistent with the residual normalization "
            "recorded by the response-panel artifact mode."
        ),
    )
    parser.add_argument("--gate-index-include", action="store_true")
    return parser


def _load_gkx_summary(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"kx_target", "residual_level", "residual_std", "tmax"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return table


def _optional_trace_metrics(
    *,
    trace_dir: Path | None,
    combined_trace_csv: Path | None,
    reference_traces: pd.DataFrame,
    kx: float,
    tail_fraction: float,
    initial_level: float | None,
) -> dict[str, float | int | None]:
    if trace_dir is None and combined_trace_csv is None:
        return {
            "trace_available": 0,
            "tail_std": None,
            "reference_tail_std": None,
            "tail_mean_abs_error": None,
            "tail_max_abs_error": None,
        }
    if combined_trace_csv is not None:
        if not combined_trace_csv.exists():
            return {
                "trace_available": 0,
                "tail_std": None,
                "reference_tail_std": None,
                "tail_mean_abs_error": None,
                "tail_max_abs_error": None,
            }
        t_raw, y_raw = load_w7x_combined_trace_csv(combined_trace_csv, kx)
    else:
        if trace_dir is None:
            raise ValueError("trace_dir is unexpectedly None")
        path = w7x_trace_path(trace_dir, kx)
        if not path.exists():
            return {
                "trace_available": 0,
                "tail_std": None,
                "reference_tail_std": None,
                "tail_mean_abs_error": None,
                "tail_max_abs_error": None,
            }
        t_raw, y_raw = load_w7x_trace_csv(path)
    t_obs, y_obs = normalize_trace(t_raw, y_raw, initial_level=initial_level)
    ref_t, ref_y = reference_mean_trace(reference_traces, kx)
    return {
        "trace_available": 1,
        **tail_trace_metrics(
            t_obs=t_obs,
            y_obs=y_obs,
            t_ref=ref_t,
            y_ref=ref_y,
            tail_fraction=float(tail_fraction),
        ),
    }


def build_comparison(
    *,
    gkx_summary: Path,
    reference_traces: Path,
    reference_residuals: Path,
    gkx_trace_dir: Path | None = None,
    gkx_traces: Path | None = None,
    residual_atol: float = 0.02,
    residual_rtol: float = 0.10,
    coverage_fraction: float = 0.98,
    tail_fraction: float = 0.10,
    envelope_atol: float = 0.03,
    trace_normalization: str = "summary_initial_level",
):
    if gkx_trace_dir is not None and gkx_traces is not None:
        raise ValueError(
            "gkx_trace_dir and gkx_traces are mutually exclusive"
        )
    summary = _load_gkx_summary(gkx_summary)
    ref_traces = pd.read_csv(reference_traces)
    ref_residuals = reference_residual_table(reference_residuals)
    ref_limits = reference_time_limits(ref_traces)
    ref = pd.merge(ref_residuals, ref_limits, on="kx", how="inner")
    rows: list[dict[str, object]] = []
    gates = []
    for kx in KX_VALUES:
        obs_matches = summary[np.isclose(summary["kx_target"], float(kx))]
        ref_matches = ref[np.isclose(ref["kx"], float(kx))]
        if obs_matches.empty:
            raise ValueError(f"missing GKX summary row for kx={kx}")
        if ref_matches.empty:
            raise ValueError(f"missing reference row for kx={kx}")
        obs = obs_matches.iloc[0]
        ref_row = ref_matches.iloc[0]
        residual_ref = float(ref_row["reference_residual"])
        residual_atol_eff = float(residual_atol) + float(
            ref_row["reference_code_spread"]
        )
        residual_gate = evaluate_scalar_gate(
            f"residual_kx{kx_token(kx)}",
            float(obs["residual_level"]),
            residual_ref,
            atol=residual_atol_eff,
            rtol=float(residual_rtol),
            notes=(
                "Residual compared with the mean of digitized stella/GENE inset values; "
                "absolute tolerance includes the inter-code spread."
            ),
        )
        coverage_ratio = min(float(obs["tmax"]) / float(ref_row["reference_tmax"]), 1.0)
        coverage_gate = evaluate_scalar_gate(
            f"time_coverage_kx{kx_token(kx)}",
            coverage_ratio,
            1.0,
            atol=1.0 - float(coverage_fraction),
            rtol=0.0,
            notes=f"Passes when GKX reaches at least {coverage_fraction:.0%} of the digitized reference window.",
        )
        gates.extend([coverage_gate, residual_gate])
        trace_initial_level = None
        trace_source_provided = (
            gkx_trace_dir is not None or gkx_traces is not None
        )
        if (
            str(trace_normalization).strip().lower().replace("-", "_")
            == "summary_initial_level"
        ):
            if trace_source_provided and "initial_level" not in obs.index:
                raise ValueError(
                    "summary_initial_level trace normalization requires an initial_level column "
                    "in the GKX summary CSV"
                )
            trace_initial_level = (
                float(obs["initial_level"]) if trace_source_provided else None
            )
        elif (
            str(trace_normalization).strip().lower().replace("-", "_")
            != "first_nonzero"
        ):
            raise ValueError(
                "trace_normalization must be one of {'summary_initial_level', 'first_nonzero'}"
            )
        trace_metrics = _optional_trace_metrics(
            trace_dir=gkx_trace_dir,
            combined_trace_csv=gkx_traces,
            reference_traces=ref_traces,
            kx=kx,
            tail_fraction=float(tail_fraction),
            initial_level=trace_initial_level,
        )
        if (
            trace_metrics["tail_std"] is not None
            and trace_metrics["reference_tail_std"] is not None
        ):
            gates.append(
                evaluate_scalar_gate(
                    f"tail_envelope_std_kx{kx_token(kx)}",
                    float(trace_metrics["tail_std"]),
                    float(trace_metrics["reference_tail_std"]),
                    atol=float(envelope_atol),
                    rtol=0.0,
                    notes="Late-window oscillation envelope compared against digitized stella/GENE mean trace.",
                )
            )
        rows.append(
            {
                "kx": float(kx),
                "gkx_residual": float(obs["residual_level"]),
                "gkx_residual_std": float(obs["residual_std"]),
                "gkx_tmax": float(obs["tmax"]),
                "reference_residual": residual_ref,
                "reference_min": float(ref_row["reference_min"]),
                "reference_max": float(ref_row["reference_max"]),
                "reference_tmax": float(ref_row["reference_tmax"]),
                "coverage_ratio": float(coverage_ratio),
                "residual_abs_error": float(
                    abs(float(obs["residual_level"]) - residual_ref)
                ),
                "residual_atol_effective": residual_atol_eff,
                **trace_metrics,
            }
        )
    report = gate_report(
        "w7x_zonal_response_reference",
        "digitized stella/GENE W7-X test-4 Fig. 11",
        tuple(gates),
    )
    return pd.DataFrame(rows), report


def _write_comparison_plot(rows: pd.DataFrame, out_png: Path) -> None:
    set_plot_style()
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.fill_between(
        x,
        np.asarray(rows["reference_min"], dtype=float),
        np.asarray(rows["reference_max"], dtype=float),
        color="#8ecae6",
        alpha=0.45,
        label="digitized stella/GENE band",
    )
    ax.plot(
        x,
        rows["reference_residual"],
        color="#1d4e89",
        marker="o",
        linewidth=2.0,
        label="reference mean",
    )
    ax.plot(
        x,
        rows["gkx_residual"],
        color="#c2410c",
        marker="s",
        linewidth=2.0,
        label="GKX",
    )
    ax.set_xticks(x, [f"{value:.2f}" for value in rows["kx"]])
    ax.set_xlabel(r"$k_x \rho_i$")
    ax.set_ylabel("late residual")
    ax.set_title("Residual Gate")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar(x, rows["coverage_ratio"], color="#2a9d55", alpha=0.85)
    ax.axhline(0.98, color="#c2410c", linestyle="--", linewidth=1.5, label="98% gate")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x, [f"{value:.2f}" for value in rows["kx"]])
    ax.set_xlabel(r"$k_x \rho_i$")
    ax.set_ylabel("covered reference window")
    ax.set_title("Time Coverage")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _main_compare(argv: list[str]) -> int:
    args = _build_compare_parser().parse_args(argv)
    rows, report = build_comparison(
        gkx_summary=args.gkx_summary,
        reference_traces=args.reference_traces,
        reference_residuals=args.reference_residuals,
        gkx_trace_dir=args.gkx_trace_dir,
        gkx_traces=args.gkx_traces,
        residual_atol=float(args.residual_atol),
        residual_rtol=float(args.residual_rtol),
        coverage_fraction=float(args.coverage_fraction),
        tail_fraction=float(args.tail_fraction),
        envelope_atol=float(args.envelope_atol),
        trace_normalization=str(args.trace_normalization),
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out_csv, index=False)
    _write_comparison_plot(rows, args.out_png)
    payload = {
        "case": "w7x_zonal_response_reference",
        "validation_status": "closed" if report.passed else "open",
        "gate_index_include": bool(args.gate_index_include),
        "gate_report": gate_report_to_dict(report),
        "gkx_summary": _repo_relative(args.gkx_summary),
        "gkx_trace_dir": None
        if args.gkx_trace_dir is None
        else _repo_relative(args.gkx_trace_dir),
        "gkx_traces": None
        if args.gkx_traces is None
        else _repo_relative(args.gkx_traces),
        "reference_traces": _repo_relative(args.reference_traces),
        "reference_residuals": _repo_relative(args.reference_residuals),
        "comparison_csv": _repo_relative(args.out_csv),
        "comparison_png": _repo_relative(args.out_png),
        "trace_normalization": str(args.trace_normalization),
        "notes": (
            "This gate compares GKX W7-X test-4 zonal-flow residuals and, when trace CSVs are available, "
            "late-window oscillation envelopes against digitized stella/GENE Figure 11 references. The current "
            "paper-normalized long-window artifact closes the time-coverage gates, but residuals remain open at "
            "three wavelengths and the late-window envelope gates remain open. This is tracked as a physics/numerics "
            "closure lane rather than a documentation-only mismatch."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_png}")
    return 0 if report.passed else 2


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        print("usage: build_w7x_zonal_reference_artifacts.py {digitize,compare} ...")
        return 2
    command, rest = tokens[0], tokens[1:]
    if command == "digitize":
        return _main_digitize(rest)
    if command == "compare":
        return _main_compare(rest)
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
