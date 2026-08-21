#!/usr/bin/env python3
"""Build tracked reference-comparison panels for docs and release artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC = REPO_ROOT / "docs" / "_static"
REQUIRED_LINEAR_COLUMNS = {"ky", "gamma", "omega", "gamma_gx", "omega_gx"}


def _resolve(path: str | Path) -> Path:
    """Resolve CLI paths relative to the repository root."""

    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _autocrop_image(
    image: np.ndarray,
    *,
    white_threshold: float = 0.985,
    pad_pixels: int = 12,
) -> np.ndarray:
    """Trim uniform white borders from an RGB/RGBA image array."""

    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return arr
    rgb = arr[..., :3]
    non_white = np.any(rgb < white_threshold, axis=2)
    rows = np.where(np.any(non_white, axis=1))[0]
    cols = np.where(np.any(non_white, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return arr
    r0 = max(int(rows[0]) - pad_pixels, 0)
    r1 = min(int(rows[-1]) + pad_pixels + 1, arr.shape[0])
    c0 = max(int(cols[0]) - pad_pixels, 0)
    c1 = min(int(cols[-1]) + pad_pixels + 1, arr.shape[1])
    return arr[r0:r1, c0:c1]


def _mean_rel(
    lhs: np.ndarray,
    rhs: np.ndarray,
    floor_fraction: float = 1.0e-8,
) -> float:
    """Return a robust mean relative mismatch with a scale-dependent floor."""

    lhs_f = np.asarray(lhs, dtype=float)
    rhs_f = np.asarray(rhs, dtype=float)
    finite = np.isfinite(lhs_f) & np.isfinite(rhs_f)
    if not np.any(finite):
        return float("nan")
    rhs_sel = rhs_f[finite]
    floor = max(float(np.nanmax(np.abs(rhs_sel))) * floor_fraction, 1.0e-30)
    denom = np.maximum(np.abs(rhs_sel), floor)
    return float(np.nanmean(np.abs(lhs_f[finite] - rhs_sel) / denom))


def _load_secondary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ky", "kx", "gamma_gx", "gamma_sp", "rel_gamma", "omega_gx", "omega_sp"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    if "abs_omega" not in df.columns:
        df["abs_omega"] = (df["omega_sp"] - df["omega_gx"]).abs()
    return df.sort_values(["ky", "kx"]).reset_index(drop=True)


def _secondary_table_rows(df: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in df.itertuples(index=False):
        rows.append(
            [
                f"({row.ky:.2f}, {row.kx:+.2f})",
                f"{row.gamma_gx:.6f}",
                f"{row.gamma_sp:.6f}",
                f"{row.rel_gamma:.2e}",
                f"{row.omega_gx:.2e}",
                f"{row.omega_sp:.2e}",
                f"{row.abs_omega:.2e}",
            ]
        )
    return rows


def _load_imported_linear(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "ky",
        "mean_abs_omega",
        "mean_rel_omega",
        "mean_abs_gamma",
        "mean_rel_gamma",
        "mean_rel_Wg",
        "mean_rel_Wphi",
        "mean_rel_Wapar",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    return df.sort_values("ky").reset_index(drop=True)


def _load_imported_linear_lastvalue(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ky", "rel_gamma", "rel_omega", "gamma", "gamma_gx", "omega", "omega_gx"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    return df.sort_values("ky").reset_index(drop=True)


def _linear_table_rows(df: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in df.itertuples(index=False):
        rows.append(
            [
                f"{row.ky:.3f}",
                f"{row.mean_abs_omega:.2e}",
                f"{row.mean_abs_gamma:.2e}",
                f"{row.mean_rel_Wg:.2e}",
                f"{row.mean_rel_Wphi:.2e}",
            ]
        )
    return rows


def _load_linear_mismatch(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_LINEAR_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    return df.sort_values("ky").reset_index(drop=True)


def _plot_imported_linear(
    ax: Axes,
    df: pd.DataFrame,
    title: str,
    *,
    lastvalue: pd.DataFrame | None = None,
    note: str | None = None,
) -> None:
    ky = np.asarray(df["ky"], dtype=float)
    ax.plot(ky, np.asarray(df["mean_rel_omega"], dtype=float), marker="o", linewidth=2.0, label="window rel ω")
    ax.plot(ky, np.asarray(df["mean_rel_gamma"], dtype=float), marker="s", linewidth=2.0, label="window rel γ")
    ax.plot(ky, np.asarray(df["mean_rel_Wg"], dtype=float), marker="^", linewidth=2.0, linestyle="--", label="window rel Wg")
    ax.plot(
        ky,
        np.asarray(df["mean_rel_Wphi"], dtype=float),
        marker="d",
        linewidth=2.0,
        linestyle="--",
        label="window rel Wphi",
    )
    if "mean_rel_Wapar" in df.columns and np.any(np.asarray(df["mean_rel_Wapar"], dtype=float) > 0.0):
        ax.plot(
            ky,
            np.asarray(df["mean_rel_Wapar"], dtype=float),
            marker="x",
            linewidth=1.8,
            linestyle=":",
            label="window rel Wapar",
        )
    if lastvalue is not None:
        ky_last = np.asarray(lastvalue["ky"], dtype=float)
        ax.plot(
            ky_last,
            np.abs(np.asarray(lastvalue["rel_omega"], dtype=float)),
            marker="o",
            linewidth=2.2,
            linestyle=":",
            color="#c2410c",
            label="late rel ω",
        )
        ax.plot(
            ky_last,
            np.abs(np.asarray(lastvalue["rel_gamma"], dtype=float)),
            marker="s",
            linewidth=2.2,
            linestyle=":",
            color="#7c3aed",
            label="late rel γ",
        )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("ky")
    ax.set_yscale("log")
    ax.set_ylabel("relative mismatch")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="upper left")
    if note:
        ax.text(
            0.03,
            0.03,
            note,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            color="#334155",
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.25"},
        )


def _plot_secondary(ax: Axes, df: pd.DataFrame, title: str) -> None:
    modes = [f"({row.ky:.2f},{row.kx:+.2f})" for row in df.itertuples(index=False)]
    x = np.arange(len(modes), dtype=float)
    ax.plot(x, np.asarray(df["gamma_gx"], dtype=float), marker="o", linewidth=2.0, label="γ reference")
    ax.plot(x, np.asarray(df["gamma_sp"], dtype=float), marker="s", linewidth=2.0, linestyle="--", label="γ GKX")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("gamma")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=25, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax_omega = cast(Axes, ax.twinx())
    ax_omega.bar(
        x,
        np.asarray(df["abs_omega"], dtype=float),
        width=0.55,
        color="0.75",
        alpha=0.5,
        label="|Δω|",
    )
    ax_omega.set_ylabel("|Δω|")
    handles_l, labels_l = ax.get_legend_handles_labels()
    handles_r, labels_r = ax_omega.get_legend_handles_labels()
    ax.legend(handles_l + handles_r, labels_l + labels_r, fontsize=8, frameon=False, loc="upper left")


def _plot_linear_metric(ax: Axes, df: pd.DataFrame, *, metric: str, title: str) -> None:
    ky = np.asarray(df["ky"], dtype=float)
    reference = np.asarray(df[f"{metric}_gx"], dtype=float)
    gkx = np.asarray(df[metric], dtype=float)
    ax.plot(ky, reference, marker="o", linewidth=2.2, color="#111111", label="reference")
    ax.plot(ky, gkx, marker="s", linewidth=2.2, color="#d1495b", linestyle="--", label="GKX")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(r"$k_y \rho_i$")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    rel = _mean_rel(gkx, reference, floor_fraction=1.0e-12)
    ax.text(
        0.03,
        0.97,
        f"mean rel err = {rel:.2e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )
    ax.legend(frameon=False, fontsize=8, loc="best")


def _add_tokamak_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cyclone-linear",
        type=Path,
        default=STATIC / "cyclone_miller_linear_mismatch.csv",
        help="Tracked Cyclone Miller linear mismatch CSV.",
    )
    parser.add_argument(
        "--kbm-linear",
        type=Path,
        default=STATIC / "comparison" / "kbm_reference_mismatch.csv",
        help="Tracked KBM linear mismatch CSV.",
    )
    parser.add_argument(
        "--cyclone-nonlinear-panel",
        type=Path,
        default=STATIC / "nonlinear_cyclone_miller_diag_compare_t122.png",
        help="Tracked Cyclone Miller nonlinear comparison figure.",
    )
    parser.add_argument(
        "--kbm-nonlinear-panel",
        type=Path,
        default=STATIC / "nonlinear_kbm_diag_compare_t400_stats.png",
        help="Tracked KBM nonlinear comparison figure.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=STATIC / "comparison" / "reference_cyclone_kbm_panel.png",
        help="Output tokamak panel path.",
    )


def _add_stellarator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cyclone-kbm-panel",
        type=Path,
        default=STATIC / "comparison" / "reference_cyclone_kbm_panel.png",
        help="Tracked Cyclone/KBM tokamak panel.",
    )
    parser.add_argument(
        "--w7x-panel",
        type=Path,
        default=STATIC / "nonlinear_w7x_diag_compare_t200.png",
        help="Tracked nonlinear W7-X comparison figure.",
    )
    parser.add_argument(
        "--hsx-panel",
        type=Path,
        default=STATIC / "hsx_nonlinear_compare_t50_true.png",
        help="Tracked nonlinear HSX comparison figure.",
    )
    parser.add_argument(
        "--w7x-linear-csv",
        type=Path,
        default=STATIC / "w7x_linear_t2_scan.csv",
        help="Tracked linear W7-X imported-geometry comparison CSV.",
    )
    parser.add_argument(
        "--hsx-linear-csv",
        type=Path,
        default=STATIC / "hsx_linear_t2_scan.csv",
        help="Tracked linear HSX imported-geometry comparison CSV.",
    )
    parser.add_argument(
        "--w7x-linear-lastvalue-csv",
        type=Path,
        default=STATIC / "w7x_linear_t2_lastvalue.csv",
        help="Tracked late-time W7-X imported-geometry comparison CSV.",
    )
    parser.add_argument(
        "--hsx-linear-lastvalue-csv",
        type=Path,
        default=STATIC / "hsx_linear_t2_lastvalue.csv",
        help="Tracked late-time HSX imported-geometry comparison CSV.",
    )


def _add_publication_args(parser: argparse.ArgumentParser) -> None:
    _add_stellarator_args(parser)
    parser.add_argument(
        "--out",
        type=Path,
        default=STATIC / "comparison" / "reference_publication_panel.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--pdf-out",
        type=Path,
        default=STATIC / "comparison" / "reference_publication_panel.pdf",
        help="Optional PDF output path for publication workflows.",
    )


def _add_summary_args(parser: argparse.ArgumentParser) -> None:
    _add_stellarator_args(parser)
    parser.add_argument(
        "--secondary-csv",
        type=Path,
        default=STATIC / "comparison" / "secondary_reference_out_compare.csv",
        help="Tracked secondary comparison CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=STATIC / "comparison" / "reference_summary_panel.png",
        help="Output PNG path.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="panel", required=True)
    _add_tokamak_args(
        subparsers.add_parser(
            "tokamak",
            help="Build the Cyclone/KBM tokamak comparison panel.",
        )
    )
    _add_publication_args(
        subparsers.add_parser(
            "publication",
            help="Build the publication-facing benchmark comparison panel.",
        )
    )
    _add_summary_args(
        subparsers.add_parser(
            "summary",
            help="Build the full tracked validation summary panel.",
        )
    )
    return parser


def build_tokamak_panel(args: argparse.Namespace) -> None:
    cyclone_linear = _load_linear_mismatch(_resolve(args.cyclone_linear))
    kbm_linear = _load_linear_mismatch(_resolve(args.kbm_linear))
    cyclone_img = _autocrop_image(mpimg.imread(_resolve(args.cyclone_nonlinear_panel)), pad_pixels=4)
    kbm_img = _autocrop_image(mpimg.imread(_resolve(args.kbm_nonlinear_panel)), pad_pixels=4)
    out = _resolve(args.out)

    fig = plt.figure(figsize=(19.5, 12.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[0.9, 1.45])
    _plot_linear_metric(fig.add_subplot(gs[0, 0]), cyclone_linear, metric="gamma", title="Cyclone (Miller) Linear γ")
    _plot_linear_metric(fig.add_subplot(gs[0, 1]), cyclone_linear, metric="omega", title="Cyclone (Miller) Linear ω")
    _plot_linear_metric(fig.add_subplot(gs[0, 2]), kbm_linear, metric="gamma", title="KBM Linear γ")
    _plot_linear_metric(fig.add_subplot(gs[0, 3]), kbm_linear, metric="omega", title="KBM Linear ω")

    ax_c = fig.add_subplot(gs[1, :2])
    ax_c.imshow(cyclone_img)
    ax_c.set_title("Cyclone (Miller) Nonlinear", fontsize=14, fontweight="bold")
    ax_c.axis("off")

    ax_k = fig.add_subplot(gs[1, 2:])
    ax_k.imshow(kbm_img)
    ax_k.set_title("KBM Nonlinear", fontsize=14, fontweight="bold")
    ax_k.axis("off")

    fig.suptitle("Tokamak Reference Validation: Cyclone and KBM", fontsize=17, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


def _load_stellarator_linear_args(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        _load_imported_linear(_resolve(args.w7x_linear_csv)),
        _load_imported_linear(_resolve(args.hsx_linear_csv)),
        _load_imported_linear_lastvalue(_resolve(args.w7x_linear_lastvalue_csv)),
        _load_imported_linear_lastvalue(_resolve(args.hsx_linear_lastvalue_csv)),
    )


def build_publication_panel(args: argparse.Namespace) -> None:
    cyclone_kbm = _resolve(args.cyclone_kbm_panel)
    w7x_panel = _resolve(args.w7x_panel)
    hsx_panel = _resolve(args.hsx_panel)
    out = _resolve(args.out)
    pdf_out = _resolve(args.pdf_out)
    w7x_linear, hsx_linear, w7x_last, hsx_last = _load_stellarator_linear_args(args)

    cyclone_img = _autocrop_image(mpimg.imread(cyclone_kbm), pad_pixels=4)
    w7x_img = _autocrop_image(mpimg.imread(w7x_panel), pad_pixels=4)
    hsx_img = _autocrop_image(mpimg.imread(hsx_panel), pad_pixels=4)
    fig = plt.figure(figsize=(20.5, 18.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.55, 1.15, 1.0])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(cyclone_img)
    ax0.set_title("Tokamak Benchmarks: Cyclone and KBM", fontsize=15, fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(w7x_img)
    ax1.set_title("W7-X Nonlinear VMEC", fontsize=14, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(hsx_img)
    ax2.set_title("HSX Nonlinear VMEC", fontsize=14, fontweight="bold")
    ax2.axis("off")

    _plot_imported_linear(
        fig.add_subplot(gs[2, 0]),
        w7x_linear,
        "W7-X Linear VMEC",
        lastvalue=w7x_last,
        note="late-time closure tracks the whole-window scan",
    )
    _plot_imported_linear(
        fig.add_subplot(gs[2, 1]),
        hsx_linear,
        "HSX Linear VMEC",
        lastvalue=hsx_last,
        note="near marginality inflates the whole-window γ average; late-time closure remains tight",
    )

    fig.suptitle("Reference-Code Validation: Publication Panel", fontsize=18, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, facecolor="white")
    fig.savefig(pdf_out, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")
    print(f"saved {pdf_out}")


def build_summary_panel(args: argparse.Namespace) -> None:
    cyclone_kbm = _resolve(args.cyclone_kbm_panel)
    w7x_panel = _resolve(args.w7x_panel)
    hsx_panel = _resolve(args.hsx_panel)
    secondary = _load_secondary(_resolve(args.secondary_csv))
    out = _resolve(args.out)
    w7x_linear, hsx_linear, w7x_last, hsx_last = _load_stellarator_linear_args(args)

    fig = plt.figure(figsize=(20.5, 21.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.45, 1.15, 0.95, 0.9])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(_autocrop_image(mpimg.imread(cyclone_kbm), pad_pixels=4))
    ax0.set_title("Cyclone / KBM", fontsize=14, fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(_autocrop_image(mpimg.imread(w7x_panel), pad_pixels=4))
    ax1.set_title("W7-X Nonlinear VMEC", fontsize=14, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(_autocrop_image(mpimg.imread(hsx_panel), pad_pixels=4))
    ax2.set_title("HSX Nonlinear VMEC", fontsize=14, fontweight="bold")
    ax2.axis("off")

    _plot_imported_linear(
        fig.add_subplot(gs[2, 0]),
        w7x_linear,
        "W7-X Linear VMEC",
        lastvalue=w7x_last,
        note="window and late-time metrics remain aligned across the tracked ky range",
    )
    _plot_imported_linear(
        fig.add_subplot(gs[2, 1]),
        hsx_linear,
        "HSX Linear VMEC",
        lastvalue=hsx_last,
        note="near-marginal lane: whole-window rel γ is stress-only, late-time closure is tighter",
    )
    _plot_secondary(fig.add_subplot(gs[3, :]), secondary, "Secondary Slab")

    fig.suptitle("Reference-Aligned Validation Summary", fontsize=18, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.panel == "tokamak":
        build_tokamak_panel(args)
    elif args.panel == "publication":
        build_publication_panel(args)
    elif args.panel == "summary":
        build_summary_panel(args)
    else:  # pragma: no cover - argparse enforces choices.
        raise SystemExit(f"unknown panel: {args.panel}")


if __name__ == "__main__":
    main()
