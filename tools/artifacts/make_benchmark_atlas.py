#!/usr/bin/env python3
"""Build the compact benchmark atlas used by the README and docs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gkx.diagnostics.validation_gates import (
    evaluate_scalar_gate,
    gate_report,
    gate_report_to_dict,
)


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "docs" / "_static"

REF_COLOR = "#111827"
SP_COLOR = "#1d4ed8"
GRID_COLOR = "#cbd5e1"
TITLE_COLOR = "#0f172a"
TEXT_COLOR = "#334155"
FONT_FAMILY = "DejaVu Sans"
SUPTITLE_SIZE = 24
TILE_TITLE_SIZE = 16
NOTE_SIZE = 11
TICK_SIZE = 12
LEGEND_SIZE = 13
PANEL_WIDTH = 18.0
LINEAR_PANEL_HEIGHT = 8.8
ATLAS_HEIGHT = 14.2
README_HEIGHT = 27.0
CONVERGENCE_HEIGHT = 6.8

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "axes.titlesize": TILE_TITLE_SIZE,
        "axes.labelsize": 10,
        "legend.fontsize": LEGEND_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
    }
)


def _atlas_manifest_path() -> Path:
    return ROOT / "tools" / "benchmark_atlas_manifest.toml"


def _load_manifest(path: Path | None = None) -> dict:
    manifest_path = _resolve(path or _atlas_manifest_path())
    with manifest_path.open("rb") as fh:
        return tomllib.load(fh)


def _resolve_asset_paths(manifest: dict) -> dict[str, dict[str, Path]]:
    groups = manifest.get("group", {})
    resolved: dict[str, dict[str, Path]] = {}
    for group_name, assets in groups.items():
        resolved[group_name] = {}
        for asset_name, rel_path in assets.items():
            resolved[group_name][asset_name] = _resolve(rel_path)
    return resolved


def _validate_manifest_assets(resolved_assets: dict[str, dict[str, Path]]) -> None:
    missing: list[Path] = []
    for assets in resolved_assets.values():
        for path in assets.values():
            if not path.exists():
                missing.append(path)
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"benchmark atlas inputs are missing:\n{joined}")


def _write_summary(
    summary_path: Path,
    *,
    manifest_path: Path,
    resolved_assets: dict[str, dict[str, Path]],
    outputs: dict[str, Path],
    gate_reports: dict[str, dict[str, object]] | None = None,
) -> None:
    payload = {
        "manifest": str(manifest_path),
        "groups": {
            group: {name: str(path) for name, path in assets.items()}
            for group, assets in resolved_assets.items()
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    if gate_reports is not None:
        payload["gate_reports"] = gate_reports
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _autocrop_image(
    image: np.ndarray,
    *,
    white_threshold: float = 0.985,
    pad_pixels: int = 14,
) -> np.ndarray:
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


def _load_image(path: Path, *, pad_pixels: int = 14) -> np.ndarray:
    return _autocrop_image(mpimg.imread(path), pad_pixels=pad_pixels)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, facecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"saved {path.with_suffix('.pdf')}")


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.45, color=GRID_COLOR, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors=TEXT_COLOR, labelsize=TICK_SIZE)


def _plot_overlay_case(
    ax_gamma: plt.Axes,
    ax_omega: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    xcol: str,
    gamma_ref: str,
    gamma_sp: str,
    omega_ref: str,
    omega_sp: str,
    x_label: str,
) -> None:
    x = np.asarray(df[xcol], dtype=float)
    gamma_ref_vals = np.asarray(df[gamma_ref], dtype=float)
    gamma_sp_vals = np.asarray(df[gamma_sp], dtype=float)
    omega_ref_vals = np.asarray(df[omega_ref], dtype=float)
    omega_sp_vals = np.asarray(df[omega_sp], dtype=float)

    _style_axis(ax_gamma)
    _style_axis(ax_omega)

    ax_gamma.plot(
        x, gamma_ref_vals, color=REF_COLOR, marker="o", linewidth=2.0, label="Reference"
    )
    ax_gamma.plot(
        x, gamma_sp_vals, color=SP_COLOR, marker="s", linewidth=2.0, label="GKX"
    )
    ax_gamma.set_title(
        title, fontsize=TILE_TITLE_SIZE, color=TITLE_COLOR, fontweight="bold"
    )
    ax_gamma.set_ylabel(r"Growth rate $\gamma$")

    ax_omega.plot(
        x, omega_ref_vals, color=REF_COLOR, marker="o", linewidth=2.0, label="Reference"
    )
    ax_omega.plot(
        x, omega_sp_vals, color=SP_COLOR, marker="s", linewidth=2.0, label="GKX"
    )
    ax_omega.set_ylabel(r"Frequency $\omega$")
    ax_omega.set_xlabel(x_label)

    if x.size == 1:
        for ax in (ax_gamma, ax_omega):
            center = float(x[0])
            width = max(abs(center) * 0.4, 0.02)
            ax.set_xlim(center - width, center + width)

    denom_gamma = np.maximum(np.abs(gamma_ref_vals), 1.0e-12)
    denom_omega = np.maximum(np.abs(omega_ref_vals), 1.0e-12)
    max_rel_gamma = float(np.max(np.abs(gamma_sp_vals - gamma_ref_vals) / denom_gamma))
    max_rel_omega = float(np.max(np.abs(omega_sp_vals - omega_ref_vals) / denom_omega))
    ax_gamma.text(
        0.03,
        0.97,
        f"max rel γ = {max_rel_gamma:.2%}\nmax rel ω = {max_rel_omega:.2%}",
        transform=ax_gamma.transAxes,
        va="top",
        ha="left",
        fontsize=NOTE_SIZE,
        color=TEXT_COLOR,
        bbox={
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "boxstyle": "round,pad=0.25",
        },
    )


def _plot_kaw_case(ax_gamma: plt.Axes, ax_omega: plt.Axes, df: pd.DataFrame) -> None:
    _plot_overlay_case(
        ax_gamma,
        ax_omega,
        df,
        title="KAW Exact Diagnostic Window",
        xcol="ky",
        gamma_ref="gamma_ref",
        gamma_sp="gamma_gkx",
        omega_ref="omega_ref",
        omega_sp="omega_gkx",
        x_label=r"$k_y \rho_i$",
    )
    row = df.iloc[0]
    ax_omega.text(
        0.03,
        0.12,
        (
            "window errors\n"
            f"free energy = {row['rel_free_energy']:.2e}\n"
            f"electrostatic field energy = {row['rel_electrostatic_energy']:.2e}\n"
            f"magnetic field energy = {row['rel_magnetic_energy']:.2e}"
        ),
        transform=ax_omega.transAxes,
        va="bottom",
        ha="left",
        fontsize=NOTE_SIZE,
        color=TEXT_COLOR,
        bbox={
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "boxstyle": "round,pad=0.25",
        },
    )


def _plot_exact_growth_case(
    ax_gamma: plt.Axes,
    ax_omega: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    gamma_ref: str,
    gamma_sp: str,
    omega_ref: str,
    omega_sp: str,
    rel_gamma: str | None = None,
    rel_omega: str | None = None,
    note_title: str = "window errors",
) -> None:
    _plot_overlay_case(
        ax_gamma,
        ax_omega,
        df,
        title=title,
        xcol="ky",
        gamma_ref=gamma_ref,
        gamma_sp=gamma_sp,
        omega_ref=omega_ref,
        omega_sp=omega_sp,
        x_label=r"$k_y \rho_i$",
    )
    row = df.iloc[0]
    lines = [note_title]
    if rel_gamma is not None and rel_gamma in row:
        lines.append(f"rel γ = {row[rel_gamma]:.2e}")
    if rel_omega is not None and rel_omega in row:
        lines.append(f"rel ω = {row[rel_omega]:.2e}")
    ax_omega.text(
        0.03,
        0.12,
        "\n".join(lines),
        transform=ax_omega.transAxes,
        va="bottom",
        ha="left",
        fontsize=NOTE_SIZE,
        color=TEXT_COLOR,
        bbox={
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "boxstyle": "round,pad=0.25",
        },
    )


def _image_tile(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image, aspect="auto")
    ax.set_title(title, fontsize=TILE_TITLE_SIZE, color=TITLE_COLOR, fontweight="bold")
    ax.axis("off")


def _image_tile_plain(ax: plt.Axes, image: np.ndarray) -> None:
    ax.imshow(image, aspect="auto")
    ax.axis("off")


def _build_convergence_panel(path: Path, assets: dict[str, Path]) -> None:
    scan = pd.read_csv(assets["cyclone_scan"]).sort_values("ky")
    rhostar = pd.read_csv(assets["cyclone_rhostar"]).sort_values("rho_star")

    fig, axes = plt.subplots(
        1, 2, figsize=(PANEL_WIDTH, CONVERGENCE_HEIGHT), constrained_layout=True
    )
    ax_scan, ax_rho = axes

    _style_axis(ax_scan)
    ax_scan.plot(
        np.asarray(scan["ky"], dtype=float),
        np.asarray(scan["rel_gamma_change"], dtype=float),
        color=SP_COLOR,
        marker="o",
        linewidth=2.2,
        label=r"$\Delta \gamma$",
    )
    ax_scan.plot(
        np.asarray(scan["ky"], dtype=float),
        np.asarray(scan["rel_omega_change"], dtype=float),
        color="#c2410c",
        marker="s",
        linewidth=2.2,
        label=r"$\Delta \omega$",
    )
    ax_scan.set_title(
        "Cyclone Resolution Convergence",
        fontsize=TILE_TITLE_SIZE,
        color=TITLE_COLOR,
        fontweight="bold",
    )
    ax_scan.set_xlabel(r"$k_y \rho_i$")
    ax_scan.set_ylabel("relative change")
    ax_scan.legend(frameon=False, fontsize=LEGEND_SIZE, loc="upper left")
    ax_scan.text(
        0.03,
        0.08,
        "tracked high-vs-low production grid check",
        transform=ax_scan.transAxes,
        fontsize=NOTE_SIZE,
        color=TEXT_COLOR,
        ha="left",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "boxstyle": "round,pad=0.25",
        },
    )

    _style_axis(ax_rho)
    rho = np.asarray(rhostar["rho_star"], dtype=float)
    ax_rho.plot(
        rho,
        np.asarray(rhostar["mean_gamma_ratio"], dtype=float),
        color=SP_COLOR,
        marker="o",
        linewidth=2.2,
        label=r"$\gamma/\gamma_{\rho_\star=1}$",
    )
    ax_rho.plot(
        rho,
        np.asarray(rhostar["mean_omega_ratio"], dtype=float),
        color="#c2410c",
        marker="s",
        linewidth=2.2,
        label=r"$\omega/\omega_{\rho_\star=1}$",
    )
    ax_rho.axhline(1.0, color=REF_COLOR, linewidth=1.4, linestyle=":")
    ax_rho.set_title(
        r"Cyclone $\rho_\star$ Sensitivity",
        fontsize=TILE_TITLE_SIZE,
        color=TITLE_COLOR,
        fontweight="bold",
    )
    ax_rho.set_xlabel(r"$\rho_\star / \rho_{\star,\mathrm{bench}}$")
    ax_rho.set_ylabel("normalized response")
    ax_rho.legend(frameon=False, fontsize=LEGEND_SIZE, loc="upper left")
    ax_rho.text(
        0.03,
        0.08,
        "benchmark panels use the locked reference normalization",
        transform=ax_rho.transAxes,
        fontsize=NOTE_SIZE,
        color=TEXT_COLOR,
        ha="left",
        va="bottom",
        bbox={
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "boxstyle": "round,pad=0.25",
        },
    )

    fig.suptitle(
        "Representative Convergence and Sensitivity Checks",
        fontsize=SUPTITLE_SIZE,
        fontweight="bold",
    )
    _save(fig, path)


def _build_convergence_gate_reports(
    assets: dict[str, Path],
    *,
    max_rel_change: float = 0.05,
) -> dict[str, dict[str, object]]:
    """Build machine-readable gates for tracked convergence-panel inputs."""

    scan = pd.read_csv(assets["cyclone_scan"]).sort_values("ky")
    threshold = float(max_rel_change)
    if threshold < 0.0:
        raise ValueError("max_rel_change must be non-negative")
    report = gate_report(
        "cyclone_resolution_convergence",
        "tracked high-vs-low production grid",
        (
            evaluate_scalar_gate(
                "max_rel_gamma_change",
                float(np.nanmax(np.asarray(scan["rel_gamma_change"], dtype=float))),
                0.0,
                atol=threshold,
                rtol=0.0,
                notes=f"Passes when high-vs-low grid gamma change <= {threshold:.6g}.",
            ),
            evaluate_scalar_gate(
                "max_rel_omega_change",
                float(np.nanmax(np.asarray(scan["rel_omega_change"], dtype=float))),
                0.0,
                atol=threshold,
                rtol=0.0,
                notes=f"Passes when high-vs-low grid omega change <= {threshold:.6g}.",
            ),
        ),
    )
    return {"cyclone_resolution_convergence": gate_report_to_dict(report)}


def _build_imported_linear_panel(path: Path, assets: dict[str, Path]) -> None:
    w7x = pd.read_csv(assets["w7x"]).sort_values("ky")
    hsx = pd.read_csv(assets["hsx"]).sort_values("ky")
    miller = pd.read_csv(assets["miller"]).sort_values("ky")
    kaw = pd.read_csv(assets["kaw"]).sort_values("ky")

    fig, axes = plt.subplots(
        2, 4, figsize=(PANEL_WIDTH, LINEAR_PANEL_HEIGHT), constrained_layout=True
    )

    _plot_overlay_case(
        axes[0, 0],
        axes[1, 0],
        w7x,
        title="W7-X VMEC",
        xcol="ky",
        gamma_ref="gamma_ref_last",
        gamma_sp="gamma_last",
        omega_ref="omega_ref_last",
        omega_sp="omega_last",
        x_label=r"$k_y \rho_i$",
    )
    _plot_overlay_case(
        axes[0, 1],
        axes[1, 1],
        hsx,
        title="HSX VMEC",
        xcol="ky",
        gamma_ref="gamma_ref_last",
        gamma_sp="gamma_last",
        omega_ref="omega_ref_last",
        omega_sp="omega_last",
        x_label=r"$k_y \rho_i$",
    )
    _plot_overlay_case(
        axes[0, 2],
        axes[1, 2],
        miller,
        title="Cyclone Miller",
        xcol="ky",
        gamma_ref="gamma_gx",
        gamma_sp="gamma",
        omega_ref="omega_gx",
        omega_sp="omega",
        x_label=r"$k_y \rho_i$",
    )
    _plot_kaw_case(axes[0, 3], axes[1, 3], kaw)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
        fontsize=LEGEND_SIZE,
    )
    fig.suptitle(
        "Imported Geometry and Exact-Diagnostic Linear Benchmarks",
        fontsize=SUPTITLE_SIZE,
        fontweight="bold",
    )
    _save(fig, path)


def _build_extended_linear_panel(path: Path, assets: dict[str, Path]) -> None:
    kinetic = pd.read_csv(assets["kinetic"]).sort_values("ky")
    tem = pd.read_csv(assets["tem"]).sort_values("ky")
    miller = pd.read_csv(assets["miller"]).sort_values("ky")

    fig, axes = plt.subplots(
        2, 3, figsize=(PANEL_WIDTH, LINEAR_PANEL_HEIGHT), constrained_layout=True
    )
    _plot_overlay_case(
        axes[0, 0],
        axes[1, 0],
        kinetic,
        title="Cyclone Kinetic Electrons",
        xcol="ky",
        gamma_ref="gamma_ref",
        gamma_sp="gamma_gkx",
        omega_ref="omega_ref",
        omega_sp="omega_gkx",
        x_label=r"$k_y \rho_i$",
    )
    _plot_overlay_case(
        axes[0, 1],
        axes[1, 1],
        tem,
        title="TEM Stress Lane",
        xcol="ky",
        gamma_ref="gamma_ref",
        gamma_sp="gamma_gkx",
        omega_ref="omega_ref",
        omega_sp="omega_gkx",
        x_label=r"$k_y \rho_i$",
    )
    _plot_exact_growth_case(
        axes[0, 2],
        axes[1, 2],
        miller,
        title="KBM Miller Late Growth Window",
        gamma_ref="gamma_gx_dump",
        gamma_sp="gamma_sp_dump",
        omega_ref="omega_gx_dump",
        omega_sp="omega_sp_dump",
        rel_gamma="rel_gamma_sp_vs_gx_dump",
        rel_omega="rel_omega_sp_vs_gx_dump",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
        fontsize=LEGEND_SIZE,
    )
    fig.suptitle(
        "Extended Linear Stress Matrix", fontsize=SUPTITLE_SIZE, fontweight="bold"
    )
    _save(fig, path)


def _build_core_linear_atlas(path: Path, assets: dict[str, Path]) -> None:
    cyclone = pd.read_csv(assets["cyclone"]).sort_values("ky")
    etg = pd.read_csv(assets["etg"]).sort_values("ky")
    kbm = pd.read_csv(assets["kbm"]).sort_values("ky")
    w7x = pd.read_csv(assets["w7x"]).sort_values("ky")
    hsx = pd.read_csv(assets["hsx"]).sort_values("ky")
    miller = pd.read_csv(assets["miller"]).sort_values("ky")
    kaw = pd.read_csv(assets["kaw"]).sort_values("ky")
    kbm_miller = pd.read_csv(assets["kbm_miller"]).sort_values("ky")

    fig = plt.figure(figsize=(21.5, 16.8), constrained_layout=True)
    outer = fig.add_gridspec(2, 4)

    def add_case(idx: int) -> tuple[plt.Axes, plt.Axes]:
        sub = outer[idx // 4, idx % 4].subgridspec(2, 1, hspace=0.05)
        return fig.add_subplot(sub[0]), fig.add_subplot(sub[1])

    cases = [
        (
            "Cyclone ITG",
            cyclone,
            "ky",
            "gamma_ref",
            "gamma_gkx",
            "omega_ref",
            "omega_gkx",
            r"$k_y \rho_i$",
        ),
        (
            "ETG",
            etg,
            "ky",
            "gamma_ref",
            "gamma_gkx",
            "omega_ref",
            "omega_gkx",
            r"$k_y \rho_i$",
        ),
        (
            "KBM",
            kbm,
            "ky",
            "gamma_ref",
            "gamma_gkx",
            "omega_ref",
            "omega_gkx",
            r"$\beta$",
        ),
        (
            "W7-X VMEC",
            w7x,
            "ky",
            "gamma_ref_last",
            "gamma_last",
            "omega_ref_last",
            "omega_last",
            r"$k_y \rho_i$",
        ),
        (
            "HSX VMEC",
            hsx,
            "ky",
            "gamma_ref_last",
            "gamma_last",
            "omega_ref_last",
            "omega_last",
            r"$k_y \rho_i$",
        ),
        (
            "Cyclone Miller",
            miller,
            "ky",
            "gamma_gx",
            "gamma",
            "omega_gx",
            "omega",
            r"$k_y \rho_i$",
        ),
    ]

    for idx, (title, df, xcol, gref, gsp, oref, osp, xlabel) in enumerate(cases):
        axg, axo = add_case(idx)
        _plot_overlay_case(
            axg,
            axo,
            df,
            title=title,
            xcol=xcol,
            gamma_ref=gref,
            gamma_sp=gsp,
            omega_ref=oref,
            omega_sp=osp,
            x_label=xlabel,
        )

    axg, axo = add_case(6)
    _plot_kaw_case(axg, axo, kaw)

    axg, axo = add_case(7)
    _plot_exact_growth_case(
        axg,
        axo,
        kbm_miller,
        title="KBM Miller Late Growth",
        gamma_ref="gamma_gx_dump",
        gamma_sp="gamma_sp_dump",
        omega_ref="omega_gx_dump",
        omega_sp="omega_sp_dump",
        rel_gamma="rel_gamma_sp_vs_gx_dump",
        rel_omega="rel_omega_sp_vs_gx_dump",
    )

    fig.suptitle(
        "Linear Benchmark Master Panel",
        fontsize=SUPTITLE_SIZE,
        fontweight="bold",
        y=1.03,
    )
    _save(fig, path)


def _build_core_nonlinear_atlas(path: Path, assets: dict[str, Path]) -> None:
    cyclone = _load_image(assets["cyclone"], pad_pixels=4)
    kbm = _load_image(assets["kbm"], pad_pixels=4)
    w7x = _load_image(assets["w7x"], pad_pixels=4)
    hsx = _load_image(assets["hsx"], pad_pixels=4)
    miller = _load_image(assets["miller"], pad_pixels=4)

    fig = plt.figure(figsize=(22.0, 18.8), constrained_layout=True)
    outer = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.08])

    tiles = [
        (cyclone, "Cyclone ITG Nonlinear"),
        (kbm, "KBM Nonlinear"),
        (w7x, "W7-X Nonlinear"),
        (hsx, "HSX Nonlinear"),
    ]
    for idx, (image, title) in enumerate(tiles):
        ax = fig.add_subplot(outer[idx // 2, idx % 2])
        _image_tile_plain(ax, image)
        ax.text(
            0.01,
            0.99,
            title,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=14,
            color=TITLE_COLOR,
            bbox={
                "facecolor": "white",
                "edgecolor": "#cbd5e1",
                "boxstyle": "round,pad=0.2",
            },
        )

    ax_miller = fig.add_subplot(outer[2, :])
    _image_tile_plain(ax_miller, miller)
    ax_miller.text(
        0.01,
        0.99,
        "Cyclone Miller Nonlinear",
        transform=ax_miller.transAxes,
        va="top",
        ha="left",
        fontsize=14,
        color=TITLE_COLOR,
        bbox={
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "boxstyle": "round,pad=0.2",
        },
    )
    fig.suptitle(
        "Nonlinear Benchmark Master Panel",
        fontsize=SUPTITLE_SIZE,
        fontweight="bold",
        y=1.02,
    )
    _save(fig, path)


def _build_readme_panel(
    path: Path,
    *,
    convergence_path: Path,
    core_linear_path: Path,
    core_nonlinear_path: Path,
) -> None:
    convergence = _load_image(convergence_path, pad_pixels=2)
    core_linear = _load_image(core_linear_path, pad_pixels=2)
    core_nonlinear = _load_image(core_nonlinear_path, pad_pixels=2)

    fig = plt.figure(figsize=(20.0, 25.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.78, 1.52, 1.58], hspace=0.02)

    for idx, image in enumerate((convergence, core_linear, core_nonlinear)):
        ax = fig.add_subplot(gs[idx, 0])
        _image_tile_plain(ax, image)

    fig.suptitle(
        "GKX Benchmark and Convergence Atlas",
        fontsize=SUPTITLE_SIZE + 1,
        fontweight="bold",
        y=0.998,
    )
    fig.subplots_adjust(top=0.985, bottom=0.015, left=0.02, right=0.98)
    _save(fig, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_atlas_manifest_path(),
        help="Atlas input manifest.",
    )
    parser.add_argument(
        "--imported-linear-out",
        type=Path,
        default=STATIC / "benchmark_imported_linear_panel.png",
        help="Output path for imported-geometry and exact-diagnostic linear scans.",
    )
    parser.add_argument(
        "--extended-linear-out",
        type=Path,
        default=STATIC / "benchmark_extended_linear_panel.png",
        help="Output path for extended linear stress scans.",
    )
    parser.add_argument(
        "--core-linear-out",
        type=Path,
        default=STATIC / "benchmark_core_linear_atlas.png",
        help="Output path for the core linear atlas.",
    )
    parser.add_argument(
        "--core-nonlinear-out",
        type=Path,
        default=STATIC / "benchmark_core_nonlinear_atlas.png",
        help="Output path for the core nonlinear atlas.",
    )
    parser.add_argument(
        "--convergence-out",
        type=Path,
        default=STATIC / "benchmark_convergence_panel.png",
        help="Output path for the representative convergence panel.",
    )
    parser.add_argument(
        "--readme-out",
        type=Path,
        default=STATIC / "benchmark_readme_panel.png",
        help="Output path for the README benchmark panel.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ROOT / "tools_out" / "benchmark_atlas_summary.json",
        help="Optional JSON summary of the atlas inputs and outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = _resolve(args.manifest)
    manifest = _load_manifest(manifest_path)
    assets = _resolve_asset_paths(manifest)
    _validate_manifest_assets(assets)
    imported_linear_out = _resolve(args.imported_linear_out)
    extended_linear_out = _resolve(args.extended_linear_out)
    core_linear_out = _resolve(args.core_linear_out)
    core_nonlinear_out = _resolve(args.core_nonlinear_out)
    convergence_out = _resolve(args.convergence_out)
    readme_out = _resolve(args.readme_out)
    summary_out = _resolve(args.summary_out)

    _build_imported_linear_panel(imported_linear_out, assets["imported_linear"])
    _build_extended_linear_panel(extended_linear_out, assets["extended_linear"])
    _build_core_linear_atlas(core_linear_out, assets["core_linear"])
    _build_core_nonlinear_atlas(core_nonlinear_out, assets["core_nonlinear"])
    _build_convergence_panel(convergence_out, assets["convergence"])
    _build_readme_panel(
        readme_out,
        convergence_path=convergence_out,
        core_linear_path=core_linear_out,
        core_nonlinear_path=core_nonlinear_out,
    )
    _write_summary(
        summary_out,
        manifest_path=manifest_path,
        resolved_assets=assets,
        outputs={
            "imported_linear": imported_linear_out,
            "extended_linear": extended_linear_out,
            "core_linear": core_linear_out,
            "core_nonlinear": core_nonlinear_out,
            "convergence": convergence_out,
            "readme": readme_out,
        },
        gate_reports=_build_convergence_gate_reports(assets["convergence"]),
    )


if __name__ == "__main__":
    main()
