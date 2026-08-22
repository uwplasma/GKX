"""Linear benchmark parity: every tracked case on one pair of axes.

Replaces the multi-case atlas for README use. The atlas shows each case in its
own pair of panels, which is right for auditing a single case and wrong for
answering "does this code agree with the references, and by how much" -- at
that size the per-panel axes differ, two cases hold a single point, and the
reader cannot compare anything across panels.

A parity plot answers that question directly: every ``(reference, GKX)`` pair on
one 1:1 line, with the per-case relative error beside it. Cases whose stored
tables carry a reference column are included; the rest are named in the console
output rather than silently dropped.

Data comes from ``tools/benchmark_atlas_manifest.toml`` -- the same tables the
atlas reads -- so this figure never disagrees with it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from gkx.utils import tomlcompat as tomllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gkx.artifacts.figure_style import (
    GKX_COLORS,
    REFERENCE_STYLE,
    annotate_reference,
    figure_style,
    panel_label,
    save_figure,
)

ROOT = Path(__file__).resolve().parents[2]

# (case label, manifest group/key, gkx column, reference column).
# Column names differ between tables because they were produced by different
# comparison scripts; normalizing them here keeps that mess out of the figure.
_CASES = (
    (
        "Cyclone ITG",
        ("core_linear", "cyclone"),
        "gamma_gkx",
        "gamma_ref",
        "omega_gkx",
        "omega_ref",
    ),
    ("ETG", ("core_linear", "etg"), "gamma_gkx", "gamma_ref", "omega_gkx", "omega_ref"),
    ("KBM", ("core_linear", "kbm"), "gamma_gkx", "gamma_ref", "omega_gkx", "omega_ref"),
    (
        "Cyclone Miller",
        ("core_linear", "miller"),
        "gamma",
        "gamma_gx",
        "omega",
        "omega_gx",
    ),
    ("KAW", ("core_linear", "kaw"), "gamma_gkx", "gamma_ref", "omega_gkx", "omega_ref"),
    (
        "W7-X",
        ("imported_linear", "w7x"),
        "gamma_last",
        "gamma_ref_last",
        "omega_last",
        "omega_ref_last",
    ),
    (
        "HSX",
        ("imported_linear", "hsx"),
        "gamma_last",
        "gamma_ref_last",
        "omega_last",
        "omega_ref_last",
    ),
)

_MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def _load_manifest() -> dict:
    with (ROOT / "tools" / "benchmark_atlas_manifest.toml").open("rb") as handle:
        return tomllib.load(handle)["group"]


def collect() -> tuple[list[dict], list[str]]:
    manifest = _load_manifest()
    collected: list[dict] = []
    skipped: list[str] = []

    for label, (group, key), g_col, g_ref, w_col, w_ref in _CASES:
        path = ROOT / manifest.get(group, {}).get(key, "")
        if not path.is_file():
            skipped.append(f"{label} (missing {path.name})")
            continue
        frame = pd.read_csv(path)
        if not {g_col, g_ref}.issubset(frame.columns):
            skipped.append(f"{label} (no reference column)")
            continue

        entry = {
            "label": label,
            "gamma": frame[g_col].to_numpy(float),
            "gamma_ref": frame[g_ref].to_numpy(float),
        }
        if {w_col, w_ref}.issubset(frame.columns):
            entry["omega"] = frame[w_col].to_numpy(float)
            entry["omega_ref"] = frame[w_ref].to_numpy(float)
        collected.append(entry)
    return collected, skipped


def _relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    scale = np.abs(reference).max()
    if scale == 0.0:
        return float("nan")
    return float(100.0 * np.abs(value - reference).max() / scale)


def build_figure(output: Path) -> dict[str, object]:
    cases, skipped = collect()

    with figure_style():
        fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6))

        for index, (ax, quantity, symbol) in enumerate(
            ((axes[0], "gamma", r"\gamma"), (axes[1], "omega", r"\omega"))
        ):
            # Each case is normalized by its own peak |reference|. The cases
            # span two orders of magnitude in gamma (ETG and KAW dwarf the
            # rest), so raw axes compress every other case into a blob at the
            # origin -- the plot then only shows that the biggest number is
            # right. Normalizing keeps the 1:1 line meaningful while making
            # every case visible.
            for case, marker, color in zip(
                cases,
                _MARKERS,
                [
                    GKX_COLORS[k]
                    for k in (
                        "blue",
                        "vermillion",
                        "green",
                        "orange",
                        "purple",
                        "sky",
                        "grey",
                    )
                ],
            ):
                if quantity not in case:
                    continue
                reference = case[f"{quantity}_ref"]
                value = case[quantity]
                scale = np.abs(reference).max()
                if scale == 0.0:
                    continue
                ax.plot(
                    reference / scale,
                    value / scale,
                    marker,
                    color=color,
                    markersize=6,
                    linestyle="none",
                    label=case["label"],
                    alpha=0.9,
                )

            span = np.array([-1.12, 1.12])
            ax.plot(span, span, label="exact agreement", **REFERENCE_STYLE)
            ax.set_xlim(*span)
            ax.set_ylim(*span)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(rf"reference ${symbol}$ / case peak")
            ax.set_ylabel(rf"GKX ${symbol}$ / case peak")
            ax.set_title(
                rf"Growth rate ${symbol}$" if index == 0 else rf"Frequency ${symbol}$"
            )
            if index == 0:
                ax.legend(loc="upper left", fontsize=8, ncol=2)
            panel_label(ax, f"({'ab'[index]})", dx=-0.16)

        # ---- (c) per-case error bars --------------------------------------
        ax = axes[2]
        labels = [case["label"] for case in cases]
        gamma_error = [_relative_error(c["gamma"], c["gamma_ref"]) for c in cases]
        omega_error = [
            _relative_error(c["omega"], c["omega_ref"]) if "omega" in c else np.nan
            for c in cases
        ]
        positions = np.arange(len(labels), dtype=float)
        ax.barh(
            positions + 0.19,
            gamma_error,
            0.36,
            label=r"$\gamma$",
            color=GKX_COLORS["blue"],
        )
        ax.barh(
            positions - 0.19,
            omega_error,
            0.36,
            label=r"$\omega$",
            color=GKX_COLORS["orange"],
        )
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("max relative difference [%]")
        ax.set_title("Agreement per case")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", fontsize=9)
        annotate_reference(
            ax,
            "normalized by the peak reference value,\n"
            "so near-zero crossings do not inflate the ratio",
            loc="upper right",
        )
        panel_label(ax, "(c)", dx=-0.30)

        fig.suptitle(
            "Linear benchmark parity against tracked reference results", fontsize=13
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(fig, output, palette_colors=256)

    return {
        "cases": labels,
        "gamma_error_percent": dict(zip(labels, gamma_error)),
        "omega_error_percent": dict(zip(labels, omega_error)),
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/_static/benchmark_linear_parity.png")
    )
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    summary = build_figure(args.output)
    print(f"{'case':>16} {'gamma %':>9} {'omega %':>9}")
    for label in summary["cases"]:
        print(
            f"{label:>16} {summary['gamma_error_percent'][label]:>9.2f}"
            f" {summary['omega_error_percent'][label]:>9.2f}"
        )
    if summary["skipped"]:
        # Never drop a case silently: a parity plot that quietly omits the
        # disagreeing cases is worse than no parity plot.
        print("skipped:", "; ".join(summary["skipped"]))
    print(f"written: {args.output}")

    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
