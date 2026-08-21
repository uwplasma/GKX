"""Runtime and peak-memory panel, rendered from the recorded benchmark CSV.

Reads the measurements produced by ``benchmarks/performance`` rather than
re-running anything, so the figure can be restyled without a fresh campaign on
the benchmark host.

Both panels are honest about what is being compared: these are cold wall times,
which include JAX startup and first-call compilation. That is the right number
for "how long does one run take" and the wrong one for "how fast is the kernel";
the profiler artifacts in docs/performance.rst carry the warm timings.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gkx.artifacts.figure_style import (
    GKX_COLORS,
    annotate_reference,
    figure_style,
    panel_label,
    save_figure,
)

# Comparison-code rows are benchmark provenance, which is the only context in
# which another code is named anywhere in this repository.
_BACKENDS = (
    ("gkx_cpu", "GKX (CPU)", GKX_COLORS["blue"]),
    ("gkx_gpu", "GKX (GPU)", GKX_COLORS["green"]),
    ("gx", "GX (reference)", GKX_COLORS["orange"]),
)


def load(
    path: Path,
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Return ordered case labels plus runtime and memory keyed by backend."""

    runtime: dict[str, dict[str, float]] = {key: {} for key, _, _ in _BACKENDS}
    memory: dict[str, dict[str, float]] = {key: {} for key, _, _ in _BACKENDS}
    order: list[str] = []

    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "success":
                continue
            label = row["label"]
            backend = row["backend"]
            if backend not in runtime:
                continue
            if label not in order:
                order.append(label)
            if row.get("runtime_s"):
                runtime[backend][label] = float(row["runtime_s"])
            if row.get("peak_rss_mb"):
                memory[backend][label] = float(row["peak_rss_mb"])
    return order, runtime, memory


def build_figure(results: Path, output: Path) -> dict[str, object]:
    order, runtime, memory = load(results)
    positions = np.arange(len(order), dtype=float)
    width = 0.26

    with figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.2))

        for ax, data, title, unit, logscale in (
            (axes[0], runtime, "Cold wall time", "s", True),
            (axes[1], memory, "Peak resident memory", "MiB", False),
        ):
            for index, (key, label, color) in enumerate(_BACKENDS):
                values = [data[key].get(case, np.nan) for case in order]
                ax.bar(
                    positions + (index - 1) * width,
                    values,
                    width,
                    label=label,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                )
            if logscale:
                ax.set_yscale("log")
                finite = [v for series in data.values() for v in series.values()]
                # Headroom so the tallest bar is not clipped by the axis top.
                ax.set_ylim(0.6 * min(finite), 4.0 * max(finite))
            ax.set_xticks(positions)
            ax.set_xticklabels(order, rotation=32, ha="right")
            ax.set_ylabel(f"{title} [{unit}]")
            ax.set_title(title)
            ax.grid(axis="x", visible=False)
            ax.legend(loc="upper left", ncol=3, fontsize=9)
            ax.margins(x=0.02)

        # Report the CPU->GPU speedup range actually present in the data rather
        # than a headline number: it varies by more than an order of magnitude
        # across cases, and quoting one figure would misrepresent the rest.
        speedups = [
            runtime["gkx_cpu"][case] / runtime["gkx_gpu"][case]
            for case in order
            if case in runtime["gkx_cpu"] and case in runtime["gkx_gpu"]
        ]
        annotate_reference(
            axes[0],
            "cold times include JAX startup and compilation\n"
            f"GKX CPU$\\rightarrow$GPU speedup spans "
            f"{min(speedups):.1f}$\\times$ to {max(speedups):.1f}$\\times$",
            loc="upper right",
        )
        panel_label(axes[0], "(a)", dx=-0.07)
        panel_label(axes[1], "(b)", dx=-0.07)

        fig.suptitle(
            "Runtime and memory across the tracked benchmark cases", fontsize=13
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save_figure(fig, output)

    return {
        "cases": order,
        "speedup_min": float(min(speedups)),
        "speedup_max": float(max(speedups)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("docs/_static/runtime_memory_results_ship_refresh.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/_static/runtime_memory_benchmark.png")
    )
    args = parser.parse_args()

    summary = build_figure(args.results, args.output)
    print(f"cases: {len(summary['cases'])}")
    print(
        f"GKX CPU->GPU speedup: {summary['speedup_min']:.2f}x to "
        f"{summary['speedup_max']:.2f}x"
    )
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
