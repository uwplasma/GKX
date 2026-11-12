"""Render the README collision-operator figure from ``docs/figures/collision_operators.json``.

    python docs/figures/collision_operators.py [--data JSON] [--out DIR]

The JSON holds the collisional trapped-electron mode growth rates of
``benchmarks/cases/tem_collisional.toml`` for the GKX operators, GS2 and stella.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gkx.artifacts.figure_style import GKX_COLORS, figure_style, save_figure

ROOT = Path(__file__).resolve().parents[2]
LINES = (  # key, label, color, marker
    ("landau", "GKX landau", GKX_COLORS["blue"], "o"),
    ("sugama", "GKX sugama, improved_sugama", GKX_COLORS["orange"], "s"),
    ("lenard_bernstein", "GKX lenard_bernstein", GKX_COLORS["purple"], "v"),
)
RC = {"font.size": 8.5, "axes.titlesize": 9.0, "legend.fontsize": 7.5, "savefig.dpi": 200}


def figure(d: dict) -> plt.Figure:
    nu = np.array(d["nu"])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, ratio in zip(axes, (False, True)):
        scale = (lambda y: np.asarray(y, float) / y[0]) if ratio else np.asarray
        for key, label, color, marker in LINES:
            ax.plot(nu, scale(d["gkx_4_8"][key]), marker=marker, color=color, label=label + " (4,8)")
        for key, label, color, marker in (
            ("gs2", "GS2", GKX_COLORS["green"], "^"),
            ("stella", "stella", GKX_COLORS["vermillion"], "D"),
        ):
            ax.plot(nu, scale(d[key]), marker=marker, color=color, ls="--", label=label)
        if not ratio:
            ax.plot(
                nu,
                d["gkx_landau_high_resolution"],
                marker="o",
                mfc="white",
                color=GKX_COLORS["blue"],
                ls=":",
                label="GKX landau, highest resolution",
            )
        ax.set(xlabel=r"electron collisionality $\nu$")
        ax.grid(alpha=0.3)
    axes[0].set(ylabel=r"$\gamma\,a/v_{ti}$", title=r"Growth rate, $k_y\rho_i=0.3$")
    axes[1].set(ylabel=r"$\gamma(\nu)/\gamma(0)$", title="Reduction by collisions")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "docs/figures/collision_operators.json"))
    ap.add_argument("--out", default=str(ROOT / "docs/_static"))
    args = ap.parse_args()
    d = json.loads(Path(args.data).read_text())
    with figure_style(**RC):
        save_figure(figure(d), Path(args.out) / "collision_operators.png", palette_colors=96)


if __name__ == "__main__":
    main()
