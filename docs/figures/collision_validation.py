"""Render the README collision-validation figure from ``docs/figures/collision_validation.json``.

    python docs/figures/collision_validation.py [--data JSON] [--out DIR]

Left, the converged collisional trapped-electron mode growth rate of GKX ``landau``
against GS2 and stella; right, zonal-flow damping of the four operators against the
Xiao-Catto-Molvig formula.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gkx.artifacts.figure_style import GKX_COLORS, figure_style, save_figure

ROOT = Path(__file__).resolve().parents[2]
RC = {"font.size": 8.5, "axes.titlesize": 9.0, "legend.fontsize": 7.5, "savefig.dpi": 200}


def formula(t, q, eps, nu):
    """<phi>(t)/<phi>(0) with the Braginskii ion collision frequency 0.532 nu."""

    beta = eps**2 / q**2
    theta = 1.635 * eps**1.5 + eps**2 / 2 + 0.36 * eps**2.5
    gamma = 0.585 * eps**0.5 + 0.529 * eps
    rate = (1 + beta) * gamma / (theta + beta) * 0.532 * nu
    return beta / (1 + beta) * (1 + (1 - theta) / (theta + beta) * np.exp(-rate * t))


def figure(d: dict) -> plt.Figure:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    tem = d["tem"]
    n = [nl * nm for nl, nm in tem["resolution"]]
    lim, err = tem["limit"]
    ax.axhspan(lim - err, lim + err, color=GKX_COLORS["blue"], alpha=0.25)
    ax.plot(n, tem["gkx_landau"], "o-", color=GKX_COLORS["blue"], label="GKX landau")
    for key, label, color, marker in (
        ("gs2", "GS2", GKX_COLORS["green"], "^"),
        ("stella_fokker_planck_test_part", "stella Fokker-Planck test part", GKX_COLORS["vermillion"], "D"),
        ("stella_dougherty", "stella Dougherty", GKX_COLORS["purple"], "v"),
    ):
        ax.axhline(tem[key], color=color, ls="--", lw=1.0, label=label)
    ax.set(
        xscale="log",
        xlabel=r"velocity-space size $N_lN_m$",
        ylabel=r"$\gamma\,a/v_{ti}$",
        title=r"Collisional TEM, $\nu_e=0.05$, $k_y\rho_i=0.3$",
    )
    ax.legend(frameon=False, loc="lower right")

    z = d["zonal"]
    mid = [(a + b) / 2 for a, b in z["windows"]]
    for key, color, marker in (
        ("landau", GKX_COLORS["blue"], "o"),
        ("sugama", GKX_COLORS["orange"], "s"),
        ("lenard_bernstein", GKX_COLORS["purple"], "v"),
    ):
        bx.plot(mid, z["gkx_6_16"][key], marker=marker, color=color, label=f"GKX {key} (6,16)")
    bx.plot(z["gkx_24_64"]["t"], z["gkx_24_64"]["landau"], "o", mfc="white", mew=1.3, color=GKX_COLORS["blue"], label="GKX landau (24,64)")
    t = np.linspace(40, 400, 200)
    bx.plot(t, formula(t, z["q"], z["epsilon"], z["nu"]), "k--", lw=1.0, label="formula")
    bx.set(
        yscale="log",
        xlabel=r"$t\,v_{ti}/a$",
        ylabel=r"$\langle\phi\rangle/\langle\phi\rangle(0)$",
        title=rf"Zonal-flow damping, $\nu={z['nu']}$",
    )
    bx.legend(frameon=False)
    for a in (ax, bx):
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "docs/figures/collision_validation.json"))
    ap.add_argument("--out", default=str(ROOT / "docs/_static"))
    args = ap.parse_args()
    d = json.loads(Path(args.data).read_text())
    with figure_style(**RC):
        save_figure(figure(d), Path(args.out) / "collision_validation.png", palette_colors=96)


if __name__ == "__main__":
    main()
