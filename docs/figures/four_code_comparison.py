"""Render the README comparison figures from ``docs/figures/four_code_comparison.json``.

    python docs/figures/four_code_comparison.py [--data JSON] [--out DIR]

The JSON holds scalars only (growth rates, frequencies, degrees of freedom, wall times,
heat-flux statistics).  GS2 and stella entries are already converted to GX/GKX units
(gamma, omega x sqrt(2); k_y / sqrt(2)).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from gkx.artifacts.figure_style import GKX_COLORS, figure_style, save_figure

ROOT = Path(__file__).resolve().parents[2]
COLOR = {
    "GKX": GKX_COLORS["blue"],
    "GX": GKX_COLORS["vermillion"],
    "GS2": GKX_COLORS["green"],
    "stella": "#9B4F96",
}
MARK = {"GKX": "o", "GX": "s", "GS2": "^", "stella": "D"}
SIZE = {"GKX": 6.5, "GX": 5.0, "GS2": 5.0, "stella": 4.5}
CODES = ("GKX", "GX", "GS2", "stella")
RC = {
    "font.size": 7.5,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.1,
    "axes.titlepad": 5.0,
    "savefig.dpi": 220,
}


def _plot(ax, code, x, y, **kw):
    hollow = code == "GX"
    ax.plot(
        x,
        y,
        marker=MARK[code],
        ms=SIZE[code],
        color=COLOR[code],
        mfc="white" if hollow else COLOR[code],
        mew=1.1 if hollow else 0.0,
        ls="-" if code == "GKX" else "--",
        lw=1.6 if code == "GKX" else 0.9,
        alpha=1.0,
        zorder={"GKX": 3, "GX": 4, "GS2": 2, "stella": 2}[code],
        label=code,
        **kw,
    )


def linear_figure(d: dict) -> plt.Figure:
    cases = [
        ("cyclone_ad", "Cyclone, adiabatic electrons", 6),
        ("cyclone_ke", "Cyclone, kinetic electrons", 8),
        ("em", r"Cyclone, kinetic e, A$_\parallel$", 16),
        ("w7x_ad", "W7-X, adiabatic electrons", 40),
    ]
    fig, axs = plt.subplots(
        3,
        4,
        figsize=(8.0, 4.7),
        sharex="col",
        gridspec_kw={
            "hspace": 0.14,
            "wspace": 0.45,
            "height_ratios": [1.0, 0.75, 0.55],
        },
    )
    for j, (key, title, dlim) in enumerate(cases):
        blk = d["em"] if key == "em" else d["linear"][key]
        for i, q in enumerate(("gamma", "omega")):
            ax = axs[i, j]
            for c in CODES:
                if c in blk["codes"]:
                    cb = blk["codes"][c]
                    xs, ys = (blk["beta"], cb[q]) if key == "em" else (cb["ky"], cb[q])
                    keep = [k for k, y in enumerate(ys) if y is not None]
                    _plot(ax, c, [xs[k] for k in keep], [ys[k] for k in keep])
            ax.set_ylabel(
                r"$\gamma\, a/v_{ti}$" if q == "gamma" else r"$\omega\, a/v_{ti}$",
                labelpad=1.5,
            )
            ax.grid(alpha=0.3, lw=0.4)
            if q == "gamma":
                ax.set_ylim(bottom=0)
        ax = axs[2, j]
        ref = blk["codes"]["GKX"]
        xs_ref = blk["beta"] if key == "em" else ref["ky"]
        for c in CODES[1:]:
            if c not in blk["codes"]:
                continue
            cb = blk["codes"][c]
            xs, ys = (
                (blk["beta"], cb["gamma"]) if key == "em" else (cb["ky"], cb["gamma"])
            )
            rmap = {x: y for x, y in zip(xs_ref, ref["gamma"]) if y is not None}
            pts = [
                (x, 100 * (y - rmap[x]) / rmap[x])
                for x, y in zip(xs, ys)
                if y is not None and x in rmap
            ]
            if pts:
                _plot(ax, c, [p[0] for p in pts], [p[1] for p in pts])
        ax.axhline(0, color=COLOR["GKX"], lw=1.0, zorder=0)
        ax.set_ylim(-dlim, dlim)
        ax.set_ylabel(r"$\Delta\gamma/\gamma_{\rm GKX}$ (%)", labelpad=1.0)
        ax.set_xlabel(
            r"$\beta_{\rm ref}$ ($k_y\rho_i = 0.3$)" if key == "em" else r"$k_y\rho_i$"
        )
        ax.grid(alpha=0.3, lw=0.4)
        axs[0, j].set_title(title, loc="left")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        handlelength=2.2,
        columnspacing=2.0,
    )
    return fig


def speed_figure(d: dict) -> plt.Figure:
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(8.0, 2.75),
        gridspec_kw={"wspace": 0.42, "width_ratios": [1.0, 1.0, 0.95]},
    )
    # (a) accuracy against time-to-solution, Cyclone adiabatic, k_y rho_i = 0.3
    ax = axs[0]
    conv = d["convergence"]["cyclone_ad_ky0.3"]
    floor = 0.02
    for c in CODES:
        pts = conv[c]
        err = [max(abs(e["err_pct"]), floor) for e in pts]
        t = [e["t"] if c in ("GKX", "GX") else e["t4"] for e in pts]
        ax.plot(t, err, ls="-", lw=0.7, color=COLOR[c], alpha=0.55, zorder=1)
        ax.plot(
            t,
            err,
            ls="none",
            marker=MARK[c],
            ms=SIZE[c],
            color=COLOR[c],
            mfc="white" if c == "GX" else COLOR[c],
            mew=1.1 if c == "GX" else 0.0,
            zorder=3,
            label=c,
        )
        if c == "GKX":
            ax.plot(
                [e["t_cold"] for e in pts],
                err,
                ls="none",
                marker="o",
                ms=3.2,
                mfc="white",
                mec=COLOR[c],
                mew=0.8,
                zorder=2,
            )
        elif c != "GX":
            ax.plot(
                [e["t1"] for e in pts],
                err,
                ls="none",
                marker=MARK[c],
                ms=3.5,
                mfc="white",
                mec=COLOR[c],
                mew=0.8,
                zorder=1,
            )
    ax.axhline(1.0, color="#555555", lw=0.7, ls=(0, (2, 2)))
    ax.text(
        0.97,
        0.97,
        "open markers:\nGKX with compilation,\nGS2 and stella on 1 core",
        transform=ax.transAxes,
        fontsize=5.6,
        va="top",
        ha="right",
        color="#444444",
    )
    ax.text(
        0.97,
        1.12,
        "1 %",
        fontsize=6.5,
        color="#555555",
        ha="right",
        va="bottom",
        transform=ax.get_yaxis_transform(),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time to solution (s)")
    ax.set_ylabel(r"$|\gamma-\gamma_{\rm conv}|/\gamma_{\rm conv}$ (%)")
    ax.set_title(r"Cyclone, $k_y\rho_i=0.3$", loc="left")
    ax.set_ylim(floor * 0.8, 8)
    ax.grid(alpha=0.3, lw=0.4, which="both")
    # (b) nonlinear heat flux
    ax = axs[1]
    nl = d["nonlinear"]["cyclone_ad"]
    for c in ("GX", "GKX"):
        r = nl[c]
        ax.plot(
            r["trace_t"],
            r["trace_q"],
            color=COLOR[c],
            lw=0.7,
            alpha=0.75,
            label=f"{c}  $Q$ = {r['Q']:.1f} $\\pm$ {r['sem']:.1f}",
        )
        ax.axhspan(
            r["Q"] - r["sem"],
            r["Q"] + r["sem"],
            xmin=0.0,
            xmax=1.0,
            color=COLOR[c],
            alpha=0.12,
            lw=0,
        )
    lit = nl["literature"]
    ax.errorbar(
        [415],
        [lit["GX_paper_Q"]],
        yerr=[lit["GX_paper_err"]],
        fmt="none",
        ecolor="#333333",
        elinewidth=1.4,
        capsize=2.5,
        zorder=5,
        clip_on=False,
    )
    ax.text(
        420,
        9.4,
        "GX paper\n7.2 $\\pm$ 0.8",
        fontsize=6.3,
        ha="right",
        va="center",
        color="#333333",
    )
    ax.set_xlim(0, 425)
    ax.set_ylim(0, 12)
    ax.set_xlabel(r"$t\, v_{ti}/a$")
    ax.set_ylabel(r"$Q_i/Q_{\rm gB}$")
    ax.set_title("Cyclone nonlinear, adiabatic e", loc="left")
    ax.legend(loc="upper left", frameon=False, handlelength=1.4)
    ax.grid(alpha=0.3, lw=0.4)
    # (c) time to solution bars
    ax = axs[2]
    tl = d["timing"]
    groups = [
        (
            "linear, $k_y\\rho_i=0.3$\n(first rung within 1 %\nover $k_y\\rho_i$ = 0.2-0.5)",
            tl["linear_scan_grade"],
        ),
        ("nonlinear to\n$t\\,v_{ti}/a=400$", tl["nonlinear"]),
    ]
    x0 = 0
    ticks, labs = [], []
    for gname, items in groups:
        for i, it in enumerate(items):
            ax.bar(
                x0 + i,
                it["t"],
                color=COLOR[it["code"]],
                width=0.8,
                hatch=it.get("hatch"),
                edgecolor="white",
                lw=0.5,
            )
            ax.text(
                x0 + i,
                it["t"] * 1.12,
                it["label"],
                ha="center",
                va="bottom",
                fontsize=5.8,
                rotation=90,
            )
        ticks.append(x0 + (len(items) - 1) / 2)
        labs.append(gname)
        x0 += len(items) + 1
    ax.set_yscale("log")
    ax.set_ylim(1, 8000)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs, fontsize=6.3)
    ax.set_ylabel("wall time (s)")
    ax.set_title("Time to solution", loc="left")
    ax.grid(alpha=0.3, lw=0.4, axis="y", which="both")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", default=str(ROOT / "docs/figures/four_code_comparison.json")
    )
    ap.add_argument("--out", default=str(ROOT / "docs/_static"))
    a = ap.parse_args()
    d = json.load(open(a.data))
    out = Path(a.out)
    with figure_style(**RC):
        save_figure(linear_figure(d), out / "four_code_linear.png", palette_colors=96)
        save_figure(speed_figure(d), out / "four_code_speed.png", palette_colors=96)


if __name__ == "__main__":
    main()
