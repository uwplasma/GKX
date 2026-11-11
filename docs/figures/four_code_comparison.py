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
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "legend.fontsize": 9.0,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
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
        ("cyclone_ad", "(a) Cyclone, adiabatic e", 6),
        ("cyclone_ke", "(b) Cyclone, kinetic e", 8),
        ("em", r"(c) kinetic e, A$_\parallel$", 16),
        ("w7x_ad", "(d) W7-X, adiabatic e", 40),
    ]
    fig, axs = plt.subplots(
        3,
        4,
        figsize=(8.0, 5.2),
        sharex="col",
        gridspec_kw={
            "hspace": 0.14,
            "wspace": 0.62,
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


def _fmt_time(s: float) -> str:
    if s < 100:
        return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h" if s < 36000 else f"{s / 3600:.0f} h"


def _dof(n: float) -> str:
    return f"{n / 1e6:.1f} M" if n >= 1e6 else f"{round(n / 1e3):,} k"


def _bars(ax, rows, xlim, ticks, note_fn):
    """Horizontal log-scale bars, one row per code, with a text note at each bar end."""
    for k, r in enumerate(rows):
        y = len(rows) - 1 - k
        c = r["code"]
        done = r.get("measured_s", r["t"])
        ax.barh(y, done, left=xlim[0], height=0.62, color=COLOR[c], zorder=3)
        if r["t"] > done * 1.02:  # extrapolated part
            ax.barh(
                y,
                r["t"] - done,
                left=done,
                height=0.62,
                facecolor="white",
                edgecolor=COLOR[c],
                hatch="////",
                lw=1.0,
                zorder=3,
            )
        top, sub = note_fn(r)
        ax.text(
            r["t"] * 1.3,
            y + 0.13,
            top,
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="bold",
            color="#111111",
        )
        ax.text(
            r["t"] * 1.3,
            y - 0.21,
            sub,
            va="center",
            ha="left",
            fontsize=8.5,
            color="#444444",
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{r['code']}\n{r['hardware']}" for r in rows[::-1]], fontsize=9.5
    )
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([n for _, n in ticks])
    ax.minorticks_off()
    ax.tick_params(axis="y", length=0)
    ax.grid(False, axis="y")
    ax.grid(alpha=0.3, lw=0.5, axis="x", zorder=0)
    ax.spines["left"].set_visible(False)


def speed_figure(d: dict) -> plt.Figure:
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(8.0, 3.3),
        gridspec_kw={"wspace": 0.62, "left": 0.115, "right": 0.985, "bottom": 0.2},
    )
    lin = d["timing"]["linear"]
    _bars(
        axs[0],
        lin,
        (1.0, 700.0),
        [(1, "1 s"), (10, "10 s"), (100, "100 s")],
        lambda r: (
            _fmt_time(r["t"]),
            f"{_dof(r['dof'])} DOF"
            + (
                f", {_fmt_time(r['t_compile'])} with compile"
                if r.get("t_compile")
                else ""
            ),
        ),
    )
    axs[0].set_title(
        "Linear: one $k_y$, 1% growth-rate accuracy", loc="left", fontsize=10.5
    )
    axs[0].set_xlabel("wall time (log scale)")
    nl = d["timing"]["nonlinear"]

    def nl_note(r):
        if r.get("cores"):
            return (
                "~" + _fmt_time(r["t"]),
                f"{r['core_hours']:,.0f} core-h; run to $t$ = {r['t_reached']:.0f}",
            )
        return _fmt_time(r["t"]), f"$Q_i$ = {r['Q']:.1f} $\\pm$ {r['sem']:.1f}"

    _bars(
        axs[1],
        nl,
        (60.0, 3.0e6),
        [
            (60, "1 min"),
            (600, "10 min"),
            (3600, "1 h"),
            (36000, "10 h"),
            (360000, "100 h"),
        ],
        nl_note,
    )
    axs[1].set_title(
        "Nonlinear Cyclone, to $t\\,v_{ti}/a = 400$", loc="left", fontsize=10.5
    )
    axs[1].set_xlabel("wall time (log scale); hatched = extrapolated from step rate")
    return fig


def trace_figure(d: dict) -> plt.Figure:
    """Nonlinear Cyclone heat flux against time for the four codes."""
    nl = d["nonlinear"]["cyclone_ad"]
    fig, ax = plt.subplots(
        figsize=(8.0, 2.7),
        gridspec_kw={"bottom": 0.2, "left": 0.08, "right": 0.985, "top": 0.9},
    )
    for c in ("GX", "GKX", "GS2", "stella"):
        r = nl[c]
        ax.plot(
            r["trace_t"],
            r["trace_q"],
            color=COLOR[c],
            lw=1.6 if c == "GKX" else 1.1,
            ls="-" if c in ("GKX", "GS2", "stella") else "--",
            alpha=0.95 if c == "GKX" else 0.85,
            label=c,
            zorder=3 if c == "GKX" else 2,
        )
    lit = nl["literature"]
    ax.axhspan(
        lit["GX_paper_Q"] - lit["GX_paper_err"],
        lit["GX_paper_Q"] + lit["GX_paper_err"],
        color="#999999",
        alpha=0.25,
        lw=0,
        label="GX paper, $t\\leq 1000$",
    )
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 30)
    ax.set_xlabel(r"$t\,v_{ti}/a$")
    ax.set_ylabel(r"$Q_i/Q_{\rm gB}$")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(ncol=5, frameon=False, loc="upper right", fontsize=9)
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
        save_figure(trace_figure(d), out / "four_code_traces.png", palette_colors=96)


if __name__ == "__main__":
    main()
