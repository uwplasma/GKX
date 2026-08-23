"""Shared color-blind-safe style and deterministic export for GKX figures."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from cycler import cycler

__all__ = [
    "GKX_COLORS",
    "SERIES",
    "REFERENCE_STYLE",
    "figure_style",
    "annotate_reference",
    "save_figure",
    "panel_label",
]

#: Okabe-Ito colour-blind safe qualitative palette.
GKX_COLORS: dict[str, str] = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#7F7F7F",
}

#: Default ordering for multi-series plots.
SERIES: tuple[str, ...] = (
    GKX_COLORS["blue"],
    GKX_COLORS["vermillion"],
    GKX_COLORS["green"],
    GKX_COLORS["orange"],
    GKX_COLORS["purple"],
    GKX_COLORS["sky"],
)

#: How a published/analytic reference curve is always drawn.
#:
#: Reference values are held to one visual convention everywhere: black, dashed,
#: behind the data. A reader should never have to check a legend to find out
#: which curve is the one being tested against.
REFERENCE_STYLE: dict[str, Any] = {
    "color": GKX_COLORS["black"],
    "linestyle": (0, (6, 3)),
    "linewidth": 1.6,
    "zorder": 2,
    "alpha": 0.85,
}

_RC: dict[str, Any] = {
    # Typography. One family, few sizes; sizes are chosen so a two-column
    # figure downscaled to README width still reads at ~9pt equivalent.
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "mathtext.fontset": "dejavusans",
    # Axes: light frame, no top/right spine, grid behind the data.
    "axes.prop_cycle": cycler(color=list(SERIES)),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "axes.titlelocation": "left",
    "axes.titlepad": 8.0,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.color": "#B0B0B0",
    "grid.alpha": 0.35,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    # Marks.
    "lines.linewidth": 1.9,
    "lines.markersize": 5.5,
    "lines.markeredgewidth": 0.0,
    "legend.frameon": False,
    "legend.handlelength": 1.9,
    "legend.columnspacing": 1.2,
    "legend.labelspacing": 0.35,
    # Ticks point out, only where there is a spine.
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    # Export. 200 dpi keeps text crisp on HiDPI without bloating the repo.
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


@contextmanager
def figure_style(**overrides: Any) -> Iterator[None]:
    """Apply the house style without changing process-global defaults."""

    # matplotlib types rc_context against a Literal of every valid key, which a
    # plain dict cannot satisfy; the keys are validated by matplotlib at runtime.
    with plt.rc_context(cast(Any, {**_RC, **overrides})):
        yield


def panel_label(
    ax: plt.Axes, text: str, *, dx: float = -0.085, dy: float = 1.06
) -> None:
    """Put a bold ``(a)``-style panel label just outside the axes corner."""

    ax.text(
        dx,
        dy,
        text,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def annotate_reference(ax: plt.Axes, text: str, *, loc: str = "lower left") -> None:
    """Attach the literature or physics anchor to a panel."""

    positions = {
        "lower left": (0.02, 0.03, "left", "bottom"),
        "lower right": (0.98, 0.03, "right", "bottom"),
        "upper left": (0.02, 0.97, "left", "top"),
        "upper right": (0.98, 0.97, "right", "top"),
    }
    x, y, ha, va = positions[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=8.5,
        color="#333333",
        ha=ha,
        va=va,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "linewidth": 0.7,
            "alpha": 0.92,
        },
        zorder=10,
    )


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    *,
    close: bool = True,
    palette_colors: int | None = None,
) -> Path:
    """Write a figure to ``path``, creating parent directories."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target)
    if palette_colors is not None and target.suffix.lower() == ".png":
        from PIL import Image

        with Image.open(target) as image:
            preview = image.convert("RGB").quantize(
                colors=palette_colors,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.NONE,
            )
            preview.save(target, optimize=True)
    if close:
        plt.close(fig)
    return target
