"""Shared Matplotlib style for SPECTRAX-GK publication figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

def set_plot_style() -> None:
    """Apply a consistent plotting style suitable for publications."""

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "figure.dpi": 120,
        }
    )



__all__ = ["set_plot_style"]
