"""Plotting utilities for publication-ready figures."""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from spectraxgk.benchmarks import CycloneReference, CycloneScanResult


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


def cyclone_reference_figure(ref: CycloneReference) -> Tuple[plt.Figure, np.ndarray]:
    """Create a two-panel Cyclone base case reference plot."""

    set_plot_style()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.0, 5.0))
    ax0, ax1 = axes

    ax0.plot(ref.ky, ref.gamma, marker="o", color="#1f77b4", label="GX")
    ax0.set_ylabel(r"$\gamma a / v_{ti}$")
    ax0.set_title("Cyclone base case (adiabatic electrons)")
    ax0.legend(loc="best")

    ax1.plot(ref.ky, ref.omega, marker="o", color="#ff7f0e", label="GX")
    ax1.set_xlabel(r"$k_y \rho_i$")
    ax1.set_ylabel(r"$\omega a / v_{ti}$")
    ax1.legend(loc="best")

    fig.tight_layout()
    return fig, axes


def cyclone_comparison_figure(
    ref: CycloneReference,
    scan: CycloneScanResult,
    label: str = "SPECTRAX-GK",
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a two-panel comparison plot between reference and solver output."""

    set_plot_style()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.0, 5.0))
    ax0, ax1 = axes

    ax0.plot(ref.ky, ref.gamma, marker="o", color="#1f77b4", label="GX")
    ax0.plot(scan.ky, scan.gamma, marker="s", color="#2ca02c", label=label)
    ax0.set_ylabel(r"$\gamma a / v_{ti}$")
    ax0.set_title("Cyclone base case (adiabatic electrons)")
    ax0.legend(loc="best")

    ax1.plot(ref.ky, ref.omega, marker="o", color="#ff7f0e", label="GX")
    ax1.plot(scan.ky, scan.omega, marker="s", color="#d62728", label=label)
    ax1.set_xlabel(r"$k_y \rho_i$")
    ax1.set_ylabel(r"$\omega a / v_{ti}$")
    ax1.legend(loc="best")

    fig.tight_layout()
    return fig, axes
