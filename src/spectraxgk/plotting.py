"""Plotting utilities for publication-ready figures."""

from __future__ import annotations

from typing import Tuple
from dataclasses import dataclass

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

    ax0.plot(ref.ky, ref.gamma, marker="o", color="#1f77b4", label="Reference")
    ax0.set_ylabel(r"$\gamma a / v_{ti}$")
    ax0.set_title("Cyclone base case (adiabatic electrons)")
    ax0.legend(loc="best")

    ax1.plot(ref.ky, ref.omega, marker="o", color="#ff7f0e", label="Reference")
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

    ax0.plot(ref.ky, ref.gamma, marker="o", color="#1f77b4", label="Reference")
    ax0.plot(scan.ky, scan.gamma, marker="s", color="#2ca02c", label=label)
    ax0.set_ylabel(r"$\gamma a / v_{ti}$")
    ax0.set_title("Cyclone base case (adiabatic electrons)")
    ax0.legend(loc="best")

    ax1.plot(ref.ky, ref.omega, marker="o", color="#ff7f0e", label="Reference")
    ax1.plot(scan.ky, scan.omega, marker="s", color="#d62728", label=label)
    ax1.set_xlabel(r"$k_y \rho_i$")
    ax1.set_ylabel(r"$\omega a / v_{ti}$")
    ax1.legend(loc="best")

    fig.tight_layout()
    return fig, axes


def etg_trend_figure(
    R_over_LTe: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    ky_target: float,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a two-panel ETG trend plot versus R/LTe."""

    set_plot_style()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.0, 5.0))
    ax0, ax1 = axes

    ax0.plot(R_over_LTe, gamma, marker="o", color="#1f77b4")
    ax0.set_ylabel(r"$\gamma a / v_{ti}$")
    ax0.set_title(fr"ETG trend at $k_y={ky_target:.2f}$")

    ax1.plot(R_over_LTe, omega, marker="o", color="#ff7f0e")
    ax1.set_xlabel(r"$R/L_{Te}$")
    ax1.set_ylabel(r"$\omega a / v_{ti}$")

    fig.tight_layout()
    return fig, axes


def mtm_trend_figure(
    nu_values: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    ky_target: float,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a two-panel MTM trend plot versus collisionality."""

    set_plot_style()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.0, 5.0))
    ax0, ax1 = axes

    ax0.plot(nu_values, gamma, marker="o", color="#2ca02c")
    ax0.set_ylabel(r"$\gamma a / v_{ti}$")
    ax0.set_title(fr"MTM trend at $k_y={ky_target:.2f}$")

    ax1.plot(nu_values, omega, marker="o", color="#d62728")
    ax1.set_xlabel(r"$\nu$")
    ax1.set_ylabel(r"$\omega a / v_{ti}$")

    fig.tight_layout()
    return fig, axes


@dataclass(frozen=True)
class LinearValidationPanel:
    name: str
    z: np.ndarray
    eigenfunction: np.ndarray
    x: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray
    x_label: str


def linear_validation_figure(
    panels: list[LinearValidationPanel],
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a multi-panel summary plot of eigenfunctions, growth rates, and frequencies."""

    if len(panels) == 0:
        raise ValueError("panels must be non-empty")
    set_plot_style()
    nrows = len(panels)
    fig, axes = plt.subplots(nrows, 3, figsize=(12.0, 3.0 * nrows), sharex="col")
    if nrows == 1:
        axes = np.asarray([axes])

    for i, panel in enumerate(panels):
        ax0, ax1, ax2 = axes[i]
        ax0.plot(panel.z, panel.eigenfunction.real, color="#1f77b4", label="Re")
        ax0.plot(panel.z, panel.eigenfunction.imag, color="#ff7f0e", linestyle="--", label="Im")
        ax0.set_ylabel(panel.name)
        ax0.set_xlabel(r"$\theta$")
        if i == 0:
            ax0.set_title("Eigenfunction")
            ax1.set_title("Growth rate")
            ax2.set_title("Frequency")
        if i == 0:
            ax0.legend(loc="best", fontsize=9)

        ax1.plot(panel.x, panel.gamma, marker="o", color="#2ca02c")
        ax1.set_xlabel(panel.x_label)
        ax1.set_ylabel(r"$\gamma a / v_{ti}$")

        ax2.plot(panel.x, panel.omega, marker="o", color="#d62728")
        ax2.set_xlabel(panel.x_label)
        ax2.set_ylabel(r"$\omega a / v_{ti}$")

    fig.tight_layout()
    return fig, axes
