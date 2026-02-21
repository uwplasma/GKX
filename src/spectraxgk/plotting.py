"""Plotting utilities for publication-ready figures."""

from __future__ import annotations

from typing import Tuple
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from spectraxgk.benchmarks import CycloneReference, CycloneScanResult
from spectraxgk.analysis import fit_growth_rate


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
    ax0.set_xscale("log")

    ax1.plot(ref.ky, ref.omega, marker="o", color="#ff7f0e", label="Reference")
    ax1.set_xlabel(r"$k_y \rho_i$")
    ax1.set_ylabel(r"$\omega a / v_{ti}$")
    ax1.legend(loc="best")
    ax1.set_xscale("log")

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
    ax0.set_xscale("log")

    ax1.plot(ref.ky, ref.omega, marker="o", color="#ff7f0e", label="Reference")
    ax1.plot(scan.ky, scan.omega, marker="s", color="#d62728", label=label)
    ax1.set_xlabel(r"$k_y \rho_i$")
    ax1.set_ylabel(r"$\omega a / v_{ti}$")
    ax1.legend(loc="best")
    ax1.set_xscale("log")

    fig.tight_layout()
    return fig, axes


def scan_comparison_figure(
    x: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    x_label: str,
    title: str,
    x_ref: np.ndarray | None = None,
    gamma_ref: np.ndarray | None = None,
    omega_ref: np.ndarray | None = None,
    label: str = "SPECTRAX-GK",
    ref_label: str = "Reference",
    log_x: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a two-panel comparison plot for a generic scan."""

    set_plot_style()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.0, 5.0))
    ax0, ax1 = axes

    ax0.plot(x, gamma, marker="o", color="#2ca02c", label=label)
    if x_ref is not None and gamma_ref is not None:
        ax0.plot(x_ref, gamma_ref, marker="o", linestyle="None", color="#1f77b4", label=ref_label)
    ax0.set_ylabel(r"$\gamma a / v_{ti}$")
    ax0.set_title(title)
    ax0.legend(loc="best")
    if log_x:
        ax0.set_xscale("log")

    ax1.plot(x, omega, marker="o", color="#d62728", label=label)
    if x_ref is not None and omega_ref is not None:
        ax1.plot(x_ref, omega_ref, marker="o", linestyle="None", color="#1f77b4", label=ref_label)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(r"$\omega a / v_{ti}$")
    ax1.legend(loc="best")
    if log_x:
        ax1.set_xscale("log")

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


@dataclass(frozen=True)
class LinearValidationPanel:
    name: str
    z: np.ndarray
    eigenfunction: np.ndarray
    x: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray
    x_label: str
    x_ref: np.ndarray | None = None
    gamma_ref: np.ndarray | None = None
    omega_ref: np.ndarray | None = None
    ref_label: str = "Reference"
    log_x: bool = False


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

        ax1.plot(panel.x, panel.gamma, marker="o", color="#2ca02c", label="SPECTRAX-GK")
        if panel.x_ref is not None and panel.gamma_ref is not None:
            ax1.plot(panel.x_ref, panel.gamma_ref, marker="o", linestyle="None", color="#1f77b4", label=panel.ref_label)
        ax1.set_xlabel(panel.x_label)
        ax1.set_ylabel(r"$\gamma a / v_{ti}$")
        if panel.log_x:
            ax1.set_xscale("log")

        ax2.plot(panel.x, panel.omega, marker="o", color="#d62728", label="SPECTRAX-GK")
        if panel.x_ref is not None and panel.omega_ref is not None:
            ax2.plot(panel.x_ref, panel.omega_ref, marker="o", linestyle="None", color="#1f77b4", label=panel.ref_label)
        ax2.set_xlabel(panel.x_label)
        ax2.set_ylabel(r"$\omega a / v_{ti}$")
        if panel.log_x:
            ax2.set_xscale("log")
        if i == 0:
            ax1.legend(loc="best", fontsize=9)
            ax2.legend(loc="best", fontsize=9)

    fig.tight_layout()
    return fig, axes


def growth_rate_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    gamma: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    cmap: str = "jet",
) -> Tuple[plt.Figure, plt.Axes]:
    """Render a growth-rate heatmap versus two gradient axes."""

    set_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    im = ax.imshow(gamma, origin="lower", aspect="auto", extent=extent, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(im, ax=ax, label=r"$\gamma a / v_{ti}$")
    fig.tight_layout()
    return fig, ax


def growth_fit_figure(
    t: np.ndarray,
    signal: np.ndarray,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
    title: str = "Growth-fit window",
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot |signal|^2 and log|signal|^2 with an optional fit window."""

    set_plot_style()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.0, 4.5))
    ax0, ax1 = axes
    energy = np.abs(signal) ** 2
    tiny = np.finfo(float).tiny
    log_energy = np.log(np.maximum(energy, tiny))
    ax0.plot(t, energy, label=r"$|s|^2$")
    ax0.set_ylabel("energy")
    ax1.plot(t, log_energy, label=r"$\log|s|^2$")
    ax1.set_ylabel("log energy")
    ax1.set_xlabel("t")
    ax0.set_title(title)

    if tmin is not None and tmax is not None and tmax > tmin:
        ax0.axvspan(tmin, tmax, color="orange", alpha=0.2, label="fit window")
        ax1.axvspan(tmin, tmax, color="orange", alpha=0.2)
        gamma, _omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)
        idx = int(np.searchsorted(t, tmin))
        log_ref = log_energy[idx] if idx < log_energy.size else log_energy[-1]
        fit_line = 2.0 * gamma * (t - tmin) + log_ref
        ax1.plot(t, fit_line, color="red", linestyle="--", label="fit line")
    ax0.legend(loc="best", fontsize=9)
    ax1.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig, axes
