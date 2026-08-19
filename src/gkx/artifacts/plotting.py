"""Publication-ready benchmark, diagnostic, runtime, and zonal plots."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np

from gkx.benchmarking.shared import CycloneReference, CycloneScanResult
from gkx.diagnostics.growth_rates import fit_growth_rate

def set_plot_style() -> None:
    """Apply the shared publication style used by generated figures."""

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
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.5, 5.0))
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
    label: str = "GKX",
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a two-panel comparison plot between reference and solver output."""

    set_plot_style()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.5, 5.0))
    ax0, ax1 = axes

    ax0.plot(
        ref.ky,
        ref.gamma,
        marker="o",
        color="#1f77b4",
        linewidth=2.0,
        label="Reference",
    )
    ax0.plot(
        scan.ky,
        scan.gamma,
        marker="s",
        markerfacecolor="none",
        markeredgewidth=1.6,
        linestyle="--",
        color="#2ca02c",
        linewidth=1.8,
        label=label,
    )
    ax0.set_ylabel(r"$\gamma a / v_{ti}$")
    ax0.set_title("Cyclone base case (adiabatic electrons)")
    ax0.legend(loc="best")

    ax1.plot(
        ref.ky,
        ref.omega,
        marker="o",
        color="#ff7f0e",
        linewidth=2.0,
        label="Reference",
    )
    ax1.plot(
        scan.ky,
        scan.omega,
        marker="s",
        markerfacecolor="none",
        markeredgewidth=1.6,
        linestyle="--",
        color="#d62728",
        linewidth=1.8,
        label=label,
    )
    ax1.set_xlabel(r"$k_y \rho_i$")
    ax1.set_ylabel(r"$\omega a / v_{ti}$")
    ax1.legend(loc="best")
    ax1.set_xticks([0.05, 0.1, 0.2, 0.3, 0.4])

    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(left=0.18)
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
    label: str = "GKX",
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
    tprim_e: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    ky_target: float,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a two-panel ETG trend plot versus the electron drive ``a/L_Te``.

    The scanned quantity is the electron species' ``tprim``, which is
    :math:`a/L_{Te}`; the axis used to be labelled :math:`R/L_{Te}`.
    """

    set_plot_style()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.0, 5.0))
    ax0, ax1 = axes

    ax0.plot(tprim_e, gamma, marker="o", color="#1f77b4")
    ax0.set_ylabel(r"$\gamma a / v_{ti}$")
    ax0.set_title(fr"ETG trend at $k_y={ky_target:.2f}$")

    ax1.plot(tprim_e, omega, marker="o", color="#ff7f0e")
    ax1.set_xlabel(r"$a/L_{Te}$")
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


@dataclass(frozen=True)
class ReferenceSeries:
    label: str
    x: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray
    color: str
    marker: str = "o"
    linestyle: str = "--"


@dataclass(frozen=True)
class MultiReferenceValidationPanel:
    name: str
    z: np.ndarray
    eigenfunction: np.ndarray
    x: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray
    x_label: str
    references: list[ReferenceSeries]
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

        ax1.plot(panel.x, panel.gamma, marker="o", color="#2ca02c", label="GKX")
        if panel.x_ref is not None and panel.gamma_ref is not None:
            ax1.plot(panel.x_ref, panel.gamma_ref, marker="o", linestyle="None", color="#1f77b4", label=panel.ref_label)
        ax1.set_xlabel(panel.x_label)
        ax1.set_ylabel(r"$\gamma a / v_{ti}$")
        if panel.log_x:
            ax1.set_xscale("log")

        ax2.plot(panel.x, panel.omega, marker="o", color="#d62728", label="GKX")
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


def linear_validation_multi_reference_figure(
    panels: list[MultiReferenceValidationPanel],
) -> Tuple[plt.Figure, np.ndarray]:
    """Create summary panels with multiple external reference curves."""

    if len(panels) == 0:
        raise ValueError("panels must be non-empty")
    set_plot_style()
    nrows = len(panels)
    # Keep each row on its own x-range so Cyclone- and ETG-scale ky scans
    # remain readable in the combined summary figure.
    fig, axes = plt.subplots(nrows, 3, figsize=(12.0, 3.0 * nrows), sharex=False)
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
            ax0.legend(loc="best", fontsize=9)

        ax1.plot(panel.x, panel.gamma, marker="o", color="#2ca02c", label="GKX")
        ax2.plot(panel.x, panel.omega, marker="o", color="#d62728", label="GKX")
        for ref in panel.references:
            ax1.plot(
                ref.x,
                ref.gamma,
                marker=ref.marker,
                linestyle=ref.linestyle,
                color=ref.color,
                label=ref.label,
            )
            ax2.plot(
                ref.x,
                ref.omega,
                marker=ref.marker,
                linestyle=ref.linestyle,
                color=ref.color,
                label=ref.label,
            )
        ax1.set_xlabel(panel.x_label)
        ax1.set_ylabel(r"$\gamma a / v_{ti}$")
        ax2.set_xlabel(panel.x_label)
        ax2.set_ylabel(r"$\omega a / v_{ti}$")
        if panel.log_x:
            ax1.set_xscale("log")
            ax2.set_xscale("log")
        if i == 0:
            ax1.legend(loc="best", fontsize=9)
            ax2.legend(loc="best", fontsize=9)

    fig.tight_layout()
    return fig, axes


def scan_multi_reference_figure(
    x: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    x_label: str,
    title: str,
    references: list[ReferenceSeries],
    *,
    log_x: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a two-panel comparison figure against multiple reference curves."""

    set_plot_style()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.5, 5.0))
    ax0, ax1 = axes
    ax0.plot(x, gamma, marker="o", color="#2ca02c", label="GKX")
    ax1.plot(x, omega, marker="o", color="#d62728", label="GKX")
    for ref in references:
        ax0.plot(
            ref.x,
            ref.gamma,
            marker=ref.marker,
            linestyle=ref.linestyle,
            color=ref.color,
            label=ref.label,
        )
        ax1.plot(
            ref.x,
            ref.omega,
            marker=ref.marker,
            linestyle=ref.linestyle,
            color=ref.color,
            label=ref.label,
        )
    ax0.set_title(title)
    ax0.set_ylabel(r"$\gamma a / v_{ti}$")
    ax1.set_ylabel(r"$\omega a / v_{ti}$")
    ax1.set_xlabel(x_label)
    if log_x:
        ax0.set_xscale("log")
        ax1.set_xscale("log")
    ax0.legend(loc="best")
    ax1.legend(loc="best")
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
    """Plot :math:`|s|^2` and :math:`\\log |s|^2` with an optional fit window."""

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
        fit_mask = (t >= tmin) & (t <= tmax)
        fit_t = t[fit_mask]
        if fit_t.size:
            log_ref = log_energy[fit_mask][0]
            fit_line = 2.0 * gamma * (fit_t - fit_t[0]) + log_ref
            ax1.plot(
                fit_t, fit_line, color="red", linestyle="--", label="fit line"
            )
    ax0.legend(loc="best", fontsize=9)
    ax1.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig, axes


def eigenfunction_overlap_summary_figure(
    ky: np.ndarray,
    overlap: np.ndarray,
    relative_l2: np.ndarray,
    *,
    title: str = "Eigenfunction overlap summary",
    x_label: str = r"$k_y \rho_i$",
    overlap_label: str = "Normalized overlap",
    rel_l2_label: str = "Relative $L^2$ error",
    log_x: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Render a compact two-panel eigenfunction-overlap summary."""

    set_plot_style()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.6, 5.2))
    ax0, ax1 = axes
    ky_arr = np.asarray(ky, dtype=float)
    overlap_arr = np.asarray(overlap, dtype=float)
    rel_l2_arr = np.asarray(relative_l2, dtype=float)

    ax0.plot(ky_arr, overlap_arr, color="#0f4c81", marker="o", linewidth=2.2, label=overlap_label)
    ax0.set_ylabel("overlap")
    ax0.set_ylim(0.0, min(1.02, max(1.0, float(np.nanmax(overlap_arr)) + 0.02)))
    ax0.set_title(title)
    ax0.legend(loc="best", frameon=False)

    ax1.plot(ky_arr, rel_l2_arr, color="#c44e52", marker="s", linewidth=2.2, label=rel_l2_label)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(r"relative $L^2$")
    ax1.legend(loc="best", frameon=False)

    if log_x:
        ax0.set_xscale("log")
        ax1.set_xscale("log")

    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig, axes


def eigenfunction_reference_overlay_figure(
    theta: np.ndarray,
    eigenfunction: np.ndarray,
    theta_ref: np.ndarray,
    reference: np.ndarray,
    *,
    title: str = "Eigenfunction overlay",
) -> Tuple[plt.Figure, np.ndarray]:
    """Render a phase-aligned raw overlay against a frozen reference mode."""

    from gkx.diagnostics.modes import (
        compare_eigenfunctions,
        phase_align_eigenfunction,
    )

    set_plot_style()
    theta_arr = np.asarray(theta, dtype=float)
    eig = np.asarray(eigenfunction, dtype=np.complex128)
    theta_ref_arr = np.asarray(theta_ref, dtype=float)
    ref = np.asarray(reference, dtype=np.complex128)
    if eig.shape != ref.shape:
        raise ValueError("eigenfunction and reference must have the same shape")

    eig_aligned, _phase = phase_align_eigenfunction(eig, ref)
    metrics = compare_eigenfunctions(eig, ref)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    ax0, ax1, ax2 = axes

    ax0.plot(theta_ref_arr, np.real(ref), color="#0f4c81", linewidth=2.4, label="Reference Re")
    ax0.plot(theta_arr, np.real(eig_aligned), color="#c44e52", linewidth=2.0, linestyle="--", label="GKX Re")
    ax0.set_xlabel(r"$\theta$")
    ax0.set_ylabel("real")
    ax0.set_title("Real part")
    ax0.legend(loc="best", frameon=False)

    ax1.plot(theta_ref_arr, np.imag(ref), color="#0f4c81", linewidth=2.4, label="Reference Im")
    ax1.plot(theta_arr, np.imag(eig_aligned), color="#c44e52", linewidth=2.0, linestyle="--", label="GKX Im")
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel("imag")
    ax1.set_title("Imaginary part")
    ax1.legend(loc="best", frameon=False)

    ax2.plot(theta_ref_arr, np.abs(ref), color="#0f4c81", linewidth=2.4, label="Reference $|\\phi|$")
    ax2.plot(theta_arr, np.abs(eig_aligned), color="#c44e52", linewidth=2.0, linestyle="--", label="GKX $|\\phi|$")
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$|\phi|$")
    ax2.set_title("Amplitude")
    ax2.legend(loc="upper right", frameon=False)
    ax2.text(
        0.03,
        0.04,
        f"overlap = {metrics.overlap:.4f}\nrel $L^2$ = {metrics.relative_l2:.4f}",
        transform=ax2.transAxes,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes




def _normalize_by_real_max(eigenfunction: np.ndarray) -> np.ndarray:
    eigen = np.asarray(eigenfunction, dtype=np.complex128)
    real_scale = float(np.max(np.abs(np.real(eigen)))) if eigen.size else 0.0
    if real_scale <= 0.0:
        abs_scale = float(np.max(np.abs(eigen))) if eigen.size else 0.0
        if abs_scale > 0.0:
            return eigen / abs_scale
        return eigen
    return eigen / real_scale


def linear_runtime_panel_figure(
    *,
    t: np.ndarray,
    signal: np.ndarray,
    z: np.ndarray,
    eigenfunction: np.ndarray,
    gamma: float,
    omega: float,
    title: str = "GKX Linear Runtime",
) -> Tuple[plt.Figure, np.ndarray]:
    """Create the default two-panel linear runtime plot."""

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    ax0, ax1 = axes

    signal_arr = np.asarray(signal, dtype=np.complex128)
    amp2 = np.maximum(np.abs(signal_arr) ** 2, 1.0e-30)
    ax0.plot(np.asarray(t, dtype=float), amp2, color="#0f4c81", linewidth=2.4)
    ax0.set_yscale("log")
    ax0.set_xlabel("t")
    ax0.set_ylabel(r"$|\phi|^2$")
    ax0.set_title("Linear growth history")
    ax0.text(
        0.04,
        0.96,
        rf"$\gamma={gamma:.5f}$" + "\n" + rf"$\omega={omega:.5f}$",
        transform=ax0.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    eigen_norm = _normalize_by_real_max(eigenfunction)
    ax1.plot(np.asarray(z, dtype=float), np.real(eigen_norm), color="#0f4c81", linewidth=2.4, label="Re")
    ax1.plot(
        np.asarray(z, dtype=float),
        np.imag(eigen_norm),
        color="#c44e52",
        linewidth=2.2,
        linestyle="--",
        label="Im",
    )
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\phi / \max |\Re(\phi)|$")
    ax1.set_title("Eigenfunction")
    ax1.legend(loc="best", frameon=False)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


def nonlinear_runtime_panel_figure(
    *,
    t: np.ndarray,
    phi2: np.ndarray | None = None,
    wphi: np.ndarray | None = None,
    heat_flux: np.ndarray | None = None,
    gamma: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    title: str = "GKX Nonlinear Runtime",
) -> Tuple[plt.Figure, np.ndarray]:
    """Create the default three-panel nonlinear runtime plot."""

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    t_arr = np.asarray(t, dtype=float)

    ax0, ax1, ax2 = axes
    if phi2 is not None:
        ax0.plot(t_arr, np.maximum(np.asarray(phi2, dtype=float), 1.0e-30), color="#0f4c81", linewidth=2.4)
        ax0.set_yscale("log")
        ax0.set_ylabel(r"$|\phi|^2$")
        ax0.set_title("Field amplitude")
    elif wphi is not None:
        ax0.plot(t_arr, np.asarray(wphi, dtype=float), color="#0f4c81", linewidth=2.4)
        ax0.set_ylabel(r"$W_\phi$")
        ax0.set_title("Electrostatic energy")

    if wphi is not None:
        ax1.plot(t_arr, np.asarray(wphi, dtype=float), color="#2a9d8f", linewidth=2.4, label=r"$W_\phi$")
    if gamma is not None:
        ax1.plot(t_arr, np.asarray(gamma, dtype=float), color="#f4a261", linewidth=2.0, linestyle="--", label=r"$\gamma$")
    if omega is not None:
        ax1.plot(t_arr, np.asarray(omega, dtype=float), color="#c44e52", linewidth=2.0, linestyle=":", label=r"$\omega$")
    ax1.set_xlabel("t")
    ax1.set_title("Resolved diagnostics")
    if wphi is not None or gamma is not None or omega is not None:
        ax1.legend(loc="best", frameon=False)

    if heat_flux is not None:
        ax2.plot(t_arr, np.asarray(heat_flux, dtype=float), color="#c44e52", linewidth=2.4)
    ax2.set_xlabel("t")
    ax2.set_ylabel("Heat flux")
    ax2.set_title("Transport")

    ax0.set_xlabel("t")
    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


def _artifact_base(path: Path) -> Path:
    name = path.name
    for suffix in (".summary.json", ".timeseries.csv", ".eigenfunction.csv", ".diagnostics.csv", ".out.nc"):
        if name.lower().endswith(suffix):
            return path.with_name(name[: -len(suffix)])
    if path.suffix.lower() in {".json", ".csv", ".nc"}:
        return path.with_suffix("")
    return path


def _load_linear_bundle(base: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    summary = json.loads(base.with_suffix(".summary.json").read_text(encoding="utf-8"))
    timeseries = np.genfromtxt(base.with_suffix(".timeseries.csv"), delimiter=",", names=True, dtype=float)
    eigen = np.genfromtxt(base.with_suffix(".eigenfunction.csv"), delimiter=",", names=True, dtype=float)
    t = np.asarray(timeseries["t"], dtype=float)
    signal = np.asarray(timeseries["signal_real"], dtype=float) + 1j * np.asarray(timeseries["signal_imag"], dtype=float)
    z = np.asarray(eigen["z"], dtype=float)
    eig = np.asarray(eigen["eigen_real"], dtype=float) + 1j * np.asarray(eigen["eigen_imag"], dtype=float)
    return summary, t, signal, z, eig


def _load_nonlinear_csv(base: Path) -> tuple[dict, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    summary = json.loads(base.with_suffix(".summary.json").read_text(encoding="utf-8"))
    diag = np.genfromtxt(base.with_suffix(".diagnostics.csv"), delimiter=",", names=True, dtype=float)
    names = set(diag.dtype.names or ())
    t = np.asarray(diag["t"], dtype=float)
    wphi = np.asarray(diag["Wphi"], dtype=float) if "Wphi" in names else None
    heat_flux = np.asarray(diag["heat_flux"], dtype=float) if "heat_flux" in names else None
    gamma = np.asarray(diag["gamma"], dtype=float) if "gamma" in names else None
    omega = np.asarray(diag["omega"], dtype=float) if "omega" in names else None
    return summary, t, wphi, heat_flux, gamma, omega


def _load_nonlinear_netcdf(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    try:
        import netCDF4
    except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit("netCDF4 is required to plot *.out.nc runtime bundles") from exc

    with netCDF4.Dataset(path) as root:
        diag = root.groups["Diagnostics"]
        t = np.asarray(diag.variables["t"][:], dtype=float)
        phi2 = np.asarray(diag.variables["Phi2_t"][:], dtype=float) if "Phi2_t" in diag.variables else None
        wphi = None
        heat_flux = None
        if "Wphi_st" in diag.variables:
            wphi = np.sum(np.asarray(diag.variables["Wphi_st"][:], dtype=float), axis=1)
        if "HeatFlux_st" in diag.variables:
            heat_flux = np.sum(np.asarray(diag.variables["HeatFlux_st"][:], dtype=float), axis=1)
    return t, phi2, wphi, heat_flux


def plot_saved_output(path: str | Path, *, out: str | Path | None = None) -> Path:
    """Plot a saved linear or nonlinear output bundle."""

    in_path = Path(path)
    base = _artifact_base(in_path)
    out_path = Path(out) if out is not None else Path(f"{base}.plot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".nc" or in_path.name.lower().endswith(".out.nc"):
        t, phi2, wphi, heat_flux = _load_nonlinear_netcdf(in_path)
        fig, _axes = nonlinear_runtime_panel_figure(
            t=t,
            phi2=phi2,
            wphi=wphi,
            heat_flux=heat_flux,
            title=f"GKX nonlinear runtime: {base.name}",
        )
    else:
        summary_path = base.with_suffix(".summary.json")
        if not summary_path.exists():
            raise FileNotFoundError(f"Could not infer runtime summary from {in_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        kind = summary.get("kind")
        if kind == "linear":
            _summary, t, signal, z, eig = _load_linear_bundle(base)
            fig, _axes = linear_runtime_panel_figure(
                t=t,
                signal=signal,
                z=z,
                eigenfunction=eig,
                gamma=float(summary["gamma"]),
                omega=float(summary["omega"]),
                title=f"GKX linear runtime: {base.name}",
            )
        elif kind == "nonlinear":
            _summary, t, wphi, heat_flux, gamma, omega = _load_nonlinear_csv(base)
            fig, _axes = nonlinear_runtime_panel_figure(
                t=t,
                wphi=wphi,
                heat_flux=heat_flux,
                gamma=gamma,
                omega=omega,
                title=f"GKX nonlinear runtime: {base.name}",
            )
        else:
            raise ValueError(f"Unsupported saved-output kind: {kind!r}")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path




def zonal_flow_response_figure(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    """Render a zonal response without importing its fit helpers at startup."""

    from gkx.artifacts.zonal_plots import zonal_flow_response_figure as render

    return render(*args, **kwargs)


__all__ = [
    "LinearValidationPanel",
    "MultiReferenceValidationPanel",
    "ReferenceSeries",
    "cyclone_comparison_figure",
    "cyclone_reference_figure",
    "eigenfunction_overlap_summary_figure",
    "eigenfunction_reference_overlay_figure",
    "etg_trend_figure",
    "growth_fit_figure",
    "growth_rate_heatmap",
    "linear_runtime_panel_figure",
    "linear_validation_figure",
    "linear_validation_multi_reference_figure",
    "nonlinear_runtime_panel_figure",
    "plot_saved_output",
    "scan_comparison_figure",
    "scan_multi_reference_figure",
    "set_plot_style",
    "zonal_flow_response_figure",
]


# ---------------------------------------------------------------------------
# Publication figures for nonlinear transport outputs.
#
# Each figure function accepts either an in-memory ``SimulationDiagnostics``
# (duck-typed: only the attributes actually used are touched) or a path to a
# saved GKX output -- a NetCDF bundle (``*.out.nc``) or a CSV diagnostics
# sidecar (``*.diagnostics.csv`` / its ``*.summary.json`` / bare base path).
# The CSV sidecars carry time traces only; the spectra figures therefore
# require the NetCDF bundle and say so in their errors.
# ---------------------------------------------------------------------------

from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

from gkx.artifacts.figure_style import (  # noqa: E402
    GKX_COLORS,
    SERIES,
    annotate_reference,
    figure_style,
    panel_label,
    save_figure,
)

_TIME_LABEL = r"$t \, c_s/a$"
_HEAT_FLUX_LABEL = r"$Q/Q_{\mathrm{gB}}$"
_PARTICLE_FLUX_LABEL = r"$\Gamma/\Gamma_{\mathrm{gB}}$"
_KY_LABEL = r"$k_y \rho_i$"
_KX_LABEL = r"$k_x \rho_i$"

_CSV_HAS_NO_SPECTRA = (
    "{name} needs k-resolved spectra, which only the NetCDF output bundle "
    "(*.out.nc) carries; a *.diagnostics.csv sidecar stores time traces only. "
    "Re-run with `gkx run-runtime-nonlinear ... --out <case>.out.nc` and point "
    "this figure at the .out.nc file."
)


def _is_netcdf_bundle_path(path: Path) -> bool:
    return path.suffix.lower() == ".nc" or path.name.lower().endswith(".out.nc")


def _structured_column(table: Any, name: str) -> np.ndarray | None:
    names = set(table.dtype.names or ())
    if name not in names:
        return None
    return np.atleast_1d(np.asarray(table[name], dtype=float))


def _structured_species_matrix(table: Any, stem: str) -> np.ndarray | None:
    """Collect ``{stem}_s0, {stem}_s1, ...`` columns into ``(time, species)``."""

    names = tuple(table.dtype.names or ())
    indexed: list[tuple[int, str]] = []
    for name in names:
        prefix = f"{stem}_s"
        if name.startswith(prefix) and name[len(prefix) :].isdigit():
            indexed.append((int(name[len(prefix) :]), name))
    if not indexed:
        return None
    cols = [
        np.atleast_1d(np.asarray(table[name], dtype=float))
        for _idx, name in sorted(indexed)
    ]
    return np.stack(cols, axis=1)


def _diagnostics_from_csv_sidecar(base: Path):
    """Rebuild a ``SimulationDiagnostics`` from the CSV diagnostics sidecar."""

    from gkx.diagnostics import SimulationDiagnostics

    csv_path = Path(f"{base}.diagnostics.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"no diagnostics sidecar found at {csv_path}")
    table = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    t = _structured_column(table, "t")
    if t is None:
        raise ValueError(f"{csv_path} has no 't' column")
    zeros = np.zeros_like(t)

    def col(name: str) -> np.ndarray:
        found = _structured_column(table, name)
        return zeros if found is None else found

    dt_t = col("dt")
    return SimulationDiagnostics(
        t=t,
        dt_t=dt_t,
        dt_mean=np.asarray(np.mean(dt_t[dt_t > 0.0]) if np.any(dt_t > 0.0) else 0.0),
        gamma_t=col("gamma"),
        omega_t=col("omega"),
        Wg_t=col("Wg"),
        Wphi_t=col("Wphi"),
        Wapar_t=col("Wapar"),
        heat_flux_t=col("heat_flux"),
        particle_flux_t=col("particle_flux"),
        energy_t=col("energy"),
        heat_flux_species_t=_structured_species_matrix(table, "heat_flux"),
        particle_flux_species_t=_structured_species_matrix(table, "particle_flux"),
        turbulent_heating_t=_structured_column(table, "turbulent_heating"),
        turbulent_heating_species_t=_structured_species_matrix(
            table, "turbulent_heating"
        ),
        resolved=None,
    )


def _netcdf_wavenumbers(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read the dealiased ``ky``/``kx`` axes from a NetCDF output bundle."""

    import netCDF4

    ky = kx = None
    with netCDF4.Dataset(path) as root:
        grids = root.groups.get("Grids")
        if grids is not None:
            if "ky" in grids.variables:
                ky = np.asarray(grids.variables["ky"][:], dtype=float)
            if "kx" in grids.variables:
                kx = np.asarray(grids.variables["kx"][:], dtype=float)
    return ky, kx


def _coerce_nonlinear_source(
    source: Any,
) -> tuple[Any, np.ndarray | None, np.ndarray | None, str]:
    """Return ``(diagnostics, ky, kx, source_kind)`` for any accepted source."""

    if isinstance(source, (str, Path)):
        path = Path(source)
        if _is_netcdf_bundle_path(path):
            from gkx.artifacts.io import load_nonlinear_netcdf_diagnostics

            ky, kx = _netcdf_wavenumbers(path)
            return load_nonlinear_netcdf_diagnostics(path), ky, kx, "netcdf"
        base = _artifact_base(path)
        return _diagnostics_from_csv_sidecar(base), None, None, "csv"
    return source, None, None, "memory"


def _time_window(
    t: np.ndarray, window: tuple[float, float] | None
) -> tuple[np.ndarray, float, float]:
    """Return ``(mask, tmin, tmax)``; default window is the second half."""

    t_arr = np.asarray(t, dtype=float)
    if t_arr.size == 0:
        raise ValueError("diagnostics carry no time samples")
    if window is None:
        tmin = float(t_arr[0] + 0.5 * (t_arr[-1] - t_arr[0]))
        tmax = float(t_arr[-1])
    else:
        tmin, tmax = float(window[0]), float(window[1])
    mask = (t_arr >= tmin) & (t_arr <= tmax)
    if not np.any(mask):
        raise ValueError(
            f"averaging window [{tmin}, {tmax}] contains no samples "
            f"(t spans [{t_arr[0]:.4g}, {t_arr[-1]:.4g}])"
        )
    return mask, tmin, tmax


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sem = (
        float(np.std(arr, ddof=1) / np.sqrt(arr.size))
        if arr.size >= 2
        else float("nan")
    )
    return mean, sem


def _species_series(diag: Any, attr: str) -> np.ndarray | None:
    """Per-species ``(time, species)`` history, only when >= 2 species exist."""

    values = getattr(diag, attr, None)
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr


def _species_label(species_labels: Any, index: int) -> str:
    if species_labels is not None and index < len(species_labels):
        return str(species_labels[index])
    return f"species {index}"


def heat_flux_time_figure(
    source: Any,
    *,
    window: tuple[float, float] | None = None,
    species_labels: list[str] | None = None,
    title: str = "Turbulent fluxes",
    out: str | Path | None = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Stacked ``Q(t)`` and ``Gamma(t)`` traces with an optional averaging window.

    ``source`` is a ``SimulationDiagnostics``, a NetCDF output bundle path
    (``*.out.nc``), or a CSV diagnostics sidecar path. Per-species traces are
    drawn when the diagnostics resolve two or more species. When ``window``
    (``(tmin, tmax)``) is given, the window is shaded and the windowed
    mean +/- SEM of the total flux is annotated on each panel; SEM here treats
    samples as independent, so quote a proper correlation-time analysis for
    numbers that leave the figure. ``out`` optionally saves the figure (which
    stays open) via :func:`gkx.artifacts.figure_style.save_figure`.
    """

    diag, _ky, _kx, _kind = _coerce_nonlinear_source(source)
    t = np.asarray(diag.t, dtype=float)
    heat = np.asarray(diag.heat_flux_t, dtype=float)
    particle = np.asarray(diag.particle_flux_t, dtype=float)
    heat_species = _species_series(diag, "heat_flux_species_t")
    particle_species = _species_series(diag, "particle_flux_species_t")

    with figure_style():
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.8, 5.8))
        ax_q, ax_g = axes

        for ax, total, species, total_label, y_label, color in (
            (ax_q, heat, heat_species, "total", _HEAT_FLUX_LABEL, GKX_COLORS["blue"]),
            (
                ax_g,
                particle,
                particle_species,
                "total",
                _PARTICLE_FLUX_LABEL,
                GKX_COLORS["vermillion"],
            ),
        ):
            if species is not None:
                for index in range(species.shape[1]):
                    ax.plot(
                        t,
                        species[:, index],
                        linewidth=1.3,
                        alpha=0.85,
                        color=SERIES[(index + 2) % len(SERIES)],
                        label=_species_label(species_labels, index),
                    )
            ax.plot(t, total, color=color, linewidth=2.1, label=total_label)
            ax.set_ylabel(y_label)
            if species is not None:
                ax.legend(loc="best", ncols=2)

        if window is not None:
            mask, tmin, tmax = _time_window(t, window)
            q_mean, q_sem = _mean_sem(heat[mask])
            g_mean, g_sem = _mean_sem(particle[mask])
            for ax in axes:
                ax.axvspan(tmin, tmax, color=GKX_COLORS["grey"], alpha=0.16, zorder=0)
            # Lower right stays clear for a saturating flux trace: the linear
            # phase hugs the lower left and the saturated level fills the top.
            annotate_reference(
                ax_q,
                rf"$\langle Q \rangle = {q_mean:.3g} \pm {q_sem:.2g}$"
                rf"  over $t \in [{tmin:.4g}, {tmax:.4g}]$",
                loc="lower right",
            )
            annotate_reference(
                ax_g,
                rf"$\langle \Gamma \rangle = {g_mean:.3g} \pm {g_sem:.2g}$"
                rf"  over $t \in [{tmin:.4g}, {tmax:.4g}]$",
                loc="lower right",
            )

        ax_q.set_title(title)
        ax_g.set_xlabel(_TIME_LABEL)
        fig.tight_layout()
        if out is not None:
            save_figure(fig, out, close=False)
    return fig, axes


def _resolved_spectra_or_error(diag: Any, *, name: str, kind: str) -> Any:
    if kind == "csv":
        raise ValueError(_CSV_HAS_NO_SPECTRA.format(name=name))
    resolved = getattr(diag, "resolved", None)
    if resolved is None:
        raise ValueError(
            f"{name} needs resolved spectra but the diagnostics carry none; "
            "use a NetCDF output bundle (*.out.nc) written with "
            "[output] resolved_diagnostics = true"
        )
    return resolved


def _spectral_axis_or_error(
    axis: np.ndarray | None, *, length: int, axis_name: str, name: str
) -> np.ndarray:
    if axis is None:
        raise ValueError(
            f"{name}: pass {axis_name}= explicitly when the source is not a "
            f"NetCDF output bundle carrying its Grids/{axis_name} axis"
        )
    arr = np.asarray(axis, dtype=float)
    if arr.size != length:
        raise ValueError(
            f"{name}: {axis_name} axis has {arr.size} points but the spectra "
            f"resolve {length}"
        )
    return arr


def _window_average(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(values, dtype=float)[mask], axis=0)


def _annotate_with_headroom(
    ax: plt.Axes, text: str, *, loc: str = "upper left", fraction: float = 0.22
) -> None:
    """Annotate a corner after clearing a strip for the box to sit in.

    :func:`annotate_reference` draws an opaque box *inside* the axes, so it
    lands on top of the data whenever the curve happens to run through that
    corner -- and a spectrum peaks exactly where a fixed corner choice puts it
    (``Q(k_y)`` rises into "upper right", ``Phi^2(k_y)`` dips into "lower
    left"; both collided on the real Cyclone output). Expanding the limit on
    the annotated side first guarantees the strip is empty whatever the data
    does, rather than relying on a corner that only looks free for one run.
    """

    lo, hi = ax.get_ylim()
    upper = loc.startswith("upper")
    if ax.get_yscale() == "log":
        if lo > 0.0 and hi > lo:
            span = np.log10(hi) - np.log10(lo)
            if upper:
                ax.set_ylim(lo, 10.0 ** (np.log10(hi) + fraction * span))
            else:
                ax.set_ylim(10.0 ** (np.log10(lo) - fraction * span), hi)
    else:
        span = hi - lo
        if span > 0.0:
            if upper:
                ax.set_ylim(lo, hi + fraction * span)
            else:
                ax.set_ylim(lo - fraction * span, hi)
    annotate_reference(ax, text, loc=loc)


def _label_log_ky_axis(ax: plt.Axes) -> None:
    """Put readable plain-number labels on a sub-decade log ``ky`` axis.

    A dealiased ``ky`` range frequently spans less than one decade (0.035 to
    0.355 for a 32-point box), where the default log locator prints the single
    label ``10^-1`` and the reader cannot tell which ``ky`` the spectrum peaks
    at. Labelling a couple of minor ticks per decade in plain notation costs
    nothing on a wide-range production run and rescues the narrow one.
    """

    if ax.get_xscale() != "log":
        return
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0), numticks=12))
    # One formatter shape for both tick levels: two independent ScalarFormatters
    # pick their precision separately and print "0.05, 0.100, 0.20" down a
    # single axis.
    plain = FuncFormatter(lambda value, _pos: f"{value:g}")
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(plain)


def _plot_ky_spectrum(
    ax: plt.Axes,
    ky: np.ndarray,
    per_species: np.ndarray,
    *,
    species_labels: list[str] | None,
    y_label: str,
) -> None:
    """Plot a per-species ``(species, ky)`` spectrum against positive ``ky``."""

    positive = ky > 0.0
    ky_plot = ky[positive]
    total = per_species.sum(axis=0)
    if per_species.shape[0] >= 2:
        for index in range(per_species.shape[0]):
            ax.plot(
                ky_plot,
                per_species[index][positive],
                linewidth=1.3,
                alpha=0.85,
                color=SERIES[(index + 2) % len(SERIES)],
                label=_species_label(species_labels, index),
            )
        ax.plot(ky_plot, total[positive], color=GKX_COLORS["blue"], label="total")
        ax.legend(loc="best")
    else:
        ax.plot(ky_plot, total[positive], color=GKX_COLORS["blue"])
    if ky_plot.size >= 2:
        ax.set_xscale("log")
        _label_log_ky_axis(ax)
    ax.axhline(0.0, color=GKX_COLORS["grey"], linewidth=0.8, alpha=0.6)
    ax.set_xlabel(_KY_LABEL)
    ax.set_ylabel(y_label)


def _plot_kx_spectrum(
    ax: plt.Axes,
    kx: np.ndarray,
    per_species: np.ndarray,
    *,
    species_labels: list[str] | None,
    y_label: str,
) -> None:
    """Plot a per-species ``(species, kx)`` spectrum on the signed ``kx`` axis."""

    order = np.argsort(kx)
    kx_plot = kx[order]
    total = per_species.sum(axis=0)
    if per_species.shape[0] >= 2:
        for index in range(per_species.shape[0]):
            ax.plot(
                kx_plot,
                per_species[index][order],
                linewidth=1.3,
                alpha=0.85,
                color=SERIES[(index + 2) % len(SERIES)],
                label=_species_label(species_labels, index),
            )
        ax.plot(kx_plot, total[order], color=GKX_COLORS["blue"], label="total")
        ax.legend(loc="best")
    else:
        ax.plot(kx_plot, total[order], color=GKX_COLORS["blue"])
    ax.axhline(0.0, color=GKX_COLORS["grey"], linewidth=0.8, alpha=0.6)
    ax.axvline(0.0, color=GKX_COLORS["grey"], linewidth=0.8, alpha=0.6)
    ax.set_xlabel(_KX_LABEL)
    ax.set_ylabel(y_label)


def flux_spectra_figure(
    source: Any,
    *,
    ky: np.ndarray | None = None,
    kx: np.ndarray | None = None,
    window: tuple[float, float] | None = None,
    species_labels: list[str] | None = None,
    title: str = "Heat-flux spectra",
    out: str | Path | None = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Time-averaged ``Q(ky)`` and ``Q(kx)`` from resolved flux spectra.

    Reads ``HeatFlux_kyst``/``HeatFlux_kxst`` from a NetCDF output bundle
    (``*.out.nc``) or an in-memory ``SimulationDiagnostics`` with resolved
    spectra (pass ``ky=``/``kx=`` for the latter). The average is taken over
    ``window`` (default: the second half of the run). The zonal ``ky = 0``
    channel carries no flux and is omitted from the logarithmic ``ky`` axis.
    Raises ``ValueError`` for CSV sidecar sources, which carry no spectra.
    """

    diag, ky_file, kx_file, kind = _coerce_nonlinear_source(source)
    resolved = _resolved_spectra_or_error(diag, name="flux_spectra_figure", kind=kind)
    heat_ky = getattr(resolved, "HeatFlux_kyst", None)
    heat_kx = getattr(resolved, "HeatFlux_kxst", None)
    if heat_ky is None and heat_kx is None:
        raise ValueError(
            "flux_spectra_figure: the source has resolved diagnostics but no "
            "HeatFlux_kyst/HeatFlux_kxst spectra; write the NetCDF output "
            "bundle (*.out.nc) with [output] resolved_diagnostics = true"
        )
    mask, tmin, tmax = _time_window(np.asarray(diag.t, dtype=float), window)

    panels: list[tuple[str, np.ndarray, np.ndarray]] = []
    if heat_ky is not None:
        avg = _window_average(heat_ky, mask)  # (species, ky)
        axis = _spectral_axis_or_error(
            ky if ky is not None else ky_file,
            length=avg.shape[-1],
            axis_name="ky",
            name="flux_spectra_figure",
        )
        panels.append(("ky", axis, avg))
    if heat_kx is not None:
        avg = _window_average(heat_kx, mask)  # (species, kx)
        axis = _spectral_axis_or_error(
            kx if kx is not None else kx_file,
            length=avg.shape[-1],
            axis_name="kx",
            name="flux_spectra_figure",
        )
        panels.append(("kx", axis, avg))

    with figure_style():
        fig, axes = plt.subplots(
            1, len(panels), figsize=(5.6 * len(panels), 4.4), squeeze=False
        )
        axes = axes[0]
        for index, (which, axis, avg) in enumerate(panels):
            ax = axes[index]
            if which == "ky":
                _plot_ky_spectrum(
                    ax,
                    axis,
                    avg,
                    species_labels=species_labels,
                    y_label=r"$Q(k_y)/Q_{\mathrm{gB}}$",
                )
            else:
                _plot_kx_spectrum(
                    ax,
                    axis,
                    avg,
                    species_labels=species_labels,
                    y_label=r"$Q(k_x)/Q_{\mathrm{gB}}$",
                )
            if len(panels) > 1:
                panel_label(ax, f"({'ab'[index]})")
        _annotate_with_headroom(
            axes[0],
            rf"time average over $t \in [{tmin:.4g}, {tmax:.4g}]$",
            loc="upper left",
        )
        fig.suptitle(title)
        fig.tight_layout()
        if out is not None:
            save_figure(fig, out, close=False)
    return fig, axes


def _zonal_split(
    resolved: Any, ky: np.ndarray | None, phi2_kxky: np.ndarray | None
) -> np.ndarray | None:
    """Zonal ``Phi^2(t)`` from the dedicated trace or the 2D spectrum."""

    zonal = getattr(resolved, "Phi2_zonal_t", None)
    if zonal is not None:
        return np.asarray(zonal, dtype=float)
    if phi2_kxky is not None and ky is not None and ky.size and ky[0] == 0.0:
        return np.asarray(phi2_kxky, dtype=float)[:, 0, :].sum(axis=-1)
    return None


def phi2_spectra_figure(
    source: Any,
    *,
    ky: np.ndarray | None = None,
    kx: np.ndarray | None = None,
    window: tuple[float, float] | None = None,
    title: str = "Potential spectra",
    out: str | Path | None = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Four-panel ``Phi^2`` spectra summary from resolved diagnostics.

    Panels: (a) time-averaged ``Phi^2(ky)``, (b) time-averaged ``Phi^2(kx)``,
    (c) the time-averaged ``Phi^2(kx, ky)`` heatmap, and (d) the zonal
    (``ky = 0``) versus nonzonal split of ``Phi^2`` over time. Sources follow
    :func:`flux_spectra_figure`; CSV sidecars raise ``ValueError`` because
    only the NetCDF output bundle (``*.out.nc``) carries spectra.
    """

    diag, ky_file, kx_file, kind = _coerce_nonlinear_source(source)
    resolved = _resolved_spectra_or_error(diag, name="phi2_spectra_figure", kind=kind)
    phi2_kxky = getattr(resolved, "Phi2_kxkyt", None)
    phi2_ky = getattr(resolved, "Phi2_kyt", None)
    phi2_kx = getattr(resolved, "Phi2_kxt", None)
    if phi2_kxky is not None:
        phi2_kxky = np.asarray(phi2_kxky, dtype=float)  # (time, ky, kx)
        if phi2_ky is None:
            phi2_ky = phi2_kxky.sum(axis=2)
        if phi2_kx is None:
            phi2_kx = phi2_kxky.sum(axis=1)
    if phi2_ky is None or phi2_kx is None:
        raise ValueError(
            "phi2_spectra_figure needs Phi2_kxkyt (or Phi2_kyt and Phi2_kxt) "
            "resolved spectra, which only the NetCDF output bundle (*.out.nc) "
            "written with [output] resolved_diagnostics = true carries"
        )
    phi2_ky = np.asarray(phi2_ky, dtype=float)
    phi2_kx = np.asarray(phi2_kx, dtype=float)

    t = np.asarray(diag.t, dtype=float)
    mask, tmin, tmax = _time_window(t, window)
    ky_axis = _spectral_axis_or_error(
        ky if ky is not None else ky_file,
        length=phi2_ky.shape[-1],
        axis_name="ky",
        name="phi2_spectra_figure",
    )
    kx_axis = _spectral_axis_or_error(
        kx if kx is not None else kx_file,
        length=phi2_kx.shape[-1],
        axis_name="kx",
        name="phi2_spectra_figure",
    )

    ky_avg = _window_average(phi2_ky, mask)
    kx_avg = _window_average(phi2_kx, mask)
    total_t = phi2_ky.sum(axis=1)
    zonal_t = _zonal_split(resolved, ky_axis, phi2_kxky)

    with figure_style():
        fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6))
        ax_ky, ax_kx, ax_2d, ax_zonal = axes.ravel()

        # (a) Phi^2(ky), zonal ky = 0 lives in panel (d).
        positive = ky_axis > 0.0
        ax_ky.plot(ky_axis[positive], ky_avg[positive], color=GKX_COLORS["blue"])
        if np.count_nonzero(positive) >= 2:
            ax_ky.set_xscale("log")
            _label_log_ky_axis(ax_ky)
        if np.all(ky_avg[positive] > 0.0) and np.any(positive):
            ax_ky.set_yscale("log")
        ax_ky.set_xlabel(_KY_LABEL)
        ax_ky.set_ylabel(r"$\Phi^2(k_y)$")
        _annotate_with_headroom(
            ax_ky,
            rf"time average over $t \in [{tmin:.4g}, {tmax:.4g}]$",
            loc="upper left",
        )

        # (b) Phi^2(kx) on the signed kx axis.
        order = np.argsort(kx_axis)
        ax_kx.plot(kx_axis[order], kx_avg[order], color=GKX_COLORS["blue"])
        if np.all(kx_avg > 0.0):
            ax_kx.set_yscale("log")
        ax_kx.axvline(0.0, color=GKX_COLORS["grey"], linewidth=0.8, alpha=0.6)
        ax_kx.set_xlabel(_KX_LABEL)
        ax_kx.set_ylabel(r"$\Phi^2(k_x)$")

        # (c) Phi^2(kx, ky) heatmap, log color scale when the data allow it.
        if phi2_kxky is not None:
            from matplotlib.colors import LogNorm

            avg_2d = _window_average(phi2_kxky, mask)[:, order]  # (ky, kx)
            positive_2d = avg_2d[avg_2d > 0.0]
            norm = None
            if positive_2d.size:
                norm = LogNorm(
                    vmin=float(positive_2d.min()), vmax=float(positive_2d.max())
                )
                avg_2d = np.where(avg_2d > 0.0, avg_2d, float(positive_2d.min()))
            mesh = ax_2d.pcolormesh(
                kx_axis[order], ky_axis, avg_2d, cmap="magma", norm=norm, shading="auto"
            )
            ax_2d.grid(False)
            bar = fig.colorbar(mesh, ax=ax_2d, fraction=0.046, pad=0.03)
            bar.set_label(r"$\Phi^2(k_x, k_y)$")
            ax_2d.set_xlabel(_KX_LABEL)
            ax_2d.set_ylabel(_KY_LABEL)
        else:
            ax_2d.text(
                0.5,
                0.5,
                "no Phi2_kxkyt in source",
                transform=ax_2d.transAxes,
                ha="center",
                va="center",
                color="#555555",
            )
            ax_2d.set_xticks([])
            ax_2d.set_yticks([])

        # (d) zonal vs nonzonal split over time.
        ax_zonal.plot(t, total_t, color=GKX_COLORS["black"], label="total")
        if zonal_t is not None:
            # Dashed, so the nonzonal trace stays readable where it coincides
            # with the total (whenever the zonal share is negligible).
            nonzonal_t = np.clip(total_t - zonal_t, 0.0, None)
            ax_zonal.plot(
                t, zonal_t, color=GKX_COLORS["green"], label=r"zonal ($k_y=0$)"
            )
            ax_zonal.plot(
                t,
                nonzonal_t,
                color=GKX_COLORS["orange"],
                linestyle="--",
                label=r"nonzonal ($k_y\neq 0$)",
            )
        finite_total = total_t[np.isfinite(total_t)]
        if finite_total.size and np.all(finite_total > 0.0):
            ax_zonal.set_yscale("log")
        ax_zonal.axvspan(tmin, tmax, color=GKX_COLORS["grey"], alpha=0.16, zorder=0)
        ax_zonal.set_xlabel(_TIME_LABEL)
        ax_zonal.set_ylabel(r"$\Phi^2$")
        ax_zonal.legend(loc="best")

        for index, ax in enumerate((ax_ky, ax_kx, ax_2d, ax_zonal)):
            panel_label(ax, f"({'abcd'[index]})")
        fig.suptitle(title)
        fig.tight_layout()
        if out is not None:
            save_figure(fig, out, close=False)
    return fig, axes


__all__ += [
    "flux_spectra_figure",
    "heat_flux_time_figure",
    "phi2_spectra_figure",
]
