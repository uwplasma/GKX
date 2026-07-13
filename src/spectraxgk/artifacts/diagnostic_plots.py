"""Diagnostic plotting for fitted growth histories and eigenfunctions."""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from spectraxgk.artifacts.plotting import set_plot_style
from spectraxgk.diagnostics.growth_rates import fit_growth_rate

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

    from spectraxgk.diagnostics.modes import (
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
    ax0.plot(theta_arr, np.real(eig_aligned), color="#c44e52", linewidth=2.0, linestyle="--", label="SPECTRAX Re")
    ax0.set_xlabel(r"$\theta$")
    ax0.set_ylabel("real")
    ax0.set_title("Real part")
    ax0.legend(loc="best", frameon=False)

    ax1.plot(theta_ref_arr, np.imag(ref), color="#0f4c81", linewidth=2.4, label="Reference Im")
    ax1.plot(theta_arr, np.imag(eig_aligned), color="#c44e52", linewidth=2.0, linestyle="--", label="SPECTRAX Im")
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel("imag")
    ax1.set_title("Imaginary part")
    ax1.legend(loc="best", frameon=False)

    ax2.plot(theta_ref_arr, np.abs(ref), color="#0f4c81", linewidth=2.4, label="Reference $|\\phi|$")
    ax2.plot(theta_arr, np.abs(eig_aligned), color="#c44e52", linewidth=2.0, linestyle="--", label="SPECTRAX $|\\phi|$")
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



__all__ = [
    "eigenfunction_overlap_summary_figure",
    "eigenfunction_reference_overlay_figure",
    "growth_fit_figure",
]
