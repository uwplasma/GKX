"""Plotting utilities for publication-ready figures."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Tuple

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
    title: str = "SPECTRAX-GK Linear Runtime",
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
    title: str = "SPECTRAX-GK Nonlinear Runtime",
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
    """Plot a saved linear or nonlinear runtime artifact bundle."""

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
            title=f"SPECTRAX-GK nonlinear runtime: {base.name}",
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
                title=f"SPECTRAX-GK linear runtime: {base.name}",
            )
        elif kind == "nonlinear":
            _summary, t, wphi, heat_flux, gamma, omega = _load_nonlinear_csv(base)
            fig, _axes = nonlinear_runtime_panel_figure(
                t=t,
                wphi=wphi,
                heat_flux=heat_flux,
                gamma=gamma,
                omega=omega,
                title=f"SPECTRAX-GK nonlinear runtime: {base.name}",
            )
        else:
            raise ValueError(f"Unsupported runtime artifact kind: {kind!r}")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


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

        ax1.plot(panel.x, panel.gamma, marker="o", color="#2ca02c", label="SPECTRAX-GK")
        ax2.plot(panel.x, panel.omega, marker="o", color="#d62728", label="SPECTRAX-GK")
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
    ax0.plot(x, gamma, marker="o", color="#2ca02c", label="SPECTRAX-GK")
    ax1.plot(x, omega, marker="o", color="#d62728", label="SPECTRAX-GK")
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
        idx = int(np.searchsorted(t, tmin))
        log_ref = log_energy[idx] if idx < log_energy.size else log_energy[-1]
        fit_line = 2.0 * gamma * (t - tmin) + log_ref
        ax1.plot(t, fit_line, color="red", linestyle="--", label="fit line")
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

    from spectraxgk.benchmarking import compare_eigenfunctions, phase_align_eigenfunction

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


def zonal_flow_response_figure(
    t: np.ndarray,
    response: np.ndarray,
    *,
    metrics=None,
    title: str = "Zonal-flow response",
    y_label: str = "normalized response",
) -> Tuple[plt.Figure, np.ndarray]:
    """Render a zonal-flow response trace and its envelope summary."""

    from spectraxgk.benchmarking import zonal_flow_response_metrics

    set_plot_style()
    t_arr = np.asarray(t, dtype=float)
    resp = np.asarray(response, dtype=float)
    if t_arr.ndim != 1 or resp.ndim != 1 or t_arr.size != resp.size:
        raise ValueError("t and response must be one-dimensional arrays of equal length")
    if metrics is None:
        metrics = zonal_flow_response_metrics(t_arr, resp)

    response_norm = resp / float(metrics.initial_level)
    residual = float(metrics.residual_level)
    env_t = np.asarray(metrics.peak_times, dtype=float)
    env_y = np.asarray(metrics.peak_envelope, dtype=float)
    fit_count = int(getattr(metrics, "peak_fit_count", env_t.size))
    fit_tmin = float(getattr(metrics, "fit_tmin", t_arr[0]))
    fit_tmax = float(getattr(metrics, "fit_tmax", t_arr[-1]))
    damping_method = str(getattr(metrics, "damping_method", "combined_envelope"))
    frequency_method = str(getattr(metrics, "frequency_method", "peak_spacing"))
    max_peak_t = np.asarray(getattr(metrics, "max_peak_times", np.asarray([], dtype=float)), dtype=float)
    max_peak_y = np.asarray(getattr(metrics, "max_peak_values", np.asarray([], dtype=float)), dtype=float)
    min_peak_t = np.asarray(getattr(metrics, "min_peak_times", np.asarray([], dtype=float)), dtype=float)
    min_peak_y = np.asarray(getattr(metrics, "min_peak_values", np.asarray([], dtype=float)), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax0, ax1 = axes

    ax0.plot(t_arr, response_norm, color="#0f4c81", linewidth=2.2, label="response")
    ax0.axhline(residual, color="#c44e52", linestyle="--", linewidth=2.0, label="residual")
    ax0.axvspan(fit_tmin, fit_tmax, color="#d9ead3", alpha=0.22, linewidth=0.0)
    if damping_method == "branchwise_extrema":
        if max_peak_t.size:
            keep = (max_peak_t >= fit_tmin) & (max_peak_t <= fit_tmax)
            ax0.plot(max_peak_t[keep], max_peak_y[keep], linestyle="none", marker="o", color="#2a9d8f", markersize=5.2, label="maxima fit points")
        if min_peak_t.size:
            keep = (min_peak_t >= fit_tmin) & (min_peak_t <= fit_tmax)
            ax0.plot(min_peak_t[keep], min_peak_y[keep], linestyle="none", marker="o", color="#7b2cbf", markersize=5.2, label="minima fit points")
    ax0.fill_between(
        t_arr,
        residual - float(metrics.residual_std),
        residual + float(metrics.residual_std),
        color="#c44e52",
        alpha=0.15,
        linewidth=0.0,
    )
    ax0.set_xlabel("t")
    ax0.set_ylabel(y_label)
    ax0.set_title("Normalized response")
    ax0.legend(loc="best", frameon=False)

    ax1.plot(t_arr, np.maximum(np.abs(response_norm - residual), 1.0e-14), color="#4c956c", linewidth=2.0, alpha=0.5)
    if env_t.size:
        ax1.plot(env_t, env_y, color="#c44e52", marker="o", linewidth=1.8, label="envelope peaks")
    fit_env_t = env_t[(env_t >= fit_tmin) & (env_t <= fit_tmax)]
    fit_env_y = env_y[(env_t >= fit_tmin) & (env_t <= fit_tmax)]
    if damping_method == "combined_envelope" and fit_count >= 2 and np.isfinite(float(metrics.gam_damping_rate)) and fit_env_t.size:
        fit_n = min(fit_count, fit_env_t.size)
        fit_t = fit_env_t[:fit_n]
        fit = fit_env_y[0] * np.exp(-float(metrics.gam_damping_rate) * (fit_t - fit_t[0]))
        label = "envelope fit" if fit_n == fit_env_t.size else f"envelope fit (first {fit_n} peaks)"
        ax1.plot(fit_t, fit, color="#2a9d8f", linestyle="--", linewidth=2.0, label=label)
    ax1.set_yscale("log")
    ax1.set_xlabel("t")
    ax1.set_ylabel("envelope")
    ax1.set_title("GAM envelope")
    if env_t.size:
        ax1.legend(loc="best", frameon=False)
    ax1.text(
        0.03,
        0.97,
        (
            f"residual = {metrics.residual_level:.4f}\n"
            f"std = {metrics.residual_std:.4f}\n"
            f"ω_GAM = {metrics.gam_frequency:.4f}\n"
            f"γ_damp = {metrics.gam_damping_rate:.4f}\n"
            f"fit_peaks = {fit_count}\n"
            f"norm = {getattr(metrics, 'initial_policy', 'window_abs_mean')}\n"
            f"damp = {damping_method}\n"
            f"freq = {frequency_method}\n"
            f"fit_t = [{fit_tmin:.1f}, {fit_tmax:.1f}]"
        ),
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes
