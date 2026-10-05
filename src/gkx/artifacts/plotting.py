"""Publication-ready benchmark, diagnostic, runtime, and zonal plots."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np

from gkx.artifacts.figure_style import _artifact_base, set_plot_style
from gkx.benchmarking_shared import CycloneReference, CycloneScanResult
from gkx.diagnostics.growth_rates import fit_growth_rate


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
        ref.ky, ref.gamma, marker="o", color="#1f77b4", linewidth=2.0, label="Reference"
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
        ref.ky, ref.omega, marker="o", color="#ff7f0e", linewidth=2.0, label="Reference"
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
        ax0.plot(
            x_ref,
            gamma_ref,
            marker="o",
            linestyle="None",
            color="#1f77b4",
            label=ref_label,
        )
    ax0.set_ylabel(r"$\gamma a / v_{ti}$")
    ax0.set_title(title)
    ax0.legend(loc="best")
    if log_x:
        ax0.set_xscale("log")

    ax1.plot(x, omega, marker="o", color="#d62728", label=label)
    if x_ref is not None and omega_ref is not None:
        ax1.plot(
            x_ref,
            omega_ref,
            marker="o",
            linestyle="None",
            color="#1f77b4",
            label=ref_label,
        )
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(r"$\omega a / v_{ti}$")
    ax1.legend(loc="best")
    if log_x:
        ax1.set_xscale("log")

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
        ax0.plot(
            panel.z,
            panel.eigenfunction.imag,
            color="#ff7f0e",
            linestyle="--",
            label="Im",
        )
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
            ax1.plot(
                panel.x_ref,
                panel.gamma_ref,
                marker="o",
                linestyle="None",
                color="#1f77b4",
                label=panel.ref_label,
            )
        ax1.set_xlabel(panel.x_label)
        ax1.set_ylabel(r"$\gamma a / v_{ti}$")
        if panel.log_x:
            ax1.set_xscale("log")

        ax2.plot(panel.x, panel.omega, marker="o", color="#d62728", label="GKX")
        if panel.x_ref is not None and panel.omega_ref is not None:
            ax2.plot(
                panel.x_ref,
                panel.omega_ref,
                marker="o",
                linestyle="None",
                color="#1f77b4",
                label=panel.ref_label,
            )
        ax2.set_xlabel(panel.x_label)
        ax2.set_ylabel(r"$\omega a / v_{ti}$")
        if panel.log_x:
            ax2.set_xscale("log")
        if i == 0:
            ax1.legend(loc="best", fontsize=9)
            ax2.legend(loc="best", fontsize=9)

    fig.tight_layout()
    return fig, axes


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
            ax1.plot(fit_t, fit_line, color="red", linestyle="--", label="fit line")
    ax0.legend(loc="best", fontsize=9)
    ax1.legend(loc="best", fontsize=9)
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
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "#cccccc",
        },
    )

    eigen_norm = _normalize_by_real_max(eigenfunction)
    ax1.plot(
        np.asarray(z, dtype=float),
        np.real(eigen_norm),
        color="#0f4c81",
        linewidth=2.4,
        label="Re",
    )
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
        ax0.plot(
            t_arr,
            np.maximum(np.asarray(phi2, dtype=float), 1.0e-30),
            color="#0f4c81",
            linewidth=2.4,
        )
        ax0.set_yscale("log")
        ax0.set_ylabel(r"$|\phi|^2$")
        ax0.set_title("Field amplitude")
    elif wphi is not None:
        ax0.plot(t_arr, np.asarray(wphi, dtype=float), color="#0f4c81", linewidth=2.4)
        ax0.set_ylabel(r"$W_\phi$")
        ax0.set_title("Electrostatic energy")

    if wphi is not None:
        ax1.plot(
            t_arr,
            np.asarray(wphi, dtype=float),
            color="#2a9d8f",
            linewidth=2.4,
            label=r"$W_\phi$",
        )
    if gamma is not None:
        ax1.plot(
            t_arr,
            np.asarray(gamma, dtype=float),
            color="#f4a261",
            linewidth=2.0,
            linestyle="--",
            label=r"$\gamma$",
        )
    if omega is not None:
        ax1.plot(
            t_arr,
            np.asarray(omega, dtype=float),
            color="#c44e52",
            linewidth=2.0,
            linestyle=":",
            label=r"$\omega$",
        )
    ax1.set_xlabel("t")
    ax1.set_title("Resolved diagnostics")
    if wphi is not None or gamma is not None or omega is not None:
        ax1.legend(loc="best", frameon=False)

    if heat_flux is not None:
        ax2.plot(
            t_arr, np.asarray(heat_flux, dtype=float), color="#c44e52", linewidth=2.4
        )
    ax2.set_xlabel("t")
    ax2.set_ylabel("Heat flux")
    ax2.set_title("Transport")

    ax0.set_xlabel("t")
    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


def scan_result_figure(result: Any) -> Tuple[plt.Figure, np.ndarray]:
    """Standard figure for a linear ``k_y`` scan result."""

    return scan_comparison_figure(
        result.ky, result.gamma, result.omega, r"$k_y \rho_i$", "GKX linear scan"
    )


def linear_result_figure(result: Any) -> Tuple[plt.Figure, np.ndarray]:
    """Standard figure for a single linear runtime result."""

    if result.z is None or result.eigenfunction is None:
        raise ValueError("linear plotting requires z and eigenfunction arrays")
    if result.t is not None and result.signal is not None:
        return linear_runtime_panel_figure(
            t=result.t,
            signal=result.signal,
            z=result.z,
            eigenfunction=result.eigenfunction,
            gamma=result.gamma,
            omega=result.omega,
        )
    panel = LinearValidationPanel(
        name="GKX",
        z=result.z,
        eigenfunction=result.eigenfunction,
        x=np.asarray([result.ky]),
        gamma=np.asarray([result.gamma]),
        omega=np.asarray([result.omega]),
        x_label=r"$k_y \rho_i$",
    )
    return linear_validation_figure([panel])


def nonlinear_result_figure(result: Any) -> Tuple[plt.Figure, np.ndarray]:
    """Standard figure for a nonlinear runtime result with time diagnostics."""

    diagnostics = result.diagnostics
    if diagnostics is None:
        raise ValueError("nonlinear plotting requires retained time diagnostics")
    return nonlinear_runtime_panel_figure(
        t=np.asarray(diagnostics.t),
        wphi=np.asarray(diagnostics.Wphi_t),
        heat_flux=np.asarray(diagnostics.heat_flux_t),
        gamma=np.asarray(diagnostics.gamma_t),
        omega=np.asarray(diagnostics.omega_t),
    )


def _sidecar(base: Path, suffix: str) -> Path:
    """Return ``base`` + ``suffix``, the way the writers actually name sidecars.

    ``Path.with_suffix`` replaces the final suffix instead of appending, so a
    run whose output path is ``foo.out.nc`` -- the convention the shipped decks
    use -- had its sidecars written as ``foo.out.nc.summary.json`` and looked
    for as ``foo.out.summary.json``. Plotting such a run failed with "Could not
    infer runtime summary".
    """

    return base.with_name(base.name + suffix)


def _load_linear_bundle(
    base: Path,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    summary = json.loads(_sidecar(base, ".summary.json").read_text(encoding="utf-8"))
    timeseries = np.genfromtxt(
        _sidecar(base, ".timeseries.csv"), delimiter=",", names=True, dtype=float
    )
    eigen = np.genfromtxt(
        _sidecar(base, ".eigenfunction.csv"), delimiter=",", names=True, dtype=float
    )
    t = np.asarray(timeseries["t"], dtype=float)
    signal = np.asarray(timeseries["signal_real"], dtype=float) + 1j * np.asarray(
        timeseries["signal_imag"], dtype=float
    )
    z = np.asarray(eigen["z"], dtype=float)
    eig = np.asarray(eigen["eigen_real"], dtype=float) + 1j * np.asarray(
        eigen["eigen_imag"], dtype=float
    )
    return summary, t, signal, z, eig


def _load_linear_scan_bundle(
    base: Path,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    summary = json.loads(_sidecar(base, ".summary.json").read_text(encoding="utf-8"))
    scan = np.genfromtxt(
        _sidecar(base, ".scan.csv"), delimiter=",", names=True, dtype=float
    )
    ky = np.atleast_1d(np.asarray(scan["ky"], dtype=float))
    gamma = np.atleast_1d(np.asarray(scan["gamma"], dtype=float))
    omega = np.atleast_1d(np.asarray(scan["omega"], dtype=float))
    return summary, ky, gamma, omega


def _load_nonlinear_csv(
    base: Path,
) -> tuple[
    dict,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    summary = json.loads(_sidecar(base, ".summary.json").read_text(encoding="utf-8"))
    diag = np.genfromtxt(
        _sidecar(base, ".diagnostics.csv"), delimiter=",", names=True, dtype=float
    )
    names = set(diag.dtype.names or ())
    t = np.asarray(diag["t"], dtype=float)
    wphi = np.asarray(diag["Wphi"], dtype=float) if "Wphi" in names else None
    heat_flux = (
        np.asarray(diag["heat_flux"], dtype=float) if "heat_flux" in names else None
    )
    gamma = np.asarray(diag["gamma"], dtype=float) if "gamma" in names else None
    omega = np.asarray(diag["omega"], dtype=float) if "omega" in names else None
    return summary, t, wphi, heat_flux, gamma, omega


def _load_nonlinear_netcdf(
    path: Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    try:
        import netCDF4
    except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(
            "netCDF4 is required to plot *.out.nc runtime bundles"
        ) from exc

    with netCDF4.Dataset(path) as root:
        diag = root.groups["Diagnostics"]
        # The time axis lives in Grids/time, which is where this and the
        # comparison code both write it; Diagnostics/t was never written by
        # either, so --plot on a real *.out.nc used to die on a KeyError.
        grids = root.groups.get("Grids")
        time_var = None
        if grids is not None and "time" in grids.variables:
            time_var = grids.variables["time"]
        elif "t" in diag.variables:
            time_var = diag.variables["t"]
        if time_var is None:
            raise KeyError(f"{Path(path).name} carries no Grids/time axis")
        t = np.asarray(time_var[:], dtype=float)
        phi2 = (
            np.asarray(diag.variables["Phi2_t"][:], dtype=float)
            if "Phi2_t" in diag.variables
            else None
        )
        wphi = None
        heat_flux = None
        if "Wphi_st" in diag.variables:
            wphi = np.sum(np.asarray(diag.variables["Wphi_st"][:], dtype=float), axis=1)
        if "HeatFlux_st" in diag.variables:
            heat_flux = np.sum(
                np.asarray(diag.variables["HeatFlux_st"][:], dtype=float), axis=1
            )
    return t, phi2, wphi, heat_flux


def plot_saved_output(path: str | Path, *, out: str | Path | None = None) -> Path:
    """Plot a saved linear or nonlinear output bundle.

    A bundle written by another gyrokinetic code is accepted alongside GKX's
    own, so a cross-code comparison is one command rather than a script; see
    :mod:`gkx.artifacts.foreign_output` for how they are told apart.
    """

    in_path = Path(path)
    base = _artifact_base(in_path)
    out_path = Path(out) if out is not None else Path(f"{base}.plot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".nc" or in_path.name.lower().endswith(".out.nc"):
        from gkx.artifacts.foreign_output import foreign_output_plotter

        plot_foreign = foreign_output_plotter(in_path)
        if plot_foreign is not None:
            return plot_foreign(in_path, out=out_path)
        t, phi2, wphi, heat_flux = _load_nonlinear_netcdf(in_path)
        fig, _axes = nonlinear_runtime_panel_figure(
            t=t,
            phi2=phi2,
            wphi=wphi,
            heat_flux=heat_flux,
            title=f"GKX nonlinear runtime: {base.name}",
        )
    else:
        summary_path = _sidecar(base, ".summary.json")
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
        elif kind == "linear_scan":
            _summary, scan_ky, scan_gamma, scan_omega = _load_linear_scan_bundle(base)
            fig, _axes = scan_comparison_figure(
                scan_ky,
                scan_gamma,
                scan_omega,
                r"$k_y \rho_i$",
                f"GKX linear scan: {base.name}",
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


__all__ = [
    "LinearValidationPanel",
    "cyclone_comparison_figure",
    "cyclone_reference_figure",
    "growth_fit_figure",
    "linear_runtime_panel_figure",
    "linear_validation_figure",
    "nonlinear_runtime_panel_figure",
    "plot_saved_output",
    "scan_comparison_figure",
    "set_plot_style",
]
