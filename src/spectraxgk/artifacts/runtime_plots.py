"""Runtime-output plotting for saved linear and nonlinear artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from spectraxgk.artifacts.plot_style import set_plot_style

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
            raise ValueError(f"Unsupported saved-output kind: {kind!r}")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path



__all__ = [
    "linear_runtime_panel_figure",
    "nonlinear_runtime_panel_figure",
    "plot_saved_output",
]
