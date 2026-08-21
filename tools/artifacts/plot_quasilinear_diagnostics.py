#!/usr/bin/env python3
"""Build quasilinear diagnostic panels and validation plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gkx.artifacts.plotting import set_plot_style  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPECTRUM_OUT = ROOT / "docs" / "_static" / "quasilinear_cyclone_spectrum.png"
DEFAULT_UQ_INPUTS = [
    ROOT / "docs" / "_static" / "quasilinear_uq_ensemble_scaling_cpu_large.json",
    ROOT / "docs" / "_static" / "quasilinear_uq_ensemble_scaling_gpu_large.json",
]
DEFAULT_UQ_PREFIX = ROOT / "docs" / "_static" / "quasilinear_uq_ensemble_scaling_large"


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


# -----------------------------------------------------------------------------
# Quasilinear ky spectrum panel


def load_quasilinear_spectrum(path: str | Path) -> dict[str, np.ndarray]:
    """Load a quasilinear spectrum CSV written by the runtime scan path."""

    arr = np.genfromtxt(path, delimiter=",", names=True)
    if arr.shape == ():
        arr = np.asarray([arr], dtype=arr.dtype)
    required = {
        "ky",
        "gamma",
        "omega",
        "kperp_eff2",
        "heat_flux_weight_total",
        "particle_flux_weight_total",
        "amplitude2",
        "saturated_heat_flux_total",
    }
    missing = sorted(required - set(arr.dtype.names or ()))
    if missing:
        raise ValueError(f"missing quasilinear spectrum column(s): {missing}")
    return {name: np.asarray(arr[name], dtype=float) for name in arr.dtype.names or ()}


def plot_quasilinear_spectrum(
    spectrum: dict[str, np.ndarray],
    *,
    title: str = "Quasilinear Transport Spectrum",
) -> tuple[plt.Figure, np.ndarray]:
    """Create a publication-style quasilinear spectrum panel."""

    set_plot_style()
    ky = np.asarray(spectrum["ky"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    ax00, ax01, ax10, ax11 = axes.ravel()

    gamma_line = ax00.plot(
        ky,
        spectrum["gamma"],
        marker="o",
        linewidth=2.2,
        color="#0f4c81",
        label=r"$\gamma$",
    )
    ax00.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
    ax00.set_xlabel(r"$k_y \rho_i$")
    ax00.set_ylabel(r"$\gamma$", color="#0f4c81")
    ax00.tick_params(axis="y", labelcolor="#0f4c81")
    ax00.set_title("Linear mode")
    ax00r = ax00.twinx()
    omega_line = ax00r.plot(
        ky,
        spectrum["omega"],
        marker="s",
        linewidth=2.0,
        color="#c44e52",
        label=r"$\omega$",
    )
    ax00r.set_ylabel(r"$\omega$", color="#c44e52")
    ax00r.tick_params(axis="y", labelcolor="#c44e52")
    lines = gamma_line + omega_line
    ax00.legend(
        lines, [line.get_label() for line in lines], frameon=False, loc="upper left"
    )

    ax01.plot(ky, spectrum["kperp_eff2"], marker="o", linewidth=2.2, color="#2a9d8f")
    ax01.set_xlabel(r"$k_y \rho_i$")
    ax01.set_ylabel(r"$k_{\perp,\mathrm{eff}}^2$")
    ax01.set_title("Eigenfunction-weighted scale")

    ax10.plot(
        ky,
        spectrum["heat_flux_weight_total"],
        marker="o",
        linewidth=2.2,
        color="#0f4c81",
        label="heat",
    )
    ax10.plot(
        ky,
        spectrum["particle_flux_weight_total"],
        marker="s",
        linewidth=2.0,
        color="#f4a261",
        label="particle",
    )
    ax10.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
    ax10.set_xlabel(r"$k_y \rho_i$")
    ax10.set_ylabel("linear flux weight")
    ax10.set_title("Amplitude-normalized weights")
    ax10.legend(frameon=False)

    sat_heat = np.asarray(spectrum["saturated_heat_flux_total"], dtype=float)
    amp2 = np.asarray(spectrum["amplitude2"], dtype=float)
    heat_lines = []
    if np.isfinite(sat_heat).any():
        heat_lines = ax11.plot(
            ky,
            sat_heat,
            marker="o",
            linewidth=2.2,
            color="#c44e52",
            label="heat estimate",
        )
    ax11.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
    ax11.set_xlabel(r"$k_y \rho_i$")
    ax11.set_ylabel("heat estimate", color="#c44e52")
    ax11.tick_params(axis="y", labelcolor="#c44e52")
    ax11.set_title("Uncalibrated rule output")
    amp_lines = []
    if np.isfinite(amp2).any():
        ax11r = ax11.twinx()
        amp_lines = ax11r.plot(
            ky,
            amp2,
            marker="^",
            linewidth=1.8,
            color="#6c757d",
            linestyle="--",
            label=r"$A^2$",
        )
        ax11r.set_ylabel(r"$A^2$", color="#6c757d")
        ax11r.tick_params(axis="y", labelcolor="#6c757d")
    lines = heat_lines + amp_lines
    if lines:
        ax11.legend(
            lines,
            [line.get_label() for line in lines],
            frameon=False,
            loc="upper right",
        )

    fig.suptitle(title, fontsize=14)
    return fig, axes


def write_quasilinear_spectrum_figure(
    spectrum_csv: str | Path,
    *,
    out: str | Path = DEFAULT_SPECTRUM_OUT,
    title: str = "Quasilinear Transport Spectrum",
) -> dict[str, str]:
    """Load a spectrum CSV and write PNG/PDF/JSON plot artifacts."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spectrum = load_quasilinear_spectrum(spectrum_csv)
    fig, _axes = plot_quasilinear_spectrum(spectrum, title=title)
    fig.savefig(out_path, dpi=220)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    meta = {
        "kind": "quasilinear_spectrum_figure",
        "source": str(spectrum_csv),
        "png": str(out_path),
        "pdf": str(pdf_path),
        "n_ky": int(np.asarray(spectrum["ky"]).size),
        "ky_min": float(np.min(spectrum["ky"])),
        "ky_max": float(np.max(spectrum["ky"])),
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(_json_clean(meta), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(json_path)}


# -----------------------------------------------------------------------------
# Quasilinear/nonlinear spectrum-shape gate


def _netcdf_variable(root: Any, path: str) -> Any:
    current = root
    parts = [part for part in path.strip("/").split("/") if part]
    if not parts:
        raise ValueError("NetCDF variable path must not be empty")
    for group in parts[:-1]:
        if group not in current.groups:
            raise KeyError(f"NetCDF group '{group}' not found in '{path}'")
        current = current.groups[group]
    name = parts[-1]
    if name not in current.variables:
        raise KeyError(f"NetCDF variable '{name}' not found in '{path}'")
    return current.variables[name]


def _l1_distribution(values: np.ndarray, *, floor: float) -> np.ndarray:
    clean = np.where(np.isfinite(values), np.abs(values), 0.0)
    total = float(np.sum(clean))
    if total <= floor:
        raise ValueError("spectrum has zero total amplitude after cleaning")
    return clean / total


def build_spectrum_shape_report(
    *,
    spectrum_csv: str | Path,
    nonlinear_netcdf: str | Path,
    ql_column: str = "heat_flux_weight_total",
    nonlinear_variable: str = "Diagnostics/HeatFlux_kyst",
    time_min: float | None = None,
    time_max: float | None = None,
    species_index: int = 0,
    tv_gate: float = 0.2,
    cosine_gate: float = 0.95,
    floor: float = 1.0e-300,
) -> dict[str, Any]:
    """Build a normalized quasilinear-vs-nonlinear spectrum-shape report."""

    try:
        import netCDF4
    except ImportError as exc:  # pragma: no cover - optional dependency path.
        raise RuntimeError(
            "netCDF4 is required to read resolved nonlinear spectra"
        ) from exc

    spectrum_path = Path(spectrum_csv)
    nonlinear_path = Path(nonlinear_netcdf)
    ql_data = np.genfromtxt(spectrum_path, delimiter=",", names=True)
    if ql_data.shape == ():
        ql_data = np.asarray([ql_data], dtype=ql_data.dtype)
    names = set(ql_data.dtype.names or ())
    if "ky" not in names:
        raise ValueError(f"{spectrum_path} is missing required column 'ky'")
    if ql_column not in names:
        raise ValueError(f"{spectrum_path} is missing quasilinear column '{ql_column}'")
    ql_ky = np.asarray(ql_data["ky"], dtype=float)
    ql_values = np.asarray(ql_data[ql_column], dtype=float)
    finite = np.isfinite(ql_ky) & np.isfinite(ql_values)
    if not np.any(finite):
        raise ValueError(f"{spectrum_path} contains no finite quasilinear samples")
    ql_ky = ql_ky[finite]
    ql_values = ql_values[finite]

    with netCDF4.Dataset(nonlinear_path) as root:
        ky = np.asarray(_netcdf_variable(root, "Grids/ky")[:], dtype=float)
        time = np.asarray(_netcdf_variable(root, "Grids/time")[:], dtype=float)
        resolved = np.asarray(_netcdf_variable(root, nonlinear_variable)[:], dtype=float)

    if resolved.ndim != 3:
        raise ValueError(f"{nonlinear_variable} must have shape (time, species, ky)")
    if species_index < 0 or species_index >= resolved.shape[1]:
        raise ValueError(
            f"species_index {species_index} is out of bounds for {resolved.shape[1]} species"
        )
    if resolved.shape[0] != time.size or resolved.shape[2] != ky.size:
        raise ValueError(
            "resolved nonlinear spectrum dimensions do not match time/ky grids"
        )
    tmask = np.isfinite(time)
    if time_min is not None:
        tmask &= time >= float(time_min)
    if time_max is not None:
        tmask &= time <= float(time_max)
    if not np.any(tmask):
        raise ValueError("selected nonlinear time window contains no samples")
    nonlinear_spectrum = np.nanmean(np.abs(resolved[tmask, species_index, :]), axis=0)

    matched = []
    matched_indices = []
    for target in ql_ky:
        idx = int(np.argmin(np.abs(ky - target)))
        matched_indices.append(idx)
        matched.append(float(nonlinear_spectrum[idx]))
    matched_values = np.asarray(matched, dtype=float)
    ql_dist = _l1_distribution(ql_values, floor=floor)
    nl_dist = _l1_distribution(matched_values, floor=floor)
    residual = ql_dist - nl_dist
    tv_distance = float(0.5 * np.sum(np.abs(residual)))
    cosine = float(
        np.dot(ql_dist, nl_dist) / (np.linalg.norm(ql_dist) * np.linalg.norm(nl_dist))
    )
    passed = bool(tv_distance <= float(tv_gate) and cosine >= float(cosine_gate))
    return {
        "kind": "quasilinear_spectrum_shape_gate",
        "passed": passed,
        "spectrum_csv": str(spectrum_path),
        "nonlinear_netcdf": str(nonlinear_path),
        "ql_column": str(ql_column),
        "nonlinear_variable": str(nonlinear_variable),
        "time_min": None if time_min is None else float(time_min),
        "time_max": None if time_max is None else float(time_max),
        "time_samples": int(np.count_nonzero(tmask)),
        "species_index": int(species_index),
        "tv_gate": float(tv_gate),
        "cosine_gate": float(cosine_gate),
        "total_variation_distance": tv_distance,
        "cosine_similarity": cosine,
        "ky": ql_ky.tolist(),
        "nonlinear_ky": ky[np.asarray(matched_indices, dtype=int)].tolist(),
        "quasilinear_raw": ql_values.tolist(),
        "nonlinear_raw": matched_values.tolist(),
        "quasilinear_distribution": ql_dist.tolist(),
        "nonlinear_distribution": nl_dist.tolist(),
        "distribution_residual": residual.tolist(),
    }


def write_spectrum_shape_figure(
    report: dict[str, Any], *, out: str | Path, title: str
) -> dict[str, str]:
    """Write PNG/PDF/JSON artifacts for a spectrum-shape report."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ky = np.asarray(report["ky"], dtype=float)
    ql = np.asarray(report["quasilinear_distribution"], dtype=float)
    nl = np.asarray(report["nonlinear_distribution"], dtype=float)
    residual = np.asarray(report["distribution_residual"], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    ax0, ax1 = axes
    ax0.plot(
        ky, ql, marker="o", linewidth=2.0, label="quasilinear weight", color="#0f4c81"
    )
    ax0.plot(
        ky,
        nl,
        marker="s",
        linewidth=2.0,
        label="nonlinear resolved heat flux",
        color="#d1495b",
    )
    ax0.set_xlabel(r"$k_y\rho_i$")
    ax0.set_ylabel("normalized spectrum")
    ax0.set_title("Shape comparison")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.25)

    width = 0.75 * float(np.min(np.diff(ky))) if ky.size > 1 else 0.02
    ax1.bar(
        ky, residual, width=width, color=np.where(residual >= 0.0, "#2a9d8f", "#f97316")
    )
    ax1.axhline(0.0, color="0.2", linewidth=1.0)
    ax1.set_xlabel(r"$k_y\rho_i$")
    ax1.set_ylabel("QL - nonlinear")
    ax1.set_title("Distribution residual")
    ax1.text(
        0.03,
        0.95,
        f"passed = {report['passed']}\n"
        f"TV = {float(report['total_variation_distance']):.3f} <= {float(report['tv_gate']):.3f}\n"
        f"cosine = {float(report['cosine_similarity']):.3f} >= {float(report['cosine_gate']):.3f}",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.92},
    )
    ax1.grid(True, axis="y", alpha=0.25)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(json_path)}


# -----------------------------------------------------------------------------
# QL/UQ ensemble scaling


def _grid_label(payload: dict[str, Any]) -> str:
    grid = payload["grid"]
    return (
        f"Nx={int(grid['Nx'])}, Ny={int(grid['Ny'])}, Nz={int(grid['Nz'])}, "
        f"Nl={int(grid['Nl'])}, Nm={int(grid['Nm'])}"
    )


def load_summary(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        inputs.append(
            {
                "path": str(path),
                "backend": str(payload["backend"]),
                "grid": payload["grid"],
                "time": payload["time"],
                "identity_passed": bool(payload["identity_passed"]),
                "claim_scope": str(payload.get("claim_scope", "")),
            }
        )
        for row in payload["rows"]:
            item = dict(row)
            item["backend"] = str(payload["backend"])
            item["source"] = str(path)
            item["grid_label"] = _grid_label(payload)
            rows.append(item)
    return _json_clean(
        {
            "kind": "quasilinear_uq_ensemble_scaling_combined",
            "claim_scope": (
                "solver-backed quasilinear/UQ ensemble strong-scaling artifact for "
                "independent CPU processes and GPU workers; the observable is a "
                "reduced mixing-length feature from real late-time linear scans, "
                "not a promoted absolute nonlinear heat-flux predictor"
            ),
            "identity_passed": all(item["identity_passed"] for item in inputs),
            "inputs": inputs,
            "rows": rows,
        }
    )


def write_artifacts(summary: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = list(summary["rows"])
    fieldnames = [
        "backend",
        "requested_devices",
        "actual_workers",
        "timed_wall_s",
        "strong_speedup_vs_1_device",
        "parallel_efficiency",
        "ensemble_mean_heat_flux_proxy",
        "ensemble_std_heat_flux_proxy",
        "max_heat_flux_proxy_rel_error",
        "max_gamma_abs_error",
        "identity_gate_pass",
        "grid_label",
        "source",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0), constrained_layout=True)
    style = {
        "cpu": {"color": "#276b8e", "marker": "o", "label": "CPU workers"},
        "gpu": {"color": "#c45a14", "marker": "s", "label": "GPU workers"},
    }
    max_devices = max(int(row["requested_devices"]) for row in rows)
    for backend in sorted({str(row["backend"]) for row in rows}):
        subset = sorted(
            (row for row in rows if str(row["backend"]) == backend),
            key=lambda row: int(row["requested_devices"]),
        )
        x = np.asarray([int(row["requested_devices"]) for row in subset], dtype=float)
        speedup = np.asarray(
            [float(row["strong_speedup_vs_1_device"]) for row in subset], dtype=float
        )
        elapsed = np.asarray([float(row["timed_wall_s"]) for row in subset], dtype=float)
        heat_rel = np.asarray(
            [float(row["max_heat_flux_proxy_rel_error"]) for row in subset], dtype=float
        )
        gamma_abs = np.asarray(
            [float(row["max_gamma_abs_error"]) for row in subset], dtype=float
        )
        color = style.get(backend, {}).get("color")
        marker = style.get(backend, {}).get("marker", "o")
        label = style.get(backend, {}).get("label", backend)
        axes[0].plot(x, speedup, marker=marker, lw=2.2, color=color, label=label)
        axes[1].semilogy(
            x,
            np.maximum(elapsed, 1.0e-16),
            marker=marker,
            lw=2.2,
            color=color,
            label=label,
        )
        axes[2].semilogy(
            x,
            np.maximum(heat_rel, 1.0e-16),
            marker=marker,
            lw=2.0,
            color=color,
            label=label + " QL",
        )
        axes[2].semilogy(
            x + 0.04,
            np.maximum(gamma_abs, 1.0e-16),
            marker=marker,
            ls="--",
            lw=1.8,
            color=color,
            label=label + r" $\gamma$",
        )
    ideal = np.arange(1, max_devices + 1)
    axes[0].plot(ideal, ideal, ":", color="0.35", lw=1.4, label="ideal")
    axes[0].set_xlabel("workers/devices")
    axes[0].set_ylabel("speedup vs one worker")
    axes[0].set_title("Quasilinear/UQ ensemble scaling")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_xlabel("workers/devices")
    axes[1].set_ylabel("median ensemble time [s]")
    axes[1].set_title("Late-time linear solves")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].set_xlabel("workers/devices")
    axes[2].set_ylabel("identity error")
    axes[2].set_title("QL/growth identity")
    axes[2].legend(frameon=False, fontsize=7, ncol=2)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


# -----------------------------------------------------------------------------
# CLI


def _add_spectrum_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spectrum", required=True, help="Input *.quasilinear_spectrum.csv")
    parser.add_argument("--out", default=str(DEFAULT_SPECTRUM_OUT), help="Output PNG path")
    parser.add_argument("--title", default="Quasilinear Transport Spectrum")


def _add_shape_gate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spectrum", required=True, help="Input quasilinear spectrum CSV")
    parser.add_argument(
        "--nonlinear", required=True, help="Input nonlinear NetCDF with resolved diagnostics"
    )
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--ql-column", default="heat_flux_weight_total")
    parser.add_argument("--nonlinear-variable", default="Diagnostics/HeatFlux_kyst")
    parser.add_argument("--time-min", type=float, default=None)
    parser.add_argument("--time-max", type=float, default=None)
    parser.add_argument("--species-index", type=int, default=0)
    parser.add_argument("--tv-gate", type=float, default=0.2)
    parser.add_argument("--cosine-gate", type=float, default=0.95)
    parser.add_argument("--title", default="Quasilinear/nonlinear ky-spectrum shape gate")


def _add_uq_scaling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_UQ_INPUTS)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_UQ_PREFIX)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_spectrum_args(
        subparsers.add_parser("spectrum", help="Plot a quasilinear ky spectrum.")
    )
    _add_shape_gate_args(
        subparsers.add_parser(
            "shape-gate", help="Compare quasilinear and nonlinear ky-spectrum shapes."
        )
    )
    _add_uq_scaling_args(
        subparsers.add_parser(
            "uq-ensemble-scaling",
            help="Combine quasilinear/UQ ensemble CPU/GPU scaling artifacts.",
        )
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "spectrum":
        paths = write_quasilinear_spectrum_figure(
            args.spectrum, out=args.out, title=args.title
        )
        print(f"saved {paths['png']}")
        print(f"saved {paths['pdf']}")
        print(f"saved {paths['json']}")
        return 0
    if args.command == "shape-gate":
        report = build_spectrum_shape_report(
            spectrum_csv=args.spectrum,
            nonlinear_netcdf=args.nonlinear,
            ql_column=args.ql_column,
            nonlinear_variable=args.nonlinear_variable,
            time_min=args.time_min,
            time_max=args.time_max,
            species_index=args.species_index,
            tv_gate=args.tv_gate,
            cosine_gate=args.cosine_gate,
        )
        paths = write_spectrum_shape_figure(report, out=args.out, title=args.title)
        print(f"saved {paths['png']}")
        print(f"saved {paths['pdf']}")
        print(f"saved {paths['json']}")
        print(
            "passed={passed} total_variation_distance={tv:.6g} cosine_similarity={cos:.6g}".format(
                passed=report["passed"],
                tv=report["total_variation_distance"],
                cos=report["cosine_similarity"],
            )
        )
        return 0 if report["passed"] else 1
    if args.command == "uq-ensemble-scaling":
        summary = load_summary([Path(path) for path in args.inputs])
        paths = write_artifacts(summary, Path(args.out_prefix))
        print(
            json.dumps(
                {"identity_passed": summary["identity_passed"], "paths": paths},
                indent=2,
            )
        )
        return 0
    raise SystemExit(f"unknown quasilinear diagnostic command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
