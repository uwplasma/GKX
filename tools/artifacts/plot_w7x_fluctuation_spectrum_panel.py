#!/usr/bin/env python3
"""Build the W7-X nonlinear fluctuation-spectrum diagnostic panel."""

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


DEFAULT_NONLINEAR = Path(
    "tools_out/final_nonlinear_audit/w7x_gkx_current_adaptive_t200.out.nc"
)
DEFAULT_GATE_SUMMARY = Path("docs/_static/nonlinear_w7x_gate_summary.json")
DEFAULT_OUT = Path("docs/_static/w7x_fluctuation_spectrum_panel.png")


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_clean(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


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


def _finite_window(
    time: np.ndarray, *, time_min: float | None, time_max: float | None
) -> np.ndarray:
    mask = np.isfinite(time)
    if time_min is not None:
        mask &= time >= float(time_min)
    if time_max is not None:
        mask &= time <= float(time_max)
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            "selected W7-X fluctuation-spectrum window needs at least two finite samples"
        )
    return mask


def _l1_normalized(values: np.ndarray) -> np.ndarray:
    clean = np.where(np.isfinite(values), np.abs(values), 0.0)
    total = float(np.sum(clean))
    if total <= 0.0:
        return np.zeros_like(clean, dtype=float)
    return clean / total


def _dominant_index(grid: np.ndarray, values: np.ndarray, *, exclude_zero: bool) -> int:
    eligible = np.isfinite(grid) & np.isfinite(values)
    if exclude_zero:
        eligible &= np.abs(grid) > 1.0e-14
    if not np.any(eligible):
        raise ValueError("no eligible finite spectrum entries")
    masked = np.where(eligible, np.abs(values), -np.inf)
    return int(np.argmax(masked))


def _temporal_power_spectrum(
    time: np.ndarray, trace: np.ndarray, *, n_points: int = 256
) -> dict[str, Any]:
    finite = np.isfinite(time) & np.isfinite(trace)
    if np.count_nonzero(finite) < 4:
        return {"frequency": [], "power_normalized": [], "dominant_frequency": None}
    t = np.asarray(time[finite], dtype=float)
    y = np.asarray(trace[finite], dtype=float)
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    unique = np.concatenate(([True], np.diff(t) > 0.0))
    t = t[unique]
    y = y[unique]
    if t.size < 4 or float(t[-1] - t[0]) <= 0.0:
        return {"frequency": [], "power_normalized": [], "dominant_frequency": None}

    n = int(max(16, n_points))
    t_uniform = np.linspace(float(t[0]), float(t[-1]), n)
    y_uniform = np.interp(t_uniform, t, y)
    y_uniform = y_uniform - float(np.mean(y_uniform))
    if not np.any(np.abs(y_uniform) > 0.0):
        return {"frequency": [], "power_normalized": [], "dominant_frequency": None}
    window = np.hanning(n)
    spectrum = np.fft.rfft(y_uniform * window)
    freq = np.fft.rfftfreq(n, d=float(t_uniform[1] - t_uniform[0]))
    power = np.abs(spectrum) ** 2
    if power.size > 0:
        power[0] = 0.0
    norm = float(np.max(power))
    if norm > 0.0:
        power = power / norm
    dominant = None
    if np.any(power > 0.0):
        dominant = float(freq[int(np.argmax(power))])
    return {
        "frequency": freq.tolist(),
        "power_normalized": power.tolist(),
        "dominant_frequency": dominant,
    }


def _load_gate_summary(path: str | Path) -> dict[str, Any]:
    gate_path = Path(path)
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    passed = bool(
        payload.get("gate_passed", False)
        or payload.get("gate_report", {}).get("passed", False)
    )
    if not passed:
        raise ValueError(f"W7-X nonlinear gate summary did not pass: {gate_path}")
    return payload


def build_w7x_fluctuation_spectrum_report(
    *,
    nonlinear: str | Path = DEFAULT_NONLINEAR,
    gate_summary: str | Path = DEFAULT_GATE_SUMMARY,
    time_min: float | None = None,
    time_max: float | None = None,
    species_index: int = 0,
) -> dict[str, Any]:
    """Return a JSON-ready W7-X nonlinear fluctuation-spectrum diagnostic report."""

    try:
        import netCDF4
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only without optional dependency.
        raise RuntimeError(
            "netCDF4 is required to read nonlinear fluctuation spectra"
        ) from exc

    nonlinear_path = Path(nonlinear)
    gate_payload = _load_gate_summary(gate_summary)
    with netCDF4.Dataset(nonlinear_path) as root:
        time = np.asarray(_netcdf_variable(root, "Grids/time")[:], dtype=float)
        kx = np.asarray(_netcdf_variable(root, "Grids/kx")[:], dtype=float)
        ky = np.asarray(_netcdf_variable(root, "Grids/ky")[:], dtype=float)
        phi2_ky_t = np.asarray(
            _netcdf_variable(root, "Diagnostics/Phi2_kyt")[:], dtype=float
        )
        phi2_kxky_t = np.asarray(
            _netcdf_variable(root, "Diagnostics/Phi2_kxkyt")[:], dtype=float
        )
        wphi_ky_t = np.asarray(
            _netcdf_variable(root, "Diagnostics/Wphi_kyst")[:], dtype=float
        )
        heat_ky_t = np.asarray(
            _netcdf_variable(root, "Diagnostics/HeatFlux_kyst")[:], dtype=float
        )
        zonal_mode = np.asarray(
            _netcdf_variable(root, "Diagnostics/Phi_zonal_mode_kxt")[:], dtype=float
        )

    if phi2_ky_t.shape != (time.size, ky.size):
        raise ValueError(
            "Diagnostics/Phi2_kyt dimensions do not match Grids/time and Grids/ky"
        )
    if phi2_kxky_t.shape != (time.size, ky.size, kx.size):
        raise ValueError(
            "Diagnostics/Phi2_kxkyt dimensions do not match Grids/time, Grids/ky, and Grids/kx"
        )
    if wphi_ky_t.ndim != 3 or heat_ky_t.ndim != 3:
        raise ValueError(
            "Wphi_kyst and HeatFlux_kyst must have shape (time, species, ky)"
        )
    if (
        species_index < 0
        or species_index >= wphi_ky_t.shape[1]
        or species_index >= heat_ky_t.shape[1]
    ):
        raise ValueError(
            f"species_index {species_index} is out of bounds for resolved W7-X spectra"
        )
    if wphi_ky_t.shape[0] != time.size or wphi_ky_t.shape[2] != ky.size:
        raise ValueError(
            "Diagnostics/Wphi_kyst dimensions do not match Grids/time and Grids/ky"
        )
    if heat_ky_t.shape[0] != time.size or heat_ky_t.shape[2] != ky.size:
        raise ValueError(
            "Diagnostics/HeatFlux_kyst dimensions do not match Grids/time and Grids/ky"
        )
    if zonal_mode.shape != (time.size, kx.size, 2):
        raise ValueError(
            "Diagnostics/Phi_zonal_mode_kxt must have shape (time, kx, ri)"
        )

    tmask = _finite_window(time, time_min=time_min, time_max=time_max)
    selected_time = time[tmask]
    phi2_ky = np.nanmean(np.abs(phi2_ky_t[tmask, :]), axis=0)
    wphi_ky = np.nanmean(np.abs(wphi_ky_t[tmask, species_index, :]), axis=0)
    heat_ky = np.nanmean(heat_ky_t[tmask, species_index, :], axis=0)
    heat_abs = np.abs(heat_ky)
    phi2_kxky = np.nanmean(np.abs(phi2_kxky_t[tmask, :, :]), axis=0)

    dominant_phi_idx = _dominant_index(ky, phi2_ky, exclude_zero=True)
    dominant_heat_idx = _dominant_index(ky, heat_abs, exclude_zero=True)
    zonal_rms = np.sqrt(
        np.nanmean(np.sum(zonal_mode[tmask, :, :] ** 2, axis=-1), axis=0)
    )
    dominant_zonal_kx_idx = _dominant_index(kx, zonal_rms, exclude_zero=True)
    phi2_frequency = _temporal_power_spectrum(
        selected_time, phi2_ky_t[tmask, dominant_phi_idx]
    )
    zonal_trace = zonal_mode[tmask, dominant_zonal_kx_idx, 0]
    zonal_frequency = _temporal_power_spectrum(selected_time, zonal_trace)

    phi2_map_max = float(np.nanmax(phi2_kxky)) if phi2_kxky.size else 0.0
    return {
        "kind": "w7x_fluctuation_spectrum_panel",
        "claim_level": "validated_nonlinear_simulation_spectrum_not_experimental_validation",
        "gate_index_include": False,
        "source_nonlinear": str(nonlinear_path),
        "source_gate_summary": str(Path(gate_summary)),
        "source_gate_passed": True,
        "source_gate_case": gate_payload.get("gate_report", {}).get(
            "case", gate_payload.get("case", "W7-X")
        ),
        "time_min": float(np.min(selected_time)),
        "time_max": float(np.max(selected_time)),
        "time_samples": int(selected_time.size),
        "species_index": int(species_index),
        "ky": ky.tolist(),
        "kx": kx.tolist(),
        "phi2_ky": phi2_ky.tolist(),
        "wphi_ky": wphi_ky.tolist(),
        "heat_flux_ky": heat_ky.tolist(),
        "phi2_ky_distribution": _l1_normalized(phi2_ky).tolist(),
        "wphi_ky_distribution": _l1_normalized(wphi_ky).tolist(),
        "heat_flux_abs_ky_distribution": _l1_normalized(heat_abs).tolist(),
        "phi2_kxky_log10_normalized": np.log10(
            np.maximum(phi2_kxky / max(phi2_map_max, 1.0e-300), 1.0e-12)
        ),
        "dominant_phi_ky": float(ky[dominant_phi_idx]),
        "dominant_heat_flux_ky": float(ky[dominant_heat_idx]),
        "dominant_zonal_kx": float(kx[dominant_zonal_kx_idx]),
        "dominant_phi_envelope_frequency": phi2_frequency["dominant_frequency"],
        "dominant_zonal_frequency": zonal_frequency["dominant_frequency"],
        "phi2_envelope_frequency_spectrum": phi2_frequency,
        "zonal_frequency_spectrum": zonal_frequency,
        "interpretation": (
            "This panel summarizes resolved nonlinear W7-X simulation spectra from a passed transport-window "
            "gate. It is a reproducible simulation diagnostic. It is not a Doppler-reflectometry validation "
            "because the experimental diagnostic transfer function and access model are not encoded here."
        ),
    }


def write_w7x_fluctuation_spectrum_artifacts(
    report: dict[str, Any],
    *,
    out: str | Path = DEFAULT_OUT,
) -> dict[str, str]:
    """Write PNG/PDF/JSON/CSV artifacts for a W7-X spectrum report."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ky = np.asarray(report["ky"], dtype=float)
    kx = np.asarray(report["kx"], dtype=float)
    phi_dist = np.asarray(report["phi2_ky_distribution"], dtype=float)
    wphi_dist = np.asarray(report["wphi_ky_distribution"], dtype=float)
    heat_dist = np.asarray(report["heat_flux_abs_ky_distribution"], dtype=float)
    heat_flux = np.asarray(report["heat_flux_ky"], dtype=float)
    map_log = np.asarray(report["phi2_kxky_log10_normalized"], dtype=float)
    phi_freq = report["phi2_envelope_frequency_spectrum"]
    zonal_freq = report["zonal_frequency_spectrum"]

    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.ravel()

    ax0.plot(ky, phi_dist, marker="o", linewidth=2.0, label=r"$|\phi|^2$")
    ax0.plot(ky, wphi_dist, marker="s", linewidth=2.0, label=r"$W_\phi$")
    ax0.plot(ky, heat_dist, marker="^", linewidth=2.0, label=r"$|Q_i|$")
    ax0.set_xlabel(r"$k_y\rho_i$")
    ax0.set_ylabel("normalized spectrum")
    ax0.set_title("Resolved ky spectra")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="best", fontsize=8)

    if kx.size > 1 and ky.size > 1:
        extent = [
            float(np.min(kx)),
            float(np.max(kx)),
            float(np.min(ky)),
            float(np.max(ky)),
        ]
        image = ax1.imshow(
            map_log,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=-8.0,
            vmax=0.0,
        )
        colorbar = fig.colorbar(image, ax=ax1, shrink=0.88)
        colorbar.set_label(r"$\log_{10}(\langle |\phi_{k_x,k_y}|^2\rangle_t / \max)$")
    ax1.set_xlabel(r"$k_x\rho_i$")
    ax1.set_ylabel(r"$k_y\rho_i$")
    ax1.set_title("kx-ky fluctuation power")

    ax2.plot(ky, heat_flux, marker="o", linewidth=2.0, color="#d1495b")
    ax2.axhline(0.0, color="0.2", linewidth=1.0)
    ax2.set_xlabel(r"$k_y\rho_i$")
    ax2.set_ylabel(r"$\langle Q_i(k_y)\rangle_t$")
    ax2.set_title("Signed heat-flux spectrum")
    ax2.grid(True, alpha=0.25)

    phi_f = np.asarray(phi_freq["frequency"], dtype=float)
    phi_p = np.asarray(phi_freq["power_normalized"], dtype=float)
    zonal_f = np.asarray(zonal_freq["frequency"], dtype=float)
    zonal_p = np.asarray(zonal_freq["power_normalized"], dtype=float)
    if phi_f.size and phi_p.size:
        ax3.plot(
            phi_f,
            phi_p,
            linewidth=1.8,
            label=rf"$|\phi|^2$ envelope, $k_y={report['dominant_phi_ky']:.3g}$",
        )
    if zonal_f.size and zonal_p.size:
        ax3.plot(
            zonal_f,
            zonal_p,
            linewidth=1.8,
            label=rf"zonal $\phi$, $k_x={report['dominant_zonal_kx']:.3g}$",
        )
    ax3.set_xlabel(r"frequency [$v_t/a$]")
    ax3.set_ylabel("normalized temporal power")
    ax3.set_title("Windowed temporal spectrum")
    ax3.set_xlim(left=0.0)
    active_freq = []
    if phi_f.size and phi_p.size:
        active_freq.extend(phi_f[phi_p > 2.0e-2].tolist())
    if zonal_f.size and zonal_p.size:
        active_freq.extend(zonal_f[zonal_p > 2.0e-2].tolist())
    if active_freq:
        full_max = max(
            float(np.max(phi_f)) if phi_f.size else 0.0,
            float(np.max(zonal_f)) if zonal_f.size else 0.0,
        )
        ax3.set_xlim(right=min(full_max, max(0.08, 1.25 * max(active_freq))))
    ax3.grid(True, alpha=0.25)
    if phi_f.size or zonal_f.size:
        ax3.legend(loc="best", fontsize=8)
    ax3.text(
        0.02,
        0.04,
        "simulation diagnostic\nnot an experimental transfer-function comparison",
        transform=ax3.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.92},
    )

    fig.suptitle(
        "W7-X nonlinear fluctuation spectrum\n"
        f"validated input gate, {report['time_samples']} samples, "
        rf"$t\in[{report['time_min']:.1f},{report['time_max']:.1f}]$",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "ky",
                "phi2_ky",
                "wphi_ky",
                "heat_flux_ky",
                "phi2_distribution",
                "wphi_distribution",
                "heat_flux_abs_distribution",
            ],
        )
        writer.writeheader()
        for idx, ky_value in enumerate(ky):
            writer.writerow(
                {
                    "ky": float(ky_value),
                    "phi2_ky": float(report["phi2_ky"][idx]),
                    "wphi_ky": float(report["wphi_ky"][idx]),
                    "heat_flux_ky": float(report["heat_flux_ky"][idx]),
                    "phi2_distribution": float(phi_dist[idx]),
                    "wphi_distribution": float(wphi_dist[idx]),
                    "heat_flux_abs_distribution": float(heat_dist[idx]),
                }
            )
    return {
        "png": str(out_path),
        "pdf": str(pdf_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nonlinear",
        default=str(DEFAULT_NONLINEAR),
        help="Input W7-X nonlinear NetCDF file",
    )
    parser.add_argument(
        "--gate-summary",
        default=str(DEFAULT_GATE_SUMMARY),
        help="Passed W7-X nonlinear gate JSON",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output PNG path")
    parser.add_argument(
        "--time-min", type=float, default=None, help="Optional minimum simulation time"
    )
    parser.add_argument(
        "--time-max", type=float, default=None, help="Optional maximum simulation time"
    )
    parser.add_argument(
        "--species-index",
        type=int,
        default=0,
        help="Species index for resolved spectra",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_w7x_fluctuation_spectrum_report(
        nonlinear=args.nonlinear,
        gate_summary=args.gate_summary,
        time_min=args.time_min,
        time_max=args.time_max,
        species_index=args.species_index,
    )
    paths = write_w7x_fluctuation_spectrum_artifacts(report, out=args.out)
    print(f"saved {paths['png']}")
    print(f"saved {paths['pdf']}")
    print(f"saved {paths['json']}")
    print(f"saved {paths['csv']}")
    print(
        "dominant_phi_ky={phi:.6g} dominant_heat_flux_ky={heat:.6g} dominant_zonal_kx={kx:.6g}".format(
            phi=report["dominant_phi_ky"],
            heat=report["dominant_heat_flux_ky"],
            kx=report["dominant_zonal_kx"],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
