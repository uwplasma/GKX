#!/usr/bin/env python3
"""Compare nonlinear diagnostic windows and term-resolved state dumps."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import re
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset

from gkx.diagnostics.validation_gates import (
    evaluate_scalar_gate,
    gate_report,
    gate_report_to_dict,
)
from gkx.benchmarking.shared import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
    _apply_reference_hypercollisions,
    _two_species_params,
)
from gkx.config import CycloneBaseCase, GridConfig, KBMBaseCase
from gkx.core.grid import build_spectral_grid, twothirds_mask
from gkx.core.velocity import laguerre_quadrature_count
from gkx.geometry import (
    SAlphaGeometry,
    apply_imported_geometry_grid_defaults,
)
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams
from gkx.runtime import (
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from gkx.terms.config import TermConfig
from gkx.terms.nonlinear import (
    _laguerre_j0_field,
    _laguerre_to_grid,
    nonlinear_em_components,
)
from gkx.workflows.runtime.toml import (
    load_runtime_from_toml,
)

try:
    from tools.comparison.compare_gx_rhs_terms import _infer_y0
except ModuleNotFoundError:  # Direct execution adds this script directory.
    from compare_gx_rhs_terms import _infer_y0


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _reduce_species_time(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim == 2:
        if name.startswith("Wapar"):
            return arr[:, 0]
        return np.sum(arr, axis=1)
    return arr


def _load_gkx_csv(path: Path) -> dict[str, np.ndarray]:
    named = np.genfromtxt(path, delimiter=",", names=True)
    if isinstance(named, np.ndarray) and named.dtype.names:
        names = set(named.dtype.names)
        if {"t", "Wg", "Wphi", "Wapar", "energy", "heat_flux", "particle_flux"}.issubset(names):
            out = {
                "t": np.asarray(named["t"], dtype=float),
                "Wg": np.asarray(named["Wg"], dtype=float),
                "Wphi": np.asarray(named["Wphi"], dtype=float),
                "Wapar": np.asarray(named["Wapar"], dtype=float),
                "energy": np.asarray(named["energy"], dtype=float),
                "heat_flux": np.asarray(named["heat_flux"], dtype=float),
                "particle_flux": np.asarray(named["particle_flux"], dtype=float),
            }
            if "gamma" in names:
                out["gamma"] = np.asarray(named["gamma"], dtype=float)
            if "omega" in names:
                out["omega"] = np.asarray(named["omega"], dtype=float)
            return out

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] == 9:
        return {
            "t": data[:, 0],
            "gamma": data[:, 1],
            "omega": data[:, 2],
            "Wg": data[:, 3],
            "Wphi": data[:, 4],
            "Wapar": data[:, 5],
            "energy": data[:, 6],
            "heat_flux": data[:, 7],
            "particle_flux": data[:, 8],
        }
    if data.shape[1] >= 10:
        return {
            "t": data[:, 0],
            "gamma": data[:, 2],
            "omega": data[:, 3],
            "Wg": data[:, 4],
            "Wphi": data[:, 5],
            "Wapar": data[:, 6],
            "energy": data[:, 7],
            "heat_flux": data[:, 8],
            "particle_flux": data[:, 9],
        }
    raise ValueError(f"unsupported GKX CSV shape {data.shape}")


def _load_gkx_diag(path: Path) -> dict[str, np.ndarray]:
    root = Dataset(path, "r")
    diag = root.groups["Diagnostics"]
    grid = root.groups["Grids"]
    out = {
        "t": np.asarray(grid.variables["time"][:], dtype=float),
        "phi2": np.asarray(diag.variables["Phi2_t"][:], dtype=float),
        "Wg": _reduce_species_time(np.asarray(diag.variables["Wg_st"][:], dtype=float), "Wg_st"),
        "Wphi": _reduce_species_time(np.asarray(diag.variables["Wphi_st"][:], dtype=float), "Wphi_st"),
        "Wapar": _reduce_species_time(np.asarray(diag.variables["Wapar_st"][:], dtype=float), "Wapar_st"),
        "heat_flux": _reduce_species_time(np.asarray(diag.variables["HeatFlux_st"][:], dtype=float), "HeatFlux_st"),
        "particle_flux": _reduce_species_time(
            np.asarray(diag.variables["ParticleFlux_st"][:], dtype=float), "ParticleFlux_st"
        ),
    }
    out["energy"] = out["Wg"] + out["Wphi"] + out["Wapar"]
    root.close()
    return out


def _load_gx_diag(path: Path) -> dict[str, np.ndarray]:
    root = Dataset(path, "r")
    diag = root.groups["Diagnostics"]
    grid = root.groups["Grids"]
    t = np.asarray(grid.variables["time"][:], dtype=float)
    out = {
        "t": t,
        "phi2": np.asarray(diag.variables["Phi2_t"][:], dtype=float),
        "Wg": _reduce_species_time(np.asarray(diag.variables["Wg_st"][:], dtype=float), "Wg_st"),
        "Wphi": _reduce_species_time(np.asarray(diag.variables["Wphi_st"][:], dtype=float), "Wphi_st"),
        "Wapar": _reduce_species_time(np.asarray(diag.variables["Wapar_st"][:], dtype=float), "Wapar_st"),
        "heat_flux": _reduce_species_time(np.asarray(diag.variables["HeatFlux_st"][:], dtype=float), "HeatFlux_st"),
        "particle_flux": _reduce_species_time(
            np.asarray(diag.variables["ParticleFlux_st"][:], dtype=float), "ParticleFlux_st"
        ),
    }
    out["energy"] = out["Wg"] + out["Wphi"] + out["Wapar"]
    root.close()
    return out


def _load_gkx(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".nc":
        return _load_gkx_diag(path)
    return _load_gkx_csv(path)


def _apply_time_window(
    series: dict[str, np.ndarray],
    *,
    tmin: float | None,
    tmax: float | None,
) -> dict[str, np.ndarray]:
    """Return a copy restricted to a common diagnostic time window."""

    t = np.asarray(series["t"], dtype=float)
    mask = np.ones(t.shape, dtype=bool)
    if tmin is not None:
        mask &= t >= float(tmin)
    if tmax is not None:
        mask &= t <= float(tmax)
    if int(np.count_nonzero(mask)) < 2:
        raise ValueError("time window must retain at least two diagnostic samples")
    return {key: np.asarray(value)[mask] for key, value in series.items()}


def _reduce_species_resolved(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"expected resolved array with shape (time, species, mode), got {arr.shape} for {name}")
    if name.startswith("Wapar"):
        return np.asarray(arr[:, 0, :], dtype=float)
    return np.asarray(np.sum(arr, axis=1), dtype=float)


def _load_resolved_diag(path: Path) -> dict[str, np.ndarray]:
    root = Dataset(path, "r")
    diag = root.groups["Diagnostics"]
    grid = root.groups["Grids"]
    out = {
        "t": np.asarray(grid.variables["time"][:], dtype=float),
        "kx": np.asarray(grid.variables["kx"][:], dtype=float),
        "ky": np.asarray(grid.variables["ky"][:], dtype=float),
        "Wphi_kx": _reduce_species_resolved(np.asarray(diag.variables["Wphi_kxst"][:], dtype=float), "Wphi_kxst"),
        "Wphi_ky": _reduce_species_resolved(np.asarray(diag.variables["Wphi_kyst"][:], dtype=float), "Wphi_kyst"),
        "HeatFlux_kx": _reduce_species_resolved(
            np.asarray(diag.variables["HeatFlux_kxst"][:], dtype=float),
            "HeatFlux_kxst",
        ),
    }
    root.close()
    return out


def _interp_summary(
    ref_t: np.ndarray,
    ref_y: np.ndarray,
    cmp_t: np.ndarray,
    cmp_y: np.ndarray,
) -> tuple[float, float, float]:
    cmp_interp = np.interp(ref_t, cmp_t, cmp_y)
    denom = np.maximum(np.abs(cmp_interp), 1.0e-30)
    rel = np.abs((ref_y - cmp_interp) / denom)
    final_rel = float((ref_y[-1] - cmp_interp[-1]) / denom[-1])
    return float(np.mean(rel)), float(np.max(rel)), final_rel


def _interp_mode_summary(
    ref_t: np.ndarray,
    ref_y: np.ndarray,
    cmp_t: np.ndarray,
    cmp_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ref_y.ndim != 2 or cmp_y.ndim != 2:
        raise ValueError(f"expected resolved 2D arrays, got {ref_y.shape} and {cmp_y.shape}")
    if ref_y.shape[1] != cmp_y.shape[1]:
        raise ValueError(f"resolved mode count mismatch: {ref_y.shape[1]} vs {cmp_y.shape[1]}")
    mean_rel = np.zeros(ref_y.shape[1], dtype=float)
    max_rel = np.zeros(ref_y.shape[1], dtype=float)
    final_rel = np.zeros(ref_y.shape[1], dtype=float)
    for idx in range(ref_y.shape[1]):
        cmp_interp = np.interp(ref_t, cmp_t, cmp_y[:, idx])
        denom = np.maximum(np.abs(cmp_interp), 1.0e-30)
        rel = np.abs((ref_y[:, idx] - cmp_interp) / denom)
        mean_rel[idx] = float(np.mean(rel))
        max_rel[idx] = float(np.max(rel))
        final_rel[idx] = float((ref_y[-1, idx] - cmp_interp[-1]) / denom[-1])
    return mean_rel, max_rel, final_rel


def _write_resolved_audit(
    *,
    gx_path: Path,
    sp_path: Path,
    out_png: Path,
    out_csv: Path | None,
    tmin: float | None,
    tmax: float | None,
) -> None:
    gx = _load_resolved_diag(gx_path)
    sp = _load_resolved_diag(sp_path)

    if tmin is not None or tmax is not None:
        gx_mask = np.ones(gx["t"].shape, dtype=bool)
        sp_mask = np.ones(sp["t"].shape, dtype=bool)
        if tmin is not None:
            gx_mask &= gx["t"] >= float(tmin)
            sp_mask &= sp["t"] >= float(tmin)
        if tmax is not None:
            gx_mask &= gx["t"] <= float(tmax)
            sp_mask &= sp["t"] <= float(tmax)
        if int(np.count_nonzero(gx_mask)) < 2 or int(np.count_nonzero(sp_mask)) < 2:
            raise ValueError("resolved audit time window must retain at least two samples")
        gx["t"] = gx["t"][gx_mask]
        sp["t"] = sp["t"][sp_mask]
        for key in ("Wphi_kx", "Wphi_ky", "HeatFlux_kx"):
            gx[key] = gx[key][gx_mask, :]
            sp[key] = sp[key][sp_mask, :]

    rows: list[dict[str, float | str | int]] = []
    audit_specs = (
        ("Wphi_kx", "kx", gx["kx"]),
        ("Wphi_ky", "ky", gx["ky"]),
        ("HeatFlux_kx", "kx", gx["kx"]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    for ax, (name, axis_name, coords) in zip(axes, audit_specs):
        mean_rel, max_rel, final_rel = _interp_mode_summary(sp["t"], sp[name], gx["t"], gx[name])
        coord_vals = np.asarray(coords, dtype=float)
        ax.plot(coord_vals, mean_rel, marker="o", linewidth=2.0, label="mean rel")
        ax.plot(coord_vals, max_rel, marker="s", linewidth=1.8, linestyle="--", label="max rel")
        ax.plot(coord_vals, np.abs(final_rel), marker="^", linewidth=1.6, linestyle=":", label="|final rel|")
        ax.set_title(name.replace("_", " "), fontsize=12, fontweight="bold")
        ax.set_xlabel(axis_name)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        if axis_name == "kx":
            ax.set_ylabel("relative error")
        worst_idx = int(np.argmax(mean_rel))
        ax.axvline(coord_vals[worst_idx], color="0.6", linewidth=1.0, linestyle=":")
        ax.text(
            coord_vals[worst_idx],
            mean_rel[worst_idx],
            f"worst {axis_name}={coord_vals[worst_idx]:.3g}",
            fontsize=8,
            ha="left",
            va="bottom",
        )
        for idx, coord in enumerate(coord_vals):
            rows.append(
                {
                    "metric": name,
                    "axis": axis_name,
                    "mode_index": idx,
                    "mode_value": float(coord),
                    "mean_rel": float(mean_rel[idx]),
                    "max_rel": float(max_rel[idx]),
                    "final_rel": float(final_rel[idx]),
                }
            )
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Resolved nonlinear diagnostic audit: GX vs GKX", fontsize=14, fontweight="bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    print(f"saved {out_png}")
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"saved {out_csv}")


def run_diagnostics(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx", type=Path, required=True, help="GX .out.nc file with diagnostics")
    parser.add_argument("--gkx", type=Path, required=True, help="GKX nonlinear CSV or GX-style .out.nc")
    parser.add_argument("--tmin", type=float, default=None, help="Optional min time for plotting and metrics")
    parser.add_argument("--tmax", type=float, default=None, help="Optional max time for plotting")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/_static/nonlinear_cyclone_diag_compare.png"),
        help="Output figure path",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Nonlinear diagnostics: GX vs GKX",
        help="Figure title.",
    )
    parser.add_argument(
        "--resolved-out",
        type=Path,
        default=None,
        help="Optional resolved-diagnostic audit figure (requires both inputs as .nc)",
    )
    parser.add_argument(
        "--resolved-csv",
        type=Path,
        default=None,
        help="Optional CSV summary for the resolved-diagnostic audit",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary with scalar gate results for the plotted nonlinear diagnostics.",
    )
    parser.add_argument(
        "--summary-case",
        type=str,
        default="nonlinear_diagnostics_window",
        help="Case name stored in --summary-json gate metadata.",
    )
    parser.add_argument(
        "--summary-source",
        type=str,
        default="runtime diagnostics",
        help="Reference/source label stored in --summary-json gate metadata.",
    )
    parser.add_argument(
        "--gate-mean-rel",
        type=float,
        default=0.10,
        help="Mean relative-mismatch tolerance used in --summary-json gate metadata.",
    )
    args = parser.parse_args(argv)

    gx = _load_gx_diag(args.gx)
    sp = _load_gkx(args.gkx)

    if args.tmin is not None or args.tmax is not None:
        gx = _apply_time_window(gx, tmin=args.tmin, tmax=args.tmax)
        sp = _apply_time_window(sp, tmin=args.tmin, tmax=args.tmax)

    import matplotlib.patheffects as pe

    gx_style = {"color": "#111827", "linewidth": 2.2, "linestyle": "--", "marker": "s", "markevery": max(len(sp["t"]) // 14, 1), "ms": 3.0, "zorder": 4}
    sp_style = {"color": "#2563eb", "linewidth": 2.2, "marker": "o", "markevery": max(len(sp["t"]) // 14, 1), "ms": 3.2, "alpha": 0.92, "zorder": 3}
    metric_specs = [
        ("Wg", gx["Wg"], sp["Wg"]),
        ("Wphi", gx["Wphi"], sp["Wphi"]),
        ("Wapar", gx["Wapar"], sp["Wapar"]),
        ("Wtot", gx["energy"], sp["energy"]),
        ("Heat flux", gx["heat_flux"], sp["heat_flux"]),
        ("Particle flux", gx["particle_flux"], sp["particle_flux"]),
    ]
    keep_specs = []
    for label, gx_y, sp_y in metric_specs:
        if label in {"Wg", "Wphi", "Wtot", "Heat flux"}:
            keep_specs.append((label, gx_y, sp_y))
            continue
        if max(float(np.nanmax(np.abs(gx_y))), float(np.nanmax(np.abs(sp_y)))) > 1.0e-10:
            keep_specs.append((label, gx_y, sp_y))

    ncols = 2
    nrows = int(np.ceil(len(keep_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.8, 2.9 * nrows + 0.8), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (label, gx_y, sp_y) in zip(axes, keep_specs):
        ax.plot(sp["t"], sp_y, label="GKX", **sp_style)
        gx_line = ax.plot(gx["t"], gx_y, label="GX", **gx_style)[0]
        gx_line.set_path_effects([pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()])
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(keep_specs):]:
        ax.remove()

    if len(keep_specs) > 0:
        axes[0].legend(frameon=False, ncol=2, loc="upper right")
    for ax in axes[max(0, len(keep_specs) - ncols):len(keep_specs)]:
        ax.set_xlabel("t")
    fig.suptitle(args.title, fontsize=13, fontweight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"saved {args.out}")

    summary_pairs = []
    if "phi2" in sp and "phi2" in gx:
        summary_pairs.append(("Phi2", sp["phi2"], gx["phi2"]))
    summary_pairs.extend(
        [
            ("Wg", sp["Wg"], gx["Wg"]),
            ("Wphi", sp["Wphi"], gx["Wphi"]),
            ("Wapar", sp["Wapar"], gx["Wapar"]),
            ("HeatFlux", sp["heat_flux"], gx["heat_flux"]),
            ("ParticleFlux", sp["particle_flux"], gx["particle_flux"]),
        ]
    )
    print("metric mean_rel_abs max_rel_abs final_rel")
    summary_rows: list[dict[str, float | str]] = []
    for name, sp_y, gx_y in summary_pairs:
        mean_rel, max_rel, final_rel = _interp_summary(sp["t"], sp_y, gx["t"], gx_y)
        print(f"{name} {mean_rel:.6e} {max_rel:.6e} {final_rel:.6e}")
        summary_rows.append(
            {
                "metric": name,
                "mean_rel_abs": float(mean_rel),
                "max_rel_abs": float(max_rel),
                "final_rel": float(final_rel),
            }
        )
    if args.summary_json is not None:
        threshold = float(args.gate_mean_rel)
        if threshold < 0.0:
            raise ValueError("--gate-mean-rel must be non-negative")
        report = gate_report(
            args.summary_case,
            args.summary_source,
            [
                evaluate_scalar_gate(
                    f"{row['metric']}_mean_rel_abs",
                    float(row["mean_rel_abs"]),
                    0.0,
                    atol=threshold,
                    rtol=0.0,
                    notes=f"Passes when mean relative mismatch <= {threshold:.6g}.",
                )
                for row in summary_rows
            ],
        )
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "gx": str(args.gx),
            "gkx": str(args.gkx),
            "case": str(args.summary_case),
            "source": str(args.summary_source),
            "tmin": None if args.tmin is None else float(args.tmin),
            "tmax": None if args.tmax is None else float(args.tmax),
            "gate_mean_rel": threshold,
            "summary": summary_rows,
            "gate_report": gate_report_to_dict(report),
            "gate_passed": bool(report.passed),
        }
        args.summary_json.write_text(
            json.dumps(
                _json_clean(payload),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        print(f"saved {args.summary_json}")
    if args.resolved_out is not None:
        if args.gx.suffix != ".nc" or args.gkx.suffix != ".nc":
            raise ValueError("--resolved-out requires both --gx and --gkx to be .nc files")
        _write_resolved_audit(
            gx_path=args.gx,
            sp_path=args.gkx,
            out_png=args.resolved_out,
            out_csv=args.resolved_csv,
            tmin=args.tmin,
            tmax=args.tmax,
        )
    return 0


def _slice_species_params(
    params: LinearParams, nspec: int, *, species_index: int = 0
) -> LinearParams:
    """Slice species-vector parameters to `nspec` entries starting at `species_index`."""
    if nspec <= 0:
        return params
    start = max(int(species_index), 0)
    stop = start + int(nspec)
    fields = (
        "charge_sign",
        "density",
        "mass",
        "temp",
        "vth",
        "rho",
        "tz",
        "fprim",
        "tprim",
        "tprim_e",
        "nu",
    )
    updates: dict[str, object] = {}
    for name in fields:
        val = getattr(params, name)
        arr = np.asarray(val)
        if arr.ndim == 0:
            continue
        if arr.shape[0] >= stop:
            updates[name] = arr[start:stop]
    return cast(LinearParams, replace(params, **updates)) if updates else params  # type: ignore[arg-type]


def _load_shape(path: Path) -> dict[str, int]:
    data: dict[str, int] = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            data[parts[0]] = int(parts[1])
    return data


def _infer_shape_from_gx_out(path: Path) -> dict[str, int]:
    """Best-effort fallback when rhs_terms_shape.txt is unavailable."""
    with Dataset(path, "r") as root:
        grids = root.groups["Grids"]
        inputs = root.groups["Inputs"]
        ky = np.asarray(grids.variables["ky"][:], dtype=float)
        kx = np.asarray(grids.variables["kx"][:], dtype=float)
        theta = np.asarray(grids.variables["theta"][:], dtype=float)

        def _read_int(name: str, default: int) -> int:
            if name not in inputs.variables:
                return default
            return int(np.asarray(inputs.variables[name][:]).reshape(()))

        nspec = _read_int("nspecies", _read_int("nspec", 1))
        nl = _read_int("nlaguerre", 1)
        nm = _read_int("nhermite", 1)
        nj = _read_int("nj", laguerre_quadrature_count(nl))
    return {
        "nspec": nspec,
        "nl": nl,
        "nm": nm,
        "nyc": int(ky.size),
        "nx": int(kx.size),
        "nz": int(theta.size),
        "nj": nj,
    }


def _infer_shape_from_gx_input(path: Path) -> dict[str, int]:
    """Parse GX .in file for nhermite/nlaguerre/nspecies values."""
    text = path.read_text()

    def _pick_int(key: str, default: int) -> int:
        pat = rf"^\s*{re.escape(key)}\s*=\s*([0-9]+)"
        m = re.search(pat, text, flags=re.MULTILINE)
        if m is None:
            return default
        return int(m.group(1))

    nl = _pick_int("nlaguerre", 1)
    nm = _pick_int("nhermite", 1)
    nspec = _pick_int("nspecies", 1)
    return {"nspec": nspec, "nl": nl, "nm": nm, "nj": laguerre_quadrature_count(nl)}


def _pick_first_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _pick_species_dump(gx_dir: Path, stem: str, species_index: int) -> Path | None:
    return _pick_first_existing(
        gx_dir / f"{stem}_s{species_index}.bin",
        gx_dir / f"{stem}.bin",
    )


def _load_bin(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.complex64)
    if raw.size != int(np.prod(shape)):
        raise ValueError(f"{path} size {raw.size} does not match expected {shape}")
    return raw.reshape(shape)


def _load_bin_complex(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    """Load complex binary as complex64, falling back to complex128."""
    expected = int(np.prod(shape))
    raw = np.fromfile(path, dtype=np.complex64)
    if raw.size == expected:
        return raw.reshape(shape)
    raw128 = np.fromfile(path, dtype=np.complex128)
    if raw128.size == expected:
        return raw128.astype(np.complex64).reshape(shape)
    raise ValueError(f"{path} size {raw.size} does not match expected {shape}")


def _load_float_bin(path: Path, n: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != n:
        raise ValueError(f"{path} size {raw.size} does not match expected {n}")
    return raw


def _load_float_bin_any(path: Path) -> np.ndarray:
    """Load float32 binary without enforcing a fixed length."""
    return np.fromfile(path, dtype=np.float32)


def _pad_spectral_kxky(
    arr: jnp.ndarray,
    *,
    kx_main: np.ndarray,
    ky_main: np.ndarray,
    kx_pad: np.ndarray,
    ky_pad: np.ndarray,
) -> jnp.ndarray:
    """Zero-pad spectral array onto a larger (kx, ky) grid by matching values."""
    nyc_main = int(ky_main.size)
    nx_main = int(kx_main.size)
    nyc_pad = int(ky_pad.size)
    nx_pad = int(kx_pad.size)
    if nyc_main == nyc_pad and nx_main == nx_pad:
        return arr
    out_shape = arr.shape[:-3] + (nyc_pad, nx_pad, arr.shape[-1])
    out = jnp.zeros(out_shape, dtype=arr.dtype)
    ky_map = []
    for val in ky_main:
        matches = np.where(np.isclose(ky_pad, val))[0]
        if matches.size == 0:
            raise ValueError(f"ky value {val} missing from pad grid")
        ky_map.append(int(matches[0]))
    kx_map = []
    for val in kx_main:
        matches = np.where(np.isclose(kx_pad, val))[0]
        if matches.size == 0:
            raise ValueError(f"kx value {val} missing from pad grid")
        kx_map.append(int(matches[0]))
    for ky_i, ky_pad_i in enumerate(ky_map):
        for kx_i, kx_pad_i in enumerate(kx_map):
            out = out.at[..., ky_pad_i, kx_pad_i, :].set(arr[..., ky_i, kx_i, :])
    return out


def _extract_spectral_kxky(
    arr: jnp.ndarray,
    *,
    kx_main: np.ndarray,
    ky_main: np.ndarray,
    kx_pad: np.ndarray,
    ky_pad: np.ndarray,
) -> jnp.ndarray:
    """Extract main-grid spectral array from a padded (kx, ky) grid."""
    nyc_main = int(ky_main.size)
    nx_main = int(kx_main.size)
    out_shape = arr.shape[:-3] + (nyc_main, nx_main, arr.shape[-1])
    out = jnp.zeros(out_shape, dtype=arr.dtype)
    ky_map = []
    for val in ky_main:
        matches = np.where(np.isclose(ky_pad, val))[0]
        if matches.size == 0:
            raise ValueError(f"ky value {val} missing from pad grid")
        ky_map.append(int(matches[0]))
    kx_map = []
    for val in kx_main:
        matches = np.where(np.isclose(kx_pad, val))[0]
        if matches.size == 0:
            raise ValueError(f"kx value {val} missing from pad grid")
        kx_map.append(int(matches[0]))
    for ky_i, ky_pad_i in enumerate(ky_map):
        for kx_i, kx_pad_i in enumerate(kx_map):
            out = out.at[..., ky_i, kx_i, :].set(arr[..., ky_pad_i, kx_pad_i, :])
    return out


def _reshape_gx(
    raw: np.ndarray,
    *,
    nspec: int,
    nl: int,
    nm: int,
    nyc: int,
    nx: int,
    nz: int,
) -> np.ndarray:
    nR = nyc * nx * nz
    # GX flattens moments with moment_idx = l + Nl * m (m-major).
    arr = raw.reshape((nspec, nm, nl, nR)).transpose(0, 2, 1, 3)
    ky_idx = np.arange(nyc)[:, None, None]
    kx_idx = np.arange(nx)[None, :, None]
    z_idx = np.arange(nz)[None, None, :]
    idxyz = ky_idx + nyc * (kx_idx + nx * z_idx)
    arr = arr[..., idxyz.ravel()]
    return arr.reshape((nspec, nl, nm, nyc, nx, nz))


def _expand_ky(arr: np.ndarray, *, nyc: int) -> np.ndarray:
    """Expand Nyc (real FFT) axis to full Ny using conjugate symmetry."""
    if arr.shape[-3] != nyc:
        raise ValueError("Expected ky axis at position -3 with length nyc")
    ny_full = 2 * (nyc - 1)
    if ny_full <= 0:
        return arr
    pos = arr
    if nyc <= 2:
        return pos
    neg = np.conj(pos[..., 1 : nyc - 1, :, :])
    neg = neg[..., ::-1, :, :]
    nx = pos.shape[-2]
    if nx > 1:
        kx_neg = np.concatenate(([0], np.arange(nx - 1, 0, -1)))
        neg = neg[..., kx_neg, :]
    return np.concatenate([pos, neg], axis=-3)


def _resolve_dealias_mask(
    mask: jnp.ndarray | np.ndarray, *, ny: int, nx: int
) -> jnp.ndarray:
    """Return a dealias mask compatible with the compared spectral field shape."""
    mask_np = np.asarray(mask, dtype=bool)
    if mask_np.shape == (ny, nx):
        return jnp.asarray(mask_np)
    return twothirds_mask(int(ny), int(nx))


def _synth_positive_ky(*, nyc: int, y0: float) -> np.ndarray:
    return np.arange(int(nyc), dtype=float) / float(y0)


def _synth_full_ky(*, nyc: int, y0: float) -> np.ndarray:
    ky_pos = _synth_positive_ky(nyc=nyc, y0=y0)
    ny_full = 2 * (int(nyc) - 1)
    if ny_full > int(nyc):
        return np.concatenate([ky_pos, -ky_pos[1 : int(nyc) - 1][::-1]])
    return ky_pos


def _synth_kx(*, nx: int, delta_kx: float) -> np.ndarray:
    idx = np.fft.fftfreq(int(nx), d=1.0 / float(nx))
    return (2.0 * np.pi / (2.0 * np.pi / float(delta_kx))) * idx


def _gx_grid_convention(g_mu: jnp.ndarray, roots: jnp.ndarray) -> jnp.ndarray:
    """Undo the ``exp(-x/2)`` GKX carries on quadrature-grid values.

    ``laguerre_transform`` splits the Gauss-Laguerre weight evenly between the
    two matrices for conditioning, so ``_laguerre_to_grid`` yields
    ``exp(-x_j/2) g(x_j)``. GX stores the unsplit ``g(x_j)``. Every comparison
    of a raw grid quantity against a GX dump has to remove the factor first;
    quantities the two codes build the same way from ``roots`` (J0*phi, b) do
    not.

    Note the rescaling reintroduces exactly the dynamic range the split was
    meant to avoid -- ``exp(x_max/2)`` is 5e16 at ``nl=16`` -- so the GX-side
    convention, and hence this comparison, runs out of float32 by ``nl~32``.
    """

    scale = jnp.exp(0.5 * jnp.asarray(roots, dtype=jnp.float64))
    return g_mu * scale[None, :, None, None, None, None].astype(g_mu.dtype)


def _summary(label: str, ref: np.ndarray, test: np.ndarray) -> None:
    if ref.ndim != test.ndim:
        raise ValueError(
            f"{label}: incompatible ranks ref.ndim={ref.ndim} test.ndim={test.ndim}"
        )
    if ref.shape != test.shape:
        common = tuple(min(r, t) for r, t in zip(ref.shape, test.shape))
        slices = tuple(slice(0, n) for n in common)
        print(
            f"{label:12s} shape mismatch ref={ref.shape} test={test.shape}; "
            f"comparing common slice={common}"
        )
        ref = ref[slices]
        test = test[slices]
    diff = test - ref
    max_ref = float(np.max(np.abs(ref)))
    max_test = float(np.max(np.abs(test)))
    max_diff = float(np.max(np.abs(diff)))
    if max_ref == 0.0:
        max_rel = float("nan")
        rms_rel = float("nan")
    else:
        thresh = max_ref * 1.0e-12
        mask = np.abs(ref) > thresh
        if not np.any(mask):
            max_rel = float("nan")
            rms_rel = float("nan")
        else:
            rel = diff[mask] / ref[mask]
            max_rel = float(np.max(np.abs(rel)))
            rms_rel = float(
                np.sqrt(np.mean(np.abs(diff[mask]) ** 2))
                / (np.sqrt(np.mean(np.abs(ref[mask]) ** 2)) + 1.0e-30)
            )
    idx_diff = np.unravel_index(int(np.argmax(np.abs(diff))), diff.shape)
    ref_at = ref[idx_diff]
    test_at = test[idx_diff]
    print(
        f"{label:12s} max|ref|={max_ref:.3e} max|test|={max_test:.3e} "
        f"max|diff|={max_diff:.3e} max|rel|={max_rel:.3e} rms_rel={rms_rel:.3e} "
        f"idx={idx_diff} ref={ref_at:.3e} test={test_at:.3e}"
    )


def _apply_kx_order(arr: np.ndarray, *, order: str, kx_axis: int) -> np.ndarray:
    if order == "native":
        return arr
    if order == "fftshift":
        return np.fft.fftshift(arr, axes=(kx_axis,))
    if order == "ifftshift":
        return np.fft.ifftshift(arr, axes=(kx_axis,))
    raise ValueError(f"Unknown kx order '{order}'")


def _apply_kx_order_1d(arr: np.ndarray, *, order: str) -> np.ndarray:
    if order == "native":
        return arr
    if order == "fftshift":
        return np.fft.fftshift(arr)
    if order == "ifftshift":
        return np.fft.ifftshift(arr)
    raise ValueError(f"Unknown kx order '{order}'")


def _broadcast_grid(grid: jnp.ndarray, ndim: int) -> jnp.ndarray:
    shape = (1,) * (ndim - 3) + grid.shape + (1,)
    return jnp.reshape(grid, shape)


def _ifft2_xy(arr: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.ifft2(arr, axes=(-3, -2))


def _bracket_real(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    ny_full: int | None = None,
) -> jnp.ndarray:
    """Return real-space bracket {G, chi} without spectral masking."""
    dchi_dx, dchi_dy = _grad_xy_real(
        chi_hat, kx_grid=kx_grid, ky_grid=ky_grid, ny_full=ny_full
    )
    dchi_dx = dchi_dx[:, :, None, ...]
    dchi_dy = dchi_dy[:, :, None, ...]
    dG_dx, dG_dy = _grad_xy_real(
        G_hat, kx_grid=kx_grid, ky_grid=ky_grid, ny_full=ny_full
    )
    bracket = dG_dx * dchi_dy - dG_dy * dchi_dx
    return bracket


def _grad_xy_real(
    F_hat: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    ny_full: int | None = None,
    fft_norm: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    imag = jnp.asarray(1j, dtype=F_hat.dtype)
    kx = jnp.asarray(kx_grid)
    ky = jnp.asarray(ky_grid)
    if ny_full is None:
        ny_full = int(ky.shape[0])
    use_rfft = int(ky.shape[0]) != int(ny_full)
    if fft_norm is None:
        fft_norm_val = float(int(kx.shape[1]) * int(ny_full))
    else:
        fft_norm_val = float(fft_norm)
    nxy = fft_norm_val
    kx_b = _broadcast_grid(kx, F_hat.ndim)
    ky_b = _broadcast_grid(ky, F_hat.ndim)
    if use_rfft:
        # ky axis is Nyc; use irfft2 with ky axis last in axes tuple.
        dF_dx = jnp.fft.irfft2(
            imag * kx_b * F_hat, s=(int(kx.shape[1]), int(ny_full)), axes=(-2, -3)
        )
        dF_dy = jnp.fft.irfft2(
            imag * ky_b * F_hat, s=(int(kx.shape[1]), int(ny_full)), axes=(-2, -3)
        )
        dF_dx = dF_dx * nxy
        dF_dy = dF_dy * nxy
    else:
        dF_dx = _ifft2_xy(imag * kx_b * F_hat) * nxy
        dF_dy = _ifft2_xy(imag * ky_b * F_hat) * nxy
    dF_dx = jnp.real(dF_dx)
    dF_dy = jnp.real(dF_dy)
    return dF_dx, dF_dy


def _build_runtime_compare_context(
    config_path: Path,
    *,
    nx: int,
    ny_full: int,
    nz: int,
    nl: int,
    nm: int,
    ky_vals_nyc: np.ndarray,
    y0_override: float | None,
):
    cfg, _data = load_runtime_from_toml(config_path)
    y0_use = float(y0_override) if y0_override is not None else _infer_y0(ky_vals_nyc)
    cfg_use = replace(
        cfg,
        grid=replace(
            cfg.grid,
            Nx=int(nx),
            Ny=int(ny_full),
            Nz=int(nz),
            y0=float(y0_use),
        ),
    )
    geom = build_runtime_geometry(cfg_use)
    grid_cfg = apply_imported_geometry_grid_defaults(geom, cfg_use.grid)
    grid = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg_use, Nm=nm, geom=geom)
    term_cfg = build_runtime_term_config(cfg_use)
    return cfg_use, geom, grid, params, term_cfg


def build_terms_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gx-dir",
        type=Path,
        required=True,
        help="Directory with GX nonlinear term dumps",
    )
    parser.add_argument(
        "--gx-out", type=Path, required=True, help="GX .out.nc file to map ky indices"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Runtime TOML config. When set, geometry/physics come from the runtime path instead of --case.",
    )
    parser.add_argument(
        "--case", type=str, default="cyclone", choices=("cyclone", "kbm")
    )
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--Nl", type=int, default=48)
    parser.add_argument("--Nm", type=int, default=16)
    parser.add_argument("--Ny", type=int, default=24)
    parser.add_argument("--Nz", type=int, default=96)
    parser.add_argument("--Lx", type=float, default=62.8)
    parser.add_argument("--Ly", type=float, default=62.8)
    parser.add_argument("--y0", type=float, default=None)
    parser.add_argument("--boundary", type=str, default="linked")
    parser.add_argument(
        "--species-index", type=int, default=0, help="Species index to compare"
    )
    parser.add_argument("--ntheta", type=int, default=None)
    parser.add_argument("--nperiod", type=int, default=None)
    parser.add_argument(
        "--kx-order",
        type=str,
        default="auto",
        help="kx ordering for spectral arrays: native, fftshift, ifftshift, or auto",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Optional npz output for GKX terms"
    )
    return parser


def run_terms(argv: list[str] | None = None) -> None:
    args = build_terms_parser().parse_args(argv)

    shape_path = args.gx_dir / "rhs_terms_shape.txt"
    if shape_path.exists():
        shape = _load_shape(shape_path)
    else:
        print(f"Missing {shape_path}; inferring shape from GX input/output files")
        shape = _infer_shape_from_gx_out(args.gx_out)
        gx_input = _pick_first_existing(*sorted(args.gx_dir.glob("*.in")))
        if gx_input is not None:
            shape.update(_infer_shape_from_gx_input(gx_input))
        kx_dump = args.gx_dir / "nl_kx.bin"
        ky_dump = args.gx_dir / "nl_ky.bin"
        if kx_dump.exists():
            shape["nx"] = int(np.fromfile(kx_dump, dtype=np.float32).size)
        if ky_dump.exists():
            ny_raw = int(np.fromfile(ky_dump, dtype=np.float32).size)
            shape["nyc"] = ny_raw
        phi_guess = _pick_first_existing(
            args.gx_dir / "nl_phi.bin", args.gx_dir / "phi.bin"
        )
        if phi_guess is not None and int(shape.get("nz", 0)) <= 0:
            nphi = int(np.fromfile(phi_guess, dtype=np.complex64).size)
            nx_guess = int(shape.get("nx", 0))
            nyc_guess = int(shape.get("nyc", 0))
            if nx_guess > 0 and nyc_guess > 0 and nphi % (nx_guess * nyc_guess) == 0:
                shape["nz"] = int(nphi // (nx_guess * nyc_guess))
        g_guess = _pick_first_existing(
            args.gx_dir / "nl_g_state_s0.bin",
            args.gx_dir / "nl_g_state.bin",
            args.gx_dir / "g_state_s0.bin",
            args.gx_dir / "g_state.bin",
        )
        if g_guess is not None:
            nraw = int(np.fromfile(g_guess, dtype=np.complex64).size)
            denom = (
                int(shape.get("nl", 1))
                * int(shape.get("nm", 1))
                * int(shape.get("nyc", 1))
                * int(shape.get("nx", 1))
                * int(shape.get("nz", 1))
            )
            if denom > 0 and nraw % denom == 0:
                shape["nspec"] = int(max(1, nraw // denom))
    nspec = shape.get("nspec", 1)
    nl = shape.get("nl", args.Nl)
    nm = shape.get("nm", args.Nm)
    nyc = shape.get("nyc", args.Ny // 2 + 1)
    nx = shape.get("nx", 1)
    nz = shape.get("nz", args.Nz)
    nj = shape.get("nj")
    if nj is None:
        nj = laguerre_quadrature_count(nl)
    gx_shape = (nspec, nl, nm, nyc, nx, nz)

    g_state_path = _pick_species_dump(args.gx_dir, "nl_g_state", args.species_index)
    if g_state_path is None:
        g_state_path = _pick_species_dump(args.gx_dir, "g_state", args.species_index)
    if g_state_path is None:
        raise FileNotFoundError(
            "Expected nl_g_state/g_state dump in GX nonlinear dump directory"
        )
    g_state_nyc = _reshape_gx(
        _load_bin(g_state_path, gx_shape),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )
    g_state = _expand_ky(g_state_nyc, nyc=nyc)

    muB_roots = None
    muB_path = args.gx_dir / "nl_muB.bin"
    if muB_path.exists():
        muB_roots = np.fromfile(muB_path, dtype=np.float32)
    rho2s_dump = None
    rho2s_path = args.gx_dir / "nl_rho2s.bin"
    if rho2s_path.exists():
        rho2s_dump = float(np.fromfile(rho2s_path, dtype=np.float32)[0])
    kperp2_dump = None
    kperp2_path = args.gx_dir / "nl_kperp2.bin"
    if kperp2_path.exists():
        raw_kperp2 = np.fromfile(kperp2_path, dtype=np.float32)
        expected_kperp2 = nyc * nx * nz
        if raw_kperp2.size == expected_kperp2:
            # GX flat index: idxyz = idy + nyc * (idx + nx * idz)
            # C-order view is (z, x, y); transpose to (y, x, z).
            kperp2_dump = raw_kperp2.reshape((nz, nx, nyc)).transpose(2, 1, 0)
        else:
            print(f"Skipping kperp2 dump: size {raw_kperp2.size} != {expected_kperp2}")

    kx_vals = None
    ky_vals = None
    kx_vals_dump = None
    ky_vals_dump = None
    kx_dump = _pick_first_existing(
        args.gx_dir / "nl_kx.bin",
        *sorted(args.gx_dir.glob("diag_state_kx_t*.bin")),
    )
    ky_dump = _pick_first_existing(
        args.gx_dir / "nl_ky.bin",
        *sorted(args.gx_dir.glob("diag_state_ky_t*.bin")),
    )
    if kx_dump is not None and ky_dump is not None:
        try:
            kx_vals_dump = _load_float_bin(kx_dump, nx).astype(float)
        except ValueError:
            kx_vals_dump = _load_float_bin_any(kx_dump).astype(float)
        try:
            ky_vals_dump = _load_float_bin(ky_dump, nyc).astype(float)
        except ValueError:
            ky_vals_dump = _load_float_bin_any(ky_dump).astype(float)
    with Dataset(args.gx_out, "r") as root:
        ky_vals = np.asarray(root.groups["Grids"].variables["ky"][:], dtype=float)
        kx_vals = np.asarray(root.groups["Grids"].variables["kx"][:], dtype=float)
    if kx_vals is None or ky_vals is None:
        raise ValueError("Failed to load GX kx/ky values")
    y0_infer = float(args.y0) if args.y0 is not None else _infer_y0(ky_vals)
    if kx_vals_dump is not None and int(kx_vals_dump.size) == nx:
        kx_vals = kx_vals_dump
    elif int(kx_vals.size) != nx:
        positive = np.abs(np.asarray(kx_vals, dtype=float))
        positive = positive[positive > 0.0]
        delta_kx = (
            float(np.min(positive)) if positive.size else float(2.0 * np.pi / args.Lx)
        )
        kx_vals = _synth_kx(nx=nx, delta_kx=delta_kx)
    if ky_vals_dump is not None and int(ky_vals_dump.size) == nyc:
        ky_vals_nyc = ky_vals_dump
        ky_vals_full = _synth_full_ky(nyc=nyc, y0=y0_infer)
        ny_full = int(ky_vals_full.size)
    elif int(ky_vals.size) == nyc:
        ky_vals_nyc = ky_vals
        ky_vals_full = _synth_full_ky(nyc=nyc, y0=y0_infer)
        ny_full = int(ky_vals_full.size)
    elif int(ky_vals.size) < nyc:
        ky_vals_nyc = _synth_positive_ky(nyc=nyc, y0=y0_infer)
        ky_vals_full = _synth_full_ky(nyc=nyc, y0=y0_infer)
        ny_full = int(ky_vals_full.size)
    else:
        ky_vals_full = np.asarray(ky_vals, dtype=float)
        ny_full = int(ky_vals_full.size)
        ky_vals_nyc = ky_vals_full[:nyc]
    if kx_vals_dump is not None:
        kx_vals_pad = kx_vals_dump
    else:
        kx_vals_pad = kx_vals
    if ky_vals_dump is not None:
        ky_vals_pad = ky_vals_dump
    else:
        ky_vals_pad = ky_vals_nyc
    nyc_pad = int(ky_vals_pad.size)
    nx_pad = int(kx_vals_pad.size)
    ny_full_pad = 2 * (nyc_pad - 1)
    ky_idx = int(np.argmin(np.abs(ky_vals_nyc - float(args.ky))))
    ky_grid_full, kx_grid_full = np.meshgrid(ky_vals_full, kx_vals, indexing="ij")
    ky_grid_nyc, kx_grid_nyc = np.meshgrid(ky_vals_nyc, kx_vals, indexing="ij")
    ky_grid_pad, kx_grid_pad = np.meshgrid(ky_vals_pad, kx_vals_pad, indexing="ij")

    phi_path = args.gx_dir / "nl_phi.bin"
    if not phi_path.exists():
        phi_path = args.gx_dir / "phi.bin"
    phi_pad = None
    phi_raw_c64 = np.fromfile(phi_path, dtype=np.complex64)
    expected_main = nyc * nx * nz
    expected_pad = nyc_pad * nx_pad * nz
    if phi_raw_c64.size == expected_pad:
        phi_pad = phi_raw_c64.reshape((nz, nx_pad, nyc_pad)).transpose(2, 1, 0)
        phi_nyc = np.asarray(
            _extract_spectral_kxky(
                jnp.asarray(phi_pad),
                kx_main=kx_vals,
                ky_main=ky_vals_nyc,
                kx_pad=kx_vals_pad,
                ky_pad=ky_vals_pad,
            )
        )
    elif phi_raw_c64.size == expected_main:
        phi_nyc = phi_raw_c64.reshape((nz, nx, nyc)).transpose(2, 1, 0)
    else:
        phi_nyc = _reshape_gx(
            _load_bin_complex(phi_path, (1, 1, 1, nyc, nx, nz)),
            nspec=1,
            nl=1,
            nm=1,
            nyc=nyc,
            nx=nx,
            nz=nz,
        )[0, 0, 0, ...]
    phi = _expand_ky(phi_nyc[None, ...], nyc=nyc)[0]
    apar_path = _pick_first_existing(
        args.gx_dir / "nl_apar.bin", args.gx_dir / "apar.bin"
    )
    bpar_path = _pick_first_existing(
        args.gx_dir / "nl_bpar.bin", args.gx_dir / "bpar.bin"
    )
    apar = None
    bpar = None
    if apar_path is not None:
        apar_raw = np.fromfile(apar_path, dtype=np.complex64)
        if apar_raw.size == expected_pad:
            apar_pad = apar_raw.reshape((nz, nx_pad, nyc_pad)).transpose(2, 1, 0)
            apar = np.asarray(
                _extract_spectral_kxky(
                    jnp.asarray(apar_pad),
                    kx_main=kx_vals,
                    ky_main=ky_vals_nyc,
                    kx_pad=kx_vals_pad,
                    ky_pad=ky_vals_pad,
                )
            )
        elif apar_raw.size == expected_main:
            apar = apar_raw.reshape((nz, nx, nyc)).transpose(2, 1, 0)
        else:
            apar = _reshape_gx(
                _load_bin_complex(apar_path, (1, 1, 1, nyc, nx, nz)),
                nspec=1,
                nl=1,
                nm=1,
                nyc=nyc,
                nx=nx,
                nz=nz,
            )[0, 0, 0, ...]
        apar = _expand_ky(apar[None, ...], nyc=nyc)[0]
    if bpar_path is not None:
        bpar_raw = np.fromfile(bpar_path, dtype=np.complex64)
        if bpar_raw.size == expected_pad:
            bpar_pad = bpar_raw.reshape((nz, nx_pad, nyc_pad)).transpose(2, 1, 0)
            bpar = np.asarray(
                _extract_spectral_kxky(
                    jnp.asarray(bpar_pad),
                    kx_main=kx_vals,
                    ky_main=ky_vals_nyc,
                    kx_pad=kx_vals_pad,
                    ky_pad=ky_vals_pad,
                )
            )
        elif bpar_raw.size == expected_main:
            bpar = bpar_raw.reshape((nz, nx, nyc)).transpose(2, 1, 0)
        else:
            bpar = _reshape_gx(
                _load_bin_complex(bpar_path, (1, 1, 1, nyc, nx, nz)),
                nspec=1,
                nl=1,
                nm=1,
                nyc=nyc,
                nx=nx,
                nz=nz,
            )[0, 0, 0, ...]
        bpar = _expand_ky(bpar[None, ...], nyc=nyc)[0]

    cfg: Any
    if args.config is not None:
        cfg, geom, grid, params, term_cfg = _build_runtime_compare_context(
            args.config,
            nx=nx,
            ny_full=ny_full,
            nz=nz,
            nl=nl,
            nm=nm,
            ky_vals_nyc=ky_vals_nyc,
            y0_override=None if args.y0 is None else float(args.y0),
        )
        ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - float(args.ky))))
        params = _slice_species_params(params, nspec, species_index=args.species_index)
    else:
        use_gx_spacing = ky_vals_nyc.size > 1 and kx_vals.size > 1
        if use_gx_spacing and ky_vals_nyc.size > 1:
            delta_ky = float(ky_vals_nyc[1] - ky_vals_nyc[0])
            y0 = 1.0 / delta_ky if delta_ky != 0.0 else args.y0
        else:
            y0 = args.y0
        if use_gx_spacing and kx_vals.size > 1:
            delta_kx = float(kx_vals[1] - kx_vals[0])
            Lx = float(2.0 * np.pi / delta_kx) if delta_kx != 0.0 else args.Lx
        else:
            Lx = args.Lx
        if args.boundary == "linked" and y0 is not None:
            geom_tmp = SAlphaGeometry.from_config(CycloneBaseCase().geometry)
            theta_tmp = jnp.linspace(-jnp.pi, jnp.pi, nz, endpoint=False)
            _gds2_tmp, gds21_tmp, gds22_tmp = geom_tmp.metric_coeffs(theta_tmp)
            gds21_min = float(gds21_tmp[0]) if np.ndim(gds21_tmp) else float(gds21_tmp)
            gds22_min = float(gds22_tmp[0]) if np.ndim(gds22_tmp) else float(gds22_tmp)
            shat = float(geom_tmp.s_hat)
            twist_shift_geo_fac = 0.0
            if gds22_min != 0.0:
                twist_shift_geo_fac = float(2.0 * shat * gds21_min / gds22_min)
            if twist_shift_geo_fac != 0.0:
                jtwist = int(np.round(twist_shift_geo_fac))
                if jtwist == 0:
                    jtwist = 1
                x0 = float(y0) * abs(jtwist) / abs(twist_shift_geo_fac)
                Lx = float(2.0 * np.pi * x0)
        default_cfg: Any
        if args.case == "kbm":
            default_cfg = KBMBaseCase()
            ntheta_use = (
                args.ntheta if args.ntheta is not None else default_cfg.grid.ntheta
            )
            nperiod_use = (
                args.nperiod if args.nperiod is not None else default_cfg.grid.nperiod
            )
            cfg = KBMBaseCase(
                grid=GridConfig(
                    Nx=nx,
                    Ny=ny_full,
                    Nz=nz,
                    Lx=Lx,
                    Ly=args.Ly,
                    boundary=args.boundary,
                    y0=y0,
                    ntheta=ntheta_use,
                    nperiod=nperiod_use,
                )
            )
            geom = SAlphaGeometry.from_config(cfg.geometry)
            grid = build_spectral_grid(cfg.grid)
            ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - float(args.ky))))
            params = _two_species_params(
                cfg.model,
                kpar_scale=float(geom.gradpar()),
                omega_d_scale=KBM_OMEGA_D_SCALE,
                omega_star_scale=KBM_OMEGA_STAR_SCALE,
                rho_star=KBM_RHO_STAR,
                nhermite=nm,
                beta_override=cfg.model.beta,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
            )
            params = _slice_species_params(
                params, nspec, species_index=args.species_index
            )
            term_cfg = TermConfig(nonlinear=1.0, apar=1.0, bpar=0.0)
        else:
            default_cfg = CycloneBaseCase()
            ntheta_use = (
                args.ntheta if args.ntheta is not None else default_cfg.grid.ntheta
            )
            nperiod_use = (
                args.nperiod if args.nperiod is not None else default_cfg.grid.nperiod
            )
            cfg = CycloneBaseCase(
                grid=GridConfig(
                    Nx=nx,
                    Ny=ny_full,
                    Nz=nz,
                    Lx=Lx,
                    Ly=args.Ly,
                    boundary=args.boundary,
                    y0=y0,
                    ntheta=ntheta_use,
                    nperiod=nperiod_use,
                )
            )
            geom = SAlphaGeometry.from_config(cfg.geometry)
            grid = build_spectral_grid(cfg.grid)
            ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - float(args.ky))))
            params = LinearParams(
                fprim=cfg.model.fprim,
                tprim=cfg.model.tprim_i,
                tprim_e=cfg.model.tprim_e,
                omega_d_scale=CYCLONE_OMEGA_D_SCALE,
                omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
                rho_star=CYCLONE_RHO_STAR,
                kpar_scale=float(geom.gradpar()),
                nu=cfg.model.nu_i,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
            )
            params = _apply_reference_hypercollisions(params, nhermite=nm)
            params = _slice_species_params(
                params, nspec, species_index=args.species_index
            )
            term_cfg = TermConfig(nonlinear=1.0, apar=1.0, bpar=1.0)
    cache = build_linear_cache(grid, geom, params, nl, nm)
    roots_ref = cache.laguerre_roots
    if muB_roots is not None and muB_roots.size == roots_ref.size:
        roots_ref = jnp.asarray(muB_roots, dtype=roots_ref.dtype)
    pad_cache = None
    if phi_pad is not None and (nx_pad != nx or ny_full_pad != ny_full):
        pad_cfg = GridConfig(
            Nx=nx_pad,
            Ny=ny_full_pad,
            Nz=nz,
            Lx=Lx,
            Ly=args.Ly,
            boundary=args.boundary,
            y0=y0,
            ntheta=cfg.grid.ntheta,
            nperiod=cfg.grid.nperiod,
        )
        pad_grid = build_spectral_grid(pad_cfg)
        pad_cache = build_linear_cache(pad_grid, geom, params, nl, nm)

    gx_dg_dx = _pick_species_dump(args.gx_dir, "nl_dg_dx", args.species_index)
    gx_dg_dy = _pick_species_dump(args.gx_dir, "nl_dg_dy", args.species_index)
    gx_dj_dx = args.gx_dir / "nl_dJ0phi_dx.bin"
    gx_dj_dy = args.gx_dir / "nl_dJ0phi_dy.bin"
    have_derivs = (
        gx_dg_dx is not None
        and gx_dg_dy is not None
        and gx_dj_dx.exists()
        and gx_dj_dy.exists()
    )
    ref_dg_dx = None
    ref_dg_dy = None
    ref_dj_dx = None
    ref_dj_dy = None
    if have_derivs:
        raw = np.fromfile(gx_dj_dx, dtype=np.float32)
        expected = nj * ny_full_pad * nx_pad * nz
        if raw.size == expected:
            # GX flat index: idxyzj = idy + ny*(idx + nx*(idz + nz*idj))
            # C-order view is (j, z, x, y); transpose to (j, y, x, z).
            ref_dj_dx = raw.reshape((nj, nz, nx_pad, ny_full_pad)).transpose(0, 3, 2, 1)
        raw = np.fromfile(gx_dj_dy, dtype=np.float32)
        if raw.size == expected:
            ref_dj_dy = raw.reshape((nj, nz, nx_pad, ny_full_pad)).transpose(0, 3, 2, 1)
        assert gx_dg_dx is not None
        raw = np.fromfile(gx_dg_dx, dtype=np.float32)
        expected_g = nm * nj * ny_full_pad * nx_pad * nz
        if raw.size == expected_g:
            # GX flat index: ig = idxyzj + (nx*ny*nz*nj) * idm
            # C-order view is (m, j, z, x, y); transpose to (m, j, y, x, z).
            ref_dg_dx = raw.reshape((nm, nj, nz, nx_pad, ny_full_pad)).transpose(
                0, 1, 4, 3, 2
            )
        assert gx_dg_dy is not None
        raw = np.fromfile(gx_dg_dy, dtype=np.float32)
        if raw.size == expected_g:
            ref_dg_dy = raw.reshape((nm, nj, nz, nx_pad, ny_full_pad)).transpose(
                0, 1, 4, 3, 2
            )
        have_derivs = (
            ref_dg_dx is not None
            and ref_dg_dy is not None
            and ref_dj_dx is not None
            and ref_dj_dy is not None
        )

    def _rms_rel(ref: np.ndarray, test: np.ndarray) -> float:
        diff = test - ref
        rms = np.sqrt(np.mean(diff * diff))
        rms_ref = np.sqrt(np.mean(ref * ref)) + 1.0e-30
        return float(rms / rms_ref)

    def _prepare_order(order: str):
        g_np = _apply_kx_order(g_state, order=order, kx_axis=-2)
        phi_np = _apply_kx_order(phi, order=order, kx_axis=-2)
        apar_np = (
            _apply_kx_order(apar, order=order, kx_axis=-2) if apar is not None else None
        )
        bpar_np = (
            _apply_kx_order(bpar, order=order, kx_axis=-2) if bpar is not None else None
        )
        kx_grid = _apply_kx_order(kx_grid_full, order=order, kx_axis=1)
        return g_np, phi_np, apar_np, bpar_np, kx_grid, ky_grid_full

    def _prepare_order_nyc(order: str):
        g_np = _apply_kx_order(g_state_nyc, order=order, kx_axis=-2)
        phi_np = _apply_kx_order(phi_nyc, order=order, kx_axis=-2)
        apar_np = (
            _apply_kx_order(apar[:nyc, ...], order=order, kx_axis=-2)
            if apar is not None
            else None
        )
        bpar_np = (
            _apply_kx_order(bpar[:nyc, ...], order=order, kx_axis=-2)
            if bpar is not None
            else None
        )
        kx_grid = _apply_kx_order(kx_grid_nyc, order=order, kx_axis=1)
        b_nyc_local = np.asarray(cache.b[:, :nyc, :, :])
        b_nyc_local = _apply_kx_order(b_nyc_local, order=order, kx_axis=-2)
        b_dump_local = None
        if kperp2_dump is not None and rho2s_dump is not None:
            b_dump_local = (
                rho2s_dump
                * _apply_kx_order(kperp2_dump[None, ...], order=order, kx_axis=-2)[0]
            )
        return (
            g_np,
            phi_np,
            apar_np,
            bpar_np,
            kx_grid,
            ky_grid_nyc,
            b_nyc_local,
            b_dump_local,
        )

    def _compare_derivs(order: str) -> float:
        (
            g_np,
            phi_np,
            _apar_np,
            _bpar_np,
            kx_grid,
            ky_grid,
            b_nyc_local,
            b_dump_local,
        ) = _prepare_order_nyc(order)
        kx_main_ord = _apply_kx_order_1d(kx_vals, order=order)
        kx_pad_ord = _apply_kx_order_1d(kx_vals_pad, order=order)
        ky_grid_pad_ord, kx_grid_pad_ord = np.meshgrid(
            ky_vals_pad, kx_pad_ord, indexing="ij"
        )
        g_mu = _gx_grid_convention(
            _laguerre_to_grid(jnp.asarray(g_np), cache.laguerre_to_grid), roots_ref
        )
        b_nyc = b_dump_local[None, ...] if b_dump_local is not None else b_nyc_local
        chi_phi = _laguerre_j0_field(
            jnp.asarray(phi_np.astype(np.complex64)),
            b_nyc,
            roots_ref,
            1.0,
        )
        g_mu_pad = _pad_spectral_kxky(
            g_mu,
            kx_main=kx_main_ord,
            ky_main=ky_vals_nyc,
            kx_pad=kx_pad_ord,
            ky_pad=ky_vals_pad,
        )
        chi_phi_pad = _pad_spectral_kxky(
            chi_phi,
            kx_main=kx_main_ord,
            ky_main=ky_vals_nyc,
            kx_pad=kx_pad_ord,
            ky_pad=ky_vals_pad,
        )
        dchi_dx, dchi_dy = _grad_xy_real(
            chi_phi_pad,
            kx_grid=jnp.asarray(kx_grid_pad_ord),
            ky_grid=jnp.asarray(ky_grid_pad_ord),
            ny_full=ny_full_pad,
        )
        dG_dx, dG_dy = _grad_xy_real(
            g_mu_pad,
            kx_grid=jnp.asarray(kx_grid_pad_ord),
            ky_grid=jnp.asarray(ky_grid_pad_ord),
            ny_full=ny_full_pad,
        )
        test_dj_dx = np.asarray(dchi_dx[0])
        test_dj_dy = np.asarray(dchi_dy[0])
        test_dg_dx = np.asarray(dG_dx[0]).transpose(1, 0, 2, 3, 4)
        test_dg_dy = np.asarray(dG_dy[0]).transpose(1, 0, 2, 3, 4)
        assert ref_dj_dx is not None
        assert ref_dj_dy is not None
        assert ref_dg_dx is not None
        assert ref_dg_dy is not None
        rms = (
            _rms_rel(ref_dj_dx, test_dj_dx)
            + _rms_rel(ref_dj_dy, test_dj_dy)
            + _rms_rel(ref_dg_dx, test_dg_dx)
            + _rms_rel(ref_dg_dy, test_dg_dy)
        )
        print(f"deriv rms sum ({order}) = {rms:.3e}")
        return rms

    order = args.kx_order.lower()
    if order == "auto" and have_derivs:
        candidates = ["native", "fftshift", "ifftshift"]
        scores = {cand: _compare_derivs(cand) for cand in candidates}
        order = min(scores, key=scores.__getitem__)
        print(f"Selected kx order: {order}")
    elif order == "auto":
        order = "native"
    kx_vals_ordered = _apply_kx_order_1d(kx_vals, order=order)
    kx_vals_pad_ordered = _apply_kx_order_1d(kx_vals_pad, order=order)
    ky_grid_pad_ordered, kx_grid_pad_ordered = np.meshgrid(
        ky_vals_pad, kx_vals_pad_ordered, indexing="ij"
    )
    (
        g_state_cmp,
        phi_cmp,
        apar_cmp,
        bpar_cmp,
        kx_grid_cmp,
        ky_grid_cmp,
    ) = _prepare_order(order)
    dealias_mask_cmp = _resolve_dealias_mask(
        cache.dealias_mask, ny=phi_cmp.shape[-3], nx=phi_cmp.shape[-2]
    )

    b_for_comps = cache.b
    if kperp2_dump is not None and rho2s_dump is not None:
        b_dump_nyc = rho2s_dump * _apply_kx_order(kperp2_dump, order=order, kx_axis=-2)
        b_dump_full = _expand_ky(b_dump_nyc[None, ...], nyc=nyc)[0]
        ns_cache = int(np.asarray(cache.b).shape[0])
        b_for_comps = jnp.asarray(
            np.broadcast_to(b_dump_full[None, ...], (ns_cache,) + b_dump_full.shape),
            dtype=cache.b.dtype,
        )
    G = jnp.asarray(g_state_cmp.astype(np.complex64))
    comps = nonlinear_em_components(
        G,
        phi=jnp.asarray(phi_cmp.astype(np.complex64)),
        apar=jnp.asarray(apar_cmp.astype(np.complex64))
        if apar_cmp is not None
        else None,
        bpar=jnp.asarray(bpar_cmp.astype(np.complex64))
        if bpar_cmp is not None
        else None,
        Jl=cache.Jl,
        JlB=cache.JlB,
        tz=jnp.asarray(params.tz),
        vth=jnp.asarray(params.vth),
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        kx_grid=jnp.asarray(kx_grid_cmp),
        ky_grid=jnp.asarray(ky_grid_cmp),
        dealias_mask=dealias_mask_cmp,
        kxfac=cache.kxfac,
        weight=term_cfg.nonlinear,
        apar_weight=term_cfg.apar,
        bpar_weight=term_cfg.bpar,
        laguerre_to_grid=cache.laguerre_to_grid,
        laguerre_to_spectral=cache.laguerre_to_spectral,
        laguerre_roots=roots_ref,
        b=b_for_comps,
    )

    # Real-space bracket comparison if GX dump is available (use Nyc path).
    (
        g_nyc,
        phi_nyc_ord,
        apar_nyc_ord,
        _bpar_nyc_ord,
        kx_grid_nyc_ord,
        ky_grid_nyc_ord,
        b_nyc_ord,
        b_dump_ord,
    ) = _prepare_order_nyc(order)
    g_mu_nyc = _gx_grid_convention(
        _laguerre_to_grid(jnp.asarray(g_nyc), cache.laguerre_to_grid), roots_ref
    )
    b_nyc = b_dump_ord[None, ...] if b_dump_ord is not None else b_nyc_ord
    if phi_pad is not None:
        phi_pad_ord = _apply_kx_order(phi_pad, order=order, kx_axis=-2)
        if pad_cache is not None:
            b_pad = pad_cache.b[:, :nyc_pad, :, :]
        else:
            b_pad = _pad_spectral_kxky(
                b_nyc,
                kx_main=kx_vals_ordered,
                ky_main=ky_vals_nyc,
                kx_pad=kx_vals_pad_ordered,
                ky_pad=ky_vals_pad,
            )
        chi_phi_nyc = _laguerre_j0_field(
            jnp.asarray(phi_pad_ord.astype(np.complex64)),
            b_pad,
            roots_ref,
            1.0,
        )
    else:
        chi_phi_nyc = _laguerre_j0_field(
            jnp.asarray(phi_nyc_ord.astype(np.complex64)),
            b_nyc,
            roots_ref,
            1.0,
        )
    g_mu_pad = _pad_spectral_kxky(
        g_mu_nyc,
        kx_main=kx_vals_ordered,
        ky_main=ky_vals_nyc,
        kx_pad=kx_vals_pad_ordered,
        ky_pad=ky_vals_pad,
    )
    if phi_pad is not None:
        chi_phi_pad = chi_phi_nyc
    else:
        chi_phi_pad = _pad_spectral_kxky(
            chi_phi_nyc,
            kx_main=kx_vals_ordered,
            ky_main=ky_vals_nyc,
            kx_pad=kx_vals_pad_ordered,
            ky_pad=ky_vals_pad,
        )
    bracket_real = _bracket_real(
        g_mu_pad,
        chi_phi_pad,
        kx_grid=jnp.asarray(kx_grid_pad_ordered),
        ky_grid=jnp.asarray(ky_grid_pad_ordered),
        ny_full=ny_full_pad,
    )
    if have_derivs:
        (
            g_nyc,
            phi_nyc_ord,
            _apar_nyc_ord,
            _bpar_nyc_ord,
            kx_grid_nyc_ord,
            ky_grid_nyc_ord,
            b_nyc_ord,
            b_dump_ord,
        ) = _prepare_order_nyc(order)
        g_mu_nyc = _gx_grid_convention(
            _laguerre_to_grid(jnp.asarray(g_nyc), cache.laguerre_to_grid), roots_ref
        )
        b_nyc = b_dump_ord[None, ...] if b_dump_ord is not None else b_nyc_ord
        if phi_pad is not None:
            phi_pad_ord = _apply_kx_order(phi_pad, order=order, kx_axis=-2)
            if pad_cache is not None:
                b_pad = pad_cache.b[:, :nyc_pad, :, :]
            else:
                b_pad = _pad_spectral_kxky(
                    b_nyc,
                    kx_main=kx_vals_ordered,
                    ky_main=ky_vals_nyc,
                    kx_pad=kx_vals_pad_ordered,
                    ky_pad=ky_vals_pad,
                )
            chi_phi_nyc = _laguerre_j0_field(
                jnp.asarray(phi_pad_ord.astype(np.complex64)),
                b_pad,
                roots_ref,
                1.0,
            )
        else:
            chi_phi_nyc = _laguerre_j0_field(
                jnp.asarray(phi_nyc_ord.astype(np.complex64)),
                b_nyc,
                roots_ref,
                1.0,
            )
        g_mu_pad = _pad_spectral_kxky(
            g_mu_nyc,
            kx_main=kx_vals_ordered,
            ky_main=ky_vals_nyc,
            kx_pad=kx_vals_pad_ordered,
            ky_pad=ky_vals_pad,
        )
        if phi_pad is not None:
            chi_phi_pad = chi_phi_nyc
        else:
            chi_phi_pad = _pad_spectral_kxky(
                chi_phi_nyc,
                kx_main=kx_vals_ordered,
                ky_main=ky_vals_nyc,
                kx_pad=kx_vals_pad_ordered,
                ky_pad=ky_vals_pad,
            )
        dchi_dx, dchi_dy = _grad_xy_real(
            chi_phi_pad,
            kx_grid=jnp.asarray(kx_grid_pad_ordered),
            ky_grid=jnp.asarray(ky_grid_pad_ordered),
            ny_full=ny_full_pad,
        )
        dG_dx, dG_dy = _grad_xy_real(
            g_mu_pad,
            kx_grid=jnp.asarray(kx_grid_pad_ordered),
            ky_grid=jnp.asarray(ky_grid_pad_ordered),
            ny_full=ny_full_pad,
        )
        test_dj_dx = np.asarray(dchi_dx[0])
        test_dj_dy = np.asarray(dchi_dy[0])
        test_dg_dx = np.asarray(dG_dx[0]).transpose(1, 0, 2, 3, 4)
        test_dg_dy = np.asarray(dG_dy[0]).transpose(1, 0, 2, 3, 4)
        if ref_dj_dx is not None and ref_dj_dy is not None:
            num_dx = np.sum(ref_dj_dx * test_dj_dx)
            den_dx = np.sum(test_dj_dx * test_dj_dx) + 1.0e-30
            scale_dx = float(num_dx / den_dx)
            num_dy = np.sum(ref_dj_dy * test_dj_dy)
            den_dy = np.sum(test_dj_dy * test_dj_dy) + 1.0e-30
            scale_dy = float(num_dy / den_dy)
            print(f"best-fit scale dJ0phi_dx={scale_dx:.3e} dJ0phi_dy={scale_dy:.3e}")
            rms_dx_flip = _rms_rel(ref_dj_dx, -test_dj_dx)
            rms_dy_flip = _rms_rel(ref_dj_dy, -test_dj_dy)
            print(
                f"rms dJ0phi_dx flip={rms_dx_flip:.3e} dJ0phi_dy flip={rms_dy_flip:.3e}"
            )
        assert ref_dj_dx is not None
        assert ref_dj_dy is not None
        assert ref_dg_dx is not None
        assert ref_dg_dy is not None
        _summary("dJ0phi_dx", ref_dj_dx, test_dj_dx)
        _summary("dJ0phi_dy", ref_dj_dy, test_dj_dy)
        _summary("dg_dx", ref_dg_dx, test_dg_dx)
        _summary("dg_dy", ref_dg_dy, test_dg_dy)
    gx_bracket_real = _pick_species_dump(
        args.gx_dir, "nl_bracket_phi_real", args.species_index
    )
    if gx_bracket_real is not None and gx_bracket_real.exists():
        raw_real = np.fromfile(gx_bracket_real, dtype=np.float32)
        denom = nj * nz * ny_full_pad * nx_pad
        m_tot = int(raw_real.size // denom) if denom > 0 else 0
        if denom == 0 or m_tot <= 0 or raw_real.size % denom != 0:
            print(
                f"Skipping bracket_phi_real: size {raw_real.size} not divisible by {denom}"
            )
        else:
            m_ghost = max(0, (m_tot - nm) // 2)
            if m_tot != nm + 2 * m_ghost:
                print(
                    f"Skipping bracket_phi_real: unexpected m_tot {m_tot} for nm {nm}"
                )
            else:
                # GX real bracket uses idxyzj memory ordering with y fastest:
                # C-order view: (m, j, z, x, y) -> compare as (m, j, y, x, z).
                ref_real = raw_real.reshape((m_tot, nj, nz, nx_pad, ny_full_pad))
                if m_ghost > 0:
                    ref_real = ref_real[m_ghost:-m_ghost]
                ref_real = ref_real.transpose(0, 1, 4, 3, 2)
                # bracket_real[0]: (Nj, Nm, Ny, Nx, Nz) -> (Nm, Nj, Ny, Nx, Nz)
                test_real = np.asarray(bracket_real[0]).transpose(1, 0, 2, 3, 4)
                _summary("bracket_real", ref_real, test_real)

    if rho2s_dump is not None:
        rho_param = params.rho
        if isinstance(rho_param, (float, int)):
            rho2_ref = float(rho_param) ** 2
        else:
            rho2_ref = float(np.asarray(rho_param)[0] ** 2)
        print(f"GX rho2s dump: {rho2s_dump:.6e}")
        print(f"GKX rho2s (species {args.species_index}): {rho2_ref:.6e}")
    if kperp2_dump is not None and rho2s_dump is not None:
        b_dump = rho2s_dump * kperp2_dump
        b_cache = np.asarray(cache.b[0, :nyc, :, :])
        _summary("b_dump", b_dump, b_cache)

    gx_j0phi = args.gx_dir / "nl_j0phi.bin"
    if gx_j0phi.exists():
        raw_j0 = np.fromfile(gx_j0phi, dtype=np.complex64)
        expected_nyc = nj * nz * nyc * nx
        expected_full = nj * nz * ny_full * nx
        if raw_j0.size == expected_nyc:
            # GX flat index: ig = idxyz + (nx*nyc*nz)*idj, idxyz = idy + nyc*(idx + nx*idz)
            # C-order view is (j, z, x, y); transpose to (j, y, x, z).
            j0phi = raw_j0.reshape((nj, nz, nx, nyc)).transpose(0, 3, 2, 1)
            ky_idx = int(np.argmin(np.abs(ky_vals_nyc - float(args.ky))))
            (
                g_nyc,
                phi_nyc_ord,
                apar_nyc_ord,
                _bpar_nyc_ord,
                _,
                _,
                b_nyc_ord,
                b_dump_ord,
            ) = _prepare_order_nyc(order)
            b_nyc = b_dump_ord[None, ...] if b_dump_ord is not None else b_nyc_ord
            chi_phi_nyc = _laguerre_j0_field(
                jnp.asarray(phi_nyc_ord.astype(np.complex64)),
                b_nyc,
                roots_ref,
                1.0,
            )
            test_j0 = np.asarray(chi_phi_nyc[0, :, ky_idx, :, :])
            ref_j0 = j0phi[:, ky_idx, :, :]
            _summary("j0phi", ref_j0, test_j0)
        elif raw_j0.size == expected_full:
            # Full-ky layout (Ny): C-order view is (j, z, x, y)
            j0phi_full = raw_j0.reshape((nj, nz, nx, ny_full)).transpose(0, 3, 2, 1)
            ky_idx = int(np.argmin(np.abs(ky_vals_full - float(args.ky))))
            (
                g_nyc,
                phi_nyc_ord,
                apar_nyc_ord,
                _bpar_nyc_ord,
                _,
                _,
                b_nyc_ord,
                b_dump_ord,
            ) = _prepare_order_nyc(order)
            b_nyc = b_dump_ord[None, ...] if b_dump_ord is not None else b_nyc_ord
            chi_phi_nyc = _laguerre_j0_field(
                jnp.asarray(phi_nyc_ord.astype(np.complex64)),
                b_nyc,
                roots_ref,
                1.0,
            )
            chi_phi_full = _expand_ky(np.asarray(chi_phi_nyc[0]), nyc=nyc)
            test_j0 = chi_phi_full[:, ky_idx, :, :]
            ref_j0 = j0phi_full[:, ky_idx, :, :]
            _summary("j0phi", ref_j0, test_j0)
        else:
            print(
                f"Skipping j0phi: size {raw_j0.size} != {expected_nyc} or {expected_full}"
            )

    gx_j0apar = args.gx_dir / "nl_j0apar.bin"
    if gx_j0apar.exists() and apar is not None:
        raw_j0 = np.fromfile(gx_j0apar, dtype=np.complex64)
        if raw_j0.size == expected_nyc:
            j0apar = raw_j0.reshape((nj, nz, nx, nyc)).transpose(0, 3, 2, 1)
            ky_idx = int(np.argmin(np.abs(ky_vals_nyc - float(args.ky))))
            (
                g_nyc,
                _phi_nyc_ord,
                apar_nyc_ord,
                _bpar_nyc_ord,
                _,
                _,
                b_nyc_ord,
                b_dump_ord,
            ) = _prepare_order_nyc(order)
            b_nyc = b_dump_ord[None, ...] if b_dump_ord is not None else b_nyc_ord
            chi_apar_nyc = _laguerre_j0_field(
                jnp.asarray(apar_nyc_ord.astype(np.complex64)),
                b_nyc,
                roots_ref,
                1.0,
            )
            test_j0 = np.asarray(chi_apar_nyc[0, :, ky_idx, :, :])
            ref_j0 = j0apar[:, ky_idx, :, :]
            _summary("j0apar", ref_j0, test_j0)
        elif raw_j0.size == expected_full:
            j0apar_full = raw_j0.reshape((nj, nz, nx, ny_full)).transpose(0, 3, 2, 1)
            ky_idx = int(np.argmin(np.abs(ky_vals_full - float(args.ky))))
            (
                g_nyc,
                _phi_nyc_ord,
                apar_nyc_ord,
                _bpar_nyc_ord,
                _,
                _,
                b_nyc_ord,
                b_dump_ord,
            ) = _prepare_order_nyc(order)
            b_nyc = b_dump_ord[None, ...] if b_dump_ord is not None else b_nyc_ord
            chi_apar_nyc = _laguerre_j0_field(
                jnp.asarray(apar_nyc_ord.astype(np.complex64)),
                b_nyc,
                roots_ref,
                1.0,
            )
            chi_apar_full = _expand_ky(np.asarray(chi_apar_nyc[0]), nyc=nyc)
            test_j0 = chi_apar_full[:, ky_idx, :, :]
            ref_j0 = j0apar_full[:, ky_idx, :, :]
            _summary("j0apar", ref_j0, test_j0)
        else:
            print(
                f"Skipping j0apar: size {raw_j0.size} != {expected_nyc} or {expected_full}"
            )

    exb_total = np.asarray(comps["exb_phi"]) + np.asarray(comps["exb_bpar"])
    gx_map = {
        "exb_total": ("nl_exb_phi", exb_total),
        "exb_bpar": ("nl_exb_bpar", np.asarray(comps["exb_bpar"])),
        "bracket_apar": ("nl_bracket_apar", np.asarray(comps["bracket_apar"])),
        "flutter": ("nl_flutter", np.asarray(comps["flutter"])),
        "total": ("nl_total", np.asarray(comps["total"])),
    }
    for key, (stem, test_arr) in gx_map.items():
        path = _pick_species_dump(args.gx_dir, stem, args.species_index)
        if path is None:
            print(f"Skipping {key}: no dump found for {stem}")
            continue
        ref = _reshape_gx(
            _load_bin(path, gx_shape),
            nspec=nspec,
            nl=nl,
            nm=nm,
            nyc=nyc,
            nx=nx,
            nz=nz,
        )
        ref = ref[:, :, :, ky_idx : ky_idx + 1, :, :]
        test_slice = test_arr[:, :, :, ky_index : ky_index + 1, :, :]
        _summary(key, ref, test_slice)

    if (
        _pick_species_dump(args.gx_dir, "nl_flutter", args.species_index) is None
        and _pick_species_dump(args.gx_dir, "nl_total", args.species_index) is not None
        and _pick_species_dump(args.gx_dir, "nl_exb_phi", args.species_index)
        is not None
    ):
        total_path = _pick_species_dump(args.gx_dir, "nl_total", args.species_index)
        exb_phi_path = _pick_species_dump(args.gx_dir, "nl_exb_phi", args.species_index)
        assert total_path is not None
        assert exb_phi_path is not None
        ref_total = _reshape_gx(
            _load_bin(total_path, gx_shape),
            nspec=nspec,
            nl=nl,
            nm=nm,
            nyc=nyc,
            nx=nx,
            nz=nz,
        )
        ref_exb_phi = _reshape_gx(
            _load_bin(exb_phi_path, gx_shape),
            nspec=nspec,
            nl=nl,
            nm=nm,
            nyc=nyc,
            nx=nx,
            nz=nz,
        )
        ref_flutter = ref_total - ref_exb_phi
        exb_bpar_path = _pick_species_dump(
            args.gx_dir, "nl_exb_bpar", args.species_index
        )
        if exb_bpar_path is not None and exb_bpar_path.exists():
            ref_exb_bpar = _reshape_gx(
                _load_bin(exb_bpar_path, gx_shape),
                nspec=nspec,
                nl=nl,
                nm=nm,
                nyc=nyc,
                nx=nx,
                nz=nz,
            )
            ref_flutter = ref_flutter - ref_exb_bpar
        ref_flutter = ref_flutter[:, :, :, ky_idx : ky_idx + 1, :, :]
        test_flutter = np.asarray(comps["flutter"])[
            :, :, :, ky_index : ky_index + 1, :, :
        ]
        _summary("flutter*", ref_flutter, test_flutter)

    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            bracket_phi=np.asarray(comps["exb_phi"]),
            bracket_bpar=np.asarray(comps["exb_bpar"]),
            bracket_apar=np.asarray(comps["bracket_apar"]),
            flutter=np.asarray(comps["flutter"]),
            nl_total=np.asarray(comps["total"]),
        )
        print(f"Wrote {out}")


def main(argv: list[str] | None = None) -> int | None:
    """Dispatch aggregate nonlinear diagnostics or term-level comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("diagnostics", "terms"))
    args, remainder = parser.parse_known_args(argv)
    if args.mode == "diagnostics":
        return run_diagnostics(remainder)
    run_terms(remainder)
    return None


if __name__ == "__main__":
    raise SystemExit(main())

