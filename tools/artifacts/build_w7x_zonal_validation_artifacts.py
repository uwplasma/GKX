#!/usr/bin/env python3
"""Build W7-X zonal response, contract, and state-convention artifacts.

This script does not run simulations. It combines the tracked GKX
W7-X zonal response artifacts with digitized stella/GENE Fig. 11 data from the
González-Jerez et al. W7-X benchmark paper. The output is deliberately
paper-facing but marked as an open audit until the residual and late-envelope
gates close under the literature normalization convention.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from math import ceil
from pathlib import Path
from typing import Any

import matplotlib
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gkx.artifacts.plotting import set_plot_style  # noqa: E402
from gkx.artifacts.io import (  # noqa: E402
    load_diagnostic_time_series,
)
from gkx.core.grid import SpectralGrid, build_spectral_grid  # noqa: E402
from gkx.diagnostics import (  # noqa: E402
    fieldline_quadrature_weights,
    zonal_phi_line_kxt,
    zonal_phi_mode_kxt,
)
from gkx.diagnostics.zonal_validation import (
    kx_token,
    load_w7x_combined_trace_csv,
    reference_mean_trace,
    zonal_flow_response_metrics,
)  # noqa: E402
from gkx.geometry import (  # noqa: E402
    apply_geometry_grid_defaults,
    ensure_flux_tube_geometry_data,
)
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.runtime import (  # noqa: E402
    _build_initial_condition,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from gkx.terms.assembly import compute_fields_cached  # noqa: E402
from gkx.workflows.runtime.config import RuntimeConfig  # noqa: E402
from gkx.workflows.runtime.artifacts import (  # noqa: E402
    run_runtime_nonlinear_with_artifacts,
)
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_OUT = ROOT / "docs" / "_static" / "w7x_zonal_contract_audit.png"
TRACE_OVERLAY_KX = (0.07, 0.30)


def _parse_contract_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-traces",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized.csv",
        help="Digitized stella/GENE Fig. 11 trace CSV.",
    )
    parser.add_argument(
        "--reference-residuals",
        type=Path,
        default=ROOT
        / "docs"
        / "_static"
        / "w7x_zonal_reference_digitized_residuals.csv",
        help="Digitized stella/GENE Fig. 11 inset residual CSV.",
    )
    parser.add_argument(
        "--gkx-summary",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_response_panel.csv",
        help="GKX W7-X zonal summary CSV.",
    )
    parser.add_argument(
        "--gkx-traces",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_response_panel.traces.csv",
        help="Combined GKX W7-X zonal trace CSV from response-panel mode.",
    )
    parser.add_argument(
        "--compare-csv",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_compare.csv",
        help=(
            "Residual/time/envelope CSV from "
            "build_w7x_zonal_reference_artifacts.py compare."
        ),
    )
    parser.add_argument(
        "--out-png", type=Path, default=DEFAULT_CONTRACT_OUT, help="Output PNG path."
    )
    parser.add_argument(
        "--out-csv", type=Path, default=None, help="Output audit CSV path."
    )
    parser.add_argument(
        "--out-json", type=Path, default=None, help="Output audit JSON path."
    )
    parser.add_argument("--residual-rtol", type=float, default=0.10)
    parser.add_argument("--envelope-atol", type=float, default=0.03)
    return parser.parse_args(argv)


def _repo_relative(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def load_audit_rows(
    compare_csv: Path,
    *,
    residual_rtol: float = 0.10,
    envelope_atol: float = 0.03,
) -> list[dict[str, object]]:
    """Build a compact audit table from the tracked W7-X comparison CSV."""

    table = pd.read_csv(compare_csv)
    required = {
        "kx",
        "gkx_residual",
        "reference_residual",
        "reference_min",
        "reference_max",
        "coverage_ratio",
        "residual_abs_error",
        "residual_atol_effective",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{compare_csv} missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for _, item in table.sort_values("kx").iterrows():
        ref = float(item["reference_residual"])
        residual_tol = float(item["residual_atol_effective"]) + float(
            residual_rtol
        ) * abs(ref)
        tail_std = (
            float(item["tail_std"])
            if "tail_std" in table.columns and pd.notna(item["tail_std"])
            else np.nan
        )
        reference_tail_std = (
            float(item["reference_tail_std"])
            if "reference_tail_std" in table.columns
            and pd.notna(item["reference_tail_std"])
            else np.nan
        )
        tail_ratio = (
            tail_std / reference_tail_std
            if np.isfinite(tail_std) and reference_tail_std > 0.0
            else np.nan
        )
        rows.append(
            {
                "kx": float(item["kx"]),
                "gkx_residual": float(item["gkx_residual"]),
                "reference_residual": ref,
                "reference_min": float(item["reference_min"]),
                "reference_max": float(item["reference_max"]),
                "residual_abs_error": float(item["residual_abs_error"]),
                "residual_tolerance": residual_tol,
                "tail_std": tail_std,
                "reference_tail_std": reference_tail_std,
                "tail_std_ratio": tail_ratio,
                "coverage_ratio": float(item["coverage_ratio"]),
                "residual_gate_passed": bool(
                    float(item["residual_abs_error"]) <= residual_tol
                ),
                "tail_gate_passed": bool(
                    np.isfinite(tail_std)
                    and np.isfinite(reference_tail_std)
                    and abs(tail_std - reference_tail_std) <= float(envelope_atol)
                ),
            }
        )
    return rows


def contract_audit_figure(
    rows: list[dict[str, object]],
    reference_traces: pd.DataFrame,
    gkx_traces: Path,
    *,
    overlay_kx: tuple[float, ...] = TRACE_OVERLAY_KX,
) -> plt.Figure:
    """Create the four-panel W7-X zonal-response contract audit figure."""

    if not rows:
        raise ValueError("no W7-X audit rows to plot")
    set_plot_style()
    kx = np.asarray([float(row["kx"]) for row in rows])
    x = np.arange(len(rows))
    xlabels = [f"{value:.2f}" for value in kx]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.fill_between(
        x,
        np.asarray([float(row["reference_min"]) for row in rows]),
        np.asarray([float(row["reference_max"]) for row in rows]),
        color="#8ecae6",
        alpha=0.42,
        label="digitized stella/GENE band",
    )
    ax.plot(
        x,
        [float(row["reference_residual"]) for row in rows],
        marker="o",
        linewidth=2.2,
        color="#1d4e89",
        label="reference mean",
    )
    ax.plot(
        x,
        [float(row["gkx_residual"]) for row in rows],
        marker="s",
        linewidth=2.2,
        color="#c2410c",
        label="GKX",
    )
    ax.set_xticks(x, xlabels)
    ax.set_xlabel(r"$k_x \rho_i$")
    ax.set_ylabel("late residual")
    ax.set_title("Residual level")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    width = 0.34
    ref_tail = np.asarray(
        [float(row["reference_tail_std"]) for row in rows], dtype=float
    )
    obs_tail = np.asarray([float(row["tail_std"]) for row in rows], dtype=float)
    ax.bar(
        x - width / 2,
        ref_tail,
        width=width,
        color="#1d4e89",
        alpha=0.78,
        label="reference",
    )
    ax.bar(
        x + width / 2,
        obs_tail,
        width=width,
        color="#c2410c",
        alpha=0.78,
        label="GKX",
    )
    ax.set_yscale("log")
    ax.set_xticks(x, xlabels)
    ax.set_xlabel(r"$k_x \rho_i$")
    ax.set_ylabel("late-window standard deviation")
    ax.set_title("Late envelope")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    for ax, this_kx in zip(axes[1, :], overlay_kx, strict=True):
        ref_t, ref_y = reference_mean_trace(reference_traces, this_kx)
        obs_t, obs_y = load_w7x_combined_trace_csv(
            gkx_traces, this_kx, normalized=True
        )
        ax.plot(ref_t, ref_y, color="#1d4e89", linewidth=2.0, label="digitized mean")
        ax.plot(
            obs_t,
            obs_y,
            color="#c2410c",
            linewidth=1.7,
            alpha=0.92,
            label="GKX",
        )
        ax.set_xlabel(r"$t v_{ti}/a$")
        ax.set_ylabel(r"$\phi_z/\phi_z(0)$")
        ax.set_title(rf"Trace overlay, $k_x \rho_i={this_kx:.2f}$")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "W7-X test-4 zonal-response contract audit",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    return fig


def write_contract_rows(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_contract_metadata(
    *,
    rows: list[dict[str, object]],
    out_json: Path,
    out_csv: Path,
    out_png: Path,
    args: argparse.Namespace,
) -> None:
    payload = {
        "case": "w7x_zonal_response_contract_audit",
        "validation_status": "open",
        "gate_index_include": False,
        "reference": "Gonzalez-Jerez et al., J. Plasma Phys. 88, 905880310 (2022), W7-X test 4",
        "reference_contract": {
            "observable": "unweighted line-averaged electrostatic potential",
            "normalization": "line-averaged potential normalized to its t=0 line-average value",
            "kx_rhoi": [float(row["kx"]) for row in rows],
        },
        "all_residual_gates_pass": all(
            bool(row["residual_gate_passed"]) for row in rows
        ),
        "all_tail_gates_pass": all(bool(row["tail_gate_passed"]) for row in rows),
        "audit_csv": _repo_relative(out_csv),
        "audit_png": _repo_relative(out_png),
        "reference_traces": _repo_relative(args.reference_traces),
        "reference_residuals": _repo_relative(args.reference_residuals),
        "gkx_summary": _repo_relative(args.gkx_summary),
        "gkx_traces": _repo_relative(args.gkx_traces),
        "compare_csv": _repo_relative(args.compare_csv),
        "rows": rows,
        "notes": (
            "This figure is a diagnostic audit, not a closed validation gate. It intentionally uses "
            "the paper-text line-first normalization rather than the clipped initial value visible in "
            "the published figure. The current artifact reaches the reference time windows, but the "
            "residual and late-envelope mismatches remain open physics/numerics work."
        ),
    }
    out_json.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def run_contract(argv: list[str] | None = None) -> int:
    args = _parse_contract_args(argv)
    out_csv = args.out_csv or args.out_png.with_suffix(".csv")
    out_json = args.out_json or args.out_png.with_suffix(".json")
    rows = load_audit_rows(
        args.compare_csv,
        residual_rtol=float(args.residual_rtol),
        envelope_atol=float(args.envelope_atol),
    )
    reference_traces = pd.read_csv(args.reference_traces)
    fig = contract_audit_figure(rows, reference_traces, args.gkx_traces)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=240, bbox_inches="tight")
    fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    write_contract_rows(rows, out_csv)
    write_contract_metadata(
        rows=rows, out_json=out_json, out_csv=out_csv, out_png=args.out_png, args=args
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {args.out_png}")
    return 0


DEFAULT_STATE_CONFIG = ROOT / "benchmarks" / "runtime_w7x_zonal_response_vmec.toml"
DEFAULT_STATE_OUT = ROOT / "docs" / "_static" / "w7x_zonal_state_convention_audit.png"


def _parse_state_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit W7-X zonal initial-state and observable conventions."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_STATE_CONFIG)
    parser.add_argument("--kx", type=float, default=0.07, help="Target zonal kx rho_i.")
    parser.add_argument("--ky", type=float, default=0.0, help="Target zonal ky rho_i.")
    parser.add_argument(
        "--Nl", type=int, default=None, help="Laguerre count; defaults to run.Nl."
    )
    parser.add_argument(
        "--Nm", type=int, default=None, help="Hermite count; defaults to run.Nm."
    )
    parser.add_argument("--out-png", type=Path, default=DEFAULT_STATE_OUT)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args(argv)


def _selected_indices(
    grid: SpectralGrid, *, ky_target: float, kx_target: float
) -> tuple[int, int]:
    ky = np.asarray(grid.ky, dtype=float)
    kx = np.asarray(grid.kx, dtype=float)
    ky_index = int(np.argmin(np.abs(ky - float(ky_target))))
    positive = np.flatnonzero(kx > 0.0)
    if positive.size:
        kx_index = int(positive[np.argmin(np.abs(kx[positive] - float(kx_target)))])
    else:
        kx_index = int(np.argmin(np.abs(kx - float(kx_target))))
    return ky_index, kx_index


def _expected_paper_phi(cfg: RuntimeConfig, z: np.ndarray) -> np.ndarray:
    center = 0.0
    width = float(cfg.init.gaussian_width)
    envelope = float(cfg.init.gaussian_envelope_constant) + float(
        cfg.init.gaussian_envelope_sine
    ) * np.sin(z - center)
    return float(cfg.init.init_amp) * envelope * np.exp(-(((z - center) / width) ** 2))


def _relative_l2(observed: np.ndarray, expected: np.ndarray) -> float:
    denom = float(np.linalg.norm(expected))
    if denom <= 0.0:
        return float(np.linalg.norm(observed))
    return float(np.linalg.norm(observed - expected) / denom)


def _relative_abs(observed: complex, expected: complex) -> float:
    denom = max(abs(expected), 1.0e-300)
    return float(abs(observed - expected) / denom)


def build_state_audit(
    cfg: RuntimeConfig,
    *,
    kx_target: float,
    ky_target: float = 0.0,
    Nl: int | None = None,
    Nm: int | None = None,
) -> dict[str, object]:
    """Build one state-level W7-X zonal convention audit from the runtime path."""

    raw_nl = 8 if Nl is None else int(Nl)
    raw_nm = 32 if Nm is None else int(Nm)
    geom = build_runtime_geometry(cfg)
    grid_cfg = apply_geometry_grid_defaults(geom, cfg.grid)
    if str(grid_cfg.boundary).lower() != "periodic":
        grid_cfg = replace(grid_cfg, boundary="periodic", jtwist=None, non_twist=True)
    grid_cfg = replace(grid_cfg, Lx=float(2.0 * np.pi / float(kx_target)))
    grid = build_spectral_grid(grid_cfg)
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    ky_index, kx_index = _selected_indices(
        grid, ky_target=ky_target, kx_target=kx_target
    )

    params = build_runtime_linear_params(cfg, Nm=raw_nm, geom=geom_eff)
    cache = build_linear_cache(grid, geom_eff, params, raw_nl, raw_nm)
    terms = build_runtime_term_config(cfg)
    g0 = _build_initial_condition(
        grid,
        geom_eff,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=raw_nl,
        Nm=raw_nm,
        nspecies=sum(1 for species in cfg.species if bool(species.kinetic)),
    )
    fields = compute_fields_cached(
        jnp.asarray(g0), cache, params, terms=terms, use_custom_vjp=False
    )
    phi = np.asarray(fields.phi)
    target_phi = np.asarray(phi[ky_index, kx_index, :])
    expected = _expected_paper_phi(cfg, np.asarray(grid.z, dtype=float))
    vol_fac, _flux_fac = fieldline_quadrature_weights(geom_eff, grid)
    vol_np = np.asarray(vol_fac, dtype=float)
    line_helper = np.asarray(zonal_phi_line_kxt(jnp.asarray(phi), grid))
    mode_helper = np.asarray(zonal_phi_mode_kxt(jnp.asarray(phi), grid, vol_fac))
    manual_line = complex(np.mean(target_phi))
    manual_mode = complex(np.sum(target_phi * vol_np))
    line_value = complex(line_helper[kx_index])
    mode_value = complex(mode_helper[kx_index])

    mask = np.ones(phi.shape, dtype=bool)
    mask[ky_index, kx_index, :] = False
    target_amp = max(float(np.max(np.abs(target_phi))), 1.0e-300)
    non_target_max = float(np.max(np.abs(phi[mask]))) if np.any(mask) else 0.0
    density_seed = np.asarray(g0)[0, 0, 0, ky_index, kx_index, :]

    row = {
        "case": "w7x_zonal_state_convention_audit",
        "kx_target": float(kx_target),
        "kx_selected": float(np.asarray(grid.kx, dtype=float)[kx_index]),
        "ky_target": float(ky_target),
        "ky_selected": float(np.asarray(grid.ky, dtype=float)[ky_index]),
        "kx_index": int(kx_index),
        "ky_index": int(ky_index),
        "Nz": int(grid.z.size),
        "Nl": int(raw_nl),
        "Nm": int(raw_nm),
        "init_amp": float(cfg.init.init_amp),
        "gaussian_width": float(cfg.init.gaussian_width),
        "profile_relative_l2": _relative_l2(target_phi.real, expected),
        "profile_max_abs_error": float(np.max(np.abs(target_phi.real - expected))),
        "profile_max_relative_error": float(
            np.max(np.abs(target_phi.real - expected))
            / max(float(np.max(np.abs(expected))), 1.0e-300)
        ),
        "profile_imag_abs_max": float(np.max(np.abs(target_phi.imag))),
        "non_target_phi_abs_max": non_target_max,
        "non_target_phi_over_target": float(non_target_max / target_amp),
        "line_helper_vs_manual_rel": _relative_abs(line_value, manual_line),
        "mode_helper_vs_manual_rel": _relative_abs(mode_value, manual_mode),
        "line_first_initial_over_init_amp": float(
            abs(manual_line) / max(abs(float(cfg.init.init_amp)), 1.0e-300)
        ),
        "volume_initial_over_init_amp": float(
            abs(manual_mode) / max(abs(float(cfg.init.init_amp)), 1.0e-300)
        ),
        "line_vs_volume_relative_difference": _relative_abs(line_value, mode_value),
        "density_seed_max_over_init_amp": float(
            np.max(np.abs(density_seed)) / max(abs(float(cfg.init.init_amp)), 1.0e-300)
        ),
        "paper_initializer": "phi(theta)=init_amp*exp[-theta^2/gaussian_width^2] for ky=0",
        "paper_observable": "unweighted line average Phi_zonal_line_kxt normalized by its first nonzero sample",
    }
    passed = (
        float(row["profile_relative_l2"]) <= 1.0e-4
        and float(row["profile_imag_abs_max"])
        <= 1.0e-8 * max(abs(float(cfg.init.init_amp)), 1.0e-300)
        and float(row["non_target_phi_over_target"]) <= 1.0e-6
        and float(row["line_helper_vs_manual_rel"]) <= 1.0e-6
        and float(row["mode_helper_vs_manual_rel"]) <= 1.0e-6
    )
    return {
        "row": row,
        "passed": bool(passed),
        "z": np.asarray(grid.z, dtype=float),
        "phi_real": target_phi.real,
        "phi_imag": target_phi.imag,
        "expected_phi": expected,
        "line_value": line_value,
        "mode_value": mode_value,
    }


def state_audit_figure(audit: dict[str, object]) -> plt.Figure:
    """Create the W7-X state-convention audit figure."""

    row = dict(audit["row"])  # type: ignore[arg-type]
    z = np.asarray(audit["z"], dtype=float)
    phi = np.asarray(audit["phi_real"], dtype=float)
    expected = np.asarray(audit["expected_phi"], dtype=float)
    scale = max(float(np.max(np.abs(expected))), 1.0e-300)

    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(
        z,
        expected / scale,
        color="#111827",
        linewidth=2.5,
        label=r"paper $e^{-\theta^2}$",
    )
    ax.plot(
        z,
        phi / scale,
        color="#0f4c81",
        linestyle="--",
        linewidth=2.0,
        label="recovered field solve",
    )
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\phi/\max|\phi_\mathrm{paper}|$")
    ax.set_title("Initial potential profile")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    keys = [
        ("line_first_initial_over_init_amp", "line avg / init_amp"),
        ("volume_initial_over_init_amp", "volume avg / init_amp"),
        ("line_vs_volume_relative_difference", "line-vs-volume rel diff"),
    ]
    ax.bar(
        [label for _key, label in keys],
        [float(row[key]) for key, _label in keys],
        color=["#0f4c81", "#2a9d8f", "#c2410c"],
    )
    ax.set_ylabel("dimensionless")
    ax.set_title("Observable convention")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=18)

    ax = axes[1, 0]
    err_keys = [
        ("profile_relative_l2", r"profile $L^2$"),
        ("profile_max_relative_error", "profile max rel"),
        ("line_helper_vs_manual_rel", "line helper"),
        ("mode_helper_vs_manual_rel", "mode helper"),
        ("non_target_phi_over_target", "off-target"),
    ]
    values = [max(float(row[key]), 1.0e-18) for key, _label in err_keys]
    ax.bar([label for _key, label in err_keys], values, color="#7b2cbf", alpha=0.82)
    ax.set_yscale("log")
    ax.set_ylabel("relative error")
    ax.set_title("State-level closure checks")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 1]
    summary = (
        f"kx selected = {float(row['kx_selected']):.5f}\n"
        f"Nl,Nm = {int(row['Nl'])},{int(row['Nm'])}\n"
        f"line-first/init_amp = {float(row['line_first_initial_over_init_amp']):.5f}\n"
        f"profile L2 rel = {float(row['profile_relative_l2']):.2e}\n"
        f"non-target/target = {float(row['non_target_phi_over_target']):.2e}\n"
        f"state convention = {'closed' if bool(audit['passed']) else 'open'}"
    )
    ax.text(
        0.04, 0.96, summary, transform=ax.transAxes, va="top", ha="left", fontsize=11
    )
    ax.set_axis_off()
    fig.suptitle(
        "W7-X test-4 zonal state and observable convention audit",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    return fig


def write_state_outputs(
    audit: dict[str, object],
    *,
    out_png: Path,
    out_csv: Path,
    out_json: Path,
    config: Path,
) -> None:
    row = dict(audit["row"])  # type: ignore[arg-type]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = state_audit_figure(audit)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(row.keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(row)
    payload = {
        "case": "w7x_zonal_state_convention_audit",
        "validation_status": "state_convention_closed"
        if bool(audit["passed"])
        else "open",
        "gate_index_include": False,
        "config": _repo_relative(config),
        "audit_csv": _repo_relative(out_csv),
        "audit_png": _repo_relative(out_png),
        "reference": "Gonzalez-Jerez et al., J. Plasma Phys. 88, 905880310 (2022), W7-X test 4",
        "row": row,
        "notes": (
            "This state-level audit checks the paper-facing W7-X test-4 convention before time evolution: "
            "the runtime phi initializer recovers the prescribed Gaussian potential profile, the selected "
            "mode is ky=0 at the requested kx, the off-target spectral field is negligible, and the "
            "unweighted line-average observable differs explicitly from the volume-weighted zonal mode. "
            "It closes the initializer/observable convention layer but does not close the separate "
            "long-time recurrence and damping mismatch against the digitized stella/GENE traces."
        ),
    }
    out_json.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def run_state_convention(argv: list[str] | None = None) -> int:
    args = _parse_state_args(argv)
    cfg, raw = load_runtime_from_toml(args.config)
    run_cfg = dict(raw.get("run", {}))
    nl = int(args.Nl) if args.Nl is not None else int(run_cfg.get("Nl", 8))
    nm = int(args.Nm) if args.Nm is not None else int(run_cfg.get("Nm", 32))
    audit = build_state_audit(
        cfg, kx_target=float(args.kx), ky_target=float(args.ky), Nl=nl, Nm=nm
    )
    out_csv = args.out_csv or args.out_png.with_suffix(".csv")
    out_json = args.out_json or args.out_png.with_suffix(".json")
    write_state_outputs(
        audit,
        out_png=args.out_png,
        out_csv=out_csv,
        out_json=out_json,
        config=args.config,
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {args.out_png}")
    return 0


W7X_TEST4_REFERENCE = {
    "paper": "Gonzalez-Jerez et al., J. Plasma Phys. 88, 905880310 (2022)",
    "configuration": "W7-X high-mirror",
    "test": 4,
    "flux_tube": "bean",
    "torflux": 0.64,
    "alpha": 0.0,
    "adiabatic_electrons": True,
    "a_over_LTi": 0.0,
    "a_over_Ln": 0.0,
    "ky": 0.0,
    "kx_rhoi_values": [0.05, 0.07, 0.10, 0.30],
    "observable": "unweighted line-averaged electrostatic potential",
    "nperiod": 4,
    "nz_reference": 512,
    "nvpar_reference": 256,
    "nmu_reference": 32,
    "normalization": "line-averaged potential normalized to its t=0 line-average value",
}


def _parse_response_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "benchmarks" / "runtime_w7x_zonal_response_vmec.toml",
        help="Runtime TOML for the W7-X test-4 zonal-response benchmark.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "tools_out" / "zonal_response" / "w7x_test4_vmec",
        help="Directory for per-kx runtime outputs and extracted traces.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_response_panel.png",
        help="Publication-facing output figure path.",
    )
    parser.add_argument(
        "--kx-values",
        type=float,
        nargs="+",
        default=[0.05, 0.07, 0.10, 0.30],
        help="kx rho_i values for the W7-X test-4 sweep.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.3,
        help="Late-time fraction used for the residual window.",
    )
    parser.add_argument(
        "--initial-fraction",
        type=float,
        default=0.1,
        help="Leading fraction used for the initial normalization window.",
    )
    parser.add_argument(
        "--initial-policy",
        choices=("first_abs", "window_abs_mean"),
        default="first_abs",
        help="Initial normalization convention for the residual/GAM metrics.",
    )
    parser.add_argument(
        "--initial-normalization",
        choices=("init_amp", "line_first"),
        default="line_first",
        help=(
            "Reference normalization for the plotted response. The W7-X test-4 "
            "text normalizes the line-averaged response to its t=0 value, so "
            "the default uses the first nonzero line-average sample. init_amp "
            "is retained for explicit normalization audits."
        ),
    )
    parser.add_argument(
        "--peak-fit-max-peaks",
        type=int,
        default=4,
        help="Maximum number of extrema per branch used in the damping fit.",
    )
    parser.add_argument(
        "--fit-window-tmax",
        type=float,
        default=12.0,
        help=(
            "Upper bound of the shared early-time GAM fit window. This is chosen "
            "to isolate the initial GAM before the slower stellarator-specific oscillation."
        ),
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to runtime times before plotting/extraction. "
            "The default is 1 because runtime samples are already interpreted "
            "on the paper's t v_ti/a axis; non-unit values are for explicit "
            "axis-calibration audits only."
        ),
    )
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help="Reuse any existing per-kx out.nc bundles instead of rerunning them.",
    )
    parser.add_argument(
        "--resume-output",
        action="store_true",
        help=(
            "Continue each per-kx bundle from its matching restart file when it exists, "
            "appending diagnostics to the existing out.nc history."
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override the runtime time step without editing the tracked benchmark TOML.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the number of fixed RK steps without editing the tracked benchmark TOML.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=None,
        help="Override the diagnostic sample stride without editing the tracked benchmark TOML.",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=None,
        help=(
            "Split fixed-step runtime generation into restartable chunks. "
            "This enables fail-fast nonfinite checks during long stability sweeps."
        ),
    )
    parser.add_argument(
        "--Nl",
        type=int,
        default=None,
        help="Override the Laguerre moment count without editing the tracked benchmark TOML.",
    )
    parser.add_argument(
        "--Nm",
        type=int,
        default=None,
        help="Override the Hermite moment count without editing the tracked benchmark TOML.",
    )
    parser.add_argument(
        "--gaussian-width",
        type=float,
        default=None,
        help=(
            "Override the Gaussian potential-initializer width. The paper-facing default is the TOML value "
            "width=1; non-unit widths are initializer audits, not validation defaults."
        ),
    )
    parser.add_argument(
        "--enable-hypercollisions",
        action="store_true",
        help="Enable the runtime hypercollision term for explicit recurrence/closure audits.",
    )
    parser.add_argument(
        "--nu-hyper-l",
        type=float,
        default=None,
        help="Override Laguerre hypercollision strength.",
    )
    parser.add_argument(
        "--nu-hyper-m",
        type=float,
        default=None,
        help="Override Hermite hypercollision strength.",
    )
    parser.add_argument(
        "--nu-hyper-lm",
        type=float,
        default=None,
        help="Override mixed Laguerre-Hermite hypercollision strength.",
    )
    parser.add_argument(
        "--nu-hyper",
        type=float,
        default=None,
        help="Override isotropic high-order hypercollision strength.",
    )
    parser.add_argument(
        "--p-hyper-l",
        type=float,
        default=None,
        help="Override Laguerre hypercollision exponent.",
    )
    parser.add_argument(
        "--p-hyper-m",
        type=float,
        default=None,
        help="Override Hermite hypercollision exponent.",
    )
    parser.add_argument(
        "--p-hyper-lm",
        type=float,
        default=None,
        help="Override mixed Laguerre-Hermite hypercollision exponent.",
    )
    parser.add_argument(
        "--hypercollisions-const",
        type=float,
        default=None,
        help=(
            "Override the constant velocity-space hypercollision source. When a nu_hyper_* closure strength is "
            "provided and neither source is set, this defaults to 1 so the requested closure is active."
        ),
    )
    parser.add_argument(
        "--hypercollisions-kz",
        type=float,
        default=None,
        help="Override the |k_parallel| Hermite hypercollision source.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Print runtime progress while generating missing per-kx bundles.",
    )
    return parser.parse_args(argv)


def _nearest_kx_index(path: Path, target_kx: float) -> tuple[int, float]:
    with nc.Dataset(path, "r") as ds:
        grids = ds.groups.get("Grids")
        if grids is None or "kx" not in grids.variables:
            raise ValueError(f"missing Grids/kx in {path}")
        kx = np.asarray(grids.variables["kx"][:], dtype=float)
    idx = int(np.argmin(np.abs(kx - float(target_kx))))
    return idx, float(kx[idx])


def _artifact_path(path: Path | str) -> str:
    """Return a stable repo-relative path for tracked metadata when possible."""

    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _finite_or_none(value: float) -> float | None:
    val = float(value)
    return val if np.isfinite(val) else None


def _format_metric(
    value: object, *, fmt: str = ".3f", missing: str = "not fitted"
) -> str:
    if value is None:
        return missing
    val = float(value)
    if not np.isfinite(val):
        return missing
    return f"{val:{fmt}}"


def _initial_level_override(args: argparse.Namespace, cfg: object) -> float | None:
    if str(args.initial_normalization) == "line_first":
        return None
    init = getattr(cfg, "init", None)
    init_amp = float(getattr(init, "init_amp", 1.0))
    if not np.isfinite(init_amp) or init_amp == 0.0:
        raise ValueError(
            "init.init_amp must be finite and non-zero for --initial-normalization=init_amp"
        )
    return abs(init_amp)


def _normalization_label(args: argparse.Namespace) -> str:
    if str(args.initial_normalization) == "init_amp":
        return r"$\langle\phi\rangle_z/|\phi_0|_{\max}$"
    return r"$\phi_\mathrm{zonal}/|\phi_\mathrm{zonal}(0)|$"


def _closure_overrides(args: argparse.Namespace) -> dict[str, float | bool | None]:
    nu = None if args.nu_hyper is None else float(args.nu_hyper)
    nu_l = None if args.nu_hyper_l is None else float(args.nu_hyper_l)
    nu_m = None if args.nu_hyper_m is None else float(args.nu_hyper_m)
    nu_lm = None if args.nu_hyper_lm is None else float(args.nu_hyper_lm)
    has_nonzero_nu = any(
        value is not None and value != 0.0 for value in (nu, nu_l, nu_m, nu_lm)
    )
    hyper_const = (
        None
        if args.hypercollisions_const is None
        else float(args.hypercollisions_const)
    )
    hyper_kz = (
        None if args.hypercollisions_kz is None else float(args.hypercollisions_kz)
    )
    if has_nonzero_nu and hyper_const is None and hyper_kz is None:
        hyper_const = 1.0
    return {
        "enable_hypercollisions": bool(args.enable_hypercollisions or has_nonzero_nu),
        "gaussian_width": None
        if args.gaussian_width is None
        else float(args.gaussian_width),
        "nu_hyper": nu,
        "nu_hyper_l": nu_l,
        "nu_hyper_m": nu_m,
        "nu_hyper_lm": nu_lm,
        "p_hyper_l": None if args.p_hyper_l is None else float(args.p_hyper_l),
        "p_hyper_m": None if args.p_hyper_m is None else float(args.p_hyper_m),
        "p_hyper_lm": None if args.p_hyper_lm is None else float(args.p_hyper_lm),
        "hypercollisions_const": hyper_const,
        "hypercollisions_kz": hyper_kz,
    }


def _apply_audit_overrides(cfg: object, args: argparse.Namespace) -> object:
    overrides = _closure_overrides(args)
    init_cfg = cfg.init
    if overrides["gaussian_width"] is not None:
        init_cfg = replace(init_cfg, gaussian_width=float(overrides["gaussian_width"]))
    collision_updates = {
        name: float(value)
        for name, value in overrides.items()
        if name
        in {
            "nu_hyper",
            "nu_hyper_l",
            "nu_hyper_m",
            "nu_hyper_lm",
            "p_hyper_l",
            "p_hyper_m",
            "p_hyper_lm",
            "hypercollisions_const",
            "hypercollisions_kz",
        }
        and value is not None
    }
    collision_cfg = (
        replace(cfg.collisions, **collision_updates)
        if collision_updates
        else cfg.collisions
    )
    if bool(overrides["enable_hypercollisions"]):
        return replace(
            cfg,
            init=init_cfg,
            collisions=collision_cfg,
            physics=replace(cfg.physics, hypercollisions=True),
            terms=replace(cfg.terms, hypercollisions=1.0),
        )
    return replace(cfg, init=init_cfg, collisions=collision_cfg)


def _plot_panel(
    cases: list[dict[str, object]],
    *,
    title: str,
    y_label: str,
) -> plt.Figure:
    set_plot_style()
    ncases = len(cases)
    ncols = 2
    nrows = ceil(ncases / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 3.7 * nrows), squeeze=False)

    for axis, case in zip(axes.flat, cases, strict=False):
        t = np.asarray(case["t"], dtype=float)
        response = np.asarray(case["response"], dtype=float)
        metrics = case["metrics"]
        response_norm = response / float(metrics.initial_level)
        axis.plot(t, response_norm, color="#0f4c81", linewidth=2.0)
        axis.axhline(
            float(metrics.residual_level),
            color="#c44e52",
            linestyle="--",
            linewidth=1.8,
        )
        axis.axvspan(
            float(metrics.fit_tmin),
            float(metrics.fit_tmax),
            color="#d9ead3",
            alpha=0.22,
            linewidth=0.0,
        )

        max_peak_t = np.asarray(metrics.max_peak_times, dtype=float)
        max_peak_y = np.asarray(metrics.max_peak_values, dtype=float)
        min_peak_t = np.asarray(metrics.min_peak_times, dtype=float)
        min_peak_y = np.asarray(metrics.min_peak_values, dtype=float)
        fit_mask_max = (max_peak_t >= float(metrics.fit_tmin)) & (
            max_peak_t <= float(metrics.fit_tmax)
        )
        fit_mask_min = (min_peak_t >= float(metrics.fit_tmin)) & (
            min_peak_t <= float(metrics.fit_tmax)
        )
        if np.any(fit_mask_max):
            axis.plot(
                max_peak_t[fit_mask_max],
                max_peak_y[fit_mask_max],
                linestyle="none",
                marker="o",
                markersize=4.5,
                color="#2a9d8f",
            )
        if np.any(fit_mask_min):
            axis.plot(
                min_peak_t[fit_mask_min],
                min_peak_y[fit_mask_min],
                linestyle="none",
                marker="o",
                markersize=4.5,
                color="#7b2cbf",
            )

        axis.set_title(rf"$k_x \rho_i = {float(case['kx_target']):.2f}$")
        axis.set_xlabel("t")
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.25)
        axis.text(
            0.03,
            0.97,
            (
                f"residual = {float(metrics.residual_level):.4f}\n"
                rf"$\omega_{{GAM}}R_0/v_{{ti}}$ = {_format_metric(case['omega_R0_over_vi'])}"
                + "\n"
                rf"$\gamma_{{GAM}}R_0/v_{{ti}}$ = {_format_metric(case['gamma_R0_over_vi'])}"
            ),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "alpha": 0.92,
                "edgecolor": "#cccccc",
            },
        )

    for axis in axes.flat[ncases:]:
        axis.set_visible(False)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def _write_combined_trace_csv(cases: list[dict[str, object]], out_csv: Path) -> None:
    fieldnames = [
        "kx_target",
        "kx_selected",
        "t_reference",
        "phi_zonal_real",
        "response_normalized",
        "initial_level",
        "initial_normalization",
        "source_path",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            t = np.asarray(case["t"], dtype=float)
            response = np.asarray(case["response"], dtype=float)
            initial_level = float(case["initial_level"])
            for time_value, response_value in zip(t, response, strict=True):
                writer.writerow(
                    {
                        "kx_target": float(case["kx_target"]),
                        "kx_selected": float(case["kx_selected"]),
                        "t_reference": float(time_value),
                        "phi_zonal_real": float(response_value),
                        "response_normalized": float(response_value) / initial_level,
                        "initial_level": initial_level,
                        "initial_normalization": str(case["initial_normalization"]),
                        "source_path": str(case["source_path"]),
                    }
                )


def run_response_panel(argv: list[str] | None = None) -> int:
    args = _parse_response_args()
    cfg, raw = load_runtime_from_toml(args.config)
    cfg = _apply_audit_overrides(cfg, args)
    audit_overrides = _closure_overrides(args)
    run_cfg = dict(raw.get("run", {}))
    ky_target = float(run_cfg.get("ky", 0.0))
    nl = int(args.Nl) if args.Nl is not None else int(run_cfg.get("Nl", 8))
    nm = int(args.Nm) if args.Nm is not None else int(run_cfg.get("Nm", 32))
    dt = (
        float(args.dt) if args.dt is not None else float(run_cfg.get("dt", cfg.time.dt))
    )
    steps = (
        int(args.steps)
        if args.steps is not None
        else int(run_cfg.get("steps", max(int(round(float(cfg.time.t_max) / dt)), 1)))
    )
    sample_stride = (
        int(args.sample_stride)
        if args.sample_stride is not None
        else int(run_cfg.get("sample_stride", cfg.time.sample_stride))
    )
    if dt <= 0.0:
        raise ValueError("--dt must be positive")
    if steps <= 0:
        raise ValueError("--steps must be positive")
    if sample_stride <= 0:
        raise ValueError("--sample-stride must be positive")
    if float(args.time_scale) <= 0.0:
        raise ValueError("--time-scale must be positive")
    if args.checkpoint_steps is not None and int(args.checkpoint_steps) <= 0:
        raise ValueError("--checkpoint-steps must be positive when provided")
    if bool(args.reuse_output) and bool(args.resume_output):
        raise ValueError("--reuse-output and --resume-output are mutually exclusive")
    if nl <= 0:
        raise ValueError("--Nl must be positive")
    if nm <= 0:
        raise ValueError("--Nm must be positive")
    diagnostics = bool(run_cfg.get("diagnostics", cfg.time.diagnostics))
    r0 = float(getattr(cfg.geometry, "R0", 1.0))
    initial_override = _initial_level_override(args, cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for kx_target in [float(val) for val in args.kx_values]:
        token = kx_token(kx_target)
        out_bundle = args.out_dir / f"w7x_test4_kx{token}.out.nc"
        if bool(args.resume_output) or not args.reuse_output or not out_bundle.exists():
            cfg_case = replace(
                cfg,
                grid=replace(
                    cfg.grid,
                    Lx=float(2.0 * np.pi / kx_target),
                    boundary="periodic",
                    jtwist=None,
                    non_twist=True,
                ),
                time=replace(
                    cfg.time,
                    nstep_restart=None
                    if args.checkpoint_steps is None
                    else int(args.checkpoint_steps),
                ),
                output=replace(
                    cfg.output,
                    path=str(out_bundle),
                    restart_if_exists=bool(args.resume_output),
                    append_on_restart=True,
                    save_for_restart=True,
                ),
            )
            run_runtime_nonlinear_with_artifacts(
                cfg_case,
                out=out_bundle,
                ky_target=ky_target,
                kx_target=kx_target,
                Nl=nl,
                Nm=nm,
                dt=dt,
                steps=steps,
                sample_stride=sample_stride,
                diagnostics=diagnostics,
                show_progress=bool(args.show_progress),
            )

        kx_index, kx_selected = _nearest_kx_index(out_bundle, kx_target)
        kx_tol = max(5.0e-4, 2.0e-2 * abs(float(kx_target)))
        if abs(float(kx_selected) - float(kx_target)) > kx_tol:
            raise ValueError(
                f"selected kx={kx_selected:.6g} differs from target {kx_target:.6g}; "
                "check the radial box and boundary settings for this zonal run"
            )
        series = load_diagnostic_time_series(
            out_bundle,
            variable="Phi_zonal_line_kxt",
            kx_index=kx_index,
            component="real",
            align_phase=True,
        )
        values = np.asarray(series.values, dtype=float)
        t_scaled = np.asarray(series.t, dtype=float) * float(args.time_scale)
        metrics = zonal_flow_response_metrics(
            t_scaled,
            values,
            tail_fraction=float(args.tail_fraction),
            initial_fraction=float(args.initial_fraction),
            initial_policy=str(args.initial_policy),
            initial_level_override=initial_override,
            peak_fit_max_peaks=int(args.peak_fit_max_peaks),
            damping_fit_mode="branchwise_extrema",
            frequency_fit_mode="hilbert_phase",
            fit_window_tmax=float(args.fit_window_tmax),
            hilbert_trim_fraction=0.2,
        )
        gam_frequency = _finite_or_none(metrics.gam_frequency)
        gam_damping_rate = _finite_or_none(metrics.gam_damping_rate)
        omega_r0_over_vi = None if gam_frequency is None else float(gam_frequency) * r0
        gamma_r0_over_vi = (
            None if gam_damping_rate is None else -float(gam_damping_rate) * r0
        )
        row = {
            "kx_target": float(kx_target),
            "kx_selected": float(kx_selected),
            "kx_index": int(kx_index),
            "source_path": _artifact_path(out_bundle),
            "initial_level": float(metrics.initial_level),
            "initial_normalization": str(args.initial_normalization),
            "initial_level_override": None
            if initial_override is None
            else float(initial_override),
            "residual_level": float(metrics.residual_level),
            "residual_std": float(metrics.residual_std),
            "response_rms": float(metrics.response_rms),
            "gam_frequency": gam_frequency,
            "gam_damping_rate": gam_damping_rate,
            "omega_R0_over_vi": omega_r0_over_vi,
            "gamma_R0_over_vi": gamma_r0_over_vi,
            "peak_count": int(metrics.peak_count),
            "peak_fit_count": int(metrics.peak_fit_count),
            "tmin": float(metrics.tmin),
            "tmax": float(metrics.tmax),
            "fit_tmin": float(metrics.fit_tmin),
            "fit_tmax": float(metrics.fit_tmax),
        }
        summary_rows.append(row)
        cases.append(
            {
                **row,
                "t": t_scaled,
                "response": values,
                "metrics": metrics,
            }
        )
        trace_csv = args.out_dir / f"w7x_test4_kx{token}.csv"
        np.savetxt(
            trace_csv,
            np.column_stack([t_scaled, values]),
            delimiter=",",
            header="t_reference,phi_zonal_real",
            comments="",
        )

    fig = _plot_panel(
        cases,
        title="W7-X bean-tube zonal-flow relaxation (test 4)",
        y_label=_normalization_label(args),
    )
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight")
    fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight")

    summary_csv = args.out_png.with_suffix(".csv")
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    traces_csv = args.out_png.with_suffix(".traces.csv")
    _write_combined_trace_csv(cases, traces_csv)

    meta_out = args.out_png.with_suffix(".json")
    meta_out.write_text(
        json.dumps(
            {
                "config": _artifact_path(args.config),
                "summary_csv": _artifact_path(summary_csv),
                "traces_csv": _artifact_path(traces_csv),
                "initial_policy": str(args.initial_policy),
                "initial_normalization": str(args.initial_normalization),
                "initial_level_override": None
                if initial_override is None
                else float(initial_override),
                "damping_method": "branchwise_extrema",
                "frequency_method": "hilbert_phase",
                "fit_window_tmax": float(args.fit_window_tmax),
                "runtime": {
                    "dt": float(dt),
                    "steps": int(steps),
                    "sample_stride": int(sample_stride),
                    "checkpoint_steps": None
                    if args.checkpoint_steps is None
                    else int(args.checkpoint_steps),
                    "resume_output": bool(args.resume_output),
                    "time_scale": float(args.time_scale),
                    "diagnostics": bool(diagnostics),
                    "show_progress": bool(args.show_progress),
                    "expected_tmax": float(dt) * float(steps),
                    "Nl": int(nl),
                    "Nm": int(nm),
                },
                "audit_overrides": audit_overrides,
                "literature_reference": dict(W7X_TEST4_REFERENCE),
                "cases": summary_rows,
                "validation_status": "open",
                "notes": (
                    "This panel follows the W7-X stella/GENE benchmark test-4 contract: "
                    "bean flux tube, torflux=0.64, alpha=0, adiabatic electrons, zero gradients, ky=0, "
                    "and Gaussian electrostatic-potential initial perturbations at four kx rho_i values. "
                    "It uses the unweighted line-averaged signed potential observable requested by the paper; "
                    "the volume-weighted Phi_zonal_mode_kxt diagnostic remains available for shaped-tokamak "
                    "and energy-consistency checks. "
                    "The paper text normalizes the line-averaged response to its t=0 value; "
                    "therefore the default --initial-normalization=line_first follows that convention. "
                    "The init_amp option is retained for explicit audits of the caption wording and "
                    "the clipped initial portion of Fig. 11, but it is not the default validation contract. "
                    "The initial GAM is extracted with separate positive/negative-extrema "
                    "damping fits plus a Hilbert-phase frequency estimate over a common early-time window. "
                    "Runtime times are multiplied by the metadata time_scale before plotting and reference "
                    "comparison; the default time_scale=1 keeps the runtime axis on the paper's t v_ti/a "
                    "axis and non-unit values are treated as calibration probes. "
                    "The default fit window cap isolates the initial GAM before the slower stellarator-specific "
                    "oscillation described in section 4.4 of the benchmark paper; this cutoff is a manuscript-policy "
                    "inference, not a quoted number from the paper itself. The metadata remains open until the "
                    "separate digitized-reference gate closes both residual and late-envelope tolerances."
                ),
                "references": [
                    "Gonzalez-Jerez et al. 2022 W7-X test-4 zonal-flow relaxation benchmark",
                    "Merlo et al. 2016 shaped-tokamak GAM benchmark for extraction-policy consistency",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("response-panel", "contract", "state-convention")
    )
    args, remainder = parser.parse_known_args(argv)
    if args.mode == "response-panel":
        return run_response_panel(remainder)
    if args.mode == "contract":
        return run_contract(remainder)
    return run_state_convention(remainder)


if __name__ == "__main__":
    raise SystemExit(main())
