#!/usr/bin/env python3
"""Gate the optional spectral Laguerre nonlinear mode against grid mode.

The spectral mode is an engineering fast path for the nonlinear bracket. This
gate keeps it opt-in by comparing end-of-run scalar diagnostics against the
default grid-mode bracket on the same bounded nonlinear case.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from gkx.config import GeometryConfig  # type: ignore[import-untyped]
from gkx.workflows.runtime.toml import load_runtime_from_toml  # type: ignore[import-untyped]
from gkx.runtime import run_runtime_nonlinear  # type: ignore[import-untyped]
from gkx.workflows.runtime.config import RuntimeConfig  # type: ignore[import-untyped]


@dataclass(frozen=True)
class CaseSpec:
    name: str
    config: Path
    ky: float
    nl: int
    nm: int
    steps: int
    dt: float | None = None
    geometry_file: Path | None = None
    geometry_model: str | None = None


DEFAULT_CASES: dict[str, CaseSpec] = {
    "cyclone": CaseSpec(
        name="cyclone",
        config=Path(
            "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_short.toml"
        ),
        ky=0.3,
        nl=4,
        nm=8,
        steps=6,
        dt=0.05,
    ),
    "kbm": CaseSpec(
        name="kbm",
        config=Path("examples/nonlinear/axisymmetric/runtime_kbm_nonlinear_short.toml"),
        ky=0.3,
        nl=3,
        nm=4,
        steps=4,
        dt=0.0003,
    ),
    "w7x": CaseSpec(
        name="w7x",
        config=Path(
            "examples/nonlinear/non-axisymmetric/runtime_w7x_nonlinear_imported_geometry.toml"
        ),
        ky=1.0 / 21.0,
        nl=3,
        nm=4,
        steps=3,
        dt=0.05,
        geometry_file=Path("tools_out/pilots/w7x_linear_local.eik.nc"),
        geometry_model="vmec-eik",
    ),
    "hsx": CaseSpec(
        name="hsx",
        config=Path(
            "examples/nonlinear/non-axisymmetric/runtime_hsx_nonlinear_vmec_geometry.toml"
        ),
        ky=1.0 / 21.0,
        nl=3,
        nm=4,
        steps=3,
        dt=0.05,
        geometry_file=Path(
            ".cache/gkx/vmec_eik/HSX_QHS_vacuum_ns201_a6ec24c48834374f.eik.nc"
        ),
        geometry_model="imported-netcdf",
    ),
}


DIAGNOSTIC_KEYS = (
    "Wg_end",
    "Wphi_end",
    "heat_end",
    "particle_end",
    "gamma_end",
    "omega_end",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(DEFAULT_CASES) + ["all"],
        default=None,
        help="Case to run. Repeat for multiple cases. Defaults to all built-in cases.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override step count for every selected case.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.15,
        help="Maximum relative scalar-diagnostic difference.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-12,
        help="Absolute floor for scalar-diagnostic differences.",
    )
    parser.add_argument(
        "--out-json", type=Path, default=Path("docs/_static/laguerre_mode_gate.json")
    )
    parser.add_argument(
        "--out-csv", type=Path, default=Path("docs/_static/laguerre_mode_gate.csv")
    )
    parser.add_argument(
        "--plot-out", type=Path, default=Path("docs/_static/laguerre_mode_gate.png")
    )
    parser.add_argument("--w7x-geometry-file", type=Path, default=None)
    parser.add_argument("--hsx-geometry-file", type=Path, default=None)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def _selected_cases(args: argparse.Namespace) -> list[CaseSpec]:
    requested = args.case or ["all"]
    names = list(DEFAULT_CASES) if "all" in requested else requested
    specs: list[CaseSpec] = []
    for name in names:
        spec = DEFAULT_CASES[name]
        geometry_file = spec.geometry_file
        if name == "w7x" and args.w7x_geometry_file is not None:
            geometry_file = args.w7x_geometry_file
        if name == "hsx" and args.hsx_geometry_file is not None:
            geometry_file = args.hsx_geometry_file
        steps = int(args.steps) if args.steps is not None else int(spec.steps)
        specs.append(replace(spec, steps=steps, geometry_file=geometry_file))
    return specs


def _with_case_overrides(cfg: RuntimeConfig, spec: CaseSpec) -> RuntimeConfig:
    time_cfg = replace(
        cfg.time,
        sample_stride=1,
        diagnostics_stride=1,
        diagnostics=True,
        progress_bar=False,
    )
    geometry_cfg = cfg.geometry
    if spec.geometry_file is not None:
        geometry_cfg = GeometryConfig(
            **{
                **asdict(cfg.geometry),
                "model": spec.geometry_model or cfg.geometry.model,
                "geometry_file": str(spec.geometry_file),
                "vmec_file": None,
            }
        )
    return replace(cfg, time=time_cfg, geometry=geometry_cfg)


def _final_scalar_diagnostics(result: Any) -> dict[str, float]:
    diag = result.diagnostics
    if diag is None:
        raise RuntimeError("nonlinear run did not return diagnostics")
    return {
        "Wg_end": float(np.asarray(diag.Wg_t[-1])),
        "Wphi_end": float(np.asarray(diag.Wphi_t[-1])),
        "heat_end": float(np.asarray(diag.heat_flux_t[-1])),
        "particle_end": float(np.asarray(diag.particle_flux_t[-1])),
        "gamma_end": float(np.asarray(diag.gamma_t[-1])),
        "omega_end": float(np.asarray(diag.omega_t[-1])),
    }


def _run_mode(cfg: RuntimeConfig, spec: CaseSpec, mode: str) -> dict[str, float]:
    start = time.perf_counter()
    result = run_runtime_nonlinear(
        cfg,
        ky_target=float(spec.ky),
        Nl=int(spec.nl),
        Nm=int(spec.nm),
        dt=spec.dt,
        steps=int(spec.steps),
        laguerre_mode=mode,
        diagnostics=True,
        resolved_diagnostics=False,
    )
    elapsed = time.perf_counter() - start
    scalars = _final_scalar_diagnostics(result)
    scalars["run_s"] = float(elapsed)
    scalars["ky_selected"] = (
        float(result.ky_selected) if result.ky_selected is not None else float("nan")
    )
    return scalars


def _compare(
    grid: dict[str, float], spectral: dict[str, float], *, atol: float
) -> dict[str, float]:
    rel: dict[str, float] = {}
    for key in DIAGNOSTIC_KEYS:
        a = float(grid[key])
        b = float(spectral[key])
        denom = max(abs(a), abs(b), float(atol))
        rel[f"{key}_rel_diff"] = abs(a - b) / denom
    rel["max_rel_diff"] = max(rel.values()) if rel else 0.0
    rel["speedup_grid_over_spectral"] = float(grid["run_s"]) / max(
        float(spectral["run_s"]), 1.0e-30
    )
    return rel


def _run_case(spec: CaseSpec, *, rtol: float, atol: float) -> dict[str, Any]:
    if not spec.config.exists():
        return {
            "case": spec.name,
            "status": "missing_config",
            "config": str(spec.config),
        }
    if spec.geometry_file is not None and not spec.geometry_file.exists():
        return {
            "case": spec.name,
            "status": "missing_geometry",
            "config": str(spec.config),
            "geometry_file": str(spec.geometry_file),
        }
    cfg, _data = load_runtime_from_toml(spec.config)
    cfg = _with_case_overrides(cfg, spec)
    try:
        grid = _run_mode(cfg, spec, "grid")
        spectral = _run_mode(cfg, spec, "spectral")
        comparison = _compare(grid, spectral, atol=atol)
        passed = bool(
            comparison["max_rel_diff"] <= rtol
            and np.isfinite(comparison["max_rel_diff"])
        )
        return {
            "case": spec.name,
            "status": "pass" if passed else "mismatch",
            "config": str(spec.config),
            "geometry_file": str(spec.geometry_file)
            if spec.geometry_file is not None
            else None,
            "ky": spec.ky,
            "Nl": spec.nl,
            "Nm": spec.nm,
            "steps": spec.steps,
            "dt": spec.dt,
            "rtol": rtol,
            "atol": atol,
            "grid": grid,
            "spectral": spectral,
            "comparison": comparison,
        }
    except (
        Exception
    ) as exc:  # pragma: no cover - exercised by operational gate failures.
        return {
            "case": spec.name,
            "status": "error",
            "config": str(spec.config),
            "geometry_file": str(spec.geometry_file)
            if spec.geometry_file is not None
            else None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "status",
        "steps",
        "dt",
        "grid_run_s",
        "spectral_run_s",
        "speedup_grid_over_spectral",
        "max_rel_diff",
    ]
    for key in DIAGNOSTIC_KEYS:
        fieldnames.extend([f"grid_{key}", f"spectral_{key}", f"{key}_rel_diff"])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            grid = row.get("grid", {})
            spectral = row.get("spectral", {})
            comparison = row.get("comparison", {})
            flat = {
                "case": row.get("case"),
                "status": row.get("status"),
                "steps": row.get("steps"),
                "dt": row.get("dt"),
                "grid_run_s": grid.get("run_s"),
                "spectral_run_s": spectral.get("run_s"),
                "speedup_grid_over_spectral": comparison.get(
                    "speedup_grid_over_spectral"
                ),
                "max_rel_diff": comparison.get("max_rel_diff"),
            }
            for key in DIAGNOSTIC_KEYS:
                flat[f"grid_{key}"] = grid.get(key)
                flat[f"spectral_{key}"] = spectral.get(key)
                flat[f"{key}_rel_diff"] = comparison.get(f"{key}_rel_diff")
            writer.writerow(flat)


def _write_plot(path: Path, rows: list[dict[str, Any]], *, rtol: float) -> None:
    passed_rows = [row for row in rows if "comparison" in row]
    if not passed_rows:
        return
    import matplotlib.pyplot as plt

    labels = [str(row["case"]) for row in passed_rows]
    x = np.arange(len(labels))
    speedup = [
        float(row["comparison"]["speedup_grid_over_spectral"]) for row in passed_rows
    ]
    rel = [float(row["comparison"]["max_rel_diff"]) for row in passed_rows]
    rel_floor = max(float(rtol) * 1.0e-4, 1.0e-12)
    rel_plot = [max(value, rel_floor) for value in rel]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    axes[0].bar(x, speedup, color="#2f6f73", edgecolor="#143638", linewidth=0.8)
    axes[0].axhline(1.0, color="0.25", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("grid runtime / spectral runtime")
    axes[0].set_title("Warm-throughput ratio")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")

    axes[1].bar(x, rel_plot, color="#c46a3a", edgecolor="#5a2d19", linewidth=0.8)
    axes[1].axhline(
        rtol, color="0.25", linestyle="--", linewidth=1.0, label=f"gate rtol={rtol:g}"
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("max scalar relative difference")
    axes[1].set_title("Grid vs spectral parity")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].legend(frameon=False, fontsize=8)
    for xpos, value, plotted in zip(x, rel, rel_plot, strict=True):
        label = "0" if value == 0.0 else f"{value:.1e}"
        axes[1].text(xpos, plotted * 1.2, label, ha="center", va="bottom", fontsize=8)
    fig.suptitle("Optional spectral Laguerre nonlinear mode gate", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    rows = [
        _run_case(spec, rtol=float(args.rtol), atol=float(args.atol))
        for spec in _selected_cases(args)
    ]
    payload = {
        "description": "Optional spectral Laguerre nonlinear mode gate against default grid-mode nonlinear bracket.",
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(args.out_csv, rows)
    _write_plot(args.plot_out, rows, rtol=float(args.rtol))
    for row in rows:
        status = row["status"]
        comparison = row.get("comparison", {})
        speedup = comparison.get("speedup_grid_over_spectral", float("nan"))
        max_rel = comparison.get("max_rel_diff", float("nan"))
        print(
            f"{row['case']}: status={status} speedup={speedup:.3g} max_rel={max_rel:.3g}"
        )
    if args.fail_on_mismatch and any(row["status"] != "pass" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
