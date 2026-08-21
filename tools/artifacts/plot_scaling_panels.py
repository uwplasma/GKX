#!/usr/bin/env python3
"""Plot scaling and parallelization panels from tracked JSON/CSV artifacts.

Subcommands:
  diffrax-speedup       Two-device diffrax scaling from scaling_speedup_data.csv.
  independent-ky        Independent ky CPU/GPU scan strong-scaling panel.
  rhs-profile           CPU/GPU nonlinear RHS kernel split profile.
  nonlinear-sharding    Nonlinear whole-state sharding diagnostic scaling panel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIFFRAX_DATA = REPO_ROOT / "docs" / "_static" / "scaling_speedup_data.csv"
DEFAULT_DIFFRAX_PNG = REPO_ROOT / "docs" / "_static" / "scaling_speedup.png"
DEFAULT_INDEPENDENT_KY_INPUTS = [
    REPO_ROOT / "docs" / "_static" / "independent_ky_scan_scaling_cpu_large.json",
    REPO_ROOT / "docs" / "_static" / "independent_ky_scan_scaling_gpu_large.json",
]
DEFAULT_INDEPENDENT_KY_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "independent_ky_scan_scaling_large"
)
DEFAULT_NONLINEAR_SHARDING_INPUTS = [
    REPO_ROOT / "docs" / "_static" / "nonlinear_sharding_strong_scaling_cpu_large.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "nonlinear_sharding_strong_scaling_gpu_xlarge.json",
]
DEFAULT_NONLINEAR_SHARDING_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "nonlinear_sharding_strong_scaling_large"
)
DEFAULT_RHS_PROFILE_INPUTS = {
    "CPU grid": REPO_ROOT / "docs" / "_static" / "nonlinear_rhs_profile_cpu.csv",
    "CPU spectral": REPO_ROOT
    / "docs"
    / "_static"
    / "nonlinear_rhs_profile_cpu_spectral.csv",
    "GPU grid": REPO_ROOT / "docs" / "_static" / "nonlinear_rhs_profile_gpu.csv",
    "GPU spectral": REPO_ROOT
    / "docs"
    / "_static"
    / "nonlinear_rhs_profile_gpu_spectral.csv",
}
DEFAULT_RHS_PROFILE_PNG = REPO_ROOT / "docs" / "_static" / "nonlinear_rhs_profile.png"


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


def _save_figure(fig, png_path: Path) -> dict[str, str]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def parse_rhs_profile_input(value: str) -> tuple[str, Path]:
    """Parse a ``LABEL=CSV`` RHS profile input specification."""

    label, sep, path = value.partition("=")
    if not sep or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--input must have the form LABEL=CSV")
    return label.strip(), Path(path.strip())


def read_rhs_profile(path: Path) -> dict[str, float]:
    with path.open(newline="") as f:
        return {row["kernel"]: float(row["seconds"]) for row in csv.DictReader(f)}


def rhs_profile_case_title(case: str) -> str:
    titles = {
        "cyclone_short": "Cyclone short case",
        "cyclone_miller_benchmark_size": "Cyclone Miller benchmark-size case",
    }
    return titles.get(case, case.replace("_", " "))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    ratio = float(numerator) / float(denominator)
    return ratio if math.isfinite(ratio) else None


def build_rhs_profile_summary(
    profiles: dict[str, dict[str, float]], *, case: str = "cyclone_short"
) -> dict[str, Any]:
    """Return a machine-readable summary of RHS split-profile CSV files."""

    rows: dict[str, dict[str, Any]] = {}
    for label, profile in profiles.items():
        full_rhs = profile.get("full_rhs")
        timed_kernels = {
            key: value for key, value in profile.items() if key != "full_rhs"
        }
        dominant = (
            max(timed_kernels, key=lambda key: timed_kernels[key])
            if timed_kernels
            else None
        )
        rows[label] = {
            "seconds": profile,
            "dominant_measured_kernel": dominant,
            "field_solve_fraction_of_full_rhs": _safe_ratio(
                profile.get("field_solve"), full_rhs
            ),
            "linear_rhs_fraction_of_full_rhs": _safe_ratio(
                profile.get("linear_rhs"), full_rhs
            ),
            "nonlinear_bracket_fraction_of_full_rhs": _safe_ratio(
                profile.get("nonlinear_bracket"), full_rhs
            ),
        }

    spectral_speedups: dict[str, dict[str, float | None]] = {}
    for backend in ("CPU", "GPU"):
        grid = profiles.get(f"{backend} grid")
        spectral = profiles.get(f"{backend} spectral")
        if grid is None or spectral is None:
            continue
        spectral_speedups[backend.lower()] = {
            "full_rhs_grid_over_spectral": _safe_ratio(
                grid.get("full_rhs"), spectral.get("full_rhs")
            ),
            "nonlinear_bracket_grid_over_spectral": _safe_ratio(
                grid.get("nonlinear_bracket"), spectral.get("nonlinear_bracket")
            ),
        }

    full_rhs_candidates = [
        (profile["full_rhs"], label)
        for label, profile in profiles.items()
        if "full_rhs" in profile and math.isfinite(float(profile["full_rhs"]))
    ]
    fastest = min(
        full_rhs_candidates, default=(None, None), key=lambda item: float(item[0])
    )
    return {
        "kind": "nonlinear_rhs_profile_summary",
        "case": case,
        "rows": rows,
        "spectral_speedups": spectral_speedups,
        "fastest_full_rhs_label": fastest[1],
        "fastest_full_rhs_seconds": fastest[0],
        "claim_scope": (
            "RHS split profile for hot-path localization. Treat speedups as bounded engineering "
            "measurements; production runtime claims require matched benchmark-size CPU/GPU sweeps."
        ),
    }


def write_rhs_profile_summary_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_rhs_profile_panel(
    profiles: dict[str, dict[str, float]],
    *,
    out_png: Path,
    case: str = "cyclone_short",
    title: str | None = None,
    summary_json: Path | None = None,
) -> dict[str, str]:
    """Plot a nonlinear RHS kernel profile and write its JSON summary."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style

    kernels = ["field_solve", "linear_rhs", "nonlinear_bracket", "full_rhs"]
    labels = list(profiles)
    x = np.arange(len(kernels))
    width = min(0.18, 0.75 / max(len(labels), 1))
    offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2.0) * width

    set_plot_style()
    colors = ["#52616b", "#8c6d31", "#2f6f73", "#c46a3a"]
    fig, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    for idx, label in enumerate(labels):
        values = [profiles[label].get(kernel, np.nan) for kernel in kernels]
        ax.bar(
            x + offsets[idx],
            values,
            width=width,
            label=label,
            color=colors[idx % len(colors)],
            edgecolor="0.18",
        )

    ax.set_yscale("log")
    ax.set_ylabel("seconds per compiled kernel call")
    ax.set_title(
        f"Nonlinear RHS kernel profile: {title or rhs_profile_case_title(str(case))}"
    )
    ax.set_xticks(x, [kernel.replace("_", "\n") for kernel in kernels])
    ax.grid(axis="y", which="major", alpha=0.25)
    ax.legend(frameon=False, ncols=2, fontsize=8)
    paths = _save_figure(fig, out_png)
    plt.close(fig)

    summary_path = summary_json if summary_json is not None else out_png.with_suffix(".json")
    write_rhs_profile_summary_json(
        build_rhs_profile_summary(profiles, case=str(case)), summary_path
    )
    paths["json"] = str(summary_path)
    return paths


def write_diffrax_speedup_panel(
    data_path: Path = DEFAULT_DIFFRAX_DATA,
    out_png: Path = DEFAULT_DIFFRAX_PNG,
) -> dict[str, str]:
    """Plot two-device diffrax speedup from the tracked CSV sweep."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style

    df = pd.read_csv(data_path)
    set_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6))

    for backend, color in [("cpu", "#1f77b4"), ("cuda", "#ff7f0e")]:
        sub = df[df["backend"] == backend]
        steps = sorted(sub["steps"].unique())
        speedups = []
        for step in steps:
            sub_step = sub[sub["steps"] == step]
            t1 = float(sub_step[sub_step["devices"] == 1]["elapsed_s"].iloc[0])
            t2 = float(sub_step[sub_step["devices"] == 2]["elapsed_s"].iloc[0])
            speedups.append(t1 / t2)
        ax.plot(
            steps,
            speedups,
            marker="o",
            linewidth=2.4,
            color=color,
            label=f"{backend.upper()} 2 devices",
        )

    ax.axhline(2.0, color="#444444", linestyle=":", linewidth=1.0, label="ideal")
    ax.set_ylabel("Speedup (1x / 2x)")
    ax.set_xlabel("Linear integration steps per run")
    ax.set_xticks(sorted(df[df["backend"].isin(["cpu", "cuda"])]["steps"].unique()))
    ax.set_title("Two-device diffrax scaling (Ny=64, Nz=128, Nl=6, Nm=6)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    paths = _save_figure(fig, out_png)
    plt.close(fig)
    return paths


def _independent_ky_grid_label(payload: dict[str, Any]) -> str:
    grid = payload["grid"]
    return (
        f"Nx={int(grid['Nx'])}, Ny={int(grid['Ny'])}, Nz={int(grid['Nz'])}, "
        f"Nl={int(grid['Nl'])}, Nm={int(grid['Nm'])}"
    )


def load_independent_ky_summary(paths: list[Path]) -> dict[str, Any]:
    """Load independent-ky scaling rows from CPU/GPU JSON summaries."""

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
            }
        )
        for row in payload["rows"]:
            item = dict(row)
            item["backend"] = str(payload["backend"])
            item["source"] = str(path)
            item["grid_label"] = _independent_ky_grid_label(payload)
            rows.append(item)
    return _json_clean(
        {
            "kind": "independent_ky_scan_scaling_combined",
            "claim_scope": (
                "solver-backed independent ky scan strong-scaling artifact for CPU processes "
                "and GPU workers; not a nonlinear domain-decomposition speedup claim"
            ),
            "identity_passed": all(item["identity_passed"] for item in inputs),
            "inputs": inputs,
            "rows": rows,
        }
    )


def write_independent_ky_artifacts(
    summary: dict[str, Any], out_prefix: Path
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style

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
        "max_gamma_rel_error",
        "max_omega_abs_error",
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
        elapsed = np.asarray(
            [float(row["timed_wall_s"]) for row in subset], dtype=float
        )
        gamma_rel = np.asarray(
            [float(row["max_gamma_rel_error"]) for row in subset], dtype=float
        )
        omega_abs = np.asarray(
            [float(row["max_omega_abs_error"]) for row in subset], dtype=float
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
            np.maximum(gamma_rel, 1.0e-16),
            marker=marker,
            lw=2.0,
            color=color,
            label=label + r" $\gamma$",
        )
        axes[2].semilogy(
            x + 0.04,
            np.maximum(omega_abs, 1.0e-16),
            marker=marker,
            ls="--",
            lw=1.8,
            color=color,
            label=label + r" $\omega$",
        )
    ideal = np.arange(1, max_devices + 1)
    axes[0].plot(ideal, ideal, ":", color="0.35", lw=1.4, label="ideal")
    axes[0].set_xlabel("workers/devices")
    axes[0].set_ylabel("speedup vs one worker")
    axes[0].set_title("Independent ky strong scaling")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_xlabel("workers/devices")
    axes[1].set_ylabel("median scan time [s]")
    axes[1].set_title("Solver throughput")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].set_xlabel("workers/devices")
    axes[2].set_ylabel("identity error")
    axes[2].set_title("Gamma/omega identity")
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


def _nonlinear_sharding_grid_label(grid: dict[str, Any]) -> str:
    return (
        f"Nx={int(grid['Nx'])}, Ny={int(grid['Ny_requested'])}, Nz={int(grid['Nz'])}, "
        f"Nl={int(grid['Nl'])}, Nm={int(grid['Nm'])}"
    )


def load_nonlinear_sharding_summary(paths: list[Path]) -> dict[str, Any]:
    """Load nonlinear whole-state sharding rows from CPU/GPU JSON summaries."""

    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        backend = str(payload["backend"])
        inputs.append(
            {
                "path": str(path),
                "backend": backend,
                "grid": payload["grid"],
                "identity_passed": bool(payload["identity_passed"]),
                "speedup_passed": bool(payload.get("speedup_passed", False)),
                "status": str(payload.get("status", "diagnostic_identity_only")),
                "speedup_blockers": list(payload.get("speedup_blockers", [])),
                "claim_scope": payload.get("claim_scope", ""),
            }
        )
        for row in payload["rows"]:
            item = dict(row)
            item["source"] = str(path)
            item["grid_label"] = _nonlinear_sharding_grid_label(payload["grid"])
            rows.append(item)
    return _json_clean(
        {
            "kind": "nonlinear_sharding_strong_scaling_combined",
            "claim_scope": (
                "large CPU/GPU nonlinear whole-state sharding artifact. It is a numerical-identity "
                "and profiler-direction result; it is not a production speedup claim unless speedup_passed "
                "is true and the separate production gate passes."
            ),
            "identity_passed": all(bool(item["identity_passed"]) for item in inputs),
            "speedup_passed": all(bool(item["speedup_passed"]) for item in inputs),
            "status": (
                "identity_and_speedup"
                if inputs and all(bool(item["speedup_passed"]) for item in inputs)
                else "diagnostic_identity_only"
            ),
            "speedup_blockers": [
                f"{item['backend']}:{blocker}"
                for item in inputs
                for blocker in item.get("speedup_blockers", [])
            ],
            "inputs": inputs,
            "rows": rows,
        }
    )


def write_nonlinear_sharding_artifacts(
    summary: dict[str, Any], out_prefix: Path
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style

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
        "actual_devices",
        "grid_label",
        "best_spec",
        "state_sharding_active",
        "identity_gate_pass",
        "parallel_median_s",
        "strong_speedup_vs_1_device",
        "same_process_speedup",
        "max_rel_state_error",
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
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1), constrained_layout=True)
    style = {
        "cpu": {"color": "#276b8e", "marker": "o", "label": "CPU logical devices"},
        "gpu": {"color": "#c45a14", "marker": "s", "label": "GPU devices"},
    }
    for backend in sorted({str(row["backend"]) for row in rows}):
        subset = sorted(
            (row for row in rows if str(row["backend"]) == backend),
            key=lambda row: int(row["requested_devices"]),
        )
        x = np.asarray([int(row["requested_devices"]) for row in subset], dtype=float)
        y = np.asarray(
            [float(row["strong_speedup_vs_1_device"]) for row in subset], dtype=float
        )
        t = np.asarray([float(row["parallel_median_s"]) for row in subset], dtype=float)
        label = f"{style.get(backend, {}).get('label', backend)}"
        color = style.get(backend, {}).get("color", None)
        marker = style.get(backend, {}).get("marker", "o")
        axes[0].plot(x, y, marker=marker, lw=2.2, color=color, label=label)
        axes[1].semilogy(
            x, np.maximum(t, 1.0e-16), marker=marker, lw=2.2, color=color, label=label
        )
    xmax = max(int(row["requested_devices"]) for row in rows)
    ideal = np.arange(1, xmax + 1)
    axes[0].plot(ideal, ideal, ":", color="0.35", lw=1.4, label="ideal")
    axes[0].axhline(1.0, color="0.5", ls="--", lw=1.0)
    axes[0].set_xlabel("devices")
    axes[0].set_ylabel("speedup vs one device")
    axes[0].set_title("Large nonlinear whole-state sharding")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_xlabel("devices")
    axes[1].set_ylabel("median fixed-step time [s]")
    axes[1].set_title("Identity-preserving timing")
    axes[1].legend(frameon=False, fontsize=8)
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


def build_diffrax_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot two-device diffrax scaling.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DIFFRAX_DATA)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_DIFFRAX_PNG)
    return parser


def build_independent_ky_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot independent-ky scan scaling.")
    parser.add_argument(
        "--inputs", type=Path, nargs="+", default=DEFAULT_INDEPENDENT_KY_INPUTS
    )
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_INDEPENDENT_KY_PREFIX
    )
    return parser


def build_nonlinear_sharding_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot nonlinear whole-state sharding strong-scaling diagnostics."
    )
    parser.add_argument(
        "--inputs", type=Path, nargs="+", default=DEFAULT_NONLINEAR_SHARDING_INPUTS
    )
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_NONLINEAR_SHARDING_PREFIX
    )
    return parser


def build_rhs_profile_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot nonlinear RHS split profiles.")
    parser.add_argument("--out", type=Path, default=DEFAULT_RHS_PROFILE_PNG)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults to the plot path with a .json suffix.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        metavar="LABEL=CSV",
        help="Optional labeled profile CSV. May be repeated. Defaults to the shipped Cyclone inputs.",
    )
    parser.add_argument(
        "--case",
        default="cyclone_short",
        help="Case label written to the JSON summary.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title suffix. Defaults to a title from --case.",
    )
    return parser


def main_diffrax_speedup(argv: list[str] | None = None) -> int:
    args = build_diffrax_parser().parse_args(argv)
    paths = write_diffrax_speedup_panel(args.data, args.out_png)
    print(f"Wrote {paths['png']}")
    return 0


def main_independent_ky(argv: list[str] | None = None) -> int:
    args = build_independent_ky_parser().parse_args(argv)
    summary = load_independent_ky_summary([Path(path) for path in args.inputs])
    paths = write_independent_ky_artifacts(summary, Path(args.out_prefix))
    print(
        json.dumps(
            {"identity_passed": summary["identity_passed"], "paths": paths}, indent=2
        )
    )
    return 0


def main_rhs_profile(argv: list[str] | None = None) -> int:
    args = build_rhs_profile_parser().parse_args(argv)
    input_paths = (
        dict(parse_rhs_profile_input(item) for item in args.input)
        if args.input is not None
        else DEFAULT_RHS_PROFILE_INPUTS
    )
    profiles = {
        label: read_rhs_profile(path)
        for label, path in input_paths.items()
        if path.exists()
    }
    if not profiles:
        raise FileNotFoundError("no nonlinear RHS profile CSV files were found")
    paths = write_rhs_profile_panel(
        profiles,
        out_png=Path(args.out),
        case=str(args.case),
        title=args.title,
        summary_json=args.summary_json,
    )
    print(f"Wrote {paths['png']}")
    print(f"Wrote {paths['json']}")
    return 0


def main_nonlinear_sharding(argv: list[str] | None = None) -> int:
    args = build_nonlinear_sharding_parser().parse_args(argv)
    summary = load_nonlinear_sharding_summary([Path(path) for path in args.inputs])
    paths = write_nonlinear_sharding_artifacts(summary, Path(args.out_prefix))
    print(
        json.dumps(
            {
                "identity_passed": summary["identity_passed"],
                "speedup_passed": summary["speedup_passed"],
                "status": summary["status"],
                "paths": paths,
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "command",
            choices=(
                "diffrax-speedup",
                "independent-ky",
                "rhs-profile",
                "nonlinear-sharding",
            ),
        )
        parser.print_help()
        return 2
    command, rest = tokens[0], tokens[1:]
    if command == "diffrax-speedup":
        return main_diffrax_speedup(rest)
    if command == "independent-ky":
        return main_independent_ky(rest)
    if command == "rhs-profile":
        return main_rhs_profile(rest)
    if command == "nonlinear-sharding":
        return main_nonlinear_sharding(rest)
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
