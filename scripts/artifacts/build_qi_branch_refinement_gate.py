#!/usr/bin/env python3
"""Build a fail-closed QI linear branch-refinement gate."""

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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gkx.artifacts.plotting import set_plot_style  # noqa: E402


DEFAULT_SPECTRUM = (
    ROOT
    / "docs/_static/quasilinear_vmec_qi_seed_linear_spectrum_scan.quasilinear_spectrum.csv"
)
DEFAULT_OUT = ROOT / "docs/_static/quasilinear_vmec_qi_seed_branch_refinement_gate.png"


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _repo_relative(path: str | Path) -> str:
    target = Path(path)
    try:
        return target.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def load_spectrum(path: str | Path) -> dict[str, np.ndarray]:
    """Load the minimal columns needed for branch refinement."""

    rows: list[dict[str, float]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"ky", "gamma", "omega"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"spectrum is missing required columns: {sorted(missing)}")
        for row in reader:
            ky = _finite_float(row.get("ky"))
            gamma = _finite_float(row.get("gamma"))
            omega = _finite_float(row.get("omega"))
            if ky is None or gamma is None or omega is None:
                continue
            rows.append({"ky": ky, "gamma": gamma, "omega": omega})
    if not rows:
        raise ValueError(f"spectrum has no finite ky/gamma/omega rows: {path}")
    rows.sort(key=lambda item: item["ky"])
    return {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in ("ky", "gamma", "omega")
    }


def _load_krylov(path: str | Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ky = _finite_float(payload.get("ky"))
    gamma = _finite_float(payload.get("gamma"))
    omega = _finite_float(payload.get("omega"))
    if ky is None or gamma is None or omega is None:
        raise ValueError(f"krylov summary lacks finite ky/gamma/omega: {path}")
    return {"ky": ky, "gamma": gamma, "omega": omega}


def _longest_positive_run(values: np.ndarray, *, threshold: float = 0.0) -> int:
    longest = 0
    current = 0
    for value in values:
        if float(value) > threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def build_qi_branch_refinement_report(
    spectrum: dict[str, np.ndarray],
    *,
    source: str | Path,
    krylov: dict[str, float] | None = None,
    min_launch_gamma: float = 0.02,
    min_points: int = 5,
    min_positive_run: int = 2,
    krylov_abs_tol: float = 2.5e-3,
) -> dict[str, Any]:
    """Return a JSON-ready QI branch-refinement gate report."""

    ky = np.asarray(spectrum["ky"], dtype=float)
    gamma = np.asarray(spectrum["gamma"], dtype=float)
    omega = np.asarray(spectrum["omega"], dtype=float)
    finite = np.isfinite(ky) & np.isfinite(gamma) & np.isfinite(omega)
    if not np.all(finite):
        ky = ky[finite]
        gamma = gamma[finite]
        omega = omega[finite]
    if ky.size == 0:
        raise ValueError("spectrum has no finite rows")
    imax = int(np.nanargmax(gamma))
    max_gamma = float(gamma[imax])
    max_ky = float(ky[imax])
    positive_run = _longest_positive_run(gamma)

    subgates: dict[str, dict[str, Any]] = {
        "finite_rows": {
            "passed": bool(ky.size >= int(min_points)),
            "value": int(ky.size),
            "threshold": int(min_points),
        },
        "positive_run": {
            "passed": bool(positive_run >= int(min_positive_run)),
            "value": int(positive_run),
            "threshold": int(min_positive_run),
        },
        "nonlinear_launch_growth": {
            "passed": bool(max_gamma >= float(min_launch_gamma)),
            "value": max_gamma,
            "threshold": float(min_launch_gamma),
        },
    }

    krylov_payload: dict[str, Any] | None = None
    if krylov is not None:
        idx = int(np.argmin(np.abs(ky - float(krylov["ky"]))))
        gamma_delta = abs(float(gamma[idx]) - float(krylov["gamma"]))
        omega_delta = abs(float(omega[idx]) - float(krylov["omega"]))
        subgates["krylov_consistency"] = {
            "passed": bool(gamma_delta <= float(krylov_abs_tol)),
            "gamma_abs_delta": gamma_delta,
            "omega_abs_delta": omega_delta,
            "threshold": float(krylov_abs_tol),
            "matched_ky": float(ky[idx]),
        }
        krylov_payload = {
            "ky": float(krylov["ky"]),
            "gamma": float(krylov["gamma"]),
            "omega": float(krylov["omega"]),
            "matched_time_branch_gamma": float(gamma[idx]),
            "matched_time_branch_omega": float(omega[idx]),
        }

    launch_ready = bool(
        subgates["finite_rows"]["passed"]
        and subgates["positive_run"]["passed"]
        and subgates["nonlinear_launch_growth"]["passed"]
        and subgates.get("krylov_consistency", {"passed": True})["passed"]
    )
    return {
        "kind": "qi_branch_refinement_gate",
        "claim_level": "linear_branch_refinement_not_nonlinear_transport_validation",
        "passed": launch_ready,
        "nonlinear_launch_ready": launch_ready,
        "absolute_flux_promoted": False,
        "source": _repo_relative(source),
        "n_ky": int(ky.size),
        "ky_min": float(np.min(ky)),
        "ky_max": float(np.max(ky)),
        "max_gamma": max_gamma,
        "max_gamma_ky": max_ky,
        "min_launch_gamma": float(min_launch_gamma),
        "positive_run_length": int(positive_run),
        "krylov_check": krylov_payload,
        "subgates": subgates,
        "notes": (
            "A finite near-marginal QI branch is useful branch-continuation evidence, "
            "but nonlinear transport holdout launches require max_gamma above the "
            "minimum launch threshold and the usual post-transient grid/window gates."
        ),
    }


def write_gate_figure(
    spectrum: dict[str, np.ndarray],
    report: dict[str, Any],
    *,
    out: str | Path = DEFAULT_OUT,
    dpi: int = 220,
    write_pdf: bool = True,
) -> dict[str, str]:
    """Write a compact branch-refinement figure plus optional PDF."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ky = np.asarray(spectrum["ky"], dtype=float)
    gamma = np.asarray(spectrum["gamma"], dtype=float)
    omega = np.asarray(spectrum["omega"], dtype=float)
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), constrained_layout=True)
    ax, ax_text = axes
    ax.plot(ky, gamma, "o-", color="#0f4c81", label=r"$\gamma$")
    ax.axhline(0.0, color="0.5", linewidth=1.0)
    ax.axhline(
        float(report["min_launch_gamma"]),
        color="#b44a3c",
        linestyle="--",
        label="launch threshold",
    )
    ax.set_xlabel(r"$k_y \rho_i$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title("QI seed linear branch")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(ky, omega, "s-", color="#c84d56", alpha=0.8, label=r"$\omega$")
    ax2.set_ylabel(r"$\omega$", color="#c84d56")
    ax2.tick_params(axis="y", colors="#c84d56")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="best")

    sub = report["subgates"]
    krylov = sub.get("krylov_consistency")
    lines = [
        f"Status: {'READY' if report['passed'] else 'BLOCKED'}",
        f"max gamma: {report['max_gamma']:.4g}",
        f"at ky: {report['max_gamma_ky']:.4g}",
        f"launch threshold: {report['min_launch_gamma']:.4g}",
        f"positive run length: {report['positive_run_length']}",
        "",
        "Subgates:",
        f"- finite rows: {sub['finite_rows']['passed']}",
        f"- positive run: {sub['positive_run']['passed']}",
        f"- launch growth: {sub['nonlinear_launch_growth']['passed']}",
    ]
    if krylov is not None:
        lines.append(f"- Krylov consistency: {krylov['passed']}")
        lines.append(f"  |delta gamma|={krylov['gamma_abs_delta']:.3g}")
    lines.extend(
        [
            "",
            "Claim boundary:",
            "linear branch refinement only;",
            "no nonlinear transport validation.",
        ]
    )
    ax_text.axis("off")
    ax_text.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=10.5,
    )
    fig.suptitle("QI seed branch-refinement gate", fontsize=14)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    outputs = {"png": str(out_path)}
    if write_pdf:
        pdf = out_path.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight")
        outputs["pdf"] = str(pdf)
    plt.close(fig)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", type=Path, default=DEFAULT_SPECTRUM)
    parser.add_argument("--krylov-summary", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-launch-gamma", type=float, default=0.02)
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument("--min-positive-run", type=int, default=2)
    parser.add_argument("--krylov-abs-tol", type=float, default=2.5e-3)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-pdf", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spectrum = load_spectrum(args.spectrum)
    krylov = _load_krylov(args.krylov_summary)
    report = build_qi_branch_refinement_report(
        spectrum,
        source=args.spectrum,
        krylov=krylov,
        min_launch_gamma=float(args.min_launch_gamma),
        min_points=int(args.min_points),
        min_positive_run=int(args.min_positive_run),
        krylov_abs_tol=float(args.krylov_abs_tol),
    )
    outputs = write_gate_figure(
        spectrum, report, out=args.out, dpi=int(args.dpi), write_pdf=not args.no_pdf
    )
    report = dict(report)
    report["png"] = _repo_relative(outputs["png"])
    if "pdf" in outputs:
        report["pdf"] = _repo_relative(outputs["pdf"])
    json_path = args.out.with_suffix(".json")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"passed": report["passed"], "json": str(json_path), **outputs},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bool(report["passed"]) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
