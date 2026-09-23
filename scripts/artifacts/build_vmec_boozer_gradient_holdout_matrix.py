#!/usr/bin/env python3
"""Summarize full-chain VMEC/Boozer gradient gates across equilibria."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gkx.artifacts.plotting import set_plot_style  # type: ignore[import-untyped]  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_boozer_gradient_holdout_matrix.png"

DEFAULT_GATES = (
    (
        "QH warm start",
        "frequency",
        ROOT / "docs/_static/vmec_boozer_solver_frequency_gradient_gate.json",
    ),
    (
        "QH warm start",
        "quasilinear",
        ROOT / "docs/_static/vmec_boozer_quasilinear_gradient_gate.json",
    ),
    (
        "QH warm start",
        "nonlinear-window estimator",
        ROOT / "docs/_static/vmec_boozer_nonlinear_window_gradient_gate.json",
    ),
    (
        "Li383",
        "frequency",
        ROOT / "docs/_static/vmec_boozer_li383_solver_frequency_gradient_gate.json",
    ),
    (
        "Li383",
        "quasilinear",
        ROOT / "docs/_static/vmec_boozer_li383_quasilinear_gradient_gate.json",
    ),
    (
        "Li383",
        "nonlinear-window estimator",
        ROOT / "docs/_static/vmec_boozer_li383_nonlinear_window_gradient_gate.json",
    ),
)

OBJECTIVE_ORDER = (
    "gamma",
    "omega",
    "kperp_eff2",
    "linear_heat_flux_weight",
    "mixing_length_heat_flux_proxy",
    "nonlinear_window_heat_flux_mean",
    "nonlinear_window_heat_flux_cv",
    "nonlinear_window_heat_flux_trend",
)

OBJECTIVE_LABELS = {
    "gamma": r"$\gamma$",
    "omega": r"$\omega$",
    "kperp_eff2": r"$\langle k_\perp^2\rangle$",
    "linear_heat_flux_weight": r"$\hat Q_i$",
    "mixing_length_heat_flux_proxy": r"$\gamma\hat Q_i/k_\perp^2$",
    "nonlinear_window_heat_flux_mean": r"$\langle Q_i\rangle_\mathrm{win}$",
    "nonlinear_window_heat_flux_cv": "window CV",
    "nonlinear_window_heat_flux_trend": "window trend",
}


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


def _finite_max(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return None if not finite else max(finite)


def _load_gate(label: str, gate_type: str, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    objective_gates = payload.get("objective_gates", [])
    if not isinstance(objective_gates, list) or not objective_gates:
        raise ValueError(f"{path} has no objective_gates list")
    rel_errors: list[float] = []
    abs_errors: list[float] = []
    objective_status: dict[str, bool] = {}
    objective_rel_error: dict[str, float | None] = {}
    for gate in objective_gates:
        if not isinstance(gate, dict):
            continue
        objective = gate.get("objective")
        if objective is None:
            continue
        objective_name = str(objective)
        objective_status[objective_name] = bool(gate.get("passed", False))
        rel_error = gate.get("rel_error")
        if rel_error is None:
            objective_rel_error[objective_name] = None
        else:
            rel_error_float = float(rel_error)
            objective_rel_error[objective_name] = rel_error_float
            rel_errors.append(rel_error_float)
        abs_error = gate.get("abs_error")
        if abs_error is not None:
            abs_errors.append(float(abs_error))
    return {
        "label": label,
        "gate_type": gate_type,
        "path": str(path),
        "case_name": str(payload.get("case_name", label)),
        "passed": bool(payload.get("passed", False)),
        "source_scope": str(payload.get("source_scope", "")),
        "mboz": payload.get("mboz"),
        "nboz": payload.get("nboz"),
        "surface_stencil_width": payload.get("surface_stencil_width"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "n_objectives": len(objective_gates),
        "max_rel_error": _finite_max(rel_errors),
        "max_abs_error": _finite_max(abs_errors),
        "objectives": objective_status,
        "objective_rel_error": objective_rel_error,
    }


def build_gradient_holdout_matrix(
    gate_specs: tuple[tuple[str, str, Path], ...] = DEFAULT_GATES,
) -> dict[str, Any]:
    """Return a JSON-ready multi-equilibrium gradient holdout matrix."""

    rows = [
        _load_gate(label, gate_type, Path(path))
        for label, gate_type, path in gate_specs
    ]
    all_mode21 = all(row["source_scope"] == "mode21_vmec_boozer_state" for row in rows)
    all_mode_counts = all(
        int(row["mboz"]) >= 21 and int(row["nboz"]) >= 21 for row in rows
    )
    all_passed = all(bool(row["passed"]) for row in rows)
    cases = sorted({str(row["case_name"]) for row in rows})
    gate_types = sorted({str(row["gate_type"]) for row in rows})
    return {
        "kind": "vmec_boozer_gradient_holdout_matrix",
        "claim_level": (
            "multi_equilibrium_reduced_linear_quasilinear_and_nonlinear_window_estimator_gradient_gate_"
            "not_production_nonlinear_optimization"
        ),
        "passed": bool(all_passed and all_mode21 and all_mode_counts),
        "summary": {
            "n_cases": len(cases),
            "cases": cases,
            "gate_types": gate_types,
            "all_gates_passed": all_passed,
            "all_mode21_source_scope": all_mode21,
            "all_mboz_nboz_at_least_21": all_mode_counts,
            "max_relative_error": _finite_max(
                [
                    float(row["max_rel_error"])
                    for row in rows
                    if row["max_rel_error"] is not None
                ]
            ),
        },
        "rows": rows,
        "notes": (
            "This matrix checks differentiability from vmex state coefficients through "
            "booz_xform_jax mode-21 equal-arc geometry into GKX linear and "
            "quasilinear solver observables plus a reduced nonlinear-window estimator. "
            "It does not validate converged nonlinear-window turbulence gradients or "
            "optimized-equilibrium nonlinear transport."
        ),
    }


def write_gradient_holdout_matrix(
    payload: dict[str, Any],
    *,
    out: str | Path = DEFAULT_OUT,
    title: str = "VMEC/Boozer full-chain gradient holdouts",
) -> dict[str, str]:
    """Write PNG/PDF/JSON/CSV artifacts for the gradient holdout matrix."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    rows = list(payload["rows"])
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "case_name",
                "gate_type",
                "passed",
                "max_rel_error",
                "max_abs_error",
                "n_objectives",
                "mboz",
                "nboz",
                "surface_stencil_width",
                "path",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    set_plot_style()
    labels = [f"{row['label']}\n{row['gate_type']}" for row in rows]
    y = np.arange(len(rows))
    max_rel = np.asarray(
        [
            1.0e-8
            if row["max_rel_error"] is None
            else max(float(row["max_rel_error"]), 1.0e-8)
            for row in rows
        ],
        dtype=float,
    )
    colors = ["#2a9d8f" if bool(row["passed"]) else "#d1495b" for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.8), constrained_layout=True)
    ax0, ax1 = axes
    ax0.barh(y, max_rel, color=colors, edgecolor="#1f2937")
    ax0.axvline(
        2.0e-2, color="#111827", linestyle="--", linewidth=1.2, label="2e-2 QL gate"
    )
    ax0.axvline(
        5.0e-2,
        color="#6b7280",
        linestyle=":",
        linewidth=1.2,
        label="5e-2 frequency gate",
    )
    ax0.axvline(
        7.5e-2,
        color="#8b5e34",
        linestyle="-.",
        linewidth=1.2,
        label="7.5e-2 estimator gate",
    )
    for yi, value in zip(y, max_rel, strict=True):
        ax0.text(value * 1.15, yi, f"{value:.2g}", va="center", fontsize=8)
    ax0.set_xscale("log")
    ax0.set_yticks(y, labels)
    ax0.invert_yaxis()
    ax0.set_xlabel("maximum AD/finite-difference relative error")
    ax0.set_title("Holdout gradient accuracy")
    ax0.grid(True, axis="x", which="both", alpha=0.22)
    ax0.legend(loc="lower right", fontsize=8)

    matrix = np.full((len(rows), len(OBJECTIVE_ORDER)), np.nan)
    rel_matrix = np.full_like(matrix, np.nan, dtype=float)
    for i, row in enumerate(rows):
        for j, objective in enumerate(OBJECTIVE_ORDER):
            if objective not in row["objectives"]:
                continue
            matrix[i, j] = 1.0 if bool(row["objectives"][objective]) else 0.0
            rel = row["objective_rel_error"].get(objective)
            if rel is not None:
                rel_matrix[i, j] = float(rel)
    masked = np.ma.masked_invalid(matrix)
    cmap = matplotlib.colors.ListedColormap(["#d1495b", "#2a9d8f"])
    cmap.set_bad("#e5e7eb")
    ax1.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax1.set_xticks(
        np.arange(len(OBJECTIVE_ORDER)),
        [OBJECTIVE_LABELS[name] for name in OBJECTIVE_ORDER],
        rotation=25,
        ha="right",
    )
    ax1.set_yticks(y, labels)
    ax1.set_title("Objective-level pass matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                text = "--"
                color = "#374151"
            else:
                text = "pass" if matrix[i, j] > 0.5 else "fail"
                if math.isfinite(rel_matrix[i, j]):
                    text += f"\n{rel_matrix[i, j]:.1e}"
                color = "white"
            ax1.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7.5,
                color=color,
                fontweight="bold" if text.startswith("pass") else "normal",
            )
    ax1.set_xlim(-0.5, len(OBJECTIVE_ORDER) - 0.5)
    ax1.set_ylim(len(rows) - 0.5, -0.5)
    ax1.set_xlabel("differentiated observable")

    fig.suptitle(title, y=1.03, fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.015,
        "Closed for reduced linear/quasilinear VMEC/Boozer objectives and reduced nonlinear-window estimators; production nonlinear gradients remain separate.",
        ha="center",
        fontsize=8.8,
        color="#333333",
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "png": str(out_path),
        "pdf": str(pdf_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--title", default="VMEC/Boozer full-chain gradient holdouts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_gradient_holdout_matrix()
    paths = write_gradient_holdout_matrix(payload, out=args.out, title=args.title)
    print(json.dumps(paths, indent=2, sort_keys=True))
    print(
        "passed={passed} cases={cases} max_relative_error={max_rel}".format(
            passed=payload["passed"],
            cases=",".join(payload["summary"]["cases"]),
            max_rel=payload["summary"]["max_relative_error"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
