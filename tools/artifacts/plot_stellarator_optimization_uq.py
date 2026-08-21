#!/usr/bin/env python3
"""Plot UQ and sensitivity diagnostics for stellarator ITG optimization examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from gkx.artifacts.plotting import set_plot_style  # type: ignore[import-untyped]  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "docs" / "_static" / "stellarator_itg_optimization_comparison.json"
)
DEFAULT_OUT = ROOT / "docs" / "_static" / "stellarator_itg_optimization_uq.png"
COLORS = {
    "growth": "#1b6f8f",
    "quasilinear_flux": "#b55a30",
    "nonlinear_heat_flux": "#386641",
}


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("optimization comparison payload must be a JSON object")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(
            "optimization comparison payload must contain non-empty results"
        )
    return payload


def _short_param(name: str) -> str:
    return {
        "minor_radius_log_shift": r"$\Delta\log a$",
        "vertical_elongation_shift": r"$\Delta\kappa$",
        "helical_ripple_amplitude": r"$\epsilon_h$",
        "magnetic_shear_shift": r"$\Delta\hat{s}$",
    }.get(name, name)


def _kind_label(kind: str) -> str:
    return {
        "growth": "growth",
        "quasilinear_flux": "QL flux",
        "nonlinear_heat_flux": "NL window",
    }.get(kind, kind.replace("_", " "))


def _ellipse_from_covariance(
    ax: plt.Axes,
    center: np.ndarray,
    cov: np.ndarray,
    *,
    color: str,
    label: str,
    n_sigma: float = 1.0,
) -> None:
    cov2 = np.asarray(cov[:2, :2], dtype=float)
    cov2 = 0.5 * (cov2 + cov2.T)
    eigvals, eigvecs = np.linalg.eigh(cov2)
    eigvals = np.maximum(eigvals, 0.0)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = (
        float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
        if eigvecs.size
        else 0.0
    )
    width, height = 2.0 * n_sigma * np.sqrt(eigvals[:2])
    patch = Ellipse(
        xy=(float(center[0]), float(center[1])),
        width=float(width),
        height=float(height),
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.18,
        lw=1.8,
        label=label,
    )
    ax.add_patch(patch)
    ax.scatter(
        [center[0]],
        [center[1]],
        color=color,
        s=46,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )


def build_uq_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract machine-readable UQ and AD/FD diagnostics from optimization results."""

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("payload must contain non-empty results")
    parameter_names = list(
        payload.get("parameter_names") or results[0].get("parameter_names") or []
    )
    if not parameter_names:
        raise ValueError("payload must provide parameter names")

    rows: list[dict[str, Any]] = []
    all_gates = True
    full_rank = True
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("each optimization result must be a JSON object")
        kind = str(result.get("objective_kind", "unknown"))
        gate = (
            result.get("gradient_gate", {})
            if isinstance(result.get("gradient_gate"), dict)
            else {}
        )
        cov = (
            result.get("covariance", {})
            if isinstance(result.get("covariance"), dict)
            else {}
        )
        jac_ad = np.asarray(gate.get("jacobian_ad"), dtype=float)
        jac_fd = np.asarray(gate.get("jacobian_fd"), dtype=float)
        covariance = np.asarray(cov.get("covariance"), dtype=float)
        if jac_ad.shape != (1, len(parameter_names)) or jac_fd.shape != jac_ad.shape:
            raise ValueError(
                f"{kind}: gradient gate must contain 1 x n_parameter AD/FD Jacobians"
            )
        if covariance.shape != (len(parameter_names), len(parameter_names)):
            raise ValueError(f"{kind}: covariance must be n_parameter x n_parameter")
        rank = int(cov.get("sensitivity_map_rank", 0))
        all_gates = all_gates and bool(gate.get("passed", False))
        full_rank = full_rank and rank == len(parameter_names)
        rows.append(
            {
                "objective_kind": kind,
                "gradient_gate_passed": bool(gate.get("passed", False)),
                "max_abs_error": float(gate.get("max_abs_error", np.nan)),
                "max_rel_error": float(gate.get("max_rel_error", np.nan)),
                "tangent_max_abs_error": float(
                    gate.get("tangent_max_abs_error", np.nan)
                ),
                "jacobian_ad": jac_ad.reshape(-1).tolist(),
                "jacobian_fd": jac_fd.reshape(-1).tolist(),
                "covariance_std": list(
                    map(float, np.asarray(cov.get("covariance_std"), dtype=float))
                ),
                "covariance_correlation": np.asarray(
                    cov.get("covariance_correlation"), dtype=float
                ).tolist(),
                "jacobian_singular_values": list(
                    map(
                        float,
                        np.asarray(cov.get("jacobian_singular_values"), dtype=float),
                    )
                ),
                "jacobian_condition_number": float(
                    cov.get("jacobian_condition_number", np.nan)
                ),
                "sensitivity_map_rank": rank,
                "uq_ellipse_area_1sigma": float(
                    cov.get("uq_ellipse_area_1sigma", np.nan)
                ),
                "initial_params": list(map(float, result.get("initial_params", []))),
                "final_params": list(map(float, result.get("final_params", []))),
                "covariance": covariance.tolist(),
            }
        )

    return {
        "kind": "stellarator_itg_optimization_uq",
        "claim_level": "reduced_objective_uq_and_sensitivity_validation_not_full_vmec_gk_optimization",
        "parameter_names": parameter_names,
        "all_gradient_gates_passed": bool(all_gates),
        "all_sensitivity_maps_full_rank": bool(full_rank),
        "parallel": payload.get(
            "parallel",
            {
                "requested_workers": 1,
                "effective_workers": 1,
                "executor": "thread",
                "finite_difference_workers": 1,
                "finite_difference_executor": "thread",
                "identity_contract": "serial payload or archived artifact",
            },
        ),
        "results": rows,
    }


def write_uq_figure(
    summary: dict[str, Any], *, out: Path = DEFAULT_OUT
) -> dict[str, str]:
    """Write PNG/PDF/JSON artifacts for the optimization UQ summary."""

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    pdf_path = out.with_suffix(".pdf")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(summary, out)
    _plot(summary, pdf_path)
    return {"png": str(out), "pdf": str(pdf_path), "json": str(json_path)}


def _plot(summary: dict[str, Any], path: Path) -> None:
    set_plot_style()
    params = list(summary["parameter_names"])
    labels = [_short_param(name) for name in params]
    results = list(summary["results"])
    kinds = [str(row["objective_kind"]) for row in results]
    colors = [COLORS.get(kind, "#52525b") for kind in kinds]
    x = np.arange(len(params))

    fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)
    fig.suptitle(
        "Stellarator ITG optimization: AD/FD, sensitivity, and UQ gates",
        fontsize=14,
        fontweight="bold",
    )

    width = 0.23
    for i, row in enumerate(results):
        ad = np.asarray(row["jacobian_ad"], dtype=float)
        fd = np.asarray(row["jacobian_fd"], dtype=float)
        offset = (i - (len(results) - 1) / 2.0) * width
        axs[0, 0].bar(
            x + offset,
            np.abs(ad - fd),
            width=width,
            color=colors[i],
            label=_kind_label(kinds[i]),
        )
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_xticks(x, labels)
    axs[0, 0].set_ylabel(r"$|\partial J_{AD}-\partial J_{FD}|$")
    axs[0, 0].set_title("Gradient parity by control")
    axs[0, 0].legend(frameon=False, fontsize=8)
    axs[0, 0].grid(axis="y", alpha=0.25)

    for i, row in enumerate(results):
        std = np.asarray(row["covariance_std"], dtype=float)
        offset = (i - (len(results) - 1) / 2.0) * width
        axs[0, 1].bar(
            x + offset, std, width=width, color=colors[i], label=_kind_label(kinds[i])
        )
    axs[0, 1].set_xticks(x, labels)
    axs[0, 1].set_ylabel("posterior std. (Gauss-Newton)")
    axs[0, 1].set_title("Control uncertainty scale")
    axs[0, 1].grid(axis="y", alpha=0.25)

    initial = np.asarray(results[0]["initial_params"], dtype=float)
    axs[1, 0].scatter(
        [initial[0]],
        [initial[1]],
        marker="x",
        s=70,
        color="#111827",
        label="shared initial",
    )
    for i, row in enumerate(results):
        final = np.asarray(row["final_params"], dtype=float)
        cov = np.asarray(row["covariance"], dtype=float)
        _ellipse_from_covariance(
            axs[1, 0],
            final[:2],
            cov[:2, :2],
            color=colors[i],
            label=_kind_label(kinds[i]),
        )
    axs[1, 0].set_xlabel(labels[0])
    axs[1, 0].set_ylabel(labels[1])
    axs[1, 0].set_title("1-sigma covariance projection")
    axs[1, 0].legend(frameon=False, fontsize=8)
    axs[1, 0].grid(alpha=0.25)

    max_corr = []
    cond = []
    rank = []
    ellipse = []
    for row in results:
        corr = np.asarray(row["covariance_correlation"], dtype=float)
        off_diag = (
            corr[~np.eye(corr.shape[0], dtype=bool)]
            if corr.ndim == 2
            else np.asarray([])
        )
        max_corr.append(float(np.max(np.abs(off_diag))) if off_diag.size else 0.0)
        cond.append(float(row["jacobian_condition_number"]))
        rank.append(float(row["sensitivity_map_rank"]))
        ellipse.append(float(row["uq_ellipse_area_1sigma"]))
    xpos = np.arange(len(results))
    axs[1, 1].bar(
        xpos - 0.20, max_corr, width=0.20, color="#64748b", label="max |corr|"
    )
    axs[1, 1].bar(
        xpos,
        np.log10(np.maximum(cond, 1.0)),
        width=0.20,
        color="#0f766e",
        label=r"$\log_{10}\kappa(J)$",
    )
    axs[1, 1].bar(xpos + 0.20, rank, width=0.20, color="#92400e", label="rank")
    axs[1, 1].set_xticks(xpos, [_kind_label(kind) for kind in kinds], rotation=0)
    axs[1, 1].set_title("Identifiability diagnostics")
    axs[1, 1].set_ylabel("diagnostic value")
    axs[1, 1].legend(frameon=False, fontsize=8)
    axs[1, 1].grid(axis="y", alpha=0.25)

    for ax in axs.ravel():
        for spine in ax.spines.values():
            spine.set_alpha(0.35)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Optimization comparison JSON payload.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Output PNG path."
    )
    args = parser.parse_args()

    summary = build_uq_summary(_load_payload(args.input))
    paths = write_uq_figure(summary, out=args.out)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
