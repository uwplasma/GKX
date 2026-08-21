#!/usr/bin/env python3
"""Build a branch-level audit for the provisional TEM parity lane."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import textwrap
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gkx.artifacts.plotting import set_plot_style  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = ROOT / "docs" / "_static" / "tem_mismatch_table.csv"
DEFAULT_REFERENCE = ROOT / "src" / "gkx" / "data" / "tem_reference.csv"
DEFAULT_BRANCH_OUT = ROOT / "docs" / "_static" / "tem_branch_parity_audit.png"

REFERENCE_PROVENANCE = (
    "Digitized low-ky TEM branch from the literature figure tracked in "
    "src/gkx/data/tem_reference.csv; this is not a direct code-to-code dump."
)

REQUIRED_COLUMNS = (
    "ky",
    "gamma_ref",
    "omega_ref",
    "gamma_gkx",
    "omega_gkx",
    "rel_gamma",
    "rel_omega",
)


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    unique = np.unique(values)
    for val in unique:
        mask = values == val
        if np.count_nonzero(mask) > 1:
            ranks[mask] = float(np.mean(ranks[mask]))
    return ranks


def _corrcoef(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    return _corrcoef(_rankdata(x), _rankdata(y))


def _load_table(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in reader:
            parsed: dict[str, float] = {}
            for column in REQUIRED_COLUMNS:
                value = _finite_float(row.get(column))
                if value is None:
                    raise ValueError(
                        f"{path} contains non-finite {column!r}: {row.get(column)!r}"
                    )
                parsed[column] = value
            rows.append(parsed)
    if not rows:
        raise ValueError(f"{path} contains no TEM mismatch rows")
    return sorted(rows, key=lambda row: row["ky"])


def _read_reference_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "n_rows": 0}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "available": True,
        "path": str(path),
        "n_rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "provenance": REFERENCE_PROVENANCE,
    }


def build_branch_audit_payload(
    *,
    table: Path = DEFAULT_TABLE,
    reference: Path = DEFAULT_REFERENCE,
) -> dict[str, Any]:
    """Return a JSON-ready TEM branch parity audit payload."""

    rows = _load_table(Path(table))
    ky = np.array([row["ky"] for row in rows], dtype=float)
    gamma_ref = np.array([row["gamma_ref"] for row in rows], dtype=float)
    omega_ref = np.array([row["omega_ref"] for row in rows], dtype=float)
    gamma_gkx = np.array([row["gamma_gkx"] for row in rows], dtype=float)
    omega_gkx = np.array([row["omega_gkx"] for row in rows], dtype=float)
    rel_gamma = np.array([row["rel_gamma"] for row in rows], dtype=float)
    rel_omega = np.array([row["rel_omega"] for row in rows], dtype=float)

    gamma_error = gamma_gkx - gamma_ref
    omega_error = omega_gkx - omega_ref
    gamma_abs_error = np.abs(gamma_error)
    omega_abs_error = np.abs(omega_error)
    gamma_sign_mismatch = np.signbit(gamma_ref) != np.signbit(gamma_gkx)
    omega_sign_mismatch = np.signbit(omega_ref) != np.signbit(omega_gkx)
    safe_omega = np.abs(omega_ref) >= 0.2
    if np.any(safe_omega):
        max_safe_rel_omega = float(np.max(np.abs(rel_omega[safe_omega])))
        worst_safe_rel_omega_ky = float(
            ky[safe_omega][int(np.argmax(np.abs(rel_omega[safe_omega])))]
        )
    else:
        max_safe_rel_omega = None
        worst_safe_rel_omega_ky = None

    omega_spearman = _spearman(omega_ref, omega_gkx)
    gamma_spearman = _spearman(gamma_ref, gamma_gkx)
    branch_inversion = omega_spearman is not None and omega_spearman < 0.0

    worst_gamma_idx = int(np.argmax(gamma_abs_error))
    worst_omega_idx = int(np.argmax(omega_abs_error))
    raw_worst_gamma_rel_idx = int(np.argmax(np.abs(rel_gamma)))
    raw_worst_omega_rel_idx = int(np.argmax(np.abs(rel_omega)))
    status = "open"
    rows_out = []
    for i, row in enumerate(rows):
        rows_out.append(
            {
                **row,
                "gamma_error": float(gamma_error[i]),
                "omega_error": float(omega_error[i]),
                "abs_gamma_error": float(gamma_abs_error[i]),
                "abs_omega_error": float(omega_abs_error[i]),
                "gamma_sign_mismatch": bool(gamma_sign_mismatch[i]),
                "omega_sign_mismatch": bool(omega_sign_mismatch[i]),
            }
        )

    return {
        "kind": "tem_branch_parity_audit",
        "status": status,
        "claim_level": "provisional_literature_digitization_not_closed_parity",
        "source_table": str(table),
        "reference": _read_reference_metadata(Path(reference)),
        "metrics": {
            "n_ky": int(ky.size),
            "ky_min": float(np.min(ky)),
            "ky_max": float(np.max(ky)),
            "max_abs_rel_gamma": float(np.max(np.abs(rel_gamma))),
            "max_abs_rel_gamma_ky": float(ky[raw_worst_gamma_rel_idx]),
            "max_abs_rel_omega_raw": float(np.max(np.abs(rel_omega))),
            "max_abs_rel_omega_raw_ky": float(ky[raw_worst_omega_rel_idx]),
            "max_abs_rel_omega_ref_ge_0p2": max_safe_rel_omega,
            "max_abs_rel_omega_ref_ge_0p2_ky": worst_safe_rel_omega_ky,
            "max_abs_gamma_error": float(gamma_abs_error[worst_gamma_idx]),
            "max_abs_gamma_error_ky": float(ky[worst_gamma_idx]),
            "max_abs_omega_error": float(omega_abs_error[worst_omega_idx]),
            "max_abs_omega_error_ky": float(ky[worst_omega_idx]),
            "gamma_rmse": float(np.sqrt(np.mean(gamma_error**2))),
            "omega_rmse": float(np.sqrt(np.mean(omega_error**2))),
            "gamma_sign_mismatch_count": int(np.count_nonzero(gamma_sign_mismatch)),
            "omega_sign_mismatch_count": int(np.count_nonzero(omega_sign_mismatch)),
            "gamma_spearman": gamma_spearman,
            "omega_spearman": omega_spearman,
            "omega_branch_inversion": bool(branch_inversion),
        },
        "rows": rows_out,
        "interpretation": (
            "The tracked GKX low-ky TEM branch does not match the digitized reference: "
            "growth-rate amplitude/sign and frequency branch orientation fail simultaneously. "
            "Because the reference is a reconstructed literature digitization rather than a direct "
            "case dump, this artifact blocks broad TEM validation claims but should not be used by "
            "itself to tune solver physics."
        ),
        "next_action": (
            "Reconstruct the full TEM case definition or obtain an independent reference dump, then "
            "rerun a solver/signal/moment-resolution branch audit before promoting W7-X kinetic-electron claims."
        ),
    }


def write_branch_artifacts(
    payload: dict[str, Any], *, out_png: Path = DEFAULT_BRANCH_OUT
) -> dict[str, str]:
    """Write PNG/PDF/JSON/CSV artifacts for a TEM audit payload."""

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_png.with_suffix(".pdf")
    out_json = out_png.with_suffix(".json")
    out_csv = out_png.with_suffix(".csv")
    out_json.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = payload["rows"]
    fieldnames = [
        "ky",
        "gamma_ref",
        "omega_ref",
        "gamma_gkx",
        "omega_gkx",
        "rel_gamma",
        "rel_omega",
        "gamma_error",
        "omega_error",
        "gamma_sign_mismatch",
        "omega_sign_mismatch",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    set_plot_style()
    ky = np.array([row["ky"] for row in rows], dtype=float)
    gamma_ref = np.array([row["gamma_ref"] for row in rows], dtype=float)
    omega_ref = np.array([row["omega_ref"] for row in rows], dtype=float)
    gamma_gkx = np.array([row["gamma_gkx"] for row in rows], dtype=float)
    omega_gkx = np.array([row["omega_gkx"] for row in rows], dtype=float)
    gamma_error = np.abs(gamma_gkx - gamma_ref)
    omega_error = np.abs(omega_gkx - omega_ref)
    metrics = payload["metrics"]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    ax = axes[0, 0]
    ax.plot(ky, gamma_ref, "o-", label="digitized reference", color="#264653")
    ax.plot(ky, gamma_gkx, "s--", label="GKX tracked", color="#e76f51")
    ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"$k_y \rho_i$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title("Growth-rate branch")
    ax.legend(frameon=False, fontsize=8.0)

    ax = axes[0, 1]
    ax.plot(ky, omega_ref, "o-", label="digitized reference", color="#264653")
    ax.plot(ky, omega_gkx, "s--", label="GKX tracked", color="#e76f51")
    ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"$k_y \rho_i$")
    ax.set_ylabel(r"$\omega$")
    ax.set_title("Frequency branch")
    ax.legend(frameon=False, fontsize=8.0)

    ax = axes[1, 0]
    width = 0.017
    ax.bar(
        ky - width / 2,
        gamma_error,
        width=width,
        label=r"$|\Delta\gamma|$",
        color="#2a9d8f",
    )
    ax.bar(
        ky + width / 2,
        omega_error,
        width=width,
        label=r"$|\Delta\omega|$",
        color="#e9c46a",
    )
    ax.set_xlabel(r"$k_y \rho_i$")
    ax.set_ylabel("absolute error")
    ax.set_title("Pointwise mismatch")
    ax.legend(frameon=False, fontsize=8.0)

    ax = axes[1, 1]
    ax.axis("off")
    text = "\n".join(
        [
            "TEM lane status: OPEN",
            f"max |rel gamma| = {metrics['max_abs_rel_gamma']:.3g}",
            f"max |rel omega|, |omega_ref|>=0.2 = {metrics['max_abs_rel_omega_ref_ge_0p2']:.3g}",
            f"max |Delta gamma| = {metrics['max_abs_gamma_error']:.3g}",
            f"max |Delta omega| = {metrics['max_abs_omega_error']:.3g}",
            f"gamma sign mismatches = {metrics['gamma_sign_mismatch_count']}",
            f"omega sign mismatches = {metrics['omega_sign_mismatch_count']}",
            f"omega Spearman = {metrics['omega_spearman']:.3g}",
            "",
            textwrap.fill(
                "Reference provenance: provisional literature digitization, not a direct case dump. "
                "This blocks TEM/W7-X kinetic-electron validation claims without being a tuning target.",
                width=48,
            ),
        ]
    )
    ax.text(0.0, 1.0, text, transform=ax.transAxes, va="top", ha="left", fontsize=9.0)

    fig.suptitle("Tracked TEM branch parity audit", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return {
        "png": str(out_png),
        "pdf": str(out_pdf),
        "json": str(out_json),
        "csv": str(out_csv),
    }




DEFAULT_W7X_OUT = ROOT / "docs" / "_static" / "w7x_tem_extension_status.png"
DEFAULT_W7X_SPECTRUM = ROOT / "docs" / "_static" / "w7x_fluctuation_spectrum_panel.json"
DEFAULT_TEM_TABLE = ROOT / "docs" / "_static" / "tem_mismatch_table.csv"
DEFAULT_TEM_AUDIT = ROOT / "docs" / "_static" / "tem_branch_parity_audit.json"

STATUS_ORDER = {"closed": 0, "partial": 1, "open": 2}
STATUS_COLORS = {"closed": "#2a9d8f", "partial": "#e9c46a", "open": "#f4a261"}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _tem_metrics(path: Path, *, audit_path: Path | None = None) -> dict[str, Any]:
    if audit_path is not None and audit_path.exists():
        audit = _read_json(audit_path)
        metrics = audit.get("metrics", {}) if audit else {}
        if not isinstance(metrics, dict):
            metrics = {}
        return {
            "available": True,
            "audit_available": True,
            "audit_status": audit.get("status") if audit else None,
            "audit_claim_level": audit.get("claim_level") if audit else None,
            "n_rows": metrics.get("n_ky"),
            "max_abs_rel_gamma": metrics.get("max_abs_rel_gamma"),
            "max_abs_rel_omega": metrics.get("max_abs_rel_omega_ref_ge_0p2"),
            "max_abs_rel_omega_raw": metrics.get("max_abs_rel_omega_raw"),
            "max_abs_gamma_error": metrics.get("max_abs_gamma_error"),
            "max_abs_omega_error": metrics.get("max_abs_omega_error"),
            "gamma_sign_mismatch_count": metrics.get("gamma_sign_mismatch_count"),
            "omega_sign_mismatch_count": metrics.get("omega_sign_mismatch_count"),
            "omega_spearman": metrics.get("omega_spearman"),
            "omega_branch_inversion": metrics.get("omega_branch_inversion"),
            "source": str(audit_path),
        }
    if not path.exists():
        return {
            "available": False,
            "audit_available": False,
            "n_rows": 0,
            "max_abs_rel_gamma": None,
            "max_abs_rel_omega": None,
            "worst_gamma_ky": None,
            "worst_omega_ky": None,
        }
    rows: list[dict[str, str]] = list(
        csv.DictReader(path.open(newline="", encoding="utf-8"))
    )
    gamma_values: list[tuple[float, float]] = []
    omega_values: list[tuple[float, float]] = []
    for row in rows:
        ky = _finite_float(row.get("ky"))
        rel_gamma = _finite_float(row.get("rel_gamma"))
        rel_omega = _finite_float(row.get("rel_omega"))
        if ky is not None and rel_gamma is not None:
            gamma_values.append((ky, abs(rel_gamma)))
        if ky is not None and rel_omega is not None:
            omega_values.append((ky, abs(rel_omega)))
    worst_gamma = (
        max(gamma_values, key=lambda item: item[1]) if gamma_values else (None, None)
    )
    worst_omega = (
        max(omega_values, key=lambda item: item[1]) if omega_values else (None, None)
    )
    return {
        "available": True,
        "audit_available": False,
        "n_rows": len(rows),
        "max_abs_rel_gamma": worst_gamma[1],
        "max_abs_rel_omega": worst_omega[1],
        "worst_gamma_ky": worst_gamma[0],
        "worst_omega_ky": worst_omega[0],
    }


def build_w7x_status_payload(
    *,
    w7x_spectrum: Path = DEFAULT_W7X_SPECTRUM,
    tem_table: Path = DEFAULT_TEM_TABLE,
    tem_audit: Path = DEFAULT_TEM_AUDIT,
) -> dict[str, Any]:
    """Return a JSON-ready W7-X/TEM validation extension status payload."""

    spectrum = _read_json(Path(w7x_spectrum))
    tem = _tem_metrics(Path(tem_table), audit_path=Path(tem_audit))
    spectrum_closed = bool(spectrum and spectrum.get("source_gate_passed") is True)
    tem_gamma = _finite_float(tem.get("max_abs_rel_gamma"))
    tem_omega = _finite_float(tem.get("max_abs_rel_omega"))
    if bool(tem.get("audit_available")):
        tem_open = str(tem.get("audit_status")) != "closed"
    else:
        tem_open = (
            tem_gamma is None or tem_omega is None or tem_gamma > 0.2 or tem_omega > 0.2
        )
    rows = [
        {
            "lane": "W7-X nonlinear fluctuation spectrum",
            "status": "closed" if spectrum_closed else "open",
            "claim_level": "validated_simulation_spectrum_not_experimental_validation",
            "primary_artifact": "docs/_static/w7x_fluctuation_spectrum_panel.json",
            "key_metrics": {
                "time_samples": None
                if spectrum is None
                else spectrum.get("time_samples"),
                "dominant_phi_ky": None
                if spectrum is None
                else spectrum.get("dominant_phi_ky"),
                "dominant_heat_flux_ky": None
                if spectrum is None
                else spectrum.get("dominant_heat_flux_ky"),
            },
            "next_action": (
                "Keep as the released simulation-spectrum diagnostic; do not use as an experimental density-spectrum claim."
            ),
        },
        {
            "lane": "TEM / kinetic-electron linear parity",
            "status": "open" if tem_open else "closed",
            "claim_level": "open_linear_mismatch_blocks_tem_extension",
            "primary_artifact": (
                "docs/_static/tem_branch_parity_audit.json"
                if bool(tem.get("audit_available"))
                else "docs/_static/tem_mismatch_table.csv"
            ),
            "key_metrics": tem,
            "next_action": (
                "Reconstruct the TEM case definition or obtain an independent reference dump, then fix the branch/frequency "
                "mismatch before using kinetic-electron W7-X scans for validation or optimization."
            ),
        },
        {
            "lane": "W7-X multi-flux-tube and multi-surface scan",
            "status": "open",
            "claim_level": "untracked_required_manuscript_extension",
            "primary_artifact": None,
            "key_metrics": {
                "required_alpha_values": [
                    0.0,
                    "field-period displaced",
                    "bad-curvature tube",
                ],
                "required_torflux_values": [0.25, 0.50, 0.64],
                "requires_nonlinear_window_gates": True,
            },
            "next_action": (
                "Run alpha/surface-resolved W7-X ITG scans and promote only after linear branch, nonlinear window, "
                "and resolved-spectrum gates pass."
            ),
        },
        {
            "lane": "W7-X kinetic-electron/TEM nonlinear window",
            "status": "open",
            "claim_level": "not_started",
            "primary_artifact": None,
            "key_metrics": {
                "requires_kinetic_electrons": True,
                "requires_density_gradient_scan": True,
                "requires_independent_reference_or_literature_window": True,
            },
            "next_action": (
                "After TEM linear parity closes, run bounded kinetic-electron W7-X pilots and add matched nonlinear gates."
            ),
        },
    ]
    return {
        "kind": "w7x_tem_extension_status",
        "claim_scope": "w7x_fluctuation_tem_multi_flux_extension_tracking",
        "rows": rows,
        "summary": {
            "n_rows": len(rows),
            "n_closed": sum(1 for row in rows if row["status"] == "closed"),
            "n_partial": sum(1 for row in rows if row["status"] == "partial"),
            "n_open": sum(1 for row in rows if row["status"] == "open"),
        },
        "notes": (
            "This artifact is a claim-scope gate. It closes the reproducible W7-X simulation-spectrum "
            "estimator only; TEM, multi-flux-tube, and kinetic-electron W7-X validation remain open."
        ),
    }


def write_w7x_status_artifacts(
    payload: dict[str, Any], *, out_png: Path = DEFAULT_W7X_OUT
) -> dict[str, str]:
    """Write PNG/PDF/JSON/CSV artifacts for the extension-status payload."""

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_png.with_suffix(".json")
    out_csv = out_png.with_suffix(".csv")
    out_pdf = out_png.with_suffix(".pdf")
    out_json.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fieldnames = ["lane", "status", "claim_level", "primary_artifact", "next_action"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow({key: row.get(key) for key in fieldnames})

    set_plot_style()
    rows = payload["rows"]
    y = np.arange(len(rows))
    values = [STATUS_ORDER[str(row["status"])] for row in rows]
    labels = [textwrap.fill(str(row["lane"]), width=34) for row in rows]
    colors = [STATUS_COLORS[str(row["status"])] for row in rows]
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.barh(y, values, color=colors, edgecolor="#333333", alpha=0.95)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 2.25)
    ax.set_xticks([0, 1, 2], ["closed", "partial", "open"])
    ax.invert_yaxis()
    ax.set_title("W7-X fluctuation, multi-flux, and TEM extension status")
    ax.grid(axis="x", alpha=0.25)
    for yi, row, value in zip(y, rows, values, strict=True):
        metrics = row.get("key_metrics", {})
        if row["lane"].startswith("W7-X nonlinear"):
            text = f"samples={metrics.get('time_samples')}, ky_phi={metrics.get('dominant_phi_ky'):.3g}"
        elif row["lane"].startswith("TEM"):
            gamma = _finite_float(metrics.get("max_abs_rel_gamma"))
            omega = _finite_float(metrics.get("max_abs_rel_omega"))
            if gamma is None:
                text = "TEM audit missing"
            elif omega is None:
                text = f"TEM max |rel gamma|={gamma:.2g}"
            else:
                text = f"TEM |rel gamma|={gamma:.2g}, |rel omega|={omega:.2g}"
        elif row["lane"].startswith("W7-X multi"):
            text = "alpha/surface missing"
        else:
            text = "kinetic-e window missing"
        if str(row["status"]) == "open":
            xpos = 1.95
            ha = "right"
        else:
            xpos = min(float(value) + 0.05, 2.05)
            ha = "left"
        ax.text(xpos, yi, text, va="center", ha=ha, fontsize=8.1)
    fig.text(
        0.5,
        0.02,
        "Only the simulation-spectrum estimator is closed; TEM and multi-flux W7-X validation remain paper-level work.",
        ha="center",
        fontsize=8.2,
        color="#333333",
    )
    fig.subplots_adjust(left=0.36, right=0.98, top=0.88, bottom=0.16)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return {
        "png": str(out_png),
        "pdf": str(out_pdf),
        "json": str(out_json),
        "csv": str(out_csv),
    }




def build_branch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the axisymmetric TEM branch audit.")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_BRANCH_OUT)
    parser.add_argument("--json-only", action="store_true")
    return parser


def build_w7x_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build W7-X TEM extension status.")
    parser.add_argument("--w7x-spectrum", type=Path, default=DEFAULT_W7X_SPECTRUM)
    parser.add_argument("--tem-table", type=Path, default=DEFAULT_TEM_TABLE)
    parser.add_argument("--tem-audit", type=Path, default=DEFAULT_TEM_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_W7X_OUT)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in {"axisymmetric-branch", "w7x-extension"}:
        raise SystemExit("mode must be 'axisymmetric-branch' or 'w7x-extension'")
    mode, mode_args = args[0], args[1:]
    if mode == "axisymmetric-branch":
        parsed = build_branch_parser().parse_args(mode_args)
        payload = build_branch_audit_payload(table=parsed.table, reference=parsed.reference)
        if parsed.json_only:
            print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        else:
            print(json.dumps(write_branch_artifacts(payload, out_png=parsed.out), indent=2, sort_keys=True))
        return 0

    parsed = build_w7x_parser().parse_args(mode_args)
    payload = build_w7x_status_payload(
        w7x_spectrum=parsed.w7x_spectrum,
        tem_table=parsed.tem_table,
        tem_audit=parsed.tem_audit,
    )
    if parsed.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                write_w7x_status_artifacts(payload, out_png=parsed.out),
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
