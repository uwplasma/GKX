#!/usr/bin/env python3
"""Nonlinear optimization and turbulence-gradient release gates.

This grouped maintainer command owns the fail-closed checks for nonlinear
turbulence-gradient evidence, production nonlinear optimization guardrails,
and overdetermined nonlinear-gradient campaign readiness.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gkx.artifacts.plotting import set_plot_style  # noqa: E402
from gkx.diagnostics.nonlinear_gradient_evidence import (  # noqa: E402
    NonlinearTurbulenceGradientEvidenceConfig,
    NonlinearTurbulenceGradientGapConfig,
    load_json_artifact,
    nonlinear_turbulence_gradient_evidence_report,
)
from gkx.diagnostics.nonlinear_transport_optimization import (  # noqa: E402
    ProductionNonlinearOptimizationGuardConfig,
    production_nonlinear_optimization_guard_report,
)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def build_gradient_evidence_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient-artifact", type=Path, required=True)
    parser.add_argument(
        "--window-artifact",
        action="append",
        type=Path,
        default=[],
        help="Replicated ensemble JSON or individual nonlinear-window convergence summary.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--gap-json-out",
        type=Path,
        help="Optional output path for the extracted fail-closed missing-campaign report.",
    )
    parser.add_argument(
        "--gap-case-slug",
        default="optimized_equilibrium_turbulence_gradient",
        help="Slug used when naming the required baseline/plus/minus campaign.",
    )
    parser.add_argument(
        "--gradient-parameter-name",
        default="vmec_state_control_or_profile_gradient",
        help="Parameter to perturb in the required paired nonlinear campaign.",
    )
    parser.add_argument("--perturbation-fraction", type=float, default=0.05)
    parser.add_argument("--analysis-tmin", type=float, default=350.0)
    parser.add_argument("--analysis-tmax", type=float, default=700.0)
    parser.add_argument("--minimum-tmax", type=float, default=700.0)
    parser.add_argument("--minimum-grid", default="n64x64x64x40x40")
    parser.add_argument(
        "--replicate-label",
        action="append",
        default=[],
        help=(
            "Required replicate label. Defaults to seed31, seed32, and dt0p04 "
            "when omitted."
        ),
    )
    parser.add_argument("--min-window-reports", type=int, default=2)
    parser.add_argument("--max-window-mean-rel-spread", type=float, default=0.15)
    parser.add_argument("--max-window-combined-sem-rel", type=float, default=0.25)
    parser.add_argument("--max-gradient-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-asymmetry-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-condition-number", type=float, default=1.0e8)
    parser.add_argument("--min-fd-response-fraction", type=float, default=0.03)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser


def main_gradient_evidence(argv: list[str] | None = None) -> int:
    args = build_gradient_evidence_parser().parse_args(argv)
    gradient = load_json_artifact(args.gradient_artifact)
    windows = [load_json_artifact(path) for path in args.window_artifact]
    report = nonlinear_turbulence_gradient_evidence_report(
        gradient,
        window_artifacts=windows,
        gradient_path=_repo_relative(args.gradient_artifact),
        window_paths=[_repo_relative(path) for path in args.window_artifact],
        config=NonlinearTurbulenceGradientEvidenceConfig(
            min_window_reports=args.min_window_reports,
            max_window_mean_rel_spread=args.max_window_mean_rel_spread,
            max_window_combined_sem_rel=args.max_window_combined_sem_rel,
            max_gradient_uncertainty_rel=args.max_gradient_uncertainty_rel,
            max_fd_asymmetry_rel=args.max_fd_asymmetry_rel,
            max_fd_condition_number=args.max_fd_condition_number,
            min_fd_response_fraction=args.min_fd_response_fraction,
        ),
        gap_config=NonlinearTurbulenceGradientGapConfig(
            case_slug=args.gap_case_slug,
            parameter_name=args.gradient_parameter_name,
            perturbation_fraction=args.perturbation_fraction,
            analysis_tmin=args.analysis_tmin,
            analysis_tmax=args.analysis_tmax,
            minimum_tmax=args.minimum_tmax,
            minimum_grid=args.minimum_grid,
            replicate_labels=tuple(
                args.replicate_label or ["seed31", "seed32", "dt0p04"]
            ),
        ),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"saved {args.json_out}")
    else:
        print(text)
    if args.gap_json_out is not None:
        args.gap_json_out.parent.mkdir(parents=True, exist_ok=True)
        args.gap_json_out.write_text(
            json.dumps(report["evidence_gap"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"saved {args.gap_json_out}")
    if args.fail_on_blocked and not bool(report["passed"]):
        print(
            "nonlinear turbulence-gradient evidence blocked: "
            + ", ".join(report["blockers"]),
            file=sys.stderr,
        )
        return 1
    return 0


# Production nonlinear optimization guard.
DEFAULT_OPTIMIZATION_ARTIFACT = (
    ROOT / "docs/_static/stellarator_itg_optimization_comparison.json"
)
DEFAULT_REDUCED_ARTIFACTS = (
    ROOT / "docs/_static/nonlinear_window_fd_audit.json",
    ROOT / "docs/_static/vmec_boozer_nonlinear_window_fd_audit.json",
)
DEFAULT_REPLICATED_ENSEMBLES = (
    ROOT
    / "docs/_static/external_vmec_dshape_replicates/dshape_replicate_t250_ensemble_gate.json",
    ROOT
    / "docs/_static/external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.json",
    ROOT
    / "docs/_static/vmec_boozer_holdout_transport/vmec_boozer_qh_torflux078_alpha120_holdout_ensemble_gate.json",
)
DEFAULT_OPTIMIZED_EQUILIBRIUM_ENSEMBLES = (
    ROOT
    / "docs/_static/optimized_equilibrium_replicates/optimized_equilibrium_replicate_t700_ensemble_gate.json",
    ROOT
    / "docs/_static/vmec_qa_t1500_replicates/growth_from_strict_baseline_t1500_ensemble_gate.json",
    ROOT
    / "docs/_static/vmec_qa_t1500_replicates/quasilinear_from_strict_baseline_t1500_ensemble_gate.json",
    ROOT
    / "docs/_static/vmec_qa_t1500_replicates/nonlinear_window_from_strict_baseline_t1500_ensemble_gate.json",
)
DEFAULT_MATCHED_OPTIMIZED_AUDITS = (
    ROOT / "docs/_static/qa_no_ess_to_optimized_nonlinear_audit.json",
    ROOT / "docs/_static/vmex_qa_projected_weight_0p0005_matched_comparison.json",
    ROOT / "docs/_static/vmex_qa_projected_weight_0p001_matched_comparison.json",
    ROOT / "docs/_static/vmec_qa_t1500_baseline_to_growth_comparison.json",
    ROOT / "docs/_static/vmec_qa_t1500_baseline_to_quasilinear_comparison.json",
    ROOT / "docs/_static/vmec_qa_t1500_baseline_to_nonlinear_window_comparison.json",
)
DEFAULT_OUT_JSON = ROOT / "docs/_static/production_nonlinear_optimization_guard.json"
DEFAULT_OUT_PNG = ROOT / "docs/_static/production_nonlinear_optimization_guard.png"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_mapping(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {_repo_relative(path): _load_json(path) for path in paths}


def _write_csv(report: dict[str, Any], path: Path) -> None:
    rows = []
    for gate in report["gates"]:
        rows.append(
            {
                "group": "safety"
                if gate["metric"] in report["safety_gate"]["requirements"]
                else "",
                "metric": gate["metric"],
                "passed": gate["passed"],
                "detail": gate["detail"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["metric", "passed", "detail"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in ("metric", "passed", "detail")})


def _write_plot(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    gates = list(report["gates"])
    labels = [str(gate["metric"]).replace("_", "\n") for gate in gates]
    values = [1 if bool(gate["passed"]) else 0 for gate in gates]
    colors = ["#0f766e" if value else "#b91c1c" for value in values]
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    x = range(len(gates))
    axs[0].bar(x, values, color=colors, alpha=0.9)
    axs[0].set_ylim(0.0, 1.15)
    axs[0].set_yticks([0, 1], ["blocked", "passed"])
    axs[0].set_xticks(list(x), labels, rotation=0, ha="center", fontsize=8)
    axs[0].set_title("Release safety and production-promotion gates")
    axs[0].grid(axis="y", alpha=0.25)

    summaries = report["summary"]
    counts = [
        summaries["qualifying_replicated_holdout_ensembles"],
        summaries["qualifying_optimized_equilibrium_ensembles"],
        summaries["qualifying_matched_optimized_transport_audits"],
    ]
    axs[1].bar(
        [0, 1, 2],
        counts,
        color=["#2563eb", "#f97316", "#7c3aed"],
        alpha=0.88,
    )
    axs[1].axhline(
        report["config"]["min_replicated_ensembles"], color="#2563eb", ls=":", lw=1.5
    )
    axs[1].axhline(
        report["config"]["min_optimized_equilibrium_ensembles"],
        color="#f97316",
        ls="--",
        lw=1.2,
    )
    axs[1].axhline(
        report["config"]["min_matched_optimized_audits"],
        color="#7c3aed",
        ls="-.",
        lw=1.2,
    )
    axs[1].set_xticks(
        [0, 1, 2],
        [
            "long-window\nholdouts",
            "optimized-equilibrium\ntransport",
            "matched\noptimization audits",
        ],
        fontsize=9,
    )
    axs[1].set_ylabel("qualifying artifacts")
    axs[1].set_title("Evidence available for nonlinear optimization")
    axs[1].grid(axis="y", alpha=0.25)
    axs[1].text(
        0.03,
        0.95,
        (
            f"matched audits total: {summaries.get('total_matched_optimized_transport_audits', 0)}\n"
            f"strict negatives: {summaries.get('failed_matched_optimized_transport_audits', 0)}"
        ),
        transform=axs[1].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.9},
    )

    status = (
        "production promoted"
        if report["production_nonlinear_optimization_promoted"]
        else "production blocked; release-safe scoped evidence"
    )
    fig.suptitle(
        f"Production nonlinear turbulent-flux optimization guard: {status}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_production_guard_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimization-artifact", type=Path, default=DEFAULT_OPTIMIZATION_ARTIFACT
    )
    parser.add_argument("--reduced-artifact", action="append", type=Path, default=[])
    parser.add_argument("--replicated-ensemble", action="append", type=Path, default=[])
    parser.add_argument(
        "--optimized-equilibrium-ensemble", action="append", type=Path, default=[]
    )
    parser.add_argument(
        "--matched-optimized-audit", action="append", type=Path, default=[]
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--out-pdf", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--min-replicated-ensembles", type=int, default=2)
    parser.add_argument("--min-reports-per-ensemble", type=int, default=2)
    parser.add_argument("--max-mean-rel-spread", type=float, default=0.15)
    parser.add_argument("--max-combined-sem-rel", type=float, default=0.25)
    parser.add_argument("--min-optimized-equilibrium-ensembles", type=int, default=3)
    parser.add_argument("--min-matched-optimized-audits", type=int, default=3)
    parser.add_argument(
        "--min-matched-optimized-relative-reduction", type=float, default=0.02
    )
    parser.add_argument(
        "--min-matched-optimized-uncertainty-sigma", type=float, default=1.0
    )
    parser.add_argument(
        "--allow-missing-optimized-equilibrium-transport",
        action="store_true",
        help="Do not require optimized-equilibrium replicated transport for production promotion.",
    )
    parser.add_argument(
        "--allow-missing-matched-optimized-audit",
        action="store_true",
        help="Do not require a matched baseline-to-optimized nonlinear transport audit for production promotion.",
    )
    parser.add_argument(
        "--fail-on-unsafe",
        action="store_true",
        help="Return non-zero if reduced/startup evidence is unsafe for release.",
    )
    parser.add_argument(
        "--fail-on-unpromoted",
        action="store_true",
        help="Return non-zero if production nonlinear optimization is not promoted.",
    )
    return parser


def main_production_guard(argv: list[str] | None = None) -> int:
    args = build_production_guard_parser().parse_args(argv)
    reduced_paths = list(args.reduced_artifact) or list(DEFAULT_REDUCED_ARTIFACTS)
    replicated_paths = list(args.replicated_ensemble) or list(
        DEFAULT_REPLICATED_ENSEMBLES
    )
    optimized_paths = list(args.optimized_equilibrium_ensemble) or list(
        DEFAULT_OPTIMIZED_EQUILIBRIUM_ENSEMBLES
    )
    matched_paths = list(args.matched_optimized_audit) or list(
        DEFAULT_MATCHED_OPTIMIZED_AUDITS
    )
    cfg = ProductionNonlinearOptimizationGuardConfig(
        min_replicated_ensembles=args.min_replicated_ensembles,
        min_reports_per_ensemble=args.min_reports_per_ensemble,
        max_mean_rel_spread=args.max_mean_rel_spread,
        max_combined_sem_rel=args.max_combined_sem_rel,
        require_optimized_equilibrium_transport=not args.allow_missing_optimized_equilibrium_transport,
        require_matched_optimized_transport_audit=not args.allow_missing_matched_optimized_audit,
        min_optimized_equilibrium_ensembles=args.min_optimized_equilibrium_ensembles,
        min_matched_optimized_audits=args.min_matched_optimized_audits,
        min_matched_optimized_relative_reduction=args.min_matched_optimized_relative_reduction,
        min_matched_optimized_uncertainty_sigma=args.min_matched_optimized_uncertainty_sigma,
    )
    report = production_nonlinear_optimization_guard_report(
        optimization_artifact=_load_json(args.optimization_artifact),
        optimization_artifact_path=_repo_relative(args.optimization_artifact),
        reduced_artifacts=_load_mapping(reduced_paths),
        replicated_ensemble_artifacts=_load_mapping(replicated_paths),
        optimized_equilibrium_artifacts=_load_mapping(optimized_paths),
        matched_optimized_transport_artifacts=_load_mapping(matched_paths),
        config=cfg,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.out_png is not None:
        _write_plot(report, args.out_png)
    if args.out_pdf is not None:
        _write_plot(report, args.out_pdf)
    _write_csv(report, args.out_csv or args.out_json.with_suffix(".csv"))
    print(
        json.dumps(
            {
                "safe_to_release": report["safe_to_release"],
                "promoted": report["production_nonlinear_optimization_promoted"],
                "summary": report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_unsafe and not bool(report["safe_to_release"]):
        return 1
    if args.fail_on_unpromoted and not bool(
        report["production_nonlinear_optimization_promoted"]
    ):
        return 1
    return 0


# Overdetermined nonlinear-gradient campaign gate.
def _repo_path(path: Path | str) -> str:
    raw = Path(path)
    try:
        return raw.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _extract_out_dir_from_command(command: str) -> Path | None:
    parts = shlex.split(command)
    try:
        raw = parts[parts.index("--out-dir") + 1]
    except (ValueError, IndexError):
        return None
    return _resolve_repo_path(raw)


def _expected_nested_manifest(control: dict[str, Any]) -> Path | None:
    explicit = control.get("expected_nonlinear_campaign_manifest")
    if explicit:
        return _resolve_repo_path(str(explicit))
    out_dir = _extract_out_dir_from_command(
        str(control.get("nonlinear_campaign_command_after_vmec_runs", ""))
    )
    return None if out_dir is None else out_dir / "gradient_campaign_manifest.json"


def _expected_runtime_outputs(nested_manifest: dict[str, Any] | None) -> list[Path]:
    if nested_manifest is None:
        return []
    state_commands = nested_manifest.get("state_ensemble_commands")
    if not isinstance(state_commands, dict):
        return []
    outputs: list[Path] = []
    for row in state_commands.values():
        if not isinstance(row, dict):
            continue
        for raw_path in row.get("expected_outputs", []):
            outputs.append(_resolve_repo_path(str(raw_path)))
    return outputs


def _required_runtime_tmax(manifest: dict[str, Any]) -> float | None:
    contract = manifest.get("run_contract")
    if not isinstance(contract, dict):
        return None
    window = contract.get("analysis_window")
    if isinstance(window, (list, tuple)) and len(window) >= 2:
        try:
            return float(window[1])
        except (TypeError, ValueError):
            return None
    horizons = contract.get("horizons")
    if isinstance(horizons, str):
        values: list[float] = []
        for raw in horizons.split(","):
            try:
                values.append(float(raw.strip()))
            except ValueError:
                continue
        return max(values) if values else None
    return None


def _read_runtime_time_max(path: Path) -> float | None:
    try:
        import netCDF4
        import numpy as np
    except ImportError:
        return None

    candidates: list[float] = []
    try:
        with netCDF4.Dataset(path) as root:
            arrays = []
            if "time" in root.variables:
                arrays.append(root.variables["time"][:])
            grids = root.groups.get("Grids")
            if grids is not None and "time" in grids.variables:
                arrays.append(grids.variables["time"][:])
            for array in arrays:
                values = np.asarray(array, dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size:
                    candidates.append(float(finite.max()))
    except Exception:
        return None
    return max(candidates) if candidates else None


def _runtime_completion_tolerance(required_tmax: float | None) -> float:
    if required_tmax is None:
        return 0.0
    return max(0.5, abs(float(required_tmax)) * 1.0e-4)


def _runtime_output_status(
    paths: list[Path], *, required_tmax: float | None
) -> dict[str, Any]:
    tolerance = _runtime_completion_tolerance(required_tmax)
    rows: list[dict[str, Any]] = []
    for path in paths:
        exists = path.exists()
        size_bytes = int(path.stat().st_size) if exists else 0
        time_max = (
            _read_runtime_time_max(path)
            if exists and size_bytes > 0 and required_tmax is not None
            else None
        )
        complete = bool(exists and size_bytes > 0)
        if complete and required_tmax is not None:
            complete = bool(
                time_max is not None and time_max >= required_tmax - tolerance
            )
        rows.append(
            {
                "path": _repo_path(path),
                "exists": exists,
                "size_bytes": size_bytes,
                "time_max": time_max,
                "complete": complete,
            }
        )
    missing = [row for row in rows if not row["exists"] or int(row["size_bytes"]) <= 0]
    incomplete = [
        row
        for row in rows
        if row["exists"] and int(row["size_bytes"]) > 0 and not row["complete"]
    ]
    return {
        "expected_count": len(rows),
        "complete_count": sum(1 for row in rows if row["complete"]),
        "missing_count": len(missing),
        "incomplete_count": len(incomplete),
        "required_tmax": required_tmax,
        "completion_tolerance": tolerance,
        "missing_outputs": [row["path"] for row in missing[:20]],
        "incomplete_outputs": [
            {
                "path": row["path"],
                "time_max": row["time_max"],
                "size_bytes": row["size_bytes"],
            }
            for row in incomplete[:20]
        ],
    }


def _state_file_status(paths: dict[str, Any] | None) -> dict[str, Any]:
    paths = paths or {}
    rows: dict[str, dict[str, Any]] = {}
    for state, raw in sorted(paths.items()):
        path = _resolve_repo_path(str(raw))
        rows[str(state)] = {
            "path": _repo_path(path),
            "exists": path.exists(),
            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        }
    return {
        "passed": bool(rows)
        and all(row["exists"] and row["size_bytes"] > 0 for row in rows.values()),
        "rows": rows,
    }


def _fd_status(path: Path) -> dict[str, Any]:
    payload = _load_optional_json(path)
    if payload is None:
        return {
            "path": _repo_path(path),
            "exists": False,
            "passed": False,
            "blockers": ["missing_fd_artifact"],
        }
    return {
        "path": _repo_path(path),
        "exists": True,
        "passed": bool(payload.get("passed", False)),
        "blockers": list(payload.get("blockers", []))
        if isinstance(payload.get("blockers", []), list)
        else [],
        "metrics": payload.get("metrics", {}),
    }


def _ranking_status(path: Path) -> dict[str, Any]:
    payload = _load_optional_json(path)
    if payload is None:
        return {
            "path": _repo_path(path),
            "exists": False,
            "passed": False,
            "recommendation": "ranking artifact is missing because not all central-FD artifacts exist yet",
        }
    return {
        "path": _repo_path(path),
        "exists": True,
        "passed": bool(payload.get("passed", False)),
        "recommendation": str(payload.get("recommendation", "")),
        "best_candidate": payload.get("best_candidate"),
    }


def _control_status(
    control: dict[str, Any], *, required_tmax: float | None
) -> dict[str, Any]:
    slug = str(control.get("coefficient_slug", "unknown"))
    vmec_inputs = _state_file_status(
        control.get("state_input_files")
        if isinstance(control.get("state_input_files"), dict)
        else {}
    )
    vmec_wouts = _state_file_status(
        control.get("expected_wout_files")
        if isinstance(control.get("expected_wout_files"), dict)
        else {}
    )
    nested_manifest_path = _expected_nested_manifest(control)
    nested_manifest = (
        _load_optional_json(nested_manifest_path) if nested_manifest_path is not None else None
    )
    runtime_outputs = _expected_runtime_outputs(nested_manifest)
    runtime_status = _runtime_output_status(
        runtime_outputs, required_tmax=required_tmax
    )
    fd_artifact = _resolve_repo_path(str(control.get("expected_fd_artifact", "")))
    fd = _fd_status(fd_artifact)
    nonlinear_manifest_exists = nested_manifest is not None
    ready_for_runtime = bool(vmec_wouts["passed"] and nonlinear_manifest_exists)
    runtime_outputs_complete = bool(runtime_outputs) and runtime_status[
        "complete_count"
    ] == len(runtime_outputs)
    passed = bool(ready_for_runtime and runtime_outputs_complete and fd["passed"])
    blockers: list[str] = []
    if not vmec_inputs["passed"]:
        blockers.append("missing_vmec_inputs")
    if not vmec_wouts["passed"]:
        blockers.append("missing_vmec_wouts")
    if not nonlinear_manifest_exists:
        blockers.append("missing_nested_nonlinear_campaign_manifest")
    if nonlinear_manifest_exists and runtime_status["missing_count"]:
        blockers.append("missing_runtime_outputs")
    if nonlinear_manifest_exists and runtime_status["incomplete_count"]:
        blockers.append("incomplete_runtime_outputs")
    if not fd["passed"]:
        blockers.append("central_fd_not_promoted")
    return {
        "coefficient": str(control.get("coefficient", slug)),
        "coefficient_slug": slug,
        "case": str(control.get("case", "")),
        "passed": passed,
        "ready_for_runtime": ready_for_runtime,
        "runtime_outputs_complete": runtime_outputs_complete,
        "blockers": blockers,
        "vmec_input_status": vmec_inputs,
        "vmec_wout_status": vmec_wouts,
        "nested_nonlinear_campaign_manifest": None
        if nested_manifest_path is None
        else {
            "path": _repo_path(nested_manifest_path),
            "exists": nonlinear_manifest_exists,
        },
        "runtime_output_status": runtime_status,
        "central_fd_status": fd,
        "vmec_run_commands": control.get("vmec_run_commands", {}),
        "write_nonlinear_campaign_command": control.get(
            "nonlinear_campaign_command_after_vmec_runs", ""
        ),
    }


def overdetermined_campaign_status_report(
    manifest: dict[str, Any], *, manifest_path: Path | None = None
) -> dict[str, Any]:
    """Return a fail-closed status report for an overdetermined campaign."""

    if (
        manifest.get("kind")
        != "overdetermined_nonlinear_turbulence_gradient_campaign_manifest"
    ):
        raise ValueError(
            "expected kind='overdetermined_nonlinear_turbulence_gradient_campaign_manifest', "
            f"got {manifest.get('kind')!r}"
        )
    controls_raw = manifest.get("controls")
    if not isinstance(controls_raw, list) or not controls_raw:
        raise ValueError("manifest must contain a non-empty controls list")
    controls = [row for row in controls_raw if isinstance(row, dict)]
    if len(controls) != len(controls_raw):
        raise ValueError("all controls must be JSON objects")

    required_tmax = _required_runtime_tmax(manifest)
    control_rows = [
        _control_status(control, required_tmax=required_tmax) for control in controls
    ]
    contract = manifest.get("promotion_contract")
    contract_map = contract if isinstance(contract, dict) else {}
    ranking_json = contract_map.get("candidate_ranking_json")
    ranking = (
        _ranking_status(_resolve_repo_path(str(ranking_json)))
        if ranking_json
        else {
            "path": "",
            "exists": False,
            "passed": False,
            "recommendation": "manifest is missing candidate_ranking_json",
        }
    )
    ready_controls = [row for row in control_rows if bool(row["ready_for_runtime"])]
    completed_controls = [
        row for row in control_rows if bool(row["runtime_outputs_complete"])
    ]
    promoted_controls = [
        row for row in control_rows if bool(row["central_fd_status"]["passed"])
    ]
    passed = bool(
        len(promoted_controls) >= 1
        and all(bool(row["passed"]) for row in control_rows)
        and bool(ranking["passed"])
    )
    next_actions: list[str] = []
    if not all(row["vmec_wout_status"]["passed"] for row in control_rows):
        next_actions.append("run the per-control VMEC re-equilibration commands")
    if any(
        row["vmec_wout_status"]["passed"]
        and not row["nested_nonlinear_campaign_manifest"]["exists"]
        for row in control_rows
    ):
        next_actions.append(
            "run nonlinear_campaign_command_after_vmec_runs for each VMEC-complete control"
        )
    if any(
        row["ready_for_runtime"] and not row["runtime_outputs_complete"]
        for row in control_rows
    ):
        next_actions.append(
            "run direct full-horizon nonlinear tasks for each nested campaign manifest"
        )
    if (
        all(row["runtime_outputs_complete"] for row in control_rows)
        and not promoted_controls
    ):
        all_fd_artifacts_exist = all(
            bool(row["central_fd_status"]["exists"]) for row in control_rows
        )
        if all_fd_artifacts_exist and bool(ranking.get("exists", False)):
            recommendation = str(ranking.get("recommendation", "")).strip()
            next_actions.append(
                recommendation
                or "keep the nonlinear-gradient claim fail-closed; no candidate passes production gates"
            )
        else:
            next_actions.append(
                "run output gates, ensemble gates, central-FD gates, then candidate ranking"
            )
    if not next_actions and not passed:
        next_actions.append(
            "inspect failed central-FD/ranking blockers before any release promotion"
        )
    return {
        "kind": "overdetermined_nonlinear_gradient_campaign_status",
        "claim_level": "multi_control_profile_gradient_status_not_simulation_claim",
        "manifest": "" if manifest_path is None else _repo_path(manifest_path),
        "case": str(manifest.get("case", "")),
        "passed": passed,
        "summary": {
            "control_count": len(control_rows),
            "ready_for_runtime_count": len(ready_controls),
            "runtime_complete_count": len(completed_controls),
            "central_fd_promoted_count": len(promoted_controls),
            "ranking_passed": bool(ranking["passed"]),
        },
        "controls": control_rows,
        "ranking_status": ranking,
        "next_actions": next_actions,
        "claim_boundary": (
            "This status can only pass after at least one real control has a passing "
            "long-window central-FD nonlinear turbulence-gradient artifact and the "
            "candidate ranking promotes it. Missing VMEC, runtime, or FD artifacts "
            "remain release blockers for the broader gradient claim."
        ),
    }


def build_overdetermined_gradient_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser


def main_overdetermined_gradient(argv: list[str] | None = None) -> int:
    args = build_overdetermined_gradient_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} does not contain a JSON object")
    report = overdetermined_campaign_status_report(
        manifest, manifest_path=manifest_path
    )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.fail_on_blocked and not bool(report["passed"]) else 0


COMMANDS = {
    "gradient-evidence": main_gradient_evidence,
    "overdetermined-gradient": main_overdetermined_gradient,
    "production-guard": main_production_guard,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=sorted(COMMANDS),
        help="Nonlinear optimization release gate to run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] in {"-h", "--help"}:
        return build_parser().parse_args(tokens) and 0
    command = tokens[0]
    if command not in COMMANDS:
        choices = ", ".join(sorted(COMMANDS))
        raise SystemExit(f"unknown nonlinear optimization gate {command!r}; choose one of: {choices}")
    return COMMANDS[command](tokens[1:])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
