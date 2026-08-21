#!/usr/bin/env python3
"""Nonlinear transport completion, convergence, and ensemble release gates.

This grouped maintainer command owns the fail-closed nonlinear transport
checks used by long-window turbulent-flux campaigns: runtime output
integrity, target-time completion, matched-matrix progress, and broad
matrix portfolio selection. Grouping these checks keeps the release
surface coherent without changing the underlying physics gates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gkx.artifacts.plotting import set_plot_style  # noqa: E402
from gkx.diagnostics.nonlinear_replicates import (  # noqa: E402
    NonlinearWindowEnsembleManifestConfig,
    nonlinear_window_ensemble_artifact_manifest,
)
from gkx.diagnostics.validation_gates import (  # noqa: E402
    matched_nonlinear_transport_report,
)
from gkx.diagnostics.transport_windows import (  # noqa: E402
    NonlinearWindowConvergenceConfig,
    NonlinearWindowEnsembleConfig,
    nonlinear_window_convergence_from_csv,
    nonlinear_window_convergence_from_summary,
    nonlinear_window_ensemble_report,
)


def _repo_relative(path: Path | str) -> str:
    raw = Path(path)
    try:
        return raw.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return raw.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _resolve_path(raw: str, *, manifest_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    repo_candidate = ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    manifest_candidate = manifest_path.parent / path
    if manifest_candidate.exists():
        return manifest_candidate
    return repo_candidate


def _bundle_base(path: Path) -> Path:
    name = path.name
    for suffix in (".out.nc", ".big.nc", ".restart.nc"):
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)])
    return path.with_suffix("") if path.suffix == ".nc" else path


def _bundle_paths(output_path: Path) -> dict[str, Path]:
    base = _bundle_base(output_path)
    return {
        "out": Path(f"{base}.out.nc"),
        "restart": Path(f"{base}.restart.nc"),
        "big": Path(f"{base}.big.nc"),
    }


def _last_finite_time(values: Any) -> float | None:
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    value = float(arr.reshape(-1)[-1])
    return value if math.isfinite(value) else None


def _read_output_tmax(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        import netCDF4  # type: ignore[import-not-found]  # noqa: PLC0415

        with netCDF4.Dataset(str(path)) as dataset:
            for group_name, variable_name in (
                ("Grids", "time"),
                ("Diagnostics", "time"),
                ("", "time"),
                ("", "t"),
            ):
                group = dataset if not group_name else dataset.groups.get(group_name)
                if group is None or variable_name not in group.variables:
                    continue
                value = _last_finite_time(group.variables[variable_name][:])
                if value is not None:
                    return value
    except Exception:
        pass
    try:
        from gkx.artifacts.io import (  # noqa: PLC0415
            load_nonlinear_netcdf_diagnostics,
        )

        diagnostics = load_nonlinear_netcdf_diagnostics(path)
        return _last_finite_time(getattr(diagnostics, "t", []))
    except Exception:
        return None


def _iter_expected_outputs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("matrix manifest is missing a samples list")
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        states = sample.get("states")
        if not isinstance(states, Mapping):
            continue
        for role, state in states.items():
            if not isinstance(state, Mapping):
                continue
            for output in state.get("final_outputs", []):
                rows.append(
                    {
                        "sample_id": sample.get("sample_id"),
                        "surface_torflux": sample.get("surface_torflux"),
                        "alpha": sample.get("alpha"),
                        "ky": sample.get("ky"),
                        "role": str(role),
                        "state_label": state.get("label"),
                        "output": str(output),
                    }
                )
    return rows


def _default_time_tolerance(cfg: Mapping[str, Any]) -> float:
    """Allow fixed-step output grids to stop just shy of the nominal horizon."""

    dt_values: list[float] = []
    for key in ("dt",):
        try:
            value = float(cfg.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            dt_values.append(value)
    raw_variants = cfg.get("dt_variants", ())
    if isinstance(raw_variants, (list, tuple)):
        for raw in raw_variants:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0.0:
                dt_values.append(value)
    if not dt_values:
        return 1.0e-9
    return max(1.0e-9, 2.0 * max(dt_values))


def build_matrix_progress_report(
    *,
    matrix_manifest: Path,
    target_time: float | None = None,
    time_tolerance: float | None = None,
    skip_time_check: bool = False,
) -> dict[str, Any]:
    manifest = _load_json(matrix_manifest)
    cfg = manifest.get("config") if isinstance(manifest.get("config"), Mapping) else {}
    effective_time_tolerance = (
        _default_time_tolerance(cfg)
        if time_tolerance is None
        else float(time_tolerance)
    )
    if target_time is None:
        window = cfg.get("window") if isinstance(cfg.get("window"), Mapping) else {}
        target_time = float(window.get("tmax", 0.0) or 0.0)
    if target_time <= 0.0:
        raise ValueError(
            "target time must be positive; pass --target-time if the manifest has no window.tmax"
        )
    expected = _iter_expected_outputs(manifest)
    rows: list[dict[str, Any]] = []
    complete_bundles = 0
    confirmed_targets = 0
    for item in expected:
        output_path = _resolve_path(str(item["output"]), manifest_path=matrix_manifest)
        bundle = _bundle_paths(output_path)
        present = {key: path.exists() for key, path in bundle.items()}
        bundle_complete = all(present.values())
        complete_bundles += int(bundle_complete)
        tmax = None if skip_time_check else _read_output_tmax(bundle["out"])
        target_confirmed = bool(
            bundle_complete
            and not skip_time_check
            and tmax is not None
            and tmax >= float(target_time) - float(effective_time_tolerance)
        )
        confirmed_targets += int(target_confirmed)
        rows.append(
            {
                **item,
                "output_path": _repo_relative(output_path),
                "bundle": {key: _repo_relative(path) for key, path in bundle.items()},
                "present": present,
                "bundle_complete": bundle_complete,
                "output_tmax": tmax,
                "target_time_confirmed": target_confirmed,
            }
        )
    expected_count = len(expected)
    return {
        "kind": "matched_nonlinear_transport_matrix_progress_report",
        "matrix_manifest": _repo_relative(matrix_manifest),
        "target_time": float(target_time),
        "time_tolerance": float(effective_time_tolerance),
        "skip_time_check": bool(skip_time_check),
        "summary": {
            "expected_outputs": expected_count,
            "complete_bundles": complete_bundles,
            "target_time_confirmed": confirmed_targets,
            "missing_or_incomplete_bundles": expected_count - complete_bundles,
            "not_confirmed_at_target_time": expected_count - confirmed_targets,
            "ready_for_postprocess": bool(
                expected_count and confirmed_targets == expected_count
            ),
            "time_check_skipped": bool(skip_time_check),
        },
        "rows": rows,
    }


def build_matrix_progress_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", required=True, type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--target-time", type=float)
    parser.add_argument(
        "--time-tolerance",
        type=float,
        help=(
            "Absolute tolerance for confirming the final time. Defaults to two "
            "matrix time steps so fixed-step output grids that stop just shy of "
            "the nominal horizon are not misclassified as incomplete."
        ),
    )
    parser.add_argument(
        "--skip-time-check",
        action="store_true",
        help=(
            "Count bundle presence without reading NetCDF time. Target-time "
            "confirmation and ready_for_postprocess remain false in this mode; "
            "run without this flag before postprocessing."
        ),
    )
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main_matrix_progress(argv: list[str] | None = None) -> int:
    args = build_matrix_progress_parser().parse_args(argv)
    report = build_matrix_progress_report(
        matrix_manifest=args.matrix_manifest,
        target_time=args.target_time,
        time_tolerance=(
            None if args.time_tolerance is None else float(args.time_tolerance)
        ),
        skip_time_check=bool(args.skip_time_check),
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if args.fail_on_incomplete and not bool(report["summary"]["ready_for_postprocess"]):
        return 1
    return 0


# Runtime output and target-time gates.
DEFAULT_HEAT_FLUX_VARIABLE = "Diagnostics/HeatFlux_st"


def build_runtime_outputs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "outputs", nargs="+", type=Path, help="Nonlinear runtime .out.nc files"
    )
    parser.add_argument("--json-out", type=Path, help="Optional JSON report path")
    parser.add_argument("--heat-flux-variable", default=DEFAULT_HEAT_FLUX_VARIABLE)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Optional required analysis-window start",
    )
    parser.add_argument(
        "--tmax", type=float, default=None, help="Optional required analysis-window end"
    )
    parser.add_argument(
        "--tmax-atol",
        type=float,
        default=None,
        help=(
            "Absolute tolerance for the required tmax coverage check. The default "
            "uses a fraction of the saved diagnostic cadence to tolerate fixed-step "
            "float32 accumulation roundoff."
        ),
    )
    parser.add_argument(
        "--min-window-samples",
        type=int,
        default=None,
        help="Minimum samples inside [tmin,tmax]; defaults to --min-samples when a window is requested.",
    )
    parser.add_argument(
        "--min-abs-window-mean",
        type=float,
        default=None,
        help="Optional fail-closed lower bound on |mean heat flux| in the requested window.",
    )
    return parser


def build_target_time_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether one nonlinear NetCDF output bundle reached a target time."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-time", required=True, type=float)
    parser.add_argument("--time-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_target_time_report(
    *,
    output: Path,
    target_time: float,
    time_tolerance: float,
) -> dict[str, object]:
    bundle = _bundle_paths(output)
    present = {key: path.exists() for key, path in bundle.items()}
    bundle_complete = all(present.values())
    output_tmax = _read_output_tmax(bundle["out"])
    target_confirmed = bool(
        bundle_complete
        and output_tmax is not None
        and float(output_tmax) >= float(target_time) - float(time_tolerance)
    )
    return {
        "kind": "nonlinear_output_target_time_check",
        "output": _repo_relative(output),
        "bundle": {key: _repo_relative(path) for key, path in bundle.items()},
        "present": present,
        "bundle_complete": bundle_complete,
        "output_tmax": output_tmax,
        "target_time": float(target_time),
        "time_tolerance": float(time_tolerance),
        "target_time_confirmed": target_confirmed,
    }


def _netcdf_variable(root: Any, variable_path: str) -> np.ndarray:
    group = root
    parts = variable_path.split("/")
    for part in parts[:-1]:
        if part not in group.groups:
            raise KeyError(f"missing NetCDF group {part!r} in {variable_path!r}")
        group = group.groups[part]
    name = parts[-1]
    if name not in group.variables:
        raise KeyError(f"missing NetCDF variable {name!r} in {variable_path!r}")
    return np.asarray(group.variables[name][:], dtype=float)


def _window_mask(
    time: np.ndarray, *, tmin: float | None, tmax: float | None
) -> np.ndarray:
    mask = np.isfinite(time)
    if tmin is not None:
        mask &= time >= float(tmin)
    if tmax is not None:
        mask &= time <= float(tmax)
    return mask


def _resolved_tmax_atol(time: np.ndarray, requested: float | None) -> float:
    if requested is not None:
        return max(float(requested), 0.0)
    finite = np.asarray(time[np.isfinite(time)], dtype=float)
    if finite.size < 2:
        return 1.0e-6
    positive_diffs = np.diff(finite)
    positive_diffs = positive_diffs[positive_diffs > 0.0]
    if positive_diffs.size == 0:
        return 1.0e-6
    return min(1.0, max(1.0e-6, 0.25 * float(np.median(positive_diffs))))


def validate_output(
    path: Path,
    *,
    heat_flux_variable: str = DEFAULT_HEAT_FLUX_VARIABLE,
    min_samples: int = 2,
    tmin: float | None = None,
    tmax: float | None = None,
    tmax_atol: float | None = None,
    min_window_samples: int | None = None,
    min_abs_window_mean: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": path.as_posix(),
        "passed": False,
        "failures": [],
        "warnings": [],
    }
    failures = row["failures"]
    warnings = row["warnings"]
    if not path.exists():
        failures.append("missing_output_file")
        return row

    try:
        import netCDF4
    except Exception as exc:  # pragma: no cover - dependency is in pyproject
        failures.append(f"netcdf4_unavailable:{type(exc).__name__}")
        return row

    try:
        with netCDF4.Dataset(path) as root:
            time = _netcdf_variable(root, "Grids/time").reshape(-1)
            heat = _netcdf_variable(root, heat_flux_variable)
    except Exception as exc:
        failures.append(f"read_error:{type(exc).__name__}:{exc}")
        return row

    row["samples"] = int(time.size)
    row["heat_flux_shape"] = list(heat.shape)
    row["time_min"] = float(np.nanmin(time)) if time.size else None
    row["time_max"] = float(np.nanmax(time)) if time.size else None
    row["tmax_atol"] = _resolved_tmax_atol(time, tmax_atol)

    if time.size < int(min_samples):
        failures.append("too_few_time_samples")
    if time.size and not bool(np.all(np.isfinite(time))):
        failures.append("nonfinite_time")
    if time.size > 1 and bool(np.any(np.diff(time) <= 0.0)):
        failures.append("nonmonotone_time")
    if heat.ndim == 0:
        failures.append("scalar_heat_flux_variable")
        heat_total = np.asarray([], dtype=float)
    elif heat.shape[0] != time.size:
        failures.append("heat_flux_time_dimension_mismatch")
        heat_total = np.asarray([], dtype=float)
    else:
        heat_total = heat.reshape((time.size, -1)).sum(axis=1)
        if not bool(np.all(np.isfinite(heat_total))):
            failures.append("nonfinite_heat_flux")
        row["heat_flux_min"] = float(np.nanmin(heat_total)) if heat_total.size else None
        row["heat_flux_max"] = float(np.nanmax(heat_total)) if heat_total.size else None
        row["heat_flux_last"] = float(heat_total[-1]) if heat_total.size else None

    if tmax is not None and (
        not time.size or float(np.nanmax(time)) + float(row["tmax_atol"]) < float(tmax)
    ):
        failures.append("does_not_reach_required_tmax")

    if tmin is not None or tmax is not None:
        required_window_samples = (
            int(min_window_samples)
            if min_window_samples is not None
            else int(min_samples)
        )
        mask = _window_mask(time, tmin=tmin, tmax=tmax)
        window_heat = (
            heat_total[mask] if heat_total.size else np.asarray([], dtype=float)
        )
        row["window"] = {
            "tmin": None if tmin is None else float(tmin),
            "tmax": None if tmax is None else float(tmax),
            "samples": int(window_heat.size),
            "mean_heat_flux": float(np.mean(window_heat)) if window_heat.size else None,
            "sem_heat_flux": (
                float(np.std(window_heat, ddof=1) / np.sqrt(window_heat.size))
                if window_heat.size > 1
                else None
            ),
        }
        if window_heat.size < required_window_samples:
            failures.append("too_few_window_samples")
        if window_heat.size and min_abs_window_mean is not None:
            mean_abs = abs(float(np.mean(window_heat)))
            if mean_abs < float(min_abs_window_mean):
                failures.append("window_mean_heat_flux_below_threshold")
        if window_heat.size and abs(float(np.mean(window_heat))) < 1.0e-12:
            warnings.append("near_zero_window_mean_heat_flux")

    row["passed"] = len(failures) == 0
    return row


def check_outputs(
    outputs: list[Path],
    *,
    heat_flux_variable: str,
    min_samples: int,
    tmin: float | None,
    tmax: float | None,
    tmax_atol: float | None,
    min_window_samples: int | None,
    min_abs_window_mean: float | None,
) -> dict[str, Any]:
    rows = [
        validate_output(
            output,
            heat_flux_variable=heat_flux_variable,
            min_samples=min_samples,
            tmin=tmin,
            tmax=tmax,
            tmax_atol=tmax_atol,
            min_window_samples=min_window_samples,
            min_abs_window_mean=min_abs_window_mean,
        )
        for output in outputs
    ]
    passed = all(bool(row["passed"]) for row in rows)
    return {
        "kind": "nonlinear_runtime_output_diagnostics_gate",
        "passed": passed,
        "heat_flux_variable": heat_flux_variable,
        "config": {
            "min_samples": int(min_samples),
            "tmin": None if tmin is None else float(tmin),
            "tmax": None if tmax is None else float(tmax),
            "tmax_atol": tmax_atol,
            "min_window_samples": min_window_samples,
            "min_abs_window_mean": min_abs_window_mean,
        },
        "summary": {
            "outputs": len(rows),
            "passed": sum(1 for row in rows if bool(row["passed"])),
            "failed": sum(1 for row in rows if not bool(row["passed"])),
        },
        "rows": rows,
    }


def main_target_time(argv: list[str] | None = None) -> int:
    args = build_target_time_parser().parse_args(argv)
    report = build_target_time_report(
        output=args.output,
        target_time=float(args.target_time),
        time_tolerance=float(args.time_tolerance),
    )
    if args.out_json is not None:
        _write_json(args.out_json, report)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["target_time_confirmed"]) else 1


def main_runtime_outputs(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "target-time":
        return main_target_time(tokens[1:])

    args = build_runtime_outputs_parser().parse_args(tokens)
    payload = check_outputs(
        args.outputs,
        heat_flux_variable=str(args.heat_flux_variable),
        min_samples=int(args.min_samples),
        tmin=args.tmin,
        tmax=args.tmax,
        tmax_atol=args.tmax_atol,
        min_window_samples=args.min_window_samples,
        min_abs_window_mean=args.min_abs_window_mean,
    )
    if args.json_out is not None:
        _write_json(args.json_out, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if bool(payload["passed"]) else 1


# Matrix portfolio gate.
def _parse_labeled_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"expected LABEL=PATH, got {raw!r}")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty label in {raw!r}")
    return label, Path(path)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _matrix_row(
    *,
    label: str,
    path: Path,
    min_total_samples: int,
    min_surfaces: int,
    min_alphas: int,
    min_ky_values: int,
    min_pass_fraction: float,
    min_mean_relative_reduction: float,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "label": label,
            "path": _repo_relative(path),
            "exists": False,
            "passed": False,
            "qualifies_for_broad_promotion": False,
            "blockers": ["missing matrix report"],
        }
    payload = _load_json(path)
    summary = (
        payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    )
    total_samples = int(_finite_float(summary.get("total_samples")) or 0)
    completed_samples = int(_finite_float(summary.get("completed_samples")) or 0)
    passed_samples = int(_finite_float(summary.get("passed_samples")) or 0)
    pass_fraction = _finite_float(summary.get("pass_fraction"))
    mean_reduction = _finite_float(summary.get("mean_relative_reduction"))
    surfaces = (
        summary.get("surfaces") if isinstance(summary.get("surfaces"), list) else []
    )
    alphas = summary.get("alphas") if isinstance(summary.get("alphas"), list) else []
    ky_values = (
        summary.get("ky_values") if isinstance(summary.get("ky_values"), list) else []
    )
    blockers: list[str] = []
    if payload.get("kind") != "matched_nonlinear_transport_matrix_report":
        blockers.append("not a matched nonlinear transport matrix report")
    if not bool(payload.get("passed", False)):
        blockers.append("matrix report failed its internal gate")
    if total_samples < int(min_total_samples):
        blockers.append(f"total_samples {total_samples} < {int(min_total_samples)}")
    if completed_samples < total_samples:
        blockers.append(
            f"completed_samples {completed_samples} < total_samples {total_samples}"
        )
    if len(set(surfaces)) < int(min_surfaces):
        blockers.append(f"surfaces {len(set(surfaces))} < {int(min_surfaces)}")
    if len(set(alphas)) < int(min_alphas):
        blockers.append(f"field_line_labels {len(set(alphas))} < {int(min_alphas)}")
    if len(set(ky_values)) < int(min_ky_values):
        blockers.append(f"ky_values {len(set(ky_values))} < {int(min_ky_values)}")
    if pass_fraction is None or pass_fraction < float(min_pass_fraction):
        blockers.append(
            "pass_fraction "
            f"{'n/a' if pass_fraction is None else f'{pass_fraction:.6g}'} "
            f"< {float(min_pass_fraction):.6g}"
        )
    if mean_reduction is None or mean_reduction < float(min_mean_relative_reduction):
        blockers.append(
            "mean_relative_reduction "
            f"{'n/a' if mean_reduction is None else f'{mean_reduction:.6g}'} "
            f"< {float(min_mean_relative_reduction):.6g}"
        )
    return {
        "label": label,
        "path": _repo_relative(path),
        "exists": True,
        "passed": bool(payload.get("passed", False)),
        "qualifies_for_broad_promotion": not blockers,
        "blockers": blockers,
        "summary": {
            "total_samples": total_samples,
            "completed_samples": completed_samples,
            "passed_samples": passed_samples,
            "pass_fraction": pass_fraction,
            "mean_relative_reduction": mean_reduction,
            "surfaces": surfaces,
            "alphas": alphas,
            "ky_values": ky_values,
        },
    }


def _excluded_row(label: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"label": label, "path": _repo_relative(path), "exists": False}
    payload = _load_json(path)
    stats = (
        payload.get("statistics")
        if isinstance(payload.get("statistics"), Mapping)
        else {}
    )
    comparison = (
        payload.get("comparison")
        if isinstance(payload.get("comparison"), Mapping)
        else {}
    )
    return {
        "label": label,
        "path": _repo_relative(path),
        "exists": True,
        "passed": bool(payload.get("passed", False)),
        "relative_reduction": _finite_float(
            comparison.get("relative_reduction", stats.get("relative_reduction"))
        ),
        "uncertainty_z_score": _finite_float(
            comparison.get("uncertainty_z_score", stats.get("uncertainty_z_score"))
        ),
        "note": "recorded as negative/diagnostic evidence; excluded from broad matrix promotion",
    }


def build_transport_matrix_portfolio_report(
    *,
    matrix_reports: Mapping[str, Path],
    excluded_comparisons: Mapping[str, Path],
    min_total_samples: int = 18,
    min_surfaces: int = 3,
    min_alphas: int = 2,
    min_ky_values: int = 3,
    min_pass_fraction: float = 1.0,
    min_mean_relative_reduction: float = 0.02,
) -> dict[str, Any]:
    rows = [
        _matrix_row(
            label=label,
            path=path,
            min_total_samples=min_total_samples,
            min_surfaces=min_surfaces,
            min_alphas=min_alphas,
            min_ky_values=min_ky_values,
            min_pass_fraction=min_pass_fraction,
            min_mean_relative_reduction=min_mean_relative_reduction,
        )
        for label, path in sorted(matrix_reports.items())
    ]
    qualified = [row for row in rows if bool(row["qualifies_for_broad_promotion"])]
    selected = max(
        qualified,
        key=lambda row: float(row["summary"].get("mean_relative_reduction") or 0.0),
        default=None,
    )
    blockers: list[str] = []
    if not rows:
        blockers.append("no matrix reports supplied")
    if selected is None:
        blockers.append("no candidate family passed the broad matrix gate")
    return {
        "kind": "nonlinear_transport_matrix_portfolio_gate",
        "claim_level": "broad_nonlinear_turbulent_flux_optimization_family_selection",
        "passed": selected is not None,
        "selected_family": None if selected is None else selected["label"],
        "selected_report": selected,
        "config": {
            "min_total_samples": int(min_total_samples),
            "min_surfaces": int(min_surfaces),
            "min_field_line_labels": int(min_alphas),
            "min_ky_values": int(min_ky_values),
            "min_pass_fraction": float(min_pass_fraction),
            "min_mean_relative_reduction": float(min_mean_relative_reduction),
        },
        "matrix_reports": rows,
        "excluded_comparisons": [
            _excluded_row(label, path)
            for label, path in sorted(excluded_comparisons.items())
        ],
        "blockers": blockers,
        "notes": [
            "A single-point matched audit is not enough for this broad promotion gate.",
            "Strict growth/QL/nonlinear-window transfer rows are listed only as excluded evidence.",
        ],
    }


def _write_figure(report: Mapping[str, Any], path: Path) -> None:
    rows = [row for row in report.get("matrix_reports", []) if isinstance(row, Mapping)]
    path.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    labels = [str(row.get("label")) for row in rows]
    reductions = [
        100.0 * float((row.get("summary") or {}).get("mean_relative_reduction") or 0.0)
        for row in rows
    ]
    colors = [
        "#0f766e" if bool(row.get("qualifies_for_broad_promotion")) else "#b45309"
        for row in rows
    ]
    if rows:
        ax.bar(
            range(len(rows)), reductions, color=colors, edgecolor="0.2", linewidth=0.6
        )
        ax.set_xticks(range(len(rows)), labels, rotation=25, ha="right")
        ax.set_ylabel("mean heat-flux reduction across matrix (%)")
        ax.axhline(0.0, color="0.2", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No matrix reports supplied", ha="center", va="center")
        ax.set_xticks([])
    status = "passes" if bool(report.get("passed", False)) else "blocked"
    selected = report.get("selected_family") or "none"
    ax.set_title(f"Nonlinear transport matrix portfolio: {status}; selected={selected}")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_transport_matrix_portfolio_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-report",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Candidate-family matrix report JSON. May be supplied multiple times.",
    )
    parser.add_argument(
        "--excluded-comparison",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Strict negative-transfer comparison JSON to record but not promote.",
    )
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-figure", type=Path)
    parser.add_argument("--min-total-samples", type=int, default=18)
    parser.add_argument("--min-surfaces", type=int, default=3)
    parser.add_argument("--min-field-line-labels", type=int, default=2)
    parser.add_argument("--min-ky-values", type=int, default=3)
    parser.add_argument("--min-pass-fraction", type=float, default=1.0)
    parser.add_argument("--min-mean-relative-reduction", type=float, default=0.02)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser


def main_matrix_portfolio(argv: list[str] | None = None) -> int:
    args = build_transport_matrix_portfolio_parser().parse_args(argv)
    matrix_reports = dict(_parse_labeled_path(raw) for raw in args.matrix_report)
    excluded = dict(_parse_labeled_path(raw) for raw in args.excluded_comparison)
    report = build_transport_matrix_portfolio_report(
        matrix_reports=matrix_reports,
        excluded_comparisons=excluded,
        min_total_samples=int(args.min_total_samples),
        min_surfaces=int(args.min_surfaces),
        min_alphas=int(args.min_field_line_labels),
        min_ky_values=int(args.min_ky_values),
        min_pass_fraction=float(args.min_pass_fraction),
        min_mean_relative_reduction=float(args.min_mean_relative_reduction),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.out_figure is not None:
        _write_figure(report, args.out_figure)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "selected_family": report["selected_family"],
                "blockers": report["blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_blocked and not bool(report["passed"]):
        return 1
    return 0


VARIANT_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "seed": ("seed", "random_seed", "rng_seed", "simulation_seed"),
    "timestep": ("timestep", "time_step", "dt"),
}


def build_window_ensemble_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check replicated nonlinear-window summaries for robustness."
    )
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="Nonlinear-window convergence JSON reports.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-png", type=Path)
    parser.add_argument("--case", default="nonlinear_window_ensemble")
    parser.add_argument("--comparison", default="replicate_uncertainty")
    parser.add_argument("--min-reports", type=int, default=2)
    parser.add_argument("--max-mean-rel-spread", type=float, default=0.15)
    parser.add_argument("--max-combined-sem-rel", type=float, default=0.25)
    parser.add_argument("--value-floor", type=float, default=1.0e-12)
    parser.add_argument(
        "--allow-failed-individual",
        action="store_true",
        help="Do not require each input window to pass the promotion-ready gate.",
    )
    return parser


def build_convergence_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check nonlinear late-window convergence metadata without rerunning a solve."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Diagnostics CSV with time trace.")
    source.add_argument(
        "--summary",
        type=Path,
        help="Window summary JSON containing a diagnostics source path.",
    )
    parser.add_argument("--diagnostics-source", default="gkx")
    parser.add_argument("--time-column", default="t")
    parser.add_argument("--value-column", default="heat_flux")
    parser.add_argument("--case", default=None)
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument("--transient-fraction", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=24)
    parser.add_argument("--min-blocks", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=256)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--max-running-mean-rel-drift", type=float, default=0.15)
    parser.add_argument("--terminal-fraction", type=float, default=0.25)
    parser.add_argument("--min-terminal-samples", type=int, default=8)
    parser.add_argument("--max-terminal-mean-rel-delta", type=float, default=0.10)
    parser.add_argument("--max-sem-rel", type=float, default=0.25)
    parser.add_argument("--value-floor", type=float, default=1.0e-12)
    parser.add_argument(
        "--allow-nonfinite",
        action="store_true",
        help="Ignore non-finite samples inside the late window instead of failing.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    return parser


def build_matched_windows_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two independently converged nonlinear transport windows."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--case", default="matched_nonlinear_transport")
    parser.add_argument("--treatment-name", default="treatment")
    parser.add_argument("--min-relative-reduction", type=float, default=0.0)
    parser.add_argument("--min-uncertainty-z-score", type=float, default=0.0)
    parser.add_argument("--value-floor", type=float, default=1.0e-12)
    return parser


def build_readiness_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert transport-window summaries into ensemble-readiness metadata."
    )
    parser.add_argument(
        "summaries",
        nargs="+",
        type=Path,
        help="Transport-window summary JSON files with trace CSV provenance.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="Optional directory for per-summary convergence report JSON files.",
    )
    parser.add_argument("--case", default="nonlinear_window_ensemble_readiness")
    parser.add_argument("--time-column", default="t")
    parser.add_argument("--value-column", default="heat_flux")
    parser.add_argument("--min-samples", type=int, default=24)
    parser.add_argument("--min-blocks", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=256)
    parser.add_argument("--max-running-mean-rel-drift", type=float, default=0.15)
    parser.add_argument("--max-terminal-mean-rel-delta", type=float, default=0.10)
    parser.add_argument("--max-sem-rel", type=float, default=0.25)
    parser.add_argument("--min-replicates-per-case", type=int, default=2)
    parser.add_argument(
        "--variant-axis",
        action="append",
        choices=tuple(VARIANT_KEY_ALIASES),
        help="Required replicated variant axis. Defaults to seed and timestep.",
    )
    parser.add_argument(
        "--allow-failed-observed-window",
        action="store_true",
        help="Record failed convergence reports but do not block on them.",
    )
    return parser


def _load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} did not contain a JSON object")
        reports.append(payload)
    return reports


def _window_load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _window_repo_relative(path: Path | str | None) -> str | None:
    if path is None:
        return None
    raw_path = Path(path)
    try:
        return raw_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _safe_report_name(summary_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", summary_path.stem)
    return f"{stem}.convergence.json"


def _nested_search(payload: dict[str, Any], aliases: tuple[str, ...]) -> Any | None:
    queue: list[dict[str, Any]] = [payload]
    seen = 0
    while queue and seen < 64:
        current = queue.pop(0)
        seen += 1
        for key in aliases:
            if key in current and current[key] not in (None, ""):
                return current[key]
        for key in (
            "variant",
            "metadata",
            "run",
            "simulation",
            "config",
            "nonlinear_config",
        ):
            nested = current.get(key)
            if isinstance(nested, dict):
                queue.append(nested)
    return None


def _variant_from_summary(
    summary: dict[str, Any], summary_path: Path
) -> dict[str, Any]:
    text = summary_path.stem.lower()
    variant: dict[str, Any] = {}
    for axis, aliases in VARIANT_KEY_ALIASES.items():
        value = _nested_search(summary, aliases)
        if value is None:
            regexes = (
                [r"(?:seed|rng)[_-]?([0-9]+)"]
                if axis == "seed"
                else [r"(?:dt|timestep)[_-]?([0-9]+(?:p[0-9]+)?)"]
            )
            for pattern in regexes:
                match = re.search(pattern, text)
                if match:
                    value = match.group(1).replace("p", ".")
                    break
        variant[axis] = value
    return variant


def _summary_case(summary: dict[str, Any], path: Path) -> str:
    return str(summary.get("case") or summary.get("case_name") or path.stem)


def _write_window_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_png(report: dict[str, Any], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style

    rows = list(report["rows"])
    labels = [str(row["case"]) for row in rows]
    means = np.asarray(
        [
            np.nan if row["late_mean"] is None else float(row["late_mean"])
            for row in rows
        ]
    )
    sem = np.asarray([0.0 if row["sem"] is None else float(row["sem"]) for row in rows])
    ensemble_mean = report["statistics"]["ensemble_mean"]
    passed = bool(report["passed"])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    x = np.arange(len(rows))
    colors = ["#0f766e" if bool(row["promotion_ready"]) else "#b91c1c" for row in rows]
    ax.bar(x, means, yerr=sem, capsize=4, color=colors, alpha=0.88)
    if ensemble_mean is not None:
        ax.axhline(
            float(ensemble_mean), color="0.2", lw=1.3, ls="--", label="ensemble mean"
        )
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("late-window mean")
    status = "passed" if passed else "failed closed"
    ax.set_title(f"{report['comparison']} ensemble gate {status}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _convergence_config(args: argparse.Namespace) -> NonlinearWindowConvergenceConfig:
    return NonlinearWindowConvergenceConfig(
        tmin=args.tmin,
        tmax=args.tmax,
        transient_fraction=args.transient_fraction,
        min_samples=args.min_samples,
        min_blocks=args.min_blocks,
        block_size=args.block_size,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        max_running_mean_rel_drift=args.max_running_mean_rel_drift,
        terminal_fraction=args.terminal_fraction,
        min_terminal_samples=args.min_terminal_samples,
        max_terminal_mean_rel_delta=args.max_terminal_mean_rel_delta,
        max_sem_rel=args.max_sem_rel,
        value_floor=args.value_floor,
        require_all_finite=not args.allow_nonfinite,
    )


def main_convergence(argv: list[str] | None = None) -> int:
    args = build_convergence_parser().parse_args(argv)
    cfg = _convergence_config(args)
    if args.csv is not None:
        report = nonlinear_window_convergence_from_csv(
            args.csv,
            time_column=args.time_column,
            value_column=args.value_column,
            case=args.case,
            config=cfg,
        )
    else:
        report = nonlinear_window_convergence_from_summary(
            args.summary,
            diagnostics_source=args.diagnostics_source,
            time_column=args.time_column,
            value_column=args.value_column,
            case=args.case,
            config=cfg,
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    stats = report["statistics"]
    print(f"wrote {args.out_json}")
    print(
        "nonlinear_window_converged={passed} late_mean={mean} sem={sem}".format(
            passed=report["passed"],
            mean=stats["late_mean"],
            sem=stats["sem"],
        )
    )
    return 0 if report["passed"] else 2


def main_readiness(argv: list[str] | None = None) -> int:
    args = build_readiness_parser().parse_args(argv)
    axes = tuple(args.variant_axis or ("seed", "timestep"))
    records: list[dict[str, Any]] = []
    for summary_path in args.summaries:
        summary = _window_load_json_object(summary_path)
        convergence_config = NonlinearWindowConvergenceConfig(
            tmin=summary.get("tmin"),
            tmax=summary.get("tmax"),
            min_samples=args.min_samples,
            min_blocks=args.min_blocks,
            bootstrap_samples=args.bootstrap_samples,
            max_running_mean_rel_drift=args.max_running_mean_rel_drift,
            max_terminal_mean_rel_delta=args.max_terminal_mean_rel_delta,
            max_sem_rel=args.max_sem_rel,
        )
        report = nonlinear_window_convergence_from_summary(
            summary_path,
            time_column=args.time_column,
            value_column=args.value_column,
            case=_summary_case(summary, summary_path),
            config=convergence_config,
        )
        report_path: Path | None = None
        if args.reports_dir is not None:
            report_path = args.reports_dir / _safe_report_name(summary_path)
            _write_window_json(report_path, report)
        records.append(
            {
                "case": _summary_case(summary, summary_path),
                "summary_artifact": _window_repo_relative(summary_path),
                "source_artifact": _window_repo_relative(
                    report["provenance"]["source_artifact"]
                ),
                "convergence_report_artifact": _window_repo_relative(report_path),
                "variant": _variant_from_summary(summary, summary_path),
                "report": report,
            }
        )

    manifest = nonlinear_window_ensemble_artifact_manifest(
        records,
        case=args.case,
        config=NonlinearWindowEnsembleManifestConfig(
            min_replicates_per_case=args.min_replicates_per_case,
            required_variant_axes=axes,
            require_observed_windows_ready=not args.allow_failed_observed_window,
        ),
    )
    _write_window_json(args.out_json, manifest)
    print(json.dumps(manifest["promotion_gate"], indent=2, sort_keys=True))
    return 0 if bool(manifest["passed"]) else 1


def main_ensemble(argv: list[str] | None = None) -> int:
    args = build_window_ensemble_parser().parse_args(argv)
    reports = _load_reports(list(args.reports))
    summary = nonlinear_window_ensemble_report(
        reports,
        case=args.case,
        comparison=args.comparison,
        config=NonlinearWindowEnsembleConfig(
            min_reports=args.min_reports,
            max_mean_rel_spread=args.max_mean_rel_spread,
            max_combined_sem_rel=args.max_combined_sem_rel,
            value_floor=args.value_floor,
            require_individual_passed=not args.allow_failed_individual,
        ),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.out_png is not None:
        _write_png(summary, args.out_png)
    print(json.dumps(summary["gate_report"], indent=2, sort_keys=True))
    return 0 if bool(summary["passed"]) else 1


def main_matched_windows(argv: list[str] | None = None) -> int:
    args = build_matched_windows_parser().parse_args(argv)
    report = matched_nonlinear_transport_report(
        _window_load_json_object(args.baseline),
        _window_load_json_object(args.treatment),
        case=args.case,
        treatment_name=args.treatment_name,
        min_relative_reduction=args.min_relative_reduction,
        min_uncertainty_z_score=args.min_uncertainty_z_score,
        value_floor=args.value_floor,
    )
    _write_window_json(args.out_json, report)
    print(json.dumps(report["gate_report"], indent=2, sort_keys=True))
    return 0 if bool(report["passed"]) else 1


COMMANDS = {
    "convergence": main_convergence,
    "ensemble": main_ensemble,
    "readiness": main_readiness,
    "runtime-outputs": main_runtime_outputs,
    "target-time": main_target_time,
    "matrix-progress": main_matrix_progress,
    "matrix-portfolio": main_matrix_portfolio,
    "matched-windows": main_matched_windows,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=sorted(COMMANDS),
        help="Nonlinear transport release gate to run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] in {"-h", "--help"}:
        return build_parser().parse_args(tokens) and 0
    command = tokens[0]
    if command not in COMMANDS:
        choices = ", ".join(sorted(COMMANDS))
        raise SystemExit(
            f"unknown nonlinear transport gate {command!r}; choose one of: {choices}"
        )
    return COMMANDS[command](tokens[1:])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
