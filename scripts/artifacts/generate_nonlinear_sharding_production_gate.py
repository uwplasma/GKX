#!/usr/bin/env python3
"""Gate whole-state nonlinear sharding before any production speedup claim."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = [
    REPO_ROOT / "docs" / "_static" / "nonlinear_sharding_strong_scaling_cpu_large.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "nonlinear_sharding_strong_scaling_gpu_xlarge.json",
]
DEFAULT_OUT_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "nonlinear_sharding_production_speedup_gate"
)
DEFAULT_MIN_SPEEDUP = 1.20
DEFAULT_MIN_EFFICIENCY = 0.50
DEFAULT_MIN_DEVICES = 2
DEFAULT_REQUIRED_BACKENDS = ("cpu", "gpu")
DEFAULT_IDENTITY_ATOL = 1.0e-5
DEFAULT_IDENTITY_RTOL = 1.0e-5
IDENTITY_BLOCKERS = {
    "identity_gate_failed",
    "identity_abs_error_missing",
    "identity_abs_error_above_tolerance",
    "identity_rel_error_missing",
    "identity_rel_error_above_tolerance",
}
REFERENCE_BLOCKERS = {
    "below_min_devices",
    "actual_devices_below_min_devices",
}


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


def _parse_backend_list(text: str) -> tuple[str, ...]:
    values = tuple(
        part.strip().lower() for part in str(text).split(",") if part.strip()
    )
    if not values:
        raise argparse.ArgumentTypeError("expected at least one backend")
    unknown = sorted(set(values) - {"cpu", "gpu"})
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unsupported backend(s): {', '.join(unknown)}"
        )
    return values


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _identity_error_blockers(
    row: dict[str, Any],
    *,
    identity_atol: float,
    identity_rtol: float,
) -> tuple[str, ...]:
    """Return blockers from explicit state-error metrics, not only booleans."""

    blockers: list[str] = []
    max_abs = _finite_float(row.get("max_abs_state_error"))
    max_rel = _finite_float(row.get("max_rel_state_error"))

    if not math.isfinite(max_abs):
        blockers.append("identity_abs_error_missing")
    elif max_abs > float(identity_atol):
        blockers.append("identity_abs_error_above_tolerance")

    if not math.isfinite(max_rel):
        blockers.append("identity_rel_error_missing")
    elif max_rel > float(identity_rtol):
        blockers.append("identity_rel_error_above_tolerance")

    return tuple(blockers)


def _repo_relative(path: Path) -> str:
    """Return a stable artifact path for generated JSON/CSV metadata."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    """Load rows from sweep or combined nonlinear sharding artifacts."""

    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        backend = str(payload.get("backend", "")).lower()
        source_kind = str(
            payload.get("kind", "nonlinear_sharding_strong_scaling_sweep")
        )
        for row in payload.get("rows", []):
            item = dict(row)
            item.setdefault("backend", backend or str(item.get("backend", "")).lower())
            item["backend"] = str(item["backend"]).lower()
            item["source"] = _repo_relative(path)
            item["source_kind"] = source_kind
            rows.append(item)
    return rows


def _row_blockers(
    row: dict[str, Any],
    *,
    min_speedup: float,
    min_efficiency: float,
    min_devices: int,
    identity_atol: float,
    identity_rtol: float,
) -> tuple[str, ...]:
    blockers: list[str] = []
    requested_devices = int(row.get("requested_devices") or 0)
    actual_devices = int(row.get("actual_devices") or 0)
    speedup = _finite_float(row.get("strong_speedup_vs_1_device"))
    denominator = actual_devices if actual_devices > 0 else requested_devices
    efficiency = (
        speedup / denominator
        if denominator > 0 and math.isfinite(speedup)
        else math.nan
    )

    if requested_devices < int(min_devices):
        blockers.append("below_min_devices")
    if actual_devices < int(min_devices):
        blockers.append("actual_devices_below_min_devices")
    if not bool(row.get("state_sharding_active", False)):
        blockers.append("state_sharding_inactive")
    if not bool(row.get("identity_gate_pass", False)):
        blockers.append("identity_gate_failed")
    blockers.extend(
        _identity_error_blockers(
            row,
            identity_atol=float(identity_atol),
            identity_rtol=float(identity_rtol),
        )
    )
    if row.get("error") not in {None, ""}:
        blockers.append("profile_row_error")
    if not math.isfinite(speedup):
        blockers.append("speedup_missing")
    elif speedup < float(min_speedup):
        blockers.append("speedup_below_threshold")
    if not math.isfinite(efficiency):
        blockers.append("parallel_efficiency_missing")
    elif efficiency < float(min_efficiency):
        blockers.append("parallel_efficiency_below_threshold")
    return tuple(blockers)


def _row_classification(
    blockers: tuple[str, ...],
    *,
    requested_devices: int,
    actual_devices: int,
    min_devices: int,
    speedup: float,
    min_speedup: float,
) -> str:
    """Classify rows for claim-boundary review without changing gate policy."""

    blocker_set = set(blockers)
    if "profile_row_error" in blocker_set:
        return "profile_error"
    if requested_devices < int(min_devices) or actual_devices < int(min_devices):
        return "reference_only"
    if IDENTITY_BLOCKERS & blocker_set:
        return "identity_failed"
    if "state_sharding_inactive" in blocker_set:
        return "inactive_or_fallback"
    if "speedup_missing" in blocker_set:
        return "timing_incomplete"
    if math.isfinite(speedup) and speedup < 1.0:
        return "identity_preserving_regression"
    if (
        "speedup_below_threshold" in blocker_set
        or "parallel_efficiency_below_threshold" in blocker_set
    ):
        return "identity_only_insufficient_speedup"
    if math.isfinite(speedup) and speedup >= float(min_speedup):
        return "production_candidate"
    return "diagnostic_only"


def _backend_summary(
    rows: list[dict[str, Any]],
    required_backends: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Summarize rows by backend for reviewer-facing claim classification."""

    summary: dict[str, dict[str, Any]] = {}
    for backend in required_backends:
        backend_rows = [row for row in rows if row["backend"] == backend]
        classifications = sorted({str(row["classification"]) for row in backend_rows})
        identity_rows = [
            row
            for row in backend_rows
            if row["classification"]
            in {
                "production_candidate",
                "identity_only_insufficient_speedup",
                "identity_preserving_regression",
            }
        ]
        best_identity = None
        if identity_rows:
            best_identity = max(
                identity_rows,
                key=lambda row: (
                    _finite_float(row.get("strong_speedup_vs_1_device")),
                    int(row.get("requested_devices") or 0),
                ),
            )
        summary[backend] = {
            "row_count": len(backend_rows),
            "classifications": classifications,
            "best_identity_preserving_row": best_identity,
            "production_candidate_count": sum(
                1
                for row in backend_rows
                if row["classification"] == "production_candidate"
            ),
        }
    return summary


def _tolerance_fraction(value: Any, tolerance: float) -> float:
    metric = _finite_float(value)
    tol = float(tolerance)
    if not math.isfinite(metric) or tol <= 0.0:
        return math.nan
    return metric / tol


def _worst_identity_error_row(
    rows: list[dict[str, Any]],
    *,
    identity_atol: float,
    identity_rtol: float,
) -> dict[str, Any] | None:
    finite_rows = [
        row
        for row in rows
        if math.isfinite(_finite_float(row.get("max_abs_state_error")))
        and math.isfinite(_finite_float(row.get("max_rel_state_error")))
    ]
    if not finite_rows:
        return None
    worst = max(
        finite_rows,
        key=lambda row: max(
            _tolerance_fraction(row.get("max_abs_state_error"), identity_atol),
            _tolerance_fraction(row.get("max_rel_state_error"), identity_rtol),
        ),
    )
    return {
        "backend": worst["backend"],
        "requested_devices": worst["requested_devices"],
        "actual_devices": worst["actual_devices"],
        "state_sharding_active": worst["state_sharding_active"],
        "identity_gate_pass": worst["identity_gate_pass"],
        "max_abs_state_error": worst["max_abs_state_error"],
        "max_rel_state_error": worst["max_rel_state_error"],
        "identity_abs_tolerance_fraction": worst["identity_abs_tolerance_fraction"],
        "identity_rel_tolerance_fraction": worst["identity_rel_tolerance_fraction"],
        "classification": worst["classification"],
        "source": worst["source"],
    }


def _identity_evidence_summary(
    rows: list[dict[str, Any]],
    required_backends: tuple[str, ...],
    *,
    identity_atol: float,
    identity_rtol: float,
) -> dict[str, dict[str, Any]]:
    """Summarize identity coverage and error margins without speedup promotion."""

    summary: dict[str, dict[str, Any]] = {}
    for backend in required_backends:
        backend_rows = [row for row in rows if row["backend"] == backend]
        finite_abs = [
            _finite_float(row.get("max_abs_state_error"))
            for row in backend_rows
            if math.isfinite(_finite_float(row.get("max_abs_state_error")))
        ]
        finite_rel = [
            _finite_float(row.get("max_rel_state_error"))
            for row in backend_rows
            if math.isfinite(_finite_float(row.get("max_rel_state_error")))
        ]
        identity_complete_rows = [
            row
            for row in backend_rows
            if not (IDENTITY_BLOCKERS & set(row["blockers"]))
        ]
        blocker_counts = {
            blocker: sum(1 for row in backend_rows if blocker in set(row["blockers"]))
            for blocker in sorted(IDENTITY_BLOCKERS)
        }
        max_abs_error = max(finite_abs) if finite_abs else math.nan
        max_rel_error = max(finite_rel) if finite_rel else math.nan
        summary[backend] = {
            "row_count": len(backend_rows),
            "identity_gate_pass_count": sum(
                1 for row in backend_rows if bool(row["identity_gate_pass"])
            ),
            "finite_error_metric_count": sum(
                1
                for row in backend_rows
                if math.isfinite(_finite_float(row.get("max_abs_state_error")))
                and math.isfinite(_finite_float(row.get("max_rel_state_error")))
            ),
            "identity_within_tolerance_count": len(identity_complete_rows),
            "active_identity_within_tolerance_count": sum(
                1
                for row in identity_complete_rows
                if bool(row["state_sharding_active"])
            ),
            "identity_blocker_counts": blocker_counts,
            "max_abs_state_error": max_abs_error,
            "max_rel_state_error": max_rel_error,
            "max_abs_tolerance_fraction": _tolerance_fraction(
                max_abs_error, identity_atol
            ),
            "max_rel_tolerance_fraction": _tolerance_fraction(
                max_rel_error, identity_rtol
            ),
            "worst_finite_error_row": _worst_identity_error_row(
                backend_rows,
                identity_atol=identity_atol,
                identity_rtol=identity_rtol,
            ),
        }
    return summary


def _count_row_values(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(field)
        values = value if isinstance(value, (list, tuple)) else (value,)
        for item in values:
            key = str(item)
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _backend_blocker_report(
    rows: list[dict[str, Any]],
    required_backends: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Explain why each backend did or did not qualify for a claim."""

    report: dict[str, dict[str, Any]] = {}
    for backend in required_backends:
        backend_rows = [row for row in rows if row["backend"] == backend]
        candidate_rows = [
            row
            for row in backend_rows
            if not (REFERENCE_BLOCKERS & set(row["blockers"]))
        ]
        passing_rows = [row for row in candidate_rows if bool(row["candidate_passed"])]
        identity_complete_rows = [
            row
            for row in candidate_rows
            if not (IDENTITY_BLOCKERS & set(row["blockers"]))
        ]
        active_identity_complete_rows = [
            row for row in identity_complete_rows if bool(row["state_sharding_active"])
        ]
        production_speedup_candidate_missing = not passing_rows
        identity_evidence_complete = bool(identity_complete_rows)
        active_identity_evidence_complete = bool(active_identity_complete_rows)

        primary_blockers: list[str] = []
        if not backend_rows:
            primary_blockers.append("backend_rows_missing")
        if not candidate_rows:
            primary_blockers.append("candidate_rows_missing")
        if not identity_evidence_complete:
            primary_blockers.append("identity_evidence_incomplete")
        if not active_identity_evidence_complete:
            primary_blockers.append("active_identity_evidence_incomplete")
        if production_speedup_candidate_missing:
            primary_blockers.append(f"{backend}_production_speedup_candidate_missing")

        report[backend] = {
            "row_count": len(backend_rows),
            "candidate_row_count": len(candidate_rows),
            "passing_candidate_count": len(passing_rows),
            "production_speedup_candidate_missing": production_speedup_candidate_missing,
            "identity_evidence_complete": identity_evidence_complete,
            "active_identity_evidence_complete": active_identity_evidence_complete,
            "classification_counts": _count_row_values(backend_rows, "classification"),
            "candidate_blocker_counts": _count_row_values(candidate_rows, "blockers"),
            "primary_blockers": tuple(primary_blockers),
            "claim_scope": (
                "Backend remains diagnostic unless at least one active candidate row "
                "has complete identity evidence and passes the speedup and efficiency gates."
            ),
        }
    return report


def evaluate_production_gate(
    rows: list[dict[str, Any]],
    *,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
    min_efficiency: float = DEFAULT_MIN_EFFICIENCY,
    min_devices: int = DEFAULT_MIN_DEVICES,
    required_backends: tuple[str, ...] = DEFAULT_REQUIRED_BACKENDS,
    identity_atol: float = DEFAULT_IDENTITY_ATOL,
    identity_rtol: float = DEFAULT_IDENTITY_RTOL,
) -> dict[str, Any]:
    """Return a fail-closed production-speedup gate summary."""

    evaluated_rows: list[dict[str, Any]] = []
    passing_by_backend: dict[str, list[dict[str, Any]]] = {
        backend: [] for backend in required_backends
    }
    for row in rows:
        requested_devices = int(row.get("requested_devices") or 0)
        actual_devices = int(row.get("actual_devices") or 0)
        speedup = _finite_float(row.get("strong_speedup_vs_1_device"))
        denominator = actual_devices if actual_devices > 0 else requested_devices
        efficiency = (
            speedup / denominator
            if denominator > 0 and math.isfinite(speedup)
            else math.nan
        )
        blockers = _row_blockers(
            row,
            min_speedup=min_speedup,
            min_efficiency=min_efficiency,
            min_devices=min_devices,
            identity_atol=identity_atol,
            identity_rtol=identity_rtol,
        )
        classification = _row_classification(
            blockers,
            requested_devices=requested_devices,
            actual_devices=actual_devices,
            min_devices=min_devices,
            speedup=speedup,
            min_speedup=min_speedup,
        )
        item = {
            "backend": str(row.get("backend", "")).lower(),
            "requested_devices": requested_devices,
            "actual_devices": actual_devices,
            "best_spec": row.get("best_spec"),
            "state_sharding_active": bool(row.get("state_sharding_active", False)),
            "identity_gate_pass": bool(row.get("identity_gate_pass", False)),
            "strong_speedup_vs_1_device": speedup,
            "parallel_efficiency": efficiency,
            "max_abs_state_error": row.get("max_abs_state_error"),
            "max_rel_state_error": row.get("max_rel_state_error"),
            "identity_abs_tolerance_fraction": _tolerance_fraction(
                row.get("max_abs_state_error"),
                identity_atol,
            ),
            "identity_rel_tolerance_fraction": _tolerance_fraction(
                row.get("max_rel_state_error"),
                identity_rtol,
            ),
            "identity_atol": float(identity_atol),
            "identity_rtol": float(identity_rtol),
            "source": row.get("source"),
            "candidate_passed": not blockers,
            "classification": classification,
            "blockers": blockers,
        }
        evaluated_rows.append(item)
        if item["candidate_passed"] and item["backend"] in passing_by_backend:
            passing_by_backend[item["backend"]].append(item)

    best_candidates: dict[str, Any] = {}
    gate_blockers: list[str] = []
    for backend in required_backends:
        candidates = passing_by_backend.get(backend, [])
        if not candidates:
            gate_blockers.append(f"{backend}_production_speedup_candidate_missing")
            best_candidates[backend] = None
            continue
        best_candidates[backend] = max(
            candidates,
            key=lambda item: (
                float(item["strong_speedup_vs_1_device"]),
                int(item["requested_devices"]),
            ),
        )

    gate_passed = not gate_blockers
    return _json_clean(
        {
            "kind": "nonlinear_sharding_production_speedup_gate",
            "gate_passed": gate_passed,
            "production_speedup_claim_allowed": gate_passed,
            "status": "production_speedup_candidate"
            if gate_passed
            else "diagnostic_only",
            "required_backends": required_backends,
            "min_devices": int(min_devices),
            "min_speedup_vs_1_device": float(min_speedup),
            "min_parallel_efficiency": float(min_efficiency),
            "identity_atol": float(identity_atol),
            "identity_rtol": float(identity_rtol),
            "best_candidates": best_candidates,
            "backend_summary": _backend_summary(evaluated_rows, required_backends),
            "identity_evidence_summary": _identity_evidence_summary(
                evaluated_rows,
                required_backends,
                identity_atol=float(identity_atol),
                identity_rtol=float(identity_rtol),
            ),
            "backend_blocker_report": _backend_blocker_report(
                evaluated_rows,
                required_backends,
            ),
            "blockers": tuple(gate_blockers),
            "rows": evaluated_rows,
            "claim_scope": (
                "Whole-state nonlinear sharding may be described as a production speedup candidate only "
                "when this gate passes. Otherwise keep it as a diagnostic identity/profiler artifact."
            ),
        }
    )


def write_artifacts(summary: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fieldnames = [
        "backend",
        "requested_devices",
        "actual_devices",
        "best_spec",
        "state_sharding_active",
        "identity_gate_pass",
        "strong_speedup_vs_1_device",
        "parallel_efficiency",
        "max_abs_state_error",
        "max_rel_state_error",
        "identity_abs_tolerance_fraction",
        "identity_rel_tolerance_fraction",
        "classification",
        "candidate_passed",
        "blockers",
        "source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary["rows"])
    return {"json": str(json_path), "csv": str(csv_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--min-speedup", type=float, default=DEFAULT_MIN_SPEEDUP)
    parser.add_argument("--min-efficiency", type=float, default=DEFAULT_MIN_EFFICIENCY)
    parser.add_argument("--min-devices", type=int, default=DEFAULT_MIN_DEVICES)
    parser.add_argument(
        "--required-backends",
        type=_parse_backend_list,
        default=DEFAULT_REQUIRED_BACKENDS,
    )
    parser.add_argument("--identity-atol", type=float, default=DEFAULT_IDENTITY_ATOL)
    parser.add_argument("--identity-rtol", type=float, default=DEFAULT_IDENTITY_RTOL)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_rows([Path(path) for path in args.inputs])
    summary = evaluate_production_gate(
        rows,
        min_speedup=float(args.min_speedup),
        min_efficiency=float(args.min_efficiency),
        min_devices=int(args.min_devices),
        required_backends=tuple(args.required_backends),
        identity_atol=float(args.identity_atol),
        identity_rtol=float(args.identity_rtol),
    )
    paths = write_artifacts(summary, Path(args.out_prefix))
    print(
        json.dumps(
            {
                "gate_passed": summary["gate_passed"],
                "status": summary["status"],
                "paths": paths,
            },
            indent=2,
        )
    )
    return 0 if bool(summary["gate_passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
