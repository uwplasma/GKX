#!/usr/bin/env python3
"""Build the multi-point VMEC/Boozer aggregate-objective FD artifact."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
import signal
import sys
import time
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gkx.artifacts.plotting import set_plot_style  # noqa: E402
from gkx.objectives.sampling import (  # noqa: E402
    solver_grid_options_from_ky_values,
)
from gkx.objectives.vmec_boozer_fd import (  # noqa: E402
    vmec_boozer_aggregate_scalar_objective_finite_difference_report,
)
from gkx.objectives.vmec_boozer_line_search import (  # noqa: E402
    vmec_boozer_aggregate_scalar_objective_line_search_report,
)
from tools.artifacts.build_solver_objective_gradient_gate import _json_clean  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_boozer_aggregate_objective_gate.png"
DEFAULT_LINE_SEARCH_OUT = (
    ROOT / "docs" / "_static" / "vmec_boozer_aggregate_line_search_gate.png"
)


DEFAULT_LINE_SEARCH_COMPARISON_OUT = (
    ROOT / "docs" / "_static" / "vmec_boozer_aggregate_line_search_comparison.png"
)
DEFAULT_MULTI_POINT_OUT = (
    ROOT / "docs" / "_static" / "vmec_boozer_multi_point_objective_gate.png"
)
DEFAULT_SECOND_EQUILIBRIUM_OUT = (
    ROOT / "docs" / "_static" / "vmec_boozer_second_equilibrium_aggregate_gate.png"
)
DEFAULT_COMPARISON_OBJECTIVES = ("growth", "quasilinear_flux")
DEFAULT_MULTI_POINT_ALPHAS = (0.0, 0.5)
DEFAULT_MULTI_POINT_SELECTED_KY_INDICES = (1, 2)
DEFAULT_SECOND_EQUILIBRIUM_CASE_NAME = "li383_low_res"
DEFAULT_SECOND_EQUILIBRIUM_ALPHAS = (0.0,)
DEFAULT_SECOND_EQUILIBRIUM_SELECTED_KY_INDICES = (1, 2)
DEFAULT_MULTI_POINT_MAX_SAMPLES = 8
DEFAULT_SECOND_EQUILIBRIUM_MAX_SAMPLES = 4
DEFAULT_VMEC_BOOZER_MAX_WALL_SECONDS = 300.0


def _finite_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _float_tuple(raw: Sequence[float] | None, *, name: str) -> tuple[float, ...]:
    if raw is None or len(raw) == 0:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(float(item) for item in raw)


def _int_tuple(raw: Sequence[int] | None, *, name: str) -> tuple[int, ...]:
    if raw is None or len(raw) == 0:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(int(item) for item in raw)


def _unique_count(values: Sequence[object]) -> int:
    return len(set(values))


def _validate_multi_alpha_or_surface(
    surfaces: tuple[int | None, ...],
    alphas: tuple[float, ...],
) -> None:
    if _unique_count(surfaces) >= 2 or _unique_count(alphas) >= 2:
        return
    raise ValueError(
        "multi-point VMEC/Boozer gate requires at least two alphas or two "
        "surface indices"
    )


def _sample_count(
    surfaces: tuple[int | None, ...],
    alphas: tuple[float, ...],
    selected_ky_indices: tuple[int, ...],
) -> int:
    return len(surfaces) * len(alphas) * len(selected_ky_indices)


def _validate_sample_bound(sample_count: int, *, max_samples: int) -> None:
    if int(max_samples) < 1:
        raise ValueError("max_samples must be at least 1")
    if int(sample_count) > int(max_samples):
        raise ValueError(
            f"requested {sample_count} aggregate samples, exceeding "
            f"--max-samples={int(max_samples)}"
        )


@contextmanager
def _wall_time_limit(
    seconds: float, *, label: str = "VMEC/Boozer aggregate gate"
) -> Iterator[None]:
    """Bound optional-backend gate generation on Unix-like CI hosts."""

    seconds_float = float(seconds)
    if seconds_float <= 0.0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"{label} exceeded {seconds_float:g} s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds_float)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _surface_indices(raw: Sequence[int | None] | None) -> tuple[int | None, ...]:
    if raw is None or len(raw) == 0:
        return (None,)
    return tuple(None if item is None else int(item) for item in raw)


def write_vmec_boozer_aggregate_objective_artifacts(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_OUT,
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for an aggregate-objective payload."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    samples = payload.get("samples", [])
    base_values = payload.get("base_sample_values", [])
    minus_values = payload.get("minus_sample_values", [])
    plus_values = payload.get("plus_sample_values", [])
    rows = samples if isinstance(samples, list) else []
    fieldnames = [
        "sample",
        "surface_index",
        "torflux",
        "surface",
        "alpha",
        "ky",
        "selected_ky_index",
        "selected_ky",
        "ky_abs_error",
        "weight",
        "minus_value",
        "base_value",
        "plus_value",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            sample = row if isinstance(row, dict) else {}
            writer.writerow(
                {
                    "sample": index,
                    "surface_index": sample.get("surface_index", ""),
                    "torflux": sample.get("torflux", ""),
                    "surface": sample.get("surface", ""),
                    "alpha": sample.get("alpha", ""),
                    "ky": sample.get("ky", ""),
                    "selected_ky_index": sample.get("selected_ky_index", ""),
                    "selected_ky": sample.get("selected_ky", ""),
                    "ky_abs_error": sample.get("ky_abs_error", ""),
                    "weight": sample.get("weight", ""),
                    "minus_value": minus_values[index]
                    if isinstance(minus_values, list) and index < len(minus_values)
                    else "",
                    "base_value": base_values[index]
                    if isinstance(base_values, list) and index < len(base_values)
                    else "",
                    "plus_value": plus_values[index]
                    if isinstance(plus_values, list) and index < len(plus_values)
                    else "",
                }
            )

    base = np.asarray(base_values, dtype=float)
    minus = np.asarray(minus_values, dtype=float)
    plus = np.asarray(plus_values, dtype=float)
    labels = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        surface = row.get("surface_index", "mid")
        if row.get("torflux") not in (None, ""):
            surface_label = f"{float(row['torflux']):.3g}"
        else:
            surface_label = "mid" if surface is None else str(surface)
        ky_label = (
            f"ky={float(row['ky']):.3g}"
            if "ky" in row and row.get("ky") not in (None, "")
            else f"ky#{int(row.get('selected_ky_index', 0))}"
        )
        labels.append(
            f"s={surface_label}, a={float(row.get('alpha', 0.0)):.2g}, {ky_label}"
        )
    if not labels:
        labels = [str(index) for index in range(base.size)]

    set_plot_style()
    fig, (ax_values, ax_summary) = plt.subplots(
        1, 2, figsize=(12.2, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )
    x = np.arange(base.size)
    width = 0.24
    ax_values.bar(
        x - width,
        minus,
        width,
        label=r"$x-h$",
        color="#8ecae6",
        edgecolor="#202020",
        linewidth=0.4,
    )
    ax_values.bar(
        x,
        base,
        width,
        label=r"$x$",
        color="#219ebc",
        edgecolor="#202020",
        linewidth=0.4,
    )
    ax_values.bar(
        x + width,
        plus,
        width,
        label=r"$x+h$",
        color="#023047",
        edgecolor="#202020",
        linewidth=0.4,
    )
    ax_values.set_xticks(x, labels, rotation=18, ha="right")
    ax_values.set_ylabel(str(payload.get("objective", "objective")))
    ax_values.set_title("Per-sample scalar objective")
    ax_values.grid(axis="y", alpha=0.25)
    ax_values.legend(frameon=False, fontsize=8)

    passed = bool(payload.get("passed"))
    status = "passed" if passed else "open"
    summary_lines = [
        f"status: {status}",
        f"objective: {payload.get('objective')}",
        f"reduction: {payload.get('reduction')}",
        f"samples: {payload.get('n_samples')}",
        f"base: {float(payload.get('base_value', float('nan'))):.6g}",
        f"central FD: {float(payload.get('central_derivative', float('nan'))):.6g}",
        f"response: {float(payload.get('response_abs', float('nan'))):.3e}",
        f"curvature ratio: {float(payload.get('curvature_ratio', float('nan'))):.3e}",
    ]
    ax_summary.axis("off")
    ax_summary.set_title("Aggregate FD gate")
    ax_summary.text(
        0.02,
        0.95,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        transform=ax_summary.transAxes,
    )
    ax_summary.text(
        0.02,
        0.16,
        "Finite-difference sensitivity through the in-memory\n"
        "VMEC/Boozer/GKX value path. This is a\n"
        "multi-point reduced linear/QL objective gate, not a\n"
        "nonlinear turbulent transport optimization claim.",
        va="top",
        ha="left",
        fontsize=8.3,
        transform=ax_summary.transAxes,
    )
    fig.suptitle(f"VMEC/Boozer multi-point aggregate-objective gate: {status}")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.22, wspace=0.28)
    fig.savefig(out_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "png": str(out_path),
        "pdf": str(pdf_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def write_vmec_boozer_aggregate_line_search_artifacts(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_LINE_SEARCH_OUT,
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for an aggregate line-search payload."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    history = payload.get("history", [])
    rows = history if isinstance(history, list) else []
    fieldnames = [
        "step",
        "delta",
        "objective",
        "central_derivative",
        "curvature_ratio",
        "accepted",
        "candidate_delta",
        "candidate_objective",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if isinstance(row, dict):
                writer.writerow({name: row.get(name, "") for name in fieldnames})

    step_labels = [
        str(row.get("step", index))
        for index, row in enumerate(rows)
        if isinstance(row, dict)
    ]
    objectives = np.asarray(
        [float(row.get("objective", np.nan)) for row in rows if isinstance(row, dict)],
        dtype=float,
    )
    candidates = np.asarray(
        [
            float(row.get("candidate_objective", np.nan))
            for row in rows
            if isinstance(row, dict)
        ],
        dtype=float,
    )
    curvature = np.asarray(
        [
            float(row.get("curvature_ratio", np.nan))
            for row in rows
            if isinstance(row, dict)
        ],
        dtype=float,
    )

    set_plot_style()
    fig, (ax_obj, ax_meta) = plt.subplots(
        1, 2, figsize=(12.0, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )
    x = np.arange(max(len(objectives), 1))
    if objectives.size:
        ax_obj.plot(x, objectives, marker="o", lw=2.0, color="#005f73", label="current")
    if candidates.size:
        ax_obj.plot(
            x, candidates, marker="s", lw=1.8, color="#ca6702", label="candidate"
        )
    ax_obj.set_xticks(x, step_labels or ["0"])
    ax_obj.set_xlabel("line-search step")
    ax_obj.set_ylabel(str(payload.get("objective", "objective")))
    ax_obj.set_title("Aggregate objective decrease")
    ax_obj.grid(alpha=0.25)
    ax_obj.legend(frameon=False, fontsize=8)

    if curvature.size and np.all(np.isfinite(curvature)):
        ax_curve = ax_obj.twinx()
        ax_curve.plot(
            x,
            curvature,
            marker="^",
            lw=1.3,
            ls="--",
            color="#6a4c93",
            label="curvature",
        )
        ax_curve.set_ylabel("curvature ratio")
        ax_curve.tick_params(axis="y", labelcolor="#6a4c93")

    passed = bool(payload.get("passed"))
    status = "passed" if passed else "open"
    summary_lines = [
        f"status: {status}",
        f"objective: {payload.get('objective')}",
        f"reduction: {payload.get('reduction')}",
        f"samples: {payload.get('n_samples')}",
        f"accepted: {payload.get('accepted_steps')}/{payload.get('max_steps')}",
        f"initial: {float(payload.get('initial_objective', float('nan'))):.6g}",
        f"final: {float(payload.get('final_objective', float('nan'))):.6g}",
        f"rel. reduction: {payload.get('relative_reduction')}",
        f"stop: {payload.get('stop_reason')}",
    ]
    ax_meta.axis("off")
    ax_meta.set_title("Line-search gate")
    ax_meta.text(
        0.02,
        0.95,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.4,
        transform=ax_meta.transAxes,
    )
    ax_meta.text(
        0.02,
        0.14,
        "Every attempted update must pass the aggregate\n"
        "finite-difference curvature gate and decrease the\n"
        "multi-point reduced objective. This is optimizer\n"
        "control-flow evidence, not a nonlinear transport claim.",
        va="top",
        ha="left",
        fontsize=8.3,
        transform=ax_meta.transAxes,
    )
    fig.suptitle(f"VMEC/Boozer aggregate-objective line-search gate: {status}")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.18, wspace=0.28)
    fig.savefig(out_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "png": str(out_path),
        "pdf": str(pdf_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def _first_history_row(report: dict[str, object]) -> dict[str, object]:
    history = report.get("history", [])
    if isinstance(history, list) and history and isinstance(history[0], dict):
        return history[0]
    return {}


def _line_search_direction(derivative: float) -> str:
    if not math.isfinite(derivative) or derivative == 0.0:
        return "stationary_or_unresolved"
    return "negative_delta" if derivative > 0.0 else "positive_delta"


def _sample_key(sample: object) -> tuple[object, ...]:
    if not isinstance(sample, dict):
        return (repr(sample),)
    return (
        sample.get("surface_index"),
        _finite_float(sample.get("alpha")),
        sample.get("selected_ky_index"),
        _finite_float(sample.get("weight")),
    )


def _samples_match(reports: dict[str, dict[str, object]]) -> bool:
    keys: list[tuple[tuple[object, ...], ...]] = []
    for report in reports.values():
        samples = report.get("samples", [])
        if not isinstance(samples, list) or not samples:
            return False
        keys.append(tuple(_sample_key(sample) for sample in samples))
    return bool(keys) and all(key == keys[0] for key in keys[1:])


def _summarize_report(objective: str, report: dict[str, object]) -> dict[str, object]:
    first = _first_history_row(report)
    derivative = _finite_float(first.get("central_derivative"))
    initial = _finite_float(report.get("initial_objective"))
    final = _finite_float(report.get("final_objective"))
    reduction = _finite_float(report.get("relative_reduction"), default=math.nan)
    return {
        "objective": str(objective),
        "passed": bool(report.get("passed", False)),
        "n_samples": int(report.get("n_samples", 0) or 0),
        "initial_objective": initial,
        "final_objective": final,
        "absolute_reduction": initial - final
        if math.isfinite(initial) and math.isfinite(final)
        else math.nan,
        "relative_reduction": reduction,
        "initial_central_derivative": derivative,
        "initial_update_direction": _line_search_direction(derivative),
        "accepted_steps": int(report.get("accepted_steps", 0) or 0),
        "max_steps": int(report.get("max_steps", 0) or 0),
        "initial_delta": _finite_float(report.get("initial_delta")),
        "final_delta": _finite_float(report.get("final_delta")),
        "stop_reason": str(report.get("stop_reason", "")),
    }


def build_vmec_boozer_aggregate_line_search_comparison_report(
    *,
    objectives: tuple[str, ...] = DEFAULT_COMPARISON_OBJECTIVES,
    case_name: str = "nfp4_QH_warm_start",
    reduction: str = "mean",
    surface_indices: tuple[int | None, ...] = (None,),
    alphas: tuple[float, ...] = (0.0,),
    selected_ky_indices: tuple[int, ...] = (1, 2),
    radial_index: int | None = None,
    mode_index: int = 1,
    perturbation_step: float = 1.0e-7,
    update_step: float = 1.0e-8,
    max_steps: int = 1,
    min_improvement: float = 0.0,
    response_atol: float = 0.0,
    max_curvature_ratio: float = 5.0,
    **kwargs: Any,
) -> dict[str, object]:
    """Run aggregate line searches for each objective and compare summaries."""

    objective_tuple = tuple(str(item) for item in objectives)
    if len(objective_tuple) < 2:
        raise ValueError("objectives must contain at least two entries")
    reports: dict[str, dict[str, object]] = {}
    for objective in objective_tuple:
        reports[objective] = vmec_boozer_aggregate_scalar_objective_line_search_report(
            case_name=case_name,
            objective=objective,  # type: ignore[arg-type]
            reduction=reduction,  # type: ignore[arg-type]
            surface_indices=surface_indices,
            alphas=alphas,
            selected_ky_indices=selected_ky_indices,
            radial_index=radial_index,
            mode_index=mode_index,
            perturbation_step=perturbation_step,
            update_step=update_step,
            max_steps=max_steps,
            min_improvement=min_improvement,
            response_atol=response_atol,
            max_curvature_ratio=max_curvature_ratio,
            **kwargs,
        )

    rows = [
        _summarize_report(objective, reports[objective])
        for objective in objective_tuple
    ]
    same_samples = _samples_match(reports)
    directions = [str(row["initial_update_direction"]) for row in rows]
    same_direction = bool(
        directions and all(direction == directions[0] for direction in directions[1:])
    )
    all_line_searches_passed = all(bool(row["passed"]) for row in rows)
    final_deltas = [_finite_float(row["final_delta"]) for row in rows]
    delta_spread = (
        float(max(final_deltas) - min(final_deltas))
        if all(math.isfinite(item) for item in final_deltas)
        else None
    )
    reductions = [_finite_float(row["relative_reduction"]) for row in rows]
    finite_reductions = [item for item in reductions if math.isfinite(item)]
    reduction_spread = (
        float(max(finite_reductions) - min(finite_reductions))
        if finite_reductions
        else None
    )

    return {
        "kind": "vmec_boozer_aggregate_line_search_comparison",
        "passed": bool(all_line_searches_passed and same_samples),
        "source_scope": "mode21_vmec_boozer_state_multi_point",
        "claim_scope": (
            "side-by-side one-parameter line-search comparison for reduced aggregate growth "
            "and quasilinear proxy objectives on the same QH VMEC/Boozer sample set; "
            "not a nonlinear turbulent transport or optimizer-convergence claim"
        ),
        "case_name": str(case_name),
        "objectives": list(objective_tuple),
        "reduction": str(reduction),
        "n_samples": int(rows[0].get("n_samples", 0) or 0) if rows else 0,
        "same_sample_set": same_samples,
        "all_line_searches_passed": all_line_searches_passed,
        "same_initial_update_direction": same_direction,
        "initial_update_directions": {
            str(row["objective"]): str(row["initial_update_direction"]) for row in rows
        },
        "final_delta_spread": delta_spread,
        "relative_reduction_spread": reduction_spread,
        "rows": rows,
        "reports": reports,
        "notes": (
            "Direction agreement is recorded as a diagnostic, not a pass/fail criterion beyond "
            "requiring both underlying line-search reports to be valid on the same samples."
        ),
        "options": {
            key: value
            for key, value in kwargs.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
    }


def write_vmec_boozer_aggregate_line_search_comparison_artifacts(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_LINE_SEARCH_COMPARISON_OUT,
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for an aggregate line-search comparison."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = payload.get("rows", [])
    row_list = rows if isinstance(rows, list) else []
    fieldnames = [
        "objective",
        "passed",
        "n_samples",
        "initial_objective",
        "final_objective",
        "absolute_reduction",
        "relative_reduction",
        "initial_central_derivative",
        "initial_update_direction",
        "accepted_steps",
        "max_steps",
        "initial_delta",
        "final_delta",
        "stop_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in row_list:
            if isinstance(row, dict):
                writer.writerow({name: row.get(name, "") for name in fieldnames})

    labels = [
        str(row.get("objective", index))
        for index, row in enumerate(row_list)
        if isinstance(row, dict)
    ]
    initial = np.asarray(
        [
            _finite_float(row.get("initial_objective"))
            for row in row_list
            if isinstance(row, dict)
        ],
        dtype=float,
    )
    final = np.asarray(
        [
            _finite_float(row.get("final_objective"))
            for row in row_list
            if isinstance(row, dict)
        ],
        dtype=float,
    )
    rel = np.asarray(
        [
            _finite_float(row.get("relative_reduction"))
            for row in row_list
            if isinstance(row, dict)
        ],
        dtype=float,
    )
    derivative = np.asarray(
        [
            _finite_float(row.get("initial_central_derivative"))
            for row in row_list
            if isinstance(row, dict)
        ],
        dtype=float,
    )
    final_delta = np.asarray(
        [
            _finite_float(row.get("final_delta"))
            for row in row_list
            if isinstance(row, dict)
        ],
        dtype=float,
    )
    finite_initial = np.where(
        np.isfinite(initial) & (np.abs(initial) > 0.0), initial, np.nan
    )
    normalized_final = final / finite_initial

    set_plot_style()
    fig = plt.figure(figsize=(13.6, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.05])
    ax_obj = fig.add_subplot(gs[0, 0])
    ax_dir = fig.add_subplot(gs[0, 1])
    ax_meta = fig.add_subplot(gs[0, 2])
    x = np.arange(max(len(labels), 1))
    width = 0.34
    if labels:
        ax_obj.bar(
            x - width / 2.0,
            np.ones_like(x, dtype=float),
            width,
            label="initial",
            color="#94d2bd",
            edgecolor="#1f2937",
        )
        ax_obj.bar(
            x + width / 2.0,
            normalized_final,
            width,
            label="final / initial",
            color="#005f73",
            edgecolor="#1f2937",
        )
        for xi, value in zip(x, rel, strict=True):
            text = f"{value:.2%}" if math.isfinite(float(value)) else "n/a"
            ax_obj.text(xi, 1.03, text, ha="center", va="bottom", fontsize=8)
    ax_obj.set_xticks(x, labels or ["objective"], rotation=15, ha="right")
    ax_obj.set_ylabel("normalized objective")
    ax_obj.set_title("Line-search outcome")
    ax_obj.grid(axis="y", alpha=0.25)
    ax_obj.legend(frameon=False, fontsize=8)

    signs = np.sign(derivative)
    colors = [
        "#ca6702" if sign > 0.0 else "#0a9396" if sign < 0.0 else "#6b7280"
        for sign in signs
    ]
    if labels:
        ax_dir.bar(x, final_delta, color=colors, edgecolor="#1f2937")
        for xi, deriv in zip(x, derivative, strict=True):
            if math.isfinite(float(deriv)):
                ax_dir.text(
                    xi,
                    0.0,
                    f"dJ/dx={deriv:.2g}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    rotation=90,
                )
    ax_dir.axhline(0.0, color="#111827", linewidth=1.0)
    ax_dir.set_xticks(x, labels or ["objective"], rotation=15, ha="right")
    ax_dir.set_ylabel("final VMEC coefficient delta")
    ax_dir.set_title("Initial descent direction")
    ax_dir.grid(axis="y", alpha=0.25)

    passed = bool(payload.get("passed"))
    status = "passed" if passed else "open"
    summary_lines = [
        f"status: {status}",
        f"case: {payload.get('case_name')}",
        f"reduction: {payload.get('reduction')}",
        f"samples: {payload.get('n_samples')}",
        f"same samples: {payload.get('same_sample_set')}",
        f"same direction: {payload.get('same_initial_update_direction')}",
        f"delta spread: {payload.get('final_delta_spread')}",
        f"rel. reduction spread: {payload.get('relative_reduction_spread')}",
    ]
    ax_meta.axis("off")
    ax_meta.set_title("Comparison scope")
    ax_meta.text(
        0.02,
        0.95,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.2,
        transform=ax_meta.transAxes,
    )
    ax_meta.text(
        0.02,
        0.22,
        "Growth and QL proxy line searches are run with\n"
        "the same QH surfaces, field lines, ky samples,\n"
        "VMEC coefficient, and finite-difference controls.\n"
        "This compares reduced directions/results only; it\n"
        "does not validate nonlinear turbulent transport.",
        va="top",
        ha="left",
        fontsize=8.3,
        transform=ax_meta.transAxes,
    )
    fig.suptitle(f"VMEC/Boozer aggregate growth-vs-QL line search: {status}", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.20, wspace=0.35)
    fig.savefig(out_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "png": str(out_path),
        "pdf": str(pdf_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def _annotate_payload(
    payload: dict[str, object],
    *,
    surfaces: tuple[int | None, ...],
    alphas: tuple[float, ...],
    selected_ky_indices: tuple[int, ...],
    max_samples: int,
    max_wall_seconds: float,
    elapsed_wall_seconds: float,
) -> dict[str, object]:
    sample_count = _sample_count(surfaces, alphas, selected_ky_indices)
    multi_alpha_or_surface = _unique_count(surfaces) >= 2 or _unique_count(alphas) >= 2
    annotated = dict(payload)
    annotated["artifact_kind"] = "vmec_boozer_multi_point_objective_gate"
    annotated["builder"] = (
        "tools/artifacts/build_vmec_boozer_aggregate_objective_gate.py multi-point"
    )
    annotated["claim_scope"] = (
        "bounded finite-difference sensitivity of a reduced linear/quasilinear "
        "VMEC/Boozer/GKX objective over multiple fixed field-line "
        "alphas and/or surfaces; not a nonlinear turbulent transport claim"
    )
    annotated["multi_point_coverage"] = {
        "surface_indices": [None if item is None else int(item) for item in surfaces],
        "alphas": [float(item) for item in alphas],
        "selected_ky_indices": [int(item) for item in selected_ky_indices],
        "n_surfaces": len(surfaces),
        "n_alphas": len(alphas),
        "n_selected_ky": len(selected_ky_indices),
        "n_samples_requested": sample_count,
        "multi_alpha_or_surface": bool(multi_alpha_or_surface),
    }
    annotated["bounded_runtime"] = {
        "max_samples": int(max_samples),
        "max_wall_seconds": float(max_wall_seconds),
        "elapsed_wall_seconds": float(elapsed_wall_seconds),
    }
    annotated["passed"] = bool(annotated.get("passed", False)) and bool(
        multi_alpha_or_surface
    )
    annotated["next_action"] = (
        "Use this artifact only for reduced linear/quasilinear objective "
        "plumbing across fixed VMEC/Boozer field-line or surface samples. "
        "Nonlinear transport optimization still requires separate long-window "
        "nonlinear gates."
    )
    return annotated


def build_vmec_boozer_multi_point_objective_payload(
    *,
    case_name: str = "nfp4_QH_warm_start",
    objective: str = "quasilinear_flux",
    reduction: str = "mean",
    surface_indices: Sequence[int | None] | None = None,
    alphas: Sequence[float] = DEFAULT_MULTI_POINT_ALPHAS,
    selected_ky_indices: Sequence[int] = DEFAULT_MULTI_POINT_SELECTED_KY_INDICES,
    radial_index: int | None = None,
    mode_index: int = 1,
    perturbation_step: float = 1.0e-7,
    max_curvature_ratio: float = 5.0,
    response_atol: float = 0.0,
    ntheta: int = 4,
    mboz: int = 21,
    nboz: int = 21,
    surface_stencil_width: int | None = 3,
    n_laguerre: int = 2,
    n_hermite: int = 3,
    nx: int = 1,
    ny: int = 6,
    max_samples: int = DEFAULT_MULTI_POINT_MAX_SAMPLES,
    max_wall_seconds: float = DEFAULT_VMEC_BOOZER_MAX_WALL_SECONDS,
) -> dict[str, object]:
    """Build a bounded multi-alpha/surface aggregate objective payload."""

    surfaces = _surface_indices(surface_indices)
    alpha_values = _float_tuple(alphas, name="alphas")
    ky_indices = _int_tuple(selected_ky_indices, name="selected_ky_indices")
    _validate_multi_alpha_or_surface(surfaces, alpha_values)
    sample_count = _sample_count(surfaces, alpha_values, ky_indices)
    _validate_sample_bound(sample_count, max_samples=int(max_samples))

    start = time.perf_counter()
    with _wall_time_limit(float(max_wall_seconds)):
        payload = vmec_boozer_aggregate_scalar_objective_finite_difference_report(
            case_name=case_name,
            objective=objective,
            reduction=reduction,
            surface_indices=surfaces,
            alphas=alpha_values,
            selected_ky_indices=ky_indices,
            radial_index=radial_index,
            mode_index=mode_index,
            perturbation_step=perturbation_step,
            response_atol=response_atol,
            max_curvature_ratio=max_curvature_ratio,
            ntheta=ntheta,
            mboz=mboz,
            nboz=nboz,
            surface_stencil_width=surface_stencil_width,
            n_laguerre=n_laguerre,
            n_hermite=n_hermite,
            nx=nx,
            ny=ny,
        )
    elapsed = time.perf_counter() - start
    return _annotate_payload(
        payload,
        surfaces=surfaces,
        alphas=alpha_values,
        selected_ky_indices=ky_indices,
        max_samples=int(max_samples),
        max_wall_seconds=float(max_wall_seconds),
        elapsed_wall_seconds=elapsed,
    )


def write_vmec_boozer_multi_point_objective_artifacts(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_MULTI_POINT_OUT,
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for the multi-point gate payload."""

    return write_vmec_boozer_aggregate_objective_artifacts(payload, out=out)


def _as_public_surfaces(surfaces: Sequence[int | None]) -> list[int | None]:
    return [None if item is None else int(item) for item in surfaces]


def _mode_bound(mboz: int, nboz: int) -> dict[str, object]:
    return {
        "mboz": int(mboz),
        "nboz": int(nboz),
        "minimum_required": 21,
        "passed": bool(int(mboz) >= 21 and int(nboz) >= 21),
    }


def _fail_closed_payload(
    *,
    case_name: str,
    blocker: BaseException,
    elapsed_wall_seconds: float,
    max_wall_seconds: float,
    mboz: int,
    nboz: int,
    surfaces: Sequence[int | None],
    alphas: Sequence[float],
    selected_ky_indices: Sequence[int],
    max_samples: int,
) -> dict[str, object]:
    sample_count = _sample_count(
        tuple(surfaces), tuple(alphas), tuple(selected_ky_indices)
    )
    return {
        "kind": "vmec_boozer_second_equilibrium_aggregate_gate",
        "builder": "tools/artifacts/build_vmec_boozer_aggregate_objective_gate.py second-equilibrium",
        "passed": False,
        "feasible": False,
        "case_name": str(case_name),
        "source_scope": "mode21_vmec_boozer_state_second_equilibrium_aggregate",
        "claim_scope": (
            "second-equilibrium reduced aggregate VMEC/Boozer/GKX "
            "finite-difference or line-search feasibility at mboz=nboz>=21; "
            "not a nonlinear turbulent transport optimization claim"
        ),
        "blocker_type": type(blocker).__name__,
        "blocker_message": str(blocker),
        "mode_bound": _mode_bound(mboz, nboz),
        "sample_bound": {
            "n_samples_requested": int(sample_count),
            "max_samples": int(max_samples),
            "passed": bool(int(sample_count) <= int(max_samples)),
        },
        "bounded_runtime": {
            "max_wall_seconds": float(max_wall_seconds),
            "elapsed_wall_seconds": float(elapsed_wall_seconds),
            "passed": bool(float(elapsed_wall_seconds) <= float(max_wall_seconds)),
        },
        "coverage": {
            "surface_indices": _as_public_surfaces(surfaces),
            "alphas": [float(item) for item in alphas],
            "selected_ky_indices": [int(item) for item in selected_ky_indices],
        },
        "next_action": (
            "Resolve the recorded fixture, memory, optional-backend API, or runtime "
            "blocker before using this second equilibrium in aggregate optimization gates."
        ),
    }


def build_vmec_boozer_second_equilibrium_aggregate_payload(
    *,
    case_name: str = DEFAULT_SECOND_EQUILIBRIUM_CASE_NAME,
    objective: str = "quasilinear_flux",
    reduction: str = "mean",
    surface_indices: Sequence[int | None] | None = None,
    alphas: Sequence[float] = DEFAULT_SECOND_EQUILIBRIUM_ALPHAS,
    selected_ky_indices: Sequence[int] = DEFAULT_SECOND_EQUILIBRIUM_SELECTED_KY_INDICES,
    radial_index: int | None = None,
    mode_index: int = 1,
    perturbation_step: float = 1.0e-7,
    update_step: float = 1.0e-8,
    max_steps: int = 1,
    min_improvement: float = 0.0,
    max_curvature_ratio: float = 5.0,
    response_atol: float = 0.0,
    ntheta: int = 4,
    mboz: int = 21,
    nboz: int = 21,
    surface_stencil_width: int | None = 3,
    n_laguerre: int = 2,
    n_hermite: int = 3,
    nx: int = 1,
    ny: int = 6,
    max_samples: int = DEFAULT_SECOND_EQUILIBRIUM_MAX_SAMPLES,
    max_wall_seconds: float = DEFAULT_VMEC_BOOZER_MAX_WALL_SECONDS,
) -> dict[str, object]:
    """Return a pass/fail aggregate gate payload for a non-QH equilibrium."""

    surfaces = _surface_indices(surface_indices)
    alpha_values = _float_tuple(alphas, name="alphas")
    ky_indices = _int_tuple(selected_ky_indices, name="selected_ky_indices")
    sample_count = _sample_count(surfaces, alpha_values, ky_indices)
    _validate_sample_bound(sample_count, max_samples=int(max_samples))
    mode_bound = _mode_bound(mboz, nboz)
    if not mode_bound["passed"]:
        raise ValueError("second-equilibrium aggregate gate requires mboz,nboz >= 21")

    start = time.perf_counter()
    common_kwargs: dict[str, Any] = {
        "case_name": str(case_name),
        "objective": str(objective),
        "reduction": str(reduction),
        "surface_indices": surfaces,
        "alphas": alpha_values,
        "selected_ky_indices": ky_indices,
        "radial_index": radial_index,
        "mode_index": int(mode_index),
        "perturbation_step": float(perturbation_step),
        "response_atol": float(response_atol),
        "max_curvature_ratio": float(max_curvature_ratio),
        "ntheta": int(ntheta),
        "mboz": int(mboz),
        "nboz": int(nboz),
        "surface_stencil_width": surface_stencil_width,
        "n_laguerre": int(n_laguerre),
        "n_hermite": int(n_hermite),
        "nx": int(nx),
        "ny": int(ny),
    }
    try:
        with _wall_time_limit(float(max_wall_seconds)):
            fd_report = vmec_boozer_aggregate_scalar_objective_finite_difference_report(
                **common_kwargs,
            )
            line_search_report = (
                vmec_boozer_aggregate_scalar_objective_line_search_report(
                    **common_kwargs,
                    update_step=float(update_step),
                    max_steps=int(max_steps),
                    min_improvement=float(min_improvement),
                )
            )
    except Exception as exc:  # noqa: BLE001 - the artifact must fail closed with exact blocker metadata.
        elapsed = time.perf_counter() - start
        return _fail_closed_payload(
            case_name=str(case_name),
            blocker=exc,
            elapsed_wall_seconds=elapsed,
            max_wall_seconds=float(max_wall_seconds),
            mboz=int(mboz),
            nboz=int(nboz),
            surfaces=surfaces,
            alphas=alpha_values,
            selected_ky_indices=ky_indices,
            max_samples=int(max_samples),
        )

    elapsed = time.perf_counter() - start
    runtime_passed = bool(
        float(max_wall_seconds) <= 0.0 or elapsed <= float(max_wall_seconds)
    )
    fd_passed = bool(fd_report.get("passed", False))
    line_search_passed = bool(line_search_report.get("passed", False))
    return {
        "kind": "vmec_boozer_second_equilibrium_aggregate_gate",
        "builder": "tools/artifacts/build_vmec_boozer_aggregate_objective_gate.py second-equilibrium",
        "passed": bool(fd_passed and line_search_passed and runtime_passed),
        "feasible": bool(fd_passed or line_search_passed),
        "case_name": str(case_name),
        "source_scope": "mode21_vmec_boozer_state_second_equilibrium_aggregate",
        "claim_scope": (
            "second-equilibrium reduced aggregate VMEC/Boozer/GKX "
            "finite-difference and one-parameter line-search gate at mboz=nboz>=21; "
            "not a nonlinear turbulent transport optimization claim"
        ),
        "objective": str(objective),
        "reduction": str(reduction),
        "coverage": {
            "surface_indices": _as_public_surfaces(surfaces),
            "alphas": [float(item) for item in alpha_values],
            "selected_ky_indices": [int(item) for item in ky_indices],
            "n_samples_requested": int(sample_count),
        },
        "mode_bound": mode_bound,
        "sample_bound": {
            "n_samples_requested": int(sample_count),
            "max_samples": int(max_samples),
            "passed": True,
        },
        "bounded_runtime": {
            "max_wall_seconds": float(max_wall_seconds),
            "elapsed_wall_seconds": float(elapsed),
            "passed": runtime_passed,
        },
        "finite_difference_passed": fd_passed,
        "line_search_passed": line_search_passed,
        "finite_difference_summary": {
            "base_value": fd_report.get("base_value"),
            "minus_value": fd_report.get("minus_value"),
            "plus_value": fd_report.get("plus_value"),
            "central_derivative": fd_report.get("central_derivative"),
            "response_abs": fd_report.get("response_abs"),
            "curvature_ratio": fd_report.get("curvature_ratio"),
            "n_samples": fd_report.get("n_samples"),
        },
        "line_search_summary": {
            "accepted_steps": line_search_report.get("accepted_steps"),
            "initial_objective": line_search_report.get("initial_objective"),
            "final_objective": line_search_report.get("final_objective"),
            "relative_reduction": line_search_report.get("relative_reduction"),
            "stop_reason": line_search_report.get("stop_reason"),
            "n_samples": line_search_report.get("n_samples"),
        },
        "finite_difference_report": fd_report,
        "line_search_report": line_search_report,
        "next_action": (
            "Use this only as a second-equilibrium reduced aggregate optimizer-plumbing "
            "gate. Promotion still requires held-out surface or field-line evidence and "
            "separate nonlinear transport-window validation."
        ),
    }


def write_vmec_boozer_second_equilibrium_aggregate_artifacts(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_SECOND_EQUILIBRIUM_OUT,
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for the second-equilibrium gate."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fd = payload.get("finite_difference_summary", {})
    ls = payload.get("line_search_summary", {})
    mode_bound = payload.get("mode_bound", {})
    runtime = payload.get("bounded_runtime", {})
    sample_bound = payload.get("sample_bound", {})
    fd_dict = fd if isinstance(fd, dict) else {}
    ls_dict = ls if isinstance(ls, dict) else {}
    mode_dict = mode_bound if isinstance(mode_bound, dict) else {}
    runtime_dict = runtime if isinstance(runtime, dict) else {}
    sample_dict = sample_bound if isinstance(sample_bound, dict) else {}
    fieldnames = [
        "case_name",
        "passed",
        "feasible",
        "mboz",
        "nboz",
        "n_samples",
        "elapsed_wall_seconds",
        "fd_passed",
        "fd_base_value",
        "fd_central_derivative",
        "fd_response_abs",
        "fd_curvature_ratio",
        "line_search_passed",
        "accepted_steps",
        "initial_objective",
        "final_objective",
        "relative_reduction",
        "stop_reason",
        "blocker_type",
        "blocker_message",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "case_name": payload.get("case_name", ""),
                "passed": payload.get("passed", ""),
                "feasible": payload.get("feasible", ""),
                "mboz": mode_dict.get("mboz", ""),
                "nboz": mode_dict.get("nboz", ""),
                "n_samples": sample_dict.get(
                    "n_samples_requested", fd_dict.get("n_samples", "")
                ),
                "elapsed_wall_seconds": runtime_dict.get("elapsed_wall_seconds", ""),
                "fd_passed": payload.get("finite_difference_passed", ""),
                "fd_base_value": fd_dict.get("base_value", ""),
                "fd_central_derivative": fd_dict.get("central_derivative", ""),
                "fd_response_abs": fd_dict.get("response_abs", ""),
                "fd_curvature_ratio": fd_dict.get("curvature_ratio", ""),
                "line_search_passed": payload.get("line_search_passed", ""),
                "accepted_steps": ls_dict.get("accepted_steps", ""),
                "initial_objective": ls_dict.get("initial_objective", ""),
                "final_objective": ls_dict.get("final_objective", ""),
                "relative_reduction": ls_dict.get("relative_reduction", ""),
                "stop_reason": ls_dict.get("stop_reason", ""),
                "blocker_type": payload.get("blocker_type", ""),
                "blocker_message": payload.get("blocker_message", ""),
            }
        )

    set_plot_style()
    fig, (ax_values, ax_meta) = plt.subplots(
        1, 2, figsize=(12.0, 5.0), gridspec_kw={"width_ratios": [1.2, 1.0]}
    )
    passed = bool(payload.get("passed"))
    status = "passed" if passed else "blocked"
    if fd_dict:
        values = np.asarray(
            [
                float(fd_dict.get("minus_value", np.nan)),
                float(fd_dict.get("base_value", np.nan)),
                float(fd_dict.get("plus_value", np.nan)),
            ],
            dtype=float,
        )
        ax_values.bar(
            [0, 1, 2],
            values,
            color=["#90be6d", "#277da1", "#f3722c"],
            edgecolor="#202020",
            linewidth=0.5,
        )
        ax_values.set_xticks([0, 1, 2], ["x-h", "x", "x+h"])
        ax_values.set_ylabel(str(payload.get("objective", "objective")))
        ax_values.set_title("Aggregate finite-difference values")
        ax_values.grid(axis="y", alpha=0.25)
    else:
        ax_values.axis("off")
        ax_values.text(
            0.05,
            0.55,
            str(payload.get("blocker_message", "blocked")),
            va="center",
            ha="left",
            wrap=True,
        )
        ax_values.set_title("Fail-closed blocker")

    summary_lines = [
        f"status: {status}",
        f"case: {payload.get('case_name')}",
        f"mboz/nboz: {mode_dict.get('mboz')}/{mode_dict.get('nboz')}",
        f"samples: {sample_dict.get('n_samples_requested')}",
        f"elapsed: {float(runtime_dict.get('elapsed_wall_seconds', float('nan'))):.3g} s",
        f"FD passed: {payload.get('finite_difference_passed', False)}",
        f"line passed: {payload.get('line_search_passed', False)}",
        f"FD deriv: {float(fd_dict.get('central_derivative', float('nan'))):.6g}",
        f"curvature: {float(fd_dict.get('curvature_ratio', float('nan'))):.3e}",
        f"rel. red.: {ls_dict.get('relative_reduction')}",
    ]
    if not passed and payload.get("blocker_type"):
        summary_lines.extend(
            [
                f"blocker: {payload.get('blocker_type')}",
                str(payload.get("blocker_message")),
            ]
        )
    ax_meta.axis("off")
    ax_meta.set_title("Second-equilibrium aggregate gate")
    ax_meta.text(
        0.02,
        0.95,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.2,
        transform=ax_meta.transAxes,
    )
    ax_meta.text(
        0.02,
        0.13,
        "Reduced VMEC/Boozer aggregate FD and line-search\n"
        "evidence on a non-QH fixture. This checks bounded\n"
        "optimizer plumbing, not nonlinear transport validity.",
        va="top",
        ha="left",
        fontsize=8.3,
        transform=ax_meta.transAxes,
    )
    fig.suptitle(f"VMEC/Boozer second-equilibrium aggregate gate: {status}")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16, wspace=0.28)
    fig.savefig(out_path, dpi=220)
    fig.savefig(pdf_path)
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
    parser.add_argument("--case-name", default="nfp4_QH_warm_start")
    parser.add_argument("--objective", default="quasilinear_flux")
    parser.add_argument(
        "--reduction", choices=["mean", "weighted_mean", "max"], default="mean"
    )
    parser.add_argument("--surface-indices", nargs="*", type=int, default=[])
    parser.add_argument(
        "--torflux-values",
        nargs="*",
        type=float,
        default=[],
        help="Optional physical normalized toroidal-flux samples. Cannot be combined with --surface-indices.",
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--selected-ky-indices", nargs="+", type=int, default=[1, 2])
    parser.add_argument(
        "--ky-values",
        nargs="*",
        type=float,
        default=[],
        help="Optional physical ky*rho_i values. When set, selected indices, Ly, and Ny are inferred.",
    )
    parser.add_argument(
        "--ky-base",
        type=float,
        default=None,
        help="Base ky spacing for --ky-values; defaults to the smallest requested ky.",
    )
    parser.add_argument("--radial-index", type=int, default=None)
    parser.add_argument("--mode-index", type=int, default=1)
    parser.add_argument("--perturbation-step", type=float, default=1.0e-7)
    parser.add_argument("--max-curvature-ratio", type=float, default=5.0)
    parser.add_argument("--response-atol", type=float, default=0.0)
    parser.add_argument("--ntheta", type=int, default=4)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--surface-stencil-width", type=int, default=3)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--json-only", action="store_true")
    return parser


def build_line_search_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the multi-point VMEC/Boozer aggregate-objective line-search artifact."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_LINE_SEARCH_OUT)
    parser.add_argument("--case-name", default="nfp4_QH_warm_start")
    parser.add_argument("--objective", default="quasilinear_flux")
    parser.add_argument(
        "--reduction", choices=["mean", "weighted_mean", "max"], default="mean"
    )
    parser.add_argument("--surface-indices", nargs="*", type=int, default=[])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--selected-ky-indices", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--radial-index", type=int, default=None)
    parser.add_argument("--mode-index", type=int, default=1)
    parser.add_argument("--perturbation-step", type=float, default=1.0e-7)
    parser.add_argument("--update-step", type=float, default=1.0e-8)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--max-curvature-ratio", type=float, default=5.0)
    parser.add_argument("--response-atol", type=float, default=0.0)
    parser.add_argument("--ntheta", type=int, default=4)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--surface-stencil-width", type=int, default=3)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--json-only", action="store_true")
    return parser


def build_line_search_comparison_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare growth and quasilinear aggregate line searches."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_LINE_SEARCH_COMPARISON_OUT)
    parser.add_argument("--case-name", default="nfp4_QH_warm_start")
    parser.add_argument(
        "--objectives", nargs="+", default=list(DEFAULT_COMPARISON_OBJECTIVES)
    )
    parser.add_argument(
        "--reduction", choices=["mean", "weighted_mean", "max"], default="mean"
    )
    parser.add_argument("--surface-indices", nargs="*", type=int, default=[])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--selected-ky-indices", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--radial-index", type=int, default=None)
    parser.add_argument("--mode-index", type=int, default=1)
    parser.add_argument("--perturbation-step", type=float, default=1.0e-7)
    parser.add_argument("--update-step", type=float, default=1.0e-8)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--max-curvature-ratio", type=float, default=5.0)
    parser.add_argument("--response-atol", type=float, default=0.0)
    parser.add_argument("--ntheta", type=int, default=4)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--surface-stencil-width", type=int, default=3)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--json-only", action="store_true")
    return parser


def build_multi_point_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a bounded multi-alpha/surface aggregate-objective artifact."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_MULTI_POINT_OUT)
    parser.add_argument("--case-name", default="nfp4_QH_warm_start")
    parser.add_argument(
        "--objective",
        choices=[
            "growth",
            "gamma",
            "frequency",
            "omega",
            "kperp_eff2",
            "linear_heat_flux_weight",
            "linear_particle_flux_weight",
            "quasilinear_flux",
            "mixing_length_heat_flux_proxy",
        ],
        default="quasilinear_flux",
    )
    parser.add_argument(
        "--reduction", choices=["mean", "weighted_mean", "max"], default="mean"
    )
    parser.add_argument("--surface-indices", nargs="*", type=int, default=[])
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=list(DEFAULT_MULTI_POINT_ALPHAS)
    )
    parser.add_argument(
        "--selected-ky-indices",
        nargs="+",
        type=int,
        default=list(DEFAULT_MULTI_POINT_SELECTED_KY_INDICES),
    )
    parser.add_argument("--radial-index", type=int, default=None)
    parser.add_argument("--mode-index", type=int, default=1)
    parser.add_argument("--perturbation-step", type=float, default=1.0e-7)
    parser.add_argument("--max-curvature-ratio", type=float, default=5.0)
    parser.add_argument("--response-atol", type=float, default=0.0)
    parser.add_argument("--ntheta", type=int, default=4)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--surface-stencil-width", type=int, default=3)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument(
        "--max-samples", type=int, default=DEFAULT_MULTI_POINT_MAX_SAMPLES
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=DEFAULT_VMEC_BOOZER_MAX_WALL_SECONDS,
        help="Set <=0 to disable the Unix wall-clock timeout.",
    )
    parser.add_argument("--json-only", action="store_true")
    return parser


def build_second_equilibrium_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a second-equilibrium aggregate objective artifact."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_SECOND_EQUILIBRIUM_OUT)
    parser.add_argument("--case-name", default=DEFAULT_SECOND_EQUILIBRIUM_CASE_NAME)
    parser.add_argument("--objective", default="quasilinear_flux")
    parser.add_argument(
        "--reduction", choices=["mean", "weighted_mean", "max"], default="mean"
    )
    parser.add_argument("--surface-indices", nargs="*", type=int, default=[])
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=list(DEFAULT_SECOND_EQUILIBRIUM_ALPHAS),
    )
    parser.add_argument(
        "--selected-ky-indices",
        nargs="+",
        type=int,
        default=list(DEFAULT_SECOND_EQUILIBRIUM_SELECTED_KY_INDICES),
    )
    parser.add_argument("--radial-index", type=int, default=None)
    parser.add_argument("--mode-index", type=int, default=1)
    parser.add_argument("--perturbation-step", type=float, default=1.0e-7)
    parser.add_argument("--update-step", type=float, default=1.0e-8)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--max-curvature-ratio", type=float, default=5.0)
    parser.add_argument("--response-atol", type=float, default=0.0)
    parser.add_argument("--ntheta", type=int, default=4)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--surface-stencil-width", type=int, default=3)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument(
        "--max-samples", type=int, default=DEFAULT_SECOND_EQUILIBRIUM_MAX_SAMPLES
    )
    parser.add_argument(
        "--max-wall-seconds", type=float, default=DEFAULT_VMEC_BOOZER_MAX_WALL_SECONDS
    )
    parser.add_argument("--json-only", action="store_true")
    return parser


def _annotate_physical_ky_samples(
    payload: dict[str, object],
    *,
    requested_ky_values: list[float],
    solver_grid_options: dict[str, object],
) -> None:
    index_to_requested = {
        int(index): float(ky)
        for index, ky in zip(
            solver_grid_options["selected_ky_indices"],
            requested_ky_values,
            strict=True,
        )
    }
    index_to_resolved = {
        int(index): float(ky)
        for index, ky in zip(
            solver_grid_options["selected_ky_indices"],
            solver_grid_options["resolved_ky_values"],
            strict=True,
        )
    }
    rows = payload.get("samples")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        selected = int(row.get("selected_ky_index", 0))
        if selected not in index_to_requested:
            continue
        requested = index_to_requested[selected]
        resolved = index_to_resolved[selected]
        row["ky"] = requested
        row["selected_ky"] = resolved
        row["ky_abs_error"] = abs(resolved - requested)


def _run_line_search(argv: list[str]) -> int:
    args = build_line_search_parser().parse_args(argv)
    payload = vmec_boozer_aggregate_scalar_objective_line_search_report(
        case_name=args.case_name,
        objective=args.objective,
        reduction=args.reduction,
        surface_indices=_surface_indices(args.surface_indices),
        alphas=tuple(args.alphas),
        selected_ky_indices=tuple(args.selected_ky_indices),
        radial_index=args.radial_index,
        mode_index=args.mode_index,
        perturbation_step=args.perturbation_step,
        update_step=args.update_step,
        max_steps=args.max_steps,
        min_improvement=args.min_improvement,
        response_atol=args.response_atol,
        max_curvature_ratio=args.max_curvature_ratio,
        ntheta=args.ntheta,
        mboz=args.mboz,
        nboz=args.nboz,
        surface_stencil_width=None
        if args.surface_stencil_width <= 0
        else args.surface_stencil_width,
        n_laguerre=args.n_laguerre,
        n_hermite=args.n_hermite,
        nx=args.nx,
        ny=args.ny,
    )
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_vmec_boozer_aggregate_line_search_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


def _run_line_search_comparison(argv: list[str]) -> int:
    args = build_line_search_comparison_parser().parse_args(argv)
    payload = build_vmec_boozer_aggregate_line_search_comparison_report(
        objectives=tuple(args.objectives),
        case_name=args.case_name,
        reduction=args.reduction,
        surface_indices=_surface_indices(args.surface_indices),
        alphas=tuple(args.alphas),
        selected_ky_indices=tuple(args.selected_ky_indices),
        radial_index=args.radial_index,
        mode_index=args.mode_index,
        perturbation_step=args.perturbation_step,
        update_step=args.update_step,
        max_steps=args.max_steps,
        min_improvement=args.min_improvement,
        response_atol=args.response_atol,
        max_curvature_ratio=args.max_curvature_ratio,
        ntheta=args.ntheta,
        mboz=args.mboz,
        nboz=args.nboz,
        surface_stencil_width=None
        if args.surface_stencil_width <= 0
        else args.surface_stencil_width,
        n_laguerre=args.n_laguerre,
        n_hermite=args.n_hermite,
        nx=args.nx,
        ny=args.ny,
    )
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_vmec_boozer_aggregate_line_search_comparison_artifacts(
        payload, out=args.out
    )
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


def _run_multi_point(argv: list[str]) -> int:
    parser = build_multi_point_parser()
    args = parser.parse_args(argv)
    try:
        payload = build_vmec_boozer_multi_point_objective_payload(
            case_name=args.case_name,
            objective=args.objective,
            reduction=args.reduction,
            surface_indices=_surface_indices(args.surface_indices),
            alphas=tuple(args.alphas),
            selected_ky_indices=tuple(args.selected_ky_indices),
            radial_index=args.radial_index,
            mode_index=args.mode_index,
            perturbation_step=args.perturbation_step,
            response_atol=args.response_atol,
            max_curvature_ratio=args.max_curvature_ratio,
            ntheta=args.ntheta,
            mboz=args.mboz,
            nboz=args.nboz,
            surface_stencil_width=None
            if args.surface_stencil_width <= 0
            else args.surface_stencil_width,
            n_laguerre=args.n_laguerre,
            n_hermite=args.n_hermite,
            nx=args.nx,
            ny=args.ny,
            max_samples=args.max_samples,
            max_wall_seconds=args.max_wall_seconds,
        )
    except (TimeoutError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_vmec_boozer_multi_point_objective_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


def _run_second_equilibrium(argv: list[str]) -> int:
    args = build_second_equilibrium_parser().parse_args(argv)
    payload = build_vmec_boozer_second_equilibrium_aggregate_payload(
        case_name=args.case_name,
        objective=args.objective,
        reduction=args.reduction,
        surface_indices=_surface_indices(args.surface_indices),
        alphas=tuple(args.alphas),
        selected_ky_indices=tuple(args.selected_ky_indices),
        radial_index=args.radial_index,
        mode_index=args.mode_index,
        perturbation_step=args.perturbation_step,
        update_step=args.update_step,
        max_steps=args.max_steps,
        min_improvement=args.min_improvement,
        response_atol=args.response_atol,
        max_curvature_ratio=args.max_curvature_ratio,
        ntheta=args.ntheta,
        mboz=args.mboz,
        nboz=args.nboz,
        surface_stencil_width=None
        if args.surface_stencil_width <= 0
        else args.surface_stencil_width,
        n_laguerre=args.n_laguerre,
        n_hermite=args.n_hermite,
        nx=args.nx,
        ny=args.ny,
        max_samples=args.max_samples,
        max_wall_seconds=args.max_wall_seconds,
    )
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0 if bool(payload.get("passed")) else 1
    paths = write_vmec_boozer_second_equilibrium_aggregate_artifacts(
        payload, out=args.out
    )
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0 if bool(payload.get("passed")) else 1


def _run_finite_difference(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.torflux_values and args.surface_indices:
        raise ValueError("use --torflux-values or --surface-indices, not both")
    selected_ky_indices = tuple(args.selected_ky_indices)
    solver_grid_options: dict[str, object] = {}
    objective_kwargs: dict[str, object] = {
        "ntheta": args.ntheta,
        "mboz": args.mboz,
        "nboz": args.nboz,
        "surface_stencil_width": None
        if args.surface_stencil_width <= 0
        else args.surface_stencil_width,
        "n_laguerre": args.n_laguerre,
        "n_hermite": args.n_hermite,
        "nx": args.nx,
        "ny": args.ny,
    }
    if args.ky_values:
        solver_grid_options = solver_grid_options_from_ky_values(
            tuple(args.ky_values),
            ky_base=args.ky_base,
            min_ny=args.ny,
        )
        selected_ky_indices = tuple(
            int(item) for item in solver_grid_options["selected_ky_indices"]
        )
        objective_kwargs["ny"] = int(solver_grid_options["ny"])
        objective_kwargs["ly"] = float(solver_grid_options["ly"])

    payload = vmec_boozer_aggregate_scalar_objective_finite_difference_report(
        case_name=args.case_name,
        objective=args.objective,
        reduction=args.reduction,
        surface_indices=(None,)
        if args.torflux_values
        else _surface_indices(args.surface_indices),
        torflux_values=tuple(args.torflux_values) if args.torflux_values else None,
        alphas=tuple(args.alphas),
        selected_ky_indices=selected_ky_indices,
        radial_index=args.radial_index,
        mode_index=args.mode_index,
        perturbation_step=args.perturbation_step,
        response_atol=args.response_atol,
        max_curvature_ratio=args.max_curvature_ratio,
        **objective_kwargs,
    )
    if solver_grid_options:
        requested_ky_values = [float(item) for item in args.ky_values]
        _annotate_physical_ky_samples(
            payload,
            requested_ky_values=requested_ky_values,
            solver_grid_options=solver_grid_options,
        )
        payload["requested_ky_values"] = requested_ky_values
        payload["ky_values"] = requested_ky_values
        payload["solver_grid_options_from_ky_values"] = solver_grid_options
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_vmec_boozer_aggregate_objective_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else list(argv)
    if raw and raw[0] == "line-search":
        return _run_line_search(raw[1:])
    if raw and raw[0] == "line-search-comparison":
        return _run_line_search_comparison(raw[1:])
    if raw and raw[0] == "multi-point":
        return _run_multi_point(raw[1:])
    if raw and raw[0] == "second-equilibrium":
        return _run_second_equilibrium(raw[1:])
    return _run_finite_difference(raw)


if __name__ == "__main__":
    raise SystemExit(main())
