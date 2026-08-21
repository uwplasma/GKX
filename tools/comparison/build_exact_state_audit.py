#!/usr/bin/env python3
"""Run and report exact-state comparison audits for multiple physics lanes.

This runner is intentionally file-path driven (manifest-based) so it can be
used on local laptops and remote machines without hardcoding office-only paths
into the repo.

It wraps the startup, diagnostic-state, and window subcommands in
``tools/comparison/compare_runtime.py``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gkx.artifacts.plotting import set_plot_style  # noqa: E402
from gkx.workflows.runtime.toml import load_toml


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_TOOL_DIR = REPO_ROOT / "tools" / "comparison"


def build_run_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="TOML manifest describing each exact-state lane.",
    )
    p.add_argument(
        "--lane", type=str, default=None, help="Optional single lane key to run."
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("tools_out") / "exact_state_audit",
        help="Output directory root.",
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use for subprocess tool calls.",
    )
    return p


def _tool_env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    entries = [str((repo_root / "src").resolve()), str(repo_root.resolve())]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _comparison_tool_path(name: str) -> Path:
    """Return one checked comparison command from the repository tool layout."""

    path = COMPARISON_TOOL_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"comparison tool does not exist: {path}")
    return path


def _run_tool(
    cmd: list[str],
    *,
    cwd: Path | None,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    env_lines: list[str] = []
    if env is not None:
        for key in ("PYTHONPATH", "W7X_VMEC_FILE", "HSX_VMEC_FILE", "JAX_PLATFORMS"):
            if key in env:
                env_lines.append(f"{key}={env[key]}")
    log_path.write_text(
        f"$ {' '.join(cmd)}\n"
        + ("env:\n" + "\n".join(env_lines) + "\n\n" if env_lines else "\n")
        + f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}\n",
        encoding="utf-8",
    )
    return {"returncode": int(proc.returncode), "log": str(log_path)}


@contextmanager
def _temporary_env(env_updates: dict[str, str] | None):
    if not env_updates:
        yield
        return
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in env_updates}
    try:
        for key, value in env_updates.items():
            os.environ[str(key)] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _lane_section(manifest: dict[str, Any]) -> dict[str, Any]:
    lanes = manifest.get("lane")
    if not isinstance(lanes, dict):
        raise ValueError("Manifest must contain a [lane.<name>] table per lane.")
    return lanes


def _resolve_manifest_path(value: str | Path, *, manifest_dir: Path) -> Path:
    raw = os.path.expandvars(str(value))
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _expand_env_value(value: object) -> str:
    text = str(value)
    for _ in range(4):
        expanded = os.path.expanduser(os.path.expandvars(text))
        if expanded == text:
            return expanded
        text = expanded
    return text


def _main_run(argv: list[str]) -> int:
    args = build_run_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest_dir = manifest_path.parent
    manifest = load_toml(manifest_path)
    lanes = _lane_section(manifest)

    selected = [args.lane] if args.lane is not None else list(lanes.keys())
    out_root = args.outdir.expanduser().resolve()
    py = str(args.python)
    summary: dict[str, Any] = {"manifest": str(manifest_path), "lanes": {}}

    for lane_key in selected:
        if lane_key not in lanes:
            raise SystemExit(f"Lane not found in manifest: {lane_key}")
        cfg = lanes[lane_key]
        if not isinstance(cfg, dict):
            raise SystemExit(f"Lane config must be a table: lane.{lane_key}")

        gx_out = _resolve_manifest_path(cfg["gx_out"], manifest_dir=manifest_dir)
        config = _resolve_manifest_path(cfg["config"], manifest_dir=manifest_dir)
        out_dir = out_root / lane_key
        out_dir.mkdir(parents=True, exist_ok=True)

        lane_summary: dict[str, Any] = {"gx_out": str(gx_out), "config": str(config)}
        env_cfg = cfg.get("env")
        env_updates = (
            {str(key): _expand_env_value(value) for key, value in env_cfg.items()}
            if isinstance(env_cfg, dict)
            else None
        )
        if env_updates:
            lane_summary["env"] = env_updates

        with _temporary_env(env_updates):
            tool_env = _tool_env(REPO_ROOT)
            if "startup" in cfg:
                st = cfg["startup"]
                cmd = [
                    py,
                    str(_comparison_tool_path("compare_runtime.py")),
                    "startup",
                    "--gx-dir",
                    str(
                        _resolve_manifest_path(st["gx_dir"], manifest_dir=manifest_dir)
                    ),
                    "--gx-out",
                    str(gx_out),
                    "--config",
                    str(config),
                    "--ky",
                    str(st["ky"]),
                ]
                if "kx_target" in st:
                    cmd += ["--kx-target", str(st["kx_target"])]
                if "y0" in st:
                    cmd += ["--y0", str(st["y0"])]
                lane_summary["startup"] = _run_tool(
                    cmd,
                    cwd=COMPARISON_TOOL_DIR,
                    log_path=out_dir / "startup.log",
                    env=tool_env,
                )

            if "diag_state" in cfg:
                ds = cfg["diag_state"]
                cmd = [
                    py,
                    str(_comparison_tool_path("compare_runtime.py")),
                    "diagnostic-state",
                    "--gx-dir",
                    str(
                        _resolve_manifest_path(ds["gx_dir"], manifest_dir=manifest_dir)
                    ),
                    "--gx-out",
                    str(gx_out),
                    "--config",
                    str(config),
                    "--time-index",
                    str(ds["time_index"]),
                    "--out",
                    str(out_dir / "diag_state.csv"),
                ]
                if "y0" in ds:
                    cmd += ["--y0", str(ds["y0"])]
                lane_summary["diag_state"] = _run_tool(
                    cmd,
                    cwd=COMPARISON_TOOL_DIR,
                    log_path=out_dir / "diag_state.log",
                    env=tool_env,
                )

            if "window" in cfg:
                w = cfg["window"]
                cmd = [
                    py,
                    str(_comparison_tool_path("compare_runtime.py")),
                    "window",
                    "--gx-dir",
                    str(_resolve_manifest_path(w["gx_dir"], manifest_dir=manifest_dir)),
                    "--gx-out",
                    str(gx_out),
                    "--config",
                    str(config),
                    "--time-index-start",
                    str(w["time_index_start"]),
                    "--time-index-stop",
                    str(w["time_index_stop"]),
                    "--out",
                    str(out_dir / "window.csv"),
                ]
                if "steps" in w:
                    cmd += ["--steps", str(w["steps"])]
                if "ky" in w:
                    cmd += ["--ky", str(w["ky"])]
                if "y0" in w:
                    cmd += ["--y0", str(w["y0"])]
                lane_summary["window"] = _run_tool(
                    cmd,
                    cwd=COMPARISON_TOOL_DIR,
                    log_path=out_dir / "window.log",
                    env=tool_env,
                )

        summary["lanes"][lane_key] = lane_summary

    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"saved {summary_path}")
    return 0


DEFAULT_AUDIT_DIR = (
    REPO_ROOT / "tools_out" / "exact_state_audit_w7x_20260424" / "w7x_vmec"
)
DEFAULT_OUT = REPO_ROOT / "docs" / "_static" / "w7x_exact_state_audit.png"
REL_FLOOR = 1.0e-12
PASS_THRESHOLD = 1.0e-4

SUMMARY_RE = re.compile(
    r"^(?P<metric>\S+)\s+"
    r"max\|ref\|=(?P<max_ref>\S+)\s+"
    r"max\|test\|=(?P<max_test>\S+)\s+"
    r"max\|diff\|=(?P<max_diff>\S+)\s+"
    r"max\|rel\|=(?P<max_rel>\S+)\s+"
    r"rms_rel=(?P<rms_rel>\S+)"
)


def build_report_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--pass-threshold", type=float, default=PASS_THRESHOLD)
    return parser


def _repo_relative(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _as_float(value: str | float | int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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


def parse_array_summary_log(path: Path, *, phase: str) -> list[dict[str, object]]:
    """Parse ``_summary`` lines from an exact-state audit log."""

    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = SUMMARY_RE.match(line.strip())
        if match is None:
            continue
        data = match.groupdict()
        rows.append(
            {
                "phase": phase,
                "kind": "array",
                "metric": data["metric"],
                "quantity": "max_rel",
                "value": _as_float(data["max_rel"]),
                "max_ref": _as_float(data["max_ref"]),
                "max_test": _as_float(data["max_test"]),
                "max_diff": _as_float(data["max_diff"]),
                "rms_rel": _as_float(data["rms_rel"]),
                "source_path": _repo_relative(path),
            }
        )
    return rows


def parse_diagnostic_csv(path: Path) -> list[dict[str, object]]:
    """Load scalar diagnostic relative errors from ``diag_state.csv``."""

    table = pd.read_csv(path)
    required = {
        "metric",
        "gx_out",
        "gkx_dump",
        "rel_dump",
        "gkx_solve",
        "rel_solve",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for item in table.to_dict(orient="records"):
        for quantity in ("rel_dump", "rel_solve"):
            rows.append(
                {
                    "phase": "late diagnostics",
                    "kind": "diagnostic",
                    "metric": str(item["metric"]),
                    "quantity": quantity,
                    "value": float(item[quantity]),
                    "max_ref": float(item["gx_out"]),
                    "max_test": float(
                        item[
                            "gkx_dump"
                            if quantity == "rel_dump"
                            else "gkx_solve"
                        ]
                    ),
                    "max_diff": abs(
                        float(item["gx_out"])
                        - float(
                            item[
                                "gkx_dump"
                                if quantity == "rel_dump"
                                else "gkx_solve"
                            ]
                        )
                    ),
                    "rms_rel": float("nan"),
                    "source_path": _repo_relative(path),
                }
            )
    return rows


def build_rows(audit_dir: Path) -> list[dict[str, object]]:
    """Build the long-form exact-state audit table from an audit directory."""

    startup_log = audit_dir / "startup.log"
    diag_log = audit_dir / "diag_state.log"
    diag_csv = audit_dir / "diag_state.csv"
    missing = [path for path in (startup_log, diag_log, diag_csv) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing exact-state audit files: {', '.join(str(path) for path in missing)}"
        )
    rows = []
    rows.extend(parse_array_summary_log(startup_log, phase="startup"))
    rows.extend(parse_array_summary_log(diag_log, phase="late arrays"))
    rows.extend(parse_diagnostic_csv(diag_csv))
    return rows


def _finite_plot_value(value: object) -> float | None:
    value_f = float(value)
    if not np.isfinite(value_f):
        return None
    return max(value_f, REL_FLOOR)


def _bar_panel(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    *,
    title: str,
    threshold: float,
    color: str,
    quantity: str = "max_rel",
) -> None:
    filtered = [
        row
        for row in rows
        if row["quantity"] == quantity and _finite_plot_value(row["value"]) is not None
    ]
    labels = [str(row["metric"]) for row in filtered]
    values = [_finite_plot_value(row["value"]) or REL_FLOOR for row in filtered]
    if not filtered:
        ax.text(
            0.5, 0.5, "no finite rows", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_axis_off()
        return
    xpos = np.arange(len(filtered))
    ax.bar(xpos, values, color=color, alpha=0.86)
    ax.axhline(
        threshold,
        color="#991b1b",
        linestyle="--",
        linewidth=1.2,
        label=f"{threshold:.0e} gate",
    )
    ax.set_yscale("log")
    ax.set_ylim(REL_FLOOR * 0.8, max(max(values) * 6.0, threshold * 4.0))
    ax.set_xticks(xpos, labels, rotation=35, ha="right")
    ax.set_ylabel("relative error")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    for xval, row, yval in zip(xpos, filtered, values, strict=True):
        if float(row["value"]) == 0.0:
            ax.text(xval, REL_FLOOR * 1.8, "0", ha="center", va="bottom", fontsize=8)


def exact_state_figure(
    rows: list[dict[str, object]], *, threshold: float
) -> plt.Figure:
    """Create the W7-X exact-state convention audit figure."""

    set_plot_style()
    startup = [
        row for row in rows if row["phase"] == "startup" and row["kind"] == "array"
    ]
    late_arrays = [
        row
        for row in rows
        if row["phase"] == "late arrays"
        and row["kind"] == "array"
        and str(row["metric"]) not in {"apar", "bpar"}
    ]
    diagnostics = [row for row in rows if row["kind"] == "diagnostic"]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), constrained_layout=True)
    _bar_panel(
        axes[0], startup, title="Startup state", threshold=threshold, color="#0f4c81"
    )
    _bar_panel(
        axes[1],
        late_arrays,
        title="Late dumped arrays",
        threshold=threshold,
        color="#2a9d8f",
    )

    diag_table = pd.DataFrame(diagnostics)
    diag_metrics = (
        list(dict.fromkeys(diag_table["metric"].astype(str)))
        if not diag_table.empty
        else []
    )
    xpos = np.arange(len(diag_metrics))
    width = 0.36
    for offset, quantity, color, label in (
        (-0.5 * width, "rel_dump", "#7b2cbf", "dumped fields"),
        (0.5 * width, "rel_solve", "#c2410c", "re-solved fields"),
    ):
        values = []
        raw_values = []
        for metric in diag_metrics:
            subset = diag_table[
                (diag_table["metric"] == metric) & (diag_table["quantity"] == quantity)
            ]
            value = float(subset["value"].iloc[0]) if not subset.empty else float("nan")
            raw_values.append(value)
            values.append(max(value, REL_FLOOR) if np.isfinite(value) else REL_FLOOR)
        axes[2].bar(
            xpos + offset, values, width=width, color=color, alpha=0.86, label=label
        )
        for xval, raw, plotted in zip(xpos + offset, raw_values, values, strict=True):
            if np.isfinite(raw) and raw == 0.0:
                axes[2].text(
                    xval, REL_FLOOR * 1.8, "0", ha="center", va="bottom", fontsize=8
                )
    axes[2].axhline(
        threshold,
        color="#991b1b",
        linestyle="--",
        linewidth=1.2,
        label=f"{threshold:.0e} gate",
    )
    axes[2].set_yscale("log")
    axes[2].set_ylim(REL_FLOOR * 0.8, threshold * 4.0)
    axes[2].set_xticks(xpos, diag_metrics, rotation=35, ha="right")
    axes[2].set_ylabel("relative error")
    axes[2].set_title("Late scalar diagnostics")
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "W7-X nonlinear exact-state convention audit",
        y=1.03,
        fontsize=14,
        fontweight="bold",
    )
    return fig


def write_outputs(
    rows: list[dict[str, object]],
    *,
    audit_dir: Path,
    out_png: Path,
    out_csv: Path,
    out_json: Path,
    threshold: float,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    finite = [float(row["value"]) for row in rows if np.isfinite(float(row["value"]))]
    max_rel = float(max(finite)) if finite else float("nan")
    payload = {
        "case": "w7x_nonlinear_exact_state_audit",
        "validation_status": "closed"
        if np.isfinite(max_rel) and max_rel <= threshold
        else "open",
        "gate_index_include": False,
        "reference": "GX W7-X nonlinear VMEC exact-state startup and late diagnostic dumps",
        "audit_dir": _repo_relative(audit_dir),
        "audit_csv": _repo_relative(out_csv),
        "audit_png": _repo_relative(out_png),
        "pass_threshold": float(threshold),
        "max_finite_relative_error": max_rel,
        "rows": rows,
        "notes": (
            "This audit checks state/geometry/fieldsolve/diagnostic conventions on exact GX W7-X nonlinear states. "
            "It closes those convention layers but does not close the separate W7-X zonal-response recurrence "
            "and damping-envelope literature lane."
        ),
    }
    out_json.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _main_report(argv: list[str]) -> int:
    args = build_report_parser().parse_args(argv)
    rows = build_rows(args.audit_dir)
    out_csv = args.out_csv or args.out_png.with_suffix(".csv")
    out_json = args.out_json or args.out_png.with_suffix(".json")
    fig = exact_state_figure(rows, threshold=float(args.pass_threshold))
    fig.savefig(args.out_png, dpi=240, bbox_inches="tight")
    fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    write_outputs(
        rows,
        audit_dir=args.audit_dir,
        out_png=args.out_png,
        out_csv=out_csv,
        out_json=out_json,
        threshold=float(args.pass_threshold),
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {args.out_png}")
    return 0


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        print("usage: build_exact_state_audit.py {run,report} ...")
        return 2
    command, rest = tokens[0], tokens[1:]
    if command == "run":
        return _main_run(rest)
    if command == "report":
        return _main_report(rest)
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
