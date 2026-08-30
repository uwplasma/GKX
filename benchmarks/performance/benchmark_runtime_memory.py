#!/usr/bin/env python3
"""Run manifest-driven runtime and memory benchmarks and plot grouped bars."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import glob
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "tools" / "runtime_memory_manifest.toml"
DEFAULT_CSV = ROOT / "tools_out" / "runtime_memory_results.csv"
DEFAULT_PNG = ROOT / "docs" / "_static" / "runtime_memory_benchmark.png"
DEFAULT_PDF = ROOT / "docs" / "_static" / "runtime_memory_benchmark.pdf"
DEFAULT_SUMMARY = ROOT / "tools_out" / "runtime_memory_summary.json"
DEFAULT_LOG_DIR = ROOT / "tools_out" / "runtime_memory_logs"
BACKEND_ORDER = ("gkx_cpu", "gkx_gpu", "gx")
BACKEND_LABELS = {
    "gkx_cpu": "GKX CPU",
    "gkx_gpu": "GKX GPU",
    "gx": "GX",
}
BACKEND_COLORS = {
    "gkx_cpu": "#1f4e79",
    "gkx_gpu": "#2e8b57",
    "gx": "#b35c1e",
}
CASE_ORDER = (
    "cyclone-linear",
    "cyclone-nonlinear",
    "etg-linear",
    "kbm-linear",
    "kbm-nonlinear",
    "w7x-linear",
    "w7x-nonlinear",
    "hsx-linear",
    "hsx-nonlinear",
    "miller-nonlinear",
)


@dataclass(frozen=True)
class RuntimeBenchRun:
    case: str
    label: str
    backend: str
    command: str
    cwd: str
    host: str | None = None
    profile_command: str | None = None
    enabled: bool = True
    wrap_time: bool = True
    profile_wrap_time: bool = False


def _resolve(path: str | Path) -> Path:
    p = Path(str(path)).expanduser()
    return p if p.is_absolute() else ROOT / p


def _render(text: str) -> str:
    return os.path.expandvars(text.replace("{root}", str(ROOT)))


def _load_manifest(path: Path) -> list[RuntimeBenchRun]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    runs: list[RuntimeBenchRun] = []
    for item in data.get("run", []):
        runs.append(
            RuntimeBenchRun(
                case=str(item["case"]),
                label=str(item.get("label", item["case"])),
                backend=str(item["backend"]),
                command=str(item["command"]),
                cwd=str(item.get("cwd", "{root}")),
                host=None if item.get("host") in (None, "") else str(item.get("host")),
                profile_command=None
                if item.get("profile_command") in (None, "")
                else str(item.get("profile_command")),
                enabled=bool(item.get("enabled", True)),
                wrap_time=bool(item.get("wrap_time", True)),
                profile_wrap_time=bool(item.get("profile_wrap_time", False)),
            )
        )
    return runs


def _select_runs(
    runs: list[RuntimeBenchRun], cases: set[str] | None, backends: set[str] | None
) -> list[RuntimeBenchRun]:
    out = [run for run in runs if run.enabled]
    if cases:
        out = [run for run in out if run.case in cases]
    if backends:
        out = [run for run in out if run.backend in backends]
    return out


def _time_wrapper_prefix(system_name: str | None = None) -> list[str]:
    system = (system_name or platform.system()).lower()
    if system == "darwin":
        return ["/usr/bin/time", "-l"]
    return ["/usr/bin/time", "-v"]


def _parse_peak_rss_mb(text: str) -> float | None:
    byte_patterns = [
        r"(?mi)^[ \t]*(\d+)[ \t]+maximum resident set size$",
        r"(?mi)^[ \t]*peak memory footprint:\s*(\d+)\s*$",
    ]
    for pattern in byte_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)) / (1024.0 * 1024.0)

    kb_patterns = [
        r"(?mi)^[ \t]*maximum resident set size\s*=\s*(\d+)\s*$",
        r"(?mi)^[ \t]*Maximum resident set size \(kbytes\):\s*(\d+)\s*$",
        r"(?mi)^[ \t]*maxresident\)k\s*=\s*(\d+)\s*$",
    ]
    for pattern in kb_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)) / 1024.0
    return None


def _parse_profile_times(text: str) -> dict[str, float]:
    fields = ("warmup_time_s", "run_time_s")
    out: dict[str, float] = {}
    for field in fields:
        match = re.search(
            rf"(?mi)(?:^|\s){re.escape(field)}=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            text,
        )
        if match:
            out[field] = float(match.group(1))
    return out


def _write_row_logs(log_dir: Path, row: dict[str, object]) -> dict[str, str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{row['case']}__{row['backend']}"
    stdout_path = log_dir / f"{stem}.stdout.log"
    stderr_path = log_dir / f"{stem}.stderr.log"
    stdout_path.write_text(str(row.get("stdout") or ""), encoding="utf-8")
    stderr_path.write_text(str(row.get("stderr") or ""), encoding="utf-8")
    return {"stdout_log": str(stdout_path), "stderr_log": str(stderr_path)}


def _log_fingerprint(text: str) -> dict[str, object]:
    """Return compact provenance for a captured process stream."""

    encoded = text.encode("utf-8")
    return {
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest() if encoded else "",
    }


def _summary_row(row: dict[str, object]) -> dict[str, object]:
    """Drop bulky process logs from tracked summaries while preserving provenance."""

    out = dict(row)
    for field in ("stdout", "stderr"):
        value = str(out.pop(field, "") or "")
        fingerprint = _log_fingerprint(value)
        out[f"{field}_bytes"] = fingerprint["bytes"]
        out[f"{field}_sha256"] = fingerprint["sha256"]
    return out


def _execute_command(
    *,
    host: str | None,
    cwd: str,
    command: str,
    wrap_time: bool,
) -> subprocess.CompletedProcess[str]:
    rendered_host = _render(host) if host else None
    rendered_cwd = _render(cwd)
    rendered_command = _render(command)
    if rendered_host:
        shell_cmd = rendered_command
        if wrap_time:
            prefix = " ".join(shlex.quote(x) for x in _time_wrapper_prefix("linux"))
            shell_cmd = f"{prefix} /bin/sh -lc {shlex.quote(rendered_command)}"
        remote_cmd = f"cd {shlex.quote(rendered_cwd)} && {shell_cmd}"
        return subprocess.run(
            ["ssh", "-x", rendered_host, remote_cmd],
            capture_output=True,
            text=True,
        )

    resolved_cwd = _resolve(rendered_cwd)
    shell_cmd = rendered_command
    if wrap_time:
        prefix = " ".join(shlex.quote(x) for x in _time_wrapper_prefix())
        shell_cmd = f"{prefix} /bin/sh -lc {shlex.quote(rendered_command)}"
    return subprocess.run(
        shell_cmd,
        shell=True,
        cwd=resolved_cwd,
        capture_output=True,
        text=True,
    )


def _run_command(run: RuntimeBenchRun) -> dict[str, object]:
    rendered_host = _render(run.host) if run.host else None
    rendered_cwd = _render(run.cwd)
    rendered_command = _render(run.command)
    start = time.perf_counter()
    proc = _execute_command(
        host=run.host, cwd=run.cwd, command=run.command, wrap_time=run.wrap_time
    )
    elapsed = time.perf_counter() - start
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    peak_rss_mb = _parse_peak_rss_mb(combined)
    row = {
        "case": run.case,
        "label": run.label,
        "backend": run.backend,
        "command": rendered_command,
        "cwd": rendered_cwd if rendered_host else str(_resolve(rendered_cwd)),
        "host": rendered_host or "",
        "wrap_time": run.wrap_time,
        "status": "success" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "runtime_s": elapsed,
        "peak_rss_mb": peak_rss_mb,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    row.update(_parse_profile_times(combined))

    if proc.returncode == 0 and run.profile_command:
        profile_proc = _execute_command(
            host=run.host,
            cwd=run.cwd,
            command=run.profile_command,
            wrap_time=run.profile_wrap_time,
        )
        profile_combined = (
            (profile_proc.stdout or "") + "\n" + (profile_proc.stderr or "")
        )
        row["stdout"] = (
            f"{proc.stdout}\n--- profile stdout ---\n{profile_proc.stdout}"
            if (proc.stdout or profile_proc.stdout)
            else ""
        )
        row["stderr"] = (
            f"{proc.stderr}\n--- profile stderr ---\n{profile_proc.stderr}"
            if (proc.stderr or profile_proc.stderr)
            else ""
        )
        row.update(_parse_profile_times(profile_combined))
        if profile_proc.returncode != 0:
            row["status"] = "failed"
            row["returncode"] = profile_proc.returncode
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "label",
        "backend",
        "status",
        "returncode",
        "runtime_s",
        "warmup_time_s",
        "run_time_s",
        "peak_rss_mb",
        "host",
        "cwd",
        "command",
        "stdout_log",
        "stderr_log",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"rows": [_summary_row(row) for row in rows]}, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_summary_rows(patterns: list[str]) -> list[dict[str, object]]:
    rows_by_key: dict[tuple[str, str], dict[str, object]] = {}
    seen: set[str] = set()
    for pattern in patterns:
        for match in sorted(glob.glob(str(_resolve(pattern)))):
            if match in seen:
                continue
            seen.add(match)
            data = json.loads(Path(match).read_text(encoding="utf-8"))
            file_rows = data.get("rows", [])
            if not isinstance(file_rows, list):
                raise ValueError(f"{match} does not contain a 'rows' list")
            for row in file_rows:
                row_dict = dict(row)
                key = (str(row_dict.get("case", "")), str(row_dict.get("backend", "")))
                rows_by_key[key] = row_dict
    return list(rows_by_key.values())


def _plot_results(csv_path: Path, png_path: Path, pdf_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pandas as pd

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "success"].copy()
    if ok.empty:
        raise ValueError("no successful runtime rows available for plotting")

    preferred = {case: idx for idx, case in enumerate(CASE_ORDER)}
    order = []
    labels = {}
    for _, row in ok.iterrows():
        if row["case"] not in labels:
            labels[row["case"]] = row["label"]
            order.append(row["case"])
    order.sort(key=lambda case: (preferred.get(case, len(preferred)), labels[case]))

    x = list(range(len(order)))
    width = 0.24
    fig, axes = plt.subplots(
        1, 2, figsize=(max(15.5, 1.45 * len(order) + 6.5), 7.4), constrained_layout=True
    )
    warm_handles: list[Line2D] = []

    for idx, backend in enumerate(BACKEND_ORDER):
        sub = ok[ok["backend"] == backend].set_index("case")
        runtime_vals = [
            float(sub.loc[case, "runtime_s"]) if case in sub.index else float("nan")
            for case in order
        ]
        memory_vals = [
            float(sub.loc[case, "peak_rss_mb"])
            if case in sub.index and not pd.isna(sub.loc[case, "peak_rss_mb"])
            else float("nan")
            for case in order
        ]
        offset = (idx - 1) * width
        axes[0].bar(
            [v + offset for v in x],
            runtime_vals,
            width=width,
            color=BACKEND_COLORS[backend],
            label=BACKEND_LABELS[backend],
        )
        axes[1].bar(
            [v + offset for v in x],
            memory_vals,
            width=width,
            color=BACKEND_COLORS[backend],
            label=BACKEND_LABELS[backend],
        )
        if "run_time_s" in sub.columns:
            warm_vals = [
                float(sub.loc[case, "run_time_s"])
                if case in sub.index and not pd.isna(sub.loc[case, "run_time_s"])
                else float("nan")
                for case in order
            ]
            warm_x = [v + offset for v, warm in zip(x, warm_vals) if not pd.isna(warm)]
            warm_y = [warm for warm in warm_vals if not pd.isna(warm)]
            if warm_x:
                axes[0].scatter(
                    warm_x,
                    warm_y,
                    marker="D",
                    s=42,
                    facecolors="white",
                    edgecolors=BACKEND_COLORS[backend],
                    linewidths=1.4,
                    zorder=4,
                )
                if not warm_handles:
                    warm_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="D",
                            color="none",
                            markerfacecolor="white",
                            markeredgecolor="#222222",
                            markersize=6,
                            linewidth=0,
                            label="Warm run",
                        )
                    )

    tick_labels = [labels[case] for case in order]
    for ax, title, ylabel in (
        (axes[0], "Runtime", "Wall time [s]"),
        (axes[1], "Peak Memory", "Peak RSS [MiB]"),
    ):
        ax.set_xticks(x, tick_labels, rotation=28, ha="right")
        ax.set_title(title, pad=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_yscale("log")
    handles, labels_ = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles + warm_handles,
        labels_ + [h.get_label() for h in warm_handles],
        loc="upper left",
        ncols=4,
        frameon=False,
    )
    fig.suptitle("Runtime and Memory Comparison", fontsize=21)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument(
        "--case",
        action="append",
        default=None,
        help="Run only the named case (repeatable).",
    )
    p.add_argument(
        "--backend",
        action="append",
        default=None,
        help="Run only the named backend (repeatable).",
    )
    p.add_argument(
        "--list", action="store_true", help="List enabled manifest rows and exit."
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them."
    )
    p.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    p.add_argument("--plot-out", type=Path, default=DEFAULT_PNG)
    p.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    p.add_argument(
        "--summary-glob",
        action="append",
        default=None,
        help="Read existing summary JSON files matching this glob and assemble the combined CSV/plot without executing commands.",
    )
    p.add_argument(
        "--skip-plot",
        action="store_true",
        help="Only write CSV/summary, do not build the plot.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining rows after a failure and return nonzero at the end if any row failed.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.summary_glob:
        summary_rows = _load_summary_rows(list(args.summary_glob))
        if not summary_rows:
            raise ValueError(
                "no summary rows matched the provided --summary-glob patterns"
            )
        _write_csv(_resolve(args.csv_out), summary_rows)
        _write_summary(_resolve(args.summary_out), summary_rows)
        if not args.skip_plot:
            plot_png = _resolve(args.plot_out)
            plot_pdf = plot_png.with_suffix(".pdf")
            _plot_results(_resolve(args.csv_out), plot_png, plot_pdf)
        return 0

    runs = _load_manifest(_resolve(args.manifest))
    selected = _select_runs(
        runs, set(args.case or []) or None, set(args.backend or []) or None
    )

    if args.list:
        for run in selected:
            prefix = f"{run.host}:" if run.host else ""
            print(f"{run.case} [{run.backend}] -> {prefix}{run.command}")
        return 0

    rows: list[dict[str, object]] = []
    any_failures = False
    for run in selected:
        rendered_command = _render(run.command)
        rendered_cwd = _render(run.cwd)
        if args.dry_run:
            if run.host:
                print(
                    f"[dry-run] {run.case} [{run.backend}] ssh {run.host} 'cd {rendered_cwd} && {rendered_command}'"
                )
            else:
                print(
                    f"[dry-run] {run.case} [{run.backend}] cd {_resolve(rendered_cwd)} && {rendered_command}"
                )
            continue
        row = _run_command(run)
        row.update(_write_row_logs(_resolve(args.log_dir), row))
        rows.append(row)
        _write_csv(_resolve(args.csv_out), rows)
        _write_summary(_resolve(args.summary_out), rows)
        runtime_obj = row["runtime_s"]
        returncode_obj = row["returncode"]
        if not isinstance(runtime_obj, (int, float)) or not isinstance(
            returncode_obj, int
        ):
            raise TypeError(
                "runtime benchmark row has invalid runtime or return code types"
            )
        runtime_s = float(runtime_obj)
        return_code = returncode_obj
        print(
            f"{row['case']} [{row['backend']}] status={row['status']} runtime_s={runtime_s:.3f} peak_rss_mb={row['peak_rss_mb']}"
        )
        if row["status"] != "success":
            any_failures = True
            if not args.continue_on_error:
                return return_code

    if rows:
        if not args.skip_plot:
            plot_png = _resolve(args.plot_out)
            plot_pdf = plot_png.with_suffix(".pdf")
            _plot_results(_resolve(args.csv_out), plot_png, plot_pdf)
    return 1 if any_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
