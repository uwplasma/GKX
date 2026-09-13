#!/usr/bin/env python3
"""Tabulate the Q3 runs from the runner JSONs and /usr/bin/time logs.

Usage: summarize.py RUN_DIR OUT_CSV

One row per registered key. Keys that were registered but not run are listed
with status "skipped". ``gamma_rel_change_from_prev_nl`` is the relative gamma
change from the next lower Nl of the same configuration (Nl24->32 on the Nl32
row, Nl32->48 on the Nl48 row).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys

KEYS = [
    ("full", 24),
    ("full", 32),
    ("full", 48),
    ("gradb0", 24),
    ("gradb0", 32),
    ("curvature0", 24),
    ("curvature0", 32),
    ("nu1e-3", 24),
    ("nu1e-3", 32),
    ("nu1e-2", 24),
    ("nu1e-2", 32),
]
SKIPPED = {"curvature0_nl24", "curvature0_nl32"}
FIELDS = [
    "key",
    "configuration",
    "Nl",
    "status",
    "gamma",
    "omega",
    "gamma_half_time_shift",
    "omega_half_time_shift",
    "settled",
    "gamma_rel_change_from_prev_nl",
    "runner_scan_seconds",
    "process_wall_seconds",
    "peak_rss_kb",
    "peak_device_mb",
]


def _time_log(path: Path) -> tuple[float | None, int | None]:
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8", errors="replace")
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    seconds = None
    if wall:
        parts = [float(p) for p in wall.group(1).split(":")]
        seconds = sum(v * 60.0**i for i, v in enumerate(reversed(parts)))
    return seconds, int(rss.group(1)) if rss else None


def main() -> None:
    run_dir, out_csv = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    gamma = {}
    for configuration, nl in KEYS:
        key = f"{configuration}_nl{nl}"
        row = {"key": key, "configuration": configuration, "Nl": nl}
        path = run_dir / "results" / f"{key}.json"
        if key in SKIPPED:
            row["status"] = "skipped"
        elif not path.exists():
            row["status"] = "missing"
        else:
            case = json.loads(path.read_text(encoding="utf-8"))["cases"][0]
            result = case["rows"][0]
            wall, rss = _time_log(run_dir / "logs" / f"{key}.stderr.txt")
            gamma[(configuration, nl)] = result["gamma_gkx"]
            row.update(
                status="ok",
                gamma=result["gamma_gkx"],
                omega=result["omega_gkx"],
                gamma_half_time_shift=result["gamma_half_time_shift"],
                omega_half_time_shift=result["omega_half_time_shift"],
                settled=result["converged"],
                runner_scan_seconds=case["cost"]["gkx_scan_seconds"],
                process_wall_seconds=wall,
                peak_rss_kb=rss,
                peak_device_mb=case["cost"]["gkx_peak_device_mb"],
            )
        rows.append(row)
    ladder = {}
    for configuration, nl in KEYS:
        ladder.setdefault(configuration, []).append(nl)
    for row in rows:
        levels = ladder[row["configuration"]]
        index = levels.index(row["Nl"])
        if index == 0:
            continue
        low = gamma.get((row["configuration"], levels[index - 1]))
        high = gamma.get((row["configuration"], row["Nl"]))
        if low is not None and high is not None and low != 0.0:
            row["gamma_rel_change_from_prev_nl"] = (high - low) / abs(low)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})
    print(out_csv.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
