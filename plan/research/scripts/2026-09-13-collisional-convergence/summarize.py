#!/usr/bin/env python3
"""Tabulate Q8 (GKX and GX) plus the reused Q3 rows and evaluate P1, P2, P4.

Usage: summarize.py RUN_DIR OUT_CSV

GKX rows come from RUN_DIR/results/<key>.json (parity runner) and the
/usr/bin/time -v logs in RUN_DIR/logs/<key>.stderr.txt; the reused Q3 rows come
from q3_reused.csv next to this script. GX rows come from RUN_DIR/gx/<key>.json
(gx_fit.py), RUN_DIR/logs/<key>.time.txt and RUN_DIR/logs/<key>.gpumem.txt.
A "-t300" rerun of a key, if present, supersedes the T=150 row in the ladders.
``gamma_rel_change_from_prev_nl`` is relative to the next lower Nl of the same
code, collision setting and hypercollision setting.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
FIELDS = [
    "code",
    "source",
    "key",
    "nu",
    "hypercollisions",
    "Nl",
    "t_end",
    "gamma",
    "omega",
    "gamma_half_time_shift",
    "settled",
    "gamma_rel_change_from_prev_nl",
    "process_wall_seconds",
    "peak_rss_kb",
    "peak_device_mb",
    "superseded",
]
NU = {"full": 0.0, "nu0": 0.0, "nu1e-3": 1e-3, "nu3e-3": 3e-3, "nu1e-2": 1e-2}
GKX_KEY = re.compile(r"^(nu0|nu1e-3|nu3e-3|nu1e-2)-nl(\d+)(-t300)?$")
Q3_KEY = re.compile(r"^(full|nu1e-3|nu1e-2)_nl(\d+)$")
GX_KEY = re.compile(r"^gx-(nu1e-2)-nl(\d+)(-nohyper)?$")


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


def _float(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


def gkx_rows(run_dir: Path) -> list[dict]:
    rows = []
    with (HERE / "q3_reused.csv").open(encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            match = Q3_KEY.match(item["key"])
            rows.append(
                {
                    "code": "GKX",
                    "source": "Q3 06606e404 (reused)",
                    "key": item["key"],
                    "nu": NU[match.group(1)],
                    "hypercollisions": "kz",
                    "Nl": int(match.group(2)),
                    "t_end": 150.0,
                    "gamma": float(item["gamma"]),
                    "omega": float(item["omega"]),
                    "gamma_half_time_shift": float(item["gamma_half_time_shift"]),
                    "settled": item["settled"] == "True",
                    "process_wall_seconds": _float(item["process_wall_seconds"]),
                    "peak_rss_kb": _float(item["peak_rss_kb"]),
                    "peak_device_mb": _float(item["peak_device_mb"]),
                }
            )
    for path in sorted((run_dir / "results").glob("*.json")):
        match = GKX_KEY.match(path.stem)
        if not match:
            continue
        case = json.loads(path.read_text(encoding="utf-8"))["cases"][0]
        result = case["rows"][0]
        wall, rss = _time_log(run_dir / "logs" / f"{path.stem}.stderr.txt")
        rows.append(
            {
                "code": "GKX",
                "source": "Q8 578b97074",
                "key": path.stem,
                "nu": NU[match.group(1)],
                "hypercollisions": "kz",
                "Nl": int(match.group(2)),
                "t_end": case["resolution"]["t_end"],
                "gamma": result["gamma_gkx"],
                "omega": result["omega_gkx"],
                "gamma_half_time_shift": result["gamma_half_time_shift"],
                "settled": result["converged"],
                "process_wall_seconds": wall,
                "peak_rss_kb": rss,
                "peak_device_mb": case["cost"]["gkx_peak_device_mb"],
            }
        )
    return rows


def gx_rows(run_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((run_dir / "gx").glob("*.json")):
        match = GX_KEY.match(path.stem)
        if not match:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        wall, rss = _time_log(run_dir / "logs" / f"{path.stem}.time.txt")
        memory = run_dir / "logs" / f"{path.stem}.gpumem.txt"
        samples = (
            [float(v) for v in memory.read_text().split()] if memory.exists() else []
        )
        rows.append(
            {
                "code": "GX",
                "source": "GX 3865a537 + repairs, binary 96a53403",
                "key": path.stem,
                "nu": record["species_nu"][0],
                "hypercollisions": "kz" if record["hypercollisions_kz"] else "off",
                "Nl": record["nlaguerre"],
                "t_end": record["t_end"],
                "gamma": record["gamma_phi2_late"],
                "omega": record["omega_second_half_mean"],
                "gamma_half_time_shift": record["gamma_half_time_shift"],
                "settled": record["settled"],
                "process_wall_seconds": wall,
                "peak_rss_kb": rss,
                "peak_device_mb": max(samples) if samples else None,
            }
        )
    return rows


def main() -> None:
    run_dir, out_csv = Path(sys.argv[1]), Path(sys.argv[2])
    rows = gkx_rows(run_dir) + gx_rows(run_dir)
    reruns = {r["key"][: -len("-t300")] for r in rows if r["key"].endswith("-t300")}
    for row in rows:
        row["superseded"] = row["key"] in reruns
    ladder: dict[tuple, dict[int, dict]] = {}
    for row in rows:
        if not row["superseded"]:
            group = (row["code"], row["nu"], row["hypercollisions"])
            ladder.setdefault(group, {})[row["Nl"]] = row
    for levels in ladder.values():
        order = sorted(levels)
        for low, high in zip(order, order[1:]):
            levels[high]["gamma_rel_change_from_prev_nl"] = (
                levels[high]["gamma"] - levels[low]["gamma"]
            ) / abs(levels[low]["gamma"])
    rows.sort(
        key=lambda r: (
            r["code"] != "GKX",
            r["hypercollisions"],
            r["nu"],
            r["Nl"],
            r["t_end"],
        )
    )
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})

    gkx = {nu: levels for (code, nu, hyp), levels in ladder.items() if code == "GKX"}
    print("P1: GKX collisional ladders")
    gamma48 = {}
    for nu in (1e-3, 3e-3, 1e-2):
        levels = gkx.get(nu, {})
        text = ", ".join(
            f"Nl{nl} {levels[nl]['gamma']:.6f}{'' if levels[nl]['settled'] else '*'}"
            for nl in sorted(levels)
        )
        print(f"  nu={nu:g}: {text}  (* = not settled)")
        if 32 in levels and 48 in levels:
            gamma48[nu] = levels[48]["gamma"]
            change = (
                abs(levels[48]["gamma"] - levels[32]["gamma"]) / levels[48]["gamma"]
            )
            print(f"    |g48-g32|/g48 = {change:.4%}  (P1 threshold 2%)")
    if len(gamma48) == 3:
        values = list(gamma48.values())
        mean = sum(values) / 3
        print(
            f"  spread of g48 over nu: (max-min)/mean = {(max(values) - min(values)) / mean:.4%}"
        )
        xs, ys = list(gamma48), values
        xbar, ybar = sum(xs) / 3, mean
        slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum(
            (x - xbar) ** 2 for x in xs
        )
        print(
            f"  linear LSQ g48(nu): intercept {ybar - slope * xbar:.6f}, slope {slope:.4f}"
        )
        two = (gamma48[1e-3] * 3e-3 - gamma48[3e-3] * 1e-3) / (3e-3 - 1e-3)
        print(f"  two-point (1e-3, 3e-3) extrapolation to nu=0: {two:.6f}")
    zero = gkx.get(0.0, {})
    print("P2: GKX collisionless ladder")
    print(
        "  "
        + ", ".join(
            f"Nl{nl} {zero[nl]['gamma']:.6f}{'' if zero[nl]['settled'] else '*'}"
            for nl in sorted(zero)
        )
    )
    if 48 in zero and 64 in zero:
        print(
            f"  g64 < g48: {zero[64]['gamma'] < zero[48]['gamma']}  change {zero[64]['gamma_rel_change_from_prev_nl']:+.4%}"
        )
    print("P4: GX vs GKX at nu=1e-2")
    for row in rows:
        if row["code"] != "GX":
            continue
        match = gkx.get(row["nu"], {}).get(row["Nl"])
        if match is None:
            continue
        print(
            f"  {row['key']}: GX gamma {row['gamma']:.6f} omega {row['omega']:.6f}; "
            f"GKX gamma {match['gamma']:.6f} omega {match['omega']:.6f}; "
            f"d gamma {(match['gamma'] - row['gamma']) / row['gamma']:+.4%} "
            f"d omega {(match['omega'] - row['omega']) / row['omega']:+.4%}"
        )


if __name__ == "__main__":
    main()
