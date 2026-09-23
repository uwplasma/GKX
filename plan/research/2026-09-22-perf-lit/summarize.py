"""Print the PERF-LIT tables from a directory of profile_perf_lit.py records.

Usage: python summarize.py <dir> [<dir> ...]
"""

from __future__ import annotations

import glob
import json
import os
import sys


def mb(x):
    return "-" if x is None else f"{x / 1e6:.1f}"


def forward(path: str) -> None:
    r = json.load(open(path))
    print(
        f"== {os.path.basename(path)} shape={r['shape']} state={mb(r['state_bytes'])} MB "
        f"load={r['environment']['loadavg_start'][0]:.0f} dev={r['environment']['devices'][0]}"
    )
    for k, g in r["graphs"].items():
        h = g.get("hlo", {})
        print(
            f"  {k:20s} warm {g['warm_median_s'] * 1e3:9.3f} ms  compile {g['compile_s']:5.1f} s  "
            f"temp {mb(g['temp_bytes']):>8} MB  fft {h.get('fft')}  transpose {h.get('transpose')}  "
            f"copy {h.get('copy')}  concat {h.get('concatenate')}  materialized {mb(h.get('materialized_bytes'))} MB  "
            f"dev_peak {mb(g['device_peak_bytes'])} MB"
        )
    print("  derived", {k: round(v, 4) for k, v in r["derived"].items()})
    t = r.get("trace_top_ops")
    if t and "by_category" in t:
        print(
            "  trace",
            t["scope"],
            {k: round(v["frac"], 3) for k, v in t["by_category"].items()},
        )
        for x in t["top"][:8]:
            print(f"     {x['frac']:.3f} n={x['count']:3d} {x['name'][:100]}")


def fft(path: str) -> None:
    r = json.load(open(path))
    print(
        f"== {os.path.basename(path)} fft_s_per_rhs {r['fft_seconds_per_rhs'] * 1e3:.3f} ms  "
        f"rhs {r['rhs_seconds'] * 1e3:.3f} ms  fft_fraction_upper {r['fft_fraction_upper']:.3f}"
    )


def arms(path: str) -> None:
    r = json.load(open(path))
    extra = {k: r[k] for k in ("steps", "shape", "Nl", "Nm", "Nz") if k in r}
    print(
        f"== {os.path.basename(path)} {extra} dev={r['environment']['devices'][0]} "
        f"load={r['environment']['loadavg_start'][0]:.0f}"
    )
    for k, a in r["arms"].items():
        keys = ("compile_s", "lower_s", "first_call_s", "warm_median_s", "call_s")
        vals = {
            x: (
                round(a[x], 3)
                if isinstance(a[x], float)
                else [round(v, 3) for v in a[x]]
            )
            for x in keys
            if x in a
        }
        print(
            f"  {k:24s} {vals} temp {mb(a.get('temp_bytes'))} MB dev_peak {mb(a.get('device_peak_bytes'))} MB "
            f"host_peak {mb(a.get('host_peak_bytes'))} MB "
            + " ".join(
                f"{x}={a[x]:.6g}"
                for x in (
                    "objective",
                    "grad_tprim_scale",
                    "grad_geometry_norm",
                    "gamma",
                    "grad_norm",
                )
                if x in a
            )
        )
    print("  derived", {k: round(v, 3) for k, v in r["derived"].items()})


def main() -> None:
    for d in sys.argv[1:]:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            cmd = json.load(open(path)).get("command")
            {"forward": forward, "fftfloor": fft}.get(cmd, arms)(path)


if __name__ == "__main__":
    main()
