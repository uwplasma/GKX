"""Q9 A/B/A/B timing table and registered gate.

Usage: python ab_table.py OUTDIR [FLAGSET ...]

For each flagset, kernel and block: per-arm median of the 7 reps, and
variant/base ratio. The gate (registered before any timing): a variant passes a
kernel if in every block its median is not above base's median by more than the
noise tolerance of that block, tol = max(0.02, relative spread of base's reps
(max-min)/median in that block). Also reports the ratio of block-pooled medians and, as a contention
diagnostic only, the ratio of fastest reps.
"""

import json
import pathlib
import statistics
import sys

out = pathlib.Path(sys.argv[1])
flagsets = sys.argv[2:] or ["pool", "nopool", "pool32"]
kernels = ("rhs", "rhs_vjp", "scan_rk3", "window_vjp")
arms = ("base", "p2t", "tonly")
report = {}
for fs in flagsets:
    files = sorted(out.glob(f"{fs}_base_b*.json"))
    if not files:
        continue
    blocks = sorted(int(f.stem.split("_b")[-1]) for f in files)
    rows = {}
    for kernel in kernels:
        per_block = []
        for b in blocks:
            data = {}
            for arm in arms:
                p = out / f"{fs}_{arm}_b{b}.json"
                if not p.exists():
                    break
                data[arm] = json.loads(p.read_text())["kernels"][kernel]
            if len(data) != len(arms):
                continue
            base_reps = data["base"]["reps_s"]
            base_med = statistics.median(base_reps)
            tol = max(0.02, (max(base_reps) - min(base_reps)) / base_med)
            entry = {"block": b, "tol": tol, "base_ms": 1e3 * base_med}
            for arm in ("p2t", "tonly"):
                med = statistics.median(data[arm]["reps_s"])
                entry[f"{arm}_ms"] = 1e3 * med
                entry[f"{arm}_ratio"] = med / base_med
                entry[f"{arm}_pass"] = med <= base_med * (1.0 + tol)
                entry[f"{arm}_min_ratio"] = min(data[arm]["reps_s"]) / min(base_reps)
            per_block.append(entry)
        pooled = {}
        for arm in arms:
            reps = [
                r
                for b in per_block
                for r in json.loads(
                    (out / f"{fs}_{arm}_b{b['block']}.json").read_text()
                )["kernels"][kernel]["reps_s"]
            ]
            pooled[arm] = statistics.median(reps) if reps else float("nan")
        rows[kernel] = {
            "blocks": per_block,
            "pooled_ms": {a: 1e3 * v for a, v in pooled.items()},
            "pooled_ratio": {a: pooled[a] / pooled["base"] for a in ("p2t", "tonly")},
            "pass_all_blocks": {
                a: bool(per_block) and all(e[f"{a}_pass"] for e in per_block)
                for a in ("p2t", "tonly")
            },
        }
    report[fs] = rows
    print(f"== {fs} ({len(blocks)} blocks)")
    print(
        f"{'kernel':11s} {'blk':>3s} {'base ms':>9s} {'S+T ms':>9s} {'ratio':>6s} {'T ms':>9s} {'ratio':>6s} {'tol':>5s}"
    )
    for kernel, row in rows.items():
        for e in row["blocks"]:
            print(
                f"{kernel:11s} {e['block']:3d} {e['base_ms']:9.1f} {e['p2t_ms']:9.1f} {e['p2t_ratio']:6.3f}"
                f"{'' if e['p2t_pass'] else '*'} {e['tonly_ms']:9.1f} {e['tonly_ratio']:6.3f}"
                f"{'' if e['tonly_pass'] else '*'} {e['tol']:5.3f}  min-rep S+T {e['p2t_min_ratio']:.3f} T {e['tonly_min_ratio']:.3f}"
            )
        pr = row["pooled_ratio"]
        print(
            f"{kernel:11s} all {row['pooled_ms']['base']:9.1f} {row['pooled_ms']['p2t']:9.1f} {pr['p2t']:6.3f}"
            f"  {row['pooled_ms']['tonly']:9.1f} {pr['tonly']:6.3f}   pass S+T={row['pass_all_blocks']['p2t']}"
            f" T={row['pass_all_blocks']['tonly']}"
        )
(out / "ab_report.json").write_text(json.dumps(report, indent=2))
