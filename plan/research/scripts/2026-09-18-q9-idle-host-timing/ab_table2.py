"""Q9 idle-host A/B/A/B table, two arms, with the Q9 gate rule unchanged.

Usage: python ab_table2.py OUTDIR [TAG ...]

Arms: base (main just before #243) and p2t (main with #243's shared
transform + transpose-free chain layout). One median per arm, kernel and
block, plus the variant/base ratio.

Gate, as registered in the 2026-09-14 Q9 entry and reused verbatim here: "a
variant passes a kernel if, in every block, its median is at most base's
median x (1 + tol), where tol = max(2%, (max - min)/median of base's reps in
that block)".

Registered before this campaign's first measurement, to let the run say more
than "not slower": the mirror of the same rule decides a speed-up. A variant
is called FASTER on a kernel only if in every block its median is at most
base's median x (1 - tol) with the same per-block tol; SLOWER if in every
block it is at least base's median x (1 + tol); otherwise
INDISTINGUISHABLE. A variant that fails the pass gate in any block is
reported as a regression on that kernel regardless of the other blocks.
"""

import json
import pathlib
import statistics
import sys

out = pathlib.Path(sys.argv[1])
tags = sys.argv[2:] or ["pool64", "nopool64", "pool32"]
kernels = ("rhs", "rhs_vjp", "scan_rk3", "window_vjp")
arms = ("base", "p2t")
report = {}
for tag in tags:
    files = sorted(out.glob(f"{tag}_base_b*.json"))
    if not files:
        continue
    blocks = sorted(int(f.stem.split("_b")[-1]) for f in files)
    rows = {}
    for kernel in kernels:
        per_block = []
        for b in blocks:
            data = {}
            for arm in arms:
                p = out / f"{tag}_{arm}_b{b}.json"
                if not p.exists():
                    break
                payload = json.loads(p.read_text())["kernels"]
                if kernel not in payload:
                    break
                data[arm] = payload[kernel]
            if len(data) != len(arms):
                continue
            base_reps = data["base"]["reps_s"]
            base_med = statistics.median(base_reps)
            tol = max(0.02, (max(base_reps) - min(base_reps)) / base_med)
            var_reps = data["p2t"]["reps_s"]
            med = statistics.median(var_reps)
            per_block.append(
                {
                    "block": b,
                    "tol": tol,
                    "base_ms": 1e3 * base_med,
                    "base_reps_ms": [1e3 * r for r in base_reps],
                    "base_spread": (max(base_reps) - min(base_reps)) / base_med,
                    "p2t_ms": 1e3 * med,
                    "p2t_reps_ms": [1e3 * r for r in var_reps],
                    "p2t_spread": (max(var_reps) - min(var_reps)) / med,
                    "p2t_ratio": med / base_med,
                    "p2t_pass": med <= base_med * (1.0 + tol),
                    "p2t_faster": med <= base_med * (1.0 - tol),
                    "p2t_slower": med >= base_med * (1.0 + tol),
                    "p2t_min_ratio": min(var_reps) / min(base_reps),
                    "n_reps": [len(base_reps), len(var_reps)],
                }
            )
        if not per_block:
            continue
        pooled = {}
        for arm in arms:
            reps = [
                r
                for e in per_block
                for r in json.loads(
                    (out / f"{tag}_{arm}_b{e['block']}.json").read_text()
                )["kernels"][kernel]["reps_s"]
            ]
            pooled[arm] = statistics.median(reps)
        passes = all(e["p2t_pass"] for e in per_block)
        if not passes:
            verdict = "REGRESSION"
        elif all(e["p2t_faster"] for e in per_block):
            verdict = "FASTER"
        elif all(e["p2t_slower"] for e in per_block):
            verdict = "SLOWER"
        else:
            verdict = "INDISTINGUISHABLE"
        rows[kernel] = {
            "blocks": per_block,
            "pooled_ms": {a: 1e3 * v for a, v in pooled.items()},
            "pooled_ratio": pooled["p2t"] / pooled["base"],
            "pass_all_blocks": passes,
            "verdict": verdict,
        }
    report[tag] = rows
    print(f"== {tag} ({len(blocks)} blocks)")
    print(
        f"{'kernel':11s} {'blk':>3s} {'reps':>5s} {'base ms':>9s} {'spread':>7s} "
        f"{'S+T ms':>9s} {'spread':>7s} {'ratio':>6s} {'tol':>5s}"
    )
    for kernel, row in rows.items():
        for e in row["blocks"]:
            flag = "" if e["p2t_pass"] else "*"
            print(
                f"{kernel:11s} {e['block']:3d} {e['n_reps'][0]:5d} {e['base_ms']:9.1f} "
                f"{e['base_spread']:7.3f} {e['p2t_ms']:9.1f} {e['p2t_spread']:7.3f} "
                f"{e['p2t_ratio']:6.3f}{flag} {e['tol']:5.3f}  min-rep {e['p2t_min_ratio']:.3f}"
            )
        print(
            f"{kernel:11s} all       {row['pooled_ms']['base']:9.1f} "
            f"        {row['pooled_ms']['p2t']:9.1f}         {row['pooled_ratio']:6.3f}"
            f"         pass={row['pass_all_blocks']} verdict={row['verdict']}"
        )
(out / "ab_report_idle.json").write_text(json.dumps(report, indent=2))
