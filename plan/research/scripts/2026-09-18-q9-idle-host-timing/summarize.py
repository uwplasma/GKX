"""Q9 idle-host campaign artifacts, from the per-arm JSON bench_q9.py wrote.

Usage: python summarize.py OUTDIR DEST

Writes into DEST:
  raw_reps.txt    every timed rep, per tag, kernel, block and arm
  threadpool.txt  default XLA CPU FFT thread pool against
                  --xla_cpu_multi_thread_eigen=false
                  intra_op_parallelism_threads=1, per arm (plan 5.3 N2)
  arm_load.txt    the runner's per-arm gate lines (load and core busy)
  block_load.txt  the same, folded to one line per tag and block
  env.txt         host, interpreter, jax, affinity and XLA flags per arm
The A/B tables themselves come from ab_table2.py, which owns the gate rule.
"""

import json
import pathlib
import re
import statistics
import sys

out = pathlib.Path(sys.argv[1])
dest = pathlib.Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)
tags = ["pool64", "nopool64", "pool32", "dense64", "dense32", "dscan64", "dscan32"]
kernels = ("rhs", "rhs_vjp", "scan_rk3", "window_vjp")
arms = ("base", "p2t")


def load(tag, arm, block):
    p = out / f"{tag}_{arm}_b{block}.json"
    return json.loads(p.read_text()) if p.exists() else None


def blocks_of(tag):
    return sorted(int(f.stem.split("_b")[-1]) for f in out.glob(f"{tag}_base_b*.json"))


lines = ["Q9 idle-host A/B/A/B: every timed rep, in milliseconds.", ""]
for tag in tags:
    bl = blocks_of(tag)
    if not bl:
        continue
    lines.append(f"== {tag} ==")
    for kernel in kernels:
        wrote = False
        for b in bl:
            for arm in arms:
                d = load(tag, arm, b)
                if not d or kernel not in d["kernels"]:
                    continue
                reps = d["kernels"][kernel]["reps_s"]
                lines.append(
                    f"{kernel:11s} b{b} {arm:4s} n={len(reps):3d} "
                    f"median={1e3 * statistics.median(reps):9.1f} "
                    f"min={1e3 * min(reps):9.1f} compile={d['kernels'][kernel]['compile_and_first_s']:6.2f}s"
                )
                lines.append("    " + " ".join(f"{1e3 * r:.1f}" for r in reps))
                wrote = True
        if wrote:
            lines.append("")
(dest / "raw_reps.txt").write_text("\n".join(lines) + "\n")

steps = {"pool64": 5, "nopool64": 2}
tp = [
    "CPU FFT thread pool, plan 5.3 N2 re-measurement (64x64x24 Nl4/Nm8).",
    "pool   = default XLA CPU flags (multi-threaded Eigen/DUCC FFT).",
    "nopool = --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1.",
    "Medians pooled over every rep of every block of that tag; the rk3 scan is",
    "reported per step because the pool tag ran 5 steps and the nopool tag 2.",
    "",
    f"{'kernel':13s} {'arm':5s} {'pool ms':>10s} {'nopool ms':>10s} {'nopool/pool':>12s}",
]
for kernel in kernels:
    for arm in arms:
        vals = {}
        for tag in ("pool64", "nopool64"):
            reps = [
                r
                for b in blocks_of(tag)
                for r in (load(tag, arm, b) or {"kernels": {}})["kernels"]
                .get(kernel, {})
                .get("reps_s", [])
            ]
            if reps:
                vals[tag] = statistics.median(reps) / (
                    steps[tag] if kernel == "scan_rk3" else 1
                )
        if len(vals) == 2:
            label = kernel + ("/step" if kernel == "scan_rk3" else "")
            tp.append(
                f"{label:13s} {arm:5s} {1e3 * vals['pool64']:10.1f} "
                f"{1e3 * vals['nopool64']:10.1f} {vals['nopool64'] / vals['pool64']:12.3f}"
            )
(dest / "threadpool.txt").write_text("\n".join(tp) + "\n")

env = ["Per-arm environment recorded by bench_q9.py.", ""]
seen = set()
for tag in tags:
    for b in blocks_of(tag):
        for arm in arms:
            d = load(tag, arm, b)
            if not d:
                continue
            key = (
                d["host"],
                d["jax"],
                tuple(d["affinity"] or ()),
                d["xla_flags"],
                tuple(d["grid"]),
                arm,
            )
            if key in seen:
                continue
            seen.add(key)
            env.append(
                f"{tag} {arm}: host={d['host']} jax={d['jax']} grid={d['grid']} "
                f"state_shape={d['state_shape']}\n  gkx={d['gkx']}\n"
                f"  xla_flags={d['xla_flags']!r} affinity={d['affinity']}"
            )
(dest / "env.txt").write_text("\n".join(env) + "\n")

load_lines = []
for tag in tags:
    p = out / f"runs_{tag}.log"
    if p.exists():
        load_lines.append(f"== runs_{tag}.log ==")
        load_lines.append(p.read_text().rstrip())
        w = out / f"wait_{tag}.log"
        load_lines.append(
            f"-- wait_{tag}.log: {'no arm ever waited on the idleness gate' if not w.exists() else w.read_text().rstrip()}"
        )
        load_lines.append("")
(dest / "arm_load.txt").write_text("\n".join(load_lines) + "\n")

pat = re.compile(
    r"block=(\d+) arm=(\w+) tag=(\w+) order=([\w ]+?) load=([\d.]+) "
    r"mean_core_busy=([\d.]+) worst_core_busy=([\d.]+)"
)
per_block: dict[tuple[str, int], dict[str, object]] = {}
for line in (dest / "arm_load.txt").read_text().splitlines():
    m = pat.search(line)
    if not m:
        continue
    b, arm, tag, order, load, mb, wb = m.groups()
    entry = per_block.setdefault((tag, int(b)), {})
    entry["first"] = order.split()[0]
    entry[arm] = (float(load), float(mb), float(wb))
bl_lines = [
    "Machine state at the start of every arm, by tag and block.",
    "first = which arm ran first in that block (the order rotates).",
    "load = 1-minute load average, dominated by this campaign's own decaying",
    "threads; busy = mean/worst fraction of the 12 benchmark cores busy over a",
    "3 s sample, which is the quantity the idleness gate actually tests.",
    "",
]
for (tag, b), v in sorted(per_block.items()):
    bl_lines.append(
        f"{tag:9s} b{b} first={v['first']:4s} "
        f"base load={v['base'][0]:5.2f} busy={v['base'][1]:.3f}/{v['base'][2]:.3f}   "
        f"p2t load={v['p2t'][0]:5.2f} busy={v['p2t'][1]:.3f}/{v['p2t'][2]:.3f}"
    )
(dest / "block_load.txt").write_text("\n".join(bl_lines) + "\n")
print("wrote", ", ".join(sorted(p.name for p in dest.iterdir())))
