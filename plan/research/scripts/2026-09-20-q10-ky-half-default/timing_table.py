"""Q10: the wall-clock half/full table, from the archived per-arm records.

Usage: python timing_table.py [OUT_DIR]

Reads the ``nopool_g{32,64}_{full,half}_b{1..4}.json`` records that
``run_timing_ab.sh`` wrote on the office host and prints the table
``docs/performance.rst`` quotes.  Each record is one fresh process on one arm;
its ``median_s`` is the median of seven timed repetitions after a warm call.
The table takes, per kernel and grid, the median over the four blocks of each
arm and reports ``half / full``.

Both arms imported the same tree (``38d7c4277``, the merge of #259, clean
``src/``) and differed only in ``--ky-layout``; the records name the tree, the
jax version, the XLA flags, the CPU affinity and the host load before and after
each arm, and this script refuses a set in which those disagree across arms.
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "out"
records = sorted(out.glob("nopool_g*_*_b*.json"))
if not records:
    raise SystemExit(f"no timing records under {out}")

medians: dict[tuple[str, str], dict[str, list[float]]] = collections.defaultdict(
    lambda: collections.defaultdict(list)
)
provenance: set[tuple[str, str, str, tuple[int, ...]]] = set()
for path in records:
    data = json.loads(path.read_text())
    _, grid, arm, _block = path.stem.split("_")
    if data["ky_layout"] != arm or data["grid_ky_layout"] != arm:
        raise SystemExit(f"{path.name}: arm {arm!r} ran on {data['grid_ky_layout']!r}")
    provenance.add(
        (data["gkx"], data["jax"], data["xla_flags"], tuple(data["affinity"]))
    )
    for kernel, timing in data["kernels"].items():
        medians[(grid, kernel)][arm].append(float(timing["median_s"]))

if len(provenance) != 1:
    raise SystemExit(f"records disagree on tree/jax/flags/affinity: {provenance}")
tree, jax_version, flags, affinity = provenance.pop()
print(f"tree {tree}\njax {jax_version}  XLA_FLAGS {flags}  cpus {list(affinity)}\n")
print(f"{'grid':5} {'kernel':11} {'full (s)':>10} {'half (s)':>10} {'half/full':>10}")
for (grid, kernel), arms in sorted(medians.items()):
    full = statistics.median(arms["full"])
    half = statistics.median(arms["half"])
    blocks = f"{len(arms['full'])}/{len(arms['half'])}"
    print(
        f"{grid:5} {kernel:11} {full:10.4f} {half:10.4f} {half / full:10.3f}  {blocks}"
    )
