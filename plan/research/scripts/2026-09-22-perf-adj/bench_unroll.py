"""Per-step cost of the window inside its scan, and the effect of scan unroll.

Reuses bench_ab's case (import side effects build it) and times the forward
window and its value_and_grad with ``jax.lax.scan`` patched to ``unroll=k``,
interleaved over rounds. Usage: python bench_unroll.py --steps 256 --ky full
"""

import os
import statistics
import sys
import time

import jax

sys.argv += ["--variants", "block", "--rounds", "1"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_ab as case  # noqa: E402

original_scan = jax.lax.scan
arms = {}
for unroll in (1, 2, 4):
    jax.clear_caches()
    jax.lax.scan = lambda *a, _u=unroll, **k: original_scan(*a, **{**k, "unroll": _u})
    arms[unroll] = case.compile_variant("block")
    jax.lax.scan = original_scan
times = {u: {"vjp": [], "fwd": []} for u in arms}
for u, (vjp, fwd, *_rest) in arms.items():
    jax.block_until_ready(vjp(case.tprim0))
    jax.block_until_ready(fwd(case.tprim0))
for _ in range(5):
    for u, (vjp, fwd, *_rest) in arms.items():
        for kind, fn in (("vjp", vjp), ("fwd", fwd)):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(case.tprim0))
            times[u][kind].append(time.perf_counter() - t0)
steps = case.args.steps
for u, t in times.items():
    fwd = statistics.median(t["fwd"])
    vjp = statistics.median(t["vjp"])
    print(
        f"unroll {u}: fwd {fwd:.4f}s ({1e3 * fwd / steps:.3f} ms/step) "
        f"vjp {vjp:.4f}s ({1e3 * vjp / steps:.3f} ms/step) compile {arms[u][2]:.1f}s temp {arms[u][3] / 2**20:.0f}MiB"
    )
