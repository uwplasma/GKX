"""One-step probe of integrate_nonlinear_sharded, split vs unsplit, on 2 real devices.

usage (from a GKX checkout; CPU needs XLA_FLAGS=--xla_force_host_platform_device_count=2):
    PYTHONPATH=src python plan/research/scripts/2026-09-22-ky-state-sharding/step_probe.py [full|half]

For ky and kx splits, euler and rk2, and the compressed-real-FFT projector on
and off, prints the max difference after one step, the step's own size, and
the ky rows / kx columns that disagree. This is how the XLA:GPU failure of the
one-region-per-call draft was localized (identity_gpu.jsonl): wrong only with
the projector on and ky split, on the conjugate rows that cross devices.
"""

import json
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests" / "unit" / "parallel"))
import test_parallel_nonlinear_routing as deck  # noqa: E402
import gkx.workflows.nonlinear as workflow  # noqa: E402
from gkx.runtime import run_runtime_nonlinear  # noqa: E402
from gkx.operators.linear.cache_builder import build_linear_cache  # noqa: E402
from gkx.core_grid import _gyrokinetic_moment_shape  # noqa: E402
from gkx.parallel import integrators as I  # noqa: E402

layout = sys.argv[1] if len(sys.argv) > 1 else "full"
captured = {}
orig = workflow._run_once


def spy(cfg, ctx, *a, **k):
    captured["ctx"] = ctx
    return orig(cfg, ctx, *a, **k)


workflow._run_once = spy
cfg = deck._live_cfg(ky_layout=layout)
run_runtime_nonlinear(cfg, **{**deck._RUN_KWARGS, "steps": 1})
workflow._run_once = orig
ctx = captured["ctx"]
nl, nm = _gyrokinetic_moment_shape(ctx.G0)
cache = build_linear_cache(ctx.grid, ctx.geom, ctx.params, nl, nm)
mesh = Mesh(np.array(jax.devices()[:2]), ("d",))
G0 = jnp.asarray(ctx.G0)
out = {"backend": jax.default_backend(), "layout": layout}
for axis, name in ((3, "ky"), (4, "kx")):
    sh = NamedSharding(mesh, P(*[("d" if i == axis else None) for i in range(G0.ndim)]))
    for method in ("euler", "rk2"):
        for crf in (False, True):
            kw = dict(
                dt=ctx.dt,
                steps=1,
                method=method,
                terms=ctx.terms,
                compressed_real_fft=crf,
                return_fields=False,
            )
            ref = np.asarray(I.integrate_nonlinear_sharded(G0, cache, ctx.params, **kw))
            got = np.asarray(
                I.integrate_nonlinear_sharded(
                    G0, cache, ctx.params, state_sharding=sh, **kw
                )
            )
            change = float(np.max(np.abs(ref - np.asarray(G0))))
            diff = float(np.max(np.abs(got - ref)))
            # which ky rows / kx columns are wrong
            bad = np.argwhere(np.abs(got - ref) > 1e-3 * max(change, 1e-30))
            out[f"{name}/{method}/crf={crf}"] = {
                "diff": diff,
                "change": change,
                "bad_ky": sorted(set(int(b[3]) for b in bad)),
                "bad_kx": sorted(set(int(b[4]) for b in bad)),
            }
print(json.dumps(out))
