"""Serial vs ``[time] state_sharding`` run of a live nonlinear deck; reads results back.

usage (from a GKX checkout, JAX 0.10.2, CPU):
    AMP=1e-2 STEPS=20 XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        PYTHONPATH=src python plan/research/scripts/2026-09-22-ky-state-sharding/state_sharding_identity.py \
        <full|half> <n_devices> <ky|kx>

The deck is the routing-test deck of tests/unit/parallel/test_parallel_nonlinear_routing.py
with Nx = 4, Lx = Ly = 31.4 and a multimode initial condition. The routing-test deck itself
(Nx = 1, single mode) does not evolve (phi = 0, state unchanged), so an identity on it is vacuous.
Prints one JSON line; identity_after.jsonl holds the recorded runs.
"""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests" / "unit" / "parallel"))
import test_parallel_nonlinear_routing as deck  # noqa: E402
import gkx.solvers_time_runners as runners  # noqa: E402
import gkx.workflows.nonlinear as workflow  # noqa: E402
from gkx.parallel.state import resolve_state_sharding  # noqa: E402
from gkx.runtime import run_runtime_nonlinear  # noqa: E402

layout, n_dev, spec = sys.argv[1], int(sys.argv[2]), sys.argv[3]

AMP = float(os.environ.get("AMP", "1e-8"))
STEPS = int(os.environ.get("STEPS", "0"))
captured = {}
orig = workflow._run_once


def spy(cfg, ctx, *a, **k):
    captured["ctx"] = ctx
    return orig(cfg, ctx, *a, **k)


workflow._run_once = spy
cfg = deck._nonlinear_cfg(ky_layout=layout)
cfg = replace(
    cfg,
    init=replace(cfg.init, init_amp=AMP, init_single=False),
    grid=replace(cfg.grid, Nx=int(os.environ.get("NX", "4")), Lx=31.4, Ly=31.4),
)
run_runtime_nonlinear(cfg, **deck._RUN_KWARGS)
workflow._run_once = orig
ctx = captured["ctx"]
devices = jax.devices()[:n_dev]
runners.resolve_state_sharding = lambda G0, s: resolve_state_sharding(
    G0, s, devices=devices
)


def run(sharding):
    tc = replace(
        cfg.time,
        dt=ctx.dt,
        t_max=ctx.dt * (STEPS or ctx.steps),
        state_sharding=sharding,
    )
    state, fields = runners.integrate_nonlinear_from_config(
        jnp.array(ctx.G0, copy=True),
        ctx.grid,
        ctx.geom,
        ctx.params,
        tc,
        terms=ctx.terms,
    )
    return state, fields


out = {
    "layout": layout,
    "n_dev": n_dev,
    "spec": spec,
    "ky_extent": int(ctx.G0.shape[-3]),
    "jax": jax.__version__,
    "x64": bool(jax.config.jax_enable_x64),
}
ref, ref_f = run(None)
ref = np.asarray(ref)
try:
    st, f = run(spec)
    out["sharding"] = str(getattr(st.sharding, "spec", st.sharding))
    got = np.asarray(st)
    out["state_shape"] = list(got.shape)
    scale = float(np.max(np.abs(ref)))
    out["max_abs_diff"] = float(np.max(np.abs(got - ref)))
    out["rel_diff"] = out["max_abs_diff"] / scale
    out["state_max"] = scale
    out["state_change"] = float(np.max(np.abs(ref - np.asarray(ctx.G0))))
    out["phi_max"] = float(np.max(np.abs(np.asarray(ref_f.phi))))
    out["phi_max_abs_diff"] = float(
        np.max(np.abs(np.asarray(f.phi) - np.asarray(ref_f.phi)))
    )
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
print(json.dumps(out))
