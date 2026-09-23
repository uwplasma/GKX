"""SHARD-PAD (plan F.6): does a ky-sharded nonlinear run partition its scan?

usage (from a GKX checkout, JAX 0.10.2, CPU):
    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        PYTHONPATH=src python plan/research/scripts/2026-09-22-shard-pad/ky_partition_probe.py \
        > plan/research/scripts/2026-09-22-shard-pad/ky_partition_probe.json

Two questions, answered on the small periodic deck of
tests/unit/parallel/test_parallel_nonlinear_routing.py (Ny = 8, so the stored
ky extent is 8 on "full" and Nyc = 5 on "half"), for 2 and 4 devices:

1. runtime `[parallel] strategy="shard_map" axis="ky"`: what sharding does
   the compiled diagnostic scan receive?  The state is placed on the ky mesh
   before the run; the probe records the sharding of `prepared.G0`, the array
   the scan's jit is called with.
2. `[time] state_sharding="ky"` (gkx.parallel.integrators, which constrains
   the state to the ky split at every step): does it run?

No timing is taken; every output is a sharding or an exception string. Each
case runs in its own process (see main).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests" / "unit" / "parallel"))

import test_parallel_nonlinear_routing as deck  # noqa: E402

import gkx.solvers_nonlinear_diagnostics as diagnostics  # noqa: E402
import gkx.solvers_time_runners as runners  # noqa: E402
import gkx.workflows.nonlinear as workflow  # noqa: E402
from gkx.config import RuntimeParallelConfig  # noqa: E402
from gkx.parallel.state import resolve_state_sharding  # noqa: E402
from gkx.runtime import run_runtime_nonlinear  # noqa: E402


def _describe(sharding) -> dict:
    return {
        "spec": str(getattr(sharding, "spec", sharding)),
        "fully_replicated": bool(sharding.is_fully_replicated),
        "n_devices": len(sharding.device_set),
    }


def runtime_route(layout: str, n_dev: int) -> dict:
    seen: dict = {}
    original_place = workflow.shard_nonlinear_state
    original_scan = diagnostics._run_explicit_diagnostic_scan_and_finalize

    def place(state, plan):
        placed = original_place(state, plan)
        seen["placed_input"] = _describe(placed.sharding)
        return placed

    def scan(prepared, *args, **kwargs):
        seen["scan_input"] = _describe(prepared.G0.sharding)
        return original_scan(prepared, *args, **kwargs)

    workflow.shard_nonlinear_state = place
    diagnostics._run_explicit_diagnostic_scan_and_finalize = scan
    try:
        cfg = deck._nonlinear_cfg(
            RuntimeParallelConfig(
                strategy="shard_map",
                axis="ky",
                num_devices=n_dev,
                strict_identity=False,
            ),
            ky_layout=layout,
        )
        result = run_runtime_nonlinear(cfg, return_state=True, **deck._RUN_KWARGS)
        seen["state_shape"] = list(np.asarray(result.state).shape)
    except Exception as exc:  # recorded, not raised: the answer is the error
        seen["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:240]}"
    finally:
        workflow.shard_nonlinear_state = original_place
        diagnostics._run_explicit_diagnostic_scan_and_finalize = original_scan
    return seen


def config_route(layout: str, n_dev: int) -> dict:
    captured: dict = {}
    original_once = workflow._run_once

    def spy(cfg, ctx, *args, **kwargs):
        captured["ctx"] = ctx
        return original_once(cfg, ctx, *args, **kwargs)

    workflow._run_once = spy
    try:
        cfg = deck._nonlinear_cfg(ky_layout=layout)
        run_runtime_nonlinear(cfg, **deck._RUN_KWARGS)
    finally:
        workflow._run_once = original_once
    ctx = captured["ctx"]
    devices = jax.devices()[:n_dev]
    original_resolve = runners.resolve_state_sharding
    runners.resolve_state_sharding = lambda G0, spec: resolve_state_sharding(
        G0, spec, devices=devices
    )
    time_cfg = replace(
        cfg.time, dt=ctx.dt, t_max=ctx.dt * ctx.steps, state_sharding="ky"
    )
    out: dict = {"ky_extent": int(ctx.G0.shape[-3])}
    try:
        state, *_ = runners.integrate_nonlinear_from_config(
            jnp.array(ctx.G0, copy=True),
            ctx.grid,
            ctx.geom,
            ctx.params,
            time_cfg,
            terms=ctx.terms,
        )
        out["final_state"] = _describe(state.sharding)
        # Dispatch is asynchronous: an execution failure surfaces only when
        # the value is read, so read it before calling the case a success.
        np.asarray(state)
        out["executed"] = True
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:240]}"
    finally:
        runners.resolve_state_sharding = original_resolve
    return out


def main() -> None:
    """Run every case in a fresh process, so no case sees another's compiles.

    Isolation keeps one case's compiled executables and failures out of the
    next case's answer.
    """

    if len(sys.argv) == 4:
        route, layout, n_dev = sys.argv[1], sys.argv[2], int(sys.argv[3])
        probe = runtime_route if route == "runtime_ky_route" else config_route
        json.dump(probe(layout, n_dev), sys.stdout)
        return
    record: dict = {
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": len(jax.devices()),
        "runtime_ky_route": {},
        "time_state_sharding_ky": {},
    }
    for route in ("runtime_ky_route", "time_state_sharding_ky"):
        for layout in ("full", "half"):
            for n_dev in (2, 4):
                done = subprocess.run(
                    [sys.executable, __file__, route, layout, str(n_dev)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                lines = done.stdout.strip().splitlines()
                record[route][f"{layout}/{n_dev}"] = (
                    json.loads(lines[-1])
                    if lines
                    else {"error": f"exit {done.returncode}: {done.stderr[-240:]}"}
                )
    json.dump(record, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
