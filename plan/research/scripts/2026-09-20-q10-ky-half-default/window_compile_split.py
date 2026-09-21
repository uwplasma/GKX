"""Q10: split the eager window gradient's wall time into compile and execution.

Usage: python window_compile_split.py OUT.json --ky-layout full|half [--reps 3]

The benchmark-host A/B (``timing_table.py``) found one kernel that moves the wrong way:
the eager checkpointed heat-flux window gradient, ``window_vjp``, at 2.31x on
the half axis at 32x32x24.  That kernel is not jitted as a whole, so every call
re-enters JAX's dispatch and may compile.  This script asks XLA how long each
call spent compiling, through the ``/jax/core/compile/*`` duration events, so
the wall time can be split into what compilation costs and what execution
costs.

It builds the kernel exactly as ``bench_ky_layout.py`` does (same deck, same
grid, same seed, same window) and times ``--reps`` calls after a first call.
For each call it records the wall time, the number of backend compilations,
and the seconds spent in them; execution is the remainder.  If every call
recompiles, the compile count per call is constant and nonzero, and that is
the defect #264 addresses, independent of the layout.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.monitoring
import jax.numpy as jnp
import numpy as np

from gkx.solvers_nonlinear_state_integration import nonlinear_heat_flux_window
from tools.profiling.profile_runtime_kernels import (
    _build_initial_condition,
    _select_nonlinear_mode_indices,
    apply_imported_geometry_grid_defaults,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
    build_spectral_grid,
    load_runtime_from_toml,
)

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--ky-layout", choices=("full", "half"), required=True)
ap.add_argument("--Nx", type=int, default=32)
ap.add_argument("--Ny", type=int, default=32)
ap.add_argument("--Nz", type=int, default=24)
ap.add_argument("--Nl", type=int, default=4)
ap.add_argument("--Nm", type=int, default=8)
ap.add_argument("--window-steps", type=int, default=6)
ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()

EVENTS = (
    "/jax/core/compile/backend_compile_duration",
    "/jax/core/compile/jaxpr_to_mlir_module_duration",
    "/jax/core/compile/jaxpr_trace_duration",
)
ledger: dict[str, list[float]] = {name: [] for name in EVENTS}


def _listen(event: str, duration: float, **_: object) -> None:
    if event in ledger:
        ledger[event].append(float(duration))


jax.monitoring.register_event_duration_secs_listener(_listen)
LOAD_BEFORE = os.getloadavg()

cfg, _ = load_runtime_from_toml(
    Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml")
)
cfg = replace(cfg, grid=replace(cfg.grid, Nx=args.Nx, Ny=args.Ny, Nz=args.Nz))
geom = build_runtime_geometry(cfg)
grid = build_spectral_grid(
    apply_imported_geometry_grid_defaults(geom, cfg.grid), ky_layout=args.ky_layout
)
if grid.ky_layout != args.ky_layout:
    raise SystemExit(f"grid reports {grid.ky_layout}, asked for {args.ky_layout}")
params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
terms = build_runtime_term_config(cfg)
ky_i, kx_i = _select_nonlinear_mode_indices(
    grid,
    ky_target=0.3,
    kx_target=None,
    use_dealias_mask=bool(cfg.time.nonlinear_dealias),
)
g0 = jnp.asarray(
    _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=ky_i,
        kx_index=kx_i,
        Nl=args.Nl,
        Nm=args.Nm,
        nspecies=len(cfg.species),
    )
)
g0 = g0 / jnp.maximum(jnp.max(jnp.abs(g0)), 1e-30) * 1e-2
dt = float(cfg.time.dt)

window_vjp = jax.value_and_grad(
    lambda tprim: nonlinear_heat_flux_window(
        g0,
        grid,
        geom,
        replace(params, tprim=tprim),
        dt,
        args.window_steps,
        terms=terms,
        method="rk3",
        checkpoint=True,
        compressed_real_fft=True,
        laguerre_mode="grid",
    )
)
tprim = jnp.asarray(params.tprim)


def _call() -> dict[str, float]:
    marks = {name: len(values) for name, values in ledger.items()}
    t0 = time.perf_counter()
    for leaf in jax.tree_util.tree_leaves(window_vjp(tprim)):
        np.asarray(leaf)
    wall = time.perf_counter() - t0
    new = {name: ledger[name][marks[name] :] for name in EVENTS}
    backend = new["/jax/core/compile/backend_compile_duration"]
    lowering = sum(new["/jax/core/compile/jaxpr_to_mlir_module_duration"])
    tracing = sum(new["/jax/core/compile/jaxpr_trace_duration"])
    compile_s = sum(backend) + lowering + tracing
    return {
        "wall_s": wall,
        "backend_compiles": len(backend),
        "backend_compile_s": sum(backend),
        "lowering_s": lowering,
        "tracing_s": tracing,
        "compile_total_s": compile_s,
        "execution_s": wall - compile_s,
    }


first = _call()
reps = [_call() for _ in range(args.reps)]
report = {
    "gkx": "src/gkx/__init__.py",
    "jax": jax.__version__,
    "host": platform.system(),
    "cpu_count": os.cpu_count(),
    "affinity": sorted(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else None,
    "load_before": LOAD_BEFORE,
    "load_after": os.getloadavg(),
    "xla_flags": os.environ.get("XLA_FLAGS", ""),
    "ky_layout": args.ky_layout,
    "grid": [args.Nx, args.Ny, args.Nz, args.Nl, args.Nm],
    "window_steps": args.window_steps,
    "state_shape": list(g0.shape),
    "first_call": first,
    "reps": reps,
    "median": {key: float(np.median([r[key] for r in reps])) for key in reps[0]},
}
args.out.write_text(json.dumps(report, indent=2))
print(json.dumps({"ky_layout": args.ky_layout, "median": report["median"]}, indent=2))
