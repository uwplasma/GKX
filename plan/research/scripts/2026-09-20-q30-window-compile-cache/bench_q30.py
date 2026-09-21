"""Q30: compiles per call and wall time for the adjoint heat-flux window.

One tree per process, whichever gkx is on PYTHONPATH. Wraps
``jax._src.compiler.backend_compile_and_load`` to count XLA module compilations
and accumulate their time, then runs ``jax.value_and_grad`` of
``nonlinear_heat_flux_window`` --- the route an optimization loop takes ---
once to warm and ``--reps`` times more, forcing each to the host. A route whose
graph is cached compiles on the first call only; the route on ``main`` compiles
thirteen modules on every call.

Usage: python bench_q30.py OUT.json [--Nx 16 --Ny 16 --Nz 12 --Nl 2 --Nm 4
                                     --window-steps 6 --reps 3 --ky-mode full|half]
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
import jax.numpy as jnp
import numpy as np

import jax._src.compiler as _jc

import gkx
from gkx.core_ky_layout import FULL, HALF
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
for name, default in (
    ("--Nx", 16),
    ("--Ny", 16),
    ("--Nz", 12),
    ("--Nl", 2),
    ("--Nm", 4),
):
    ap.add_argument(name, type=int, default=default)
ap.add_argument("out", type=Path)
ap.add_argument("--window-steps", type=int, default=6)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--ky-mode", type=str, default="full", choices=("full", "half"))
args = ap.parse_args()

# --- compile counter -------------------------------------------------------
_COUNT = {"n": 0, "s": 0.0}
_orig = _jc.backend_compile_and_load


def _counting(*a, **k):
    t0 = time.perf_counter()
    try:
        return _orig(*a, **k)
    finally:
        _COUNT["n"] += 1
        _COUNT["s"] += time.perf_counter() - t0


_jc.backend_compile_and_load = _counting

cfg, _ = load_runtime_from_toml(
    Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml")
)
cfg = replace(cfg, grid=replace(cfg.grid, Nx=args.Nx, Ny=args.Ny, Nz=args.Nz))
geom = build_runtime_geometry(cfg)
grid = build_spectral_grid(
    apply_imported_geometry_grid_defaults(geom, cfg.grid),
    ky_layout=FULL if args.ky_mode == "full" else HALF,
)
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


def objective(tprim):
    return nonlinear_heat_flux_window(
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


fn = jax.value_and_grad(objective)
tprim = jnp.asarray(params.tprim)

report = {
    "gkx": str(Path(gkx.__file__).resolve().relative_to(Path.cwd())),
    "jax": jax.__version__,
    "platform": platform.system(),
    "platforms": os.environ.get("JAX_PLATFORMS", ""),
    "x64": bool(jax.config.jax_enable_x64),
    "grid": [args.Nx, args.Ny, args.Nz, args.Nl, args.Nm],
    "ky_mode": args.ky_mode,
    "ky_size": int(np.asarray(grid.ky).size),
    "window_steps": args.window_steps,
    "state_shape": list(g0.shape),
    "calls": [],
}

for rep in range(args.reps + 1):
    _COUNT["n"] = 0
    _COUNT["s"] = 0.0
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(tprim))
    wall = time.perf_counter() - t0
    value, gradient = out
    report["calls"].append(
        {
            "call": rep,
            "wall_s": wall,
            "compiles": _COUNT["n"],
            "compile_s": _COUNT["s"],
            "execution_s": wall - _COUNT["s"],
            "value": repr(np.asarray(value).tolist()),
            "grad": repr(np.asarray(gradient).tolist()),
        }
    )
    print(
        f"call {rep}: wall {wall:.3f}s compiles {_COUNT['n']} "
        f"compile {_COUNT['s']:.3f}s exec {wall - _COUNT['s']:.3f}s",
        flush=True,
    )

args.out.write_text(json.dumps(report, indent=2))
