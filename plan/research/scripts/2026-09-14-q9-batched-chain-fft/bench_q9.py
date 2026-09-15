"""Q9 timing kernels for whichever gkx is on PYTHONPATH (one tree per process).

Usage: python bench_q9.py OUT.json --Nx 64 --Ny 64 --Nz 24 --Nl 4 --Nm 8 [--reps 7]

Kernels (Cyclone nonlinear deck, complex64 state, cache/params as jit args):
  rhs         jit nonlinear_rhs_cached
  rhs_vjp     jit grad of Re<c, rhs(G)> wrt G
  scan_rk3    jit integrate_nonlinear rk3, --scan-steps fixed steps
  window_vjp  eager value_and_grad of nonlinear_heat_flux_window (rk3,
              checkpointed, --window-steps) wrt tprim
Each kernel: one compile call (timed separately), then --reps timed calls,
each forced to host (--window-reps for the eager window gradient). Records affinity, load averages, XLA_FLAGS and tree.
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

import gkx
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.solvers_nonlinear_state_integration import (
    integrate_nonlinear,
    nonlinear_heat_flux_window,
    nonlinear_rhs_cached,
)
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
for name, default in (
    ("--Nx", 64),
    ("--Ny", 64),
    ("--Nz", 24),
    ("--Nl", 4),
    ("--Nm", 8),
):
    ap.add_argument(name, type=int, default=default)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--window-reps", type=int, default=None)
ap.add_argument("--scan-steps", type=int, default=5)
ap.add_argument("--window-steps", type=int, default=6)
ap.add_argument("--kernels", type=str, default="rhs,rhs_vjp,scan_rk3,window_vjp")
args = ap.parse_args()

cfg, _ = load_runtime_from_toml(
    Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml")
)
cfg = replace(cfg, grid=replace(cfg.grid, Nx=args.Nx, Ny=args.Ny, Nz=args.Nz))
geom = build_runtime_geometry(cfg)
grid = build_spectral_grid(apply_imported_geometry_grid_defaults(geom, cfg.grid))
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
cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
rng = np.random.default_rng(0)
cot = jnp.asarray(
    rng.standard_normal(g0.shape) + 1j * rng.standard_normal(g0.shape), g0.dtype
)
dt = float(cfg.time.dt)


def rhs_fn(g, c, p):
    return nonlinear_rhs_cached(
        g, c, p, terms, compressed_real_fft=True, laguerre_mode="grid"
    )[0]


kernels = {
    "rhs": (jax.jit(rhs_fn), (g0, cache, params)),
    "rhs_vjp": (
        jax.jit(jax.grad(lambda g, c, p: jnp.real(jnp.vdot(cot, rhs_fn(g, c, p))))),
        (g0, cache, params),
    ),
    "scan_rk3": (
        jax.jit(
            lambda g, c, p: integrate_nonlinear(
                g,
                grid,
                geom,
                p,
                dt=dt,
                steps=args.scan_steps,
                method="rk3",
                cache=c,
                terms=terms,
                compressed_real_fft=True,
                laguerre_mode="grid",
                return_fields=False,
            )[0]
        ),
        (g0, cache, params),
    ),
    # Eager value_and_grad, as callers use it: the window rebuilds its linked
    # cache from the geometry, which an outer jit would trace (twist-shift
    # jtwist is an integer read off the shear and refuses a traced geometry).
    "window_vjp": (
        jax.value_and_grad(
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
        ),
        (jnp.asarray(params.tprim),),
    ),
}


def force(value):
    for leaf in jax.tree_util.tree_leaves(value):
        np.asarray(leaf)


report = {
    "gkx": gkx.__file__,
    "jax": jax.__version__,
    "host": platform.node(),
    "xla_flags": os.environ.get("XLA_FLAGS", ""),
    "affinity": sorted(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else None,
    "load_before": os.getloadavg(),
    "grid": [args.Nx, args.Ny, args.Nz, args.Nl, args.Nm],
    "state_shape": list(g0.shape),
    "kernels": {},
}
for name in [k for k in args.kernels.split(",") if k]:
    fn, fn_args = kernels[name]
    t0 = time.perf_counter()
    force(fn(*fn_args))
    compile_s = time.perf_counter() - t0
    reps = []
    n_reps = (
        args.window_reps if name == "window_vjp" and args.window_reps else args.reps
    )
    for _ in range(n_reps):
        t0 = time.perf_counter()
        force(fn(*fn_args))
        reps.append(time.perf_counter() - t0)
    report["kernels"][name] = {
        "compile_and_first_s": compile_s,
        "reps_s": reps,
        "median_s": float(np.median(reps)),
        "min_s": float(np.min(reps)),
    }
    print(
        name,
        f"compile {compile_s:.2f}s median {np.median(reps) * 1e3:.1f} ms min {np.min(reps) * 1e3:.1f} ms",
        flush=True,
    )
    report["load_after"] = os.getloadavg()
    args.out.write_text(json.dumps(report, indent=2))
