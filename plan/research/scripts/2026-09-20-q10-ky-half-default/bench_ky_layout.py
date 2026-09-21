"""Q10 wall-clock kernels for the ``ky`` layout A/B (plan 5.3 N3).

Usage: python bench_ky_layout.py OUT.json --ky-layout full|half [--Nx 64 ...]

Both arms run **the same tree**: the layout is an argument to
``build_spectral_grid``, not a code difference, so nothing here can be
explained by one arm having different source.  That is the whole point --
#259 established that the half layout wins every load-independent measure,
and the only open question is whether the extra strided in-fusion reads it
buys (``fused_interior_bytes`` +146/+171 per cent on rk3) cost wall clock.

Kernels are Q9's verbatim (``bench_q9.py``), so the two campaigns are
comparable:

  rhs         jit ``nonlinear_rhs_cached``
  rhs_vjp     jit grad of ``Re<c, rhs(G)>`` wrt ``G``
  scan_rk3    jit ``integrate_nonlinear`` rk3, ``--scan-steps`` fixed steps
  window_vjp  eager ``value_and_grad`` of ``nonlinear_heat_flux_window``
              (rk3, checkpointed, ``--window-steps``) wrt ``tprim``

Each kernel: one compile call (timed separately), then ``--reps`` timed
calls, each forced to host.  Affinity, load averages, XLA_FLAGS, the tree and
the realised state shape are recorded so a reader can check the arm was what
it claims to be.

The cotangent and the initial state are seeded identically in both arms and
are *not* comparable element by element across layouts (they have different
shapes); this script measures time, and identity is gated elsewhere
(``run_identity.sh``).
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
ap.add_argument("--ky-layout", choices=("full", "half"), required=True)
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
grid = build_spectral_grid(
    apply_imported_geometry_grid_defaults(geom, cfg.grid), ky_layout=args.ky_layout
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
    "gkx": "src/gkx/__init__.py",
    "jax": jax.__version__,
    "platform": platform.system(),
    "ky_layout": args.ky_layout,
    "grid_ky_layout": grid.ky_layout,
    "xla_flags": os.environ.get("XLA_FLAGS", ""),
    "affinity": sorted(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else None,
    "load_before": os.getloadavg(),
    "grid": [args.Nx, args.Ny, args.Nz, args.Nl, args.Nm],
    "state_shape": list(g0.shape),
    "kernels": {},
}
if report["grid_ky_layout"] != args.ky_layout:
    raise SystemExit(
        f"grid reports layout {report['grid_ky_layout']}, asked for {args.ky_layout}"
    )
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
        f"compile {compile_s:.2f}s median {np.median(reps) * 1e3:.1f} ms "
        f"min {np.min(reps) * 1e3:.1f} ms",
        flush=True,
    )
    report["load_after"] = os.getloadavg()
    args.out.write_text(json.dumps(report, indent=2))
