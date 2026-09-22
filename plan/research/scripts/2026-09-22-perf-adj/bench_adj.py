"""PERF-ADJ harness: split the adjoint-window cost of gkx.nonlinear_heat_flux_window.

One layout per process. Reports, for value_and_grad wrt tprim:
  e2e cold/steady wall, compiles per call,
  prep-only (host-side cache build under JVP) wall,
  jitted window VJP alone (cache as operands) compile/exec and memory analysis,
  forward-only window exec.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jax._src.compiler as _jc

ap = argparse.ArgumentParser()
for name, default in (("--Nx", 32), ("--Ny", 32), ("--Nz", 24), ("--Nl", 4), ("--Nm", 8)):
    ap.add_argument(name, type=int, default=default)
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--ky", choices=("full", "half"), default="full")
ap.add_argument("--method", default="rk3")
ap.add_argument("--parts", default="e2e,window,fwd")
ap.add_argument("--out", type=Path, default=None)
ap.add_argument("--profile-dir", type=Path, default=None)
ap.add_argument("--hlo-dir", type=Path, default=None)
args = ap.parse_args()

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

import gkx  # noqa: E402
from gkx.core_ky_layout import FULL, HALF  # noqa: E402
from gkx import solvers_nonlinear_state_integration as sni  # noqa: E402
from gkx.geometry import ensure_flux_tube_geometry_data  # noqa: E402
from gkx.operators.linear.cache_builder import build_linear_cache  # noqa: E402
from gkx.operators.linear.linked import mask_supplied_state  # noqa: E402
from gkx.operators.moments import fieldline_quadrature_weights  # noqa: E402
from gkx.operators.nonlinear.projection import hermitian_projector_signature  # noqa: E402

sys.path.insert(0, os.getcwd())
from tools.profiling.profile_runtime_kernels import (  # noqa: E402
    _build_initial_condition,
    _select_nonlinear_mode_indices,
    apply_imported_geometry_grid_defaults,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
    build_spectral_grid,
    load_runtime_from_toml,
)

cfg, _ = load_runtime_from_toml(
    Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml")
)
cfg = replace(cfg, grid=replace(cfg.grid, Nx=args.Nx, Ny=args.Ny, Nz=args.Nz))
geom = build_runtime_geometry(cfg)
grid = build_spectral_grid(
    apply_imported_geometry_grid_defaults(geom, cfg.grid),
    ky_layout=FULL if args.ky == "full" else HALF,
)
params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
terms = build_runtime_term_config(cfg)
ky_i, kx_i = _select_nonlinear_mode_indices(
    grid, ky_target=0.3, kx_target=None, use_dealias_mask=bool(cfg.time.nonlinear_dealias)
)
g0 = jnp.asarray(
    _build_initial_condition(
        grid, geom, cfg, ky_index=ky_i, kx_index=kx_i, Nl=args.Nl, Nm=args.Nm,
        nspecies=len(cfg.species),
    )
)
g0 = g0 / jnp.maximum(jnp.max(jnp.abs(g0)), 1e-30) * 1e-2
dt = float(cfg.time.dt)
steps = args.steps


def objective(tprim):
    return sni.nonlinear_heat_flux_window(
        g0, grid, geom, replace(params, tprim=tprim), dt, steps,
        terms=terms, method=args.method, checkpoint=True,
        compressed_real_fft=True, laguerre_mode="grid", divergence_knee_steps=None,
    )


tprim0 = jnp.asarray(params.tprim)
report = {
    "gkx": gkx.__file__, "jax": jax.__version__, "host": platform.node(),
    "x64": bool(jax.config.jax_enable_x64),
    "grid": [args.Nx, args.Ny, args.Nz, args.Nl, args.Nm], "ky": args.ky,
    "state_shape": list(g0.shape), "steps": steps, "method": args.method,
}


def timed(fn, *a, reps=args.reps):
    rows = []
    for rep in range(reps + 1):
        _COUNT["n"] = 0
        _COUNT["s"] = 0.0
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn(*a))
        wall = time.perf_counter() - t0
        rows.append({"wall": wall, "compiles": _COUNT["n"], "compile_s": _COUNT["s"]})
    steady = [r["wall"] for r in rows[1:]]
    return out, {"cold": rows[0], "steady_median": statistics.median(steady),
                 "steady": steady, "steady_compiles": [r["compiles"] for r in rows[1:]]}


parts = args.parts.split(",")
if "e2e" in parts:
    out, stats = timed(jax.value_and_grad(objective), tprim0)
    stats["value"] = repr(np.asarray(out[0]).tolist())
    stats["grad"] = repr(np.asarray(out[1]).tolist())
    report["e2e"] = stats
    print("e2e", json.dumps(stats), flush=True)

# --- window VJP alone, cache/params as operands of one outer jit -----------
count = tail = steps
term_cfg = terms
geometry = ensure_flux_tube_geometry_data(geom, grid.z)
Nl, Nm = g0.shape[1:3]
cache = build_linear_cache(grid, geometry, params, Nl=Nl, Nm=Nm)
_vf, flux_factor = fieldline_quadrature_weights(geometry, grid)
sig = hermitian_projector_signature(np.asarray(grid.ky), int(np.asarray(grid.kx).size))
init = jax.lax.stop_gradient(mask_supplied_state(g0, cache))
heat0 = jnp.zeros((), dtype=jnp.result_type(jnp.real(init), flux_factor))
static = dict(dt=dt, count=count, tail=tail, method=args.method, term_cfg=term_cfg,
              compressed_real_fft=True, laguerre_mode="grid", collision_operator=None,
              checkpoint=True, projector_signature=sig)


def window_of_params(p, c):
    return sni._nonlinear_heat_flux_window_total(
        init, c, grid, p, flux_factor, heat0, jnp.arange(count), **static)


if "window" in parts:
    # differentiate wrt params (tprim) with the cache fixed: isolates the scan VJP
    def wobj(tp):
        return window_of_params(replace(params, tprim=tp), cache)

    f = jax.jit(jax.value_and_grad(wobj))
    t0 = time.perf_counter()
    lowered = f.lower(tprim0)
    t_lower = time.perf_counter() - t0
    t0 = time.perf_counter()
    compiled = lowered.compile()
    t_compile = time.perf_counter() - t0
    ma = compiled.memory_analysis()
    if args.hlo_dir is not None:
        args.hlo_dir.mkdir(parents=True, exist_ok=True)
        (args.hlo_dir / f"window_vjp_{args.ky}.hlo.txt").write_text(compiled.as_text())
    out, stats = timed(compiled, tprim0)
    stats.update(lower_s=t_lower, compile_s=t_compile,
                 temp_bytes=int(ma.temp_size_in_bytes), arg_bytes=int(ma.argument_size_in_bytes),
                 value=repr(np.asarray(out[0]).tolist()), grad=repr(np.asarray(out[1]).tolist()))
    report["window_vjp"] = stats
    print("window_vjp", json.dumps(stats), flush=True)
    if args.profile_dir is not None:
        jax.profiler.start_trace(str(args.profile_dir / f"vjp_{args.ky}"))
        for _ in range(2):
            jax.block_until_ready(compiled(tprim0))
        jax.profiler.stop_trace()

if "fwd" in parts:
    f = jax.jit(lambda tp: window_of_params(replace(params, tprim=tp), cache))
    compiled = f.lower(tprim0).compile()
    ma = compiled.memory_analysis()
    out, stats = timed(compiled, tprim0)
    stats.update(temp_bytes=int(ma.temp_size_in_bytes), value=repr(np.asarray(out).tolist()))
    report["window_fwd"] = stats
    print("window_fwd", json.dumps(stats), flush=True)

if "prep" in parts:
    # host-side preparation under JVP: cost of rebuilding the cache from traced tprim
    def prep(tp):
        p = replace(params, tprim=tp)
        c = build_linear_cache(grid, geometry, p, Nl=Nl, Nm=Nm)
        return jax.tree_util.tree_reduce(
            lambda a, b: a + b,
            [jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(c)
             if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.inexact)])

    _, stats = timed(jax.value_and_grad(prep), tprim0)
    report["prep_grad"] = stats
    print("prep_grad", json.dumps(stats), flush=True)

report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
print("peak_rss_mib", report["peak_rss_mib"])
if args.out is not None:
    args.out.write_text(json.dumps(report, indent=2))
