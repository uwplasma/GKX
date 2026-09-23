"""A/B/A/B the adjoint heat-flux window VJP across step-policy and checkpoint variants.

All variants are compiled in one process and then called in alternating rounds,
so host contention drifts hit every arm alike. Variants:

  main      step without stage barriers, nested checkpoint schedule (origin/main)
  barrier   stage barriers, nested schedule
  block     stage barriers, block-only schedule under the default budget
  stage     barriers on stage states only (not on stage derivatives), nested
  mainblock no barriers, block-only schedule

Usage (from the repository root, PYTHONPATH=src):
  python bench_ab.py --Nx 16 --Ny 16 --Nz 16 --steps 64 --ky half --rounds 3
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ap = argparse.ArgumentParser()
for name, default in (("--Nx", 16), ("--Ny", 16), ("--Nz", 16), ("--Nl", 4), ("--Nm", 8)):
    ap.add_argument(name, type=int, default=default)
ap.add_argument("--steps", type=int, default=64)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--ky", choices=("full", "half"), default="full")
ap.add_argument("--method", default="rk3")
ap.add_argument("--variants", default="main,barrier,block")
ap.add_argument("--budget", type=float, default=None, help="bytes for the block arm")
ap.add_argument("--out", type=Path, default=None)
args = ap.parse_args()

from gkx import solvers_nonlinear_explicit as sne  # noqa: E402
from gkx import solvers_nonlinear_state_integration as sni  # noqa: E402
from gkx.core_ky_layout import FULL, HALF  # noqa: E402
from gkx.core_grid import build_spectral_grid  # noqa: E402
from gkx.geometry import apply_imported_geometry_grid_defaults, ensure_flux_tube_geometry_data  # noqa: E402
from gkx.operators.linear.cache_builder import build_linear_cache  # noqa: E402
from gkx.operators.linear.linked import mask_supplied_state  # noqa: E402
from gkx.operators.moments import fieldline_quadrature_weights  # noqa: E402
from gkx.operators.nonlinear.projection import hermitian_projector_signature  # noqa: E402
from gkx.runtime import build_runtime_geometry, build_runtime_linear_params, build_runtime_term_config  # noqa: E402
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402

cfg, _ = load_runtime_from_toml(Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml"))
cfg = replace(cfg, grid=replace(cfg.grid, Nx=args.Nx, Ny=args.Ny, Nz=args.Nz, ntheta=None))
geom = build_runtime_geometry(cfg)
grid = build_spectral_grid(apply_imported_geometry_grid_defaults(geom, cfg.grid),
                           ky_layout=FULL if args.ky == "full" else HALF)
params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
terms = build_runtime_term_config(cfg)
geometry = ensure_flux_tube_geometry_data(geom, grid.z)
cache = build_linear_cache(grid, geometry, params, Nl=args.Nl, Nm=args.Nm)
real = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
cplx = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64

# A deterministic real-field state: draw on the full axis, keep the reality
# condition by construction, then store the layout's rows. Values match across
# layouts up to the storage of the negative rows.
rng = np.random.default_rng(7)
ny, nx, nz = args.Ny, args.Nx, int(grid.z.size)
shape_real = (1, args.Nl, args.Nm, ny, nx, nz)
real_space = rng.standard_normal(shape_real)
spec = np.fft.fft2(real_space, axes=(3, 4)) / (ny * nx)
mask = np.asarray(grid.dealias_mask)
if args.ky == "half":
    spec = spec[:, :, :, : ny // 2 + 1]
spec = spec * mask[None, None, None, :, :, None]
g0 = jnp.asarray(1.0e-3 * spec, dtype=cplx)
init = jax.lax.stop_gradient(mask_supplied_state(g0, cache))
_vf, flux_factor = fieldline_quadrature_weights(geometry, grid)
sig = hermitian_projector_signature(np.asarray(grid.ky), int(np.asarray(grid.kx).size))
heat0 = jnp.zeros((), dtype=real)
count = args.steps
tprim0 = jnp.asarray(params.tprim)

original_advance = sne.advance_explicit_nonlinear_state


def unbarriered_advance(G, dG, dt_local, *, method, rhs_fn, project_state, state_dtype):
    G_new = sne._explicit_stage_update(G, dG, dt_local, method=method, rhs_fn=rhs_fn,
                                       project_state=project_state)
    return jnp.asarray(project_state(G_new), dtype=state_dtype)


def stage_only_advance(G, dG, dt_local, *, method, rhs_fn, project_state, state_dtype):
    def projection(state):
        return jax.lax.optimization_barrier(project_state(state))

    G_new = sne._explicit_stage_update(G, dG, dt_local, method=method, rhs_fn=rhs_fn,
                                       project_state=projection)
    return jnp.asarray(project_state(G_new), dtype=state_dtype)


def compile_variant(name):
    jax.clear_caches()
    sni.advance_explicit_nonlinear_state = {
        "main": unbarriered_advance, "mainblock": unbarriered_advance,
        "stage": stage_only_advance}.get(name, original_advance)
    budget = None
    if name in ("block", "mainblock"):
        budget = int(args.budget) if args.budget else sni.ADJOINT_MEMORY_BUDGET_BYTES

    def objective(tp):
        return sni._nonlinear_heat_flux_window_total(
            init, cache, grid, replace(params, tprim=tp), flux_factor, heat0,
            jnp.arange(count), dt=float(cfg.time.dt), count=count, tail=count,
            method=args.method, term_cfg=terms, compressed_real_fft=True,
            laguerre_mode="grid", collision_operator=None, checkpoint=True,
            projector_signature=sig, memory_budget_bytes=budget)

    t0 = time.perf_counter()
    compiled = jax.jit(jax.value_and_grad(objective)).lower(tprim0).compile()
    compile_s = time.perf_counter() - t0
    fwd = jax.jit(objective).lower(tprim0).compile()
    sni.advance_explicit_nonlinear_state = original_advance
    return compiled, fwd, compile_s, int(compiled.memory_analysis().temp_size_in_bytes)


names = args.variants.split(",")
arms = {n: compile_variant(n) for n in names}
results = {n: {"compile_s": arms[n][2], "temp_bytes": arms[n][3], "vjp": [], "fwd": []} for n in names}
for n in names:  # warm
    out = jax.block_until_ready(arms[n][0](tprim0))
    results[n]["value"] = float(out[0])
    results[n]["grad"] = float(np.asarray(out[1]).reshape(-1)[0])
    jax.block_until_ready(arms[n][1](tprim0))
for _ in range(args.rounds):
    for n in names:
        t0 = time.perf_counter()
        jax.block_until_ready(arms[n][0](tprim0))
        results[n]["vjp"].append(time.perf_counter() - t0)
    for n in names:
        t0 = time.perf_counter()
        jax.block_until_ready(arms[n][1](tprim0))
        results[n]["fwd"].append(time.perf_counter() - t0)
ref = names[0]
for n in names:
    r = results[n]
    r["vjp_median"] = statistics.median(r["vjp"])
    r["fwd_median"] = statistics.median(r["fwd"])
    r["value_rel_to_" + ref] = abs(r["value"] - results[ref]["value"]) / abs(results[ref]["value"])
    r["grad_rel_to_" + ref] = abs(r["grad"] - results[ref]["grad"]) / abs(results[ref]["grad"])
    print(f"{args.ky} {n:8s} vjp {r['vjp_median']:8.3f}s fwd {r['fwd_median']:7.3f}s "
          f"temp {r['temp_bytes']/2**20:8.1f}MiB compile {r['compile_s']:6.1f}s "
          f"dval {r['value_rel_to_' + ref]:.1e} dgrad {r['grad_rel_to_' + ref]:.1e}", flush=True)
record = {"jax": jax.__version__, "host": platform.node(), "devices": [str(d) for d in jax.devices()],
          "x64": bool(jax.config.jax_enable_x64), "args": vars(args) | {"out": None},
          "state_shape": list(g0.shape), "results": results}
if args.out is not None:
    args.out.write_text(json.dumps(record, indent=2))
