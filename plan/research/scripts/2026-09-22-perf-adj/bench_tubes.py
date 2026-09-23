"""F.5 step 2 costs: warm-started re-saturation and a vmapped multi-tube window.

Cyclone s-alpha deck at the QA example's resolution (8x8x16, Nl4/Nm8, dt 0.05,
rk3); tubes differ by a drift-profile scale standing in for field lines. Times
(synchronized, compile separated):

  cold/warm spin-up wall time for the cold (8000) and warm (2000) budgets
  window    1024-step window value+grad (drift scale, tprim) at geometry B
            from: the seed after 1x and 3x the cold budget, 3x plus one warm
            budget (spread along one trajectory), and A's 3x state after one
            and three warm budgets (warm starts)
  tubes     value+grad of T tubes: T single calls vs one jax.vmap call

Usage (repository root, PYTHONPATH=src): python bench_tubes.py --out rec.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gkx
from gkx.geometry import ensure_flux_tube_geometry_data
from gkx.runtime import (
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from gkx.workflows.runtime.toml import load_runtime_from_toml

ap = argparse.ArgumentParser()
ap.add_argument("--Nx", type=int, default=8)
ap.add_argument("--Ny", type=int, default=8)
ap.add_argument("--Nz", type=int, default=16)
ap.add_argument("--Nl", type=int, default=4)
ap.add_argument("--Nm", type=int, default=8)
ap.add_argument("--dt", type=float, default=0.05)
ap.add_argument("--cold", type=int, default=8000)
ap.add_argument("--warm", type=int, default=2000)
ap.add_argument("--window", type=int, default=1024)
ap.add_argument("--tubes", default="1,2,4")
ap.add_argument(
    "--shift", type=float, default=0.03, help="relative drift change A -> B"
)
ap.add_argument("--out", type=Path, default=None)
args = ap.parse_args()

cfg, _ = load_runtime_from_toml(
    Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml")
)
cfg = replace(
    cfg, grid=replace(cfg.grid, Nx=args.Nx, Ny=args.Ny, Nz=args.Nz, ntheta=None)
)
grid = gkx.build_spectral_grid(cfg.grid)
geom = ensure_flux_tube_geometry_data(build_runtime_geometry(cfg), grid.z)
params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
terms = build_runtime_term_config(cfg)
cplx = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64

rng = np.random.default_rng(3)
shape = (1, args.Nl, args.Nm, args.Ny, args.Nx, grid.z.size)
spec = np.fft.fft2(rng.standard_normal(shape), axes=(3, 4)) / (args.Ny * args.Nx)
seed = jnp.asarray(
    1.0e-3 * spec * np.asarray(grid.dealias_mask)[None, None, None, :, :, None], cplx
)


def tube(scale):
    return replace(
        geom, gb_profile=geom.gb_profile * scale, cv_profile=geom.cv_profile * scale
    )


def timed(fn, *a):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*a))
    return out, time.perf_counter() - t0


def spin(state, scale, steps):
    return gkx.integrate_nonlinear(
        state,
        grid,
        tube(scale),
        params,
        args.dt,
        steps,
        method="rk3",
        terms=terms,
        checkpoint=False,
        return_fields=False,
    )


def window(state, scale, rlt):
    return gkx.nonlinear_heat_flux_window(
        state,
        grid,
        tube(scale),
        replace(params, tprim=rlt),
        args.dt,
        args.window,
        terms=terms,
        method="rk3",
    )


record = {
    "jax": jax.__version__,
    "host": platform.node(),
    "devices": [str(d) for d in jax.devices()],
    "x64": bool(jax.config.jax_enable_x64),
    "args": vars(args) | {"out": None},
}
a, b = 1.0, 1.0 + args.shift
spin(seed, a, 10)  # compile the spin-up graph (steps is static: compile per length)
_, record["cold_first_s"] = timed(spin, seed, a, args.cold)
_, record["cold_s"] = timed(spin, seed, a, args.cold)
_, record["warm_s"] = timed(spin, seed, a, args.warm)
# Saturation references: B from the seed at the cold budget, at 3x it, and
# continued by one warm budget (the window's own trajectory-to-trajectory
# spread); warm starts from A's long state after one and three warm budgets.
states = {"coldB_1x": spin(seed, b, args.cold)}
states["coldB_3x"] = spin(states["coldB_1x"], b, 2 * args.cold)
states["coldB_3x_plus"] = spin(states["coldB_3x"], b, args.warm)
state_a = spin(seed, a, 3 * args.cold)
states["warmB_1"] = spin(state_a, b, args.warm)
states["warmB_3"] = spin(states["warmB_1"], b, 2 * args.warm)

vg = jax.value_and_grad(window, argnums=(1, 2))
rlt = jnp.asarray(params.tprim)
_, record["window_first_s"] = timed(vg, states["coldB_3x"], b, rlt)
_, record["window_s"] = timed(vg, states["coldB_3x"], b, rlt)
record["window"] = {}
for name, state in states.items():
    value, grads = vg(state, b, rlt)
    record["window"][name] = [
        float(value),
        float(grads[0]),
        *map(float, np.ravel(grads[1])),
    ]
    print(name, record["window"][name], flush=True)
state_b = states["coldB_3x"]

record["tubes"] = []
batched_vg = jax.vmap(vg, in_axes=(0, 0, None))
for count in map(int, args.tubes.split(",")):
    scales = jnp.asarray(
        1.0 + args.shift * np.linspace(-1.0, 1.0, count) if count > 1 else [b]
    )
    tube_states = jnp.stack([state_b] * count)
    _, first = timed(batched_vg, tube_states, scales, rlt)
    out, batched = timed(batched_vg, tube_states, scales, rlt)
    loop = 0.0
    loop_values = []
    for i in range(count):
        single, seconds = timed(vg, state_b, scales[i], rlt)
        loop += seconds
        loop_values.append(float(single[0]))
    row = {
        "tubes": count,
        "vmap_first_s": first,
        "vmap_s": batched,
        "loop_s": loop,
        "vmap_per_tube_s": batched / count,
        "loop_per_tube_s": loop / count,
        "vmap_vs_loop_value_rel": float(
            np.max(np.abs(np.asarray(out[0]) - loop_values) / np.abs(loop_values))
        ),
    }
    record["tubes"].append(row)
    print(json.dumps(row), flush=True)
print(json.dumps({k: v for k, v in record.items() if k != "tubes"}, indent=1))
if args.out is not None:
    args.out.write_text(json.dumps(record, indent=2))
