#!/usr/bin/env python3
"""Cost of the collision term alone: Dougherty vs linearized Landau.

    python rhs_cost.py <Nl> <Nm> [Ny] [Nz]

Cyclone s-alpha grid (Nx=8), one kinetic species, nu = 0.01. Times the jitted
collision term for each kernel, median of 20
calls, models alternated A/B/A/B. Prints RESULT lines.
"""

import json
import platform
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from gkx.config import GridConfig
from gkx.core_grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.collisions import apply_linearized_landau
from gkx.operators.linear.dissipation import collisions_contribution
from gkx.operators.linear.params import linear_params_for_geometry

nl, nm = int(sys.argv[1]), int(sys.argv[2])
ny = int(sys.argv[3]) if len(sys.argv) > 3 else 16
nz = int(sys.argv[4]) if len(sys.argv) > 4 else 32
grid = build_spectral_grid(GridConfig(Nx=8, Ny=ny, Nz=nz, Lx=62.8, Ly=62.8))
geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778)
shape = (1, nl, nm, grid.ky.size, grid.kx.size, grid.z.size)
state = jnp.asarray(np.random.default_rng(0).normal(size=shape), jnp.complex128)
caches = {}
for model in ("dougherty", "landau"):
    params = linear_params_for_geometry(geom, nu=0.01, collision_model=model)
    start = time.perf_counter()
    caches[model] = (build_linear_cache(grid, geom, params, nl, nm), params)
    caches[model] += (time.perf_counter() - start,)
cache_d = caches["dougherty"][0]
nu = jnp.asarray([0.01])
term = {
    "dougherty": jax.jit(
        lambda h, jl, jlb, b, lb: collisions_contribution(
            h, G=h, Jl=jl, JlB=jlb, b=b, nu=nu, lb_lam=lb, weight=jnp.asarray(1.0)
        )
    ),
    "landau": jax.jit(
        lambda h, pairs: apply_linearized_landau(
            h, pairs, nu=nu, density=jnp.ones(1), charge=jnp.ones(1)
        )
    ),
}
args = {
    "dougherty": (cache_d.Jl, cache_d.JlB, cache_d.b, cache_d.lb_lam),
    "landau": (caches["landau"][0].landau_pairs,),
}


def median_ms(fn):
    fn().block_until_ready()
    times = []
    for _ in range(20):
        start = time.perf_counter()
        fn().block_until_ready()
        times.append(time.perf_counter() - start)
    return round(1e3 * float(np.median(times)), 3)


for model in ("dougherty", "landau") * 2:
    cache, params, build = caches[model]
    print(
        "RESULT",
        json.dumps(
            dict(
                model=model,
                Nl=nl,
                Nm=nm,
                shape=list(shape),
                collision_ms=median_ms(lambda: term[model](state, *args[model])),
                cache_build_s=round(build, 2),
                host=f"{platform.system()}-{platform.machine()}-{jax.devices()[0].platform}",
            )
        ),
        flush=True,
    )
