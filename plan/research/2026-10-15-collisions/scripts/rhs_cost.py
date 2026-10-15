#!/usr/bin/env python3
"""Share of the collision term in one linear RHS: Dougherty vs linearized Landau.

    python rhs_cost.py <Nl> <Nm> <species: 1|2> [Ny] [Nz]

Cyclone s-alpha grid (Nx=8), electrostatic. One species (adiabatic electrons) or
ions plus kinetic electrons (m_e/m_i = 1/1836). Each model gets one jitted
function, warmed once; the models are then timed interleaved A/B/A/B with
block_until_ready, median of 20 calls each round. Prints RESULT lines.
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
from gkx.operators.linear.params import linear_params_for_geometry
from gkx.terms.assembly import assemble_rhs_cached
from gkx.terms.config import TermConfig

nl, nm, ns = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
ny = int(sys.argv[4]) if len(sys.argv) > 4 else 16
nz = int(sys.argv[5]) if len(sys.argv) > 5 else 32
grid = build_spectral_grid(GridConfig(Nx=8, Ny=ny, Nz=nz, Lx=62.8, Ly=62.8))
geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778)
species = dict(
    charge_sign=jnp.asarray([1.0, -1.0][:ns]),
    density=jnp.ones(ns),
    mass=jnp.asarray([1.0, 1.0 / 1836.0][:ns]),
    temp=jnp.ones(ns),
    tz=jnp.asarray([1.0, -1.0][:ns]),
    vth=jnp.asarray([1.0, 1836.0**0.5][:ns]),
    rho=jnp.asarray([1.0, 1836.0**-0.5][:ns]),
    fprim=jnp.full(ns, 0.8),
    tprim=jnp.full(ns, 2.49),
    nu=jnp.asarray([0.01, 0.01 * 1836.0**0.5][:ns]),
)
shape = (ns, nl, nm, grid.ky.size, grid.kx.size, grid.z.size)
state = jnp.asarray(np.random.default_rng(0).normal(size=shape), jnp.complex128)
terms = TermConfig(apar=0.0, bpar=0.0)
runs = {}
for model in ("dougherty", "landau"):
    params = linear_params_for_geometry(geom, collision_model=model, **species)
    cache = build_linear_cache(grid, geom, params, nl, nm)
    for weight in (1.0, 0.0):
        fn = jax.jit(
            lambda g, c, p, w=weight: assemble_rhs_cached(
                g, c, p, terms=TermConfig(apar=0.0, bpar=0.0, collisions=w)
            )[0]
        )
        runs[model, weight] = (fn, cache, params)
        fn(state, cache, params).block_until_ready()

times = {key: [] for key in runs}
for _ in range(2):
    for key, (fn, cache, params) in runs.items():
        for _ in range(10):
            start = time.perf_counter()
            fn(state, cache, params).block_until_ready()
            times[key].append(time.perf_counter() - start)
host = f"{platform.system()}-{platform.machine()}-{jax.devices()[0].platform}"
for model in ("dougherty", "landau"):
    full = 1e3 * float(np.median(times[model, 1.0]))
    bare = 1e3 * float(np.median(times[model, 0.0]))
    print(
        "RESULT",
        json.dumps(
            dict(
                model=model,
                Nl=nl,
                Nm=nm,
                species=ns,
                shape=list(shape),
                rhs_ms=round(full, 3),
                rhs_without_collisions_ms=round(bare, 3),
                collision_share=round((full - bare) / full, 3),
                host=host,
            )
        ),
        flush=True,
    )
