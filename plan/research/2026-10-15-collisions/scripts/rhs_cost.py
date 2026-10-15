#!/usr/bin/env python3
"""Cost of one linear RHS with the Dougherty and the linearized Landau kernel.

    python rhs_cost.py <Nl> <Nm> [Ny] [Nz]

Cyclone s-alpha ITG grid, one kinetic species with nu = 0.01. Prints one RESULT
line per kernel with the median wall time of 20 jitted RHS calls and the time
to build the Landau matrices.
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
from gkx.operators.linear.rhs import linear_rhs_cached

nl, nm = int(sys.argv[1]), int(sys.argv[2])
ny = int(sys.argv[3]) if len(sys.argv) > 3 else 16
nz = int(sys.argv[4]) if len(sys.argv) > 4 else 32
grid = build_spectral_grid(GridConfig(Nx=8, Ny=ny, Nz=nz, Lx=62.8, Ly=62.8))
geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778)
state = jnp.ones((nl, nm, grid.ky.size, grid.kx.size, grid.z.size), jnp.complex128)
for model in ("dougherty", "landau"):
    params = linear_params_for_geometry(geom, nu=0.01, collision_model=model)
    start = time.perf_counter()
    cache = build_linear_cache(grid, geom, params, nl, nm)
    build = time.perf_counter() - start
    linear_rhs_cached(state, cache, params)[0].block_until_ready()
    times = []
    for _ in range(20):
        start = time.perf_counter()
        linear_rhs_cached(state, cache, params)[0].block_until_ready()
        times.append(time.perf_counter() - start)
    print(
        "RESULT",
        json.dumps(
            dict(
                model=model,
                Nl=nl,
                Nm=nm,
                shape=list(state.shape),
                rhs_ms=round(1e3 * float(np.median(times)), 3),
                cache_build_s=round(build, 2),
                host=f"{platform.system()}-{platform.machine()}-{jax.devices()[0].platform}",
            )
        ),
        flush=True,
    )
