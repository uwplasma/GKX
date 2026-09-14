"""Q6: does the tiny selected-ky linked case certify with the adaptive route? (before/after)"""

import time
import jax.numpy as jnp
import numpy as np
import gkx
from gkx.config import CycloneBaseCase, GridConfig
from gkx.core_grid import build_spectral_grid, select_ky_grid
from gkx.geometry import SAlphaGeometry
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams, LinearTerms
from gkx.solvers_linear_krylov import dominant_eigenpair

print("gkx", gkx.__file__, flush=True)
cfg = CycloneBaseCase(
    grid=GridConfig(
        Nx=8, Ny=4, Nz=8, Ly=2.0 * np.pi * 10.0, boundary="linked", y0=10.0, jtwist=1
    )
)
grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
geom = SAlphaGeometry.from_config(cfg.geometry)
params = LinearParams(damp_ends_amp=0.1)
cache = build_linear_cache(grid, geom, params, Nl=2, Nm=4)
shape = (2, 4, grid.ky.size, grid.kx.size, grid.z.size)
index = jnp.arange(int(np.prod(shape)), dtype=jnp.float64)
seed = jnp.reshape(jnp.exp(1j * (index + 1.0) * 0.6180339887498948), shape)
chain = np.zeros(shape, dtype=bool)
chain[..., [0, 1, 2, 6, 7], :] = True
for label, start in (("fullseed", seed), ("chainseed", jnp.where(chain, seed, 0))):
    msgs = []
    t0 = time.perf_counter()
    try:
        value, vector = dominant_eigenpair(
            start,
            cache,
            params,
            terms=LinearTerms(),
            method="adaptive",
            status_callback=msgs.append,
        )
        v = np.asarray(vector)
        print(
            f"[{label}] eig={complex(np.asarray(value))} off-chain max={np.max(np.abs(v[~chain])):.3e} t={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    except Exception as e:
        print(
            f"[{label}] raised {type(e).__name__}: {e} t={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    for m in msgs:
        if "adaptive solve" in m:
            print("   ", m, flush=True)
