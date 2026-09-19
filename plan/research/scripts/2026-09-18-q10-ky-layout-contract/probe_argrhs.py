"""Dump the optimized HLO of the argument-passing nonlinear RHS at 32x32x24."""

import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gkx
from gkx.core_grid import build_spectral_grid
from gkx.geometry import apply_imported_geometry_grid_defaults
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.runtime import (
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from gkx.solvers_nonlinear_state_integration import nonlinear_rhs_cached
from gkx.workflows.runtime.toml import load_runtime_from_toml

out = Path(sys.argv[1])
cfg, _ = load_runtime_from_toml(
    Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml")
)
cfg = replace(cfg, grid=replace(cfg.grid, Nx=32, Ny=32, Nz=24))
geom = build_runtime_geometry(cfg)
grid = build_spectral_grid(apply_imported_geometry_grid_defaults(geom, cfg.grid))
params = build_runtime_linear_params(cfg, Nm=4, geom=geom)
terms = build_runtime_term_config(cfg)
cache = build_linear_cache(grid, geom, params, 2, 4)
shape = (len(cfg.species), 2, 4, grid.ky.size, grid.kx.size, grid.z.size)
rng = np.random.default_rng(20260914)
g = jnp.asarray(
    1e-3 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)),
    jnp.complex64,
)
fn = jax.jit(
    lambda s, c, p: nonlinear_rhs_cached(
        s, c, p, terms, compressed_real_fft=True, laguerre_mode="grid"
    )[0]
)
out.write_text(fn.lower(g, cache, params).compile().as_text(), encoding="utf-8")
print(gkx.__file__)
