"""Hash every grid/cache/params leaf for the nl32 identity case."""

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

import gkx
from gkx.core_grid import build_spectral_grid
from gkx.geometry import apply_imported_geometry_grid_defaults
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.runtime import (
    build_runtime_geometry,
    build_runtime_linear_params,
)
from gkx.workflows.runtime.toml import load_runtime_from_toml

cfg, _ = load_runtime_from_toml(
    Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml")
)
cfg = replace(cfg, grid=replace(cfg.grid, Nx=32, Ny=32, Nz=24))
geom = build_runtime_geometry(cfg)
grid = build_spectral_grid(apply_imported_geometry_grid_defaults(geom, cfg.grid))
params = build_runtime_linear_params(cfg, Nm=4, geom=geom)
cache = build_linear_cache(grid, geom, params, 2, 4)
out = {}
for name, tree in (("grid", grid), ("params", params), ("cache", cache)):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    out[f"{name}/treedef"] = str(treedef)
    for i, leaf in enumerate(leaves):
        arr = np.asarray(leaf)
        out[f"{name}/{i}"] = [
            str(arr.dtype),
            list(arr.shape),
            hashlib.sha256(arr.tobytes()).hexdigest()[:16],
        ]
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
print(gkx.__file__)
