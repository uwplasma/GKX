#!/usr/bin/env python3
"""Preflight for the Q3 drift ablation: prove the term switches reach the RHS.

For the full fixture and the gradb=0 / curvature=0 copies this resolves the
runtime config exactly as the parity runner does, asserts the term weights and
the absorber strength (damp_ends_amp/dt = 50 at dt=.002), then evaluates the
cached linear RHS on one fixed random state at ky=.55, Nl24, Nm96 and reports
how far each ablated RHS is from the full one. Measurement only; no source
change. Run from a checkout root with PYTHONPATH=$PWD/src:$PWD and
GX_PARITY_REF_DIR set.
"""

# ruff: noqa: E402
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import jax.numpy as jnp

from gkx import load_runtime_from_toml
from gkx.core_grid import build_spectral_grid, select_ky_grid
from gkx.diagnostics.modes import select_ky_index
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.workflows.runtime.startup import (
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_linear_terms,
)

HERE = "plan/research/scripts/2026-09-13-drift-ablation"
CONFIGS = {
    "full": ("tools/comparison/fixtures/parity/cyclone_salpha_itg.toml", 1.0, 1.0),
    "gradb0": (f"{HERE}/cyclone_salpha_itg_gradb0.toml", 0.0, 1.0),
    "curvature0": (f"{HERE}/cyclone_salpha_itg_curvature0.toml", 1.0, 0.0),
}
KY, NL, NM, DT = 0.550000011920929, 24, 96, 0.002


def main() -> int:
    real = jnp.float64 if jnp.zeros(()).dtype == jnp.float64 else jnp.float32
    print(f"real dtype {jnp.dtype(real).name}")
    rhs = {}
    state = None
    for name, (path, want_gradb, want_curv) in CONFIGS.items():
        cfg, _ = load_runtime_from_toml(Path(path))
        terms = build_runtime_linear_terms(cfg)
        assert float(terms.gradb) == want_gradb, (name, terms.gradb)
        assert float(terms.curvature) == want_curv, (name, terms.curvature)
        grid = build_spectral_grid(cfg.grid)
        ky_i = select_ky_index(np.asarray(grid.ky), KY)
        sub = select_ky_grid(grid, ky_i)
        geom = build_runtime_geometry(cfg)
        params = build_runtime_linear_params(cfg, Nm=NM, geom=geom)
        strength = float(params.end_damping_strength(DT, real))
        assert abs(strength - 50.0) < 1e-9, (name, strength)
        cache = build_linear_cache(sub, geom, params, NL, NM)
        shape = (len(cfg.species), NL, NM, 1, int(np.asarray(sub.kx).size), cfg.grid.Nz)
        if state is None:
            rng = np.random.default_rng(20260913)
            state = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            state = jnp.asarray(
                state, dtype=jnp.complex128 if real == jnp.float64 else jnp.complex64
            )
        dG, _ = linear_rhs_cached(state, cache, params, terms=terms, dt=DT)
        rhs[name] = np.asarray(dG)
        print(
            f"{name:10s} ky={float(np.asarray(sub.ky)[0]):.6f} shape={shape} "
            f"terms(curv,gradb,stream,mirror,diamag,hyper,end)="
            f"({terms.curvature},{terms.gradb},{terms.streaming},{terms.mirror},"
            f"{terms.diamagnetic},{terms.hypercollisions},{terms.end_damping}) "
            f"end_damping_strength={strength:g} |RHS|={np.linalg.norm(rhs[name]):.6e}"
        )
    full = rhs["full"]
    ok = True
    for name in ("gradb0", "curvature0"):
        rel = np.linalg.norm(rhs[name] - full) / np.linalg.norm(full)
        print(f"relative RHS difference {name} vs full: {rel:.6e}")
        ok = ok and np.isfinite(rel) and rel > 1e-6
    diff = np.linalg.norm(rhs["gradb0"] - rhs["curvature0"]) / np.linalg.norm(full)
    print(f"relative RHS difference gradb0 vs curvature0: {diff:.6e}")
    ok = ok and diff > 1e-6
    print("PREFLIGHT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
