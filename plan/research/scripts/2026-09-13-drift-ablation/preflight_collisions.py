#!/usr/bin/env python3
"""Preflight for the Q3 collision follow-up: prove species nu reaches the RHS.

For the collisionless fixture and the [[species]] nu = 1e-3 / 1e-2 copies this
resolves the runtime config as the parity runner does and asserts: the
collision term weight (0 when every nu is 0, 1 otherwise), params.nu, the
absorber strength 50, and that [time] collision_operator resolves to the
built-in diagonal Lenard-Bernstein path (no dense moment operator). It then
evaluates the cached linear RHS on one fixed random state at ky=.55, Nl24,
Nm96 and reports the RHS change against the collisionless fixture; the change
must be nonzero and scale linearly with nu (ratio ~10). Measurement only; run
from a checkout root with PYTHONPATH=$PWD/src:$PWD and GX_PARITY_REF_DIR set.
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
from gkx.solvers_time_runners import _resolve_config_collision_operator
from gkx.workflows.runtime.startup import (
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_linear_terms,
)

HERE = "plan/research/scripts/2026-09-13-drift-ablation"
CONFIGS = {
    "nu0": ("tools/comparison/fixtures/parity/cyclone_salpha_itg.toml", 0.0),
    "nu1e-3": (f"{HERE}/cyclone_salpha_itg_nu1e-3.toml", 1e-3),
    "nu1e-2": (f"{HERE}/cyclone_salpha_itg_nu1e-2.toml", 1e-2),
}
KY, NL, NM, DT = 0.550000011920929, 24, 96, 0.002


def main() -> int:
    real = jnp.float64 if jnp.zeros(()).dtype == jnp.float64 else jnp.float32
    complex_dtype = jnp.complex128 if real == jnp.float64 else jnp.complex64
    print(f"real dtype {jnp.dtype(real).name}")
    rhs = {}
    state = None
    for name, (path, nu) in CONFIGS.items():
        cfg, _ = load_runtime_from_toml(Path(path))
        terms = build_runtime_linear_terms(cfg)
        assert float(terms.collisions) == (0.0 if nu == 0.0 else 1.0), (name, terms)
        assert float(cfg.species[0].nu) == nu, (name, cfg.species[0].nu)
        grid = build_spectral_grid(cfg.grid)
        sub = select_ky_grid(grid, select_ky_index(np.asarray(grid.ky), KY))
        geom = build_runtime_geometry(cfg)
        params = build_runtime_linear_params(cfg, Nm=NM, geom=geom)
        assert float(np.asarray(params.nu).reshape(-1)[0]) == nu, (name, params.nu)
        strength = float(params.end_damping_strength(DT, real))
        assert abs(strength - 50.0) < 1e-9, (name, strength)
        operator = _resolve_config_collision_operator(cfg.time, params)
        assert operator is None, (name, cfg.time.collision_operator, operator)
        cache = build_linear_cache(sub, geom, params, NL, NM)
        shape = (len(cfg.species), NL, NM, 1, int(np.asarray(sub.kx).size), cfg.grid.Nz)
        if state is None:
            rng = np.random.default_rng(20260913)
            state = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            state = jnp.asarray(state, dtype=complex_dtype)
        dG, _ = linear_rhs_cached(state, cache, params, terms=terms, dt=DT)
        rhs[name] = np.asarray(dG)
        print(
            f"{name:7s} nu={nu:g} terms.collisions={terms.collisions} "
            f"collision_operator={cfg.time.collision_operator!r} (built-in LB) "
            f"nu_hermite={cfg.collisions.nu_hermite} nu_laguerre={cfg.collisions.nu_laguerre} "
            f"end_damping_strength={strength:g} |RHS|={np.linalg.norm(rhs[name]):.6e}"
        )
    base = rhs["nu0"]
    norm = np.linalg.norm(base)
    d3 = np.linalg.norm(rhs["nu1e-3"] - base) / norm
    d2 = np.linalg.norm(rhs["nu1e-2"] - base) / norm
    print(f"relative RHS change nu1e-3 vs nu0: {d3:.6e}")
    print(f"relative RHS change nu1e-2 vs nu0: {d2:.6e}")
    print(f"ratio (nu1e-2 / nu1e-3): {d2 / d3:.6f}")
    ok = bool(np.isfinite(d2) and d3 > 1e-8 and abs(d2 / d3 - 10.0) < 1e-3)
    print("PREFLIGHT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
