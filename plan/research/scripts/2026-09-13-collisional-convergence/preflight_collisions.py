#!/usr/bin/env python3
"""CPU preflight for Q8: resolved collision settings, nu scaling, Sugama basis.

Part A. For the collisionless fixture and the [[species]] nu = 1e-3 / 3e-3 /
1e-2 copies, resolve the runtime config as the parity runner does and assert
the collision term weight, params.nu, the absorber strength 50 and that
[time] collision_operator resolves to the built-in term (None). Evaluate the
cached linear RHS on one fixed random state at ky=.55, Nl24, Nm96 and require
the RHS change against the collisionless fixture to be linear in nu.

Part B. With [time] collision_operator = "sugama" at nu=3e-3, try to resolve
the operator on an Nl32 x Nm96 state and, if it resolves, report the RHS
change against the built-in term. Part C repeats that on the operator's own
Nl2 x Nm4 basis. B and C are reported, not asserted.

Measurement only; run from a checkout root with PYTHONPATH=$PWD/src:$PWD,
JAX_PLATFORMS=cpu, JAX_ENABLE_X64=true and GX_PARITY_REF_DIR set.
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

HERE = "plan/research/scripts/2026-09-13-collisional-convergence"
FIXTURE = "tools/comparison/fixtures/parity/cyclone_salpha_itg.toml"
CONFIGS = {
    "nu0": (FIXTURE, 0.0),
    "nu1e-3": (f"{HERE}/cyclone_salpha_itg_nu1e-3.toml", 1e-3),
    "nu3e-3": (f"{HERE}/cyclone_salpha_itg_nu3e-3.toml", 3e-3),
    "nu1e-2": (f"{HERE}/cyclone_salpha_itg_nu1e-2.toml", 1e-2),
}
SUGAMA = f"{HERE}/cyclone_salpha_itg_nu3e-3_sugama.toml"
KY, DT = 0.550000011920929, 0.002


def _setup(path: str, nl: int, nm: int):
    cfg, _ = load_runtime_from_toml(Path(path))
    terms = build_runtime_linear_terms(cfg)
    grid = build_spectral_grid(cfg.grid)
    sub = select_ky_grid(grid, select_ky_index(np.asarray(grid.ky), KY))
    geom = build_runtime_geometry(cfg)
    params = build_runtime_linear_params(cfg, Nm=nm, geom=geom)
    cache = build_linear_cache(sub, geom, params, nl, nm)
    shape = (len(cfg.species), nl, nm, 1, int(np.asarray(sub.kx).size), cfg.grid.Nz)
    return cfg, terms, params, cache, shape


def _state(shape: tuple[int, ...]) -> jnp.ndarray:
    real = jnp.zeros(()).dtype
    complex_dtype = jnp.complex128 if real == jnp.float64 else jnp.complex64
    rng = np.random.default_rng(20260913)
    value = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return jnp.asarray(value, dtype=complex_dtype)


def _rhs(state, cache, params, terms, operator=None) -> np.ndarray:
    dG, _ = linear_rhs_cached(
        state, cache, params, terms=terms, dt=DT, collision_operator=operator
    )
    return np.asarray(dG)


def part_a() -> bool:
    print("Part A: built-in collision term, ky=.55, Nl24, Nm96")
    rhs = {}
    state = None
    for name, (path, nu) in CONFIGS.items():
        cfg, terms, params, cache, shape = _setup(path, 24, 96)
        assert float(terms.collisions) == (0.0 if nu == 0.0 else 1.0), (name, terms)
        assert float(cfg.species[0].nu) == nu, (name, cfg.species[0].nu)
        assert float(np.asarray(params.nu).reshape(-1)[0]) == nu, (name, params.nu)
        real = jnp.zeros(()).dtype
        strength = float(params.end_damping_strength(DT, real))
        assert abs(strength - 50.0) < 1e-9, (name, strength)
        operator = _resolve_config_collision_operator(cfg.time, params)
        assert operator is None, (name, cfg.time.collision_operator, operator)
        state = _state(shape) if state is None else state
        rhs[name] = _rhs(state, cache, params, terms)
        print(
            f"  {name:7s} nu={nu:g} terms.collisions={float(terms.collisions)} "
            f"collision_operator={cfg.time.collision_operator!r} -> built-in "
            f"nu_hermite={cfg.collisions.nu_hermite} "
            f"nu_laguerre={cfg.collisions.nu_laguerre} "
            f"hypercollisions_kz={cfg.collisions.hypercollisions_kz} "
            f"hypercollisions_const={cfg.collisions.hypercollisions_const} "
            f"end_damping_strength={strength:g} |RHS|={np.linalg.norm(rhs[name]):.6e}"
        )
    base = rhs["nu0"]
    norm = np.linalg.norm(base)
    change = {k: np.linalg.norm(rhs[k] - base) / norm for k in CONFIGS if k != "nu0"}
    for key, value in change.items():
        print(f"  relative RHS change {key} vs nu0: {value:.6e}")
    r3 = change["nu3e-3"] / change["nu1e-3"]
    r10 = change["nu1e-2"] / change["nu1e-3"]
    print(f"  ratios nu3e-3/nu1e-3 = {r3:.6f}, nu1e-2/nu1e-3 = {r10:.6f}")
    return bool(
        change["nu1e-3"] > 1e-8 and abs(r3 - 3.0) < 1e-3 and abs(r10 - 10.0) < 1e-3
    )


def sugama(nl: int, nm: int) -> str:
    print(
        f"Part {'B' if nl > 2 else 'C'}: collision_operator='sugama', nu=3e-3, Nl{nl}, Nm{nm}"
    )
    cfg, terms, params, cache, shape = _setup(SUGAMA, nl, nm)
    state = _state(shape)
    print(
        f"  config collision_operator={cfg.time.collision_operator!r} state shape={shape}"
    )
    try:
        operator = _resolve_config_collision_operator(cfg.time, params, state)
    except Exception as exc:  # reported, not asserted
        message = " ".join(str(exc).split())
        print(f"  REJECTED at resolution: {type(exc).__name__}: {message}")
        return "rejected"
    matrix = getattr(operator, "matrix", None)
    print(
        f"  resolved {type(operator).__name__} matrix shape="
        f"{None if matrix is None else tuple(matrix.shape)}"
    )
    try:
        with_operator = _rhs(state, cache, params, terms, operator)
    except Exception as exc:  # reported, not asserted
        message = " ".join(str(exc).split())
        print(f"  REJECTED in RHS: {type(exc).__name__}: {message}")
        return "rejected"
    builtin_cfg, builtin_terms, builtin_params, builtin_cache, _ = _setup(
        CONFIGS["nu3e-3"][0], nl, nm
    )
    builtin = _rhs(state, builtin_cache, builtin_params, builtin_terms)
    zero_cfg, zero_terms, zero_params, zero_cache, _ = _setup(CONFIGS["nu0"][0], nl, nm)
    collisionless = _rhs(state, zero_cache, zero_params, zero_terms)
    norm = np.linalg.norm(collisionless)
    print(
        f"  relative RHS change vs collisionless: sugama "
        f"{np.linalg.norm(with_operator - collisionless) / norm:.6e}, built-in "
        f"{np.linalg.norm(builtin - collisionless) / norm:.6e}; sugama vs built-in "
        f"{np.linalg.norm(with_operator - builtin) / norm:.6e}"
    )
    return "runs"


def main() -> int:
    print(f"real dtype {jnp.dtype(jnp.zeros(()).dtype).name}")
    ok = part_a()
    print("PART_A", "PASS" if ok else "FAIL")
    print("SUGAMA_NL32_NM96", sugama(32, 96).upper())
    print("SUGAMA_NL2_NM4", sugama(2, 4).upper())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
