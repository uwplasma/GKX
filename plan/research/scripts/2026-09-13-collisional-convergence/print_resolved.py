#!/usr/bin/env python3
"""Print the resolved collision settings of one Q8 manifest case.

Usage: print_resolved.py MANIFEST KEY. Loads the case's config exactly as
tools/comparison/build_gx_parity_matrix.py does and prints the collision,
hypercollision and absorber settings the run uses. CPU only, no solve.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tomllib

import numpy as np
import jax.numpy as jnp

from gkx import load_runtime_from_toml
from gkx.workflows.runtime.startup import (
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_linear_terms,
)


def main() -> None:
    manifest, key = Path(sys.argv[1]), sys.argv[2]
    case = next(
        c for c in tomllib.loads(manifest.read_text())["case"] if c["key"] == key
    )
    cfg, _ = load_runtime_from_toml(Path(case["config"]))
    terms = build_runtime_linear_terms(cfg)
    geom = build_runtime_geometry(cfg)
    params = build_runtime_linear_params(cfg, Nm=int(case["Nm"]), geom=geom)
    strength = float(
        params.end_damping_strength(float(case["dt"]), jnp.zeros(()).dtype)
    )
    col = cfg.collisions
    print(
        f"resolved {key}: config={case['config']} Nl={case['Nl']} Nm={case['Nm']} "
        f"steps={case['steps']} dt={case['dt']} method={case['method']} "
        f"species_nu={cfg.species[0].nu} params_nu={np.asarray(params.nu).reshape(-1).tolist()} "
        f"collision_operator={cfg.time.collision_operator!r} "
        f"terms.collisions={float(terms.collisions)} nu_hermite={col.nu_hermite} "
        f"nu_laguerre={col.nu_laguerre} hypercollisions_kz={col.hypercollisions_kz} "
        f"hypercollisions_const={col.hypercollisions_const} "
        f"terms.hypercollisions={float(terms.hypercollisions)} "
        f"end_damping_strength={strength:g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
