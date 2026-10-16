# ruff: noqa: E402
"""Dominant eigenvalue of the end-mode gate operator: python gate_eig.py Nm rate..."""

import sys
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, ".")
import tests.validation.physics_gates.test_collision_physics as T
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearTerms, linear_params_for_geometry
from gkx.solvers_linear_krylov import dominant_eigenpair

grid, geom = T._linked_salpha_flux_tube()
nm = int(sys.argv[1])
for r in sys.argv[2:]:
    params = linear_params_for_geometry(
        geom,
        tprim=2.49,
        fprim=0.8,
        damp_ends_widthfrac=0.125,
        nu_hermite=1.0,
        nu_laguerre=2.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=1.0,
        damp_ends_rate=float(r),
    )
    cache = build_linear_cache(grid, geom, params, Nl=T._NL, Nm=nm)
    terms = LinearTerms(collisions=0.0, hyperdiffusion=0.0, apar=0.0, bpar=0.0)
    v0 = jnp.asarray(
        np.random.default_rng(0).standard_normal((T._NL, nm, 1, 1, T._NZ)) + 0j
    )
    lam, vec = dominant_eigenpair(v0, cache, params, terms)[:2]
    v = np.abs(np.asarray(vec))
    pm = (v**2).sum(axis=(0, 2, 3, 4))
    pz = (v**2).sum(axis=(0, 1, 2, 3))
    print(
        "EIG",
        nm,
        r,
        complex(lam),
        "m-tail",
        float(pm[-nm // 4 :].sum() / pm.sum()),
        "end-share",
        float(pz[:12].sum() + pz[-12:].sum()) / pz.sum(),
        flush=True,
    )
