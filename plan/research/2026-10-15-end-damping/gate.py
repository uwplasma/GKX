# ruff: noqa: E402
"""End-mode gate (tests/validation/physics_gates) at explicit rates: python gate.py r1 r2 ..."""

import sys
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, ".")
import tests.validation.physics_gates.test_collision_physics as T
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearTerms, linear_params_for_geometry
from gkx.solvers_linear_integrators import integrate_linear_diagnostics
import os

T._NM = int(os.environ.get("NM", T._NM))
T._DT = float(os.environ.get("DT", T._DT))
T._STEPS = int(round(40.0 / T._DT))
T._SAMPLE_STRIDE = int(os.environ.get("STRIDE", T._SAMPLE_STRIDE))
grid, geom = T._linked_salpha_flux_tube()
for r in sys.argv[1:]:
    params = linear_params_for_geometry(
        geom,
        tprim=2.49,
        fprim=0.8,
        damp_ends_widthfrac=0.125,
        nu_hermite=1.0,
        nu_laguerre=2.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=1.0,
        damp_ends_rate=None if r == "default" else float(r),
    )
    cache = build_linear_cache(grid, geom, params, Nl=T._NL, Nm=T._NM)
    terms = LinearTerms(collisions=0.0, hyperdiffusion=0.0, apar=0.0, bpar=0.0)
    z = np.arange(T._NZ)
    seed = np.zeros((T._NL, T._NM, 1, 1, T._NZ), complex)
    seed[0, 0, 0, 0] = T._SEED_AMPLITUDE * np.exp(
        -0.5 * ((z - T._NZ / 2) / (0.1 * T._NZ)) ** 2
    )
    _G, phi_t, _ = integrate_linear_diagnostics(
        jnp.asarray(seed),
        grid,
        geom,
        params,
        T._DT,
        T._STEPS,
        method="imex2",
        terms=terms,
        sample_stride=T._SAMPLE_STRIDE,
        species_index=0,
        cache=cache,
    )
    amp = np.max(np.abs(np.asarray(phi_t)[:, 0, 0, :]), axis=1)
    t = T._DT * T._SAMPLE_STRIDE * np.arange(1, amp.size + 1)
    late = slice(int(0.6 * amp.size), None)
    print(
        "GATE",
        T._NM,
        r,
        float(np.polyfit(t[late], np.log(amp[late]), 1)[0]),
        float(amp.max()),
        flush=True,
    )
