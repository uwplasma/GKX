"""Q9 identity gate on RHS terms and their VJPs for whichever gkx is on PYTHONPATH.

Usage: python rhs_identity.py OUT.npz
Run once per tree (base, prototype) and per precision, then compare with
compare_npz.py. Cases:
  pilot      linked Cyclone linear pilot (Nx=8, Ny=16, Nz=16, jtwist=1, Nl4/Nm8),
             deck terms
  pilot_hkz  the same with hypercollisions_kz=1 so the shared route is exercised
  nl32       Cyclone nonlinear deck 32x32x24 Nl2/Nm4 (5 chain classes)
  nl64       Cyclone nonlinear deck 64x64x24 Nl4/Nm8 (7 chain classes)
Per case: every named term of assemble_rhs_terms_cached, the total, the full
nonlinear RHS, and the VJP of Re<c, rhs(G)> wrt G and wrt (tprim, nu_hyper_m)
through the jitted linear RHS. Records whether the shared route traced.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gkx
import gkx.terms.assembly as assembly
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.solvers_nonlinear_state_integration import nonlinear_rhs_cached
from gkx.terms.assembly import assemble_rhs_cached, assemble_rhs_terms_cached
from tools.profiling.profile_runtime_kernels import (
    apply_imported_geometry_grid_defaults,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
    build_spectral_grid,
    load_runtime_from_toml,
)

out = Path(sys.argv[1])
X64 = bool(jax.config.read("jax_enable_x64"))
CDT = jnp.complex128 if X64 else jnp.complex64
print("gkx", gkx.__file__, "x64", X64, flush=True)

SHARED_TRACES = {"n": 0}
if hasattr(assembly, "_shared_linked_streaming_hypercollisions"):
    _orig = assembly._shared_linked_streaming_hypercollisions

    def _probe(*args, **kwargs):
        result = _orig(*args, **kwargs)
        if result is not None:
            SHARED_TRACES["n"] += 1
        return result

    assembly._shared_linked_streaming_hypercollisions = _probe


def setup(name):
    if name.startswith("pilot"):
        cfg, _ = load_runtime_from_toml(Path("examples/linear/axisymmetric/cyclone.toml"))
        cfg = replace(
            cfg,
            grid=replace(cfg.grid, Nx=8, Ny=16, Nz=16, ntheta=16, nperiod=1, jtwist=1),
        )
        nl, nm = 4, 8
    else:
        n, nl, nm = (32, 2, 4) if name == "nl32" else (64, 4, 8)
        cfg, _ = load_runtime_from_toml(
            Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml")
        )
        cfg = replace(cfg, grid=replace(cfg.grid, Nx=n, Ny=n, Nz=24))
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(apply_imported_geometry_grid_defaults(geom, cfg.grid))
    params = build_runtime_linear_params(cfg, Nm=nm, geom=geom)
    if name == "pilot_hkz":
        params = replace(params, hypercollisions_kz=1.0, nu_hyper_m=0.5)
    terms = build_runtime_term_config(cfg)
    cache = build_linear_cache(grid, geom, params, nl, nm)
    shape = (len(cfg.species), nl, nm, grid.ky.size, grid.kx.size, grid.z.size)
    rng = np.random.default_rng(20260914)
    g = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    cot = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return cfg, cache, params, terms, jnp.asarray(1e-3 * g, CDT), jnp.asarray(cot, CDT)


results = {}
meta = {"gkx": gkx.__file__, "x64": X64, "jax": jax.__version__, "cases": {}}
for name in ("pilot", "pilot_hkz", "nl32", "nl64"):
    cfg, cache, params, terms, g, cot = setup(name)
    before = SHARED_TRACES["n"]
    total, _fields, contrib = jax.jit(
        lambda s, c, p: assemble_rhs_terms_cached(s, c, p, terms=terms)
    )(g, cache, params)
    results[f"{name}/total"] = np.asarray(total)
    for key, value in contrib.items():
        results[f"{name}/{key}"] = np.asarray(value)
    lin_terms = replace(terms, nonlinear=0.0)
    if name.startswith("nl"):
        full = jax.jit(
            lambda s, c, p: nonlinear_rhs_cached(
                s, c, p, terms, compressed_real_fft=True, laguerre_mode="grid"
            )[0]
        )(g, cache, params)
        results[f"{name}/nonlinear_rhs"] = np.asarray(full)

    def linear_scalar(s, tprim, nu_hyper_m, c=cache, p=params):
        rhs = assemble_rhs_cached(
            s, c, replace(p, tprim=tprim, nu_hyper_m=nu_hyper_m), terms=lin_terms
        )[0]
        return jnp.real(jnp.vdot(cot, rhs))

    value, grads = jax.jit(jax.value_and_grad(linear_scalar, argnums=(0, 1, 2)))(
        g, jnp.asarray(params.tprim), jnp.asarray(params.nu_hyper_m)
    )
    results[f"{name}/vjp_value"] = np.asarray(value)
    results[f"{name}/vjp_G"] = np.asarray(grads[0])
    results[f"{name}/vjp_tprim"] = np.asarray(grads[1])
    results[f"{name}/vjp_nu_hyper_m"] = np.asarray(grads[2])
    meta["cases"][name] = {
        "shape": list(g.shape),
        "dtype": str(g.dtype),
        "classes": [list(np.shape(i)) for i in cache.linked_indices],
        "shared_route_traces": SHARED_TRACES["n"] - before,
        "hypercollisions_kz": float(np.asarray(params.hypercollisions_kz)),
    }
    print(name, meta["cases"][name], flush=True)

np.savez(out, **results)
out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
print("saved", out, flush=True)
