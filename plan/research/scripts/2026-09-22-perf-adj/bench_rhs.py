"""Time the pieces of the differentiable nonlinear RHS in one ky layout."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ap = argparse.ArgumentParser()
for name, default in (("--Nx", 16), ("--Ny", 16), ("--Nz", 16), ("--Nl", 4), ("--Nm", 8)):
    ap.add_argument(name, type=int, default=default)
ap.add_argument("--ky", choices=("full", "half"), default="full")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--hlo-dir", type=Path, default=None)
args = ap.parse_args()

from gkx.core_ky_layout import FULL, HALF  # noqa: E402
from gkx import solvers_nonlinear_state_integration as sni  # noqa: E402
from gkx.geometry import ensure_flux_tube_geometry_data  # noqa: E402
from gkx.operators.linear.cache_builder import build_linear_cache  # noqa: E402
from gkx.terms.config import TermConfig  # noqa: E402

sys.path.insert(0, os.getcwd())
from tools.profiling.profile_runtime_kernels import (  # noqa: E402
    _build_initial_condition, _select_nonlinear_mode_indices,
    apply_imported_geometry_grid_defaults, build_runtime_geometry,
    build_runtime_linear_params, build_runtime_term_config, build_spectral_grid,
    load_runtime_from_toml,
)

cfg, _ = load_runtime_from_toml(Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml"))
cfg = replace(cfg, grid=replace(cfg.grid, Nx=args.Nx, Ny=args.Ny, Nz=args.Nz))
geom = build_runtime_geometry(cfg)
grid = build_spectral_grid(apply_imported_geometry_grid_defaults(geom, cfg.grid),
                           ky_layout=FULL if args.ky == "full" else HALF)
params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
terms = build_runtime_term_config(cfg)
ky_i, kx_i = _select_nonlinear_mode_indices(grid, ky_target=0.3, kx_target=None,
                                            use_dealias_mask=bool(cfg.time.nonlinear_dealias))
g0 = jnp.asarray(_build_initial_condition(grid, geom, cfg, ky_index=ky_i, kx_index=kx_i,
                                          Nl=args.Nl, Nm=args.Nm, nspecies=len(cfg.species)))
g0 = g0 / jnp.maximum(jnp.max(jnp.abs(g0)), 1e-30) * 1e-2
geometry = ensure_flux_tube_geometry_data(geom, grid.z)
cache = build_linear_cache(grid, geometry, params, Nl=args.Nl, Nm=args.Nm)
lin_terms = replace(terms, nonlinear=0.0)


def bench(name, fn, *a):
    c = jax.jit(fn).lower(*a).compile()
    jax.block_until_ready(c(*a))
    ts = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        jax.block_until_ready(c(*a))
        ts.append(time.perf_counter() - t0)
    print(f"{args.ky} {name:24s} median {1e3*statistics.median(ts):8.2f} ms  min {1e3*min(ts):8.2f} ms", flush=True)
    if args.hlo_dir is not None:
        args.hlo_dir.mkdir(parents=True, exist_ok=True)
        (args.hlo_dir / f"{name}_{args.ky}.hlo.txt").write_text(c.as_text())


def rhs_full(G, c):
    return sni.nonlinear_rhs_cached(G, c, params, terms, compressed_real_fft=True,
                                    laguerre_mode="grid", differentiable=True)[0]


def rhs_lin(G, c):
    return sni.nonlinear_rhs_cached(G, c, params, lin_terms, compressed_real_fft=True,
                                    laguerre_mode="grid", differentiable=True)[0]


def rhs_vjp(G, c):
    out, pull = jax.vjp(lambda x: rhs_full(x, c), G)
    return pull(out)[0]


print("state", g0.shape, g0.dtype)
bench("rhs_nonlinear", rhs_full, g0, cache)
bench("rhs_linear", rhs_lin, g0, cache)
bench("rhs_vjp_state", rhs_vjp, g0, cache)

from gkx.operators.fluxes import heat_flux_total  # noqa: E402
from gkx.operators.moments import fieldline_quadrature_weights  # noqa: E402
from gkx.solvers_nonlinear_explicit import advance_explicit_nonlinear_state  # noqa: E402
from gkx.operators.nonlinear.projection import hermitian_projector_signature, hermitian_projector_for_signature  # noqa: E402

_vf, flux_factor = fieldline_quadrature_weights(geometry, grid)
sig = hermitian_projector_signature(np.asarray(grid.ky), int(np.asarray(grid.kx).size))
proj = hermitian_projector_for_signature(sig)


def fields_only(G, c):
    return sni.nonlinear_rhs_cached(G, c, params, terms, compressed_real_fft=True,
                                    laguerre_mode="grid", differentiable=True)[1].phi


def heat(G, c):
    phi = fields_only(G, c)
    z = jnp.zeros_like(phi)
    return heat_flux_total(G, phi, z, z, c, grid, params, flux_factor)


def step(G, c):
    rhs = lambda s: sni.nonlinear_rhs_cached(s, c, params, terms, compressed_real_fft=True,
                                             laguerre_mode="grid", differentiable=True)
    G = proj(G)
    d, _ = rhs(G)
    return advance_explicit_nonlinear_state(G, d, jnp.asarray(0.01), method="rk3", rhs_fn=rhs,
                                            project_state=proj, state_dtype=G.dtype)


bench("fields_only", fields_only, g0, cache)
bench("heat", heat, g0, cache)
bench("project", lambda G: proj(G), g0)
bench("rk3_step", step, g0, cache)

proj_barrier = (lambda G: jax.lax.optimization_barrier(G)) if args.ky == "half" else proj


def step_b(G, c):
    rhs = lambda s: sni.nonlinear_rhs_cached(s, c, params, terms, compressed_real_fft=True,
                                             laguerre_mode="grid", differentiable=True)
    G = proj_barrier(G)
    d, _ = rhs(G)
    return advance_explicit_nonlinear_state(G, d, jnp.asarray(0.01), method="rk3", rhs_fn=rhs,
                                            project_state=proj_barrier, state_dtype=G.dtype)


bench("rk3_step_barrier", step_b, g0, cache)
a = jax.jit(step)(g0, cache); b = jax.jit(step_b)(g0, cache)
print("bitwise", bool(jnp.all(a == b)), float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(a))))


def step_c(G, c):
    def rhs(s):
        d, f = sni.nonlinear_rhs_cached(s, c, params, terms, compressed_real_fft=True,
                                        laguerre_mode="grid", differentiable=True)
        return jax.lax.optimization_barrier(d), f
    G = proj_barrier(G)
    d, _ = rhs(G)
    return advance_explicit_nonlinear_state(G, d, jnp.asarray(0.01), method="rk3", rhs_fn=rhs,
                                            project_state=proj_barrier, state_dtype=G.dtype)


bench("rk3_step_barrier_rhs", step_c, g0, cache)
cc = jax.jit(step_c)(g0, cache)
print("bitwise_c", bool(jnp.all(a == cc)))

if os.environ.get("PROFILE_DIR"):
    fn = jax.jit(step_c if os.environ.get("PROFILE_VARIANT") == "barrier" else step)
    jax.block_until_ready(fn(g0, cache))
    jax.profiler.start_trace(os.environ["PROFILE_DIR"])
    for _ in range(3):
        jax.block_until_ready(fn(g0, cache))
    jax.profiler.stop_trace()
