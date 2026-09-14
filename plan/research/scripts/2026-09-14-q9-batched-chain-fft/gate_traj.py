"""Q4 gate: trajectories on the Cyclone nonlinear deck for whichever gkx is on PYTHONPATH.

Usage: python gate.py OUT.npz [case,case,...]
Cases: scan, runtime, sharded, species_hermite, window
"""

import json
import os
import sys
import time
import traceback
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
    _build_initial_condition,
    _select_nonlinear_mode_indices,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
    run_runtime_nonlinear,
)
from gkx.solvers_nonlinear_state_integration import (
    integrate_nonlinear,
    nonlinear_heat_flux_window,
    nonlinear_rhs_cached,
)
from gkx.parallel.integrators import (
    integrate_nonlinear_sharded,
    integrate_nonlinear_species_hermite,
)
from gkx.terms.config import TermConfig
from gkx.workflows.runtime.toml import load_runtime_from_toml

DECK = Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml")
out_path = Path(sys.argv[1])
cases = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else {
    "scan",
    "runtime",
    "sharded",
    "species_hermite",
    "window",
}
X64 = bool(jax.config.read("jax_enable_x64"))
NX, NY, NZ, NL, NM = (int(v) for v in os.environ.get("GATE_GRID", "16,16,12,2,4").split(","))
STEPS = int(os.environ.get("GATE_STEPS", "100"))
DT = float(os.environ.get("GATE_DT", "0.01"))
AMP = float(os.environ.get("GATE_AMP", "0.05"))
WSTEPS = int(os.environ.get("GATE_WINDOW_STEPS", "20"))
print("gkx", gkx.__file__, "jax", jax.__version__, "x64", X64, flush=True)

cfg, _ = load_runtime_from_toml(DECK)
cfg = replace(
    cfg,
    grid=replace(cfg.grid, Nx=NX, Ny=NY, Nz=NZ),
    init=replace(cfg.init, init_amp=AMP),
    time=replace(cfg.time, run_to="t_max"),
)
geom = build_runtime_geometry(cfg)
grid = build_spectral_grid(apply_imported_geometry_grid_defaults(geom, cfg.grid))
params = build_runtime_linear_params(cfg, Nm=NM, geom=geom)
term_cfg = build_runtime_term_config(cfg)
lmode = cfg.time.laguerre_nonlinear_mode
kyi, kxi = _select_nonlinear_mode_indices(
    grid, ky_target=0.3, kx_target=None, use_dealias_mask=bool(cfg.time.nonlinear_dealias)
)
G0 = jnp.asarray(
    _build_initial_condition(
        grid, geom, cfg, ky_index=kyi, kx_index=kxi, Nl=NL, Nm=NM, nspecies=len(cfg.species)
    )
)
if X64:
    G0 = G0.astype(jnp.complex128)
cache = build_linear_cache(grid, geom, params, NL, NM)
results: dict[str, np.ndarray] = {}
meta: dict[str, object] = {
    "gkx": gkx.__file__,
    "x64": X64,
    "grid": [NX, NY, NZ, NL, NM],
    "steps": STEPS,
    "dt": DT,
    "amp": AMP,
    "ky_index": int(kyi),
    "kx_index": int(kxi),
    "state_shape": list(G0.shape),
    "state_dtype": str(G0.dtype),
    "errors": {},
    "seconds": {},
}

# Size of the nonlinear term relative to the linear one at the initial state.
lin_terms = replace(term_cfg, nonlinear=0.0)
full = nonlinear_rhs_cached(G0, cache, params, term_cfg)[0]
lin = nonlinear_rhs_cached(G0, cache, params, lin_terms)[0]
meta["nonlinear_over_linear_rhs_norm"] = float(
    jnp.linalg.norm(full - lin) / jnp.linalg.norm(lin)
)
print("NL/L ratio", meta["nonlinear_over_linear_rhs_norm"], flush=True)


def record(name, fn):
    t0 = time.perf_counter()
    try:
        out = fn()
        for key, value in out.items():
            if value is None:
                continue
            results[f"{name}/{key}"] = np.asarray(value)
        print(f"{name}: ok {time.perf_counter() - t0:.1f}s", flush=True)
    except Exception as exc:  # recorded, not hidden
        meta["errors"][name] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        print(f"{name}: ERROR {exc}", flush=True)
    meta["seconds"][name] = time.perf_counter() - t0


if "scan" in cases:
    for method in ("euler", "rk2", "rk3", "rk3_classic", "rk4", "sspx3", "k10"):

        def run_scan(method=method):
            G, fields = integrate_nonlinear(
                jnp.array(G0),  # the scan donates its state
                grid,
                geom,
                params,
                dt=DT,
                steps=STEPS,
                method=method,
                cache=cache,
                terms=term_cfg,
                compressed_real_fft=True,
                laguerre_mode=lmode,
                return_fields=True,
            )
            return {"G": G, "phi": fields.phi}

        record(f"scan/{method}", run_scan)

if "runtime" in cases:
    variants = {
        "adaptive": cfg,
        "collision_split": replace(
            cfg, time=replace(cfg.time, collision_split=True, collision_scheme="implicit")
        ),
        "fixed_mode": replace(
            cfg, expert=replace(cfg.expert, fixed_mode=True, iky_fixed=int(kyi), ikx_fixed=1)
        ),
    }
    for vname, vcfg in variants.items():
        for method in ("rk3", "rk4"):

            def run_runtime(vcfg=vcfg, method=method):
                res = run_runtime_nonlinear(
                    vcfg,
                    ky_target=0.3,
                    Nl=NL,
                    Nm=NM,
                    dt=DT,
                    steps=STEPS,
                    method=method,
                    sample_stride=1,
                    diagnostics_stride=1,
                    diagnostics=True,
                    return_state=True,
                )
                d = res.diagnostics
                return {
                    "state": res.state,
                    "t": d.t,
                    "dt_t": d.dt_t,
                    "Wg_t": d.Wg_t,
                    "Wphi_t": d.Wphi_t,
                    "heat_flux_t": d.heat_flux_t,
                    "phi_mode_t": d.phi_mode_t,
                }

            record(f"runtime/{vname}/{method}", run_runtime)

if "sharded" in cases:
    for method in ("rk3", "rk3_classic", "rk4"):

        def run_sharded(method=method):
            return {
                "G": integrate_nonlinear_sharded(
                    jnp.array(G0),
                    cache,
                    params,
                    dt=DT,
                    steps=STEPS,
                    method=method,
                    terms=term_cfg,
                    state_sharding=None,
                    return_fields=False,
                )
            }

        record(f"sharded/{method}", run_sharded)

if "species_hermite" in cases:
    for method in ("rk3", "rk4"):

        def run_species(method=method):
            try:
                terms = term_cfg
                run = integrate_nonlinear_species_hermite(
                    jnp.array(G0), cache, params, dt=DT, steps=STEPS, method=method, terms=terms, num_devices=1
                )
            except (ValueError, NotImplementedError) as exc:
                meta.setdefault("species_hermite_fallback", str(exc))
                terms = TermConfig(nonlinear=1.0, apar=0.0, bpar=0.0)
                run = integrate_nonlinear_species_hermite(
                    jnp.array(G0), cache, params, dt=DT, steps=STEPS, method=method, terms=terms, num_devices=1
                )
            return {"G": run.state}

        record(f"species_hermite/{method}", run_species)

if "window" in cases:
    for method in ("rk3", "rk4"):

        def run_window(method=method):
            def objective(tprim):
                return nonlinear_heat_flux_window(
                    jnp.array(G0),
                    grid,
                    geom,
                    replace(params, tprim=tprim),
                    DT,
                    WSTEPS,
                    terms=term_cfg,
                    method=method,
                    checkpoint=True,
                    compressed_real_fft=True,
                    laguerre_mode=lmode,
                )

            value, grad = jax.value_and_grad(objective)(jnp.asarray(params.tprim))
            return {"value": value, "grad_tprim": grad}

        record(f"window/{method}", run_window)

meta["classes"] = [list(np.shape(i)) for i in cache.linked_indices]
np.savez(out_path, **results)
out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
print("saved", out_path, "errors", meta["errors"], flush=True)
