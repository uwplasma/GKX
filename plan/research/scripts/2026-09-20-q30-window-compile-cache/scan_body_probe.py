"""Q30: the scan bodies the adjoint window stages, for whichever gkx is on PYTHONPATH.

Wraps ``scan_p.bind`` and records every scan body the window traces: equation
count, invar/outvar counts, and a SHA-256 of the printed jaxpr with source
locations stripped (two trees enter through different Python frames). Use
``--wrt`` to change which parameter carries the tangent; on ``main`` the body is
the same size for ``tprim``, ``beta``, ``tau_e`` and ``rho_star``,
which is what rules the linear cache out as the cause of the branch's larger
body.

Usage: python scan_body_probe.py [--grad] [--wrt tprim] [--dump DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
from jax._src.lax.control_flow import loops as _loops

from gkx.core_ky_layout import FULL, HALF
from gkx.solvers_nonlinear_state_integration import nonlinear_heat_flux_window
from tools.profiling.profile_runtime_kernels import (
    _build_initial_condition,
    _select_nonlinear_mode_indices,
    apply_imported_geometry_grid_defaults,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
    build_spectral_grid,
    load_runtime_from_toml,
)

ap = argparse.ArgumentParser()
for name, default in (
    ("--Nx", 16),
    ("--Ny", 16),
    ("--Nz", 12),
    ("--Nl", 2),
    ("--Nm", 4),
):
    ap.add_argument(name, type=int, default=default)
ap.add_argument("--window-steps", type=int, default=6)
ap.add_argument("--method", type=str, default="rk3")
ap.add_argument("--ky-mode", type=str, default="full")
ap.add_argument("--grad", action="store_true")
ap.add_argument("--dump", type=str, default=None)
ap.add_argument("--wrt", type=str, default="tprim")
args = ap.parse_args()

cfg, _ = load_runtime_from_toml(
    Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml")
)
cfg = replace(cfg, grid=replace(cfg.grid, Nx=args.Nx, Ny=args.Ny, Nz=args.Nz))
geom = build_runtime_geometry(cfg)
grid = build_spectral_grid(
    apply_imported_geometry_grid_defaults(geom, cfg.grid),
    ky_layout=FULL if args.ky_mode == "full" else HALF,
)
params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
terms = build_runtime_term_config(cfg)
ky_i, kx_i = _select_nonlinear_mode_indices(
    grid,
    ky_target=0.3,
    kx_target=None,
    use_dealias_mask=bool(cfg.time.nonlinear_dealias),
)
g0 = jnp.asarray(
    _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=ky_i,
        kx_index=kx_i,
        Nl=args.Nl,
        Nm=args.Nm,
        nspecies=len(cfg.species),
    )
)
g0 = g0 / jnp.maximum(jnp.max(jnp.abs(g0)), 1e-30) * 1e-2
dt = float(cfg.time.dt)


def window(value):
    return nonlinear_heat_flux_window(
        g0,
        grid,
        geom,
        replace(params, **{args.wrt: value}),
        dt,
        args.window_steps,
        terms=terms,
        method=args.method,
        checkpoint=True,
        compressed_real_fft=True,
        laguerre_mode="grid",
    )


records = []

_orig_bind = _loops.scan_p.bind


def _record(*a, **k):
    body = k.get("jaxpr")
    if body is not None:
        inner = body.jaxpr if hasattr(body, "jaxpr") else body
        text = re.sub(r"\s*\[[^\]]*\.py:[0-9]+[^\]]*\]", "", str(body))
        records.append(
            {
                "eqns": len(inner.eqns),
                "invars": len(inner.invars),
                "outvars": len(inner.outvars),
                "length": k.get("length"),
                "num_consts": k.get("num_consts"),
                "num_carry": k.get("num_carry"),
                "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
            }
        )
        dump = Path(args.dump) if args.dump else None
        if dump is not None:
            dump.mkdir(parents=True, exist_ok=True)
            (dump / f"scan_{len(records):02d}_{len(inner.eqns)}.txt").write_text(text)
    return _orig_bind(*a, **k)


_loops.scan_p.bind = _record

if args.grad:
    jax.block_until_ready(jax.grad(window)(jnp.asarray(getattr(params, args.wrt))))
else:
    jax.block_until_ready(window(jnp.asarray(getattr(params, args.wrt))))

records.sort(key=lambda r: (r["eqns"], r["sha256"]))
print(
    json.dumps(
        {"grad": args.grad, "wrt": args.wrt, "scans": len(records), "bodies": records},
        indent=1,
    )
)
