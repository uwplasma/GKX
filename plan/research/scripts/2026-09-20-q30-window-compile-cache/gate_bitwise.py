"""Q30: the adjoint window's value and gradient, bit for bit.

One tree per process, whichever gkx is on PYTHONPATH. Differentiates the
windowed heat flux with respect to both drive gradients and writes the value
and the two adjoint components to an ``.npz``, plus a SHA-256 over their raw
bytes. Two trees agree when the digests match; nothing here rounds or formats a
number before comparing it.

Usage: python gate_bitwise.py OUT.npz [--Nx .. --ky-mode full|half --method rk3
                                       --no-checkpoint --deck PATH ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gkx
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
ap.add_argument("out", type=Path)
for name, default in (
    ("--Nx", 16),
    ("--Ny", 16),
    ("--Nz", 12),
    ("--Nl", 2),
    ("--Nm", 4),
):
    ap.add_argument(name, type=int, default=default)
ap.add_argument("--window-steps", type=int, default=6)
ap.add_argument("--tail-steps", type=int, default=None)
ap.add_argument("--method", type=str, default="rk3")
ap.add_argument("--ky-mode", type=str, default="full", choices=("full", "half"))
ap.add_argument(
    "--deck",
    type=str,
    default="examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml",
)
ap.add_argument("--no-checkpoint", action="store_true")
ap.add_argument("--no-compressed-fft", action="store_true")
ap.add_argument("--laguerre-mode", type=str, default="grid")
args = ap.parse_args()

cfg, _ = load_runtime_from_toml(Path(args.deck))
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


def window(tprim, fprim):
    return nonlinear_heat_flux_window(
        g0,
        grid,
        geom,
        replace(params, tprim=tprim, fprim=fprim),
        dt,
        args.window_steps,
        terms=terms,
        method=args.method,
        tail_steps=args.tail_steps,
        checkpoint=not args.no_checkpoint,
        compressed_real_fft=not args.no_compressed_fft,
        laguerre_mode=args.laguerre_mode,
    )


tprim = jnp.asarray(params.tprim)
fprim = jnp.asarray(params.fprim)
value, grads = jax.value_and_grad(window, argnums=(0, 1))(tprim, fprim)
arrays = {
    "value": np.asarray(value),
    "grad_tprim": np.asarray(grads[0]),
    "grad_fprim": np.asarray(grads[1]),
}
np.savez(args.out, **arrays)
digest = hashlib.sha256()
for key in sorted(arrays):
    digest.update(key.encode())
    digest.update(np.ascontiguousarray(arrays[key]).tobytes())
print(
    json.dumps(
        {
            "gkx": gkx.__file__,
            "x64": bool(jax.config.jax_enable_x64),
            "sha256": digest.hexdigest(),
            "value": arrays["value"].tolist(),
            "grad_tprim": arrays["grad_tprim"].tolist(),
            "grad_fprim": arrays["grad_fprim"].tolist(),
        }
    )
)
