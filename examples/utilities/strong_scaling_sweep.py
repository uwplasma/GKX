#!/usr/bin/env python3
"""Strong-scaling sweep of the sharded linear RK2 loop over CPU devices.

Integrates one large linear Cyclone-like state with the ky-sharded RK2
integrator on 1, 2, 4, and 8 devices, prints the elapsed wall time per device
count, and writes a CSV of the sweep.  Needs the requested number of visible
JAX devices (e.g. ``XLA_FLAGS=--xla_force_host_platform_device_count=8``).
A few minutes total on a multi-core CPU.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from gkx import build_linear_cache, build_linear_params
from gkx.config import CycloneBaseCase, GridConfig
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid
from gkx.parallel.state import resolve_state_sharding
from gkx.parallel.integrators import integrate_linear_sharded
from gkx.operators.linear.params import Species

NY = 128  # binormal grid points
NZ = 256  # parallel grid points
NL = 8  # Laguerre moments
NM = 8  # Hermite moments
STEPS = 120  # timed RK2 steps per device count
DT = 0.1  # time step
DEVICES = [1, 2, 4, 8]  # device counts to sweep
BACKEND = "cpu_sharded_large"  # label recorded in the CSV
OUT = Path("tools_out/strong_scaling_sweep.csv")

OUT.parent.mkdir(parents=True, exist_ok=True)

cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=NY, Nz=NZ, Lx=6.28, Ly=6.28))
grid = build_spectral_grid(cfg.grid)
geom = SAlphaGeometry.from_config(cfg.geometry)
params = build_linear_params([Species(1.0, 1.0, 1.0, 1.0, 2.0, 0.8)], tau_e=1.0)

G0 = jnp.zeros((NL, NM, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
G0 = G0.at[0, 0, grid.ky.size // 2, 0, :].set(1.0e-3 + 0.0j)
cache = build_linear_cache(grid, geom, params, NL, NM)

rows = []
for n in DEVICES:
    devices = jax.devices()[:n]
    if len(devices) < n:
        raise RuntimeError(
            f"Requested {n} devices but only {len(devices)} are visible."
        )
    state_sharding = resolve_state_sharding(G0, "ky", devices=devices)
    warm = integrate_linear_sharded(
        G0, cache, params, dt=DT, steps=2, state_sharding=state_sharding
    )
    jax.block_until_ready(warm)
    t0 = time.time()
    out = integrate_linear_sharded(
        G0,
        cache,
        params,
        dt=DT,
        steps=STEPS,
        state_sharding=state_sharding,
    )
    jax.block_until_ready(out)
    elapsed = time.time() - t0
    print(f"devices={n} steps={STEPS} elapsed={elapsed:.2f}s")
    rows.append(
        {
            "backend": BACKEND,
            "steps": STEPS,
            "devices": n,
            "elapsed_s": elapsed,
            "ny": NY,
            "nz": NZ,
            "nl": NL,
            "nm": NM,
            "dt": DT,
            "notes": "sharded linear RK2 sweep",
        }
    )

with OUT.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {OUT}")
