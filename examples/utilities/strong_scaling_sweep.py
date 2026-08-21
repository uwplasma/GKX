#!/usr/bin/env python3
"""Run a strong-scaling sweep with the sharded linear RK2 loop."""

from __future__ import annotations

import argparse
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


def _parse_devices(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nz", type=int, default=256)
    parser.add_argument("--nl", type=int, default=8)
    parser.add_argument("--nm", type=int, default=8)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--devices", type=_parse_devices, default="1,2,4,8")
    parser.add_argument("--backend", type=str, default="cpu_sharded_large")
    parser.add_argument("--out", type=Path, default=Path("tools_out/strong_scaling_sweep.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=args.ny, Nz=args.nz, Lx=6.28, Ly=6.28))
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = build_linear_params([Species(1.0, 1.0, 1.0, 1.0, 2.0, 0.8)], tau_e=1.0)

    G0 = jnp.zeros((args.nl, args.nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    G0 = G0.at[0, 0, grid.ky.size // 2, 0, :].set(1.0e-3 + 0.0j)
    cache = build_linear_cache(grid, geom, params, args.nl, args.nm)

    rows = []
    for n in args.devices:
        devices = jax.devices()[:n]
        if len(devices) < n:
            raise RuntimeError(f"Requested {n} devices but only {len(devices)} are visible.")
        state_sharding = resolve_state_sharding(G0, "ky", devices=devices)
        warm = integrate_linear_sharded(G0, cache, params, dt=args.dt, steps=2, state_sharding=state_sharding)
        jax.block_until_ready(warm)
        t0 = time.time()
        out = integrate_linear_sharded(
            G0,
            cache,
            params,
            dt=args.dt,
            steps=args.steps,
            state_sharding=state_sharding,
        )
        jax.block_until_ready(out)
        elapsed = time.time() - t0
        print(f"devices={n} steps={args.steps} elapsed={elapsed:.2f}s")
        rows.append(
            {
                "backend": args.backend,
                "steps": args.steps,
                "devices": n,
                "elapsed_s": elapsed,
                "ny": args.ny,
                "nz": args.nz,
                "nl": args.nl,
                "nm": args.nm,
                "dt": args.dt,
                "notes": "sharded linear RK2 sweep",
            }
        )

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
