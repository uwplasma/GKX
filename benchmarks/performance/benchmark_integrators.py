"""Measure fixed-step linear integrator runtime and host memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
import tracemalloc

import jax
import jax.numpy as jnp

from gkx.config import CycloneBaseCase
from gkx.solvers.time.diffrax_linear import integrate_linear_diffrax
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid
from gkx.linear import LinearParams, build_linear_cache, integrate_linear


def _block(tree):
    return jax.tree_util.tree_map(jax.block_until_ready, tree)


def _measure(func, *, repeat: int) -> dict[str, object]:
    times = []
    peaks = []
    for _ in range(repeat):
        tracemalloc.start()
        t0 = time.perf_counter()
        _block(func())
        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(t1 - t0)
        peaks.append(peak / 1.0e6)
    return {
        "times_s": times,
        "mean_s": float(statistics.mean(times)),
        "median_s": float(statistics.median(times)),
        "min_s": float(min(times)),
        "host_peak_mb": float(max(peaks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed-step linear integrators on a Cyclone state."
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--diffrax-solver", default="Heun")
    parser.add_argument("--diffrax-adaptive", action="store_true")
    parser.add_argument("--diffrax-rtol", type=float, default=1.0e-3)
    parser.add_argument("--diffrax-atol", type=float, default=1.0e-6)
    parser.add_argument("--diffrax-max-steps", type=int, default=20000)
    parser.add_argument("--no-diffrax-jit", action="store_true")
    parser.add_argument("--skip-diffrax", action="store_true")
    parser.add_argument("--method", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--Nl", type=int, default=2)
    parser.add_argument("--Nm", type=int, default=2)
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--label", default="current")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    cfg = CycloneBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    steps = (
        int(round(cfg.time.t_max / cfg.time.dt))
        if args.steps is None
        else int(args.steps)
    )
    dt = float(cfg.time.dt if args.dt is None else args.dt)
    method = str(cfg.time.method if args.method is None else args.method)

    G0 = jnp.zeros(
        (args.Nl, args.Nm, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=jnp.complex64,
    )
    ky_index = int(jnp.argmin(jnp.abs(jnp.asarray(grid.ky) - float(args.ky))))
    G0 = G0.at[0, 0, ky_index, 0, :].set(1.0e-3 + 0.0j)
    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)

    def run_custom():
        return integrate_linear(
            G0,
            grid,
            geom,
            params,
            dt=dt,
            steps=steps,
            method=method,
            cache=cache,
        )

    def run_diffrax():
        return integrate_linear_diffrax(
            G0,
            grid,
            geom,
            params,
            dt=dt,
            steps=steps,
            method=args.diffrax_solver,
            adaptive=args.diffrax_adaptive,
            rtol=args.diffrax_rtol,
            atol=args.diffrax_atol,
            max_steps=args.diffrax_max_steps,
            progress_bar=False,
            jit=not args.no_diffrax_jit,
        )

    for _ in range(max(args.warmup, 0)):
        _block(run_custom())
        if not args.skip_diffrax:
            _block(run_diffrax())

    custom = _measure(run_custom, repeat=args.repeat)
    final_state, field_history = _block(run_custom())
    custom["final_state_norm"] = float(jnp.linalg.norm(final_state))
    custom["field_history_norm"] = float(jnp.linalg.norm(field_history))
    custom["finite"] = bool(
        jnp.all(jnp.isfinite(final_state)) and jnp.all(jnp.isfinite(field_history))
    )
    diffrax = None if args.skip_diffrax else _measure(run_diffrax, repeat=args.repeat)

    print("Cyclone defaults benchmark")
    print(f"steps={steps} dt={dt} method={method} Nl={args.Nl} Nm={args.Nm}")
    print(
        f"custom:  median={custom['median_s']:.3f}s  "
        f"host_peak={custom['host_peak_mb']:.2f} MB"
    )
    if diffrax is not None:
        print(
            f"diffrax: median={diffrax['median_s']:.3f}s  "
            f"host_peak={diffrax['host_peak_mb']:.2f} MB"
        )
    print("Note: host_peak is Python tracemalloc (device memory not captured).")
    if args.out_json is not None:
        payload = {
            "kind": "linear_integrator_profile",
            "label": str(args.label),
            "backend": jax.default_backend(),
            "devices": len(jax.devices()),
            "method": method,
            "steps": steps,
            "dt": dt,
            "Nl": int(args.Nl),
            "Nm": int(args.Nm),
            "Nx": int(grid.kx.size),
            "Ny": int(grid.ky.size),
            "Nz": int(grid.z.size),
            "ky": float(grid.ky[ky_index]),
            "custom": custom,
            "diffrax": diffrax,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
