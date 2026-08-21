#!/usr/bin/env python3
"""Benchmark nonlinear Cyclone runtime for GKX and parse GX runtimes.

Example:
  python benchmarks/performance/benchmark_nonlinear_suite.py --steps 200 --dt 0.0377 \
    --out /tmp/gkx_nl_bench.csv
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark nonlinear Cyclone runtime.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/linear/axisymmetric/runtime_cyclone_nonlinear.toml"),
    )
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--diagnostics-stride", type=int, default=None)
    parser.add_argument("--laguerre-mode", type=str, default=None)
    parser.add_argument("--gx-log", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _parse_gx_runtime(path: Path) -> tuple[float | None, float | None]:
    text = path.read_text(errors="ignore")
    match = re.search(r"Total runtime\s*=\s*([0-9.Ee+-]+)\s*min\s*\(([^)]+)\)", text)
    if not match:
        return None, None
    runtime_min = float(match.group(1))
    step_match = re.search(r"([0-9.Ee+-]+)\s*s\s*/\s*timestep", match.group(2))
    s_per_step = float(step_match.group(1)) if step_match else None
    return runtime_min * 60.0, s_per_step


def main() -> None:
    args = _parse_args()

    from gkx.workflows.runtime.toml import load_runtime_from_toml
    from gkx.runtime import run_runtime_nonlinear

    cfg, _data = load_runtime_from_toml(args.config)

    def _run():
        return run_runtime_nonlinear(
            cfg,
            ky_target=args.ky,
            Nl=args.Nl,
            Nm=args.Nm,
            dt=args.dt,
            steps=args.steps,
            method=args.method,
            sample_stride=args.sample_stride,
            diagnostics_stride=args.diagnostics_stride,
            laguerre_mode=args.laguerre_mode,
            diagnostics=True,
            resolved_diagnostics=False,
        )

    t0 = time.perf_counter()
    _warmup = _run()
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    res = _run()
    t3 = time.perf_counter()

    diag = res.diagnostics
    Wg_end = float(np.asarray(diag.Wg_t[-1])) if diag is not None else float("nan")
    Wphi_end = float(np.asarray(diag.Wphi_t[-1])) if diag is not None else float("nan")
    heat_end = (
        float(np.asarray(diag.heat_flux_t[-1])) if diag is not None else float("nan")
    )

    run_time = t3 - t2
    warmup_time = t1 - t0
    s_per_step = run_time / float(args.steps)

    rows = [
        {
            "backend": "gkx",
            "warmup_s": warmup_time,
            "run_s": run_time,
            "steps": args.steps,
            "s_per_step": s_per_step,
            "Wg_end": Wg_end,
            "Wphi_end": Wphi_end,
            "heat_end": heat_end,
        }
    ]

    if args.gx_log is not None and args.gx_log.exists():
        gx_runtime_s, gx_s_per_step = _parse_gx_runtime(args.gx_log)
        rows.append(
            {
                "backend": "gx",
                "warmup_s": float("nan"),
                "run_s": gx_runtime_s if gx_runtime_s is not None else float("nan"),
                "steps": float("nan"),
                "s_per_step": gx_s_per_step
                if gx_s_per_step is not None
                else float("nan"),
                "Wg_end": float("nan"),
                "Wphi_end": float("nan"),
                "heat_end": float("nan"),
            }
        )

    for row in rows:
        print(
            f"{row['backend']}: run_s={row['run_s']:.3f} s_per_step={row['s_per_step']:.5f}"
        )

    if args.out is not None:
        import csv

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
