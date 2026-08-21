#!/usr/bin/env python3
"""Run the config-backed KBM nonlinear example."""

from __future__ import annotations

import argparse
from pathlib import Path

from gkx import run_nonlinear_case

CONFIG = Path(__file__).resolve().parent / "runtime_kbm_nonlinear_short.toml"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ky", type=float, default=None)
    p.add_argument("--Nl", type=int, default=None)
    p.add_argument("--Nm", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--sample-stride", type=int, default=None)
    p.add_argument("--diagnostics-stride", type=int, default=None)
    args = p.parse_args()
    return run_nonlinear_case(
        CONFIG,
        ky=args.ky,
        Nl=args.Nl,
        Nm=args.Nm,
        steps=args.steps,
        dt=args.dt,
        sample_stride=args.sample_stride,
        diagnostics_stride=args.diagnostics_stride,
    )


if __name__ == "__main__":
    raise SystemExit(main())
