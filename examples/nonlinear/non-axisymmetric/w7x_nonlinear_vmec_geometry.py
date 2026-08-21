#!/usr/bin/env python3
"""Run the config-backed W7-X nonlinear VMEC example."""

from __future__ import annotations

import argparse
from pathlib import Path

from gkx import run_nonlinear_case

CONFIG = Path(__file__).resolve().parent / "runtime_w7x_nonlinear_vmec_geometry.toml"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--ky", type=float, default=None)
    p.add_argument("--Nl", type=int, default=None)
    p.add_argument("--Nm", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--dt", type=float, default=None)
    args = p.parse_args()
    return run_nonlinear_case(args.config, ky=args.ky, Nl=args.Nl, Nm=args.Nm, steps=args.steps, dt=args.dt)


if __name__ == "__main__":
    raise SystemExit(main())
