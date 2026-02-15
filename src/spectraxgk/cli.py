"""Command line interface for SPECTRAX-GK."""

from __future__ import annotations

import argparse
import jax.numpy as jnp

from spectraxgk.config import CycloneBaseCase
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid


def _cmd_cyclone_info(_: argparse.Namespace) -> int:
    cfg = CycloneBaseCase()
    print("Cyclone base case")
    print(cfg.to_dict())
    return 0


def _cmd_cyclone_kperp(args: argparse.Namespace) -> int:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    theta = grid.z
    kx0 = jnp.array(args.kx0)
    ky = jnp.array(args.ky)
    kperp2 = geom.k_perp2(kx0, ky, theta)
    print(f"k_perp^2(theta) min={kperp2.min():.6g} max={kperp2.max():.6g}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectrax-gk")
    sub = parser.add_subparsers(dest="cmd", required=True)

    info = sub.add_parser("cyclone-info", help="Print Cyclone base case defaults")
    info.set_defaults(func=_cmd_cyclone_info)

    kperp = sub.add_parser("cyclone-kperp", help="Compute k_perp^2(theta)")
    kperp.add_argument("--kx0", type=float, default=0.0)
    kperp.add_argument("--ky", type=float, default=0.3)
    kperp.set_defaults(func=_cmd_cyclone_kperp)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
