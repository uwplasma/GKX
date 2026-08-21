#!/usr/bin/env python
"""Build the reduced multi-surface/alpha/ky ITG portfolio gate artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stellarator_itg_plotting import write_portfolio_gate_artifacts  # noqa: E402
from gkx import (  # noqa: E402
    StellaratorITGOptimizationConfig,
    StellaratorITGSampleSet,
    default_stellarator_initial_params,
    stellarator_itg_portfolio_gate_payload,
)


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "stellarator_itg_portfolio_gate",
        help="Output base path without extension.",
    )
    parser.add_argument("--surfaces", type=_float_tuple, default=(0.50, 0.64, 0.78))
    parser.add_argument("--alphas", type=_float_tuple, default=(0.0, 1.0471975511965976))
    parser.add_argument("--ky-values", type=_float_tuple, default=(0.10, 0.30, 0.50))
    parser.add_argument(
        "--objectives",
        default="growth,quasilinear_flux",
        help="Comma-separated objective columns.",
    )
    parser.add_argument(
        "--finite-difference-workers",
        type=int,
        default=1,
        help="Thread workers for finite-difference columns.",
    )
    args = parser.parse_args()

    objectives = tuple(item.strip() for item in args.objectives.split(",") if item.strip())
    cfg = StellaratorITGOptimizationConfig()
    sample_set = StellaratorITGSampleSet(
        surfaces=args.surfaces,
        alphas=args.alphas,
        ky_values=args.ky_values,
    )
    params = default_stellarator_initial_params()
    payload = stellarator_itg_portfolio_gate_payload(
        params,
        objectives,
        cfg,
        sample_set,
        finite_difference_workers=args.finite_difference_workers,
    )
    write_portfolio_gate_artifacts(payload, args.out)
    print(
        "portfolio gate: "
        f"passed={payload['passed']}, samples={sample_set.n_samples}, "
        f"objectives={','.join(objectives)}, artifacts={args.out}"
    )


if __name__ == "__main__":
    main()
