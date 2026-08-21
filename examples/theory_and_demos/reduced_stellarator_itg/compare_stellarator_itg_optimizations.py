#!/usr/bin/env python
"""Compare growth, quasilinear-flux, and nonlinear-window QA optimizations.

This script runs the same explicit reduced objective workflow used by the three
single-objective examples, mirrors the VMEC-JAX ``QA_optimization.py`` teaching
style, then assembles one publication-style panel with objective histories,
nonlinear-window comparisons, reduced LCFS |B| surfaces, and Boozer-coordinate
LCFS |B| maps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stellarator_itg_plotting import write_comparison_artifacts  # noqa: E402
from _stellarator_itg_workflow import (  # noqa: E402
    add_common_stellarator_itg_arguments,
    compare_scripted_stellarator_itg_objectives,
    config_from_args,
)
from gkx import StellaratorITGOptimizationConfig  # noqa: E402


OBJECTIVE_KINDS = ("growth", "quasilinear_flux", "nonlinear_heat_flux")
OUTPUT_BASE = ROOT / "docs" / "_static" / "stellarator_itg_optimization_comparison"

BASE_CONFIG = StellaratorITGOptimizationConfig(
    target_aspect=7.0,
    target_iota=0.41,
    max_mode=1,
    aspect_weight=0.25,
    iota_weight=25.0,
    qa_weight=5.0,
    turbulence_weight=1.0,
    regularization=2.0e-3,
    quasilinear_csat=0.75,
    nonlinear_dt=0.18,
    nonlinear_steps=520,
    nonlinear_tail_fraction=0.25,
    reference_density_gradient=2.2,
    reference_temperature_gradient=6.0,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT_BASE, help="Output base path without extension.")
    parser.add_argument("--workers", type=int, default=1, help="Independent objective workers; preserves ordering.")
    parser.add_argument(
        "--parallel-executor",
        choices=("thread", "process"),
        default="thread",
        help="Executor for independent objective workers.",
    )
    add_common_stellarator_itg_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Shared editable baseline; each objective applies its own conservative
    # default step count and learning rate unless overridden on the CLI.
    cfg = config_from_args(args, base_config=BASE_CONFIG, objective_kind="growth")
    payload = compare_scripted_stellarator_itg_objectives(
        OBJECTIVE_KINDS,
        config=cfg,
        workers=args.workers,
        parallel_executor=args.parallel_executor,
        finite_difference_workers=args.finite_difference_workers,
        finite_difference_executor=args.finite_difference_executor,
    )
    write_comparison_artifacts(payload, args.out)
    print(f"comparison artifacts={args.out}")


if __name__ == "__main__":
    main()
