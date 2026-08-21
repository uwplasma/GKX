#!/usr/bin/env python
"""Optimize a QA max-mode-1 stellarator for small adiabatic-electron ITG growth.

This example intentionally mirrors the editable style of VMEC-JAX
``examples/optimization/QA_optimization.py``: problem constants are visible in
this file, the objective is assembled explicitly, then the optimizer, AD/FD
gates, and plots are run as separate script blocks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stellarator_itg_plotting import write_result_artifacts  # noqa: E402
from _stellarator_itg_workflow import (  # noqa: E402
    add_common_stellarator_itg_arguments,
    add_portfolio_arguments,
    config_from_args,
    run_stellarator_itg_adam,
    write_optional_portfolio_artifacts,
)
from gkx import StellaratorITGOptimizationConfig  # noqa: E402


OBJECTIVE_KIND = "growth"
OUTPUT_BASE = ROOT / "docs" / "_static" / "stellarator_itg_growth_optimization"

# Problem parameters.  Edit these directly for exploratory runs, as in the
# VMEC-JAX QA optimization examples.
BASE_CONFIG = StellaratorITGOptimizationConfig(
    target_aspect=7.0,
    target_iota=0.41,
    max_mode=1,
    aspect_weight=0.25,
    iota_weight=25.0,
    qa_weight=5.0,
    turbulence_weight=1.0,
    regularization=2.0e-3,
    reference_density_gradient=2.2,
    reference_temperature_gradient=6.0,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT_BASE, help="Output base path without extension.")
    add_common_stellarator_itg_arguments(parser)
    add_portfolio_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = config_from_args(args, base_config=BASE_CONFIG, objective_kind=OBJECTIVE_KIND)

    print("\nObjective blocks:")
    print("  QA constraints: aspect, mean iota, quasisymmetry, regularization")
    print("  Transport term: dominant reduced ITG growth rate gamma")
    result = run_stellarator_itg_adam(
        OBJECTIVE_KIND,
        config=cfg,
        finite_difference_workers=args.finite_difference_workers,
        finite_difference_executor=args.finite_difference_executor,
    )
    write_result_artifacts(
        result,
        args.out,
        title="QA stellarator optimization for small ITG growth rate",
    )
    portfolio_out = write_optional_portfolio_artifacts(args=args, result=result, out_base=args.out)
    print(
        "growth optimization: "
        f"objective {result.initial_objective:.4e} -> {result.final_objective:.4e}, "
        f"AD/FD gate={result.gradient_gate['passed']}, artifacts={args.out}"
        + ("" if portfolio_out is None else f", portfolio_artifacts={portfolio_out}")
    )


if __name__ == "__main__":
    main()
