#!/usr/bin/env python
"""Optimize a QA max-mode-1 stellarator for a reduced nonlinear ITG window.

The optimized quantity is a differentiable late-window envelope estimator.  It
is useful for AD/FD and optimizer plumbing, but it is not a production
long-time turbulent heat-flux optimization claim.  The script layout follows
the editable VMEC-JAX ``QA_optimization.py`` style.
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
    config_from_args,
    run_stellarator_itg_adam,
)
from gkx import StellaratorITGOptimizationConfig  # noqa: E402


OBJECTIVE_KIND = "nonlinear_heat_flux"
OUTPUT_BASE = ROOT / "docs" / "_static" / "stellarator_itg_nonlinear_optimization"

BASE_CONFIG = StellaratorITGOptimizationConfig(
    target_aspect=7.0,
    target_iota=0.41,
    max_mode=1,
    aspect_weight=0.25,
    iota_weight=25.0,
    qa_weight=5.0,
    turbulence_weight=1.0,
    regularization=2.0e-3,
    nonlinear_dt=0.18,
    nonlinear_steps=520,
    nonlinear_tail_fraction=0.25,
    reference_density_gradient=2.2,
    reference_temperature_gradient=6.0,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT_BASE, help="Output base path without extension.")
    add_common_stellarator_itg_arguments(parser)
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Print the nonlinear portfolio scope boundary; no production nonlinear portfolio artifact is written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = config_from_args(args, base_config=BASE_CONFIG, objective_kind=OBJECTIVE_KIND)
    if args.portfolio:
        print(
            "portfolio gate not written for nonlinear_heat_flux: production nonlinear evidence still requires "
            "long post-transient transport windows, replicate/seed audits, and optimized-equilibrium nonlinear "
            "transport validation; this script only reports a reduced nonlinear-window estimator."
        )

    print("\nObjective blocks:")
    print("  QA constraints: aspect, mean iota, quasisymmetry, regularization")
    print("  Transport term: late-window mean of a smooth reduced nonlinear ITG envelope")
    result = run_stellarator_itg_adam(
        OBJECTIVE_KIND,
        config=cfg,
        finite_difference_workers=args.finite_difference_workers,
        finite_difference_executor=args.finite_difference_executor,
    )
    write_result_artifacts(
        result,
        args.out,
        title="QA stellarator optimization for a reduced nonlinear ITG window",
    )
    trace = result.nonlinear_trace or {}
    final_window = trace.get("final_window", {})
    print(
        "reduced nonlinear-window optimization: "
        f"objective {result.initial_objective:.4e} -> {result.final_objective:.4e}, "
        f"AD/FD gate={result.gradient_gate['passed']}, "
        f"tail CV={final_window.get('cv', float('nan')):.3e}, "
        f"tail trend={final_window.get('trend', float('nan')):.3e}, "
        f"artifacts={args.out}"
    )


if __name__ == "__main__":
    main()
