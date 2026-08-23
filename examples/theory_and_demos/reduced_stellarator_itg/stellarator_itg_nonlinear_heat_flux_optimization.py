#!/usr/bin/env python
"""Optimize a QA max-mode-1 stellarator for a reduced nonlinear ITG window.

The optimized quantity is a differentiable late-window envelope estimator.  It
is useful for AD/FD and optimizer plumbing, but it is not a production
long-time turbulent heat-flux optimization claim.  The script layout follows
the editable VMEC-JAX ``QA_optimization.py`` style.  Writes the result
artifacts (JSON + figure) under ``OUT``.  Roughly ten minutes on a laptop CPU
at the default step count.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stellarator_itg_plotting import write_result_artifacts  # noqa: E402
from _stellarator_itg_workflow import (  # noqa: E402
    run_stellarator_itg_adam,
)
from gkx import StellaratorITGOptimizationConfig  # noqa: E402


OBJECTIVE_KIND = "nonlinear_heat_flux"
OUT = ROOT / "docs" / "_static" / "stellarator_itg_nonlinear_optimization"

# Problem parameters.  Edit these directly for exploratory runs; optimizer
# settings (steps, learning rate, fd_step) come from ``with_kind_defaults``.
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

FINITE_DIFFERENCE_WORKERS = 1  # thread workers for the FD gradient-gate columns
FINITE_DIFFERENCE_EXECUTOR = "thread"  # "thread" or "process"

# No production nonlinear portfolio artifact is written for this objective:
# production nonlinear evidence still requires long post-transient transport
# windows, replicate/seed audits, and optimized-equilibrium nonlinear
# transport validation; this script only reports a reduced nonlinear-window
# estimator.

cfg = BASE_CONFIG.with_kind_defaults(OBJECTIVE_KIND)

print("\nObjective blocks:")
print("  QA constraints: aspect, mean iota, quasisymmetry, regularization")
print("  Transport term: late-window mean of a smooth reduced nonlinear ITG envelope")
result = run_stellarator_itg_adam(
    OBJECTIVE_KIND,
    config=cfg,
    finite_difference_workers=FINITE_DIFFERENCE_WORKERS,
    finite_difference_executor=FINITE_DIFFERENCE_EXECUTOR,
)
write_result_artifacts(
    result,
    OUT,
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
    f"artifacts={OUT}"
)
