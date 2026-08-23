#!/usr/bin/env python
"""Optimize a QA max-mode-1 stellarator for small quasilinear ITG heat flux.

The workflow is intentionally explicit and mirrors the VMEC-JAX
``QA_optimization.py`` style: visible problem constants, direct objective
assembly, Adam optimization, AD/FD gradient gates, and explicit artifact
generation.  Writes the result artifacts (JSON + figure) under ``OUT``.
Roughly ten minutes on a laptop CPU at the default step count.
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
    write_optional_portfolio_artifacts,
)
from gkx import StellaratorITGOptimizationConfig  # noqa: E402


OBJECTIVE_KIND = "quasilinear_flux"
OUT = ROOT / "docs" / "_static" / "stellarator_itg_quasilinear_optimization"

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
    quasilinear_csat=0.75,
    reference_density_gradient=2.2,
    reference_temperature_gradient=6.0,
)

FINITE_DIFFERENCE_WORKERS = 1  # thread workers for the FD gradient-gate columns
FINITE_DIFFERENCE_EXECUTOR = "thread"  # "thread" or "process"

# Optional reduced multi-surface/alpha/ky growth+QL portfolio gate at the
# optimized point; None values fall back to the packaged sample set.
PORTFOLIO = False
SURFACES = None  # e.g. (0.50, 0.64, 0.78)
ALPHAS = None  # e.g. (0.0, 1.0471975511965976)
KY_VALUES = None  # e.g. (0.10, 0.30, 0.50)
OBJECTIVE_WEIGHTS = None  # optional portfolio weights: growth,quasilinear_flux

cfg = BASE_CONFIG.with_kind_defaults(OBJECTIVE_KIND)

print("\nObjective blocks:")
print("  QA constraints: aspect, mean iota, quasisymmetry, regularization")
print("  Transport term: mixing-length quasilinear ITG heat-flux proxy")
result = run_stellarator_itg_adam(
    OBJECTIVE_KIND,
    config=cfg,
    finite_difference_workers=FINITE_DIFFERENCE_WORKERS,
    finite_difference_executor=FINITE_DIFFERENCE_EXECUTOR,
)
write_result_artifacts(
    result,
    OUT,
    title="QA stellarator optimization for small quasilinear ITG heat flux",
)
portfolio_out = write_optional_portfolio_artifacts(
    result=result,
    out_base=OUT,
    portfolio=PORTFOLIO,
    surfaces=SURFACES,
    alphas=ALPHAS,
    ky_values=KY_VALUES,
    objective_weights=OBJECTIVE_WEIGHTS,
    finite_difference_workers=FINITE_DIFFERENCE_WORKERS,
    finite_difference_executor=FINITE_DIFFERENCE_EXECUTOR,
)
print(
    "quasilinear-flux optimization: "
    f"objective {result.initial_objective:.4e} -> {result.final_objective:.4e}, "
    f"AD/FD gate={result.gradient_gate['passed']}, artifacts={OUT}"
    + ("" if portfolio_out is None else f", portfolio_artifacts={portfolio_out}")
)
