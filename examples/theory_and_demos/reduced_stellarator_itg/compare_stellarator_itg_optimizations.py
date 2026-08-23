#!/usr/bin/env python
"""Compare growth, quasilinear-flux, and nonlinear-window QA optimizations.

This script runs the same explicit reduced objective workflow used by the three
single-objective examples, mirrors the VMEC-JAX ``QA_optimization.py`` teaching
style, then assembles one publication-style panel with objective histories,
nonlinear-window comparisons, reduced LCFS |B| surfaces, and Boozer-coordinate
LCFS |B| maps.  Writes the comparison artifacts under ``OUT``.  Roughly half an
hour on a laptop CPU (three optimizations run back to back).
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stellarator_itg_plotting import write_comparison_artifacts  # noqa: E402
from _stellarator_itg_workflow import (  # noqa: E402
    compare_scripted_stellarator_itg_objectives,
)
from gkx import StellaratorITGOptimizationConfig  # noqa: E402


OBJECTIVE_KINDS = ("growth", "quasilinear_flux", "nonlinear_heat_flux")
OUT = ROOT / "docs" / "_static" / "stellarator_itg_optimization_comparison"

# Shared editable baseline; each objective applies its own conservative
# default step count and learning rate through ``with_kind_defaults``.
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

WORKERS = 1  # independent objective workers; preserves ordering
PARALLEL_EXECUTOR = "thread"  # "thread" or "process"
FINITE_DIFFERENCE_WORKERS = 1  # thread workers for the FD gradient-gate columns
FINITE_DIFFERENCE_EXECUTOR = "thread"  # "thread" or "process"

cfg = BASE_CONFIG.with_kind_defaults("growth")
payload = compare_scripted_stellarator_itg_objectives(
    OBJECTIVE_KINDS,
    config=cfg,
    workers=WORKERS,
    parallel_executor=PARALLEL_EXECUTOR,
    finite_difference_workers=FINITE_DIFFERENCE_WORKERS,
    finite_difference_executor=FINITE_DIFFERENCE_EXECUTOR,
)
write_comparison_artifacts(payload, OUT)
print(f"comparison artifacts={OUT}")
