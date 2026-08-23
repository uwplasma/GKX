#!/usr/bin/env python
"""Build the reduced multi-surface/alpha/ky ITG portfolio gate artifact.

Evaluates the growth and quasilinear-flux objectives on a small grid of flux
surfaces, field-line labels, and ky values at the packaged initial parameters,
audits AD against finite differences per sample, and writes the gate payload
(JSON + figure) under ``OUT``.  A few minutes on a laptop CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stellarator_itg_plotting import write_portfolio_gate_artifacts  # noqa: E402
from gkx import (  # noqa: E402
    StellaratorITGOptimizationConfig,
    StellaratorITGSampleSet,
    default_stellarator_initial_params,
    stellarator_itg_portfolio_gate_payload,
)


OUT = ROOT / "docs" / "_static" / "stellarator_itg_portfolio_gate"
SURFACES = (0.50, 0.64, 0.78)  # normalized flux surfaces
ALPHAS = (0.0, 1.0471975511965976)  # field-line alpha values
KY_VALUES = (0.10, 0.30, 0.50)  # ky*rho_i values
OBJECTIVES = ("growth", "quasilinear_flux")  # objective columns
FINITE_DIFFERENCE_WORKERS = 1  # thread workers for finite-difference columns

cfg = StellaratorITGOptimizationConfig()
sample_set = StellaratorITGSampleSet(
    surfaces=SURFACES,
    alphas=ALPHAS,
    ky_values=KY_VALUES,
)
params = default_stellarator_initial_params()
payload = stellarator_itg_portfolio_gate_payload(
    params,
    OBJECTIVES,
    cfg,
    sample_set,
    finite_difference_workers=FINITE_DIFFERENCE_WORKERS,
)
write_portfolio_gate_artifacts(payload, OUT)
print(
    "portfolio gate: "
    f"passed={payload['passed']}, samples={sample_set.n_samples}, "
    f"objectives={','.join(OBJECTIVES)}, artifacts={OUT}"
)
