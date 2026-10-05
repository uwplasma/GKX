"""Public runtime solve facade: one entry point per solve.

The solve owners live in :mod:`gkx.workflows.linear`,
:mod:`gkx.workflows.nonlinear` and
:mod:`gkx.workflows.runtime.orchestration_scan`; this module only re-exports
them and sits above them, so nothing below imports it.
"""

from __future__ import annotations

from typing import Any

from gkx.config import Case
from gkx.workflows.linear import run_runtime_linear
from gkx.workflows.nonlinear import (
    _runtime_external_phi,
    _select_nonlinear_mode_indices,
    build_runtime_nonlinear_diagnostics_kwargs,
    prepare,
    run_runtime_nonlinear,
)
from gkx.workflows.runtime.orchestration_scan import (
    RuntimeIndependentParallelPlan,
    run_runtime_scan,
)
from gkx.workflows.runtime.results import (
    RuntimeLinearResult,
    RuntimeLinearScanResult,
    RuntimeNonlinearResult,
)
from gkx.workflows.runtime.startup import (
    _build_initial_condition,
    _load_initial_state_from_file,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_linear_terms,
    build_runtime_term_config,
)


def solve(case: Case, **options: Any) -> RuntimeLinearResult | RuntimeNonlinearResult:
    """Solve one case through its existing linear or nonlinear runtime owner."""
    if case.physics.nonlinear:
        return run_runtime_nonlinear(case, **options)
    if case.physics.linear:
        return run_runtime_linear(case, **options)
    raise ValueError("case must enable linear or nonlinear physics")


__all__ = [
    "RuntimeIndependentParallelPlan",
    "RuntimeLinearResult",
    "RuntimeLinearScanResult",
    "RuntimeNonlinearResult",
    "_build_initial_condition",
    "_load_initial_state_from_file",
    "_runtime_external_phi",
    "_select_nonlinear_mode_indices",
    "build_runtime_geometry",
    "build_runtime_linear_params",
    "build_runtime_linear_terms",
    "build_runtime_nonlinear_diagnostics_kwargs",
    "build_runtime_term_config",
    "prepare",
    "run_runtime_linear",
    "run_runtime_nonlinear",
    "run_runtime_scan",
    "solve",
]
