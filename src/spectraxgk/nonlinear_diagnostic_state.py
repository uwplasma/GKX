"""Compatibility facade for nonlinear diagnostic-state assembly.

Implementation lives in :mod:`spectraxgk.operators.nonlinear.diagnostic_state`.
"""

from __future__ import annotations

from spectraxgk.operators.nonlinear.diagnostic_state import (
    NonlinearDiagnosticKernels,
    compute_nonlinear_diagnostic_tuple,
)

__all__ = ["NonlinearDiagnosticKernels", "compute_nonlinear_diagnostic_tuple"]
