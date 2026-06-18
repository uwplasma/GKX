"""Stable facade for logical nonlinear spectral identity gates."""

from __future__ import annotations

from spectraxgk.operators.nonlinear.spectral_identity_integrator import (
    _append_spectral_observables,
    _spectral_integrator_observables,
    integrate_logical_decomposed_nonlinear_spectral,
    nonlinear_spectral_integrator_identity_gate,
)
from spectraxgk.operators.nonlinear.spectral_identity_reports import (
    _nonlinear_spectral_report_blockers,
    _nonlinear_spectral_rhs_report_blockers,
    nonlinear_spectral_communication_identity_gate,
    nonlinear_spectral_communication_identity_report,
    nonlinear_spectral_rhs_identity_report,
)
from spectraxgk.operators.nonlinear.spectral_identity_rhs import (
    _logical_sharded_nonlinear_spectral_rhs,
    logical_decomposed_nonlinear_spectral_rhs,
    nonlinear_spectral_rhs_identity_gate,
)

__all__ = [
    "_append_spectral_observables",
    "_logical_sharded_nonlinear_spectral_rhs",
    "_nonlinear_spectral_report_blockers",
    "_nonlinear_spectral_rhs_report_blockers",
    "_spectral_integrator_observables",
    "integrate_logical_decomposed_nonlinear_spectral",
    "logical_decomposed_nonlinear_spectral_rhs",
    "nonlinear_spectral_communication_identity_gate",
    "nonlinear_spectral_communication_identity_report",
    "nonlinear_spectral_integrator_identity_gate",
    "nonlinear_spectral_rhs_identity_gate",
    "nonlinear_spectral_rhs_identity_report",
]
