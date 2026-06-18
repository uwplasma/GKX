"""Compatibility facade for VMEC-JAX transport admission gates.

The policy dataclasses, nonlinear audit gates, sample-coverage helpers, and
candidate-selection logic live in focused modules under
``spectraxgk.validation.stellarator``. This facade keeps the historical import
path stable for tools, tests, and public API exports.
"""

from __future__ import annotations

# ruff: noqa: F401

from spectraxgk.validation.stellarator.transport_nonlinear import (
    _ensemble_blockers,
    _ensemble_statistics,
    _sample_statistics_summary,
    build_nonlinear_audit_redesign_report,
    build_nonlinear_campaign_admission_report,
    build_nonlinear_landscape_admission_report,
    build_reduced_nonlinear_audit_prelaunch_report,
)
from spectraxgk.validation.stellarator.transport_policies import (
    DEFAULT_TRANSPORT_METRIC_KEYS,
    VMECJAXNonlinearAuditPolicy,
    VMECJAXNonlinearCampaignPolicy,
    VMECJAXReducedPrelaunchPolicy,
    VMECJAXTransportAdmissionPolicy,
    _finite_float_or_none,
)
from spectraxgk.validation.stellarator.transport_samples import (
    _finite_sequence,
    _ky_values_single_grid_compatible,
    _sample_values,
    candidate_transport_metric,
    transport_objective_sample_summary,
)
from spectraxgk.validation.stellarator.transport_selection import (
    _physical_gate_blockers,
    _relative_improvement,
    build_transport_admission_report,
    select_admitted_transport_candidate,
)

__all__ = [
    "DEFAULT_TRANSPORT_METRIC_KEYS",
    "VMECJAXNonlinearAuditPolicy",
    "VMECJAXNonlinearCampaignPolicy",
    "VMECJAXReducedPrelaunchPolicy",
    "VMECJAXTransportAdmissionPolicy",
    "build_nonlinear_landscape_admission_report",
    "build_nonlinear_campaign_admission_report",
    "build_nonlinear_audit_redesign_report",
    "build_reduced_nonlinear_audit_prelaunch_report",
    "build_transport_admission_report",
    "candidate_transport_metric",
    "select_admitted_transport_candidate",
    "transport_objective_sample_summary",
]
