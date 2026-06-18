"""Compatibility facade for nonlinear stellarator transport admission reports."""

from __future__ import annotations

# ruff: noqa: F401

from spectraxgk.validation.stellarator.transport_audit import (
    build_nonlinear_audit_redesign_report,
)
from spectraxgk.validation.stellarator.transport_campaign import (
    build_nonlinear_campaign_admission_report,
)
from spectraxgk.validation.stellarator.transport_landscape import (
    _ensemble_blockers,
    _ensemble_statistics,
    build_nonlinear_landscape_admission_report,
)
from spectraxgk.validation.stellarator.transport_prelaunch import (
    _sample_statistics_summary,
    build_reduced_nonlinear_audit_prelaunch_report,
)

__all__ = [
    "_ensemble_blockers",
    "_ensemble_statistics",
    "_sample_statistics_summary",
    "build_nonlinear_audit_redesign_report",
    "build_nonlinear_campaign_admission_report",
    "build_nonlinear_landscape_admission_report",
    "build_reduced_nonlinear_audit_prelaunch_report",
]
