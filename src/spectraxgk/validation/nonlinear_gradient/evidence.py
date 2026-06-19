"""Claim-boundary gates for nonlinear turbulence-gradient evidence.

This module is intentionally data-only.  It does not run nonlinear solves and
does not infer production turbulence-gradient support from startup finite
differences, reduced nonlinear-window estimators, or single late-window
summaries.  The default behavior is fail-closed unless an artifact explicitly
records production long-window gradient scope, finite-difference conditioning,
gradient uncertainty, and replicated nonlinear-window uncertainty evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from spectraxgk.validation.nonlinear_gradient.evidence_core import (
    NON_PRODUCTION_SCOPE_MARKERS,
    NonlinearTurbulenceGradientBracketSweepConfig,
    NonlinearTurbulenceGradientCandidateRankingConfig,
    NonlinearTurbulenceGradientEvidenceConfig,
    NonlinearTurbulenceGradientFiniteDifferenceConfig,
    NonlinearTurbulenceGradientGapConfig,
    _artifact_passed as _artifact_passed,
    _explicit_production_scope as _explicit_production_scope,
    _finite_float as _finite_float,
    _gate as _gate,
    _gradient_conditioning_summary as _gradient_conditioning_summary,
    _json_number as _json_number,
    _scope_blockers as _scope_blockers,
)
from spectraxgk.validation.nonlinear_gradient.evidence_classification import (
    classify_gradient_artifact,
)
from spectraxgk.validation.nonlinear_gradient.evidence_gap import (
    _required_run_rows as _required_run_rows,
    nonlinear_turbulence_gradient_evidence_gap_report,
    nonlinear_turbulence_gradient_evidence_report,
)
from spectraxgk.validation.nonlinear_gradient.evidence_brackets import (
    _bracket_sweep_recommendation as _bracket_sweep_recommendation,
    _bracket_sweep_row as _bracket_sweep_row,
    _delta_key as _delta_key,
    _paired_same_sign_fraction as _paired_same_sign_fraction,
    _paired_uncertainty_rel as _paired_uncertainty_rel,
    nonlinear_turbulence_gradient_bracket_sweep_report,
)
from spectraxgk.validation.nonlinear_gradient.evidence_screening import (
    _candidate_next_action as _candidate_next_action,
    nonlinear_turbulence_gradient_candidate_ranking_report,
)
from spectraxgk.validation.nonlinear_gradient.evidence_scoring import (
    _metric_margin as _metric_margin,
)
from spectraxgk.validation.nonlinear_gradient.evidence_fd import (
    nonlinear_turbulence_gradient_finite_difference_report,
)
from spectraxgk.validation.nonlinear_gradient.evidence_windows import (
    _ensemble_row as _ensemble_row,
    summarize_window_evidence,
)


def load_json_artifact(path: str | Path) -> dict[str, Any]:
    """Load a JSON object artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


__all__ = [
    "NON_PRODUCTION_SCOPE_MARKERS",
    "NonlinearTurbulenceGradientBracketSweepConfig",
    "NonlinearTurbulenceGradientCandidateRankingConfig",
    "NonlinearTurbulenceGradientEvidenceConfig",
    "NonlinearTurbulenceGradientFiniteDifferenceConfig",
    "NonlinearTurbulenceGradientGapConfig",
    "classify_gradient_artifact",
    "load_json_artifact",
    "nonlinear_turbulence_gradient_bracket_sweep_report",
    "nonlinear_turbulence_gradient_candidate_ranking_report",
    "nonlinear_turbulence_gradient_evidence_gap_report",
    "nonlinear_turbulence_gradient_evidence_report",
    "nonlinear_turbulence_gradient_finite_difference_report",
    "summarize_window_evidence",
]
