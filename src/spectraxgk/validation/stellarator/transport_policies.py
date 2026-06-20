"""Policy dataclasses for VMEC-JAX transport admission gates."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np


DEFAULT_TRANSPORT_METRIC_KEYS = (
    "transport_objective_final",
    "spectrax_objective_final",
    "transport_metric_final",
    "objective_final",
)


def _finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


@dataclass(frozen=True)
class VMECJAXTransportAdmissionPolicy:
    """Fail-closed policy for selecting transport-aware VMEC candidates."""

    metric_keys: tuple[str, ...] = DEFAULT_TRANSPORT_METRIC_KEYS
    minimum_relative_improvement: float = 0.0
    lower_is_better: bool = True
    require_authoritative_gate: bool = True
    allow_baseline_fallback: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "metric_keys": list(self.metric_keys),
            "minimum_relative_improvement": float(self.minimum_relative_improvement),
            "lower_is_better": bool(self.lower_is_better),
            "require_authoritative_gate": bool(self.require_authoritative_gate),
            "allow_baseline_fallback": bool(self.allow_baseline_fallback),
        }


@dataclass(frozen=True)
class VMECJAXNonlinearAuditPolicy:
    """Policy for promoting or redesigning VMEC-JAX transport candidates.

    Reduced growth/quasilinear/nonlinear-window objectives are useful only if
    they transfer to late-window nonlinear transport.  This policy encodes the
    minimum replicated-audit evidence and sample coverage required before a
    candidate can be promoted beyond local reduced-metric admission.
    """

    minimum_relative_reduction: float = 0.02
    minimum_uncertainty_z_score: float = 1.0
    maximum_combined_sem_rel: float = 0.25
    minimum_replicate_count: int = 3
    minimum_surface_count: int = 3
    minimum_alpha_count: int = 2
    minimum_ky_count: int = 3
    minimum_sample_count: int = 12
    recommended_surfaces: tuple[float, ...] = (0.45, 0.64, 0.78)
    recommended_alphas: tuple[float, ...] = (0.0, pi / 4.0)
    recommended_ky_values: tuple[float, ...] = (0.10, 0.30, 0.50)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "minimum_relative_reduction": float(self.minimum_relative_reduction),
            "minimum_uncertainty_z_score": float(self.minimum_uncertainty_z_score),
            "maximum_combined_sem_rel": float(self.maximum_combined_sem_rel),
            "minimum_replicate_count": int(self.minimum_replicate_count),
            "minimum_surface_count": int(self.minimum_surface_count),
            "minimum_alpha_count": int(self.minimum_alpha_count),
            "minimum_ky_count": int(self.minimum_ky_count),
            "minimum_sample_count": int(self.minimum_sample_count),
            "recommended_surfaces": [float(item) for item in self.recommended_surfaces],
            "recommended_alphas": [float(item) for item in self.recommended_alphas],
            "recommended_ky_values": [float(item) for item in self.recommended_ky_values],
        }


@dataclass(frozen=True)
class VMECJAXReducedPrelaunchPolicy:
    """Fail-closed reduced-objective gate before expensive nonlinear audits."""

    metric_key: str = "nonlinear_window_heat_flux"
    minimum_relative_reduction: float = 0.04
    failed_reference_safety_factor: float = 1.5
    require_sample_coverage: bool = True
    maximum_cross_sample_sem_rel: float = 0.35

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "metric_key": str(self.metric_key),
            "minimum_relative_reduction": float(self.minimum_relative_reduction),
            "failed_reference_safety_factor": float(self.failed_reference_safety_factor),
            "require_sample_coverage": bool(self.require_sample_coverage),
            "maximum_cross_sample_sem_rel": float(self.maximum_cross_sample_sem_rel),
        }


@dataclass(frozen=True)
class VMECJAXNonlinearCampaignPolicy:
    """Admission limits for launching the next nonlinear optimizer campaign.

    This gate sits between a reduced candidate screen and a broader optimizer
    campaign.  Passing it means the next campaign is worth launching; it does
    not promote a production nonlinear turbulent-flux optimization claim.
    """

    minimum_landscape_relative_reduction: float = 0.10
    minimum_landscape_uncertainty_z_score: float = 3.0
    maximum_landscape_sem_rel: float = 0.05
    minimum_landscape_replicate_count: int = 3
    require_reduced_prelaunch_passed: bool = True
    require_reduced_cross_sample_gate: bool = True
    require_landscape_admission_passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        return {
            "minimum_landscape_relative_reduction": float(
                self.minimum_landscape_relative_reduction
            ),
            "minimum_landscape_uncertainty_z_score": float(
                self.minimum_landscape_uncertainty_z_score
            ),
            "maximum_landscape_sem_rel": float(self.maximum_landscape_sem_rel),
            "minimum_landscape_replicate_count": int(
                self.minimum_landscape_replicate_count
            ),
            "require_reduced_prelaunch_passed": bool(
                self.require_reduced_prelaunch_passed
            ),
            "require_reduced_cross_sample_gate": bool(
                self.require_reduced_cross_sample_gate
            ),
            "require_landscape_admission_passed": bool(
                self.require_landscape_admission_passed
            ),
        }



__all__ = [
    "DEFAULT_TRANSPORT_METRIC_KEYS",
    "VMECJAXNonlinearAuditPolicy",
    "VMECJAXNonlinearCampaignPolicy",
    "VMECJAXReducedPrelaunchPolicy",
    "VMECJAXTransportAdmissionPolicy",
]
