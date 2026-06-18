"""Contracts for reduced QA low-turbulence optimization diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


QA_LOW_TURBULENCE_DESIGN_NAMES = (
    "qa_constraints",
    "qa_plus_nonlinear_heat_flux",
)

QA_LOW_TURBULENCE_OBSERVABLE_NAMES = (
    "aspect",
    "mean_iota",
    "iota_floor_violation",
    "iota_operating_floor_violation",
    "qa_residual",
    "growth_rate",
    "kperp_eff2",
    "linear_heat_flux_weight",
    "quasilinear_heat_flux",
    "nonlinear_heat_flux_mean",
    "nonlinear_heat_flux_cv",
    "nonlinear_heat_flux_trend",
)


@dataclass(frozen=True)
class QALowTurbulenceConfig:
    """Configuration for the reduced QA low-turbulence comparison."""

    target_aspect: float = 6.0
    min_iota: float = 0.41
    iota_operating_floor: float = 0.70
    max_mode: int = 1
    aspect_weight: float = 8.0
    iota_floor_weight: float = 160.0
    iota_operating_weight: float = 70.0
    qa_weight: float = 8.0
    target_helical_amplitude: float = 0.16
    helical_shaping_weight: float = 24.0
    regularization: float = 2.0e-3
    nonlinear_weight: float = 8.0
    learning_rate: float = 0.030
    steps: int = 60
    nonlinear_dt: float = 0.20
    nonlinear_steps: int = 2000
    nonlinear_tail_fraction: float = 0.50
    long_window_min_time: float = 300.0
    long_window_max_cv: float = 0.03
    long_window_max_trend: float = 0.02
    long_window_max_half_mean_rel_change: float = 0.02
    fixed_density_gradient: float = 2.2
    fixed_temperature_gradient: float = 6.0
    scan_density_gradients: tuple[float, ...] = (
        0.6,
        1.0,
        1.4,
        1.8,
        2.2,
        2.8,
        3.4,
        4.0,
        4.8,
    )
    fd_step: float = 1.0e-4
    surface_ntheta: int = 72
    surface_nzeta: int = 72
    n_field_periods: int = 2


@dataclass(frozen=True)
class QALowTurbulenceResult:
    """JSON-ready result for one reduced QA optimization."""

    design_name: str
    includes_nonlinear_heat_flux: bool
    parameter_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    initial_params: tuple[float, ...]
    final_params: tuple[float, ...]
    initial_objective: float
    final_objective: float
    initial_observables: tuple[float, ...]
    final_observables: tuple[float, ...]
    history: tuple[dict[str, Any], ...]
    residual_gradient_gate: dict[str, Any]
    scalar_gradient_gate: dict[str, Any]
    observable_gradient_gate: dict[str, Any]
    covariance: dict[str, Any]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-friendly payload."""

        return {
            "design_name": self.design_name,
            "includes_nonlinear_heat_flux": self.includes_nonlinear_heat_flux,
            "parameter_names": list(self.parameter_names),
            "observable_names": list(self.observable_names),
            "initial_params": list(self.initial_params),
            "final_params": list(self.final_params),
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "initial_observables": list(self.initial_observables),
            "final_observables": list(self.final_observables),
            "history": list(self.history),
            "residual_gradient_gate": self.residual_gradient_gate,
            "scalar_gradient_gate": self.scalar_gradient_gate,
            "observable_gradient_gate": self.observable_gradient_gate,
            "covariance": self.covariance,
            "config": self.config,
            "claim_level": (
                "reduced_differentiable_qa_low_turbulence_comparison_"
                "not_full_vmec_nonlinear_transport_optimization"
            ),
        }


__all__ = [
    "QA_LOW_TURBULENCE_DESIGN_NAMES",
    "QA_LOW_TURBULENCE_OBSERVABLE_NAMES",
    "QALowTurbulenceConfig",
    "QALowTurbulenceResult",
]
