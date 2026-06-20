"""Contracts for reduced stellarator ITG optimization examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np

from spectraxgk.objectives.portfolio_contracts import PortfolioReduction


StellaratorObjectiveKind = Literal["growth", "quasilinear_flux", "nonlinear_heat_flux"]


PARAMETER_NAMES = (
    "minor_radius_log_shift",
    "vertical_elongation_shift",
    "helical_ripple_amplitude",
    "magnetic_shear_shift",
)


OBSERVABLE_NAMES = (
    "aspect",
    "mean_iota",
    "qa_residual",
    "kperp_eff2",
    "growth_rate",
    "frequency",
    "linear_heat_flux_weight",
    "quasilinear_heat_flux",
    "nonlinear_heat_flux_mean",
    "nonlinear_heat_flux_cv",
    "nonlinear_heat_flux_trend",
)


@dataclass(frozen=True)
class StellaratorITGOptimizationConfig:
    """Configuration for the QA max-mode-1 ITG optimization examples."""

    target_aspect: float = 7.0
    target_iota: float = 0.41
    max_mode: int = 1
    aspect_weight: float = 0.25
    iota_weight: float = 25.0
    qa_weight: float = 5.0
    turbulence_weight: float = 1.0
    regularization: float = 2.0e-3
    learning_rate: float = 0.035
    steps: int = 90
    nonlinear_dt: float = 0.18
    nonlinear_steps: int = 520
    nonlinear_tail_fraction: float = 0.25
    quasilinear_csat: float = 0.75
    reference_density_gradient: float = 2.2
    reference_temperature_gradient: float = 6.0
    scan_density_gradients: tuple[float, ...] = (0.8, 1.2, 1.6, 2.2, 3.0, 3.8, 4.8)
    fd_step: float = 1.0e-4

    def with_kind_defaults(self, kind: StellaratorObjectiveKind) -> "StellaratorITGOptimizationConfig":
        """Return conservative optimizer defaults for one objective family."""

        if kind == "growth":
            return replace(self, learning_rate=0.045, steps=max(self.steps, 80), turbulence_weight=1.0)
        if kind == "quasilinear_flux":
            return replace(self, learning_rate=0.030, steps=max(self.steps, 95), turbulence_weight=1.0)
        if kind == "nonlinear_heat_flux":
            return replace(self, learning_rate=0.025, steps=max(self.steps, 110), turbulence_weight=1.0)
        raise ValueError(f"unknown stellarator objective kind {kind!r}")


@dataclass(frozen=True)
class StellaratorITGSampleSet:
    """Reduced multi-surface/multi-alpha/multi-``k_y`` ITG portfolio contract."""

    surfaces: tuple[float, ...] = (0.50, 0.64, 0.78)
    alphas: tuple[float, ...] = (0.0, 1.0471975511965976)
    ky_values: tuple[float, ...] = (0.10, 0.30, 0.50)
    surface_weights: tuple[float, ...] | None = None
    alpha_weights: tuple[float, ...] | None = None
    ky_weights: tuple[float, ...] | None = None
    reduction: PortfolioReduction = "weighted_mean"

    def __post_init__(self) -> None:
        for name, values in (
            ("surfaces", self.surfaces),
            ("alphas", self.alphas),
            ("ky_values", self.ky_values),
        ):
            arr = np.asarray(values, dtype=float)
            if arr.ndim != 1 or arr.size < 1 or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be a non-empty finite vector")
        if np.any(np.asarray(self.ky_values, dtype=float) <= 0.0):
            raise ValueError("ky_values must be positive")
        for name, weights, expected in (
            ("surface_weights", self.surface_weights, len(self.surfaces)),
            ("alpha_weights", self.alpha_weights, len(self.alphas)),
            ("ky_weights", self.ky_weights, len(self.ky_values)),
        ):
            if weights is None:
                continue
            arr = np.asarray(weights, dtype=float)
            if arr.ndim != 1 or arr.size != expected or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be a finite length-{expected} vector")
            if np.any(arr < 0.0) or float(np.sum(arr)) <= 0.0:
                raise ValueError(f"{name} must be non-negative with positive sum")
        if self.reduction not in ("weighted_mean", "mean", "max"):
            raise ValueError("reduction must be weighted_mean, mean, or max")

    @property
    def n_samples(self) -> int:
        """Number of surface/alpha/ky samples in the rectangular portfolio."""

        return len(self.surfaces) * len(self.alphas) * len(self.ky_values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "surfaces": list(self.surfaces),
            "alphas": list(self.alphas),
            "ky_values": list(self.ky_values),
            "surface_weights": None if self.surface_weights is None else list(self.surface_weights),
            "alpha_weights": None if self.alpha_weights is None else list(self.alpha_weights),
            "ky_weights": None if self.ky_weights is None else list(self.ky_weights),
            "reduction": self.reduction,
            "n_samples": self.n_samples,
        }


@dataclass(frozen=True)
class StellaratorITGOptimizationResult:
    """JSON-friendly result for one differentiable stellarator objective."""

    objective_kind: StellaratorObjectiveKind
    parameter_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    initial_params: tuple[float, ...]
    final_params: tuple[float, ...]
    initial_objective: float
    final_objective: float
    initial_observables: tuple[float, ...]
    final_observables: tuple[float, ...]
    history: tuple[dict[str, Any], ...]
    gradient_gate: dict[str, Any]
    covariance: dict[str, Any]
    nonlinear_trace: dict[str, Any] | None
    config: dict[str, Any]
    backend_info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""

        payload = asdict(self)
        payload["parameter_names"] = list(self.parameter_names)
        payload["observable_names"] = list(self.observable_names)
        payload["initial_params"] = list(self.initial_params)
        payload["final_params"] = list(self.final_params)
        payload["initial_observables"] = list(self.initial_observables)
        payload["final_observables"] = list(self.final_observables)
        payload["history"] = list(self.history)
        if self.objective_kind == "nonlinear_heat_flux":
            payload["claim_level"] = (
                "reduced_nonlinear_window_estimator_optimization_not_transport_average"
            )
            payload["nonlinear_transport_scope"] = {
                "model": "smooth_logistic_heat_flux_envelope_from_linear_observables",
                "transport_average_gate": False,
                "production_nonlinear_optimization_claim": False,
                "requires_for_production": [
                    "long post-transient nonlinear transport window",
                    "seed/initial-condition and timestep replicate ensemble",
                    "optimized-equilibrium nonlinear audit",
                ],
            }
        else:
            payload["claim_level"] = "reduced_linear_or_quasilinear_objective_optimization"
        return payload
