"""Public facade for backend-free stellarator objective portfolios.

Implementation is split by responsibility:

- :mod:`spectraxgk.objectives.portfolio_contracts` validates objective tables,
  sample weights, and scalar reductions.
- :mod:`spectraxgk.objectives.portfolio_sensitivity` owns AD/FD,
  conditioning, and covariance gates.
- :mod:`spectraxgk.objectives.portfolio_artifacts` checks real VMEC/Boozer
  reduced-portfolio artifacts before they can support release or manuscript
  claims.

This facade preserves the historical import path for users and tools.
"""

from __future__ import annotations

from spectraxgk.objectives.portfolio_artifacts import (
    ReducedPortfolioArtifactGuardConfig,
    reduced_portfolio_artifact_guard_report,
)
from spectraxgk.objectives.portfolio_contracts import (
    PortfolioReduction,
    StellaratorObjectivePortfolioContract,
    aggregate_objective_portfolio,
    portfolio_objective_weight_vector,
    portfolio_sample_weight_tensor,
    validate_objective_portfolio_contract,
)
from spectraxgk.objectives.portfolio_sensitivity import objective_portfolio_sensitivity_report

__all__ = [
    "PortfolioReduction",
    "ReducedPortfolioArtifactGuardConfig",
    "StellaratorObjectivePortfolioContract",
    "aggregate_objective_portfolio",
    "objective_portfolio_sensitivity_report",
    "portfolio_objective_weight_vector",
    "portfolio_sample_weight_tensor",
    "reduced_portfolio_artifact_guard_report",
    "validate_objective_portfolio_contract",
]
