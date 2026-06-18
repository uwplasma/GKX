"""Stable facade for nonlinear parallelization contracts and reports."""

from __future__ import annotations

from spectraxgk.operators.nonlinear.parallel_contracts_domain import (
    _NONLINEAR_DOMAIN_CLAIM_SCOPE,
    _NONLINEAR_DOMAIN_GATE_NAME,
    _NONLINEAR_DOMAIN_TRANSPORT_CLAIM_SCOPE,
    _NONLINEAR_DOMAIN_TRANSPORT_GATE_NAME,
    _nonlinear_domain_identity_blockers,
    _nonlinear_domain_plan_blockers,
    NonlinearDomainDecompositionPlan,
    NonlinearDomainIdentityReport,
    NonlinearDomainTransportWindowReport,
)
from spectraxgk.operators.nonlinear.parallel_contracts_spectral import (
    NonlinearSpectralCommunicationReport,
    NonlinearSpectralDevicePencilFFTBatchModel,
    NonlinearSpectralDevicePencilRHSIdentityReport,
    NonlinearSpectralDevicePencilTransportWindowReport,
    NonlinearSpectralDomainWorkModel,
    NonlinearSpectralIntegratorIdentityReport,
    NonlinearSpectralPencilRHSIdentityReport,
    NonlinearSpectralPencilTransportWindowReport,
    NonlinearSpectralPencilWorkModel,
    NonlinearSpectralRHSIdentityReport,
)
from spectraxgk.operators.nonlinear.parallel_contracts_strategy import (
    _STRATEGIES,
    _STRATEGY_BY_NAME,
    NonlinearParallelStrategy,
    NonlinearParallelStrategyName,
    ParallelReadiness,
)

__all__ = [
    "NonlinearDomainDecompositionPlan",
    "NonlinearDomainIdentityReport",
    "NonlinearDomainTransportWindowReport",
    "NonlinearParallelStrategy",
    "NonlinearParallelStrategyName",
    "NonlinearSpectralCommunicationReport",
    "NonlinearSpectralDevicePencilFFTBatchModel",
    "NonlinearSpectralDevicePencilRHSIdentityReport",
    "NonlinearSpectralDevicePencilTransportWindowReport",
    "NonlinearSpectralDomainWorkModel",
    "NonlinearSpectralIntegratorIdentityReport",
    "NonlinearSpectralPencilRHSIdentityReport",
    "NonlinearSpectralPencilTransportWindowReport",
    "NonlinearSpectralPencilWorkModel",
    "NonlinearSpectralRHSIdentityReport",
    "ParallelReadiness",
    "_NONLINEAR_DOMAIN_CLAIM_SCOPE",
    "_NONLINEAR_DOMAIN_GATE_NAME",
    "_NONLINEAR_DOMAIN_TRANSPORT_CLAIM_SCOPE",
    "_NONLINEAR_DOMAIN_TRANSPORT_GATE_NAME",
    "_STRATEGIES",
    "_STRATEGY_BY_NAME",
    "_nonlinear_domain_identity_blockers",
    "_nonlinear_domain_plan_blockers",
]
