"""Public facade for nonlinear parallelization contracts and identity gates.

The production-facing paths in this module remain policy metadata. Numerical
routes are conservative diagnostic local-stencil routes: they only enable decomposed
updates after direct numerical identity against serial reference operations.
"""

from __future__ import annotations


from gkx.operators.nonlinear.parallel_contracts_domain import (
    NonlinearDomainDecompositionPlan,
    NonlinearDomainIdentityReport,
    NonlinearDomainTransportWindowReport,
)
from gkx.operators.nonlinear.parallel_contracts_spectral import (
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
from gkx.operators.nonlinear.parallel_contracts_strategy import (
    _STRATEGIES,
    _STRATEGY_BY_NAME,
    NonlinearParallelStrategy,
    NonlinearParallelStrategyName,
    ParallelReadiness,
)


from gkx.operators.nonlinear.domain_decomposition import (
    build_nonlinear_domain_decomposition_plan,
    deterministic_nonlinear_domain_state,
    nonlinear_domain_identity_report,
    nonlinear_domain_parallel_identity_gate,
    nonlinear_domain_transport_window_identity_gate,
    local_stencil_nonlinear_domain_decomposed_step,
    local_stencil_nonlinear_domain_serial_step,
)


from gkx.operators.nonlinear.spectral_core import (
    deterministic_nonlinear_spectral_state,
    device_z_pencil_fft_batch_pressure_model,
    nonlinear_spectral_domain_work_model,
    nonlinear_spectral_pencil_work_model,
)

from gkx.operators.nonlinear.spectral_identity_integrator import (
    integrate_logical_decomposed_nonlinear_spectral,
    nonlinear_spectral_integrator_identity_gate,
    nonlinear_spectral_pencil_transport_window_identity_gate,
)
from gkx.operators.nonlinear.spectral_identity_reports import (
    nonlinear_spectral_communication_identity_gate,
    nonlinear_spectral_communication_identity_report,
    nonlinear_spectral_rhs_identity_report,
)
from gkx.operators.nonlinear.spectral_identity_rhs import (
    logical_decomposed_nonlinear_spectral_rhs,
    nonlinear_spectral_pencil_rhs_identity_gate,
    nonlinear_spectral_rhs_identity_gate,
    pencil_decomposed_nonlinear_spectral_rhs,
)

from gkx.operators.nonlinear.device_z import (
    device_z_pencil_nonlinear_spectral_rhs,
    device_z_pencil_nonlinear_spectral_transport_window_identity_gate,
)


def nonlinear_parallel_strategies() -> tuple[NonlinearParallelStrategy, ...]:
    """Return all nonlinear parallelization strategy contracts."""

    return _STRATEGIES


def nonlinear_parallel_strategy(
    name: NonlinearParallelStrategyName,
) -> NonlinearParallelStrategy:
    """Return the contract for a named nonlinear parallelization strategy."""

    try:
        return _STRATEGY_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown nonlinear parallelization strategy {name!r}"
        ) from exc


def classify_nonlinear_parallel_strategy(
    name: NonlinearParallelStrategyName,
) -> ParallelReadiness:
    """Return the release-readiness classification for a strategy."""

    return nonlinear_parallel_strategy(name).readiness


def release_ready_nonlinear_parallel_strategies() -> tuple[
    NonlinearParallelStrategy, ...
]:
    """Return production-facing strategies that do not alter solver layout."""

    return tuple(strategy for strategy in _STRATEGIES if strategy.release_ready)


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
    "build_nonlinear_domain_decomposition_plan",
    "classify_nonlinear_parallel_strategy",
    "deterministic_nonlinear_domain_state",
    "deterministic_nonlinear_spectral_state",
    "device_z_pencil_fft_batch_pressure_model",
    "device_z_pencil_nonlinear_spectral_rhs",
    "device_z_pencil_nonlinear_spectral_transport_window_identity_gate",
    "integrate_logical_decomposed_nonlinear_spectral",
    "nonlinear_domain_identity_report",
    "nonlinear_domain_parallel_identity_gate",
    "nonlinear_domain_transport_window_identity_gate",
    "nonlinear_parallel_strategies",
    "nonlinear_parallel_strategy",
    "nonlinear_spectral_communication_identity_gate",
    "nonlinear_spectral_communication_identity_report",
    "nonlinear_spectral_domain_work_model",
    "logical_decomposed_nonlinear_spectral_rhs",
    "nonlinear_spectral_integrator_identity_gate",
    "nonlinear_spectral_pencil_rhs_identity_gate",
    "nonlinear_spectral_pencil_transport_window_identity_gate",
    "nonlinear_spectral_pencil_work_model",
    "nonlinear_spectral_rhs_identity_gate",
    "nonlinear_spectral_rhs_identity_report",
    "pencil_decomposed_nonlinear_spectral_rhs",
    "local_stencil_nonlinear_domain_decomposed_step",
    "local_stencil_nonlinear_domain_serial_step",
    "release_ready_nonlinear_parallel_strategies",
]
