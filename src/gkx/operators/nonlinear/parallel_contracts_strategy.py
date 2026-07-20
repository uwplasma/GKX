"""Readiness-policy contracts for nonlinear parallelization strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

NonlinearParallelStrategyName = Literal[
    "independent_ky_scan",
    "uq_ensemble",
    "whole_state_kx_ky",
    "velocity_species_hermite",
    "fft_axis_domain",
]
ParallelReadiness = Literal["release_ready", "diagnostic", "blocked"]


@dataclass(frozen=True)
class NonlinearParallelStrategy:
    """Readiness contract for one nonlinear parallelization candidate."""

    name: NonlinearParallelStrategyName
    readiness: ParallelReadiness
    independent_work: bool
    changes_solver_layout: bool
    identity_gates: tuple[str, ...]
    physics_gates: tuple[str, ...]
    profiler_gates: tuple[str, ...]
    notes: str

    @property
    def release_ready(self) -> bool:
        """Whether this strategy is allowed for production-facing execution."""

        return self.readiness == "release_ready"

    @property
    def diagnostic_only(self) -> bool:
        """Whether this strategy is limited to correctness/profiling artifacts."""

        return self.readiness == "diagnostic"

    @property
    def blocked(self) -> bool:
        """Whether this strategy is unavailable until required gates exist."""

        return self.readiness == "blocked"

    @property
    def required_gates(self) -> tuple[str, ...]:
        """All identity, physics, and profiler gates required by this policy."""

        return self.identity_gates + self.physics_gates + self.profiler_gates

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the contract."""

        return asdict(self)


_STRATEGIES: tuple[NonlinearParallelStrategy, ...] = (
    NonlinearParallelStrategy(
        name="independent_ky_scan",
        readiness="release_ready",
        independent_work=True,
        changes_solver_layout=False,
        identity_gates=(
            "serial_vs_parallel_ky_scan_growth_rate_identity",
            "serial_vs_parallel_ky_scan_eigenfunction_norm_identity",
        ),
        physics_gates=("linear_ky_scan_reference_physics_gate",),
        profiler_gates=("bounded_independent_ky_scan_scaling_profile",),
        notes="Independent ky tasks preserve solver state layout and ordering.",
    ),
    NonlinearParallelStrategy(
        name="uq_ensemble",
        readiness="release_ready",
        independent_work=True,
        changes_solver_layout=False,
        identity_gates=(
            "serial_vs_parallel_uq_observable_identity",
            "serial_vs_parallel_uq_covariance_identity",
        ),
        physics_gates=("uq_member_physics_gate",),
        profiler_gates=("bounded_uq_ensemble_scaling_profile",),
        notes="Independent UQ members can use batch/thread/process scheduling.",
    ),
    NonlinearParallelStrategy(
        name="whole_state_kx_ky",
        readiness="diagnostic",
        independent_work=False,
        changes_solver_layout=True,
        identity_gates=(
            "whole_state_kx_ky_final_state_identity",
            "whole_state_kx_ky_final_field_identity",
            "whole_state_kx_ky_final_rhs_identity",
        ),
        physics_gates=(
            "nonlinear_window_diagnostic_identity_gate",
            "nonlinear_domain_transport_window_identity",
        ),
        profiler_gates=("matched_cpu_gpu_whole_state_scaling_profile",),
        notes="Current pjit whole-state kx/ky sharding is a correctness/profiler artifact, not a speedup claim.",
    ),
    NonlinearParallelStrategy(
        name="velocity_species_hermite",
        readiness="diagnostic",
        independent_work=False,
        changes_solver_layout=True,
        identity_gates=(
            "hermite_ghost_exchange_identity",
            "velocity_field_reduce_broadcast_identity",
            "velocity_species_linear_rhs_identity",
        ),
        physics_gates=(
            "species_moment_reduction_physics_gate",
            "nonlinear_fixed_step_identity_gate",
        ),
        profiler_gates=("matched_velocity_species_hermite_scaling_profile",),
        notes="Velocity-space production candidate; promotion requires end-to-end nonlinear identity gates.",
    ),
    NonlinearParallelStrategy(
        name="fft_axis_domain",
        readiness="diagnostic",
        independent_work=False,
        changes_solver_layout=True,
        identity_gates=(
            "distributed_fft_forward_inverse_identity",
            "distributed_fft_nonlinear_bracket_identity",
            "distributed_fft_field_solve_identity",
            "logical_sharded_nonlinear_spectral_rhs_identity",
            "logical_sharded_nonlinear_spectral_integrator_identity",
            "pencil_fft_fused_nonlinear_rhs_identity",
            "pencil_fft_physical_transport_window_identity",
            "device_z_pencil_fused_nonlinear_rhs_identity",
            "device_z_pencil_physical_transport_window_identity",
        ),
        physics_gates=("fft_axis_nonlinear_window_physics_gate",),
        profiler_gates=("distributed_fft_scaling_profile",),
        notes=(
            "Diagnostic split/reassemble spectral communication, logical spectral "
            "RHS, and pencil-FFT fused-bracket identity gates exist; production "
            "promotion still requires device-level transport-window routing, "
            "conservation, transport-window, and profiler speedup gates."
        ),
    ),
)

_STRATEGY_BY_NAME: dict[NonlinearParallelStrategyName, NonlinearParallelStrategy] = {
    strategy.name: strategy for strategy in _STRATEGIES
}


__all__ = [
    "NonlinearParallelStrategy",
    "NonlinearParallelStrategyName",
    "ParallelReadiness",
    "_STRATEGIES",
    "_STRATEGY_BY_NAME",
]
