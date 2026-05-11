"""Planning contract for nonlinear parallelization strategies.

This module is metadata only. It makes the current release policy explicit
without selecting a solver path, moving arrays, or changing runtime defaults.
"""

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
        physics_gates=("nonlinear_window_diagnostic_identity_gate",),
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
        notes="GX-inspired production candidate; promotion requires end-to-end nonlinear identity gates.",
    ),
    NonlinearParallelStrategy(
        name="fft_axis_domain",
        readiness="blocked",
        independent_work=False,
        changes_solver_layout=True,
        identity_gates=(
            "distributed_fft_forward_inverse_identity",
            "distributed_fft_nonlinear_bracket_identity",
            "distributed_fft_field_solve_identity",
        ),
        physics_gates=("fft_axis_nonlinear_window_physics_gate",),
        profiler_gates=("distributed_fft_scaling_profile",),
        notes="Blocked until distributed FFT identity gates exist.",
    ),
)

_STRATEGY_BY_NAME: dict[NonlinearParallelStrategyName, NonlinearParallelStrategy] = {
    strategy.name: strategy for strategy in _STRATEGIES
}


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
    "NonlinearParallelStrategy",
    "NonlinearParallelStrategyName",
    "ParallelReadiness",
    "classify_nonlinear_parallel_strategy",
    "nonlinear_parallel_strategies",
    "nonlinear_parallel_strategy",
    "release_ready_nonlinear_parallel_strategies",
]
