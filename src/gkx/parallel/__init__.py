"""Parallel execution, decomposition, and sharding helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "identity": ("ParallelIdentityReport", "parallel_identity_report"),
    "batch": (
        "batch_map",
        "batch_map_identity_report",
        "ky_scan_batches",
        "pad_to_multiple",
        "split_evenly",
    ),
    "independent": (
        "IndependentEnsembleProvenanceReport",
        "IndependentMapExecutionError",
        "IndependentWorkerMetadata",
        "independent_ensemble_provenance_gate",
        "independent_map",
        "independent_map_identity_report",
        "independent_worker_metadata",
    ),
    "decomposition": (
        "ClaimLevel",
        "DecompositionContract",
        "DecompositionWorkload",
        "DiagnosticWorkload",
        "IndependentWorkload",
        "ReconstructionIdentityReport",
        "ShardAssignment",
        "build_diagnostic_nonlinear_domain_decomposition",
        "build_independent_portfolio_decomposition",
        "reconstruct_serial",
        "serial_reconstruction_identity_report",
        "shard_sequence",
    ),
    "state": ("resolve_state_sharding",),
    "velocity": (
        "VelocityShardingPlan",
        "build_velocity_sharding_plan",
        "curvature_gradb_drift_reference",
        "curvature_gradb_drift_shard_map",
        "diamagnetic_drive_reference",
        "diamagnetic_drive_shard_map",
        "electrostatic_phi_reference",
        "electrostatic_phi_shard_map",
        "hermite_neighbor_reference",
        "hermite_neighbor_shard_map",
        "hermite_shift_reference",
        "hermite_shift_shard_map",
        "hermite_streaming_ladder_reference",
        "hermite_streaming_ladder_shard_map",
        "mirror_drift_reference",
        "mirror_drift_shard_map",
        "periodic_streaming_reference",
        "periodic_streaming_shard_map",
        "velocity_field_reduce_reference",
        "velocity_field_reduce_shard_map",
    ),
    "integrators": ("integrate_linear_sharded", "integrate_nonlinear_sharded"),
}

__all__ = [name for names in _MODULE_EXPORTS.values() for name in names]
if len(__all__) != len(set(__all__)):
    raise RuntimeError("parallel public exports must have one owning module")
_EXPORT_MODULES = {
    name: module_name
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}


def __getattr__(name: str) -> Any:
    """Lazily resolve parallel exports without importing unused JAX kernels."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"gkx.parallel.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
