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
    "state": (
        "HERMITE_MESH_AXIS",
        "SPECIES_MESH_AXIS",
        "build_species_hermite_mesh",
        "resolve_species_hermite_sharding",
        "resolve_state_sharding",
        "species_hermite_state_spec",
    ),
    "velocity": (
        "VelocityShardingPlan",
        "build_species_hermite_mesh_plan",
        "build_velocity_sharding_plan",
        "curvature_gradb_drift_reference",
        "curvature_gradb_drift_shard_map",
        "diamagnetic_drive_reference",
        "diamagnetic_drive_shard_map",
        "electrostatic_phi_reference",
        "electrostatic_phi_shard_map",
        "hermite_field_moment_head",
        "hermite_halo_extend",
        "hermite_halo_interior",
        "hermite_neighbor_reference",
        "hermite_neighbor_shard_map",
        "hermite_shift_reference",
        "hermite_shift_shard_map",
        "hermite_streaming_ladder_reference",
        "hermite_streaming_ladder_shard_map",
        "hermite_window_cache_arrays",
        "hermite_window_indices",
        "mirror_drift_reference",
        "mirror_drift_shard_map",
        "periodic_streaming_reference",
        "periodic_streaming_shard_map",
        "species_hermite_device_counts",
        "velocity_field_reduce_reference",
        "velocity_field_reduce_shard_map",
    ),
    "integrators": (
        "SpeciesHermiteRun",
        "stage_from_host",
        "integrate_linear_sharded",
        "integrate_nonlinear_sharded",
        "integrate_nonlinear_species_hermite",
        "species_hermite_nonlinear_rhs",
    ),
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
