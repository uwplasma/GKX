"""Runtime orchestration facade.

Scan batching, progress formatting, and nonlinear artifact/restart handoff live
in focused owner modules. This facade preserves the established runtime import
path without mixing independent workflow policies in one file.
"""

from __future__ import annotations

from spectraxgk.workflows.runtime.orchestration_artifacts import (
    NonlinearArtifactPolicy,
    RuntimeArtifactHandoffDeps,
    resolve_nonlinear_artifact_policy,
    run_runtime_nonlinear_artifact_handoff,
)
from spectraxgk.workflows.runtime.orchestration_progress import (
    RuntimeProgressSnapshot,
    build_runtime_progress_message,
    format_duration,
)
from spectraxgk.workflows.runtime.orchestration_scan import (
    RuntimeScanBatchDeps,
    RuntimeScanDeps,
    build_runtime_scan_batch_deps,
    build_runtime_scan_orchestration_deps,
    run_runtime_scan_batch,
    run_runtime_scan_orchestration,
)

__all__ = [
    "NonlinearArtifactPolicy",
    "RuntimeArtifactHandoffDeps",
    "RuntimeProgressSnapshot",
    "RuntimeScanBatchDeps",
    "RuntimeScanDeps",
    "build_runtime_progress_message",
    "build_runtime_scan_batch_deps",
    "build_runtime_scan_orchestration_deps",
    "format_duration",
    "resolve_nonlinear_artifact_policy",
    "run_runtime_nonlinear_artifact_handoff",
    "run_runtime_scan_batch",
    "run_runtime_scan_orchestration",
]
