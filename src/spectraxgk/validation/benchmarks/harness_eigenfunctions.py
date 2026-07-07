"""Eigenfunction normalization, alignment, and reference bundles."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from spectraxgk.diagnostics.validation_gates import (
    EigenfunctionComparisonMetrics,
    EigenfunctionReferenceBundle,
)


def normalize_eigenfunction(eigenfunction: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Normalize an eigenfunction by its value at theta=0 (nearest z=0)."""

    idx = int(np.argmin(np.abs(z)))
    scale = eigenfunction[idx]
    if scale == 0:
        return eigenfunction
    return eigenfunction / scale


def phase_align_eigenfunction(
    eigenfunction: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, float]:
    """Phase-align ``eigenfunction`` to ``reference`` using the global complex phase."""

    lhs = np.asarray(eigenfunction, dtype=np.complex128)
    rhs = np.asarray(reference, dtype=np.complex128)
    if lhs.shape != rhs.shape:
        raise ValueError("eigenfunction and reference must have the same shape")
    phase = np.vdot(lhs, rhs)
    if abs(phase) <= 1.0e-30:
        return lhs, 0.0
    phase_shift = float(np.angle(phase))
    return lhs * np.exp(1j * phase_shift), phase_shift


def compare_eigenfunctions(
    eigenfunction: np.ndarray, reference: np.ndarray
) -> EigenfunctionComparisonMetrics:
    """Return normalized overlap and relative L2 error after global phase alignment."""

    lhs = np.asarray(eigenfunction, dtype=np.complex128)
    rhs = np.asarray(reference, dtype=np.complex128)
    if lhs.shape != rhs.shape:
        raise ValueError("eigenfunction and reference must have the same shape")
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return EigenfunctionComparisonMetrics(
            overlap=float("nan"),
            relative_l2=float("nan"),
            phase_shift=0.0,
        )
    lhs_aligned, phase_shift = phase_align_eigenfunction(lhs, rhs)
    overlap = float(np.abs(np.vdot(lhs, rhs)) / (lhs_norm * rhs_norm))
    rel_l2 = float(np.linalg.norm(lhs_aligned - rhs) / rhs_norm)
    return EigenfunctionComparisonMetrics(
        overlap=overlap,
        relative_l2=rel_l2,
        phase_shift=phase_shift,
    )


def save_eigenfunction_reference_bundle(
    path: str | Path,
    *,
    theta: np.ndarray,
    mode: np.ndarray,
    source: str,
    case: str,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Write a frozen reference eigenfunction bundle as ``.npz``."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        theta=np.asarray(theta, dtype=float),
        mode=np.asarray(mode, dtype=np.complex128),
        source=np.asarray(str(source)),
        case=np.asarray(str(case)),
        metadata_json=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    )
    return out


def load_eigenfunction_reference_bundle(
    path: str | Path,
) -> EigenfunctionReferenceBundle:
    """Load a frozen reference eigenfunction bundle."""

    data = np.load(Path(path), allow_pickle=False)
    metadata_json = (
        str(np.asarray(data["metadata_json"]).item())
        if "metadata_json" in data
        else "{}"
    )
    return EigenfunctionReferenceBundle(
        theta=np.asarray(data["theta"], dtype=float),
        mode=np.asarray(data["mode"], dtype=np.complex128),
        source=str(np.asarray(data["source"]).item()),
        case=str(np.asarray(data["case"]).item()),
        metadata=json.loads(metadata_json),
    )


__all__ = [
    "compare_eigenfunctions",
    "load_eigenfunction_reference_bundle",
    "normalize_eigenfunction",
    "phase_align_eigenfunction",
    "save_eigenfunction_reference_bundle",
]
