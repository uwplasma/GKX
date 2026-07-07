"""Mode-selection and eigenfunction extraction diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from spectraxgk.diagnostics.validation_gates import (
    EigenfunctionComparisonMetrics,
    EigenfunctionReferenceBundle,
)

@dataclass(frozen=True)
class ModeSelection:
    ky_index: int
    kx_index: int
    z_index: int = 0


@dataclass(frozen=True)
class ModeSelectionBatch:
    ky_indices: np.ndarray
    kx_index: int
    z_index: int = 0


def select_ky_index(ky: np.ndarray, ky_target: float) -> int:
    """Return the best ky index for a requested target.

    For nonzero requests, prefer a nonzonal mode with the closest absolute
    magnitude, then prefer a sign match when one exists. This avoids collapsing
    sparse signed grids such as ``[0, -k]`` onto the zonal row when the user
    requests ``+k``.
    """

    ky_arr = np.asarray(ky, dtype=float)
    if ky_arr.ndim != 1 or ky_arr.size == 0:
        raise ValueError("ky must be a non-empty one-dimensional array")

    if np.isclose(ky_target, 0.0):
        return int(np.argmin(np.abs(ky_arr)))

    abs_target = abs(float(ky_target))
    magnitude_error = np.abs(np.abs(ky_arr) - abs_target)
    nonzero_penalty = np.isclose(ky_arr, 0.0).astype(int)
    sign_penalty = np.where(np.signbit(ky_arr) == np.signbit(ky_target), 0, 1)
    direct_error = np.abs(ky_arr - float(ky_target))

    order = np.lexsort((direct_error, sign_penalty, nonzero_penalty, magnitude_error))
    return int(order[0])


def extract_mode_time_series(
    phi_t: np.ndarray, sel: ModeSelection, method: str = "z_index"
) -> np.ndarray:
    """Extract a complex mode time series from phi_t(t, ky, kx, z)."""

    data = phi_t[:, sel.ky_index, sel.kx_index, :]
    z_index = min(max(int(sel.z_index), 0), max(data.shape[1] - 1, 0))

    def _late_time_mode(arr: np.ndarray, *, mode_method: str) -> np.ndarray:
        n = arr.shape[0]
        tail_start = int(0.6 * n)
        tail = arr[tail_start:] if tail_start < n else arr
        finite_rows = np.isfinite(tail).all(axis=1)
        if finite_rows.any():
            tail = tail[finite_rows]
        else:
            finite_all = np.isfinite(arr).all(axis=1)
            if finite_all.any():
                tail = arr[finite_all]
        if tail.shape[0] == 0:
            raise ValueError("not enough finite samples for late-time mode extraction")
        if mode_method == "snapshot":
            ref_idx = int(np.argmax(np.linalg.norm(tail, axis=1)))
            return tail[ref_idx]
        if mode_method == "svd":
            _u, _s, vh = np.linalg.svd(tail, full_matrices=False)
            mode = vh[0]
            ref_idx = int(np.argmax(np.linalg.norm(tail, axis=1)))
            ref = tail[ref_idx]
            phase = np.vdot(mode, ref)
            if phase != 0.0:
                mode = mode * np.exp(-1j * np.angle(phase))
            return mode
        raise ValueError("mode_method must be 'snapshot' or 'svd'")

    def _project_onto_mode(arr: np.ndarray, mode: np.ndarray) -> np.ndarray:
        denom = np.vdot(mode, mode)
        if not np.isfinite(denom) or abs(denom) <= 0.0:
            return arr[:, z_index]
        return (arr @ mode.conj()) / denom

    if method == "z_index":
        return data[:, z_index]
    if method == "max":
        idx = np.argmax(np.abs(data), axis=1)
        return data[np.arange(data.shape[0]), idx]
    if method == "project":
        try:
            mode = _late_time_mode(data, mode_method="snapshot")
        except ValueError:
            return data[:, z_index]
        return _project_onto_mode(data, mode)
    if method == "svd":
        if not np.isfinite(data).all():
            return extract_mode_time_series(phi_t, sel, method="z_index")
        try:
            mode = _late_time_mode(data, mode_method="svd")
        except np.linalg.LinAlgError:
            return extract_mode_time_series(phi_t, sel, method="project")
        return _project_onto_mode(data, mode)
    raise ValueError("method must be one of {'z_index', 'max', 'project', 'svd'}")


def extract_mode(phi_t: np.ndarray, sel: ModeSelection) -> np.ndarray:
    """Extract a complex mode time series from phi_t(t, ky, kx, z)."""

    return extract_mode_time_series(phi_t, sel, method="z_index")


def density_moment(
    G: np.ndarray,
    Jl: np.ndarray,
    *,
    species_index: int | None = None,
) -> np.ndarray:
    """Compute the m=0 density moment for a selected species (or summed if None)."""

    if G.ndim == 5:
        Gm0 = G[:, 0, ...]
        return np.sum(Jl * Gm0, axis=0)
    if G.ndim == 6:
        if species_index is None:
            Gm0 = G[:, :, 0, ...]
            return np.sum(Jl[None, ...] * Gm0, axis=1).sum(axis=0)
        Gm0 = G[species_index, :, 0, ...]
        return np.sum(Jl * Gm0, axis=0)
    raise ValueError("G must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")


def extract_eigenfunction(
    phi_t: np.ndarray,
    t: np.ndarray,
    sel: ModeSelection,
    z: np.ndarray | None = None,
    method: str = "svd",
    tmin: float | None = None,
    tmax: float | None = None,
) -> np.ndarray:
    """Extract a normalized eigenfunction in z from phi_t(t, ky, kx, z)."""

    if phi_t.ndim != 4:
        raise ValueError("phi_t must have shape (t, ky, kx, z)")
    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if t.shape[0] != phi_t.shape[0]:
        raise ValueError("t and phi_t must have consistent time dimension")

    mask = np.ones_like(t, dtype=bool)
    if tmin is not None:
        mask &= t >= tmin
    if tmax is not None:
        mask &= t <= tmax
    data = phi_t[mask, sel.ky_index, sel.kx_index, :]
    if data.shape[0] == 0:
        raise ValueError("empty time window for eigenfunction extraction")

    def _snapshot_mode(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1)
        idx = int(np.argmax(norms))
        return arr[idx]

    if method == "snapshot":
        finite_rows = np.isfinite(data).all(axis=1)
        data_finite = data[finite_rows] if finite_rows.any() else data
        mode = _snapshot_mode(data_finite)
    elif method == "svd":
        if not np.isfinite(data).all():
            finite_rows = np.isfinite(data).all(axis=1)
            data_finite = data[finite_rows] if finite_rows.any() else data
            mode = _snapshot_mode(data_finite)
        else:
            try:
                _u, _s, vh = np.linalg.svd(data, full_matrices=False)
                mode = vh[0]
                ref = _snapshot_mode(data)
                phase = np.vdot(mode, ref)
                if phase != 0.0:
                    mode = mode * np.exp(-1j * np.angle(phase))
            except np.linalg.LinAlgError:
                mode = _snapshot_mode(data)
    else:
        raise ValueError("method must be one of {'svd', 'snapshot'}")

    if z is not None:
        if z.ndim != 1:
            raise ValueError("z must be 1D when provided")
        if z.shape[0] != mode.shape[0]:
            raise ValueError("z must have the same length as the eigenfunction")
        idx0 = int(np.argmin(np.abs(z)))
        ref = mode[idx0]
        if ref != 0.0:
            mode = mode / ref
        else:
            scale = np.max(np.abs(mode))
            if scale > 0.0:
                mode = mode / scale
    else:
        scale = np.max(np.abs(mode))
        if scale > 0.0:
            mode = mode / scale
    return mode


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
    "ModeSelection",
    "ModeSelectionBatch",
    "compare_eigenfunctions",
    "density_moment",
    "extract_eigenfunction",
    "extract_mode",
    "extract_mode_time_series",
    "load_eigenfunction_reference_bundle",
    "normalize_eigenfunction",
    "phase_align_eigenfunction",
    "save_eigenfunction_reference_bundle",
    "select_ky_index",
]
