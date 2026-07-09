"""Generic runtime artifact path, validation, and file-writing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from spectraxgk.workflows.runtime.diagnostic_arrays import (
    validate_finite_runtime_diagnostics,
)

_RUNTIME_FIELD_NAMES = ("phi", "apar", "bpar")


def _artifact_base(path: Path) -> Path:
    if path.suffix.lower() in {".json", ".csv", ".npy", ".npz"}:
        return path.with_suffix("")
    return path


def _is_netcdf_output_target(path: Path) -> bool:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    return bool(suffixes and suffixes[-1] == ".nc")


def _netcdf_bundle_base(path: Path) -> Path:
    name = path.name
    for suffix in (".out.nc", ".big.nc", ".restart.nc"):
        if name.lower().endswith(suffix):
            return path.with_name(name[: -len(suffix)])
    if path.suffix.lower() == ".nc":
        return path.with_suffix("")
    return path


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _flatten_series(series: np.ndarray) -> np.ndarray:
    arr = np.asarray(series)
    if arr.ndim == 1:
        return arr
    arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] == 1:
        return arr[:, 0]
    return np.mean(arr, axis=1)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, headers: list[str], cols: list[np.ndarray]) -> None:
    _ensure_parent(path)
    data_out = np.column_stack(cols)
    np.savetxt(path, data_out, delimiter=",", header=",".join(headers), comments="")


def _write_state(base: Path, state: np.ndarray | None) -> Path | None:
    if state is None:
        return None
    path = Path(f"{base}.state.npy")
    _ensure_parent(path)
    np.save(path, np.asarray(state))
    return path


def validate_finite_array(value: Any, *, label: str) -> None:
    """Raise if an optional artifact array contains NaN or infinite values."""

    if value is None:
        return
    arr = np.asarray(value)
    if arr.size == 0 or np.isfinite(arr).all():
        return
    raise RuntimeError(f"{label} contains non-finite values")


def validate_finite_runtime_result(result: Any, *, label: str) -> None:
    """Validate nonlinear runtime result payloads before artifact writes."""

    if result.diagnostics is not None:
        validate_finite_runtime_diagnostics(result.diagnostics, label=label)
    validate_finite_array(result.state, label=f"{label} state")
    fields = result.fields
    if fields is None:
        return
    for name in _RUNTIME_FIELD_NAMES:
        validate_finite_array(getattr(fields, name, None), label=f"{label} {name}")


__all__ = [
    "_artifact_base",
    "_ensure_parent",
    "_flatten_series",
    "_netcdf_bundle_base",
    "_is_netcdf_output_target",
    "_write_csv",
    "_write_json",
    "_write_state",
    "validate_finite_array",
    "validate_finite_runtime_result",
]
