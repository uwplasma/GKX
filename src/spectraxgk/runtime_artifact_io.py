"""Generic runtime artifact path and file-writing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

def _artifact_base(path: Path) -> Path:
    if path.suffix.lower() in {".json", ".csv", ".npy", ".npz"}:
        return path.with_suffix("")
    return path

def _is_gx_netcdf_target(path: Path) -> bool:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    return bool(suffixes and suffixes[-1] == ".nc")

def _gx_bundle_base(path: Path) -> Path:
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

__all__ = [
    "_artifact_base",
    "_ensure_parent",
    "_flatten_series",
    "_gx_bundle_base",
    "_is_gx_netcdf_target",
    "_write_csv",
    "_write_json",
    "_write_state",
]
