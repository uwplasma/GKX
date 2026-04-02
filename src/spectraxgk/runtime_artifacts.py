"""Structured runtime artifact writers for CLI and benchmark tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from spectraxgk.diagnostics import GXDiagnostics


def _artifact_base(path: Path) -> Path:
    if path.suffix.lower() in {".json", ".csv", ".npy", ".npz"}:
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _nonlinear_summary(result: Any) -> dict[str, Any]:
    diag = result.diagnostics
    payload: dict[str, Any] = {
        "kind": "nonlinear",
        "ky_selected": None if result.ky_selected is None else float(result.ky_selected),
        "kx_selected": None if result.kx_selected is None else float(result.kx_selected),
        "n_state_shape": None if result.state is None else list(np.asarray(result.state).shape),
    }
    if diag is not None:
        payload.update(
            {
                "n_samples": int(np.asarray(diag.t).size),
                "t_last": float(np.asarray(diag.t)[-1]) if np.asarray(diag.t).size else 0.0,
                "dt_mean": float(np.asarray(diag.dt_mean)),
                "gamma_last": float(np.asarray(diag.gamma_t)[-1]) if np.asarray(diag.gamma_t).size else 0.0,
                "omega_last": float(np.asarray(diag.omega_t)[-1]) if np.asarray(diag.omega_t).size else 0.0,
                "Wg_last": float(np.asarray(diag.Wg_t)[-1]) if np.asarray(diag.Wg_t).size else 0.0,
                "Wphi_last": float(np.asarray(diag.Wphi_t)[-1]) if np.asarray(diag.Wphi_t).size else 0.0,
                "Wapar_last": float(np.asarray(diag.Wapar_t)[-1]) if np.asarray(diag.Wapar_t).size else 0.0,
                "heat_flux_last": (
                    float(np.asarray(diag.heat_flux_t)[-1]) if np.asarray(diag.heat_flux_t).size else 0.0
                ),
                "particle_flux_last": (
                    float(np.asarray(diag.particle_flux_t)[-1]) if np.asarray(diag.particle_flux_t).size else 0.0
                ),
            }
        )
    elif result.phi2 is not None:
        payload.update(
            {
                "n_samples": 0,
                "t_last": 0.0,
                "phi2_last": float(np.asarray(result.phi2)),
            }
        )
    return payload


def write_runtime_linear_artifacts(out: str | Path, result: Any) -> dict[str, str]:
    """Write summary/timeseries/state artifacts for a linear runtime run."""

    out_path = Path(out)
    base = _artifact_base(out_path)
    summary_path = out_path if out_path.suffix.lower() == ".json" else Path(f"{base}.summary.json")
    csv_path = out_path if out_path.suffix.lower() == ".csv" else Path(f"{base}.timeseries.csv")

    summary = {
        "kind": "linear",
        "ky": float(result.ky),
        "gamma": float(result.gamma),
        "omega": float(result.omega),
        "selection": {
            "ky_index": int(result.selection.ky_index),
            "kx_index": int(result.selection.kx_index),
            "z_index": int(result.selection.z_index),
        },
        "n_samples": 0 if result.t is None else int(np.asarray(result.t).size),
        "n_state_shape": None if result.state is None else list(np.asarray(result.state).shape),
    }
    _write_json(summary_path, summary)

    paths = {"summary": str(summary_path)}
    if result.t is not None and result.signal is not None:
        _write_csv(
            csv_path,
            headers=["t", "signal"],
            cols=[_flatten_series(np.asarray(result.t)), _flatten_series(np.asarray(result.signal))],
        )
        paths["timeseries"] = str(csv_path)

    state_path = _write_state(base, None if result.state is None else np.asarray(result.state))
    if state_path is not None:
        paths["state"] = str(state_path)
    return paths


def write_runtime_nonlinear_artifacts(out: str | Path, result: Any) -> dict[str, str]:
    """Write summary/diagnostics/state artifacts for a nonlinear runtime run."""

    out_path = Path(out)
    base = _artifact_base(out_path)
    summary_path = out_path if out_path.suffix.lower() == ".json" else Path(f"{base}.summary.json")
    csv_path = out_path if out_path.suffix.lower() == ".csv" else Path(f"{base}.diagnostics.csv")

    _write_json(summary_path, _nonlinear_summary(result))
    paths = {"summary": str(summary_path)}
    diag: GXDiagnostics | None = result.diagnostics
    if diag is not None:
        cols = [
            _flatten_series(np.asarray(diag.t)),
            _flatten_series(np.asarray(diag.dt_t)),
            _flatten_series(np.asarray(diag.gamma_t)),
            _flatten_series(np.asarray(diag.omega_t)),
            _flatten_series(np.asarray(diag.Wg_t)),
            _flatten_series(np.asarray(diag.Wphi_t)),
            _flatten_series(np.asarray(diag.Wapar_t)),
            _flatten_series(np.asarray(diag.energy_t)),
            _flatten_series(np.asarray(diag.heat_flux_t)),
            _flatten_series(np.asarray(diag.particle_flux_t)),
        ]
        headers = [
            "t",
            "dt",
            "gamma",
            "omega",
            "Wg",
            "Wphi",
            "Wapar",
            "energy",
            "heat_flux",
            "particle_flux",
        ]
        if diag.heat_flux_species_t is not None:
            heat_s = np.asarray(diag.heat_flux_species_t)
            if heat_s.ndim == 1:
                heat_s = heat_s[:, None]
            for i in range(heat_s.shape[1]):
                cols.append(heat_s[:, i])
                headers.append(f"heat_flux_s{i}")
        if diag.particle_flux_species_t is not None:
            pflux_s = np.asarray(diag.particle_flux_species_t)
            if pflux_s.ndim == 1:
                pflux_s = pflux_s[:, None]
            for i in range(pflux_s.shape[1]):
                cols.append(pflux_s[:, i])
                headers.append(f"particle_flux_s{i}")
        _write_csv(csv_path, headers=headers, cols=cols)
        paths["diagnostics"] = str(csv_path)

    state_path = _write_state(base, None if result.state is None else np.asarray(result.state))
    if state_path is not None:
        paths["state"] = str(state_path)
    return paths
