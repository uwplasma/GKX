"""Linear and quasilinear runtime artifact writers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spectraxgk.artifacts.io import (
    _artifact_base,
    _flatten_series,
    _write_csv,
    _write_json,
    _write_state,
)


@dataclass(frozen=True)
class _LinearArtifactTargets:
    base: Path
    summary_path: Path
    timeseries_path: Path


@dataclass(frozen=True)
class _LinearScanTargets:
    base: Path
    summary_path: Path
    scan_path: Path


@dataclass(frozen=True)
class _LinearScanArrays:
    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray


@dataclass(frozen=True)
class _QuasilinearSpectrumColumns:
    ky: np.ndarray
    mode_ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray
    kperp_eff2: np.ndarray
    heat: np.ndarray
    particle: np.ndarray
    amp2: np.ndarray
    saturated_heat: np.ndarray
    saturated_particle: np.ndarray


def write_quasilinear_artifacts(
    out: str | Path, quasilinear: dict[str, Any]
) -> dict[str, str]:
    """Write quasilinear summary and species tables."""

    out_path = Path(out)
    base = _artifact_base(out_path)
    summary_path = (
        out_path
        if out_path.suffix.lower() == ".json"
        else Path(f"{base}.quasilinear.summary.json")
    )
    _write_json(summary_path, quasilinear)
    paths = {"quasilinear_summary": str(summary_path)}

    heat = np.asarray(quasilinear.get("heat_flux_weight_species", []), dtype=float)
    particle = np.asarray(
        quasilinear.get("particle_flux_weight_species", []), dtype=float
    )
    if heat.size or particle.size:
        n = max(int(heat.size), int(particle.size))
        heat_col = np.full(n, np.nan, dtype=float)
        particle_col = np.full(n, np.nan, dtype=float)
        if heat.size:
            heat_col[: heat.size] = heat
        if particle.size:
            particle_col[: particle.size] = particle

        sat_heat = np.full(n, np.nan, dtype=float)
        sat_particle = np.full(n, np.nan, dtype=float)
        sat_heat_raw = quasilinear.get("saturated_heat_flux_species")
        sat_particle_raw = quasilinear.get("saturated_particle_flux_species")
        if sat_heat_raw is not None:
            sat = np.asarray(sat_heat_raw, dtype=float)
            sat_heat[: sat.size] = sat
        if sat_particle_raw is not None:
            sat = np.asarray(sat_particle_raw, dtype=float)
            sat_particle[: sat.size] = sat

        species_path = Path(f"{base}.quasilinear_species.csv")
        _write_csv(
            species_path,
            [
                "species_index",
                "heat_flux_weight",
                "particle_flux_weight",
                "saturated_heat_flux",
                "saturated_particle_flux",
            ],
            [
                np.arange(n, dtype=float),
                heat_col,
                particle_col,
                sat_heat,
                sat_particle,
            ],
        )
        paths["quasilinear_species"] = str(species_path)
    return paths


def _runtime_linear_scan_targets(out: str | Path) -> _LinearScanTargets:
    out_path = Path(out)
    base = _artifact_base(out_path)
    summary_path = (
        out_path if out_path.suffix.lower() == ".json" else Path(f"{base}.summary.json")
    )
    scan_path = (
        out_path if out_path.suffix.lower() == ".csv" else Path(f"{base}.scan.csv")
    )
    return _LinearScanTargets(base=base, summary_path=summary_path, scan_path=scan_path)


def _runtime_linear_scan_arrays(result: Any) -> _LinearScanArrays:
    return _LinearScanArrays(
        ky=np.asarray(result.ky, dtype=float),
        gamma=np.asarray(result.gamma, dtype=float),
        omega=np.asarray(result.omega, dtype=float),
    )


def _runtime_linear_scan_summary(
    *,
    arrays: _LinearScanArrays,
    result: Any,
    ql_payloads: tuple[Any, ...],
) -> dict[str, Any]:
    summary = {
        "kind": "linear_scan",
        "n_ky": int(arrays.ky.size),
        "ky_min": None if arrays.ky.size == 0 else float(np.min(arrays.ky)),
        "ky_max": None if arrays.ky.size == 0 else float(np.max(arrays.ky)),
        "has_quasilinear": bool(ql_payloads),
    }
    parallel = getattr(result, "parallel", None)
    if isinstance(parallel, dict):
        summary["parallel"] = parallel
    return summary


def _payload_float(payload: Any, key: str) -> float:
    return float(payload.get(key, np.nan))


def _payload_optional_float(payload: Any, key: str) -> float:
    value = payload.get(key)
    return np.nan if value is None else float(value)


def _quasilinear_scan_columns(
    *,
    ky: np.ndarray,
    ql_payloads: tuple[Any, ...],
) -> _QuasilinearSpectrumColumns:
    # The scan coordinate is the user-requested target ky. Individual linear
    # payloads also carry the selected signed grid-mode ky, which can differ
    # for linked-boundary layouts.
    ql_ky = (
        np.asarray(ky, dtype=float)
        if len(ql_payloads) == int(ky.size)
        else np.asarray([_payload_float(p, "ky") for p in ql_payloads], dtype=float)
    )
    return _QuasilinearSpectrumColumns(
        ky=ql_ky,
        mode_ky=np.asarray([_payload_float(p, "ky") for p in ql_payloads], dtype=float),
        gamma=np.asarray([_payload_float(p, "gamma") for p in ql_payloads], dtype=float),
        omega=np.asarray([_payload_float(p, "omega") for p in ql_payloads], dtype=float),
        kperp_eff2=np.asarray(
            [_payload_float(p, "kperp_eff2") for p in ql_payloads], dtype=float
        ),
        heat=np.asarray(
            [_payload_float(p, "heat_flux_weight_total") for p in ql_payloads],
            dtype=float,
        ),
        particle=np.asarray(
            [_payload_float(p, "particle_flux_weight_total") for p in ql_payloads],
            dtype=float,
        ),
        amp2=np.asarray(
            [_payload_optional_float(p, "amplitude2") for p in ql_payloads],
            dtype=float,
        ),
        saturated_heat=np.asarray(
            [
                _payload_optional_float(p, "saturated_heat_flux_total")
                for p in ql_payloads
            ],
            dtype=float,
        ),
        saturated_particle=np.asarray(
            [
                _payload_optional_float(p, "saturated_particle_flux_total")
                for p in ql_payloads
            ],
            dtype=float,
        ),
    )


def _write_quasilinear_scan_spectrum(
    *,
    base: Path,
    ky: np.ndarray,
    ql_payloads: tuple[Any, ...],
) -> str | None:
    if not ql_payloads:
        return None
    ql_path = Path(f"{base}.quasilinear_spectrum.csv")
    columns = _quasilinear_scan_columns(ky=ky, ql_payloads=ql_payloads)
    _write_csv(
        ql_path,
        [
            "ky",
            "mode_ky",
            "gamma",
            "omega",
            "kperp_eff2",
            "heat_flux_weight_total",
            "particle_flux_weight_total",
            "amplitude2",
            "saturated_heat_flux_total",
            "saturated_particle_flux_total",
        ],
        [
            columns.ky,
            columns.mode_ky,
            columns.gamma,
            columns.omega,
            columns.kperp_eff2,
            columns.heat,
            columns.particle,
            columns.amp2,
            columns.saturated_heat,
            columns.saturated_particle,
        ],
    )
    return str(ql_path)


def _runtime_linear_artifact_targets(out: str | Path) -> _LinearArtifactTargets:
    out_path = Path(out)
    base = _artifact_base(out_path)
    summary_path = (
        out_path if out_path.suffix.lower() == ".json" else Path(f"{base}.summary.json")
    )
    timeseries_path = (
        out_path
        if out_path.suffix.lower() == ".csv"
        else Path(f"{base}.timeseries.csv")
    )
    return _LinearArtifactTargets(
        base=base,
        summary_path=summary_path,
        timeseries_path=timeseries_path,
    )


def _runtime_linear_summary(result: Any) -> dict[str, Any]:
    summary = {
        "kind": "linear",
        "ky": float(result.ky),
        "gamma": float(result.gamma),
        "omega": float(result.omega),
        "fit_window_tmin": None
        if result.fit_window_tmin is None
        else float(result.fit_window_tmin),
        "fit_window_tmax": None
        if result.fit_window_tmax is None
        else float(result.fit_window_tmax),
        "fit_signal_used": result.fit_signal_used,
        "selection": {
            "ky_index": int(result.selection.ky_index),
            "kx_index": int(result.selection.kx_index),
            "z_index": int(result.selection.z_index),
        },
        "n_samples": 0 if result.t is None else int(np.asarray(result.t).size),
        "n_state_shape": None
        if result.state is None
        else list(np.asarray(result.state).shape),
        "has_eigenfunction": bool(
            result.z is not None and result.eigenfunction is not None
        ),
        "has_quasilinear": bool(getattr(result, "quasilinear", None) is not None),
    }
    if getattr(result, "quasilinear", None) is not None:
        summary["quasilinear"] = result.quasilinear
    return summary


def _write_runtime_linear_timeseries(path: Path, result: Any) -> str | None:
    if result.t is None or result.signal is None:
        return None
    signal = _flatten_series(np.asarray(result.signal))
    _write_csv(
        path,
        headers=["t", "signal_real", "signal_imag", "signal_abs"],
        cols=[
            _flatten_series(np.asarray(result.t)),
            np.real(signal),
            np.imag(signal),
            np.abs(signal),
        ],
    )
    return str(path)


def _write_runtime_linear_eigenfunction(base: Path, result: Any) -> str | None:
    if result.z is None or result.eigenfunction is None:
        return None
    eig_path = Path(f"{base}.eigenfunction.csv")
    eig = np.asarray(result.eigenfunction)
    _write_csv(
        eig_path,
        headers=["z", "eigen_real", "eigen_imag", "eigen_abs"],
        cols=[
            np.asarray(result.z, dtype=float),
            np.real(eig),
            np.imag(eig),
            np.abs(eig),
        ],
    )
    return str(eig_path)


def _write_runtime_linear_state(base: Path, result: Any) -> str | None:
    state_path = _write_state(
        base, None if result.state is None else np.asarray(result.state)
    )
    return None if state_path is None else str(state_path)


def _write_runtime_linear_optional_artifacts(
    *,
    targets: _LinearArtifactTargets,
    result: Any,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    timeseries = _write_runtime_linear_timeseries(targets.timeseries_path, result)
    if timeseries is not None:
        paths["timeseries"] = timeseries
    eigenfunction = _write_runtime_linear_eigenfunction(targets.base, result)
    if eigenfunction is not None:
        paths["eigenfunction"] = eigenfunction
    state = _write_runtime_linear_state(targets.base, result)
    if state is not None:
        paths["state"] = state
    if getattr(result, "quasilinear", None) is not None:
        paths.update(write_quasilinear_artifacts(targets.base, result.quasilinear))
    return paths


def write_runtime_linear_scan_artifacts(out: str | Path, result: Any) -> dict[str, str]:
    """Write ky-scan growth/frequency and optional quasilinear spectra."""

    targets = _runtime_linear_scan_targets(out)
    arrays = _runtime_linear_scan_arrays(result)
    ql_payloads = tuple(getattr(result, "quasilinear", None) or ())
    _write_json(
        targets.summary_path,
        _runtime_linear_scan_summary(
            arrays=arrays,
            result=result,
            ql_payloads=ql_payloads,
        ),
    )
    _write_csv(
        targets.scan_path,
        ["ky", "gamma", "omega"],
        [arrays.ky, arrays.gamma, arrays.omega],
    )
    paths = {"summary": str(targets.summary_path), "scan": str(targets.scan_path)}
    ql_spectrum = _write_quasilinear_scan_spectrum(
        base=targets.base,
        ky=arrays.ky,
        ql_payloads=ql_payloads,
    )
    if ql_spectrum is not None:
        paths["quasilinear_spectrum"] = ql_spectrum
    return paths


def write_runtime_linear_artifacts(out: str | Path, result: Any) -> dict[str, str]:
    """Write summary/timeseries/state artifacts for a linear runtime run."""

    targets = _runtime_linear_artifact_targets(out)
    _write_json(targets.summary_path, _runtime_linear_summary(result))
    paths = {"summary": str(targets.summary_path)}
    paths.update(
        _write_runtime_linear_optional_artifacts(targets=targets, result=result)
    )
    return paths


__all__ = [
    "write_quasilinear_artifacts",
    "write_runtime_linear_artifacts",
    "write_runtime_linear_scan_artifacts",
]
