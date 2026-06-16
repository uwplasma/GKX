"""Linear and quasilinear runtime artifact writers."""

from __future__ import annotations

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


def write_runtime_linear_scan_artifacts(out: str | Path, result: Any) -> dict[str, str]:
    """Write ky-scan growth/frequency and optional quasilinear spectra."""

    out_path = Path(out)
    base = _artifact_base(out_path)
    summary_path = (
        out_path if out_path.suffix.lower() == ".json" else Path(f"{base}.summary.json")
    )
    csv_path = (
        out_path if out_path.suffix.lower() == ".csv" else Path(f"{base}.scan.csv")
    )
    ky = np.asarray(result.ky, dtype=float)
    gamma = np.asarray(result.gamma, dtype=float)
    omega = np.asarray(result.omega, dtype=float)
    ql_payloads = tuple(getattr(result, "quasilinear", None) or ())
    summary = {
        "kind": "linear_scan",
        "n_ky": int(ky.size),
        "ky_min": None if ky.size == 0 else float(np.min(ky)),
        "ky_max": None if ky.size == 0 else float(np.max(ky)),
        "has_quasilinear": bool(ql_payloads),
    }
    parallel = getattr(result, "parallel", None)
    if isinstance(parallel, dict):
        summary["parallel"] = parallel
    _write_json(summary_path, summary)
    _write_csv(csv_path, ["ky", "gamma", "omega"], [ky, gamma, omega])
    paths = {"summary": str(summary_path), "scan": str(csv_path)}

    if ql_payloads:
        ql_path = Path(f"{base}.quasilinear_spectrum.csv")
        # The scan coordinate is the user-requested target ky.  Individual
        # linear payloads also carry the selected signed grid-mode ky, which
        # can differ for linked-boundary layouts.  Keep both so publication
        # spectra remain ordered by requested ky without losing mode metadata.
        ql_ky = (
            np.asarray(ky, dtype=float)
            if len(ql_payloads) == int(ky.size)
            else np.asarray(
                [float(p.get("ky", np.nan)) for p in ql_payloads], dtype=float
            )
        )
        ql_mode_ky = np.asarray(
            [float(p.get("ky", np.nan)) for p in ql_payloads], dtype=float
        )
        ql_gamma = np.asarray(
            [float(p.get("gamma", np.nan)) for p in ql_payloads], dtype=float
        )
        ql_omega = np.asarray(
            [float(p.get("omega", np.nan)) for p in ql_payloads], dtype=float
        )
        kperp_eff2 = np.asarray(
            [float(p.get("kperp_eff2", np.nan)) for p in ql_payloads], dtype=float
        )
        heat = np.asarray(
            [float(p.get("heat_flux_weight_total", np.nan)) for p in ql_payloads],
            dtype=float,
        )
        particle = np.asarray(
            [float(p.get("particle_flux_weight_total", np.nan)) for p in ql_payloads],
            dtype=float,
        )
        amp2 = np.asarray(
            [
                np.nan if p.get("amplitude2") is None else float(p.get("amplitude2"))
                for p in ql_payloads
            ],
            dtype=float,
        )
        sat_heat = np.asarray(
            [
                np.nan
                if p.get("saturated_heat_flux_total") is None
                else float(p.get("saturated_heat_flux_total"))
                for p in ql_payloads
            ],
            dtype=float,
        )
        sat_particle = np.asarray(
            [
                np.nan
                if p.get("saturated_particle_flux_total") is None
                else float(p.get("saturated_particle_flux_total"))
                for p in ql_payloads
            ],
            dtype=float,
        )
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
                ql_ky,
                ql_mode_ky,
                ql_gamma,
                ql_omega,
                kperp_eff2,
                heat,
                particle,
                amp2,
                sat_heat,
                sat_particle,
            ],
        )
        paths["quasilinear_spectrum"] = str(ql_path)
    return paths


def write_runtime_linear_artifacts(out: str | Path, result: Any) -> dict[str, str]:
    """Write summary/timeseries/state artifacts for a linear runtime run."""

    out_path = Path(out)
    base = _artifact_base(out_path)
    summary_path = (
        out_path if out_path.suffix.lower() == ".json" else Path(f"{base}.summary.json")
    )
    csv_path = (
        out_path
        if out_path.suffix.lower() == ".csv"
        else Path(f"{base}.timeseries.csv")
    )

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
    _write_json(summary_path, summary)

    paths = {"summary": str(summary_path)}
    if result.t is not None and result.signal is not None:
        signal = _flatten_series(np.asarray(result.signal))
        _write_csv(
            csv_path,
            headers=["t", "signal_real", "signal_imag", "signal_abs"],
            cols=[
                _flatten_series(np.asarray(result.t)),
                np.real(signal),
                np.imag(signal),
                np.abs(signal),
            ],
        )
        paths["timeseries"] = str(csv_path)

    if result.z is not None and result.eigenfunction is not None:
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
        paths["eigenfunction"] = str(eig_path)

    state_path = _write_state(
        base, None if result.state is None else np.asarray(result.state)
    )
    if state_path is not None:
        paths["state"] = str(state_path)
    if getattr(result, "quasilinear", None) is not None:
        paths.update(write_quasilinear_artifacts(base, result.quasilinear))
    return paths


__all__ = [
    "write_quasilinear_artifacts",
    "write_runtime_linear_artifacts",
    "write_runtime_linear_scan_artifacts",
]
